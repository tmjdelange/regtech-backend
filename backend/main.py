import csv
import io
import json
from pathlib import Path

from fastapi import FastAPI, Depends, File, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import checklist
import countries
import extraction
from database import get_db
from embeddings import embedding_text, get_embedding
from models import Document
from schemas import (
    DocumentCreate,
    DocumentOut,
    SearchResult,
    AdminSearchResult,
    AdminDocumentOut,
    AdminDocumentDetail,
    DocumentUpdate,
    VerifiedUpdate,
    ChecklistProfile,
    ChecklistResponse,
    CountriesResponse,
    DestinationCode,
    OriginCode,
    ExtractionResponse,
)
from auth import verify_api_key, verify_admin_key

app = FastAPI()

DEFAULT_MAX_DISTANCE = 0.7

_RULES_DIR = Path(__file__).parent / "rules"


def _load_checklist_rules() -> dict:
    """One rules file per destination, named rules/<code>_beverages.json."""
    rules = {}
    for code in countries.DESTINATIONS:
        path = _RULES_DIR / f"{code.lower()}_beverages.json"
        if path.exists():
            with path.open(encoding="utf-8") as f:
                rules[code] = json.load(f)
    return rules


CHECKLIST_RULES_BY_DESTINATION = _load_checklist_rules()
CHECKLIST_RULES = CHECKLIST_RULES_BY_DESTINATION["UK"]


def _fetch_checklist_documents(db: Session, rules: dict) -> dict:
    titles = sorted({title for rule in rules["rules"] for title in rule["sources"]})
    documents = {}
    for title in titles:
        row = db.query(Document).filter(Document.title == title).first()
        if row is not None:
            documents[title] = {
                "source_url": row.source_url,
                "content": row.content,
                "verified": row.verified,
            }
    return documents


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentOut, dependencies=[Depends(verify_api_key)])
def create_document(doc: DocumentCreate, db: Session = Depends(get_db)):
    embedding = get_embedding(embedding_text(doc.title, doc.content))
    db_doc = Document(
        content=doc.content,
        embedding=embedding,
        title=doc.title,
        country=doc.country,
        category=doc.category,
        source_url=doc.source_url,
        verified=False,
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc


@app.get("/countries", response_model=CountriesResponse, dependencies=[Depends(verify_api_key)])
def list_countries(db: Session = Depends(get_db)):
    # Counted live, so a destination appears as soon as one of its rows is
    # verified. EU rows are shared rows, not a market, so they never count.
    counts = {}
    for row in db.query(Document.country).filter(Document.verified == True).all():
        if row.country in countries.DESTINATIONS:
            counts[row.country] = counts.get(row.country, 0) + 1
    return {
        "origins": [
            {"code": code, "name": name, "default": code == countries.DEFAULT_ORIGIN}
            for code, name in countries.ORIGINS.items()
        ],
        "destinations": [
            {
                "code": code,
                "name": countries.DESTINATIONS[code],
                "document_count": count,
                "checklist_available": code in CHECKLIST_RULES_BY_DESTINATION,
                "eu_member": code in countries.EU_MEMBER_STATES,
            }
            for code, count in sorted(counts.items(), key=lambda item: countries.DESTINATIONS[item[0]])
        ],
    }


@app.get("/documents/search", response_model=list[SearchResult], dependencies=[Depends(verify_api_key)])
def search_documents(
    query: str,
    destination: DestinationCode,
    # Validated but not used for filtering: there is no origin-side data yet.
    origin: OriginCode = countries.DEFAULT_ORIGIN,
    limit: int = 5,
    category: str | None = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    db: Session = Depends(get_db),
):
    query_embedding = get_embedding(query)
    distance = Document.embedding.cosine_distance(query_embedding)
    q = db.query(
        Document.id,
        Document.title,
        Document.country,
        Document.content,
        distance.label("distance"),
    ).filter(Document.verified == True)
    # IN excludes rows with a null country, which stay visible to admins.
    q = q.filter(Document.country.in_(countries.search_countries(destination)))
    if category is not None:
        q = q.filter(Document.category == category)
    q = q.filter(distance <= max_distance)
    results = q.order_by("distance").limit(limit).all()
    return [
        SearchResult(id=r.id, title=r.title, country=r.country, content=r.content, distance=r.distance)
        for r in results
    ]


def _is_json_upload(file: UploadFile) -> bool:
    filename = (file.filename or "").lower()
    if filename.endswith(".json"):
        return True
    if filename.endswith(".csv"):
        return False
    content_type = (file.content_type or "").lower()
    return "json" in content_type


async def _read_upload_rows(file: UploadFile) -> list:
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")

    if _is_json_upload(file):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
        return parsed if isinstance(parsed, list) else [parsed]
    return list(csv.DictReader(io.StringIO(text)))


def _clean(value):
    """Strip a string field; blank or missing becomes None."""
    if value is None:
        return None
    return str(value).strip() or None


# Fields an existing row is compared on. Title and country are the row's
# identity, so a re-upload of the same file finds the rows it created.
_BULK_COMPARED_FIELDS = ("content", "category", "source_url")
_BULK_RESULTS = ("inserted", "updated", "skipped", "failed")


def _import_rows(db: Session, rows: list, *, dry_run: bool) -> dict:
    """Insert new rows, update rows whose content, category or source_url
    changed, and skip unchanged ones. Every inserted or updated row ends up
    unverified. With dry_run, report what would happen without writing or
    calling the embedding API."""
    report_rows = []
    seen = set()

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            report_rows.append({"row": i, "title": None, "country": None,
                                "result": "failed", "reason": "not an object", "changed": []})
            continue

        fields = {key: _clean(row.get(key))
                  for key in ("title", "country", "category", "content", "source_url")}
        entry = {"row": i, "title": fields["title"], "country": fields["country"],
                 "result": None, "reason": None, "changed": []}
        report_rows.append(entry)

        missing = [key for key in ("title", "content") if fields[key] is None]
        if missing:
            entry["result"], entry["reason"] = "failed", "missing " + ", ".join(missing)
            continue

        identity = (fields["title"], fields["country"])
        if identity in seen:
            entry["result"], entry["reason"] = "failed", "duplicate title and country in file"
            continue
        seen.add(identity)

        country_filter = (
            Document.country.is_(None) if fields["country"] is None
            else Document.country == fields["country"]
        )
        existing = db.query(Document).filter(Document.title == fields["title"], country_filter).all()
        if len(existing) > 1:
            entry["result"] = "failed"
            entry["reason"] = f"{len(existing)} existing rows have this title and country"
            continue

        if existing:
            doc = existing[0]
            changed = [key for key in _BULK_COMPARED_FIELDS if getattr(doc, key) != fields[key]]
            if not changed:
                entry["result"], entry["reason"] = "skipped", "unchanged"
                continue
            entry["changed"] = changed
            if not dry_run:
                try:
                    if "content" in changed:
                        doc.embedding = get_embedding(embedding_text(fields["title"], fields["content"]))
                except Exception as exc:
                    entry["result"], entry["reason"] = "failed", f"embedding failed ({type(exc).__name__})"
                    continue
                for key in changed:
                    setattr(doc, key, fields[key])
                doc.verified = False
            entry["result"] = "updated"
            continue

        if not dry_run:
            try:
                embedding = get_embedding(embedding_text(fields["title"], fields["content"]))
            except Exception as exc:
                entry["result"], entry["reason"] = "failed", f"embedding failed ({type(exc).__name__})"
                continue
            # always unverified on bulk import, regardless of any "verified" value in the file
            db.add(Document(embedding=embedding, verified=False, **fields))
        entry["result"] = "inserted"

    if not dry_run:
        db.commit()

    by_country = {}
    for entry in report_rows:
        counts = by_country.setdefault(entry["country"] or "(none)",
                                       {result: 0 for result in _BULK_RESULTS})
        counts[entry["result"]] += 1

    totals = {result: sum(1 for e in report_rows if e["result"] == result) for result in _BULK_RESULTS}
    return {"dry_run": dry_run, **totals, "by_country": by_country, "rows": report_rows}


@app.post("/admin/documents/bulk", dependencies=[Depends(verify_admin_key)])
async def bulk_upload_documents(file: UploadFile = File(...), db: Session = Depends(get_db)):
    rows = await _read_upload_rows(file)
    return _import_rows(db, rows, dry_run=False)


# A separate route rather than a ?dry_run flag: a backend that predates it
# answers 404 instead of silently ignoring the flag and importing for real.
@app.post("/admin/documents/bulk/dry-run", dependencies=[Depends(verify_admin_key)])
async def bulk_upload_documents_dry_run(file: UploadFile = File(...), db: Session = Depends(get_db)):
    rows = await _read_upload_rows(file)
    return _import_rows(db, rows, dry_run=True)


@app.get("/admin/documents/search", response_model=list[AdminSearchResult], dependencies=[Depends(verify_admin_key)])
def admin_search_documents(
    query: str,
    limit: int = 5,
    country: str | None = None,
    destination: DestinationCode | None = None,
    category: str | None = None,
    verified: bool | None = None,
    max_distance: float | None = None,
    db: Session = Depends(get_db),
):
    query_embedding = get_embedding(query)
    distance = Document.embedding.cosine_distance(query_embedding)
    q = db.query(
        Document.id,
        Document.title,
        Document.country,
        Document.content,
        Document.verified,
        distance.label("distance"),
    )
    if country is not None:
        q = q.filter(Document.country == country)
    # Same country scoping as public search, so the two views compare like for like.
    if destination is not None:
        q = q.filter(Document.country.in_(countries.search_countries(destination)))
    if category is not None:
        q = q.filter(Document.category == category)
    if verified is not None:
        q = q.filter(Document.verified == verified)
    # No cutoff by default, so the admin view can still see distant rows.
    if max_distance is not None:
        q = q.filter(distance <= max_distance)
    results = q.order_by("distance").limit(limit).all()
    return [
        AdminSearchResult(
            id=r.id, title=r.title, country=r.country, content=r.content,
            distance=r.distance, verified=r.verified,
        )
        for r in results
    ]


@app.get("/admin/documents", response_model=list[AdminDocumentOut], dependencies=[Depends(verify_admin_key)])
def list_admin_documents(limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    return (
        db.query(Document)
        .order_by(Document.id)
        .offset(offset)
        .limit(limit)
        .all()
    )


@app.patch("/admin/documents/{document_id}/verified", response_model=AdminDocumentOut, dependencies=[Depends(verify_admin_key)])
def update_document_verified(document_id: int, payload: VerifiedUpdate, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    doc.verified = payload.verified
    db.commit()
    db.refresh(doc)
    return doc


def _get_document_or_404(db: Session, document_id: int) -> Document:
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.get("/admin/documents/{document_id}", response_model=AdminDocumentDetail, dependencies=[Depends(verify_admin_key)])
def get_admin_document(document_id: int, db: Session = Depends(get_db)):
    return _get_document_or_404(db, document_id)


# Changing any of these means the document says something its reviewer never
# saw, so the update drops it back to unverified (and out of public search and
# the public checklist) until someone reviews it again.
_REVIEWED_FIELDS = {"title", "content", "source_url"}


@app.patch("/admin/documents/{document_id}", response_model=AdminDocumentDetail, dependencies=[Depends(verify_admin_key)])
def update_document(document_id: int, payload: DocumentUpdate, db: Session = Depends(get_db)):
    doc = _get_document_or_404(db, document_id)

    changes = {}
    for field, value in payload.model_dump(exclude_unset=True).items():
        if isinstance(value, str):
            value = value.strip() or None
        if field == "content" and value is None:
            raise HTTPException(status_code=422, detail="Content can't be empty")
        if getattr(doc, field) != value:
            changes[field] = value

    for field, value in changes.items():
        setattr(doc, field, value)
    if changes.keys() & {"title", "content"}:
        doc.embedding = get_embedding(embedding_text(doc.title, doc.content))
    if changes.keys() & _REVIEWED_FIELDS:
        doc.verified = False

    db.commit()
    db.refresh(doc)
    return doc


@app.delete("/admin/documents/{document_id}", status_code=204, dependencies=[Depends(verify_admin_key)])
def delete_document(document_id: int, db: Session = Depends(get_db)):
    doc = _get_document_or_404(db, document_id)
    db.delete(doc)
    db.commit()
    return Response(status_code=204)


def _evaluate_checklist(profile, destination, origin, db, *, include_unverified):
    route = {"origin": origin, "destination": destination}
    rules = CHECKLIST_RULES_BY_DESTINATION.get(destination)
    if rules is None:
        # Never fall back to another destination's rules.
        return {
            **route,
            "checklist_available": False,
            "message": "Checklist not available for this destination yet. You can still search the requirements.",
        }
    profile_dict = profile.model_dump()
    if checklist.is_out_of_scope(rules, profile_dict):
        return {**route, **checklist.out_of_scope_response(rules)}
    documents = _fetch_checklist_documents(db, rules)
    result = checklist.evaluate(rules, profile_dict, documents, include_unverified=include_unverified)
    return {**route, **result}


@app.post("/checklist/evaluate", response_model=ChecklistResponse, dependencies=[Depends(verify_api_key)])
def evaluate_checklist(
    profile: ChecklistProfile,
    destination: DestinationCode,
    origin: OriginCode = countries.DEFAULT_ORIGIN,
    db: Session = Depends(get_db),
):
    return _evaluate_checklist(profile, destination, origin, db, include_unverified=False)


@app.post("/admin/checklist/evaluate", response_model=ChecklistResponse, dependencies=[Depends(verify_admin_key)])
def evaluate_checklist_admin(
    profile: ChecklistProfile,
    destination: DestinationCode,
    origin: OriginCode = countries.DEFAULT_ORIGIN,
    db: Session = Depends(get_db),
):
    return _evaluate_checklist(profile, destination, origin, db, include_unverified=True)


def _client_ip(request: Request) -> str:
    # Requests arrive through the frontend's server-side proxy, which passes
    # the visitor's address on. Only callers holding the API key get here, so
    # the header is trusted.
    forwarded = request.headers.get("x-client-ip", "").strip()
    if forwarded:
        return forwarded
    return request.client.host if request.client else "unknown"


# Multipart is parsed by hand rather than with UploadFile parameters:
# Starlette writes uploads over 1MB to a temp file, and these must stay in
# memory. See extraction.py.
@app.post("/extract", response_model=ExtractionResponse, dependencies=[Depends(verify_api_key)])
async def extract_from_documents(request: Request):
    try:
        if not extraction.is_configured():
            raise extraction.ExtractionError(503, "Extraction is not available right now")
        uploads = await extraction.read_uploads(request.headers.get("content-type", ""), request.stream())
        content, report_text = await run_in_threadpool(extraction.build_content, uploads)
        refused = extraction.rate_limiter.acquire(_client_ip(request))
        if refused:
            raise extraction.ExtractionError(429, refused)
        raw = await extraction.call_model(content)
        return extraction.apply_rules(
            raw,
            has_label=any(u.field == "label" for u in uploads),
            has_report=any(u.field == "lab_report" for u in uploads),
            report_text=report_text,
        )
    except extraction.ExtractionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
