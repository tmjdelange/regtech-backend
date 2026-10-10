import csv
import io
import json
from fastapi import FastAPI, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from database import get_db
from embeddings import embedding_text, get_embedding
from models import Document
from schemas import (
    DocumentCreate,
    DocumentOut,
    SearchResult,
    AdminSearchResult,
    AdminDocumentOut,
    VerifiedUpdate,
)
from auth import verify_api_key, verify_admin_key

app = FastAPI()

DEFAULT_MAX_DISTANCE = 0.7


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


@app.get("/documents/search", response_model=list[SearchResult], dependencies=[Depends(verify_api_key)])
def search_documents(
    query: str,
    limit: int = 5,
    country: str | None = None,
    category: str | None = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    db: Session = Depends(get_db),
):
    query_embedding = get_embedding(query)
    distance = Document.embedding.cosine_distance(query_embedding)
    q = db.query(
        Document.id,
        Document.content,
        distance.label("distance"),
    ).filter(Document.verified == True)
    if country is not None:
        q = q.filter(Document.country == country)
    if category is not None:
        q = q.filter(Document.category == category)
    q = q.filter(distance <= max_distance)
    results = q.order_by("distance").limit(limit).all()
    return [
        SearchResult(id=r.id, content=r.content, distance=r.distance)
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


@app.post("/admin/documents/bulk", dependencies=[Depends(verify_admin_key)])
async def bulk_upload_documents(file: UploadFile = File(...), db: Session = Depends(get_db)):
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
        rows = parsed if isinstance(parsed, list) else [parsed]
    else:
        rows = list(csv.DictReader(io.StringIO(text)))

    inserted = 0
    skipped = []
    for i, row in enumerate(rows):
        content = (row.get("content") or "").strip() if isinstance(row, dict) else ""
        if not content:
            skipped.append({"row": i, "reason": "missing content"})
            continue
        title = row.get("title") or None
        embedding = get_embedding(embedding_text(title, content))
        db_doc = Document(
            content=content,
            embedding=embedding,
            title=title,
            country=row.get("country") or None,
            category=row.get("category") or None,
            source_url=row.get("source_url") or None,
            verified=False,  # always unverified on bulk import, regardless of any "verified" value in the file
        )
        db.add(db_doc)
        inserted += 1

    db.commit()
    return {"inserted": inserted, "skipped": skipped}


@app.get("/admin/documents/search", response_model=list[AdminSearchResult], dependencies=[Depends(verify_admin_key)])
def admin_search_documents(
    query: str,
    limit: int = 5,
    country: str | None = None,
    category: str | None = None,
    verified: bool | None = None,
    max_distance: float | None = None,
    db: Session = Depends(get_db),
):
    query_embedding = get_embedding(query)
    distance = Document.embedding.cosine_distance(query_embedding)
    q = db.query(
        Document.id,
        Document.content,
        Document.verified,
        distance.label("distance"),
    )
    if country is not None:
        q = q.filter(Document.country == country)
    if category is not None:
        q = q.filter(Document.category == category)
    if verified is not None:
        q = q.filter(Document.verified == verified)
    # No cutoff by default, so the admin view can still see distant rows.
    if max_distance is not None:
        q = q.filter(distance <= max_distance)
    results = q.order_by("distance").limit(limit).all()
    return [
        AdminSearchResult(id=r.id, content=r.content, distance=r.distance, verified=r.verified)
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
