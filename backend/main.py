import csv
import io
import json
import os
from fastapi import FastAPI, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session
from openai import OpenAI

from database import get_db
from models import Document
from schemas import DocumentCreate, DocumentOut, SearchResult
from auth import verify_api_key, verify_admin_key

app = FastAPI()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentOut, dependencies=[Depends(verify_api_key)])
def create_document(doc: DocumentCreate, db: Session = Depends(get_db)):
    embedding = get_embedding(doc.content)
    db_doc = Document(
        content=doc.content,
        embedding=embedding,
        country=doc.country,
        category=doc.category,
        source_url=doc.source_url,
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
    db: Session = Depends(get_db),
):
    query_embedding = get_embedding(query)
    q = db.query(
        Document.id,
        Document.content,
        Document.embedding.cosine_distance(query_embedding).label("distance"),
    )
    if country is not None:
        q = q.filter(Document.country == country)
    if category is not None:
        q = q.filter(Document.category == category)
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
        embedding = get_embedding(content)
        db_doc = Document(
            content=content,
            embedding=embedding,
            country=row.get("country") or None,
            category=row.get("category") or None,
            source_url=row.get("source_url") or None,
        )
        db.add(db_doc)
        inserted += 1

    db.commit()
    return {"inserted": inserted, "skipped": skipped}
