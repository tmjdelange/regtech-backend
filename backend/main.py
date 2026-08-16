import os
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from openai import OpenAI

from database import get_db
from models import Document
from schemas import DocumentCreate, DocumentOut, SearchResult
from auth import verify_api_key

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
    db_doc = Document(content=doc.content, embedding=embedding)
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc


@app.get("/documents/search", response_model=list[SearchResult], dependencies=[Depends(verify_api_key)])
def search_documents(query: str, limit: int = 5, db: Session = Depends(get_db)):
    query_embedding = get_embedding(query)
    results = (
        db.query(
            Document.id,
            Document.content,
            Document.embedding.cosine_distance(query_embedding).label("distance"),
        )
        .order_by("distance")
        .limit(limit)
        .all()
    )
    return [
        SearchResult(id=r.id, content=r.content, distance=r.distance)
        for r in results
    ]
