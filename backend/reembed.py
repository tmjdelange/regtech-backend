"""Re-embed every existing document with the current embedding text.

Run inside the backend container, where DATABASE_URL and OPENAI_API_KEY
are already set:

    docker compose run --rm backend python reembed.py
"""

from database import SessionLocal
from embeddings import embedding_text, get_embedding
from models import Document


def main() -> None:
    db = SessionLocal()
    try:
        ids = [row.id for row in db.query(Document.id).order_by(Document.id).all()]
        print(f"Re-embedding {len(ids)} document(s)...")

        for n, doc_id in enumerate(ids, start=1):
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if doc is None:
                continue
            doc.embedding = get_embedding(embedding_text(doc.title, doc.content))
            # Commit per row so an API failure part-way through doesn't
            # discard the rows already re-embedded.
            db.commit()
            print(f"  [{n}/{len(ids)}] id={doc_id}")

        print("Done.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
