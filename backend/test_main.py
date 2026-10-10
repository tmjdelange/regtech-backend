import json
from datetime import datetime, timezone

import pytest

from conftest import ADMIN_API_KEY, API_KEY, DEFAULT_EMBEDDING
from embeddings import embedding_text
from models import Document

# Unit vectors chosen so cosine distance from DEFAULT_EMBEDDING (the query
# vector the mocked embedder returns) is easy to reason about.
NEAR_EMBEDDING = list(DEFAULT_EMBEDDING)                 # distance 0.0
FAR_EMBEDDING = [0.0, 1.0] + [0.0] * 1534                # orthogonal: distance 1.0


def _seed_document(fake_db, **kwargs):
    defaults = dict(
        content="Some content",
        embedding=list(DEFAULT_EMBEDDING),
        title=None,
        country=None,
        category=None,
        source_url=None,
        verified=False,
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    doc = Document(**defaults)
    fake_db.add(doc)
    return doc


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_create_document_requires_auth(client):
    resp = client.post("/documents", json={"content": "test doc"})
    assert resp.status_code == 422  # missing header entirely


def test_create_document_rejects_wrong_key(client):
    resp = client.post(
        "/documents",
        json={"content": "test doc"},
        headers={"x-api-key": "wrong-key"},
    )
    assert resp.status_code == 401  # header present, but invalid


def test_search_requires_auth(client):
    resp = client.get("/documents/search?query=test")
    assert resp.status_code == 422  # missing header entirely


def test_search_rejects_wrong_key(client):
    resp = client.get(
        "/documents/search?query=test",
        headers={"x-api-key": "wrong-key"},
    )
    assert resp.status_code == 401  # header present, but invalid


def test_bulk_upload_requires_admin_key(client, fake_db, mock_embeddings):
    csv_bytes = b"content\nSome content\n"
    files = {"file": ("docs.csv", csv_bytes, "text/csv")}

    # No x-api-key header at all: FastAPI's required Header(...) rejects the
    # request before the dependency body runs (same as verify_api_key).
    resp = client.post("/admin/documents/bulk", files=files)
    assert resp.status_code == 422

    # Wrong key: the *regular* API_KEY must not satisfy the admin dependency.
    resp = client.post(
        "/admin/documents/bulk",
        headers={"x-api-key": API_KEY},
        files=files,
    )
    assert resp.status_code == 401
    assert fake_db.added == []


def test_bulk_upload_csv_success(client, fake_db, mock_embeddings):
    csv_bytes = (
        b"country,category,title,content,source_url\n"
        b"US,tax,Title A,Some regulatory content A,https://example.com/a\n"
        b"UK,finance,Title B,Some regulatory content B,https://example.com/b\n"
    )
    files = {"file": ("docs.csv", csv_bytes, "text/csv")}

    resp = client.post(
        "/admin/documents/bulk",
        headers={"x-api-key": ADMIN_API_KEY},
        files=files,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"inserted": 2, "skipped": []}
    assert len(fake_db.added) == 2
    assert fake_db.added[0].content == "Some regulatory content A"
    assert fake_db.added[0].country == "US"
    assert fake_db.added[0].category == "tax"
    assert fake_db.added[0].source_url == "https://example.com/a"
    assert mock_embeddings.call_count == 2


def test_bulk_upload_saves_title(client, fake_db, mock_embeddings):
    csv_bytes = (
        b"country,category,title,content,source_url\n"
        b"US,tax,Title A,Some regulatory content A,https://example.com/a\n"
    )
    files = {"file": ("docs.csv", csv_bytes, "text/csv")}

    resp = client.post(
        "/admin/documents/bulk",
        headers={"x-api-key": ADMIN_API_KEY},
        files=files,
    )

    assert resp.status_code == 200
    assert resp.json() == {"inserted": 1, "skipped": []}
    assert len(fake_db.added) == 1
    assert fake_db.added[0].title == "Title A"


def test_bulk_upload_json_success(client, fake_db, mock_embeddings):
    rows = [
        {
            "country": "US",
            "category": "tax",
            "title": "Title A",
            "content": "Some regulatory content A",
            "source_url": "https://example.com/a",
        },
        {
            "country": "UK",
            "category": "finance",
            "title": "Title B",
            "content": "Some regulatory content B",
            "source_url": "https://example.com/b",
        },
    ]
    files = {"file": ("docs.json", json.dumps(rows).encode("utf-8"), "application/json")}

    resp = client.post(
        "/admin/documents/bulk",
        headers={"x-api-key": ADMIN_API_KEY},
        files=files,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"inserted": 2, "skipped": []}
    assert len(fake_db.added) == 2
    assert fake_db.added[1].content == "Some regulatory content B"
    assert fake_db.added[1].country == "UK"
    assert mock_embeddings.call_count == 2


def test_bulk_upload_skips_row_missing_content(client, fake_db, mock_embeddings):
    csv_bytes = (
        b"country,category,title,content,source_url\n"
        b"US,tax,Title A,,https://example.com/a\n"
        b"UK,finance,Title B,Real content,https://example.com/b\n"
    )
    files = {"file": ("docs.csv", csv_bytes, "text/csv")}

    resp = client.post(
        "/admin/documents/bulk",
        headers={"x-api-key": ADMIN_API_KEY},
        files=files,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["inserted"] == 1
    assert body["skipped"] == [{"row": 0, "reason": "missing content"}]
    assert len(fake_db.added) == 1
    assert fake_db.added[0].content == "Real content"


def test_public_search_hides_unverified_shows_verified(client, fake_db, mock_embeddings):
    verified_doc = _seed_document(fake_db, id=1, content="Verified content", verified=True)
    _seed_document(fake_db, id=2, content="Unverified content", verified=False)

    resp = client.get(
        "/documents/search?query=test",
        headers={"x-api-key": API_KEY},
    )

    assert resp.status_code == 200
    assert [r["id"] for r in resp.json()] == [verified_doc.id]


def test_admin_search_returns_both(client, fake_db, mock_embeddings):
    _seed_document(fake_db, id=1, content="Verified content", verified=True)
    _seed_document(fake_db, id=2, content="Unverified content", verified=False)

    resp = client.get(
        "/admin/documents/search?query=test",
        headers={"x-api-key": ADMIN_API_KEY},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert {r["id"] for r in body} == {1, 2}
    assert {r["id"]: r["verified"] for r in body} == {1: True, 2: False}


def test_admin_search_verified_filter(client, fake_db, mock_embeddings):
    _seed_document(fake_db, id=1, content="Verified content", verified=True)
    _seed_document(fake_db, id=2, content="Unverified content", verified=False)

    resp = client.get(
        "/admin/documents/search?query=test&verified=true",
        headers={"x-api-key": ADMIN_API_KEY},
    )
    assert resp.status_code == 200
    assert [r["id"] for r in resp.json()] == [1]

    resp = client.get(
        "/admin/documents/search?query=test&verified=false",
        headers={"x-api-key": ADMIN_API_KEY},
    )
    assert resp.status_code == 200
    assert [r["id"] for r in resp.json()] == [2]


def test_patch_verified_flips_flag_and_public_search_finds_it(client, fake_db, mock_embeddings):
    _seed_document(fake_db, id=1, content="Some content", verified=False)

    resp = client.get("/documents/search?query=test", headers={"x-api-key": API_KEY})
    assert resp.json() == []

    resp = client.patch(
        "/admin/documents/1/verified",
        headers={"x-api-key": ADMIN_API_KEY},
        json={"verified": True},
    )
    assert resp.status_code == 200
    assert resp.json()["verified"] is True

    resp = client.get("/documents/search?query=test", headers={"x-api-key": API_KEY})
    assert [r["id"] for r in resp.json()] == [1]


def test_patch_verified_missing_id_returns_404(client, fake_db, mock_embeddings):
    resp = client.patch(
        "/admin/documents/999/verified",
        headers={"x-api-key": ADMIN_API_KEY},
        json={"verified": True},
    )
    assert resp.status_code == 404


def test_admin_endpoints_reject_regular_api_key(client, fake_db, mock_embeddings):
    resp = client.get("/admin/documents/search?query=test", headers={"x-api-key": API_KEY})
    assert resp.status_code == 401

    resp = client.get("/admin/documents", headers={"x-api-key": API_KEY})
    assert resp.status_code == 401

    resp = client.patch(
        "/admin/documents/1/verified",
        headers={"x-api-key": API_KEY},
        json={"verified": True},
    )
    assert resp.status_code == 401


def test_bulk_upload_ignores_verified_in_file(client, fake_db, mock_embeddings):
    csv_bytes = (
        b"country,category,title,content,source_url,verified\n"
        b"US,tax,Title A,Some regulatory content A,https://example.com/a,true\n"
    )
    files = {"file": ("docs.csv", csv_bytes, "text/csv")}

    resp = client.post(
        "/admin/documents/bulk",
        headers={"x-api-key": ADMIN_API_KEY},
        files=files,
    )

    assert resp.status_code == 200
    assert resp.json() == {"inserted": 1, "skipped": []}
    assert len(fake_db.added) == 1
    assert fake_db.added[0].verified is False


# --- Relevance cutoff ---


def test_public_search_applies_default_max_distance(client, fake_db, mock_embeddings):
    _seed_document(fake_db, id=1, embedding=NEAR_EMBEDDING, verified=True)
    _seed_document(fake_db, id=2, embedding=FAR_EMBEDDING, verified=True)

    resp = client.get("/documents/search?query=test", headers={"x-api-key": API_KEY})

    assert resp.status_code == 200
    # The orthogonal row sits at distance 1.0, past the 0.7 default.
    assert [r["id"] for r in resp.json()] == [1]


def test_public_search_respects_explicit_max_distance(client, fake_db, mock_embeddings):
    _seed_document(fake_db, id=1, embedding=NEAR_EMBEDDING, verified=True)
    _seed_document(fake_db, id=2, embedding=FAR_EMBEDDING, verified=True)

    resp = client.get(
        "/documents/search?query=test&max_distance=1.5",
        headers={"x-api-key": API_KEY},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [r["id"] for r in body] == [1, 2]
    assert body[0]["distance"] == pytest.approx(0.0)
    assert body[1]["distance"] == pytest.approx(1.0)

    resp = client.get(
        "/documents/search?query=test&max_distance=0.0",
        headers={"x-api-key": API_KEY},
    )
    assert resp.status_code == 200
    assert [r["id"] for r in resp.json()] == [1]


def test_admin_search_has_no_cutoff_by_default(client, fake_db, mock_embeddings):
    _seed_document(fake_db, id=1, embedding=NEAR_EMBEDDING, verified=True)
    _seed_document(fake_db, id=2, embedding=FAR_EMBEDDING, verified=False)

    resp = client.get(
        "/admin/documents/search?query=test",
        headers={"x-api-key": ADMIN_API_KEY},
    )

    assert resp.status_code == 200
    body = resp.json()
    # Both rows come back even though id=2 is past the public 0.7 cutoff.
    assert [r["id"] for r in body] == [1, 2]
    assert body[1]["distance"] == pytest.approx(1.0)


def test_admin_search_respects_max_distance(client, fake_db, mock_embeddings):
    _seed_document(fake_db, id=1, embedding=NEAR_EMBEDDING, verified=True)
    _seed_document(fake_db, id=2, embedding=FAR_EMBEDDING, verified=True)

    resp = client.get(
        "/admin/documents/search?query=test&max_distance=0.5",
        headers={"x-api-key": ADMIN_API_KEY},
    )

    assert resp.status_code == 200
    assert [r["id"] for r in resp.json()] == [1]


# --- Embedding text ---


def test_embedding_text_prepends_title():
    assert embedding_text("A Title", "Body text") == "A Title\nBody text"


def test_embedding_text_without_title_is_content_only():
    assert embedding_text(None, "Body text") == "Body text"
    assert embedding_text("", "Body text") == "Body text"


def test_create_document_embeds_title_and_content(client, fake_db, mock_embeddings):
    resp = client.post(
        "/documents",
        headers={"x-api-key": API_KEY},
        json={"title": "My Title", "content": "My content"},
    )

    assert resp.status_code == 200
    assert mock_embeddings.call_args.args[0] == "My Title\nMy content"


def test_bulk_upload_embeds_title_and_content(client, fake_db, mock_embeddings):
    csv_bytes = b"title,content\nRow Title,Row content\n"
    files = {"file": ("docs.csv", csv_bytes, "text/csv")}

    resp = client.post(
        "/admin/documents/bulk",
        headers={"x-api-key": ADMIN_API_KEY},
        files=files,
    )

    assert resp.status_code == 200
    assert mock_embeddings.call_args.args[0] == "Row Title\nRow content"
