import json
import os
from unittest.mock import patch

# Dependencies (auth.py, database.py, main.py) read these from the environment
# at import time, so they must be set before `main` is imported.
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest
from fastapi.testclient import TestClient

from database import get_db
from main import app

API_KEY = os.environ["API_KEY"]
ADMIN_API_KEY = os.environ["ADMIN_API_KEY"]


class FakeSession:
    """Stand-in for a SQLAlchemy Session that only needs to support
    add/commit/refresh, so tests don't require a real database connection."""

    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        pass

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = len(self.added)


@pytest.fixture
def fake_db():
    session = FakeSession()
    app.dependency_overrides[get_db] = lambda: session
    yield session
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def mock_embeddings():
    with patch("main.get_embedding", return_value=[0.0] * 1536) as mocked:
        yield mocked


@pytest.fixture
def client():
    return TestClient(app)


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
