import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

# Dependencies (auth.py, database.py, main.py) read these from the environment
# at import time, so they must be set before `main` is imported.
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.sql.elements import True_, False_

from database import get_db
from main import app
from models import Document

API_KEY = os.environ["API_KEY"]
ADMIN_API_KEY = os.environ["ADMIN_API_KEY"]


def _criterion_value(criterion):
    """Pull the literal value out of a `Column == value` BinaryExpression.
    Boolean literals compile to True_()/False_() singletons rather than a
    regular BindParameter, so they need special-casing."""
    right = criterion.right
    if isinstance(right, True_):
        return True
    if isinstance(right, False_):
        return False
    return right.value


class FakeQuery:
    """Stand-in for a SQLAlchemy Query that supports the subset of the API
    main.py actually uses (filter/order_by/offset/limit/first/all), evaluated
    in-memory against whatever has been added to the FakeSession so far."""

    def __init__(self, rows, entities):
        self._rows = list(rows)
        self._entities = entities

    def filter(self, *criteria):
        rows = self._rows
        for criterion in criteria:
            key = criterion.left.key
            value = _criterion_value(criterion)
            rows = [r for r in rows if getattr(r, key) == value]
        return FakeQuery(rows, self._entities)

    def order_by(self, *args):
        return self  # ordering doesn't matter for these tests

    def offset(self, n):
        return FakeQuery(self._rows[n:], self._entities)

    def limit(self, n):
        return FakeQuery(self._rows[:n], self._entities)

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return [self._project(r) for r in self._rows]

    def _project(self, row):
        if len(self._entities) == 1 and self._entities[0] is Document:
            return row
        values = {}
        for ent in self._entities:
            key = getattr(ent, "key", None) or getattr(ent, "name", None)
            values[key] = 0.0 if key == "distance" else getattr(row, key)
        return SimpleNamespace(**values)


class FakeSession:
    """Stand-in for a SQLAlchemy Session that only needs to support
    add/commit/refresh/query, so tests don't require a real database
    connection."""

    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        pass

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = len(self.added)

    def query(self, *entities):
        return FakeQuery(self.added, entities)


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


def _seed_document(fake_db, **kwargs):
    defaults = dict(
        content="Some content",
        embedding=[0.0] * 1536,
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
