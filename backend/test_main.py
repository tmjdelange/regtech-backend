import json
from datetime import datetime, timezone

import pytest

from conftest import ADMIN_API_KEY, API_KEY, DEFAULT_EMBEDDING
from embeddings import embedding_text
from main import CHECKLIST_RULES
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
        country="UK",
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
    resp = client.get("/documents/search?destination=UK&query=test")
    assert resp.status_code == 422  # missing header entirely


def test_search_rejects_wrong_key(client):
    resp = client.get(
        "/documents/search?destination=UK&query=test",
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
    assert (body["inserted"], body["updated"], body["skipped"], body["failed"]) == (2, 0, 0, 0)
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
    assert resp.json()["inserted"] == 1
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
    assert (body["inserted"], body["updated"], body["skipped"], body["failed"]) == (2, 0, 0, 0)
    assert len(fake_db.added) == 2
    assert fake_db.added[1].content == "Some regulatory content B"
    assert fake_db.added[1].country == "UK"
    assert mock_embeddings.call_count == 2


def _bulk_rows():
    rows = [
        {"title": "DE rule", "country": "DE", "category": "beverages_non_alcoholic",
         "content": "German content", "source_url": "https://example.com/de"},
        {"title": "FR rule", "country": "FR", "category": "beverages_non_alcoholic",
         "content": "French content", "source_url": "https://example.com/fr"},
        {"title": "US rule", "country": "US", "category": "beverages_non_alcoholic",
         "content": "US content", "source_url": "https://example.com/us"},
    ]
    return rows


def _post_bulk(client, rows, *, dry_run=False):
    path = "/admin/documents/bulk/dry-run" if dry_run else "/admin/documents/bulk"
    files = {"file": ("docs.json", json.dumps(rows).encode("utf-8"), "application/json")}
    resp = client.post(path, headers={"x-api-key": ADMIN_API_KEY}, files=files)
    assert resp.status_code == 200
    return resp.json()


def _counts(body):
    return body["inserted"], body["updated"], body["skipped"], body["failed"]


def test_bulk_import_valid_file_reports_counts_per_country(client, fake_db, mock_embeddings):
    body = _post_bulk(client, _bulk_rows())

    assert body["dry_run"] is False
    assert _counts(body) == (3, 0, 0, 0)
    assert body["by_country"]["DE"] == {"inserted": 1, "updated": 0, "skipped": 0, "failed": 0}
    assert {doc.country for doc in fake_db.added} == {"DE", "FR", "US"}
    assert mock_embeddings.call_args_list[0].args[0] == "DE rule\nGerman content"


def test_bulk_import_row_missing_required_field_fails(client, fake_db, mock_embeddings):
    rows = _bulk_rows()
    del rows[0]["title"]
    rows[1]["content"] = "   "

    body = _post_bulk(client, rows)

    assert _counts(body) == (1, 0, 0, 2)
    assert body["rows"][0]["reason"] == "missing title"
    assert body["rows"][1]["reason"] == "missing content"
    assert [doc.title for doc in fake_db.added] == ["US rule"]


def test_bulk_import_is_idempotent(client, fake_db, mock_embeddings):
    _post_bulk(client, _bulk_rows())
    embed_calls = mock_embeddings.call_count

    body = _post_bulk(client, _bulk_rows())

    assert _counts(body) == (0, 0, 3, 0)
    assert len(fake_db.added) == 3
    assert mock_embeddings.call_count == embed_calls


def test_bulk_import_same_title_other_country_is_a_new_row(client, fake_db, mock_embeddings):
    _post_bulk(client, _bulk_rows())
    rows = _bulk_rows()
    rows[0]["country"] = "AT"

    body = _post_bulk(client, rows)

    assert _counts(body) == (1, 0, 2, 0)
    assert len(fake_db.added) == 4


def test_bulk_import_duplicate_in_file_fails(client, fake_db, mock_embeddings):
    rows = _bulk_rows()
    rows.append(dict(rows[0]))

    body = _post_bulk(client, rows)

    assert _counts(body) == (3, 0, 0, 1)
    assert body["rows"][3]["reason"] == "duplicate title and country in file"


def test_bulk_import_changed_content_updates_reembeds_and_unverifies(client, fake_db, mock_embeddings):
    _post_bulk(client, _bulk_rows())
    for doc in fake_db.added:
        doc.verified = True
    rows = _bulk_rows()
    rows[0]["content"] = "Revised German content"
    rows[1]["source_url"] = "https://example.com/fr-new"
    mock_embeddings.reset_mock()

    body = _post_bulk(client, rows)

    assert _counts(body) == (0, 2, 1, 0)
    de, fr, us = fake_db.added
    assert de.content == "Revised German content"
    assert de.verified is False
    assert fr.source_url == "https://example.com/fr-new"
    assert fr.verified is False
    assert us.verified is True  # unchanged, so its review stands
    # only the content change needs a new embedding
    assert [c.args[0] for c in mock_embeddings.call_args_list] == ["DE rule\nRevised German content"]
    assert body["rows"][0]["changed"] == ["content"]
    assert body["rows"][1]["changed"] == ["source_url"]


def test_bulk_import_never_sets_verified(client, fake_db, mock_embeddings):
    rows = _bulk_rows()
    for row in rows:
        row["verified"] = True

    _post_bulk(client, rows)

    assert len(fake_db.added) == 3
    assert all(doc.verified is False for doc in fake_db.added)


def test_bulk_import_dry_run_writes_nothing(client, fake_db, mock_embeddings):
    _post_bulk(client, _bulk_rows()[:1])
    fake_db.added[0].verified = True
    rows = _bulk_rows()
    rows[0]["content"] = "Revised German content"
    mock_embeddings.reset_mock()

    body = _post_bulk(client, rows, dry_run=True)

    assert body["dry_run"] is True
    assert _counts(body) == (2, 1, 0, 0)
    assert len(fake_db.added) == 1
    assert fake_db.added[0].content == "German content"
    assert fake_db.added[0].verified is True
    mock_embeddings.assert_not_called()


def test_bulk_import_embedding_failure_counts_as_failed(client, fake_db, mock_embeddings):
    mock_embeddings.side_effect = [list(DEFAULT_EMBEDDING), RuntimeError("boom"), list(DEFAULT_EMBEDDING)]

    body = _post_bulk(client, _bulk_rows())

    assert _counts(body) == (2, 0, 0, 1)
    assert body["rows"][1]["reason"] == "embedding failed (RuntimeError)"
    assert [doc.title for doc in fake_db.added] == ["DE rule", "US rule"]


def test_bulk_import_dry_run_requires_admin_key(client, fake_db):
    files = {"file": ("docs.json", json.dumps(_bulk_rows()).encode("utf-8"), "application/json")}
    resp = client.post("/admin/documents/bulk/dry-run", headers={"x-api-key": API_KEY}, files=files)
    assert resp.status_code == 401


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
    assert body["failed"] == 1
    assert body["rows"][0]["result"] == "failed"
    assert body["rows"][0]["reason"] == "missing content"
    assert len(fake_db.added) == 1
    assert fake_db.added[0].content == "Real content"


def test_public_search_hides_unverified_shows_verified(client, fake_db, mock_embeddings):
    verified_doc = _seed_document(fake_db, id=1, content="Verified content", verified=True)
    _seed_document(fake_db, id=2, content="Unverified content", verified=False)

    resp = client.get(
        "/documents/search?destination=UK&query=test",
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

    resp = client.get("/documents/search?destination=UK&query=test", headers={"x-api-key": API_KEY})
    assert resp.json() == []

    resp = client.patch(
        "/admin/documents/1/verified",
        headers={"x-api-key": ADMIN_API_KEY},
        json={"verified": True},
    )
    assert resp.status_code == 200
    assert resp.json()["verified"] is True

    resp = client.get("/documents/search?destination=UK&query=test", headers={"x-api-key": API_KEY})
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


def test_admin_get_document_returns_content(client, fake_db, mock_embeddings):
    _seed_document(fake_db, id=1, title="T", content="Full content", verified=True)

    resp = client.get("/admin/documents/1", headers={"x-api-key": ADMIN_API_KEY})
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == "Full content"
    assert body["title"] == "T"
    assert body["verified"] is True

    resp = client.get("/admin/documents/999", headers={"x-api-key": ADMIN_API_KEY})
    assert resp.status_code == 404


def test_update_metadata_only_keeps_verified_and_embedding(client, fake_db, mock_embeddings):
    doc = _seed_document(fake_db, id=1, country="UK", verified=True, embedding=list(FAR_EMBEDDING))

    resp = client.patch(
        "/admin/documents/1",
        headers={"x-api-key": ADMIN_API_KEY},
        json={"country": "IE", "category": "labelling"},
    )
    assert resp.status_code == 200
    assert resp.json()["country"] == "IE"
    assert resp.json()["category"] == "labelling"
    assert resp.json()["verified"] is True
    assert doc.embedding == FAR_EMBEDDING
    mock_embeddings.assert_not_called()


def test_update_content_reembeds_and_resets_verified(client, fake_db, mock_embeddings):
    doc = _seed_document(fake_db, id=1, title="Title", content="Old", verified=True)

    resp = client.patch(
        "/admin/documents/1",
        headers={"x-api-key": ADMIN_API_KEY},
        json={"content": "New content"},
    )
    assert resp.status_code == 200
    assert resp.json()["content"] == "New content"
    assert resp.json()["verified"] is False
    assert mock_embeddings.call_args.args[0] == "Title\nNew content"
    assert doc.verified is False


def test_update_source_url_resets_verified_without_reembedding(client, fake_db, mock_embeddings):
    _seed_document(fake_db, id=1, source_url="https://old.example", verified=True)

    resp = client.patch(
        "/admin/documents/1",
        headers={"x-api-key": ADMIN_API_KEY},
        json={"source_url": "https://new.example"},
    )
    assert resp.json()["verified"] is False
    mock_embeddings.assert_not_called()


def test_update_with_unchanged_values_keeps_verified(client, fake_db, mock_embeddings):
    _seed_document(fake_db, id=1, title="Title", content="Same", verified=True)

    resp = client.patch(
        "/admin/documents/1",
        headers={"x-api-key": ADMIN_API_KEY},
        json={"title": "Title", "content": "Same"},
    )
    assert resp.json()["verified"] is True
    mock_embeddings.assert_not_called()


def test_update_empty_string_clears_optional_field(client, fake_db, mock_embeddings):
    _seed_document(fake_db, id=1, category="labelling")

    resp = client.patch(
        "/admin/documents/1",
        headers={"x-api-key": ADMIN_API_KEY},
        json={"category": "  "},
    )
    assert resp.json()["category"] is None


def test_update_rejects_empty_content(client, fake_db, mock_embeddings):
    doc = _seed_document(fake_db, id=1, content="Keep me")

    for content in ("", "   ", None):
        resp = client.patch(
            "/admin/documents/1",
            headers={"x-api-key": ADMIN_API_KEY},
            json={"content": content},
        )
        assert resp.status_code == 422
    assert doc.content == "Keep me"


def test_update_missing_id_returns_404(client, fake_db, mock_embeddings):
    resp = client.patch(
        "/admin/documents/999",
        headers={"x-api-key": ADMIN_API_KEY},
        json={"country": "UK"},
    )
    assert resp.status_code == 404


def test_delete_removes_document_from_listing_and_search(client, fake_db, mock_embeddings):
    _seed_document(fake_db, id=1, verified=True)
    _seed_document(fake_db, id=2, verified=True)

    resp = client.delete("/admin/documents/1", headers={"x-api-key": ADMIN_API_KEY})
    assert resp.status_code == 204

    resp = client.get("/admin/documents", headers={"x-api-key": ADMIN_API_KEY})
    assert [d["id"] for d in resp.json()] == [2]
    resp = client.get("/documents/search?destination=UK&query=test", headers={"x-api-key": API_KEY})
    assert [r["id"] for r in resp.json()] == [2]

    resp = client.delete("/admin/documents/1", headers={"x-api-key": ADMIN_API_KEY})
    assert resp.status_code == 404


def test_document_edit_endpoints_reject_regular_api_key(client, fake_db, mock_embeddings):
    _seed_document(fake_db, id=1)

    assert client.get("/admin/documents/1", headers={"x-api-key": API_KEY}).status_code == 401
    resp = client.patch(
        "/admin/documents/1", headers={"x-api-key": API_KEY}, json={"country": "UK"}
    )
    assert resp.status_code == 401
    assert client.delete("/admin/documents/1", headers={"x-api-key": API_KEY}).status_code == 401
    assert len(fake_db.added) == 1


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
    assert resp.json()["inserted"] == 1
    assert len(fake_db.added) == 1
    assert fake_db.added[0].verified is False


# --- Relevance cutoff ---


def test_public_search_applies_default_max_distance(client, fake_db, mock_embeddings):
    _seed_document(fake_db, id=1, embedding=NEAR_EMBEDDING, verified=True)
    _seed_document(fake_db, id=2, embedding=FAR_EMBEDDING, verified=True)

    resp = client.get("/documents/search?destination=UK&query=test", headers={"x-api-key": API_KEY})

    assert resp.status_code == 200
    # The orthogonal row sits at distance 1.0, past the 0.7 default.
    assert [r["id"] for r in resp.json()] == [1]


def test_public_search_respects_explicit_max_distance(client, fake_db, mock_embeddings):
    _seed_document(fake_db, id=1, embedding=NEAR_EMBEDDING, verified=True)
    _seed_document(fake_db, id=2, embedding=FAR_EMBEDDING, verified=True)

    resp = client.get(
        "/documents/search?destination=UK&query=test&max_distance=1.5",
        headers={"x-api-key": API_KEY},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [r["id"] for r in body] == [1, 2]
    assert body[0]["distance"] == pytest.approx(0.0)
    assert body[1]["distance"] == pytest.approx(1.0)

    resp = client.get(
        "/documents/search?destination=UK&query=test&max_distance=0.0",
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


# --- Checklist: rules/uk_beverages.json + checklist.py ---

# Exact titles of the source documents in the production database, copied
# from there, not from the rules file - otherwise a typo in both places
# passes the title-match test below while the live rule is never assessed.
SDIL_SCOPE_TITLE = "Soft Drinks Industry Levy (SDIL) — scope and rates"
SDIL_EXEMPTION_TITLE = "Soft Drinks Industry Levy — small producer exemption"
FBO_TITLE = "UK Food Business Operator (FBO) address requirement"
EORI_TITLE = "EORI number required to import into the UK"
LABEL_TITLE = "Mandatory label information under UK FIC (Food Information for Consumers)"
ORIGIN_TITLE = "Rules of Origin for tariff-free UK-EU trade"
INVOICE_TITLE = "Commercial invoice requirements for UK customs clearance"

ALL_CHECKLIST_TITLES = [
    SDIL_SCOPE_TITLE,
    SDIL_EXEMPTION_TITLE,
    FBO_TITLE,
    EORI_TITLE,
    LABEL_TITLE,
    ORIGIN_TITLE,
    INVOICE_TITLE,
]


def _seed_checklist_documents(fake_db, *, skip_titles=(), unverified_titles=()):
    """Seed every document the checklist rules cite, verified by default.
    `skip_titles` omits a document entirely (as if it were never uploaded);
    `unverified_titles` seeds it but with verified=False."""
    for i, title in enumerate(ALL_CHECKLIST_TITLES, start=100):
        if title in skip_titles:
            continue
        _seed_document(
            fake_db,
            id=i,
            title=title,
            content=f"Content for {title}",
            source_url=f"https://example.com/{i}",
            verified=title not in unverified_titles,
        )


def _checklist_profile(**overrides):
    profile = dict(
        abv_percent=0.5,
        added_sugar_g_per_100ml=0.0,
        annual_volume="unknown",
        uk_importer="not_decided",
        label_uk_address="unknown",
        importer_has_eori="unknown",
        origin_proof="unknown",
        invoice_ready="unknown",
        label_elements=["english_name", "ingredients_list_with_allergens", "origin_statement"],
    )
    profile.update(overrides)
    return profile


def _evaluate_checklist(client, profile, *, admin=False, destination="UK"):
    path = "/admin/checklist/evaluate" if admin else "/checklist/evaluate"
    key = ADMIN_API_KEY if admin else API_KEY
    return client.post(path, params={"destination": destination}, headers={"x-api-key": key}, json=profile)


def _items_by_id(body):
    return {item["id"]: item for item in body["items"]}


def test_checklist_evaluate_requires_auth(client):
    resp = client.post("/checklist/evaluate", json=_checklist_profile())
    assert resp.status_code == 422  # missing header entirely


def test_checklist_evaluate_rejects_wrong_key(client):
    resp = client.post(
        "/checklist/evaluate",
        headers={"x-api-key": "wrong-key"},
        json=_checklist_profile(),
    )
    assert resp.status_code == 401


def test_admin_checklist_evaluate_rejects_regular_api_key(client, fake_db):
    resp = _evaluate_checklist(client, _checklist_profile(), admin=False)
    # sanity: the public call with the regular key succeeds...
    assert resp.status_code == 200
    # ...but the admin endpoint rejects that same regular key.
    resp = client.post(
        "/admin/checklist/evaluate",
        headers={"x-api-key": API_KEY},
        json=_checklist_profile(),
    )
    assert resp.status_code == 401


def test_checklist_out_of_scope_above_1_2_abv(client, fake_db):
    resp = _evaluate_checklist(client, _checklist_profile(abv_percent=1.3))
    assert resp.status_code == 200
    assert resp.json() == {
        "origin": "NL",
        "destination": "UK",
        "checklist_available": True,
        "message": (
            "This checker covers drinks up to 1.2% ABV. Higher-ABV drinks "
            "fall under different rules that aren't covered here."
        ),
    }


def test_checklist_in_scope_at_exactly_1_2_abv(client, fake_db):
    _seed_checklist_documents(fake_db)
    resp = _evaluate_checklist(client, _checklist_profile(abv_percent=1.2))
    assert resp.status_code == 200
    assert "items" in resp.json()


def test_checklist_sdil_below_threshold_not_applicable(client, fake_db):
    _seed_checklist_documents(fake_db)
    resp = _evaluate_checklist(client, _checklist_profile(added_sugar_g_per_100ml=4.9))
    body = resp.json()
    assert _items_by_id(body)["sdil"]["status"] == "not_applicable"


def test_checklist_sdil_lower_band_large_volume_action_needed(client, fake_db):
    _seed_checklist_documents(fake_db)
    resp = _evaluate_checklist(
        client,
        _checklist_profile(added_sugar_g_per_100ml=6.0, annual_volume="1m_or_more"),
    )
    item = _items_by_id(resp.json())["sdil"]
    assert item["status"] == "action_needed"
    assert "lower rate" in item["action"]


def test_checklist_sdil_higher_band_small_volume_confirm(client, fake_db):
    _seed_checklist_documents(fake_db)
    resp = _evaluate_checklist(
        client,
        _checklist_profile(added_sugar_g_per_100ml=8.0, annual_volume="under_1m"),
    )
    item = _items_by_id(resp.json())["sdil"]
    assert item["status"] == "confirm"
    assert "higher rate" in item["action"]


def test_checklist_fbo_no_importer_action_needed(client, fake_db):
    _seed_checklist_documents(fake_db)
    resp = _evaluate_checklist(client, _checklist_profile(uk_importer="no"))
    assert _items_by_id(resp.json())["fbo"]["status"] == "action_needed"


def test_checklist_fbo_not_decided_action_needed(client, fake_db):
    _seed_checklist_documents(fake_db)
    resp = _evaluate_checklist(client, _checklist_profile(uk_importer="not_decided"))
    assert _items_by_id(resp.json())["fbo"]["status"] == "action_needed"


def test_checklist_fbo_importer_yes_no_address_action_needed(client, fake_db):
    _seed_checklist_documents(fake_db)
    resp = _evaluate_checklist(
        client, _checklist_profile(uk_importer="yes", label_uk_address="no")
    )
    assert _items_by_id(resp.json())["fbo"]["status"] == "action_needed"


def test_checklist_fbo_importer_yes_address_unknown_confirm(client, fake_db):
    _seed_checklist_documents(fake_db)
    resp = _evaluate_checklist(
        client, _checklist_profile(uk_importer="yes", label_uk_address="unknown")
    )
    assert _items_by_id(resp.json())["fbo"]["status"] == "confirm"


def test_checklist_fbo_importer_yes_address_yes_covered(client, fake_db):
    _seed_checklist_documents(fake_db)
    resp = _evaluate_checklist(
        client, _checklist_profile(uk_importer="yes", label_uk_address="yes")
    )
    assert _items_by_id(resp.json())["fbo"]["status"] == "covered"


@pytest.mark.parametrize(
    "field, rule_id",
    [
        ("importer_has_eori", "eori"),
        ("origin_proof", "origin_proof"),
        ("invoice_ready", "invoice"),
    ],
)
@pytest.mark.parametrize(
    "value, expected_status",
    [("no", "action_needed"), ("unknown", "confirm"), ("yes", "covered")],
)
def test_checklist_simple_yes_no_unknown_rules(
    client, fake_db, field, rule_id, value, expected_status
):
    _seed_checklist_documents(fake_db)
    resp = _evaluate_checklist(client, _checklist_profile(**{field: value}))
    assert _items_by_id(resp.json())[rule_id]["status"] == expected_status


def test_checklist_label_elements_missing_english_name_action_needed(client, fake_db):
    _seed_checklist_documents(fake_db)
    resp = _evaluate_checklist(
        client,
        _checklist_profile(
            label_elements=["ingredients_list_with_allergens", "origin_statement"]
        ),
    )
    item = _items_by_id(resp.json())["label_elements"]
    assert item["status"] == "action_needed"
    assert "English name" in item["action"]


def test_checklist_label_elements_only_origin_missing_confirm(client, fake_db):
    _seed_checklist_documents(fake_db)
    resp = _evaluate_checklist(
        client,
        _checklist_profile(
            label_elements=["english_name", "ingredients_list_with_allergens"]
        ),
    )
    assert _items_by_id(resp.json())["label_elements"]["status"] == "confirm"


def test_checklist_label_elements_all_present_covered(client, fake_db):
    _seed_checklist_documents(fake_db)
    resp = _evaluate_checklist(
        client,
        _checklist_profile(
            label_elements=[
                "english_name",
                "ingredients_list_with_allergens",
                "origin_statement",
            ]
        ),
    )
    assert _items_by_id(resp.json())["label_elements"]["status"] == "covered"


def test_checklist_unknown_answers_never_produce_covered(client, fake_db):
    _seed_checklist_documents(fake_db)
    resp = _evaluate_checklist(
        client,
        _checklist_profile(
            added_sugar_g_per_100ml=6.0,
            annual_volume="unknown",
            uk_importer="yes",
            label_uk_address="unknown",
            importer_has_eori="unknown",
            origin_proof="unknown",
            invoice_ready="unknown",
        ),
    )
    items = _items_by_id(resp.json())
    # label_elements and sdil have no "unknown" input of their own, so they're
    # outside the scope of this guard; everything driven by an "unknown"
    # answer must not be "covered".
    for rule_id in ("fbo", "eori", "origin_proof", "invoice"):
        assert items[rule_id]["status"] != "covered"
        assert items[rule_id]["status"] == "confirm"


def test_checklist_headline_never_contains_forbidden_words(client, fake_db):
    _seed_checklist_documents(fake_db)
    profiles = [
        _checklist_profile(),
        _checklist_profile(
            added_sugar_g_per_100ml=10,
            annual_volume="1m_or_more",
            uk_importer="yes",
            label_uk_address="yes",
            importer_has_eori="yes",
            origin_proof="yes",
            invoice_ready="yes",
        ),
        _checklist_profile(
            uk_importer="no",
            label_uk_address="no",
            importer_has_eori="no",
            origin_proof="no",
            invoice_ready="no",
        ),
    ]
    for profile in profiles:
        resp = _evaluate_checklist(client, profile)
        headline = resp.json()["headline"].lower()
        for forbidden in ("compliant", "certified", "ready"):
            assert forbidden not in headline


def test_checklist_rules_file_source_titles_match_seeded_documents():
    # Titles are matched exactly, so a typo or a different dash in either
    # place silently turns a rule into not_assessed.
    seeded = set(ALL_CHECKLIST_TITLES)
    cited = set()
    for rule in CHECKLIST_RULES["rules"]:
        for title in [*rule["sources"], rule["primary_source"]]:
            assert title in seeded, f"rule {rule['id']!r} cites unknown title {title!r}"
            cited.add(title)
    assert seeded == cited, f"seeded titles no rule cites: {sorted(seeded - cited)}"


def test_checklist_rule_skipped_when_source_unverified_public_vs_admin(client, fake_db):
    _seed_checklist_documents(fake_db, unverified_titles=[EORI_TITLE])
    profile = _checklist_profile(importer_has_eori="no")

    public_resp = _evaluate_checklist(client, profile, admin=False)
    public_ids = _items_by_id(public_resp.json())
    assert public_ids["eori"]["status"] == "not_assessed"
    assert public_ids["eori"]["not_assessed_reason"] == "source_unverified"
    # the other rules, with verified sources, still come through
    assert public_ids["fbo"]["status"] == "action_needed"

    admin_resp = _evaluate_checklist(client, profile, admin=True)
    admin_ids = _items_by_id(admin_resp.json())
    assert admin_ids["eori"]["status"] == "action_needed"
    assert admin_ids["eori"]["reviewed"] is False
    assert admin_ids["fbo"]["reviewed"] is True


def test_checklist_skipped_rule_appears_as_not_assessed(client, fake_db):
    _seed_checklist_documents(fake_db, unverified_titles=[EORI_TITLE])
    resp = _evaluate_checklist(client, _checklist_profile())
    body = resp.json()

    eori = _items_by_id(body)["eori"]
    assert eori["status"] == "not_assessed"
    assert eori["title"] == EORI_TITLE
    assert eori["action"] == "Not assessed: the source entry is awaiting review."
    # nothing from the unreviewed document reaches the public report
    assert eori["source_url"] is None
    assert eori["source_excerpt"] == ""

    assert body["counts"]["not_assessed"] == 1
    assert body["checks_total"] == len(CHECKLIST_RULES["rules"])
    assert body["checks_assessed"] == body["checks_total"] - 1
    assert body["items"][-1]["id"] == "eori"


def test_checklist_rule_not_assessed_when_source_document_missing(client, fake_db):
    # Not just unverified - never uploaded, or its title doesn't match. The
    # admin endpoint can't evaluate it either, and says which title is missing
    # so it isn't confused with an entry that is merely awaiting review.
    _seed_checklist_documents(fake_db, skip_titles=[EORI_TITLE])
    profile = _checklist_profile()

    public = _items_by_id(_evaluate_checklist(client, profile).json())["eori"]
    assert public["status"] == "not_assessed"
    assert public["not_assessed_reason"] == "source_missing"
    assert public["action"] == "Not assessed: the source entry is awaiting review."

    admin = _items_by_id(_evaluate_checklist(client, profile, admin=True).json())["eori"]
    assert admin["status"] == "not_assessed"
    assert admin["not_assessed_reason"] == "source_missing"
    assert f'"{EORI_TITLE}"' in admin["action"]
    assert "title mismatch" in admin["action"]


def test_checklist_sdil_not_assessed_when_one_of_two_sources_missing(client, fake_db):
    _seed_checklist_documents(fake_db, skip_titles=[SDIL_EXEMPTION_TITLE])
    profile = _checklist_profile(added_sugar_g_per_100ml=6.0)

    for admin in (False, True):
        resp = _evaluate_checklist(client, profile, admin=admin)
        sdil = _items_by_id(resp.json())["sdil"]
        assert sdil["status"] == "not_assessed"
        assert sdil["not_assessed_reason"] == "source_missing"

    admin_sdil = _items_by_id(_evaluate_checklist(client, profile, admin=True).json())["sdil"]
    assert SDIL_EXEMPTION_TITLE in admin_sdil["action"]
    assert SDIL_SCOPE_TITLE not in admin_sdil["action"]


def test_checklist_items_ordered_action_needed_confirm_covered_not_applicable(
    client, fake_db
):
    _seed_checklist_documents(fake_db)
    resp = _evaluate_checklist(
        client,
        _checklist_profile(
            uk_importer="yes",
            label_uk_address="no",
            origin_proof="no",
            invoice_ready="no",
            importer_has_eori="unknown",
            annual_volume="under_1m",
            added_sugar_g_per_100ml=6.0,
        ),
    )
    statuses = [item["status"] for item in resp.json()["items"]]
    order = {"action_needed": 0, "confirm": 1, "covered": 2, "not_applicable": 3, "not_assessed": 4}
    assert statuses == sorted(statuses, key=lambda s: order[s])


def test_checklist_scenario_from_brief(client, fake_db):
    _seed_checklist_documents(fake_db)
    profile = _checklist_profile(
        abv_percent=0.4,
        added_sugar_g_per_100ml=6.5,
        annual_volume="under_1m",
        uk_importer="yes",
        label_uk_address="no",
        importer_has_eori="unknown",
        origin_proof="no",
        invoice_ready="no",
    )
    resp = _evaluate_checklist(client, profile)
    assert resp.status_code == 200
    body = resp.json()

    assert body["headline"] == (
        "Based on your answers: 3 actions needed, 2 points to confirm, "
        "in the 6 checks assessed."
    )
    assert body["checks_assessed"] == 6
    assert body["checks_total"] == 6

    items = _items_by_id(body)
    assert items["fbo"]["status"] == "action_needed"
    assert items["origin_proof"]["status"] == "action_needed"
    assert items["invoice"]["status"] == "action_needed"
    assert items["sdil"]["status"] == "confirm"
    assert items["eori"]["status"] == "confirm"
    assert items["label_elements"]["status"] == "covered"

    assert [item["id"] for item in body["items"]] == [
        "fbo",
        "origin_proof",
        "invoice",
        "sdil",
        "eori",
        "label_elements",
    ]
    assert all(item["reviewed"] is True for item in body["items"])


def test_checklist_report_includes_origin_and_destination(client, fake_db):
    _seed_checklist_documents(fake_db)
    body = _evaluate_checklist(client, _checklist_profile()).json()
    assert body["origin"] == "NL"
    assert body["destination"] == "UK"
    assert body["checklist_available"] is True


def test_checklist_destination_without_rules_does_not_run_uk_rules(client, fake_db, monkeypatch):
    _seed_checklist_documents(fake_db)
    import checklist as checklist_module

    def must_not_run(*args, **kwargs):
        raise AssertionError("UK rules must not run for another destination")

    monkeypatch.setattr(checklist_module, "evaluate", must_not_run)
    monkeypatch.setattr(checklist_module, "is_out_of_scope", must_not_run)

    for admin in (False, True):
        resp = _evaluate_checklist(client, _checklist_profile(), admin=admin, destination="DE")
        assert resp.status_code == 200
        assert resp.json() == {
            "origin": "NL",
            "destination": "DE",
            "checklist_available": False,
            "message": "Checklist not available for this destination yet. You can still search the requirements.",
        }


def test_checklist_requires_known_destination_and_origin(client, fake_db):
    profile = _checklist_profile()
    headers = {"x-api-key": API_KEY}
    assert client.post("/checklist/evaluate", headers=headers, json=profile).status_code == 422
    for params in ({"destination": "EU"}, {"destination": "XX"}, {"destination": "UK", "origin": "BE"}):
        resp = client.post("/checklist/evaluate", params=params, headers=headers, json=profile)
        assert resp.status_code == 422, params


def test_checklist_wording_never_contains_forbidden_words_for_any_destination(client, fake_db):
    _seed_checklist_documents(fake_db)
    for destination in ("UK", "DE", "FR", "AE", "US", "CN"):
        for abv in (0.5, 5.0):
            text = json.dumps(
                _evaluate_checklist(client, _checklist_profile(abv_percent=abv), destination=destination).json()
            ).lower()
            for forbidden in ("compliant", "certified", "ready"):
                assert forbidden not in text, (destination, abv, forbidden)


# --- Countries and destination-scoped search ---


def _countries(client):
    resp = client.get("/countries", headers={"x-api-key": API_KEY})
    assert resp.status_code == 200
    return resp.json()


def test_countries_requires_api_key(client):
    assert client.get("/countries").status_code == 422
    assert client.get("/countries", headers={"x-api-key": "wrong-key"}).status_code == 401


def test_countries_lists_only_destinations_with_verified_rows(client, fake_db):
    _seed_document(fake_db, id=1, country="UK", verified=True)
    _seed_document(fake_db, id=2, country="UK", verified=True)
    _seed_document(fake_db, id=3, country="DE", verified=True)
    _seed_document(fake_db, id=4, country="FR", verified=False)
    _seed_document(fake_db, id=5, country="EU", verified=True)
    _seed_document(fake_db, id=6, country=None, verified=True)
    _seed_document(fake_db, id=7, country="XX", verified=True)

    body = _countries(client)

    assert body["origins"] == [{"code": "NL", "name": "The Netherlands", "default": True}]
    assert body["destinations"] == [
        {"code": "DE", "name": "Germany", "document_count": 1, "checklist_available": False, "eu_member": True},
        {"code": "UK", "name": "United Kingdom", "document_count": 2, "checklist_available": True, "eu_member": False},
    ]


def test_countries_reflects_verification_without_restart(client, fake_db):
    _seed_document(fake_db, id=1, country="UK", verified=True)
    _seed_document(fake_db, id=2, country="DE", verified=False)
    assert [d["code"] for d in _countries(client)["destinations"]] == ["UK"]

    resp = client.patch("/admin/documents/2/verified", headers={"x-api-key": ADMIN_API_KEY}, json={"verified": True})
    assert resp.status_code == 200

    assert [d["code"] for d in _countries(client)["destinations"]] == ["DE", "UK"]


def _seed_country_rows(fake_db):
    for i, (country, verified) in enumerate(
        [("UK", True), ("DE", True), ("FR", True), ("EU", True), ("AE", True),
         ("US", True), ("CN", True), (None, True), ("DE", False), ("EU", False)],
        start=1,
    ):
        _seed_document(fake_db, id=i, country=country, verified=verified, content=f"row {i}")


def _search_countries(client, destination, *, admin=False, **params):
    path = "/admin/documents/search" if admin else "/documents/search"
    key = ADMIN_API_KEY if admin else API_KEY
    resp = client.get(
        path, params={"query": "test", "destination": destination, "limit": 50, **params},
        headers={"x-api-key": key},
    )
    assert resp.status_code == 200, resp.text
    return sorted((r["country"], r["id"]) for r in resp.json())


def test_search_eu_member_destination_includes_eu_rows(client, fake_db, mock_embeddings):
    _seed_country_rows(fake_db)
    assert _search_countries(client, "DE") == [("DE", 2), ("EU", 4)]
    assert _search_countries(client, "FR") == [("EU", 4), ("FR", 3)]


def test_search_non_eu_destinations_exclude_eu_rows(client, fake_db, mock_embeddings):
    _seed_country_rows(fake_db)
    assert _search_countries(client, "UK") == [("UK", 1)]
    assert _search_countries(client, "AE") == [("AE", 5)]
    assert _search_countries(client, "US") == [("US", 6)]
    assert _search_countries(client, "CN") == [("CN", 7)]


def test_search_returns_title_and_country(client, fake_db, mock_embeddings):
    _seed_document(fake_db, id=1, country="DE", title="German rule", verified=True)
    body = client.get(
        "/documents/search", params={"query": "test", "destination": "DE"}, headers={"x-api-key": API_KEY}
    ).json()
    assert body[0]["title"] == "German rule"
    assert body[0]["country"] == "DE"


def test_search_requires_destination(client, fake_db, mock_embeddings):
    resp = client.get("/documents/search", params={"query": "test"}, headers={"x-api-key": API_KEY})
    assert resp.status_code == 422
    mock_embeddings.assert_not_called()


def test_search_rejects_unknown_codes(client, fake_db, mock_embeddings):
    for params in ({"destination": "XX"}, {"destination": "EU"}, {"destination": "de"},
                   {"destination": "DE", "origin": "BE"}):
        resp = client.get(
            "/documents/search", params={"query": "test", **params}, headers={"x-api-key": API_KEY}
        )
        assert resp.status_code == 422, params
    mock_embeddings.assert_not_called()


def test_search_origin_defaults_to_nl_and_accepts_nl(client, fake_db, mock_embeddings):
    _seed_country_rows(fake_db)
    assert _search_countries(client, "UK", origin="NL") == _search_countries(client, "UK")


def test_admin_search_destination_filter_includes_unverified_but_not_null(client, fake_db, mock_embeddings):
    _seed_country_rows(fake_db)
    assert _search_countries(client, "DE", admin=True) == [("DE", 2), ("DE", 9), ("EU", 4), ("EU", 10)]


def test_admin_search_without_destination_stays_unfiltered(client, fake_db, mock_embeddings):
    _seed_country_rows(fake_db)
    resp = client.get(
        "/admin/documents/search", params={"query": "test", "limit": 50}, headers={"x-api-key": ADMIN_API_KEY}
    )
    assert len(resp.json()) == 10
    resp = client.get("/admin/documents", headers={"x-api-key": ADMIN_API_KEY})
    assert len(resp.json()) == 10
