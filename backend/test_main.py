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


# --- Checklist: rules/uk_beverages.json + checklist.py ---

SDIL_SCOPE_TITLE = "Soft Drinks Industry Levy — scope and rates"
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


def _evaluate_checklist(client, profile, *, admin=False):
    path = "/admin/checklist/evaluate" if admin else "/checklist/evaluate"
    key = ADMIN_API_KEY if admin else API_KEY
    return client.post(path, headers={"x-api-key": key}, json=profile)


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
        "message": (
            "This checker covers drinks up to 1.2% ABV. Higher-ABV drinks "
            "fall under different rules that aren't covered here."
        )
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


def test_checklist_rule_skipped_when_source_unverified_public_vs_admin(client, fake_db):
    _seed_checklist_documents(fake_db, unverified_titles=[EORI_TITLE])
    profile = _checklist_profile(importer_has_eori="no")

    public_resp = _evaluate_checklist(client, profile, admin=False)
    public_ids = _items_by_id(public_resp.json())
    assert "eori" not in public_ids
    # the other rules, with verified sources, still come through
    assert "fbo" in public_ids

    admin_resp = _evaluate_checklist(client, profile, admin=True)
    admin_ids = _items_by_id(admin_resp.json())
    assert "eori" in admin_ids
    assert admin_ids["eori"]["reviewed"] is False
    assert admin_ids["fbo"]["reviewed"] is True


def test_checklist_rule_skipped_entirely_when_source_document_missing(client, fake_db):
    # Not just unverified - never uploaded at all. Must be skipped even by
    # the admin endpoint, since there's no excerpt/url to show for it.
    _seed_checklist_documents(fake_db, skip_titles=[EORI_TITLE])
    profile = _checklist_profile()

    for admin in (False, True):
        resp = _evaluate_checklist(client, profile, admin=admin)
        assert "eori" not in _items_by_id(resp.json())


def test_checklist_sdil_skipped_when_one_of_two_sources_missing(client, fake_db):
    _seed_checklist_documents(fake_db, skip_titles=[SDIL_EXEMPTION_TITLE])
    profile = _checklist_profile(added_sugar_g_per_100ml=6.0)

    for admin in (False, True):
        resp = _evaluate_checklist(client, profile, admin=admin)
        assert "sdil" not in _items_by_id(resp.json())


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
    order = {"action_needed": 0, "confirm": 1, "covered": 2, "not_applicable": 3}
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
        "in the 1 checks covered."
    )

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
