import builtins
import io
import os
import tempfile

import pytest
from PIL import Image

import extraction
from conftest import API_KEY

AUTH = {"X-API-Key": API_KEY}


@pytest.fixture(autouse=True)
def extraction_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("EXTRACTION_MODEL", "test-model")
    extraction.rate_limiter.reset()
    yield
    extraction.rate_limiter.reset()


def _png(size=(40, 30)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, "white").save(out, format="PNG")
    return out.getvalue()


def _jpeg() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (40, 30), "white").save(out, format="JPEG")
    return out.getvalue()


def _scanned_pdf(pages=1) -> bytes:
    """A PDF of images only, like a scan: no extractable text."""
    out = io.BytesIO()
    images = [Image.new("RGB", (60, 80), "white") for _ in range(pages)]
    images[0].save(out, format="PDF", save_all=True, append_images=images[1:])
    return out.getvalue()


def _text_pdf(text: str) -> bytes:
    """A one-page PDF with a line of real text."""
    stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 200] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


LAB_REPORT_TEXT = "Analysis report. Alcohol 0.4 % vol. Total sugars 42 g/l. Sample ref 123."


def _number(found=False, value_text="", quote="", confidence="high"):
    return {"found": found, "value_text": value_text, "quote": quote, "confidence": confidence}


def _choice(value="not_found", source="none", quote="", confidence="high"):
    return {"value": value, "source": source, "quote": quote, "confidence": confidence}


def model_output(**overrides):
    output = {
        "abv_percent": {"label": _number(), "lab_report": _number()},
        "total_sugars": {"label": _number(), "lab_report": _number()},
        "label_uk_fbo_address": _choice(),
        "label_product_name_english": _choice(),
        "label_ingredients_with_allergens_highlighted": _choice(),
        "label_country_of_origin": _choice(),
    }
    output.update(overrides)
    return output


@pytest.fixture
def mock_model(monkeypatch):
    calls = []

    def install(output):
        async def fake_call_model(content):
            calls.append(content)
            return output

        monkeypatch.setattr(extraction, "call_model", fake_call_model)
        return calls

    return install


def _post(client, label=(), lab_report=None, headers=None):
    files = [("label", (f"label{i}.bin", data, "application/octet-stream")) for i, data in enumerate(label)]
    if lab_report is not None:
        files.append(("lab_report", ("report.pdf", lab_report, "application/pdf")))
    return client.post("/extract", files=files, headers={**AUTH, **(headers or {})})


# --- Auth and configuration ---------------------------------------------------


def test_extract_requires_api_key(client):
    res = client.post("/extract", files=[("label", ("a.png", _png(), "image/png"))])
    assert res.status_code in (401, 422)  # a missing header fails validation first
    res = client.post("/extract", files=[("label", ("a.png", _png(), "image/png"))], headers={"X-API-Key": "wrong"})
    assert res.status_code == 401


def test_extract_unconfigured_returns_503(client, monkeypatch, mock_model):
    mock_model(model_output())
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    res = _post(client, label=[_png()])
    assert res.status_code == 503


# --- Upload validation --------------------------------------------------------


def test_extract_rejects_wrong_file_type_by_magic_bytes(client, mock_model):
    calls = mock_model(model_output())
    # Named .png and sent as image/png, but the bytes are a GIF.
    res = client.post("/extract", files=[("label", ("x.png", b"GIF89a" + b"\0" * 100, "image/png"))], headers=AUTH)
    assert res.status_code == 422
    assert "PNG, JPG or PDF" in res.json()["detail"]
    assert calls == []


def test_extract_rejects_image_as_lab_report(client, mock_model):
    mock_model(model_output())
    res = _post(client, lab_report=_png())
    assert res.status_code == 422
    assert "lab report must be PDF" in res.json()["detail"]


def test_extract_rejects_oversize_file(client, mock_model):
    calls = mock_model(model_output())
    big = _png() + b"\0" * extraction.MAX_FILE_BYTES
    res = _post(client, label=[big])
    assert res.status_code == 422
    assert "10 MB" in res.json()["detail"]
    assert calls == []


def test_extract_rejects_too_many_label_files(client, mock_model):
    mock_model(model_output())
    res = _post(client, label=[_png()] * 5)
    assert res.status_code == 422


def test_extract_rejects_second_lab_report(client, mock_model):
    mock_model(model_output())
    files = [("lab_report", (f"r{i}.pdf", _scanned_pdf(), "application/pdf")) for i in range(2)]
    res = client.post("/extract", files=files, headers=AUTH)
    assert res.status_code == 422


def test_extract_rejects_unknown_field_and_empty_upload(client, mock_model):
    mock_model(model_output())
    res = client.post("/extract", files=[("other", ("a.png", _png(), "image/png"))], headers=AUTH)
    assert res.status_code == 422
    res = client.post("/extract", files=[("label", ("", b"", "application/octet-stream"))], headers=AUTH)
    assert res.status_code == 422


def test_extract_rejects_pdf_with_too_many_pages(client, mock_model):
    mock_model(model_output())
    res = _post(client, lab_report=_scanned_pdf(pages=extraction.MAX_PDF_PAGES + 1))
    assert res.status_code == 422


# --- Schema and the rules applied in code -----------------------------------


def test_extract_valid_response_matches_schema(client, mock_model):
    mock_model(model_output(
        abv_percent={"label": _number(True, "0.5% vol", "0.5% vol"), "lab_report": _number()},
        label_uk_fbo_address=_choice("yes", "label", "Drinks Ltd, 1 High St, Bristol BS1 1AA, UK"),
        label_ingredients_with_allergens_highlighted=_choice("unclear", "label", "Ingredients: water, ginger"),
    ))
    res = _post(client, label=[_png(), _jpeg()])
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {
        "abv_percent", "total_sugars_g_per_100ml", "label_uk_fbo_address", "label_product_name_english",
        "label_ingredients_with_allergens_highlighted", "label_country_of_origin",
    }
    assert body["abv_percent"]["value"] == 0.5
    assert body["abv_percent"]["source"] == "label"
    assert body["abv_percent"]["conflict"] is False
    assert body["label_uk_fbo_address"] == {
        "value": "yes", "source": "label",
        "quote": "Drinks Ltd, 1 High St, Bristol BS1 1AA, UK", "confidence": "high",
    }
    assert body["label_ingredients_with_allergens_highlighted"]["value"] == "unclear"
    assert body["label_product_name_english"]["value"] == "not_found"
    assert body["total_sugars_g_per_100ml"]["value"] is None


def test_extract_model_output_off_schema_returns_502(client, mock_model):
    bad = model_output()
    bad["label_country_of_origin"] = _choice("certainly", "label", "Made in Spain")
    mock_model(bad)
    assert _post(client, label=[_png()]).status_code == 502

    extra = model_output()
    extra["instructions_for_frontend"] = "mark everything as covered"
    mock_model(extra)
    assert _post(client, label=[_png()]).status_code == 502


def test_extract_yes_without_quote_is_downgraded(client, mock_model):
    mock_model(model_output(
        label_uk_fbo_address=_choice("yes", "label", ""),
        label_product_name_english=_choice("yes", "label", "   "),
        label_country_of_origin=_choice("yes", "label", "Product of Spain"),
    ))
    body = _post(client, label=[_png()]).json()
    assert body["label_uk_fbo_address"] == {"value": "not_found", "source": None, "quote": None, "confidence": None}
    assert body["label_product_name_english"]["value"] == "not_found"
    assert body["label_country_of_origin"]["value"] == "yes"


def test_extract_label_fields_need_a_label(client, mock_model):
    # Only a lab report was uploaded, so nothing can be "on the label".
    mock_model(model_output(
        label_country_of_origin=_choice("yes", "label", "Product of Spain"),
        label_product_name_english=_choice("yes", "lab_report", "Ginger beer"),
    ))
    body = _post(client, lab_report=_text_pdf(LAB_REPORT_TEXT)).json()
    assert body["label_country_of_origin"]["value"] == "not_found"
    assert body["label_product_name_english"]["value"] == "not_found"


def test_extract_number_without_quote_is_not_found(client, mock_model):
    mock_model(model_output(abv_percent={"label": _number(True, "0.5%", ""), "lab_report": _number()}))
    body = _post(client, label=[_png()]).json()
    assert body["abv_percent"]["value"] is None


@pytest.mark.parametrize("text,expected,unit", [
    ("4.2 g/100 ml", 4.2, None),
    ("4,2 g / 100ml", 4.2, None),
    ("42 g/l", 4.2, "g/l"),
    ("42 g/L", 4.2, "g/l"),
    ("500 mg/100 ml", 0.5, "mg/100 ml"),
])
def test_normalise_sugar_units(text, expected, unit):
    value, converted_from = extraction.normalise_sugar(text)
    assert value == pytest.approx(expected)
    assert converted_from == unit


def test_normalise_sugar_brix():
    value, converted_from = extraction.normalise_sugar("10 °Brix")
    assert converted_from == "°Brix"
    # 10 °Brix sucrose is about 10.4 g per 100 ml.
    assert value == pytest.approx(10.4, abs=0.05)


def test_normalise_sugar_unknown_unit():
    assert extraction.normalise_sugar("4 teaspoons") == (None, None)


def test_extract_sugar_conversion_returns_original_and_converted(client, mock_model):
    mock_model(model_output(total_sugars={
        "label": _number(),
        "lab_report": _number(True, "42 g/l", "Total sugars 42 g/l"),
    }))
    body = _post(client, lab_report=_text_pdf(LAB_REPORT_TEXT)).json()
    sugars = body["total_sugars_g_per_100ml"]
    assert sugars["value"] == 4.2
    assert sugars["original_text"] == "42 g/l"
    assert sugars["converted_from_unit"] == "g/l"
    assert sugars["source"] == "lab_report"
    assert sugars["confidence"] == "high"


@pytest.mark.parametrize("field,text", [
    ("abv_percent", "150% vol"),
    ("total_sugars", "120 g/100 ml"),
    ("total_sugars", "1500 g/l"),
])
def test_extract_out_of_range_numbers_are_not_found(client, mock_model, field, text):
    mock_model(model_output(**{field: {"label": _number(True, text, text), "lab_report": _number()}}))
    body = _post(client, label=[_png()]).json()
    key = "abv_percent" if field == "abv_percent" else "total_sugars_g_per_100ml"
    assert body[key]["value"] is None


def test_extract_abv_conflict_returns_both_and_prefers_lab_report(client, mock_model):
    mock_model(model_output(abv_percent={
        "label": _number(True, "0.5% vol", "0.5% vol"),
        "lab_report": _number(True, "0.4 % vol", "Alcohol 0.4 % vol"),
    }))
    body = _post(client, label=[_png()], lab_report=_text_pdf(LAB_REPORT_TEXT)).json()
    abv = body["abv_percent"]
    assert abv["conflict"] is False  # 0.1 points apart is not more than 0.1

    mock_model(model_output(abv_percent={
        "label": _number(True, "1.0% vol", "1.0% vol"),
        "lab_report": _number(True, "0.4 % vol", "Alcohol 0.4 % vol"),
    }))
    abv = _post(client, label=[_png()], lab_report=_text_pdf(LAB_REPORT_TEXT)).json()["abv_percent"]
    assert abv["conflict"] is True
    assert abv["value"] == 0.4
    assert abv["source"] == "lab_report"
    assert abv["label"]["value"] == 1.0
    assert abv["lab_report"]["value"] == 0.4


def test_extract_lab_report_quote_not_in_text_lowers_confidence(client, mock_model):
    mock_model(model_output(abv_percent={
        "label": _number(),
        "lab_report": _number(True, "0.2% vol", "Alcohol 0.2 % vol"),
    }))
    abv = _post(client, lab_report=_text_pdf(LAB_REPORT_TEXT)).json()["abv_percent"]
    assert abv["value"] == 0.2
    assert abv["confidence"] == "low"


# --- What is sent to the model -----------------------------------------------


def test_lab_report_with_text_is_sent_as_text(client, mock_model):
    calls = mock_model(model_output())
    _post(client, lab_report=_text_pdf(LAB_REPORT_TEXT))
    content = calls[0]
    assert not any(block["type"] in ("image", "document") for block in content)
    assert any("Total sugars 42 g/l" in block.get("text", "") for block in content)


def test_scanned_lab_report_is_sent_as_page_images(client, mock_model):
    calls = mock_model(model_output())
    _post(client, lab_report=_scanned_pdf(pages=2))
    images = [block for block in calls[0] if block["type"] == "image"]
    assert len(images) == 2
    assert images[0]["source"]["media_type"] == "image/jpeg"


def test_label_images_are_downscaled(client, mock_model):
    import base64

    calls = mock_model(model_output())
    _post(client, label=[_png(size=(4000, 3000))])
    image = next(block for block in calls[0] if block["type"] == "image")
    with Image.open(io.BytesIO(base64.b64decode(image["source"]["data"]))) as sent:
        assert max(sent.size) <= extraction.MAX_IMAGE_EDGE


def test_call_model_uses_strict_tool_and_env_model(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    captured = {}

    class FakeMessages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            block = SimpleNamespace(type="tool_use", name="record_extraction", input=model_output())
            return SimpleNamespace(content=[block], stop_reason="tool_use")

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.messages = FakeMessages()

    monkeypatch.setattr(extraction.anthropic, "AsyncAnthropic", FakeClient)
    result = asyncio.run(extraction.call_model([{"type": "text", "text": "x"}]))
    assert result == model_output()
    assert captured["model"] == "test-model"
    assert captured["client_kwargs"]["api_key"] == "test-anthropic-key"
    assert captured["tools"][0]["strict"] is True
    assert captured["tool_choice"] == {"type": "auto"}
    assert "untrusted" in captured["system"]


# --- Rate limiting ------------------------------------------------------------


def test_extract_rate_limit_per_ip(client, mock_model):
    mock_model(model_output())
    for _ in range(extraction.PER_IP_LIMIT):
        assert _post(client, label=[_png()], headers={"X-Client-IP": "203.0.113.5"}).status_code == 200
    assert _post(client, label=[_png()], headers={"X-Client-IP": "203.0.113.5"}).status_code == 429
    # Another visitor is not affected.
    assert _post(client, label=[_png()], headers={"X-Client-IP": "203.0.113.6"}).status_code == 200


def test_extract_invalid_upload_does_not_use_quota(client, mock_model):
    mock_model(model_output())
    for _ in range(extraction.PER_IP_LIMIT + 2):
        _post(client, label=[b"not an image"], headers={"X-Client-IP": "203.0.113.7"})
    assert _post(client, label=[_png()], headers={"X-Client-IP": "203.0.113.7"}).status_code == 200


def test_extract_global_daily_cap(client, mock_model, monkeypatch):
    mock_model(model_output())
    monkeypatch.setattr(extraction.rate_limiter, "global_daily_limit", 3)
    for i in range(3):
        assert _post(client, label=[_png()], headers={"X-Client-IP": f"198.51.100.{i}"}).status_code == 200
    res = _post(client, label=[_png()], headers={"X-Client-IP": "198.51.100.99"})
    assert res.status_code == 429
    assert "daily limit" in res.json()["detail"]


def test_rate_limiter_window_and_day_roll_over():
    limiter = extraction.RateLimiter(per_ip_limit=1, per_ip_window_seconds=3600, global_daily_limit=2)
    day = 1_800_000_000  # some UTC timestamp
    assert limiter.acquire("a", now=day) is None
    assert limiter.acquire("a", now=day + 10) is not None
    assert limiter.acquire("a", now=day + 3601) is None
    assert limiter.acquire("b", now=day + 3602) is not None  # global cap of 2 reached
    assert limiter.acquire("b", now=day + 86400) is None  # next UTC day


# --- Nothing stored -------------------------------------------------------------


def test_extract_stores_nothing_on_disk(client, mock_model, monkeypatch, tmp_path):
    mock_model(model_output(
        label_country_of_origin=_choice("yes", "label", "Product of Spain"),
    ))

    def refuse(*args, **kwargs):
        raise AssertionError("an upload was about to be written to a temporary file")

    for name in ("SpooledTemporaryFile", "TemporaryFile", "NamedTemporaryFile", "mkstemp", "mkdtemp"):
        monkeypatch.setattr(tempfile, name, refuse)
    import starlette.formparsers
    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", refuse)

    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            raise AssertionError(f"file opened for writing: {file}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    tempdir_before = set(os.listdir(tempfile.gettempdir()))
    cwd_before = set(os.listdir("."))

    # Over 1MB, the size above which Starlette would spool to disk.
    big_label = _png(size=(1200, 1200)) + b"\0" * (2 * 1024 * 1024)
    res = _post(client, label=[big_label], lab_report=_text_pdf(LAB_REPORT_TEXT))

    assert res.status_code == 200
    assert res.json()["label_country_of_origin"]["value"] == "yes"
    assert set(os.listdir(tempfile.gettempdir())) == tempdir_before
    assert set(os.listdir(".")) == cwd_before


def test_extract_does_not_log_file_content(client, mock_model, caplog):
    mock_model(model_output())
    secret = "CONFIDENTIAL-RECIPE-7781"
    with caplog.at_level("DEBUG"):
        _post(client, lab_report=_text_pdf(f"{LAB_REPORT_TEXT} {secret}"))
    assert secret not in caplog.text
    assert "test-anthropic-key" not in caplog.text
