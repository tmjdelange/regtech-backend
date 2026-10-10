"""Pure evaluation logic for the UK beverages import checklist.

No I/O: the rules config (parsed from rules/uk_beverages.json) and the
lookup of which cited documents exist and whether they're verified are both
handed in by the caller. That keeps this module trivially unit-testable and
keeps real file/database access where the rest of the app already does it
(main.py).
"""

ACTION_NEEDED = "action_needed"
CONFIRM = "confirm"
COVERED = "covered"
NOT_APPLICABLE = "not_applicable"

_STATUS_ORDER = {ACTION_NEEDED: 0, CONFIRM: 1, COVERED: 2, NOT_APPLICABLE: 3}

_ANNUAL_VOLUME_DESCRIPTIONS = {
    "under_1m": "under 1 million litres",
    "1m_or_more": "1 million litres or more",
    "unknown": "an unknown volume",
}


def _eval_sdil(rule, profile):
    sugar = profile["added_sugar_g_per_100ml"]
    thresholds = rule["thresholds"]
    lower_min = thresholds["lower_band_min_g_per_100ml"]
    higher_min = thresholds["higher_band_min_g_per_100ml"]
    text = rule["text"]

    if sugar < lower_min:
        return NOT_APPLICABLE, text["not_applicable"].format(
            lower_band_min_g_per_100ml=lower_min
        )

    band = text["band_higher"] if sugar >= higher_min else text["band_lower"]
    annual_volume_desc = _ANNUAL_VOLUME_DESCRIPTIONS[profile["annual_volume"]]

    if profile["annual_volume"] == "1m_or_more":
        status, template = ACTION_NEEDED, text["action_needed"]
    else:  # under_1m or unknown
        status, template = CONFIRM, text["confirm"]

    return status, template.format(band=band, sugar=sugar, annual_volume_desc=annual_volume_desc)


def _eval_fbo(rule, profile):
    text = rule["text"]
    uk_importer = profile["uk_importer"]

    if uk_importer in ("no", "not_decided"):
        return ACTION_NEEDED, text["action_needed_no_importer"]

    label_uk_address = profile["label_uk_address"]
    if label_uk_address == "no":
        return ACTION_NEEDED, text["action_needed_no_address"]
    if label_uk_address == "unknown":
        return CONFIRM, text["confirm_address_unknown"]
    return COVERED, text["covered"]


def _eval_yes_no_unknown(rule, profile, field):
    text = rule["text"]
    value = profile[field]
    if value == "no":
        return ACTION_NEEDED, text["action_needed"]
    if value == "unknown":
        return CONFIRM, text["confirm"]
    return COVERED, text["covered"]


def _eval_eori(rule, profile):
    return _eval_yes_no_unknown(rule, profile, "importer_has_eori")


def _eval_origin_proof(rule, profile):
    return _eval_yes_no_unknown(rule, profile, "origin_proof")


def _eval_invoice(rule, profile):
    return _eval_yes_no_unknown(rule, profile, "invoice_ready")


def _eval_label_elements(rule, profile):
    text = rule["text"]
    labels = rule["element_labels"]
    present = set(profile["label_elements"])

    required_missing = [
        key for key in ("english_name", "ingredients_list_with_allergens") if key not in present
    ]
    if required_missing:
        missing_desc = ", ".join(labels[key] for key in required_missing)
        return ACTION_NEEDED, text["action_needed"].format(missing=missing_desc)

    if "origin_statement" not in present:
        return CONFIRM, text["confirm"]

    return COVERED, text["covered"]


_EVALUATORS = {
    "sdil": _eval_sdil,
    "fbo": _eval_fbo,
    "eori": _eval_eori,
    "label_elements": _eval_label_elements,
    "origin_proof": _eval_origin_proof,
    "invoice": _eval_invoice,
}


def is_out_of_scope(config, profile):
    return profile["abv_percent"] > config["scope_max_abv_percent"]


def out_of_scope_response(config):
    return {"message": config["scope_stop_message"]}


def evaluate(config, profile, documents, *, include_unverified):
    """Evaluate every rule in `config["rules"]` against `profile`.

    `documents` maps an exact document title to
    {"source_url": ..., "content": ..., "verified": ...}. A rule is skipped
    entirely if any document it cites is missing from this dict. Otherwise,
    when `include_unverified` is False (the public endpoint), the rule is
    also skipped unless every cited document is verified; when True (the
    admin endpoint), it's included regardless, with `reviewed` reflecting
    whether every cited document was in fact verified.

    Caller is expected to have already checked `is_out_of_scope`.
    """
    items = []

    for rule in config["rules"]:
        sources = rule["sources"]
        if any(title not in documents for title in sources):
            continue

        all_verified = all(documents[title]["verified"] for title in sources)
        if not include_unverified and not all_verified:
            continue

        status, action = _EVALUATORS[rule["id"]](rule, profile)

        primary_doc = documents[rule["primary_source"]]
        items.append(
            {
                "id": rule["id"],
                "title": rule["primary_source"],
                "status": status,
                "action": action,
                "source_url": primary_doc["source_url"],
                "source_excerpt": primary_doc["content"],
                "reviewed": all_verified,
            }
        )

    items.sort(key=lambda item: _STATUS_ORDER[item["status"]])

    counts = {status: 0 for status in (ACTION_NEEDED, CONFIRM, COVERED, NOT_APPLICABLE)}
    for item in items:
        counts[item["status"]] += 1

    headline = (
        f"Based on your answers: {counts[ACTION_NEEDED]} actions needed, "
        f"{counts[CONFIRM]} points to confirm, in the {counts[COVERED]} checks covered."
    )

    return {
        "headline": headline,
        "counts": counts,
        "items": items,
        "checks_covered": counts[COVERED],
        "checks_total": len(items),
        "scope": config["scope_description"],
        "disclaimer": config["disclaimer"],
    }
