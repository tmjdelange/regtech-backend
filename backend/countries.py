"""Country codes used to scope public search and the checklist.

Codes match what documents.country holds, so the UK is "UK" rather than
ISO's "GB". "EU" rows are shared rows that apply to every EU member state;
it is never a destination in its own right.
"""

EU = "EU"

# All 27 member states, so a new EU destination only needs adding to
# DESTINATIONS below. Only DE and FR have documents today.
EU_MEMBER_STATES = frozenset({
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GR", "HR", "HU",
    "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK",
})

# Destinations a user may pick; anything else is rejected with a 422.
DESTINATIONS = {
    "UK": "United Kingdom",
    "DE": "Germany",
    "FR": "France",
    "AE": "United Arab Emirates",
    "US": "United States",
    "CN": "China",
}

ORIGINS = {"NL": "The Netherlands"}
DEFAULT_ORIGIN = "NL"


def search_countries(destination: str) -> list[str]:
    """documents.country values that apply when exporting to `destination`."""
    if destination in EU_MEMBER_STATES:
        return [destination, EU]
    return [destination]
