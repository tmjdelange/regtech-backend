from pydantic import BaseModel
from datetime import datetime
from typing import List, Literal, Optional, Union

import countries

# Literal over the allowlists, so an unknown code is a 422 with the allowed
# values listed, and the OpenAPI schema shows them as an enum.
DestinationCode = Literal[tuple(countries.DESTINATIONS)]
OriginCode = Literal[tuple(countries.ORIGINS)]

class DocumentCreate(BaseModel):
    content: str
    title: Optional[str] = None
    country: Optional[str] = None
    category: Optional[str] = None
    source_url: Optional[str] = None

class DocumentOut(BaseModel):
    id: int
    content: str
    title: Optional[str] = None
    country: Optional[str] = None
    category: Optional[str] = None
    source_url: Optional[str] = None

    class Config:
        from_attributes = True

class SearchResult(BaseModel):
    id: int
    title: Optional[str] = None
    country: Optional[str] = None
    content: str
    distance: float

class AdminSearchResult(SearchResult):
    verified: bool

class CountryOrigin(BaseModel):
    code: str
    name: str
    default: bool

class CountryDestination(BaseModel):
    code: str
    name: str
    document_count: int
    checklist_available: bool
    eu_member: bool

class CountriesResponse(BaseModel):
    origins: List[CountryOrigin]
    destinations: List[CountryDestination]

class AdminDocumentOut(BaseModel):
    id: int
    title: Optional[str] = None
    country: Optional[str] = None
    category: Optional[str] = None
    source_url: Optional[str] = None
    verified: bool
    created_at: datetime

    class Config:
        from_attributes = True

class AdminDocumentDetail(AdminDocumentOut):
    content: str

class DocumentUpdate(BaseModel):
    """Partial update: only the fields present in the request are changed.
    Sending null (or an empty string) clears an optional field."""
    title: Optional[str] = None
    content: Optional[str] = None
    country: Optional[str] = None
    category: Optional[str] = None
    source_url: Optional[str] = None

class VerifiedUpdate(BaseModel):
    verified: bool

YesNoUnknown = Literal["yes", "no", "unknown"]

class ChecklistProfile(BaseModel):
    abv_percent: float
    added_sugar_g_per_100ml: float
    annual_volume: Literal["under_1m", "1m_or_more", "unknown"]
    uk_importer: Literal["yes", "no", "not_decided"]
    label_uk_address: YesNoUnknown
    importer_has_eori: YesNoUnknown
    origin_proof: YesNoUnknown
    invoice_ready: YesNoUnknown
    label_elements: List[
        Literal["english_name", "ingredients_list_with_allergens", "origin_statement"]
    ]

class ChecklistItem(BaseModel):
    id: str
    title: str
    status: Literal["action_needed", "confirm", "covered", "not_applicable", "not_assessed"]
    action: str
    source_url: Optional[str] = None
    source_excerpt: str
    reviewed: bool
    not_assessed_reason: Optional[Literal["source_missing", "source_unverified"]] = None

class ChecklistCounts(BaseModel):
    action_needed: int
    confirm: int
    covered: int
    not_applicable: int
    not_assessed: int

class ChecklistRoute(BaseModel):
    origin: str
    destination: str

class ChecklistResult(ChecklistRoute):
    checklist_available: Literal[True] = True
    headline: str
    counts: ChecklistCounts
    items: List[ChecklistItem]
    checks_assessed: int
    checks_total: int
    scope: str
    disclaimer: str

class ChecklistOutOfScope(ChecklistRoute):
    checklist_available: Literal[True] = True
    message: str

class ChecklistUnavailable(ChecklistRoute):
    checklist_available: Literal[False] = False
    message: str

ChecklistResponse = Union[ChecklistResult, ChecklistOutOfScope, ChecklistUnavailable]
