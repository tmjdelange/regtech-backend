from pydantic import BaseModel
from datetime import datetime
from typing import List, Literal, Optional, Union

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
    content: str
    distance: float

class AdminSearchResult(BaseModel):
    id: int
    content: str
    distance: float
    verified: bool

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
    status: Literal["action_needed", "confirm", "covered", "not_applicable"]
    action: str
    source_url: Optional[str] = None
    source_excerpt: str
    reviewed: bool

class ChecklistCounts(BaseModel):
    action_needed: int
    confirm: int
    covered: int
    not_applicable: int

class ChecklistResult(BaseModel):
    headline: str
    counts: ChecklistCounts
    items: List[ChecklistItem]
    checks_covered: int
    checks_total: int
    scope: str
    disclaimer: str

class ChecklistOutOfScope(BaseModel):
    message: str

ChecklistResponse = Union[ChecklistResult, ChecklistOutOfScope]
