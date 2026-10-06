from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

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
