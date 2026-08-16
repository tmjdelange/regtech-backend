from pydantic import BaseModel
from typing import List

class DocumentCreate(BaseModel):
    content: str

class DocumentOut(BaseModel):
    id: int
    content: str

    class Config:
        from_attributes = True

class SearchResult(BaseModel):
    id: int
    content: str
    distance: float
