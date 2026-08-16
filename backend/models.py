from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    content = Column(String, nullable=False)
    embedding = Column(Vector(1536))  # 1536 = OpenAI/most common embedding size; adjust to whatever model you use
    created_at = Column(DateTime(timezone=True), server_default=func.now())
