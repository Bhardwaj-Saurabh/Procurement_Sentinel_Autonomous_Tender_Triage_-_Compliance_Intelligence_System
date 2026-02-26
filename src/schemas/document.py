"""Document schemas for RAG indexing."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Type of document."""

    PDF = "pdf"
    HTML = "html"
    DOCX = "docx"
    TEXT = "text"
    UNKNOWN = "unknown"


class DocumentMetadata(BaseModel):
    """Metadata about the document source."""

    source_url: Optional[str] = None
    source_file: Optional[str] = None
    tender_id: str
    document_title: Optional[str] = None
    document_type: DocumentType = DocumentType.UNKNOWN
    page_count: Optional[int] = None
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Document(BaseModel):
    """Document for RAG processing and indexing."""

    id: str
    content: str
    metadata: DocumentMetadata

    @property
    def char_count(self) -> int:
        """Get character count of content."""
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.content.split())


class DocumentChunk(BaseModel):
    """A chunk of a document for embedding and indexing."""

    id: str
    document_id: str
    tender_id: str
    content: str
    chunk_index: int
    total_chunks: int

    # Metadata for citations
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None

    # Embedding (filled by indexer)
    embedding: Optional[list[float]] = None

    @property
    def char_count(self) -> int:
        """Get character count of chunk."""
        return len(self.content)
