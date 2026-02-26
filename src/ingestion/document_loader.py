"""Document loader for downloading and extracting text from tender documents."""

import hashlib
import httpx
import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional
from datetime import datetime

from src.schemas.document import Document, DocumentMetadata, DocumentType


class DocumentLoader:
    """
    Loads and extracts text from tender documents.

    Supports:
    - PDF files (via PyMuPDF)
    - HTML content (basic text extraction)
    - Plain text
    - Local files and remote URLs
    """

    def __init__(self, timeout: float = 30.0):
        """
        Initialize the document loader.

        Args:
            timeout: HTTP request timeout in seconds
        """
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)

    def load_from_url(
        self,
        url: str,
        tender_id: str,
        document_title: Optional[str] = None,
    ) -> Document:
        """
        Download and extract text from a document URL.

        Args:
            url: URL of the document to download
            tender_id: ID of the associated tender
            document_title: Optional title for the document

        Returns:
            Document with extracted text content
        """
        response = self.client.get(url)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        doc_type = self._detect_type_from_content_type(content_type)

        # If can't detect from content-type, try URL extension
        if doc_type == DocumentType.UNKNOWN:
            doc_type = self._detect_type_from_url(url)

        text_content = self._extract_text(response.content, doc_type)
        page_count = self._get_page_count(response.content, doc_type)

        doc_id = self._generate_id(tender_id, url)

        metadata = DocumentMetadata(
            source_url=url,
            tender_id=tender_id,
            document_title=document_title,
            document_type=doc_type,
            page_count=page_count,
        )

        return Document(
            id=doc_id,
            content=text_content,
            metadata=metadata,
        )

    def load_from_file(
        self,
        file_path: str | Path,
        tender_id: str,
        document_title: Optional[str] = None,
    ) -> Document:
        """
        Load and extract text from a local file.

        Args:
            file_path: Path to the local file
            tender_id: ID of the associated tender
            document_title: Optional title for the document

        Returns:
            Document with extracted text content
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = path.read_bytes()
        doc_type = self._detect_type_from_extension(path.suffix)

        text_content = self._extract_text(content, doc_type)
        page_count = self._get_page_count(content, doc_type)

        doc_id = self._generate_id(tender_id, str(path))

        metadata = DocumentMetadata(
            source_file=str(path),
            tender_id=tender_id,
            document_title=document_title or path.stem,
            document_type=doc_type,
            page_count=page_count,
        )

        return Document(
            id=doc_id,
            content=text_content,
            metadata=metadata,
        )

    def load_from_text(
        self,
        text: str,
        tender_id: str,
        document_title: Optional[str] = None,
        source_url: Optional[str] = None,
    ) -> Document:
        """
        Create a Document from raw text content.

        Args:
            text: Raw text content
            tender_id: ID of the associated tender
            document_title: Optional title for the document
            source_url: Optional source URL

        Returns:
            Document with the provided text content
        """
        doc_id = self._generate_id(tender_id, document_title or "text")

        metadata = DocumentMetadata(
            source_url=source_url,
            tender_id=tender_id,
            document_title=document_title,
            document_type=DocumentType.TEXT,
            page_count=1,
        )

        return Document(
            id=doc_id,
            content=text,
            metadata=metadata,
        )

    def load_from_html(
        self,
        html: str,
        tender_id: str,
        document_title: Optional[str] = None,
        source_url: Optional[str] = None,
    ) -> Document:
        """
        Extract text from HTML content.

        Args:
            html: HTML content string
            tender_id: ID of the associated tender
            document_title: Optional title for the document
            source_url: Optional source URL

        Returns:
            Document with extracted text content
        """
        text_content = self._extract_text_from_html(html)
        doc_id = self._generate_id(tender_id, document_title or "html")

        metadata = DocumentMetadata(
            source_url=source_url,
            tender_id=tender_id,
            document_title=document_title,
            document_type=DocumentType.HTML,
            page_count=1,
        )

        return Document(
            id=doc_id,
            content=text_content,
            metadata=metadata,
        )

    def _extract_text(self, content: bytes, doc_type: DocumentType) -> str:
        """Extract text based on document type."""
        if doc_type == DocumentType.PDF:
            return self._extract_text_from_pdf(content)
        elif doc_type == DocumentType.HTML:
            return self._extract_text_from_html(content.decode("utf-8", errors="ignore"))
        elif doc_type == DocumentType.TEXT:
            return content.decode("utf-8", errors="ignore")
        else:
            # Try to decode as text
            return content.decode("utf-8", errors="ignore")

    def _extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF using PyMuPDF."""
        text_parts = []

        with fitz.open(stream=content, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")

        return "\n\n".join(text_parts)

    def _extract_text_from_html(self, html: str) -> str:
        """Extract text from HTML by removing tags."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Get text
        text = soup.get_text(separator="\n")

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]

        return "\n".join(lines)

    def _get_page_count(self, content: bytes, doc_type: DocumentType) -> Optional[int]:
        """Get page count for PDF documents."""
        if doc_type != DocumentType.PDF:
            return None

        try:
            with fitz.open(stream=content, filetype="pdf") as doc:
                return len(doc)
        except Exception:
            return None

    def _detect_type_from_content_type(self, content_type: str) -> DocumentType:
        """Detect document type from HTTP Content-Type header."""
        if "pdf" in content_type:
            return DocumentType.PDF
        elif "html" in content_type:
            return DocumentType.HTML
        elif "text" in content_type:
            return DocumentType.TEXT
        elif "word" in content_type or "docx" in content_type:
            return DocumentType.DOCX
        return DocumentType.UNKNOWN

    def _detect_type_from_url(self, url: str) -> DocumentType:
        """Detect document type from URL extension."""
        url_lower = url.lower()
        if url_lower.endswith(".pdf"):
            return DocumentType.PDF
        elif url_lower.endswith((".html", ".htm")):
            return DocumentType.HTML
        elif url_lower.endswith((".docx", ".doc")):
            return DocumentType.DOCX
        elif url_lower.endswith(".txt"):
            return DocumentType.TEXT
        return DocumentType.UNKNOWN

    def _detect_type_from_extension(self, extension: str) -> DocumentType:
        """Detect document type from file extension."""
        ext = extension.lower().lstrip(".")
        mapping = {
            "pdf": DocumentType.PDF,
            "html": DocumentType.HTML,
            "htm": DocumentType.HTML,
            "txt": DocumentType.TEXT,
            "docx": DocumentType.DOCX,
            "doc": DocumentType.DOCX,
        }
        return mapping.get(ext, DocumentType.UNKNOWN)

    def _generate_id(self, tender_id: str, source: str) -> str:
        """Generate a unique document ID."""
        content = f"{tender_id}:{source}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
