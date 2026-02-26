"""Document chunker for RAG indexing."""

import hashlib
from typing import Optional

from src.schemas.document import Document, DocumentChunk


class DocumentChunker:
    """
    Recursive character text splitter for tender documents.

    Splits text by logical units (paragraphs → lines → sentences → words)
    while maintaining overlap for context continuity.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[list[str]] = None,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            separators: List of separators to split on, in order of preference
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", ", ", " "]

    def chunk_document(self, document: Document) -> list[DocumentChunk]:
        """
        Split a document into chunks for indexing.

        Args:
            document: Document to chunk

        Returns:
            List of DocumentChunk objects ready for embedding
        """
        text = document.content
        if not text.strip():
            return []

        # Split text into raw chunks
        raw_chunks = self._split_text(text)

        # Convert to DocumentChunk objects
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = self._generate_chunk_id(document.id, i)

            # Try to extract page number from chunk content
            page_number = self._extract_page_number(chunk_text)

            chunk = DocumentChunk(
                id=chunk_id,
                document_id=document.id,
                tender_id=document.metadata.tender_id,
                content=chunk_text,
                chunk_index=i,
                total_chunks=len(raw_chunks),
                source_file=document.metadata.source_file,
                page_number=page_number,
            )
            chunks.append(chunk)

        return chunks

    def _split_text(self, text: str) -> list[str]:
        """
        Recursively split text using separators.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text, trying each separator in order.

        Args:
            text: Text to split
            separators: Remaining separators to try

        Returns:
            List of text chunks
        """
        if not text.strip():
            return []

        # If text is small enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        # No more separators - force split at chunk_size
        if not separators:
            return self._force_split(text)

        # Try current separator
        separator = separators[0]
        remaining_separators = separators[1:]

        if separator not in text:
            # Separator not found, try next one
            return self._recursive_split(text, remaining_separators)

        # Split by current separator
        parts = text.split(separator)

        # Merge parts into chunks
        chunks = []
        current_chunk = ""

        for part in parts:
            part_with_sep = part + separator if part != parts[-1] else part

            # If adding this part exceeds chunk size
            if len(current_chunk) + len(part_with_sep) > self.chunk_size:
                # Save current chunk if not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # If single part is too large, recursively split it
                if len(part_with_sep) > self.chunk_size:
                    sub_chunks = self._recursive_split(part_with_sep, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part_with_sep
            else:
                current_chunk += part_with_sep

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Add overlap between chunks
        return self._add_overlap(chunks)

    def _force_split(self, text: str) -> list[str]:
        """
        Force split text at chunk_size when no separators work.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.chunk_overlap if end < len(text) else end

        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """
        Add overlap between chunks for context continuity.

        Args:
            chunks: List of chunks without overlap

        Returns:
            List of chunks with overlap added
        """
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks

        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk: add suffix from next chunk
                overlap_text = chunks[i + 1][:self.chunk_overlap] if len(chunks) > 1 else ""
                result.append(chunk)
            elif i == len(chunks) - 1:
                # Last chunk: add prefix from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                result.append(overlap_text + " " + chunk)
            else:
                # Middle chunks: add prefix from previous
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                result.append(overlap_text + " " + chunk)

        return result

    def _extract_page_number(self, chunk_text: str) -> Optional[int]:
        """
        Extract page number from chunk if it contains page marker.

        Args:
            chunk_text: Text of the chunk

        Returns:
            Page number if found, None otherwise
        """
        import re
        match = re.search(r"--- Page (\d+) ---", chunk_text)
        if match:
            return int(match.group(1))
        return None

    def _generate_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """
        Generate a unique chunk ID.

        Args:
            document_id: Parent document ID
            chunk_index: Index of this chunk

        Returns:
            Unique chunk ID
        """
        content = f"{document_id}:{chunk_index}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:
    """
    Convenience function to chunk multiple documents.

    Args:
        documents: List of documents to chunk
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        List of all chunks from all documents
    """
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    return all_chunks
