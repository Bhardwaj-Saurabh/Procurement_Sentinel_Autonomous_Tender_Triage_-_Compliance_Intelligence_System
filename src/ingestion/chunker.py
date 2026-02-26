"""Document chunker for RAG indexing with sentence-aware splitting."""

import hashlib
import re
from typing import Optional

from src.schemas.document import Document, DocumentChunk


class DocumentChunker:
    """
    Sentence-aware text splitter for tender documents.

    Splits text by sentences while respecting paragraph boundaries.
    Never breaks mid-sentence or mid-word.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap_sentences: int = 1,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size for each chunk in characters
            overlap_sentences: Number of sentences to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences

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

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using regex.

        Handles common abbreviations and edge cases.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Pattern to split on sentence boundaries
        # Matches: . ! ? followed by space and capital letter, or end of string
        # Avoids splitting on common abbreviations like Mr. Mrs. Dr. etc.

        # First, protect common abbreviations
        protected = text
        abbreviations = [
            r'Mr\.', r'Mrs\.', r'Ms\.', r'Dr\.', r'Prof\.',
            r'Inc\.', r'Ltd\.', r'Co\.', r'Corp\.',
            r'vs\.', r'etc\.', r'e\.g\.', r'i\.e\.',
            r'No\.', r'Vol\.', r'Rev\.', r'Ed\.',
        ]

        # Replace abbreviation periods with placeholder
        for abbr in abbreviations:
            protected = re.sub(abbr, abbr.replace(r'\.', '<PERIOD>'), protected)

        # Split on sentence boundaries
        # Match period/question/exclamation followed by space(s) and uppercase
        # or followed by newline
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n', protected)

        # Restore periods in abbreviations
        sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences]

        # Filter empty strings
        return [s for s in sentences if s.strip()]

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """
        Split text into paragraphs.

        Args:
            text: Text to split

        Returns:
            List of paragraphs
        """
        # Split on double newlines or multiple newlines
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_text(self, text: str) -> list[str]:
        """
        Split text into chunks respecting sentence and paragraph boundaries.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        # First split into paragraphs
        paragraphs = self._split_into_paragraphs(text)

        # Build list of all sentences with paragraph markers
        all_sentences = []
        for para in paragraphs:
            sentences = self._split_into_sentences(para)
            if sentences:
                all_sentences.extend(sentences)
                # Add paragraph break marker (will be converted to newline)
                all_sentences.append("<PARA_BREAK>")

        # Remove trailing paragraph break
        if all_sentences and all_sentences[-1] == "<PARA_BREAK>":
            all_sentences.pop()

        # Group sentences into chunks
        chunks = []
        current_sentences = []
        current_length = 0

        for sentence in all_sentences:
            if sentence == "<PARA_BREAK>":
                # Don't count para breaks toward length, but include them
                if current_sentences:
                    current_sentences.append(sentence)
                continue

            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_sentences:
                # Save current chunk
                chunk_text = self._join_sentences(current_sentences)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                overlap_start = max(0, len(current_sentences) - self.overlap_sentences)
                # Filter out para breaks from overlap
                overlap_sentences = [
                    s for s in current_sentences[overlap_start:]
                    if s != "<PARA_BREAK>"
                ]
                current_sentences = overlap_sentences.copy()
                current_length = sum(len(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_length += sentence_length

        # Don't forget the last chunk
        if current_sentences:
            chunk_text = self._join_sentences(current_sentences)
            chunks.append(chunk_text)

        return chunks

    def _join_sentences(self, sentences: list[str]) -> str:
        """
        Join sentences back into text.

        Args:
            sentences: List of sentences (may include <PARA_BREAK> markers)

        Returns:
            Joined text
        """
        result_parts = []
        current_para = []

        for s in sentences:
            if s == "<PARA_BREAK>":
                if current_para:
                    result_parts.append(" ".join(current_para))
                    current_para = []
            else:
                current_para.append(s)

        if current_para:
            result_parts.append(" ".join(current_para))

        return "\n\n".join(result_parts)

    def _extract_page_number(self, chunk_text: str) -> Optional[int]:
        """
        Extract page number from chunk if it contains page marker.

        Args:
            chunk_text: Text of the chunk

        Returns:
            Page number if found, None otherwise
        """
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
    overlap_sentences: int = 1,
) -> list[DocumentChunk]:
    """
    Convenience function to chunk multiple documents.

    Args:
        documents: List of documents to chunk
        chunk_size: Target chunk size
        overlap_sentences: Number of sentences to overlap

    Returns:
        List of all chunks from all documents
    """
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        overlap_sentences=overlap_sentences,
    )

    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    return all_chunks
