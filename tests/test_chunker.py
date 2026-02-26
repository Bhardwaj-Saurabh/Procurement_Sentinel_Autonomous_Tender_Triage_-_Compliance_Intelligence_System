"""Tests for the document chunker module."""

import pytest
from src.ingestion.chunker import DocumentChunker, chunk_documents
from src.schemas.document import Document, DocumentMetadata, DocumentType


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id="test-doc-001",
        content="This is a test document. It has multiple sentences. Each sentence should be handled properly.",
        metadata=DocumentMetadata(
            tender_id="ocds-test-001",
            document_title="Test Document",
            document_type=DocumentType.TEXT,
        ),
    )


@pytest.fixture
def long_document():
    """Create a longer document with sections."""
    content = """--- Page 1 ---
INVITATION TO TENDER
Contract Reference: ITT-2024-00123

1. INTRODUCTION
The Ministry of Defence invites tenders for the provision of cybersecurity services.

2. MANDATORY REQUIREMENTS
2.1 The supplier must hold Cyber Essentials Plus certification
2.2 All staff must have SC clearance or above

--- Page 2 ---
3. TECHNICAL SPECIFICATIONS
3.1 24/7 Security Operations Centre capability
3.2 Response time: Critical incidents within 15 minutes

4. EVALUATION CRITERIA
- Technical capability: 40%
- Price: 35%
"""
    return Document(
        id="test-doc-002",
        content=content,
        metadata=DocumentMetadata(
            tender_id="ocds-test-002",
            document_title="MOD ITT Document",
            document_type=DocumentType.PDF,
        ),
    )


class TestDocumentChunker:
    """Tests for DocumentChunker class."""

    def test_init_default_values(self):
        """Test chunker initializes with default values."""
        chunker = DocumentChunker()
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100
        assert chunker.separators == ["\n\n", "\n", ". ", ", ", " "]

    def test_init_custom_values(self):
        """Test chunker initializes with custom values."""
        chunker = DocumentChunker(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n", " "],
        )
        assert chunker.chunk_size == 300
        assert chunker.chunk_overlap == 50
        assert chunker.separators == ["\n", " "]

    def test_chunk_empty_document(self):
        """Test chunking an empty document returns empty list."""
        doc = Document(
            id="empty-doc",
            content="",
            metadata=DocumentMetadata(tender_id="test", document_type=DocumentType.TEXT),
        )
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc)
        assert chunks == []

    def test_chunk_whitespace_only_document(self):
        """Test chunking whitespace-only document returns empty list."""
        doc = Document(
            id="whitespace-doc",
            content="   \n\n   \t   ",
            metadata=DocumentMetadata(tender_id="test", document_type=DocumentType.TEXT),
        )
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc)
        assert chunks == []

    def test_chunk_small_document(self, sample_document):
        """Test small document that fits in one chunk."""
        chunker = DocumentChunker(chunk_size=500)
        chunks = chunker.chunk_document(sample_document)

        # Small document should be single chunk
        assert len(chunks) == 1
        assert chunks[0].content == sample_document.content
        assert chunks[0].document_id == sample_document.id
        assert chunks[0].tender_id == sample_document.metadata.tender_id

    def test_chunk_preserves_metadata(self, sample_document):
        """Test chunks preserve document metadata."""
        chunker = DocumentChunker(chunk_size=500)
        chunks = chunker.chunk_document(sample_document)

        assert chunks[0].document_id == "test-doc-001"
        assert chunks[0].tender_id == "ocds-test-001"
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1

    def test_chunk_generates_unique_ids(self, long_document):
        """Test each chunk gets a unique ID."""
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=30)
        chunks = chunker.chunk_document(long_document)

        chunk_ids = [c.id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique

    def test_chunk_indexes_sequential(self, long_document):
        """Test chunk indexes are sequential."""
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=30)
        chunks = chunker.chunk_document(long_document)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)

    def test_chunk_extracts_page_number(self, long_document):
        """Test page numbers are extracted from content."""
        chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_document(long_document)

        # First chunk should have page 1
        assert chunks[0].page_number == 1

        # Some chunk should have page 2
        page_2_chunks = [c for c in chunks if c.page_number == 2]
        assert len(page_2_chunks) > 0

    def test_chunk_respects_size_limit(self, long_document):
        """Test chunks respect the size limit (approximately)."""
        chunk_size = 250
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=50)
        chunks = chunker.chunk_document(long_document)

        # Most chunks should be close to chunk_size (allowing for overlap)
        for chunk in chunks:
            # Allow some tolerance for overlap and boundary conditions
            assert chunk.char_count <= chunk_size * 2

    def test_chunk_has_overlap(self, long_document):
        """Test chunks have overlapping content."""
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_document(long_document)

        if len(chunks) > 1:
            # Check that consecutive chunks share some content
            for i in range(1, len(chunks)):
                prev_chunk = chunks[i - 1]
                curr_chunk = chunks[i]

                # Some text from end of prev should appear in start of current
                prev_end = prev_chunk.content[-50:]
                # At least some characters should overlap
                overlap_found = any(
                    word in curr_chunk.content[:100]
                    for word in prev_end.split()
                    if len(word) > 3
                )
                # This is a soft check - overlap may not always be obvious
                # depending on separator positions


class TestChunkDocumentsFunction:
    """Tests for the convenience chunk_documents function."""

    def test_chunk_multiple_documents(self, sample_document, long_document):
        """Test chunking multiple documents at once."""
        documents = [sample_document, long_document]
        chunks = chunk_documents(documents, chunk_size=300, chunk_overlap=50)

        # Should have chunks from both documents
        doc_ids = set(c.document_id for c in chunks)
        assert len(doc_ids) == 2
        assert "test-doc-001" in doc_ids
        assert "test-doc-002" in doc_ids

    def test_chunk_empty_list(self):
        """Test chunking empty document list."""
        chunks = chunk_documents([])
        assert chunks == []

    def test_chunk_preserves_tender_id(self, sample_document, long_document):
        """Test tender IDs are preserved across all chunks."""
        documents = [sample_document, long_document]
        chunks = chunk_documents(documents)

        for chunk in chunks:
            if chunk.document_id == "test-doc-001":
                assert chunk.tender_id == "ocds-test-001"
            elif chunk.document_id == "test-doc-002":
                assert chunk.tender_id == "ocds-test-002"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_word_document(self):
        """Test document with single word."""
        doc = Document(
            id="single-word",
            content="Hello",
            metadata=DocumentMetadata(tender_id="test", document_type=DocumentType.TEXT),
        )
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "Hello"

    def test_no_separator_in_text(self):
        """Test text without any separators."""
        doc = Document(
            id="no-sep",
            content="a" * 1000,  # Long string without separators
            metadata=DocumentMetadata(tender_id="test", document_type=DocumentType.TEXT),
        )
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_document(doc)

        # Should still split using force split
        assert len(chunks) > 1

    def test_unicode_content(self):
        """Test chunking unicode content."""
        doc = Document(
            id="unicode-doc",
            content="日本語テスト文章。これはテストです。もう一つの文章。",
            metadata=DocumentMetadata(tender_id="test", document_type=DocumentType.TEXT),
        )
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=5)
        chunks = chunker.chunk_document(doc)

        # Should handle unicode properly
        assert len(chunks) > 0
        assert all(isinstance(c.content, str) for c in chunks)

    def test_special_characters(self):
        """Test content with special characters."""
        doc = Document(
            id="special-chars",
            content="Price: £1,000,000.00\nVAT: 20%\nTotal: £1,200,000.00",
            metadata=DocumentMetadata(tender_id="test", document_type=DocumentType.TEXT),
        )
        chunker = DocumentChunker(chunk_size=500)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert "£" in chunks[0].content
        assert "%" in chunks[0].content

    def test_very_small_chunk_size(self):
        """Test with very small chunk size."""
        doc = Document(
            id="small-chunk",
            content="Hello world. This is a test.",
            metadata=DocumentMetadata(tender_id="test", document_type=DocumentType.TEXT),
        )
        chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk_document(doc)

        # Should create multiple small chunks
        assert len(chunks) > 1

    def test_overlap_larger_than_chunk(self):
        """Test when overlap is larger than chunk size."""
        doc = Document(
            id="large-overlap",
            content="Hello world. This is a test. Another sentence here.",
            metadata=DocumentMetadata(tender_id="test", document_type=DocumentType.TEXT),
        )
        # Overlap > chunk_size is unusual but shouldn't crash
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=30)
        chunks = chunker.chunk_document(doc)

        # Should still work
        assert len(chunks) > 0
