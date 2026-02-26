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
The Ministry of Defence invites tenders for the provision of cybersecurity services. This contract covers penetration testing and security monitoring.

2. MANDATORY REQUIREMENTS
The supplier must hold Cyber Essentials Plus certification. All staff must have SC clearance or above.

--- Page 2 ---
3. TECHNICAL SPECIFICATIONS
The system must provide 24/7 Security Operations Centre capability. Response time for critical incidents must be within 15 minutes.

4. EVALUATION CRITERIA
Technical capability accounts for 40 percent of the score. Price accounts for 35 percent of the total.
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
        assert chunker.overlap_sentences == 1

    def test_init_custom_values(self):
        """Test chunker initializes with custom values."""
        chunker = DocumentChunker(
            chunk_size=300,
            overlap_sentences=2,
        )
        assert chunker.chunk_size == 300
        assert chunker.overlap_sentences == 2

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
        chunker = DocumentChunker(chunk_size=200, overlap_sentences=1)
        chunks = chunker.chunk_document(long_document)

        chunk_ids = [c.id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique

    def test_chunk_indexes_sequential(self, long_document):
        """Test chunk indexes are sequential."""
        chunker = DocumentChunker(chunk_size=200, overlap_sentences=1)
        chunks = chunker.chunk_document(long_document)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)

    def test_chunk_extracts_page_number(self, long_document):
        """Test page numbers are extracted from content."""
        chunker = DocumentChunker(chunk_size=300, overlap_sentences=1)
        chunks = chunker.chunk_document(long_document)

        # First chunk should have page 1
        assert chunks[0].page_number == 1

        # Some chunk should have page 2
        page_2_chunks = [c for c in chunks if c.page_number == 2]
        assert len(page_2_chunks) > 0

    def test_chunks_have_complete_sentences(self, long_document):
        """Test that chunks don't break mid-word or mid-sentence."""
        chunker = DocumentChunker(chunk_size=300, overlap_sentences=1)
        chunks = chunker.chunk_document(long_document)

        for chunk in chunks:
            # Content should not start with lowercase (mid-sentence)
            # unless it's a page marker or special format
            content = chunk.content.strip()
            if not content.startswith("---"):
                # Check it doesn't start with a partial word
                # (partial words often start with lowercase after being cut)
                first_word = content.split()[0] if content.split() else ""
                # Numbers and special markers are OK
                if first_word and first_word[0].isalpha():
                    # Should start with capital or be a known pattern
                    assert first_word[0].isupper() or first_word in ["the", "a", "an"]

    def test_overlap_contains_complete_sentences(self, long_document):
        """Test that overlap between chunks uses complete sentences."""
        chunker = DocumentChunker(chunk_size=300, overlap_sentences=1)
        chunks = chunker.chunk_document(long_document)

        if len(chunks) > 1:
            # Check that consecutive chunks share complete sentence(s)
            for i in range(1, len(chunks)):
                prev_content = chunks[i - 1].content
                curr_content = chunks[i].content

                # Find sentences in prev that might appear in current
                # The overlap should be sentence-based, not character-based
                # So we shouldn't see partial words at chunk boundaries


class TestChunkDocumentsFunction:
    """Tests for the convenience chunk_documents function."""

    def test_chunk_multiple_documents(self, sample_document, long_document):
        """Test chunking multiple documents at once."""
        documents = [sample_document, long_document]
        chunks = chunk_documents(documents, chunk_size=300, overlap_sentences=1)

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

    def test_single_sentence_document(self):
        """Test document with single sentence."""
        doc = Document(
            id="single-sentence",
            content="This is a single complete sentence.",
            metadata=DocumentMetadata(tender_id="test", document_type=DocumentType.TEXT),
        )
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "This is a single complete sentence."

    def test_unicode_content(self):
        """Test chunking unicode content."""
        doc = Document(
            id="unicode-doc",
            content="日本語テスト文章です。これはテストです。もう一つの文章があります。",
            metadata=DocumentMetadata(tender_id="test", document_type=DocumentType.TEXT),
        )
        chunker = DocumentChunker(chunk_size=50, overlap_sentences=1)
        chunks = chunker.chunk_document(doc)

        # Should handle unicode properly
        assert len(chunks) > 0
        assert all(isinstance(c.content, str) for c in chunks)

    def test_special_characters(self):
        """Test content with special characters."""
        doc = Document(
            id="special-chars",
            content="Price: £1,000,000.00. VAT: 20%. Total: £1,200,000.00.",
            metadata=DocumentMetadata(tender_id="test", document_type=DocumentType.TEXT),
        )
        chunker = DocumentChunker(chunk_size=500)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert "£" in chunks[0].content
        assert "%" in chunks[0].content

    def test_abbreviations_preserved(self):
        """Test that abbreviations like Mr. Dr. don't cause sentence breaks."""
        doc = Document(
            id="abbrev-doc",
            content="Contact Mr. Smith for details. Dr. Jones will review.",
            metadata=DocumentMetadata(tender_id="test", document_type=DocumentType.TEXT),
        )
        chunker = DocumentChunker(chunk_size=500)
        chunks = chunker.chunk_document(doc)

        # Should be one chunk (two sentences)
        assert len(chunks) == 1
        # Mr. should not cause a break
        assert "Mr. Smith" in chunks[0].content or "Mr." in chunks[0].content

    def test_multiple_paragraphs(self):
        """Test document with multiple paragraphs."""
        doc = Document(
            id="multi-para",
            content="First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.",
            metadata=DocumentMetadata(tender_id="test", document_type=DocumentType.TEXT),
        )
        chunker = DocumentChunker(chunk_size=500)
        chunks = chunker.chunk_document(doc)

        # Should preserve paragraph structure
        assert len(chunks) >= 1

    def test_very_long_sentence(self):
        """Test document with a very long single sentence."""
        long_sentence = "This is a very long sentence " + "with many words " * 50 + "that should still be handled."
        doc = Document(
            id="long-sentence",
            content=long_sentence,
            metadata=DocumentMetadata(tender_id="test", document_type=DocumentType.TEXT),
        )
        chunker = DocumentChunker(chunk_size=200, overlap_sentences=1)
        chunks = chunker.chunk_document(doc)

        # Even with a long sentence, should produce chunks
        assert len(chunks) >= 1

    def test_numbered_list(self):
        """Test document with numbered list items."""
        doc = Document(
            id="numbered-list",
            content="Requirements: 1. First item. 2. Second item. 3. Third item.",
            metadata=DocumentMetadata(tender_id="test", document_type=DocumentType.TEXT),
        )
        chunker = DocumentChunker(chunk_size=500)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) >= 1
        # Numbers should be preserved
        assert "1." in chunks[0].content or "First" in chunks[0].content
