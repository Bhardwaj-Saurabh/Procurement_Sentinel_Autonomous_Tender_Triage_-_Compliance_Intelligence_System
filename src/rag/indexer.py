"""RAG indexer for storing document chunks in Azure AI Search."""

import structlog
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchableField,
    SimpleField,
)
from azure.core.credentials import AzureKeyCredential

from src.config import get_settings
from src.schemas.document import DocumentChunk

logger = structlog.get_logger()


class RAGIndexer:
    """
    Indexes document chunks into Azure AI Search with embeddings.

    Uses Azure OpenAI for embedding generation and Azure AI Search
    for hybrid (vector + keyword) retrieval.
    """

    def __init__(self):
        """Initialize the indexer with Azure clients."""
        settings = get_settings()

        # Azure OpenAI client for embeddings
        self.openai_client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self.embedding_deployment = settings.azure_openai_embedding_deployment
        self.embedding_dimensions = 3072  # text-embedding-3-large

        # Azure AI Search clients
        self.search_endpoint = settings.azure_search_endpoint
        self.search_key = settings.azure_search_key
        self.index_name = settings.azure_search_index

        credential = AzureKeyCredential(self.search_key)

        self.index_client = SearchIndexClient(
            endpoint=self.search_endpoint,
            credential=credential,
        )

        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=credential,
        )

    def create_index(self, delete_if_exists: bool = False) -> None:
        """
        Create the search index with vector search configuration.

        Args:
            delete_if_exists: If True, delete existing index first
        """
        if delete_if_exists:
            try:
                self.index_client.delete_index(self.index_name)
                logger.info("deleted_existing_index", index=self.index_name)
            except Exception:
                pass  # Index didn't exist

        # Define fields
        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
            ),
            SimpleField(
                name="document_id",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SimpleField(
                name="tender_id",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                analyzer_name="en.microsoft",
            ),
            SimpleField(
                name="chunk_index",
                type=SearchFieldDataType.Int32,
            ),
            SimpleField(
                name="total_chunks",
                type=SearchFieldDataType.Int32,
            ),
            SimpleField(
                name="source_file",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SimpleField(
                name="page_number",
                type=SearchFieldDataType.Int32,
            ),
            SimpleField(
                name="section",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.embedding_dimensions,
                vector_search_profile_name="default-profile",
            ),
        ]

        # Vector search configuration
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="default-algorithm"),
            ],
            profiles=[
                VectorSearchProfile(
                    name="default-profile",
                    algorithm_configuration_name="default-algorithm",
                ),
            ],
        )

        # Create the index
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
        )

        self.index_client.create_or_update_index(index)
        logger.info("created_search_index", index=self.index_name)

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a text using Azure OpenAI.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (3072 dimensions)
        """
        response = self.openai_client.embeddings.create(
            model=self.embedding_deployment,
            input=text,
        )
        return response.data[0].embedding

    def generate_embeddings_batch(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.openai_client.embeddings.create(
                model=self.embedding_deployment,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            logger.debug(
                "generated_embeddings_batch",
                batch_start=i,
                batch_size=len(batch),
            )

        return all_embeddings

    def index_chunks(
        self,
        chunks: list[DocumentChunk],
        generate_embeddings: bool = True,
    ) -> int:
        """
        Index document chunks into Azure AI Search.

        Args:
            chunks: List of chunks to index
            generate_embeddings: If True, generate embeddings for chunks

        Returns:
            Number of chunks indexed
        """
        if not chunks:
            return 0

        # Generate embeddings if needed
        if generate_embeddings:
            texts = [chunk.content for chunk in chunks]
            embeddings = self.generate_embeddings_batch(texts)

            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding

        # Prepare documents for indexing
        documents = []
        for chunk in chunks:
            doc = {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "tender_id": chunk.tender_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "section": chunk.section,
                "embedding": chunk.embedding,
            }
            documents.append(doc)

        # Upload to Azure AI Search
        result = self.search_client.upload_documents(documents)

        success_count = sum(1 for r in result if r.succeeded)
        logger.info(
            "indexed_chunks",
            total=len(chunks),
            success=success_count,
            failed=len(chunks) - success_count,
        )

        return success_count

    def delete_by_tender_id(self, tender_id: str) -> int:
        """
        Delete all chunks for a specific tender.

        Args:
            tender_id: Tender ID to delete chunks for

        Returns:
            Number of chunks deleted
        """
        # Search for all chunks with this tender_id
        results = self.search_client.search(
            search_text="*",
            filter=f"tender_id eq '{tender_id}'",
            select=["id"],
        )

        doc_ids = [{"id": doc["id"]} for doc in results]

        if not doc_ids:
            return 0

        result = self.search_client.delete_documents(doc_ids)
        deleted_count = sum(1 for r in result if r.succeeded)

        logger.info(
            "deleted_tender_chunks",
            tender_id=tender_id,
            deleted=deleted_count,
        )

        return deleted_count

    def get_index_stats(self) -> dict:
        """
        Get statistics about the search index.

        Returns:
            Dictionary with index statistics
        """
        try:
            index = self.index_client.get_index(self.index_name)
            # Get document count via search
            results = self.search_client.search(
                search_text="*",
                include_total_count=True,
            )
            return {
                "index_name": self.index_name,
                "document_count": results.get_count(),
                "fields": len(index.fields),
            }
        except Exception as e:
            logger.error("failed_to_get_index_stats", error=str(e))
            return {"error": str(e)}


def index_tender_documents(chunks: list[DocumentChunk]) -> int:
    """
    Convenience function to index chunks for a tender.

    Args:
        chunks: Document chunks to index

    Returns:
        Number of chunks indexed
    """
    indexer = RAGIndexer()
    return indexer.index_chunks(chunks)
