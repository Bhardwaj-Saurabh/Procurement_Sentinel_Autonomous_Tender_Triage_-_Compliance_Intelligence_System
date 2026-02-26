"""RAG retriever for searching document chunks in Azure AI Search."""

from typing import Optional
import structlog
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from pydantic import BaseModel

from src.config import get_settings

logger = structlog.get_logger()


class SearchResult(BaseModel):
    """A single search result from RAG retrieval."""

    id: str
    document_id: str
    tender_id: str
    content: str
    chunk_index: int
    total_chunks: int
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    score: float

    @property
    def citation(self) -> str:
        """Generate a citation string for this result."""
        parts = []
        if self.source_file:
            parts.append(self.source_file)
        if self.page_number:
            parts.append(f"Page {self.page_number}")
        parts.append(f"Chunk {self.chunk_index + 1}/{self.total_chunks}")
        return " | ".join(parts)


class RAGRetriever:
    """
    Retrieves relevant document chunks using hybrid search.

    Combines vector similarity search with keyword search.
    """

    def __init__(self):
        """Initialize the retriever with Azure clients."""
        settings = get_settings()

        # Azure OpenAI client for query embedding
        self.openai_client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self.embedding_deployment = settings.azure_openai_embedding_deployment

        # Azure AI Search client
        credential = AzureKeyCredential(settings.azure_search_key)
        self.search_client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index,
            credential=credential,
        )

    def _embed_query(self, query: str) -> list[float]:
        """Generate embedding for the search query."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_deployment,
            input=query,
        )
        return response.data[0].embedding

    def search(
        self,
        query: str,
        tender_id: Optional[str] = None,
        top_k: int = 5,
        use_vector: bool = True,
        use_text: bool = True,
    ) -> list[SearchResult]:
        """
        Search for relevant document chunks.

        Args:
            query: Search query
            tender_id: Optional filter by tender ID
            top_k: Number of results to return
            use_vector: Use vector similarity search
            use_text: Use keyword text search

        Returns:
            List of search results ranked by relevance
        """
        # Build filter
        filter_expr = None
        if tender_id:
            filter_expr = f"tender_id eq '{tender_id}'"

        # Prepare vector query if enabled
        vector_queries = []
        if use_vector:
            query_embedding = self._embed_query(query)
            vector_queries.append(
                VectorizedQuery(
                    vector=query_embedding,
                    k_nearest_neighbors=top_k,
                    fields="embedding",
                )
            )

        # Execute search
        search_text = query if use_text else None

        results = self.search_client.search(
            search_text=search_text,
            vector_queries=vector_queries if vector_queries else None,
            filter=filter_expr,
            top=top_k,
            select=[
                "id",
                "document_id",
                "tender_id",
                "content",
                "chunk_index",
                "total_chunks",
                "source_file",
                "page_number",
                "section",
            ],
        )

        # Convert to SearchResult objects
        search_results = []
        for result in results:
            search_results.append(
                SearchResult(
                    id=result["id"],
                    document_id=result["document_id"],
                    tender_id=result["tender_id"],
                    content=result["content"],
                    chunk_index=result["chunk_index"],
                    total_chunks=result["total_chunks"],
                    source_file=result.get("source_file"),
                    page_number=result.get("page_number"),
                    section=result.get("section"),
                    score=result["@search.score"],
                )
            )

        logger.debug(
            "search_completed",
            query=query[:50],
            results=len(search_results),
        )

        return search_results

    def get_all_chunks_for_tender(
        self,
        tender_id: str,
        max_chunks: int = 100,
    ) -> list[SearchResult]:
        """
        Get all indexed chunks for a specific tender.

        Args:
            tender_id: Tender ID to retrieve chunks for
            max_chunks: Maximum chunks to return

        Returns:
            All chunks for the tender, ordered by chunk_index
        """
        results = self.search_client.search(
            search_text="*",
            filter=f"tender_id eq '{tender_id}'",
            top=max_chunks,
            select=[
                "id",
                "document_id",
                "tender_id",
                "content",
                "chunk_index",
                "total_chunks",
                "source_file",
                "page_number",
                "section",
            ],
            order_by=["chunk_index"],
        )

        return [
            SearchResult(
                id=r["id"],
                document_id=r["document_id"],
                tender_id=r["tender_id"],
                content=r["content"],
                chunk_index=r["chunk_index"],
                total_chunks=r["total_chunks"],
                source_file=r.get("source_file"),
                page_number=r.get("page_number"),
                section=r.get("section"),
                score=1.0,
            )
            for r in results
        ]


def retrieve_context(
    query: str,
    tender_id: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """
    Convenience function to retrieve context as formatted string.

    Args:
        query: Search query
        tender_id: Optional tender filter
        top_k: Number of chunks to retrieve

    Returns:
        Formatted context string for LLM consumption
    """
    retriever = RAGRetriever()
    results = retriever.search(query=query, tender_id=tender_id, top_k=top_k)

    if not results:
        return "No relevant context found."

    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(f"[{i}] {result.citation}\n{result.content}")

    return "\n\n---\n\n".join(context_parts)
