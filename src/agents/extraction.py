"""Extraction agent for identifying requirements from tender documents."""

import json
import structlog
from openai import AzureOpenAI

from src.config import get_settings
from src.rag.retriever import RAGRetriever
from src.schemas.decision import ExtractedRequirement, Citation

logger = structlog.get_logger()

EXTRACTION_PROMPT = """You are an expert at analyzing UK public procurement tender documents.

Your task is to extract ALL mandatory and desirable requirements from the provided tender document chunks.

For each requirement, identify:
1. A unique ID (e.g., REQ-001, REQ-002)
2. The exact requirement description
3. The category: financial, technical, legal, experience, certification, or other
4. Whether it is mandatory (must have) or desirable (nice to have)
5. The source text (quote) where you found this requirement

IMPORTANT GUIDELINES:
- Extract EVERY requirement you can find, including implicit ones
- Mandatory requirements often use words like: "must", "shall", "required", "essential"
- Desirable requirements use words like: "should", "preferred", "desirable", "advantageous"
- Include certification requirements (ISO, Cyber Essentials, etc.)
- Include experience requirements (years of experience, past contracts)
- Include financial requirements (turnover, insurance, financial stability)
- Include legal requirements (compliance, regulations, data protection)

Return your response as a JSON object with a "requirements" array:
{{
  "requirements": [
    {{
      "requirement_id": "REQ-001",
      "description": "Supplier must hold ISO 9001 certification",
      "category": "certification",
      "is_mandatory": true,
      "source_quote": "The supplier shall hold a valid ISO 9001:2015 certification"
    }}
  ]
}}

TENDER DOCUMENT CHUNKS:
{context}

Extract all requirements from the above text. Return ONLY valid JSON."""


class ExtractionAgent:
    """
    Agent that extracts requirements from tender document chunks.

    Uses RAG to retrieve relevant chunks and LLM to extract
    structured requirements.
    """

    def __init__(self):
        """Initialize the extraction agent."""
        settings = get_settings()

        self.llm_client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self.deployment = settings.azure_openai_deployment
        self.retriever = RAGRetriever()

    def extract_requirements(
        self,
        tender_id: str,
        additional_context: str = "",
    ) -> list[ExtractedRequirement]:
        """
        Extract requirements from indexed tender documents.

        Args:
            tender_id: ID of the tender to extract from
            additional_context: Optional additional text to include

        Returns:
            List of extracted requirements
        """
        # Get all chunks for this tender
        chunks = self.retriever.get_all_chunks_for_tender(tender_id)

        if not chunks:
            logger.warning("no_chunks_found", tender_id=tender_id)
            return []

        # Build context from chunks
        context_parts = []
        for chunk in chunks:
            context_parts.append(f"[Chunk {chunk.chunk_index + 1}]\n{chunk.content}")

        context = "\n\n---\n\n".join(context_parts)

        if additional_context:
            context = f"{additional_context}\n\n---\n\n{context}"

        # Call LLM for extraction
        prompt = EXTRACTION_PROMPT.format(context=context)

        response = self.llm_client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,  # Low temperature for consistent extraction
        )

        # Parse response
        content = response.choices[0].message.content
        requirements = self._parse_response(content, chunks)

        logger.info(
            "extracted_requirements",
            tender_id=tender_id,
            count=len(requirements),
        )

        return requirements

    def extract_from_text(self, text: str, tender_id: str = "manual") -> list[ExtractedRequirement]:
        """
        Extract requirements from raw text (not indexed).

        Args:
            text: Raw text to extract from
            tender_id: Optional tender ID for reference

        Returns:
            List of extracted requirements
        """
        prompt = EXTRACTION_PROMPT.format(context=text)

        response = self.llm_client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        content = response.choices[0].message.content
        requirements = self._parse_response(content, [])

        logger.info(
            "extracted_requirements_from_text",
            tender_id=tender_id,
            count=len(requirements),
        )

        return requirements

    def _parse_response(
        self,
        content: str,
        chunks: list,
    ) -> list[ExtractedRequirement]:
        """
        Parse LLM response into ExtractedRequirement objects.

        Args:
            content: Raw JSON response from LLM
            chunks: Source chunks for citation lookup

        Returns:
            List of ExtractedRequirement objects
        """
        try:
            data = json.loads(content)

            # Handle both array and object with "requirements" key
            if isinstance(data, dict):
                data = data.get("requirements", [])

            requirements = []
            for item in data:
                # Build citation if we have source info
                citation = None
                if item.get("source_quote"):
                    # Try to find which chunk contains this quote
                    chunk_info = self._find_chunk_for_quote(
                        item["source_quote"],
                        chunks,
                    )
                    citation = Citation(
                        document_name=chunk_info.get("source_file") or "tender_document",
                        page_number=chunk_info.get("page_number"),
                        quote=item["source_quote"],
                    )

                req = ExtractedRequirement(
                    requirement_id=item.get("requirement_id", f"REQ-{len(requirements)+1:03d}"),
                    description=item.get("description", ""),
                    category=item.get("category", "other"),
                    is_mandatory=item.get("is_mandatory", True),
                    citation=citation,
                )
                requirements.append(req)

            return requirements

        except json.JSONDecodeError as e:
            logger.error("failed_to_parse_extraction", error=str(e))
            return []

    def _find_chunk_for_quote(
        self,
        quote: str,
        chunks: list,
    ) -> dict:
        """
        Find which chunk contains a quote.

        Args:
            quote: Quote text to search for
            chunks: List of chunks to search

        Returns:
            Dict with chunk metadata
        """
        quote_lower = quote.lower()[:50]  # First 50 chars for matching

        for chunk in chunks:
            if quote_lower in chunk.content.lower():
                return {
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                }

        return {}


def extract_tender_requirements(tender_id: str) -> list[ExtractedRequirement]:
    """
    Convenience function to extract requirements from a tender.

    Args:
        tender_id: ID of the tender

    Returns:
        List of extracted requirements
    """
    agent = ExtractionAgent()
    return agent.extract_requirements(tender_id)
