"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment: str
    azure_openai_embedding_deployment: str
    azure_openai_api_version: str = "2025-01-01-preview"

    # Azure AI Search
    azure_search_endpoint: str
    azure_search_key: str
    azure_search_index: str = "tender-documents"

    # Database
    database_url: str = "sqlite:///./procurement_sentinel.db"

    # UK Find-a-Tender API (no auth required)
    tender_api_base_url: str = "https://www.contractsfinder.service.gov.uk/Published/Notices/OCDS"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
