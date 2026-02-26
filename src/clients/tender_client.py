"""UK Contracts Finder / Find-a-Tender API client."""

import httpx
from typing import Optional
from datetime import datetime, timedelta, timezone

from src.config import get_settings
from src.schemas.tender import Tender, TenderValue, TenderPeriod, Buyer


class TenderClient:
    """Client for UK Contracts Finder API (OCDS format)."""

    def __init__(self):
        settings = get_settings()
        self.base_url = settings.tender_api_base_url
        self.client = httpx.Client(timeout=30.0)

    def search_tenders(
        self,
        published_from: Optional[datetime] = None,
        published_to: Optional[datetime] = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Search for published tenders.

        Args:
            published_from: Start date for search
            published_to: End date for search
            limit: Maximum number of results

        Returns:
            List of tender release packages
        """
        if published_from is None:
            published_from = datetime.now(timezone.utc) - timedelta(days=7)
        if published_to is None:
            published_to = datetime.now(timezone.utc)

        params = {
            "publishedFrom": published_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "publishedTo": published_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "size": limit,
        }

        response = self.client.get(
            f"{self.base_url}/Search",
            params=params,
        )
        response.raise_for_status()

        data = response.json()
        # API returns releases directly at root level
        return data.get("releases", [])

    def get_tender_by_ocid(self, ocid: str) -> Optional[dict]:
        """
        Get a specific tender by OCID.

        Args:
            ocid: Open Contracting ID

        Returns:
            Tender release package or None
        """
        response = self.client.get(f"{self.base_url}/Record/{ocid}")

        if response.status_code == 404:
            return None

        response.raise_for_status()
        return response.json()

    def parse_tender(self, release_data: dict) -> Tender:
        """
        Parse raw OCDS release into Tender schema.

        Args:
            release_data: Raw OCDS release data

        Returns:
            Parsed Tender object
        """
        tender_data = release_data.get("tender", {})
        buyer_data = release_data.get("buyer", {})

        value = None
        if tender_data.get("value"):
            value = TenderValue(
                amount=tender_data["value"].get("amount"),
                currency=tender_data["value"].get("currency", "GBP"),
            )

        period = None
        if tender_data.get("tenderPeriod"):
            period = TenderPeriod(
                start_date=tender_data["tenderPeriod"].get("startDate"),
                end_date=tender_data["tenderPeriod"].get("endDate"),
            )

        buyer = None
        if buyer_data:
            buyer = Buyer(
                name=buyer_data.get("name"),
                id=buyer_data.get("id"),
            )

        return Tender(
            ocid=release_data.get("ocid", ""),
            id=release_data.get("id", ""),
            title=tender_data.get("title"),
            description=tender_data.get("description"),
            status=tender_data.get("status"),
            value=value,
            tender_period=period,
            buyer=buyer,
            published_date=release_data.get("date"),
        )

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def fetch_recent_tenders(days: int = 7, limit: int = 10) -> list[Tender]:
    """
    Convenience function to fetch recent tenders.

    Args:
        days: Number of days to look back
        limit: Maximum number of tenders

    Returns:
        List of parsed Tender objects
    """
    with TenderClient() as client:
        published_from = datetime.now(timezone.utc) - timedelta(days=days)
        releases = client.search_tenders(
            published_from=published_from,
            limit=limit,
        )

        tenders = []
        for release in releases:
            tender = client.parse_tender(release)
            tenders.append(tender)

        return tenders
