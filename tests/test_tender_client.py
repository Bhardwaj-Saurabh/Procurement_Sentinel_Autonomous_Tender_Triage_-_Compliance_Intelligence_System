"""Tests for UK Contracts Finder API client."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from decimal import Decimal

from src.clients.tender_client import TenderClient, fetch_recent_tenders
from src.schemas.tender import Tender


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_release():
    """Sample OCDS release data from API."""
    return {
        "ocid": "ocds-b5fd17-test-123",
        "id": "1",
        "date": "2026-02-26T10:00:00Z",
        "tender": {
            "title": "Cloud Infrastructure Services",
            "description": "Provision of cloud services for government department",
            "status": "active",
            "value": {
                "amount": 500000,
                "currency": "GBP"
            },
            "tenderPeriod": {
                "startDate": "2026-02-26T00:00:00Z",
                "endDate": "2026-04-15T23:59:59Z"
            }
        },
        "buyer": {
            "name": "NHS Digital",
            "id": "GB-NHS-X26"
        }
    }


@pytest.fixture
def minimal_release():
    """Minimal release with only required fields."""
    return {
        "ocid": "ocds-b5fd17-minimal",
        "id": "1",
        "tender": {}
    }


@pytest.fixture
def release_with_nulls():
    """Release with null/missing optional fields."""
    return {
        "ocid": "ocds-b5fd17-nulls",
        "id": "1",
        "tender": {
            "title": None,
            "description": None,
            "value": None,
            "tenderPeriod": None
        },
        "buyer": None
    }


@pytest.fixture
def mock_api_response(sample_release):
    """Mock successful API response."""
    return {
        "uri": "https://example.com",
        "version": "1.1",
        "releases": [sample_release]
    }


@pytest.fixture
def empty_api_response():
    """Mock empty API response."""
    return {
        "uri": "https://example.com",
        "version": "1.1",
        "releases": []
    }


# =============================================================================
# TenderClient Tests
# =============================================================================


class TestTenderClientInit:
    """Tests for TenderClient initialization."""

    def test_client_initializes_with_config(self):
        """Client should load config on init."""
        with patch("src.clients.tender_client.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                tender_api_base_url="https://test-api.gov.uk"
            )
            client = TenderClient()

            assert client.base_url == "https://test-api.gov.uk"
            assert client.client is not None

    def test_client_context_manager(self):
        """Client should work as context manager."""
        with patch("src.clients.tender_client.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                tender_api_base_url="https://test-api.gov.uk"
            )

            with TenderClient() as client:
                assert client is not None

            # Client should be closed after context


class TestSearchTenders:
    """Tests for search_tenders method."""

    @patch("src.clients.tender_client.get_settings")
    def test_search_with_default_dates(self, mock_settings, mock_api_response):
        """Search should use last 7 days by default."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        with patch("httpx.Client.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_api_response
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            with TenderClient() as client:
                results = client.search_tenders(limit=10)

            # Check API was called
            mock_get.assert_called_once()
            call_args = mock_get.call_args

            # Verify params include date range
            assert "params" in call_args.kwargs
            params = call_args.kwargs["params"]
            assert "publishedFrom" in params
            assert "publishedTo" in params
            assert params["size"] == 10

    @patch("src.clients.tender_client.get_settings")
    def test_search_with_custom_dates(self, mock_settings, mock_api_response):
        """Search should accept custom date range."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        with patch("httpx.Client.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_api_response
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            published_from = datetime(2026, 1, 1, tzinfo=timezone.utc)
            published_to = datetime(2026, 1, 31, tzinfo=timezone.utc)

            with TenderClient() as client:
                results = client.search_tenders(
                    published_from=published_from,
                    published_to=published_to,
                    limit=5
                )

            call_args = mock_get.call_args
            params = call_args.kwargs["params"]
            assert "2026-01-01" in params["publishedFrom"]
            assert "2026-01-31" in params["publishedTo"]

    @patch("src.clients.tender_client.get_settings")
    def test_search_returns_releases(self, mock_settings, mock_api_response):
        """Search should return releases list."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        with patch("httpx.Client.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_api_response
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            with TenderClient() as client:
                results = client.search_tenders()

            assert isinstance(results, list)
            assert len(results) == 1

    @patch("src.clients.tender_client.get_settings")
    def test_search_empty_results(self, mock_settings, empty_api_response):
        """Search should handle empty results."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        with patch("httpx.Client.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = empty_api_response
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            with TenderClient() as client:
                results = client.search_tenders()

            assert results == []


class TestGetTenderByOcid:
    """Tests for get_tender_by_ocid method."""

    @patch("src.clients.tender_client.get_settings")
    def test_get_existing_tender(self, mock_settings, sample_release):
        """Should return tender data for valid OCID."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        with patch("httpx.Client.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_release
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            with TenderClient() as client:
                result = client.get_tender_by_ocid("ocds-b5fd17-test-123")

            assert result is not None
            assert result["ocid"] == "ocds-b5fd17-test-123"

    @patch("src.clients.tender_client.get_settings")
    def test_get_nonexistent_tender_returns_none(self, mock_settings):
        """Should return None for 404 response."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        with patch("httpx.Client.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response

            with TenderClient() as client:
                result = client.get_tender_by_ocid("ocds-nonexistent")

            assert result is None


class TestParseTender:
    """Tests for parse_tender method."""

    @patch("src.clients.tender_client.get_settings")
    def test_parse_complete_tender(self, mock_settings, sample_release):
        """Should parse all fields from complete release."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        with TenderClient() as client:
            tender = client.parse_tender(sample_release)

        assert isinstance(tender, Tender)
        assert tender.ocid == "ocds-b5fd17-test-123"
        assert tender.title == "Cloud Infrastructure Services"
        assert tender.description == "Provision of cloud services for government department"
        assert tender.status == "active"

        # Check value
        assert tender.value is not None
        assert tender.value.amount == 500000
        assert tender.value.currency == "GBP"

        # Check buyer
        assert tender.buyer is not None
        assert tender.buyer.name == "NHS Digital"
        assert tender.buyer.id == "GB-NHS-X26"

        # Check tender period
        assert tender.tender_period is not None

    @patch("src.clients.tender_client.get_settings")
    def test_parse_minimal_tender(self, mock_settings, minimal_release):
        """Should parse tender with only required fields."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        with TenderClient() as client:
            tender = client.parse_tender(minimal_release)

        assert tender.ocid == "ocds-b5fd17-minimal"
        assert tender.title is None
        assert tender.description is None
        assert tender.value is None
        assert tender.buyer is None
        assert tender.tender_period is None

    @patch("src.clients.tender_client.get_settings")
    def test_parse_tender_with_nulls(self, mock_settings, release_with_nulls):
        """Should handle null values gracefully."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        with TenderClient() as client:
            tender = client.parse_tender(release_with_nulls)

        assert tender.ocid == "ocds-b5fd17-nulls"
        assert tender.title is None
        assert tender.value is None
        assert tender.buyer is None

    @patch("src.clients.tender_client.get_settings")
    def test_parse_tender_missing_ocid(self, mock_settings):
        """Should handle missing OCID with empty string."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        release = {"id": "1", "tender": {}}

        with TenderClient() as client:
            tender = client.parse_tender(release)

        assert tender.ocid == ""

    @patch("src.clients.tender_client.get_settings")
    def test_parse_tender_partial_value(self, mock_settings):
        """Should handle partial value data (amount but no currency)."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        release = {
            "ocid": "test",
            "id": "1",
            "tender": {
                "value": {
                    "amount": 100000
                    # currency missing
                }
            }
        }

        with TenderClient() as client:
            tender = client.parse_tender(release)

        assert tender.value is not None
        assert tender.value.amount == 100000
        assert tender.value.currency == "GBP"  # Default

    @patch("src.clients.tender_client.get_settings")
    def test_parse_tender_empty_buyer(self, mock_settings):
        """Should handle empty buyer object."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        release = {
            "ocid": "test",
            "id": "1",
            "tender": {},
            "buyer": {}  # Empty dict is falsy in Python
        }

        with TenderClient() as client:
            tender = client.parse_tender(release)

        # Empty dict {} is falsy, so buyer is None
        assert tender.buyer is None


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestFetchRecentTenders:
    """Tests for fetch_recent_tenders function."""

    @patch("src.clients.tender_client.get_settings")
    def test_fetch_returns_tender_objects(self, mock_settings, mock_api_response):
        """Should return list of Tender objects."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        with patch("httpx.Client.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_api_response
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            tenders = fetch_recent_tenders(days=7, limit=10)

        assert isinstance(tenders, list)
        assert len(tenders) == 1
        assert isinstance(tenders[0], Tender)

    @patch("src.clients.tender_client.get_settings")
    def test_fetch_empty_results(self, mock_settings, empty_api_response):
        """Should handle empty results."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        with patch("httpx.Client.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = empty_api_response
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            tenders = fetch_recent_tenders()

        assert tenders == []

    @patch("src.clients.tender_client.get_settings")
    def test_fetch_with_custom_days(self, mock_settings, mock_api_response):
        """Should accept custom days parameter."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        with patch("httpx.Client.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_api_response
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            tenders = fetch_recent_tenders(days=30, limit=5)

        # Verify the call was made
        mock_get.assert_called_once()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual data."""

    @patch("src.clients.tender_client.get_settings")
    def test_very_long_title(self, mock_settings):
        """Should handle very long title strings."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        long_title = "A" * 10000  # Very long title
        release = {
            "ocid": "test",
            "id": "1",
            "tender": {
                "title": long_title
            }
        }

        with TenderClient() as client:
            tender = client.parse_tender(release)

        assert tender.title == long_title
        assert len(tender.title) == 10000

    @patch("src.clients.tender_client.get_settings")
    def test_unicode_in_fields(self, mock_settings):
        """Should handle unicode characters."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        release = {
            "ocid": "test",
            "id": "1",
            "tender": {
                "title": "Procurement für München — £500,000",
                "description": "日本語テスト 中文测试 العربية"
            },
            "buyer": {
                "name": "Côte d'Ivoire Ministry"
            }
        }

        with TenderClient() as client:
            tender = client.parse_tender(release)

        assert "München" in tender.title
        assert "£" in tender.title
        assert "日本語" in tender.description
        assert "Côte d'Ivoire" in tender.buyer.name

    @patch("src.clients.tender_client.get_settings")
    def test_zero_value_tender(self, mock_settings):
        """Should handle zero value tender."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        release = {
            "ocid": "test",
            "id": "1",
            "tender": {
                "value": {
                    "amount": 0,
                    "currency": "GBP"
                }
            }
        }

        with TenderClient() as client:
            tender = client.parse_tender(release)

        assert tender.value.amount == 0

    @patch("src.clients.tender_client.get_settings")
    def test_negative_value_tender(self, mock_settings):
        """Should handle negative value (edge case, shouldn't happen)."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        release = {
            "ocid": "test",
            "id": "1",
            "tender": {
                "value": {
                    "amount": -1000,
                    "currency": "GBP"
                }
            }
        }

        with TenderClient() as client:
            tender = client.parse_tender(release)

        # Should still parse, validation is separate concern
        assert tender.value.amount == -1000

    @patch("src.clients.tender_client.get_settings")
    def test_large_value_tender(self, mock_settings):
        """Should handle very large tender values."""
        mock_settings.return_value = Mock(
            tender_api_base_url="https://test-api.gov.uk"
        )

        release = {
            "ocid": "test",
            "id": "1",
            "tender": {
                "value": {
                    "amount": 999999999999.99,
                    "currency": "GBP"
                }
            }
        }

        with TenderClient() as client:
            tender = client.parse_tender(release)

        # Value is stored as Decimal for precision
        assert tender.value.amount == Decimal("999999999999.99")


# =============================================================================
# Integration Test (requires network - skip by default)
# =============================================================================


@pytest.mark.integration
@pytest.mark.skip(reason="Requires network access to real API")
class TestIntegration:
    """Integration tests with real API."""

    def test_real_api_search(self):
        """Test actual API call."""
        with TenderClient() as client:
            results = client.search_tenders(limit=1)

        assert isinstance(results, list)
        if results:
            assert "ocid" in results[0]

    def test_real_api_parse(self):
        """Test parsing real API data."""
        with TenderClient() as client:
            results = client.search_tenders(limit=1)
            if results:
                tender = client.parse_tender(results[0])
                assert tender.ocid is not None
