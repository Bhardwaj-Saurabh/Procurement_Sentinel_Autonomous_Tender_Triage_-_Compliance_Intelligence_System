"""Company profile schemas."""

from datetime import date
from decimal import Decimal
from typing import Optional
from pydantic import BaseModel, Field


class Certification(BaseModel):
    """Company certification."""

    name: str
    expiry: Optional[date] = None
    issuer: Optional[str] = None


class PastContract(BaseModel):
    """Past contract/project reference."""

    title: str
    client: str
    value: Optional[Decimal] = None
    year: int
    sector: Optional[str] = None
    description: Optional[str] = None


class CompanyProfile(BaseModel):
    """Company profile for compliance matching."""

    company_name: str
    annual_revenue: Decimal
    currency: str = "GBP"
    employee_count: Optional[int] = None

    certifications: list[Certification] = Field(default_factory=list)
    past_contracts: list[PastContract] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)

    min_contract_value: Optional[Decimal] = None
    max_contract_value: Optional[Decimal] = None

    geographic_coverage: list[str] = Field(default_factory=list)
    risk_appetite: str = "moderate"  # conservative, moderate, aggressive

    def has_certification(self, cert_name: str) -> bool:
        """Check if company has a specific certification."""
        cert_name_lower = cert_name.lower()
        return any(
            cert_name_lower in cert.name.lower() for cert in self.certifications
        )

    def get_certification(self, cert_name: str) -> Optional[Certification]:
        """Get certification by name (partial match)."""
        cert_name_lower = cert_name.lower()
        for cert in self.certifications:
            if cert_name_lower in cert.name.lower():
                return cert
        return None

    def has_sector_experience(self, sector: str) -> bool:
        """Check if company has experience in a sector."""
        sector_lower = sector.lower()
        return any(sector_lower in s.lower() for s in self.sectors)
