"""Decision and compliance schemas."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ComplianceStatus(str, Enum):
    """Compliance check status."""

    COMPLIANT = "COMPLIANT"
    POTENTIAL_RISK = "POTENTIAL_RISK"
    NON_COMPLIANT = "NON_COMPLIANT"
    MISSING_DATA = "MISSING_DATA"


class RiskLevel(str, Enum):
    """Risk severity level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionType(str, Enum):
    """Bid decision type."""

    BID = "BID"
    NO_BID = "NO_BID"
    CONDITIONAL_BID = "CONDITIONAL_BID"


class Citation(BaseModel):
    """Citation from source document."""

    document_name: str
    section: Optional[str] = None
    page_number: Optional[int] = None
    quote: str


class ExtractedRequirement(BaseModel):
    """Requirement extracted from tender documents."""

    requirement_id: str
    description: str
    category: str  # financial, technical, legal, experience
    is_mandatory: bool = True
    citation: Optional[Citation] = None


class ComplianceCheck(BaseModel):
    """Result of checking one requirement against company profile."""

    requirement_id: str
    requirement_description: str
    status: ComplianceStatus
    company_evidence: Optional[str] = None
    gap_description: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.MEDIUM
    recommendation: Optional[str] = None


class Risk(BaseModel):
    """Identified risk."""

    risk_id: str
    description: str
    level: RiskLevel
    mitigation: Optional[str] = None
    citation: Optional[Citation] = None


class ActionItem(BaseModel):
    """Action item for bid preparation."""

    priority: str  # urgent, high, medium, low
    description: str
    owner: Optional[str] = None


class BidDecision(BaseModel):
    """Final bid/no-bid decision."""

    decision: DecisionType
    confidence: float = Field(ge=0.0, le=1.0)

    headline: str
    executive_summary: str

    strategic_fit_score: int = Field(ge=0, le=100)
    risk_score: int = Field(ge=0, le=100)

    requirements: list[ExtractedRequirement] = Field(default_factory=list)
    compliance_checks: list[ComplianceCheck] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)

    reasoning: str
    processed_at: datetime = Field(default_factory=datetime.utcnow)


class TenderAnalysisState(BaseModel):
    """State object for LangGraph workflow."""

    tender_id: str
    tender_data: Optional[dict] = None
    documents_text: list[str] = Field(default_factory=list)
    requirements: list[ExtractedRequirement] = Field(default_factory=list)
    compliance_results: list[ComplianceCheck] = Field(default_factory=list)
    decision: Optional[BidDecision] = None
    error: Optional[str] = None
