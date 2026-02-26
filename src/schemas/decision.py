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


# =============================================================================
# Human-in-the-Loop: Gap Analysis
# =============================================================================


class GapCategory(str, Enum):
    """Category of gap identified."""

    CERTIFICATION = "certification"
    EXPERIENCE = "experience"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    LEGAL = "legal"
    OTHER = "other"


class ResponseType(str, Enum):
    """Type of response expected from user."""

    YES_NO = "yes_no"
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    PROJECT_DETAILS = "project_details"
    DOCUMENT_UPLOAD = "document_upload"


class GapItem(BaseModel):
    """A gap between tender requirement and company profile."""

    gap_id: str
    category: GapCategory
    requirement_description: str
    current_status: str  # What we know from profile
    question_for_user: str  # What to ask
    response_type: ResponseType = ResponseType.TEXT
    user_response: Optional[str] = None
    resolved: bool = False
    citation: Optional[Citation] = None


class HumanInput(BaseModel):
    """Additional information provided by user."""

    gap_id: str
    response_value: str
    supporting_evidence: Optional[str] = None
    additional_notes: Optional[str] = None
    provided_at: datetime = Field(default_factory=datetime.utcnow)


class GapAnalysisResult(BaseModel):
    """Result of gap analysis before final decision."""

    gaps: list[GapItem] = Field(default_factory=list)
    total_gaps: int = 0
    critical_gaps: int = 0  # Gaps that block decision
    requires_human_input: bool = False
    auto_decidable: bool = True  # Can decide without human?
    human_inputs: list[HumanInput] = Field(default_factory=list)

    def add_gap(self, gap: GapItem) -> None:
        """Add a gap and update counts."""
        self.gaps.append(gap)
        self.total_gaps = len(self.gaps)
        self.requires_human_input = True
        self.auto_decidable = False

    def resolve_gap(self, gap_id: str, response: HumanInput) -> bool:
        """Resolve a gap with human input."""
        for gap in self.gaps:
            if gap.gap_id == gap_id:
                gap.user_response = response.response_value
                gap.resolved = True
                self.human_inputs.append(response)
                return True
        return False

    @property
    def unresolved_gaps(self) -> list[GapItem]:
        """Get list of unresolved gaps."""
        return [g for g in self.gaps if not g.resolved]

    @property
    def all_resolved(self) -> bool:
        """Check if all gaps are resolved."""
        return all(g.resolved for g in self.gaps)


class TenderAnalysisState(BaseModel):
    """State object for LangGraph workflow."""

    tender_id: str
    tender_data: Optional[dict] = None
    documents_text: list[str] = Field(default_factory=list)
    requirements: list[ExtractedRequirement] = Field(default_factory=list)
    compliance_results: list[ComplianceCheck] = Field(default_factory=list)
    gap_analysis: Optional[GapAnalysisResult] = None  # Human-in-the-loop
    awaiting_human_input: bool = False  # Pause workflow for human
    decision: Optional[BidDecision] = None
    error: Optional[str] = None
