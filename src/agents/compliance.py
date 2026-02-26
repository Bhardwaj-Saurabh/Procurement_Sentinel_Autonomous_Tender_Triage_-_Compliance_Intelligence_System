""" Compliance agent for checking requirements against company profile."""

import json
from pathlib import Path
import structlog
from openai import AzureOpenAI

from src.config import get_settings
from src.schemas.decision import (
    ExtractedRequirement,
    ComplianceCheck,
    ComplianceStatus,
    RiskLevel,
    GapItem,
    GapCategory,
    ResponseType,
    GapAnalysisResult,
)

logger = structlog.get_logger()

COMPLIANCE_PROMPT = """You are an expert at evaluating company compliance with tender requirements.

Given a requirement and a company profile, determine if the company can meet the requirement.

COMPLIANCE STATUSES:
- COMPLIANT: Company clearly meets this requirement with evidence
- POTENTIAL_RISK: Company might meet it, but there are concerns or gaps
- NON_COMPLIANT: Company clearly cannot meet this requirement
- MISSING_DATA: Cannot determine - need more information

For each requirement, provide:
1. status: One of COMPLIANT, POTENTIAL_RISK, NON_COMPLIANT, MISSING_DATA
2. company_evidence: What in the company profile supports/contradicts this
3. gap_description: If not compliant, what is missing
4. risk_level: low, medium, high, or critical
5. recommendation: What action to take

COMPANY PROFILE:
{company_profile}

REQUIREMENTS TO CHECK:
{requirements}

Return a JSON object with "compliance_checks" array:
{{
  "compliance_checks": [
    {{
      "requirement_id": "REQ-001",
      "status": "COMPLIANT",
      "company_evidence": "Company holds ISO 9001 certification valid until 2026-11-30",
      "gap_description": null,
      "risk_level": "low",
      "recommendation": "Include certification in bid submission"
    }}
  ]
}}

Analyze each requirement carefully and return ONLY valid JSON."""


class ComplianceAgent:
    """
    Agent that checks tender requirements against company profile.

    Identifies gaps and risks before bid decision.
    """

    def __init__(self, company_profile_path: str = "data/company_profile.json"):
        """
        Initialize the compliance agent.

        Args:
            company_profile_path: Path to company profile JSON
        """
        settings = get_settings()

        self.llm_client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self.deployment = settings.azure_openai_deployment

        # Load company profile
        profile_path = Path(company_profile_path)
        if profile_path.exists():
            self.company_profile = json.loads(profile_path.read_text())
        else:
            logger.warning("company_profile_not_found", path=company_profile_path)
            self.company_profile = {}

    def check_compliance(
        self,
        requirements: list[ExtractedRequirement],
    ) -> tuple[list[ComplianceCheck], GapAnalysisResult]:
        """
        Check requirements against company profile.

        Args:
            requirements: List of requirements to check

        Returns:
            Tuple of (compliance checks, gap analysis result)
        """
        if not requirements:
            return [], GapAnalysisResult()

        # Format requirements for prompt
        req_text = ""
        for req in requirements:
            mandatory = "MANDATORY" if req.is_mandatory else "DESIRABLE"
            req_text += f"- {req.requirement_id} [{mandatory}]: {req.description}\n"

        # Call LLM
        prompt = COMPLIANCE_PROMPT.format(
            company_profile=json.dumps(self.company_profile, indent=2),
            requirements=req_text,
        )

        response = self.llm_client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        content = response.choices[0].message.content
        compliance_checks = self._parse_response(content, requirements)
        gap_analysis = self._build_gap_analysis(compliance_checks, requirements)

        logger.info(
            "compliance_check_complete",
            total=len(compliance_checks),
            compliant=sum(1 for c in compliance_checks if c.status == ComplianceStatus.COMPLIANT),
            gaps=gap_analysis.total_gaps,
        )

        return compliance_checks, gap_analysis

    def _parse_response(
        self,
        content: str,
        requirements: list[ExtractedRequirement],
    ) -> list[ComplianceCheck]:
        """Parse LLM response into ComplianceCheck objects."""
        try:
            data = json.loads(content)
            checks_data = data.get("compliance_checks", [])

            # Create lookup for requirement descriptions
            req_lookup = {r.requirement_id: r.description for r in requirements}

            checks = []
            for item in checks_data:
                req_id = item.get("requirement_id", "")

                # Map status string to enum
                status_str = item.get("status", "MISSING_DATA").upper()
                try:
                    status = ComplianceStatus(status_str)
                except ValueError:
                    status = ComplianceStatus.MISSING_DATA

                # Map risk level
                risk_str = item.get("risk_level", "medium").lower()
                try:
                    risk_level = RiskLevel(risk_str)
                except ValueError:
                    risk_level = RiskLevel.MEDIUM

                check = ComplianceCheck(
                    requirement_id=req_id,
                    requirement_description=req_lookup.get(req_id, ""),
                    status=status,
                    company_evidence=item.get("company_evidence"),
                    gap_description=item.get("gap_description"),
                    risk_level=risk_level,
                    recommendation=item.get("recommendation"),
                )
                checks.append(check)

            return checks

        except json.JSONDecodeError as e:
            logger.error("failed_to_parse_compliance", error=str(e))
            return []

    def _build_gap_analysis(
        self,
        compliance_checks: list[ComplianceCheck],
        requirements: list[ExtractedRequirement],
    ) -> GapAnalysisResult:
        """Build gap analysis from compliance checks."""
        gap_analysis = GapAnalysisResult()

        # Find requirements lookup for mandatory flag
        req_lookup = {r.requirement_id: r for r in requirements}

        for check in compliance_checks:
            req = req_lookup.get(check.requirement_id)
            is_mandatory = req.is_mandatory if req else True

            # Create gaps for non-compliant and missing data
            if check.status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.MISSING_DATA, ComplianceStatus.POTENTIAL_RISK]:

                # Determine gap category from requirement
                category = self._determine_gap_category(check.requirement_description)

                # Determine question type
                response_type = self._determine_response_type(category, check.status)

                # Generate question for user
                question = self._generate_question(check)

                gap = GapItem(
                    gap_id=f"GAP-{check.requirement_id}",
                    category=category,
                    requirement_description=check.requirement_description,
                    current_status=check.company_evidence or "No data available",
                    question_for_user=question,
                    response_type=response_type,
                )

                gap_analysis.add_gap(gap)

                # Mark critical if mandatory and non-compliant
                if is_mandatory and check.status == ComplianceStatus.NON_COMPLIANT:
                    gap_analysis.critical_gaps += 1

        return gap_analysis

    def _determine_gap_category(self, description: str) -> GapCategory:
        """Determine gap category from requirement description."""
        desc_lower = description.lower()

        if any(word in desc_lower for word in ["iso", "certification", "certified", "accredited"]):
            return GapCategory.CERTIFICATION
        elif any(word in desc_lower for word in ["experience", "years", "track record", "previous"]):
            return GapCategory.EXPERIENCE
        elif any(word in desc_lower for word in ["turnover", "revenue", "financial", "insurance"]):
            return GapCategory.FINANCIAL
        elif any(word in desc_lower for word in ["technical", "system", "software", "hardware"]):
            return GapCategory.TECHNICAL
        elif any(word in desc_lower for word in ["legal", "compliance", "regulation", "gdpr"]):
            return GapCategory.LEGAL
        else:
            return GapCategory.OTHER

    def _determine_response_type(
        self,
        category: GapCategory,
        status: ComplianceStatus,
    ) -> ResponseType:
        """Determine what type of response we need from user."""
        if status == ComplianceStatus.MISSING_DATA:
            if category == GapCategory.CERTIFICATION:
                return ResponseType.YES_NO
            elif category == GapCategory.EXPERIENCE:
                return ResponseType.PROJECT_DETAILS
            elif category == GapCategory.FINANCIAL:
                return ResponseType.NUMBER
            else:
                return ResponseType.TEXT
        else:
            return ResponseType.TEXT

    def _generate_question(self, check: ComplianceCheck) -> str:
        """Generate a question for the user about this gap."""
        if check.status == ComplianceStatus.NON_COMPLIANT:
            return f"This requirement cannot be met: '{check.requirement_description}'. Do you have any additional evidence or plans to address this?"
        elif check.status == ComplianceStatus.MISSING_DATA:
            return f"We need more information about: '{check.requirement_description}'. Can you provide details?"
        elif check.status == ComplianceStatus.POTENTIAL_RISK:
            return f"There's a potential risk with: '{check.requirement_description}'. Can you confirm or provide clarification?"
        else:
            return f"Please verify: {check.requirement_description}"


def check_tender_compliance(
    requirements: list[ExtractedRequirement],
) -> tuple[list[ComplianceCheck], GapAnalysisResult]:
    """
    Convenience function to check compliance.

    Args:
        requirements: List of requirements to check

    Returns:
        Tuple of (compliance checks, gap analysis)
    """
    agent = ComplianceAgent()
    return agent.check_compliance(requirements)
