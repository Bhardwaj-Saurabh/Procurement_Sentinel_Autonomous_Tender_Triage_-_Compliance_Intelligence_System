"""Decision agent for generating bid/no-bid recommendations."""


import json
from datetime import datetime, timezone
import structlog
from openai import AzureOpenAI

from src.config import get_settings
from src.schemas.decision import (
    ExtractedRequirement,
    ComplianceCheck,
    ComplianceStatus,
    GapAnalysisResult,
    BidDecision,
    DecisionType,
    Risk,
    RiskLevel,
    ActionItem,
)

logger = structlog.get_logger()

DECISION_PROMPT = """You are an expert bid/no-bid decision advisor for UK public procurement.

Based on the compliance analysis and gap assessment, provide a clear recommendation.

DECISION RULES:
1. If ANY mandatory requirement is NON_COMPLIANT with no mitigation → NO_BID
2. If all mandatory requirements are COMPLIANT → BID (if strategically aligned)
3. If there are POTENTIAL_RISK items that can be mitigated → CONDITIONAL_BID
4. Consider the company's risk appetite: {risk_appetite}

INPUTS:

REQUIREMENTS ({req_count} total):
{requirements}

COMPLIANCE RESULTS:
{compliance_results}

GAP ANALYSIS:
- Total gaps: {total_gaps}
- Critical gaps: {critical_gaps}
- Human inputs received: {human_inputs}

Provide your decision as JSON:
{{
  "decision": "BID" or "NO_BID" or "CONDITIONAL_BID",
  "confidence": 0.0 to 1.0,
  "headline": "Brief 10-word summary",
  "executive_summary": "2-3 sentence summary for executives",
  "strategic_fit_score": 0 to 100,
  "risk_score": 0 to 100,
  "reasoning": "Detailed explanation of the decision",
  "risks": [
    {{
      "risk_id": "RISK-001",
      "description": "Description of the risk",
      "level": "low" or "medium" or "high" or "critical",
      "mitigation": "How to address this risk"
    }}
  ],
  "action_items": [
    {{
      "priority": "urgent" or "high" or "medium" or "low",
      "description": "What needs to be done",
      "owner": "Who should do it"
    }}
  ]
}}

Return ONLY valid JSON."""


class DecisionAgent:
    """
    Agent that generates bid/no-bid decisions.

    Considers compliance results, gap analysis, and company strategy.
    """

    def __init__(self, risk_appetite: str = "moderate"):
        """
        Initialize the decision agent.

        Args:
            risk_appetite: Company risk tolerance (conservative, moderate, aggressive)
        """
        settings = get_settings()

        self.llm_client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self.deployment = settings.azure_openai_deployment
        self.risk_appetite = risk_appetite

    def make_decision(
        self,
        requirements: list[ExtractedRequirement],
        compliance_checks: list[ComplianceCheck],
        gap_analysis: GapAnalysisResult,
    ) -> BidDecision:
        """
        Generate a bid/no-bid decision.

        Args:
            requirements: Extracted requirements
            compliance_checks: Compliance check results
            gap_analysis: Gap analysis with any human inputs

        Returns:
            BidDecision with recommendation and reasoning
        """
        # Check for automatic NO_BID conditions
        auto_decision = self._check_auto_decision(compliance_checks, gap_analysis)
        if auto_decision:
            return auto_decision

        # Format inputs for LLM
        req_text = self._format_requirements(requirements)
        compliance_text = self._format_compliance(compliance_checks)
        human_inputs_text = self._format_human_inputs(gap_analysis)

        prompt = DECISION_PROMPT.format(
            risk_appetite=self.risk_appetite,
            req_count=len(requirements),
            requirements=req_text,
            compliance_results=compliance_text,
            total_gaps=gap_analysis.total_gaps,
            critical_gaps=gap_analysis.critical_gaps,
            human_inputs=human_inputs_text,
        )

        response = self.llm_client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        content = response.choices[0].message.content
        decision = self._parse_response(content, requirements, compliance_checks)

        logger.info(
            "decision_made",
            decision=decision.decision.value,
            confidence=decision.confidence,
            risk_score=decision.risk_score,
        )

        return decision

    def _check_auto_decision(
        self,
        compliance_checks: list[ComplianceCheck],
        gap_analysis: GapAnalysisResult,
    ) -> BidDecision | None:
        """Check if we can make an automatic decision without LLM."""

        # Count compliance statuses
        non_compliant_mandatory = sum(
            1 for c in compliance_checks
            if c.status == ComplianceStatus.NON_COMPLIANT
        )

        # If critical gaps and no human input provided
        if gap_analysis.critical_gaps > 0 and not gap_analysis.all_resolved:
            return BidDecision(
                decision=DecisionType.NO_BID,
                confidence=0.95,
                headline="Critical compliance gaps prevent bidding",
                executive_summary=f"There are {gap_analysis.critical_gaps} critical gaps that have not been resolved. Cannot proceed with bid.",
                strategic_fit_score=0,
                risk_score=100,
                requirements=[],
                compliance_checks=compliance_checks,
                risks=[
                    Risk(
                        risk_id="RISK-AUTO-001",
                        description="Unresolved critical compliance gaps",
                        level=RiskLevel.CRITICAL,
                        mitigation="Resolve gaps or do not bid",
                    )
                ],
                action_items=[
                    ActionItem(
                        priority="urgent",
                        description="Review and resolve critical gaps before reconsidering",
                        owner="Bid Manager",
                    )
                ],
                reasoning=f"Automatic NO_BID due to {gap_analysis.critical_gaps} unresolved critical gaps.",
                processed_at=datetime.now(timezone.utc),
            )

        return None

    def _format_requirements(self, requirements: list[ExtractedRequirement]) -> str:
        """Format requirements for prompt."""
        lines = []
        for req in requirements:
            mandatory = "MANDATORY" if req.is_mandatory else "DESIRABLE"
            lines.append(f"- {req.requirement_id} [{req.category}] ({mandatory}): {req.description}")
        return "\n".join(lines) if lines else "No requirements extracted"

    def _format_compliance(self, compliance_checks: list[ComplianceCheck]) -> str:
        """Format compliance results for prompt."""
        lines = []
        for check in compliance_checks:
            lines.append(f"- {check.requirement_id}: {check.status.value}")
            if check.company_evidence:
                lines.append(f"  Evidence: {check.company_evidence[:100]}")
            if check.gap_description:
                lines.append(f"  Gap: {check.gap_description[:100]}")
        return "\n".join(lines) if lines else "No compliance checks performed"

    def _format_human_inputs(self, gap_analysis: GapAnalysisResult) -> str:
        """Format human inputs for prompt."""
        if not gap_analysis.human_inputs:
            return "None provided"

        lines = []
        for inp in gap_analysis.human_inputs:
            lines.append(f"- Gap {inp.gap_id}: {inp.response_value}")
            if inp.supporting_evidence:
                lines.append(f"  Evidence: {inp.supporting_evidence}")
        return "\n".join(lines)

    def _parse_response(
        self,
        content: str,
        requirements: list[ExtractedRequirement],
        compliance_checks: list[ComplianceCheck],
    ) -> BidDecision:
        """Parse LLM response into BidDecision."""
        try:
            data = json.loads(content)

            # Parse decision type
            decision_str = data.get("decision", "NO_BID").upper()
            try:
                decision_type = DecisionType(decision_str)
            except ValueError:
                decision_type = DecisionType.NO_BID

            # Parse risks
            risks = []
            for r in data.get("risks", []):
                try:
                    level = RiskLevel(r.get("level", "medium").lower())
                except ValueError:
                    level = RiskLevel.MEDIUM

                risks.append(Risk(
                    risk_id=r.get("risk_id", f"RISK-{len(risks)+1:03d}"),
                    description=r.get("description", ""),
                    level=level,
                    mitigation=r.get("mitigation"),
                ))

            # Parse action items
            action_items = []
            for a in data.get("action_items", []):
                action_items.append(ActionItem(
                    priority=a.get("priority", "medium"),
                    description=a.get("description", ""),
                    owner=a.get("owner"),
                ))

            return BidDecision(
                decision=decision_type,
                confidence=float(data.get("confidence", 0.5)),
                headline=data.get("headline", "Decision generated"),
                executive_summary=data.get("executive_summary", ""),
                strategic_fit_score=int(data.get("strategic_fit_score", 50)),
                risk_score=int(data.get("risk_score", 50)),
                requirements=requirements,
                compliance_checks=compliance_checks,
                risks=risks,
                action_items=action_items,
                reasoning=data.get("reasoning", ""),
                processed_at=datetime.now(timezone.utc),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.error("failed_to_parse_decision", error=str(e))
            return BidDecision(
                decision=DecisionType.NO_BID,
                confidence=0.0,
                headline="Error in decision processing",
                executive_summary="Failed to generate decision due to processing error.",
                strategic_fit_score=0,
                risk_score=100,
                requirements=requirements,
                compliance_checks=compliance_checks,
                risks=[],
                action_items=[],
                reasoning=f"Error parsing LLM response: {str(e)}",
                processed_at=datetime.now(timezone.utc),
            )


def make_bid_decision(
    requirements: list[ExtractedRequirement],
    compliance_checks: list[ComplianceCheck],
    gap_analysis: GapAnalysisResult,
) -> BidDecision:
    """
    Convenience function to make a bid decision.

    Args:
        requirements: Extracted requirements
        compliance_checks: Compliance results
        gap_analysis: Gap analysis

    Returns:
        BidDecision
    """
    agent = DecisionAgent()
    return agent.make_decision(requirements, compliance_checks, gap_analysis)