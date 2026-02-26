"""LangGraph workflow for tender analysis."""

from typing import TypedDict, Annotated, Literal
import structlog
from langgraph.graph import StateGraph, END

from src.clients.tender_client import TenderClient
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker
from src.rag.indexer import RAGIndexer
from src.agents.extraction import ExtractionAgent
from src.agents.compliance import ComplianceAgent
from src.agents.decision import DecisionAgent
from src.schemas.decision import (
    ExtractedRequirement,
    ComplianceCheck,
    GapAnalysisResult,
    BidDecision,
    HumanInput,
)

logger = structlog.get_logger()


class TenderAnalysisState(TypedDict):
    """State for the tender analysis workflow."""

    # Input
    tender_id: str
    tender_ocid: str | None

    # Tender data
    tender_title: str | None
    tender_description: str | None

    # Processing results
    documents_indexed: int
    requirements: list[ExtractedRequirement]
    compliance_checks: list[ComplianceCheck]
    gap_analysis: GapAnalysisResult | None

    # Human-in-the-loop
    awaiting_human_input: bool
    human_inputs: list[HumanInput]

    # Output
    decision: BidDecision | None
    error: str | None


def ingest_node(state: TenderAnalysisState) -> TenderAnalysisState:
    """
    Ingest tender data and index documents.

    Fetches tender from API, extracts text, chunks and indexes.
    """
    logger.info("ingest_node_start", tender_id=state["tender_id"])

    try:
        # Fetch tender from API
        with TenderClient() as client:
            tender_data = client.get_tender_by_ocid(state["tender_id"])

            if not tender_data:
                return {
                    **state,
                    "error": f"Tender not found: {state['tender_id']}",
                }

            # Parse tender
            releases = tender_data.get("releases", [tender_data])
            if releases:
                tender = client.parse_tender(releases[0])
            else:
                return {**state, "error": "No releases found in tender data"}

        # Create document from tender description
        loader = DocumentLoader()
        doc = loader.load_from_text(
            text=tender.description or "No description available",
            tender_id=tender.ocid,
            document_title=tender.title,
        )

        # Chunk and index
        chunker = DocumentChunker(chunk_size=500, overlap_sentences=1)
        chunks = chunker.chunk_document(doc)

        indexer = RAGIndexer()
        indexed = indexer.index_chunks(chunks)

        logger.info(
            "ingest_node_complete",
            tender_id=tender.ocid,
            chunks_indexed=indexed,
        )

        return {
            **state,
            "tender_ocid": tender.ocid,
            "tender_title": tender.title,
            "tender_description": tender.description,
            "documents_indexed": indexed,
        }

    except Exception as e:
        logger.error("ingest_node_error", error=str(e))
        return {**state, "error": f"Ingest failed: {str(e)}"}


def extract_node(state: TenderAnalysisState) -> TenderAnalysisState:
    """Extract requirements from indexed tender documents."""
    logger.info("extract_node_start", tender_id=state.get("tender_ocid"))

    if state.get("error"):
        return state

    try:
        agent = ExtractionAgent()
        requirements = agent.extract_requirements(
            tender_id=state["tender_ocid"] or state["tender_id"]
        )

        logger.info("extract_node_complete", requirements_count=len(requirements))

        return {**state, "requirements": requirements}

    except Exception as e:
        logger.error("extract_node_error", error=str(e))
        return {**state, "error": f"Extraction failed: {str(e)}"}


def compliance_node(state: TenderAnalysisState) -> TenderAnalysisState:
    """Check compliance of requirements against company profile."""
    logger.info("compliance_node_start")

    if state.get("error"):
        return state

    try:
        agent = ComplianceAgent()
        checks, gap_analysis = agent.check_compliance(state["requirements"])

        logger.info(
            "compliance_node_complete",
            checks=len(checks),
            gaps=gap_analysis.total_gaps,
        )

        return {
            **state,
            "compliance_checks": checks,
            "gap_analysis": gap_analysis,
            "awaiting_human_input": gap_analysis.requires_human_input,
        }

    except Exception as e:
        logger.error("compliance_node_error", error=str(e))
        return {**state, "error": f"Compliance check failed: {str(e)}"}


def gap_analysis_node(state: TenderAnalysisState) -> TenderAnalysisState:
    """
    Process gap analysis and determine if human input is needed.

    This node checks if there are gaps that require human attention
    before proceeding to the decision.
    """
    logger.info("gap_analysis_node_start")

    if state.get("error"):
        return state

    gap_analysis = state.get("gap_analysis")
    if not gap_analysis:
        return {**state, "awaiting_human_input": False}

    # Apply any human inputs that were provided
    for human_input in state.get("human_inputs", []):
        gap_analysis.resolve_gap(human_input.gap_id, human_input)

    # Check if all gaps are resolved
    if gap_analysis.all_resolved or not gap_analysis.requires_human_input:
        logger.info("gap_analysis_complete", all_resolved=True)
        return {**state, "awaiting_human_input": False, "gap_analysis": gap_analysis}

    # Still have unresolved gaps
    unresolved = len(gap_analysis.unresolved_gaps)
    logger.info("gap_analysis_awaiting_input", unresolved_gaps=unresolved)

    return {**state, "awaiting_human_input": True, "gap_analysis": gap_analysis}


def decision_node(state: TenderAnalysisState) -> TenderAnalysisState:
    """Generate final bid/no-bid decision."""
    logger.info("decision_node_start")

    if state.get("error"):
        return state

    try:
        agent = DecisionAgent()
        decision = agent.make_decision(
            requirements=state["requirements"],
            compliance_checks=state["compliance_checks"],
            gap_analysis=state["gap_analysis"] or GapAnalysisResult(),
        )

        logger.info(
            "decision_node_complete",
            decision=decision.decision.value,
            confidence=decision.confidence,
        )

        return {**state, "decision": decision}

    except Exception as e:
        logger.error("decision_node_error", error=str(e))
        return {**state, "error": f"Decision failed: {str(e)}"}


def should_wait_for_human(state: TenderAnalysisState) -> Literal["wait", "decide"]:
    """Determine if workflow should wait for human input or proceed to decision."""
    if state.get("error"):
        return "decide"  # Go to decision to handle error

    if state.get("awaiting_human_input", False):
        return "wait"

    return "decide"


def create_workflow() -> StateGraph:
    """
    Create the tender analysis workflow graph.

    Flow:
    1. ingest -> extract -> comply -> gap_analysis
    2. gap_analysis decides: wait (END) or decide
    3. decide -> END

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(TenderAnalysisState)

    # Add nodes
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("comply", compliance_node)
    workflow.add_node("gap_analysis", gap_analysis_node)
    workflow.add_node("decide", decision_node)

    # Set entry point
    workflow.set_entry_point("ingest")

    # Add edges
    workflow.add_edge("ingest", "extract")
    workflow.add_edge("extract", "comply")
    workflow.add_edge("comply", "gap_analysis")

    # Conditional edge from gap_analysis
    workflow.add_conditional_edges(
        "gap_analysis",
        should_wait_for_human,
        {"wait": END, "decide": "decide"},
    )

    workflow.add_edge("decide", END)

    return workflow.compile()


def analyze_tender(tender_id: str) -> TenderAnalysisState:
    """
    Analyze a tender and return the final state.

    Args:
        tender_id: OCID of the tender to analyze

    Returns:
        Final workflow state with decision
    """
    workflow = create_workflow()

    initial_state: TenderAnalysisState = {
        "tender_id": tender_id,
        "tender_ocid": None,
        "tender_title": None,
        "tender_description": None,
        "documents_indexed": 0,
        "requirements": [],
        "compliance_checks": [],
        "gap_analysis": None,
        "awaiting_human_input": False,
        "human_inputs": [],
        "decision": None,
        "error": None,
    }

    # Run workflow
    final_state = workflow.invoke(initial_state)

    return final_state


def resume_with_human_input(
    state: TenderAnalysisState,
    human_inputs: list[HumanInput],
) -> TenderAnalysisState:
    """
    Resume workflow after human provides input.

    Args:
        state: Previous workflow state (awaiting input)
        human_inputs: Human responses to gaps

    Returns:
        Updated workflow state
    """
    # Add human inputs to state
    updated_state = {
        **state,
        "human_inputs": state.get("human_inputs", []) + human_inputs,
        "awaiting_human_input": False,
    }

    # Create workflow and run from gap_analysis
    workflow = create_workflow()

    # Re-run gap analysis with new inputs
    updated_state = gap_analysis_node(updated_state)

    # If still awaiting input, return
    if updated_state.get("awaiting_human_input"):
        return updated_state

    # Otherwise proceed to decision
    final_state = decision_node(updated_state)

    return final_state
