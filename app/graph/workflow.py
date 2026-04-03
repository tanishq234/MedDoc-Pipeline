"""
workflow.py - LangGraph pipeline:
START -> segregator -> [id_agent, discharge_agent, bill_agent] -> aggregator -> END
"""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional, TypedDict
from langgraph.graph import StateGraph, END, START
from app.models import ClaimState
from app.agents import run_segregator, run_id_agent, run_discharge_agent, run_bill_agent

logger = logging.getLogger(__name__)

class PipelineState(TypedDict, total=False):
    claim_id: str
    pdf_bytes: bytes
    total_pages: int
    page_classification: Dict[str, list]
    identity_data: Optional[Dict[str, Any]]
    discharge_data: Optional[Dict[str, Any]]
    bill_data: Optional[Dict[str, Any]]
    final_output: Optional[Dict[str, Any]]

def segregator_node(state: PipelineState) -> dict:
    s = ClaimState(claim_id=state["claim_id"], pdf_bytes=state["pdf_bytes"])
    result = run_segregator(s)
    return {"total_pages": result.total_pages, "page_classification": result.page_classification}

def id_node(state: PipelineState) -> dict:
    s = ClaimState(
        claim_id=state["claim_id"], pdf_bytes=state["pdf_bytes"],
        total_pages=state.get("total_pages", 0),
        page_classification=state.get("page_classification", {}),
    )
    result = run_id_agent(s)
    return {"identity_data": result.identity_data}

def discharge_node(state: PipelineState) -> dict:
    s = ClaimState(
        claim_id=state["claim_id"], pdf_bytes=state["pdf_bytes"],
        total_pages=state.get("total_pages", 0),
        page_classification=state.get("page_classification", {}),
    )
    result = run_discharge_agent(s)
    return {"discharge_data": result.discharge_data}

def bill_node(state: PipelineState) -> dict:
    s = ClaimState(
        claim_id=state["claim_id"], pdf_bytes=state["pdf_bytes"],
        total_pages=state.get("total_pages", 0),
        page_classification=state.get("page_classification", {}),
    )
    result = run_bill_agent(s)
    return {"bill_data": result.bill_data}

def aggregator_node(state: PipelineState) -> dict:
    logger.info("[Aggregator] Combining results for claim %s", state.get("claim_id"))
    identity  = state.get("identity_data")
    discharge = state.get("discharge_data")
    bill      = state.get("bill_data")

    agents_invoked = []
    if identity  and "note" not in identity  and "error" not in identity:  agents_invoked.append("id_agent")
    if discharge and "note" not in discharge and "error" not in discharge: agents_invoked.append("discharge_summary_agent")
    if bill      and "note" not in bill      and "error" not in bill:      agents_invoked.append("itemized_bill_agent")

    final = {
        "claim_id": state.get("claim_id"),
        "status": "success",
        "page_classification": state.get("page_classification", {}),
        "extracted_data": {
            "identity":          identity,
            "discharge_summary": discharge,
            "itemized_bill":     bill,
        },
        "processing_metadata": {
            "total_pages":    state.get("total_pages", 0),
            "agents_invoked": agents_invoked,
            "model_used":     "gemini-2.0-flash-lite",
        },
    }
    logger.info("[Aggregator] Done - agents: %s", agents_invoked)
    return {"final_output": final}

def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("segregator",      segregator_node)
    g.add_node("id_agent",        id_node)
    g.add_node("discharge_agent", discharge_node)
    g.add_node("bill_agent",      bill_node)
    g.add_node("aggregator",      aggregator_node)

    g.add_edge(START,            "segregator")
    g.add_edge("segregator",     "id_agent")
    g.add_edge("segregator",     "discharge_agent")
    g.add_edge("segregator",     "bill_agent")
    g.add_edge("id_agent",       "aggregator")
    g.add_edge("discharge_agent","aggregator")
    g.add_edge("bill_agent",     "aggregator")
    g.add_edge("aggregator",      END)
    return g.compile()

def run_claim_pipeline(claim_id: str, pdf_bytes: bytes) -> Dict[str, Any]:
    graph = build_graph()
    result = graph.invoke({"claim_id": claim_id, "pdf_bytes": pdf_bytes})
    output = result.get("final_output")
    if not output:
        raise RuntimeError("Pipeline produced no output")
    return output
