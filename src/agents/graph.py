

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from src.agents.nodes import (
    make_node_agent_decide,
    make_node_generate_answer,
    make_node_retrieve,
    make_node_rewrite_query,
    node_evaluate,
    node_quick_definition,
)
from src.agents.state import MedAssistState
from src.config.settings import settings
from src.generation.llm import get_llm
from src.retrieval.faiss_store import load_faiss_retriever

logger = logging.getLogger(__name__)


# ── Conditional edge functions ────────────────────────────────────────

def route_after_decide(state: MedAssistState) -> str:
    if state.get("tool") == "quick_definition":
        return "quick_definition"
    if state.get("needs_rewrite"):
        return "rewrite_query"
    return "retrieve"


def route_after_quick_def(state: MedAssistState) -> str:
    
    if state.get("tool") == "literature_search":
        return "retrieve"
    return END


def route_after_evaluate(state: MedAssistState) -> str:
    
    score   = state.get("confidence", 1.0)
    retries = state.get("retry_count", 0)

    if score < settings.confidence.threshold and retries < settings.agent.max_retries:
        logger.info("[ROUTE] Low confidence (%.3f) — retry #%d.", score, retries)
        return "rewrite_query"
    return END



def build_graph(faiss_index_path: str | None = None):
    
    logger.info("Building MedAssist graph …")

    llm       = get_llm()
    retriever = load_faiss_retriever(faiss_index_path)

    # Create node functions via factories (dependency injection via closure)
    _agent_decide    = make_node_agent_decide(llm)
    _rewrite_query   = make_node_rewrite_query(llm)
    _retrieve        = make_node_retrieve(retriever)
    _generate_answer = make_node_generate_answer(llm)

    builder = StateGraph(MedAssistState)

    
    builder.add_node("agent_decide",     _agent_decide)
    builder.add_node("rewrite_query",    _rewrite_query)
    builder.add_node("retrieve",         _retrieve)
    builder.add_node("quick_definition", node_quick_definition)
    builder.add_node("generate_answer",  _generate_answer)
    builder.add_node("evaluate",         node_evaluate)

    # ── Entry point ───────────────────────────────────────────────────
    builder.set_entry_point("agent_decide")

    # ── Conditional edges ─────────────────────────────────────────────
    builder.add_conditional_edges(
        "agent_decide",
        route_after_decide,
        {
            "quick_definition": "quick_definition",
            "rewrite_query":    "rewrite_query",
            "retrieve":         "retrieve",
        },
    )

    builder.add_conditional_edges(
        "quick_definition",
        route_after_quick_def,
        {
            "retrieve": "retrieve",
            END:        END,
        },
    )

    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "rewrite_query": "rewrite_query",
            END:             END,
        },
    )

    # ── Linear edges ──────────────────────────────────────────────────
    builder.add_edge("rewrite_query",   "retrieve")
    builder.add_edge("retrieve",        "generate_answer")
    builder.add_edge("generate_answer", "evaluate")

    graph = builder.compile()
    logger.info(
        "Graph compiled — nodes: %s",
        list(graph.get_graph().nodes.keys()),
    )
    return graph
