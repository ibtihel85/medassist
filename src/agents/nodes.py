"""

Node overview
─────────────
  node_agent_decide      — classify query, choose tool, flag rewrite
  node_rewrite_query     — expand vague queries into retrieval-ready form
  node_retrieve          — fetch top-k documents from FAISS (MMR)
  node_quick_definition  — instant lookup for medical/statistical terms
  node_generate_answer   — synthesise a grounded answer from abstracts
  node_evaluate          — score grounding confidence, build final answer
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from src.agents.state import MedAssistState
from src.config.settings import settings
from src.generation.prompts import generation_prompt, rewrite_prompt, routing_prompt
from src.generation.schemas import (
    AgentDecision,
    GeneratedAnswer,
    RewrittenQuery,
    parse_json_response,
)
from src.utils.confidence import apply_confidence_gate, compute_confidence, format_citations
from src.utils.constants import MEDICAL_DISCLAIMER, QUICK_DEFINITIONS, REFUSE_MSG

logger = logging.getLogger(__name__)


def make_node_agent_decide(llm):
    

    def node_agent_decide(state: MedAssistState) -> dict:
        
        question = state["question"]
        logger.info("[DECIDE] %s", question[:80])

        raw    = llm.invoke(routing_prompt(question))
        parsed = parse_json_response(raw, AgentDecision)

        if parsed is None:
            # Heuristic fallback
            words         = question.lower().split()
            vague_signals = {"it", "they", "this", "that", "those", "these"}
            def_signals   = {"what", "define", "definition", "mean", "meaning"}
            needs_rewrite = bool(vague_signals & set(words)) or len(words) < 5
            tool = (
                "quick_definition"
                if (len(words) <= 6 and def_signals & set(words))
                else "literature_search"
            )
            logger.warning(
                "[DECIDE] JSON parse failed — heuristic: tool=%s, rewrite=%s",
                tool, needs_rewrite,
            )
            return {"needs_rewrite": needs_rewrite, "tool": tool}

        logger.info(
            "[DECIDE] tool=%s | rewrite=%s | %s",
            parsed["tool"], parsed["needs_rewrite"], parsed["reasoning"],
        )
        return {"needs_rewrite": parsed["needs_rewrite"], "tool": parsed["tool"]}

    return node_agent_decide


def make_node_rewrite_query(llm):
    

    def node_rewrite_query(state: MedAssistState) -> dict:
        
        question = state["question"]
        logger.info("[REWRITE] Original: %s", question)

        raw    = llm.invoke(rewrite_prompt(question))
        parsed = parse_json_response(raw, RewrittenQuery)

        if parsed is None or not parsed.get("rewritten_query"):
            logger.warning("[REWRITE] Failed — keeping original question.")
            return {"rewritten_query": question}

        rewritten = parsed["rewritten_query"]
        # Strip any boolean operators that slipped through
        for operator in [" AND ", " OR ", " NOT ", "[", "]"]:
            rewritten = rewritten.replace(operator, " ")
        rewritten = rewritten.strip()

        logger.info("[REWRITE] → %s", rewritten)
        return {"rewritten_query": rewritten}

    return node_rewrite_query


def make_node_retrieve(faiss_retriever):
    

    def node_retrieve(state: MedAssistState) -> dict:
        
        query = state.get("rewritten_query") or state["question"]
        logger.info("[RETRIEVE] %s", query[:80])

        try:
            docs = faiss_retriever.get_relevant_documents(query)
            logger.info("[RETRIEVE] %d documents retrieved.", len(docs))
            return {"documents": docs}
        except Exception as exc:
            logger.error("[RETRIEVE] Error: %s", exc)
            return {"documents": [], "error": str(exc)}

    return node_retrieve


def node_quick_definition(state: MedAssistState) -> dict:
    raw_term = state["question"].lower().strip()

    # Strip common question prefixes to isolate the core term
    for stopword in [
        "what is", "what does", "define", "definition of",
        "mean", "meaning of", "?",
    ]:
        raw_term = raw_term.replace(stopword, "").strip()

    # Exact match
    if raw_term in QUICK_DEFINITIONS:
        answer = f"DEFINITION — {raw_term.upper()}:\n{QUICK_DEFINITIONS[raw_term]}"
    else:
        # Partial match (substring)
        match = next(
            (v for k, v in QUICK_DEFINITIONS.items() if k in raw_term or raw_term in k),
            None,
        )
        if match:
            answer = f"DEFINITION (closest match):\n{match}"
        else:
            # Term unknown — escalate to full literature search
            logger.info(
                "[QUICK DEF] Term '%s' not found — escalating to literature_search.", raw_term
            )
            return {"tool": "literature_search", "raw_answer": None}

    logger.info("[QUICK DEF] %s", answer[:80])
    return {
        "raw_answer":      answer,
        "final_answer":    answer + MEDICAL_DISCLAIMER,
        "confidence":      1.0,
        "is_grounded":     True,
        "has_enough_info": True,
        "key_findings":    [answer],
    }


def make_node_generate_answer(llm):
    

    def node_generate_answer(state: MedAssistState) -> dict:
       
        question = state.get("rewritten_query") or state["question"]
        docs     = state.get("documents") or []
        logger.info("[GENERATE] Synthesising from %d docs …", len(docs))

        if not docs:
            return {
                "raw_answer":      REFUSE_MSG,
                "has_enough_info": False,
                "key_findings":    [],
            }

        # Build context from top-3 docs
        context_parts = [
            f"[Abstract {i}] Source: {doc.metadata.get('source', '?')}\n"
            f"{doc.page_content[:600]}"
            for i, doc in enumerate(docs[:3], 1)
        ]
        context = "\n\n".join(context_parts)

        raw    = llm.invoke(generation_prompt(question, context))
        parsed = parse_json_response(raw, GeneratedAnswer)

        if parsed is None:
            # Fallback: strip special tokens and use raw output
            clean = re.sub(r"<\|.*?\|>", "", raw).strip()
            logger.warning("[GENERATE] JSON parse failed — using raw text.")
            return {
                "raw_answer":      clean[:1000] if clean else REFUSE_MSG,
                "has_enough_info": bool(clean),
                "key_findings":    [],
            }

        logger.info("[GENERATE] has_enough_info=%s", parsed["has_enough_info"])
        return {
            "raw_answer":      parsed["answer"],
            "has_enough_info": parsed["has_enough_info"],
            "key_findings":    parsed["key_findings"],
        }

    return node_generate_answer


def node_evaluate(state: MedAssistState) -> dict:
    
    answer  = state.get("raw_answer") or ""
    docs    = state.get("documents") or []
    retries = state.get("retry_count", 0)

    score       = compute_confidence(answer, docs)
    is_grounded = score >= settings.confidence.threshold

    logger.info(
        "[EVALUATE] confidence=%.3f | grounded=%s | retries_used=%d",
        score, is_grounded, retries,
    )

    # Build the final, user-facing answer
    gated_answer, _ = apply_confidence_gate(answer, docs)
    citations       = format_citations(docs)
    final           = gated_answer + citations

    should_retry = (not is_grounded) and (retries < settings.agent.max_retries)

    return {
        "confidence":   score,
        "is_grounded":  is_grounded,
        "final_answer": final,
        "retry_count":  retries + (1 if should_retry else 0),
    }
