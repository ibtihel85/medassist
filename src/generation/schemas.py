from __future__ import annotations

import json
import logging
import re
from typing import List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Per-node output schemas ───────────────────────────────────────────


class AgentDecision(BaseModel):
    needs_rewrite: bool = Field(
        description="True if query is vague or ambiguous and should be rewritten."
    )
    tool: str = Field(
        description="Tool to call: 'literature_search' or 'quick_definition'."
    )
    reasoning: str = Field(description="One-sentence justification for the decision.")


class RewrittenQuery(BaseModel):
    rewritten_query: str = Field(
        description="Specific, retrieval-optimised medical question in plain English."
    )
    reasoning: str = Field(description="Explanation of what was vague and how it was fixed.")


class GeneratedAnswer(BaseModel):
   

    answer: str = Field(
        description="Factual answer grounded strictly in the retrieved context."
    )
    key_findings: List[str] = Field(
        description="Up to three bullet-point key findings extracted from the abstracts."
    )
    has_enough_info: bool = Field(
        description="False if the context is insufficient to answer the question reliably."
    )


class EvaluationResult(BaseModel):
    
    confidence: float = Field(description="Grounding score 0–1 (cosine similarity).")
    is_grounded: bool = Field(
        description="True if confidence meets or exceeds the configured threshold."
    )
    should_retry: bool = Field(
        description="True if confidence is low and retry budget remains."
    )
    verdict: str = Field(description="One-sentence quality verdict.")


# ── JSON extraction helper ────────────────────────────────────────────


def parse_json_response(text: str, schema: type) -> dict | None:
    
    if not text:
        return None

    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")

    # Extract the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        logger.debug("parse_json_response: no JSON block found in LLM output.")
        return None

    try:
        data      = json.loads(match.group())
        validated = schema(**data)
        return validated.model_dump()
    except Exception as exc:
        logger.debug("parse_json_response: validation failed — %s", exc)
        return None
