

from __future__ import annotations

from typing import List, Optional

from langchain.docstore.document import Document
from typing_extensions import TypedDict


class MedAssistState(TypedDict):
    
    # ── Input ─────────────────────────────────────────────────────────
    question:         str

    # ── Query rewriting ───────────────────────────────────────────────
    rewritten_query:  Optional[str]
    needs_rewrite:    Optional[bool]

    # ── Retrieval ─────────────────────────────────────────────────────
    documents:        Optional[List[Document]]

    # ── Generation ────────────────────────────────────────────────────
    raw_answer:       Optional[str]
    key_findings:     Optional[List[str]]
    has_enough_info:  Optional[bool]

    # ── Evaluation ────────────────────────────────────────────────────
    confidence:       Optional[float]
    is_grounded:      Optional[bool]

    # ── Control ───────────────────────────────────────────────────────
    retry_count:      int
    tool:             Optional[str]

    # ── Output ────────────────────────────────────────────────────────
    final_answer:     Optional[str]
    error:            Optional[str]


def empty_state(question: str) -> MedAssistState:
    
    return MedAssistState(
        question=question,
        rewritten_query=None,
        needs_rewrite=None,
        documents=None,
        raw_answer=None,
        key_findings=None,
        has_enough_info=None,
        confidence=None,
        is_grounded=None,
        retry_count=0,
        tool=None,
        final_answer=None,
        error=None,
    )
