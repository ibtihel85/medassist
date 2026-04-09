"""
src/utils/confidence.py
─────────────────────────────────────────────────────────────────────
Grounding-confidence scoring and answer post-processing helpers.

``compute_confidence``
    Measures how well the generated answer is supported by the
    retrieved documents using mean cosine similarity.

``apply_confidence_gate``
    Applies a threshold-based gate: adds a low-confidence warning,
    or replaces the answer with a refusal, depending on the score.

``format_citations``
    Builds a structured citations block from LangChain Documents.
"""

from __future__ import annotations

import logging

import numpy as np
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config.settings import settings
from src.utils.constants import LOW_CONF_PREFIX, MEDICAL_DISCLAIMER, REFUSE_MSG

logger = logging.getLogger(__name__)

# Singleton scorer — same model as the embeddings for consistency
_scorer: SentenceTransformer | None = None


def _get_scorer() -> SentenceTransformer:
    global _scorer
    if _scorer is None:
        logger.info("Loading confidence scorer: %s", settings.model.embedding_model)
        _scorer = SentenceTransformer(settings.model.embedding_model)
    return _scorer


def compute_confidence(answer: str, source_docs: list[Document]) -> float:
    """
    Compute a grounding confidence score for *answer* w.r.t. *source_docs*.

    Encodes the answer and each document's text, then returns the mean
    cosine similarity — a proxy for how well the answer is supported by
    the retrieved evidence.

    Parameters
    ----------
    answer      : str           — The generated answer text.
    source_docs : list[Document]— Documents retrieved from the index.

    Returns
    -------
    float
        Mean cosine similarity in [0, 1].  Returns 0.0 if either input
        is empty.
    """
    if not answer or not source_docs:
        return 0.0

    scorer = _get_scorer()
    texts  = [d.page_content for d in source_docs]

    answer_emb  = scorer.encode([answer], normalize_embeddings=True)
    context_emb = scorer.encode(texts,   normalize_embeddings=True)

    score = float(np.mean(cosine_similarity(answer_emb, context_emb)[0]))
    return score


def apply_confidence_gate(
    answer: str,
    source_docs: list[Document],
) -> tuple[str, float]:
    """
    Apply the threshold-based confidence gate and append the medical
    disclaimer to the answer.

    Behaviour
    ---------
    - score < ``confidence.refuse``    → replace answer with REFUSE_MSG
    - score < ``confidence.threshold`` → prepend LOW_CONF_PREFIX
    - otherwise                        → pass through unchanged
    In all cases, MEDICAL_DISCLAIMER is appended.

    Parameters
    ----------
    answer      : str
    source_docs : list[Document]

    Returns
    -------
    (final_answer, score) : tuple[str, float]
    """
    conf = settings.confidence
    score = compute_confidence(answer, source_docs)

    if score < conf.refuse:
        final = REFUSE_MSG
    elif score < conf.threshold:
        final = LOW_CONF_PREFIX + answer
    else:
        final = answer

    final += MEDICAL_DISCLAIMER
    return final, score


def format_citations(source_docs: list[Document]) -> str:
    """
    Build a numbered citation block from retrieved source documents.

    Deduplicates by the first 100 characters of each document's content
    to avoid showing the same passage twice.

    Parameters
    ----------
    source_docs : list[Document]

    Returns
    -------
    str
        A formatted ``SOURCES:`` block, or an empty string if *source_docs*
        is empty.
    """
    if not source_docs:
        return ""

    seen: set[str] = set()
    lines: list[str] = []

    for doc in source_docs:
        key = doc.page_content[:100]
        if key in seen:
            continue
        seen.add(key)

        meta     = doc.metadata
        src      = meta.get("source",      "Unknown")
        question = meta.get("question",    "").strip()
        decision = meta.get("decision",    "").strip()
        gold     = meta.get("gold_answer", "").strip()

        line = f"  [{len(lines) + 1}] {src}"
        if question:
            line += f"\n       Q: {question[:120]}"
        if decision:
            line += f" | Decision: {decision}"
        if gold:
            line += f"\n       Summary: {gold[:150]}…"
        lines.append(line)

    return "\n\nSOURCES:\n" + "\n".join(lines)
