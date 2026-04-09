

from __future__ import annotations

import logging

import numpy as np
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config.settings import settings
from src.utils.constants import LOW_CONF_PREFIX, MEDICAL_DISCLAIMER, REFUSE_MSG

logger = logging.getLogger(__name__)


_scorer: SentenceTransformer | None = None


def _get_scorer() -> SentenceTransformer:
    global _scorer
    if _scorer is None:
        logger.info("Loading confidence scorer: %s", settings.model.embedding_model)
        _scorer = SentenceTransformer(settings.model.embedding_model)
    return _scorer


def compute_confidence(answer: str, source_docs: list[Document]) -> float:
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
