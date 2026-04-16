from __future__ import annotations

import logging

import numpy as np
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.50


def compute_context_precision(
    question: str,
    docs: list[Document],
    embedder: SentenceTransformer,
    threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """Fraction of retrieved docs that are relevant to the question."""
    if not docs:
        return 0.0
    q_emb = embedder.encode([question], normalize_embeddings=True)
    doc_texts = [d.page_content[:500] for d in docs]
    d_emb = embedder.encode(doc_texts, normalize_embeddings=True)
    sims = cosine_similarity(q_emb, d_emb)[0]
    return float(np.sum(sims >= threshold) / len(docs))


def compute_context_recall(
    reference_answer: str,
    docs: list[Document],
    embedder: SentenceTransformer,
) -> float:
    """Max cosine similarity between reference answer and any retrieved chunk."""
    if not docs or not reference_answer.strip():
        return 0.0
    ref_emb = embedder.encode([reference_answer], normalize_embeddings=True)
    doc_texts = [d.page_content[:500] for d in docs]
    d_emb = embedder.encode(doc_texts, normalize_embeddings=True)
    sims = cosine_similarity(ref_emb, d_emb)[0]
    return float(np.max(sims))


def compute_context_relevance_mean(
    question: str,
    docs: list[Document],
    embedder: SentenceTransformer,
) -> float:
    """Mean cosine similarity between query and all retrieved chunks."""
    if not docs:
        return 0.0
    q_emb = embedder.encode([question], normalize_embeddings=True)
    doc_texts = [d.page_content[:500] for d in docs]
    d_emb = embedder.encode(doc_texts, normalize_embeddings=True)
    return float(np.mean(cosine_similarity(q_emb, d_emb)[0]))


def compute_mrr(
    question: str,
    docs: list[Document],
    embedder: SentenceTransformer,
    threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """Mean Reciprocal Rank — 1/rank of the first relevant document."""
    if not docs:
        return 0.0
    q_emb = embedder.encode([question], normalize_embeddings=True)
    doc_texts = [d.page_content[:500] for d in docs]
    d_emb = embedder.encode(doc_texts, normalize_embeddings=True)
    sims = cosine_similarity(q_emb, d_emb)[0]
    for rank, sim in enumerate(sims, start=1):
        if sim >= threshold:
            return 1.0 / rank
    return 0.0
