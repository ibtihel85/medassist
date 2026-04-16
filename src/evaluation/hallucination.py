from __future__ import annotations

import re

import numpy as np
from langchain.docstore.document import Document
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

HALLUCINATION_THRESHOLD = 0.45
HALLUCINATION_WARN = 0.15
HALLUCINATION_RISK = 0.30


def detect_hallucinations_sentence_level(
    answer: str,
    docs: list[Document],
    embedder: SentenceTransformer,
    threshold: float = HALLUCINATION_THRESHOLD,
) -> dict:
    """
    Sentence-level hallucination detection.

    For each sentence in the generated answer, computes cosine similarity
    against all retrieved context chunks. Sentences with max similarity
    below threshold are flagged as unsupported (hallucination candidates).

    Returns:
        hallucination_rate: float [0, 1]
        unsupported_sentences: list[str]
        supported_sentences: list[str]
        sentence_sim_scores: list[float]
        n_sentences: int
    """
    if not answer or not docs:
        return {
            "hallucination_rate": 1.0,
            "unsupported_sentences": [],
            "supported_sentences": [],
            "sentence_sim_scores": [],
            "n_sentences": 0,
        }

    # Strip disclaimer and low-confidence prefix blocks
    clean_answer = re.sub(r"---\n⚕️.*$", "", answer, flags=re.DOTALL).strip()
    clean_answer = re.sub(r"⚠️.*?\n\n", "", clean_answer).strip()

    sentences = [s.strip() for s in sent_tokenize(clean_answer) if len(s.strip()) > 20]
    if not sentences:
        return {
            "hallucination_rate": 0.0,
            "unsupported_sentences": [],
            "supported_sentences": [],
            "sentence_sim_scores": [],
            "n_sentences": 0,
        }

    doc_texts = [d.page_content[:500] for d in docs]
    sent_embs = embedder.encode(sentences, normalize_embeddings=True)
    doc_embs = embedder.encode(doc_texts, normalize_embeddings=True)

    sim_matrix = cosine_similarity(sent_embs, doc_embs)
    max_sims = sim_matrix.max(axis=1)

    unsupported = [s for s, sim in zip(sentences, max_sims) if sim < threshold]
    supported = [s for s, sim in zip(sentences, max_sims) if sim >= threshold]

    return {
        "hallucination_rate": len(unsupported) / len(sentences),
        "unsupported_sentences": unsupported,
        "supported_sentences": supported,
        "sentence_sim_scores": max_sims.tolist(),
        "n_sentences": len(sentences),
    }
