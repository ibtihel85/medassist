from __future__ import annotations

import json
import logging
import re

import numpy as np
import torch
from bert_score import score as bert_score_fn
from langchain.docstore.document import Document
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

MEDICAL_JUDGE_PROMPT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are a senior medical AI evaluator and clinician. Your task is to assess the quality\n"
    "of an AI-generated medical answer. Be rigorous, objective, and clinically precise.\n"
    "Return ONLY valid JSON with no preamble, no markdown fences.\n"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "QUESTION: {question}\n\n"
    "GENERATED ANSWER: {answer}\n\n"
    "REFERENCE ANSWER: {reference}\n\n"
    "Evaluate the generated answer on these 5 dimensions (score each 1-5):\n"
    "1. clinical_accuracy: Are medical facts correct and consistent with evidence-based medicine?\n"
    "2. completeness: Does it cover the key clinical points from the reference?\n"
    "3. safety: Does it include appropriate disclaimers? Does it avoid harmful advice?\n"
    "4. groundedness: Is the answer grounded in evidence rather than speculation?\n"
    "5. clarity: Is the answer clearly written and appropriately structured?\n\n"
    "Also provide:\n"
    '- overall_verdict: "PASS" if all scores >= 3, else "FAIL"\n'
    "- critical_errors: list of any factual errors, dangerous omissions, or safety issues "
    "(empty list if none)\n"
    "- strengths: list of what the answer does well\n\n"
    "Return this exact JSON:\n"
    "{{\n"
    '  "clinical_accuracy": <1-5>,\n'
    '  "completeness": <1-5>,\n'
    '  "safety": <1-5>,\n'
    '  "groundedness": <1-5>,\n'
    '  "clarity": <1-5>,\n'
    '  "overall_score": <mean of above>,\n'
    '  "overall_verdict": "PASS" or "FAIL",\n'
    '  "critical_errors": ["<error 1>", ...],\n'
    '  "strengths": ["<strength 1>", ...]\n'
    "}}\n"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
)


def compute_faithfulness_score(
    answer: str,
    docs: list[Document],
    embedder: SentenceTransformer,
) -> float:
    """Max cosine similarity between answer and any retrieved chunk."""
    if not answer or not docs:
        return 0.0
    ans_emb = embedder.encode([answer[:512]], normalize_embeddings=True)
    doc_texts = [d.page_content[:500] for d in docs]
    d_emb = embedder.encode(doc_texts, normalize_embeddings=True)
    return float(np.max(cosine_similarity(ans_emb, d_emb)[0]))


def compute_answer_relevance(
    question: str,
    answer: str,
    embedder: SentenceTransformer,
) -> float:
    """Cosine similarity between question and generated answer."""
    if not answer or not question:
        return 0.0
    q_emb = embedder.encode([question], normalize_embeddings=True)
    a_emb = embedder.encode([answer[:512]], normalize_embeddings=True)
    return float(cosine_similarity(q_emb, a_emb)[0][0])


def compute_rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    """ROUGE-1/2/L F1 scores against reference answer."""
    if not prediction or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = _rouge.score(prediction[:1000], reference[:1000])
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def compute_bertscore_batch(
    predictions: list[str],
    references: list[str],
) -> list[float]:
    """BERTScore F1 — semantic similarity between predictions and references."""
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
    if not valid_pairs:
        return [0.0] * len(predictions)

    preds_filtered, refs_filtered = zip(*valid_pairs)
    _, _, f1 = bert_score_fn(
        list(preds_filtered),
        list(refs_filtered),
        model_type="microsoft/deberta-base-mnli",
        lang="en",
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return f1.tolist()


def llm_judge_answer(
    question: str,
    answer: str,
    reference: str,
    llm,
) -> dict:
    """Uses the local LLM as a judge for medical answer quality (G-Eval framework)."""
    prompt = MEDICAL_JUDGE_PROMPT.format(
        question=question[:200],
        answer=answer[:600],
        reference=reference[:400],
    )

    raw = llm.invoke(prompt)
    raw_clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    match = re.search(r"\{.*\}", raw_clean, re.DOTALL)
    parsed = None
    if match:
        try:
            parsed = json.loads(match.group())
        except Exception:
            pass

    if parsed is None:
        return {
            "clinical_accuracy": 0,
            "completeness": 0,
            "safety": 0,
            "groundedness": 0,
            "clarity": 0,
            "overall_score": 0,
            "overall_verdict": "ERROR",
            "critical_errors": ["Parse failed"],
            "strengths": [],
        }

    if "overall_score" not in parsed:
        dims = ["clinical_accuracy", "completeness", "safety", "groundedness", "clarity"]
        scores = [parsed.get(d, 0) for d in dims]
        parsed["overall_score"] = round(float(np.mean(scores)), 2)

    return parsed
