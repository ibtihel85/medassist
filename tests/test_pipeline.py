
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain.docstore.document import Document

from src.generation.schemas import (
    AgentDecision,
    GeneratedAnswer,
    RewrittenQuery,
    parse_json_response,
)
from src.utils.confidence import compute_confidence, format_citations
from src.utils.constants import QUICK_DEFINITIONS


# ── parse_json_response ───────────────────────────────────────────────

class TestParseJsonResponse:
    def test_valid_agent_decision(self):
        raw = '{"needs_rewrite": false, "tool": "literature_search", "reasoning": "clear question"}'
        result = parse_json_response(raw, AgentDecision)
        assert result is not None
        assert result["tool"] == "literature_search"
        assert result["needs_rewrite"] is False

    def test_strips_markdown_fences(self):
        raw = "```json\n{\"needs_rewrite\": true, \"tool\": \"quick_definition\", \"reasoning\": \"x\"}\n```"
        result = parse_json_response(raw, AgentDecision)
        assert result is not None
        assert result["needs_rewrite"] is True

    def test_returns_none_on_missing_json(self):
        assert parse_json_response("no json here", AgentDecision) is None

    def test_returns_none_on_empty_string(self):
        assert parse_json_response("", AgentDecision) is None

    def test_valid_rewritten_query(self):
        raw = '{"rewritten_query": "effects of metformin on HbA1c", "reasoning": "expanded"}'
        result = parse_json_response(raw, RewrittenQuery)
        assert result["rewritten_query"] == "effects of metformin on HbA1c"


# ── QUICK_DEFINITIONS ─────────────────────────────────────────────────

class TestQuickDefinitions:
    def test_rct_defined(self):
        assert "rct" in QUICK_DEFINITIONS
        assert "Randomized" in QUICK_DEFINITIONS["rct"]

    def test_p_value_defined(self):
        assert "p-value" in QUICK_DEFINITIONS

    def test_sensitivity_defined(self):
        assert "sensitivity" in QUICK_DEFINITIONS


# ── compute_confidence ────────────────────────────────────────────────

class TestComputeConfidence:
    def test_empty_answer_returns_zero(self):
        docs = [Document(page_content="some context")]
        assert compute_confidence("", docs) == 0.0

    def test_empty_docs_returns_zero(self):
        assert compute_confidence("an answer", []) == 0.0

    def test_score_in_valid_range(self):
        """Smoke test: score should be between 0 and 1."""
        docs   = [Document(page_content="Metformin reduces blood glucose in type 2 diabetes.")]
        answer = "Metformin lowers blood glucose levels."
        score  = compute_confidence(answer, docs)
        assert 0.0 <= score <= 1.0


# ── format_citations ──────────────────────────────────────────────────

class TestFormatCitations:
    def test_empty_docs_returns_empty_string(self):
        assert format_citations([]) == ""

    def test_returns_sources_header(self):
        doc = Document(
            page_content="Metformin reduces glucose.",
            metadata={"source": "PubMedQA_artificial_0", "question": "What is metformin?"},
        )
        citations = format_citations([doc])
        assert "SOURCES:" in citations
        assert "PubMedQA_artificial_0" in citations

    def test_deduplicates_identical_docs(self):
        doc = Document(page_content="x" * 200, metadata={"source": "src_1"})
        citations = format_citations([doc, doc])
        # Should appear only once
        assert citations.count("[1]") == 1
