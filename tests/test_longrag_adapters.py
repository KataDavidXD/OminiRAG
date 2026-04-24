"""Unit tests for LongRAG adapters."""

from __future__ import annotations

import pytest

from rag_contracts import GenerationResult, RetrievalResult
from longRAG_example.longrag_langgraph.adapters import LongRAGGeneration


# ═══════════════════════════════════════════════════════════════════════════════
# LongRAGGeneration adapter
# ═══════════════════════════════════════════════════════════════════════════════


class _FakeLLMInference:
    """Mimics GPTInference / ClaudeInference interface."""

    def predict_nq(self, context, query, titles):
        return (f"long answer about {query}", f"short-{query}")

    def predict_hotpotqa(self, context, query, titles):
        return (f"hotpot long answer about {query}", f"hotpot-{query}")


class TestLongRAGGeneration:
    def test_nq_generation(self):
        adapter = LongRAGGeneration(llm_inference=_FakeLLMInference())
        ctx = [
            RetrievalResult(
                source_id="s1",
                content="some context",
                title="T1",
                metadata={"context_titles": ["T1"]},
            )
        ]
        result = adapter.generate("who is X?", ctx, instruction="dataset=nq")
        assert isinstance(result, GenerationResult)
        assert result.output == "short-who is X?"
        assert result.metadata["dataset"] == "nq"
        assert result.metadata["long_answer"] == "long answer about who is X?"

    def test_hotpotqa_generation(self):
        adapter = LongRAGGeneration(llm_inference=_FakeLLMInference())
        result = adapter.generate(
            "where is Y?",
            [RetrievalResult(source_id="s1", content="ctx")],
            instruction="dataset=hotpotqa",
        )
        assert result.output == "hotpot-where is Y?"
        assert result.metadata["dataset"] == "hotpotqa"

    def test_default_is_nq_without_keyword(self):
        adapter = LongRAGGeneration(llm_inference=_FakeLLMInference())
        result = adapter.generate("q?", [], instruction="")
        assert result.metadata["dataset"] == "nq"

    def test_handles_exception(self):
        class _BadLLM:
            def predict_nq(self, *args):
                raise ValueError("LLM error")

        adapter = LongRAGGeneration(llm_inference=_BadLLM())
        result = adapter.generate("q?", [], instruction="")
        assert result.output == ""
        assert result.metadata["long_answer"] == ""

    def test_citations_from_context(self):
        adapter = LongRAGGeneration(llm_inference=_FakeLLMInference())
        ctx = [
            RetrievalResult(source_id="a", content="c"),
            RetrievalResult(source_id="b", content="c"),
        ]
        result = adapter.generate("q", ctx)
        assert result.citations == ["a", "b"]

    def test_titles_from_metadata(self):
        adapter = LongRAGGeneration(llm_inference=_FakeLLMInference())
        ctx = [
            RetrievalResult(
                source_id="s1",
                content="c",
                title="fallback",
                metadata={"context_titles": ["real_title"]},
            )
        ]
        result = adapter.generate("q", ctx)
        assert result.output == "short-q"
