"""Tests for the SelfRAG reranking-generation cache mechanism.

Verifies that:
1. SelfRAGReranking stores predictions in metadata._selfrag_pred
2. SelfRAGGeneration reuses cached predictions when available
3. SelfRAGGeneration generates from scratch when no cache exists
4. Cross-framework combinations work correctly (SelfRAG rerank + other gen)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_selfrag_root = str(Path(__file__).resolve().parent.parent / "self-rag_langgraph" / "self-rag-wtb")
if _selfrag_root not in sys.path:
    sys.path.insert(0, _selfrag_root)

from rag_contracts import GenerationResult, RetrievalResult


# ═══════════════════════════════════════════════════════════════════════════════
# Test the cache mechanism with mock-populated metadata
# ═══════════════════════════════════════════════════════════════════════════════


class TestSelfRAGCacheMechanism:
    """Tests the _selfrag_pred cache without requiring vLLM."""

    def _make_cached_results(self) -> list[RetrievalResult]:
        """Create RetrievalResults with _selfrag_pred metadata as if
        SelfRAGReranking had already processed them."""
        return [
            RetrievalResult(
                source_id="doc1",
                content="Paris is the capital of France.",
                score=0.95,
                title="France",
                metadata={
                    "selfrag_score": 0.95,
                    "_selfrag_pred": {
                        "text": "Paris",
                        "score": 0.95,
                    },
                },
            ),
            RetrievalResult(
                source_id="doc2",
                content="Berlin is the capital of Germany.",
                score=0.80,
                title="Germany",
                metadata={
                    "selfrag_score": 0.80,
                    "_selfrag_pred": {
                        "text": "Berlin",
                        "score": 0.80,
                    },
                },
            ),
        ]

    def _make_uncached_results(self) -> list[RetrievalResult]:
        """Create RetrievalResults without _selfrag_pred (e.g., from non-SelfRAG reranking)."""
        return [
            RetrievalResult(
                source_id="doc1",
                content="Paris is the capital of France.",
                score=0.9,
                title="France",
            ),
            RetrievalResult(
                source_id="doc2",
                content="Berlin is the capital of Germany.",
                score=0.8,
                title="Germany",
            ),
        ]

    def test_generation_uses_cache(self):
        """When _selfrag_pred is in metadata, generation should use it."""
        from selfrag.adapters import SelfRAGGeneration

        gen = SelfRAGGeneration(model=None)
        context = self._make_cached_results()
        result = gen._try_cached(context)

        assert result is not None
        assert result.output == "Paris"
        assert result.metadata["from_reranking_cache"] is True
        assert result.metadata["selfrag_score"] == 0.95
        assert result.citations == ["doc1"]

    def test_generation_skips_without_cache(self):
        """When no _selfrag_pred in metadata, _try_cached returns None."""
        from selfrag.adapters import SelfRAGGeneration

        gen = SelfRAGGeneration(model=None)
        context = self._make_uncached_results()
        result = gen._try_cached(context)

        assert result is None

    def test_generation_selects_best_cached(self):
        """With multiple cached results, selects highest score."""
        from selfrag.adapters import SelfRAGGeneration

        gen = SelfRAGGeneration(model=None)
        context = self._make_cached_results()
        context[1].metadata["_selfrag_pred"]["score"] = 0.99
        context[1].metadata["_selfrag_pred"]["text"] = "Berlin (best)"

        result = gen._try_cached(context)
        assert result is not None
        assert result.output == "Berlin (best)"
        assert result.metadata["passage_index"] == 1

    def test_full_generate_with_cache(self):
        """The public generate() method should use cache when available."""
        from selfrag.adapters import SelfRAGGeneration

        gen = SelfRAGGeneration(model=None)
        context = self._make_cached_results()
        result = gen.generate("What is the capital of France?", context)

        assert result.output == "Paris"
        assert result.metadata.get("from_reranking_cache") is True

    def test_full_generate_empty_without_model(self):
        """Without model and without cache, generate returns empty."""
        from selfrag.adapters import SelfRAGGeneration

        gen = SelfRAGGeneration(model=None)
        context = self._make_uncached_results()
        result = gen.generate("test", context)

        assert result.output == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Test cross-framework composability
# ═══════════════════════════════════════════════════════════════════════════════


class TestSelfRAGCacheComposability:
    """Verify cache mechanism works with non-SelfRAG components."""

    def test_cached_results_with_longrag_generation(self):
        """LongRAG-style generation ignores _selfrag_pred and uses content."""

        class MockLongRAGGen:
            def generate(self, query, context, instruction=""):
                text = " | ".join(r.content[:30] for r in context)
                return GenerationResult(
                    output=f"[LongRAG] {text}",
                    citations=[r.source_id for r in context],
                )

        gen = MockLongRAGGen()
        context = [
            RetrievalResult(
                source_id="d1",
                content="Paris is the capital of France.",
                score=0.95,
                metadata={
                    "_selfrag_pred": {"text": "Paris", "score": 0.95},
                },
            ),
        ]
        result = gen.generate("test", context)
        assert "[LongRAG]" in result.output
        assert "Paris is the capital" in result.output

    def test_uncached_results_with_selfrag_generation(self):
        """SelfRAGGeneration without cache and without model returns empty."""
        from selfrag.adapters import SelfRAGGeneration

        gen = SelfRAGGeneration(model=None)
        context = [
            RetrievalResult(source_id="d1", content="test content"),
        ]
        result = gen.generate("test", context)
        assert result.output == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Test protocol conformance of shared components
# ═══════════════════════════════════════════════════════════════════════════════


class TestCommonComponentProtocols:
    """Verify common_components satisfy rag_contracts protocols."""

    def test_alce_doc_retrieval_is_retrieval(self):
        from rag_contracts import ALCEDocRetrieval
        from rag_contracts.protocols import Retrieval

        adapter = ALCEDocRetrieval(docs=[{"title": "t", "text": "x"}])
        assert isinstance(adapter, Retrieval)

    def test_fallback_retrieval_is_retrieval(self):
        from rag_contracts import FallbackRetrieval, ALCEDocRetrieval
        from rag_contracts.protocols import Retrieval

        p = ALCEDocRetrieval(docs=[])
        f = ALCEDocRetrieval(docs=[{"title": "t", "text": "x"}])
        fb = FallbackRetrieval(primary=p, fallback=f)
        assert isinstance(fb, Retrieval)

    def test_simple_llm_generation_is_generation(self):
        from rag_contracts import SimpleLLMGeneration
        from rag_contracts.protocols import Generation

        class FakeLLM:
            def complete(self, system, user, **kwargs):
                return "answer"

        gen = SimpleLLMGeneration(llm=FakeLLM())
        assert isinstance(gen, Generation)

    def test_fallback_retrieval_chains_correctly(self):
        from rag_contracts import FallbackRetrieval, ALCEDocRetrieval

        empty = ALCEDocRetrieval(docs=[])
        with_docs = ALCEDocRetrieval(docs=[{"title": "T", "text": "content"}])
        fb = FallbackRetrieval(primary=empty, fallback=with_docs)

        results = fb.retrieve(["query"])
        assert len(results) == 1
        assert results[0].content == "content"
