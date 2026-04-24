"""Cross-project swap tests for LightRAG components.

Verifies that LightRAG components can be swapped with LongRAG/Self-RAG
components in both directions:
  - LightRAG retrieval in LongRAG pipeline
  - LongRAG generation in LightRAG pipeline
  - Self-RAG reranking in LightRAG pipeline
  - Mixed 3-way configurations
"""

from __future__ import annotations

import asyncio

import pytest

from rag_contracts import (
    GenerationResult,
    IdentityReranking,
    QueryContext,
    RetrievalResult,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Mock components simulating each framework's behavior
# ═══════════════════════════════════════════════════════════════════════════════


class MockLightRAGQuery:
    """Simulates LightRAG's keyword-based query expansion."""

    NAME = "lightrag-keywords"

    def process(self, query: str, context: QueryContext) -> list[str]:
        return [query, f"{query} entity", f"{query} relation"]


class MockLightRAGRetrieval:
    """Simulates LightRAG's hybrid vector+KG retrieval."""

    NAME = "lightrag-hybrid"

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        results = []
        for i, q in enumerate(queries):
            results.append(RetrievalResult(
                source_id=f"lightrag-chunk://{i}",
                content=f"KG-enriched context for {q}",
                score=0.9 - i * 0.05,
                title=f"LightRAG-KG-Doc-{i}",
                metadata={"retrieval_mode": "hybrid", "has_kg": True},
            ))
        return results[:top_k]


class MockLightRAGReranking:
    """Simulates LightRAG's context compression as reranking."""

    NAME = "lightrag-compress"

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 10
    ) -> list[RetrievalResult]:
        compressed = f"Compressed evidence for: {query}"
        return [
            RetrievalResult(
                source_id=r.source_id,
                content=r.content,
                score=r.score,
                title=r.title,
                metadata={**r.metadata, "compressed_context": compressed},
            )
            for r in results[:top_k]
        ]


class MockLightRAGGeneration:
    """Simulates LightRAG's answer generation."""

    NAME = "lightrag-answer"

    def generate(
        self, query: str, context: list[RetrievalResult], instruction: str = "",
    ) -> GenerationResult:
        return GenerationResult(
            output=f"[LightRAG] answer to '{query}' from {len(context)} sources",
            citations=[r.source_id for r in context],
            metadata={"framework": "lightrag"},
        )


class MockLongRAGRetrieval:
    """Simulates LongRAG's dataset-backed retrieval."""

    NAME = "longrag-tevatron"

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                source_id=f"longrag://{i}",
                content=f"LongRAG 4K context for {q}",
                score=1.0,
                title=f"LongRAG-Doc-{i}",
            )
            for i, q in enumerate(queries[:top_k])
        ]


class MockLongRAGGeneration:
    """Simulates LongRAG's LLM reader."""

    NAME = "longrag-reader"

    def generate(
        self, query: str, context: list[RetrievalResult], instruction: str = "",
    ) -> GenerationResult:
        return GenerationResult(
            output=f"[LongRAG] answer to '{query}'",
            citations=[r.source_id for r in context],
            metadata={"framework": "longrag"},
        )


class MockSelfRAGReranking:
    """Simulates Self-RAG's evidence scoring reranking."""

    NAME = "selfrag-evidence-scorer"

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 10
    ) -> list[RetrievalResult]:
        scored = []
        for r in results:
            isrel = 0.8 if "context" in r.content.lower() else 0.3
            scored.append(RetrievalResult(
                source_id=r.source_id,
                content=r.content,
                score=isrel,
                title=r.title,
                metadata={**r.metadata, "ISREL": isrel, "ISSUP": 0.7, "ISUSE": 0.6},
            ))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


class MockSelfRAGGeneration:
    """Simulates Self-RAG's generate-and-score generation."""

    NAME = "selfrag-generator"

    def generate(
        self, query: str, context: list[RetrievalResult], instruction: str = "",
    ) -> GenerationResult:
        return GenerationResult(
            output=f"[SelfRAG] evidence-scored answer to '{query}'",
            citations=[r.source_id for r in context],
            metadata={"framework": "selfrag"},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline builders
# ═══════════════════════════════════════════════════════════════════════════════


def _build_longrag_pipeline(retrieval, generation, reranking=None, query=None):
    from longRAG_example.longrag_langgraph.main_pipeline import build_graph
    return build_graph(
        retrieval=retrieval,
        generation=generation,
        reranking=reranking,
        query=query,
    )


def _build_lightrag_pipeline(retrieval, generation, reranking=None, query=None):
    from lightrag_langgraph.main_pipeline import build_query_graph
    return build_query_graph(
        retrieval=retrieval,
        generation=generation,
        reranking=reranking,
        query=query,
    )


def _run(coro):
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: LightRAG retrieval in LongRAG pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestLightRAGRetrievalInLongRAG:
    """Swap LightRAG's KG retrieval into LongRAG's pipeline."""

    def test_lightrag_retrieval_longrag_generation(self):
        graph = _build_longrag_pipeline(
            retrieval=MockLightRAGRetrieval(),
            generation=MockLongRAGGeneration(),
        )
        state = _run(graph.ainvoke({
            "query": "What causes climate change?",
            "query_id": "lr1",
            "answers": [],
            "test_data_name": "nq",
        }))
        gen = state["generation_result"]
        assert "[LongRAG]" in gen.output
        assert any("lightrag-chunk" in c for c in gen.citations)

    def test_lightrag_retrieval_with_selfrag_reranking(self):
        graph = _build_longrag_pipeline(
            retrieval=MockLightRAGRetrieval(),
            generation=MockLongRAGGeneration(),
            reranking=MockSelfRAGReranking(),
        )
        state = _run(graph.ainvoke({
            "query": "Explain quantum computing",
            "query_id": "lr2",
            "answers": [],
            "test_data_name": "nq",
        }))
        gen = state["generation_result"]
        assert "[LongRAG]" in gen.output
        for r in state["retrieval_results"]:
            assert "ISREL" in r.metadata


# ═══════════════════════════════════════════════════════════════════════════════
# Test: LongRAG components in LightRAG pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestLongRAGInLightRAGPipeline:
    """Swap LongRAG retrieval/generation into LightRAG's pipeline."""

    def test_longrag_retrieval_in_lightrag_pipeline(self):
        graph = _build_lightrag_pipeline(
            retrieval=MockLongRAGRetrieval(),
            generation=MockLightRAGGeneration(),
        )
        state = _run(graph.ainvoke({"query": "What is BERT?"}))
        gen = state["generation_result"]
        assert "[LightRAG]" in gen.output
        assert any("longrag" in c for c in gen.citations)

    def test_longrag_generation_in_lightrag_pipeline(self):
        graph = _build_lightrag_pipeline(
            retrieval=MockLightRAGRetrieval(),
            generation=MockLongRAGGeneration(),
        )
        state = _run(graph.ainvoke({"query": "How do transformers work?"}))
        gen = state["generation_result"]
        assert "[LongRAG]" in gen.output
        assert any("lightrag" in c for c in gen.citations)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Self-RAG reranking in LightRAG pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestSelfRAGRerankingInLightRAG:
    """Self-RAG's evidence scoring as reranking in LightRAG pipeline."""

    def test_selfrag_reranking_lightrag_everything_else(self):
        graph = _build_lightrag_pipeline(
            retrieval=MockLightRAGRetrieval(),
            generation=MockLightRAGGeneration(),
            reranking=MockSelfRAGReranking(),
        )
        state = _run(graph.ainvoke({"query": "What is RAG?"}))
        gen = state["generation_result"]
        assert "[LightRAG]" in gen.output
        for r in state["retrieval_results"]:
            assert "ISREL" in r.metadata

    def test_selfrag_reranking_with_lightrag_query(self):
        graph = _build_lightrag_pipeline(
            retrieval=MockLightRAGRetrieval(),
            generation=MockLightRAGGeneration(),
            reranking=MockSelfRAGReranking(),
            query=MockLightRAGQuery(),
        )
        state = _run(graph.ainvoke({"query": "Knowledge graphs"}))
        assert len(state.get("expanded_queries", [])) == 3
        gen = state["generation_result"]
        assert "[LightRAG]" in gen.output


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Full 3-way mix
# ═══════════════════════════════════════════════════════════════════════════════


class TestThreeWayMix:
    """Mix components from all three frameworks in a single pipeline."""

    def test_lightrag_query_longrag_retrieval_selfrag_reranking_lightrag_gen(self):
        graph = _build_lightrag_pipeline(
            retrieval=MockLongRAGRetrieval(),
            generation=MockLightRAGGeneration(),
            reranking=MockSelfRAGReranking(),
            query=MockLightRAGQuery(),
        )
        state = _run(graph.ainvoke({"query": "Multi-hop reasoning"}))
        assert len(state["expanded_queries"]) == 3
        gen = state["generation_result"]
        assert "[LightRAG]" in gen.output
        assert any("longrag" in c for c in gen.citations)
        for r in state["retrieval_results"]:
            assert "ISREL" in r.metadata

    def test_lightrag_retrieval_selfrag_gen_in_longrag_pipeline(self):
        graph = _build_longrag_pipeline(
            retrieval=MockLightRAGRetrieval(),
            generation=MockSelfRAGGeneration(),
            reranking=MockLightRAGReranking(),
        )
        state = _run(graph.ainvoke({
            "query": "Domain adaptation",
            "query_id": "3way1",
            "answers": [],
            "test_data_name": "hotpotqa",
        }))
        gen = state["generation_result"]
        assert "[SelfRAG]" in gen.output
        assert any("lightrag" in c for c in gen.citations)
        for r in state["retrieval_results"]:
            assert "compressed_context" in r.metadata


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Protocol duck typing for LightRAG components
# ═══════════════════════════════════════════════════════════════════════════════


class TestLightRAGProtocolConformance:
    """Verify that mock LightRAG components satisfy rag_contracts protocols."""

    def test_query_protocol(self):
        from rag_contracts.protocols import Query
        assert isinstance(MockLightRAGQuery(), Query)

    def test_retrieval_protocol(self):
        from rag_contracts.protocols import Retrieval
        assert isinstance(MockLightRAGRetrieval(), Retrieval)

    def test_reranking_protocol(self):
        from rag_contracts.protocols import Reranking
        assert isinstance(MockLightRAGReranking(), Reranking)

    def test_generation_protocol(self):
        from rag_contracts.protocols import Generation
        assert isinstance(MockLightRAGGeneration(), Generation)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: ALCE adapter with mixed pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestALCEAdapterIntegration:
    """Test that ALCE adapter works with cross-project pipelines."""

    def test_evaluate_generation_with_lightrag_gen(self):
        from benchmark.alce_adapter import (
            ALCEBenchmarkAdapter,
            compute_f1,
            compute_exact,
        )

        adapter = ALCEBenchmarkAdapter()
        mock_data = [
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "docs": [
                    {"title": "France", "text": "Paris is the capital of France."},
                    {"title": "Europe", "text": "France is in Western Europe."},
                ],
            },
        ]

        gen = MockLightRAGGeneration()
        results = adapter.evaluate_generation(mock_data, gen, max_docs=2)
        assert results.num_items == 1
        assert len(results.per_item) == 1
        assert results.per_item[0]["citations"]

    def test_f1_and_exact_metrics(self):
        from benchmark.alce_adapter import compute_f1, compute_exact
        assert compute_f1("Paris", "Paris") == 1.0
        assert compute_exact("Paris", "paris") == 1
        assert compute_exact("Paris", "London") == 0
        assert compute_f1("the capital is Paris", "Paris is the capital") > 0.5
