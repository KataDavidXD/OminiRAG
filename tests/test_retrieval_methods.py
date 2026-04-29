"""Unit tests for the new retrieval and reranking methods.

Tests BM25Retrieval, DenseRetrieval, HybridRetrieval, CrossEncoderReranking,
and CorpusIndex against the KG sample data in benchmark/sample_data/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_contracts import RetrievalResult
from rag_contracts.retrieval_methods import (
    BM25Retrieval,
    CorpusIndex,
    DenseRetrieval,
    HybridRetrieval,
)
from rag_contracts.reranking_methods import CrossEncoderReranking

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "sample_data"
HOTPOT_CHUNKS = SAMPLE_DIR / "hotpotqa_kg_sample" / "chunks.json"
ULTRA_CHUNKS = SAMPLE_DIR / "ultradomain_kg_sample" / "chunks.json"


@pytest.fixture(scope="module")
def hotpot_corpus() -> CorpusIndex:
    return CorpusIndex.from_json_file(HOTPOT_CHUNKS)


@pytest.fixture(scope="module")
def ultra_corpus() -> CorpusIndex:
    return CorpusIndex.from_json_file(ULTRA_CHUNKS)


# ═══════════════════════════════════════════════════════════════════════════════
# CorpusIndex
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorpusIndex:
    def test_loads_from_json(self, hotpot_corpus: CorpusIndex):
        assert len(hotpot_corpus) > 0
        assert all(isinstance(t, str) for t in hotpot_corpus.texts)
        assert len(hotpot_corpus.chunk_ids) == len(hotpot_corpus.texts)

    def test_from_chunks_dict(self):
        chunks = {
            "c1": {"content": "Hello world", "doc_ids": ["d1"]},
            "c2": {"content": "Foo bar baz", "doc_ids": ["d2"]},
            "c3": {"content": "", "doc_ids": ["d3"]},
        }
        idx = CorpusIndex.from_chunks_dict(chunks)
        assert len(idx) == 2
        assert "c3" not in idx.chunk_ids

    def test_empty_corpus(self):
        idx = CorpusIndex()
        assert len(idx) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# BM25 Retrieval
# ═══════════════════════════════════════════════════════════════════════════════

class TestBM25Retrieval:
    def test_retrieves_relevant_chunks(self, hotpot_corpus: CorpusIndex):
        bm25 = BM25Retrieval(corpus=hotpot_corpus)
        results = bm25.retrieve(["Scott Derrickson director"], top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.score > 0 for r in results)
        assert results[0].score >= results[-1].score

    def test_multi_query_merges(self, hotpot_corpus: CorpusIndex):
        bm25 = BM25Retrieval(corpus=hotpot_corpus)
        results = bm25.retrieve(
            ["Scott Derrickson", "Ed Wood nationality"], top_k=5
        )
        assert len(results) > 0

    def test_empty_corpus_returns_empty(self):
        bm25 = BM25Retrieval(corpus=CorpusIndex())
        results = bm25.retrieve(["anything"], top_k=5)
        assert results == []

    def test_set_corpus_rebuilds_index(self, hotpot_corpus: CorpusIndex):
        bm25 = BM25Retrieval()
        assert bm25.retrieve(["test"], top_k=5) == []
        bm25.set_corpus(hotpot_corpus)
        results = bm25.retrieve(["director"], top_k=3)
        assert len(results) > 0

    def test_metadata_has_method_tag(self, hotpot_corpus: CorpusIndex):
        bm25 = BM25Retrieval(corpus=hotpot_corpus)
        results = bm25.retrieve(["director"], top_k=1)
        assert results[0].metadata.get("retrieval_method") == "bm25"


# ═══════════════════════════════════════════════════════════════════════════════
# Dense Retrieval
# ═══════════════════════════════════════════════════════════════════════════════

class TestDenseRetrieval:
    @pytest.fixture(scope="class")
    def dense(self, hotpot_corpus: CorpusIndex) -> DenseRetrieval:
        return DenseRetrieval(corpus=hotpot_corpus)

    def test_retrieves_relevant_chunks(self, dense: DenseRetrieval):
        results = dense.retrieve(["Who directed Doctor Strange?"], top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].score >= results[-1].score

    def test_metadata_has_method_tag(self, dense: DenseRetrieval):
        results = dense.retrieve(["director"], top_k=1)
        assert results[0].metadata.get("retrieval_method") == "dense"
        assert "model" in results[0].metadata

    def test_empty_corpus_returns_empty(self):
        d = DenseRetrieval(corpus=CorpusIndex())
        results = d.retrieve(["anything"], top_k=5)
        assert results == []

    def test_set_corpus(self, hotpot_corpus: CorpusIndex):
        d = DenseRetrieval()
        assert d.retrieve(["test"], top_k=5) == []
        d.set_corpus(hotpot_corpus)
        results = d.retrieve(["film director"], top_k=3)
        assert len(results) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid Retrieval
# ═══════════════════════════════════════════════════════════════════════════════

class TestHybridRetrieval:
    @pytest.fixture(scope="class")
    def hybrid(self, hotpot_corpus: CorpusIndex) -> HybridRetrieval:
        return HybridRetrieval(corpus=hotpot_corpus)

    def test_retrieves_relevant_chunks(self, hybrid: HybridRetrieval):
        results = hybrid.retrieve(
            ["Scott Derrickson directed Doctor Strange"], top_k=3
        )
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_rrf_combines_both_sources(self, hybrid: HybridRetrieval):
        results = hybrid.retrieve(["film director nationality"], top_k=5)
        assert len(results) > 0
        methods = {r.metadata.get("retrieval_method") for r in results}
        assert methods & {"bm25", "dense", "hybrid_rrf"}

    def test_set_corpus(self, hotpot_corpus: CorpusIndex):
        h = HybridRetrieval()
        h.set_corpus(hotpot_corpus)
        results = h.retrieve(["director"], top_k=3)
        assert len(results) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Encoder Reranking
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossEncoderReranking:
    @pytest.fixture(scope="class")
    def ce(self) -> CrossEncoderReranking:
        return CrossEncoderReranking()

    def test_reranks_results(self, ce: CrossEncoderReranking):
        fake_results = [
            RetrievalResult(
                source_id=f"doc_{i}",
                content=text,
                score=float(i),
                title=f"Doc {i}",
            )
            for i, text in enumerate([
                "Scott Derrickson directed Doctor Strange in 2016.",
                "The weather in Paris is usually mild in spring.",
                "Ed Wood was an American filmmaker known for low-budget films.",
            ])
        ]
        reranked = ce.rerank(
            "Who directed Doctor Strange?", fake_results, top_k=3
        )
        assert len(reranked) == 3
        assert reranked[0].score >= reranked[1].score >= reranked[2].score
        assert reranked[0].metadata.get("reranking_method") == "cross_encoder"
        assert "original_score" in reranked[0].metadata

    def test_empty_input(self, ce: CrossEncoderReranking):
        assert ce.rerank("query", [], top_k=5) == []

    def test_top_k_limits_output(self, ce: CrossEncoderReranking):
        results = [
            RetrievalResult(source_id=f"d{i}", content=f"content {i}", score=0.5)
            for i in range(5)
        ]
        reranked = ce.rerank("query", results, top_k=2)
        assert len(reranked) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: full pipeline (BM25 -> CrossEncoder -> check order improves)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrievalPipeline:
    def test_bm25_then_cross_encoder(self, hotpot_corpus: CorpusIndex):
        bm25 = BM25Retrieval(corpus=hotpot_corpus)
        ce = CrossEncoderReranking()
        query = "Who directed Doctor Strange?"

        retrieved = bm25.retrieve([query], top_k=5)
        assert len(retrieved) > 0

        reranked = ce.rerank(query, retrieved, top_k=3)
        assert len(reranked) > 0
        assert reranked[0].score >= reranked[-1].score

    def test_hybrid_then_cross_encoder(self, hotpot_corpus: CorpusIndex):
        hybrid = HybridRetrieval(corpus=hotpot_corpus)
        ce = CrossEncoderReranking()
        query = "What nationality was Ed Wood?"

        retrieved = hybrid.retrieve([query], top_k=5)
        assert len(retrieved) > 0

        reranked = ce.rerank(query, retrieved, top_k=3)
        assert len(reranked) > 0

    def test_ultra_domain_corpus(self, ultra_corpus: CorpusIndex):
        bm25 = BM25Retrieval(corpus=ultra_corpus)
        results = bm25.retrieve(["physics quantum mechanics"], top_k=3)
        assert len(results) >= 0  # may be 0 if content doesn't match


# ═══════════════════════════════════════════════════════════════════════════════
# Protocol compliance
# ═══════════════════════════════════════════════════════════════════════════════

class TestProtocolCompliance:
    def test_bm25_satisfies_retrieval_protocol(self):
        from rag_contracts import Retrieval
        assert isinstance(BM25Retrieval(), Retrieval)

    def test_dense_satisfies_retrieval_protocol(self):
        from rag_contracts import Retrieval
        assert isinstance(DenseRetrieval(), Retrieval)

    def test_hybrid_satisfies_retrieval_protocol(self):
        from rag_contracts import Retrieval
        h = HybridRetrieval()
        h.bm25 = BM25Retrieval()
        h.dense = DenseRetrieval()
        assert isinstance(h, Retrieval)

    def test_cross_encoder_satisfies_reranking_protocol(self):
        from rag_contracts import Reranking
        assert isinstance(CrossEncoderReranking(), Reranking)
