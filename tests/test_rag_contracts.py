"""Unit tests for rag_contracts: types, protocols, and identity implementations."""

from __future__ import annotations

import pytest

from rag_contracts import (
    Chunk,
    Document,
    GenerationResult,
    QueryContext,
    RetrievalResult,
)
from rag_contracts import IdentityEmbedding, IdentityQuery, IdentityReranking


# ═══════════════════════════════════════════════════════════════════════════════
# Data-type construction
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataTypes:
    def test_document_defaults(self):
        doc = Document(doc_id="d1", content="hello")
        assert doc.doc_id == "d1"
        assert doc.metadata == {}

    def test_chunk_defaults(self):
        chunk = Chunk(chunk_id="c1", doc_id="d1", content="piece")
        assert chunk.doc_id == "d1"
        assert chunk.metadata == {}

    def test_retrieval_result_defaults(self):
        rr = RetrievalResult(source_id="s1", content="ctx")
        assert rr.score == 0.0
        assert rr.title == ""
        assert rr.metadata == {}

    def test_generation_result_defaults(self):
        gr = GenerationResult(output="answer")
        assert gr.citations == []
        assert gr.metadata == {}

    def test_query_context_defaults(self):
        qc = QueryContext()
        assert qc.topic == ""
        assert qc.history == []


# ═══════════════════════════════════════════════════════════════════════════════
# Identity / passthrough implementations
# ═══════════════════════════════════════════════════════════════════════════════


class TestIdentityQuery:
    def test_returns_original_query(self):
        iq = IdentityQuery()
        result = iq.process("who is X?", QueryContext(topic="X"))
        assert result == ["who is X?"]

    def test_empty_query(self):
        iq = IdentityQuery()
        assert iq.process("", QueryContext()) == [""]


class TestIdentityEmbedding:
    def test_returns_empty_vectors(self):
        ie = IdentityEmbedding()
        vecs = ie.embed(["hello", "world"])
        assert len(vecs) == 2
        assert all(v == [] for v in vecs)

    def test_empty_input(self):
        ie = IdentityEmbedding()
        assert ie.embed([]) == []


class TestIdentityReranking:
    def test_passthrough(self):
        ir = IdentityReranking()
        items = [
            RetrievalResult(source_id=f"s{i}", content=f"c{i}", score=float(i))
            for i in range(5)
        ]
        result = ir.rerank("q", items, top_k=3)
        assert len(result) == 3
        assert [r.source_id for r in result] == ["s0", "s1", "s2"]

    def test_top_k_larger_than_results(self):
        ir = IdentityReranking()
        items = [RetrievalResult(source_id="s0", content="c0")]
        result = ir.rerank("q", items, top_k=10)
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Protocol structural conformance
# ═══════════════════════════════════════════════════════════════════════════════


class TestProtocolConformance:
    """Verify identity impls satisfy their corresponding Protocol at runtime."""

    def test_identity_query_is_query(self):
        from rag_contracts.protocols import Query
        obj = IdentityQuery()
        assert isinstance(obj, Query)

    def test_identity_embedding_is_embedding(self):
        from rag_contracts.protocols import Embedding
        obj = IdentityEmbedding()
        assert isinstance(obj, Embedding)

    def test_identity_reranking_is_reranking(self):
        from rag_contracts.protocols import Reranking
        obj = IdentityReranking()
        assert isinstance(obj, Reranking)
