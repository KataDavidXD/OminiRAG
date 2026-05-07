"""Layer 2: Component protocol conformance tests.

Validates that each concrete component satisfies its ``rag_contracts``
protocol: Retrieval, Generation, Reranking.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag_contracts import GenerationResult, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_selfrag_path():
    sr = str(Path(__file__).resolve().parents[1] / "self-rag_langgraph" / "self-rag-wtb")
    if sr not in sys.path:
        sys.path.insert(0, sr)


def _fake_vllm_shim():
    """Install minimal vLLM shim so selfrag adapters can import."""
    import types
    if "vllm" not in sys.modules:
        from dataclasses import dataclass

        fv = types.ModuleType("vllm")

        @dataclass
        class _SP:
            temperature: float = 0.0
            top_p: float = 1.0
            max_tokens: int = 100
            logprobs: int = 0

        fv.SamplingParams = _SP
        sys.modules["vllm"] = fv

    if "torch" not in sys.modules:
        ft = types.ModuleType("torch")
        ft.no_grad = lambda: MagicMock(
            __enter__=lambda s: None, __exit__=lambda s, *a: None,
        )
        sys.modules["torch"] = ft


_fake_vllm_shim()
_ensure_selfrag_path()

SAMPLE_CONTEXT = [
    RetrievalResult(source_id="s1", content="Paris is the capital of France.", score=0.9, title="Paris"),
    RetrievalResult(source_id="s2", content="Berlin is the capital of Germany.", score=0.8, title="Berlin"),
]


# ---------------------------------------------------------------------------
# Retrieval protocol
# ---------------------------------------------------------------------------

class TestRetrievalProtocol:

    def test_all_retrieval_impls_have_retrieve_method(self):
        from rag_contracts import DuckDuckGoRetrieval, LLMRetrieval

        impls = [DuckDuckGoRetrieval, LLMRetrieval]
        for cls in impls:
            assert hasattr(cls, "retrieve"), f"{cls.__name__} missing retrieve()"

    def test_identity_reranking_has_rerank(self):
        from rag_contracts import IdentityReranking
        rr = IdentityReranking()
        assert hasattr(rr, "rerank")
        result = rr.rerank("q", SAMPLE_CONTEXT, top_k=1)
        assert len(result) == 1
        assert isinstance(result[0], RetrievalResult)

    def test_duckduckgo_retrieval_signature(self):
        from rag_contracts import DuckDuckGoRetrieval
        r = DuckDuckGoRetrieval(k=2)
        assert callable(r.retrieve)

    def test_selfrag_retrieval_has_retrieve(self):
        from selfrag.adapters import SelfRAGRetrieval
        r = SelfRAGRetrieval()
        assert hasattr(r, "retrieve")
        result = r.retrieve(["test"], top_k=5)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Generation protocol
# ---------------------------------------------------------------------------

class TestGenerationProtocol:

    def test_all_generation_impls_have_generate_method(self):
        from rag_contracts import SimpleLLMGeneration
        from rag_contracts.identity import IdentityGeneration

        impls = [IdentityGeneration]
        for cls in impls:
            assert hasattr(cls, "generate"), f"{cls.__name__} missing generate()"

    def test_identity_generation_returns_generation_result(self):
        from rag_contracts.identity import IdentityGeneration
        gen = IdentityGeneration()
        result = gen.generate(query="test", context=SAMPLE_CONTEXT)
        assert isinstance(result, GenerationResult)
        assert isinstance(result.output, str)
        assert isinstance(result.citations, list)

    def test_simple_llm_generation_has_generate(self):
        from rag_contracts import SimpleLLMGeneration
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "answer text"
        gen = SimpleLLMGeneration(llm=mock_llm)
        assert hasattr(gen, "generate")

    def test_simple_llm_generation_calls_llm(self):
        from rag_contracts import SimpleLLMGeneration
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "answer text"
        gen = SimpleLLMGeneration(llm=mock_llm)
        result = gen.generate(query="What is Paris?", context=SAMPLE_CONTEXT)
        assert isinstance(result, GenerationResult)
        assert mock_llm.complete.called

    def test_selfrag_generation_has_generate(self):
        from selfrag.adapters import SelfRAGGeneration
        gen = SelfRAGGeneration()
        assert hasattr(gen, "generate")

    def test_selfrag_generation_no_model_returns_empty(self):
        from selfrag.adapters import SelfRAGGeneration
        gen = SelfRAGGeneration(model=None)
        result = gen.generate(query="test", context=SAMPLE_CONTEXT)
        assert isinstance(result, GenerationResult)
        assert result.output == ""


# ---------------------------------------------------------------------------
# Reranking protocol
# ---------------------------------------------------------------------------

class TestRerankingProtocol:

    def test_all_reranking_impls_have_rerank_method(self):
        from rag_contracts import IdentityReranking
        impls = [IdentityReranking]
        for cls in impls:
            assert hasattr(cls, "rerank"), f"{cls.__name__} missing rerank()"

    def test_selfrag_reranking_has_rerank(self):
        from selfrag.adapters import SelfRAGReranking
        rr = SelfRAGReranking()
        assert hasattr(rr, "rerank")

    def test_selfrag_reranking_no_model_passthrough(self):
        from selfrag.adapters import SelfRAGReranking
        rr = SelfRAGReranking(model=None)
        result = rr.rerank("q", SAMPLE_CONTEXT, top_k=10)
        assert len(result) == 2
        assert result[0].source_id == "s1"


# ---------------------------------------------------------------------------
# RetrievalResult / GenerationResult field completeness
# ---------------------------------------------------------------------------

class TestFieldCompleteness:

    def test_retrieval_result_fields_populated(self):
        for r in SAMPLE_CONTEXT:
            assert r.source_id, "source_id should be non-empty"
            assert r.content, "content should be non-empty"
            assert isinstance(r.score, (int, float))

    def test_generation_result_fields_populated(self):
        from rag_contracts.identity import IdentityGeneration
        gen = IdentityGeneration()
        result = gen.generate(query="test", context=SAMPLE_CONTEXT)
        assert isinstance(result.output, str)
        assert isinstance(result.citations, list)
        assert isinstance(result.metadata, dict)
