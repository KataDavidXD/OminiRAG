"""Layer 3: Cross-project swap with real data tests.

Runs pipeline swaps using a small slice (2-3 items) of real HotpotQA data
to verify that swapped components work with real-world context.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dotenv import load_dotenv

_env = Path(__file__).resolve().parents[1] / ".env"
if _env.exists():
    load_dotenv(_env)

from rag_contracts import (
    GenerationResult,
    IdentityReranking,
    LLMRetrieval,
    RetrievalResult,
    SimpleLLMGeneration,
)

HOTPOTQA_DIR = Path("/data1/ragworkspace/dataset/all_data/hotpotqa")
ULTRADOMAIN_DIR = Path("/data1/ragworkspace/dataset/UltraDomain")

skip_no_data = pytest.mark.skipif(
    not HOTPOTQA_DIR.exists(), reason="Real HotpotQA dataset not available"
)
skip_no_api_key = pytest.mark.skipif(
    not os.environ.get("LLM_API_KEY"), reason="LLM_API_KEY not set"
)


def _ensure_selfrag():
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
    sr = str(Path(__file__).resolve().parents[1] / "self-rag_langgraph" / "self-rag-wtb")
    if sr not in sys.path:
        sys.path.insert(0, sr)


_ensure_selfrag()


def _load_real_hotpotqa(n=2):
    from benchmark.hotpotqa_adapter import load_hotpotqa_real
    return load_hotpotqa_real(HOTPOTQA_DIR, max_items=n)


def _build_simple_gen():
    from rag_contracts.component_registry import build_simple_llm
    return SimpleLLMGeneration(llm=build_simple_llm())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_no_data
class TestSwapWithRealData:

    def test_longrag_pipe_with_real_hotpotqa_context(self):
        """LongRAG pipeline with real HotpotQA context (no live LLM needed)."""
        from benchmark.hotpotqa_adapter import HotpotQABenchmarkAdapter, load_hotpotqa_real

        items = load_hotpotqa_real(HOTPOTQA_DIR, max_items=2)
        mock_gen = MagicMock()
        mock_gen.generate.return_value = GenerationResult(output="test", citations=[])

        adapter = HotpotQABenchmarkAdapter()
        result = adapter.evaluate_generation(items, mock_gen)
        assert result.num_items == 2
        for call in mock_gen.generate.call_args_list:
            ctx = call.kwargs.get("context", [])
            assert len(ctx) > 0, "Real context should be non-empty"
            assert all(isinstance(r, RetrievalResult) for r in ctx)

    @skip_no_api_key
    def test_simple_llm_with_real_hotpotqa(self):
        """SimpleLLM generation with real HotpotQA data end-to-end."""
        from benchmark.hotpotqa_adapter import HotpotQABenchmarkAdapter, load_hotpotqa_real

        items = load_hotpotqa_real(HOTPOTQA_DIR, max_items=2)
        gen = _build_simple_gen()
        adapter = HotpotQABenchmarkAdapter()
        result = adapter.evaluate_generation(items, gen)
        assert result.num_items == 2
        for pi in result.per_item:
            assert pi["output"], "Generation output should not be empty"

    def test_selfrag_gen_with_real_context_mock_model(self):
        """SelfRAG generation with real context but mock vLLM model."""
        from selfrag.adapters import SelfRAGGeneration
        from selfrag.constants import load_special_tokens

        items = _load_real_hotpotqa(2)
        context = items[0]["context_results"][:3]

        class _FakeTok:
            def convert_tokens_to_ids(self, t):
                return hash(t) % 10000

        tok = _FakeTok()
        _, rel, grd, ut = load_special_tokens(tok, True, True)

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.text = "[Relevant]Test answer[Fully supported][Utility:5]"
        mock_output.token_ids = [1, 2, 3]
        mock_output.logprobs = [{}, {}, {}]
        mock_output.cumulative_logprob = -1.0
        mock_pred = MagicMock()
        mock_pred.outputs = [mock_output]
        mock_model.generate.return_value = [mock_pred] * len(context)

        gen = SelfRAGGeneration(
            model=mock_model, rel_tokens=rel, grd_tokens=grd, ut_tokens=ut,
        )
        result = gen.generate(query=items[0]["question"], context=context)
        assert isinstance(result, GenerationResult)

    def test_cross_swap_identity_reranking_in_pipeline(self):
        """IdentityReranking swapped into evaluation with real data."""
        items = _load_real_hotpotqa(2)
        rr = IdentityReranking()
        context = items[0]["context_results"]
        reranked = rr.rerank(items[0]["question"], context, top_k=5)
        assert len(reranked) == min(5, len(context))
        assert all(isinstance(r, RetrievalResult) for r in reranked)


@skip_no_data
class TestSwapWithUltraDomain:

    def test_ultradomain_real_context_in_generation(self):
        from benchmark.ultradomain_adapter import load_ultradomain_real

        items = load_ultradomain_real(ULTRADOMAIN_DIR, domain="mix", max_items=2)
        mock_gen = MagicMock()
        mock_gen.generate.return_value = GenerationResult(output="test", citations=[])

        for item in items:
            ctx = item["context_results"]
            assert len(ctx) > 0
            mock_gen.generate(query=item["question"], context=ctx)
        assert mock_gen.generate.call_count == 2
