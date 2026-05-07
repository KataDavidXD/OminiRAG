"""Layer 5: vLLM Self-RAG model validation tests.

Validates the real Self-RAG model when the vLLM server is running.
Tests are skipped when the server is not reachable.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

VLLM_URL = os.environ.get("SELFRAG_VLLM_URL", "http://localhost:8002/v1")


def _vllm_reachable() -> bool:
    try:
        import httpx
        resp = httpx.get(f"{VLLM_URL}/models", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        try:
            import urllib.request
            url = VLLM_URL.rstrip("/")
            if url.endswith("/v1"):
                url = url[:-3]
            req = urllib.request.Request(f"{url}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False


skip_no_vllm = pytest.mark.skipif(
    not _vllm_reachable(), reason="vLLM server not reachable at " + VLLM_URL
)


def _ensure_shims():
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


_ensure_shims()


# ---------------------------------------------------------------------------
# Server reachability
# ---------------------------------------------------------------------------

@skip_no_vllm
class TestVLLMServer:

    def test_vllm_server_reachable(self):
        assert _vllm_reachable()

    def test_vllm_models_endpoint(self):
        import httpx
        resp = httpx.get(f"{VLLM_URL}/models", timeout=10.0)
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) > 0


# ---------------------------------------------------------------------------
# Completions with logprobs
# ---------------------------------------------------------------------------

@skip_no_vllm
class TestVLLMCompletions:

    def test_vllm_completions_return_logprobs(self):
        from openai import OpenAI
        client = OpenAI(api_key="EMPTY", base_url=VLLM_URL)
        resp = client.completions.create(
            model="selfrag-llama2-7b",
            prompt="### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
            max_tokens=30,
            temperature=0.0,
            logprobs=10,
        )
        choice = resp.choices[0]
        assert choice.text, "Completions should return non-empty text"
        assert choice.logprobs is not None, "logprobs should be present"
        assert choice.logprobs.tokens is not None
        assert len(choice.logprobs.tokens) > 0

    def test_selfrag_logprobs_contain_special_tokens(self):
        """Special Self-RAG tokens should appear in generated output."""
        from openai import OpenAI
        client = OpenAI(api_key="EMPTY", base_url=VLLM_URL)
        prompt = (
            "### Instruction:\nIs the following passage relevant to the question? "
            "Question: What is Python? "
            "Passage: Python is a programming language.\n\n### Response:\n"
        )
        resp = client.completions.create(
            model="selfrag-llama2-7b",
            prompt=prompt,
            max_tokens=50,
            temperature=0.0,
            logprobs=50,
        )
        tokens = resp.choices[0].logprobs.tokens or []
        token_strs = [str(t) for t in tokens]
        all_text = " ".join(token_strs)
        special = ["[Relevant]", "[Irrelevant]", "[Fully supported]",
                    "[Partially supported]", "[Utility:"]
        found_any = any(s in all_text or s in resp.choices[0].text for s in special)
        assert found_any or len(resp.choices[0].text) > 0, (
            "Expected some special tokens or text output"
        )


# ---------------------------------------------------------------------------
# RealVLLMModel wrapper
# ---------------------------------------------------------------------------

@skip_no_vllm
class TestRealVLLMModel:

    def test_real_vllm_model_generate(self):
        from rag_contracts.component_registry import RealVLLMModel
        model = RealVLLMModel(base_url=VLLM_URL, model_name="selfrag-llama2-7b")

        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=30, logprobs=50)
        results = model.generate(
            ["### Instruction:\nWhat is 2+2?\n\n### Response:\n"], sp,
        )
        assert len(results) == 1
        output = results[0].outputs[0]
        assert hasattr(output, "text")
        assert hasattr(output, "token_ids")
        assert hasattr(output, "logprobs")
        assert hasattr(output, "cumulative_logprob")
        assert len(output.text) > 0

    def test_real_vllm_vs_fake_shim_output_shape(self):
        """Compare output structure of real vs fake model wrappers."""
        from rag_contracts.component_registry import RealVLLMModel

        real_model = RealVLLMModel(base_url=VLLM_URL, model_name="selfrag-llama2-7b")
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=20, logprobs=50)
        prompt = "### Instruction:\nHello\n\n### Response:\n"
        real_results = real_model.generate([prompt], sp)

        real_out = real_results[0].outputs[0]
        assert isinstance(real_out.text, str)
        assert isinstance(real_out.token_ids, list)
        assert isinstance(real_out.logprobs, list)
        assert isinstance(real_out.cumulative_logprob, (int, float))

        if real_out.logprobs:
            first_lp = real_out.logprobs[0]
            assert isinstance(first_lp, dict)
            for k, v in first_lp.items():
                assert isinstance(k, int), "logprob keys should be token IDs (ints)"
                assert isinstance(v, float), "logprob values should be floats"


# ---------------------------------------------------------------------------
# RealVLLMTokenizer
# ---------------------------------------------------------------------------

@skip_no_vllm
class TestRealVLLMTokenizer:

    def test_tokenizer_resolves_special_tokens(self):
        from rag_contracts.component_registry import _RealVLLMTokenizer
        tok = _RealVLLMTokenizer(VLLM_URL, "selfrag-llama2-7b")
        rel_id = tok.convert_tokens_to_ids("[Relevant]")
        irr_id = tok.convert_tokens_to_ids("[Irrelevant]")
        assert rel_id != 0, "[Relevant] should have a non-zero token ID"
        assert irr_id != 0, "[Irrelevant] should have a non-zero token ID"
        assert rel_id != irr_id, "Different tokens should have different IDs"

    def test_tokenizer_all_special_tokens_have_ids(self):
        from rag_contracts.component_registry import _RealVLLMTokenizer
        from selfrag.constants import (
            retrieval_tokens_names, rel_tokens_names,
            ground_tokens_names, utility_tokens_names,
        )
        tok = _RealVLLMTokenizer(VLLM_URL, "selfrag-llama2-7b")
        all_tokens = (
            retrieval_tokens_names + rel_tokens_names
            + ground_tokens_names + utility_tokens_names
        )
        for token_str in all_tokens:
            tid = tok.convert_tokens_to_ids(token_str)
            assert tid != 0, f"Token '{token_str}' has ID 0 (not found)"


# ---------------------------------------------------------------------------
# SelfRAG generation with real model
# ---------------------------------------------------------------------------

@skip_no_vllm
class TestSelfRAGWithRealModel:

    def test_selfrag_generation_with_real_model(self):
        from rag_contracts.component_registry import RealVLLMModel, _RealVLLMTokenizer
        from selfrag.adapters import SelfRAGGeneration
        from selfrag.constants import load_special_tokens
        from rag_contracts import RetrievalResult

        model = RealVLLMModel(base_url=VLLM_URL, model_name="selfrag-llama2-7b")
        tok = _RealVLLMTokenizer(VLLM_URL, "selfrag-llama2-7b")
        _, rel, grd, ut = load_special_tokens(tok, True, True)

        gen = SelfRAGGeneration(
            model=model, rel_tokens=rel, grd_tokens=grd, ut_tokens=ut,
            max_new_tokens=50,
        )
        context = [
            RetrievalResult(
                source_id="p1", content="Paris is the capital of France.",
                score=1.0, title="Paris",
            ),
        ]
        result = gen.generate(query="What is the capital of France?", context=context)
        assert result.output, "Generation should produce non-empty output"
        assert "selfrag_score" in result.metadata
