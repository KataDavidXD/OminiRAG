"""Canonical config-to-component builder for OminiRAG pipelines.

Single source of truth for mapping RAG configuration slot values to
protocol-compliant adapter instances.  Both ``AG-UCT/rag_pipeline_search``
and ``ominirag_wtb/graph_factories`` delegate to this module.

Also contains the LLM shims (LongRAG inference, SelfRAG vLLM mock,
SimpleLLM wrapper) that allow running framework adapters against
OpenAI-compatible endpoints without the full native dependencies.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from .retrieval_methods import BM25Retrieval, CorpusIndex, DenseRetrieval, HybridRetrieval
from .reranking_methods import CrossEncoderReranking
from .identity import IdentityGeneration, IdentityQuery, IdentityReranking

# Legacy name mappings for backward compatibility
_LEGACY_FRAME_TO_CHUNKING = {
    "longrag": "longrag_4k",
    "lightrag": "kg_extraction",
    "selfrag": "standard_passage",
}
_LEGACY_RETRIEVAL = {
    "longrag_dataset": "bm25",
    "lightrag_chunk": "lightrag_hybrid",
}
_LEGACY_RERANKING = {
    "selfrag_evidence": "selfrag_critique",
}


def _project_root() -> Path:
    """Return the OminiRAG project root (parent of ``rag_contracts/``)."""
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# LLM helper builders
# ---------------------------------------------------------------------------

def build_simple_llm() -> Any:
    """Build a simple LLM object with a ``complete()`` method."""
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        class _Noop:
            def complete(self, system, user, **kw):
                return ""
        return _Noop()

    class _SimpleLLM:
        def __init__(self):
            from openai import OpenAI
            base = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE", "")
            self.client = OpenAI(api_key=api_key, base_url=base or None)
            self.model = os.environ.get("DEFAULT_LLM", "gpt-4o-mini")

        def complete(self, system, user, **kw):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=kw.get("temperature", 0.1),
                max_tokens=kw.get("max_tokens", 300),
            )
            return (resp.choices[0].message.content or "").strip()

    return _SimpleLLM()


def build_longrag_generation() -> Any:
    """Build a LongRAGGeneration with a SimpleLLM-backed inference shim.

    ``LongRAGGeneration`` requires an ``llm_inference`` object exposing
    ``predict_nq()`` and ``predict_hotpotqa()``.  When the full LongRAG
    inference module is unavailable, this creates a lightweight shim that
    delegates to OpenAI chat completions.
    """
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return IdentityGeneration()

    class _LLMInferenceShim:
        """Mimics LongRAG's inference API using OpenAI chat completions."""

        def __init__(self):
            from openai import OpenAI
            base = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE", "")
            self.client = OpenAI(api_key=api_key, base_url=base or None)
            self.model = os.environ.get("DEFAULT_LLM", "gpt-4o-mini")

        def _ask(self, context, query, titles):
            titles_str = ", ".join(titles) if titles else "N/A"
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                max_tokens=300,
                messages=[
                    {"role": "system",
                     "content": "You are an expert reader. Extract the answer "
                     "from the provided context. Be concise and precise."},
                    {"role": "user",
                     "content": f"Titles: {titles_str}\n\n"
                     f"Context:\n{context[:4000]}\n\n"
                     f"Question: {query}\n\nAnswer:"},
                ],
            )
            ans = (resp.choices[0].message.content or "").strip()
            return ans, ans

        def predict_nq(self, context, query, titles):
            return self._ask(context, query, titles)

        def predict_hotpotqa(self, context, query, titles):
            return self._ask(context, query, titles)

    from longRAG_example.longrag_langgraph.adapters import LongRAGGeneration
    return LongRAGGeneration(llm_inference=_LLMInferenceShim())


def _ensure_selfrag_path() -> None:
    """Add self-rag_langgraph/self-rag-wtb to sys.path if needed."""
    sr_root = str(_project_root() / "self-rag_langgraph" / "self-rag-wtb")
    if sr_root not in sys.path:
        sys.path.insert(0, sr_root)


def _ensure_vllm_shim() -> None:
    """Install minimal vLLM / torch shims if the real packages are absent."""
    import types
    from unittest.mock import MagicMock

    if "vllm" not in sys.modules:
        from dataclasses import dataclass as _dc

        _fv = types.ModuleType("vllm")

        @_dc
        class _SP:
            temperature: float = 0.0
            top_p: float = 1.0
            max_tokens: int = 100
            logprobs: int = 0

        _fv.SamplingParams = _SP
        sys.modules["vllm"] = _fv

    if "torch" not in sys.modules:
        _ft = types.ModuleType("torch")
        _ft.no_grad = lambda: MagicMock(
            __enter__=lambda s: None, __exit__=lambda s, *a: None,
        )
        sys.modules["torch"] = _ft


def build_selfrag_components() -> tuple:
    """Build SelfRAG reranking and generation adapters.

    When ``SELFRAG_VLLM_URL`` is set (e.g. ``http://localhost:8002/v1``),
    uses a real vLLM server for inference with real logprobs.  Otherwise
    falls back to an OpenAI chat-completions shim with synthetic logprobs.

    Returns ``(reranking, generation)`` or ``(None, None)`` on failure.
    """
    _ensure_vllm_shim()
    _ensure_selfrag_path()

    try:
        from selfrag.adapters import SelfRAGGeneration, SelfRAGReranking
        from selfrag.constants import (
            ground_tokens_names,
            load_special_tokens,
            rel_tokens_names,
            retrieval_tokens_names,
            utility_tokens_names,
        )
    except ImportError:
        return None, None

    vllm_url = os.environ.get("SELFRAG_VLLM_URL", "")

    if vllm_url:
        # ── Real vLLM server mode ──
        model_name = os.environ.get("SELFRAG_MODEL_NAME", "selfrag-llama2-7b")
        num_logprobs = int(os.environ.get("SELFRAG_NUM_LOGPROBS", "100"))
        model = RealVLLMModel(
            base_url=vllm_url, model_name=model_name,
            num_logprobs=num_logprobs,
        )
        tok = _RealVLLMTokenizer(vllm_url, model_name)
    else:
        # ── Fake shim mode (original behaviour) ──
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return None, None
        tok = _FakeSelfRAGTokenizer(retrieval_tokens_names, rel_tokens_names,
                                    ground_tokens_names, utility_tokens_names)
        model = _FakeSelfRAGModel(api_key, retrieval_tokens_names, rel_tokens_names,
                                  ground_tokens_names, utility_tokens_names)

    _, rel, grd, ut = load_special_tokens(tok, use_grounding=True, use_utility=True)
    reranking = SelfRAGReranking(model=model, rel_tokens=rel, grd_tokens=grd, ut_tokens=ut)
    generation = SelfRAGGeneration(model=model, rel_tokens=rel, grd_tokens=grd, ut_tokens=ut)
    return reranking, generation


# ---------------------------------------------------------------------------
# Fake SelfRAG model / tokenizer (synthetic logprobs via OpenAI chat)
# ---------------------------------------------------------------------------

class _FakeSelfRAGTokenizer:
    """Maps special token strings to sequential fake IDs starting at 2000."""

    def __init__(self, *token_lists):
        self._map: dict[str, int] = {}
        idx = 2000
        for lst in token_lists:
            for tok in lst:
                self._map[tok] = idx
                idx += 1

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._map.get(token, 0)


class _FakeSelfRAGModel:
    """Wraps OpenAI chat completions behind vLLM generate() with synthetic logprobs."""

    def __init__(self, api_key, retrieval_names, rel_names, grd_names, ut_names):
        from openai import OpenAI
        base = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE", "")
        self._client = OpenAI(api_key=api_key, base_url=base or None)
        self._model = os.environ.get("DEFAULT_LLM", "gpt-4o-mini")
        all_special = retrieval_names + rel_names + grd_names + ut_names
        self._tmap = {tok: 2000 + i for i, tok in enumerate(all_special)}
        self._ret = retrieval_names
        self._rel = rel_names
        self._grd = grd_names
        self._ut = ut_names

    def generate(self, prompts, sp=None):
        out = []
        for p in prompts:
            try:
                r = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system",
                         "content": "You are a Self-RAG assistant. Given a question and "
                         "optionally a paragraph of evidence, produce a SHORT factual "
                         "answer wrapped in Self-RAG control tokens.\n"
                         "Format: [Relevant]<answer>[Fully supported][Utility:5]"},
                        {"role": "user", "content": p},
                    ],
                    temperature=getattr(sp, "temperature", 0.0),
                    max_tokens=getattr(sp, "max_tokens", 150),
                )
                txt = r.choices[0].message.content or ""
            except Exception:
                txt = "[Relevant]Error[Partially supported][Utility:3]"
            out.append(self._wrap(txt))
        return out

    def _wrap(self, text):
        found_rel = next((t for t in self._rel if t in text), "[Relevant]")
        found_grd = next((t for t in self._grd if t in text), "[Fully supported]")
        found_ut = next((t for t in self._ut if t in text), "[Utility:5]")
        n = max(5, len(text.split()) // 2)
        token_ids = [self._tmap.get(found_rel, 888)]
        token_ids.extend([888] * n)
        gp = len(token_ids)
        token_ids.append(self._tmap.get(found_grd, 888))
        up = len(token_ids)
        token_ids.append(self._tmap.get(found_ut, 888))
        logprobs = []
        for pos in range(len(token_ids)):
            e: dict[int, float] = {}
            for t in self._rel:
                e[self._tmap[t]] = -0.1 if (pos == 0 and t == found_rel) else -5.0
            for t in self._ret:
                e[self._tmap[t]] = -5.0
            for t in self._grd:
                e[self._tmap[t]] = -0.1 if (pos == gp and t == found_grd) else -5.0
            for t in self._ut:
                e[self._tmap[t]] = -0.1 if (pos == up and t == found_ut) else -5.0
            logprobs.append(e)

        class _O:
            pass
        o = _O()
        o.text = text
        o.token_ids = token_ids
        o.logprobs = logprobs
        o.cumulative_logprob = -2.0

        class _P:
            pass
        p = _P()
        p.outputs = [o]
        return p


# ---------------------------------------------------------------------------
# Real vLLM model / tokenizer (calls vLLM OpenAI-compatible server)
# ---------------------------------------------------------------------------

class _RealVLLMTokenizer:
    """Fetches real token IDs from a running vLLM server's /tokenize endpoint.

    Falls back gracefully if the server is unreachable, assigning sequential
    IDs starting at 32000 (the typical location of added special tokens in
    selfrag_llama2_7b's vocabulary).
    """

    def __init__(self, base_url: str, model_name: str):
        self._cache: dict[str, int] = {}
        self._base_url = base_url.rstrip("/")
        if self._base_url.endswith("/v1"):
            self._api_base = self._base_url[:-3]
        else:
            self._api_base = self._base_url
        self._model = model_name
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        try:
            from selfrag.constants import (
                retrieval_tokens_names, rel_tokens_names,
                ground_tokens_names, utility_tokens_names,
            )
        except ImportError:
            return

        all_tokens = (
            retrieval_tokens_names + rel_tokens_names
            + ground_tokens_names + utility_tokens_names
        )

        try:
            import httpx
            client = httpx.Client(timeout=10.0)
        except ImportError:
            import urllib.request
            import json as _json
            client = None

        for idx, tok_str in enumerate(all_tokens):
            try:
                if client is not None:
                    resp = client.post(
                        f"{self._api_base}/tokenize",
                        json={"model": self._model, "prompt": tok_str},
                    )
                    if resp.status_code == 200:
                        ids = resp.json().get("tokens", [])
                        if ids:
                            self._cache[tok_str] = ids[-1]
                            continue
                else:
                    req = urllib.request.Request(
                        f"{self._api_base}/tokenize",
                        data=_json.dumps({"model": self._model, "prompt": tok_str}).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        data = _json.loads(resp.read())
                        ids = data.get("tokens", [])
                        if ids:
                            self._cache[tok_str] = ids[-1]
                            continue
            except Exception:
                pass
            self._cache[tok_str] = 32000 + idx

        if client is not None:
            client.close()

    def convert_tokens_to_ids(self, token: str) -> int:
        self._ensure_initialized()
        return self._cache.get(token, 0)


class RealVLLMModel:
    """Calls a real vLLM server via the OpenAI-compatible completions API.

    Translates the response into the ``pred.outputs[0].{text, token_ids,
    logprobs, cumulative_logprob}`` structure that ``selfrag/adapters.py``
    expects for Self-RAG scoring.

    Unlike the fake shim, logprobs here are **real model outputs**, enabling
    accurate relevance / grounding / utility scoring.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str = "selfrag-llama2-7b",
        num_logprobs: int = 100,
    ):
        from openai import OpenAI
        self._client = OpenAI(api_key="EMPTY", base_url=base_url)
        self._model = model_name
        self._num_logprobs = num_logprobs
        self._tokenizer = _RealVLLMTokenizer(base_url, model_name)

    def generate(self, prompts, sp=None):
        results = []
        max_tokens = getattr(sp, "max_tokens", 100)
        temperature = getattr(sp, "temperature", 0.0)
        top_p = getattr(sp, "top_p", 1.0)

        for prompt in prompts:
            try:
                resp = self._client.completions.create(
                    model=self._model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    logprobs=self._num_logprobs,
                )
                choice = resp.choices[0]
                text = choice.text or ""

                lp = choice.logprobs
                if lp and lp.tokens:
                    token_ids = [
                        self._tokenizer.convert_tokens_to_ids(t)
                        for t in lp.tokens
                    ]
                    logprobs_list: list[dict[int, float]] = []
                    for top_lps in (lp.top_logprobs or []):
                        entry: dict[int, float] = {}
                        if top_lps:
                            for tok_str, log_p in top_lps.items():
                                tid = self._tokenizer.convert_tokens_to_ids(tok_str)
                                if tid != 0:
                                    entry[tid] = log_p
                        logprobs_list.append(entry)
                    cum_lp = sum(
                        p for p in (lp.token_logprobs or []) if p is not None
                    )
                else:
                    token_ids = []
                    logprobs_list = []
                    cum_lp = 0.0

                output = _RealOutput(text, token_ids, logprobs_list, cum_lp)
            except Exception as exc:
                print(f"    [vLLM ERROR] {exc}")
                output = _RealOutput("", [], [], 0.0)

            results.append(_RealPrediction(output))
        return results


class _RealOutput:
    __slots__ = ("text", "token_ids", "logprobs", "cumulative_logprob")

    def __init__(self, text, token_ids, logprobs, cumulative_logprob):
        self.text = text
        self.token_ids = token_ids
        self.logprobs = logprobs
        self.cumulative_logprob = cumulative_logprob


class _RealPrediction:
    __slots__ = ("outputs",)

    def __init__(self, output):
        self.outputs = [output]


# ---------------------------------------------------------------------------
# Main pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline_from_config(
    choices: tuple[str, ...],
    benchmark: str = "hotpotqa",
    corpus: Optional[CorpusIndex] = None,
) -> dict[str, Any]:
    """Build a dict of rag_contracts components from a config tuple.

    Accepts a 5-tuple ``(chunking, query, retrieval, post_retrieval, generation)``
    or a legacy 4-tuple / old-style 5-tuple with frame names.

    Parameters
    ----------
    choices:
        Config slot values.
    benchmark:
        Benchmark name (informational, not currently used).
    corpus:
        Optional pre-loaded ``CorpusIndex`` to attach to BM25/Dense/Hybrid
        retrieval components.  When ``None``, retrievers are unindexed.

    Returns
    -------
    dict with keys ``chunking``, ``query``, ``retrieval``,
    ``post_retrieval``, ``generation``.
    """
    from .common_components import SimpleLLMGeneration

    if len(choices) == 5:
        chunking_name, query_name, retrieval_name, post_ret_name, generation_name = choices
    else:
        chunking_name = "standard_passage"
        query_name, retrieval_name, post_ret_name, generation_name = choices

    if chunking_name in _LEGACY_FRAME_TO_CHUNKING:
        chunking_name = _LEGACY_FRAME_TO_CHUNKING[chunking_name]
        retrieval_name = _LEGACY_RETRIEVAL.get(retrieval_name, retrieval_name)
        post_ret_name = _LEGACY_RERANKING.get(post_ret_name, post_ret_name)

    components: dict[str, Any] = {"chunking": chunking_name}

    # -- Query --
    if query_name == "lightrag_keywords":
        from lightrag_langgraph.adapters import LightRAGQuery
        components["query"] = LightRAGQuery()
    else:
        components["query"] = IdentityQuery()

    # -- Retrieval --
    if retrieval_name == "bm25":
        components["retrieval"] = BM25Retrieval(corpus=corpus)
    elif retrieval_name == "dense_e5":
        components["retrieval"] = DenseRetrieval(corpus=corpus)
    elif retrieval_name == "bm25_dense_hybrid":
        components["retrieval"] = HybridRetrieval(corpus=corpus)
    elif retrieval_name in ("lightrag_hybrid", "lightrag_graph"):
        from lightrag_langgraph.adapters import LightRAGRetrieval
        mode_map = {"lightrag_hybrid": "hybrid", "lightrag_graph": "graph"}
        components["retrieval"] = LightRAGRetrieval(mode=mode_map[retrieval_name])
    else:
        components["retrieval"] = BM25Retrieval(corpus=corpus)

    # -- Post-Retrieval --
    if post_ret_name == "identity":
        components["post_retrieval"] = IdentityReranking()
    elif post_ret_name == "cross_encoder":
        components["post_retrieval"] = CrossEncoderReranking()
    elif post_ret_name == "lightrag_compress":
        from lightrag_langgraph.adapters import LightRAGReranking
        components["post_retrieval"] = LightRAGReranking()
    elif post_ret_name == "selfrag_critique":
        selfrag_rr, _ = build_selfrag_components()
        components["post_retrieval"] = selfrag_rr if selfrag_rr else IdentityReranking()
    else:
        components["post_retrieval"] = IdentityReranking()

    # -- Generation --
    if generation_name == "longrag_reader":
        components["generation"] = build_longrag_generation()
    elif generation_name == "lightrag_answer":
        from lightrag_langgraph.adapters import LightRAGGeneration
        components["generation"] = LightRAGGeneration()
    elif generation_name == "selfrag_generator":
        _, selfrag_gen = build_selfrag_components()
        components["generation"] = selfrag_gen if selfrag_gen else IdentityGeneration()
    elif generation_name == "simple_llm":
        components["generation"] = SimpleLLMGeneration(llm=build_simple_llm())
    elif generation_name == "identity":
        components["generation"] = IdentityGeneration()
    else:
        components["generation"] = IdentityGeneration()

    return components
