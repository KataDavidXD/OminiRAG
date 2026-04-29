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
    """Build SelfRAG reranking and generation adapters using an OpenAI vLLM shim.

    Wraps OpenAI chat completions behind the vLLM ``generate()`` interface
    so that ``SelfRAGReranking`` / ``SelfRAGGeneration`` can score passages
    with synthetic logprobs.

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

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None, None

    ALL_SPECIAL = (
        retrieval_tokens_names + rel_tokens_names
        + ground_tokens_names + utility_tokens_names
    )
    TOKEN_MAP = {tok: 2000 + i for i, tok in enumerate(ALL_SPECIAL)}

    class _Tok:
        def convert_tokens_to_ids(self, token: str) -> int:
            return TOKEN_MAP.get(token, 0)

    class _Out:
        def __init__(self, text):
            self.text = text
            found_rel = next((t for t in rel_tokens_names if t in text), "[Relevant]")
            found_grd = next((t for t in ground_tokens_names if t in text), "[Fully supported]")
            found_ut = next((t for t in utility_tokens_names if t in text), "[Utility:5]")
            n = max(5, len(text.split()) // 2)
            self.token_ids = [TOKEN_MAP.get(found_rel, 888)]
            self.token_ids.extend([888] * n)
            gp = len(self.token_ids)
            self.token_ids.append(TOKEN_MAP.get(found_grd, 888))
            up = len(self.token_ids)
            self.token_ids.append(TOKEN_MAP.get(found_ut, 888))
            self.logprobs = []
            for pos in range(len(self.token_ids)):
                e = {}
                for t in rel_tokens_names:
                    e[TOKEN_MAP[t]] = -0.1 if (pos == 0 and t == found_rel) else -5.0
                for t in retrieval_tokens_names:
                    e[TOKEN_MAP[t]] = -5.0
                for t in ground_tokens_names:
                    e[TOKEN_MAP[t]] = -0.1 if (pos == gp and t == found_grd) else -5.0
                for t in utility_tokens_names:
                    e[TOKEN_MAP[t]] = -0.1 if (pos == up and t == found_ut) else -5.0
                self.logprobs.append(e)
            self.cumulative_logprob = -2.0

    class _Pred:
        def __init__(self, o):
            self.outputs = [o]

    class _Model:
        def __init__(self):
            from openai import OpenAI
            base = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE", "")
            self.client = OpenAI(api_key=api_key, base_url=base or None)
            self.model = os.environ.get("DEFAULT_LLM", "gpt-4o-mini")

        def generate(self, prompts, sp=None):
            out = []
            for p in prompts:
                try:
                    r = self.client.chat.completions.create(
                        model=self.model,
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
                out.append(_Pred(_Out(txt)))
            return out

    tok = _Tok()
    _, rel, grd, ut = load_special_tokens(tok, use_grounding=True, use_utility=True)
    model = _Model()

    reranking = SelfRAGReranking(model=model, rel_tokens=rel, grd_tokens=grd, ut_tokens=ut)
    generation = SelfRAGGeneration(model=model, rel_tokens=rel, grd_tokens=grd, ut_tokens=ut)
    return reranking, generation


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
