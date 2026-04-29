"""
Graph factory layer -- maps RAGConfig to compiled LangGraph pipelines.

Pipeline frame is an **explicit 5th slot**, not a heuristic derived from
component names.  ``config_to_graph_factory`` returns a zero-argument
callable suitable for ``WorkflowProject(graph_factory=...)``.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict

from .config_types import RAGConfig


# ---------------------------------------------------------------------------
# Component builder (reuses existing adapter instantiation logic)
# ---------------------------------------------------------------------------

def build_pipeline_components(config: RAGConfig) -> Dict[str, Any]:
    """Instantiate protocol-compliant adapter objects for a RAGConfig.

    Delegates to the component-building logic already present in
    ``AG-UCT/uct_engine/examples/rag_pipeline_search.py`` but
    parameterised by a ``RAGConfig`` instead of a raw tuple.

    Returns ``{"query": ..., "retrieval": ..., "reranking": ...,
    "generation": ...}``.
    """
    from rag_contracts import IdentityQuery, IdentityReranking, IdentityGeneration

    components: Dict[str, Any] = {}

    # -- Query --
    if config.query == "identity":
        components["query"] = IdentityQuery()
    elif config.query == "lightrag_keywords":
        from lightrag_langgraph.adapters import LightRAGQuery
        components["query"] = LightRAGQuery()
    else:
        components["query"] = IdentityQuery()

    # -- Retrieval --
    if config.retrieval == "longrag_dataset":
        from longRAG_example.longrag_langgraph.adapters import HFDatasetRetrieval
        components["retrieval"] = HFDatasetRetrieval()
    elif config.retrieval in ("lightrag_hybrid", "lightrag_chunk", "lightrag_graph"):
        from lightrag_langgraph.adapters import LightRAGRetrieval
        mode_map = {
            "lightrag_hybrid": "hybrid",
            "lightrag_chunk": "chunk",
            "lightrag_graph": "graph",
        }
        components["retrieval"] = LightRAGRetrieval(mode=mode_map[config.retrieval])
    elif config.retrieval == "alce_docs":
        from rag_contracts.common_components import ALCEDocRetrieval
        components["retrieval"] = ALCEDocRetrieval()
    elif config.retrieval == "llm_context":
        from rag_contracts import LLMRetrieval
        components["retrieval"] = LLMRetrieval()
    elif config.retrieval == "duckduckgo":
        from rag_contracts import DuckDuckGoRetrieval
        components["retrieval"] = DuckDuckGoRetrieval()
    else:
        components["retrieval"] = IdentityQuery()

    # -- Reranking --
    if config.reranking == "identity":
        components["reranking"] = IdentityReranking()
    elif config.reranking == "lightrag_compress":
        from lightrag_langgraph.adapters import LightRAGReranking
        components["reranking"] = LightRAGReranking()
    elif config.reranking == "selfrag_evidence":
        selfrag_rr, _ = _build_selfrag_components()
        components["reranking"] = selfrag_rr if selfrag_rr else IdentityReranking()
    else:
        components["reranking"] = IdentityReranking()

    # -- Generation --
    if config.generation == "longrag_reader":
        components["generation"] = _build_longrag_generation()
    elif config.generation == "lightrag_answer":
        from lightrag_langgraph.adapters import LightRAGGeneration
        components["generation"] = LightRAGGeneration()
    elif config.generation == "selfrag_generator":
        _, selfrag_gen = _build_selfrag_components()
        components["generation"] = selfrag_gen if selfrag_gen else IdentityGeneration()
    elif config.generation == "simple_llm":
        from rag_contracts import SimpleLLMGeneration
        components["generation"] = SimpleLLMGeneration()
    elif config.generation == "identity":
        components["generation"] = IdentityGeneration()
    else:
        components["generation"] = IdentityGeneration()

    return components


# ---------------------------------------------------------------------------
# Frame -> pipeline builder mapping (explicit, no heuristic)
# ---------------------------------------------------------------------------

_FRAME_BUILDERS: Dict[str, Callable] = {}


def _get_frame_builder(frame: str) -> Callable:
    """Return the LangGraph pipeline builder for the given frame name."""
    if not _FRAME_BUILDERS:
        from longRAG_example.longrag_langgraph.main_pipeline import (
            build_graph,
        )
        from lightrag_langgraph.main_pipeline import build_query_graph
        _sr_root = str(
            Path(__file__).resolve().parent.parent
            / "self-rag_langgraph" / "self-rag-wtb"
        )
        if _sr_root not in sys.path:
            sys.path.insert(0, _sr_root)
        from selfrag.modular_pipeline import build_selfrag_modular_graph

        _FRAME_BUILDERS["longrag"] = build_graph
        _FRAME_BUILDERS["lightrag"] = build_query_graph
        _FRAME_BUILDERS["selfrag"] = build_selfrag_modular_graph

    if frame not in _FRAME_BUILDERS:
        raise ValueError(
            f"Unknown frame '{frame}'. Must be one of: "
            f"{sorted(_FRAME_BUILDERS)}"
        )
    return _FRAME_BUILDERS[frame]


def config_to_graph_factory(config: RAGConfig) -> Callable[[], Any]:
    """Return a zero-argument factory that builds a compiled LangGraph.

    The returned callable is suitable for
    ``WorkflowProject(graph_factory=factory)``.  Each call produces a
    *fresh* compiled graph (no caching -- ``WorkflowProject`` handles
    its own ``_graph_cache``).
    """
    def factory() -> Any:
        builder = _get_frame_builder(config.frame)
        components = build_pipeline_components(config)
        return builder(
            retrieval=components["retrieval"],
            generation=components["generation"],
            reranking=components.get("reranking"),
            query=components.get("query"),
        )
    return factory


# ---------------------------------------------------------------------------
# Helpers imported from existing code (avoid circular import)
# ---------------------------------------------------------------------------

def _build_longrag_generation() -> Any:
    """Lightweight shim -- delegates to SimpleLLMGeneration when the full
    LongRAG inference module is unavailable."""
    import os
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        from rag_contracts import IdentityGeneration
        return IdentityGeneration()

    class _LLMInferenceShim:
        def __init__(self):
            from openai import OpenAI
            base = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE", "")
            self.client = OpenAI(api_key=api_key, base_url=base or None)
            self.model = os.environ.get("DEFAULT_LLM", "gpt-4o-mini")

        def _ask(self, context, query, titles):
            titles_str = ", ".join(titles) if titles else "N/A"
            resp = self.client.chat.completions.create(
                model=self.model, temperature=0.1, max_tokens=300,
                messages=[
                    {"role": "system",
                     "content": "You are an expert reader. Extract the answer "
                                "from the provided context. Be concise and precise."},
                    {"role": "user",
                     "content": f"Titles: {titles_str}\n\nContext:\n{context[:4000]}"
                                f"\n\nQuestion: {query}\n\nAnswer:"},
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


def _build_selfrag_components() -> tuple:
    """Build SelfRAG reranking + generation adapters.

    Returns ``(reranking, generation)`` or ``(None, None)`` on failure.
    Re-uses the vLLM/torch shim approach from ``rag_pipeline_search.py``.
    """
    import os
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
            __enter__=lambda s: None, __exit__=lambda s, *a: None
        )
        sys.modules["torch"] = _ft

    _sr_root = str(
        Path(__file__).resolve().parent.parent
        / "self-rag_langgraph" / "self-rag-wtb"
    )
    if _sr_root not in sys.path:
        sys.path.insert(0, _sr_root)

    try:
        from selfrag.adapters import SelfRAGGeneration, SelfRAGReranking
        from selfrag.constants import (
            ground_tokens_names, load_special_tokens,
            rel_tokens_names, retrieval_tokens_names,
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
