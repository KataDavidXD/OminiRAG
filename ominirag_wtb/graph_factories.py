"""
Graph factory layer -- maps RAGConfig to compiled LangGraph pipelines.

The 5-dimension configuration (chunking, query, retrieval, post_retrieval,
generation) determines which adapter components are injected into the
LangGraph pipeline builder.  ``config_to_graph_factory`` returns a
zero-argument callable suitable for ``WorkflowProject(graph_factory=...)``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config_types import RAGConfig


# ---------------------------------------------------------------------------
# Component builder -- delegates to the canonical registry
# ---------------------------------------------------------------------------

def build_pipeline_components(
    config: RAGConfig,
    corpus: Optional[Any] = None,
) -> Dict[str, Any]:
    """Instantiate protocol-compliant adapter objects for a RAGConfig.

    Parameters
    ----------
    config:
        Fully-specified RAG pipeline configuration.
    corpus:
        Optional ``CorpusIndex`` to attach to BM25/Dense/Hybrid retrievers.

    Returns ``{"chunking": str, "query": ..., "retrieval": ...,
    "post_retrieval": ..., "generation": ...}``.
    """
    from rag_contracts.component_registry import build_pipeline_from_config

    return build_pipeline_from_config(config.slots(), corpus=corpus)


# ---------------------------------------------------------------------------
# Frame -> pipeline builder mapping (explicit, no heuristic)
# ---------------------------------------------------------------------------

_FRAME_BUILDERS: Dict[str, Callable] = {}


def _infer_frame(config: RAGConfig) -> str:
    """Infer the pipeline frame (graph topology) from configuration.

    - kg_extraction chunking + lightrag retrieval -> lightrag frame
    - selfrag_critique + selfrag_generator -> selfrag frame
    - default -> longrag frame (simplest 4-node topology)
    """
    if config.chunking == "kg_extraction" and config.retrieval.startswith("lightrag"):
        return "lightrag"
    if config.post_retrieval == "selfrag_critique" and config.generation == "selfrag_generator":
        return "selfrag"
    return "longrag"


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

    The pipeline frame (graph topology) is inferred from the configuration
    rather than stored as an explicit slot.
    """
    def factory() -> Any:
        frame = _infer_frame(config)
        builder = _get_frame_builder(frame)
        components = build_pipeline_components(config)
        return builder(
            retrieval=components["retrieval"],
            generation=components["generation"],
            reranking=components.get("post_retrieval"),
            query=components.get("query"),
        )
    return factory


