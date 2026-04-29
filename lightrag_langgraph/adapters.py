"""Adapters between LightRAG internals and canonical rag_contracts.

Forward direction: wrap the split LightRAG modules (query_module,
retrieval_module, reranking_module, generation_module) to satisfy
canonical protocols so they can be injected into any rag_contracts pipeline.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_contracts import (
    GenerationResult,
    QueryContext,
    RetrievalResult,
)

_LIGHTRAG_SRC = str(
    Path(__file__).resolve().parent.parent
    / "A-Simplified-Core-Workflow-for-Enhancing-RAG"
    / "lightrag_core_simplified"
    / "src"
)
if _LIGHTRAG_SRC not in sys.path:
    sys.path.insert(0, str(Path(_LIGHTRAG_SRC).parent))


def _get_config():
    """Lazy-import LightRAG Config to avoid import-time side effects."""
    from lightrag_core_simplified.src.config import Config
    return Config


# ═══════════════════════════════════════════════════════════════════════════════
# Query adapter -- keyword extraction + query expansion
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LightRAGQuery:
    """``rag_contracts.Query`` wrapping LightRAG's keyword-based query expansion.

    Returns the original query plus extracted high/low-level keywords
    joined by ``|`` as additional retrieval queries.

    After ``process()`` runs, the full ``query_module.run()`` result dict
    is cached in ``_last_query_result`` so the pipeline can pass it to
    ``LightRAGRetrieval`` and avoid a redundant LLM+embedding call.
    """

    config: Any = None
    _last_query_result: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.config is None:
            self.config = _get_config()()

    def process(self, query: str, context: QueryContext) -> list[str]:
        from lightrag_core_simplified.src.modules import query_module

        result = query_module.run(self.config, query)
        self._last_query_result = result
        keywords = result["keywords"]
        expanded = [query]
        for kw in keywords.get("high_level_keywords", []):
            if kw and kw != query:
                expanded.append(kw)
        for kw in keywords.get("low_level_keywords", []):
            if kw and kw != query:
                expanded.append(kw)
        return expanded


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval adapter -- vector + KG hybrid retrieval
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LightRAGRetrieval:
    """``rag_contracts.Retrieval`` wrapping LightRAG's hybrid vector+KG retrieval.

    Requires pre-built LightRAG stores (chunks, entities, relations, graph, kv).

    When a pre-computed ``query_result`` is set (via ``set_query_result()``
    or by the retrieval node reading it from pipeline state), the adapter
    skips the redundant ``query_module.run()`` call -- saving one LLM call
    and one embedding call per query.
    """

    config: Any = None
    mode: str | None = None
    _precomputed_query_result: dict | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.config is None:
            self.config = _get_config()()

    def set_query_result(self, query_result: dict | None) -> None:
        """Inject a pre-computed query_module result to skip re-running it."""
        self._precomputed_query_result = query_result

    def retrieve(
        self, queries: list[str], top_k: int = 10
    ) -> list[RetrievalResult]:
        from lightrag_core_simplified.src.modules import query_module, retrieval_module

        if not queries:
            return []

        primary_query = queries[0]

        if self._precomputed_query_result is not None:
            query_result = self._precomputed_query_result
            self._precomputed_query_result = None
        else:
            query_result = query_module.run(self.config, primary_query)

        raw = retrieval_module.retrieve(
            self.config, primary_query, query_result, mode=self.mode,
        )

        all_results: list[RetrievalResult] = []
        seen: set[str] = set()

        for chunk_info in raw.get("context_chunks", []):
            cid = chunk_info.get("chunk_id", "")
            if cid in seen:
                continue
            seen.add(cid)
            doc_ids = chunk_info.get("doc_ids", [])
            all_results.append(RetrievalResult(
                source_id=cid,
                content=chunk_info.get("content", ""),
                score=chunk_info.get("score", 0.0),
                title=doc_ids[0] if doc_ids else cid,
                metadata={
                    "doc_ids": doc_ids,
                    "reference_id": chunk_info.get("reference_id", ""),
                    "retrieval_mode": raw.get("mode", "hybrid"),
                    "kg_entities": [
                        {"name": n["entity_name"], "type": n.get("entity_type", "")}
                        for n in raw.get("entities_structured", [])
                    ] if isinstance(raw.get("entities_structured"), list) else [],
                },
            ))

        if not raw.get("context_chunks"):
            for idx, (chunk_id, score) in enumerate(raw.get("chunks", [])):
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                all_results.append(RetrievalResult(
                    source_id=chunk_id,
                    content=raw.get("raw_context", ""),
                    score=score,
                    title=chunk_id,
                    metadata={"retrieval_mode": raw.get("mode", "hybrid")},
                ))

        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# Reranking adapter -- LLM-based context compression
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LightRAGReranking:
    """``rag_contracts.Reranking`` wrapping LightRAG's context compression.

    NOTE: This is context compression / distillation, not traditional
    cross-encoder reranking.  It compresses the full context into a focused
    evidence brief via LLM, rather than re-scoring individual passages.
    The compressed text is attached to each result's metadata.
    For standard cross-encoder reranking, see ``CrossEncoderReranking``
    in ``rag_contracts.reranking_methods``.
    """

    config: Any = None

    def __post_init__(self):
        if self.config is None:
            self.config = _get_config()()

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 10
    ) -> list[RetrievalResult]:
        from lightrag_core_simplified.src.modules import reranking_module

        if not results:
            return []

        full_context = "\n\n".join(r.content for r in results)
        compressed = reranking_module.run(self.config, query, full_context)

        reranked = []
        for r in results[:top_k]:
            reranked.append(RetrievalResult(
                source_id=r.source_id,
                content=r.content,
                score=r.score,
                title=r.title,
                metadata={
                    **r.metadata,
                    "compressed_context": compressed,
                },
            ))
        return reranked


# ═══════════════════════════════════════════════════════════════════════════════
# Generation adapter -- LLM answer generation
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LightRAGGeneration:
    """``rag_contracts.Generation`` wrapping LightRAG's answer generation.

    Builds structured context from RetrievalResults and generates an answer.
    If compressed context is available in metadata (from LightRAGReranking),
    it is included as reasoning notes.
    """

    config: Any = None

    def __post_init__(self):
        if self.config is None:
            self.config = _get_config()()

    def generate(
        self,
        query: str,
        context: list[RetrievalResult],
        instruction: str = "",
    ) -> GenerationResult:
        from lightrag_core_simplified.src.modules import generation_module

        raw_context = "\n\n".join(r.content for r in context)

        compressed = ""
        for r in context:
            c = r.metadata.get("compressed_context", "")
            if c:
                compressed = c
                break

        answer = generation_module.run(
            self.config, query, raw_context, compressed or "N/A",
        )

        return GenerationResult(
            output=answer,
            citations=[r.source_id for r in context],
            metadata={
                "raw_context_length": len(raw_context),
                "has_compressed_context": bool(compressed),
            },
        )
