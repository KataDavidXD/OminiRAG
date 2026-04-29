"""
Data types for the WTB x OminiRAG bipartite cache-reuse system.

RAGConfig     -- 5-slot pipeline configuration (new taxonomy)
BenchmarkQuestion -- question identity + payload for cache keying
WorkItem      -- binding of (config, question, reuse depth) for batch scheduling

Taxonomy (5 dimensions):
  chunking -> query -> retrieval -> post_retrieval -> generation
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Optional, Tuple

# Canonical node execution order shared by all pipeline topologies.
NODE_ORDER: Tuple[str, ...] = (
    "query_processing",
    "retrieval",
    "post_retrieval",
    "generation",
)

VALID_CHUNKING = frozenset({
    "standard_passage", "longrag_4k", "kg_extraction",
})

VALID_QUERY = frozenset({
    "identity", "lightrag_keywords",
})

VALID_RETRIEVAL = frozenset({
    "bm25", "dense_e5", "bm25_dense_hybrid",
    "lightrag_hybrid", "lightrag_graph",
})

VALID_POST_RETRIEVAL = frozenset({
    "identity", "cross_encoder", "lightrag_compress", "selfrag_critique",
})

VALID_GENERATION = frozenset({
    "longrag_reader", "lightrag_answer", "selfrag_generator", "simple_llm",
})

# Hard constraint: these retrieval methods require kg_extraction chunking
CHUNKING_RETRIEVAL_CONSTRAINTS: Dict[str, str] = {
    "lightrag_hybrid": "kg_extraction",
    "lightrag_graph": "kg_extraction",
}

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


@dataclass(frozen=True)
class RAGConfig:
    """A fully-specified RAG pipeline configuration.

    Five dimensions aligned with RAG survey taxonomy:
    chunking -> query -> retrieval -> post_retrieval -> generation

    The tuple order matches the LangGraph node execution order so that
    ``prefix(d)`` corresponds to the state after executing the first *d*
    pipeline stages.
    """

    chunking: str
    query: str
    retrieval: str
    post_retrieval: str
    generation: str

    def slots(self) -> Tuple[str, ...]:
        return (self.chunking, self.query, self.retrieval, self.post_retrieval, self.generation)

    def prefix(self, depth: int) -> Tuple[str, ...]:
        """Return the first *depth* slot values (0 <= depth <= 5)."""
        return self.slots()[:depth]

    def config_key(self) -> str:
        """Stable slash-separated key for storage and display."""
        return "/".join(self.slots())

    def state_key(self) -> Hashable:
        """Content-addressable key compatible with AG-UCT SearchState."""
        return self.slots()

    @classmethod
    def from_tuple(cls, t: Tuple[str, ...]) -> "RAGConfig":
        if len(t) != 5:
            raise ValueError(f"Expected 5-tuple, got {len(t)}: {t}")
        # Detect legacy format: first element is an old frame name
        if t[0] in _LEGACY_FRAME_TO_CHUNKING:
            chunking = _LEGACY_FRAME_TO_CHUNKING[t[0]]
            retrieval = _LEGACY_RETRIEVAL.get(t[2], t[2])
            post_ret = _LEGACY_RERANKING.get(t[3], t[3])
            return cls(chunking=chunking, query=t[1], retrieval=retrieval,
                       post_retrieval=post_ret, generation=t[4])
        return cls(chunking=t[0], query=t[1], retrieval=t[2],
                   post_retrieval=t[3], generation=t[4])


@dataclass(frozen=True)
class BenchmarkQuestion:
    """Identity + payload of a single benchmark question.

    ``question_id`` is the bipartite reuse key's right-hand side.
    Convention: ``{benchmark}::{stratum}::{index}`` for deterministic
    naming from SamplingEngine seeds.
    """

    question_id: str
    cluster_id: str
    stratum: str
    question: str
    payload: Dict[str, Any]
    target: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_benchmark_item(cls, item: Any, cluster_id: str) -> "BenchmarkQuestion":
        """Adapt a ``bsamp.sampling.types.BenchmarkItem`` to this type."""
        return cls(
            question_id=f"{cluster_id}::{item.stratum}::{item.item_id}",
            cluster_id=cluster_id,
            stratum=item.stratum,
            question=item.payload.get("question", item.payload.get("query", "")),
            payload=dict(item.payload),
            target=dict(item.target) if item.target else {},
            metadata=dict(item.metadata) if item.metadata else {},
        )


@dataclass
class WorkItem:
    """A scheduled unit of work: one (config, question) pair with reuse info.

    Populated during the partition phase of ``run_batch_with_reuse``.
    """

    config: RAGConfig
    question: BenchmarkQuestion
    reuse_depth: int = 0
    reuse_entry: Optional[Any] = None  # MaterializedEntry or None

    @property
    def is_full_hit(self) -> bool:
        return self.reuse_depth >= len(NODE_ORDER) + 1  # depth 5 = all 4 nodes + frame

    @property
    def is_partial_hit(self) -> bool:
        return 0 < self.reuse_depth < len(NODE_ORDER) + 1

    @property
    def is_full_miss(self) -> bool:
        return self.reuse_depth == 0


def state_content_hash(state: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 of a LangGraph state dict.

    Used to validate that a forked checkpoint matches the expected
    intermediate state (guards against false cache hits).
    """
    canonical = json.dumps(state, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
