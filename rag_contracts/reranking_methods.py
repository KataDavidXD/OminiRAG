"""Real reranking methods for rag_contracts pipelines.

These are standard post-retrieval reranking components that re-score
(query, passage) pairs using cross-encoder models -- the canonical
Reranking stage in RAG survey taxonomy.

- CrossEncoderReranking: cross-encoder re-scoring via sentence-transformers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import RetrievalResult


@dataclass
class CrossEncoderReranking:
    """Cross-encoder reranking via sentence-transformers.

    Default model: ``cross-encoder/ms-marco-MiniLM-L-12-v2`` (fast, accurate,
    widely used MSMARCO-trained cross-encoder).  Scores each ``(query, passage)``
    pair independently, then reorders by descending score.
    """

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    _model: Any = field(default=None, init=False, repr=False)

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 10
    ) -> list[RetrievalResult]:
        if not results:
            return []

        self._ensure_model()

        pairs = [[query, r.content] for r in results]
        scores = self._model.predict(pairs)

        scored = []
        for r, s in zip(results, scores):
            scored.append(RetrievalResult(
                source_id=r.source_id,
                content=r.content,
                score=float(s),
                title=r.title,
                metadata={**r.metadata, "reranking_method": "cross_encoder",
                          "original_score": r.score},
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
