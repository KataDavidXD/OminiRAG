"""Real retrieval methods for rag_contracts pipelines.

These are corpus-search components that build an index from documents
and retrieve by query similarity -- the standard Retrieval stage in
RAG survey taxonomy.

- BM25Retrieval: lexical matching via BM25Okapi (rank_bm25)
- DenseRetrieval: semantic matching via sentence-transformers embeddings
- HybridRetrieval: reciprocal rank fusion of BM25 + dense
- CorpusIndex: helper to load chunks from sample_data into the above
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .types import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class CorpusIndex:
    """In-memory corpus built from a ``chunks.json`` mapping.

    Expected format: ``{chunk_id: {content: str, doc_ids: list, ...}}``.
    After construction, ``chunk_ids``, ``texts``, and ``metadata`` are
    aligned parallel lists.
    """

    chunk_ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_chunks_dict(cls, chunks: dict[str, dict[str, Any]]) -> "CorpusIndex":
        ids, txts, metas = [], [], []
        for cid, info in chunks.items():
            content = info.get("content", "")
            if not content:
                continue
            ids.append(cid)
            txts.append(content)
            metas.append({k: v for k, v in info.items() if k != "content"})
        return cls(chunk_ids=ids, texts=txts, metadata=metas)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "CorpusIndex":
        with open(path, encoding="utf-8") as f:
            chunks = json.load(f)
        return cls.from_chunks_dict(chunks)

    def __len__(self) -> int:
        return len(self.chunk_ids)


@dataclass
class BM25Retrieval:
    """Lexical retrieval via BM25Okapi (``rank_bm25`` library).

    Build from a ``CorpusIndex`` or a raw list of texts.
    Tokenisation is whitespace + lowercasing (sufficient for English).
    """

    corpus: CorpusIndex | None = None
    _bm25: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.corpus is not None:
            self._build_index()

    def set_corpus(self, corpus: CorpusIndex) -> None:
        self.corpus = corpus
        self._build_index()

    def _build_index(self) -> None:
        if not self.corpus.texts:
            return
        from rank_bm25 import BM25Okapi

        tokenized = [t.lower().split() for t in self.corpus.texts]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(
        self, queries: list[str], top_k: int = 10
    ) -> list[RetrievalResult]:
        if self._bm25 is None or self.corpus is None or len(self.corpus) == 0:
            return []

        scores = np.zeros(len(self.corpus), dtype=np.float64)
        for q in queries:
            q_tokens = q.lower().split()
            scores += self._bm25.get_scores(q_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: list[RetrievalResult] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            results.append(RetrievalResult(
                source_id=self.corpus.chunk_ids[idx],
                content=self.corpus.texts[idx],
                score=float(scores[idx]),
                title=_first_doc_id(self.corpus.metadata[idx]),
                metadata={**self.corpus.metadata[idx], "retrieval_method": "bm25"},
            ))
        return results


@dataclass
class DenseRetrieval:
    """Dense semantic retrieval via sentence-transformers.

    Default model: ``intfloat/multilingual-e5-small`` (118M params, CPU-friendly,
    strong on BEIR, multilingual). Encodes corpus offline; cosine similarity
    at query time.
    """

    model_name: str = "intfloat/multilingual-e5-small"
    corpus: CorpusIndex | None = None
    _model: Any = field(default=None, init=False, repr=False)
    _corpus_embeddings: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.corpus is not None:
            self._build_index()

    def set_corpus(self, corpus: CorpusIndex) -> None:
        self.corpus = corpus
        self._build_index()

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def _build_index(self) -> None:
        self._ensure_model()
        passages = [f"passage: {t}" for t in self.corpus.texts]
        self._corpus_embeddings = self._model.encode(
            passages, normalize_embeddings=True, show_progress_bar=False,
        )

    def retrieve(
        self, queries: list[str], top_k: int = 10
    ) -> list[RetrievalResult]:
        if self._corpus_embeddings is None or self.corpus is None or len(self.corpus) == 0:
            return []

        self._ensure_model()
        prefixed = [f"query: {q}" for q in queries]
        q_embs = self._model.encode(
            prefixed, normalize_embeddings=True, show_progress_bar=False,
        )
        if q_embs.ndim == 1:
            q_embs = q_embs.reshape(1, -1)

        scores = q_embs @ self._corpus_embeddings.T
        merged = scores.max(axis=0)

        top_indices = np.argsort(merged)[::-1][:top_k]
        results: list[RetrievalResult] = []
        for idx in top_indices:
            if merged[idx] <= 0:
                break
            results.append(RetrievalResult(
                source_id=self.corpus.chunk_ids[idx],
                content=self.corpus.texts[idx],
                score=float(merged[idx]),
                title=_first_doc_id(self.corpus.metadata[idx]),
                metadata={**self.corpus.metadata[idx], "retrieval_method": "dense",
                          "model": self.model_name},
            ))
        return results


@dataclass
class HybridRetrieval:
    """Reciprocal Rank Fusion of BM25 + Dense retrieval.

    Runs both retrievers, merges with ``1 / (k + rank)`` scoring
    (k=60 by default, standard RRF constant). No tuning needed.
    """

    bm25: BM25Retrieval | None = None
    dense: DenseRetrieval | None = None
    rrf_k: int = 60
    corpus: CorpusIndex | None = None

    def __post_init__(self) -> None:
        if self.corpus is not None:
            if self.bm25 is None:
                self.bm25 = BM25Retrieval(corpus=self.corpus)
            if self.dense is None:
                self.dense = DenseRetrieval(corpus=self.corpus)

    def set_corpus(self, corpus: CorpusIndex) -> None:
        self.corpus = corpus
        if self.bm25 is None:
            self.bm25 = BM25Retrieval()
        if self.dense is None:
            self.dense = DenseRetrieval()
        self.bm25.set_corpus(corpus)
        self.dense.set_corpus(corpus)

    def retrieve(
        self, queries: list[str], top_k: int = 10
    ) -> list[RetrievalResult]:
        fetch_k = max(top_k * 3, 30)
        bm25_results = self.bm25.retrieve(queries, top_k=fetch_k) if self.bm25 else []
        dense_results = self.dense.retrieve(queries, top_k=fetch_k) if self.dense else []

        rrf_scores: dict[str, float] = {}
        result_map: dict[str, RetrievalResult] = {}

        for rank, r in enumerate(bm25_results):
            rrf_scores[r.source_id] = rrf_scores.get(r.source_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            result_map[r.source_id] = r

        for rank, r in enumerate(dense_results):
            rrf_scores[r.source_id] = rrf_scores.get(r.source_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            if r.source_id not in result_map:
                result_map[r.source_id] = r

        sorted_ids = sorted(rrf_scores, key=lambda sid: rrf_scores[sid], reverse=True)
        results: list[RetrievalResult] = []
        for sid in sorted_ids[:top_k]:
            base = result_map[sid]
            results.append(RetrievalResult(
                source_id=base.source_id,
                content=base.content,
                score=rrf_scores[sid],
                title=base.title,
                metadata={**base.metadata, "retrieval_method": "hybrid_rrf"},
            ))
        return results


def _first_doc_id(meta: dict[str, Any]) -> str:
    doc_ids = meta.get("doc_ids", [])
    if isinstance(doc_ids, list) and doc_ids:
        return str(doc_ids[0])
    return ""
