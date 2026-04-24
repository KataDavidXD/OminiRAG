"""Passthrough / identity implementations for pipelines that skip a stage."""

from __future__ import annotations

from .types import Chunk, Document, GenerationResult, QueryContext, RetrievalResult


class IdentityChunking:
    """Treats each document as a single chunk (no splitting)."""

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=doc.doc_id,
                doc_id=doc.doc_id,
                content=doc.content,
                metadata=dict(doc.metadata),
            )
            for doc in documents
        ]


class IdentityQuery:
    """Returns the original query unchanged (no expansion)."""

    def process(self, query: str, context: QueryContext) -> list[str]:
        return [query]


class IdentityEmbedding:
    """Returns empty vectors (placeholder when embedding is not needed)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


class IdentityReranking:
    """Returns retrieval results unchanged (no reranking)."""

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 10
    ) -> list[RetrievalResult]:
        return results[:top_k]


class IdentityGeneration:
    """Concatenates context as-is (placeholder when generation is not needed)."""

    def generate(
        self,
        query: str,
        context: list[RetrievalResult],
        instruction: str = "",
    ) -> GenerationResult:
        output = "\n\n".join(r.content for r in context) if context else ""
        return GenerationResult(
            output=output,
            citations=[r.source_id for r in context],
        )
