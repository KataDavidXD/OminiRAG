from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Chunk, Document, GenerationResult, QueryContext, RetrievalResult


@runtime_checkable
class Chunking(Protocol):
    """Stage 1 -- Split raw documents into retrieval units."""

    def chunk(self, documents: list[Document]) -> list[Chunk]: ...


@runtime_checkable
class Embedding(Protocol):
    """Stage 2 -- Convert text chunks into vector representations."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Query(Protocol):
    """Stage 3 -- Expand / decompose a user query into retrieval-ready queries."""

    def process(self, query: str, context: QueryContext) -> list[str]: ...


@runtime_checkable
class Retrieval(Protocol):
    """Stage 4 -- First-stage retrieval of candidate chunks.

    **Multi-query semantics**: when *queries* contains more than one
    entry the implementation should retrieve for each query and merge the
    results into a single list.  Deduplication (by ``source_id``) and
    the final ``top_k`` cap are applied to the merged set, not per-query.
    Implementations that only handle a single query should iterate
    internally or document the limitation.
    """

    def retrieve(
        self, queries: list[str], top_k: int = 10
    ) -> list[RetrievalResult]: ...


@runtime_checkable
class Reranking(Protocol):
    """Stage 5 -- Second-stage reordering / filtering of retrieval candidates."""

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 10
    ) -> list[RetrievalResult]: ...


@runtime_checkable
class Generation(Protocol):
    """Stage 6 -- Produce final output from query + reranked context."""

    def generate(
        self,
        query: str,
        context: list[RetrievalResult],
        instruction: str = "",
    ) -> GenerationResult: ...
