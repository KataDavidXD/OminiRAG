from __future__ import annotations

from typing import Optional, TypedDict

from rag_contracts import GenerationResult, RetrievalResult


class LightRAGGraphState(TypedDict, total=False):
    # Input
    query: str
    mode: str

    # Stage outputs
    expanded_queries: list[str]
    query_result: dict
    retrieval_results: list[RetrievalResult]
    generation_result: GenerationResult

    # Error tracking
    error: Optional[str]
