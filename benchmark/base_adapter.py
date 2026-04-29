"""Shared helpers and base utilities for benchmark adapters.

Consolidates code duplicated across hotpotqa_adapter, ultradomain_adapter,
and alce_adapter: chunk-to-RetrievalResult conversion and async graph
invocation.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rag_contracts import GenerationResult, RetrievalResult


def sample_chunks_to_retrieval_results(chunks: dict) -> list[RetrievalResult]:
    """Convert KG sample chunks dict to RetrievalResult list.

    Expected format: ``{chunk_id: {content: str, doc_ids: list, ...}}``.
    """
    results = []
    for cid, info in chunks.items():
        doc_ids = info.get("doc_ids", [])
        results.append(RetrievalResult(
            source_id=cid,
            content=info.get("content", ""),
            score=1.0,
            title=doc_ids[0] if doc_ids else cid,
            metadata={"doc_ids": doc_ids},
        ))
    return results


def invoke_graph_sync(graph: Any, payload: dict) -> dict:
    """Invoke a LangGraph ``ainvoke`` synchronously, handling nested loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(graph.ainvoke(payload))
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, graph.ainvoke(payload)).result()
