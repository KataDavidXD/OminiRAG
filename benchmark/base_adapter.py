"""Shared helpers and base utilities for benchmark adapters.

Consolidates code duplicated across hotpotqa_adapter, ultradomain_adapter,
and alce_adapter: chunk-to-RetrievalResult conversion, real-dataset context
conversion, and async graph invocation.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
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


def get_context_for_item(item: dict) -> list[RetrievalResult]:
    """Resolve context from an eval item, supporting both real and sample data.

    Priority:
      1. ``context_results`` -- pre-built ``list[RetrievalResult]`` (real data)
      2. ``chunks`` -- KG sample dict (stub data)
      3. empty list
    """
    if "context_results" in item:
        return item["context_results"]
    chunks = item.get("chunks", {})
    if chunks:
        return sample_chunks_to_retrieval_results(chunks)
    return []


# ---------------------------------------------------------------------------
# Real-dataset context converters
# ---------------------------------------------------------------------------

def hotpotqa_context_to_retrieval_results(
    context: list,
) -> list[RetrievalResult]:
    """Convert HotpotQA's ``context`` field to ``list[RetrievalResult]``.

    Real HotpotQA format: ``[[title, [sent1, sent2, ...]], ...]``.
    Each entry becomes one ``RetrievalResult`` with sentences joined.
    """
    results: list[RetrievalResult] = []
    for idx, entry in enumerate(context):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        title = entry[0]
        sentences = entry[1]
        if isinstance(sentences, list):
            text = " ".join(sentences)
        else:
            text = str(sentences)
        results.append(RetrievalResult(
            source_id=f"hotpotqa_ctx_{idx}",
            content=text,
            score=1.0 - idx * 0.01,
            title=str(title),
            metadata={"passage_index": idx},
        ))
    return results


def ultradomain_context_to_retrieval_results(
    context: str,
    max_chunk_chars: int = 4000,
) -> list[RetrievalResult]:
    """Convert UltraDomain's long ``context`` string to ``list[RetrievalResult]``.

    Splits the context into chunks of *max_chunk_chars* so downstream
    components receive manageable passages.
    """
    if not context:
        return []
    results: list[RetrievalResult] = []
    for idx in range(0, len(context), max_chunk_chars):
        chunk = context[idx : idx + max_chunk_chars]
        results.append(RetrievalResult(
            source_id=f"ud_ctx_{idx // max_chunk_chars}",
            content=chunk,
            score=1.0 - (idx // max_chunk_chars) * 0.01,
            title=f"UltraDomain context chunk {idx // max_chunk_chars}",
            metadata={"char_offset": idx},
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
