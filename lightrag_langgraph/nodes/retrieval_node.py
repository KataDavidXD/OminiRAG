from __future__ import annotations

from rag_contracts import Retrieval


def _deduplicate(results, top_k):
    seen: dict[str, object] = {}
    for r in results:
        prev = seen.get(r.source_id)
        if prev is None or r.score > prev.score:
            seen[r.source_id] = r
    deduped = sorted(seen.values(), key=lambda r: r.score, reverse=True)
    return deduped[:top_k]


def build_node(retrieval: Retrieval):
    async def node(state):
        queries = state.get("expanded_queries", [state["query"]])
        precomputed = state.get("query_result")
        if precomputed and hasattr(retrieval, "set_query_result"):
            retrieval.set_query_result(precomputed)
        results = retrieval.retrieve(queries, top_k=10)
        return {"retrieval_results": _deduplicate(results, top_k=10)}

    return node
