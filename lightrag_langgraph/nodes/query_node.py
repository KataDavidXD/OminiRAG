from __future__ import annotations

from rag_contracts import IdentityQuery, Query, QueryContext


def build_node(query: Query | None = None):
    _query = query or IdentityQuery()

    async def node(state):
        raw_query = state["query"]
        context = QueryContext(topic=raw_query)
        expanded = _query.process(raw_query, context)
        result: dict = {"expanded_queries": expanded}
        cached = getattr(_query, "_last_query_result", None)
        if cached:
            result["query_result"] = cached
        return result

    return node
