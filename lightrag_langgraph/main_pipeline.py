"""LightRAG pipeline rebuilt with LangGraph and canonical rag_contracts.

Provides two pipelines:
- build_query_graph(): online query pipeline (query -> retrieve -> rerank -> generate)
- build_index_graph(): offline indexing (chunk -> KG extract -> embed)
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from rag_contracts import Generation, Query, Reranking, Retrieval

from .nodes.generation_node import build_node as generation_node
from .nodes.query_node import build_node as query_node
from .nodes.reranking_node import build_node as reranking_node
from .nodes.retrieval_node import build_node as retrieval_node
from .state import LightRAGGraphState


def build_query_graph(
    *,
    retrieval: Retrieval,
    generation: Generation,
    reranking: Reranking | None = None,
    query: Query | None = None,
):
    """Build the LightRAG online query pipeline with DI.

    Four nodes:
      query_processing -> retrieval -> reranking -> generation -> END
    """
    graph = StateGraph(LightRAGGraphState)
    graph.add_node("query_processing", query_node(query))
    graph.add_node("retrieval", retrieval_node(retrieval))
    graph.add_node("reranking", reranking_node(reranking))
    graph.add_node("generation", generation_node(generation))

    graph.set_entry_point("query_processing")
    graph.add_edge("query_processing", "retrieval")
    graph.add_edge("retrieval", "reranking")
    graph.add_edge("reranking", "generation")
    graph.add_edge("generation", END)
    return graph.compile()


def build_index_graph(config=None):
    """Build the LightRAG offline indexing pipeline.

    Wraps the existing index pipeline from lightrag_core_simplified.
    Returns the compiled LangGraph for: chunk -> graph_extract -> embed -> END

    Order matters: graph extraction produces the KG delta (entities/relations)
    that the embedding node needs to build the 3-way vector stores.
    """
    import sys
    from pathlib import Path

    src = str(
        Path(__file__).resolve().parent.parent
        / "A-Simplified-Core-Workflow-for-Enhancing-RAG"
        / "lightrag_core_simplified"
        / "src"
    )
    if src not in sys.path:
        sys.path.insert(0, str(Path(src).parent))

    from lightrag_core_simplified.src.config import Config
    from lightrag_core_simplified.src.nodes.chunk_node import build_node as chunk_node
    from lightrag_core_simplified.src.nodes.embedding_node import build_node as embed_node
    from lightrag_core_simplified.src.nodes.graph_node import build_node as graph_node

    if config is None:
        config = Config()

    g = StateGraph(dict)
    g.add_node("chunk", chunk_node(config))
    g.add_node("graph_extract", graph_node(config))
    g.add_node("embed", embed_node(config))

    g.set_entry_point("chunk")
    g.add_edge("chunk", "graph_extract")
    g.add_edge("graph_extract", "embed")
    g.add_edge("embed", END)
    return g.compile()
