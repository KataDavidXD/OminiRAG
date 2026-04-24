"""Canonical RAG component contracts.

Defines the 6 replaceable component protocols and their shared data types.
Pipeline stages: Chunking -> Embedding -> Query -> Retrieval -> Reranking -> Generation
"""

from .identity import (
    IdentityChunking,
    IdentityEmbedding,
    IdentityGeneration,
    IdentityQuery,
    IdentityReranking,
)
from .protocols import Chunking, Embedding, Generation, Query, Reranking, Retrieval
from .types import Chunk, Document, GenerationResult, QueryContext, RetrievalResult

__all__ = [
    # Data types
    "Document",
    "Chunk",
    "RetrievalResult",
    "GenerationResult",
    "QueryContext",
    # Protocols
    "Chunking",
    "Embedding",
    "Query",
    "Retrieval",
    "Reranking",
    "Generation",
    # Identity / passthrough
    "IdentityChunking",
    "IdentityQuery",
    "IdentityEmbedding",
    "IdentityReranking",
    "IdentityGeneration",
]
