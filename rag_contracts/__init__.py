"""Canonical RAG component contracts.

Defines the 6 replaceable component protocols and their shared data types.
Pipeline stages: Chunking -> Embedding -> Query -> Retrieval -> Post-Retrieval -> Generation

The taxonomy distinguishes:
- **Real retrieval** (BM25, Dense, Hybrid, LightRAG graph+vector) -- corpus search
- **Utility components** (LLMRetrieval, DuckDuckGo, ALCE/HFDataset adapters) -- not comparable
- **Post-retrieval** (CrossEncoder, LightRAG compress, SelfRAG critique) -- reranking/filtering
"""

from .common_components import (
    ALCEDocRetrieval,
    DuckDuckGoRetrieval,
    FallbackRetrieval,
    LLMRetrieval,
    SimpleLLMGeneration,
)
from .identity import (
    IdentityChunking,
    IdentityEmbedding,
    IdentityGeneration,
    IdentityQuery,
    IdentityReranking,
)
from .protocols import Chunking, Embedding, Generation, Query, Reranking, Retrieval
from .reranking_methods import CrossEncoderReranking
from .retrieval_methods import (
    BM25Retrieval,
    CorpusIndex,
    DenseRetrieval,
    HybridRetrieval,
)
from .component_registry import build_pipeline_from_config, build_simple_llm
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
    # Real retrieval methods (corpus search)
    "BM25Retrieval",
    "DenseRetrieval",
    "HybridRetrieval",
    "CorpusIndex",
    # Standard reranking methods
    "CrossEncoderReranking",
    # Utility / demo components (not comparable retrieval methods)
    "LLMRetrieval",
    "DuckDuckGoRetrieval",
    "FallbackRetrieval",
    "ALCEDocRetrieval",
    "SimpleLLMGeneration",
    # Pipeline builder
    "build_pipeline_from_config",
    "build_simple_llm",
]
