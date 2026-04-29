"""
OminiRAG x WTB Integration -- Bipartite Cache-Reuse Batch Execution.

Bridges AG-UCT's logical reuse graph (materialized_keys / Path_t) with
WTB's physical checkpoint storage so that shared config prefixes across
RAG pipeline configurations avoid redundant computation.
"""

from .config_types import RAGConfig, BenchmarkQuestion, WorkItem, NODE_ORDER
from .reuse_ledger import ReuseLedger, MaterializedEntry
from .graph_factories import config_to_graph_factory, build_pipeline_components
from .batch_runner import run_batch_with_reuse, record_checkpoints
from .cache_aware_evaluator import RAGCacheAwareEvaluator

__all__ = [
    "RAGConfig",
    "BenchmarkQuestion",
    "WorkItem",
    "NODE_ORDER",
    "ReuseLedger",
    "MaterializedEntry",
    "config_to_graph_factory",
    "build_pipeline_components",
    "run_batch_with_reuse",
    "record_checkpoints",
    "RAGCacheAwareEvaluator",
]
