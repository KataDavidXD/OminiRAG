"""bsamp.scoring -- Unified scoring and evaluation for RAG benchmarks.

Pure string-in / number-out scoring functions with zero dependency on
rag_contracts or any OminiRAG module.  Any RAG system can use this package
to evaluate its outputs against HotpotQA, ALCE (ASQA / QAMPARI), and
UltraDomain benchmarks.
"""

from bsamp.scoring.types import EvaluationResult, ItemScore
from bsamp.scoring.scoring import (
    normalize_answer,
    compute_f1,
    compute_exact,
    exact_presence,
    remove_citations,
)
from bsamp.scoring.metrics import (
    ShortFormMetrics,
    ASQAMetrics,
    QampariMetrics,
    LLMJudgeMetrics,
    PairwiseJudgeMetrics,
)
from bsamp.scoring.evaluator import (
    DatasetEvaluator,
    HotpotQAEvaluator,
    ASQAEvaluator,
    QampariEvaluator,
    UltraDomainEvaluator,
)

__all__ = [
    "EvaluationResult",
    "ItemScore",
    "normalize_answer",
    "compute_f1",
    "compute_exact",
    "exact_presence",
    "remove_citations",
    "ShortFormMetrics",
    "ASQAMetrics",
    "QampariMetrics",
    "LLMJudgeMetrics",
    "PairwiseJudgeMetrics",
    "DatasetEvaluator",
    "HotpotQAEvaluator",
    "ASQAEvaluator",
    "QampariEvaluator",
    "UltraDomainEvaluator",
]
