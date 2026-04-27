"""Per-dataset evaluators that know each benchmark's data shape.

Each evaluator wraps a metric strategy and converts raw dicts
(with ``prediction``, ``answer``, etc.) into ``ItemScore`` / ``EvaluationResult``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from bsamp.scoring.metrics import (
    ASQAMetrics,
    LLMJudgeMetrics,
    PairwiseJudgeMetrics,
    QampariMetrics,
    ShortFormMetrics,
)
from bsamp.scoring.scoring import compute_f1, remove_citations
from bsamp.scoring.types import EvaluationResult, ItemScore


class DatasetEvaluator(ABC):
    """Base class for all dataset evaluators.

    Subclasses must implement ``score_item`` and may override ``score_batch``
    for custom aggregation logic.
    """

    benchmark: str = ""

    @abstractmethod
    def score_item(self, item: dict[str, Any]) -> ItemScore: ...

    def score_batch(self, items: list[dict[str, Any]]) -> EvaluationResult:
        scored = [self.score_item(it) for it in items]
        agg = self.aggregate([s.metrics for s in scored])
        return EvaluationResult(
            benchmark=self.benchmark,
            num_items=len(scored),
            aggregate=agg,
            per_item=scored,
        )

    @staticmethod
    @abstractmethod
    def aggregate(item_scores: list[dict[str, float]]) -> dict[str, float]: ...


class HotpotQAEvaluator(DatasetEvaluator):
    """Score HotpotQA predictions with EM + F1."""

    benchmark = "hotpotqa"

    def score_item(self, item: dict[str, Any]) -> ItemScore:
        pred = str(item.get("prediction", "")).strip()
        gold = item.get("answer", "")
        if isinstance(gold, list):
            gold = gold[0] if gold else ""
        scores = ShortFormMetrics.score_item(gold, pred)
        return ItemScore(
            item_id=str(item.get("query_id", "")),
            metrics=scores,
            prediction=pred,
            gold=gold,
            metadata={
                k: item[k] for k in ("question", "_stratum") if k in item
            },
        )

    @staticmethod
    def aggregate(item_scores: list[dict[str, float]]) -> dict[str, float]:
        return ShortFormMetrics.aggregate(item_scores)


class ASQAEvaluator(DatasetEvaluator):
    """Score ALCE/ASQA predictions with STR-EM + token-F1."""

    benchmark = "asqa"

    def score_item(self, item: dict[str, Any]) -> ItemScore:
        pred = str(item.get("prediction", "")).strip()
        pred_first_line = pred.split("\n")[0]
        gold = item.get("answer", "")
        if isinstance(gold, list):
            gold = gold[0] if gold else ""
        qa_pairs = item.get("qa_pairs")
        scores = ASQAMetrics.score_item(gold, pred_first_line, qa_pairs=qa_pairs)
        return ItemScore(
            item_id=str(item.get("query_id", "")),
            metrics=scores,
            prediction=pred_first_line,
            gold=gold,
            metadata={k: item[k] for k in ("question",) if k in item},
        )

    @staticmethod
    def aggregate(item_scores: list[dict[str, float]]) -> dict[str, float]:
        return ASQAMetrics.aggregate(item_scores)


class QampariEvaluator(DatasetEvaluator):
    """Score ALCE/QAMPARI predictions with precision/recall/F1.

    Expects ``item["answers"]`` to be a list-of-lists (answer alias groups)
    and ``item["prediction"]`` to be a comma-separated string.
    """

    benchmark = "qampari"

    def __init__(self, cot: bool = False):
        self.cot = cot

    def score_item(self, item: dict[str, Any]) -> ItemScore:
        pred = str(item.get("prediction", "")).strip()
        answers = item.get("answers", [])
        scores = QampariMetrics.score_item(answers, pred, cot=self.cot)
        return ItemScore(
            item_id=str(item.get("query_id", "")),
            metrics=scores,
            prediction=pred,
            gold=answers,
            metadata={k: item[k] for k in ("question",) if k in item},
        )

    @staticmethod
    def aggregate(item_scores: list[dict[str, float]]) -> dict[str, float]:
        return QampariMetrics.aggregate(item_scores)


class UltraDomainEvaluator(DatasetEvaluator):
    """Score UltraDomain predictions with LLM-as-judge (+ optional F1).

    Parameters
    ----------
    llm_fn :
        ``(system_prompt, user_prompt, **kwargs) -> str``.  When *None*,
        LLM-judge dimensions default to 0.
    mode :
        ``"direct"`` for 1-5 scoring (OminiRAG default) or ``"pairwise"``
        for A-vs-B comparison matching the LightRAG paper protocol.
        In pairwise mode, each item must have ``"answer_b"`` (the baseline).
    """

    benchmark = "ultradomain"

    def __init__(
        self,
        llm_fn: Callable[..., str] | None = None,
        mode: str = "direct",
    ):
        self.llm_fn = llm_fn
        self.mode = mode

    def score_item(self, item: dict[str, Any]) -> ItemScore:
        pred = str(item.get("prediction", "")).strip()
        question = item.get("question", "")
        domain = item.get("domain", "general")

        judge_scores = LLMJudgeMetrics.score_item(
            question, pred, domain=domain, llm_fn=self.llm_fn,
        )

        ref_answer = item.get("answer", "")
        if isinstance(ref_answer, list):
            ref_answer = ref_answer[0] if ref_answer else ""
        f1 = compute_f1(ref_answer, pred) if ref_answer else 0.0

        metrics = {**judge_scores, "f1": f1, "length": float(len(pred.split()))}
        return ItemScore(
            item_id=str(item.get("query_id", "")),
            metrics=metrics,
            prediction=pred,
            gold=ref_answer,
            metadata={
                k: item[k] for k in ("question", "domain") if k in item
            },
        )

    def score_item_pairwise(
        self,
        item: dict[str, Any],
        answer_b: str,
    ) -> ItemScore:
        """Pairwise comparison of ``item["prediction"]`` (A) vs *answer_b* (B)."""
        pred = str(item.get("prediction", "")).strip()
        question = item.get("question", "")
        winners = PairwiseJudgeMetrics.score_item(
            question, pred, answer_b, llm_fn=self.llm_fn,
        )
        return ItemScore(
            item_id=str(item.get("query_id", "")),
            metrics={k: 0.0 for k in winners},
            prediction=pred,
            gold=answer_b,
            metadata={**winners, "question": question},
        )

    @staticmethod
    def aggregate(item_scores: list[dict[str, float]]) -> dict[str, float]:
        agg = LLMJudgeMetrics.aggregate(item_scores)
        n = len(item_scores) or 1
        agg["avg_f1"] = 100 * sum(s.get("f1", 0) for s in item_scores) / n
        agg["avg_length"] = sum(s.get("length", 0) for s in item_scores) / n
        return agg

    def score_batch(self, items: list[dict[str, Any]]) -> EvaluationResult:
        if self.mode == "pairwise":
            return self._score_batch_pairwise(items)
        return super().score_batch(items)

    def _score_batch_pairwise(self, items: list[dict[str, Any]]) -> EvaluationResult:
        scored = []
        for it in items:
            answer_b = it.get("answer_b", it.get("answer", ""))
            scored.append(self.score_item_pairwise(it, answer_b))
        winner_dicts = [s.metadata for s in scored]
        agg = PairwiseJudgeMetrics.aggregate(winner_dicts)
        return EvaluationResult(
            benchmark="ultradomain_pairwise",
            num_items=len(scored),
            aggregate=agg,
            per_item=scored,
        )
