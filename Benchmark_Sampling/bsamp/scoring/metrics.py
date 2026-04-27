"""Metric strategy classes for different benchmark types.

Each class exposes ``score_item(...)`` returning a ``dict[str, float]`` and
``aggregate(item_scores)`` returning averaged metrics.  All functions are
pure string-in / number-out with no model or LLM dependencies (except
``LLMJudgeMetrics`` and ``PairwiseJudgeMetrics`` which accept a callable).
"""

from __future__ import annotations

import json
from typing import Any, Callable

from bsamp.scoring.scoring import (
    compute_exact,
    compute_f1,
    exact_presence,
    normalize_answer,
    remove_citations,
)


class ShortFormMetrics:
    """EM + token-F1 for short-answer QA (HotpotQA, PopQA, TriviaQA)."""

    @staticmethod
    def score_item(gold: str, pred: str) -> dict[str, float]:
        return {
            "em": float(compute_exact(gold, pred)),
            "f1": compute_f1(gold, pred),
        }

    @staticmethod
    def aggregate(item_scores: list[dict[str, float]]) -> dict[str, float]:
        n = len(item_scores) or 1
        return {
            "avg_em": 100 * sum(s["em"] for s in item_scores) / n,
            "avg_f1": 100 * sum(s["f1"] for s in item_scores) / n,
        }


class ASQAMetrics:
    """STR-EM + token-F1 for ALCE/ASQA long-form QA with qa_pairs."""

    @staticmethod
    def score_item(
        gold: str,
        pred: str,
        qa_pairs: list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        pred_clean = remove_citations(pred)
        f1 = compute_f1(gold, pred_clean)
        em = float(compute_exact(gold, pred_clean))

        str_em = 0.0
        str_hit = 0.0
        if qa_pairs:
            loc_acc = [
                float(exact_presence(qp.get("short_answers", []), pred_clean))
                for qp in qa_pairs
            ]
            str_em = sum(loc_acc) / len(loc_acc) if loc_acc else 0.0
            str_hit = float(str_em == 1.0)

        return {
            "f1": f1,
            "exact": em,
            "str_em": str_em,
            "str_hit": str_hit,
            "length": float(len(pred_clean.split())),
        }

    @staticmethod
    def aggregate(item_scores: list[dict[str, float]]) -> dict[str, float]:
        n = len(item_scores) or 1
        return {
            "avg_f1": 100 * sum(s["f1"] for s in item_scores) / n,
            "avg_exact": 100 * sum(s["exact"] for s in item_scores) / n,
            "avg_str_em": 100 * sum(s["str_em"] for s in item_scores) / n,
            "avg_str_hit": 100 * sum(s["str_hit"] for s in item_scores) / n,
            "avg_length": sum(s["length"] for s in item_scores) / n,
        }


class QampariMetrics:
    """Precision / Recall / F1 for ALCE/QAMPARI list-answer evaluation.

    Matches the ALCE ``compute_qampari_f1()`` logic exactly: split on
    commas, normalize, compute against gold answer-alias lists.
    """

    @staticmethod
    def score_item(
        answers: list[list[str]],
        pred: str,
        cot: bool = False,
    ) -> dict[str, float]:
        if cot:
            o = ":".join(pred.split(":")[1:]) if ":" in pred else ""
        else:
            o = pred

        preds = [
            normalize_answer(x.strip())
            for x in o.rstrip().rstrip(".").rstrip(",").split(",")
        ]
        preds = [p for p in preds if len(p) > 0]

        flat_answers = [
            normalize_answer(alias)
            for group in answers
            for alias in group
        ]
        norm_answers = [
            [normalize_answer(alias) for alias in group]
            for group in answers
        ]

        num_preds = float(len(preds))
        precision = (
            sum(p in flat_answers for p in preds) / len(preds)
            if preds
            else 0.0
        )
        recall = (
            sum(any(alias in preds for alias in group) for group in norm_answers)
            / len(norm_answers)
            if norm_answers
            else 0.0
        )
        recall_top5 = (
            min(5, sum(any(alias in preds for alias in group) for group in norm_answers))
            / min(5, len(norm_answers))
            if norm_answers
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1_top5 = (
            2 * precision * recall_top5 / (precision + recall_top5)
            if (precision + recall_top5) > 0
            else 0.0
        )

        return {
            "num_preds": num_preds,
            "precision": precision,
            "recall": recall,
            "recall_top5": recall_top5,
            "f1": f1,
            "f1_top5": f1_top5,
        }

    @staticmethod
    def aggregate(item_scores: list[dict[str, float]]) -> dict[str, float]:
        n = len(item_scores) or 1
        return {
            "avg_num_preds": sum(s["num_preds"] for s in item_scores) / n,
            "avg_precision": 100 * sum(s["precision"] for s in item_scores) / n,
            "avg_recall": 100 * sum(s["recall"] for s in item_scores) / n,
            "avg_recall_top5": 100 * sum(s["recall_top5"] for s in item_scores) / n,
            "avg_f1": 100 * sum(s["f1"] for s in item_scores) / n,
            "avg_f1_top5": 100 * sum(s["f1_top5"] for s in item_scores) / n,
        }


# ---------------------------------------------------------------------------
# LLM-based metrics (accept a callable, never import openai themselves)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are evaluating an AI assistant's answer to a domain-specific question.
Score the answer on three dimensions (each 1-5):

1. **Comprehensiveness**: Does the answer cover all important aspects of the question?
2. **Diversity**: Does the answer draw from varied perspectives, methods, or viewpoints?
3. **Empowerment**: Does the answer help the user understand deeply and act on the information?

Output ONLY a JSON object: {"comprehensiveness": N, "diversity": N, "empowerment": N}
"""

_JUDGE_USER = """\
Domain: {domain}
Question: {question}
Answer: {answer}
"""


class LLMJudgeMetrics:
    """Direct 1-5 scoring per dimension via an LLM judge."""

    @staticmethod
    def score_item(
        question: str,
        pred: str,
        domain: str = "general",
        llm_fn: Callable[..., str] | None = None,
    ) -> dict[str, float]:
        if llm_fn is None:
            return {"comprehensiveness": 0.0, "diversity": 0.0, "empowerment": 0.0}
        user_text = _JUDGE_USER.format(
            domain=domain, question=question, answer=pred,
        )
        try:
            raw = llm_fn(_JUDGE_SYSTEM, user_text, temperature=0.0, max_tokens=100)
            scores = json.loads(raw.strip())
            return {
                "comprehensiveness": float(scores.get("comprehensiveness", 0)),
                "diversity": float(scores.get("diversity", 0)),
                "empowerment": float(scores.get("empowerment", 0)),
            }
        except Exception:
            return {"comprehensiveness": 0.0, "diversity": 0.0, "empowerment": 0.0}

    @staticmethod
    def aggregate(item_scores: list[dict[str, float]]) -> dict[str, float]:
        n = len(item_scores) or 1
        return {
            "avg_comprehensiveness": sum(s["comprehensiveness"] for s in item_scores) / n,
            "avg_diversity": sum(s["diversity"] for s in item_scores) / n,
            "avg_empowerment": sum(s["empowerment"] for s in item_scores) / n,
        }


_PAIRWISE_SYSTEM = """\
---Role---
You are an expert tasked with evaluating two answers to the same question \
based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**."""

_PAIRWISE_USER = """\
You will evaluate two answers to the same question based on three criteria: \
**Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all \
aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different \
perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and \
make informed judgments about the topic?

For each criterion, choose the better answer (either Answer 1 or Answer 2) \
and explain why.  Then, select an overall winner based on these three categories.

Here is the question:
{query}

Here are the two answers:

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the three criteria listed above and provide \
detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{{"Comprehensiveness": {{"Winner": "[Answer 1 or Answer 2]", \
"Explanation": "[Provide explanation here]"}}, \
"Diversity": {{"Winner": "[Answer 1 or Answer 2]", \
"Explanation": "[Provide explanation here]"}}, \
"Empowerment": {{"Winner": "[Answer 1 or Answer 2]", \
"Explanation": "[Provide explanation here]"}}, \
"Overall Winner": {{"Winner": "[Answer 1 or Answer 2]", \
"Explanation": "[Summarize why this answer is the overall winner]"}}}}
"""


class PairwiseJudgeMetrics:
    """Pairwise comparison matching the LightRAG paper evaluation protocol.

    Compares answer_a vs answer_b and returns per-dimension winners.
    Alternates answer order and averages to mitigate order bias.
    """

    @staticmethod
    def score_item(
        question: str,
        answer_a: str,
        answer_b: str,
        llm_fn: Callable[..., str] | None = None,
    ) -> dict[str, str]:
        if llm_fn is None:
            return {
                "comp_winner": "tie",
                "div_winner": "tie",
                "emp_winner": "tie",
                "overall_winner": "tie",
            }

        def _call(a1: str, a2: str) -> dict[str, str]:
            user_text = _PAIRWISE_USER.format(
                query=question, answer1=a1, answer2=a2,
            )
            try:
                raw = llm_fn(_PAIRWISE_SYSTEM, user_text, temperature=0.0, max_tokens=500)
                result = json.loads(raw.strip())
                return {
                    "comp": result.get("Comprehensiveness", {}).get("Winner", ""),
                    "div": result.get("Diversity", {}).get("Winner", ""),
                    "emp": result.get("Empowerment", {}).get("Winner", ""),
                    "overall": result.get("Overall Winner", {}).get("Winner", ""),
                }
            except Exception:
                return {"comp": "", "div": "", "emp": "", "overall": ""}

        fwd = _call(answer_a, answer_b)
        rev = _call(answer_b, answer_a)

        def _resolve(fwd_val: str, rev_val: str) -> str:
            fwd_is_a = "1" in fwd_val
            rev_is_a = "2" in rev_val
            if fwd_is_a and rev_is_a:
                return "A"
            if not fwd_is_a and not rev_is_a:
                return "B"
            return "tie"

        return {
            "comp_winner": _resolve(fwd["comp"], rev["comp"]),
            "div_winner": _resolve(fwd["div"], rev["div"]),
            "emp_winner": _resolve(fwd["emp"], rev["emp"]),
            "overall_winner": _resolve(fwd["overall"], rev["overall"]),
        }

    @staticmethod
    def aggregate(item_scores: list[dict[str, str]], label_a: str = "A") -> dict[str, float]:
        n = len(item_scores) or 1
        result = {}
        for dim in ("comp", "div", "emp", "overall"):
            key = f"{dim}_winner"
            wins = sum(1 for s in item_scores if s.get(key) == label_a)
            result[f"{dim}_win_rate"] = 100 * wins / n
        return result
