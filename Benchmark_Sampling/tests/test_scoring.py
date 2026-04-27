"""Tests for bsamp.scoring -- scoring primitives, metrics, and evaluators.

All tests use synthetic data; no LLM calls or model loading required.
"""

from __future__ import annotations

import json

import pytest

from bsamp.scoring.scoring import (
    compute_exact,
    compute_f1,
    exact_presence,
    normalize_answer,
    remove_citations,
)
from bsamp.scoring.metrics import (
    ASQAMetrics,
    LLMJudgeMetrics,
    QampariMetrics,
    ShortFormMetrics,
)
from bsamp.scoring.evaluator import (
    ASQAEvaluator,
    HotpotQAEvaluator,
    QampariEvaluator,
    UltraDomainEvaluator,
)
from bsamp.scoring.types import EvaluationResult, ItemScore


# ═══════════════════════════════════════════════════════════════════════════════
# scoring.py primitives
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_remove_articles(self):
        assert normalize_answer("the cat") == "cat"
        assert normalize_answer("a dog") == "dog"
        assert normalize_answer("an apple") == "apple"

    def test_remove_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_whitespace_fix(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_combined(self):
        assert normalize_answer("The City of Paris!") == "city of paris"


class TestComputeF1:
    def test_exact_match(self):
        assert compute_f1("Paris", "Paris") == 1.0

    def test_no_overlap(self):
        assert compute_f1("Paris", "London") == 0.0

    def test_partial_overlap(self):
        f1 = compute_f1("the city of Paris", "Paris is great")
        assert 0.0 < f1 < 1.0

    def test_empty(self):
        assert compute_f1("", "") == 1.0
        assert compute_f1("hello", "") == 0.0
        assert compute_f1("", "hello") == 0.0


class TestComputeExact:
    def test_match(self):
        assert compute_exact("Yes", "yes") == 1
        assert compute_exact("Paris", "paris") == 1

    def test_no_match(self):
        assert compute_exact("Paris", "London") == 0

    def test_article_removal(self):
        assert compute_exact("the cat", "cat") == 1


class TestExactPresence:
    def test_present(self):
        assert exact_presence(["Paris", "France"], "I visited Paris last year") is True

    def test_absent(self):
        assert exact_presence(["Tokyo"], "I visited Paris last year") is False

    def test_empty_answers(self):
        assert exact_presence([], "some text") is False


class TestRemoveCitations:
    def test_basic(self):
        result = remove_citations("Hello [1] world [2]")
        assert "Hello" in result
        assert "[1]" not in result
        assert "[2]" not in result

    def test_no_citations(self):
        assert remove_citations("Hello world") == "Hello world"


# ═══════════════════════════════════════════════════════════════════════════════
# ShortFormMetrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestShortFormMetrics:
    def test_score_item_exact(self):
        scores = ShortFormMetrics.score_item("Yes", "yes")
        assert scores["em"] == 1.0
        assert scores["f1"] == 1.0

    def test_score_item_no_match(self):
        scores = ShortFormMetrics.score_item("Paris", "London")
        assert scores["em"] == 0.0
        assert scores["f1"] == 0.0

    def test_aggregate(self):
        items = [
            {"em": 1.0, "f1": 1.0},
            {"em": 0.0, "f1": 0.5},
        ]
        agg = ShortFormMetrics.aggregate(items)
        assert agg["avg_em"] == 50.0
        assert agg["avg_f1"] == 75.0


# ═══════════════════════════════════════════════════════════════════════════════
# ASQAMetrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestASQAMetrics:
    def test_score_item_with_qa_pairs(self):
        scores = ASQAMetrics.score_item(
            gold="Paris is the capital of France",
            pred="The capital of France is Paris, known for the Eiffel Tower",
            qa_pairs=[
                {"short_answers": ["Paris"]},
                {"short_answers": ["France"]},
            ],
        )
        assert scores["str_em"] == 1.0
        assert scores["f1"] > 0.0
        assert scores["length"] > 0

    def test_score_item_no_qa_pairs(self):
        scores = ASQAMetrics.score_item(gold="test", pred="test")
        assert scores["str_em"] == 0.0
        assert scores["f1"] == 1.0

    def test_aggregate(self):
        items = [
            {"f1": 0.8, "exact": 0.0, "str_em": 1.0, "str_hit": 1.0, "length": 10.0},
            {"f1": 0.4, "exact": 0.0, "str_em": 0.5, "str_hit": 0.0, "length": 20.0},
        ]
        agg = ASQAMetrics.aggregate(items)
        assert abs(agg["avg_f1"] - 60.0) < 0.01
        assert abs(agg["avg_str_em"] - 75.0) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# QampariMetrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestQampariMetrics:
    def test_perfect_recall(self):
        answers = [["Paris"], ["London"], ["Berlin"]]
        pred = "Paris, London, Berlin"
        scores = QampariMetrics.score_item(answers, pred)
        assert scores["precision"] == 1.0
        assert scores["recall"] == 1.0
        assert scores["f1"] == 1.0

    def test_partial(self):
        answers = [["Paris"], ["London"], ["Berlin"]]
        pred = "Paris, Tokyo"
        scores = QampariMetrics.score_item(answers, pred)
        assert scores["precision"] == 0.5
        assert abs(scores["recall"] - 1 / 3) < 0.01

    def test_empty_prediction(self):
        answers = [["Paris"]]
        pred = ""
        scores = QampariMetrics.score_item(answers, pred)
        assert scores["precision"] == 0.0
        assert scores["recall"] == 0.0

    def test_alias_match(self):
        answers = [["NYC", "New York City"], ["LA", "Los Angeles"]]
        pred = "New York City, LA"
        scores = QampariMetrics.score_item(answers, pred)
        assert scores["recall"] == 1.0

    def test_recall_top5(self):
        answers = [["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"]]
        pred = "a, b, c, d, e, f, g"
        scores = QampariMetrics.score_item(answers, pred)
        assert scores["recall"] > 0.8
        assert scores["recall_top5"] == 1.0

    def test_cot_mode(self):
        answers = [["Paris"]]
        pred = "Let me think: Paris"
        scores = QampariMetrics.score_item(answers, pred, cot=True)
        assert scores["recall"] == 1.0

    def test_aggregate(self):
        items = [
            {"num_preds": 3.0, "precision": 1.0, "recall": 1.0,
             "recall_top5": 1.0, "f1": 1.0, "f1_top5": 1.0},
            {"num_preds": 2.0, "precision": 0.5, "recall": 0.5,
             "recall_top5": 0.5, "f1": 0.5, "f1_top5": 0.5},
        ]
        agg = QampariMetrics.aggregate(items)
        assert agg["avg_f1"] == 75.0


# ═══════════════════════════════════════════════════════════════════════════════
# LLMJudgeMetrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMJudgeMetrics:
    def test_no_llm(self):
        scores = LLMJudgeMetrics.score_item("Q?", "Answer", llm_fn=None)
        assert scores["comprehensiveness"] == 0.0

    def test_with_mock_llm(self):
        def mock_llm(system, user, **kwargs):
            return '{"comprehensiveness": 4, "diversity": 3, "empowerment": 5}'

        scores = LLMJudgeMetrics.score_item("Q?", "Answer", llm_fn=mock_llm)
        assert scores["comprehensiveness"] == 4.0
        assert scores["diversity"] == 3.0
        assert scores["empowerment"] == 5.0

    def test_parse_error(self):
        def bad_llm(system, user, **kwargs):
            return "not json"

        scores = LLMJudgeMetrics.score_item("Q?", "A", llm_fn=bad_llm)
        assert scores["comprehensiveness"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PairwiseJudgeMetrics
# ═══════════════════════════════════════════════════════════════════════════════

from bsamp.scoring.metrics import PairwiseJudgeMetrics


class TestPairwiseJudgeMetrics:
    def test_no_llm_returns_ties(self):
        result = PairwiseJudgeMetrics.score_item("Q?", "A1", "A2", llm_fn=None)
        assert result["comp_winner"] == "tie"
        assert result["overall_winner"] == "tie"

    def test_with_mock_llm_consistent_winner(self):
        def mock_llm(system, user, **kwargs):
            return json.dumps({
                "Comprehensiveness": {"Winner": "Answer 1", "Explanation": "Better"},
                "Diversity": {"Winner": "Answer 1", "Explanation": "Better"},
                "Empowerment": {"Winner": "Answer 1", "Explanation": "Better"},
                "Overall Winner": {"Winner": "Answer 1", "Explanation": "Overall better"},
            })

        result = PairwiseJudgeMetrics.score_item("Q?", "good", "bad", llm_fn=mock_llm)
        assert result["comp_winner"] in ("A", "B", "tie")
        assert result["overall_winner"] in ("A", "B", "tie")

    def test_aggregate_win_rate(self):
        items = [
            {"comp_winner": "A", "div_winner": "A", "emp_winner": "B", "overall_winner": "A"},
            {"comp_winner": "B", "div_winner": "A", "emp_winner": "A", "overall_winner": "A"},
            {"comp_winner": "A", "div_winner": "tie", "emp_winner": "A", "overall_winner": "B"},
        ]
        agg = PairwiseJudgeMetrics.aggregate(items, label_a="A")
        assert abs(agg["comp_win_rate"] - 200 / 3) < 0.1
        assert abs(agg["overall_win_rate"] - 200 / 3) < 0.1

    def test_parse_error_fallback(self):
        def bad_llm(system, user, **kwargs):
            return "invalid json"

        result = PairwiseJudgeMetrics.score_item("Q?", "A1", "A2", llm_fn=bad_llm)
        assert result["overall_winner"] in ("A", "B", "tie")


# ═══════════════════════════════════════════════════════════════════════════════
# DatasetEvaluator ABC
# ═══════════════════════════════════════════════════════════════════════════════

from bsamp.scoring.evaluator import DatasetEvaluator


class TestDatasetEvaluatorABC:
    def test_all_evaluators_are_subclasses(self):
        assert issubclass(HotpotQAEvaluator, DatasetEvaluator)
        assert issubclass(ASQAEvaluator, DatasetEvaluator)
        assert issubclass(QampariEvaluator, DatasetEvaluator)
        assert issubclass(UltraDomainEvaluator, DatasetEvaluator)

    def test_polymorphic_usage(self):
        evaluator: DatasetEvaluator = HotpotQAEvaluator()
        items = [
            {"prediction": "Yes", "answer": "Yes", "query_id": "1"},
        ]
        result = evaluator.score_batch(items)
        assert result.benchmark == "hotpotqa"
        assert result.num_items == 1


# ═══════════════════════════════════════════════════════════════════════════════
# HotpotQAEvaluator
# ═══════════════════════════════════════════════════════════════════════════════

class TestHotpotQAEvaluator:
    def test_score_batch(self):
        evaluator = HotpotQAEvaluator()
        items = [
            {"prediction": "Yes", "answer": "Yes", "question": "Is this?", "query_id": "q1"},
            {"prediction": "Paris", "answer": "London", "question": "Capital?", "query_id": "q2"},
        ]
        result = evaluator.score_batch(items)
        assert result.benchmark == "hotpotqa"
        assert result.num_items == 2
        assert result.aggregate["avg_em"] == 50.0
        assert len(result.per_item) == 2
        assert result.per_item[0].metrics["em"] == 1.0
        assert result.per_item[1].metrics["em"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ASQAEvaluator
# ═══════════════════════════════════════════════════════════════════════════════

class TestASQAEvaluator:
    def test_score_batch(self):
        evaluator = ASQAEvaluator()
        items = [
            {
                "prediction": "Paris is in France",
                "answer": "Paris is the capital of France",
                "question": "Where is Paris?",
                "qa_pairs": [{"short_answers": ["Paris"]}],
                "query_id": "a1",
            },
        ]
        result = evaluator.score_batch(items)
        assert result.benchmark == "asqa"
        assert result.num_items == 1
        assert result.per_item[0].metrics["str_em"] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# QampariEvaluator
# ═══════════════════════════════════════════════════════════════════════════════

class TestQampariEvaluator:
    def test_score_batch(self):
        evaluator = QampariEvaluator()
        items = [
            {
                "prediction": "Paris, London, Berlin",
                "answers": [["Paris"], ["London"], ["Berlin"]],
                "question": "Name 3 European cities",
                "query_id": "qm1",
            },
        ]
        result = evaluator.score_batch(items)
        assert result.benchmark == "qampari"
        assert result.per_item[0].metrics["f1"] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# UltraDomainEvaluator
# ═══════════════════════════════════════════════════════════════════════════════

class TestUltraDomainEvaluator:
    def test_no_llm(self):
        evaluator = UltraDomainEvaluator(llm_fn=None)
        items = [
            {
                "prediction": "Deep learning is a subset of ML",
                "answer": "Deep learning is a subset of machine learning",
                "question": "What is deep learning?",
                "domain": "cs",
                "query_id": "u1",
            },
        ]
        result = evaluator.score_batch(items)
        assert result.benchmark == "ultradomain"
        assert result.aggregate["avg_comprehensiveness"] == 0.0
        assert result.aggregate["avg_f1"] > 0.0

    def test_with_mock_llm(self):
        def mock_llm(system, user, **kwargs):
            return '{"comprehensiveness": 4, "diversity": 3, "empowerment": 5}'

        evaluator = UltraDomainEvaluator(llm_fn=mock_llm)
        items = [
            {
                "prediction": "answer text",
                "question": "question?",
                "domain": "cs",
                "query_id": "u2",
            },
        ]
        result = evaluator.score_batch(items)
        assert result.aggregate["avg_comprehensiveness"] == 4.0
        assert result.aggregate["avg_diversity"] == 3.0
        assert result.aggregate["avg_empowerment"] == 5.0

    def test_pairwise_mode_no_llm(self):
        evaluator = UltraDomainEvaluator(llm_fn=None, mode="pairwise")
        items = [
            {
                "prediction": "answer A",
                "answer_b": "answer B",
                "question": "question?",
                "query_id": "u3",
            },
        ]
        result = evaluator.score_batch(items)
        assert result.benchmark == "ultradomain_pairwise"
        assert result.num_items == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════════

class TestTypes:
    def test_item_score_defaults(self):
        s = ItemScore(item_id="x")
        assert s.metrics == {}
        assert s.prediction == ""

    def test_evaluation_result_defaults(self):
        r = EvaluationResult()
        assert r.num_items == 0
        assert r.per_item == []
