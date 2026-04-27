"""ALCE benchmark evaluation adapter for rag_contracts pipelines.

Supports ASQA (long-form QA with STR-EM) and QAMPARI (list-answer F1).
Scoring logic lives in ``bsamp.scoring``; this module is the thin glue
between OminiRAG's ``Generation`` protocol and the scoring SDK.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_contracts import GenerationResult, RetrievalResult

from bsamp.scoring import (
    ASQAEvaluator,
    QampariEvaluator,
    compute_exact,
    compute_f1,
    exact_presence,
    normalize_answer,
    remove_citations,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Re-exports for backward compatibility
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "normalize_answer",
    "compute_f1",
    "compute_exact",
    "exact_presence",
    "remove_citations",
    "load_alce_data",
    "alce_item_to_retrieval_results",
    "ALCEEvaluationResult",
    "ALCEBenchmarkAdapter",
    "QampariEvaluationResult",
    "QampariBenchmarkAdapter",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_alce_data(path: str | Path) -> list[dict]:
    """Load an ALCE JSON file.

    Expected format: ``{"data": [...], "config": {...}}``.
    Each item has ``question``, ``answer``, ``docs`` (list of {title, text}),
    and optionally ``qa_pairs``, ``annotations``, ``claims``.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = raw.get("data", raw) if isinstance(raw, dict) else raw
    if isinstance(data, dict):
        data = list(data.values())
    return data


def alce_item_to_retrieval_results(item: dict) -> list[RetrievalResult]:
    """Convert ALCE docs into rag_contracts RetrievalResults."""
    results = []
    for idx, doc in enumerate(item.get("docs", [])):
        results.append(RetrievalResult(
            source_id=str(idx + 1),
            content=doc.get("text", ""),
            score=1.0 - idx * 0.01,
            title=doc.get("title", ""),
            metadata={
                "doc_index": idx,
                "summary": doc.get("summary", ""),
                "extraction": doc.get("extraction", ""),
            },
        ))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ASQA evaluation (STR-EM + token-F1)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ALCEEvaluationResult:
    """Aggregated evaluation results across all items."""
    avg_f1: float = 0.0
    avg_exact: float = 0.0
    avg_str_em: float = 0.0
    avg_str_hit: float = 0.0
    avg_length: float = 0.0
    num_items: int = 0
    per_item: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ALCEBenchmarkAdapter:
    """Evaluate any rag_contracts Generation against ALCE/ASQA data.

    Usage::

        adapter = ALCEBenchmarkAdapter()
        data = load_alce_data("path/to/asqa.json")
        results = adapter.evaluate_generation(data, my_generation)
    """

    def evaluate_generation(
        self,
        data: list[dict],
        generation: Any,
        max_docs: int = 5,
    ) -> ALCEEvaluationResult:
        """Run generation on each ALCE item and compute lightweight metrics."""
        evaluator = ASQAEvaluator()
        scored_items: list[dict] = []

        for item in data:
            question = item.get("question", "")
            context = alce_item_to_retrieval_results(item)[:max_docs]

            gen_result: GenerationResult = generation.generate(
                query=question,
                context=context,
                instruction="alce",
            )
            output = gen_result.output.strip()

            answer = item.get("answer", "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""

            scored_items.append({
                "prediction": output,
                "answer": answer,
                "question": question,
                "qa_pairs": item.get("qa_pairs"),
                "query_id": item.get("query_id", ""),
                "_citations": gen_result.citations,
            })

        result = evaluator.score_batch(scored_items)

        per_item = []
        for si, score_obj in zip(scored_items, result.per_item):
            per_item.append({
                "question": si["question"],
                "answer": si["answer"],
                "output": score_obj.prediction,
                "output_clean": remove_citations(score_obj.prediction),
                "f1": score_obj.metrics["f1"],
                "exact": score_obj.metrics["exact"],
                "str_em": score_obj.metrics["str_em"],
                "str_hit": score_obj.metrics["str_hit"],
                "length": score_obj.metrics["length"],
                "citations": si["_citations"],
            })

        return ALCEEvaluationResult(
            avg_f1=result.aggregate["avg_f1"],
            avg_exact=result.aggregate["avg_exact"],
            avg_str_em=result.aggregate["avg_str_em"],
            avg_str_hit=result.aggregate["avg_str_hit"],
            avg_length=result.aggregate["avg_length"],
            num_items=result.num_items,
            per_item=per_item,
        )

    def evaluate_pipeline(
        self,
        data: list[dict],
        graph_or_factory: Any,
        max_docs: int = 5,
    ) -> ALCEEvaluationResult:
        """Run an entire LangGraph pipeline per item and compute metrics."""
        import asyncio

        from rag_contracts import ALCEDocRetrieval

        is_factory = callable(graph_or_factory) and not hasattr(
            graph_or_factory, "ainvoke"
        )

        def _invoke_graph(g, payload):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(g.ainvoke(payload))
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                return pool.submit(asyncio.run, g.ainvoke(payload)).result()

        evaluator = ASQAEvaluator()
        scored_items: list[dict] = []

        for item in data:
            question = item.get("question", "")

            if is_factory:
                docs = item.get("docs", [])[:max_docs]
                retrieval = ALCEDocRetrieval(docs=docs)
                graph = graph_or_factory(retrieval=retrieval)
            else:
                graph = graph_or_factory

            state = _invoke_graph(graph, {"query": question})

            gen_result: GenerationResult = state.get("generation_result", GenerationResult(output=""))
            output = gen_result.output.strip()

            answer = item.get("answer", "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""

            scored_items.append({
                "prediction": output,
                "answer": answer,
                "question": question,
                "qa_pairs": item.get("qa_pairs"),
                "query_id": item.get("query_id", ""),
                "_citations": gen_result.citations,
            })

        result = evaluator.score_batch(scored_items)

        per_item = []
        for si, score_obj in zip(scored_items, result.per_item):
            per_item.append({
                "question": si["question"],
                "answer": si["answer"],
                "output": score_obj.prediction,
                "output_clean": remove_citations(score_obj.prediction),
                "f1": score_obj.metrics["f1"],
                "exact": score_obj.metrics["exact"],
                "str_em": score_obj.metrics["str_em"],
                "str_hit": score_obj.metrics["str_hit"],
                "length": score_obj.metrics["length"],
                "citations": si["_citations"],
            })

        return ALCEEvaluationResult(
            avg_f1=result.aggregate["avg_f1"],
            avg_exact=result.aggregate["avg_exact"],
            avg_str_em=result.aggregate["avg_str_em"],
            avg_str_hit=result.aggregate["avg_str_hit"],
            avg_length=result.aggregate["avg_length"],
            num_items=result.num_items,
            per_item=per_item,
        )

    def save_results(
        self,
        results: ALCEEvaluationResult,
        output_path: str | Path,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "avg_f1": results.avg_f1,
                    "avg_exact": results.avg_exact,
                    "avg_str_em": results.avg_str_em,
                    "avg_str_hit": results.avg_str_hit,
                    "avg_length": results.avg_length,
                    "num_items": results.num_items,
                },
                "per_item": results.per_item,
            }, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# QAMPARI evaluation (precision / recall / F1)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QampariEvaluationResult:
    """Aggregated QAMPARI evaluation results."""
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_recall_top5: float = 0.0
    avg_f1: float = 0.0
    avg_f1_top5: float = 0.0
    avg_num_preds: float = 0.0
    num_items: int = 0
    per_item: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class QampariBenchmarkAdapter:
    """Evaluate any rag_contracts Generation against ALCE/QAMPARI data.

    Usage::

        adapter = QampariBenchmarkAdapter()
        results = adapter.evaluate_generation(data, my_generation)
    """

    def evaluate_generation(
        self,
        data: list[dict],
        generation: Any,
        max_docs: int = 5,
    ) -> QampariEvaluationResult:
        evaluator = QampariEvaluator()
        scored_items: list[dict] = []

        for item in data:
            question = item.get("question", "")
            context = alce_item_to_retrieval_results(item)[:max_docs]

            gen_result: GenerationResult = generation.generate(
                query=question,
                context=context,
                instruction="qampari",
            )
            output = gen_result.output.strip()

            scored_items.append({
                "prediction": output,
                "answers": item.get("answers", []),
                "question": question,
                "query_id": item.get("query_id", ""),
                "_citations": gen_result.citations,
            })

        result = evaluator.score_batch(scored_items)

        per_item = []
        for si, score_obj in zip(scored_items, result.per_item):
            per_item.append({
                "question": si["question"],
                "output": si["prediction"],
                "answers": si["answers"],
                "precision": score_obj.metrics["precision"],
                "recall": score_obj.metrics["recall"],
                "recall_top5": score_obj.metrics["recall_top5"],
                "f1": score_obj.metrics["f1"],
                "f1_top5": score_obj.metrics["f1_top5"],
                "num_preds": score_obj.metrics["num_preds"],
                "query_id": si["query_id"],
                "citations": si["_citations"],
            })

        return QampariEvaluationResult(
            avg_precision=result.aggregate["avg_precision"],
            avg_recall=result.aggregate["avg_recall"],
            avg_recall_top5=result.aggregate["avg_recall_top5"],
            avg_f1=result.aggregate["avg_f1"],
            avg_f1_top5=result.aggregate["avg_f1_top5"],
            avg_num_preds=result.aggregate["avg_num_preds"],
            num_items=result.num_items,
            per_item=per_item,
        )

    def evaluate_pipeline(
        self,
        data: list[dict],
        graph_or_factory: Any,
        max_docs: int = 5,
    ) -> QampariEvaluationResult:
        """Run an entire LangGraph pipeline per item and compute QAMPARI metrics."""
        import asyncio

        from rag_contracts import ALCEDocRetrieval

        is_factory = callable(graph_or_factory) and not hasattr(
            graph_or_factory, "ainvoke"
        )

        def _invoke_graph(g, payload):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(g.ainvoke(payload))
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                return pool.submit(asyncio.run, g.ainvoke(payload)).result()

        evaluator = QampariEvaluator()
        scored_items: list[dict] = []

        for item in data:
            question = item.get("question", "")

            if is_factory:
                docs = item.get("docs", [])[:max_docs]
                retrieval = ALCEDocRetrieval(docs=docs)
                graph = graph_or_factory(retrieval=retrieval)
            else:
                graph = graph_or_factory

            state = _invoke_graph(graph, {"query": question})

            gen_result: GenerationResult = state.get("generation_result", GenerationResult(output=""))
            output = gen_result.output.strip()

            scored_items.append({
                "prediction": output,
                "answers": item.get("answers", []),
                "question": question,
                "query_id": item.get("query_id", ""),
                "_citations": gen_result.citations,
            })

        result = evaluator.score_batch(scored_items)

        per_item = []
        for si, score_obj in zip(scored_items, result.per_item):
            per_item.append({
                "question": si["question"],
                "output": si["prediction"],
                "answers": si["answers"],
                "precision": score_obj.metrics["precision"],
                "recall": score_obj.metrics["recall"],
                "recall_top5": score_obj.metrics["recall_top5"],
                "f1": score_obj.metrics["f1"],
                "f1_top5": score_obj.metrics["f1_top5"],
                "num_preds": score_obj.metrics["num_preds"],
                "query_id": si["query_id"],
                "citations": si["_citations"],
            })

        return QampariEvaluationResult(
            avg_precision=result.aggregate["avg_precision"],
            avg_recall=result.aggregate["avg_recall"],
            avg_recall_top5=result.aggregate["avg_recall_top5"],
            avg_f1=result.aggregate["avg_f1"],
            avg_f1_top5=result.aggregate["avg_f1_top5"],
            avg_num_preds=result.aggregate["avg_num_preds"],
            num_items=result.num_items,
            per_item=per_item,
        )

    def save_results(
        self,
        results: QampariEvaluationResult,
        output_path: str | Path,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "avg_precision": results.avg_precision,
                    "avg_recall": results.avg_recall,
                    "avg_recall_top5": results.avg_recall_top5,
                    "avg_f1": results.avg_f1,
                    "avg_f1_top5": results.avg_f1_top5,
                    "avg_num_preds": results.avg_num_preds,
                    "num_items": results.num_items,
                },
                "per_item": results.per_item,
            }, f, ensure_ascii=False, indent=2)
