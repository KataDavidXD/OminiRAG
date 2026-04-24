"""ALCE benchmark evaluation adapter for rag_contracts pipelines.

Converts ALCE dataset items into rag_contracts types, runs them through any
pipeline conforming to the canonical protocols, and evaluates outputs using
ALCE's lightweight metrics (F1, exact match, ROUGE, STR-EM).

Heavy metrics like AutoAIS and QA-pipeline are available but require GPU models.
"""

from __future__ import annotations

import collections
import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_contracts import GenerationResult, RetrievalResult


# ═══════════════════════════════════════════════════════════════════════════════
# Text normalization (from ALCE utils.py, inlined to avoid path manipulation)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def remove_citations(sent: str) -> str:
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


# ═══════════════════════════════════════════════════════════════════════════════
# Lightweight metrics (no model loading required)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_f1(gold: str, pred: str) -> float:
    gold_toks = normalize_answer(gold).split()
    pred_toks = normalize_answer(pred).split()

    if not gold_toks or not pred_toks:
        return float(gold_toks == pred_toks)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def exact_presence(short_answers: list[str], context: str) -> bool:
    n_context = normalize_answer(context)
    return any(normalize_answer(sa) in n_context for sa in short_answers)


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
# Evaluation runner
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
    """Evaluate any rag_contracts Generation against ALCE data.

    Usage::

        adapter = ALCEBenchmarkAdapter()
        data = load_alce_data("path/to/asqa.json")
        results = adapter.evaluate(data, my_generation)

    Or run a full pipeline::

        from lightrag_langgraph.main_pipeline import build_query_graph
        graph = build_query_graph(retrieval=my_ret, generation=my_gen)
        results = adapter.evaluate_pipeline(data, graph)
    """

    def evaluate_generation(
        self,
        data: list[dict],
        generation: Any,
        max_docs: int = 5,
    ) -> ALCEEvaluationResult:
        """Run generation on each ALCE item and compute lightweight metrics."""
        per_item = []
        f1_scores = []
        exact_scores = []
        str_em_scores = []
        str_hit_scores = []
        lengths = []

        for item in data:
            question = item.get("question", "")
            context = alce_item_to_retrieval_results(item)[:max_docs]

            gen_result: GenerationResult = generation.generate(
                query=question,
                context=context,
                instruction="alce",
            )

            output = gen_result.output.strip().split("\n")[0]
            output_clean = remove_citations(output)

            answer = item.get("answer", "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""

            f1 = compute_f1(answer, output_clean)
            em = compute_exact(answer, output_clean)

            qa_pairs = item.get("qa_pairs")
            item_str_em = 0.0
            item_str_hit = 0
            if qa_pairs:
                loc_acc = [
                    exact_presence(qp.get("short_answers", []), output_clean)
                    for qp in qa_pairs
                ]
                item_str_em = sum(loc_acc) / len(loc_acc) if loc_acc else 0.0
                item_str_hit = int(item_str_em == 1.0)

            length = len(output_clean.split())

            f1_scores.append(f1)
            exact_scores.append(em)
            str_em_scores.append(item_str_em)
            str_hit_scores.append(item_str_hit)
            lengths.append(length)

            per_item.append({
                "question": question,
                "answer": answer,
                "output": output,
                "output_clean": output_clean,
                "f1": f1,
                "exact": em,
                "str_em": item_str_em,
                "str_hit": item_str_hit,
                "length": length,
                "citations": gen_result.citations,
            })

        n = len(data) or 1
        return ALCEEvaluationResult(
            avg_f1=100 * sum(f1_scores) / n,
            avg_exact=100 * sum(exact_scores) / n,
            avg_str_em=100 * sum(str_em_scores) / n,
            avg_str_hit=100 * sum(str_hit_scores) / n,
            avg_length=sum(lengths) / n,
            num_items=len(data),
            per_item=per_item,
        )

    def evaluate_pipeline(
        self,
        data: list[dict],
        graph_or_factory: Any,
        max_docs: int = 5,
    ) -> ALCEEvaluationResult:
        """Run an entire LangGraph pipeline per item and compute metrics.

        ``graph_or_factory`` can be either:

        - A pre-compiled LangGraph (legacy; retrieval ignores ALCE docs).
        - A callable ``graph_factory(retrieval=...)`` that returns a compiled
          graph.  When provided, an ``ALCEDocRetrieval`` is created per item
          so the pipeline sees the correct ALCE documents.

        The pipeline must produce a ``generation_result`` field in its
        output state.
        """
        import asyncio

        from rag_contracts import ALCEDocRetrieval

        is_factory = callable(graph_or_factory) and not hasattr(
            graph_or_factory, "ainvoke"
        )

        def _invoke_graph(g, payload):
            """Run graph.ainvoke safely from sync or async contexts."""
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(g.ainvoke(payload))
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                return pool.submit(asyncio.run, g.ainvoke(payload)).result()

        per_item = []
        f1_scores = []
        exact_scores = []
        str_em_scores = []
        str_hit_scores = []
        lengths = []

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
            output = gen_result.output.strip().split("\n")[0]
            output_clean = remove_citations(output)

            answer = item.get("answer", "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""

            f1 = compute_f1(answer, output_clean)
            em = compute_exact(answer, output_clean)

            qa_pairs = item.get("qa_pairs")
            item_str_em = 0.0
            item_str_hit = 0
            if qa_pairs:
                loc_acc = [
                    exact_presence(qp.get("short_answers", []), output_clean)
                    for qp in qa_pairs
                ]
                item_str_em = sum(loc_acc) / len(loc_acc) if loc_acc else 0.0
                item_str_hit = int(item_str_em == 1.0)

            length = len(output_clean.split())

            f1_scores.append(f1)
            exact_scores.append(em)
            str_em_scores.append(item_str_em)
            str_hit_scores.append(item_str_hit)
            lengths.append(length)

            per_item.append({
                "question": question,
                "answer": answer,
                "output": output,
                "output_clean": output_clean,
                "f1": f1,
                "exact": em,
                "str_em": item_str_em,
                "str_hit": item_str_hit,
                "length": length,
                "citations": gen_result.citations,
            })

        n = len(data) or 1
        return ALCEEvaluationResult(
            avg_f1=100 * sum(f1_scores) / n,
            avg_exact=100 * sum(exact_scores) / n,
            avg_str_em=100 * sum(str_em_scores) / n,
            avg_str_hit=100 * sum(str_hit_scores) / n,
            avg_length=sum(lengths) / n,
            num_items=len(data),
            per_item=per_item,
        )

    def save_results(
        self,
        results: ALCEEvaluationResult,
        output_path: str | Path,
    ) -> None:
        """Persist evaluation results to JSON."""
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
