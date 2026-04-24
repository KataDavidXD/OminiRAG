"""UltraDomain benchmark evaluation adapter for rag_contracts pipelines.

Loads UltraDomain data (from KG sample or local JSONL), runs it through any
pipeline conforming to canonical protocols, and evaluates outputs using
LLM-as-judge metrics (Comprehensiveness, Diversity, Empowerment) as well as
basic token-F1 for cross-metric comparison.
"""

from __future__ import annotations

import collections
import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from rag_contracts import GenerationResult, RetrievalResult


# ═══════════════════════════════════════════════════════════════════════════════
# Text normalization
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


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-as-judge scoring
# ═══════════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = """\
You are evaluating an AI assistant's answer to a domain-specific question.
Score the answer on three dimensions (each 1-5):

1. **Comprehensiveness**: Does the answer cover all important aspects of the question?
2. **Diversity**: Does the answer draw from varied perspectives, methods, or viewpoints?
3. **Empowerment**: Does the answer help the user understand deeply and act on the information?

Output ONLY a JSON object: {"comprehensiveness": N, "diversity": N, "empowerment": N}
"""

JUDGE_USER_TEMPLATE = """\
Domain: {domain}
Question: {question}
Answer: {answer}
"""


def default_llm_judge(
    llm_complete: Callable[..., str],
    question: str,
    answer: str,
    domain: str = "general",
) -> dict[str, float]:
    """Call an LLM judge to score the answer on 3 dimensions.

    ``llm_complete`` must accept (system_prompt, user_prompt, **kwargs) -> str.
    Returns {comprehensiveness, diversity, empowerment} each in [1, 5].
    Falls back to zeros on parse errors.
    """
    user_text = JUDGE_USER_TEMPLATE.format(
        domain=domain, question=question, answer=answer,
    )
    try:
        raw = llm_complete(JUDGE_SYSTEM_PROMPT, user_text, temperature=0.0, max_tokens=100)
        scores = json.loads(raw.strip())
        return {
            "comprehensiveness": float(scores.get("comprehensiveness", 0)),
            "diversity": float(scores.get("diversity", 0)),
            "empowerment": float(scores.get("empowerment", 0)),
        }
    except Exception:
        return {"comprehensiveness": 0.0, "diversity": 0.0, "empowerment": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_ultradomain_sample(sample_dir: str | Path) -> list[dict]:
    """Load from the KG sample data directory.

    Reads ``queries.jsonl`` and ``chunks.json`` to produce items with
    ``question``, ``domain``, and ``context_chunks``.
    """
    sample_dir = Path(sample_dir)
    queries = []
    with open(sample_dir / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    chunks = {}
    chunks_path = sample_dir / "chunks.json"
    if chunks_path.exists():
        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)

    items = []
    for q in queries:
        items.append({
            "question": q["query"],
            "domain": q.get("domain", "general"),
            "query_id": q.get("query_id", ""),
            "chunks": chunks,
        })
    return items


def load_ultradomain_jsonl(path: str | Path) -> list[dict]:
    """Load UltraDomain from a JSONL file."""
    items = []
    with open(Path(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def sample_chunks_to_retrieval_results(chunks: dict) -> list[RetrievalResult]:
    """Convert KG sample chunks dict to RetrievalResult list."""
    results = []
    for cid, info in chunks.items():
        doc_ids = info.get("doc_ids", [])
        results.append(RetrievalResult(
            source_id=cid,
            content=info.get("content", ""),
            score=1.0,
            title=doc_ids[0] if doc_ids else cid,
            metadata={"doc_ids": doc_ids},
        ))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation runner
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UltraDomainEvaluationResult:
    """Aggregated evaluation results across all items."""
    avg_comprehensiveness: float = 0.0
    avg_diversity: float = 0.0
    avg_empowerment: float = 0.0
    avg_f1: float = 0.0
    avg_length: float = 0.0
    num_items: int = 0
    per_item: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class UltraDomainBenchmarkAdapter:
    """Evaluate any rag_contracts Generation against UltraDomain data.

    If ``llm_complete`` is provided, uses LLM-as-judge for the three domain
    metrics. Otherwise only computes token-F1 (useful for quick testing).

    Usage::

        adapter = UltraDomainBenchmarkAdapter(llm_complete=my_llm.complete)
        data = load_ultradomain_sample("benchmark/sample_data/ultradomain_kg_sample")
        results = adapter.evaluate_generation(data, my_generation)
    """

    llm_complete: Optional[Callable[..., str]] = None

    def evaluate_generation(
        self,
        data: list[dict],
        generation: Any,
    ) -> UltraDomainEvaluationResult:
        """Run generation on each UltraDomain item and compute metrics."""
        per_item = []
        comp_scores = []
        div_scores = []
        emp_scores = []
        f1_scores = []
        lengths = []

        for item in data:
            question = item["question"]
            domain = item.get("domain", "general")
            chunks = item.get("chunks", {})
            context = sample_chunks_to_retrieval_results(chunks)

            gen_result: GenerationResult = generation.generate(
                query=question,
                context=context,
            )

            output = gen_result.output.strip()
            length = len(output.split())

            judge_scores = {"comprehensiveness": 0.0, "diversity": 0.0, "empowerment": 0.0}
            if self.llm_complete is not None:
                judge_scores = default_llm_judge(
                    self.llm_complete, question, output, domain,
                )

            ref_answer = item.get("answer", "")
            if isinstance(ref_answer, list):
                ref_answer = ref_answer[0] if ref_answer else ""
            f1 = compute_f1(ref_answer, output) if ref_answer else 0.0

            comp_scores.append(judge_scores["comprehensiveness"])
            div_scores.append(judge_scores["diversity"])
            emp_scores.append(judge_scores["empowerment"])
            f1_scores.append(f1)
            lengths.append(length)

            per_item.append({
                "question": question,
                "domain": domain,
                "output": output,
                "comprehensiveness": judge_scores["comprehensiveness"],
                "diversity": judge_scores["diversity"],
                "empowerment": judge_scores["empowerment"],
                "f1": f1,
                "length": length,
                "query_id": item.get("query_id", ""),
                "citations": gen_result.citations,
            })

        n = len(data) or 1
        return UltraDomainEvaluationResult(
            avg_comprehensiveness=sum(comp_scores) / n,
            avg_diversity=sum(div_scores) / n,
            avg_empowerment=sum(emp_scores) / n,
            avg_f1=100 * sum(f1_scores) / n,
            avg_length=sum(lengths) / n,
            num_items=len(data),
            per_item=per_item,
        )

    def evaluate_pipeline(
        self,
        data: list[dict],
        graph: Any,
    ) -> UltraDomainEvaluationResult:
        """Run an entire LangGraph pipeline per item and compute metrics."""
        import asyncio

        per_item = []
        comp_scores = []
        div_scores = []
        emp_scores = []
        f1_scores = []
        lengths = []

        for item in data:
            question = item["question"]
            domain = item.get("domain", "general")
            state = asyncio.run(graph.ainvoke({"query": question}))

            gen_result: GenerationResult = state.get(
                "generation_result", GenerationResult(output="")
            )
            output = gen_result.output.strip()
            length = len(output.split())

            judge_scores = {"comprehensiveness": 0.0, "diversity": 0.0, "empowerment": 0.0}
            if self.llm_complete is not None:
                judge_scores = default_llm_judge(
                    self.llm_complete, question, output, domain,
                )

            ref_answer = item.get("answer", "")
            if isinstance(ref_answer, list):
                ref_answer = ref_answer[0] if ref_answer else ""
            f1 = compute_f1(ref_answer, output) if ref_answer else 0.0

            comp_scores.append(judge_scores["comprehensiveness"])
            div_scores.append(judge_scores["diversity"])
            emp_scores.append(judge_scores["empowerment"])
            f1_scores.append(f1)
            lengths.append(length)

            per_item.append({
                "question": question,
                "domain": domain,
                "output": output,
                "comprehensiveness": judge_scores["comprehensiveness"],
                "diversity": judge_scores["diversity"],
                "empowerment": judge_scores["empowerment"],
                "f1": f1,
                "length": length,
                "query_id": item.get("query_id", ""),
                "citations": gen_result.citations,
            })

        n = len(data) or 1
        return UltraDomainEvaluationResult(
            avg_comprehensiveness=sum(comp_scores) / n,
            avg_diversity=sum(div_scores) / n,
            avg_empowerment=sum(emp_scores) / n,
            avg_f1=100 * sum(f1_scores) / n,
            avg_length=sum(lengths) / n,
            num_items=len(data),
            per_item=per_item,
        )

    def save_results(
        self,
        results: UltraDomainEvaluationResult,
        output_path: str | Path,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "avg_comprehensiveness": results.avg_comprehensiveness,
                    "avg_diversity": results.avg_diversity,
                    "avg_empowerment": results.avg_empowerment,
                    "avg_f1": results.avg_f1,
                    "avg_length": results.avg_length,
                    "num_items": results.num_items,
                },
                "per_item": results.per_item,
            }, f, ensure_ascii=False, indent=2)
