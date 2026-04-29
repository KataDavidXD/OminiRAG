"""Real LLM demo: run key pipeline combinations against all 3 benchmarks.

Uses the KG sample data and benchmark adapters to demonstrate cross-framework
pipelines with actual LLM calls (via OpenAI-compatible API).

Requires: OPENAI_API_KEY or OPENAI_BASE_URL + OPENAI_API_KEY env vars.

Usage:
    python demos/run_benchmark_demo.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_contracts import (
    RetrievalResult,
    SimpleLLMGeneration,
)
from benchmark.hotpotqa_adapter import (
    HotpotQABenchmarkAdapter,
    load_hotpotqa_sample,
    sample_chunks_to_retrieval_results,
)
from benchmark.ultradomain_adapter import (
    UltraDomainBenchmarkAdapter,
    load_ultradomain_sample,
)
from benchmark.alce_adapter import ALCEBenchmarkAdapter

SEPARATOR = "=" * 70
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "sample_data"

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "")


class _LLM:
    """Minimal OpenAI-compatible LLM wrapper."""

    def __init__(self, model: str = MODEL, base_url: str = BASE_URL):
        from openai import OpenAI

        kwargs = {"api_key": os.environ.get("OPENAI_API_KEY", "sk-placeholder")}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model

    def complete(
        self, system: str, user: str, temperature: float = 0.1, max_tokens: int = 300
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


class SampleDataRetrieval:
    """Retrieval component that returns chunks from KG sample data."""

    def __init__(self, chunks: dict):
        self.chunks = chunks

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        return sample_chunks_to_retrieval_results(self.chunks)[:top_k]


def _run_hotpotqa_demo(llm: _LLM) -> None:
    print(f"\n{SEPARATOR}")
    print("  BENCHMARK: HotpotQA (Multi-hop QA, EM/F1)")
    print(SEPARATOR)

    data = load_hotpotqa_sample(SAMPLE_DIR / "hotpotqa_kg_sample")
    adapter = HotpotQABenchmarkAdapter()

    gen = SimpleLLMGeneration(llm=llm)

    print("\n  Config 1: Identity Query -> Sample Chunks -> Identity Reranking -> LLM Gen")
    result = adapter.evaluate_generation(data, gen)
    print(f"    EM: {result.avg_em:.1f}%  F1: {result.avg_f1:.1f}%")
    for item in result.per_item:
        print(f"    Q: {item['question'][:60]}...")
        print(f"    A: {item['output'][:80]}")
        print(f"    Gold: {item['answer']}  (EM={item['em']}, F1={item['f1']:.2f})")
        print()


def _run_ultradomain_demo(llm: _LLM) -> None:
    print(f"\n{SEPARATOR}")
    print("  BENCHMARK: UltraDomain (Domain-specific QA, LLM-judge)")
    print(SEPARATOR)

    data = load_ultradomain_sample(SAMPLE_DIR / "ultradomain_kg_sample")
    adapter = UltraDomainBenchmarkAdapter(llm_complete=llm.complete)

    gen = SimpleLLMGeneration(llm=llm)

    print("\n  Config 6: LLM Gen with LLM-as-judge evaluation")
    result = adapter.evaluate_generation(data, gen)
    print(f"    Comprehensiveness: {result.avg_comprehensiveness:.1f}/5")
    print(f"    Diversity: {result.avg_diversity:.1f}/5")
    print(f"    Empowerment: {result.avg_empowerment:.1f}/5")
    print(f"    Avg Length: {result.avg_length:.0f} tokens")
    for item in result.per_item:
        print(f"\n    [{item['domain']}] Q: {item['question'][:60]}...")
        print(f"    Output ({item['length']} words): {item['output'][:100]}...")
        print(f"    Scores: C={item['comprehensiveness']:.0f} D={item['diversity']:.0f} E={item['empowerment']:.0f}")


def _run_alce_demo(llm: _LLM) -> None:
    print(f"\n{SEPARATOR}")
    print("  BENCHMARK: ALCE (Per-document segmented QA, F1/STR-EM)")
    print(SEPARATOR)

    queries_path = SAMPLE_DIR / "alce_kg_sample" / "queries.jsonl"
    docs_path = SAMPLE_DIR / "alce_kg_sample" / "alce_docs.json"

    with open(docs_path, encoding="utf-8") as f:
        all_docs = json.load(f)

    data = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                data.append({
                    "question": q["query"],
                    "answer": q.get("ground_truth", ""),
                    "docs": all_docs.get(q["query_id"], []),
                    "query_id": q["query_id"],
                })

    adapter = ALCEBenchmarkAdapter()
    gen = SimpleLLMGeneration(llm=llm)

    print("\n  Config 10: ALCE Docs -> Identity Reranking -> LLM Gen")
    result = adapter.evaluate_generation(data, gen)
    print(f"    F1: {result.avg_f1:.1f}%  EM: {result.avg_exact:.1f}%")
    print(f"    STR-EM: {result.avg_str_em:.1f}%  Avg Length: {result.avg_length:.0f} words")
    for item in result.per_item:
        print(f"\n    Q: {item['question'][:60]}...")
        print(f"    Output: {item['output'][:100]}...")
        print(f"    Gold: {str(item['answer'])[:80]}")
        print(f"    F1={item['f1']:.2f}  EM={item['exact']}")


def main() -> None:
    print(SEPARATOR)
    print("  OMINIRAG BENCHMARK DEMO")
    print(f"  Model: {MODEL}")
    print(f"  Base URL: {BASE_URL or '(default OpenAI)'}")
    print(SEPARATOR)

    try:
        llm = _LLM()
    except Exception as e:
        print(f"\n  ERROR: Could not initialize LLM: {e}")
        print("  Set OPENAI_API_KEY and optionally OPENAI_BASE_URL environment variables.")
        sys.exit(1)

    _run_hotpotqa_demo(llm)
    _run_ultradomain_demo(llm)
    _run_alce_demo(llm)

    print(f"\n{SEPARATOR}")
    print("  DEMO COMPLETE")
    print(f"  All 3 benchmarks evaluated with live LLM ({MODEL}).")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
