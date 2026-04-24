"""Cross-project WTB demo: register STORM and LongRAG as WorkflowProjects
and demonstrate swapping canonical components between them.

Usage:
    python wtb_demo.py
"""

from __future__ import annotations

import sys

from rag_contracts import (
    GenerationResult,
    IdentityReranking,
    QueryContext,
    RetrievalResult,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Minimal mock implementations for demonstration
# ═══════════════════════════════════════════════════════════════════════════════


class MockRetrieval:
    """Simple retrieval that returns canned results (for demo only)."""

    def __init__(self, name: str = "mock"):
        self.name = name

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        results = []
        for i, q in enumerate(queries[:top_k]):
            results.append(
                RetrievalResult(
                    source_id=f"{self.name}://{i}",
                    content=f"Mock context for query: {q}",
                    score=1.0 - i * 0.1,
                    title=f"Mock Doc {i}",
                )
            )
        return results


class MockGeneration:
    """Simple generation that echoes context (for demo only)."""

    def __init__(self, name: str = "mock"):
        self.name = name

    def generate(
        self,
        query: str,
        context: list[RetrievalResult],
        instruction: str = "",
    ) -> GenerationResult:
        ctx_summary = "; ".join(r.title for r in context[:3])
        return GenerationResult(
            output=f"[{self.name}] Answer to '{query}' using [{ctx_summary}]",
            citations=[r.source_id for r in context],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    from wtb.sdk import WTBTestBench, WorkflowProject

    # --- 1. Create canonical components ---
    storm_retrieval = MockRetrieval(name="storm-web-search")
    longrag_retrieval = MockRetrieval(name="longrag-tevatron")
    storm_generation = MockGeneration(name="storm-section-writer")
    longrag_generation = MockGeneration(name="longrag-llm-reader")
    identity_reranking = IdentityReranking()

    # --- 2. Build STORM project with its default components ---
    from storm.storm_langgraph.wtb_integration import create_storm_graph_factory

    storm_mock_components = dict(
        question_asker=None,
        query_generator=None,
        retriever=None,
        answer_synthesizer=None,
        outline_generator=None,
        section_writer=None,
        polisher=None,
    )

    # For this demo, we use the LongRAG pipeline since it's simpler and
    # fully wired through canonical protocols.
    from longRAG_example.longrag_langgraph.wtb_integration import (
        create_longrag_graph_factory,
        create_longrag_project,
    )

    # --- 3. Register LongRAG project with its default components ---
    longrag_project = create_longrag_project(
        name="longrag_default",
        retrieval=longrag_retrieval,
        generation=longrag_generation,
        reranking=identity_reranking,
    )
    print(f"Registered project: {longrag_project.name} (id={longrag_project.id[:8]}...)")

    # --- 4. Register cross-project variant: STORM's retriever in LongRAG ---
    longrag_with_storm_retrieval = create_longrag_graph_factory(
        retrieval=storm_retrieval,
        generation=longrag_generation,
        reranking=identity_reranking,
    )

    longrag_project.register_variant(
        node="retrieval",
        name="storm_web_search",
        implementation=longrag_with_storm_retrieval,
        description="Use STORM's web-search retriever inside the LongRAG pipeline",
    )
    print("  Registered variant: retrieval/storm_web_search")

    # --- 5. Register cross-project variant: STORM's generation in LongRAG ---
    longrag_with_storm_gen = create_longrag_graph_factory(
        retrieval=longrag_retrieval,
        generation=storm_generation,
        reranking=identity_reranking,
    )

    longrag_project.register_variant(
        node="generation",
        name="storm_section_writer",
        implementation=longrag_with_storm_gen,
        description="Use STORM's section writer as the generator in LongRAG pipeline",
    )
    print("  Registered variant: generation/storm_section_writer")

    # --- 6. List all registered variants ---
    variants = longrag_project.list_variants()
    print(f"\nAll variants for {longrag_project.name}:")
    for node_id, variant_names in variants.items():
        for vname in variant_names:
            print(f"  {node_id}/{vname}")

    # --- 7. Run the default pipeline through WTB ---
    bench = WTBTestBench.create(mode="memory")
    bench.register_project(longrag_project)

    initial_state = {
        "query": "What is the capital of France?",
        "query_id": "demo_1",
        "answers": ["Paris"],
        "test_data_name": "nq",
    }

    print("\nRunning default LongRAG pipeline...")
    run_result = bench.run(project=longrag_project.name, initial_state=initial_state)
    print(f"  Run completed: {run_result}")

    print("\nDone. Cross-project component swap demonstrated successfully.")


if __name__ == "__main__":
    main()
