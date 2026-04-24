from __future__ import annotations

from rag_contracts import Generation


def build_node(generation: Generation):
    async def node(state):
        query = state["query"]
        context = state.get("retrieval_results", [])
        mode = state.get("mode", "hybrid")
        instruction = f"mode={mode}"
        result = generation.generate(query=query, context=context, instruction=instruction)
        return {"generation_result": result}

    return node
