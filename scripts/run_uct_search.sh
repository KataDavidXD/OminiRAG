#!/usr/bin/env bash
# Run AG-UCT search over the 5-dimension RAG configuration space.
#
# Usage:
#   # Quick smoke test (5 iterations, simulated rewards)
#   bash scripts/run_uct_search.sh --smoke
#
#   # Real evaluation with fullwiki data (5 iterations, budget=5)
#   bash scripts/run_uct_search.sh --real-quick
#
#   # Full search (500 iterations, budget=30, real evaluation)
#   bash scripts/run_uct_search.sh --full
#
#   # Custom
#   bash scripts/run_uct_search.sh --real --max-iterations 100 --budget 20
#
# Prerequisites:
#   1. Run: python scripts/build_corpus_index.py
#   2. Run: python scripts/prepare_lightrag_stores.py --fullwiki-only
#   3. Ensure LLM endpoint is reachable (check .env: LLM_BASE_URL)
#   4. (Optional) Launch Self-RAG: bash scripts/launch_selfrag_vllm.sh 4

set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".venv/bin/python"
if [ ! -f "$VENV" ]; then
    VENV="python3"
fi

export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/AG-UCT:$(pwd)/Benchmark_Sampling:$(pwd)/A-Simplified-Core-Workflow-for-Enhancing-RAG"

MODE="${1:-}"

case "$MODE" in
    --smoke)
        echo "=== UCT Smoke Test (5 iters, simulated) ==="
        $VENV AG-UCT/uct_engine/examples/rag_pipeline_search.py \
            --max-iterations 5 --seed 42
        ;;
    --real-quick)
        echo "=== UCT Real Quick (5 iters, budget=5, fullwiki data) ==="
        $VENV AG-UCT/uct_engine/examples/rag_pipeline_search.py \
            --real \
            --data-dir /data1/ragworkspace/train \
            --budget 5 \
            --max-iterations 5 \
            --seed 42
        ;;
    --full)
        echo "=== UCT Full Search (500 iters, budget=30) ==="
        $VENV AG-UCT/uct_engine/examples/rag_pipeline_search.py \
            --real \
            --data-dir /data1/ragworkspace/train \
            --budget 30 \
            --max-iterations 500 \
            --seed 42
        ;;
    --wtb)
        echo "=== UCT with WTB Cache (500 iters, budget=30) ==="
        $VENV AG-UCT/uct_engine/examples/rag_pipeline_search.py \
            --real \
            --data-dir /data1/ragworkspace/train \
            --budget 30 \
            --max-iterations 500 \
            --wtb-reuse \
            --seed 42
        ;;
    *)
        echo "=== UCT Custom ==="
        shift 2>/dev/null || true
        $VENV AG-UCT/uct_engine/examples/rag_pipeline_search.py "$@"
        ;;
esac
