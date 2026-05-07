#!/usr/bin/env bash
# Launch the Self-RAG (selfrag_llama2_7b) vLLM server on a free GPU.
#
# Usage:
#   bash scripts/launch_selfrag_vllm.sh          # GPU 5, port 8002
#   bash scripts/launch_selfrag_vllm.sh 4 8003   # GPU 4, port 8003
#
# After launching, set the environment variable so OminiRAG connects:
#   export SELFRAG_VLLM_URL=http://localhost:8002/v1
#
# Health check:
#   curl http://localhost:8002/v1/models

set -euo pipefail

GPU_ID="${1:-5}"
PORT="${2:-8002}"
MODEL_PATH="/data1/ragworkspace/self-rag/model_cache/models--selfrag--selfrag_llama2_7b/snapshots/190261383b0779ff66d2f95a73c7ad267d94b820"
VLLM_PYTHON="/data1/ragworkspace/download_model/.venv/bin/python"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$VLLM_PYTHON" ]; then
    echo "ERROR: vLLM Python not found: $VLLM_PYTHON"
    exit 1
fi

echo "=============================================="
echo "  Self-RAG vLLM Server"
echo "  GPU:   $GPU_ID"
echo "  Port:  $PORT"
echo "  Model: selfrag-llama2-7b"
echo "=============================================="

CUDA_VISIBLE_DEVICES="$GPU_ID" exec "$VLLM_PYTHON" \
    -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 --port "$PORT" \
    --served-model-name selfrag-llama2-7b \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 16 \
    --max-logprobs 32016
