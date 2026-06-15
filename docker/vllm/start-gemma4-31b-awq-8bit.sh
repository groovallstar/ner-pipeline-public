#!/bin/bash
# cyankiwi/gemma-4-31B-it-AWQ-8bit (AWQ 8bit, Dense 31B, TP=1, GPU1, port 8081)

set -e
cd "$(dirname "$0")"

export VLLM_MODEL=cyankiwi/gemma-4-31B-it-AWQ-8bit
export VLLM_GPU_MEM_UTIL=0.92
export VLLM_MAX_MODEL_LEN=8192
export VLLM_TENSOR_PARALLEL=1
export VLLM_DTYPE=auto
export VLLM_PORT=8081
export VLLM_EXTRA_ARGS=""
export VLLM_IMAGE=vllm/vllm-openai:gemma4
export VLLM_CONTAINER=vllm-gemma4-31b-awq8
export CUDA_VISIBLE_DEVICES=1

echo "기존 컨테이너 정리..."
docker compose -p vllm-gemma4-31b-awq8 -f docker-compose.yml down 2>/dev/null || true

echo "모델: $VLLM_MODEL (GPU 1, 포트 $VLLM_PORT, gpu_mem_util=$VLLM_GPU_MEM_UTIL)"
docker compose -p vllm-gemma4-31b-awq8 -f docker-compose.yml up -d
echo "로그 확인: docker logs -f $VLLM_CONTAINER"
