#!/bin/bash
# cyankiwi/Qwen3.5-27B-AWQ-4bit (AWQ 4bit, TP=1, GPU2, port 8082)

set -e
cd "$(dirname "$0")"

export VLLM_MODEL=cyankiwi/Qwen3.5-27B-AWQ-4bit
export VLLM_GPU_MEM_UTIL=0.9
export VLLM_MAX_MODEL_LEN=8192
export VLLM_TENSOR_PARALLEL=1
export VLLM_DTYPE=auto
export VLLM_PORT=8082
export VLLM_EXTRA_ARGS=""
export VLLM_IMAGE=vllm/vllm-openai:v0.19.0
export VLLM_CONTAINER=vllm-qwen
export CUDA_VISIBLE_DEVICES=2

echo "기존 컨테이너 정리..."
docker compose -p vllm-qwen -f docker-compose.yml down 2>/dev/null || true

echo "모델: $VLLM_MODEL (GPU 2, 포트 $VLLM_PORT)"
docker compose -p vllm-qwen -f docker-compose.yml up -d
echo "로그 확인: docker logs -f $VLLM_CONTAINER"
