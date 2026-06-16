#!/bin/bash
# google/gemma-4-31B-it 모델 시작 (BF16, TP=2)

set -e
cd "$(dirname "$0")"

export VLLM_MODEL=google/gemma-4-31B-it
export VLLM_GPU_MEM_UTIL=0.9
export VLLM_MAX_MODEL_LEN=8192
export VLLM_TENSOR_PARALLEL=2
export VLLM_DTYPE=bfloat16
export VLLM_PORT=8081
export VLLM_IMAGE=vllm/vllm-openai:v0.23.0
export CUDA_VISIBLE_DEVICES=1,2

echo "기존 컨테이너 정리..."
docker compose -f docker-compose.yml down 2>/dev/null || true

echo "모델: $VLLM_MODEL"
docker compose -f docker-compose.yml up -d
echo "로그 확인: bash logs.sh"
