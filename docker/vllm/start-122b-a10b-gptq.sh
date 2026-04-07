#!/bin/bash
# Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 모델 시작
# 기존 컨테이너가 있으면 먼저 중지 후 새로 시작
# GPTQ-Int4 양자화 → --quantization moe_wna16

set -e
cd "$(dirname "$0")"

export VLLM_MODEL=Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
export VLLM_GPU_MEM_UTIL=0.9
export VLLM_MAX_MODEL_LEN=32768
export VLLM_TENSOR_PARALLEL=2
export VLLM_DTYPE=auto
export VLLM_PORT=8081
export CUDA_VISIBLE_DEVICES=1,2
export VLLM_EXTRA_ARGS="--quantization moe_wna16 --trust-remote-code"

echo "기존 컨테이너 정리..."
docker compose -f docker-compose.yml down 2>/dev/null || true

echo "모델: $VLLM_MODEL"
echo "추가 인자: $VLLM_EXTRA_ARGS"
docker compose -f docker-compose.yml up -d
echo "로그 확인: bash logs.sh"
