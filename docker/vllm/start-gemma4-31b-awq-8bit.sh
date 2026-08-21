#!/bin/bash
# cyankiwi/gemma-4-31B-it-AWQ-8bit (AWQ 8bit, Dense 31B, TP=1, GPU1, port 8081)
# gpu_mem_util 이 0.94 로 높은 이유 — 가중치만 33.5 GiB 라 48 GB 카드에서 KV 캐시로
# 갈 여유가 10 GiB 남짓이다. 이 값을 낮추면 동시 처리 가능한 요청 수가 그대로 줄어든다.

set -e
cd "$(dirname "$0")"

export VLLM_MODEL=cyankiwi/gemma-4-31B-it-AWQ-8bit
export VLLM_GPU_MEM_UTIL=0.94
export VLLM_MAX_MODEL_LEN=8192
export VLLM_TENSOR_PARALLEL=1
export VLLM_DTYPE=auto
export VLLM_PORT=8081
export VLLM_EXTRA_ARGS=""
export VLLM_IMAGE=vllm/vllm-openai:v0.26.0
export VLLM_CONTAINER=vllm-gemma4-31b-awq-8bit
export CUDA_VISIBLE_DEVICES=1

echo "기존 컨테이너 정리..."
docker compose -p vllm-gemma4-31b-awq-8bit -f docker-compose.yml down 2>/dev/null || true

echo "모델: $VLLM_MODEL (GPU 1, 포트 $VLLM_PORT, gpu_mem_util=$VLLM_GPU_MEM_UTIL)"
docker compose -p vllm-gemma4-31b-awq-8bit -f docker-compose.yml up -d
echo "로그 확인: docker logs -f $VLLM_CONTAINER"
echo "준비 확인: curl -s localhost:$VLLM_PORT/v1/models"
