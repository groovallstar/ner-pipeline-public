#!/bin/bash
# RedHatAI/gemma-2-9b-it-quantized.w4a16 (W4A16 4bit, 9B, TP=1) — 웹 데모 번역용
# 경량 엔진(~6GB). NER 추론과 카드를 분리해 쓰는 배포를 상정한다.
# 근거·선정 이유: docs/reports/translate-engine-lightweight-benchmark.md
#
# **Ampere 이상 전용** — 이 모델은 vLLM 에서 bf16 전용이라 Turing(2080Ti,
# cc 7.5)에는 올라가지 않는다(float16 은 모델 타입 수준에서 거부, float32 는
# Marlin 커널에서 실패). Turing 배포는 llama.cpp(GGUF) 백엔드나 fp16 안전
# 계열 모델로 가야 한다 — 리포트 §권고 참조.
#
# 환경변수로 덮어쓸 수 있다 — GPU/포트/메모리/컨텍스트:
#   CUDA_VISIBLE_DEVICES=0 VLLM_PORT=8093 ./start-gemma2-9b.sh

set -e
cd "$(dirname "$0")"

export VLLM_MODEL=RedHatAI/gemma-2-9b-it-quantized.w4a16
# 전담 카드 기준. NER 과 한 카드를 나눠 쓰면 낮춘다.
export VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.85}
# KV 예산이 빠듯한 소형 카드 기본값 — 여유 있으면 8192 로 올린다.
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
export VLLM_TENSOR_PARALLEL=1
# auto = bf16. 다른 dtype 은 선택지가 아니다 — float16 은 vLLM 이 모델 타입
# 수준에서 거부하고(수치 불안정), float32 는 Marlin 커널에서 실패한다(실측).
export VLLM_DTYPE=${VLLM_DTYPE:-auto}
export VLLM_PORT=${VLLM_PORT:-8081}
export VLLM_EXTRA_ARGS=""
# 상시 운영 2종과 이미지를 맞춘다. 다만 이 모델로는 아직 기동을 확인하지 않았다 —
# 웹 데모용 선택 스크립트라 상시로 돌지 않는다. 처음 띄울 때 로그를 확인할 것.
export VLLM_IMAGE=vllm/vllm-openai:v0.26.0
export VLLM_CONTAINER=vllm-gemma2-9b
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

echo "기존 컨테이너 정리..."
docker compose -p vllm-gemma2-9b -f docker-compose.yml down 2>/dev/null || true

echo "모델: $VLLM_MODEL (GPU $CUDA_VISIBLE_DEVICES, 포트 $VLLM_PORT," \
     "gpu_mem_util=$VLLM_GPU_MEM_UTIL, dtype=$VLLM_DTYPE," \
     "max_model_len=$VLLM_MAX_MODEL_LEN)"
docker compose -p vllm-gemma2-9b -f docker-compose.yml up -d
echo "로그 확인: docker logs -f $VLLM_CONTAINER"
echo "준비 확인: curl -s localhost:$VLLM_PORT/v1/models"
