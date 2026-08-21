#!/bin/bash
# philbert440/Qwen3.8-27B-W4A16-AWQ (Dense 27.8B, hybrid attention + vision, W4A16, TP=1, GPU2, port 8082)
#
# 체크포인트를 philbert440 으로 고른 이유 — 같은 원본의 cyankiwi/Qwen3.8-27B-AWQ-INT4
# 대비 KO NER Exact-F1 이 0.034 높고 속도는 같으며 본체가 2.3 GB 작다. group_size 는
# 오히려 cyankiwi 가 촘촘한데(32 vs 128) 뒤지는데, AWQ 품질은 granularity 보다
# 활성값 기반 보호(smoothing)가 좌우하고 philbert440 쪽이 이 모델의 linear attention
# 경로까지 제대로 훑었기 때문으로 보인다.
#
# --enable-prefix-caching 을 명시하는 이유 — 이 모델은 hybrid 구조(linear attention +
# mamba 상태)라 vLLM 이 기본으로 끈다. 켜면 긴 지시문을 공유하는 라벨링에서 prefill 을
# 건너뛰어 2.7배 빨라진다. mamba 경로는 실험 단계로 표시돼 있으나, KO 300샘플에서
# 출력이 바뀌지 않는 것을 확인했다.

set -e
cd "$(dirname "$0")"

export VLLM_MODEL=philbert440/Qwen3.8-27B-W4A16-AWQ
export VLLM_GPU_MEM_UTIL=0.9
export VLLM_MAX_MODEL_LEN=8192
export VLLM_TENSOR_PARALLEL=1
export VLLM_DTYPE=auto
export VLLM_PORT=8082
export VLLM_EXTRA_ARGS="--enable-prefix-caching"
export VLLM_IMAGE=vllm/vllm-openai:v0.26.0
# compose 프로젝트 이름에는 점을 못 쓴다 — 컨테이너 이름만 모델명 그대로 두고
# 프로젝트는 점을 뺀 이름을 쓴다. stop.sh 에 넘기는 것은 프로젝트 이름이다.
export VLLM_CONTAINER=vllm-qwen3.8-27b-w4a16-awq
VLLM_PROJECT=vllm-qwen38-27b-w4a16-awq
export CUDA_VISIBLE_DEVICES=2

echo "기존 컨테이너 정리..."
docker compose -p "$VLLM_PROJECT" -f docker-compose.yml down 2>/dev/null || true

echo "모델: $VLLM_MODEL (GPU 2, 포트 $VLLM_PORT, gpu_mem_util=$VLLM_GPU_MEM_UTIL)"
docker compose -p "$VLLM_PROJECT" -f docker-compose.yml up -d
echo "로그 확인: docker logs -f $VLLM_CONTAINER"
echo "준비 확인: curl -s localhost:$VLLM_PORT/v1/models"
