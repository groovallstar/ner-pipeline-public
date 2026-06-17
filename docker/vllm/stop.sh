#!/bin/bash
# vLLM 컨테이너 중지 및 제거
# 사용법: stop.sh [project...] (기본: vllm-gemma vllm-qwen 모두)

set -e
cd "$(dirname "$0")"

PROJECTS="${*:-vllm-gemma vllm-qwen}"
for p in $PROJECTS; do
  docker compose -p "$p" -f docker-compose.yml down 2>/dev/null \
    && echo "중지: $p" || true
done
echo "vLLM 컨테이너 중지 완료"
