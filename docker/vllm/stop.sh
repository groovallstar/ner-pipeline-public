#!/bin/bash
# vLLM 컨테이너 중지 및 제거
# 사용법: stop.sh [project...] (기본: 상시 운영 2종)
# 인자는 compose 프로젝트 이름이다 — 컨테이너 이름과 다를 수 있다
# (점을 쓸 수 없어 qwen3.8 은 프로젝트가 vllm-qwen38-... 이다).

set -e
cd "$(dirname "$0")"

PROJECTS="${*:-vllm-gemma4-31b-awq-8bit vllm-qwen38-27b-w4a16-awq}"
for p in $PROJECTS; do
  docker compose -p "$p" -f docker-compose.yml down 2>/dev/null \
    && echo "중지: $p" || true
done
echo "vLLM 컨테이너 중지 완료"
