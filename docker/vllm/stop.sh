#!/bin/bash
# vLLM 컨테이너 중지 및 제거
# 사용법: stop.sh [project...] (기본: 상시 운영 2종)
# 인자는 compose 프로젝트 이름이다 — 컨테이너 이름과 다를 수 있다
# (점을 쓸 수 없어 qwen3.8 은 프로젝트가 vllm-qwen38-... 이다).

set -e
cd "$(dirname "$0")"

PROJECTS="${*:-vllm-gemma4-31b-awq-8bit vllm-qwen38-27b-w4a16-awq}"
status=0
for p in $PROJECTS; do
  if docker compose -p "$p" -f docker-compose.yml down; then
    echo "Stopped: $p"
  else
    project_status=$?
    echo "Failed to stop: $p (exit $project_status)" >&2
    # 나머지 프로젝트도 종료하되 첫 실패 코드는 보존한다.
    if [ "$status" -eq 0 ]; then
      status=$project_status
    fi
  fi
done
if [ "$status" -ne 0 ]; then
  echo "Some vLLM projects could not be stopped." >&2
  exit "$status"
fi
echo "All vLLM projects stopped."
