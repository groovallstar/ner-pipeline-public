#!/bin/bash
# ner-server 중지·제거.
# 사용법: stop.sh [project]
#   인자를 주면 그 프로젝트를 내린다(compose 밖에서 만든 이름까지 지정 가능).
#   안 주면 compose 가 .env·셸의 COMPOSE_PROJECT_NAME 또는 docker-compose.yml
#   의 name: 으로 정한 프로젝트라, start.sh 가 올린 것과 같은 대상이 된다.
set -e
cd "$(dirname "$0")"
. ./resolve.sh

if [ -n "${1:-}" ]; then
  PROJECT_ARGS=(-p "$1")
  PROJECT="$1"
else
  PROJECT_ARGS=()
  PROJECT=''
  if resolve_target docker-compose.yml; then
    PROJECT="$COMPOSE_TARGET_PROJECT"
  fi
fi

if docker compose "${PROJECT_ARGS[@]}" -f docker-compose.yml down; then
  echo "Stopped: ${PROJECT:-unresolved project name}"
else
  status=$?
  echo "Failed to stop: ${PROJECT:-unresolved project name} (exit $status)" >&2
  exit "$status"
fi
