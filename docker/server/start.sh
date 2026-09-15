#!/bin/bash
# ner-server 빌드·기동 (단일 GPU, 호스트 /data 모델 마운트).
# 사용법: start.sh [--no-build]
#   기본은 --build(이미지 재빌드). 서버 코드 변경이 없으면 --no-build 로
#   기존 이미지를 빠르게 띄운다(uv sync 레이어 재실행 회피). GPU 는
#   NER_SERVER_GPU(기본 0).
#
#   프로젝트·컨테이너·포트는 이 스크립트가 정하지 않는다. compose 가 .env 와
#   셸을 읽어 정한 값을 그대로 쓰고, 마지막 안내도 그 값으로 찍으므로 안내
#   주소와 실제 기동 대상이 어긋날 수 없다.
set -e
cd "$(dirname "$0")"
. ./resolve.sh

case "${1:-}" in
  ""|--build) BUILD="--build" ;;
  --no-build) BUILD="" ;;
  *) echo "Usage: start.sh [--no-build]" >&2; exit 2 ;;
esac

echo "Removing existing containers..."
if docker compose -f docker-compose.yml down; then
  :
else
  status=$?
  echo "Cleanup failed; startup aborted (exit $status)." >&2
  exit "$status"
fi

echo "Starting${BUILD:+ (build)}..."
docker compose -f docker-compose.yml up -d $BUILD

if resolve_target docker-compose.yml; then
  echo "Target: project=$COMPOSE_TARGET_PROJECT container=$COMPOSE_TARGET_CONTAINER port=$COMPOSE_TARGET_PORT"
  echo "Logs: bash logs.sh   |   Health: curl -s localhost:$COMPOSE_TARGET_PORT/health"
else
  echo "Failed to resolve target. Check docker compose -f docker-compose.yml config." >&2
  echo "Logs: bash logs.sh"
fi
