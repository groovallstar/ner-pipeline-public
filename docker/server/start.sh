#!/bin/bash
# ner-server 빌드·기동 (단일 GPU, 호스트 /data 모델 마운트).
# 사용법: start.sh [--no-build]
#   기본은 --build(이미지 재빌드). 서버 코드 변경이 없으면 --no-build 로
#   기존 이미지를 빠르게 띄운다(uv sync 레이어 재실행 회피). .env 있으면
#   compose 가 자동 로드; GPU 는 NER_SERVER_GPU(기본 0).
set -e
cd "$(dirname "$0")"

case "${1:-}" in
  ""|--build) BUILD="--build" ;;
  --no-build) BUILD="" ;;
  *) echo "Usage: start.sh [--no-build]" >&2; exit 2 ;;
esac

PROJECT="${NER_SERVER_PROJECT:-ner-server}"

echo "기존 컨테이너 정리..."
docker compose -p "$PROJECT" -f docker-compose.yml down 2>/dev/null || true

echo "기동 (project=$PROJECT${BUILD:+, build})..."
docker compose -p "$PROJECT" -f docker-compose.yml up -d $BUILD

echo "로그: bash logs.sh   |   헬스: curl -s localhost:${NER_SERVER_PORT:-8000}/health"
