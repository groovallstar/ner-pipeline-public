#!/bin/bash
# ner-server 빌드·기동 (단일 GPU, 호스트 /data 모델 마운트).
# 사용법: start.sh   (.env 있으면 compose 가 자동 로드; GPU 지정은
#   CUDA_VISIBLE_DEVICES 를 .env 또는 셸 환경에 둔다)
set -e
cd "$(dirname "$0")"

PROJECT="${NER_SERVER_PROJECT:-ner-server}"

echo "기존 컨테이너 정리..."
docker compose -p "$PROJECT" -f docker-compose.yml down 2>/dev/null || true

echo "빌드·기동 (project=$PROJECT)..."
docker compose -p "$PROJECT" -f docker-compose.yml up -d --build

echo "로그: bash logs.sh   |   헬스: curl -s localhost:${NER_SERVER_PORT:-8000}/health"
