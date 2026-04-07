#!/bin/bash
# vLLM 컨테이너 중지 및 제거

set -e
cd "$(dirname "$0")"

docker compose -f docker-compose.yml down 2>/dev/null || true
echo "vLLM 컨테이너 중지 완료"
