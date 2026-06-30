#!/bin/bash
# ner-server 중지·제거.
# 사용법: stop.sh [project] (기본: ner-server)
set -e
cd "$(dirname "$0")"

PROJECT="${1:-${NER_SERVER_PROJECT:-ner-server}}"
docker compose -p "$PROJECT" -f docker-compose.yml down 2>/dev/null \
  && echo "중지: $PROJECT" || true
