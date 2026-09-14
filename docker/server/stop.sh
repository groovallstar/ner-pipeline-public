#!/bin/bash
# ner-server 중지·제거.
# 사용법: stop.sh [project] (기본: ner-server)
set -e
cd "$(dirname "$0")"

PROJECT="${1:-${NER_SERVER_PROJECT:-ner-server}}"
if docker compose -p "$PROJECT" -f docker-compose.yml down; then
  echo "Stopped: $PROJECT"
else
  status=$?
  echo "Failed to stop: $PROJECT (exit $status)" >&2
  exit "$status"
fi
