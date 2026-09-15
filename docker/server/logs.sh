#!/bin/bash
# ner-server 로그 tail.
# 사용법: logs.sh [컨테이너명]
#   인자가 없으면 compose 서비스(ner-server)에 붙는다 — 컨테이너 이름을 이
#   스크립트가 알 필요가 없어, .env 에서 NER_SERVER_CONTAINER 를 바꿔도 대상이
#   어긋나지 않는다. 인자를 주면 그 컨테이너에 직접 붙는다(compose 밖에서 띄운
#   컨테이너를 볼 때).
#   compose 는 -f 로 준 파일이 있는 디렉토리에서 .env 를 읽으므로, cd 없이도
#   실행 위치와 무관하다.
if [ -n "${1:-}" ]; then
  docker logs -f "$1"
else
  docker compose -f "$(dirname "$0")/docker-compose.yml" logs -f --no-log-prefix ner-server
fi
