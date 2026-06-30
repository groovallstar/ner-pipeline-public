#!/bin/bash
# ner-server 로그 tail.
# 사용법: logs.sh [컨테이너명] (기본: ner-server)
docker logs -f "${1:-ner-server}"
