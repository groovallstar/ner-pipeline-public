#!/bin/bash
# compose 가 해석한 실행 대상(프로젝트·컨테이너·게시 포트)을 읽어 온다.
#
# 스크립트가 .env 를 직접 파싱하지 않는 것은, 설정을 푸는 곳이 둘이 되면 둘이
# 서로 다른 대상을 고를 수 있기 때문이다. 실제로 컨테이너를 만드는 compose 에게
# 물어보면 안내와 실행 대상이 같은 출처에서 나온다.
#
# 사용법: `. resolve.sh` 로 불러들인 뒤 `resolve_target <compose 파일>` 을 부르면
# COMPOSE_TARGET_PROJECT·COMPOSE_TARGET_CONTAINER·COMPOSE_TARGET_PORT 가 채워진다.
# 해석에 실패하면 셋을 빈 문자열로 두고 1 을 반환한다 — 추측한 기본값을 채우면
# 바로 그것이 이 스크립트가 없애려는 어긋남이 된다.
resolve_target() {
  local compose_file="$1" summary
  COMPOSE_TARGET_PROJECT=''
  COMPOSE_TARGET_CONTAINER=''
  COMPOSE_TARGET_PORT=''
  summary="$(docker compose -f "$compose_file" config --format json 2>/dev/null \
    | python3 -c '
import json, sys

try:
    cfg = json.load(sys.stdin)
except Exception:
    sys.exit(1)
svc = cfg.get("services", {}).get("ner-server") or {}
ports = svc.get("ports") or []
published = str(ports[0].get("published", "")) if ports else ""
print(cfg.get("name", ""), svc.get("container_name", ""), published, sep="\t")
' 2>/dev/null)" || return 1
  [ -n "$summary" ] || return 1
  IFS=$'\t' read -r COMPOSE_TARGET_PROJECT COMPOSE_TARGET_CONTAINER COMPOSE_TARGET_PORT <<<"$summary"
  [ -n "$COMPOSE_TARGET_PORT" ] || return 1
  return 0
}
