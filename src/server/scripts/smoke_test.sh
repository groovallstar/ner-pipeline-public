#!/usr/bin/env bash
# ja·vi NER REST API 실서버 스모크 테스트.
#
# uvicorn 으로 서버를 기동한 뒤 curl 로 /health·/v1/ner(단일·자동감지)·잘못된
# lang 에러를 검증하고, 모두 통과하면 0, 하나라도 실패하면 1 로 종료한다.
# in-process TestClient 가 못 보는 실제 포트 바인딩·네트워크 경로 확인용.
#
# 사용: bash src/server/scripts/smoke_test.sh [PORT]   (기본 PORT=8137)
set -euo pipefail

PORT="${1:-8137}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE="http://127.0.0.1:${PORT}"
LOG="$(mktemp -t ner_server_smoke.XXXXXX.log)"
cd "$ROOT"

uv run python -m server --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true' EXIT

# 모델 로드 + 포트 바인딩까지 대기(연결 거부 시 재시도).
if ! curl -s --retry 120 --retry-delay 1 --retry-connrefused \
        --retry-max-time 180 --max-time 10 "${BASE}/health" -o /dev/null; then
    echo "FAIL: server not ready"; tail -20 "$LOG"; exit 1
fi

fail=0
expect_code() {  # desc, actual, expected
    if [ "$2" = "$3" ]; then echo "  ok: $1"
    else echo "  FAIL: $1 (got $2, want $3)"; fail=1; fi
}

code=$(curl -s -o /dev/null -w '%{http_code}' "${BASE}/health")
expect_code "GET /health -> 200" "$code" "200"

resp=$(curl -s -X POST "${BASE}/v1/ner" -H 'Content-Type: application/json' \
    -d '{"text":"織田信長は東京都に住んでいた。"}')
if echo "$resp" | grep -q '"lang":"ja"'; then echo "  ok: ja autodetected"
else echo "  FAIL: ja detect ($resp)"; fail=1; fi
if echo "$resp" | grep -q '"label"'; then echo "  ok: entities present"
else echo "  FAIL: no entities ($resp)"; fail=1; fi

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${BASE}/v1/ner" \
    -H 'Content-Type: application/json' -d '{"text":"x","lang":"ko"}')
expect_code "bad lang -> 400" "$code" "400"

if [ "$fail" = 0 ]; then echo "SMOKE PASS"; else echo "SMOKE FAIL"; exit 1; fi
