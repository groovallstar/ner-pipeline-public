#!/bin/bash
# 로컬(호스트) NER 서버 기동 — GPU 0 고정(컨테이너와 동일), uv editable.
#
# 컨테이너 없이 호스트에서 바로 띄워 테스트하는 용도. 컨테이너가 8005 를
# 점유 중이면 포트를 옮긴다(예: --port 9000). GPU 는 CUDA_VISIBLE_DEVICES
# 로 고정하며 미설정 시 0(컨테이너 device_ids 기본과 일치).
#
# 사용: bash src/server/scripts/run_local.sh [--host H] [--port P] ...
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "Local NER server on GPU ${CUDA_VISIBLE_DEVICES} (uv editable)"
exec env -u PYTHONPATH uv run python -m server "$@"
