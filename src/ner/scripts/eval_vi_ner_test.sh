#!/usr/bin/env bash
# eval_vi_ner_test.sh — VI NER test 추론을 uv 환경에서 실행하는 래퍼.
#
# 본체 eval_vi_ner_test.py(argparse·절대경로 검증)를 `uv run` 으로 호출해
# 프로젝트 .venv 를 자동 동기화·적용한다. editable install 이 .pth 로 src/ 를
# 등록하므로 ner.* import 가 venv 안에서 해결된다.
#
# 사용:
#   ./src/ner/scripts/eval_vi_ner_test.sh
#   ./src/ner/scripts/eval_vi_ner_test.sh \
#       --model-dir /abs/model --test /abs/test.jsonl
set -euo pipefail

# 스크립트 위치 기준으로 리포지토리 루트로 이동 (어느 cwd 에서 실행해도 동작)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."

# uv run → 프로젝트 .venv 자동 적용 후 본체 .py 실행 (인자 그대로 전달)
exec uv run python src/ner/scripts/eval_vi_ner_test.py "$@"
