#!/bin/bash
# 모델 다운로드 → vLLM 로드 → 벤치마크 실행 자동화
# 사용법: bash load-and-bench.sh <모델명> [벤치마크 출력 파일명]
# 예시:  bash load-and-bench.sh Qwen/Qwen3.5-35B-A3B

set -e

if [ -z "$1" ]; then
  echo "사용법: $0 <모델명> [출력파일명]"
  exit 1
fi

MODEL="$1"
OUTPUT="${2:-results/error_analysis_$(echo $MODEL | tr '/' '_').json}"
VLLM_URL="http://172.22.0.2:8081/v1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "  모델: $MODEL"
echo "  출력: $OUTPUT"
echo "=========================================="

# ── Step 1: 모델 다운로드 ──────────────────────────────────────────
echo ""
echo "[1/4] 모델 다운로드 중..."
cd "$PROJECT_DIR"
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', cache_dir='/work/.huggingface')
print('Download complete')
"

# ── Step 2: vLLM 모델 교체 ─────────────────────────────────────────
echo ""
echo "[2/4] vLLM 모델 교체 중..."
cd "$SCRIPT_DIR"
sed -i "s|^VLLM_MODEL=.*|VLLM_MODEL=${MODEL}|" config/.env

# 베이스 모델 tokenizer가 필요한 경우 감지 (TokenizersBackend 호환 문제)
SNAP_DIR=$(ls -d /work/.huggingface/hub/models--$(echo "$MODEL" | tr '/' '--')/snapshots/*/ 2>/dev/null | head -1)
if [ -n "$SNAP_DIR" ] && grep -q "TokenizersBackend" "$SNAP_DIR/tokenizer_config.json" 2>/dev/null; then
  echo "  TokenizersBackend 감지 → --tokenizer + --trust-remote-code 추가"
  # 베이스 모델 추론: config.json의 _name_or_path 또는 기본값
  BASE_TOKENIZER="Qwen/Qwen3.5-35B-A3B"
  sed -i '/^VLLM_EXTRA_ARGS=/d' config/.env
  echo "VLLM_EXTRA_ARGS=--tokenizer ${BASE_TOKENIZER} --trust-remote-code" >> config/.env
else
  sed -i '/^VLLM_EXTRA_ARGS=/d' config/.env
fi

docker compose --env-file config/.env -f docker-compose.yml down
docker compose --env-file config/.env -f docker-compose.yml up -d

# ── Step 3: 모델 로딩 대기 ─────────────────────────────────────────
echo ""
echo "[3/4] 모델 로딩 대기 중..."
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
for i in $(seq 1 120); do
  result=$(curl -s "$VLLM_URL/models" 2>/dev/null || true)
  if echo "$result" | grep -q "$MODEL_SHORT"; then
    echo "  모델 로딩 완료! (${i}0초 소요)"
    break
  fi
  if [ $i -eq 120 ]; then
    echo "  ERROR: 모델 로딩 타임아웃 (20분)"
    echo "  로그 확인: docker logs vllm-server --tail 20"
    docker logs vllm-server --tail 20 2>&1
    exit 1
  fi
  if [ $((i % 6)) -eq 0 ]; then
    echo "  대기 중... (${i}0초)"
  fi
  sleep 10
done

# ── Step 4: 벤치마크 실행 ──────────────────────────────────────────
echo ""
echo "[4/4] 벤치마크 실행 중..."
cd "$PROJECT_DIR"
.venv/bin/python -m llm_eval.error_analysis \
  --models "vllm:$MODEL" \
  --vllm-url "$VLLM_URL" \
  --max-samples 50 \
  --output "$OUTPUT"

echo ""
echo "=========================================="
echo "  완료! 결과: $OUTPUT"
echo "=========================================="
