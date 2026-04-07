#!/bin/bash
# 3개 모델 순차 벤치마크: 언로드 → 로드 → 대기 → 벤치마크
# Usage: bash bench-3models.sh [MAX_SAMPLES]

set -e
cd "$(dirname "$0")"

MAX_SAMPLES=${1:-500}
VLLM_URL="http://localhost:8081/v1"
VLLM_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="/work/git/ner_pipeline/src"
OUT_DIR="/work/git/ner_pipeline/results"
mkdir -p "$OUT_DIR"

wait_for_vllm() {
    local model_name="$1"
    local max_wait=300  # 5분
    local elapsed=0
    echo "  vLLM 서버 대기 중 ($model_name)..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "$VLLM_URL/models" 2>/dev/null | grep -q "$model_name"; then
            echo "  vLLM 준비 완료! (${elapsed}초)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo "  ... ${elapsed}초 경과"
    done
    echo "  ERROR: vLLM 서버 타임아웃 (${max_wait}초)"
    return 1
}

run_benchmark() {
    local model="$1"
    local output="$2"
    echo "  벤치마크 시작: $model (${MAX_SAMPLES} samples)"
    cd "$BENCH_DIR"
    python -m evaluators --lang ko \
        --models "vllm:${model}" \
        --max-samples "$MAX_SAMPLES" \
        --vllm-url "$VLLM_URL" \
        --concurrency 8 \
        --no-bertscore \
        --output "$output" 2>&1
    echo "  결과 저장: $output"
}

echo "============================================"
echo "  3-Model Korean NER Benchmark (${MAX_SAMPLES} samples)"
echo "============================================"

# --- Model 1: Opus Distilled (현재 로드됨) ---
MODEL1="Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled"
echo ""
echo "[1/3] $MODEL1"
# 이미 로드되어 있는지 확인, 아니면 로드
if ! curl -s "$VLLM_URL/models" 2>/dev/null | grep -q "Opus-Reasoning-Distilled"; then
    echo "  모델 로드 중..."
    bash "$VLLM_DIR/start-opus-distilled.sh"
    wait_for_vllm "Opus-Reasoning-Distilled"
else
    echo "  이미 로드됨"
fi
run_benchmark "$MODEL1" "$OUT_DIR/ko_bench_500_opus_distilled.json"

# --- Model 2: Qwen3.5-35B-A3B ---
MODEL2="Qwen/Qwen3.5-35B-A3B"
echo ""
echo "[2/3] $MODEL2"
bash "$VLLM_DIR/start-35b-a3b.sh"
wait_for_vllm "Qwen3.5-35B-A3B"
run_benchmark "$MODEL2" "$OUT_DIR/ko_bench_500_35b_a3b.json"

# --- Model 3: Qwen3.5-27B ---
MODEL3="Qwen/Qwen3.5-27B"
echo ""
echo "[3/3] $MODEL3"
bash "$VLLM_DIR/start-27b.sh"
wait_for_vllm "Qwen3.5-27B"
run_benchmark "$MODEL3" "$OUT_DIR/ko_bench_500_27b.json"

echo ""
echo "============================================"
echo "  벤치마크 완료!"
echo "  결과 파일:"
echo "    $OUT_DIR/ko_bench_500_opus_distilled.json"
echo "    $OUT_DIR/ko_bench_500_35b_a3b.json"
echo "    $OUT_DIR/ko_bench_500_27b.json"
echo "============================================"
