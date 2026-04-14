#!/usr/bin/env bash
# eval_spans.sh — bio_dataset.load() → vLLM labeling → span evaluation
#
# Usage:
#   ./src/scripts/eval_spans.sh                          # 기본값: klue, 20건
#   ./src/scripts/eval_spans.sh --dataset klue --samples 500
#   ./src/scripts/eval_spans.sh --dataset "nlp-kmu/kor_ner" --samples 100
#   ./src/scripts/eval_spans.sh --model "Qwen/Qwen3.5-27B" --port 8082
#
# 환경변수:
#   VLLM_URL    — vLLM 서버 URL (기본: http://localhost:8081/v1)
#   CONCURRENCY — 동시 요청 수 (기본: 32)

set -euo pipefail
cd "$(dirname "$0")/../.."

# 기본값
DATASET="klue"
SPLIT="validation"
SAMPLES=100
MODEL=""
PORT=""
VLLM_URL="${VLLM_URL:-http://localhost:8081/v1}"
CONCURRENCY="${CONCURRENCY:-32}"

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)  DATASET="$2";  shift 2 ;;
        --split)    SPLIT="$2";    shift 2 ;;
        --samples)  SAMPLES="$2";  shift 2 ;;
        --model)    MODEL="$2";    shift 2 ;;
        --port)     PORT="$2";     shift 2 ;;
        --url)      VLLM_URL="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--dataset NAME] [--split SPLIT] [--samples N] [--model MODEL] [--port PORT]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --port 지정 시 URL 재구성
if [[ -n "$PORT" ]]; then
    VLLM_URL="http://localhost:${PORT}/v1"
fi

# vLLM 서버에서 모델 자동 감지
if [[ -z "$MODEL" ]]; then
    MODEL=$(python3 -c "
import json, urllib.request
resp = urllib.request.urlopen('${VLLM_URL}/models')
data = json.loads(resp.read())
print(data['data'][0]['id'])
" 2>/dev/null || true)
    if [[ -z "$MODEL" ]]; then
        echo "ERROR: vLLM 서버에 연결할 수 없습니다: ${VLLM_URL}"
        exit 1
    fi
fi

echo "=== Span Evaluation ==="
echo "  Dataset:     ${DATASET} (${SPLIT})"
echo "  Samples:     ${SAMPLES}"
echo "  Model:       ${MODEL}"
echo "  vLLM URL:    ${VLLM_URL}"
echo "  Concurrency: ${CONCURRENCY}"
echo ""

PYTHONPATH=src python3 -c "
from labelers.bio_dataset import load
from labelers.ko.vllm_ner_labeler import VllmNERLabeler
from llm_eval.span_evaluator import evaluate

gold = load('${DATASET}', '${SPLIT}', max_samples=${SAMPLES})
print(f'Loaded {len(gold)} gold records')

labeler = VllmNERLabeler(
    base_url='${VLLM_URL}',
    model='${MODEL}',
    concurrency=${CONCURRENCY},
)

result = evaluate(gold, labeler, show_progress=True)

print()
print('=== Results ===')
print(f'Exact Match:   P={result[\"exact\"][\"overall\"][\"precision\"]:.4f}  R={result[\"exact\"][\"overall\"][\"recall\"]:.4f}  F1={result[\"exact\"][\"overall\"][\"f1\"]:.4f}')
print(f'Relaxed Match: P={result[\"relaxed\"][\"overall\"][\"precision\"]:.4f}  R={result[\"relaxed\"][\"overall\"][\"recall\"]:.4f}  F1={result[\"relaxed\"][\"overall\"][\"f1\"]:.4f}')
print()
print(f'Gold: {result[\"counts\"][\"gold_spans\"]}  Pred: {result[\"counts\"][\"pred_spans\"]}  Exact: {result[\"counts\"][\"exact_matches\"]}  Relaxed: {result[\"counts\"][\"relaxed_matches\"]}  Errors: {result[\"counts\"][\"errors\"]}')
print(f'Latency: {result[\"latency\"][\"total_seconds\"]:.2f}s')
print()
print('Per-entity (exact):')
for etype, m in sorted(result['exact']['per_entity'].items()):
    print(f'  {etype:4s}: P={m[\"precision\"]:.4f} R={m[\"recall\"]:.4f} F1={m[\"f1\"]:.4f} (n={m[\"support\"]})')
"
