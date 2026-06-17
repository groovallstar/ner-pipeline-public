"""배포된 JA NER 모델 test 추론 — 문장별 태깅 표시 + 평가지표 출력.

학습/검증 없이 고정 test 세트와 저장된 per-class 임계값(thresholds.json)만
사용한다(실제 추론 기능). 각 test 문장이 어떻게 태깅됐는지 먼저 보여주고,
단계별 소요 시간(모델·토크나이저 로드 / 추론 전체 / 1문장당 추론)을 초 단위로
출력한 뒤, 임계값 적용 후 Precision / Recall / F1-Score 를 출력한다. 모델·
test·임계값 경로는 전체(절대) 경로만 받는다(상대 경로 거부).

사용:
    python src/ner/scripts/eval_ja_ner_test.py
    python src/ner/scripts/eval_ja_ner_test.py \
        --model-dir /abs/model --test /abs/test.jsonl \
        --thresholds /abs/thresholds.json
"""
import argparse
import os
import time

from transformers import AutoTokenizer

from ner.classifier.abstention import apply_thresholds, load_thresholds
from ner.classifier.data_utils import (
    build_label_maps,
    encode_dataset,
    load_jsonl,
)
from ner.classifier.train_eval import evaluate_model
from ner.metrics.span_metrics import compute_offset_span_f1

DEFAULT_MODEL_DIR = '/data/ner/ja/model'
DEFAULT_TEST = '/data/ner/ja/data/test.jsonl'
DEFAULT_THRESHOLDS = '/data/ner/ja/thresholds.json'


def require_abs(path, label):
    """데이터 경로는 전체(절대) 경로만 허용 — 상대 경로면 종료."""
    if not os.path.isabs(path):
        raise SystemExit(
            f'Error: {label} must be an absolute path (got: {path})')


def show_tagging(rows, pred_spans_list):
    """문장별 예측 태깅(엔티티 type:표면형) 표시. surface 는 char offset 추출."""
    print(f'=== 태깅 결과 (테스트 {len(rows)}문장) ===')
    for i, (row, spans) in enumerate(zip(rows, pred_spans_list), 1):
        text = row['text']
        if spans:
            ents = '  '.join(
                f"{s['type']}:{text[s['start']:s['end']]}"
                for s in sorted(spans, key=lambda s: s['start']))
        else:
            ents = '(없음)'
        print(f'[{i:>3}] {text}')
        print(f'      {ents}')


def main():
    p = argparse.ArgumentParser(
        description='Evaluate the deployed JA NER model on a fixed test '
                    'set: show per-sentence tagging, then print Precision / '
                    'Recall / F1-Score at the abstention operating point. '
                    'Paths must be absolute.')
    p.add_argument('--model-dir', default=DEFAULT_MODEL_DIR,
                   help='Absolute model dir (with tokenizer)')
    p.add_argument('--test', default=DEFAULT_TEST,
                   help='Absolute test JSONL path')
    p.add_argument('--thresholds', default=DEFAULT_THRESHOLDS,
                   help='Absolute thresholds.json path')
    args = p.parse_args()

    require_abs(args.model_dir, 'model dir')
    require_abs(args.test, 'test data path')
    require_abs(args.thresholds, 'thresholds path')

    label2id, id2label = build_label_maps()

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    tok_load_sec = time.perf_counter() - t0

    rows = load_jsonl(args.test)
    feats, offs = encode_dataset(rows, tok, label2id, 'ja', 256)
    res = evaluate_model(
        model_path=args.model_dir, eval_features=feats, eval_offsets=offs,
        eval_rows=rows, id2label=id2label, return_spans=True,
        capture_scores=True, capture_timing=True)

    thr = load_thresholds(args.thresholds)
    pred = apply_thresholds(res['pred_spans_list'], thr)

    # 1) 샘플 추론 내용
    show_tagging(rows, pred)

    # 2) 추론 시간 — 모델·토크나이저 로드 / 추론 전체 / 1문장당 추론
    load_sec = tok_load_sec + res['load_seconds']
    infer_sec = res['infer_seconds']
    n = len(rows)
    per_sent = infer_sec / n if n else 0.0
    print('\n' + '=' * 60)
    print('=== 추론 시간 ===')
    print(f'Model + tokenizer load: {load_sec:.3f} sec')
    print(f'Inference (total):      {infer_sec:.3f} sec  ({n} sentences)')
    print(f'Inference (per sentence): {per_sent:.4f} sec')

    # 3) 평가지표
    op = compute_offset_span_f1(res['gold_spans_list'], pred)['overall']
    print('\n' + '=' * 60)
    print('=== 평가지표 ===')
    print(f"Precision: {op['precision']:.4f}")
    print(f"Recall:    {op['recall']:.4f}")
    print(f"F1-Score:  {op['f1']:.4f}")


if __name__ == '__main__':
    main()
