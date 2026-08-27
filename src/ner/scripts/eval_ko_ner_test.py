"""배포된 KO NER 모델 test 추론 — 문장별 태깅 표시 + 평가지표 출력.

학습·검증 없이 고정 test 세트만 사용한다(실제 추론 기능). 각 test 문장이
어떻게 태깅됐는지 먼저 보여주고, 단계별 소요 시간(모델·토크나이저 로드 /
추론 전체 / 1문장당 추론)을 초 단위로 출력한 뒤, Precision / Recall /
F1-Score(strict span)를 출력한다. 저장된 per-class 임계값(thresholds.json)이
있으면 적용하고, 없으면 raw 모델 출력 그대로 평가한다 — KO 배포 패키지는
vi·en 과 마찬가지로 임계값을 쓰지 않으므로 폴백이 정상 경로다. 모델·test·
임계값 경로는 전체(절대) 경로만 받는다(상대 경로 거부).

**평가 문장 수와 태깅 표시 수를 나눠 받는다.** KO 배포 test 는 ja·vi 의
100문장 홀드아웃과 달리 수천 문장이라, 하나로 묶으면 지표를 전수로 재는 일과
태깅을 눈으로 보는 일이 서로를 막는다. `--limit` 은 평가 대상(기본 0=전수),
`--show` 는 화면에 찍을 문장 수(기본 100)다.

사용:
    python src/ner/scripts/eval_ko_ner_test.py
    python src/ner/scripts/eval_ko_ner_test.py --limit 500 --show 20
    python src/ner/scripts/eval_ko_ner_test.py \\
        --model-dir /abs/model --test /abs/test.jsonl \\
        --thresholds /abs/thresholds.json
"""
import argparse
import os
import time

from transformers import AutoTokenizer

from ner.classifier.confidence_threshold import (
    apply_thresholds,
    load_thresholds,
)
from ner.classifier.data_utils import (
    build_label_maps,
    encode_dataset,
    load_jsonl,
)
from ner.classifier.train_eval import evaluate_model
from ner.metrics.span_metrics import compute_offset_span_f1

DEFAULT_MODEL_DIR = '/data/ner/ko/model'
DEFAULT_TEST = '/data/ner/ko/data/test.jsonl'
DEFAULT_THRESHOLDS = '/data/ner/ko/thresholds.json'
# 화면에 찍을 문장 수 기본값 — 지표는 전수로 재고 표시만 자른다.
DEFAULT_SHOW = 100


def require_abs(path, label):
    """데이터 경로는 전체(절대) 경로만 허용 — 상대 경로면 종료."""
    if not os.path.isabs(path):
        raise SystemExit(
            f'Error: {label} must be an absolute path (got: {path})')


def show_tagging(rows, pred_spans_list, limit):
    """문장별 예측 태깅(엔티티 type:표면형) 표시. surface 는 char offset 추출."""
    shown = rows if limit <= 0 else rows[:limit]
    total = len(rows)
    head = f'=== 태깅 결과 ({len(shown)}문장'
    head += f' / 전체 {total})' if len(shown) < total else ')'
    print(head)
    for i, (row, spans) in enumerate(zip(shown, pred_spans_list), 1):
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
        description='Evaluate the deployed KO NER model on a fixed test '
                    'set: show per-sentence tagging, then print Precision / '
                    'Recall / F1-Score (strict span). Applies confidence '
                    'thresholds when a thresholds file exists, otherwise '
                    'evaluates raw output. Paths must be absolute.')
    p.add_argument('--model-dir', default=DEFAULT_MODEL_DIR,
                   help='Absolute model dir (with tokenizer)')
    p.add_argument('--test', default=DEFAULT_TEST,
                   help='Absolute test JSONL path')
    p.add_argument('--thresholds', default=DEFAULT_THRESHOLDS,
                   help='Absolute thresholds.json path (raw output if '
                        'missing)')
    p.add_argument('--limit', type=int, default=0,
                   help='Evaluate only the first N sentences (0 = all)')
    p.add_argument('--show', type=int, default=DEFAULT_SHOW,
                   help=f'Print tagging for the first N sentences '
                        f'(default {DEFAULT_SHOW}, 0 = all)')
    p.add_argument('--batch-size', type=int, default=64,
                   help='Inference batch size')
    args = p.parse_args()

    require_abs(args.model_dir, 'model dir')
    require_abs(args.test, 'test data path')
    require_abs(args.thresholds, 'thresholds path')

    label2id, id2label = build_label_maps()

    t0 = time.perf_counter()
    try:
        tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    except (TypeError, ValueError, OSError):
        tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    tok_load_sec = time.perf_counter() - t0

    rows = load_jsonl(args.test)
    if args.limit > 0 and len(rows) > args.limit:
        print(f'Evaluating the first {args.limit} of {len(rows)} sentences.')
        rows = rows[:args.limit]
    feats, offs = encode_dataset(rows, tok, label2id, 'ko', 256)
    res = evaluate_model(
        model_path=args.model_dir, eval_features=feats, eval_offsets=offs,
        eval_rows=rows, id2label=id2label, return_spans=True,
        capture_scores=True, capture_timing=True,
        batch_size=args.batch_size)

    # 임계값 파일이 있으면 신뢰도 임계 적용, 없으면 raw 출력으로 폴백.
    if os.path.exists(args.thresholds):
        thr = load_thresholds(args.thresholds)
        pred = apply_thresholds(res['pred_spans_list'], thr)
    else:
        print(f'Note: no thresholds file at {args.thresholds}; evaluating '
              'raw model output (the KO package ships without thresholds).')
        pred = res['pred_spans_list']

    # 1) 샘플 추론 내용
    show_tagging(rows, pred, args.show)

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

    # 3) 평가지표 (strict span, 임계값 적용 — 파일 없으면 raw)
    scored = compute_offset_span_f1(res['gold_spans_list'], pred)
    op = scored['overall']
    print('\n' + '=' * 60)
    print('=== 평가지표 ===')
    print(f"Precision: {op['precision']:.4f}")
    print(f"Recall:    {op['recall']:.4f}")
    print(f"F1-Score:  {op['f1']:.4f}")
    print(f"Support:   {op['support']}")

    print('\n=== 타입별 (strict) ===')
    for t, e in sorted(scored['per_entity'].items()):
        print(f"  {t:<12} P={e['precision']:.4f}  R={e['recall']:.4f}  "
              f"F1={e['f1']:.4f}  n={e['support']}")


if __name__ == '__main__':
    main()
