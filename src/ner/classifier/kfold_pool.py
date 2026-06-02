"""K-fold pooled 평가 CLI.

각 fold output_dir 의 test_predictions.json 을 모아, 전체 fold 의 gold/pred
span 을 합친 뒤 strict + relaxed span F1 을 한 번에 계산한다 (pooled).

사용 예:
    python -m ner.classifier.kfold_pool \
        --fold-dirs results/.../fold0 results/.../fold1 ... \
        --output results/.../pooled_metrics.json
"""

import argparse
import json
import os
from typing import List

from ner.metrics.span_metrics import (
    compute_offset_span_f1,
    compute_offset_span_f1_relaxed,
)


def pool_fold_predictions(fold_dirs: List[str]) -> dict:
    """각 fold dir 의 test_predictions.json 을 합쳐 pooled span F1 계산.

    pooled = 전체 fold 의 test 문장을 하나의 코퍼스로 합친 micro-average
    (fold 별 F1 의 평균이 아님). 호출자는 fold_dirs 가 전체 fold 를 빠짐없이
    포함하는지 직접 확인해야 한다 — 누락된 fold 는 감지되지 않는다.

    무결성 검증: fold 간 중복 text 가 있으면 ValueError. 같은 문장이 두 번
    test 되는 것은 분할 오류 또는 데이터 중복 (leak) 신호다. id 는 원본
    데이터셋에서 고유하지 않을 수 있으므로 (한 원문에서 파생된 여러 행이
    id 를 공유) 검증 기준으로 쓰지 않는다.

    Args:
        fold_dirs: 각 fold 의 output_dir (안에 test_predictions.json 존재)

    Returns:
        {"strict": {overall, per_entity},
         "relaxed": {overall, per_entity},
         "n_sentences": int, "n_folds": int}
    """
    gold_spans_list: List[List[dict]] = []
    pred_spans_list: List[List[dict]] = []
    seen_texts: set = set()

    for fold_dir in fold_dirs:
        preds_path = os.path.join(fold_dir, 'test_predictions.json')
        with open(preds_path, encoding='utf-8') as f:
            records = json.load(f)
        for rec in records:
            text = rec.get('text')
            if text is not None:
                if text in seen_texts:
                    raise ValueError(
                        f'duplicate test sentence across folds '
                        f'(in {preds_path}): {text[:50]!r}'
                    )
                seen_texts.add(text)
            gold_spans_list.append(rec['gold_spans'])
            pred_spans_list.append(rec['pred_spans'])

    n_sentences = len(gold_spans_list)
    strict = compute_offset_span_f1(gold_spans_list, pred_spans_list)
    relaxed = compute_offset_span_f1_relaxed(
        gold_spans_list, pred_spans_list
    )
    return {
        'strict': strict,
        'relaxed': relaxed,
        'n_sentences': n_sentences,
        'n_folds': len(fold_dirs),
    }


def _print_metrics_block(label: str, m: dict) -> None:
    """strict / relaxed 메트릭 블록 콘솔 출력."""
    o = m['overall']
    print(f"\n  [{label}]")
    print(f"  F1={o['f1']:.4f}  Precision={o['precision']:.4f}  "
          f"Recall={o['recall']:.4f}  Support={o['support']}")
    print(f"  {'Entity':<14} {'F1':>8} {'Prec':>8} {'Recall':>8} "
          f"{'Support':>8}")
    print(f"  {'-'*52}")
    for etype, em in sorted(m['per_entity'].items(),
                            key=lambda x: -x[1]['f1']):
        print(f"  {etype:<14} {em['f1']:>8.4f} {em['precision']:>8.4f} "
              f"{em['recall']:>8.4f} {em['support']:>8}")


def main():
    parser = argparse.ArgumentParser(
        description='Pooled span F1 over K-fold test predictions'
    )
    parser.add_argument(
        '--fold-dirs', nargs='+', required=True,
        help='Fold output dirs, each containing test_predictions.json',
    )
    parser.add_argument(
        '--output', default=None,
        help='Output JSON path (default: pooled_metrics.json in the '
             'parent dir of the first fold dir)',
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        parent = os.path.dirname(os.path.normpath(args.fold_dirs[0]))
        output = os.path.join(parent, 'pooled_metrics.json')

    result = pool_fold_predictions(args.fold_dirs)
    print(f"\n{'='*72}")
    print(f"  Pooled K-fold span F1  "
          f"(n_folds={result['n_folds']}, "
          f"n_sentences={result['n_sentences']})")
    print(f"{'='*72}")
    _print_metrics_block('strict', result['strict'])
    _print_metrics_block("relaxed (SemEval'13 Partial)", result['relaxed'])

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {output}")


if __name__ == '__main__':
    main()
