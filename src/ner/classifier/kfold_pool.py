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


def pool_fold_predictions(fold_dirs: List[str],
                          require_no_leak: bool = True) -> dict:
    """각 fold dir 의 test_predictions.json 을 합쳐 pooled span F1 계산.

    pooled = 전체 fold 의 test 문장을 하나의 코퍼스로 합친 micro-average
    (fold 별 F1 의 평균이 아님). 호출자는 fold_dirs 가 전체 fold 를 빠짐없이
    포함하는지 직접 확인해야 한다 — 누락된 fold 는 감지되지 않는다.

    무결성 검증(두 축):
      - 원문(orig) 단위 cross-fold 누출: record 에 'orig'(주입 전 원문)이
        있으면 같은 원문이 두 fold 의 test 에 걸쳐 등장하는지 본다. 같은
        fold 안에서 같은 원문이 여러 번 나오는 것은 group 분할의 정상 동작
        이라 허용하고, fold 가 갈리면 누출로 본다.
      - 레거시 전체문장 중복: 'orig' 가 없는 코퍼스는 'text' 전체로 fold 간
        중복을 본다 (예전 동작).
    require_no_leak=True 면 누출 발견 시 ValueError. 누출 baseline 을 일부러
    측정할 때만 False 로 내려 ValueError 대신 cross_fold_orig_dups 카운트만
    돌려받는다. id 는 원본 데이터셋에서 비고유(한 원문 파생 다행 공유)라
    검증 기준으로 쓰지 않는다.

    Args:
        fold_dirs: 각 fold 의 output_dir (안에 test_predictions.json 존재)
        require_no_leak: True 면 누출에 ValueError, False 면 카운트만.

    Returns:
        {"strict": {overall, per_entity},
         "relaxed": {overall, per_entity},
         "n_sentences": int, "n_folds": int,
         "cross_fold_orig_dups": int}
    """
    gold_spans_list: List[List[dict]] = []
    pred_spans_list: List[List[dict]] = []
    seen_texts: set = set()
    orig_fold: dict = {}  # orig -> 처음 등장한 fold_dir
    cross_fold_orig_dups = 0

    for fold_dir in fold_dirs:
        preds_path = os.path.join(fold_dir, 'test_predictions.json')
        with open(preds_path, encoding='utf-8') as f:
            records = json.load(f)
        for rec in records:
            orig = rec.get('orig')
            if orig is not None:
                prev = orig_fold.get(orig)
                if prev is not None and prev != fold_dir:
                    cross_fold_orig_dups += 1
                    if require_no_leak:
                        raise ValueError(
                            f'cross-fold original-text leak (in {preds_path}): '
                            f'orig in {prev} and {fold_dir}: {orig[:50]!r}'
                        )
                else:
                    orig_fold.setdefault(orig, fold_dir)
            else:
                text = rec.get('text')
                if text is not None:
                    if text in seen_texts:
                        if require_no_leak:
                            raise ValueError(
                                f'duplicate test sentence across folds '
                                f'(in {preds_path}): {text[:50]!r}'
                            )
                    else:
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
        'cross_fold_orig_dups': cross_fold_orig_dups,
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
    parser.add_argument(
        '--allow-cross-fold-leak', action='store_true',
        help='Do not raise on cross-fold original-text leak; only count it '
             '(cross_fold_orig_dups). Use to measure a leaked baseline.',
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        parent = os.path.dirname(os.path.normpath(args.fold_dirs[0]))
        output = os.path.join(parent, 'pooled_metrics.json')

    result = pool_fold_predictions(
        args.fold_dirs, require_no_leak=not args.allow_cross_fold_leak
    )
    print(f"\n{'='*72}")
    print(f"  Pooled K-fold span F1  "
          f"(n_folds={result['n_folds']}, "
          f"n_sentences={result['n_sentences']}, "
          f"cross_fold_orig_dups={result['cross_fold_orig_dups']})")
    print(f"{'='*72}")
    _print_metrics_block('strict', result['strict'])
    _print_metrics_block("relaxed (SemEval'13 Partial)", result['relaxed'])

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {output}")


if __name__ == '__main__':
    main()
