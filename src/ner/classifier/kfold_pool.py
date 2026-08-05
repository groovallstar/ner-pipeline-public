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
from typing import List, Optional, Tuple

from ner.metrics.span_metrics import (
    compute_offset_span_f1,
    compute_offset_span_f1_relaxed,
)
from ner.validity.comparability import RULER_FIELDS


def _run_ruler(fold_dirs: List[str]) -> Optional[dict]:
    """이 실행이 어떤 자로 쟀는지를 metrics.json 이 있는 첫 fold 에서 읽는다.

    **왜 pooled 로 옮겨 싣나.** 자(seed·stratify·데이터 지문)는 fold 별
    metrics.json 에만 적히는데 그 폴더는 휘발 scratch 라, 실행이 정리되면 나중에
    "이 두 실험이 같은 자로 쟀나" 를 물을 수단이 사라진다. 실제로 그렇게 잃은
    적이 있고, 그때 사후 감사가 두 시나리오를 병기하는 것으로 끝났다 — 어느
    쪽인지 고를 근거가 없었다. pooled 는 인용 근거로 승격되므로 여기 실으면
    자가 값과 함께 남는다.

    RULER_FIELDS 를 `validity` 에서 가져오는 이유는 자의 정의가 두 곳으로
    갈리지 않게 하기 위해서다 — 판정하는 쪽이 정본이다. 다만 판정은 여전히
    `fold*/metrics.json` 을 읽으므로(`validity.comparability.run_config`), 여기
    실리는 값은 **기록**이지 아직 판정 경로가 아니다.

    옛 실행에는 metrics.json 이 없거나 필드가 빠져 있을 수 있다. 하나도 못 읽으면
    None 이고, 일부만 있으면 **있는 것만** 싣는다 — 없는 것을 0 이나 기본값으로
    채우면 자를 모르는 실행이 아는 실행처럼 보인다. 그래서 **non-null 이라고 자를 다
    아는 것은 아니다**: `stratify` 나 `data_fingerprint` 가 빠진 부분 기록도 non-null
    이고, 하필 그 둘의 부재가 이 승격을 하게 만든 사건이다. 읽는 쪽은 값의 유무가
    아니라 필요한 키가 있는지를 봐야 한다.
    """
    # `metrics.json` 이 있는 첫 fold 를 대표로 쓴다 — RULER 는 run 전체에서
    # 상수라 어느 fold 든 같고, 없는 fold 는 건너뛴다.
    for fold_dir in fold_dirs:
        path = os.path.join(fold_dir, 'metrics.json')
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as fp:
            cfg = json.load(fp)
        present = {f: cfg[f] for f in RULER_FIELDS if f in cfg}
        return present or None
    return None


def _leak_mode(rec: dict, group_key: Optional[str]) -> str:
    """이 레코드를 어떤 모드로 볼지 정한다.

    'none' 은 행 단위 분할을 명시적으로 선택했다는 뜻(미측정), 'declared' 는
    그룹 키가 선언됐다는 뜻, 'legacy' 는 그룹 키를 모르는 옛 예측 파일이다.
    """
    if group_key == 'auto':
        recorded = rec.get('group_key')
        if recorded is None:
            return 'legacy'
        return 'none' if recorded == 'none' else 'declared'
    return 'none' if group_key is None else 'declared'


def _leak_basis(rec: dict, mode: str) -> Tuple[str, object]:
    """레코드에서 누출 판정에 쓸 (근거, 값) 을 고른다.

    우선순위: group(선언된 그룹 키의 값) > orig(레거시 원문) > text(문장 전체).
    'text' 는 문장을 재작성하는 증강 코퍼스에서 형제 행을 알아보지 못하므로
    신뢰 근거가 아니다 — 카운트는 나오지만 0 을 "이상 없음" 으로 읽어선 안 되고,
    판정 쪽(`validity`)이 근거를 보고 걸러야 한다.

    mode='none' 이면 근거는 'none' 이고 누출은 미측정이다. 행 단위 분할을
    명시적으로 선택했다는 뜻이라 0 으로 위장해선 안 된다.
    """
    if mode == 'none':
        return 'none', None
    if mode == 'declared':
        value = rec.get('group')
        if value is not None:
            return 'group', value
        # 그룹 키가 선언됐는데 값이 없다 → 옛 예측 파일. 아래로 물러선다.
    orig = rec.get('orig')
    if orig is not None:
        return 'orig', orig
    text = rec.get('text')
    if text is not None:
        return 'text', text
    return 'none', None


def pool_fold_predictions(fold_dirs: List[str],
                          require_no_leak: bool = True,
                          group_key: Optional[str] = 'auto') -> dict:
    """각 fold dir 의 test_predictions.json 을 합쳐 pooled span F1 계산.

    pooled = 전체 fold 의 test 문장을 하나의 코퍼스로 합친 micro-average
    (fold 별 F1 의 평균이 아님). 호출자는 fold_dirs 가 전체 fold 를 빠짐없이
    포함하는지 직접 확인해야 한다 — 누락된 fold 는 감지되지 않는다.

    cross-fold 누출 검증: 같은 그룹 값이 두 fold 의 test 에 걸쳐 등장하면
    누출이다. 같은 fold 안의 반복은 group 분할의 정상 동작이라 허용한다.
    판정 근거는 `leak_check_basis` 로 남긴다 — 카운트 0 이 "이상 없음" 인지
    "볼 수단이 없었음" 인지 구분하기 위해서다.

    require_no_leak=True 면 누출 발견 시 ValueError. 누출 baseline 을 일부러
    측정할 때만 False 로 내려 카운트만 돌려받는다.

    Args:
        fold_dirs: 각 fold 의 output_dir (안에 test_predictions.json 존재)
        require_no_leak: True 면 누출에 ValueError, False 면 카운트만.
        group_key: 그룹 키 필드명. 'auto'(기본) 면 예측 파일에 기록된
            `group_key` 를 쓰고, 없으면 레거시 orig/text 근거로 물러선다.
            None 이면 명시적 opt-out 이라 누출을 미측정(null)으로 남긴다.

    Returns:
        {"strict", "relaxed", "n_sentences", "n_folds",
         "group_key": str|None, "leak_check_basis": "group"|"orig"|"text"|"none",
         "cross_fold_group_dups": int|None,
         "cross_fold_orig_dups": int|None,   # 구 키 (하위호환, 같은 값)
         "ruler": dict|None}   # fold*/metrics.json 의 RULER_FIELDS 사본.
                               # 기록이 없는 옛 실행은 None (§_run_ruler)
    """
    gold_spans_list: List[List[dict]] = []
    pred_spans_list: List[List[dict]] = []
    seen_fold: dict = {}  # 그룹 값 -> 처음 등장한 fold_dir
    dups = 0
    basis_seen: set = set()
    declared: Optional[str] = None

    for fold_dir in fold_dirs:
        preds_path = os.path.join(fold_dir, 'test_predictions.json')
        with open(preds_path, encoding='utf-8') as f:
            records = json.load(f)
        for rec in records:
            mode = _leak_mode(rec, group_key)
            if mode == 'declared' and declared is None:
                declared = (rec.get('group_key') if group_key == 'auto'
                            else group_key)
            basis, value = _leak_basis(rec, mode)
            basis_seen.add(basis)
            if value is not None:
                prev = seen_fold.get(value)
                if prev is not None and prev != fold_dir:
                    dups += 1
                    if require_no_leak:
                        raise ValueError(
                            f'cross-fold sibling leak (in {preds_path}, '
                            f'basis={basis}): value seen in {prev} and '
                            f'{fold_dir}: {str(value)[:50]!r}'
                        )
                else:
                    seen_fold.setdefault(value, fold_dir)
            gold_spans_list.append(rec['gold_spans'])
            pred_spans_list.append(rec['pred_spans'])

    # 여러 근거가 섞였으면 가장 약한 것으로 보고한다 (보수적)
    basis = 'none'
    for weakest in ('none', 'text', 'orig', 'group'):
        if weakest in basis_seen:
            basis = weakest
            break

    # 미측정: 명시적 opt-out(none). 이 경우 0 이 아니라 null 을 남긴다.
    measured = basis != 'none'
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
        'group_key': declared,
        'leak_check_basis': basis,
        'cross_fold_group_dups': dups if measured else None,
        'cross_fold_orig_dups': dups if measured else None,
        # 비교 가능성 지문 — fold 폴더가 지워져도 자가 값과 함께 남는다.
        'ruler': _run_ruler(fold_dirs),
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
        help='Do not raise on cross-fold sibling leak; only count it '
             '(cross_fold_group_dups). Use to measure a leaked baseline.',
    )
    parser.add_argument(
        '--group-key', default='auto',
        help='Field holding the sibling-group value. Default "auto" reads '
             'the group_key recorded in test_predictions.json. Pass "none" '
             'to report the leak counter as unmeasured (null) instead of 0.',
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        parent = os.path.dirname(os.path.normpath(args.fold_dirs[0]))
        output = os.path.join(parent, 'pooled_metrics.json')

    group_key = None if args.group_key == 'none' else args.group_key
    result = pool_fold_predictions(
        args.fold_dirs, require_no_leak=not args.allow_cross_fold_leak,
        group_key=group_key,
    )
    dups = result['cross_fold_group_dups']
    print(f"\n{'='*72}")
    print(f"  Pooled K-fold span F1  "
          f"(n_folds={result['n_folds']}, "
          f"n_sentences={result['n_sentences']}, "
          f"cross_fold_group_dups={'unmeasured' if dups is None else dups}, "
          f"basis={result['leak_check_basis']})")
    if result['leak_check_basis'] in ('text', 'none'):
        print("  WARNING: leak check is not trustworthy "
              f"(basis={result['leak_check_basis']}). A count of 0 here means "
              "'not observable', not 'no leak'.")
    print(f"{'='*72}")
    _print_metrics_block('strict', result['strict'])
    _print_metrics_block("relaxed (SemEval'13 Partial)", result['relaxed'])

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {output}")


if __name__ == '__main__':
    main()
