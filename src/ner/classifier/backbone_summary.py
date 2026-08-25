"""백본 벤치마크 집계 — seed 여러 번 돌린 결과를 후보별 한 줄로 접는다.

사용 예::

    python -m ner.classifier.backbone_summary \
        --runs-dir results/classifier/en_bench \
        --timing-dir results/classifier/en_bench/timing \
        --output results/classifier/en_bench/median_summary.json

## 왜 median 인가

백본끼리 F1 이 1~2pp 로 붙으면 seed 한 번의 순위는 실력이 아니라 추첨을
적는 것이 된다. 그래서 같은 후보를 seed 를 바꿔 여러 번 돌리고 가운데 값으로
순위를 매긴다 — 평균이 아니라 median 인 것은 한 seed 가 크게 튀었을 때 그
이상치가 순위를 끌지 못하게 하기 위해서다. 폭이 순위 간격보다 큰지 읽을 수
있도록 표준편차와 최소·최대를 함께 낸다.

## 왜 NER-5 만 보나

PII 5 종(`DAT`·`EMAIL`·`PHONE`·`ID_NUM`·`CREDIT_CARD`)은 합성 주입값이라
패턴이 인위적으로 깨끗해 대부분 포화한다. 백본 변별력이 없어 선정 기준에서
빼고 참고값으로만 낸다.

## micro 를 다시 세는 방법

`metrics.json` 은 타입별 P·R·support 만 주므로 NER-5 만의 micro 를 다시
세려면 맞은 개수를 되짚어야 한다: 맞은 것 = R × support, 놓친 것 = support -
맞은 것, 잘못 부른 것 = 맞은 것 × (1-P) / P. P 가 0 인 타입은 마지막 식이
0 으로 나누기라 잘못 부른 개수를 복원할 수 없다 — 그 타입 이름을
`unknown_fp_types` 에 남겨 micro precision 이 낙관적으로 나왔음을 표시한다.

## 산출물이 자기 로그와 맞는지

`--check` 를 주면 집계 전에 run 마다 시각 정합을 본다. 두 프로세스가 같은
출력 디렉토리를 쓰면 나중 것이 앞 것의 결과를 덮거나, 앞 것이 남긴 결과가
나중 것의 로그와 짝이 안 맞는 상태로 남는데, 값만 조용히 바뀌므로 눈으로는
안 보인다. 로그의 첫 시각 + `train_time_sec` 은 `metrics.json` 이 쓰인 시각과
맞아야 하고, 그 사이에는 평가와 체크포인트 저장만 있으므로 차이는 작은
양수여야 한다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ner.classifier.data_utils import NER_TYPES, PII_TYPES

# 평가 + best 저장에 드는 여유(초). 넘거나 음수면 그 결과는 그 로그가 남긴
# 것이 아니다.
GAP_MIN, GAP_MAX = 0.0, 400.0


def micro(per_entity: dict, types: Sequence[str]) -> Tuple[
    float, float, float, int, List[str]
]:
    """타입별 P·R·support 에서 그 묶음만의 micro P/R/F1 을 다시 센다."""
    tp = fp = fn = 0.0
    unknown_fp: List[str] = []
    for label in types:
        m = per_entity.get(label)
        if not m:
            continue
        support, prec, rec = m['support'], m['precision'], m['recall']
        hit = rec * support
        tp += hit
        fn += support - hit
        if prec > 0:
            fp += hit * (1 - prec) / prec
        elif support > 0:
            unknown_fp.append(label)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return f1, p, r, int(round(tp + fn)), unknown_fp


def macro(per_entity: dict, types: Sequence[str]) -> float:
    vals = [per_entity[t]['f1'] for t in types if t in per_entity]
    return sum(vals) / len(vals) if vals else 0.0


def _log_start(log_path: Path) -> Optional[dt.datetime]:
    if not log_path.exists():
        return None
    with log_path.open(errors='ignore') as fh:
        for line in fh:
            m = re.match(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', line)
            if m:
                return dt.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
    return None


def check_runs(runs_dir: Path, pattern: str) -> List[dict]:
    """run 마다 산출물이 자기 로그와 시각으로 맞는지 본다."""
    out = []
    for run in sorted(runs_dir.glob(pattern)):
        metrics = run / 'metrics.json'
        row = {'run': run.name, 'gap_sec': None, 'ok': False,
               'reason': None}
        if not metrics.exists():
            row['reason'] = 'metrics.json missing'
            out.append(row)
            continue
        start = _log_start(runs_dir / 'logs' / f'{run.name}.log')
        if start is None:
            row['reason'] = 'no log timestamp'
            out.append(row)
            continue
        train = json.loads(metrics.read_text()).get('train_time_sec') or 0
        written = dt.datetime.fromtimestamp(metrics.stat().st_mtime)
        gap = (written - (start + dt.timedelta(seconds=train))).total_seconds()
        row['gap_sec'] = round(gap, 1)
        row['ok'] = GAP_MIN <= gap <= GAP_MAX
        if not row['ok']:
            row['reason'] = 'result does not match its own log'
        out.append(row)
    return out


def load_timing(timing_dir: Optional[Path]) -> Dict[str, float]:
    """모델별 1-epoch 독점 학습시간. 없으면 빈 표."""
    if timing_dir is None or not timing_dir.exists():
        return {}
    out = {}
    for run in sorted(timing_dir.iterdir()):
        metrics = run / 'metrics.json'
        if not metrics.exists():
            continue
        d = json.loads(metrics.read_text())
        out[d['model_name']] = d.get('train_time_sec')
    return out


def summarize(runs_dir: Path, pattern: str,
              timing: Dict[str, float]) -> List[dict]:
    by_model = defaultdict(list)
    fingerprints = set()
    for run in sorted(runs_dir.glob(pattern)):
        metrics = run / 'metrics.json'
        if not metrics.exists():
            continue
        d = json.loads(metrics.read_text())
        pe = d['per_entity']
        f1, p, r, support, unknown = micro(pe, NER_TYPES)
        fingerprints.add(d.get('data_fingerprint'))
        by_model[d['model_name']].append({
            'run': run.name,
            'train_seed': d.get('train_seed'),
            'ner5_micro_f1': f1,
            'ner5_micro_precision': p,
            'ner5_micro_recall': r,
            'ner5_macro_f1': macro(pe, NER_TYPES),
            'ner5_support': support,
            'pii5_micro_f1': micro(pe, PII_TYPES)[0],
            'overall10_f1': d['overall']['f1'],
            'per_type_f1': {t: pe.get(t, {}).get('f1') for t in NER_TYPES},
            'unknown_fp_types': unknown,
        })

    rows = []
    for model, runs in by_model.items():
        runs.sort(key=lambda x: x['train_seed'] or 0)
        micros = [x['ner5_micro_f1'] for x in runs]
        macros = [x['ner5_macro_f1'] for x in runs]
        rows.append({
            'model_name': model,
            'n_seeds': len(runs),
            'seeds': [x['train_seed'] for x in runs],
            'ner5_micro_median': st.median(micros),
            'ner5_micro_std': st.stdev(micros) if len(micros) > 1 else 0.0,
            'ner5_micro_min': min(micros),
            'ner5_micro_max': max(micros),
            'ner5_macro_median': st.median(macros),
            'pii5_micro_median': st.median(
                [x['pii5_micro_f1'] for x in runs]
            ),
            'per_type_median': {
                t: st.median([x['per_type_f1'][t] for x in runs
                              if x['per_type_f1'][t] is not None])
                for t in NER_TYPES
                if any(x['per_type_f1'][t] is not None for x in runs)
            },
            'epoch_train_sec_exclusive': timing.get(model),
            'runs': runs,
        })
    rows.sort(key=lambda x: -x['ner5_micro_median'])
    return rows, sorted(f for f in fingerprints if f)


def print_table(rows: List[dict]) -> None:
    print(f"{'model':<38} {'n':>2} {'median':>8} {'std':>7} "
          f"{'min':>8} {'max':>8} {'macro':>8} {'1ep_s':>7}  "
          f"{'/'.join(NER_TYPES)} (median)")
    for x in rows:
        per_type = ' '.join(
            f"{x['per_type_median'].get(t, float('nan')):.3f}"
            for t in NER_TYPES
        )
        secs = x['epoch_train_sec_exclusive']
        print(f"{x['model_name']:<38} {x['n_seeds']:>2} "
              f"{x['ner5_micro_median']:>8.4f} {x['ner5_micro_std']:>7.4f} "
              f"{x['ner5_micro_min']:>8.4f} {x['ner5_micro_max']:>8.4f} "
              f"{x['ner5_macro_median']:>8.4f} "
              f"{(f'{secs:.0f}' if secs else '-'):>7}  {per_type}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m ner.classifier.backbone_summary',
        description='Fold multi-seed backbone runs into one row per '
                    'candidate (NER-5 median, std, range).',
    )
    p.add_argument('--runs-dir', type=Path, required=True,
                   help='Directory holding <run>/metrics.json')
    p.add_argument('--pattern', default='*_seed*',
                   help='Glob for run directories under --runs-dir')
    p.add_argument('--timing-dir', type=Path, default=None,
                   help='Directory of exclusive 1-epoch timing runs')
    p.add_argument('--output', type=Path, default=None,
                   help='Write the summary JSON here')
    p.add_argument('--check', action='store_true',
                   help='Verify each run against its own log before '
                        'summarizing; exit non-zero on mismatch')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    checks = None
    if args.check:
        checks = check_runs(args.runs_dir, args.pattern)
        print(f"{'run':<30} {'gap_s':>8}  verdict")
        for c in checks:
            gap = '-' if c['gap_sec'] is None else f"{c['gap_sec']:.0f}"
            verdict = 'ok' if c['ok'] else f"MISMATCH ({c['reason']})"
            print(f"{c['run']:<30} {gap:>8}  {verdict}")
        bad = [c['run'] for c in checks if not c['ok']]
        if bad:
            print(f'\n{len(bad)} run(s) failed the check: {", ".join(bad)}')
            return 1
        print('\nall runs consistent with their logs\n')

    timing = load_timing(args.timing_dir)
    rows, fingerprints = summarize(args.runs_dir, args.pattern, timing)
    print_table(rows)
    print(f'\ndata fingerprints: {fingerprints}')

    if args.output:
        payload = {
            'data_fingerprints': fingerprints,
            'integrity_checks': checks,
            'candidates': rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
        print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
