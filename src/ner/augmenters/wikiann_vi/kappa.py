"""두 모델 재라벨 결과 간 span-level Cohen's kappa 측정 유틸.

각 레코드에서 두 모델이 라벨한 (start, end) span의 **합집합**을 만들고,
offset별로 모델 A와 모델 B의 타입을 쌍으로 수집한다. 한쪽만 라벨한 span은
없는 쪽을 `O`로 간주해 pair에 포함한다. 이렇게 얻은 pair 리스트에 대해
Cohen's kappa, agreement ratio, per-type 일치율을 계산한다.

검증 레이어의 cross-model agreement 지표를 담당한다.
상세 기준: `docs/manual/data/vietnamese-ner.md` §3.
"""
import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

_ABSENT = 'O'


def _load_jsonl(path: Path) -> List[dict]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _to_span_map(spans: List[dict]) -> Dict[Tuple[int, int], str]:
    """(start, end) → type 매핑. 중복 오프셋은 마지막이 승리."""
    return {
        (int(s['start']), int(s['end'])): s['type']
        for s in spans
        if 'start' in s and 'end' in s and 'type' in s
    }


def _collect_pairs(
    records_a: List[dict], records_b: List[dict],
    span_key: str = 'gold_spans_relabel',
) -> Tuple[List[str], List[str], int]:
    """두 JSONL에서 동일 id 레코드의 span 오프셋 합집합 기반 pair 수집.

    Returns:
        (labels_a, labels_b, common_record_count)
    """
    a_by_id = {str(r['id']): r for r in records_a}
    b_by_id = {str(r['id']): r for r in records_b}
    common_ids = sorted(set(a_by_id) & set(b_by_id))

    labels_a: List[str] = []
    labels_b: List[str] = []
    for rid in common_ids:
        a_map = _to_span_map(a_by_id[rid].get(span_key, []))
        b_map = _to_span_map(b_by_id[rid].get(span_key, []))
        keys = sorted(set(a_map) | set(b_map))
        for key in keys:
            labels_a.append(a_map.get(key, _ABSENT))
            labels_b.append(b_map.get(key, _ABSENT))
    return labels_a, labels_b, len(common_ids)


def cohen_kappa(labels_a: List[str], labels_b: List[str]) -> Dict:
    """Cohen's kappa (categorical, unweighted).

    kappa = (po - pe) / (1 - pe)
    - po = observed agreement ratio
    - pe = chance agreement (마진 확률의 내적)
    """
    n = len(labels_a)
    if n == 0:
        return {'kappa': float('nan'), 'po': float('nan'),
                'pe': float('nan'), 'n': 0}
    if len(labels_a) != len(labels_b):
        raise ValueError('label length mismatch')

    agreed = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = agreed / n

    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    classes = set(count_a) | set(count_b)
    pe = sum(
        (count_a[c] / n) * (count_b[c] / n) for c in classes
    )

    if abs(1 - pe) < 1e-12:
        kappa = 0.0
    else:
        kappa = (po - pe) / (1 - pe)
    return {'kappa': kappa, 'po': po, 'pe': pe, 'n': n}


def confusion_matrix(
    labels_a: List[str], labels_b: List[str],
) -> Dict[str, Dict[str, int]]:
    """type(A) × type(B) 혼동 행렬. 중첩 dict로 반환."""
    matrix: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for a, b in zip(labels_a, labels_b):
        matrix[a][b] += 1
    return {k: dict(v) for k, v in matrix.items()}


def per_type_agreement(
    labels_a: List[str], labels_b: List[str],
) -> Dict[str, Dict[str, float]]:
    """타입별 일치율 (agreement ratio when A == type).

    type == 'O' 제외. 한쪽만 라벨한 경우(O 포함)는 불일치로 계산된다.
    """
    types_a = Counter(labels_a)
    agree_by_type: Counter[str] = Counter()
    for a, b in zip(labels_a, labels_b):
        if a == b and a != _ABSENT:
            agree_by_type[a] += 1
    results: Dict[str, Dict[str, float]] = {}
    for type_name, total_a in types_a.items():
        if type_name == _ABSENT or total_a == 0:
            continue
        ratio = agree_by_type[type_name] / total_a
        results[type_name] = {
            'agreed': agree_by_type[type_name],
            'total_a': total_a,
            'ratio': ratio,
        }
    return results


def compute(
    records_a: List[dict], records_b: List[dict],
    span_key: str = 'gold_spans_relabel',
) -> Dict:
    """재라벨 JSONL 2종에 대해 kappa·혼동 행렬·per-type을 한 번에 계산."""
    labels_a, labels_b, common_n = _collect_pairs(
        records_a, records_b, span_key=span_key,
    )
    return {
        'common_records': common_n,
        'paired_spans': len(labels_a),
        **cohen_kappa(labels_a, labels_b),
        'per_type_agreement': per_type_agreement(labels_a, labels_b),
        'confusion_matrix': confusion_matrix(labels_a, labels_b),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m ner.augmenters.wikiann_vi.kappa',
        description='Compute cross-model Cohen kappa on 5-type relabel',
    )
    parser.add_argument('--a', required=True, help='JSONL path for model A')
    parser.add_argument('--b', required=True, help='JSONL path for model B')
    parser.add_argument(
        '--span-key', default='gold_spans_relabel',
        help='Field name holding 5-type spans',
    )
    parser.add_argument(
        '--json-out', default=None,
        help='Optional path to write full result as JSON',
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    records_a = _load_jsonl(Path(args.a))
    records_b = _load_jsonl(Path(args.b))
    result = compute(records_a, records_b, span_key=args.span_key)

    print(f'A: {args.a} ({len(records_a)} records)')
    print(f'B: {args.b} ({len(records_b)} records)')
    print(f"Common records: {result['common_records']}")
    print(f"Paired spans (A∪B offsets): {result['paired_spans']}")
    print(
        f"Cohen kappa: {result['kappa']:.4f} "
        f"(po={result['po']:.4f} pe={result['pe']:.4f})"
    )
    print("Per-type agreement (A-side perspective):")
    per = result['per_type_agreement']
    for t in sorted(per, key=lambda k: -per[k]['total_a']):
        d = per[t]
        print(
            f"  {t}: {d['agreed']}/{d['total_a']} = {d['ratio']:.4f}"
        )

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Wrote full result -> {args.json_out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
