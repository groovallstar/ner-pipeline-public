"""VI PROD 자연-코퍼스 re-silver — 배포 text 직접 재라벨 + additive PROD 회복.

#108 의 suffix 전용 `resilver_vi_isolated.py`(경계 로직)와 달리, 자연 주입
코퍼스(#116, `pii_all.jsonl`)는 배포 문장 자체를 Gemma+Qwen 으로 재라벨하고
`recall_strict_prod` 병합의 PROD(high+gemma_only)만 **기존 엔티티에 겹치지
않게 additive 로 추가**한다. 기존 PII·PER·LOC·ORG·EVT 는 불변 — 정밀도
헤드룸(창작물 PROD silver 누락)만 겨냥하고 EVT/PII 회귀를 구조적으로 차단.

모드:
  relabel  배포 text(고유) → 새 §3 프롬프트 재라벨 (vLLM)          [GPU 필요]
  apply    gemma+qwen 재라벨 → recall_strict_prod PROD additive 추가 → 새 gold
           + 추가분 diff(spot-audit 용)

사용:
    python src/ner/augmenters/wikiann_vi/resilver_vi_natural.py relabel \
        --pii data/wikiann_vi/pii_all.jsonl \
        --base-url http://localhost:8081/v1 \
        --model cyankiwi/gemma-4-31B-it-AWQ-8bit \
        --out results/classifier/vi/audit/natural_relabel_gemma.jsonl
    python src/ner/augmenters/wikiann_vi/resilver_vi_natural.py apply \
        --pii data/wikiann_vi/pii_all.jsonl \
        --gemma ...gemma.jsonl --qwen ...qwen.jsonl \
        --out data/wikiann_vi/pii_all_prodrecover.jsonl \
        --added-out results/classifier/vi/audit/added_prod.jsonl
"""
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

from ner.augmenters.wikiann_vi.merge_confidence import (
    _filter_by_policy, categorize_spans,
)


def _load_jsonl(path: str) -> List[dict]:
    return [json.loads(ln) for ln in open(path, encoding='utf-8')]


def _write_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """두 char 구간 [start, end) 가 겹치면 True."""
    return a[0] < b[1] and b[0] < a[1]


def apply_recovery(
    pii: List[dict],
    gemma_map: Dict[str, List[dict]],
    qwen_map: Dict[str, List[dict]],
    policy: str = 'recall_strict_prod',
    keep_surfaces: Optional[set] = None,
) -> Tuple[List[dict], List[dict]]:
    """기존 엔티티에 겹치지 않는 PROD span 만 additive 추가.

    반환: (새 gold 행 리스트, 추가된 PROD diff 리스트).
    gemma/qwen 재라벨을 categorize → policy 필터 → PROD 만 추출,
    기존 엔티티/이미 추가분과 offset 겹침·정확중복은 skip. keep_surfaces 가
    주어지면 그 표면형 PROD 만 추가(Wikidata 종-FP 필터). 원본 행 필드
    (orig 등)는 보존 — leak-free group-kfold(--group-key orig) 유지에 필수.
    """
    new_rows: List[dict] = []
    added: List[dict] = []
    for r in pii:
        text = r['text']
        entities = list(r.get('entities', []))
        existing = [(int(e['start_char']), int(e['end_char']))
                    for e in entities]
        cat = categorize_spans(
            gemma_map.get(text, []), qwen_map.get(text, []))
        prod = [s for s in _filter_by_policy(cat, policy)
                if s['type'] == 'PROD']
        for s in prod:
            surface = text[int(s['start']):int(s['end'])]
            if keep_surfaces is not None and surface not in keep_surfaces:
                continue
            span = (int(s['start']), int(s['end']))
            if any(_overlaps(span, e) for e in existing):
                continue
            new_ent = {
                'label': 'PROD',
                'start_char': span[0],
                'end_char': span[1],
                'text': surface,
            }
            entities.append(new_ent)
            existing.append(span)
            added.append({
                'id': r.get('id'), 'text': text,
                'surface': surface,
                'start': span[0], 'end': span[1],
                'confidence': s.get('confidence'),
            })
        entities.sort(key=lambda e: int(e['start_char']))
        new_rows.append({**r, 'entities': entities})
    return new_rows, added


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)

    pr = sub.add_parser('relabel', help='Relabel deployed texts via vLLM')
    pr.add_argument('--pii', required=True)
    pr.add_argument('--out', required=True)
    pr.add_argument('--base-url', default='http://localhost:8081/v1')
    pr.add_argument('--model', default='cyankiwi/gemma-4-31B-it-AWQ-8bit')
    pr.add_argument('--concurrency', type=int, default=24)
    pr.add_argument('--max-tokens', type=int, default=2048)
    pr.add_argument('--limit', type=int, default=0,
                    help='Relabel only first N unique texts (0=all, debug)')

    pa = sub.add_parser('apply', help='Additive PROD recovery into gold')
    pa.add_argument('--pii', required=True)
    pa.add_argument('--gemma', required=True)
    pa.add_argument('--qwen', required=True)
    pa.add_argument('--policy', default='recall_strict_prod')
    pa.add_argument('--keep-surfaces', default=None,
                    help='JSON list of surfaces to keep (Wikidata filter)')
    pa.add_argument('--out', required=True)
    pa.add_argument('--added-out', required=True)

    args = p.parse_args()

    if args.cmd == 'relabel':
        from ner.augmenters.wikiann_vi.relabel_8type import Relabeler
        pii = _load_jsonl(args.pii)
        uniq = sorted({r['text'] for r in pii if r.get('text')})
        if args.limit:
            uniq = uniq[:args.limit]
        records = [{'id': str(i), 'text': t} for i, t in enumerate(uniq)]
        relabeler = Relabeler(
            base_url=args.base_url, model=args.model,
            max_tokens=args.max_tokens, concurrency=args.concurrency)
        print(f'relabeling {len(records)} unique texts @ {args.model}')
        out = relabeler.relabel_sync(records)
        _write_jsonl(args.out, [
            {'text': t, 'spans': rec.get('gold_spans_8type', [])}
            for rec, t in zip(out, uniq)
        ])
        n_prod = sum(
            1 for rec in out
            for s in rec.get('gold_spans_8type', []) if s['type'] == 'PROD')
        print(f'wrote {len(out)} relabeled -> {args.out} '
              f'(PROD spans={n_prod})')

    elif args.cmd == 'apply':
        pii = _load_jsonl(args.pii)
        gemma = {r['text']: r['spans'] for r in _load_jsonl(args.gemma)}
        qwen = {r['text']: r['spans'] for r in _load_jsonl(args.qwen)}
        keep = None
        if args.keep_surfaces:
            with open(args.keep_surfaces, encoding='utf-8') as f:
                keep = set(json.load(f))
        new_rows, added = apply_recovery(
            pii, gemma, qwen, args.policy, keep_surfaces=keep)
        _write_jsonl(args.out, new_rows)
        _write_jsonl(args.added_out, added)
        n_old = sum(1 for r in pii for e in r.get('entities', [])
                    if (e.get('label') or e.get('type')) == 'PROD')
        n_new = sum(1 for r in new_rows for e in r['entities']
                    if e['label'] == 'PROD')
        print(f'rows={len(new_rows)} PROD gold {n_old} -> {n_new} '
              f'(+{len(added)} added) -> {args.out}')
        print(f'added PROD diff -> {args.added_out}')


if __name__ == '__main__':
    main()
