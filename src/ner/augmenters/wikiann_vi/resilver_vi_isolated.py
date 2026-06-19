"""VI PROD/EVT re-silver (isolated relabel) — #108 remediation 효과 측정.

새 §3 프롬프트로 *원문 NER 라벨만* 재생성하고, 주입 PII/PER/LOC·텍스트·
5-fold split 은 고정해 프롬프트 효과를 격리한다(변수 = 프롬프트 하나).

원문 복원: all.jsonl prefix 매칭(주) → 주입 marker 경계(보조) → 전체 텍스트.
원문 영역 [0, boundary) 의 NER 만 교체하고, 경계 밖 주입 entity 는 보존한다.
검증으로 경계를 가로지르는 entity 0건 확인됨(suffix 주입 구조).

모드:
  build-source  pii_all + all.jsonl → 행별 (원문, 경계, 보존 entity) JSONL
  relabel       고유 원문 → 새 §3 프롬프트 재라벨 (vLLM)        [GPU 필요]
  assemble      재라벨 + 보존 entity overlay → pii_all_v2.jsonl

사용:
    python src/ner/augmenters/wikiann_vi/resilver_vi_isolated.py build-source \
        --pii data/wikiann_vi/pii_all.jsonl --all data/wikiann_vi/all.jsonl \
        --out results/classifier/vi/audit/resilver_source.jsonl
    python src/ner/augmenters/wikiann_vi/resilver_vi_isolated.py relabel \
        --source results/classifier/vi/audit/resilver_source.jsonl \
        --out results/classifier/vi/audit/resilver_relabel.jsonl
    python src/ner/augmenters/wikiann_vi/resilver_vi_isolated.py assemble \
        --pii data/wikiann_vi/pii_all.jsonl \
        --source results/classifier/vi/audit/resilver_source.jsonl \
        --relabel results/classifier/vi/audit/resilver_relabel.jsonl \
        --out data/wikiann_vi/pii_all_v2.jsonl
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

NER_TYPES = ('PER', 'LOC', 'ORG', 'PROD', 'EVT')
# suffix 주입 scaffolding marker (augmenters/pii/injector.py SUFFIX_TEMPLATES)
MARKERS = (' Liên hệ:', ' SĐT:', ' Địa chỉ:', ' Ngày:', ' CCCD:',
           ' Email:', ' Thẻ:')


def _load_jsonl(path: str) -> List[dict]:
    return [json.loads(ln) for ln in open(path, encoding='utf-8')]


def _build_prefix_index(all_rows: List[dict]) -> Dict[str, List[str]]:
    """all.jsonl 텍스트를 첫 12자 prefix 로 버킷팅(longest-prefix 매칭용)."""
    idx: Dict[str, List[str]] = defaultdict(list)
    for t in {r['text'] for r in all_rows}:
        idx[t[:12]].append(t)
    return idx


def _recover_original(text: str, idx: Dict[str, List[str]]) -> str:
    """원문(주입 전 문장) 복원 — all.jsonl prefix(주) → marker(보조) → 전체."""
    best: Optional[str] = None
    for t in idx.get(text[:12], []):
        if text.startswith(t) and (best is None or len(t) > len(best)):
            best = t
    if best is not None:
        return best
    positions = [text.find(m) for m in MARKERS if text.find(m) > 0]
    if positions:
        return text[:min(positions)]
    return text


def build_source(pii: List[dict], all_rows: List[dict]) -> List[dict]:
    """행별 원문·경계·보존 entity(주입분) 산출. 경계 가로지름은 보고."""
    idx = _build_prefix_index(all_rows)
    out: List[dict] = []
    crossing = 0
    for i, r in enumerate(pii):
        text = r['text']
        original = _recover_original(text, idx)
        boundary = len(original)
        kept = [e for e in r['entities'] if e['start_char'] >= boundary]
        cross = [e for e in r['entities']
                 if e['start_char'] < boundary < e['end_char']]
        if cross:
            crossing += 1
        out.append({
            'idx': i, 'id': r.get('id'),
            'original_text': original, 'boundary': boundary,
            'kept_entities': kept,
        })
    if crossing:
        print(f'WARNING: {crossing} rows have entity crossing boundary')
    return out


def _norm_span(s: dict, text: str) -> Optional[dict]:
    """재라벨 span({type,start,end}/{label,start_char,end_char}) → 표준 형식."""
    label = s.get('label') or s.get('type')
    start = s.get('start_char', s.get('start'))
    end = s.get('end_char', s.get('end'))
    if label not in NER_TYPES or start is None or end is None:
        return None
    return {'label': label, 'start_char': start, 'end_char': end,
            'text': text[start:end]}


def assemble(pii: List[dict], source: List[dict],
             relabel_map: Dict[str, List[dict]]) -> List[dict]:
    """재라벨 NER(원문 영역) + 보존 주입 entity → 새 gold 행."""
    out: List[dict] = []
    missing = 0
    for r, s in zip(pii, source):
        original = s['original_text']
        new_ner = relabel_map.get(original)
        if new_ner is None:
            # 재라벨 누락 → v1 원문 NER 유지(경계 안 entity)
            missing += 1
            b = s['boundary']
            new_ner = [e for e in r['entities'] if e['end_char'] <= b]
        else:
            new_ner = [n for n in
                       (_norm_span(x, r['text']) for x in new_ner)
                       if n is not None]
        entities = sorted(
            new_ner + s['kept_entities'], key=lambda e: e['start_char'])
        out.append({'text': r['text'], 'entities': entities,
                    'id': r.get('id')})
    if missing:
        print(f'NOTE: {missing} rows kept v1 NER (relabel missing)')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)

    pb = sub.add_parser('build-source')
    pb.add_argument('--pii', required=True)
    pb.add_argument('--all', dest='all_path', required=True)
    pb.add_argument('--out', required=True)

    pr = sub.add_parser('relabel')
    pr.add_argument('--source', required=True)
    pr.add_argument('--out', required=True)
    pr.add_argument('--base-url', default='http://localhost:8081/v1')
    pr.add_argument('--model', default='cyankiwi/gemma-4-31B-it-AWQ-8bit')
    pr.add_argument('--concurrency', type=int, default=16)
    pr.add_argument('--max-tokens', type=int, default=2048)

    pm = sub.add_parser('merge')
    pm.add_argument('--gemma', required=True, help='relabel output (Gemma)')
    pm.add_argument('--qwen', required=True, help='relabel output (Qwen)')
    pm.add_argument('--policy', default='recall_strict')
    pm.add_argument('--out', required=True)

    pa = sub.add_parser('assemble')
    pa.add_argument('--pii', required=True)
    pa.add_argument('--source', required=True)
    pa.add_argument('--relabel', required=True)
    pa.add_argument('--out', required=True)

    args = p.parse_args()

    if args.cmd == 'build-source':
        src = build_source(_load_jsonl(args.pii), _load_jsonl(args.all_path))
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            for r in src:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        uniq = {r['original_text'] for r in src}
        print(f'rows={len(src)} unique_originals={len(uniq)} -> {args.out}')

    elif args.cmd == 'relabel':
        from ner.augmenters.wikiann_vi.relabel_8type import Relabeler
        src = _load_jsonl(args.source)
        uniq = sorted({r['original_text'] for r in src})
        records = [{'id': str(i), 'text': t} for i, t in enumerate(uniq)]
        relabeler = Relabeler(
            base_url=args.base_url, model=args.model,
            max_tokens=args.max_tokens, concurrency=args.concurrency)
        print(f'relabeling {len(records)} unique originals @ {args.model}')
        relabeled = relabeler.relabel_sync(records)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            for rec, text in zip(relabeled, uniq):
                f.write(json.dumps(
                    {'text': text,
                     'spans': rec.get('gold_spans_8type', [])},
                    ensure_ascii=False) + '\n')
        print(f'wrote {len(relabeled)} relabeled -> {args.out}')

    elif args.cmd == 'merge':
        from ner.augmenters.wikiann_vi.merge_confidence import (
            _filter_by_policy, categorize_spans)
        gemma = {r['text']: r['spans'] for r in _load_jsonl(args.gemma)}
        qwen = {r['text']: r['spans'] for r in _load_jsonl(args.qwen)}
        out = []
        for text, gspans in gemma.items():
            cat = categorize_spans(gspans, qwen.get(text, []))
            merged = _filter_by_policy(cat, args.policy)
            out.append({'text': text, 'spans': merged})
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        n_spans = sum(len(r['spans']) for r in out)
        print(f'merged {len(out)} texts ({args.policy}), '
              f'{n_spans} spans -> {args.out}')

    elif args.cmd == 'assemble':
        relabel_map = {r['text']: r['spans']
                       for r in _load_jsonl(args.relabel)}
        out = assemble(_load_jsonl(args.pii), _load_jsonl(args.source),
                       relabel_map)
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'wrote {len(out)} rows -> {args.out}')


if __name__ == '__main__':
    main()
