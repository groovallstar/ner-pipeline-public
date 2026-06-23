"""gemma-검증 silver-갭 additive 삽입 — leak-free group-kfold 유지."""
from typing import Dict, List, Tuple


def apply_silver_gap(
    adjudicated: List[dict], gold_rows: List[dict], etype: str = 'EVT',
    additive_only: bool = True,
) -> Tuple[List[dict], List[dict], Dict[str, int]]:
    """gemma-검증 silver-갭(FP→TP)을 gold 에 additive 삽입.

    선택 케이스: direction=FP, model_label=etype, adj.correct=etype,
    not ambiguous (gemma §3 가 'gold 누락 legit' 으로 독립 확인한 FP).
    additive_only 면 silver_label=='O'(기존 gold span 과 비-겹침)만 —
    ORG/LOC 등 cross-type 재라벨은 제외(기존 라벨을 바꾸는 위험 작업).
    (sentence_id, start, end) 중복 제거 후 gold id 매칭 행의 entities 에
    surface 일치·비-겹침을 재검증하여 삽입한다. 원본 행 필드(orig 등)는
    보존 — leak-free group-kfold(--group-key orig) 유지에 필수.

    반환: (새 gold rows, 추가된 span diff, skip 카운트).
    """
    seen = set()
    cases: List[dict] = []
    for c in adjudicated:
        adj = c.get('adj') or {}
        if (c.get('direction') != 'FP' or c.get('model_label') != etype
                or adj.get('correct') != etype or adj.get('ambiguous')):
            continue
        if additive_only and c.get('silver_label') != 'O':
            continue
        key = (c['sentence_id'], c['start'], c['end'])
        if key in seen:
            continue
        seen.add(key)
        cases.append(c)

    by_id: Dict[str, List[dict]] = {}
    for c in cases:
        by_id.setdefault(c['sentence_id'], []).append(c)

    new_rows: List[dict] = []
    added: List[dict] = []
    skipped = {'surface': 0, 'overlap': 0}
    for r in gold_rows:
        row_cases = by_id.get(r.get('id'))
        if not row_cases:
            new_rows.append(r)
            continue
        text = r['text']
        entities = list(r.get('entities', []))
        existing = [(int(e['start_char']), int(e['end_char']))
                    for e in entities]
        for c in row_cases:
            s, e = c['start'], c['end']
            if text[s:e] != c['surface']:
                skipped['surface'] += 1
                continue
            if any(s < b and a < e for a, b in existing):
                skipped['overlap'] += 1
                continue
            entities.append({'label': etype, 'start_char': s,
                             'end_char': e, 'text': c['surface']})
            existing.append((s, e))
            added.append({'id': r.get('id'), 'text': text,
                          'surface': c['surface'], 'start': s, 'end': e,
                          'rule': (c.get('adj') or {}).get('rule')})
        entities.sort(key=lambda x: int(x['start_char']))
        new_rows.append({**r, 'entities': entities})
    return new_rows, added, skipped
