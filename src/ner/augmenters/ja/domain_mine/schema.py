"""domain_mine 공용 스키마·변환·검증 유틸 (순수 함수, 외부 의존 없음).

마이닝한 문장을 classifier contract JSONL 로 변환·검증하는 토대.
B2~B5 가 본 모듈의 함수를 재사용한다.

classifier contract (data_utils.load_jsonl 가 소비):
    {"text": str,
     "entities": [{"label": str, "start_char": int,
                   "end_char": int, "text": str}],
     "id": str}

재라벨 span (wikiann_vi Relabeler 출력) 은 {type,start,end,text} 키라
필드명이 다르다 → span_to_entity 로 변환한다.
"""

from typing import Dict, List, Tuple

from ner.classifier.data_utils import CANONICAL_LABELS

# 약점 4도메인 (식품은 license 분기로 본 라운드 제외). prod_seed 와 동일 키.
DOMAINS = ('law', 'book', 'transit_card', 'music_work')

# 마이닝한 도메인 명칭의 anchor 라벨 — 4도메인 모두 canonical PROD.
ANCHOR_LABEL = 'PROD'

_VALID_LABELS = set(CANONICAL_LABELS)


def span_to_entity(span: Dict) -> Dict:
    """재라벨 span {type,start,end,text} → contract entity 로 변환."""
    return {
        'label': span['type'],
        'start_char': span['start'],
        'end_char': span['end'],
        'text': span['text'],
    }


def spans_to_entities(spans: List[Dict]) -> List[Dict]:
    """재라벨 span 리스트를 contract entity 리스트로 일괄 변환."""
    return [span_to_entity(s) for s in spans]


def make_record(rec_id: str, text: str,
                entities: List[Dict]) -> Dict:
    """classifier contract 레코드 한 건 생성."""
    return {'id': str(rec_id), 'text': text, 'entities': entities}


def find_occurrences(text: str, name: str) -> List[Tuple[int, int]]:
    """text 안 name 의 비중첩 출현 위치 (start,end) 를 전부 반환."""
    if not name:
        return []
    spans = []
    start = 0
    while True:
        idx = text.find(name, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(name)))
        start = idx + len(name)
    return spans


def make_anchor_entity(text: str, name: str, start_char: int) -> Dict:
    """마이닝 명칭을 PROD anchor entity 로 생성 (offset 명시)."""
    end_char = start_char + len(name)
    return {
        'label': ANCHOR_LABEL,
        'start_char': start_char,
        'end_char': end_char,
        'text': text[start_char:end_char],
    }


def validate_labels(entities: List[Dict]) -> List[str]:
    """entity label 이 canonical 10종 안에 있는지 검증."""
    errors = []
    for i, e in enumerate(entities):
        if e.get('label') not in _VALID_LABELS:
            errors.append(
                f'entity[{i}]: unknown label {e.get("label")!r}'
            )
    return errors


def validate_offsets(text: str, entities: List[Dict]) -> List[str]:
    """offset 무결성 검증: 경계 범위 + text[start:end] == entity text.

    Phase 3 의 offset shift 손상 (번역 혼입으로 span 좌표 어긋남) 을
    재발 방지하는 핵심 게이트.
    """
    errors = []
    n = len(text)
    for i, e in enumerate(entities):
        s, t = e.get('start_char'), e.get('end_char')
        if not isinstance(s, int) or not isinstance(t, int):
            errors.append(f'entity[{i}]: non-int offset')
            continue
        if not (0 <= s < t <= n):
            errors.append(
                f'entity[{i}]: offset out of range '
                f'[{s},{t}) len={n}'
            )
            continue
        if text[s:t] != e.get('text'):
            errors.append(
                f'entity[{i}]: text mismatch '
                f'{text[s:t]!r} != {e.get("text")!r}'
            )
    return errors


def find_overlaps(entities: List[Dict]) -> List[Tuple[int, int]]:
    """span 중첩 쌍 (i,j) 반환. flat schema 는 비중첩이어야 한다."""
    spans = [(e['start_char'], e['end_char'], i)
             for i, e in enumerate(entities)]
    spans.sort()
    overlaps = []
    for a in range(len(spans)):
        e_a = spans[a][1]
        i_a = spans[a][2]
        for b in range(a + 1, len(spans)):
            s_b, _, i_b = spans[b]
            if s_b >= e_a:
                break
            overlaps.append((i_a, i_b))
    return overlaps


def validate_record(rec: Dict) -> List[str]:
    """레코드 전체 검증: 필수 키 + 라벨 + offset + 중첩. 통과 시 빈 리스트."""
    errors = []
    for key in ('id', 'text', 'entities'):
        if key not in rec:
            errors.append(f'missing key: {key}')
    if errors:
        return errors
    text, entities = rec['text'], rec['entities']
    errors.extend(validate_labels(entities))
    offset_errors = validate_offsets(text, entities)
    errors.extend(offset_errors)
    if not offset_errors:
        overlaps = find_overlaps(entities)
        if overlaps:
            errors.append(f'overlapping spans: {overlaps}')
    return errors
