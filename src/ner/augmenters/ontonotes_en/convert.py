"""OntoNotes5 원본 레코드를 canonical JSONL 레코드로 옮긴다.

원본은 줄당 `{"tokens": [...], "tags": [...]}` 이고 `tags` 는 `label.json`
(원천이 함께 배포)의 id 다. 여기서 하는 일은 셋이다 — BIO 를 토큰 span 으로
풀고, 원본 타입을 canonical 라벨로 옮기고(`mapping`), 자연문 복원에서 받은
토큰별 offset 으로 char-offset span 을 만든다(`detokenize`).

원본 `FAC` 만 타입이 아니라 **표면**으로 갈린다 — 개별 구조물은 `ORG`,
여러 지점을 잇는 경로는 `LOC` 이며 판정은 표가 갖는다(§3 인프라). 그래서
`resolve` 에 표면 키를 함께 넘기고, 그 키는 자연문이 아니라 원본 토큰의
공백 조인이다(`fac_surface`).

산출 레코드는 기존 JSONL contract 를 그대로 따르고 두 필드를 더한다.

- `split` — OntoNotes 원본 배정(train/valid/test). 저장소 관례인 80/10/10
  재분할을 하지 않으므로 어느 split 이었는지가 데이터에 남아야 한다.
- `orig` — 원문 식별자. PII 주입이 한 원문에서 형제 행을 만들기 때문에,
  형제를 한 split 에 묶을 그룹 키가 필요하다. 여기서 심어두지 않으면
  후속 학습에서 `--group-key` 를 줄 수단이 없다.

`orig` 는 행 일련번호가 아니라 **같은 문장을 공유하는 행끼리 같은 값**이다
(`assign_text_groups`). OntoNotes 는 같은 문장을 여러 번 담는다 — 방송·전화
대화 도메인의 `yeah`·`Uh-huh.` 같은 짧은 발화 때문이며 train 59,924 행 중
고유 문장은 55,154 개뿐이다. 행마다 고유한 값을 `orig` 로 주면 모든 unit 이
싱글턴이 되어 그룹 보호가 no-op 이 되는데, 같은 필드로 누출을 세면 중복이
0 이라 "누출 없음" 으로 읽힌다 — 보호는 없는데 게이트는 초록이다.
`validate_group_key` 가 바로 이 경우를 거부한다.
"""
from __future__ import annotations

import collections
import json
from collections.abc import Sequence
from pathlib import Path

from ner.augmenters.ontonotes_en.detokenize import detokenize, normalize_token
from ner.augmenters.ontonotes_en.mapping import (
    BY_SURFACE,
    ONTONOTES_TO_CANONICAL,
    is_dropped,
    resolve,
)

# 표면 키가 필요한 원본 타입 — 지금은 `FAC` 하나다. 리터럴로 적지 않는
# 것은 매핑표가 정본이기 때문이다.
BY_SURFACE_TYPES = frozenset(
    t for t, v in ONTONOTES_TO_CANONICAL.items() if v == BY_SURFACE
)


def load_id2label(label_json: Path) -> dict[int, str]:
    """원천이 배포하는 `label.json`(label→id)을 뒤집어 읽는다.

    id→label 표를 코드에 복제하지 않는 이유는 사본이 원천보다 낡을 수 있기
    때문이다. 원천 파일을 그대로 읽으면 갈릴 자리가 없다.
    """
    raw = json.loads(label_json.read_text(encoding='utf-8'))
    return {int(v): k for k, v in raw.items()}


def source_types(id2label: dict[int, str]) -> set[str]:
    """라벨 표에 등장하는 원본 엔티티 타입 집합 (`O` 제외)."""
    return {
        lab.split('-', 1)[1] for lab in id2label.values() if lab != 'O'
    }


def decode_bio(
    tags: Sequence[int], id2label: dict[int, str],
) -> list[tuple[int, int, str]]:
    """BIO 태그 열을 `(첫 토큰, 마지막 토큰, 원본 타입)` 목록으로 푼다.

    마지막 토큰 인덱스는 **포함**이다. `I-` 로 시작하거나 타입이 바뀌며
    이어지는 비정형 열은 새 span 의 시작으로 읽는다 — 버리면 gold 가 조용히
    줄어들기 때문이다.
    """
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    cur: str | None = None

    for i, tag_id in enumerate(tags):
        label = id2label[tag_id]
        if label == 'O':
            if start is not None:
                spans.append((start, i - 1, cur))
                start, cur = None, None
            continue
        prefix, typ = label.split('-', 1)
        if prefix == 'B' or cur != typ:
            if start is not None:
                spans.append((start, i - 1, cur))
            start, cur = i, typ
    if start is not None:
        spans.append((start, len(tags) - 1, cur))
    return spans


def count_source_spans(
    tags: Sequence[int], id2label: dict[int, str],
) -> collections.Counter:
    """원본 타입별 span 수를 센다 — 매핑표를 거치지 않는다.

    쓰임은 **테스트가 이 수를 못 박는 것**이다(`GOLDEN_SOURCE_SPANS`).
    canonical 쪽 골든 넘버와 달리 매핑표에 의존하지 않으므로, 둘을 함께
    고정하면 실패가 어느 단계에서 났는지 갈린다 — 원본 수만 움직이면 BIO
    디코드, canonical 수만 움직이면 매핑이다.

    **런타임 등식으로 쓰지 말 것.** 한때 "살아남은 span + 버린 span = 원본
    span" 을 CLI 에서 검사했는데, 버린 쪽을 검사기와 같은 `resolve()` 술어로
    파생시켜 **정의상 참**이었다 — 매핑 한 줄을 `DROP` 으로 뒤집어도 통과했다.
    잃어버린 span 을 실제로 잡는 것은 테스트에 손으로 박힌 골든 넘버뿐이다.
    """
    counts: collections.Counter = collections.Counter()
    for _, _, source_type in decode_bio(tags, id2label):
        counts[source_type] += 1
    return counts


def count_fac_surfaces(
    tokens: Sequence[str], tags: Sequence[int], id2label: dict[int, str],
) -> collections.Counter:
    """표면별 판정 타입의 표면을 센다 — 피복 검사가 쓰는 관측치.

    `resolve` 는 표에 없는 표면을 span 단위로 세우므로 "원본에만 있는 표면"
    은 변환 도중에 이미 걸린다. 여기서 걷는 값이 메우는 것은 반대 방향이다 —
    **표에만 있고 코퍼스에 없는 줄**은 span 이 하나도 안 지나가 아무 데도
    안 걸리므로, 관측 집합을 따로 모아 끝에서 맞춰야 보인다.
    """
    counts: collections.Counter = collections.Counter()
    for start, end, source_type in decode_bio(tags, id2label):
        if source_type in BY_SURFACE_TYPES:
            counts[fac_surface(tokens, start, end)] += 1
    return counts


def strip_ws(text: str) -> str:
    """공백을 전부 뺀 문자열 — 엔티티↔토큰 대조의 정규형.

    자연문 복원이 바꿔도 되는 것은 띄어쓰기뿐이다. 글자가 늘거나 줄면
    복원이 아니라 변조다.
    """
    return ''.join(text.split())


def fac_surface(
    tokens: Sequence[str], tok_start: int, tok_end: int,
) -> str:
    """판정 표를 찾을 표면 키 — 원본 토큰의 공백 조인 (끝 인덱스 포함).

    자연문 복원(`detokenize`)이 아니라 **원본 토큰**으로 키를 만드는 것이
    요점이다. 복원 규칙은 붙여쓰기·따옴표 처리가 바뀔 수 있는 코드인데,
    그게 바뀌면 같은 span 이 다른 키를 내어 표 전체가 한꺼번에 미판정으로
    돌아선다. 토큰 배열은 원천이 준 그대로라 안 흔들린다.
    """
    return ' '.join(tokens[tok_start:tok_end + 1])


def convert_record(
    tokens: Sequence[str],
    tags: Sequence[int],
    id2label: dict[int, str],
    record_id: str,
    split: str,
) -> dict:
    """원본 한 줄을 canonical JSONL 레코드 한 개로 옮긴다."""
    text, offsets = detokenize(tokens)
    entities = []
    for tok_start, tok_end, src_type in decode_bio(tags, id2label):
        label = resolve(src_type, fac_surface(tokens, tok_start, tok_end))
        if label is None:
            continue
        start_char = offsets[tok_start][0]
        end_char = offsets[tok_end][1]
        entities.append({
            'label': label,
            'start_char': start_char,
            'end_char': end_char,
            'text': text[start_char:end_char],
        })
    return {
        'text': text,
        'entities': entities,
        'id': record_id,
        'orig': record_id,
        'split': split,
    }


def assign_text_groups(records: list[dict]) -> None:
    """같은 문장을 가진 행끼리 `orig` 를 공유하게 다시 매긴다 (제자리 수정).

    한 split 안에서만 묶는다. split 을 가로지르는 중복은 원본 split 을 그대로
    쓰기로 한 이상 묶을 수 없다. 그 유지 근거는 "외부 공개 수치와 비교" 가
    **아니다** — 18→6 매핑으로 라벨 공간이 이미 달라져 published
    OntoNotes NER F1 과 나란히 못 놓는다. 남은 근거는 공식 split 을 그대로
    쓰는 편이 재현·인용에 유리하고 노출이 test span 의 0.81% 로 작다는 것이며,
    규모가 커지면 재검토 대상이다. 규모는 CLI 가 매 실행 보고한다.
    """
    first_seen: dict[str, str] = {}
    for record in records:
        first_seen.setdefault(record['text'], record['id'])
        record['orig'] = first_seen[record['text']]


def check_entity_token_match(
    record: dict, tokens: Sequence[str], tags: Sequence[int],
    id2label: dict[int, str],
) -> list[str]:
    """엔티티 표면이 원본 토큰과 (공백을 뺀 채) 같은지 대조한다.

    비교 상대가 우리가 만들지 않은 원본 `tokens` 배열이라, `detokenize` 의
    offset 산술과 BIO 디코드의 span 인덱스를 실제로 검증한다.

    **이 검사가 못 보는 것을 분명히 해 둔다.**

    - `text[start:end] == entity['text']` 는 **항진명제**다 — `convert_record`
      가 `entity['text']` 를 그 슬라이스로 대입하기 때문이다. 변환 산출물에
      대해서는 이 대조가 아무것도 걸러내지 못한다(PII 주입은 문장을 다시
      쓰므로 그 산출물에서는 자명하지 않다).
    - `normalize_token` 이 양쪽에 걸리므로 **정규화 표 자체는 검증되지
      않는다.** 그 표는 `test_detokenize.py` 가 따로 고정한다.
    - **잃어버린 span 은 못 본다.** 여기 오는 것은 살아남은 span 뿐이라
      매핑 한 줄이 통째로 빠져도 남은 것끼리는 다 맞는다. 그 자리를 메우는
      것은 테스트에 손으로 박힌 골든 넘버다 — 원본 타입별 수(매핑과 독립)와
      이 이슈가 안 건드리는 canonical 타입(`PER`·`PROD`·`EVT`·`DAT`)의 수다.
      `LOC`·`ORG` 는 판정 표가 정하는 값이라 골든이 아니고, 표에서 도출한
      기대 수와 대조한다. 런타임 등식으로 메우려던 시도는 정의상 참이 되어
      실패했다(위 `count_source_spans` 참조).
    """
    problems: list[str] = []
    kept = []
    for start, end, source_type in decode_bio(tags, id2label):
        # 타입만으로 끝나는 드롭은 표면을 만들지 않는다 — `is_dropped` 가
        # 표면 없이 답하는 자리다. 나머지만 표면을 만들어 표에 묻는다.
        if is_dropped(source_type):
            continue
        if resolve(source_type,
                   fac_surface(tokens, start, end)) is None:
            continue
        kept.append((start, end, source_type))
    if len(kept) != len(record['entities']):
        problems.append(
            f'{record["id"]}: span count {len(record["entities"])} '
            f'!= source {len(kept)}'
        )
        return problems
    for (tok_start, tok_end, _), ent in zip(kept, record['entities']):
        expected = strip_ws(''.join(
            normalize_token(t) for t in tokens[tok_start:tok_end + 1]
        ))
        actual = strip_ws(ent['text'])
        if expected != actual:
            problems.append(
                f'{record["id"]}: {actual!r} != source {expected!r}'
            )
    return problems
