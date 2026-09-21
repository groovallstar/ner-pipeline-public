"""EMAIL 주소를 공백이 아닌 글자 뒤에 붙이는 문맥 치환.

바이트 BPE 토크나이저(RoBERTa)는 공백 뒤 단어에 `Ġ` 를 붙여 다른 토큰으로
쓴다. `at alice@x.com` 의 로컬파트는 `Ġalice` 이고 `<alice@x.com>` 의
로컬파트는 `al` + `ice` 다. LLM 자연 주입은 주소를 거의 언제나 공백 뒤에 두므로,
뒤쪽 모양을 학습에서 보지 못한 모델은 앞 조각을 떨어뜨린 채 주소를 잡는다.

이 모듈은 이미 주입된 주소의 일부를 괄호·따옴표·`mailto:`로 감싸거나, 앞의
단서어(`email`)와 콜론으로 잇는다. 주소를 문두·문말로 옮기지 않고 LLM 이 둔
자리를 그대로 두므로 위치 분포와 단서어는 바뀌지 않는다.

목표 비율은 `EMAIL_WRAP_WEIGHTS`·`EMAIL_CUE_COLON_RATE` 가 단일 출처다.
"""
from __future__ import annotations

import copy
import random
import re
from typing import Any, Dict, List, Optional, Tuple

EMAIL_LABEL = 'EMAIL'

# 감싸기 종류 → (여는 글자, 닫는 글자). `mailto:` 는 닫는 글자가 없다.
EMAIL_WRAPS: Dict[str, Tuple[str, str]] = {
    'angle': ('<', '>'),
    'paren': ('(', ')'),
    'quote': ('"', '"'),
    'mailto': ('mailto:', ''),
}

# 감쌀 수 있는 주소 가운데 각 종류로 감싸는 비율(%). 합 13% 를 뺀 나머지는
# 그대로 둔다. `<주소>` 는 메일 머리글·서명에서 가장 흔해 가장 크게 둔다.
EMAIL_WRAP_WEIGHTS: Dict[str, float] = {
    'angle': 5,
    'paren': 3,
    'quote': 3,
    'mailto': 2,
}

# 바로 앞이 단서어 `email`·`e-mail` + 공백 하나인 주소에서, 그 공백을 콜론으로
# 바꾸는 비율. 단서어 없이 콜론만 넣으면 `at:alice@…` 같은 없는 표기가 된다.
EMAIL_CUE_COLON_RATE = 0.3
_CUE_BEFORE = re.compile(r'(?i)(?:^|\s)e-?mail $')


def _wrap_eligible(text: str, start: int, end: int) -> bool:
    """앞이 공백·문자열 시작이고 뒤가 주소를 잇는 글자가 아닐 때만 감싼다."""
    left_ok = start == 0 or text[start - 1].isspace()
    right = text[end:end + 1]
    right_ok = not right or not (right.isalnum() or right == '@')
    return left_ok and right_ok


def _draw_wrap(weights: Dict[str, float], roll: float) -> Optional[str]:
    """0~100 사이 `roll` 을 누적 비율에 대어 감싸기 종류를 고른다."""
    acc = 0.0
    for form, weight in weights.items():
        acc += weight
        if roll < acc:
            return form
    return None


def glue_email_contexts(
    row: Dict[str, Any],
    rng: random.Random,
    wrap_weights: Optional[Dict[str, float]] = None,
    cue_colon_rate: float = EMAIL_CUE_COLON_RATE,
) -> Tuple[Dict[str, Any], List[str]]:
    """레코드의 EMAIL 주소 일부를 붙은 문맥으로 바꾼 새 레코드와 적용 종류를 돌려준다.

    적용 종류는 EMAIL span 마다 하나씩, 오프셋 순서로 `'none'`·`'colon'`·
    감싸기 종류 이름 가운데 하나다. 주소마다 난수를 두 번 뽑아 자격과 무관하게
    같은 양을 쓰므로, 같은 입력과 seed 는 같은 결과를 낸다.

    gold span 표면이 `text` 와 어긋나거나 엔티티가 오프셋 오름차순이 아니면
    ValueError 로 멈춘다. 어긋난 코퍼스를 조용히 고치면 그 손상이 치환 결과에
    섞여 원인을 가릴 수 없다.
    """
    weights = EMAIL_WRAP_WEIGHTS if wrap_weights is None else wrap_weights
    entities = row.get('entities', [])
    if not any(e.get('label') == EMAIL_LABEL for e in entities):
        return row, []

    row = copy.deepcopy(row)
    text = row['text']
    shift = 0
    prev_start = -1
    forms: List[str] = []
    for entity in row['entities']:
        if entity['start_char'] < prev_start:
            raise ValueError(
                f'entities are not sorted by offset: id={row.get("id")}')
        prev_start = entity['start_char']
        start = entity['start_char'] + shift
        end = entity['end_char'] + shift
        entity['start_char'], entity['end_char'] = start, end
        if 'text' in entity and text[start:end] != entity['text']:
            raise ValueError(
                f'span text mismatch: id={row.get("id")} '
                f'gold={entity["text"]!r} text={text[start:end]!r}')
        if entity['label'] != EMAIL_LABEL:
            continue

        colon_roll, wrap_roll = rng.random(), rng.random() * 100
        if (colon_roll < cue_colon_rate
                and _CUE_BEFORE.search(text[:start])
                and _wrap_eligible(text, start, end)):
            text = text[:start - 1] + ':' + text[start:]
            forms.append('colon')
            continue
        form = _draw_wrap(weights, wrap_roll)
        if form is None or not _wrap_eligible(text, start, end):
            forms.append('none')
            continue
        opener, closer = EMAIL_WRAPS[form]
        text = text[:start] + opener + text[start:end] + closer + text[end:]
        entity['start_char'] = start + len(opener)
        entity['end_char'] = end + len(opener)
        shift += len(opener) + len(closer)
        forms.append(form)

    row['text'] = text
    return row, forms


def glued_share(rows: List[Dict[str, Any]]) -> float:
    """EMAIL span 가운데 바로 앞 글자가 공백이 아닌 것의 비율.

    문자열 첫 자리는 세지 않는다. 그 자리는 토크나이저의 앞 공백 설정이
    맡아 코퍼스 치환의 대상이 아니다.
    """
    total = glued = 0
    for row in rows:
        text = row['text']
        for e in row.get('entities', []):
            if e['label'] != EMAIL_LABEL:
                continue
            total += 1
            start = e['start_char']
            glued += start > 0 and not text[start - 1].isspace()
    return glued / total if total else 0.0
