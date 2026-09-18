"""EN EMAIL 로컬파트 형태 × 문맥 probe — 실제 `/data/ner/en` 모델 의존(로컬 전용).

배포 test 는 주입기가 만든 분포 위에서만 재므로, 그 분포 밖 표면형을
놓쳐도 metric 이 멀쩡해 보인다. 이 probe 는 로컬파트 형태 셋(점 없는
소문자·점 있음·숫자 포함)과 문맥 넷(문중·문말·괄호·단독)을 교차해, 서버
추론 경로(`LangModel.predict`)가 주소를 경계까지 정확히 잡는지 strict 로 잰다.
주소를 조각으로 잡으면 실패로 센다 — 마스킹에서 조각 사이 글자는 그대로
새기 때문이다.

**probe 는 수정 전에 고정했다.** 통과하도록 문장을 고르면 이 기준이 자기
참조가 된다. 도메인에는 주입 목록에 없는 `bluepine.io` 같은 이름을 섞었다.
"""

import os
from collections import Counter

import pytest

from server.config import ServerConfig
from server.inference import LangModel

_CONFIG = ServerConfig.from_env()
_EN_DIR = _CONFIG.model_dir('en')

ADDRESSES = {
    'dotless_lower': [
        'alice@example.com', 'jsmith@gmail.com', 'emilythompson@yahoo.com',
        'dkowalski@bluepine.io', 'marcus@northgate.edu', 'okafor@acme.co.uk',
    ],
    'dotted': [
        'alice.kim@example.com', 'john.doe@acme.co.uk',
        'Maria.Gonzalez@outlook.com', 'felix.bergstrom@northgate.edu',
        'r.patel@bluepine.io', 'Alice.Kim@gmail.com',
    ],
    'digit': [
        'alice42@example.com', 'jdoe1987@gmail.com', 'emily.t88@yahoo.com',
        'mkim07@bluepine.io', '32rebecca@northgate.edu', 'sam_lee99@acme.co.uk',
    ],
}

# 문맥마다 틀 둘을 주소 순서대로 번갈아 쓴다. 단독은 입력 전체가 주소다.
CONTEXTS = {
    'mid': ['Contact her at {e} today.', 'You can reach {e} for the invoice.'],
    'end': ['Please send the report to {e}.', 'His address is {e}'],
    'paren': ['Contact her (email {e}) before Friday.',
              'The organizer ({e}) confirmed the room.'],
    'alone': ['{e}', '{e}'],
}

# 이슈 수락 기준. 전체와, 이 probe 를 만든 이유인 점 없는 소문자에 따로 건다.
MIN_RECALL = 0.90


def _probe():
    items = []
    for form, addresses in ADDRESSES.items():
        for i, address in enumerate(addresses):
            for ctx, templates in CONTEXTS.items():
                text = templates[i % 2].format(e=address)
                start = text.index(address)
                items.append((form, ctx, text, start, start + len(address)))
    return items


def test_probe_is_the_full_cross():
    """형태 3 × 주소 6 × 문맥 4 = 72 문장 — 칸이 빠지면 recall 이 부풀어 오른다."""
    items = _probe()
    assert len(items) == 72
    assert Counter(form for form, *_ in items) == {
        'dotless_lower': 24, 'dotted': 24, 'digit': 24}
    assert Counter(ctx for _, ctx, *_ in items) == {
        'mid': 18, 'end': 18, 'paren': 18, 'alone': 18}


@pytest.mark.skipif(not os.path.isdir(_EN_DIR),
                    reason=f'en model not present at {_EN_DIR}')
def test_en_email_probe_strict_recall():
    model = LangModel('en', _EN_DIR, _CONFIG.thresholds_path('en'),
                      _CONFIG.max_length)
    items = _probe()
    predictions = model.predict_many([text for _, _, text, _, _ in items])

    hits, totals, misses = Counter(), Counter(), []
    for (form, ctx, text, start, end), spans in zip(items, predictions):
        found = [(s['start_char'], s['end_char'])
                 for s in spans if s['label'] == 'EMAIL']
        ok = (start, end) in found
        for key in ('all', form, ctx):
            hits[key] += ok
            totals[key] += 1
        if not ok:
            misses.append(
                (text, [text[s:e] for s, e in found]))

    recall = {key: hits[key] / totals[key] for key in totals}
    report = ' '.join(f'{k}={recall[k]:.3f}' for k in sorted(recall))
    detail = '\n'.join(f'  {t!r} -> {f}' for t, f in misses)
    assert recall['all'] >= MIN_RECALL, f'{report}\n{detail}'
    assert recall['dotless_lower'] >= MIN_RECALL, f'{report}\n{detail}'
