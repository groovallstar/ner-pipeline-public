"""EN EMAIL 로컬파트 형태 × 문맥 probe — 실제 `/data/ner/en` 모델 의존(로컬 전용).

배포 test 는 주입기가 만든 분포 위에서만 재므로, 그 분포 밖 표면형을
놓쳐도 metric 이 멀쩡해 보인다. 이 probe 는 로컬파트 형태 셋(점 없는
소문자·점 있음·숫자 포함)과 문맥 넷(문중·문말·괄호·단독)을 교차해, 서버
추론 경로(`LangModel.predict`)가 주소를 경계까지 정확히 잡는지 strict 로 잰다.
주소를 조각으로 잡으면 실패로 센다 — 마스킹에서 조각 사이 글자는 그대로
새기 때문이다.

**probe 는 수정 전에 고정했다.** 통과하도록 문장을 고르면 이 기준이 자기
참조가 된다. 도메인에는 주입 목록에 없는 `bluepine.io` 같은 이름을 섞었다.

붙은 문맥 probe 는 같은 주소를 공백이 아닌 글자 뒤에 둔다. 바이트 BPE 는
공백 뒤 단어에만 `Ġ` 를 붙이므로, 이 자리에서는 로컬파트 첫 토큰이 문중과
다른 모양이 된다. 겨냥한 문맥과 겨냥하지 않은 문맥을 나눠, 뒤쪽은 하한 없이
일반화 증거로만 보고한다. 음성 probe 는 이메일이 아닌 내용을 같은 자리에
넣어 "괄호 안·문두 = EMAIL" 같은 지름길을 잡는다.
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
# 문두(단독)와 문중·문말에도 같은 하한을 건다.
MIN_RECALL = 0.90

# 공백이 아닌 글자 뒤에 붙은 주소. 앞 다섯은 학습 코퍼스 치환이, 뒤 셋은
# 토큰화 직전 공백 정규화가 겨냥한다. 하한은 이 여덟 문맥 전체에 건다.
GLUED_CONTEXTS = {
    'angle': ['Send the contract to <{e}> by Monday.', 'Maria Lopez <{e}>'],
    'paren': ['Ask the registrar ({e}) about the fee.',
              'Our new intern ({e}) starts on Monday.'],
    'quote': ['Her new address is "{e}" now.', 'Type "{e}" into the login box.'],
    'mailto': ['Click mailto:{e} to write to us.', 'mailto:{e}'],
    'colon': ['Email:{e}', 'Please email:{e} with questions.'],
    'newline': ['Contact:\n{e}', 'Best regards,\nMaria Lopez\n{e}'],
    'crlf': ['Contact:\r\n{e}', 'Name: Maria Lopez\r\n{e}\r\n'],
    'tab': ['Maria Lopez\t{e}', 'id\t{e}\tactive'],
}

# 어느 수정도 겨냥하지 않은 붙은 문맥 — 하한 없이 보고만 한다.
HELD_OUT_CONTEXTS = {
    'bracket': ['See [{e}] for details.', '[{e}]'],
    'single_quote': ["Use '{e}' to sign in.", "'{e}'"],
    'equals': ['user={e}', 'from={e} status=sent'],
}

# 이메일이 없는 입력. 붙은 문맥·문두 자리에 다른 내용을 넣었다. EMAIL 이
# 하나라도 나오면 모델이 주소 모양이 아니라 자리를 보고 있다는 뜻이다.
NEGATIVE_TEXTS = [
    'Maria Lopez <Head of Sales>', 'Send it to <the main office> today.',
    'Ask the registrar (Seoul campus) about the fee.',
    'The report (page 12) is due.', 'Her new title is "senior editor" now.',
    'Type "password" into the login box.', 'Click mailto: to write to us.',
    'Email:', 'Email:none', 'Please email:us with questions.',
    'Contact:\nMaria Lopez', 'Best regards,\nMaria Lopez\nSeoul',
    'Contact:\r\nMaria Lopez', 'Maria Lopez\tSeoul', 'id\t42\tactive',
    'See [the appendix] for details.', "Use 'admin' to sign in.",
    'user=admin', 'from=sales status=sent', 'alice.okafor',
    'Alice Okafor went home.', 'Bluepine announced a merger.',
]


def _probe(contexts=CONTEXTS):
    items = []
    for form, addresses in ADDRESSES.items():
        for i, address in enumerate(addresses):
            for ctx, templates in contexts.items():
                text = templates[i % 2].format(e=address)
                start = text.index(address)
                items.append((form, ctx, text, start, start + len(address)))
    return items


def _model():
    return LangModel('en', _EN_DIR, _CONFIG.thresholds_path('en'),
                     _CONFIG.max_length)


def _strict_recall(model, items):
    """(키별 recall, 놓친 입력 설명) — 키는 'all'·형태·문맥이다."""
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
    return recall, f'{report}\n{detail}'


def test_probe_is_the_full_cross():
    """형태 3 × 주소 6 × 문맥 4 = 72 문장 — 칸이 빠지면 recall 이 부풀어 오른다."""
    items = _probe()
    assert len(items) == 72
    assert Counter(form for form, *_ in items) == {
        'dotless_lower': 24, 'dotted': 24, 'digit': 24}
    assert Counter(ctx for _, ctx, *_ in items) == {
        'mid': 18, 'end': 18, 'paren': 18, 'alone': 18}


def test_glued_probe_puts_every_address_after_a_non_space():
    """붙은 문맥은 주소 바로 앞 글자가 공백이 아니거나 입력의 첫 글자다.

    공백 뒤에 오면 로컬파트가 `Ġ` 토큰이 되어 이 probe 가 재려는 자리가
    아니다. 문맥 11 × 주소 18 = 198 문장이다.
    """
    items = _probe({**GLUED_CONTEXTS, **HELD_OUT_CONTEXTS})
    assert len(items) == 198
    for _, ctx, text, start, _ in items:
        assert start == 0 or text[start - 1] != ' ', (ctx, text)
    assert all('@' not in t for t in NEGATIVE_TEXTS)


_needs_model = pytest.mark.skipif(not os.path.isdir(_EN_DIR),
                                  reason=f'en model not present at {_EN_DIR}')


@_needs_model
def test_en_email_probe_strict_recall():
    recall, report = _strict_recall(_model(), _probe())
    below = [key for key in ('all', 'dotless_lower', 'alone', 'mid', 'end')
             if recall[key] < MIN_RECALL]
    assert not below, f'below {MIN_RECALL}: {below}\n{report}'


@_needs_model
def test_en_email_glued_probe_strict_recall():
    """겨냥한 여덟 문맥 전체에 하한을 건다. 겨냥하지 않은 셋은 보고만 한다."""
    model = _model()
    recall, report = _strict_recall(model, _probe(GLUED_CONTEXTS))
    _, held_out = _strict_recall(model, _probe(HELD_OUT_CONTEXTS))
    assert recall['all'] >= MIN_RECALL, f'{report}\nheld-out: {held_out}'


@_needs_model
def test_en_email_negative_probe_has_no_email():
    predictions = _model().predict_many(NEGATIVE_TEXTS)
    found = [(text, s['text']) for text, spans in zip(NEGATIVE_TEXTS, predictions)
             for s in spans if s['label'] == 'EMAIL']
    assert not found, found
