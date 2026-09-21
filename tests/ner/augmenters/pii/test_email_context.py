"""EMAIL 붙은 문맥 치환 수락 검사.

치환이 바꿔도 되는 것은 주소를 둘러싼 글자뿐이다. 주소 표면·다른 엔티티
표면·그 밖의 문맥이 흔들리면 재학습 전후 차이를 붙은 문맥 때문이라고 말할
수 없으므로 그 불변식을 여기서 고정한다.
"""
import random

import pytest

from ner.augmenters.pii.email_context import (
    EMAIL_CUE_COLON_RATE,
    EMAIL_WRAP_WEIGHTS,
    EMAIL_WRAPS,
    glue_email_contexts,
    glued_share,
)


def _row(text, entities, rid='r0'):
    """(라벨, 표면) 쌍으로 레코드를 만든다 — 오프셋은 텍스트에서 센다."""
    spans = []
    for label, value in entities:
        start = text.index(value)
        spans.append({
            'label': label, 'start_char': start,
            'end_char': start + len(value), 'text': value,
        })
    return {'text': text, 'entities': spans, 'id': rid, 'orig': rid}


def _surfaces_hold(row):
    return all(row['text'][e['start_char']:e['end_char']] == e['text']
               for e in row['entities'])


@pytest.mark.parametrize('form', sorted(EMAIL_WRAPS))
def test_each_wrap_surrounds_the_address_and_keeps_it(form):
    """감싼 뒤에도 EMAIL span 은 주소만 가리키고, 바로 앞 글자는 공백이 아니다."""
    row = _row('Contact her at alice@example.com today.',
               [('EMAIL', 'alice@example.com')])
    out, forms = glue_email_contexts(row, random.Random(0),
                                     wrap_weights={form: 100},
                                     cue_colon_rate=0.0)
    opener, closer = EMAIL_WRAPS[form]
    assert forms == [form]
    assert out['text'] == f'Contact her at {opener}alice@example.com{closer} today.'
    ent = out['entities'][0]
    assert ent['text'] == 'alice@example.com'
    assert _surfaces_hold(out)
    assert out['text'][ent['start_char'] - 1] != ' '


def test_following_entities_shift_and_keep_their_surface():
    """주소 뒤 엔티티는 더한 글자 수만큼 밀리고 앞 엔티티는 그대로다."""
    text = 'John Smith mails alice@example.com and calls 555-0100 daily.'
    row = _row(text, [('PER', 'John Smith'), ('EMAIL', 'alice@example.com'),
                      ('PHONE', '555-0100')])
    out, _ = glue_email_contexts(row, random.Random(0),
                                 wrap_weights={'mailto': 100},
                                 cue_colon_rate=0.0)
    assert out['text'] == ('John Smith mails mailto:alice@example.com '
                           'and calls 555-0100 daily.')
    assert out['entities'][0]['start_char'] == 0
    assert out['entities'][2]['start_char'] == row['entities'][2]['start_char'] + 7
    assert _surfaces_hold(out)


def test_cue_word_glues_the_address_with_a_colon():
    """`email alice@…` 은 공백 자리에 콜론을 넣어 길이를 바꾸지 않는다."""
    row = _row('Please email alice@example.com with questions.',
               [('EMAIL', 'alice@example.com')])
    out, forms = glue_email_contexts(row, random.Random(0),
                                     wrap_weights={}, cue_colon_rate=1.0)
    assert forms == ['colon']
    assert out['text'] == 'Please email:alice@example.com with questions.'
    assert out['entities'] == row['entities']


def test_colon_needs_the_cue_word():
    """단서어가 없으면 콜론을 넣지 않는다 — `at:alice@…` 는 만들지 않는다."""
    row = _row('Contact her at alice@example.com today.',
               [('EMAIL', 'alice@example.com')])
    out, forms = glue_email_contexts(row, random.Random(0),
                                     wrap_weights={}, cue_colon_rate=1.0)
    assert forms == ['none']
    assert out == row


def test_an_already_glued_address_is_left_alone():
    """이미 공백 아닌 글자 뒤에 있는 주소는 한 번 더 감싸지 않는다."""
    row = _row('The organizer (alice@example.com) confirmed.',
               [('EMAIL', 'alice@example.com')])
    out, forms = glue_email_contexts(row, random.Random(0),
                                     wrap_weights={'angle': 100},
                                     cue_colon_rate=1.0)
    assert forms == ['none']
    assert out == row


def test_rows_without_email_are_unchanged():
    row = _row('John Smith visited Seoul.', [('PER', 'John Smith'),
                                             ('LOC', 'Seoul')])
    out, forms = glue_email_contexts(row, random.Random(0),
                                     wrap_weights={'angle': 100})
    assert forms == []
    assert out == row


def test_the_input_row_is_not_mutated():
    row = _row('Contact her at alice@example.com today.',
               [('EMAIL', 'alice@example.com')])
    before = repr(row)
    glue_email_contexts(row, random.Random(0), wrap_weights={'angle': 100})
    assert repr(row) == before


def test_same_seed_same_output():
    rows = [_row(f'Contact {i} at user{i}@example.com now.',
                 [('EMAIL', f'user{i}@example.com')], rid=str(i))
            for i in range(200)]
    run = [[glue_email_contexts(r, rng)[0] for r in rows]
           for rng in (random.Random(7), random.Random(7))]
    assert run[0] == run[1]


def test_default_rates_match_the_declared_weights():
    """기본값으로 돌린 비율이 선언한 표에 붙는다."""
    rng = random.Random(3)
    n = 20000
    wrap_forms, cue_forms = [], []
    for i in range(n):
        wrap_forms += glue_email_contexts(
            _row('Contact her at a@example.com now.',
                 [('EMAIL', 'a@example.com')]), rng)[1]
        cue_forms += glue_email_contexts(
            _row('Please email a@example.com now.',
                 [('EMAIL', 'a@example.com')]), rng)[1]
    for form, weight in EMAIL_WRAP_WEIGHTS.items():
        assert wrap_forms.count(form) / n == pytest.approx(weight / 100,
                                                            abs=0.006)
    assert cue_forms.count('colon') / n == pytest.approx(
        EMAIL_CUE_COLON_RATE, abs=0.015)


def test_mismatched_surface_stops():
    row = _row('Contact her at alice@example.com today.',
               [('EMAIL', 'alice@example.com')])
    row['entities'][0]['text'] = 'bob@example.com'
    with pytest.raises(ValueError, match='mismatch'):
        glue_email_contexts(row, random.Random(0))


def test_unsorted_entities_stop():
    row = _row('John mails alice@example.com today.',
               [('PER', 'John'), ('EMAIL', 'alice@example.com')])
    row['entities'].reverse()
    with pytest.raises(ValueError, match='sorted'):
        glue_email_contexts(row, random.Random(0))


def test_glued_share_counts_addresses_after_a_non_space():
    rows = [
        _row('at <a@x.com> now', [('EMAIL', 'a@x.com')]),
        _row('at b@x.com now', [('EMAIL', 'b@x.com')]),
        _row('c@x.com', [('EMAIL', 'c@x.com')]),
        _row('email:d@x.com', [('EMAIL', 'd@x.com')]),
    ]
    # 문자열 첫 자리는 붙은 자리로 세지 않는다 — 토크나이저 앞 공백이 맡는다.
    assert glued_share(rows) == pytest.approx(2 / 4)
