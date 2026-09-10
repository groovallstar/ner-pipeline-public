"""EMAIL 로컬파트 치환 스크립트 수락 검사.

코퍼스를 다시 만들 때 바뀌어야 하는 것은 EMAIL 로컬파트 하나뿐이다. 도메인·
문맥·다른 라벨이 함께 흔들리면 재학습 전후 차이의 원인을 EMAIL 표면형으로
좁힐 수 없으므로, 그 불변식을 여기서 고정한다.
"""
import json
import random

import pytest

from ner.scripts import rewrite_email_localpart as rw


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


def test_email_span_gets_a_new_local_part_and_keeps_the_domain():
    """치환 뒤에도 `@` 뒤 도메인은 그대로다."""
    row = _row('Contact 32Rebecca.Thompson@gmail.com now.',
               [('EMAIL', '32Rebecca.Thompson@gmail.com')])
    out, _ = rw.rewrite_rows([row], 'en', random.Random(1))
    value = out[0]['entities'][0]['text']
    assert value.rsplit('@', 1)[1] == 'gmail.com'
    assert value.rsplit('@', 1)[0] != '32Rebecca.Thompson'


def test_span_offsets_point_at_the_new_surface():
    """엔티티 오프셋이 새 텍스트의 같은 표면을 가리킨다."""
    row = _row('Contact 32Rebecca.Thompson@gmail.com now.',
               [('EMAIL', '32Rebecca.Thompson@gmail.com')])
    out, _ = rw.rewrite_rows([row], 'en', random.Random(1))
    ent = out[0]['entities'][0]
    assert out[0]['text'][ent['start_char']:ent['end_char']] == ent['text']


def test_following_entity_offsets_shift_by_the_length_delta():
    """뒤따르는 엔티티가 길이 차이만큼 밀린다."""
    text = 'Mail 15James.Gon.zalez@yahoo.com and ID 901122501 today.'
    row = _row(text, [('EMAIL', '15James.Gon.zalez@yahoo.com'),
                      ('ID_NUM', '901122501')])
    out, _ = rw.rewrite_rows([row], 'en', random.Random(2))
    tail = out[0]['entities'][1]
    assert tail['text'] == '901122501'
    assert out[0]['text'][tail['start_char']:tail['end_char']] == '901122501'


def test_text_outside_the_span_is_untouched():
    """문맥 글자는 한 자도 바뀌지 않는다."""
    text = 'Mail 15James.Gon.zalez@yahoo.com and ID 901122501 today.'
    row = _row(text, [('EMAIL', '15James.Gon.zalez@yahoo.com'),
                      ('ID_NUM', '901122501')])
    out, _ = rw.rewrite_rows([row], 'en', random.Random(2))
    new_email = out[0]['entities'][0]['text']
    assert out[0]['text'] == text.replace('15James.Gon.zalez@yahoo.com',
                                          new_email)


def test_rows_without_email_are_returned_unchanged():
    """EMAIL 이 없는 레코드는 그대로 통과한다."""
    row = _row('Seoul is a city.', [('LOC', 'Seoul')])
    out, stats = rw.rewrite_rows([row], 'ko', random.Random(3))
    assert out[0] == row
    assert stats['email_spans'] == 0


def test_span_text_mismatch_is_refused():
    """gold 와 텍스트가 어긋난 레코드는 조용히 고치지 않고 멈춘다."""
    row = _row('Contact a@b.com now.', [('EMAIL', 'a@b.com')])
    row['entities'][0]['text'] = 'zzz@b.com'
    with pytest.raises(ValueError, match='span text mismatch'):
        rw.rewrite_rows([row], 'en', random.Random(4))


def test_rewrite_is_deterministic_for_a_seed():
    """같은 seed 면 같은 코퍼스가 나온다."""
    rows = [_row(f'Mail a{i}.b@gmail.com now.',
                 [('EMAIL', f'a{i}.b@gmail.com')], f'r{i}')
            for i in range(20)]
    first, _ = rw.rewrite_rows(json.loads(json.dumps(rows)), 'en',
                               random.Random(5))
    second, _ = rw.rewrite_rows(json.loads(json.dumps(rows)), 'en',
                                random.Random(5))
    assert first == second


def test_stats_report_before_and_after_digit_prefix_rate():
    """원장에 남길 전후 분포를 산출한다."""
    rows = [_row(f'Mail {i}5James.Gonzalez@yahoo.com now.',
                 [('EMAIL', f'{i}5James.Gonzalez@yahoo.com')], f'r{i}')
            for i in range(50)]
    _, stats = rw.rewrite_rows(rows, 'en', random.Random(6))
    assert stats['email_spans'] == 50
    assert stats['before']['digit_prefix'] == 1.0
    assert stats['after']['digit_prefix'] <= 0.1
