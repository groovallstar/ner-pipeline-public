"""VI DAT 표면형 치환 스크립트 수락 검사.

코퍼스를 다시 만들 때 바뀌어야 하는 것은 DAT 표면형 하나뿐이다. 문맥·다른
라벨이 함께 흔들리면 재학습 전후 차이의 원인을 DAT 표면형으로 좁힐 수 없으므로,
그 불변식을 여기서 고정한다.

`ngày` 를 둘러싼 규칙 둘이 이 스크립트의 핵심이라 자리마다 따로 잠근다. 서술형
`D tháng M năm Y` 는 앞의 `ngày` 를 span 밖에 남기고, 월-연 `tháng M năm Y` 는
그 `ngày` 를 먹고 들어간다. 먹어야 하는 이유는 `ngày tháng 5 năm 1988` 이 "날 5월
1988년" 이라 말이 안 되기 때문이다.
"""
import random

import pytest

from ner.scripts import rewrite_vi_dat_surface as rw

_ALWAYS_FULL = {'numeric': 0, 'descriptive_full': 100, 'month_year': 0}
_ALWAYS_MONTH_YEAR = {'numeric': 0, 'descriptive_full': 0, 'month_year': 100}
_ALWAYS_NUMERIC = {'numeric': 100, 'descriptive_full': 0, 'month_year': 0}


def _row(text, entities, rid='r0'):
    """(라벨, 표면) 쌍으로 레코드를 만든다 — 오프셋은 텍스트에서 센다."""
    spans = []
    cursor = 0
    for label, value in entities:
        start = text.index(value, cursor)
        spans.append({
            'label': label, 'start_char': start,
            'end_char': start + len(value), 'text': value,
        })
        cursor = start + len(value)
    return {'text': text, 'entities': spans, 'id': rid, 'orig': rid}


def _rewrite(row, weights, seed=1):
    out, stats = rw.rewrite_rows([row], random.Random(seed), weights)
    return out[0], stats


def test_numeric_span_becomes_a_descriptive_date():
    """숫자 날짜가 `D tháng M năm Y` 로 바뀐다."""
    row = _row('Giấy phép được cấp ngày 26/05/1988.', [('DAT', '26/05/1988')])
    out, _ = _rewrite(row, _ALWAYS_FULL)
    assert out['entities'][0]['text'] == '26 tháng 5 năm 1988'
    assert out['text'] == 'Giấy phép được cấp ngày 26 tháng 5 năm 1988.'


@pytest.mark.parametrize('value,expected', [
    ('26/05/1988', '26 tháng 5 năm 1988'),
    ('1988-05-26', '26 tháng 5 năm 1988'),
    ('26-05-1988', '26 tháng 5 năm 1988'),
])
def test_all_three_numeric_forms_parse_to_the_same_date(value, expected):
    """세 숫자 표기가 같은 날짜로 읽힌다 — 자리 순서를 잘못 읽으면 날짜가 바뀐다."""
    row = _row(f'Giấy phép được cấp ngày {value}.', [('DAT', value)])
    out, _ = _rewrite(row, _ALWAYS_FULL)
    assert out['entities'][0]['text'] == expected


def test_ngay_stays_outside_the_descriptive_span():
    """서술형 span 은 앞의 `ngày` 를 포함하지 않는다.

    현 코퍼스가 `ngày` + 숫자 날짜 3,708 자리를 예외 없이 그렇게 두고 있어,
    관례를 따르는 쪽이 모델에 주는 신호가 일관된다.
    """
    row = _row('Giấy phép được cấp ngày 26/05/1988.', [('DAT', '26/05/1988')])
    out, _ = _rewrite(row, _ALWAYS_FULL)
    ent = out['entities'][0]
    assert not ent['text'].lower().startswith('ngày')
    assert out['text'][:ent['start_char']].endswith('ngày ')


def test_month_year_eats_the_leading_ngay():
    """월-연 표기는 앞의 `ngày` 를 먹고 들어간다."""
    row = _row('Giấy phép được cấp ngày 26/05/1988.', [('DAT', '26/05/1988')])
    out, _ = _rewrite(row, _ALWAYS_MONTH_YEAR)
    assert out['text'] == 'Giấy phép được cấp tháng 5 năm 1988.'
    ent = out['entities'][0]
    assert ent['text'] == 'tháng 5 năm 1988'
    assert out['text'][ent['start_char']:ent['end_char']] == ent['text']


def test_month_year_keeps_sentence_initial_capitalization():
    """문두의 `Ngày` 를 먹으면 `Tháng` 으로 올려 쓴다."""
    row = _row('Ngày 26/05/1988 là hạn chót.', [('DAT', '26/05/1988')])
    out, _ = _rewrite(row, _ALWAYS_MONTH_YEAR)
    assert out['text'] == 'Tháng 5 năm 1988 là hạn chót.'
    assert out['entities'][0]['text'] == 'Tháng 5 năm 1988'


def test_month_year_falls_back_to_numeric_without_a_leading_ngay():
    """앞에 `ngày` 가 없으면 월-연으로 바꾸지 않고 숫자 표기로 남긴다.

    `diễn ra vào tháng 5 năm 1988` 자체는 말이 되지만, 먹을 `ngày` 가 없는데
    span 만 월-연으로 바꾸면 날짜의 일(日) 정보가 문맥 없이 사라진다.
    """
    row = _row('Sự kiện diễn ra vào 02/03/1957.', [('DAT', '02/03/1957')])
    out, stats = _rewrite(row, _ALWAYS_MONTH_YEAR)
    assert out['text'] == 'Sự kiện diễn ra vào 02/03/1957.'
    assert stats['fallback']['month_year_without_ngay'] == 1


def test_site_after_nam_is_left_alone():
    """앞 낱말이 `năm` 인 자리는 건드리지 않는다.

    바꾸면 `năm 26 tháng 5 năm 1988` 이 되어 `năm` 이 두 번 나온다.
    """
    row = _row('Thành viên sinh năm 1973-07-07 tại Huế.',
               [('DAT', '1973-07-07')])
    out, stats = _rewrite(row, _ALWAYS_FULL)
    assert out['text'] == 'Thành viên sinh năm 1973-07-07 tại Huế.'
    assert stats['fallback']['preceded_by_time_word'] == 1


def test_following_entity_offsets_shift_by_the_length_delta():
    """뒤따르는 엔티티가 길이 차이만큼 밀린다."""
    text = 'Cấp ngày 26/05/1988 tại Hà Nội cho Nguyen Van An.'
    row = _row(text, [('DAT', '26/05/1988'), ('LOC', 'Hà Nội'),
                      ('PER', 'Nguyen Van An')])
    out, _ = _rewrite(row, _ALWAYS_FULL)
    for ent in out['entities']:
        assert out['text'][ent['start_char']:ent['end_char']] == ent['text']
    assert [e['text'] for e in out['entities']][1:] == [
        'Hà Nội', 'Nguyen Van An']


def test_text_outside_the_span_is_untouched():
    """문맥 글자는 한 자도 바뀌지 않는다."""
    text = 'Cấp ngày 26/05/1988 tại Hà Nội cho Nguyen Van An.'
    row = _row(text, [('DAT', '26/05/1988'), ('LOC', 'Hà Nội'),
                      ('PER', 'Nguyen Van An')])
    out, _ = _rewrite(row, _ALWAYS_FULL)
    assert out['text'] == text.replace('26/05/1988', '26 tháng 5 năm 1988')


def test_rows_without_dat_are_returned_unchanged():
    """DAT 이 없는 레코드는 그대로 통과한다."""
    row = _row('Hà Nội là thủ đô.', [('LOC', 'Hà Nội')])
    out, stats = _rewrite(row, _ALWAYS_FULL)
    assert out == row
    assert stats['dat_spans'] == 0


def test_numeric_choice_leaves_the_row_byte_identical():
    """숫자 표기를 고른 자리는 한 자도 안 바뀐다."""
    row = _row('Giấy phép được cấp ngày 26/05/1988.', [('DAT', '26/05/1988')])
    out, _ = _rewrite(row, _ALWAYS_NUMERIC)
    assert out == row


def test_unsorted_entities_raise():
    """오프셋 오름차순이 아닌 레코드는 예외로 멈춘다."""
    row = _row('Cấp ngày 26/05/1988 tại Hà Nội.',
               [('DAT', '26/05/1988'), ('LOC', 'Hà Nội')])
    row['entities'].reverse()
    with pytest.raises(ValueError, match='sorted'):
        _rewrite(row, _ALWAYS_FULL)


def test_span_text_mismatch_raises():
    """gold 표면과 텍스트가 어긋나면 예외로 멈춘다."""
    row = _row('Giấy phép được cấp ngày 26/05/1988.', [('DAT', '26/05/1988')])
    row['entities'][0]['text'] = '27/05/1988'
    with pytest.raises(ValueError, match='mismatch'):
        _rewrite(row, _ALWAYS_FULL)


def test_unparsable_dat_is_left_alone():
    """숫자 표기로 안 읽히는 DAT 은 건드리지 않고 세어만 둔다."""
    row = _row('Cấp ngày mai.', [('DAT', 'mai')])
    out, stats = _rewrite(row, _ALWAYS_FULL)
    assert out['text'] == 'Cấp ngày mai.'
    assert stats['fallback']['unparsed'] == 1


def _many_rows(n):
    return [_row(f'Giấy phép số {i} được cấp ngày 26/05/1988.',
                 [('DAT', '26/05/1988')], rid=f'r{i}')
            for i in range(n)]


def test_target_shares_are_met():
    """형식별 비율이 목표(숫자 60 · 서술형 30 · 월연 10)에 맞는다."""
    out, stats = rw.rewrite_rows(_many_rows(6000), random.Random(42))
    after = stats['after']
    assert 0.57 <= after['numeric'] <= 0.63, after
    assert 0.27 <= after['descriptive_full'] <= 0.33, after
    assert 0.08 <= after['month_year'] <= 0.12, after
    assert stats['before']['numeric'] == 1.0


def test_every_rewritten_span_slices_back_from_the_text():
    """치환 뒤 모든 엔티티 오프셋이 실제 텍스트 슬라이스와 일치한다."""
    out, _ = rw.rewrite_rows(_many_rows(2000), random.Random(7))
    for row in out:
        for ent in row['entities']:
            assert row['text'][ent['start_char']:ent['end_char']] == ent['text']


def test_seed_determinism():
    """같은 seed 는 같은 코퍼스를 낸다."""
    a, _ = rw.rewrite_rows(_many_rows(200), random.Random(3))
    b, _ = rw.rewrite_rows(_many_rows(200), random.Random(3))
    assert a == b
