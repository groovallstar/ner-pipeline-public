"""VI DAT 표면형 × 문맥 probe — 실제 `/data/ner/vi` 모델 의존(로컬 전용).

배포 test 는 주입기가 만든 분포 위에서만 재는데, vi 의 DAT 은 사람 gold 가 아니라
PII 합성 주입분이라 그 분포가 곧 주입기의 출력 형식이다. 주입기가 `15/03/2024` 류
숫자 표기만 내므로 `ngày 15 tháng 3 năm 2024` 같은 베트남어 일상 표기는 학습에도
평가에도 한 번도 안 나온다. metric 이 멀쩡한 채로 실사용 입력에서 날짜가 조각나는
이유가 이것이다.

이 probe 는 표면형 넷과 문맥 넷을 교차해, 서버 추론 경로(`LangModel.predict_many`)가
날짜를 경계까지 한 덩어리로 잡는지 strict 로 잰다. 조각 매치는 실패로 센다 —
`15` 만 잡히면 마스킹이 `tháng 3 năm 2024` 를 그대로 흘린다.

**probe 는 수정 전에 고정했다.** 통과하도록 문장을 고르면 이 기준이 자기 참조가 된다.

## span 경계

앞에 붙는 시간 낱말은 그것을 떼고도 날짜로 읽히면 span 밖에 둔다. `ngày 15/03/2024`
3,708 자리가 예외 없이 `15/03/2024` 만 달고 있는 현 코퍼스 관례가 이것이다. 반대로
떼면 날짜가 아니게 되는 자리는 span 안에 넣는다 — `tháng 3 năm 2024` 에서 `tháng` 을
빼면 `3 năm 2024` 가 되어 날짜로 안 읽힌다.

| 표면 | span | 앞 낱말을 뗀 나머지 |
|---|---|---|
| `ngày 15 tháng 3 năm 2024` | `15 tháng 3 năm 2024` | 날짜로 읽힌다 |
| `ngày 15/03/2024` | `15/03/2024` | 날짜로 읽힌다 |
| `tháng 3 năm 2024` | 전체 | `3 năm 2024` — 아니다 |
| `năm 2024` | 전체 | `2024` — 아니다 |

## 하한을 거는 자리

`năm Y` 단독은 하한 밖이다. 베트남어 사건명이 `Bầu cử liên bang Úc năm 2004` 처럼
연도로 끝나는 꼴이 흔해 그 자리의 옳은 라벨이 EVT 이고, issue-265 가 이 형식을 주입
대상에서 뺐다. 수치는 하한 있는 테스트의 실패 메시지에 함께 실어 나중에 이 판단을
되짚을 실측으로 남긴다.
"""

import os
from collections import Counter

import pytest

from server.config import ServerConfig
from server.inference import LangModel

_CONFIG = ServerConfig.from_env()
_VI_DIR = _CONFIG.model_dir('vi')

# 최근·과거 연도와 한 자리·두 자리 월일을 섞는다. 주입기의 연도 범위가
# 1950~2005 라 2024·2026 이 학습 밖인지도 이 목록이 가른다.
DATES = [
    (15, 3, 2024), (1, 12, 1988), (26, 5, 1973),
    (8, 10, 2026), (30, 9, 2001), (3, 7, 1995),
]


def _descriptive_full(d, m, y):
    return f'ngày {d} tháng {m} năm {y}', 5


def _numeric_ngay(d, m, y):
    return f'ngày {d:02d}/{m:02d}/{y:04d}', 5


def _month_year(d, m, y):
    return f'tháng {m} năm {y}', 0


def _year_only(d, m, y):
    return f'năm {y}', 0


# 하한을 거는 형식 셋. `ngày D/M/Y` 는 이미 되는 형식이라 무회귀 감시다.
BOUNDED_FORMS = {
    'descriptive_full': _descriptive_full,
    'numeric_ngay': _numeric_ngay,
    'month_year': _month_year,
}

# 하한 없이 수치만 보는 형식.
OBSERVED_FORMS = {'year_only': _year_only}

# 문맥마다 틀 둘을 날짜 순서대로 번갈아 쓴다. `start` 는 첫 글자를 올린다.
CONTEXTS = {
    'mid': ['Cuộc họp diễn ra vào {d} tại Hà Nội.',
            'Hợp đồng được ký vào {d} tại trụ sở công ty.'],
    'end': ['Giấy phép được cấp {d}.', 'Hồ sơ phải nộp trước {d}.'],
    'start': ['{d} là hạn chót nộp hồ sơ.',
              '{d} công ty chính thức đi vào hoạt động.'],
    'alone': ['{d}', '{d}'],
}

MIN_RECALL = 0.90

# 날짜가 아닌 숫자. DAT 이 하나라도 나오면 모델이 날짜 모양이 아니라 네 자리
# 숫자를 보고 있다는 뜻이다.
NEGATIVE_TEXTS = [
    'Công ty có 2024 nhân viên đang làm việc.',
    'Phòng số 1988 nằm ở tầng ba.',
    'Cuốn sách này dày 1973 trang.',
    'Mã sản phẩm là 2026 và còn hàng.',
]

# 사건명이 연도로 끝나는 자리. 옳은 라벨은 EVT 이고 DAT 이 아니다. issue-265 가
# `năm Y` 를 주입에서 뺀 이유가 이 자리이며, 재학습이 이 경계를 밀지 않는지 본다.
EVENT_TAIL_TEXTS = [
    'Chiến dịch mùa xuân năm 1975 được ghi nhớ đến hôm nay.',
    'Cuộc Tổng tuyển cử ở Thái Lan năm 2007 đã kết thúc.',
    'Bầu cử liên bang Úc năm 2004 diễn ra suôn sẻ.',
    'Cuộc đột kích Abu Kamal năm 2008 gây nhiều tranh cãi.',
]


def _probe(forms=BOUNDED_FORMS):
    """(형식, 문맥, 입력, span 시작, span 끝) 목록."""
    items = []
    for form, build in forms.items():
        for i, (d, m, y) in enumerate(DATES):
            surface, span_offset = build(d, m, y)
            for ctx, templates in CONTEXTS.items():
                text = templates[i % 2]
                if ctx == 'start':
                    surface_in_text = surface[0].upper() + surface[1:]
                else:
                    surface_in_text = surface
                filled = text.format(d=surface_in_text)
                base = filled.index(surface_in_text)
                start = base + span_offset
                end = base + len(surface_in_text)
                items.append((form, ctx, filled, start, end))
    return items


def _model():
    return LangModel('vi', _VI_DIR, _CONFIG.thresholds_path('vi'),
                     _CONFIG.max_length)


def _strict_recall(model, items):
    """(키별 recall, 놓친 입력 설명) — 키는 'all'·형식·문맥이다."""
    predictions = model.predict_many([text for _, _, text, _, _ in items])
    hits, totals, misses = Counter(), Counter(), []
    for (form, ctx, text, start, end), spans in zip(items, predictions):
        found = [(s['start_char'], s['end_char'])
                 for s in spans if s['label'] == 'DAT']
        ok = (start, end) in found
        for key in ('all', form, ctx):
            hits[key] += ok
            totals[key] += 1
        if not ok:
            misses.append((text, [text[s:e] for s, e in found]))
    recall = {key: hits[key] / totals[key] for key in totals}
    report = ' '.join(f'{k}={recall[k]:.3f}' for k in sorted(recall))
    detail = '\n'.join(f'  {t!r} -> {f}' for t, f in misses)
    return recall, f'{report}\n{detail}'


def test_probe_is_the_full_cross():
    """형식 3 × 날짜 6 × 문맥 4 = 72 문장 — 칸이 빠지면 recall 이 부풀어 오른다."""
    items = _probe()
    assert len(items) == 72
    assert Counter(form for form, *_ in items) == {
        'descriptive_full': 24, 'numeric_ngay': 24, 'month_year': 24}
    assert Counter(ctx for _, ctx, *_ in items) == {
        'mid': 18, 'end': 18, 'start': 18, 'alone': 18}


def test_probe_spans_slice_back_to_the_date():
    """기대 span 이 실제 입력에서 날짜 부분과 정확히 맞는다.

    문맥 틀을 고치다 offset 이 어긋나면 recall 이 조용히 0 이 되므로 따로 잠근다.
    """
    for form, ctx, text, start, end in _probe({**BOUNDED_FORMS,
                                               **OBSERVED_FORMS}):
        span = text[start:end]
        assert span, (form, ctx, text)
        assert not span.lower().startswith('ngày'), (form, ctx, span)
        assert span[-4:].isdigit(), (form, ctx, span)
        if form in ('month_year', 'year_only'):
            assert span.lower().startswith(('tháng', 'năm')), (form, span)


def test_negative_probe_has_no_date():
    """음성 probe 입력에 날짜가 없다 — probe 자체의 무결성 검사다."""
    for text in NEGATIVE_TEXTS:
        assert 'ngày' not in text.lower(), text
        assert 'tháng' not in text.lower(), text


_needs_model = pytest.mark.skipif(not os.path.isdir(_VI_DIR),
                                  reason=f'vi model not present at {_VI_DIR}')


@_needs_model
def test_vi_dat_probe_strict_recall():
    """서술형·숫자·월연 세 형식에 하한을 건다. `năm Y` 는 수치만 함께 싣는다."""
    model = _model()
    recall, report = _strict_recall(model, _probe())
    _, observed = _strict_recall(model, _probe(OBSERVED_FORMS))
    below = [key for key in ('all', 'descriptive_full', 'numeric_ngay',
                             'month_year', 'mid', 'end', 'start', 'alone')
             if recall[key] < MIN_RECALL]
    assert not below, (f'below {MIN_RECALL}: {below}\n{report}\n'
                       f'year_only(관찰용): {observed}')


@_needs_model
def test_vi_dat_negative_probe_has_no_dat():
    """날짜가 아닌 네 자리 숫자를 DAT 으로 잡지 않는다."""
    predictions = _model().predict_many(NEGATIVE_TEXTS)
    found = [(text, s['text'])
             for text, spans in zip(NEGATIVE_TEXTS, predictions)
             for s in spans if s['label'] == 'DAT']
    assert not found, found


@_needs_model
def test_vi_dat_does_not_eat_event_tail_year():
    """사건명 꼬리의 연도를 DAT 으로 떼어내지 않는다.

    `năm Y` 를 주입에서 뺀 판단이 지켜지는 자리다. 여기서 DAT 이 나오면 재학습이
    EVT 경계를 밀었다는 뜻이다.
    """
    predictions = _model().predict_many(EVENT_TAIL_TEXTS)
    found = [(text, s['text'])
             for text, spans in zip(EVENT_TAIL_TEXTS, predictions)
             for s in spans if s['label'] == 'DAT']
    assert not found, found
