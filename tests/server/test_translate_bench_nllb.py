"""번역 벤치 NLLB 엔진의 모델 무의존 로직 — 문장 분할 + 생성 붕괴 탐지.

두 로직 모두 실측에서 나온 실패를 막는 장치라 회귀가 조용히 지나가면 안 된다.
문장 분할이 새면 NLLB 가 뒷문장을 통째로 버리고, 붕괴 탐지가 없으면 한 어절을
수십 번 되풀이한 출력이 평균 지표에 흡수돼 보이지 않는다. 모델 로드가 필요한
경로는 여기서 다루지 않는다.
"""

from server.scripts.translate_bench.nllb import (
    ASCII_SENTINEL,
    max_token_repeat,
    split_sentences,
)
from server.translate import DEFAULT_SENTINEL, _mask, _restore


# ---------- 문장 분할 ----------

def test_ja_splits_on_fullwidth_stop():
    """ja 는 구두점 뒤에 공백이 없어도 문장이 갈린다."""
    assert split_sentences('今日は晴れです。明日は雨だ。', 'ja') == [
        '今日は晴れです。', '明日は雨だ。']


def test_vi_splits_on_period_space():
    """vi 는 마침표 + 공백이 기본 경계다."""
    assert split_sentences('Hôm nay trời đẹp. Ngày mai sẽ mưa.', 'vi') == [
        'Hôm nay trời đẹp.', 'Ngày mai sẽ mưa.']


def test_single_letter_abbreviation_is_not_a_boundary():
    """한 글자 약어 뒤 마침표는 문장 끝이 아니다 — 되붙인다."""
    assert split_sentences('Ông T. Nguyen đã nói vậy.', 'vi') == [
        'Ông T. Nguyen đã nói vậy.']


def test_decimal_and_year_do_not_split():
    """소수점·연도처럼 뒤에 공백이 없는 마침표는 경계가 아니다."""
    assert split_sentences('Giá là 3.14 triệu vào 2019.', 'vi') == [
        'Giá là 3.14 triệu vào 2019.']


def test_sentinel_survives_splitting():
    """마스킹된 sentinel 이 문장 분할로 쪼개지지 않는다."""
    phone = '0904-123-456'
    text = f'Liên hệ {phone}. Cảm ơn.'
    p = text.index(phone)
    spans = [{'label': 'PHONE', 'start_char': p, 'end_char': p + len(phone),
              'text': phone}]
    masked, id2val = _mask(text, spans, ASCII_SENTINEL)
    parts = split_sentences(masked, 'vi')
    token = ASCII_SENTINEL.token(0)
    assert any(token in part for part in parts)     # 한 조각 안에 온전히 있다
    restored, n_ok, n_drop = _restore(' '.join(parts), id2val, ASCII_SENTINEL)
    assert phone in restored and (n_ok, n_drop) == (1, 0)


def test_empty_and_unsplittable_text_returns_whole():
    """못 나누는 입력은 통짜로 돌려준다(빈 배치로 모델을 부르지 않는다)."""
    assert split_sentences('공백없는한문장', 'ja') == ['공백없는한문장']
    assert split_sentences('mot cau', 'vi') == ['mot cau']


# ---------- 생성 붕괴 ----------

def test_repeat_detector_flags_collapse():
    """같은 어절 반복은 반복 횟수로 잡힌다."""
    assert max_token_repeat('본격적으로 ' * 40) == 40


def test_repeat_detector_ignores_normal_text():
    """정상 문장은 낮은 값이라 붕괴 기준(5회)에 못 미친다."""
    assert max_token_repeat('오늘은 날씨가 좋고 내일은 비가 온다') < 5
    assert max_token_repeat('') == 0


def test_repeat_detector_counts_only_consecutive():
    """떨어져 나온 같은 어절은 반복으로 세지 않는다."""
    assert max_token_repeat('가 나 가 다 가 라 가') == 1


# ---------- 엔진 속성 ----------

def test_ascii_sentinel_differs_from_default_only_in_brackets():
    """NLLB 표기는 괄호만 다르고 nonce·번호 부분은 기본과 같다."""
    ascii_token = ASCII_SENTINEL.token(7)
    default_token = DEFAULT_SENTINEL.token(7)
    assert ascii_token == f'[{default_token[1:-1]}]'
