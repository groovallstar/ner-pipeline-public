"""언어 자동 감지 단위 테스트 — ja·vi 양성 감지 + 미지원 명시.

가나→ja, vi-변별 코드포인트(horn·hook·dot·đ)→vi, 그 외→unsupported.
범-라틴 부호(fr/pt/de/es/tr)·romaji·không-dấu·ASCII 는 vi 신호가 없어
unsupported 로 떨어진다(false-accept 0 을 위한 좁은 술어의 수용된 한계).
"""

import unicodedata

import pytest

from server import detect
from server.detect import (
    UNSUPPORTED, detect_lang, has_vi_mark)


def test_hiragana_detected_as_ja():
    """히라가나 포함 → ja."""
    assert detect_lang('東京は晴れです。') == 'ja'


def test_katakana_detected_as_ja():
    """가타카나 포함 → ja."""
    assert detect_lang('トヨタ自動車') == 'ja'


def test_vi_horn_detected_as_vi():
    """horn(ư) 보유 → vi (người)."""
    assert detect_lang('Tôi là người Việt Nam.') == 'vi'


def test_vi_dot_below_and_d_stroke_detected_as_vi():
    """dot-below(ộ)·đ 보유 → vi (Hà Nội là thủ đô)."""
    assert detect_lang('Hà Nội là thủ đô.') == 'vi'


def test_vi_with_han_not_misdetected_as_ja():
    """vi(đ 보유)에 한자 혼입돼도 가나가 없으므로 vi (오분류 방지)."""
    assert detect_lang('Tả Đê Hầu thiền vu ( 且鞮侯單于 )') == 'vi'


def test_vi_decomposed_nfd_input_detected_as_vi():
    """분해조합(NFD) 입력도 사전조합과 동일하게 vi 로 잡는다."""
    nfd = unicodedata.normalize('NFD', 'Việt')
    assert detect_lang(nfd) == 'vi'


@pytest.mark.parametrize('text', [
    'hello world',               # ASCII 영어
    'Bonjour à tous, ça va ?',   # fr: grave·cedilla(범-라틴)
    'Grüße aus München',         # de: 움라우트·ß
    'Olá, coração, São Paulo',   # pt: acute·tilde·circumflex
    'El niño está aquí',         # es: tilde-n·acute
    'Doğum günü kutlu olsun',    # tr: breve·dotless-i·움라우트
    '안녕하세요 여러분',          # ko: 한글
    '这是一个中文句子',          # zh: 한자(가나 아님)
    'Viet Nam la mot quoc gia',  # 무부호 vi(không dấu)
    'Watashi wa gakusei desu',   # romaji-ja
])
def test_no_ja_vi_signal_is_unsupported(text):
    """가나·vi-변별 부호가 없으면 모두 unsupported(en 단정 안 함)."""
    assert detect_lang(text) == UNSUPPORTED


@pytest.mark.parametrize(
    'text', ['café', 'naïve', 'Zürich', 'mañana', 'João'])
def test_pan_latin_marks_are_not_vi_signal(text):
    """범-라틴 부호(acute·diaeresis·움라우트·tilde-n·tilde)는 vi 신호 아님."""
    assert has_vi_mark(text) is False


def test_mark_sparse_vi_is_accepted_limitation():
    """변별 부호 없는 짧은 vi('Xin chào')는 unsupported — 수용된 한계.

    chào 는 grave(à)뿐이라 horn·hook·dot·đ 가 없어 잡히지 않는다. 좁은
    술어가 pan-Latin false-accept 0 을 보장하는 대가의 경계 사례.
    """
    assert detect_lang('Xin chào') == UNSUPPORTED


def test_empty_text_is_unsupported():
    """빈 문자열은 신호가 없어 unsupported."""
    assert detect_lang('') == UNSUPPORTED


def test_registry_extension_is_local(monkeypatch):
    """새 스크립트 언어 추가가 DETECTORS 한 줄 등록으로 끝나는지.

    한글 신호 detector 를 레지스트리에 추가하면 그전까지 unsupported 이던
    한국어가 ko 로 감지된다 — 다른 코드 변경 없이 국소 등록만으로. 가나·vi
    의 기존 우선순위는 그대로 유지된다.
    """
    assert detect_lang('안녕하세요') == UNSUPPORTED

    def has_hangul(text):
        return any(0xAC00 <= ord(c) <= 0xD7A3 for c in text)

    monkeypatch.setattr(
        detect, 'DETECTORS', detect.DETECTORS + [(has_hangul, 'ko')])
    assert detect_lang('안녕하세요') == 'ko'
    assert detect_lang('東京は') == 'ja'
    assert detect_lang('Hà Nội đô') == 'vi'
