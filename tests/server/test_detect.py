"""언어 자동 감지 단위 테스트 — 언어 고유 신호 양성 감지 + 라틴 폴백.

가나→ja, 한글→ko, vi-변별 코드포인트(horn·hook·dot·đ)→vi, 셋이 모두
실패하고 라틴 글자가 있으면 en, 라틴도 없으면 unsupported. 범-라틴 부호
(fr/pt/de/es/tr)·romaji·không dấu·ASCII 는 vi 신호가 없어 폴백까지
내려가 en 이 된다 — false-accept 0 을 위한 좁은 술어가 치르는 대가다.
한자만 있는 텍스트는 라틴 글자도 없어 unsupported 로 남는다 — ja·ko 가
한자를 공유해 어느 쪽도 가리키지 않는다.
"""

import unicodedata

import pytest

from server import detect
from server.detect import (
    DETECTORS, LATIN_FALLBACK, UNSUPPORTED, _has_latin, detect_lang,
    has_vi_mark)


def test_hiragana_detected_as_ja():
    """히라가나 포함 → ja."""
    assert detect_lang('東京は晴れです。') == 'ja'


def test_katakana_detected_as_ja():
    """가타카나 포함 → ja."""
    assert detect_lang('トヨタ自動車') == 'ja'


def test_hangul_syllables_detected_as_ko():
    """한글 음절 포함 → ko."""
    assert detect_lang('삼성전자는 수원에 있다.') == 'ko'


def test_hangul_jamo_alone_detected_as_ko():
    """자모 단독 표기(ㄱㄴㄷ·호환자모)도 ko — 음절 블록만 보면 놓친다."""
    assert detect_lang('ㅋㅋㅋ') == 'ko'


def test_ko_with_han_detected_as_ko():
    """한글+한자 혼용문은 가나가 없으므로 ko."""
    assert detect_lang('大韓民國 국민은 모두 평등하다.') == 'ko'


def test_ja_wins_over_ko_when_both_scripts_present():
    """가나와 한글이 함께 있으면 ja — DETECTORS 순서가 정한다."""
    assert detect_lang('한국어와 日本語のテキスト') == 'ja'


def test_han_only_text_is_unsupported():
    """한자만 있으면 unsupported — ja·ko 가 공유해 어느 쪽도 아니다."""
    assert detect_lang('東京都千代田区') == UNSUPPORTED


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
    'Viet Nam la mot quoc gia',  # 무부호 vi(không dấu)
    'Watashi wa gakusei desu',   # romaji-ja
])
def test_latin_text_falls_back_to_en(text):
    """언어 고유 신호가 없고 라틴 글자가 있으면 en 폴백.

    여덟 중 실제 영어는 하나뿐이다 — fr·de·pt·es·tr 는 다른 언어이고
    뒤 둘은 베트남어·일본어다. 그래도 en 으로 보내는 것이 폴백이 받은
    대가이며, 그 대가로 라틴 텍스트가 빈 결과로 끝나지 않는다.
    """
    assert detect_lang(text) == LATIN_FALLBACK


@pytest.mark.parametrize('text', [
    '这是一个中文句子',    # zh: 한자(가나·한글 아님)
    '東京都千代田区',      # 한자만
    'Привет мир',         # 키릴
    'مرحبا بالعالم',        # 아랍
    '12345 !!!',          # 숫자·기호만
    '   ',                # 공백만
])
def test_non_latin_residue_stays_unsupported(text):
    """라틴 글자까지 없으면 종전대로 unsupported.

    빈 문자열은 `test_empty_text_is_unsupported` 가 소유하므로 여기 두지
    않는다 — 같은 케이스를 두 곳에 두면 다음에 고칠 때 한쪽만 고친다.
    """
    assert detect_lang(text) == UNSUPPORTED


@pytest.mark.parametrize(
    'text', ['café', 'naïve', 'Zürich', 'mañana', 'João'])
def test_pan_latin_marks_are_not_vi_signal(text):
    """범-라틴 부호(acute·diaeresis·움라우트·tilde-n·tilde)는 vi 신호 아님."""
    assert has_vi_mark(text) is False


def test_mark_sparse_vi_now_falls_back_to_en():
    """변별 부호 없는 짧은 vi('Xin chào')는 en — 한계의 형태가 바뀌었다.

    chào 는 grave(à)뿐이라 horn·hook·dot·đ 가 없어 잡히지 않는다. 좁은
    술어가 pan-Latin false-accept 0 을 보장하는 대가는 그대로인데, 그
    대가의 결말이 unsupported(빈 결과)에서 en 오분류로 바뀌었다 — 이제
    베트남어 문장이 영어 모델을 탄다.
    """
    assert detect_lang('Xin chào') == LATIN_FALLBACK


def test_empty_text_is_unsupported():
    """빈 문자열은 신호가 없어 unsupported."""
    assert detect_lang('') == UNSUPPORTED


def test_registry_extension_is_local(monkeypatch):
    """새 스크립트 언어 추가가 DETECTORS 한 줄 등록으로 끝나는지.

    키릴 신호 detector 를 레지스트리에 추가하면 그전까지 unsupported 이던
    러시아어가 ru 로 감지된다 — 다른 코드 변경 없이 국소 등록만으로.
    레지스트리에 이미 등록된 세 신호의 우선순위는 그대로 유지된다.
    """
    assert detect_lang('Привет мир') == UNSUPPORTED

    def has_cyrillic(text):
        return any(0x0400 <= ord(c) <= 0x04FF for c in text)

    monkeypatch.setattr(
        detect, 'DETECTORS', detect.DETECTORS + [(has_cyrillic, 'ru')])
    assert detect_lang('Привет мир') == 'ru'
    assert detect_lang('東京は') == 'ja'
    assert detect_lang('안녕하세요') == 'ko'
    assert detect_lang('Hà Nội đô') == 'vi'


def test_registry_detector_wins_over_latin_fallback(monkeypatch):
    """새로 등록한 감지기가 라틴 폴백보다 먼저 돈다 — 배선 판별자.

    폴백을 `DETECTORS` 끝에 술어로 넣으면 나중에 append 한 감지기가
    catch-all 뒤로 밀려, 라틴 글자가 섞인 입력을 en 이 먼저 삼킨다. 폴백을
    루프 *뒤* 에 두면 그 일이 없다. 기존 확장 테스트는 라틴이 없는 키릴만
    써서 두 배선을 구별하지 못하므로, 여기서는 키릴에 라틴을 섞는다.
    """
    def has_cyrillic(text):
        return any(0x0400 <= ord(c) <= 0x04FF for c in text)

    monkeypatch.setattr(
        detect, 'DETECTORS', detect.DETECTORS + [(has_cyrillic, 'ru')])
    assert detect_lang('Привет мир and hello') == 'ru'


def test_detectors_hold_only_exclusive_script_signals():
    """레지스트리는 언어 고유의 양성 신호만 담는다.

    누가 `(_has_latin_ish, 'id')` 같은 폴백성 술어를 append 하면 배선
    판별자는 키릴을 쓰므로 여전히 통과하지만 이 단언이 실패한다. 선택지 B
    가 조용히 A 로 되돌아가는 유일한 경로를 닫는 한 줄이다.
    """
    for signal, _lang in DETECTORS:
        assert signal('hello world') is False


@pytest.mark.parametrize('text,expected', [
    ('a', True),        # Basic Latin
    ('Ø', True),        # U+00D8 Latin-1 Supplement
    ('ğ', True),        # U+011F Latin Extended-A
    ('ế', True),        # U+1EBF Latin Extended Additional (vi 사전조합)
    ('ḃ', True),        # U+1E02 같은 블록의 점 부호 라틴
    ('Ａ', True),        # U+FF21 전각 라틴
    ('東', False),
    ('ア', False),
    ('가', False),
    ('П', False),
    ('٣', False),
    ('×', False),       # U+00D7 — Latin-1 구간 안이지만 글자가 아니다
    ('÷', False),       # U+00F7 — 같은 이유
    ('', False),
])
def test_has_latin_unit(text, expected):
    """`_has_latin` 은 라틴 *글자* 만 본다 — 수식 기호는 아니다.

    ×·÷ 가 False 여야 한다는 것이 Latin-1 Supplement 를 통짜가 아니라 세
    구간으로 쪼갠 이유의 집행이다. 통짜면 '× ÷' 같은 입력이 en 으로 가
    영어 모델을 탄다.
    """
    assert _has_latin(text) is expected
