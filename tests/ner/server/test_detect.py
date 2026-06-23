"""언어 자동 감지 단위 테스트 — 가나→ja, 한자 혼입 vi→vi."""

from ner.server.detect import detect_lang


def test_hiragana_detected_as_ja():
    """히라가나(は·です) 포함 → ja."""
    assert detect_lang('東京は晴れです。') == 'ja'


def test_katakana_detected_as_ja():
    """가타카나 포함 → ja."""
    assert detect_lang('トヨタ自動車') == 'ja'


def test_vietnamese_detected_as_vi():
    """베트남어 성조부호 라틴 → vi."""
    assert detect_lang('Hà Nội là thủ đô.') == 'vi'


def test_vi_with_han_not_misdetected_as_ja():
    """vi 코퍼스의 한자 혼입은 가나가 없으므로 vi (오분류 방지)."""
    assert detect_lang('Tả Đê Hầu thiền vu ( 且鞮侯單于 )') == 'vi'


def test_plain_ascii_defaults_to_vi():
    """가나·성조부호 없는 ASCII 는 기본값 vi."""
    assert detect_lang('hello world') == 'vi'


def test_default_override():
    """default 인자로 폴백 언어를 바꿀 수 있다."""
    assert detect_lang('hello', default='ja') == 'ja'
