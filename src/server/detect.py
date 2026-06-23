"""언어 자동 감지 — ja/vi 판별.

판별 규칙: 텍스트에 히라가나·가타카나가 하나라도 있으면 `ja`, 아니면
`vi`. 가나는 일본어 전용이라 베트남어엔 나타나지 않으므로 견고하다. 반대로
베트남어 코퍼스에는 한자(chữ Hán)가 섞일 수 있어(예: 고유명사 병기)
"한자 존재 → ja" 식 판별은 vi 텍스트를 오분류한다 — 그래서 한자가 아니라
가나만 ja 신호로 삼는다.

한계: 가나·베트남어 성조부호가 모두 없는 한자/ASCII 전용 텍스트는 기본값
(`vi`)으로 떨어진다. 명시 `lang` 이 주어지면 감지를 우회한다(API 레이어).
"""

# 히라가나 / 가타카나 / 가타카나 음성확장 / 반각 가타카나 코드포인트 범위
_KANA_RANGES = (
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0x31F0, 0x31FF),   # Katakana Phonetic Extensions
    (0xFF66, 0xFF9D),   # Halfwidth Katakana
)


def _has_kana(text: str) -> bool:
    """텍스트에 가나(히라가나·가타카나) 문자가 하나라도 있는지."""
    for ch in text:
        cp = ord(ch)
        for lo, hi in _KANA_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def detect_lang(text: str, default: str = 'vi') -> str:
    """텍스트의 언어를 추정한다 — 가나 존재 시 `ja`, 아니면 `default`."""
    return 'ja' if _has_kana(text) else default
