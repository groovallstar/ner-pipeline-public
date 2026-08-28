"""언어 자동 감지 — ja·ko·vi 양성 감지 + 미지원 명시.

각 언어를 *양성 신호*로 감지한다 — 가나(일본어 전용)·한글(한국어 전용)·
vi-변별 코드포인트(베트남어 전용). 어느 신호도 없으면 `unsupported`
(셋 다 아님)로 명시한다. 'vi 기본값으로 떨어뜨리지' 않는다 — 무부호 ASCII 는
en·id·romaji·không-dấu-vi·노이즈 등 무엇이든 가능하므로 vi 단정은 거짓을
만든다.

확장: `DETECTORS` 는 (신호 함수, 언어 코드) 순서 레지스트리다. 결정적
스크립트 신호를 가진 언어(가나→ja, 한글→ko, 키릴→ru …)는 여기에 한 줄
추가하면 되고, 위에서부터 처음 맞은 언어로 확정한다. 라틴 스크립트면서
고유 코드포인트가 없는 언어(en·id)는 스크립트만으로 변별 불가라 양성 감지
대상이 아니다(설계상 unsupported).

한자만 있는 텍스트는 어느 신호에도 안 걸려 `unsupported` 다 — ja·ko 가 한자를
공유하므로 한자의 존재는 어느 쪽도 가리키지 않는다. 가나·한글이라는 *배타적*
스크립트만 신호로 쓰는 이유가 이것이다.

vi-변별 술어는 가나·한글이 결정적인 것과 달리 *부호*에 의존하므로, 부호를 뗀
베트남어(không dấu)는 잡지 못한다 — 수용된 한계로 unsupported 가 된다.
방법론·혼동행렬 근거: docs/reports/language-detection-benchmark.md.
"""

import unicodedata
from typing import Callable, List, Tuple

# 미지원 명시값 — ja·ko·vi 어느 신호도 없을 때
UNSUPPORTED = 'unsupported'

# 히라가나 / 가타카나 / 음성확장 / 반각 가타카나 — 일본어 전용 신호
_KANA_RANGES = (
    (0x3040, 0x309F),
    (0x30A0, 0x30FF),
    (0x31F0, 0x31FF),
    (0xFF66, 0xFF9D),
)

# 한글 — 자모 / 호환자모 / 자모확장A / 음절 / 자모확장B / 반각 한글.
# 음절(가–힣)만 보면 자모 단독 표기(ㄱㄴㄷ·옛한글)를 놓치므로 블록 전체를
# 잡는다. 반각 한글(FFA0–FFDC)은 반각 가타카나(FF66–FF9D)와 겹치지 않는다.
_HANGUL_RANGES = (
    (0x1100, 0x11FF),
    (0x3130, 0x318F),
    (0xA960, 0xA97F),
    (0xAC00, 0xD7A3),
    (0xD7B0, 0xD7FF),
    (0xFFA0, 0xFFDC),
)

# vi-변별 결합부호(NFD 분해 후): horn·hook-above·dot-below. 라틴 문자 중
# 베트남어 전용이라, 범-라틴 부호(circumflex·acute·grave·tilde·움라우트·
# cedilla)와 달리 fr/pt/de/es/tr 를 false-accept 하지 않는다.
_VI_COMBINING = frozenset({
    '\N{COMBINING HORN}',        # U+031B (ơ ư)
    '\N{COMBINING HOOK ABOVE}',  # U+0309 (ả ẻ ỉ …)
    '\N{COMBINING DOT BELOW}',   # U+0323 (ạ ẹ ị …)
})
# vi-변별 단독 문자: đ Đ — NFD 로 분해되지 않는 독립 자모
_VI_LETTERS = frozenset({'đ', 'Đ'})


def _in_ranges(text: str, ranges: Tuple[Tuple[int, int], ...]) -> bool:
    """텍스트에 주어진 코드포인트 구간의 글자가 하나라도 있는지."""
    for ch in text:
        cp = ord(ch)
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return True
    return False


def _has_kana(text: str) -> bool:
    """텍스트에 가나(히라가나·가타카나)가 하나라도 있는지."""
    return _in_ranges(text, _KANA_RANGES)


def _has_hangul(text: str) -> bool:
    """텍스트에 한글(음절 또는 자모)이 하나라도 있는지."""
    return _in_ranges(text, _HANGUL_RANGES)


def has_vi_mark(text: str) -> bool:
    """vi-변별 코드포인트(horn·hook·dot 결합부호 또는 đ)가 있는지.

    NFD 로 정규화해 결합부호를 분리한 뒤 검사하므로 사전조합(ạ)·분해조합
    (a+◌̣) 입력을 모두 동일하게 잡는다. 범-라틴 부호만 가진 텍스트
    (fr/pt/de/es/tr)는 걸리지 않는다.
    """
    nfd = unicodedata.normalize('NFD', text)
    for ch in nfd:
        if ch in _VI_COMBINING or ch in _VI_LETTERS:
            return True
    return False


# 언어 감지기 레지스트리 — (신호 함수, 언어 코드). 위에서부터 처음 맞은
# 언어로 확정한다. 셋의 신호는 서로소라 순서가 결과를 바꾸지 않지만, 코드
# 스위칭(한 문장에 둘 이상)에서는 위쪽이 이긴다 — 가나+한글은 ja, 한글+vi
# 부호는 ko. 스크립트 신호가 부호 신호보다 강하다는 기존 원칙 그대로다.
DETECTORS: List[Tuple[Callable[[str], bool], str]] = [
    (_has_kana, 'ja'),
    (_has_hangul, 'ko'),
    (has_vi_mark, 'vi'),
]


def detect_lang(text: str) -> str:
    """텍스트 언어를 양성 감지한다 — `ja`·`ko`·`vi` 또는 `unsupported`.

    `DETECTORS` 를 순서대로 적용해 처음 맞은 언어를 반환하고, 어느 신호도
    없으면 `unsupported`. 미지원을 `en` 등으로 단정하지 않는다 —
    스크립트만으로는 지원 언어 신호의 *부재*만 알 수 있다.
    """
    for signal, lang in DETECTORS:
        if signal(text):
            return lang
    return UNSUPPORTED
