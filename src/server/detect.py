"""언어 자동 감지 — 배타적 스크립트 양성 감지 + 라틴 폴백.

두 단이다. 먼저 각 언어를 *양성 신호*로 감지한다 — 가나→ `ja`,
한글(음절·자모)→ `ko`, vi-변별 결합부호와 đ→ `vi`. 셋 다 배타적
스크립트라 서로를 오인하지 않는다. 셋이 모두 실패하면 둘째 단으로
넘어가, 라틴 글자가 하나라도 있으면 → `en`, 라틴도 없으면 →
`unsupported` 다.

둘째 단은 감지가 아니라 폴백이다. 영어는 라틴 스크립트에 고유
코드포인트가 없어 스크립트만으로는 변별되지 않으므로, `en` 은 "영어라는
신호를 봤다" 가 아니라 "지원 언어의 신호가 없는데 라틴 글자는 있다" 를
뜻한다. 대가는 오분류다 — 인도네시아어, 로마자로 적은 일본어, 부호를 뗀
베트남어(không dấu)가 모두 `en` 으로 간다. 그 대가를 받는 대신 라틴
텍스트가 빈 결과로 끝나지 않게 했다.

확장: `DETECTORS` 는 (신호 함수, 언어 코드) 순서 레지스트리다. 결정적
스크립트 신호를 가진 언어(키릴→ru …)는 여기에 한 줄 추가하면 되고,
위에서부터 처음 맞은 언어로 확정한다. 라틴 폴백을 이 리스트가 아니라
`detect_lang` 의 루프 *뒤* 에 둔 이유가 그 한 줄이다 — 리스트 끝에 두면
나중에 추가한 감지기가 catch-all 뒤로 밀려, 라틴 글자가 한 자만 섞여도
폴백이 먼저 삼킨다.

한자만 있는 텍스트는 라틴 글자도 없어 `unsupported` 다 — ja·ko 가 한자를
공유하므로 한자의 존재는 어느 쪽도 가리키지 않는다. 가나·한글이라는
*배타적* 스크립트만 신호로 쓰는 이유가 이것이다.

방법론·혼동행렬 근거: docs/reports/language-detection-benchmark.md.
"""

import unicodedata
from typing import Callable, List, Tuple

# 미지원 명시값 — 배타적 스크립트 신호도 라틴 글자도 없을 때
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


# 라틴 폴백 언어 — 양성 신호가 아니라 "지원 언어 신호 부재 + 라틴 글자
# 존재" 를 뜻한다. 이름을 `en` 으로 두되 감지기 목록에는 넣지 않는다.
LATIN_FALLBACK = 'en'

# 라틴 글자 구간 — Basic Latin 영문자 / Latin-1 Supplement 문자 / Latin
# Extended-A·B / Latin Extended Additional / 전각 영문자. Latin-1 을 세
# 구간으로 쪼갠 것은 ×(U+00D7) 과 ÷(U+00F7) 를 빼기 위해서다 — 통짜 C0–FF
# 로 잡으면 수식 기호만 있는 입력이 폴백에 걸린다. Latin Extended
# Additional(1E00–1EFF)을 넣는 것은 베트남어 사전조합 글자(ế·ộ)와 점 부호
# 라틴(ḃ·ẹ)이 거기 살기 때문이다 — 빼면 ASCII 를 한 자도 안 낀 그런 문장이
# "라틴 글자가 있으면 en" 이라는 서술과 달리 unsupported 로 간다.
# IPA 확장·Latin Ext-C/D/E 는 자연어 문장에서 단독으로 오지 않아 뺐다.
# 글자마다 UCD 이름을 조회하는 것보다 정수 비교 아홉 번이 싸서
# `unicodedata.name` 대신 구간을 쓴다.
_LATIN_RANGES = (
    (0x0041, 0x005A),
    (0x0061, 0x007A),
    (0x00C0, 0x00D6),
    (0x00D8, 0x00F6),
    (0x00F8, 0x00FF),
    (0x0100, 0x024F),
    (0x1E00, 0x1EFF),
    (0xFF21, 0xFF3A),
    (0xFF41, 0xFF5A),
)


def _has_latin(text: str) -> bool:
    """텍스트에 라틴 글자가 하나라도 있는지."""
    return _in_ranges(text, _LATIN_RANGES)


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
    """텍스트 언어를 두 단으로 정한다.

    첫째 단은 `DETECTORS` 를 순서대로 적용해 처음 맞은 언어를 낸다 —
    가나→ `ja`, 한글→ `ko`, vi-변별 부호→ `vi`. 셋이 모두 실패하면 둘째
    단으로 내려가, 라틴 글자가 있으면 `LATIN_FALLBACK`, 라틴도 없으면
    `UNSUPPORTED` 다.

    둘째 단이 `DETECTORS` 안이 아니라 루프 뒤에 있는 것은 의도다. 리스트
    끝에 폴백 술어를 두면 나중에 append 하는 감지기가 catch-all 뒤로 밀려
    영영 안 돈다. 루프 뒤에 두면 새 감지기가 언제나 폴백보다 앞에서 돌아,
    "스크립트 언어 추가는 한 줄" 이라는 확장 계약이 유지된다.
    """
    for signal, lang in DETECTORS:
        if signal(text):
            return lang
    if _has_latin(text):
        return LATIN_FALLBACK
    return UNSUPPORTED
