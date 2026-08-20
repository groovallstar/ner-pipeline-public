"""사전 토큰화된 OntoNotes 토큰 배열을 자연스러운 영문으로 되돌린다.

OntoNotes 는 자연문이 아니라 토큰 배열로 배포되므로(`['time', '-',
'consuming']`), 이 저장소의 공용 통화인 char-offset span 을 만들려면 문장을
먼저 복원해야 한다. 영어는 이 복원 규칙이 결정적이라 가능하다 — 띄어쓰기가
없는 일본어였다면 성립하지 않는다.

규칙은 추측이 아니라 코퍼스 76,714 문장 전수 측정에서 나왔다.

- `-` 는 10,659 회 중 **10,403 회(97.6%)가 단어와 단어 사이**다
  (`sales - tax`·`time - consuming`). 그래서 양쪽에 붙인다. `/` 도 같다.
  **`&` 는 반대라 띄어 쓴다** — 아래 `_TIGHT_INFIX` 주석에 근거가 있다.
- PTB 브래킷 이스케이프(`-LRB-` 1,792 · `-RRB-` 1,813 등)는 **엔티티 안에도
  나타난다**(각 50·48 회). 실제 영문에 없는 표기라 되돌리지 않으면 그대로
  학습 데이터에 남는다.
- 발화 전사 마커 `/.`(9,485) `/?`(1,010) `/-`(326) 과 비유창성 마커
  `%um`(694) `%uh`(639) 는 방송·전화 대화 도메인의 표기다.

## 문자를 바꾸는 정규화와 검사의 관계

`_normalize` 는 **문자 자체를 바꾼다**(`-LRB-` → `(`). 그래서 엔티티↔토큰
대조 검사도 같은 정규화를 거친 토큰과 비교해야 하고, 그만큼 그 검사는
정규화 표 자체는 검증하지 못한다. 검사가 잡는 것은 span 인덱스 오류와
offset 계산 오류이고, 정규화 표(8 항목)는 단위 테스트가 따로 고정한다.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

# PTB 브래킷 이스케이프 — 실제 영문에 없는 표기라 반드시 되돌린다.
_PTB_UNESCAPE = {
    '-LRB-': '(', '-RRB-': ')',
    '-LSB-': '[', '-RSB-': ']',
    '-LCB-': '{', '-RCB-': '}',
}

# 방향이 있는 따옴표 → 대칭 따옴표.
_OPEN_QUOTE = frozenset({'``', '`'})
_CLOSE_QUOTE = frozenset({"''"})

# 발화 전사의 문장부호 마커.
_SPEECH_MARKER = {'/.': '.', '/?': '?', '/-': '-'}

# 비유창성 마커 — 선행 `%` 만 떼면 자연스러운 표기가 된다.
_DISFLUENCY = re.compile(r'^%[a-z]{1,3}$')

# 앞 토큰에 붙는 것 (앞에 공백을 두지 않는다).
_ATTACH_LEFT = frozenset({
    ',', '.', ';', ':', '!', '?', '%', ')', ']', '}',
    '...', '..', '/.', '/?', '-RRB-', '-RSB-', '-RCB-',
    "'s", "'re", "'ve", "'ll", "'d", "'m", "n't", "'",
    "'S", "'RE", "'VE", "'LL", "'D", "'M", "N'T",
})

# 뒤 토큰이 붙는 것 (뒤에 공백을 두지 않는다).
_ATTACH_RIGHT = frozenset({
    '(', '[', '{', '$', '#', '-LRB-', '-LSB-', '-LCB-',
})

# 양쪽이 단어일 때만 양쪽에 붙는 중위 기호.
#
# `&` 는 **일부러 뺐다.** 단독 `&` 토큰 345 건의 문맥은 `Fleet & Leasing`·
# `Dooling & Co.`·`Liu & Tong` 처럼 띄어 쓰는 접속이고, 붙여 쓰는 `AT&T`·
# `S&P` 류는 애초에 쪼개지지 않아 한 토큰(143 건)으로 온다. 붙이면 실제
# 영문에 없는 `Fleet&Leasing` 이 만들어진다.
_TIGHT_INFIX = frozenset({'-', '/'})

_WORDLIKE = re.compile(r'[A-Za-z0-9]')


def normalize_token(token: str) -> str:
    """토큰 하나를 실제 영문에 쓰이는 표기로 되돌린다.

    엔티티↔토큰 대조 검사가 같은 함수를 써야 양쪽이 같은 자로 비교된다.
    """
    if token in _PTB_UNESCAPE:
        return _PTB_UNESCAPE[token]
    if token in _OPEN_QUOTE or token in _CLOSE_QUOTE:
        return '"'
    if token in _SPEECH_MARKER:
        return _SPEECH_MARKER[token]
    if _DISFLUENCY.match(token):
        return token[1:]
    return token


def _is_wordlike(token: str | None) -> bool:
    """영숫자를 포함한 토큰인가 — 중위 기호 판정에 쓴다."""
    return bool(token and _WORDLIKE.search(token))


def _needs_space(
    prev: str | None, cur: str, nxt: str | None, quote_open: bool,
) -> bool:
    """`cur` 앞에 공백을 둘지 정한다."""
    if prev is None:
        return False
    # 중위 기호는 양쪽이 단어일 때만 붙는다 (`sales - tax` → `sales-tax`).
    if cur in _TIGHT_INFIX:
        return not (_is_wordlike(prev) and _is_wordlike(nxt))
    if prev in _TIGHT_INFIX:
        # 앞이 붙는 중위 기호였으면 이쪽도 붙는다.
        return False
    if cur in _ATTACH_LEFT:
        return False
    if prev in _ATTACH_RIGHT:
        return False
    # 대칭 따옴표는 여는 쪽이면 뒤가 붙고 닫는 쪽이면 앞이 붙는다.
    if cur in _CLOSE_QUOTE or (cur == '"' and quote_open):
        return False
    if prev in _OPEN_QUOTE or (prev == '"' and quote_open):
        return False
    return True


def detokenize(tokens: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    """토큰 배열을 자연문으로 잇고 토큰별 char-offset 을 함께 돌려준다.

    반환하는 offset 은 반-개구간 `[start, end)` 이며, `text[start:end]` 가
    그 토큰의 정규화된 표기와 정확히 같다. span 을 만들 때 첫 토큰의 start 와
    마지막 토큰의 end 를 그대로 쓰면 된다.
    """
    chunks: list[str] = []
    offsets: list[tuple[int, int]] = []
    pos = 0
    quote_open = False
    # 앞 토큰의 *원본* 표기를 본다 — 공백 판단이 원본 형태에 걸려 있다.
    prev_raw: str | None = None

    for i, raw in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if _needs_space(prev_raw, raw, nxt, quote_open):
            chunks.append(' ')
            pos += 1
        surface = normalize_token(raw)
        chunks.append(surface)
        offsets.append((pos, pos + len(surface)))
        pos += len(surface)
        # 방향 없는 `"` 는 상태를 뒤집고, 방향 있는 따옴표는 상태를 못 박는다.
        if raw in _OPEN_QUOTE:
            quote_open = True
        elif raw in _CLOSE_QUOTE:
            quote_open = False
        elif raw == '"':
            quote_open = not quote_open
        prev_raw = raw

    return ''.join(chunks), offsets
