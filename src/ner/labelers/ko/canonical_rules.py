"""canonical §5.3 이 정한 KO EVT 규칙을 프롬프트 쪽에서 읽기 위한 파서.

**이 파일이 상수 사본을 두지 않는 이유.** 규칙은 기준 파일(사람 승인·반박자를
타는 곳)에 있고 이 파일은 잠금 밖이다. 예전에는 감사 모듈이 목록을 상수로 복사해
두고 양방향 동기 테스트가 canonical 과 대조했는데, 그 감사 모듈과 테스트가 함께
폐기되면 사본만 남아 아무도 안 보는 상태가 된다. 그래서 사본을 없애고 canonical
표를 그때그때 읽는다 — 목록을 좁히려면 기준 파일을 고쳐야 하고, 그건 승인을 탄다.

읽는 것은 `SINGLE_PROMPT_TEMPLATE` ↔ canonical 동기 검사가 쓰는 셋뿐이다. 명절
어근·머리 목록, 축1-복합 head 목록, 그리고 span 뒤에 붙은 기간 머리를 찾는 함수다.
"""

from __future__ import annotations

import functools
import re
from typing import Dict, Tuple

from ner.labelers.ko.ko_evt_r2_audit import CANONICAL_PATH, canonical_section_rows

_BACKTICKED = re.compile(r"`([^`]+)`")

# ── canonical §5.3 명절 표 ─────────────────────────────────────────────

_ROOT_MARKER = re.compile(r"\*\*명절 어근\*\*:\s*(.+)$")
_PERIOD_MARKER = re.compile(r"\*\*구간 이름\*\*:\s*(.+)$")
_DAY_HEAD_MARKER = re.compile(r"\*\*하루 머리\*\*:\s*(.+)$")
_SPAN_HEAD_MARKER = re.compile(r"\*\*기간 머리\*\*:\s*(.+)$")
_CATEGORY_MARKER = re.compile(r"\*\*범주 머리\*\*:\s*(.+)$")


def parse_canonical_holiday(path: str = CANONICAL_PATH) -> Dict[str, Tuple[str, ...]]:
    """canonical §5.3 표에서 명절 어근·구간 이름·하루/기간/범주 머리를 읽어 온다.

    다섯 중 하나라도 표에서 못 찾으면 예외다 — 빈 목록은 "규칙이 실제로 비어 있다"
    와 구별되지 않고, 그러면 어근 0 개짜리 검사가 정상처럼 통과한다.
    """
    found: Dict[str, Tuple[str, ...]] = {}
    markers = (
        ("roots", _ROOT_MARKER),
        ("period_names", _PERIOD_MARKER),
        ("day_heads", _DAY_HEAD_MARKER),
        ("span_heads", _SPAN_HEAD_MARKER),
        ("category_heads", _CATEGORY_MARKER),
    )
    for line in canonical_section_rows(path):
        for key, marker in markers:
            match = marker.search(line)
            if match:
                found[key] = tuple(_BACKTICKED.findall(match.group(1)))
    missing = [key for key, _ in markers if key not in found]
    if missing:
        raise ValueError(
            f"canonical holiday markers not found in {path}: {missing}; "
            "the prompt sync check reads its population from that table"
        )
    return found


# ── canonical §5.3 축1-복합 head ───────────────────────────────────────

_AXIS1_HEAD_MARKER = re.compile(r"\*\*축1 head[^*]*\*\*:\s*(.+)$")


def canonical_axis1_heads(path: str = CANONICAL_PATH) -> Tuple[str, ...]:
    """축1-복합 head 목록. 고유명이 조사 없이 앞설 때 EVT 가 되는 사건 머리다.

    표에서 못 찾으면 예외다 — 빈 목록이면 프롬프트가 head 를 하나도 안 가르쳐도
    동기 검사가 통과해 버린다.
    """
    for line in canonical_section_rows(path):
        match = _AXIS1_HEAD_MARKER.search(line)
        if match:
            return tuple(_BACKTICKED.findall(match.group(1)))
    raise ValueError(
        f"canonical axis-1 head marker not found in {path}; "
        "the prompt sync check reads its head list from that table"
    )


# ── span 뒤에 붙는 기간 머리 ───────────────────────────────────────────

# 등위 접속 — `설과 추석 연휴` 처럼 머리를 나눠 갖는 자리를 따라가기 위한 것.
# 한글 접속조사는 뒤에 공백을 요구한다 — 안 그러면 `크리스마스` 뒤의 `나는` 을
# 접속으로 읽어 엉뚱한 자리를 따라간다. 구두점은 공백 없이도 접속이다.
_COORDINATOR = re.compile(r"^\s*(?:(?:과|와|및|이나|나)\s+|[,·‧・/]\s*)")
# 건너뛴 어절 끝에 붙은 접속 표지 — `추석과` 를 `추석` 으로 되돌려 다음 홉이
# 그 조사를 다시 접속으로 읽게 한다. 이게 없으면 `과` 연쇄가 1 홉에서 끊긴다.
_COORD_TAIL = re.compile(r"(?:과|와|및|[,·‧・/])+$")
_MAX_COORD_HOPS = 4


@functools.lru_cache(maxsize=8)
def _span_heads(path: str) -> Tuple[str, ...]:
    """긴 머리부터 맞추도록 정렬해 둔다 — 목록 순서대로면 `연휴` 가 `연휴기간` 을
    가려, 판정은 같아도 어느 머리가 걸렸는지가 실제와 달라진다."""
    return tuple(sorted(parse_canonical_holiday(path)["span_heads"],
                        key=len, reverse=True))


def following_span_head(text: str, end: int, path: str = CANONICAL_PATH) -> str:
    """span 바로 뒤에 붙는 기간 머리를 돌려준다. 없으면 빈 문자열.

    왜 span *밖*을 보는가. gold 의 경계가 같은 표현에서 갈린다 — `추석 연휴` 를
    한 span 으로 단 행이 있고 `추석` 만 달고 `연휴` 를 안 단 행이 있다. 이름만
    `EVT` 로 옮기면 **경계 차이가 타입 차이로 바뀐다** — 같은 문맥에서 한 행은
    `EVT`, 다른 행은 `DAT` 가 되고 그건 이 규칙이 닫으려던 바로 그 비일관이다.

    **등위로 머리를 나눠 갖는 자리까지 본다.** `설과 추석 연휴` 에서 `연휴` 는 두
    접속항에 걸리므로 `설` 도 기간 머리 아래다. 바로 뒤만 보면 뒤 접속항만 걸러져
    **한 명사구 안에서 타입이 갈린다** — 행 사이 불일치보다 나쁘다.
    """
    heads = _span_heads(path)
    pos = end
    for _ in range(_MAX_COORD_HOPS):
        tail = text[pos:pos + 16].lstrip()
        for head in heads:
            if tail.startswith(head):
                return head
        match = _COORDINATOR.match(text[pos:pos + 4])
        if not match:
            return ""
        rest = text[pos + match.end():]
        step = _COORD_TAIL.sub("", rest.split(" ", 1)[0])
        if not step:
            return ""
        pos += match.end() + len(step)
    return ""
