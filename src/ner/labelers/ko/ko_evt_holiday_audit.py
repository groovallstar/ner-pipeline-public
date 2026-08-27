"""한국어 명절·기념일·절기 이름의 `EVT` 판정 감사 + gold 재라벨.

canonical §3.3 때-머리 원칙은 **머리가 그 하루를 부르는지, 그 하루 둘레의
기간을 부르는지**로 가른다. `크리스마스`·`설날`·`동지` 는 달력의 한 날을 부르는
고유명이라 `EVT` 이고, `크리스마스 시즌`·`설 연휴` 는 머리가 기간이라 `DAT` 다.

**이 판정은 형태만으로 못 한다.** `설날` 과 `이날` 은 같은 머리를 갖는데 gold 의
`날` 머리 `DAT` 는 `이날`·`전날`·`다음날` 이 압도한다. 그래서 어느 어근이 명절
이름인지는 canonical §5.3 이 열거하고, 이 파일은 그 목록을 집행만 한다.

세 모드가 한 계보를 이룬다:

- ``survey`` — 후보 표면형을 전수 열거해 판정용 JSONL 로 낸다. 후보는 어근
               매칭만으로 뽑지 않는다 — 어근 목록이 개방집합이라 그것만으로는
               빠뜨린 이름이 조용히 남는다. 명절 꼴 접미사 스윕을 함께 돌려
               **어근에 없는 이름도 후보에 올린다.** `--residue` 는 반대로
               후보에 **안** 든 표면형을 세어 준다 — 놓친 이름을 사람이 눈으로
               찾는 자리다.
- ``gate``   — 원장이 후보를 남김없이 판정했는지(미배정 0), 원장이 적은 자리가
               gold 에 그대로 있는지, 그 판정이 gold 라벨과 맞는지를 본다.
               재라벨 전후에 같은 명령으로 돌린다.
- ``apply``  — 원장을 소비해 gold 를 재라벨한다. 개수가 아니라 **자리 집합**으로
               단언하므로, 옮길 자리를 옮기면서 다른 자리를 지우는 변경은
               통과하지 못한다. 적용 뒤 `gate` 를 새 gold 에 다시 돌려 그 결과를
               provenance 에 넣는다.

**게이트가 조이는 범위를 넘겨 읽지 말 것.** 이 게이트는 **원장이 판정한 표면형**
안에서만 조인다. 후보에 안 든 `DAT`·`EVT` span 을 옮기거나 지우는 변경은 여기서
안 걸리며, 그건 `apply` 의 자리-집합 단언이 맡는다. 축1·R2 게이트의 사각을 이
게이트가 통째로 메운다고 적으면 다음 라운드가 더 넓은 방어를 안 만든다.

**어근 목록의 완전성은 기계가 보증하지 못한다.** 그래서 `sweep` 앵커를 둔다 —
`DAT`·`EVT` 표면형 전체의 지문을 원장에 적어 두고, gold 에 새 표면형이 생기면
게이트가 막아 사람이 잔여를 다시 훑게 한다. 재라벨은 라벨만 바꾸고 표면형 집합은
안 바꾸므로 이 지문은 재라벨 전후로 같다. **후보 그물은 숫자 있는 표면형을 빼지만
이 앵커는 안 뺀다** — 그물은 사람이 판정할 모집단이라 좁혀야 하고 앵커는 해시라
넓혀도 비용이 없는데, 좁히면 `DAT`+`EVT` span 의 64.6%가 감시 밖에 남는다.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import pathlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

from ner.labelers.ko.ko_evt_r2_audit import (
    CANONICAL_PATH,
    canonical_section_rows,
    dump_gold,
    file_sha256,
    load_gold,
)

logger = logging.getLogger(__name__)

DAT = "DAT"
EVT = "EVT"

# ── canonical §5.3 명절 어근 — 정본은 canonical 표이고 아래는 사본이다 ──────
#
# 규칙은 기준 파일(바꾸면 사람 승인·반박자를 타는 곳)에 있는데 그것을 집행하는 이
# 파일은 잠금 밖이다. 어근을 조용히 좁히면 후보 모집단이 줄어 게이트가 아무 일 없이
# 통과한다 — 자기 규칙을 자기 자로 재는 구조다. `parse_canonical_holiday()` 가 표를
# 파싱하고 `tests/ner/labelers/test_ko_evt_holiday_audit.py` 가 이 상수와 양방향으로
# 대조해 그 경로를 막는다(좁혀도 넓혀도 FAIL).
HOLIDAY_ROOTS: Tuple[str, ...] = (
    "크리스마스", "성탄", "추석", "설", "단오", "동지", "입춘", "한식", "칠석",
    "초파일", "석가", "부활", "핼러윈", "할로윈", "밸런타인", "발렌타인",
    "추수감사", "대보름", "제야", "새해", "신년", "신정", "구정", "명절",
    "공휴일", "한글", "어린이", "어버이", "스승", "국군", "근로자", "현충",
    "광복", "제헌", "개천", "삼일", "식목", "노동절", "독립기념", "화이트데이",
    "빼빼로", "삼겹살", "가래떡", "연말연시", "한가위", "핼로윈", "춘제",
    "春節", "광군제", "라마단",
)

# 머리가 여전히 **그 하루**를 부르므로 이름에 붙어도 EVT 로 남는다
DAY_HEADS: Tuple[str, ...] = ("날", "이브", "첫날", "첫 날", "전야")

# 머리가 그 하루가 아니라 **둘레의 기간·범위**를 부르므로 DAT 로 돌아간다
SPAN_HEADS: Tuple[str, ...] = (
    "시즌", "연휴", "연휴기간", "기간", "때", "시절", "동안", "무렵", "즈음",
)

# 하루가 아니라 **구간**을 부르지만 그 구간의 이름이라 EVT 인 것들. 파생형이
# 아니므로 기간 머리 규칙이 안 걸린다 — `새해`(이름) ↔ `설 연휴`(`설` + 기간 머리).
PERIOD_NAMES: Tuple[str, ...] = ("새해", "신년", "연말연시", "라마단")

# 이름이 아니라 **범주**를 부르는 말 — 고유명이 아니라 DAT 다. span 의 머리가
# 이것일 때만 그 행이며, 이름 뒤에 따라온다고 자동으로 DAT 로 끌지는 않는다 —
# 머리일 수도(`작년 설 명절`) 서술일 수도(`한글날 공휴일 되찾기`) 있어서다.
CATEGORY_HEADS: Tuple[str, ...] = ("명절", "공휴일", "휴일")
CATEGORY_WORDS = CATEGORY_HEADS  # 옛 이름 — canonical 마커와 짝이 맞는 쪽이 정본이다

# 명절 꼴 접미사 — 어근 목록이 놓친 이름을 후보에 올리는 그물이다. 판정이 아니라
# **후보 생성**에만 쓴다. `이날`·`일요일`·`고등학교 시절` 이 대량으로 딸려 오지만,
# 그물이 좁아 이름을 빠뜨리는 것보다 넓어 오탐이 섞이는 편이 안전하다 — 오탐은
# 원장에서 `NOT_HOLIDAY` 로 한 번 판정하면 끝나고, 누락은 아무도 못 본다.
_CANDIDATE_SUFFIX = re.compile(
    r"(날|절|일|데이|기념일|보름|이브|연휴|연시|시즌|시절|명절|공휴일|때)$"
)
_HAS_DIGIT = re.compile(r"\d")
# 등위 접속 — `설과 추석 연휴` 처럼 머리를 나눠 갖는 자리를 따라가기 위한 것.
# 한글 접속조사는 뒤에 공백을 요구한다 — 안 그러면 `크리스마스` 뒤의 `나는` 을
# 접속으로 읽어 엉뚱한 자리를 따라간다. 구두점은 공백 없이도 접속이다.
_COORDINATOR = re.compile(r"^\s*(?:(?:과|와|및|이나|나)\s+|[,·‧・/]\s*)")
# 건너뛴 어절 끝에 붙은 접속 표지 — `추석과` 를 `추석` 으로 되돌려 다음 홉이
# 그 조사를 다시 접속으로 읽게 한다. 이게 없으면 `과` 연쇄가 1 홉에서 끊긴다.
_COORD_TAIL = re.compile(r"(?:과|와|및|[,·‧・/])+$")
_MAX_COORD_HOPS = 4

# 원장 판정 코드 — 앞의 셋이 gold 라벨을 구속한다
#
# `EVT` 와 `EVT_KEEP` 이 갈리는 것은 **재라벨이 건드리는가**가 다르기 때문이다.
# 둘을 한 코드로 묶으면 재라벨 전 상태에서 "옮길 자리는 전부 DAT" 라는 위상 검사가
# 성립하지 않는다 — 이미 EVT 인 자리가 섞여 들어 전량도 0 도 아닌 수가 나오고,
# 그러면 절반만 적용된 gold 와 구별되지 않는다.
VERDICT_EVT = "EVT"                  # 명절 이름이고 현재 DAT 다 — 재라벨이 옮긴다
VERDICT_EVT_KEEP = "EVT_KEEP"        # 명절 계열이고 이미 EVT 다 — 안 건드린다
VERDICT_DAT = "DAT"                  # 명절 계열이지만 DAT 로 남는다
VERDICT_NOT_HOLIDAY = "NOT_HOLIDAY"  # 명절 계열이 아니다 — 현 라벨을 그대로 둔다
VERDICTS = (VERDICT_EVT, VERDICT_EVT_KEEP, VERDICT_DAT, VERDICT_NOT_HOLIDAY)

# 판정 사유 — 어느 규칙이 그 판정을 냈는지. canonical §3.3 의 갈래와 1:1 이다
REASONS: Dict[str, str] = {
    "name": "이름 자체 — 달력의 한 날을 부르는 고유명",
    "period_name": "이름 자체 — 하루가 아니라 절기·연시 구간을 부르는 고유명",
    "day_head": "이름 + 하루 머리 — 머리가 여전히 그 하루를 부른다",
    "modified": "회차·날짜·상대 수식 — 머리가 안 바뀐다",
    "event_name": "명절에서 파생된 행사 이름 — 이미 EVT 이고 그대로 둔다",
    "span_head": "이름 + 기간·범위 머리 — 머리가 둘레의 기간을 부른다",
    "category": "범주어 — 이름이 아니라 갈래를 부른다",
    "homomorph": "어근 동형 — 형태만 겹치고 명절이 아니다",
    "unrelated": "접미 동형 — 명절 어근이 없다",
}


# ── 후보 생성 ──────────────────────────────────────────────────────────


@dataclass
class Candidate:
    """한 표면형과 그것이 gold 에서 나타나는 모든 자리."""

    surface: str
    labels: collections.Counter = field(default_factory=collections.Counter)
    sites: List[Tuple[str, int, int]] = field(default_factory=list)
    matched_by: Set[str] = field(default_factory=set)

    @property
    def count(self) -> int:
        return len(self.sites)


def is_candidate(surface: str) -> Set[str]:
    """후보로 뽑는 이유를 돌려준다. 빈 집합이면 후보가 아니다.

    두 그물을 겹쳐 친다. **어근**은 `크리스마스`처럼 접미사가 없는 이름을 잡고,
    **접미**는 어근 목록이 아직 모르는 이름을 잡는다. 접미 그물은 숫자 없는
    표면형에만 건다 — `10월 9일`·`지난19일` 류 날짜가 `일` 로 끝나 통째로 딸려
    오면 후보가 163 종에서 **732 종**(추가 569 종 / 3,769 span)으로 늘고, 그
    569 종은 전수 확인 결과 전부 날짜 표현이다. 숫자가 붙은 명절
    (`8.15 광복절`·`2월 설`)은 어근 그물이 이미 잡는다.

    **그 대가로 숫자 있는 표면형은 어근에 안 걸리면 후보가 못 된다.** 어근에
    없는 새 이름이 숫자를 달고 나타나면(`2026 광군절`) 두 그물 다 놓친다.
    그래서 `surface_inventory()` 앵커는 숫자 있는 표면형까지 덮는다 — 후보
    모집단은 좁게 유지하되 **완전성 감시는 좁히지 않는다.**
    """
    why: Set[str] = set()
    if any(root in surface for root in HOLIDAY_ROOTS):
        why.add("root")
    if not _HAS_DIGIT.search(surface) and _CANDIDATE_SUFFIX.search(surface):
        why.add("suffix")
    return why


def holiday_candidates(rows: Sequence[dict]) -> Dict[str, Candidate]:
    """gold 의 `DAT`·`EVT` span 중 후보 표면형을 전수 모은다.

    `DAT` 와 `EVT` 만 보는 것은 이 판정이 그 둘 사이의 이동이기 때문이다.
    `PROD`(`8월의 크리스마스` 작품명)·`LOC` 는 다른 규칙의 소관이라 건드리지
    않으며, 그래서 후보에서도 뺀다.
    """
    out: Dict[str, Candidate] = {}
    for row in rows:
        for ent in row["entities"]:
            if ent["label"] not in (DAT, EVT):
                continue
            why = is_candidate(ent["text"])
            if not why:
                continue
            cand = out.setdefault(ent["text"], Candidate(surface=ent["text"]))
            cand.labels[ent["label"]] += 1
            cand.sites.append((row["id"], ent["start_char"], ent["end_char"]))
            cand.matched_by |= why
    return out


def following_span_head(text: str, end: int) -> str:
    """span 바로 뒤에 붙는 기간 머리를 돌려준다. 없으면 빈 문자열.

    **이 함수가 canonical §5.3 의 기간 머리 목록을 실행 경로에 배선한다.**
    없으면 그 목록은 파싱돼 동기 테스트만 통과하고 아무 판정에도 안 쓰인다 —
    규칙이 기준 파일에 적혀 있는데 그것을 세는 코드가 없는 상태다.

    왜 span *밖*을 보는가. gold 의 경계가 같은 표현에서 갈린다 — `추석 연휴` 를
    한 span 으로 단 행이 있고 `추석` 만 달고 `연휴` 를 안 단 행이 있다. 재라벨
    전에는 둘 다 `DAT` 라 경계만 달랐지만, 앞엣것만 `EVT` 로 옮기면 **경계 차이가
    타입 차이로 바뀐다** — 같은 문맥에서 한 행은 `EVT`, 다른 행은 `DAT` 가 되고
    그건 이 규칙이 닫으려던 바로 그 비일관이다.
    """
    heads = sorted(SPAN_HEADS, key=len, reverse=True)
    pos = end
    for _ in range(_MAX_COORD_HOPS):
        tail = text[pos:pos + 16].lstrip()
        # 긴 머리부터 맞춘다 — 목록 순서대로면 `연휴` 가 `연휴기간` 을 가려,
        # 판정은 같아도 원장에 적히는 머리가 실제와 달라진다.
        for head in heads:
            if tail.startswith(head):
                return head
        # **등위로 머리를 나눠 갖는 자리까지 본다.** `설과 추석 연휴` 에서
        # `연휴` 는 두 접속항에 걸리므로 `설` 도 기간 머리 아래다. 바로 뒤만
        # 보면 뒤 접속항만 걸러져 **한 명사구 안에서 타입이 갈린다** — 행
        # 사이 불일치보다 나쁘다.
        match = _COORDINATOR.match(text[pos:pos + 4])
        if not match:
            return ""
        rest = text[pos + match.end():]
        step = _COORD_TAIL.sub("", rest.split(" ", 1)[0])
        if not step:
            return ""
        pos += match.end() + len(step)
    return ""


def following_head(text: str, end: int) -> Tuple[str, str]:
    """span 뒤에 붙는 머리를 **부류까지** 돌려준다. `("", "")` 이면 없음.

    `following_span_head()` 는 기간 머리만 본다. 그런데 완전성을 그 함수로 재면
    **자기 탐지기의 시야로 자기 완전성을 재는** 꼴이라, 아직 자로 인정 안 한
    머리 부류가 있어도 0 이 나온다. 그래서 이 함수는 기간·범주·하루 머리를 모두
    훑고, 게이트는 그중 기간 머리만 예외를 강제하되 나머지도 **원장이 봤다고
    적었는지**를 요구한다. 스윕이 강제보다 넓어야 다음 부류가 드러난다.
    """
    span = following_span_head(text, end)
    if span:
        return span, "span"
    tail = text[end:end + 12].lstrip()
    for word in sorted(CATEGORY_HEADS, key=len, reverse=True):
        if tail.startswith(word):
            return word, "category"
    for word in sorted(DAY_HEADS, key=len, reverse=True):
        if tail.startswith(word):
            return word, "day"
    return "", ""


def _context(row: dict, start: int, end: int, window: int = 30) -> str:
    lo = max(0, start - window)
    return row["text"][lo:end + window].replace("\n", " ")


# ── 완전성 앵커 ────────────────────────────────────────────────────────


def surface_inventory(rows: Sequence[dict]) -> List[str]:
    """`DAT`·`EVT` 표면형 전체 — **후보 밖까지, 숫자 있는 것까지** 포함한 목록.

    후보 생성기의 두 그물은 어근 목록에 기대는데 그 목록이 개방집합이라, 그물이
    못 잡은 이름은 미배정으로도 안 잡힌다(`unclassified: 0` 이 생성기 자신의 자로
    잰 0 이 된다). 실제로 `한가위`·`핼로윈`·`춘제`·`광군제`·`라마단` 이 그렇게
    빠졌다 — `핼로윈` 은 어근에 `핼러윈`·`할로윈` 두 변이가 이미 있는데도 셋째
    변이가 새어 나간 자리다.

    그래서 이 목록의 지문을 원장에 적어 둔다. gold 에 새 표면형이 생기면 지문이
    달라지고 게이트가 막으므로, 사람이 잔여(`survey --residue`)를 다시 훑게 된다.
    **재라벨은 라벨만 바꾸고 표면형 집합은 안 바꾸므로 이 지문은 재라벨 전후로
    같다** — 위상을 안 타는 앵커라 한 값으로 양쪽을 다 지킨다.

    **후보 그물과 달리 숫자 있는 표면형을 빼지 않는다.** 그물은 사람이 판정할
    모집단이라 좁혀야 하지만 앵커는 해시 하나라 넓혀도 비용이 없고, 좁히면
    숫자 있는 표면형 1,787 종(전체 `DAT`+`EVT` span 의 64.6%)이 감시 밖에 남아
    `2026 광군절` 같은 새 이름이 그물도 앵커도 안 건드리고 들어온다.
    """
    out = set()
    for row in rows:
        for ent in row["entities"]:
            if ent["label"] in (DAT, EVT):
                out.add(ent["text"])
    return sorted(out)


def inventory_sha256(rows: Sequence[dict]) -> str:
    payload = "\n".join(surface_inventory(rows)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stale_sites(rows: Sequence[dict], judgements: Sequence[dict]) -> List[dict]:
    """원장이 적은 자리가 gold 에 그대로 있는지 본다.

    라벨 분포만 세면 **자리를 옮긴 변경이 안 걸린다** — `크리스마스` span 하나를
    지우고 다른 행에 넣으면 개수가 그대로다. 원장의 좌표는 재라벨 `apply` 가
    소비할 입력이므로, 그 좌표가 낡았는지를 여기서 확인하지 않으면 어긋난 자리에
    삽입해도 게이트가 초록불을 낸다.
    """
    by_id = {row["id"]: row for row in rows}
    out: List[dict] = []
    for item in judgements:
        for site in item["sites"]:
            rid, start, end = site[0], site[1], site[2]
            row = by_id.get(rid)
            found = None
            if row is not None:
                for ent in row["entities"]:
                    if ent["start_char"] == start and ent["end_char"] == end:
                        found = ent
                        break
            if found is None or found["text"] != item["surface"]:
                out.append({"surface": item["surface"], "row": rid,
                            "start": start, "end": end,
                            "found": None if found is None else found["text"]})
    return out


# ── 원장 ───────────────────────────────────────────────────────────────


LEDGER_KEYS = ("gold_sha256", "sweep", "reviewed_heads", "judgements")


def load_ledger(path: str) -> dict:
    """원장 전체를 돌려준다 — 판정 목록만이 아니라 지문·스윕 앵커까지.

    앵커를 선택 항목으로 두면 그것이 빠진 원장이 조용히 통과하고, 그러면 완전성
    검사가 있다는 기록만 남는다. 그래서 세 칸을 필수로 못 박는다.
    """
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a holiday ledger: expected an object")
    missing = [key for key in LEDGER_KEYS if key not in payload]
    if missing:
        raise ValueError(f"{path} is not a holiday ledger: missing {missing}")
    return payload


def ledger_index(judgements: Sequence[dict]) -> Dict[str, dict]:
    """표면형 → 판정. 같은 표면형이 두 번 나오면 예외.

    중복을 조용히 덮어쓰면 나중 항목이 이긴다. 어느 쪽이 이길지가 파일 안 순서로
    정해지는 규칙은 사람이 읽어서 알 수 없으므로, 통과시키지 않고 멈춘다.
    """
    out: Dict[str, dict] = {}
    for item in judgements:
        surface = item["surface"]
        if surface in out:
            raise ValueError(
                f"duplicate judgement for surface {surface!r} in the ledger"
            )
        if item["verdict"] not in VERDICTS:
            raise ValueError(
                f"unknown verdict {item['verdict']!r} for {surface!r}; "
                f"expected one of {VERDICTS}"
            )
        if item.get("reason") not in REASONS:
            raise ValueError(
                f"unknown reason {item.get('reason')!r} for {surface!r}; "
                f"expected one of {tuple(REASONS)}"
            )
        # 자리는 선택 항목이 아니다. `sites` 를 빼면 자리 대조가 그 표면형을
        # 통째로 건너뛰는데, 리포트의 빈 `stale_sites` 는 "이상 없음" 과
        # "아무것도 안 봤음" 을 구별하지 못한다 — 감사 대상이 스스로 검사를
        # 끄는 길이 열린다. 개수도 함께 못 박는다: 자리가 중복되거나 라벨
        # 합계와 어긋나면 실제 자리 하나가 조용히 미검증으로 남는다.
        sites = item.get("sites")
        if not isinstance(sites, list) or not sites:
            raise ValueError(f"{surface!r} has no sites; the ledger must "
                             "record every (row_id, start, end) it judged")
        keys = [tuple(site[:3]) for site in sites]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{surface!r} repeats a site; a duplicate hides "
                             "one real site from the site check")
        exceptions = item.get("site_exceptions", [])
        if exceptions and item["verdict"] != VERDICT_EVT:
            raise ValueError(
                f"{surface!r} carries site exceptions on a {item['verdict']} "
                "verdict; only EVT verdicts consume them, so these would be "
                "dead entries that read as judgements"
            )
        known = {tuple(site[:3]) for site in sites}
        for exc in exceptions:
            if tuple(exc["site"][:3]) not in known:
                raise ValueError(
                    f"{surface!r} excepts a site it does not list: {exc['site']}")
            if exc["verdict"] not in VERDICTS or exc.get("reason") not in REASONS:
                raise ValueError(
                    f"{surface!r} has a bad site exception: {exc}")
        count = item.get("count")
        if count != len(sites) or count != sum(item.get("labels", {}).values()):
            raise ValueError(
                f"{surface!r} disagrees with itself: count={count}, "
                f"sites={len(sites)}, labels={sum(item.get('labels', {}).values())}"
            )
        out[surface] = item
    return out


# ── canonical 동기 ─────────────────────────────────────────────────────

_ROOT_MARKER = re.compile(r"\*\*명절 어근\*\*:\s*(.+)$")
_PERIOD_MARKER = re.compile(r"\*\*구간 이름\*\*:\s*(.+)$")
_DAY_HEAD_MARKER = re.compile(r"\*\*하루 머리\*\*:\s*(.+)$")
_SPAN_HEAD_MARKER = re.compile(r"\*\*기간 머리\*\*:\s*(.+)$")
_CATEGORY_MARKER = re.compile(r"\*\*범주 머리\*\*:\s*(.+)$")
_BACKTICKED = re.compile(r"`([^`]+)`")


def parse_canonical_holiday(path: str = CANONICAL_PATH) -> Dict[str, Tuple[str, ...]]:
    """canonical §5.3 표에서 명절 어근·구간 이름·하루/기간/범주 머리를 읽어 온다.

    이 파서가 있는 이유는 편의가 아니라 **집행 주체**다. 규칙은 기준 파일에 있고
    그것을 집행하는 이 모듈의 상수는 잠금 밖이라, 상수만 조용히 고치면 어긋남을
    볼 것이 없다. 파싱 결과와 상수를 테스트가 양방향으로 대조한다.

    다섯 중 하나라도 표에서 못 찾으면 예외다 — 빈 목록은 "규칙이 실제로 비어 있다"
    와 구별되지 않고, 그러면 어근 0 개짜리 후보 생성기가 정상처럼 통과한다.
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
            "the candidate generator reads its population from that table"
        )
    return found


# ── 게이트 ─────────────────────────────────────────────────────────────


def expected_labels(item: dict, phase: str) -> collections.Counter:
    """원장이 기록한 라벨 분포에서, 그 위상에서 기대되는 분포를 만든다.

    `EVT` 판정만 위상을 탄다 — 재라벨이 그 자리의 `DAT` 를 `EVT` 로 옮기기
    때문이다. 나머지 셋은 재라벨이 안 건드리므로 판정 시점 분포가 그대로 남아야
    한다. **`NOT_HOLIDAY` 까지 묶는 것이 요점이다** — 명절과 무관한 표면형의
    라벨을 안 묶으면 `이날` 767 자리를 통째로 `EVT` 로 바꿔도 게이트가 통과한다.
    """
    recorded: collections.Counter = collections.Counter(item["labels"])
    if item["verdict"] == VERDICT_EVT and phase == "applied":
        held = sum(1 for exc in item.get("site_exceptions", [])
                   if exc["verdict"] != VERDICT_EVT)
        moved = recorded.pop(DAT, 0) - held
        if held:
            recorded[DAT] = held
        if moved:
            recorded[EVT] += moved
    return recorded


def check_gate(rows: Sequence[dict], ledger: dict) -> dict:
    """원장 완전성과 gold 정합을 함께 본다.

    셋을 센다:

    - **미배정** — gold 에 있는 후보인데 원장이 판정하지 않은 표면형. 0 이어야
      한다. 이것이 0 이 아니면 다음 둘은 "판정한 것 안에서만" 맞는 수라 뜻이 없다.
    - **유령 판정** — 원장에는 있는데 gold 에 없는 표면형. 원장이 gold 를 앞질러
      부풀었다는 뜻이라 함께 센다.
    - **라벨 이탈** — 판정한 표면형이 원장에 적힌 라벨 분포에서 벗어난 자리.

    **위상을 먼저 정하고 그 위상에서 기대되는 분포와 대조한다.** `EVT` 판정의
    자리는 재라벨 전이면 전부 `DAT`, 뒤면 전부 `EVT` 다. 그 사이 값은 적용이
    일부만 먹었다는 뜻이라 `partial` 로 잡아 통과시키지 않는다 — 이 조건이
    없으면 절반만 옮긴 gold 가 "진행 중" 으로 보여 그대로 굳는다.

    나머지 세 판정은 재라벨이 안 건드리므로 판정 시점 분포가 그대로여야 한다.
    분포를 세므로 라벨이 바뀐 것뿐 아니라 **span 이 지워진 것도 잡힌다** —
    `추석` 14 자리를 옮기면서 다른 자리를 삭제하는 변경이 여기서 걸린다.
    """
    judgements = ledger["judgements"]
    cands = holiday_candidates(rows)
    index = ledger_index(judgements)

    unclassified = sorted(set(cands) - set(index))
    phantom = sorted(set(index) - set(cands))

    planned = planned_moves(judgements)
    by_id = {row["id"]: row for row in rows}
    label_at = {}
    for row in rows:
        for ent in row["entities"]:
            label_at[(row["id"], ent["start_char"], ent["end_char"])] = ent["label"]
    evt_sites = len(planned)
    evt_pending = sum(1 for key in planned if label_at.get(key) == DAT)

    # canonical §5.3 의 기간 머리를 실제로 세는 유일한 자리. 옮기려는 자리 바로
    # 뒤에 기간 머리가 붙어 있으면 그 자리의 옳은 span 은 더 긴 DAT 한 덩어리라,
    # bare 이름만 EVT 로 올리면 같은 표현이 행마다 다른 타입을 갖게 된다.
    unmarked_span_head = []
    unreviewed_head = []
    reviewed = {tuple(item["site"][:3]) for item in ledger["reviewed_heads"]}
    for key in sorted(planned):
        row = by_id.get(key[0])
        if row is None:
            continue
        head, kind = following_head(row["text"], key[2])
        if not head:
            continue
        entry = {"row": key[0], "surface": row["text"][key[1]:key[2]],
                 "head": head, "kind": kind,
                 "context": row["text"][max(0, key[1] - 10):key[2] + 12]}
        if kind == "span":
            unmarked_span_head.append(entry)
        elif key not in reviewed:
            # 기간 머리가 아닌 머리는 타입을 안 바꾸지만 **경계**는 바꾼다.
            # 자로 인정할지는 사람이 정하되, 안 본 채로 지나가지는 못하게 한다.
            unreviewed_head.append(entry)

    if evt_pending == 0:
        phase = "applied"
    elif evt_pending == evt_sites:
        phase = "pre-apply"
    else:
        phase = "partial"

    drift: List[dict] = []
    for surface, cand in sorted(cands.items()):
        item = index.get(surface)
        if item is None:
            continue
        expected = expected_labels(item, phase)
        if collections.Counter(cand.labels) != expected:
            drift.append({
                "surface": surface,
                "verdict": item["verdict"],
                "expected": dict(sorted(expected.items())),
                "actual": dict(sorted(cand.labels.items())),
            })

    recorded = ledger["sweep"].get("inventory_sha256")
    observed = inventory_sha256(rows)

    verdict_counts = collections.Counter(i["verdict"] for i in index.values())
    return {
        "candidates": {"surfaces": len(cands),
                       "spans": sum(c.count for c in cands.values())},
        "judged": dict(verdict_counts),
        "unclassified": unclassified,
        "phantom": phantom,
        "evt": {"sites": evt_sites, "pending": evt_pending, "phase": phase},
        "label_drift": drift,
        "sites": {"verified": sum(len(i["sites"]) for i in judgements),
                  "stale": stale_sites(rows, judgements),
                  "unmarked_span_head": unmarked_span_head,
                  "unreviewed_head": unreviewed_head},
        "sweep": {"recorded": recorded, "observed": observed,
                  "matches": recorded == observed},
    }


def gate_failures(report: dict) -> List[str]:
    """게이트가 막아야 할 사유를 모은다. 빈 목록이면 통과."""
    out: List[str] = []
    if report["unclassified"]:
        out.append(
            f"{len(report['unclassified'])} candidate surfaces unclassified: "
            + ", ".join(report["unclassified"][:10])
        )
    if report["phantom"]:
        out.append(
            f"{len(report['phantom'])} ledger surfaces absent from gold: "
            + ", ".join(report["phantom"][:10])
        )
    if report["evt"]["phase"] == "partial":
        out.append(
            f"EVT verdicts are half applied: {report['evt']['pending']} of "
            f"{report['evt']['sites']} sites still carry DAT"
        )
    if report["label_drift"]:
        out.append(
            f"{len(report['label_drift'])} surfaces drifted from the labels the "
            "ledger recorded: "
            + ", ".join(d["surface"] for d in report["label_drift"][:10])
        )
    if report["sites"]["stale"]:
        stale = report["sites"]["stale"]
        out.append(
            f"{len(stale)} of {report['sites']['verified']} ledger sites no "
            "longer hold the surface they recorded: "
            + ", ".join(f"{s['surface']}@{s['row']}" for s in stale[:10])
        )
    if report["sites"]["unmarked_span_head"]:
        bad = report["sites"]["unmarked_span_head"]
        out.append(
            f"{len(bad)} sites would move to EVT while a period head follows "
            "them, which turns a boundary difference into a type difference: "
            + ", ".join(f"{b['surface']}+{b['head']}@{b['row']}" for b in bad[:10])
            + " — judge each site in the ledger's site_exceptions"
        )
    if report["sites"]["unreviewed_head"]:
        bad = report["sites"]["unreviewed_head"]
        out.append(
            f"{len(bad)} moved sites carry a non-period head the ledger never "
            "reviewed: "
            + ", ".join(f"{b['surface']}+{b['head']}({b['kind']})@{b['row']}"
                        for b in bad[:10])
            + " — record each in the ledger's reviewed_heads"
        )
    if not report["sweep"]["matches"]:
        out.append(
            "the DAT/EVT surface inventory moved "
            f"({report['sweep']['recorded']} -> {report['sweep']['observed']}); "
            "re-sweep the residue so a new holiday name cannot slip past the "
            "candidate nets"
        )
    return out


# ── 재라벨 ─────────────────────────────────────────────────────────────


def _sites_by_label(rows: Sequence[dict]) -> Dict[str, Set[Tuple[str, int, int]]]:
    """라벨 → 그 라벨이 붙은 자리 집합. 개수가 아니라 집합으로 비교하려는 것이다."""
    out: Dict[str, Set[Tuple[str, int, int]]] = collections.defaultdict(set)
    for row in rows:
        for ent in row["entities"]:
            out[ent["label"]].add((row["id"], ent["start_char"], ent["end_char"]))
    return out


def site_exceptions(item: dict) -> Dict[Tuple[str, int, int], dict]:
    """표면형 판정과 다르게 가는 자리들. 표면형 단위 판정의 탈출구다."""
    return {tuple(exc["site"][:3]): exc
            for exc in item.get("site_exceptions", [])}


def planned_moves(judgements: Sequence[dict]) -> Set[Tuple[str, int, int]]:
    """원장이 `EVT` 로 판정한 자리 전부. 이 집합이 재라벨의 유일한 입력이다.

    자리별 예외가 걸린 자리는 뺀다 — 한 표면형이 문맥으로 갈리는 자리가 실재하고
    (`추석`(EVT) ↔ `추석` + `연휴`(DAT)), 표면형 하나에 답 하나만 허용하면 그
    자리를 담을 데가 없어 규칙이 금지한 값이 gold 로 들어간다.
    """
    out: Set[Tuple[str, int, int]] = set()
    for item in judgements:
        if item["verdict"] != VERDICT_EVT:
            continue
        excepted = site_exceptions(item)
        for site in item["sites"]:
            key = (site[0], site[1], site[2])
            if excepted.get(key, {}).get("verdict", VERDICT_EVT) == VERDICT_EVT:
                out.add(key)
    return out


def apply_ledger(rows: Sequence[dict], ledger: dict,
                 gold_sha256: str) -> Tuple[List[dict], dict]:
    """원장을 소비해 gold 를 재라벨하고 **자리 집합**으로 단언한다.

    개수 단언(`옮긴 수 == 판정 수`)은 옮길 자리를 옮기면서 다른 자리를 지우는
    변경을 통과시킨다 — 지운 만큼 다른 데서 채우면 수가 맞기 때문이다. 그래서
    두 등식을 집합으로 건다:

        DAT_after == DAT_before − 원장EVT
        EVT_after == EVT_before ∪ 원장EVT

    여기에 보존 단언을 더한다 — 행 수·`id`·`text`·행별 엔티티 수가 그대로이고,
    나머지 여덟 타입의 자리 집합이 안 움직이며, 옮긴 span 은 `label` 을 뺀 모든
    필드가 byte-identical 이다. 원장이 판정한 gold 와 지금 gold 의 지문이 다르면
    좌표를 다른 코퍼스에서 잰 것이므로 시작하지 않는다(축1 `apply` 와 같은 취급).
    """
    if ledger["gold_sha256"] != gold_sha256:
        raise SystemExit(
            f"FAIL: the ledger judged gold {ledger['gold_sha256']} but this "
            f"gold hashes to {gold_sha256}; its site offsets were taken "
            "against a different corpus"
        )
    planned = planned_moves(ledger["judgements"])
    before = _sites_by_label(rows)
    if not planned <= before[DAT]:
        missing = sorted(planned - before[DAT])[:10]
        raise SystemExit(
            f"FAIL: {len(planned - before[DAT])} planned sites are not DAT in "
            f"this gold: {missing}"
        )

    new_rows = json.loads(json.dumps(rows))
    moved = 0
    for row in new_rows:
        for ent in row["entities"]:
            if (row["id"], ent["start_char"], ent["end_char"]) in planned:
                ent["label"] = EVT
                moved += 1

    after = _sites_by_label(new_rows)
    if after[DAT] != before[DAT] - planned:
        raise SystemExit("FAIL: the DAT site set is not exactly what the "
                         "ledger removed from it")
    if after[EVT] != before[EVT] | planned:
        raise SystemExit("FAIL: the EVT site set is not exactly what the "
                         "ledger added to it")
    for label in set(before) | set(after):
        if label not in (DAT, EVT) and before[label] != after[label]:
            raise SystemExit(f"FAIL: {label} sites moved; the relabel must "
                             "touch DAT and EVT only")
    if len(new_rows) != len(rows):
        raise SystemExit("FAIL: the row count changed")
    for old, new in zip(rows, new_rows):
        if old["id"] != new["id"] or old["text"] != new["text"]:
            raise SystemExit(f"FAIL: row {old['id']} lost its id or text")
        if len(old["entities"]) != len(new["entities"]):
            raise SystemExit(f"FAIL: row {old['id']} changed entity count")
        for a, b in zip(old["entities"], new["entities"]):
            if {k: v for k, v in a.items() if k != "label"} != \
                    {k: v for k, v in b.items() if k != "label"}:
                raise SystemExit(
                    f"FAIL: row {old['id']} span {a.get('text')!r} changed a "
                    "field other than its label")

    # 실제로 옮긴 자리만 센다 — 판정한 자리에서 자리별 예외를 뺀 수라야 합이
    # `moved` 와 맞고, 안 그러면 사유별 표가 총계와 어긋나 어느 쪽이 참인지 모른다.
    reasons: collections.Counter = collections.Counter()
    for item in ledger["judgements"]:
        if item["verdict"] != VERDICT_EVT:
            continue
        held = site_exceptions(item)
        for site in item["sites"]:
            key = (site[0], site[1], site[2])
            if held.get(key, {}).get("verdict", VERDICT_EVT) == VERDICT_EVT:
                reasons[item["reason"]] += 1
    held_total = sum(
        1 for item in ledger["judgements"] if item["verdict"] == VERDICT_EVT
        for exc in item.get("site_exceptions", []) if exc["verdict"] != VERDICT_EVT)
    if sum(reasons.values()) != moved:
        raise SystemExit("FAIL: the per-reason counts do not sum to the moves")
    provenance = {
        "moved": moved,
        "held_by_site_exception": held_total,
        "by_reason": dict(sorted(reasons.items())),
        "label_totals": {
            "before": {DAT: len(before[DAT]), EVT: len(before[EVT])},
            "after": {DAT: len(after[DAT]), EVT: len(after[EVT])},
        },
    }
    return new_rows, provenance


# ── CLI ────────────────────────────────────────────────────────────────


def cmd_survey(args: argparse.Namespace) -> None:
    rows = load_gold(args.gold)
    by_id = {row["id"]: row for row in rows}
    cands = holiday_candidates(rows)
    if args.residue:
        counts: collections.Counter = collections.Counter()
        for row in rows:
            for ent in row["entities"]:
                if ent["label"] in (DAT, EVT) and ent["text"] not in cands:
                    counts[ent["text"]] += 1
        with open(args.out, "w", encoding="utf-8") as fh:
            for surface, count in counts.most_common():
                fh.write(json.dumps({"surface": surface, "count": count},
                                    ensure_ascii=False) + "\n")
        logger.info("Residue (DAT/EVT surfaces outside both nets): "
                    "%d surfaces / %d spans -> %s",
                    len(counts), sum(counts.values()), args.out)
        return
    with open(args.out, "w", encoding="utf-8") as fh:
        for surface, cand in sorted(
                cands.items(), key=lambda kv: (-kv[1].count, kv[0])):
            samples = [_context(by_id[rid], start, end)
                       for rid, start, end in cand.sites[:3]]
            fh.write(json.dumps({
                "surface": surface,
                "count": cand.count,
                "labels": dict(cand.labels),
                "matched_by": sorted(cand.matched_by),
                "samples": samples,
            }, ensure_ascii=False) + "\n")
    logger.info("Holiday candidates: %d surfaces / %d spans -> %s",
                len(cands), sum(c.count for c in cands.values()), args.out)


def cmd_gate(args: argparse.Namespace) -> None:
    rows = load_gold(args.gold)
    ledger = load_ledger(args.ledger)
    report = check_gate(rows, ledger)
    observed = file_sha256(args.gold)
    # 재라벨 전이면 gold 가 판정 시점 그대로여야 한다. 재라벨 뒤의 기대 sha 는
    # 아직 아무도 모르므로(그건 apply 의 provenance 가 적을 값이다) 기록만 한다.
    report["gold_sha256"] = {"judged": ledger["gold_sha256"],
                             "observed": observed,
                             "matches": ledger["gold_sha256"] == observed}
    if report["evt"]["phase"] == "pre-apply" and not report["gold_sha256"][
            "matches"]:
        raise SystemExit(
            f"FAIL: gold {observed} is not the gold the ledger judged "
            f"({ledger['gold_sha256']}), yet no EVT verdict has been applied"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = gate_failures(report)
    if failures:
        raise SystemExit("FAIL: " + " | ".join(failures))
    logger.info("Holiday gate passed (%s): %d surfaces judged",
                report["evt"]["phase"], report["candidates"]["surfaces"])


def cmd_apply(args: argparse.Namespace) -> None:
    rows = load_gold(args.gold)
    ledger = load_ledger(args.ledger)
    new_rows, provenance = apply_ledger(rows, ledger, file_sha256(args.gold))
    dump_gold(new_rows, args.out)
    provenance["gold_sha256"] = {"before": file_sha256(args.gold),
                                 "after": file_sha256(args.out)}
    provenance["ledger_sha256"] = file_sha256(args.ledger)
    # 새 gold 에 게이트를 다시 돌려 결과를 함께 남긴다. 적용이 다른 자리의
    # 판정을 깨뜨렸다면 여기서 드러나야 하고, provenance 안에 있어야 나중에
    # "그때 무엇이 초록이었나" 를 다시 재지 않고 읽을 수 있다.
    report = check_gate(new_rows, ledger)
    failures = gate_failures(report)
    provenance["gate_after"] = {
        "phase": report["evt"]["phase"],
        "unclassified": len(report["unclassified"]),
        "verified_sites": report["sites"]["verified"],
        "stale_sites": len(report["sites"]["stale"]),
        "label_drift": len(report["label_drift"]),
        "sweep_matches": report["sweep"]["matches"],
        "failures": failures,
    }
    pathlib.Path(args.provenance).write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    if failures:
        raise SystemExit("FAIL after apply: " + " | ".join(failures))
    logger.info("Applied %d holiday moves: DAT %d -> %d, EVT %d -> %d",
                provenance["moved"],
                provenance["label_totals"]["before"][DAT],
                provenance["label_totals"]["after"][DAT],
                provenance["label_totals"]["before"][EVT],
                provenance["label_totals"]["after"][EVT])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="KO holiday EVT rule audit and gold relabel")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_su = sub.add_parser(
        "survey", help="enumerate candidate surfaces for judgement")
    p_su.add_argument("--gold", required=True)
    p_su.add_argument("--out", required=True, help="candidate JSONL")
    p_su.add_argument(
        "--residue", action="store_true",
        help="instead list the DAT/EVT surfaces NEITHER net caught — where "
             "a missed holiday name would be hiding")
    p_su.set_defaults(func=cmd_survey)

    p_ga = sub.add_parser(
        "gate", help="ledger completeness + gold conformance")
    p_ga.add_argument("--gold", required=True)
    p_ga.add_argument("--ledger", required=True, help="holiday judgement ledger")
    p_ga.add_argument("--out", help="write gate report JSON here")
    p_ga.set_defaults(func=cmd_gate)

    p_ap = sub.add_parser(
        "apply", help="consume the ledger and relabel gold")
    p_ap.add_argument("--gold", required=True)
    p_ap.add_argument("--ledger", required=True, help="holiday judgement ledger")
    p_ap.add_argument("--out", required=True, help="write the new gold here")
    p_ap.add_argument("--provenance", required=True,
                      help="write the apply provenance JSON here")
    p_ap.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
