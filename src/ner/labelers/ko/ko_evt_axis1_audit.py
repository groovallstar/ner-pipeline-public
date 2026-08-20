"""한국어 EVT 축1-복합(고유명 + 사건 head) head 후보의 결정적 열거.

canonical `docs/manual/data/canonical-entity-schema.md` §5.3 은 축1 을 "1회성
named 사건" 으로, 축2 를 "단일 generic 사건명사 단독은 **named·수식 시에만** EVT"
로 정해 두었다. 두 조항이 같은 말을 하는데 **집행할 수 있는 형태가 없다** — 어떤
head 가 사건 head 인지, 무엇이 고유명 동반을 만족시키는지가 적혀 있지 않아 회수도
감사도 사람의 눈대중에 걸린다.

이 모듈은 그 목록을 만드는 절차를 고정한다. **핵심은 후보 집합을 저자가 고르지
않는다는 것이다.** 저자가 떠올린 head 만 표에 올리면 표는 엄밀해 보이지만 여집합이
검토되지 않고, head 를 좁힐수록 그것을 분모로 쓰는 감사가 쉬워진다 — 자기 규칙을
자기 자로 재는 구조다. 그래서 후보는 gold 가 제안한다:

1. gold EVT span 에 든 **모든 어절**을 전수 집계한다.
2. ``min_count`` 회 이상 등장하고, 코퍼스에 단독 명사로 쓰이며, 고유명으로 라벨된
   이력이 없는 어절을 후보로 삼는다.
3. 후보마다 **코퍼스 실측**을 붙인다: 동형 명사가 몇 종인가 · 그중 gold EVT 이력이
   있는 것은 몇 종인가 · 다른 타입으로 라벨된 것은 무엇인가 · 더 짧은 후보에 이미
   덮이나.

**마지막 어절만 보면 안 된다.** 축1-복합이 head 로 끝나는 것은 맞지만, gold 가 그
복합을 *더 긴* span 으로 잡아 두면 head 가 span 중간에 묻힌다 — `마라톤` 은
`보스턴 마라톤 대회 폭발 사건` 안에 4 회 있는데 어떤 span 에서도 마지막이 아니라,
마지막 어절 집계로는 존재 자체가 안 보인다. `폭발`(span-final 0 회)·`붕괴`(1)·
`화재`(1)도 같은 이유로 문턱 아래로 사라진다. 고유명 라벨 이력으로 거르면 같은 span 의 수식부(`보스턴`)는
자동으로 빠지므로, 모집단을 넓히면서 정밀도는 유지된다.

채택·제외는 이 모듈이 정하지 않는다. **canonical 이 정하고 이 표는 그 근거다** —
제외마다 사유코드가 붙어야 나중에 "좁혀서 통과시킨 것" 과 구별된다.

실측이 두 가지를 섞는다는 점에 주의해야 한다. gold EVT 이력 비율이 낮은 것은
*어휘경계*(head 가 사건 아닌 명사에도 붙는다 — `분위기`·`우주비행사`)일 수도
*대량 미회수*(head 는 사건인데 gold 가 안 달았다 — `교통사고`·`원전사고`)일 수도
있고, **비율만으로는 갈리지 않는다.** 동형 목록을 사람이 읽어야 갈린다.
"""

from __future__ import annotations

import argparse
import collections
import json
import hashlib
import logging
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

# 조사 처리·삽입 무결성 검사는 R2 감사의 구현을 그대로 쓴다. 어절에서 조사를 어떻게
# 떼는지가, 또 불변 타입이 안 움직였는지를 무엇으로 판정하는지가 두 벌이 되면 같은
# 자리를 두 모듈이 다르게 읽어, 한쪽 게이트가 통과시킨 것을 다른 쪽이 못 보게 된다
# — 자가 둘이 되는 것과 같다.
from ner.labelers.ko.ko_evt_r2_audit import (
    CANONICAL_PATH,
    EVT,
    _load_jsonl,
    _strip_particle,
    dump_gold,
    extract_noun_candidates,
    file_sha256,
    load_gold,
    verify_invariants,
)

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"\S+")
_EDGE = re.compile(r"^[^가-힣A-Za-z0-9]+|[^가-힣A-Za-z0-9]+$")
_WORD_CHAR = re.compile(r"[가-힣A-Za-z0-9]")

# 고유명 판정에 쓰는 타입 — canonical 이 고유명으로 다루는 넷이다.
PROPER_LABELS = ("PER", "LOC", "ORG", "PROD")
# 전역 라벨 이력을 근거로 쓸 때의 가드. 길이만 보면 1 글자 성씨(`이`)가, 비율을
# 안 보면 `경찰`(고유명 비율 0.01) 같은 저비율 표면형이 딸려 들어온다.
DEFAULT_MIN_PROPER_LEN = 2
DEFAULT_MIN_PROPER_RATIO = 0.3

# 후보로 볼 최소 등장 횟수. 낮추면 표가 커지고(사람이 사유코드를 붙여야 한다),
# 높이면 드문 사건 head 가 조용히 빠진다 — `마라톤`·`폭발`·`화재` 는 3~4 회다.
DEFAULT_MIN_COUNT = 3
# 접미가 head 로 인정받는 데 필요한 생산성(서로 다른 EVT span 어절 종수).
DEFAULT_MIN_TYPES = 3
# 1 글자 어절은 형태 매칭이 어휘 경계를 식별하지 못해 head 로 쓸 수 없다.
MIN_HEAD_LEN = 2


# ── canonical §5.3 축1-복합 — 고유명이 앞설 때 EVT 가 되는 사건 head ──────
#
# **정본은 canonical 표이고 아래는 사본이다.** 규칙은 기준 파일(사람 승인·반박자를
# 타는 곳)에 있는데 그것을 집행하는 이 파일은 잠금 밖이라, head 를 조용히 좁히면
# 자리 스캔의 모집단이 줄어 뒤따르는 게이트가 그냥 통과한다.

AXIS1_HEADS: Tuple[str, ...] = (
    # 사건·재해
    "사건", "사고", "참사", "사태", "테러", "전쟁", "해전", "위기",
    "침몰", "붕괴", "폭발", "화재", "대지진", "충돌", "민주화운동",
    # 대회·행사
    "월드컵", "올림픽", "아시안게임", "아시안컵", "챔피언십", "세계선수권",
    "마라톤", "콩쿠르", "페스티벌", "축제", "영화제", "박람회", "모터쇼",
    "콘서트", "언팩",
)

# 제외 — canonical 이 사유코드를 정하고 이 코드는 집행만 한다. 왜 빠졌는지가
# 남아야 나중에 "좁혀서 통과시킨 것" 과 구별된다.
AXIS1_EXCLUDE: Dict[str, Tuple[str, ...]] = {
    "타규칙(R2)": (
        "기자회견", "청문회", "토론회", "포럼", "간담회", "공청회", "설명회",
        "발표회", "회담", "총회", "접촉", "위원회", "학회",
    ),
    "타규칙(R2c)": (
        "시상식", "개막식", "영결식", "추모식", "기념식", "갈라쇼", "취임식",
        "연기대상", "연예대상", "방송연예대상",   # 시상식 소관 — `대상`(對象)과 갈린다
    ),
    "타규칙(축2)": (
        "대선", "수능", "총선", "국정감사", "정기국회", "임시국회", "대회",
        "영장실질심사", "대정부질문", "선거", "조사", "투표", "시험", "보선",
        "슈퍼볼", "상봉",
    ),
    "타규칙(R3)": ("예선", "플레이오프", "경기"),
    "타규칙(리그)": ("리그", "시리즈"),
    "비1회성": ("운동", "훈련"),
    "비사건행위": ("합의", "규탄", "기념", "회항"),
    "어휘경계": ("행사", "대상", "화제", "스포"),
    "수식어": (
        "국회의원", "이산가족", "남북", "한일", "한중", "글로벌", "청소년",
        "국가대표", "모자", "여대생", "원전", "폭탄", "삼성", "적십자",
        "해병대캠프", "테크노밸리", "환풍구", "대구역", "땅콩", "여객선",
        "여객기", "헬기", "버스", "항공", "대교", "주년", "월드", "코리아",
        "오브", "스핀", "골든", "세미", "나이트", "파이트", "레이스", "오픈",
        "대통령", "의원",
    ),
    "개별사건명": (
        "하이옌", "세계대전", "촛불집회", "민중총궐기", "철도파업", "지뢰도발",
        "대종상", "발롱도르", "연고전", "가나전", "한국전", "산청엑스포",
        "미스코리아",
    ),
    "고유명": ("소치", "판교", "자그레브", "평창", "개성공단", "선댄스"),
    "수치토큰": ("2013",),
}

CODE_ADOPTED = "채택"
CODE_COVERED = "상위head포함"

_BACKTICK = re.compile(r"`([^`]+)`")
_AXIS1_HEAD_MARKER = re.compile(r"\*\*축1 head[^*]*\*\*:\s*(.+)$")
_AXIS1_EXCLUDE_MARKER = re.compile(r"축1-복합 제외 `([^`]+)`")
_PROPER_MARKER = re.compile(r"\*\*고유명 판정\*\*.*?`([A-Z/]+)`")
_GUARD_MARKER = re.compile(r"길이 ≥(\d+)\s*·\s*고유명 라벨 비율 ≥([\d.]+)")


def parse_canonical_axis1(path: str = CANONICAL_PATH) -> Dict[str, object]:
    """canonical §5.3 에서 모집단 파라미터 **넷**을 읽어 온다.

    head 목록만 읽으면 부족하다 — 고유명으로 볼 타입, 전역 이력을 쓸 때의 길이·비율
    가드, 사유코드의 이름이 모두 모집단을 바꾸는데 그것들이 잠금 밖에 남으면 조용히
    좁힐 수 있다. 규칙은 기준 파일(사람 승인·반박자를 타는 곳)에 있고 그것을 세는 이
    모듈은 잠금 밖이라, 테스트가 이 결과와 모듈 상수를 대조해 어긋남을 실패로 만든다.

    **읽는 것은 이 넷뿐이다** — head 목록·고유명 타입·길이/비율 가드·사유코드 이름.
    §5.3 이 산문으로 정하는 나머지(head 매칭 방식·고유명 판정 근거의 구성·인접
    조건·축3 경계 조건절·제외 어휘 목록)는 이 모듈이 코드로 집행하며 여기서 안 읽는다.
    `canonical_rule_sha256()` 이 그 사실을 그대로 물려받으므로 그쪽 설명을 함께 볼 것.
    """
    heads: Tuple[str, ...] = ()
    proper_labels: Tuple[str, ...] = ()
    codes: List[str] = []
    min_len = 0
    min_ratio = 0.0
    in_ko_table = False
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("### 5.3"):
            in_ko_table = True
            continue
        if in_ko_table and line.startswith("## "):
            break
        if not in_ko_table or not line.startswith("|"):
            continue
        marker = _AXIS1_HEAD_MARKER.search(line)
        if marker:
            heads = tuple(_BACKTICK.findall(marker.group(1)))
        proper = _PROPER_MARKER.search(line)
        if proper:
            proper_labels = tuple(proper.group(1).split("/"))
        guard = _GUARD_MARKER.search(line)
        if guard:
            min_len = int(guard.group(1))
            min_ratio = float(guard.group(2))
        codes.extend(_AXIS1_EXCLUDE_MARKER.findall(line))
    return {
        "heads": heads,
        "proper_labels": proper_labels,
        "min_proper_len": min_len,
        "min_proper_ratio": min_ratio,
        "exclude_codes": tuple(dict.fromkeys(codes)),
    }


@dataclass
class HeadCandidate:
    """head 후보 하나와 그 채택·제외를 판단할 코퍼스 실측.

    ``covered_by`` 가 비어 있지 않으면 더 짧은 후보가 이미 그 자리를 덮는다
    (`침몰사고` 는 `사고` 가 덮는다) — 별도 등재가 필요 없다는 근거다.
    """

    head: str
    source: str             # `어절` | `접미` | `어절∪접미`
    span_count: int         # gold EVT span 안에서의 등장 횟수 (그 어절 자체)
    last_count: int         # 그중 span 의 마지막 어절이었던 횟수
    productive_types: int   # 이 head 로 끝나는 서로 다른 EVT span 어절 종수
    homomorph_types: int    # 이 head 로 끝나는 코퍼스 명사 종수 (자기 자신 제외)
    evt_history_types: int  # 그중 gold EVT 표면형 이력이 있는 종수
    other_label_types: Tuple[str, ...] = ()   # 그중 다른 타입으로 라벨된 표면형
    covered_by: Tuple[str, ...] = ()          # 이 head 를 덮는 더 짧은 후보
    span_samples: Tuple[str, ...] = ()        # 이 head 를 품은 gold EVT span 예시
    homomorph_samples: Tuple[str, ...] = ()   # 동형 명사 예시 (EVT 이력 없는 쪽)

    @property
    def evt_history_ratio(self) -> float:
        """동형 명사 중 gold EVT 이력 비율.

        **낮다고 어휘경계가 아니다** — 미회수도 낮게 나온다(모듈 docstring).
        """
        return (self.evt_history_types / self.homomorph_types
                if self.homomorph_types else 0.0)


def evt_span_words(rows: Sequence[dict]) -> Dict[str, Dict[str, int]]:
    """gold EVT span 에 든 모든 어절을 집계한다 → {어절: 통계}.

    ``span`` 은 전체 등장 횟수, ``last`` 는 그중 span 의 마지막 어절이었던 횟수다.
    마지막이 아닌 자리도 세는 것이 이 함수의 요점이다 — gold 가 복합을 더 긴 span
    으로 잡아 두면 head 가 중간에 묻히기 때문이다(모듈 docstring).
    """
    out: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: {"span": 0, "last": 0})
    for row in rows:
        for ent in row.get("entities", []):
            if ent.get("label") != "EVT":
                continue
            tokens = str(ent.get("text", "")).split()
            for index, token in enumerate(tokens):
                word = _EDGE.sub("", token)
                noun = _strip_particle(word) or word
                if len(noun) < MIN_HEAD_LEN:
                    continue
                out[noun]["span"] += 1
                if index == len(tokens) - 1:
                    out[noun]["last"] += 1
    return dict(out)


def survey_head_candidates(
    rows: Sequence[dict], min_count: int = DEFAULT_MIN_COUNT,
    min_types: int = DEFAULT_MIN_TYPES, samples: int = 6,
) -> List[HeadCandidate]:
    """head 후보를 결정적으로 열거하고 후보마다 코퍼스 실측을 붙인다.

    세 필터가 순서대로 걸린다 — 빈도 ``min_count`` · 코퍼스에 단독 명사로 쓰임 ·
    **고유명으로 라벨된 이력 없음**. 마지막 필터가 같은 span 의 수식부를 걷어낸다:
    `보스턴 마라톤 대회 폭발 사건` 에서 `보스턴` 은 LOC 이력이 있어 빠지고 `마라톤`
    은 남는다. 이게 없으면 후보에 고유명이 절반 섞여 표를 읽을 수 없다.
    """
    words = evt_span_words(rows)
    nouns = set(extract_noun_candidates(rows))

    evt_surfaces = set()
    labels: Dict[str, set] = collections.defaultdict(set)
    span_texts: Dict[str, List[str]] = collections.defaultdict(list)
    for row in rows:
        for ent in row.get("entities", []):
            text = str(ent.get("text", ""))
            labels[text].add(ent.get("label"))
            if ent.get("label") == "EVT":
                evt_surfaces.add(text)

    # 후보 출처 둘을 합친다. 어절만 보면 `한국전쟁`·`걸프전쟁` 이 따로 세어져
    # 공통 head `전쟁` 이 문턱 아래로 흩어지고, 접미만 보면 한 낱말에만 붙는 head
    # (`마라톤`)가 생산성 문턱에 걸려 빠진다. 둘 다 놓치면 안 되는 종류다.
    by_suffix: Dict[str, set] = collections.defaultdict(set)
    for word in words:
        for length in range(MIN_HEAD_LEN, len(word) + 1):
            by_suffix[word[-length:]].add(word)

    whole = {w for w, stat in words.items() if stat["span"] >= min_count}
    productive = {s for s, forms in by_suffix.items() if len(forms) >= min_types}
    # 고유명은 head 가 아니라 같은 span 의 수식부다. 판정에는 자리 스캔이 쓰는
    # 어휘를 그대로 쓴다 — 자가 둘이 되면 한쪽이 통과시킨 것을 다른 쪽이 못 본다.
    # 이진 "라벨 이력 있음" 으로 거르면 작품명으로 한 번 쓰인 `전쟁` 까지 죽는다.
    lexicon = proper_noun_lexicon(rows)
    picked = sorted(
        (h for h in (whole | productive) if h in nouns and h not in lexicon),
        key=lambda h: (-words.get(h, {"span": 0})["span"], h),
    )
    picked_set = set(picked)

    for row in rows:
        for ent in row.get("entities", []):
            if ent.get("label") != "EVT":
                continue
            text = str(ent.get("text", ""))
            for token in text.split():
                noun = _strip_particle(_EDGE.sub("", token)) or _EDGE.sub("", token)
                if noun in picked_set and len(span_texts[noun]) < samples:
                    span_texts[noun].append(text)

    out: List[HeadCandidate] = []
    for head in picked:
        homomorph = sorted(
            n for n in nouns if len(n) > len(head) and n.endswith(head)
        )
        history = [n for n in homomorph if n in evt_surfaces]
        other = tuple(
            f"{n}({'/'.join(sorted(labels[n]))})"
            for n in homomorph
            if n not in evt_surfaces and labels.get(n) and "EVT" not in labels[n]
        )
        stat = words.get(head, {"span": 0, "last": 0})
        out.append(HeadCandidate(
            head=head,
            source=("어절∪접미" if head in whole and head in productive
                    else "어절" if head in whole else "접미"),
            span_count=stat["span"],
            last_count=stat["last"],
            productive_types=len(by_suffix.get(head, ())),
            homomorph_types=len(homomorph),
            evt_history_types=len(history),
            other_label_types=other[:samples],
            covered_by=tuple(
                t for t in picked_set
                if t != head and len(t) < len(head) and head.endswith(t)
            ),
            span_samples=tuple(span_texts.get(head, ())),
            homomorph_samples=tuple(
                n for n in homomorph if n not in evt_surfaces
            )[:samples],
        ))
    return out


def survey_report(
    rows: Sequence[dict], min_count: int = DEFAULT_MIN_COUNT,
    min_types: int = DEFAULT_MIN_TYPES,
) -> dict:
    """열거 산출물 — 표를 재현하는 데 필요한 파라미터를 값과 함께 남긴다.

    문턱이 결과를 바꾸므로 표만 커밋하면 다음 사람이 같은 표를 못 만든다.
    """
    cands = survey_head_candidates(rows, min_count=min_count,
                                   min_types=min_types)
    return {
        "params": {
            "min_count": min_count,
            "min_types": min_types,
            "min_head_len": MIN_HEAD_LEN,
            "proper_labels": list(PROPER_LABELS),
        },
        "gold_rows": len(rows),
        "evt_spans": sum(
            1 for row in rows for ent in row.get("entities", [])
            if ent.get("label") == "EVT"
        ),
        "candidates": [
            {
                "head": c.head,
                "source": c.source,
                "span_count": c.span_count,
                "last_count": c.last_count,
                "productive_types": c.productive_types,
                "homomorph_types": c.homomorph_types,
                "evt_history_types": c.evt_history_types,
                "evt_history_ratio": round(c.evt_history_ratio, 4),
                "other_label_types": list(c.other_label_types),
                "covered_by": sorted(c.covered_by),
                "span_samples": list(c.span_samples),
                "homomorph_samples": list(c.homomorph_samples),
            }
            for c in cands
        ],
    }


@dataclass
class Axis1Site:
    """축1-복합 후보 **자리** 하나.

    표면형이 아니라 자리(row, span)다 — 같은 표면형이 행마다 다른 경계·다른 판정을
    갖는 것이 실측되므로(`보스턴 마라톤` 은 네 행에서 네 상태다), 표면형 단위로 세면
    한 행의 정당한 회수가 다른 행을 사면한다.
    """

    row_index: int
    start: int
    end: int
    surface: str
    head: str
    proper: str
    proper_basis: str      # `동행라벨` | `전역이력`
    attachment: str        # `어절내` | `선행어절`
    gold_evt_overlap: str  # `none` | `exact` | `inside_longer` | `partial`
    insert_start: int = -1   # canonical 축3 조건절을 적용한 실제 삽입 경계
    insert_end: int = -1
    boundary_reason: str = ""      # 왼쪽 경계를 무엇이 확정했나
    clash_labels: Tuple[str, ...] = ()   # 삽입 경계와 겹치는 비-EVT gold 라벨


def proper_noun_lexicon(
    rows: Sequence[dict],
    min_len: int = DEFAULT_MIN_PROPER_LEN,
    min_ratio: float = DEFAULT_MIN_PROPER_RATIO,
) -> Dict[str, float]:
    """전역 라벨 이력으로 고유명 어휘를 만든다 → {표면형: 고유명 라벨 비율}.

    **동행 라벨만 쓰면 안 된다** — gold 가 그 행에서 고유명을 안 달았을 뿐인 자리가
    통째로 빠지고, 그게 이 이슈의 대표 사례다(`보스턴 마라톤` 행의 `보스턴` 은
    무라벨). 그렇다고 전역 이력을 그대로 쓰면 1 글자 성씨와 저비율 표면형이
    딸려오므로 길이·비율 가드를 건다.
    """
    occurrences: collections.Counter = collections.Counter()
    proper: collections.Counter = collections.Counter()
    for row in rows:
        for match in _TOKEN.finditer(row["text"]):
            word = _EDGE.sub("", match.group())
            noun = _strip_particle(word) or word
            if noun:
                occurrences[noun] += 1
        for ent in row.get("entities", []):
            if ent.get("label") in PROPER_LABELS:
                proper[str(ent.get("text", ""))] += 1
    return {
        surface: count / max(occurrences.get(surface, 0), 1)
        for surface, count in proper.items()
        if len(surface) >= min_len
        and count / max(occurrences.get(surface, 0), 1) >= min_ratio
    }


def _evt_spans(row: dict) -> List[Tuple[int, int]]:
    return [
        (int(e["start_char"]), int(e["end_char"]))
        for e in row.get("entities", [])
        if e.get("label") == "EVT"
    ]


def _overlap_kind(span: Tuple[int, int], evt: Sequence[Tuple[int, int]]) -> str:
    """후보 자리가 기존 gold EVT span 과 어떤 관계인가.

    **열거는 관계를 재고 정책은 정하지 않는다.** 겹친 자리를 여기서 버리면 `더 긴
    EVT span 안` 버킷이 세어지지 않아, 회수량과 흡수량이 구별되지 않는다.
    """
    start, end = span
    for e_start, e_end in evt:
        if (start, end) == (e_start, e_end):
            return "exact"
        if e_start <= start and end <= e_end:
            return "inside_longer"
        if start < e_end and e_start < end:
            return "partial"
    return "none"


def find_axis1_sites(
    rows: Sequence[dict],
    heads: Sequence[str],
    min_len: int = DEFAULT_MIN_PROPER_LEN,
    min_ratio: float = DEFAULT_MIN_PROPER_RATIO,
) -> List[Axis1Site]:
    """`고유명 + 사건 head` 자리를 전수 열거한다.

    고유명이 붙는 형태는 둘이다 — 한 어절 안(`서울월드컵`)과 앞 어절(`보스턴 테러`).
    **앞 어절에 조사가 붙어 있으면 제외한다**: `파리에서 테러`·`세월호는 사고` 는
    복합명사가 아니라 문장 성분이라, 그 자리를 회수하면 gold 에 없는 경계를 만든다.
    """
    lexicon = proper_noun_lexicon(rows, min_len=min_len, min_ratio=min_ratio)
    ordered = sorted(heads, key=len, reverse=True)
    out: List[Axis1Site] = []
    for index, row in enumerate(rows):
        text = row["text"]
        evt = _evt_spans(row)
        row_proper = {
            str(e.get("text", "")) for e in row.get("entities", [])
            if e.get("label") in PROPER_LABELS and len(str(e.get("text", ""))) >= min_len
        }
        tokens = [(m.group(), m.start()) for m in _TOKEN.finditer(text)]
        for position, (raw, offset) in enumerate(tokens):
            word = _EDGE.sub("", raw)
            shift = raw.find(word) if word else 0
            noun = _strip_particle(word) or word
            start = offset + shift
            for head in ordered:
                if not noun.endswith(head):
                    continue
                end = start + len(noun)
                prefix = noun[: -len(head)]
                site = None
                if prefix and (prefix in row_proper or prefix in lexicon):
                    site = (start, end, prefix, "어절내")
                elif not prefix and position > 0:
                    prev_raw, prev_offset = tokens[position - 1]
                    prev = _EDGE.sub("", prev_raw)
                    # 조사가 붙어 있으면 수식이 아니라 문장 성분이다
                    if _strip_particle(prev) is None and (
                            prev in row_proper or prev in lexicon):
                        site = (prev_offset + prev_raw.find(prev), end, prev,
                                "선행어절")
                if site is None:
                    break
                s_start, s_end, proper, attachment = site
                basis = "동행라벨" if proper in row_proper else "전역이력"
                head_start = s_end - len(head)
                # canonical 축3 경계 조건절 — gold 는 평면 BIO 라 중첩이 표현되지
                # 않는다. 라벨된 인접 고유명은 삼킬 수 없으므로 그만큼 왼쪽을 자르고,
                # 잘라도 head 까지 덮여 있으면 삽입 자체가 불가능한 타입 충돌이다.
                # **부착 형태로 가리지 않는다** — `서울월드컵` 처럼 한 어절 안에서도
                # gold 가 앞부분(`서울`)만 LOC 로 잡아 두는 자리가 있다.
                covering = [
                    e for e in row.get("entities", [])
                    if e.get("label") != "EVT"
                    and s_start < int(e["end_char"]) and int(e["start_char"]) < s_end
                ]
                clash = tuple(sorted({
                    str(e.get("label")) for e in covering
                    if head_start < int(e["end_char"])
                    and int(e["start_char"]) < s_end
                }))
                i_start, reason = s_start, "고유명 포함 — 인접 고유명 무라벨"
                if clash:
                    reason = "삽입 불가 — head 까지 다른 타입이 덮는다"
                elif covering:
                    i_start = max(int(e["end_char"]) for e in covering)
                    # 공백만 건너뛰면 `\'` 같은 인용부호가 span 머리에 남는다
                    while i_start < s_end and not _WORD_CHAR.match(text[i_start]):
                        i_start += 1
                    reason = "head 만 — 인접 고유명이 이미 라벨됨"
                out.append(Axis1Site(
                    row_index=index,
                    start=s_start,
                    end=s_end,
                    surface=text[s_start:s_end],
                    head=head,
                    proper=proper,
                    proper_basis=basis,
                    attachment=attachment,
                    gold_evt_overlap=_overlap_kind((i_start, s_end), evt),
                    insert_start=i_start,
                    insert_end=s_end,
                    boundary_reason=reason,
                    clash_labels=clash,
                ))
                break
    return out


def heads_sha256(heads: Sequence[str]) -> str:
    """head 목록의 지문. 목록이 한 종만 달라져도 값이 바뀐다."""
    joined = "\n".join(sorted(heads))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def canonical_rule_sha256(path: str = CANONICAL_PATH) -> str:
    """canonical §5.3 이 정하는 **규칙 내용**의 지문.

    **왜 파일 전체 해시로는 부족한가.** 사전등록이 잡아야 할 것은 head 를
    넓히거나 가드를 푸는 일인데, 파일 해시는 그것과 실측 주석 한 줄 추가를 같은
    신호로 만든다. 그러면 문서를 정당하게 손볼 때마다 사전등록이 깨지고, 깨진
    것을 푸는 유일한 길이 지문 갱신이라 결국 "언제든 갱신 가능한 값" 이 된다 —
    집행하려던 것이 사라진다.

    그래서 `parse_canonical_axis1()` 이 읽는 것만 정규화해 잰다.

    **이 지문이 재는 것은 정확히 넷이다** — head 목록 · 고유명으로 볼 라벨 타입 ·
    전역 이력 가드의 길이/비율 임계 · 사유코드 이름. 정확히는 그 넷의 **파싱된
    형태**라, 파서가 알아보는 문구로 적힌 것만 잰다 — 같은 내용을 파서가 못 읽는
    형태로 옮기면 값이 안 움직인다(그쪽은 파싱↔모듈 상수 동기 테스트가 잡는다).

    **그 밖의 §5.3 규칙은 이 값이 재지 않는다.** 못 보는 것을 목록으로 적으면 그
    목록을 다 닫았을 때 "이제 안전하다" 로 읽히는데, §5.3 은 산문이라 규칙이 몇
    가지인지 자체가 열려 있다. 실제로 head 매칭 방식(`끝나는` ↔ `포함하는`)·고유명
    판정 근거의 구성(동행 라벨 ∪ 전역 이력)·인접 조건(조사 배제·수식어 개재)·축3
    경계 조건절·제외 어휘 목록을 바꿔도 이 값은 안 움직인다. 확인된 것만 다섯이고
    닫힌 목록이 아니다.

    **그러므로 이 값의 불변은 "규칙이 안 바뀌었다" 의 증명이 아니다** — 위 넷이
    그대로라는 뜻일 뿐이다. §5.3 의 다른 조항에 기대는 작업(예: #202 가 남긴 인접
    정의 완화)은 이 지문으로 잠기지 않으므로 그때 잠글 수단을 따로 만들어야 한다.
    옛 파일 전체 해시는 그것까지 잡았지만 실측 주석 한 줄에도 깨져 갱신을 강요했다 —
    넓게 새는 쪽 대신 좁게 확실한 쪽을 골랐고, 그 대가가 이 범위 제한이다.
    """
    rule = parse_canonical_axis1(path)
    payload = json.dumps(rule, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sites_report(
    rows: Sequence[dict],
    heads: Sequence[str],
    min_len: int = DEFAULT_MIN_PROPER_LEN,
    min_ratio: float = DEFAULT_MIN_PROPER_RATIO,
    gold_path: Optional[str] = None,
) -> dict:
    """모집단 산출물 — **재현 파라미터를 값과 함께 남긴다.**

    head 목록·가드 임계·gold 가 모집단을 정하므로, 값만 커밋하면 다음 사람이 모집단이
    왜 달라졌는지 판단할 수 없고 좁히기가 사고와 구별되지 않는다. head 지문과 gold
    지문을 함께 남기는 이유가 이것이다 — 둘 중 무엇이 움직였는지 값으로 갈린다.
    """
    sites = find_axis1_sites(rows, heads, min_len=min_len, min_ratio=min_ratio)
    buckets: collections.Counter = collections.Counter(
        s.gold_evt_overlap for s in sites
    )
    # 부착 형태 × 고유명 판정 근거의 교차 분포. 개별 자리에만 있고 집계가 없으면
    # "인접 정의를 완화하면 정밀도가 어떻게 되나" 를 재려는 다음 사람이 매번 스캔을
    # 다시 돌려야 하고, 그 수는 어디에도 커밋돼 있지 않아 비교 기준이 되지 못한다.
    attachment_x_basis: Dict[str, int] = {}
    for s in sites:
        key = f"{s.attachment}×{s.proper_basis}"
        attachment_x_basis[key] = attachment_x_basis.get(key, 0) + 1
    return {
        "params": {
            "heads": sorted(heads),
            "heads_sha256": heads_sha256(heads),
            "gold_sha256": file_sha256(gold_path) if gold_path else None,
            "proper_labels": list(PROPER_LABELS),
            "min_proper_len": min_len,
            "min_proper_ratio": min_ratio,
        },
        "gold_rows": len(rows),
        "sites": len(sites),
        "gold_evt_overlap": dict(buckets),
        "attachment_x_basis": dict(sorted(attachment_x_basis.items())),
        "population": buckets.get("none", 0),
        "items": [
            {
                "row_index": s.row_index,
                "start": s.start,
                "end": s.end,
                "surface": s.surface,
                "head": s.head,
                "proper": s.proper,
                "proper_basis": s.proper_basis,
                "attachment": s.attachment,
                "gold_evt_overlap": s.gold_evt_overlap,
            }
            for s in sites
        ],
    }


def assign_codes(cands: Sequence[HeadCandidate]) -> Dict[str, str]:
    """후보마다 canonical 사유코드를 붙인다 — **미부여가 남으면 안 된다.**

    두 단계 상속이 있다.

    - `상위head포함` 은 **채택된** 더 짧은 head 가 덮을 때만 쓴다. 덮는 head 가
      제외됐으면 긴 쪽은 자기 코드가 필요하다 — `영화제` 는 `화제`(어휘경계)가
      덮지만 그 자체로 head 다.
    - 제외된 head 가 덮으면 **그 조항이 긴 형태도 소유한다** (`여론조사` 는 `조사`
      와 같은 축2 소관). 상속을 안 하면 같은 규칙에 속한 것이 코드 없이 남아
      "빠뜨린 것" 과 구별되지 않는다.
    """
    code: Dict[str, str] = {}
    known = {c.head: c for c in cands}
    for head in AXIS1_HEADS:
        if head in known:
            code[head] = CODE_ADOPTED
    for name, heads in AXIS1_EXCLUDE.items():
        for head in heads:
            if head in known:
                code[head] = name
    adopted = {h for h, c in code.items() if c == CODE_ADOPTED}
    for cand in cands:
        if cand.head not in code and any(t in adopted for t in cand.covered_by):
            code[cand.head] = CODE_COVERED
    for _ in range(len(AXIS1_EXCLUDE)):      # 다단 상속 (재보궐선거 ← 보궐선거 ← 선거)
        changed = False
        for cand in cands:
            if cand.head in code:
                continue
            owners = [code[t] for t in cand.covered_by
                      if code.get(t, "").startswith("타규칙")]
            if owners:
                code[cand.head] = owners[0]
                changed = True
        if not changed:
            break
    return code


def codes_report(
    rows: Sequence[dict], min_count: int = DEFAULT_MIN_COUNT,
    min_types: int = DEFAULT_MIN_TYPES,
) -> dict:
    """사유코드 배정 원장 — 어느 후보가 어느 코드를 받았나.

    **이 원장이 없으면 "후보 전부에 코드를 붙였다" 가 반증 불가능한 주장이 된다.**
    열거는 재현되는데 배정은 사람이 하므로, 선택 편향이 살 수 있는 자리가 여기다.
    """
    cands = survey_head_candidates(rows, min_count=min_count, min_types=min_types)
    code = assign_codes(cands)
    unassigned = sorted(c.head for c in cands if c.head not in code)
    unknown = sorted(
        h for h in list(AXIS1_HEADS) + [x for v in AXIS1_EXCLUDE.values() for x in v]
        if h not in {c.head for c in cands}
    )
    summary: collections.Counter = collections.Counter(code.values())
    return {
        "params": {"min_count": min_count, "min_types": min_types},
        "candidates": len(cands),
        "assigned": len(code),
        "unassigned": unassigned,
        "not_in_candidates": unknown,
        "summary": dict(summary.most_common()),
        "codes": {
            c.head: {
                "code": code.get(c.head, ""),
                "span_count": c.span_count,
                "last_count": c.last_count,
                "covered_by": sorted(c.covered_by),
            }
            for c in cands
        },
    }


def cmd_codes(args: argparse.Namespace) -> None:
    rows = load_gold(args.gold)
    report = codes_report(rows)
    logger.info("candidates %d / assigned %d / unassigned %d",
                report["candidates"], report["assigned"],
                len(report["unassigned"]))
    if report["unassigned"]:
        logger.error("unassigned candidates: %s", " ".join(report["unassigned"]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        logger.info("wrote %s", args.out)
    if report["unassigned"] and args.gate:
        raise SystemExit(1)


STATUS_ALREADY = "already_evt"
STATUS_RECOVERED = "recovered"
STATUS_EXCLUDED = "excluded"
STATUS_JUDGED_NOT = "judged_not"
STATUS_COVERED = "covered_by_longer_evt"
STATUS_TYPE_CLASH = "type_clash_recorded"
STATUS_UNCLASSIFIED = "unclassified"


def _site_key(site: Axis1Site) -> Tuple[int, int, int]:
    """자리의 신원은 **탐지 span** 이다.

    삽입 span 이 아니라 탐지 span 으로 잡는 이유는 경계 규칙이 바뀌어도 같은 자리를
    가리키기 위해서다 — 신원이 삽입 경계에 묶이면 규칙을 손볼 때마다 판정이 통째로
    떨어져 나간다.
    """
    return (site.row_index, site.start, site.end)


def ledger_verdicts(ledger: Sequence[dict]) -> Dict[Tuple[int, int, int], str]:
    """판정 원장을 자리별로 읽는다 — 표면형이 아니라 자리다.

    같은 표면형이 행마다 다른 판정을 받는 것이 실측되므로(`세월호 사고` 는 한 행에서
    사건이고 다른 행에서 대책위 이름의 일부다), 표면형 단위로 접으면 한 행의 정당한
    판정이 다른 행을 사면한다.
    """
    out: Dict[Tuple[int, int, int], str] = {}
    for rec in ledger:
        try:
            key = (int(rec["row_index"]), int(rec["start"]), int(rec["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = str(rec.get("verdict", "")).upper()
    return out


def classify_axis1_sites(
    sites: Sequence[Axis1Site], ledger: Sequence[dict] = (),
    excluded_surfaces: Sequence[str] = (),
) -> List[Tuple[Axis1Site, str]]:
    """자리마다 상태를 붙인다 — 어디에도 안 들면 미분류다.

    순서가 의미를 갖는다. 이미 gold EVT 인 자리와 더 긴 EVT 안에 든 자리는 판정
    대상이 아니고, 삽입 자체가 불가능한 타입 충돌은 **범위 밖으로 선언하되 세어서**
    남긴다 — 이 출구가 없으면 게이트를 통과시키는 유일한 길이 "원장에 NOT 이라고
    쓰는 것" 뿐이라, 사람이 판정해도 결론이 강제돼 판정이 형식이 된다.
    """
    verdicts = ledger_verdicts(ledger)
    excluded = set(excluded_surfaces)
    out: List[Tuple[Axis1Site, str]] = []
    for site in sites:
        key = _site_key(site)
        verdict = verdicts.get(key, "")
        if site.gold_evt_overlap == "exact":
            status = STATUS_ALREADY
        elif site.gold_evt_overlap == "inside_longer":
            status = STATUS_COVERED
        elif site.clash_labels:
            status = STATUS_TYPE_CLASH
        elif verdict == "EVT":
            status = STATUS_RECOVERED
        elif verdict == "NOT":
            status = STATUS_JUDGED_NOT
        elif site.surface in excluded or site.head in excluded:
            status = STATUS_EXCLUDED
        else:
            status = STATUS_UNCLASSIFIED
        out.append((site, status))
    return out


def check_axis1_gate(
    rows: Sequence[dict],
    heads: Sequence[str] = AXIS1_HEADS,
    ledger: Sequence[dict] = (),
    gold_path: Optional[str] = None,
) -> dict:
    """주 게이트 — 축1-복합 자리가 전부 분류됐나. 미분류가 0 이어야 통과한다.

    **회수 건수를 함께 보고하는 이유**: 모든 자리가 제외·흡수·충돌로 분류되면
    **0 건 회수도 전 기준을 통과한다.** 0 에 가까우면 그 자체가 재검토 신호다.
    """
    sites = find_axis1_sites(rows, heads)
    excluded = [f for forms in AXIS1_EXCLUDE.values() for f in forms]
    classified = classify_axis1_sites(sites, ledger, excluded)
    counts: collections.Counter = collections.Counter(s for _, s in classified)
    judged = counts[STATUS_RECOVERED] + counts[STATUS_JUDGED_NOT]
    pending = [
        {"row_index": s.row_index, "start": s.start, "end": s.end,
         "surface": s.surface, "head": s.head,
         "insert": [s.insert_start, s.insert_end],
         "boundary_reason": s.boundary_reason}
        for s, st in classified if st == STATUS_UNCLASSIFIED
    ]
    return {
        "params": {
            "heads_sha256": heads_sha256(heads),
            "gold_sha256": file_sha256(gold_path) if gold_path else None,
            "min_proper_len": DEFAULT_MIN_PROPER_LEN,
            "min_proper_ratio": DEFAULT_MIN_PROPER_RATIO,
        },
        "sites": len(sites),
        "status_counts": dict(counts.most_common()),
        "unclassified": pending,
        "recovered": counts[STATUS_RECOVERED],
        "judged": judged,
        # ④ 가 대량 흡수 통로가 되면 회수량이 아니라 흡수량을 보고 있는 셈이라,
        # 비율이 과반이면 사람이 표본을 확인해야 한다.
        "covered_ratio": (counts[STATUS_COVERED] / len(sites)) if sites else 0.0,
        "covered_by_longer_evt": [
            {"row_index": s.row_index, "surface": s.surface}
            for s, st in classified if st == STATUS_COVERED
        ],
        "type_clash_recorded": [
            {"row_index": s.row_index, "surface": s.surface,
             "labels": list(s.clash_labels)}
            for s, st in classified if st == STATUS_TYPE_CLASH
        ],
    }


# ── 비순환 재채점 ──────────────────────────────────────────────────────


def load_pred_spans(paths: Sequence[str]) -> List[Tuple[str, Dict[str, List[dict]]]]:
    """보존된 예측 span 을 fold 별로 읽는다 → [(fold 이름, {row id: [span]})]."""
    out: List[Tuple[str, Dict[str, List[dict]]]] = []
    for path in paths:
        raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        name = pathlib.Path(path).parent.name
        out.append((name, {
            str(row_id): [{"type": t, "start": int(s), "end": int(e)}
                          for t, s, e in spans]
            for row_id, spans in raw.items()
        }))
    return out


def _gold_spans(row: dict) -> List[dict]:
    return [{"type": str(e["label"]), "start": int(e["start_char"]),
             "end": int(e["end_char"])} for e in row.get("entities", [])]


def _score(gold: Sequence[List[dict]], pred: Sequence[List[dict]]) -> dict:
    from ner.metrics.span_metrics import compute_offset_span_f1
    return compute_offset_span_f1(list(gold), list(pred))


def rescore_arms(
    rows: Sequence[dict],
    base_folds: Sequence[Tuple[str, Dict[str, List[dict]]]],
    head_folds: Sequence[Tuple[str, Dict[str, List[dict]]]],
    skip_ids: Sequence[str] = (),
) -> dict:
    """두 팔의 예측을 **같은 gold** 로 다시 채점한다.

    base 팔은 옛 gold 로 학습·채점됐다. 그 수치를 신 gold 수치와 나란히 놓으면 자가
    둘이라 비교가 성립하지 않는다 — 회수가 EVT 분모(support)를 바꿨기 때문이다.
    그래서 예측은 그대로 두고 채점만 신 gold 로 다시 한다.

    ``skip_ids`` 는 회수가 손댄 행이다. 그 행을 뺀 부분집합의 Δ 가 **회수 무관 Δ** 로,
    "gold 에 답을 넣어 준 자리" 밖에서도 나아졌나를 본다.

    **pooled Δ 와 paired Δ 는 성립 조건이 다르다.** pooled 는 두 팔이 같은 행 집합을
    한 번씩 덮으면 되므로 fold 배정과 무관하다. paired 는 fold 가 서로 대응해야 하는데,
    gold 를 바꾸면 층화 분할이 통째로 움직여 대응이 깨진다. 깨진 채로 짝지어 계산하면
    **서로 다른 문장 집합을 비교한 수가 그럴듯한 모습으로 나온다** — 그래서 대응이
    깨졌으면 paired 를 내지 않고 안 낸 이유와 실측 겹침을 남긴다.
    """
    by_id = {str(row.get("id")): row for row in rows}
    skip = {str(i) for i in skip_ids}
    base = {i: spans for _, fold in base_folds for i, spans in fold.items()}
    head = {i: spans for _, fold in head_folds for i, spans in fold.items()}
    for arm, folds, merged in (("base", base_folds, base), ("head", head_folds, head)):
        entries = sum(len(fold) for _, fold in folds)
        if entries != len(merged):
            raise SystemExit(
                f"FAIL: {arm} arm has {entries - len(merged)} rows in more than one fold")
    if set(base) != set(head):
        raise SystemExit(
            f"FAIL: arms cover different rows (base {len(base)}, head {len(head)})")
    # gold 에 없는 행을 조용히 빼면 분모가 줄어든 채로 F1 이 나온다 — 값은 그럴듯하고
    # 무엇이 빠졌는지는 어디에도 안 남는다.
    missing = set(base) - set(by_id)
    if missing:
        raise SystemExit(
            f"FAIL: {len(missing)} predicted rows are absent from gold "
            f"(e.g. {sorted(missing)[:3]})")

    ids = [i for i in sorted(base) if i not in skip]
    pooled_gold = [_gold_spans(by_id[i]) for i in ids]
    scores = {"base": _score(pooled_gold, [base[i] for i in ids]),
              "head": _score(pooled_gold, [head[i] for i in ids])}

    head_by_name = dict(head_folds)
    matched = [(name, fold, head_by_name[name]) for name, fold in base_folds
               if name in head_by_name and set(head_by_name[name]) == set(fold)]
    per_fold: Dict[str, Dict[str, float]] = {}
    if len(matched) == len(base_folds) == len(head_folds):
        for name, fold, other in matched:
            fold_ids = [i for i in sorted(fold) if i not in skip]
            gold = [_gold_spans(by_id[i]) for i in fold_ids]
            per_fold[name] = {
                "base": _score(gold, [fold[i] for i in fold_ids])
                        ["per_entity"].get(EVT, {}).get("f1", 0.0),
                "head": _score(gold, [other[i] for i in fold_ids])
                        ["per_entity"].get(EVT, {}).get("f1", 0.0),
            }
    types = sorted(set(scores["base"]["per_entity"]) | set(scores["head"]["per_entity"]))
    per_entity = {}
    for label in types + ["overall"]:
        def pick(arm: str) -> dict:
            block = scores[arm]
            return block["overall"] if label == "overall" else \
                block["per_entity"].get(label, {})
        base_f1 = round(pick("base").get("f1", 0.0), 4)
        head_f1 = round(pick("head").get("f1", 0.0), 4)
        per_entity[label] = {
            "base": base_f1, "head": head_f1,
            "delta": round(head_f1 - base_f1, 4),
            # 판정에 쓰이는 것은 Δ 가 아니라 Δ 의 크기와 σ 대비 비다. 문서가 그걸
            # 손으로 계산하면 산출물에 없는 수가 표에 앉는다 — 여기서 함께 낸다.
            "abs_delta": round(abs(head_f1 - base_f1), 4),
            "support": pick("head").get("support", 0),
        }
    if per_fold:
        deltas = [v["head"] - v["base"] for v in per_fold.values()]
        mean = sum(deltas) / len(deltas)
        spread = (sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)) ** 0.5 \
            if len(deltas) > 1 else 0.0
        paired = {
            "available": True,
            "mean": round(mean, 4), "stdev": round(spread, 4),
            "wins": sum(1 for d in deltas if d > 0),
            "losses": sum(1 for d in deltas if d < 0),
            "per_fold": {k: {a: round(v, 4) for a, v in vals.items()}
                         for k, vals in sorted(per_fold.items())},
        }
    else:
        overlaps = [len(set(fold) & set(head_by_name.get(name, {}))) / max(len(fold), 1)
                    for name, fold in base_folds]
        paired = {
            "available": False,
            "reason": "fold partitions do not correspond between the arms — "
                      "the stratified split moved when gold changed",
            "mean_fold_overlap": round(sum(overlaps) / max(len(overlaps), 1), 4),
        }
    return {
        "rows_scored": len(pooled_gold),
        "per_entity": per_entity,
        "paired_evt": paired,
    }


def cmd_rescore(args: argparse.Namespace) -> None:
    rows = load_gold(args.gold)
    ledger = _load_jsonl(args.ledger) if args.ledger else []
    recovered = [str(rec.get("row_id")) for rec in ledger
                 if str(rec.get("verdict", "")).upper() == EVT]
    base = load_pred_spans(args.base_preds)
    head = load_pred_spans(args.head_preds)
    sigma = json.loads(pathlib.Path(args.sigma).read_text(encoding="utf-8")) \
        if args.sigma else {}
    report = {
        "method": "base 팔 예측을 회수 후 gold 로 재채점해 head 팔과 같은 자로 비교한다. "
                  "구 gold 기준 수치와는 나란히 놓지 않는다.",
        "sigma_source": args.sigma,
        # 사전등록과 이어 붙이는 지문 — σ 나 예측이 바뀌면 이 표가 스스로 어긋난다
        "gold_sha256": file_sha256(args.gold),
        "sigma_sha256": file_sha256(args.sigma) if args.sigma else None,
        "pred_sha256": {"base": [file_sha256(p) for p in args.base_preds],
                        "head": [file_sha256(p) for p in args.head_preds]},
        "full": rescore_arms(rows, base, head),
        # 회수가 손댄 행을 뺀다 — gold 에 답을 넣어 준 자리 밖에서도 나아졌나
        "recovery_free_subset": rescore_arms(rows, base, head, skip_ids=recovered),
        "recovered_sites": len(recovered),
    }
    for block in ("full", "recovery_free_subset"):
        for label, values in report[block]["per_entity"].items():
            std = sigma.get(label, {}).get("std")
            if std:
                values["sigma_pre"] = std
                # 밴드 안인가를 결정하는 값 — 사전등록 σ 대비 Δ 의 크기
                values["sigma_ratio"] = round(values["abs_delta"] / std, 2)
    logger.info("EVT delta full %+.4f / recovery-free %+.4f (sigma_pre %.4f)",
                report["full"]["per_entity"][EVT]["delta"],
                report["recovery_free_subset"]["per_entity"][EVT]["delta"],
                sigma.get(EVT, {}).get("std", 0.0))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        logger.info("wrote %s", args.out)


# ── 모델 FP 사각 진단 ──────────────────────────────────────────────────


def _fp_compound_shape(
    text: str, start: int, end: int, row_proper: Sequence[str],
    lexicon: Dict[str, float],
) -> Optional[Tuple[str, str]]:
    """FP span 이 `고유명 + 사건 head` 모양인가 → (고유명, head).

    **head 목록을 쓰지 않는다.** 이 진단은 그 목록의 사각을 재는 것이라, 목록으로
    모양을 판정하면 목록이 못 본 head 는 모양 판정에서부터 빠져 사각이 0 으로 나온다
    — 자기 시야로 자기 시야를 재는 구조다. 그래서 head 는 열린 값으로 둔다: FP span
    의 마지막 어절에서 조사를 뗀 것이면 무엇이든 head 후보다.
    """
    inner = [(m.group(), start + m.start())
             for m in _TOKEN.finditer(text[start:end])]
    if not inner:
        return None
    tail = _EDGE.sub("", inner[-1][0])
    head = _strip_particle(tail) or tail
    if len(head) < MIN_HEAD_LEN or head in lexicon or head in row_proper:
        return None
    if len(inner) > 1:
        prefix = _EDGE.sub("", inner[-2][0])
    else:
        # 한 어절이면 그 안에서 갈린다 — `서울월드컵` 형
        prefix = ""
        for cut in range(len(head) - MIN_HEAD_LEN, MIN_HEAD_LEN - 1, -1):
            if head[:cut] in row_proper or head[:cut] in lexicon:
                prefix, head = head[:cut], head[cut:]
                break
    if not prefix or (prefix not in row_proper and prefix not in lexicon):
        return None
    return prefix, head


def diagnose_fp_blindspot(
    rows: Sequence[dict],
    predictions: Sequence[dict],
    heads: Sequence[str] = AXIS1_HEADS,
    min_len: int = DEFAULT_MIN_PROPER_LEN,
    min_ratio: float = DEFAULT_MIN_PROPER_RATIO,
) -> dict:
    """모델 EVT FP 중 복합 named 모양이 **자리 스캔 모집단에 없는 수**를 센다.

    모델 예측은 head 후보의 출처가 아니다(선택 편향 — gold 가 모델이 이미 발화한
    자리에서만 늘어 recall 상승이 구조적으로 보장된다). 여기서의 쓰임은 진단뿐이다:
    규칙이 못 본 형태가 무엇인지 이름을 얻는다. 넓히는 것은 F1 관측 **전에만**
    허용되고, 그 순서는 산출물에 박히는 해시가 집행한다.
    """
    lexicon = proper_noun_lexicon(rows, min_len=min_len, min_ratio=min_ratio)
    nouns = extract_noun_candidates(rows)
    by_id = {str(row.get("id")): (index, row) for index, row in enumerate(rows)}
    population: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
    for site in find_axis1_sites(rows, heads, min_len=min_len, min_ratio=min_ratio):
        population[site.row_index].append((site.start, site.end))
    excluded = {f for forms in AXIS1_EXCLUDE.values() for f in forms}

    seen: Set[Tuple[str, int, int]] = set()
    stats: collections.Counter = collections.Counter()
    items: List[dict] = []
    for pred in predictions:
        found = by_id.get(str(pred.get("id")))
        if found is None:
            stats["row_id_not_in_gold"] += 1
            continue
        index, row = found
        gold = {(int(s["start"]), int(s["end"])) for s in pred.get("gold_spans", [])
                if s.get("type") == EVT}
        text = row["text"]
        row_proper = [
            str(e.get("text", "")) for e in row.get("entities", [])
            if e.get("label") in PROPER_LABELS
            and len(str(e.get("text", ""))) >= min_len
        ]
        for span in pred.get("pred_spans", []):
            if span.get("type") != EVT:
                continue
            start, end = int(span["start"]), int(span["end"])
            if (start, end) in gold:
                continue
            stats["evt_fp"] += 1
            key = (str(pred.get("id")), start, end)
            if key in seen:
                stats["duplicate_fp"] += 1
                continue
            seen.add(key)
            shape = _fp_compound_shape(text, start, end, row_proper, lexicon)
            if shape is None:
                continue
            proper, head = shape
            stats["compound_fp"] += 1
            covered = any(s < end and start < e for s, e in population[index])
            if covered:
                stats["in_population"] += 1
                continue
            stats["absent_from_population"] += 1
            items.append({
                "row_index": index, "row_id": row.get("id"),
                "span": [start, end], "surface": text[start:end],
                "proper": proper, "head": head,
                "head_in_list": head in heads or any(head.endswith(h) for h in heads),
                "head_excluded": head in excluded,
                # 모델 span 이 낱말을 자르면 head 도 조각이 된다(`(EAS`·`원회`·`+3`).
                # 조각은 사각의 이름이 아니라 경계 오류라, 세되 갈라 둔다 — 판별은
                # 자리 스캔의 head 후보와 같은 기준(코퍼스 단독 명사)으로 한다.
                "head_is_corpus_noun": head in nouns,
                "context": text[max(0, start - 30):end + 30],
            })
    absent_heads = collections.Counter(i["head"] for i in items)
    return {
        "params": {
            "heads_sha256": heads_sha256(heads),
            "min_proper_len": min_len,
            "min_proper_ratio": min_ratio,
        },
        "predictions": len(predictions),
        "stats": dict(stats.most_common()),
        # 사각의 이름 — 이 head 들이 목록에 없어서 모집단이 그 자리를 못 만들었다
        "absent_by_head": dict(absent_heads.most_common()),
        # 목록에 있는데도 모집단 밖이면 head 가 아니라 가드·부착 조건이 걸렀다는 뜻
        "absent_though_head_listed": sum(1 for i in items if i["head_in_list"]),
        "absent_head_excluded_by_code": sum(1 for i in items if i["head_excluded"]),
        "absent_head_not_a_corpus_noun": sum(
            1 for i in items if not i["head_is_corpus_noun"]),
        "items": items,
    }


def cmd_fp_blindspot(args: argparse.Namespace) -> None:
    rows = load_gold(args.gold)
    predictions: List[dict] = []
    for path in args.predictions:
        predictions.extend(json.loads(
            pathlib.Path(path).read_text(encoding="utf-8")))
    report = diagnose_fp_blindspot(rows, predictions)
    # 사전등록 — 이 산출물이 head 팔 학습보다 **먼저** 커밋됐음을 지문으로 남긴다.
    # 나중에 F1 을 보고 head 를 넓히면 해시가 어긋나 산출물 자체가 반박한다.
    report["prereg"] = {
        "gold_sha256": file_sha256(args.gold),
        "canonical_sha256": file_sha256(CANONICAL_PATH),
        "heads_sha256": heads_sha256(AXIS1_HEADS),
        "prediction_sha256": {p: file_sha256(p) for p in args.predictions},
    }
    logger.info("EVT FP %d — compound-shaped %d, absent from population %d %s",
                report["stats"].get("evt_fp", 0),
                report["stats"].get("compound_fp", 0),
                report["stats"].get("absent_from_population", 0),
                report["absent_by_head"])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        logger.info("wrote %s", args.out)


# ── 잔여 불일치 계수기 ─────────────────────────────────────────────────

# 583(최대-span) 위반 — 인접 고유명이 그 행에서 무라벨인데 EVT span 이 안 품었다.
RESIDUE_UNDER = "과축소"
# 586(겹침 정책) 위반 — 라벨된 고유명을 EVT span 이 삼켰다. 실측 0 이 전제다.
RESIDUE_SWALLOW = "고유명삼킴"


def find_boundary_residue(
    rows: Sequence[dict],
    min_len: int = DEFAULT_MIN_PROPER_LEN,
    min_ratio: float = DEFAULT_MIN_PROPER_RATIO,
) -> List[dict]:
    """확정된 축3 경계 조건절을 **기존 EVT span 전수**에 적용해 불일치를 센다.

    소급 교정은 하지 않는다(canonical 584) — 트림 경계 판정은 이미 폐기된 비결정
    구간이다. 그래서 **보고 자체가 게이트**다: 회수된 자리는 gold EVT 이력을 얻어
    자리 스캔의 모집단에서 영구히 빠지므로, 오삽입은 회수를 몇 번 더 돌려도 스스로는
    안 보인다. 이 계수기만 그 뒤를 본다.

    **모집단을 축1 head 로 좁히지 않는다.** 좁히면 규칙이 못 보는 곳을 규칙의 시야로
    재는 셈이라, 넓힌 자리에서 생긴 불일치가 분모 밖으로 빠진다.
    """
    lexicon = proper_noun_lexicon(rows, min_len=min_len, min_ratio=min_ratio)
    out: List[dict] = []
    for index, row in enumerate(rows):
        text = row["text"]
        entities = row.get("entities", [])
        row_proper = {
            str(e.get("text", "")) for e in entities
            if e.get("label") in PROPER_LABELS
            and len(str(e.get("text", ""))) >= min_len
        }
        labeled: Set[int] = set()
        for ent in entities:
            if ent.get("label") != EVT:
                labeled.update(range(int(ent["start_char"]), int(ent["end_char"])))
        tokens = [(m.group(), m.start()) for m in _TOKEN.finditer(text)]
        starts = {offset: position for position, (_, offset) in enumerate(tokens)}
        for start, end in sorted(_evt_spans(row)):
            swallowed = sorted({
                str(e.get("label")) for e in entities
                if e.get("label") != EVT
                and start < int(e["end_char"]) and int(e["start_char"]) < end
            })
            if swallowed:
                out.append({"row_index": index, "row_id": row.get("id"),
                            "kind": RESIDUE_SWALLOW, "span": [start, end],
                            "surface": text[start:end], "candidate": "",
                            "labels": swallowed, "expected": ""})
                continue
            # 왼쪽에 무엇이 붙어 있나 — 어절 안이면 그 앞부분, 어절 머리면 앞 어절.
            position = starts.get(start)
            if position is not None:
                if position == 0:
                    continue
                prev_raw, prev_offset = tokens[position - 1]
                candidate = _EDGE.sub("", prev_raw)
                # 조사가 붙어 있으면 수식이 아니라 문장 성분이다
                if not candidate or _strip_particle(candidate) is not None:
                    continue
                cand_start = prev_offset + prev_raw.find(candidate)
            else:
                token = max((t for t in tokens if t[1] < start), key=lambda t: t[1],
                            default=None)
                if token is None:
                    continue
                raw, offset = token
                if offset + len(raw) < end:
                    continue          # span 이 어절 경계와 어긋난다 — 다른 관심사다
                candidate = _EDGE.sub("", text[offset:start])
                if not candidate:
                    continue
                cand_start = offset + text[offset:start].find(candidate)
            if candidate not in row_proper and candidate not in lexicon:
                continue
            # 그 고유명이 라벨돼 있으면 안 삼킨 것이 규칙대로다 — 위반이 아니다.
            if labeled & set(range(cand_start, cand_start + len(candidate))):
                continue
            out.append({"row_index": index, "row_id": row.get("id"),
                        "kind": RESIDUE_UNDER, "span": [start, end],
                        "surface": text[start:end], "candidate": candidate,
                        "labels": [], "expected": text[cand_start:end]})
    return out


def residue_report(
    rows: Sequence[dict], gold_path: Optional[str] = None,
    min_len: int = DEFAULT_MIN_PROPER_LEN, min_ratio: float = DEFAULT_MIN_PROPER_RATIO,
) -> dict:
    items = find_boundary_residue(rows, min_len=min_len, min_ratio=min_ratio)
    spans = [(index, s, e) for index, row in enumerate(rows)
             for s, e in _evt_spans(row)]
    multi = sum(1 for index, s, e in spans
                if len(_TOKEN.findall(rows[index]["text"][s:e])) > 1)
    return {
        "params": {
            "gold_sha256": file_sha256(gold_path) if gold_path else None,
            "min_proper_len": min_len,
            "min_proper_ratio": min_ratio,
        },
        "evt_spans": len(spans),
        "evt_spans_multiword": multi,
        "residue": len(items),
        "by_kind": dict(collections.Counter(i["kind"] for i in items).most_common()),
        "by_candidate": dict(collections.Counter(
            i["candidate"] for i in items if i["candidate"]).most_common()),
        "items": items,
    }


def cmd_residue(args: argparse.Namespace) -> None:
    rows = load_gold(args.gold)
    report = residue_report(rows, gold_path=args.gold)
    logger.info("EVT spans %d (multiword %d) — boundary residue %d %s",
                report["evt_spans"], report["evt_spans_multiword"],
                report["residue"], report["by_kind"])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        logger.info("wrote %s", args.out)
    # 과축소는 소급 교정 대상이 아니라 보고 항목이다. 삼킴은 다르다 — 평면 BIO 가
    # 표현할 수 없는 상태라 실측 0 이 전제이고, 깨지면 삽입이 만든 것이다.
    swallowed = report["by_kind"].get(RESIDUE_SWALLOW, 0)
    if swallowed:
        raise SystemExit(f"FAIL: {swallowed} EVT spans swallow a labeled proper noun")


# ── 적용 ───────────────────────────────────────────────────────────────

# 삽입을 막아야 하는 어긋남. 전부 "원장과 현 규칙이 같은 자리를 가리키지 않는다" 는
# 뜻이라, 세고 넘어가면 회수 건수 주장이 조용히 거짓이 된다.
APPLY_BLOCKERS = ("site_missing", "insert_span_drift", "surface_mismatch",
                  "insert_clash")


def ledger_gold_sha(ledger: Sequence[dict]) -> Tuple[str, ...]:
    """원장이 **무엇을 보고 판정했는지** 기록한 gold 지문.

    둘 이상이면 서로 다른 gold 를 본 판정이 한 원장에 섞인 것이라, 그대로 적용하면
    어느 쪽 기준으로 회수했는지가 사라진다.
    """
    return tuple(dict.fromkeys(
        str(rec["gold_sha256_before"]) for rec in ledger
        if rec.get("gold_sha256_before")
    ))


def apply_axis1_decisions(
    rows: Sequence[dict],
    ledger: Sequence[dict],
    heads: Sequence[str] = AXIS1_HEADS,
) -> Tuple[List[dict], dict]:
    """원장의 EVT 판정을 gold 에 삽입한다 — **경계는 원장과 현 규칙이 둘 다 동의해야 한다.**

    삽입 span 을 원장에서만 읽으면 경계 규칙을 나중에 고쳐도 원장이 조용히 낡고, 현
    규칙으로 다시 계산하기만 하면 사람이 승인한 경계와 다른 자리가 소리 없이 들어간다.
    그래서 자리를 다시 스캔해 **탐지 span 으로** 원장 행을 찾고, 삽입 경계가 어긋나면
    삽입하지 않고 센다 — 어긋남은 통과가 아니라 실패로 다뤄야 하기 때문이다.
    """
    stats: collections.Counter = collections.Counter()
    sites = {_site_key(s): s for s in find_axis1_sites(rows, heads)}
    # 판정은 row_id 로 되돌린다. 위치(row_index)로만 붙이면 gold 를 재생성해 행
    # 순서가 달라졌을 때 엉뚱한 문장에 span 이 조용히 박힌다.
    index_of = {str(row.get("id", i)): i for i, row in enumerate(rows)}
    by_row: Dict[int, List[Axis1Site]] = collections.defaultdict(list)
    for rec in ledger:
        verdict = str(rec.get("verdict", "")).upper()
        if verdict != EVT:
            stats["skipped_" + (verdict.lower() or "blank")] += 1
            continue
        row_index = int(rec["row_index"])
        idx = index_of.get(str(rec.get("row_id")), row_index)
        if idx != row_index:
            stats["row_index_drift"] += 1
        site = sites.get((idx, int(rec["start"]), int(rec["end"])))
        if site is None:
            stats["site_missing"] += 1
            continue
        if (site.insert_start, site.insert_end) != (
                int(rec["insert_start"]), int(rec["insert_end"])):
            stats["insert_span_drift"] += 1
            continue
        text = rows[idx]["text"]
        if text[site.insert_start:site.insert_end] != rec.get("insert_surface"):
            stats["surface_mismatch"] += 1
            continue
        by_row[idx].append(site)

    out: List[dict] = []
    for idx, row in enumerate(rows):
        ents = list(row["entities"])
        occupied: Set[int] = set()
        for ent in ents:
            occupied.update(range(int(ent["start_char"]), int(ent["end_char"])))
        for site in sorted(by_row[idx], key=lambda s: s.insert_start):
            span = set(range(site.insert_start, site.insert_end))
            if span & occupied:
                stats["insert_clash"] += 1
                continue
            ents.append({"label": EVT, "start_char": site.insert_start,
                         "end_char": site.insert_end,
                         "text": row["text"][site.insert_start:site.insert_end]})
            occupied |= span
            stats["recovered"] += 1
        ents.sort(key=lambda e: (e["start_char"], e["end_char"]))
        out.append({**row, "entities": ents})
    return out, dict(stats)


def cmd_apply(args: argparse.Namespace) -> None:
    rows = load_gold(args.gold)
    ledger = _load_jsonl(args.ledger)
    gold_before = file_sha256(args.gold)
    recorded = ledger_gold_sha(ledger)
    # 판정 당시의 gold 와 지금의 gold 가 다르면 offset 이 그대로일 보장이 없다.
    if recorded and (len(recorded) > 1 or recorded[0] != gold_before):
        raise SystemExit(
            f"FAIL: ledger judged gold {list(recorded)} but --gold is {gold_before}")
    expected = sum(1 for rec in ledger
                   if str(rec.get("verdict", "")).upper() == EVT)
    new_rows, stats = apply_axis1_decisions(rows, ledger)
    blocked = {k: stats[k] for k in APPLY_BLOCKERS if stats.get(k)}
    if blocked:
        raise SystemExit(f"FAIL: ledger disagrees with the current site scan {blocked}")
    if stats.get("recovered", 0) != expected:
        raise SystemExit(
            f"FAIL: inserted {stats.get('recovered', 0)} spans for {expected} EVT verdicts")
    checks = verify_invariants(rows, new_rows)
    if not checks["frozen_types_identical"]:
        raise SystemExit("FAIL: non-EVT types changed")
    if checks["span_text_mismatch"] or checks["entity_overlap"]:
        raise SystemExit(f"FAIL: integrity broken {checks}")
    # 적용 뒤 같은 게이트를 다시 돌린다. 회수한 자리가 gold EVT 로 안 잡히거나 삽입이
    # 다른 자리의 분류를 무너뜨리면 여기서 드러난다 — 삽입 자체가 잔여를 만든다.
    after = check_axis1_gate(new_rows, AXIS1_HEADS, ledger)
    if after["unclassified"]:
        raise SystemExit(
            f"FAIL: {len(after['unclassified'])} sites unclassified after apply")
    dump_gold(new_rows, args.out)
    payload = {
        "stats": stats,
        "checks": checks,
        # gold 는 버전 관리 밖이라 이 회수가 어떤 diff 에도 남지 않는다. 이 원장이
        # 유일한 감사 흔적이므로 입력 넷의 지문을 전부 함께 박는다.
        "gold_sha256": {"before": gold_before, "after": file_sha256(args.out)},
        "ledger_sha256": file_sha256(args.ledger),
        "heads_sha256": heads_sha256(AXIS1_HEADS),
        "canonical_sha256": file_sha256(CANONICAL_PATH),
        "gate_after": {"sites": after["sites"],
                       "status_counts": after["status_counts"],
                       "unclassified": len(after["unclassified"])},
    }
    with open(args.provenance, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    logger.info("recovered %d sites — EVT %d -> %d, wrote %s",
                stats.get("recovered", 0), checks["evt_before"],
                checks["evt_after"], args.out)


def cmd_gate(args: argparse.Namespace) -> None:
    rows = load_gold(args.gold)
    ledger = _load_jsonl(args.ledger) if args.ledger else []
    report = check_axis1_gate(rows, AXIS1_HEADS, ledger, gold_path=args.gold)
    logger.info("sites %d / unclassified %d / recovered %d",
                report["sites"], len(report["unclassified"]), report["recovered"])
    if report["covered_ratio"] > 0.5:
        logger.warning(
            "covered_by_longer_evt is %.0f%% of all sites — sample-check it",
            report["covered_ratio"] * 100)
    if report["recovered"] == 0 and ledger:
        logger.warning("no site recovered — every site was excluded or absorbed")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        logger.info("wrote %s", args.out)
    if report["unclassified"]:
        for item in report["unclassified"]:
            logger.error("unclassified: row %d %s", item["row_index"], item["surface"])
        raise SystemExit(1)


def cmd_sites(args: argparse.Namespace) -> None:
    rows = load_gold(args.gold)
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    report = sites_report(rows, heads, min_len=args.min_proper_len,
                          min_ratio=args.min_proper_ratio, gold_path=args.gold)
    logger.info("axis-1 sites: %d (population without gold EVT overlap: %d)",
                report["sites"], report["population"])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        logger.info("wrote %s", args.out)


def cmd_head_survey(args: argparse.Namespace) -> None:
    rows = load_gold(args.gold)
    report = survey_report(rows, min_count=args.min_count,
                           min_types=args.min_types)
    buried = [c for c in report["candidates"] if c["last_count"] == 0]
    logger.info(
        "head candidates: %d (never a span-final eojeol: %d)",
        len(report["candidates"]), len(buried),
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        logger.info("wrote %s", args.out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="KO EVT axis-1 compound head candidate survey")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_hs = sub.add_parser(
        "head-survey",
        help="enumerate head candidates from gold EVT spans with corpus evidence")
    p_hs.add_argument("--gold", required=True)
    p_hs.add_argument("--out", help="write survey JSON here")
    p_hs.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT,
                      help="minimum occurrences inside gold EVT spans")
    p_hs.add_argument("--min-types", type=int, default=DEFAULT_MIN_TYPES,
                      help="minimum distinct span words a suffix must end")
    p_hs.set_defaults(func=cmd_head_survey)

    p_si = sub.add_parser(
        "sites",
        help="enumerate proper-noun + event-head sites for a given head list")
    p_si.add_argument("--gold", required=True)
    p_si.add_argument("--heads", required=True,
                      help="comma-separated event heads")
    p_si.add_argument("--out", help="write site JSON here")
    p_si.add_argument("--min-proper-len", type=int,
                      default=DEFAULT_MIN_PROPER_LEN,
                      help="minimum length for a global proper-noun surface")
    p_si.add_argument("--min-proper-ratio", type=float,
                      default=DEFAULT_MIN_PROPER_RATIO,
                      help="minimum proper-label ratio for a global surface")
    p_si.set_defaults(func=cmd_sites)

    p_co = sub.add_parser(
        "codes",
        help="reason-code ledger — every head candidate must carry a canonical code")
    p_co.add_argument("--gold", required=True)
    p_co.add_argument("--out", help="write ledger JSON here")
    p_co.add_argument("--gate", action="store_true",
                      help="exit non-zero if any candidate is unassigned")
    p_co.set_defaults(func=cmd_codes)

    p_ga = sub.add_parser(
        "gate",
        help="main gate — every axis-1 site must be classified (site-level)")
    p_ga.add_argument("--gold", required=True)
    p_ga.add_argument("--ledger", help="per-site judgement ledger JSONL")
    p_ga.add_argument("--out", help="write gate report JSON here")
    p_ga.set_defaults(func=cmd_gate)

    p_rs = sub.add_parser(
        "rescore",
        help="re-score both arms against the same gold and report the "
             "recovery-free subset delta")
    p_rs.add_argument("--gold", required=True)
    p_rs.add_argument("--base-preds", nargs="+", required=True,
                      help="base-arm fold*/pred_spans.json dump")
    p_rs.add_argument("--head-preds", nargs="+", required=True,
                      help="head-arm fold*/pred_spans.json dump")
    p_rs.add_argument("--ledger", help="per-site judgement ledger JSONL")
    p_rs.add_argument("--sigma", help="pre-registered fold_sigma.json")
    p_rs.add_argument("--out", help="write the non-circular report JSON here")
    p_rs.set_defaults(func=cmd_rescore)

    p_fp = sub.add_parser(
        "fp-blindspot",
        help="diagnose which compound-named EVT false positives the head list misses")
    p_fp.add_argument("--gold", required=True)
    p_fp.add_argument("--predictions", nargs="+", required=True,
                      help="fold test_predictions.json files")
    p_fp.add_argument("--out", help="write blind-spot report JSON here")
    p_fp.set_defaults(func=cmd_fp_blindspot)

    p_re = sub.add_parser(
        "residue",
        help="count boundary-clause mismatches across every existing EVT span")
    p_re.add_argument("--gold", required=True)
    p_re.add_argument("--out", help="write residue report JSON here")
    p_re.set_defaults(func=cmd_residue)

    p_ap = sub.add_parser(
        "apply", help="insert the ledger's EVT verdicts into gold")
    p_ap.add_argument("--gold", required=True)
    p_ap.add_argument("--ledger", required=True,
                      help="per-site judgement ledger JSONL")
    p_ap.add_argument("--out", required=True, help="write the new gold here")
    p_ap.add_argument("--provenance", required=True,
                      help="write apply provenance JSON here — gold is outside "
                           "version control, so this is the only audit trail")
    p_ap.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
