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
from typing import Dict, List, Optional, Sequence, Tuple

# 조사 처리는 R2 감사의 구현을 그대로 쓴다. 어절에서 조사를 어떻게 떼는지가 두 벌이
# 되면 같은 자리를 두 모듈이 다르게 읽어, 한쪽 게이트가 통과시킨 것을 다른 쪽이
# 못 보게 된다 — 자가 둘이 되는 것과 같다.
from ner.labelers.ko.ko_evt_r2_audit import (
    CANONICAL_PATH,
    _strip_particle,
    extract_noun_candidates,
    file_sha256,
    load_gold,
)

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"\S+")
_EDGE = re.compile(r"^[^가-힣A-Za-z0-9]+|[^가-힣A-Za-z0-9]+$")

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
    """canonical §5.3 에서 **모집단을 정하는 파라미터 전부**를 읽어 온다.

    head 목록만 읽으면 부족하다 — 고유명으로 볼 타입, 전역 이력을 쓸 때의 길이·비율
    가드, 사유코드의 이름이 모두 모집단을 바꾸는데 그것들이 잠금 밖에 남으면 조용히
    좁힐 수 있다. 규칙은 기준 파일(사람 승인·반박자를 타는 곳)에 있고 그것을 세는 이
    모듈은 잠금 밖이라, 테스트가 이 결과와 모듈 상수를 대조해 어긋남을 실패로 만든다.
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
                out.append(Axis1Site(
                    row_index=index,
                    start=s_start,
                    end=s_end,
                    surface=text[s_start:s_end],
                    head=head,
                    proper=proper,
                    proper_basis=("동행라벨" if proper in row_proper else "전역이력"),
                    attachment=attachment,
                    gold_evt_overlap=_overlap_kind((s_start, s_end), evt),
                ))
                break
    return out


def heads_sha256(heads: Sequence[str]) -> str:
    """head 목록의 지문. 목록이 한 종만 달라져도 값이 바뀐다."""
    joined = "\n".join(sorted(heads))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
