"""한국어 EVT gold 의 canonical 규칙 정합성 감사.

canonical `docs/manual/data/canonical-entity-schema.md` §5.3 의 EVT 결정표
(축1~축3 · R1~R3)에 대해 gold 를 **전수 감사**한다. 규칙이 EVT 라 정한 표면형이
gold 에서 빠진 곳(완전성 결손)과 규칙을 어긴 span(정합성 위반)을 함께 센다.

두 모드 모두 재실행 가능한 기계 게이트다:

- ``audit``     — 규칙 반영률 · 정합성 위반 · 회수 후보를 계산해 JSON 리포트로 낸다.
- ``homomorph`` — R2 head 를 공유하는 표면형이 전부 분류됐는지 본다. 미분류가 남으면
                  실패한다.

판정 원장(``--ledger``)은 사람이 NOT 으로 정한 자리를 후보와 분모에서 뺄 때만 읽는다.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import pathlib
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# ── canonical §5.3 R2 — 고유명 없이도 EVT 인 회의·회견 사건 ─────────────
#
# **정본은 canonical 표이고 아래는 사본이다.** 규칙은 기준 파일(변경 시 사람
# 승인·반박자를 타는 곳)에 있는데 그것을 집행하는 이 파일은 잠금 밖이다. 그래서
# head 를 조용히 좁히면 동형 검사(§기준 3)가 아무 일 없이 통과해 버린다 —
# 자기 규칙을 자기 자로 재는 구조다. `parse_canonical_r2()` 가 표를 파싱하고
# `tests/ner/labelers/test_ko_evt_r2_audit.py` 가 이 상수와 대조해 그 경로를 막는다.

# head 로 끝나는 복합명사는 제도화·임의 개최를 불문하고 bare 여도 EVT
R2_HEADS: Tuple[str, ...] = (
    "회의", "회담", "회견", "국회", "총회", "접촉", "청문회", "공청회",
    "토론회", "간담회", "세미나", "발표회", "설명회", "포럼", "심포지엄",
)
# head 자체가 회의 형식을 지시해 복합 없이 단독으로도 EVT
R2_BARE_STANDALONE: Tuple[str, ...] = (
    "청문회", "공청회", "토론회", "간담회", "세미나", "발표회", "설명회",
    "포럼", "심포지엄",
)
# R2c 의식 — **어휘 목록이며 head 원칙이 아니다.** `식`·`쇼` 는 1 글자라 형태
# 매칭이 어휘 경계를 식별하지 못한다(`식` 동형 184 종 중 gold EVT 이력 9 종).
R2C_CEREMONY: Tuple[str, ...] = (
    "개막식", "폐막식", "취임식", "시상식", "영결식", "추모식", "기념식", "갈라쇼",
)
# R2 제외 — canonical 이 사유코드를 정하고 이 코드는 집행만 한다. `generic단독`·
# `상설조직`·`어휘경계` 는 head 목록에 애초에 없어 자동으로 빠지지만, 왜 빠졌는지가
# 표에 남아야 나중에 "좁혀서 통과시킨 것" 과 구별된다.
R2_EXCLUDE: Dict[str, Tuple[str, ...]] = {
    "generic단독": ("회의", "회담", "회견", "국회", "총회", "접촉"),
    "상설조직": ("위원회", "협의회"),
    "어휘경계": ("회", "식", "쇼", "제"),
    "타개체": ("군법회의", "시국회"),
    "비회의행위": ("신체접촉", "언론접촉"),
}
_EXCLUDED_SURFACES = frozenset(
    form for forms in R2_EXCLUDE.values() for form in forms
)

# bare 로 등장했을 때 EVT 인 어휘 — 감사 버킷이 쓰는 표면형 목록
R2_FORMS: Dict[str, str] = {}
for _form in R2_BARE_STANDALONE:
    R2_FORMS[_form] = "R2단독"
for _form in R2C_CEREMONY:
    R2_FORMS[_form] = "R2c"

# canonical §5.3 축2 에 *이미* 등재돼 있던 복합 행사명사 — R2 반영률의 대조군이다.
# R2 커버리지 분모에서 빼야 한다. 두 집합이 겹친 채로 세면 이미 잘 반영된 등재분이
# 분자를 밀어올려 baseline 이 실제보다 높게 나오고 게이트가 헐거워진다.
AXIS2_LISTED = frozenset({
    "총선", "여론조사", "국정조사", "정상회담", "정기국회", "실무회담",
})
# 제도화된 절차 — gold 에 EVT 로 이미 있으나 canonical 결정표엔 행이 없다.
# 이슈 #198 범위 밖(별도 판단)이라 회수 후보에서 기본 제외하고 버킷으로만 보고한다.
AXIS2_PROC_EXTENSION = frozenset({
    "사전투표", "영장실질심사", "역학조사", "재보선", "보궐선거",
})

# canonical §5.3 축2 — 단일 generic 사건명사 단독은 비-entity (named·수식 시에만 EVT)
AXIS2_GENERIC = frozenset({
    "테러", "조사", "선거", "투표", "대회", "경기", "사고", "전쟁", "회담", "회의",
    "행사", "시위", "사건", "사태", "축제", "재판", "공연", "파업", "훈련", "작전",
})
# canonical §5.3 R3 — 순수 경기 단계명 단독은 비-entity
R3_STAGE = frozenset({
    "결승전", "준결승", "준결승전", "결승", "포스트시즌", "플레이오프",
    "8강", "16강", "4강", "예선전", "와일드카드",
})
# R1 — EVT 표면 뒤에 이 head 가 오면 그 span 은 다른 개체(시설·조직·팀·도로)의 일부다
R1_OTHER_HEAD = (
    "공원", "경기장", "구장", "대표팀", "선수단", "조직위", "위원회", "기념관",
    "박물관", "북로", "남로", "동로", "서로", "대로", "거리", "로터리", "타워",
    "빌리지", "스타디움", "본부", "사무국", "재단", "협회", "연맹", "촌",
)
# R2 파생 — 장소·문서 파생 명사는 그 행사가 아니다 (기자회견장·본회의장·기자회견문)
DERIV_SUFFIX = frozenset("장실석자단문록비후중때날용차")

# 한국어 조사·서술격 — 긴 것부터 봐야 `으로` 가 `로` 로 잘리지 않는다
PARTICLES: Tuple[str, ...] = tuple(sorted(
    (
        "으로서", "으로써", "이라는", "이라고", "에서는", "에서도", "에게서",
        "으로는", "으로도", "이라며", "이라도", "이었", "이라", "라고", "라는",
        "라며", "에서", "에게", "에는", "에도", "에만", "까지", "부터", "처럼",
        "보다", "조차", "마저", "밖에", "대로", "같이", "이랑", "이며", "으로",
        "이나", "이란", "은", "는", "이", "가", "을", "를", "에", "와", "과",
        "의", "로", "만", "도", "랑", "뿐", "인", "며", "나", "란",
    ),
    key=len,
    reverse=True,
))
BOUNDARY_CHARS = frozenset(" \t.,·\"'()[]<>「」”’…!?;:/~-–—")
_HANGUL_OR_ALNUM = re.compile(r"[가-힣A-Za-z0-9]")
# KLUE 원본 인라인 마크업 잔재(`한<일:;LC>월드컵`·`2편:QT>은`). 이게 앞에 붙으면
# 복합어가 끊긴 것처럼 보여 `한일월드컵` 의 뒤쪽만 독립 언급으로 오인된다.
_MARKUP_RESIDUE = re.compile(r":;?[A-Z]{2}>$")

EVT = "EVT"


def r2_rule_of(surface: str) -> Optional[str]:
    """canonical R2 가 이 표면형을 bare EVT 로 정하나. 아니면 None.

    제외를 head 매칭보다 **먼저** 본다 — `군법회의` 는 `회의` head 로 끝나지만
    사법 절차라 canonical 이 `타개체` 로 제외했고, 순서가 뒤집히면 그 제외가
    형태 매칭에 덮인다.
    """
    if surface in _EXCLUDED_SURFACES:
        return None
    if surface in R2_FORMS:
        return R2_FORMS[surface]
    for head in R2_HEADS:
        if len(surface) > len(head) and surface.endswith(head):
            return "R2원칙"
    return None


# ── 형태소 경계 판정 ────────────────────────────────────────────────────


def is_standalone_mention(text: str, start: int, surface: str) -> bool:
    """``text[start:start+len(surface)]`` 가 독립된 개체 언급인지.

    부분문자열 매칭만으로 세면 `기자회견장`·`제작발표회` 같은 파생·복합어까지
    회수 후보로 잡혀 gold 를 오염시킨다. 앞이 한글·영숫자로 이어지면 복합어이고,
    뒤가 조사·문장부호가 아니면(특히 `장`·`문` 같은 파생 접미면) 다른 낱말이다.

    KLUE 마크업 잔재도 복합어로 본다 — `한<일:;LC>월드컵` 은 원문이 깨진 것이지
    `월드컵` 이 독립 언급인 게 아니다. 마크업이 낱말을 갈라놓은 탓에 앞 글자가
    `>` 로 보여 위 한글 검사를 그냥 통과해 버린다.
    """
    if start > 0 and _HANGUL_OR_ALNUM.match(text[start - 1]):
        return False
    if _MARKUP_RESIDUE.search(text[:start]):
        return False
    rest = text[start + len(surface):]
    if not rest:
        return True
    if rest[0] in BOUNDARY_CHARS:
        return True
    if rest[0] in DERIV_SUFFIX:
        return False
    return any(rest.startswith(p) for p in PARTICLES)


def followed_by_other_head(text: str, end: int) -> Optional[str]:
    """R1 — span 직후에 다른 개체의 head 가 오면 그 head 를 돌려준다."""
    tail = text[end:end + 12].lstrip()
    for head in R1_OTHER_HEAD:
        if tail.startswith(head):
            return head
    return None


# ── gold IO ────────────────────────────────────────────────────────────


def _load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_gold(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def covered_offsets(row: dict) -> Set[int]:
    out: Set[int] = set()
    for ent in row["entities"]:
        out.update(range(ent["start_char"], ent["end_char"]))
    return out


def evt_surface_counts(rows: Sequence[dict]) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for row in rows:
        for ent in row["entities"]:
            if ent["label"] == EVT:
                counts[ent["text"]] += 1
    return counts


# ── 감사 ───────────────────────────────────────────────────────────────


@dataclass
class Candidate:
    """회수 후보 한 건 — gold 에 없지만 규칙상 EVT 여야 하는 문맥 출현."""

    row_index: int
    row_id: str
    start: int
    end: int
    surface: str
    bucket: str
    rule: str
    context: str
    # span 직후가 주는 복합명사 근거 — `없음` 이면 속격 잔여일 수 있어 개별 확인 대상
    evidence: str = ""


@dataclass
class Violation:
    """정합성 위반 한 건 — gold 에 있으나 규칙에 어긋나는 EVT span."""

    row_index: int
    row_id: str
    start: int
    end: int
    surface: str
    kind: str
    detail: str
    fix: str
    replacement: Optional[Tuple[int, int]] = None


@dataclass
class AuditReport:
    evt_total: int = 0
    evt_forms: int = 0
    coverage: Dict[str, dict] = field(default_factory=dict)
    candidates: List[Candidate] = field(default_factory=list)
    violations: List[Violation] = field(default_factory=list)
    bucket_totals: Dict[str, int] = field(default_factory=dict)
    bare_asymmetry: List[dict] = field(default_factory=list)
    genitive_residue: List[dict] = field(default_factory=list)

    def coverage_rate(self, group: str) -> float:
        block = self.coverage.get(group)
        if not block:
            return 1.0
        labeled, missing = block["labeled"], block["missing"]
        return labeled / (labeled + missing) if labeled + missing else 1.0


def find_unlabeled_mentions(
    rows: Sequence[dict], surfaces: Iterable[str]
) -> List[Candidate]:
    """규칙상 EVT 인 표면형이 gold 에서 라벨되지 않은 채 나타난 지점을 모은다.

    R1(다른 개체 head 선행부)은 여기서 제외한다 — `올림픽 공원`의 `올림픽` 은
    회수 대상이 아니라 비-entity 다.
    """
    wanted = [s for s in surfaces if len(s) >= 2]
    out: List[Candidate] = []
    for idx, row in enumerate(rows):
        text = row["text"]
        covered = covered_offsets(row)
        for surface in wanted:
            if surface not in text:
                continue
            pos = text.find(surface)
            while pos != -1:
                end = pos + len(surface)
                if not (covered & set(range(pos, end))):
                    if is_standalone_mention(text, pos, surface) and not followed_by_other_head(text, end):
                        rule = r2_rule_of(surface)
                        if surface in AXIS2_PROC_EXTENSION:
                            bucket = "B2b_proc"
                        elif rule:
                            bucket = "B1_r2"
                        else:
                            bucket = "B2a_hole"
                        out.append(Candidate(
                            row_index=idx,
                            row_id=str(row.get("id", idx)),
                            start=pos,
                            end=end,
                            surface=surface,
                            bucket=bucket,
                            rule=rule or "축1/축2",
                            context=text[max(0, pos - 40):end + 40],
                        ))
                pos = text.find(surface, pos + 1)
    return out


def find_violations(rows: Sequence[dict]) -> List[Violation]:
    """gold 의 EVT span 중 canonical 규칙에 어긋나는 것을 찾는다."""
    forms = evt_surface_counts(rows)
    longer = sorted((f for f in forms if len(f) >= 3), key=len, reverse=True)
    out: List[Violation] = []
    for idx, row in enumerate(rows):
        text = row["text"]
        others = [e for e in row["entities"] if e["label"] != EVT]
        for ent in row["entities"]:
            if ent["label"] != EVT:
                continue
            surface, start, end = ent["text"], ent["start_char"], ent["end_char"]
            base = dict(row_index=idx, row_id=str(row.get("id", idx)),
                        start=start, end=end, surface=surface)
            if surface in AXIS2_GENERIC:
                out.append(Violation(**base, kind="A1_axis2_generic",
                                     detail="단일 generic 사건명사 단독", fix="drop"))
                continue
            if surface in R3_STAGE:
                out.append(Violation(**base, kind="A2_r3_stage",
                                     detail="순수 경기 단계명 단독", fix="drop"))
                continue
            head = followed_by_other_head(text, end)
            if head:
                out.append(Violation(**base, kind="A4_r1_other_entity",
                                     detail=f"직후 head={head}", fix="drop"))
                continue
            # 축3 — 같은 문장에 더 긴 EVT 표면형이 실재하면 트림된 것이다
            expand = _find_trim_target(text, surface, start, end, longer, others)
            if expand:
                new_start, new_end = expand
                out.append(Violation(**base, kind="A3_axis3_trim",
                                     detail=f"→ {text[new_start:new_end]!r}",
                                     fix="expand", replacement=(new_start, new_end)))
    return out


def _find_trim_target(
    text: str,
    surface: str,
    start: int,
    end: int,
    longer: Sequence[str],
    others: Sequence[dict],
) -> Optional[Tuple[int, int]]:
    """이 span 을 감싸는 더 긴 EVT 표면형이 같은 문장에 있으면 그 범위를 준다.

    겹침 정책(축3)상 이미 라벨된 다른 타입의 span 을 삼키는 확장은 기각한다.
    """
    for cand in longer:
        if len(cand) <= len(surface) or surface not in cand:
            continue
        pos = text.find(cand)
        while pos != -1:
            if pos <= start and end <= pos + len(cand):
                new_start, new_end = pos, pos + len(cand)
                clash = any(
                    o["start_char"] < new_end and new_start < o["end_char"]
                    for o in others
                )
                return None if clash else (new_start, new_end)
            pos = text.find(cand, pos + 1)
    return None


def find_bare_asymmetry(rows: Sequence[dict]) -> List[dict]:
    """`[수식어] [R2형]` 꼴로 이미 라벨된 EVT span 을 **보고 전용**으로 모은다.

    R2 회수는 고유명 없는 R2 형을 bare 로 삽입하는데, gold 에는 수식어를 낀 긴
    span 이 이미 있다. 둘이 같은 구성에서 다른 경계를 가르칠 수 있어 노출한다.

    게이트가 아니다 — 여기 잡히는 다수는 정당하다(`아카데미 시상식`·`세월호 청문회`
    는 고유명이 명칭 일부라 축3 최대-span). 트림 판정은 #163 이 폐기한 협소 재정의
    시도에서 두 LLM 이 16%(131/840) 불일치한 비결정 구간이라 규칙으로 세우지 않고,
    대신 **세어서 보이게만** 한다.

    공백 없는 표면형은 제외한다. `인사청문회`·`기자간담회` 는 한 낱말이라 옳은
    제외지만 `경제동향간담회` 는 붙여 쓴 일반 수식어라 놓친다 — 이 휴리스틱의
    알려진 한계이며, 그래서 미해결 일반-수식어 잔여는 3 이 아니라 **4** 다.
    """
    out: List[dict] = []
    for idx, row in enumerate(rows):
        for ent in row["entities"]:
            if ent["label"] != EVT:
                continue
            surface = ent["text"]
            for form in R2_FORMS:
                if not surface.endswith(form) or len(surface) <= len(form):
                    continue
                modifier = surface[:-len(form)].strip()
                if modifier and surface[-len(form) - 1] == " ":
                    out.append({"row_id": str(row.get("id", idx)),
                                "span": surface, "modifier": modifier, "form": form})
                break
    return out


def not_judged_sites(ledger: Sequence[dict]) -> Set[Tuple[str, int, int]]:
    """판정 원장에서 NOT 이 난 자리 — (row_id, start, end).

    표면형이 아니라 **자리** 단위다. 같은 표면형이라도 문맥에 따라 갈릴 수 있어
    표면형으로 빼면 판정하지 않은 자리까지 조용히 면제된다.

    자리가 없는 판정(타입 충돌처럼 삽입할 독립 자리가 아예 없는 것)은 건너뛴다 —
    어차피 회수 후보로 잡히지 않아 뺄 것이 없고, 억지로 매칭하면 엉뚱한 자리를
    면제하게 된다.
    """
    sites: Set[Tuple[str, int, int]] = set()
    for rec in ledger:
        if str(rec.get("verdict", "")).upper() != "NOT":
            continue
        if rec.get("start") is None or rec.get("end") is None:
            continue
        sites.add((str(rec.get("row_id")), int(rec["start"]), int(rec["end"])))
    return sites


def run_audit(
    rows: Sequence[dict],
    include_proc: bool = False,
    ledger: Sequence[dict] = (),
) -> AuditReport:
    """gold 를 canonical EVT 규칙에 대해 전수 감사한다.

    ``include_proc`` 는 이슈 #198 범위 밖인 제도화 절차(`사전투표` 등)를 회수
    후보에 넣을지다. 기본 False — 버킷 집계에는 남되 회수·판정 대상에서 빠진다.

    ``ledger`` 는 사람 판정 원장이다. NOT 이 난 자리는 규칙상 후보로 보이지만
    사람이 이미 아니라고 정한 곳이라 회수 대상도 반영률 분모도 아니다 — 안 빼면
    판정할수록 반영률이 내려가 원장이 게이트를 깎는다.
    """
    forms = evt_surface_counts(rows)
    report = AuditReport(evt_total=sum(forms.values()), evt_forms=len(forms))

    # head 파생은 gold 에 한 번도 안 나왔어도 규칙상 EVT 다 — 탐색 대상에서 빠지면
    # 후보로도 분모로도 안 잡혀 반영률이 실제보다 높게 나온다
    targets = set(forms) | set(R2_FORMS) | set(head_derived_surfaces(rows))
    found = find_unlabeled_mentions(rows, targets)
    # 축2 generic 단독은 무라벨이 정상이라 회수 후보가 아니다
    found = [c for c in found if c.surface not in AXIS2_GENERIC]
    judged_not = not_judged_sites(ledger)
    if judged_not:
        found = [c for c in found
                 if (c.row_id, c.start, c.end) not in judged_not]
    report.bucket_totals = dict(collections.Counter(c.bucket for c in found))
    report.candidates = [
        c for c in found if include_proc or c.bucket != "B2b_proc"
    ]

    missing = collections.Counter(c.surface for c in found)
    groups = {
        # R2 가 EVT 라 정한 표면형 전체 — 어휘(bare)든 head 파생이든. 이미 등재된
        # 축2 군집은 대조군이라 뺀다
        "R2": {f for f in targets if r2_rule_of(f)} - AXIS2_LISTED,
        "axis2_listed": set(AXIS2_LISTED),
    }
    for group, members in groups.items():
        labeled = sum(forms.get(f, 0) for f in members)
        gap = sum(missing.get(f, 0) for f in members)
        report.coverage[group] = {"labeled": labeled, "missing": gap,
                                  "rate": labeled / (labeled + gap) if labeled + gap else 1.0}
    report.violations = find_violations(rows)
    report.bare_asymmetry = find_bare_asymmetry(rows)
    report.genitive_residue = find_genitive_residue(rows, ledger)
    return report


# ── canonical 파싱 · head 동형 검사 ────────────────────────────────────

# 저장소 루트 기준 절대경로로 잡는다 — 상대경로면 cwd 에 따라 파서가 파일을
# 못 찾고, 그러면 canonical ↔ 상수 동기 테스트가 규칙 불일치가 아니라 실행 위치
# 때문에 깨져 신호가 흐려진다. (src/ner/labelers/ko/ 에서 네 단계 위가 루트)
CANONICAL_PATH = str(
    pathlib.Path(__file__).resolve().parents[4]
    / "docs" / "manual" / "data" / "canonical-entity-schema.md"
)
_BACKTICK = re.compile(r"`([^`]+)`")
_PAREN = re.compile(r"\([^)]*\)")
_HEAD_MARKER = re.compile(r"\*\*R2 head\*\*:\s*(.+)$")
_EXCLUDE_MARKER = re.compile(r"^R2 제외 `([^`]+)`")
_WORD_EDGE = re.compile(r"^[^가-힣A-Za-z0-9]+|[^가-힣A-Za-z0-9]+$")
_HANGUL_WORD = re.compile(r"[가-힣]+")

KO_SECTION = "### 5.3"


def canonical_section_rows(
    path: str = CANONICAL_PATH, heading: str = KO_SECTION,
) -> List[str]:
    """`heading` 절 안의 표 줄만 돌려준다. 절이 없거나 표가 없으면 예외.

    **빈 목록이 아니라 예외인 이유.** 파서가 읽는 절이 사라지거나 이름이 바뀌면
    결과가 빈 구조가 되는데, 빈 구조는 "규칙이 실제로 비어 있다" 와 구별되지
    않는다. 그러면 `canonical_rule_sha256()` 이 그 빈 규칙의 해시를 정상 값처럼
    만들어 내므로, 파싱이 깨진 채 사전등록을 다시 뜨면 빈 규칙이 정본으로 굳는다.
    규칙을 세는 쪽이 규칙을 못 읽었을 때는 조용한 0 이 아니라 시끄러운 실패여야
    한다 — 안전한 방향으로 실패하는 쪽이 과도한 차단이라 그쪽을 고른다.

    절의 끝은 다음 `## ` 또는 `### ` 이다. 같은 층의 다음 절에서 멈추지 않으면
    `### 5.4` 가 생겼을 때 그 표의 행이 KO 규칙으로 섞여 든다.
    """
    rows: List[str] = []
    found = False
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(heading):
            found = True
            continue
        if found and (line.startswith("## ") or line.startswith("### ")):
            break
        if found and line.startswith("|"):
            rows.append(line)
    if not found:
        raise ValueError(
            f"canonical section {heading!r} not found in {path}; "
            "the rule parser reads its population from that section, and an "
            "empty result would be indistinguishable from an empty rule set"
        )
    if not rows:
        raise ValueError(
            f"canonical section {heading!r} in {path} has no table rows; "
            "the rule parser reads its population from that table"
        )
    return rows


def parse_canonical_r2(path: str = CANONICAL_PATH) -> Dict[str, object]:
    """canonical §5.3 표에서 R2 head·단독 어휘·제외 목록을 읽어 온다.

    이 파서가 있는 이유는 편의가 아니라 **집행 주체**다. 규칙은 기준 파일(바꾸면
    사람 승인과 반박자를 타는 곳)에 있는데 그 규칙을 세는 이 모듈은 잠금 밖이라,
    둘이 어긋나도 아무도 모른다 — head 를 조용히 좁히면 동형 검사가 그냥 통과한다.
    테스트가 이 결과와 모듈 상수를 대조해 어긋남을 실패로 만든다.
    """
    heads: Tuple[str, ...] = ()
    bare: Tuple[str, ...] = ()
    ceremony: Tuple[str, ...] = ()
    exclude: Dict[str, Tuple[str, ...]] = {}
    for line in canonical_section_rows(path):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            continue
        surface_cell, _verdict, why = cells
        # 표면형 칸의 괄호는 한정어·예시(`(단독)`·`(영화)`)라 목록이 아니다
        surfaces = tuple(_BACKTICK.findall(_PAREN.sub("", surface_cell)))
        marker = _HEAD_MARKER.search(why)
        if marker:
            heads = tuple(_BACKTICK.findall(marker.group(1)))
        if why.startswith("R2 단독"):
            bare = surfaces
        elif why.startswith("R2c"):
            ceremony = surfaces
        else:
            excl = _EXCLUDE_MARKER.match(why)
            if excl:
                exclude[excl.group(1)] = surfaces
    return {"heads": heads, "bare_standalone": bare,
            "ceremony": ceremony, "exclude": exclude}


def _strip_particle(word: str, skip_genitive: bool = False) -> Optional[str]:
    """어절 끝 조사를 하나 떼어 돌려준다. 뗄 게 없으면 None.

    조사를 뗀 뒤 가장자리를 한 번 더 정리한다 — `원탁회의'는` 처럼 어절 안쪽에
    인용부호가 있으면 조사만 떼서는 한글 전용 검사에서 탈락한다.
    """
    for particle in PARTICLES:
        if skip_genitive and particle == "의":
            continue
        if len(word) > len(particle) + 1 and word.endswith(particle):
            return _WORD_EDGE.sub("", word[: -len(particle)])
    return None


def extract_noun_candidates(rows: Sequence[dict]) -> collections.Counter:
    """어절에서 한글 명사 후보를 센다.

    형태소 분석기를 쓰지 않는다 — 의존성이 느는 것보다, 분석기의 판정이 또 하나의
    자가 되어 canonical 밖에서 경계가 정해지는 쪽이 문제다. 대신 조사 제거 + 한글
    전용이라는 보수적 근사를 쓴다.

    **어려운 곳은 1 글자 조사 `의` 하나다.** 복합명사의 끝 음절과 형태로 구별되지
    않아, 무조건 떼면 `점검회의` 가 `점검회` 로 잘려 모집단에서 사라지고, 무조건
    두면 `사회의`·`위원회의` 가 `회의` head 로 잘못 걸린다. 그래서 두 번 훑는다 —
    먼저 `의` 를 빼고 어휘를 모은 뒤(`사회를`·`사회는` 에서 `사회` 가 모인다),
    `~의` 어절은 뗀 나머지가 그 어휘에 이미 있으면 조사로, 없으면 낱말의 일부로
    본다. 1 패스에서 `의` 를 떼면 `점검회` 가 어휘에 들어가 자기 자신을 근거로
    조사 판정을 받으므로 순서가 뒤집히면 안 된다.
    """
    genitive: List[str] = []
    counts: collections.Counter = collections.Counter()
    for row in rows:
        for raw in row["text"].split():
            word = _WORD_EDGE.sub("", raw)
            if word.endswith("의") and len(word) >= 3:
                genitive.append(word)
                continue
            noun = _strip_particle(word, skip_genitive=True)
            if noun is None:
                noun = word
            if len(noun) >= 2 and _HANGUL_WORD.fullmatch(noun):
                counts[noun] += 1

    # gold 가 라벨한 표면형도 어휘 근거다. `경찰위원회의` 처럼 어절이 늘 `~의` 로만
    # 나오면 어간이 코퍼스에 독립 등장하지 않아 낱말로 오인되는데, gold 는 이미
    # `경찰위원회` 를 ORG 로 잡아 "이건 명사다" 라고 말하고 있다.
    known = set(counts)
    known.update(
        row["text"][ent["start_char"]:ent["end_char"]]
        for row in rows for ent in row["entities"]
    )
    for word in genitive:
        stem = _WORD_EDGE.sub("", word[:-1])
        if stem in known:
            counts[stem] += 1                              # `사회의` = 조사
        elif len(word) >= 2 and _HANGUL_WORD.fullmatch(word):
            counts[word] += 1                              # `점검회의` = 낱말
    return counts


def head_derived_surfaces(rows: Sequence[dict]) -> Dict[str, str]:
    """R2 head 로 끝나는데 gold 에 EVT 로 한 번도 안 나온 표면형 → head 매핑.

    감사의 반영률 분모와 동형 게이트의 모집단이 **같은 함수**에서 나와야 한다.
    분모가 'gold 에 이미 있는 표면형' 으로만 만들어지면 규칙상 EVT 인데 한 번도
    라벨된 적 없는 형태가 통째로 빠져, 반영률이 실제보다 높게 나온다.
    """
    evt_surfaces = set(evt_surface_counts(rows))
    out: Dict[str, str] = {}
    for noun in extract_noun_candidates(rows):
        if noun in evt_surfaces:
            continue
        for head in R2_HEADS:
            if len(noun) > len(head) and noun.endswith(head):
                out.setdefault(noun, head)
                break
    return out


HOMO_EXCLUDED = "excluded"
HOMO_JUDGED_NOT = "judged_not"
HOMO_COVERED_BY_EVT = "covered_by_longer_evt"
HOMO_UNCLASSIFIED = "unclassified"


@dataclass
class HomomorphItem:
    """R2 head 를 공유하나 gold 에 EVT 이력이 없는 표면형과 그 처리 상태."""

    surface: str
    head: str
    standalone_hits: int
    clash_hits: int
    clash_types: Tuple[str, ...] = ()
    status: str = HOMO_UNCLASSIFIED
    reason: str = ""


def find_head_homomorph(rows: Sequence[dict]) -> List[HomomorphItem]:
    """R2 head 를 공유하는데 gold 에 EVT 로 한 번도 안 나온 표면형을 전수로 모은다.

    **이 모집단이 일관성 게이트의 분모다.** 원칙화가 자의성을 정말 없앴는지를
    "목록 안팎으로 갈리나" 로 물으면 답이 나오지 않는다 — 목록을 없앤 순간 안팎이
    라는 대립 자체가 사라져 "분기 0" 이 재작성의 부산물로 참이 되기 때문이다.
    그래서 gold 실측을 모집단으로 삼는다: 같은 head 를 쓰는데 한쪽만 EVT 인 자리가
    실제 코퍼스에 남아 있나. 이건 데이터가 반증할 수 있다.

    회수된 표면형은 gold EVT 이력을 갖게 되어 모집단에서 저절로 빠진다.
    """
    novel = head_derived_surfaces(rows)
    standalone: collections.Counter = collections.Counter()
    clash: collections.Counter = collections.Counter()
    clash_types: Dict[str, Set[str]] = collections.defaultdict(set)
    for row in rows:
        text = row["text"]
        covered = covered_offsets(row)
        for surface in novel:
            pos = text.find(surface)
            while pos != -1:
                end = pos + len(surface)
                if is_standalone_mention(text, pos, surface):
                    if covered & set(range(pos, end)):
                        clash[surface] += 1
                        # 겹친 gold 가 EVT 인지 다른 타입인지가 분류를 가른다 —
                        # 전자는 canonical 이 정한 축3 최대-span 이고, 후자는
                        # 규칙이 EVT 라 하는 자리를 gold 가 다른 타입으로 쓰는
                        # 정면 충돌이라 사람이 판정해야 한다
                        clash_types[surface].update(
                            e["label"] for e in row["entities"]
                            if not (e["end_char"] <= pos or e["start_char"] >= end)
                        )
                    else:
                        standalone[surface] += 1
                pos = text.find(surface, pos + 1)
    return [
        HomomorphItem(surface=s, head=h, standalone_hits=standalone[s],
                      clash_hits=clash[s],
                      clash_types=tuple(sorted(clash_types[s])))
        for s, h in sorted(novel.items())
    ]


def classify_homomorph(
    items: Sequence[HomomorphItem], ledger: Sequence[dict]
) -> List[HomomorphItem]:
    """동형 표면형을 셋 중 하나로 분류한다 — 어디에도 안 들면 미분류다.

    ① canonical 이 사유코드로 명시 제외 ② 판정 원장에 NOT 으로 기록 ③ 독립·무라벨
    자리가 없고 겹친 gold 가 **전부 EVT** — canonical R2 삽입 단서가 "이미 라벨된
    긴 span 을 되돌아가 트림하지 않는다" 로 정한 축3 최대-span 이다.

    **겹친 gold 에 EVT 아닌 타입이 있으면 ③ 이 아니다.** 규칙이 EVT 라 하는 자리를
    gold 가 ORG·PROD 로 쓰는 정면 충돌이라, 자리가 없다는 이유로 통과시키면 그
    충돌이 판정 없이 묻힌다 — 라벨러 프롬프트는 그 head 를 EVT 로 가르치므로
    체계적 오류로 되돌아온다.
    """
    not_judged = {
        str(rec.get("surface")) for rec in ledger
        if str(rec.get("verdict", "")).upper() == "NOT"
    }
    out: List[HomomorphItem] = []
    for item in items:
        non_evt = tuple(t for t in item.clash_types if t != EVT)
        if item.surface in _EXCLUDED_SURFACES:
            code = next(c for c, forms in R2_EXCLUDE.items()
                        if item.surface in forms)
            status, reason = HOMO_EXCLUDED, f"canonical 제외 `{code}`"
        elif item.surface in not_judged:
            status, reason = HOMO_JUDGED_NOT, "판정 원장 NOT"
        elif item.standalone_hits == 0 and not non_evt:
            status, reason = (HOMO_COVERED_BY_EVT,
                              f"더 긴 EVT span 안 (겹침 {item.clash_hits}) — "
                              "canonical R2 삽입 단서")
        elif item.standalone_hits == 0:
            status, reason = (HOMO_UNCLASSIFIED,
                              f"규칙은 EVT 인데 gold 는 {'/'.join(non_evt)} — "
                              "타입 충돌이라 판정 필요")
        else:
            status, reason = HOMO_UNCLASSIFIED, "회수도 제외도 판정도 되지 않음"
        out.append(HomomorphItem(item.surface, item.head, item.standalone_hits,
                                 item.clash_hits, item.clash_types,
                                 status, reason))
    return out


def check_homomorph(rows: Sequence[dict], ledger: Sequence[dict]) -> dict:
    """head 동형 표면형이 전부 분류됐나 — 미분류가 0 이어야 통과."""
    items = classify_homomorph(find_head_homomorph(rows), ledger)
    counts = collections.Counter(item.status for item in items)
    return {
        "population": len(items),
        "by_status": dict(counts),
        "unclassified": counts[HOMO_UNCLASSIFIED],
        "items": [
            {"surface": i.surface, "head": i.head,
             "standalone_hits": i.standalone_hits, "clash_hits": i.clash_hits,
             "clash_types": list(i.clash_types),
             "status": i.status, "reason": i.reason}
            for i in items
        ],
    }


_QUOTE_TRAIL = "'\"’”)]}」』>"
# 개최를 가리키는 활용형을 2 글자 이상으로 적는다. 짧은 대안을 두면 무관한 자리가
# 근거를 얻어 속격 검사 대상에서 빠진다 — 1 글자 `했` 이 "못**했**을"(`추가회담`)을,
# bare `참석` 이 "**참석**자"(`주교회의`)를 물었다. 근거를 넉넉히 주는 실수는 조용히
# 통과시키는 쪽으로 실패한다. 부분문자열 검색이라 과대매칭이 **닫힌 것은 아니고
# 좁혀진 것**이다 — 새 오탐이 보이면 여기를 먼저 본다.
_HOLD_VERB = re.compile(
    r"(열리|열려|열어|열고|열린|열었|개최|주재|소집|가진|가졌|갖고|참석한|참석했|진행)"
)


def span_evidence(text: str, end: int) -> str:
    """span 직후가 그 자리를 복합명사로 읽게 하는 근거를 준다.

    `~의` 로 끝나는 표면형은 복합명사(`전체회의`)일 수도 속격 구성(`귀족사회`+`의`)
    일 수도 있고 **형태만으로는 갈리지 않는다.** 갈라 주는 것은 직후다 — 조사가
    바로 붙으면(`전체회의를`) 그 앞이 통째로 명사이므로 복합명사이고, 공백 뒤
    명사구가 오면(`귀족사회의 완벽한 고증`) 속격 독법이 열린다.

    근거가 `없음` 인 자리는 자동 통과시키면 안 된다 — 실측하면 속격 잔여가 전부
    거기 모인다. 기계가 사람을 그 목록으로 몰아주는 것이 이 함수의 쓸모다.
    """
    after = text[end:end + 16].lstrip(_QUOTE_TRAIL)
    for particle in PARTICLES:
        if after.startswith(particle):
            return "조사직결"
    if _HOLD_VERB.search(after):
        return "개최동사"
    return "없음"


def find_genitive_residue(
    rows: Sequence[dict], ledger: Sequence[dict] = ()
) -> List[dict]:
    """이번 회수로 넣은 EVT 중 복합명사 근거가 없는 자리를 모은다.

    **이 검사가 필요한 이유는 동형 게이트가 자기 오류를 못 보기 때문이다.**
    `귀족사회의` 같은 속격 구성이 EVT 로 삽입되면 그 표면형은 gold EVT 이력을
    얻고, 모집단은 "EVT 이력 없는 표면형" 이라 그 순간 영구히 빠진다. 게이트를
    몇 번 돌려도 미분류 0 이 유지되므로, 들어간 쪽을 따로 세지 않으면 아무도 못 본다.

    범위를 **판정 원장이 넣은 자리**로 좁힌다. gold 에 원래 있던 `국무회의`·`본회의`
    까지 세면 24 건이 보고돼 정작 이번에 들어간 4 건이 묻힌다 — 이 변경이 책임질
    범위는 이번에 삽입한 것뿐이다. 원장을 안 주면 gold 전체를 본다.

    게이트가 아니라 **보고**다. `전체회의 보고자료` 처럼 근거가 없어도 정당한
    자리가 있어 자동 판정할 수 없다. 목록을 눈에 띄게 남겨, 사람이 원장의 자리별
    사유와 대조하게 하는 것이 목적이다.
    """
    scope: Optional[Set[Tuple[str, int, int]]] = None
    if ledger:
        scope = {
            (str(rec.get("row_id")), int(rec["start"]), int(rec["end"]))
            for rec in ledger
            if str(rec.get("verdict", "")).upper() == "EVT"
            and rec.get("start") is not None and rec.get("end") is not None
        }
    out: List[dict] = []
    for idx, row in enumerate(rows):
        text = row["text"]
        row_id = str(row.get("id", idx))
        for ent in row["entities"]:
            if ent["label"] != EVT or not ent["text"].endswith("의"):
                continue
            key = (row_id, ent["start_char"], ent["end_char"])
            if scope is not None and key not in scope:
                continue
            if span_evidence(text, ent["end_char"]) != "없음":
                continue
            out.append({
                "row_index": idx, "row_id": row_id,
                "start": ent["start_char"], "end": ent["end_char"],
                "surface": ent["text"], "stem": ent["text"][:-1],
                "context": text[max(0, ent["start_char"] - 30):ent["end_char"] + 30],
            })
    return out


# ── CLI ────────────────────────────────────────────────────────────────


def _report_dict(rep: AuditReport) -> dict:
    return {
        "evt_total": rep.evt_total,
        "evt_forms": rep.evt_forms,
        "coverage": rep.coverage,
        "candidates": {
            "total": len(rep.candidates),
            "by_bucket": dict(collections.Counter(c.bucket for c in rep.candidates)),
            "by_surface": dict(collections.Counter(c.surface for c in rep.candidates).most_common()),
        },
        "buckets_found": rep.bucket_totals,
        # 게이트 아님 — R2 bare 삽입과 기존 긴 span 의 경계 비대칭을 보이게만 한다
        "reported_only": {
            "bare_asymmetry": len(rep.bare_asymmetry),
            "bare_asymmetry_spans": sorted({a["span"] for a in rep.bare_asymmetry}),
            # 동형 게이트는 gold 에 **들어간** 오류를 못 본다 — 삽입되는 순간
            # EVT 이력이 생겨 모집단에서 빠지기 때문이다. 그래서 따로 센다
            "genitive_residue": len(rep.genitive_residue),
            "genitive_residue_spans": sorted(
                {g["surface"] for g in rep.genitive_residue}),
        },
        "violations": {
            "total": len(rep.violations),
            "by_kind": dict(collections.Counter(v.kind for v in rep.violations)),
        },
    }


def cmd_audit(args: argparse.Namespace) -> None:
    rows = load_gold(args.gold)
    ledger = _load_jsonl(args.ledger) if args.ledger else []
    rep = run_audit(rows, include_proc=args.include_proc, ledger=ledger)
    payload = _report_dict(rep)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    if args.candidates_out:
        with open(args.candidates_out, "w", encoding="utf-8") as fh:
            for cand in rep.candidates:
                fh.write(json.dumps(cand.__dict__, ensure_ascii=False) + "\n")
    rate = rep.coverage_rate("R2")
    logger.info("R2 coverage rate: %.4f (gate >= %.2f)", rate, args.gate_rate)
    if args.gate_rate and rate < args.gate_rate:
        raise SystemExit(
            f"FAIL: R2 coverage {rate:.4f} < gate {args.gate_rate:.2f}"
        )
    if args.gate_violations is not None and len(rep.violations) > args.gate_violations:
        raise SystemExit(
            f"FAIL: {len(rep.violations)} violations > gate {args.gate_violations}"
        )


def cmd_homomorph(args: argparse.Namespace) -> None:
    rows = load_gold(args.gold)
    ledger = _load_jsonl(args.ledger) if args.ledger else []
    result = check_homomorph(rows, ledger)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if result["unclassified"]:
        unresolved = [i["surface"] for i in result["items"]
                      if i["status"] == HOMO_UNCLASSIFIED]
        raise SystemExit(
            f"FAIL: {result['unclassified']} head-homomorph surfaces "
            f"unclassified: {', '.join(unresolved)}"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="KO EVT canonical rule audit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_au = sub.add_parser("audit", help="audit gold against canonical EVT rules")
    p_au.add_argument("--gold", required=True)
    p_au.add_argument("--out", help="write report JSON here")
    p_au.add_argument("--candidates-out", help="write recovery candidates JSONL here")
    p_au.add_argument("--gate-rate", type=float, default=0.0,
                      help="fail if R2 coverage rate is below this")
    p_au.add_argument("--gate-violations", type=int, default=None,
                      help="fail if violation count exceeds this")
    p_au.add_argument("--include-proc", action="store_true",
                      help="also count institutional-procedure forms (out of issue #198 scope)")
    p_au.add_argument("--ledger",
                      help="judgement ledger JSONL; NOT verdicts drop out of "
                           "candidates and the coverage denominator")
    p_au.set_defaults(func=cmd_audit)

    p_hm = sub.add_parser(
        "homomorph",
        help="consistency gate — every head-homomorph surface must be classified")
    p_hm.add_argument("--gold", required=True)
    p_hm.add_argument("--ledger", help="judgement ledger JSONL (NOT verdicts)")
    p_hm.add_argument("--out", help="write result JSON here")
    p_hm.set_defaults(func=cmd_homomorph)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
