"""한국어 EVT gold 의 canonical 규칙 정합성 감사 + R2 회수 파이프라인.

canonical `docs/manual/data/canonical-entity-schema.md` §5.3 의 EVT 결정표
(축1~축3 · R1~R3)에 대해 gold 를 **전수 감사**하고, 규칙이 EVT 라 정한 표면형이
gold 에서 빠진 곳(완전성 결손)을 dual-LLM 문맥 판정으로 회수한다.

세 모드가 한 계보를 이룬다:

- ``audit``  — 규칙 반영률 · 정합성 위반 · 회수 후보를 계산해 JSON 리포트로 낸다.
              재실행 가능한 기계 게이트라 회수 전후에 같은 명령으로 검증한다.
- ``judge``  — 회수 후보를 두 LLM 이 독립 판정한다. 합의분만 남기고 불일치(HOLD)는
              버린다 — 틀린 라벨을 주입하느니 불확실분을 빼는 쪽이 well-posed 다.
- ``apply``  — 합의 판정 + 기계 결정적 교정을 gold 에 반영하고 provenance 를 남긴다.

**감사와 적용이 같은 판정 함수를 쓴다** — 둘이 갈리면 감사가 통과시킨 gold 를
적용이 다르게 해석해 반영률이 조용히 안 오른다.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# ── canonical §5.3 R2 — 고유명 없이도 EVT 인 복합 행사명사 ──────────────

R2A_PROCEDURE = ("청문회", "국정조사", "정기국회", "본회의", "임시국회", "공청회")
R2B_MEETING = (
    "기자회견", "기자간담회", "간담회", "최고위원회의", "국무회의",
    "토론회", "세미나", "발표회",
)
R2C_CEREMONY = (
    "개막식", "폐막식", "취임식", "시상식", "영결식", "추모식", "기념식", "갈라쇼",
)
R2_FORMS: Dict[str, str] = {}
for _form in R2A_PROCEDURE:
    R2_FORMS[_form] = "R2a"
for _form in R2B_MEETING:
    R2_FORMS[_form] = "R2b"
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


def load_gold(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def dump_gold(rows: Sequence[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


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
                        if surface in AXIS2_PROC_EXTENSION:
                            bucket = "B2b_proc"
                        elif surface in R2_FORMS:
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
                            rule=R2_FORMS.get(surface, "축1/축2"),
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


def run_audit(rows: Sequence[dict], include_proc: bool = False) -> AuditReport:
    """gold 를 canonical EVT 규칙에 대해 전수 감사한다.

    ``include_proc`` 는 이슈 #198 범위 밖인 제도화 절차(`사전투표` 등)를 회수
    후보에 넣을지다. 기본 False — 버킷 집계에는 남되 회수·판정 대상에서 빠진다.
    """
    forms = evt_surface_counts(rows)
    report = AuditReport(evt_total=sum(forms.values()), evt_forms=len(forms))

    targets = set(forms) | set(R2_FORMS)
    found = find_unlabeled_mentions(rows, targets)
    # 축2 generic 단독은 무라벨이 정상이라 회수 후보가 아니다
    found = [c for c in found if c.surface not in AXIS2_GENERIC]
    report.bucket_totals = dict(collections.Counter(c.bucket for c in found))
    report.candidates = [
        c for c in found if include_proc or c.bucket != "B2b_proc"
    ]

    missing = collections.Counter(c.surface for c in found)
    groups = {
        # 이슈 #198 이 승격하는 R2 전용분 — 이미 등재된 축2 군집은 대조군이라 뺀다
        "R2": set(R2_FORMS) - AXIS2_LISTED,
        "axis2_listed": set(AXIS2_LISTED),
    }
    for group, members in groups.items():
        labeled = sum(forms.get(f, 0) for f in members)
        gap = sum(missing.get(f, 0) for f in members)
        report.coverage[group] = {"labeled": labeled, "missing": gap,
                                  "rate": labeled / (labeled + gap) if labeled + gap else 1.0}
    report.violations = find_violations(rows)
    report.bare_asymmetry = find_bare_asymmetry(rows)
    return report


# ── dual-LLM 판정 ──────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "당신은 한국어 개체명 인식(NER) 라벨 심판입니다. "
    "주어진 문맥에서 표시된 표면형이 EVT(행사·사건) 개체명인지 판정합니다. "
    "반드시 EVT 또는 NOT 한 단어만 출력하세요."
)

"""판정 rubric 의 R2 목록은 **상수에서 만든다** — 손으로 나열하면 canonical 표와
어긋난 채로 판정이 돌고, 그 어긋남이 gold 에 그대로 굳는다."""
_R2_RUBRIC = " / ".join(
    "·".join(group) for group in (R2A_PROCEDURE, R2B_MEETING, R2C_CEREMONY)
)

JUDGE_TEMPLATE = """규칙:
- EVT = 행사·사건. 회의·회견·의식은 고유명이 없어도 EVT다
  (""" + _R2_RUBRIC + """).
- 단, 그 행사의 장소·문서를 가리키는 파생어(기자회견장·본회의장·시상식장·기자회견문)는 EVT가 아니다(NOT).
- 다른 개체(시설·조직·팀·도로)의 이름 일부이면 EVT가 아니다(NOT). 예: "올림픽 공원"의 올림픽, "월드컵 북로"의 월드컵.
- 실제로 열린/열릴 특정 행사를 가리켜야 EVT다. 일반 개념·비유·부정문 속 막연한 언급이면 NOT.

문장: {context}
표면형: 「{surface}」

이 문맥에서 「{surface}」는 EVT인가? EVT 또는 NOT 한 단어만 출력:"""


async def _ask(client, model: str, prompt: str) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=8,
    )
    raw = (resp.choices[0].message.content or "").strip().upper()
    return "EVT" if "EVT" in raw else "NOT" if "NOT" in raw else "?"


async def _judge_all(
    cands: Sequence[Candidate], endpoints: Sequence[Tuple[str, str]], concurrency: int
) -> List[dict]:
    from openai import AsyncOpenAI

    clients = [(AsyncOpenAI(base_url=url, api_key="EMPTY"), model)
               for url, model in endpoints]
    sem = asyncio.Semaphore(concurrency)

    async def one(cand: Candidate) -> dict:
        prompt = JUDGE_TEMPLATE.format(context=cand.context, surface=cand.surface)
        async with sem:
            votes = await asyncio.gather(
                *[_ask(cli, mdl, prompt) for cli, mdl in clients],
                return_exceptions=True,
            )
        clean = [v if isinstance(v, str) else "?" for v in votes]
        agree = len(set(clean)) == 1 and clean[0] in ("EVT", "NOT")
        return {
            "row_index": cand.row_index, "row_id": cand.row_id,
            "start": cand.start, "end": cand.end, "surface": cand.surface,
            "bucket": cand.bucket, "rule": cand.rule, "context": cand.context,
            "votes": clean, "verdict": clean[0] if agree else "HOLD",
        }

    return await asyncio.gather(*[one(c) for c in cands])


# ── 적용 ───────────────────────────────────────────────────────────────


def apply_decisions(
    rows: List[dict], decisions: Sequence[dict], violations: Sequence[Violation]
) -> Tuple[List[dict], dict]:
    """합의 EVT 판정을 삽입하고 정합성 위반을 교정한다 (불변 타입 무영향)."""
    stats = collections.Counter()
    # 판정은 **row_id 로** 되돌린다. 위치(row_index)로 붙이면 gold 를 재생성해 행
    # 순서가 달라졌을 때 엉뚱한 문장에 span 이 조용히 박힌다.
    index_of = {str(row.get("id", i)): i for i, row in enumerate(rows)}
    by_row: Dict[int, List[dict]] = collections.defaultdict(list)
    for dec in decisions:
        if dec["verdict"] != "EVT":
            stats["skipped_" + dec["verdict"].lower()] += 1
            continue
        idx = index_of.get(str(dec.get("row_id")), dec["row_index"])
        if idx != dec["row_index"]:
            stats["row_index_drift"] += 1
        if rows[idx]["text"][dec["start"]:dec["end"]] != dec["surface"]:
            stats["surface_mismatch_skipped"] += 1
            continue
        by_row[idx].append(dec)

    drops: Dict[int, Set[Tuple[int, int]]] = collections.defaultdict(set)
    expands: Dict[int, Dict[Tuple[int, int], Tuple[int, int]]] = collections.defaultdict(dict)
    for vio in violations:
        if vio.fix == "drop":
            drops[vio.row_index].add((vio.start, vio.end))
        elif vio.fix == "expand" and vio.replacement:
            expands[vio.row_index][(vio.start, vio.end)] = vio.replacement

    out: List[dict] = []
    for idx, row in enumerate(rows):
        ents = []
        for ent in row["entities"]:
            key = (ent["start_char"], ent["end_char"])
            if ent["label"] == EVT and key in drops[idx]:
                stats["violation_dropped"] += 1
                continue
            if ent["label"] == EVT and key in expands[idx]:
                new_start, new_end = expands[idx][key]
                ent = dict(ent, start_char=new_start, end_char=new_end,
                           text=row["text"][new_start:new_end])
                stats["violation_expanded"] += 1
            ents.append(ent)

        occupied: Set[int] = set()
        for ent in ents:
            occupied.update(range(ent["start_char"], ent["end_char"]))
        for dec in sorted(by_row[idx], key=lambda d: d["start"]):
            span = set(range(dec["start"], dec["end"]))
            if span & occupied:
                stats["insert_clash"] += 1
                continue
            ents.append({"label": EVT, "start_char": dec["start"],
                         "end_char": dec["end"],
                         "text": row["text"][dec["start"]:dec["end"]]})
            occupied |= span
            stats["recovered"] += 1
        ents.sort(key=lambda e: (e["start_char"], e["end_char"]))
        out.append({**row, "entities": ents})
    return out, dict(stats)


def verify_invariants(before: Sequence[dict], after: Sequence[dict]) -> dict:
    """EVT 외 9 종이 byte-identical 인지, span↔원문이 어긋나지 않는지 검사."""
    assert len(before) == len(after), "row count changed"
    frozen_before, frozen_after = [], []
    mismatch = overlap = 0
    for old, new in zip(before, after):
        assert old["text"] == new["text"] and old.get("id") == new.get("id"), "row identity changed"
        frozen_before.append([e for e in old["entities"] if e["label"] != EVT])
        frozen_after.append([e for e in new["entities"] if e["label"] != EVT])
        spans = sorted((e["start_char"], e["end_char"]) for e in new["entities"])
        for (s, e), (ns, _) in zip(spans, spans[1:]):
            if ns < e:
                overlap += 1
        for ent in new["entities"]:
            if new["text"][ent["start_char"]:ent["end_char"]] != ent["text"]:
                mismatch += 1
    canon = json.dumps(frozen_before, ensure_ascii=False, sort_keys=True)
    return {
        "frozen_types_identical": canon == json.dumps(frozen_after, ensure_ascii=False, sort_keys=True),
        "span_text_mismatch": mismatch,
        "entity_overlap": overlap,
        "evt_before": sum(1 for r in before for e in r["entities"] if e["label"] == EVT),
        "evt_after": sum(1 for r in after for e in r["entities"] if e["label"] == EVT),
    }


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
        },
        "violations": {
            "total": len(rep.violations),
            "by_kind": dict(collections.Counter(v.kind for v in rep.violations)),
        },
    }


def cmd_audit(args: argparse.Namespace) -> None:
    rows = load_gold(args.gold)
    rep = run_audit(rows, include_proc=args.include_proc)
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


def cmd_judge(args: argparse.Namespace) -> None:
    rows = load_gold(args.gold)
    rep = run_audit(rows, include_proc=args.include_proc)
    cands = rep.candidates
    if args.limit:
        cands = cands[:args.limit]
    endpoints = [(args.url_a, args.model_a), (args.url_b, args.model_b)]
    logger.info("Judging %d candidates with %d models", len(cands), len(endpoints))
    results = asyncio.run(_judge_all(cands, endpoints, args.concurrency))
    with open(args.out, "w", encoding="utf-8") as fh:
        for res in results:
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
    tally = collections.Counter(r["verdict"] for r in results)
    total = len(results) or 1
    logger.info("Verdicts: %s | agreement=%.3f",
                dict(tally), (total - tally["HOLD"]) / total)


def cmd_apply(args: argparse.Namespace) -> None:
    rows = load_gold(args.gold)
    with open(args.decisions, encoding="utf-8") as fh:
        decisions = [json.loads(line) for line in fh if line.strip()]
    violations = find_violations(rows) if args.fix_violations else []
    new_rows, stats = apply_decisions(rows, decisions, violations)
    checks = verify_invariants(rows, new_rows)
    if not checks["frozen_types_identical"]:
        raise SystemExit("FAIL: non-EVT types changed")
    if checks["span_text_mismatch"] or checks["entity_overlap"]:
        raise SystemExit(f"FAIL: integrity broken {checks}")
    dump_gold(new_rows, args.out)
    if args.provenance:
        with open(args.provenance, "w", encoding="utf-8") as fh:
            json.dump({"stats": stats, "checks": checks,
                       "violations": [v.__dict__ for v in violations]},
                      fh, ensure_ascii=False, indent=2)
    print(json.dumps({"stats": stats, "checks": checks}, ensure_ascii=False, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="KO EVT canonical rule audit and R2 recovery")
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
    p_au.set_defaults(func=cmd_audit)

    p_ju = sub.add_parser("judge", help="dual-LLM judgement of recovery candidates")
    p_ju.add_argument("--gold", required=True)
    p_ju.add_argument("--out", required=True, help="per-instance provenance JSONL")
    p_ju.add_argument("--url-a", default="http://localhost:8081/v1")
    p_ju.add_argument("--model-a", default="cyankiwi/gemma-4-31B-it-AWQ-8bit")
    p_ju.add_argument("--url-b", default="http://localhost:8082/v1")
    p_ju.add_argument("--model-b", default="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
    p_ju.add_argument("--concurrency", type=int, default=16)
    p_ju.add_argument("--limit", type=int, default=0, help="pilot: first N")
    p_ju.add_argument("--include-proc", action="store_true",
                      help="also judge institutional-procedure forms (out of issue #198 scope)")
    p_ju.set_defaults(func=cmd_judge)

    p_ap = sub.add_parser("apply", help="apply consensus decisions to gold")
    p_ap.add_argument("--gold", required=True)
    p_ap.add_argument("--decisions", required=True, help="judge output JSONL")
    p_ap.add_argument("--out", required=True, help="new gold JSONL")
    p_ap.add_argument("--provenance", help="write apply provenance JSON here")
    p_ap.add_argument("--fix-violations", action="store_true",
                      help="also apply deterministic rule-violation fixes")
    p_ap.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
