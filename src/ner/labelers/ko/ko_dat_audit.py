"""한국어 `DAT` 의 span 경계 정합 + 무라벨 자리 회수.

canonical §5.3 은 명절 어근에 기간 머리가 붙은 자리를 `DAT` 로 판정하면서
**그 span 이 머리까지 덮어야 한다는 조항을 오래 두지 않았다.** 그래서 gold 에
`설 연휴` 를 한 덩어리로 단 자리와 `설` 만 달고 `연휴` 를 밖에 둔 자리가 함께
남았고, 판정이 같은데 경계만 갈리는 상태가 다음 재라벨에서 타입 차이로 번진다.
`경계 조항` 이 그 자리를 닫고 이 모듈이 그것을 집행한다.

`회수 조항` 은 다른 축이다. §1 이 `DAT` 를 "모든 날짜" 로 정하고 어디까지
라벨되는지를 원천에 맡긴 탓에, 같은 표면형이 어떤 자리에선 달리고 어떤 자리에선
안 달린 채로 굳었다. 회수는 그 갈림을 좁힌다.

**두 축의 모집단은 기계로 갈라 둔다.** 명절 표면형은 명절 원장이 이미 판정했고
그 판정이 자리 수까지 못 박고 있으므로, 완전성 회수가 같은 표면형을 건드리면
원장이 기술하는 gold 와 실제 gold 가 어긋난다. 반대 방향은 더 나쁘다 — 명절
이름에 회수 사유코드가 붙으면 §5.3 이 `EVT` 로 보내라는 자리에 `DAT` 가 들어간다.
그래서 `completeness_population()` 이 원장 표면형을 빼고, 뺐다는 사실을 스스로
단언한다.

**매니페스트는 눈가림이 아니라 목표 명세다.** gold 는 버전 관리 밖이라 편집 시점이
diff 에 안 남고, 커밋 순서를 기계가 확인할 방법도 없다. 이 파일이 사는 이유는
apply 결과를 **자리 집합으로** 대조할 대상을 주는 것이다 — 개수 일치는 옮길 자리를
옮기면서 다른 자리를 지우는 변경을 통과시킨다.

서브커맨드:

- ``survey``   — 두 축의 후보를 열거해 사람이 사유코드를 붙일 JSON 을 낸다.
- ``manifest`` — 사유코드가 붙은 판정을 받아 편집 매니페스트를 굳힌다. 표면형
                 인벤토리의 편집 후 기대 지문을 **옛 gold 에서** 계산해 넣는다 —
                 편집 후 gold 에서 재계산하면 감시 대상에서 파생된 값이 되어,
                 §5.3 이 어근 개방집합의 앵커로 삼은 지문이 항진명제가 된다.
- ``scan``     — 열거를 재실행해 매니페스트 자리 집합과 대조한다.
- ``apply``    — 매니페스트대로 gold 를 고치고 불변식과 등식을 검증한다.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import re
import tempfile
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from ner.labelers.ko.ko_evt_holiday_audit import (
    CANONICAL_PATH,
    following_span_head,
    is_candidate,
    parse_canonical_holiday,
)
from ner.labelers.ko.ko_evt_r2_audit import canonical_section_rows

# 기본 원장 — 제외 표면형을 정하는 파일이라 자리가 하나뿐이다.
_DEFAULT_LEDGER = (
    pathlib.Path(__file__).resolve().parent / "data" / "evt_holiday_prereg.json")

# ── 어절 경계·조사 ──────────────────────────────────────────────────────

# 어절을 이루는 글자. 이 밖의 문자는 경계로 본다.
WORD_CHAR = re.compile(r"[가-힣A-Za-z0-9]")

# 표면형 뒤에 붙어도 같은 낱말로 보는 꼬리. 이 목록이 모집단 생성기이자
# 완전성 측정기라, 좁히면 무라벨 자리가 줄어 회수가 공짜로 끝난다. 그래서
# 매니페스트가 오늘 값을 그대로 적어 두고 변이 시험이 그것을 baseline 으로 쓴다.
JOSA: Tuple[str, ...] = (
    "", "은", "는", "이", "가", "을", "를", "의", "에", "도", "만", "과", "와",
    "로", "으로", "에서", "에게", "한테", "께", "부터", "까지", "보다", "처럼",
    "마다", "조차", "마저", "밖에", "라도", "이라도", "이나", "나", "든", "든지",
    "야", "여", "이여", "랑", "이랑", "하고", "며", "이며", "이다", "다", "인",
    "이라", "라", "이란", "란", "요", "에는", "에도", "에선", "에서는", "으로는",
    "로는", "와의", "과의", "의는", "이고", "고", "만을", "만은", "만이", "만에",
    "까지도", "부터는", "에서도", "에게도", "으로도", "로도", "이라는", "라는",
    "서", "께서", "에다", "대로", "만큼", "같이", "치고", "로서", "으로서",
    "로써", "으로써", "이야", "야말로", "이야말로", "든가", "거나", "에요",
    "예요", "입니다", "이었", "였", "였다", "이었다", "인데", "이지", "지",
    "조차도", "뿐", "뿐만", "및",
)
_JOSA_SET = frozenset(JOSA)

VERDICTS = ("RECOVER", "DROP", "PER_SITE")


def josa_sha256(josa: Sequence[str] = JOSA) -> str:
    """조사 목록의 지문. 매니페스트가 이 값을 적어 변이 시험의 baseline 으로 쓴다."""
    payload = "\n".join(sorted(josa)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def left_boundary(text: str, start: int) -> bool:
    """표면형 앞이 어절 경계인가."""
    return start == 0 or not WORD_CHAR.match(text[start - 1])


def right_boundary(text: str, end: int) -> bool:
    """표면형 뒤가 어절 경계이거나 남은 꼬리가 조사인가."""
    if end >= len(text) or not WORD_CHAR.match(text[end]):
        return True
    cursor = end
    while cursor < len(text) and WORD_CHAR.match(text[cursor]):
        cursor += 1
    tail = text[end:cursor]
    return all("가" <= ch <= "힣" for ch in tail) and tail in _JOSA_SET


# ── canonical 동기 ─────────────────────────────────────────────────────

_INCLUDE_MARKER = re.compile(r"\*\*경계 포함 머리\*\*:\s*(.+?)(?:\s*·\s*\*\*|$)")
_EXCEPT_MARKER = re.compile(r"\*\*경계 예외\*\*:\s*(.+?)(?:\s*\||$)")
_CODE_MARKER = re.compile(r"\*\*회수 사유코드\*\*:\s*(.+?)(?:\s*·\s*\*\*|$)")
_THRESHOLD_MARKER = re.compile(r"\*\*회수 문턱\*\*:\s*(.+?)(?:\s*·\s*\*\*|$)")
_MIN_LENGTH_MARKER = re.compile(r"\*\*모집단 최소 길이\*\*:\s*(.+?)(?:\s*—|\s*\||$)")
_BACKTICKED = re.compile(r"`([^`]+)`")


def parse_canonical_dat(path: str = CANONICAL_PATH) -> Dict[str, object]:
    """canonical §5.3 의 경계 조항·회수 조항에서 집행 파라미터를 읽는다.

    **이 파서가 조항을 실행 경로에 배선한다.** 없으면 조항은 문서에만 남고
    강제 대상은 코드 상수가 정하게 되어, 기준 파일을 고쳐도 판정이 안 바뀐다.
    넷 중 하나라도 못 찾으면 예외다 — 빈 값은 "조항이 실제로 비었다" 와
    구별되지 않아 강제 대상 0 개짜리 게이트가 정상처럼 통과한다.
    """
    include: List[str] = []
    exclude: List[str] = []
    codes: List[str] = []
    threshold: List[str] = []
    min_length: List[str] = []
    for row in canonical_section_rows(path):
        for marker, sink in (
            (_INCLUDE_MARKER, include),
            (_EXCEPT_MARKER, exclude),
            (_CODE_MARKER, codes),
            (_THRESHOLD_MARKER, threshold),
            (_MIN_LENGTH_MARKER, min_length),
        ):
            found = marker.search(row)
            if found and not sink:
                sink.extend(_BACKTICKED.findall(found.group(1)))
    missing = [
        name
        for name, got in (
            ("경계 포함 머리", include),
            ("경계 예외", exclude),
            ("회수 사유코드", codes),
            ("회수 문턱", threshold),
            ("모집단 최소 길이", min_length),
        )
        if not got
    ]
    if missing:
        raise ValueError(
            "canonical KO section is missing DAT clause markers: " + ", ".join(missing))
    if sorted(codes) != sorted(VERDICTS):
        raise ValueError(
            f"canonical recovery codes {codes} disagree with module {list(VERDICTS)}")
    return {
        "include_head": tuple(include),
        "exclude_attachment": tuple(exclude),
        "codes": tuple(codes),
        "threshold": float(threshold[0]),
        "min_surface_length": int(min_length[0]),
    }


# ── 경계 축 ────────────────────────────────────────────────────────────

# canonical `경계 예외` 가 쓰는 어휘. 코드가 영문 이름을 내면 마커와 문자열이
# 안 맞아 예외 지정이 조용히 무효가 된다 — 강제 대상이 늘어도 아무도 못 본다.
ADJACENT = "인접"
COORDINATION = "등위"


def attachment_of(text: str, end: int, head: str) -> str:
    """머리가 span 바로 뒤에 붙는지(`adjacent`) 등위로 나눠 갖는지(`coordination`).

    바로 뒤 공백만 건너뛴 자리에서 머리가 시작하면 인접이고, 그 사이에 다른
    접속항이 끼면 등위다. 등위는 경계 조항의 강제 대상이 아니다 — 첫 접속항이
    머리를 포함하려면 안쪽 이름 span 을 삼켜 지워야 하고, 평면 BIO 에서 그것은
    개체 하나를 없애는 일이다.
    """
    return ADJACENT if text[end:].lstrip().startswith(head) else COORDINATION


def boundary_sites(rows: Sequence[dict], spec: Dict[str, object]) -> List[dict]:
    """경계 조항이 보는 자리를 열거한다.

    강제 대상은 canonical 이 정한다 — `경계 포함 머리` 가 `기간머리` 이므로
    기간 머리만 보고, `경계 예외` 가 `등위` 이므로 등위 부착은 강제에서 뺀다.
    마커를 뒤집으면 이 함수가 내는 자리 수가 바뀐다.
    """
    if "기간머리" not in spec["include_head"]:
        raise ValueError(
            f"unsupported boundary head class: {spec['include_head']}")
    excluded = set(spec["exclude_attachment"])
    unknown = excluded - {ADJACENT, COORDINATION}
    if unknown:
        raise ValueError(
            f"canonical names an attachment this module cannot produce: {sorted(unknown)}")
    out: List[dict] = []
    for row in rows:
        text = row["text"]
        for ent in row["entities"]:
            if ent["label"] not in ("DAT", "EVT"):
                continue
            if not is_candidate(ent["text"]):
                continue
            head = following_span_head(text, ent["end_char"])
            if not head:
                continue
            attachment = attachment_of(text, ent["end_char"], head)
            out.append({
                "row_id": row["id"],
                "start": ent["start_char"],
                "end": ent["end_char"],
                "label": ent["label"],
                "surface": ent["text"],
                "head": head,
                "attachment": attachment,
                "enforced": attachment not in excluded,
            })
    out.sort(key=lambda s: (int(s["row_id"]), s["start"]))
    return out


# ── 완전성 축 ──────────────────────────────────────────────────────────

LEDGER_SURFACE_COUNT = 163

# 한 글자 표면형은 모집단에서 뺀다 — `설`·`봄`·`월` 은 어절 경계만으로 다른 낱말의
# 조각과 안 갈린다. **이 가드도 조사 목록과 같은 성질이라** 매니페스트가 값을 못
# 박고 변이 시험이 그것을 baseline 으로 쓴다. 안 박으면 좁혀서 회수를 줄일 수 있다.
MIN_SURFACE_LENGTH = 2


def ledger_surfaces(ledger_path: str) -> Set[str]:
    """명절 원장이 판정한 표면형. 종수를 커밋된 값과 대조하고 아니면 멈춘다.

    빼고 나서 교집합이 비었는지 묻는 것은 항진명제다 — 같은 함수가 빼고 같은
    함수가 확인하므로 빼는 쪽이 좁아져도 함께 움직여 통과한다. 원장이 실제로
    몇 종을 판정했는지를 밖에서 못 박아야 그 축소가 시끄러워진다.
    """
    with open(ledger_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    surfaces = {item["surface"] for item in payload["judgements"]}
    if len(surfaces) != LEDGER_SURFACE_COUNT:
        raise ValueError(
            f"holiday ledger judges {len(surfaces)} surfaces, "
            f"expected {LEDGER_SURFACE_COUNT}")
    return surfaces


def completeness_population(rows: Sequence[dict],
                            ledger_surfaces: Set[str],
                            min_length: int = None) -> Set[str]:
    """회수 후보가 될 표면형을 gold 가 제안하게 한다.

    gold 에 `DAT` 로 달린 표면형에서 명절 원장이 이미 판정한 것을 뺀다. 빼지
    않으면 원장이 자리 수까지 못 박은 표면형을 회수해 원장과 gold 가 어긋나거나,
    명절 이름에 회수 코드가 붙어 §5.3 이 `EVT` 로 보내라는 자리에 `DAT` 가 든다.
    """
    floor = MIN_SURFACE_LENGTH if min_length is None else min_length
    labelled = {
        ent["text"]
        for row in rows for ent in row["entities"]
        if ent["label"] == "DAT" and len(ent["text"]) >= floor
    }
    return labelled - ledger_surfaces


def unlabelled_sites(rows: Sequence[dict],
                     surfaces: Set[str]) -> Dict[str, List[Tuple[str, int, int]]]:
    """표면형이 **어떤 라벨 span 에도 문자 하나 걸치지 않는** 자리를 모은다.

    걸치는 자리를 세면 삽입이 기존 span 안쪽으로 들어가 중첩이 생기고, 평면
    BIO 는 그것을 표현하지 못한다. 긴 표면형을 먼저 맞춰 짧은 것이 그 안에서
    다시 잡히지 않게 한다.
    """
    by_length: Dict[int, Set[str]] = collections.defaultdict(set)
    for surface in surfaces:
        by_length[len(surface)].add(surface)
    lengths = sorted(by_length, reverse=True)

    found: Dict[str, List[Tuple[str, int, int]]] = collections.defaultdict(list)
    for row in rows:
        text = row["text"]
        size = len(text)
        occupied = bytearray(size)
        taken = bytearray(size)
        for ent in row["entities"]:
            for i in range(ent["start_char"], min(ent["end_char"], size)):
                occupied[i] = 1
        for length in lengths:
            if length > size:
                continue
            pool = by_length[length]
            for start in range(0, size - length + 1):
                if taken[start]:
                    continue
                chunk = text[start:start + length]
                if chunk not in pool:
                    continue
                if not left_boundary(text, start) or not right_boundary(text, start + length):
                    continue
                if not any(occupied[start:start + length]):
                    found[chunk].append((row["id"], start, start + length))
                for i in range(start, start + length):
                    taken[i] = 1
    return dict(found)


def completeness_survey(rows: Sequence[dict], ledger_surfaces: Set[str],
                        threshold: float) -> List[dict]:
    """표면형마다 라벨·무라벨·반영률을 붙여 사람이 사유코드를 적을 표를 낸다."""
    population = completeness_population(rows, ledger_surfaces)
    labelled = collections.Counter(
        ent["text"]
        for row in rows for ent in row["entities"]
        if ent["label"] == "DAT"
    )
    sites = unlabelled_sites(rows, population)
    out: List[dict] = []
    for surface in sorted(population):
        gaps = sites.get(surface, [])
        if not gaps:
            continue
        kept = labelled[surface]
        rate = kept / (kept + len(gaps))
        out.append({
            "surface": surface,
            "labelled": kept,
            "unlabelled": len(gaps),
            "rate": round(rate, 4),
            "above_threshold": rate >= threshold,
            "sites": [list(s) for s in gaps],
        })
    out.sort(key=lambda r: (-r["unlabelled"], r["surface"]))
    return out


# ── 표면형 인벤토리 ────────────────────────────────────────────────────

def inventory(rows: Iterable[dict]) -> List[str]:
    """`DAT`·`EVT` 표면형 전체. 숫자 있는 것도 뺀 것 없이 담는다."""
    return sorted({
        ent["text"]
        for row in rows for ent in row["entities"]
        if ent["label"] in ("DAT", "EVT")
    })


def inventory_sha256(rows: Iterable[dict]) -> str:
    return hashlib.sha256("\n".join(inventory(rows)).encode("utf-8")).hexdigest()


def projected_inventory_sha256(rows: Sequence[dict], removed: Iterable[str],
                               added: Iterable[str]) -> str:
    """편집 **전** gold 에서 편집 후 인벤토리 지문을 미리 구한다.

    편집 후 gold 에서 재계산하면 그 지문은 감시 대상에서 파생된 값이 되어,
    §5.3 이 어근 개방집합의 앵커로 삼은 검사가 항진명제로 바뀐다. 그래서
    매니페스트가 이 값을 미리 적고 편집 뒤 실측과 대조한다.
    """
    surfaces = set(inventory(rows))
    surfaces -= set(removed)
    surfaces |= set(added)
    return hashlib.sha256("\n".join(sorted(surfaces)).encode("utf-8")).hexdigest()


def read_gold(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gold_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ── 매니페스트 ─────────────────────────────────────────────────────────

def build_manifest(rows: Sequence[dict], spec: Dict[str, object],
                   gold_path: str, boundary: Sequence[dict],
                   holiday_gaps: Sequence[dict],
                   completeness: Sequence[dict],
                   r2_queue_expected: Dict[str, int] = None,
                   below_threshold: Sequence[dict] = (),
                   population_size: Dict[str, int] = None) -> dict:
    """편집의 목표 명세를 굳힌다.

    표면형 인벤토리의 편집 후 지문을 **편집 전 gold 에서** 계산해 넣는 것이
    이 함수의 핵심이다. 편집 후 gold 에서 재계산하면 그 지문은 감시 대상에서
    나온 값이라, canonical §5.3 이 어근 개방집합의 앵커로 삼은 검사가 스스로를
    증명하는 꼴이 된다.
    """
    removed: Set[str] = set()
    added: Set[str] = set()
    surviving: collections.Counter = collections.Counter()
    for row in rows:
        for ent in row["entities"]:
            if ent["label"] in ("DAT", "EVT"):
                surviving[ent["text"]] += 1
    for site in boundary:
        if not site["enforced"]:
            continue
        surviving[site["surface"]] -= 1
        if surviving[site["surface"]] == 0:
            removed.add(site["surface"])
        added.add(site["target"])
    for gap in holiday_gaps:
        added.add(gap["text"])
    for item in completeness:
        if item["verdict"] == "DROP":
            continue
        added.add(item["surface"])

    present = set(inventory(rows))
    added = {surface for surface in added if surface not in present} | (
        added & removed)
    return {
        "gold_sha256": gold_sha256(gold_path),
        "span_heads": list(spec["include_head"]),
        "attachment_excluded": list(spec["exclude_attachment"]),
        "recovery_codes": list(spec["codes"]),
        "recovery_threshold": spec["threshold"],
        "min_surface_length": MIN_SURFACE_LENGTH,
        "completeness_population": dict(population_size or {}),
        "completeness_below_threshold": list(below_threshold),
        "josa": list(JOSA),
        "josa_sha256": josa_sha256(),
        "word_char_pattern": WORD_CHAR.pattern,
        "ledger_surface_count": LEDGER_SURFACE_COUNT,
        "boundary": {
            "enforced": [s for s in boundary if s["enforced"]],
            "excluded": [s for s in boundary if not s["enforced"]],
        },
        "holiday_gaps": list(holiday_gaps),
        "completeness": list(completeness),
        "inventory_sha256_before": inventory_sha256(rows),
        "inventory_sha256_after_expected": projected_inventory_sha256(
            rows, removed, added),
        "r2_queue_expected_delta": dict(r2_queue_expected or {}),
        "inventory_removed": sorted(removed),
        "inventory_added": sorted(added),
        "note": (
            "This file is a target spec, not a blinded pre-registration. "
            "The recovery threshold was chosen after seeing the rate "
            "distribution, and gold is untracked so commit order cannot be "
            "verified. What it buys is a set to compare the applied gold "
            "against; counting alone would pass an edit that moves one site "
            "while deleting another."
        ),
    }


def planned_edits(manifest: dict) -> Tuple[List[dict], List[dict]]:
    """매니페스트가 지시하는 확장 자리와 삽입 자리를 돌려준다."""
    expands = [
        {
            "row_id": s["row_id"], "start": s["start"],
            "end_before": s["end"], "end_after": s["start"] + len(s["target"]),
            "target": s["target"],
        }
        for s in manifest["boundary"]["enforced"]
    ]
    inserts = [
        {"row_id": g["row_id"], "start": g["start"], "end": g["end"],
         "text": g["text"], "origin": "holiday_gap"}
        for g in manifest["holiday_gaps"]
    ]
    for item in manifest["completeness"]:
        if item["verdict"] == "DROP":
            continue
        if item["verdict"] == "RECOVER":
            chosen = [tuple(s) for s in item["sites"]]
        else:
            chosen = [
                tuple(sv["site"]) for sv in item["site_verdicts"]
                if sv["verdict"] == "RECOVER"
            ]
        for row_id, start, end in chosen:
            inserts.append({
                "row_id": row_id, "start": start, "end": end,
                "text": item["surface"], "origin": "completeness",
            })
    return expands, inserts


# ── gold 편집 ──────────────────────────────────────────────────────────

def apply_manifest(rows: Sequence[dict], manifest: dict) -> List[dict]:
    """매니페스트대로 gold 를 고친다. 자리를 못 찾으면 멈춘다."""
    expands, inserts = planned_edits(manifest)
    by_row = {row["id"]: row for row in rows}
    out = [
        {**row, "entities": [dict(e) for e in row["entities"]]}
        for row in rows
    ]
    index = {row["id"]: row for row in out}

    for edit in expands:
        row = index.get(edit["row_id"])
        if row is None:
            raise ValueError(f"manifest names a missing row: {edit['row_id']}")
        target = None
        for ent in row["entities"]:
            if ent["start_char"] == edit["start"] and ent["end_char"] == edit["end_before"]:
                target = ent
                break
        if target is None:
            raise ValueError(
                f"row {edit['row_id']} has no span at "
                f"({edit['start']}, {edit['end_before']}) to expand")
        text = by_row[edit["row_id"]]["text"]
        if text[edit["start"]:edit["end_after"]] != edit["target"]:
            raise ValueError(
                f"row {edit['row_id']} text does not match the manifest target "
                f"{edit['target']!r}")
        target["end_char"] = edit["end_after"]
        target["text"] = edit["target"]

    for edit in inserts:
        row = index.get(edit["row_id"])
        if row is None:
            raise ValueError(f"manifest names a missing row: {edit['row_id']}")
        text = by_row[edit["row_id"]]["text"]
        if text[edit["start"]:edit["end"]] != edit["text"]:
            raise ValueError(
                f"row {edit['row_id']} text does not match the manifest "
                f"insert {edit['text']!r}")
        row["entities"].append({
            "label": "DAT", "start_char": edit["start"],
            "end_char": edit["end"], "text": edit["text"],
        })

    for row in out:
        row["entities"].sort(key=lambda e: (e["start_char"], e["end_char"]))
    return out


def adjacent_pairs(rows: Sequence[dict]) -> Set[Tuple]:
    """공백 하나 이하로 맞닿은 span 쌍. 새로 생기는 것만 결함이다.

    인접 자체는 gold 에 흔하다 — 서로 다른 개체가 나란히 서는 자리가 수천 개다.
    문제는 **이번 편집이 만드는 `DAT` 끼리의** 인접이다. 한 덩어리여야 할 때
    표현을 회수가 둘로 쪼개면(`8월` + `15일`) 경계를 정해야 하는데, 명절 아닌
    `DAT` 의 경계는 이 조항 밖이라 정할 자가 없다. 타입이 다른 인접은 별개
    개체가 나란히 선 것이라 쪼갬이 아니다. 그래서 같은 타입만, 편집 전후의
    차집합만 본다.
    """
    pairs: Set[Tuple] = set()
    for row in rows:
        text = row["text"]
        spans = sorted(
            (e["start_char"], e["end_char"], e["label"])
            for e in row["entities"])
        for (_, a_end, a_label), (b_start, _, b_label) in zip(spans, spans[1:]):
            if a_label != "DAT" or b_label != "DAT":
                continue
            if 0 <= b_start - a_end <= 1 and not text[a_end:b_start].strip():
                pairs.add((row["id"], a_end, b_start))
    return pairs


def check_invariants(rows: Sequence[dict]) -> List[str]:
    """평면 BIO 가 표현할 수 있는 gold 인지 전수로 본다.

    `text` 정합·offset 순서·겹침 없음 셋이다. 인접은 여기서 안 본다 — 기존
    gold 가 이미 수천 자리에서 인접하므로 전역 금지는 편집과 무관하게 실패한다.
    """
    problems: List[str] = []
    for row in rows:
        text = row["text"]
        spans = sorted(
            ((e["start_char"], e["end_char"], e) for e in row["entities"]))
        for start, end, ent in spans:
            if not 0 <= start < end <= len(text):
                problems.append(f"row {row['id']}: bad offsets {start}..{end}")
                continue
            if text[start:end] != ent["text"]:
                problems.append(
                    f"row {row['id']}: span text {ent['text']!r} != "
                    f"{text[start:end]!r}")
        for (a_start, a_end, _), (b_start, b_end, _) in zip(spans, spans[1:]):
            if b_start < a_end:
                problems.append(
                    f"row {row['id']}: spans overlap at {a_start}..{a_end} "
                    f"and {b_start}..{b_end}")
    return problems


def _span_keys(rows: Sequence[dict], label: str = None) -> Set[Tuple]:
    return {
        (row["id"], ent["start_char"], ent["end_char"], ent["label"])
        for row in rows for ent in row["entities"]
        if label is None or ent["label"] == label
    }


def verify_apply(before: Sequence[dict], after: Sequence[dict],
                 manifest: dict) -> List[str]:
    """편집이 매니페스트가 말한 것과 **집합으로** 같은지 본다."""
    problems: List[str] = []
    expands, inserts = planned_edits(manifest)

    if len(before) != len(after):
        problems.append("row count changed")
    for old, new in zip(before, after):
        if old["id"] != new["id"] or old["text"] != new["text"]:
            problems.append(f"row {old['id']}: id or text changed")

    frozen_before = {k for k in _span_keys(before) if k[3] != "DAT"}
    frozen_after = {k for k in _span_keys(after) if k[3] != "DAT"}
    if frozen_before != frozen_after:
        problems.append(
            f"non-DAT spans moved: "
            f"{sorted(frozen_before ^ frozen_after)[:5]}")

    expected = {k for k in _span_keys(before) if k[3] == "DAT"}
    for edit in expands:
        expected.discard((edit["row_id"], edit["start"], edit["end_before"], "DAT"))
        expected.add((edit["row_id"], edit["start"], edit["end_after"], "DAT"))
    for edit in inserts:
        expected.add((edit["row_id"], edit["start"], edit["end"], "DAT"))
    actual = {k for k in _span_keys(after) if k[3] == "DAT"}
    if expected != actual:
        problems.append(
            f"DAT set does not match the manifest equation: "
            f"missing={sorted(expected - actual)[:5]} "
            f"unexpected={sorted(actual - expected)[:5]}")

    total_before = sum(len(r["entities"]) for r in before)
    total_after = sum(len(r["entities"]) for r in after)
    if total_after != total_before + len(inserts):
        problems.append(
            f"entity total is {total_after}, expected "
            f"{total_before} + {len(inserts)}")

    problems.extend(check_invariants(after))

    new_pairs = adjacent_pairs(after) - adjacent_pairs(before)
    if new_pairs:
        problems.append(
            f"the edit creates {len(new_pairs)} new adjacent DAT pairs, "
            f"which split a phrase that should be one span: "
            f"{sorted(new_pairs)[:5]}")

    observed = inventory_sha256(after)
    if observed != manifest["inventory_sha256_after_expected"]:
        problems.append(
            f"inventory fingerprint is {observed}, manifest projected "
            f"{manifest['inventory_sha256_after_expected']}")
    return problems


def revert_apply(after: Sequence[dict], manifest: dict) -> List[dict]:
    """편집을 되돌린다. gold 는 버전 관리 밖이라 이것이 유일한 앵커다."""
    expands, inserts = planned_edits(manifest)
    inserted = {(e["row_id"], e["start"], e["end"]) for e in inserts}
    shrink = {
        (e["row_id"], e["start"], e["end_after"]): e["end_before"]
        for e in expands
    }
    out = []
    for row in after:
        kept = []
        for ent in row["entities"]:
            key = (row["id"], ent["start_char"], ent["end_char"])
            if ent["label"] == "DAT" and key in inserted:
                continue
            ent = dict(ent)
            if ent["label"] == "DAT" and key in shrink:
                ent["end_char"] = shrink[key]
                ent["text"] = row["text"][ent["start_char"]:ent["end_char"]]
            kept.append(ent)
        out.append({**row, "entities": kept})
    return out


# ── 세대 원장 ──────────────────────────────────────────────────────────

def transform_ledger(old: dict, manifest: dict, new_gold_path: str,
                     new_rows: Sequence[dict]) -> dict:
    """옛 명절 원장 + 매니페스트 → 이번 세대 원장. 손으로 넣는 값은 없다.

    옛 원장을 고치지 않고 새 세대를 내는 이유는 그것이 **그때 무엇이 초록이었는지**
    의 기록이기 때문이다. 그러나 경계를 고치면 옛 원장이 기술하던 gold 가 더는
    없으므로, 그 원장으로 게이트를 돌려 통과시키려 하면 게이트 자신을 느슨하게
    하는 길밖에 남지 않는다 — 그 파일은 기준 파일 목록 밖이라 아무도 못 본다.
    그래서 새 세대를 내고, 옛 원장이 새 gold 에서 **실패한다는 것**을 따로 못 박는다.

    `sweep` 지문은 재계산하지 않고 매니페스트가 편집 전에 계산해 둔 값을 옮긴다.
    편집 후 gold 에서 다시 구하면 감시 대상에서 나온 값이라 앵커가 항진명제가 된다.
    """
    expands, _ = planned_edits(manifest)
    moved = {
        (e["row_id"], e["start"], e["end_before"]):
            (e["row_id"], e["start"], e["end_after"], e["target"])
        for e in expands
    }
    gap_sites = [
        (g["row_id"], g["start"], g["end"], g["text"])
        for g in manifest["holiday_gaps"]
    ]

    # 라벨은 verdict 에서 추정하지 않고 편집된 gold 에서 읽는다. 원장의 `labels`
    # 는 "이 표면형이 gold 에서 실제로 무엇으로 달려 있나" 이고, verdict 는 "무엇
    # 이어야 하나" 다. 예외 자리를 둔 표면형에서 그 둘이 갈리므로 추정하면 어긋난다.
    label_at = {
        (row["id"], ent["start_char"], ent["end_char"]): ent["label"]
        for row in new_rows for ent in row["entities"]
    }

    by_surface: Dict[str, dict] = {}
    for item in old["judgements"]:
        clone = json.loads(json.dumps(item))
        clone["sites"] = []
        # 확장으로 경계가 맞아진 자리의 예외는 사라지고, 등위처럼 강제 대상이
        # 아닌 자리의 예외는 그대로 남는다.
        kept = [
            exc for exc in (clone.get("site_exceptions") or [])
            if (exc["site"][0], exc["site"][1], exc["site"][2]) not in moved
        ]
        if kept:
            clone["site_exceptions"] = kept
        else:
            clone.pop("site_exceptions", None)
        by_surface[item["surface"]] = clone

    arrivals: Dict[str, List[Tuple[str, int, int]]] = collections.defaultdict(list)
    for item in old["judgements"]:
        for site in item["sites"]:
            key = (site[0], site[1], site[2])
            if key in moved:
                row_id, start, end, target = moved[key]
                arrivals[target].append((row_id, start, end))
            else:
                by_surface[item["surface"]]["sites"].append(list(site))
    for row_id, start, end, text in gap_sites:
        arrivals[text].append((row_id, start, end))

    for surface, sites in arrivals.items():
        entry = by_surface.get(surface)
        if entry is None:
            entry = {
                "surface": surface,
                "verdict": "DAT",
                "reason": "span_head",
                "count": 0,
                "labels": {},
                "sites": [],
            }
            by_surface[surface] = entry
        for site in sites:
            entry["sites"].append(list(site))

    judgements = []
    for surface in sorted(by_surface):
        entry = by_surface[surface]
        entry["sites"].sort(key=lambda s: (int(s[0]), s[1]))
        if not entry["sites"]:
            continue
        entry["count"] = len(entry["sites"])
        counts: collections.Counter = collections.Counter()
        for site in entry["sites"]:
            label = label_at.get((site[0], site[1], site[2]))
            if label is None:
                raise ValueError(
                    f"{surface!r} keeps a site the edited gold does not label: {site}")
            counts[label] += 1
        entry["labels"] = dict(counts)
        judgements.append(entry)

    new = json.loads(json.dumps(old))
    new["gold_sha256"] = gold_sha256(new_gold_path)
    new["judgements"] = judgements
    new["candidates"] = {
        "surfaces": len(judgements),
        "spans": sum(item["count"] for item in judgements),
    }
    sweep = dict(new.get("sweep") or {})
    sweep["inventory_sha256"] = manifest["inventory_sha256_after_expected"]
    new["sweep"] = sweep
    new["generation"] = {
        "edited_from": manifest["gold_sha256"],
        "previous_ledger_described": old["gold_sha256"],
        "note": (
            "Produced by transform_ledger() from the previous generation plus "
            "the edit manifest. Nothing here is entered by hand; re-running the "
            "transform on the committed inputs must reproduce this file byte for "
            "byte. The two fingerprints differ on purpose: the manifest edited "
            "the gold this ledger now describes, while the previous ledger "
            "described the generation before that one. Applying the manifest to "
            "`previous_ledger_described` would not produce this gold."
        ),
    }
    return new


def write_gold(rows: Sequence[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── CLI ────────────────────────────────────────────────────────────────

def _load_boundary(rows, spec, ledger_path):
    """경계 자리를 열거하고 옛 원장의 예외 note 에서 목표 문자열을 옮긴다.

    목표를 새로 짓지 않는 것이 중요하다 — 그 값들은 앞 세대가 숫자를 보기 전에
    고정했고 코퍼스 실재가 이미 단언돼 있다. 여기서 다시 지으면 저자가 고른 값이
    된다. note 의 마지막 백틱이 목표이며 형태가 균일하다.
    """
    backticked = re.compile(r"`([^`]+)`")
    with open(ledger_path, encoding="utf-8") as handle:
        ledger = json.load(handle)
    targets = {}
    for item in ledger["judgements"]:
        for exc in (item.get("site_exceptions") or []):
            if "head" in exc:
                site = exc["site"]
                targets[(site[0], site[1], site[2])] = backticked.findall(
                    exc["note"])[-1]
    sites = boundary_sites(rows, spec)
    for site in sites:
        key = (site["row_id"], site["start"], site["end"])
        if site["enforced"]:
            site["target"] = targets[key]
        else:
            site["reason"] = (
                "평면 BIO 는 중첩 span 을 표현 못 해 안쪽 이름 span 을 지워야 한다")
    return sites


def cmd_survey(args: argparse.Namespace) -> None:
    rows = read_gold(args.gold)
    spec = parse_canonical_dat()
    surfaces = ledger_surfaces(args.ledger)
    survey = completeness_survey(rows, surfaces, spec["threshold"])
    payload = {
        "boundary": _load_boundary(rows, spec, args.ledger),
        "completeness": [s for s in survey if s["above_threshold"]],
        "completeness_below_threshold": [
            s for s in survey if not s["above_threshold"]],
        "population": {
            "surfaces": len(completeness_population(rows, surfaces)),
            "with_gaps": len(survey),
        },
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    print(f"INFO survey wrote {args.out}")


def cmd_manifest(args: argparse.Namespace) -> None:
    """사유코드가 붙은 판정을 받아 편집의 목표 명세를 굳힌다."""
    rows = read_gold(args.gold)
    spec = parse_canonical_dat()
    surfaces = ledger_surfaces(args.ledger)
    with open(args.verdicts, encoding="utf-8") as handle:
        verdicts = json.load(handle)
    survey = completeness_survey(rows, surfaces, spec["threshold"])
    manifest = build_manifest(
        rows, spec, args.gold,
        _load_boundary(rows, spec, args.ledger),
        verdicts["holiday_gaps"],
        verdicts["completeness"],
        r2_queue_expected=verdicts.get("r2_queue_expected_delta"),
        below_threshold=[s for s in survey if not s["above_threshold"]],
        population_size={
            "surfaces": len(completeness_population(rows, surfaces)),
            "with_gaps": len(survey),
        },
    )
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=1)
    print(f"INFO manifest wrote {args.out}")


def cmd_scan(args: argparse.Namespace) -> None:
    """열거를 다시 돌려 매니페스트 자리 집합과 대조한다. 개수 일치는 통과가 아니다."""
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    rows = read_gold(args.gold)
    before = revert_apply(rows, manifest) if args.applied else rows
    spec = parse_canonical_dat()
    problems: List[str] = []

    # **되감기가 매니페스트를 입력으로 쓴다.** 그래서 매니페스트에서 자리를 지우면
    # 그 자리가 안 되감겨 gold 에 라벨된 채 남고, 무라벨 열거에서도 함께 빠져
    # 대조가 자기일관이 된다 — 지운 자리를 아무도 못 본다. 되감은 결과가 매니페스트가
    # 말한 세대인지 먼저 확인해 그 순환을 끊는다.
    #
    # **이 검사는 한 세대짜리다.** 다음 gold 편집이 얹히면 `--applied` 되감기는 그
    # 편집까지 되돌리지 못해 여기서 영구히 운다. 그때는 이 매니페스트를 그 세대의
    # 되감기 결과에 대고 `--pristine` 으로 돌린다.
    if args.applied:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reverted.jsonl")
            write_gold(before, path)
            reverted = gold_sha256(path)
        if reverted != manifest["gold_sha256"]:
            problems.append(
                f"reverting with this manifest yields {reverted[:16]}…, but it "
                f"records {manifest['gold_sha256'][:16]}… — the manifest and the "
                f"gold disagree about what this edit was")

    if manifest["josa_sha256"] != josa_sha256():
        problems.append("josa list drifted from the manifest")
    if manifest["word_char_pattern"] != WORD_CHAR.pattern:
        problems.append("word boundary rule drifted from the manifest")
    if manifest.get("min_surface_length") != MIN_SURFACE_LENGTH:
        problems.append("minimum surface length drifted from the manifest")

    fresh = _load_boundary(before, spec, args.ledger)
    fresh_keys = {
        (s["row_id"], s["start"], s["end"], s["enforced"]) for s in fresh}

    # **목표 문자열도 손으로 적을 수 있는 자리다.** `apply` 가 gold 본문과 맞춰 보므로
    # 아무 문자열이나 들어가진 않지만, 같은 자리의 *다른* 유효한 확장(`추석 연휴를`)은
    # 통과한다. 목표는 앞 세대 원장의 예외 note 가 이미 지정한 값이라, 그 재도출과
    # 대조하면 저자가 고른 값이 들어갈 문이 닫힌다.
    derived = {
        (s["row_id"], s["start"], s["end"]): s["target"]
        for s in fresh if s["enforced"]
    }
    for site in manifest["boundary"]["enforced"]:
        key = (site["row_id"], site["start"], site["end"])
        if key in derived and site.get("target") != derived[key]:
            problems.append(
                f"boundary target {site.get('target')!r} at {key} differs from "
                f"the note the previous ledger recorded ({derived[key]!r})")
    pinned = {
        (s["row_id"], s["start"], s["end"], True)
        for s in manifest["boundary"]["enforced"]
    } | {
        (s["row_id"], s["start"], s["end"], False)
        for s in manifest["boundary"]["excluded"]
    }
    if fresh_keys != pinned:
        problems.append(
            f"boundary enumeration differs from the manifest: "
            f"{sorted(fresh_keys ^ pinned)[:5]}")

    surfaces = ledger_surfaces(args.ledger)
    population = completeness_population(before, surfaces)
    recorded = manifest.get("completeness_population") or {}
    if recorded.get("surfaces") != len(population):
        problems.append(
            f"completeness population is {len(population)}, manifest "
            f"recorded {recorded.get('surfaces')}")
    survey = completeness_survey(before, surfaces, manifest["recovery_threshold"])

    # **표면형이 아니라 자리로 맞춘다.** 표면형 집합만 보면 한 표면형의 자리를
    # 하나 지워도 통과하는데, 삽입의 대부분이 여기서 온다 — 편집의 거의 전부가
    # 대조 밖에 남는다. 경계 축은 처음부터 좌표로 맞췄고 이 축만 안 맞췄다.
    def _sites(items):
        return {
            (item["surface"], site[0], site[1], site[2])
            for item in items for site in item["sites"]
        }

    fresh_above = [s for s in survey if s["above_threshold"]]
    fresh_below = [s for s in survey if not s["above_threshold"]]
    if _sites(fresh_above) != _sites(manifest["completeness"]):
        diff = _sites(fresh_above) ^ _sites(manifest["completeness"])
        problems.append(
            f"completeness sites differ from the manifest: {sorted(diff)[:5]}")
    pinned_below = manifest.get("completeness_below_threshold", [])
    if _sites(fresh_below) != _sites(pinned_below):
        diff = _sites(fresh_below) ^ _sites(pinned_below)
        problems.append(
            f"below-threshold sites differ from the manifest: {sorted(diff)[:5]}")

    # 회수하기로 한 자리가 실제 편집과 같은지도 본다 — 판정 파일은 커밋 밖이라
    # 매니페스트가 그 유일한 기록이다.
    _, inserts = planned_edits(manifest)
    recovered = {
        (edit["text"], edit["row_id"], edit["start"], edit["end"])
        for edit in inserts if edit["origin"] == "completeness"
    }
    judged = set()
    for item in manifest["completeness"]:
        if item["verdict"] == "RECOVER":
            judged |= {(item["surface"], *site) for site in item["sites"]}
        elif item["verdict"] == "PER_SITE":
            judged |= {
                (item["surface"], *sv["site"])
                for sv in item["site_verdicts"] if sv["verdict"] == "RECOVER"
            }
    if recovered != judged:
        problems.append(
            f"planned recovery does not match the verdicts: "
            f"{sorted(recovered ^ judged)[:5]}")

    # **회수 자리는 기계가 방금 낸 열거 안에 있어야 한다.** 위 두 검사는 매니페스트의
    # 서로 다른 배열끼리 맞추는 것이라, `site_verdicts` 에 열거기가 제안한 적 없는
    # 좌표를 적으면 셋 다 조용하다 — 되감기조차 같은 위조 매니페스트로 정의되니
    # 일관되게 통과한다. 자유 좌표가 들어갈 수 있는 유일한 자리가 여기이므로,
    # 매니페스트가 아니라 **gold 에서 나온 집합**에 부분집합으로 건다.
    stray = recovered - _sites(fresh_above)
    if stray:
        problems.append(
            f"recovery names sites the enumerator never proposed: "
            f"{sorted(stray)[:5]}")

    # **명절 무라벨 자리도 같은 문이다.** 이 배열은 커밋 밖 판정 파일에서 실려 오고
    # `planned_edits` 가 그대로 삽입으로 바꾼다. 표면형 목록으로는 못 찾는 자리라
    # 원문을 어근으로 훑는 열거기가 따로 있고, 여기에 그 결과를 건다.
    fresh_gaps = {
        (site["row_id"], site["start"], site["end"])
        for site in holiday_gap_sites(before)
    }
    pinned_gaps = {
        (gap["row_id"], gap["start"], gap["end"])
        for gap in manifest["holiday_gaps"]
    }
    if pinned_gaps != fresh_gaps:
        problems.append(
            f"holiday gaps differ from the enumerator: "
            f"{sorted(pinned_gaps ^ fresh_gaps)[:5]}")
    for gap in manifest["holiday_gaps"]:
        row = next((r for r in before if r["id"] == gap["row_id"]), None)
        if row is None:
            problems.append(f"holiday gap names a missing row: {gap['row_id']}")
            continue
        text = row["text"]
        if not left_boundary(text, gap["start"]) or not right_boundary(text, gap["end"]):
            problems.append(f"holiday gap is not on word boundaries: {gap}")
        if not any(gap["text"].startswith(surface) for surface in surfaces):
            problems.append(
                f"holiday gap does not start with a judged holiday surface: {gap}")

    unjudged = [
        item["surface"] for item in manifest["completeness"]
        if item["verdict"] == "PER_SITE"
        and len(item.get("site_verdicts") or []) != len(item["sites"])
    ]
    if unjudged:
        problems.append(f"PER_SITE surfaces with unjudged sites: {unjudged}")

    for line in problems:
        print(f"FAIL {line}")
    if problems:
        raise SystemExit(1)
    print(
        f"INFO scan passed: boundary {len(fresh)} sites "
        f"({sum(1 for s in fresh if s['enforced'])} enforced) · "
        f"completeness {len(fresh_above)} above ({len(recovered)} recovered) / "
        f"{len(fresh_below)} below threshold · "
        f"population {len(population)} surfaces")


def cmd_apply(args: argparse.Namespace) -> None:
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    rows = read_gold(args.gold)
    before = revert_apply(rows, manifest) if args.applied else rows
    applied = apply_manifest(before, manifest)
    problems = verify_apply(before, applied, manifest)
    for line in problems:
        print(f"FAIL {line}")
    if problems:
        raise SystemExit(1)
    if args.write:
        write_gold(applied, args.gold)
        print(f"INFO applied to {args.gold} ({gold_sha256(args.gold)[:16]}…)")
    else:
        print("INFO verify only, gold untouched")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    survey = sub.add_parser("survey", help="enumerate candidates for judgement")
    survey.add_argument("--gold", required=True)
    survey.add_argument("--ledger", required=True)
    survey.add_argument("--out", required=True)
    survey.set_defaults(func=cmd_survey)

    manifest_cmd = sub.add_parser(
        "manifest", help="freeze the edit target spec from judged verdicts")
    manifest_cmd.add_argument("--gold", required=True)
    manifest_cmd.add_argument("--ledger", required=True)
    manifest_cmd.add_argument("--verdicts", required=True)
    manifest_cmd.add_argument("--out", required=True)
    manifest_cmd.set_defaults(func=cmd_manifest)

    scan = sub.add_parser("scan", help="re-enumerate and compare to the manifest")
    scan.add_argument("--gold", required=True)
    scan.add_argument("--manifest", required=True)
    scan.add_argument("--ledger", default=str(_DEFAULT_LEDGER),
                      help="holiday ledger that fixes the excluded surfaces")
    scan.add_argument("--applied", action="store_true", default=True,
                      help="gold already carries the edit (default)")
    scan.add_argument("--pristine", dest="applied", action="store_false",
                      help="gold predates the edit")
    scan.set_defaults(func=cmd_scan)

    apply_cmd = sub.add_parser("apply", help="edit gold and verify the result")
    apply_cmd.add_argument("--gold", required=True)
    apply_cmd.add_argument("--manifest", required=True)
    apply_cmd.add_argument("--applied", action="store_true", default=True)
    apply_cmd.add_argument("--pristine", dest="applied", action="store_false")
    mode = apply_cmd.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true",
                      help="write the edited gold back")
    mode.add_argument("--verify", action="store_true",
                      help="verify without writing (the default)")
    apply_cmd.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    args.func(args)


def holiday_gap_sites(rows: Sequence[dict]) -> List[dict]:
    """gold 가 아무 라벨도 안 단 명절+기간머리 자리를 원문에서 찾는다.

    **표면형 목록으로는 이 자리를 못 찾는다.** 회수 후보를 gold 의 라벨된 표면형에서
    뽑으면 라벨된 적 없는 자리는 애초에 모집단 밖이다. 그래서 어근을 원문에 직접
    걸고, 그 뒤에 기간 머리가 붙는지를 같은 함수(`following_span_head()`)로 본다.

    왼쪽 어절 경계가 오탐을 거른다 — `소설 때문에`·`맞설때`·`들어설 때마다` 는 모두
    `설` 앞이 한글이라 걸러진다. 이 조건이 없으면 어근이 다른 낱말의 조각으로
    잡혀 자리 수가 부풀고, 그 자리는 사람이 판정할 목록에 섞여 든다.
    """
    spec = parse_canonical_holiday()
    roots = sorted(spec["roots"], key=len, reverse=True)
    out: List[dict] = []
    for row in rows:
        text = row["text"]
        occupied = bytearray(len(text))
        for ent in row["entities"]:
            for i in range(ent["start_char"], min(ent["end_char"], len(text))):
                occupied[i] = 1
        seen: Set[int] = set()
        for root in roots:
            start = text.find(root)
            while start != -1:
                end = start + len(root)
                if start not in seen and left_boundary(text, start):
                    head = following_span_head(text, end)
                    if head:
                        tail = text.index(head, end) + len(head)
                        if not any(occupied[start:tail]):
                            out.append({
                                "row_id": row["id"], "start": start, "end": tail,
                                "text": text[start:tail], "root": root, "head": head,
                            })
                            seen.update(range(start, tail))
                start = text.find(root, start + 1)
    out.sort(key=lambda s: (int(s["row_id"]), s["start"]))
    return out


if __name__ == "__main__":
    main()
