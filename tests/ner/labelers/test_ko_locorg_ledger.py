"""KO LOC/ORG 원장 — canonical §2.5 판정을 gold 실측에 묶는다.

**gold 를 직접 기준으로 쓰지 않는 이유.** `data/` 는 `.gitignore` 라 커밋되지
않고 `RULER_PATHS` 에도 없다. 잠긴 §2.5 를 잠기지 않은·버전 없는 파일로 검증하면,
둘이 어긋났을 때 gold 는 손댈 수 없으니(재분류는 별건) **조정 방향이 규칙 쪽으로만
열린다** — 안 잠긴 것이 잠긴 것을 지배하는 구조다. 그래서 대조 시점의 실측을
커밋되는 원장에 박아 두고, 검사는 현 gold 를 그 원장에 대조한다. gold 가 바뀌면
지문 불일치로 **거부**되므로 바뀐 사실이 diff 에 드러난다(`evt_axis1_apply.json`
과 같은 형태).

판정 기준은 **"다른 타입으로 라벨된 적이 없다"** 이다. "모든 등장이 라벨됐다" 로
읽으면 미회수(`백두산` 1 등장 / 0 라벨)가 규칙 위반으로 잡혀, 회수 작업이 규칙
위반 목록을 늘리는 이상한 구조가 된다.
"""

import hashlib
import json
import pathlib
import warnings

import pytest

from ner.labelers.ko.ner_prompts import SINGLE_PROMPT_TEMPLATE  # noqa: F401
from tests.ner.labelers.test_ko_ner_prompts import (
    _canonical_examples,
    _canonical_marker,
    _canonical_verdicts,
)

_ROOT = pathlib.Path(__file__).resolve().parents[3]
LEDGER_PATH = _ROOT / "src" / "ner" / "labelers" / "ko" / "data" / "ko_locorg_ledger.json"


def _ledger():
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def _gold_path():
    ledger = _ledger()
    return _ROOT / ledger["gold_path"]


def _holds(verdict, labeled):
    """다른 타입으로 라벨된 적이 없는가."""
    return set(labeled) <= (set() if verdict == "non-entity" else {verdict})


def _exempt():
    return {r["surface"] for r in _ledger()["known_divergence"]}


def test_ledger_entries_satisfy_their_own_criterion():
    """원장 자체의 정합성 — gold 없이도 돈다.

    known_divergence 에 오른 표면형만 면제된다. 면제가 아니라 조용한 통과였다면
    어긋난 자리가 원장 어디에도 안 남아, 나중에 규칙을 고칠 근거가 사라진다.
    """
    ledger = _ledger()
    exempt = _exempt()
    for row in ledger["expect"] + ledger["prose_expect"] + ledger["change_log"]:
        if row["surface"] in exempt:
            continue
        assert _holds(row["verdict"], row["labeled"]), row


def test_every_ledger_surface_is_a_canonical_example():
    """표 예시 모집단이 §2.5 에서 나온다 — §2.5 를 고치면 원장도 따라와야 한다.

    모집단을 손으로 고르게 두면 불리한 표면형을 조용히 빼는 길이 열린다.
    """
    for verdict in ("ORG", "LOC", "non-entity"):
        expected = _canonical_examples(verdict)
        got = {r["surface"] for r in _ledger()["expect"] if r["verdict"] == verdict}
        assert got == expected, (verdict, expected ^ got)


def test_prose_population_comes_from_the_canonical_markers():
    """**산문 규칙도 gold 대조 안에 있다.**

    §2.5 의 단독 표면형·시설 접미 예외는 표가 아니라 산문이라, 모집단을 표 예시로만
    잡으면 그 규칙 전체가 실측 밖에 남는다 — 규칙이 gold 와 어긋나도 아무도 모른다.
    """
    named = (set(_canonical_verdicts("단독 표면형"))
             | _canonical_marker("시설 접미 예외")
             | _canonical_marker("환유 정부건물명 allowlist"))
    named -= {r["surface"] for r in _ledger()["expect"]}   # 표 예시 쪽이 이미 덮는다
    got = {r["surface"] for r in _ledger()["prose_expect"]}
    assert got == named, named ^ got


def test_known_divergences_actually_violate_the_criterion():
    """어긋남 칸에 준수 행을 넣어 두면 기록이 헐거워진다.

    준수하는 변경 기록은 `change_log` 로 간다 — 둘을 한 칸에 담으면 "어긋남 0" 인지
    "기록만 있는지" 를 구별할 수 없다.
    """
    for row in _ledger()["known_divergence"]:
        assert not _holds(row["verdict"], row["labeled"]), row
        assert row.get("note"), row


def test_live_gold_matches_the_ledger():
    """현 gold 를 원장에 대조한다. 지문이 다르면 **거부**한다."""
    gold = _gold_path()
    if not gold.exists():
        warnings.warn(
            f"KO gold 대조를 건너뛴다 — {gold} 가 없다. `data/` 는 gitignore 라"
            " 워크트리마다 없을 수 있고, 그때 이 검사는 돌지 않는다.",
            stacklevel=1)
        pytest.skip(f"gold 없음: {gold}")

    ledger = _ledger()
    digest = hashlib.sha256(gold.read_bytes()).hexdigest()
    assert digest == ledger["gold_sha256"], (
        "gold 가 원장이 기록한 판본과 다르다 — 대조를 거부한다. gold 를 바꿨다면"
        " 원장을 다시 떠서 그 변경을 커밋에 남겨라."
        f" ledger={ledger['gold_sha256'][:12]} actual={digest[:12]}")

    rows = (ledger["expect"] + ledger["prose_expect"]
            + ledger["known_divergence"] + ledger["change_log"])
    surfaces = {r["surface"] for r in rows}
    occ = dict.fromkeys(surfaces, 0)
    labeled = {s: {} for s in surfaces}
    for line in gold.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for surface in surfaces:
            occ[surface] += row["text"].count(surface)
        for entity in row.get("entities", []):
            text = entity.get("text") or row["text"][
                entity["start_char"]:entity["end_char"]]
            if text in surfaces:
                counts = labeled[text]
                counts[entity["label"]] = counts.get(entity["label"], 0) + 1

    for row in rows:
        surface = row["surface"]
        assert occ[surface] == row["occurrences"], (surface, occ[surface])
        assert labeled[surface] == row["labeled"], (surface, labeled[surface])
