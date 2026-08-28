"""KO 명절 EVT 판정 하네스 테스트.

**게이트가 무엇을 막는지가 핵심이다.** 이 이슈의 기존 두 게이트(축1·R2)는
`DAT`↔`EVT` 이동에 구조적으로 눈이 없어, 명절과 무관한 `DAT` 를 아무거나
`EVT` 로 바꿔도 똑같이 통과한다. 그래서 통과가 증거가 되려면 **잘못된 재라벨을
실제로 막는지**를 고정해야 하고, 아래 테스트 대부분이 그 반증 시나리오다.
"""

import collections
import hashlib
import json
import pathlib
import re
import tempfile
import warnings

import pytest

from ner.labelers.ko.ko_evt_holiday_audit import (
    CATEGORY_HEADS,
    DAY_HEADS,
    HOLIDAY_ROOTS,
    PERIOD_NAMES,
    SPAN_HEADS,
    apply_ledger,
    following_head,
    following_span_head,
    load_ledger,
    planned_moves,
    check_gate,
    gate_failures,
    holiday_candidates,
    inventory_sha256,
    is_candidate,
    ledger_index,
    parse_canonical_holiday,
    surface_inventory,
)
from ner.labelers.ko.ko_evt_r2_audit import dump_gold, load_gold

# ── 합성 gold — 판정 네 갈래를 한 줄씩 담는다 ──────────────────────────


def _row(rid, text, ents):
    return {"id": rid, "text": text,
            "entities": [{"label": lab, "start_char": s, "end_char": s + len(t),
                          "text": t} for lab, s, t in ents]}


def _gold():
    """네 판정 + 무관 표면형이 한 벌씩 든 최소 gold."""
    return [
        _row("1", "크리스마스에 눈이 왔다", [("DAT", 0, "크리스마스")]),
        _row("2", "설날 아침이다", [("DAT", 0, "설날")]),
        _row("3", "크리스마스 시즌 매출", [("DAT", 0, "크리스마스 시즌")]),
        _row("4", "공휴일 근무 수당", [("DAT", 0, "공휴일")]),
        _row("5", "강릉 단오제 개막", [("EVT", 0, "강릉 단오제")]),
        _row("6", "이날 오후 발표했다", [("DAT", 0, "이날")]),
        _row("7", "설명회 자리에서", [("EVT", 0, "설명회")]),
    ]


def _judgements():
    return [
        {"surface": "크리스마스", "verdict": "EVT", "reason": "name",
         "count": 1, "labels": {"DAT": 1}, "sites": [["1", 0, 5]]},
        {"surface": "설날", "verdict": "EVT", "reason": "name",
         "count": 1, "labels": {"DAT": 1}, "sites": [["2", 0, 2]]},
        {"surface": "크리스마스 시즌", "verdict": "DAT", "reason": "span_head",
         "count": 1, "labels": {"DAT": 1}, "sites": [["3", 0, 8]]},
        {"surface": "공휴일", "verdict": "DAT", "reason": "category",
         "count": 1, "labels": {"DAT": 1}, "sites": [["4", 0, 3]]},
        {"surface": "강릉 단오제", "verdict": "EVT_KEEP", "reason": "event_name",
         "count": 1, "labels": {"EVT": 1}, "sites": [["5", 0, 6]]},
        {"surface": "이날", "verdict": "NOT_HOLIDAY", "reason": "unrelated",
         "count": 1, "labels": {"DAT": 1}, "sites": [["6", 0, 2]]},
        {"surface": "설명회", "verdict": "NOT_HOLIDAY", "reason": "homomorph",
         "count": 1, "labels": {"EVT": 1}, "sites": [["7", 0, 3]]},
    ]


def _ledger(judgements=None, rows=None):
    """판정 목록에 지문·스윕 앵커를 붙여 원장 한 벌로 만든다."""
    rows = _gold() if rows is None else rows
    return {
        "gold_sha256": "0" * 64,
        "sweep": {"inventory_sha256": inventory_sha256(rows)},
        "reviewed_heads": [],
        "judgements": _judgements() if judgements is None else judgements,
    }


def _relabel(rows, surfaces):
    out = json.loads(json.dumps(rows))
    for row in out:
        for ent in row["entities"]:
            if ent["label"] == "DAT" and ent["text"] in surfaces:
                ent["label"] = "EVT"
    return out


# ── 후보 생성 — 두 그물이 각자 무엇을 잡는지 ───────────────────────────


def test_root_net_catches_names_with_no_holiday_suffix():
    """`크리스마스`·`추석` 은 접미사가 없어 어근 그물만 잡는다."""
    assert is_candidate("크리스마스") == {"root"}
    assert is_candidate("추석") == {"root"}


def test_suffix_net_catches_names_the_root_list_does_not_know():
    """어근 목록은 개방집합이라 그물이 하나뿐이면 모르는 이름이 조용히 빠진다.

    `세계실종아동의 날`·`와퍼의 날` 은 어근 목록에 없는 기념일인데 접미 그물이
    후보로 올린다. 이 경로가 없으면 원장은 그 자리를 판정할 기회조차 못 얻는다.
    """
    assert "suffix" in is_candidate("와퍼의 날")
    assert not any(root in "와퍼의 날" for root in HOLIDAY_ROOTS)


def test_suffix_net_skips_digit_bearing_dates():
    """`지난19일`·`10월 9일` 이 후보로 들어오면 판정이 형해화한다.

    숫자가 붙은 명절(`8.15 광복절`·`2월 설`)은 어근 그물이 이미 잡으므로,
    접미 그물을 숫자 없는 표면형으로 좁혀도 이름을 빠뜨리지 않는다.
    """
    assert is_candidate("지난19일") == set()
    assert "root" in is_candidate("8.15 광복절")
    assert "root" in is_candidate("2월 설")


def test_candidates_only_come_from_dat_and_evt():
    """`8월의 크리스마스`(PROD 작품명)는 다른 규칙 소관이라 후보가 아니다."""
    rows = [_row("1", "영화 8월의 크리스마스", [("PROD", 3, "8월의 크리스마스")])]
    assert holiday_candidates(rows) == {}


# ── 게이트 — 통과해야 하는 것 ──────────────────────────────────────────


def test_gate_passes_before_the_relabel():
    report = check_gate(_gold(), _ledger())
    assert report["evt"]["phase"] == "pre-apply"
    assert gate_failures(report) == []


def test_gate_passes_after_the_exact_relabel():
    rows = _relabel(_gold(), {"크리스마스", "설날"})
    report = check_gate(rows, _ledger())
    assert report["evt"]["phase"] == "applied"
    assert gate_failures(report) == []


# ── 게이트 — 막아야 하는 것 ────────────────────────────────────────────


def test_gate_blocks_unrelated_dat_moved_to_evt():
    """이 하네스가 존재하는 이유다.

    축1 `gate` 와 R2 `audit` 은 `이날` 을 `EVT` 로 바꿔도 판정이 한 글자도 안
    바뀐다 — 자리 모집단이 본문 텍스트에서 나오고, EVT 가 늘면 자리가
    `already_evt` 로 흡수만 되기 때문이다. 그래서 `NOT_HOLIDAY` 판정의 라벨까지
    원장에 묶어야 이 변경이 걸린다.

    **잡히는 것은 원장이 판정한 표면형뿐이다** — 그 밖의 자리는 아래
    `test_gate_is_blind_outside_the_candidate_nets` 가 고정한다.
    """
    rows = _relabel(_gold(), {"크리스마스", "설날", "이날"})
    failures = gate_failures(check_gate(rows, _ledger()))
    assert failures
    assert "이날" in " ".join(failures)


def test_gate_blocks_a_deleted_span():
    """옮길 자리를 정확히 옮기면서 다른 자리를 지우는 변경.

    "여덟 타입이 byte-identical" 류의 불변식은 `DAT`↔`EVT` 축을 비켜 가므로
    이것을 못 잡는다. 원장이 라벨 분포를 세므로 여기서 걸린다.
    """
    rows = _relabel(_gold(), {"크리스마스", "설날"})
    rows = [r for r in rows if r["id"] != "2"]
    failures = gate_failures(check_gate(rows, _ledger()))
    assert failures
    assert "설날" in " ".join(failures)


def test_gate_blocks_a_half_applied_relabel():
    """절반만 옮긴 gold 는 '진행 중' 으로 보여 그대로 굳는다."""
    rows = _relabel(_gold(), {"크리스마스"})
    report = check_gate(rows, _ledger())
    assert report["evt"]["phase"] == "partial"
    assert gate_failures(report)


def test_gate_blocks_a_dat_verdict_moved_to_evt():
    """`크리스마스 시즌` 은 머리가 기간이라 `DAT` 로 남아야 한다."""
    rows = _relabel(_gold(), {"크리스마스", "설날", "크리스마스 시즌"})
    failures = gate_failures(check_gate(rows, _ledger()))
    assert failures
    assert "크리스마스 시즌" in " ".join(failures)


def test_gate_blocks_an_unclassified_candidate():
    """원장이 판정을 빠뜨리면 나머지 수는 '판정한 것 안에서만' 맞는 수다."""
    ledger = [j for j in _judgements() if j["surface"] != "크리스마스"]
    failures = gate_failures(check_gate(_gold(), _ledger(ledger)))
    assert failures
    assert "크리스마스" in " ".join(failures)


def test_gate_blocks_a_ledger_surface_absent_from_gold():
    """gold 에 없는 표면형을 원장이 들고 있으면 원장이 앞질러 부푼 것이다."""
    ledger = _judgements() + [
        {"surface": "추석", "verdict": "EVT", "reason": "name",
         "count": 1, "labels": {"DAT": 1}, "sites": [["99", 0, 2]]}]
    failures = gate_failures(check_gate(_gold(), _ledger(ledger)))
    assert failures
    assert "추석" in " ".join(failures)


def test_gate_blocks_an_evt_keep_that_lost_its_label():
    """이미 EVT 인 명절 행사 이름이 `DAT` 로 내려가면 걸린다."""
    rows = json.loads(json.dumps(_gold()))
    for row in rows:
        for ent in row["entities"]:
            if ent["text"] == "강릉 단오제":
                ent["label"] = "DAT"
    failures = gate_failures(check_gate(rows, _ledger()))
    assert failures
    assert "강릉 단오제" in " ".join(failures)


# ── 원장 형식 ──────────────────────────────────────────────────────────


def test_ledger_rejects_a_duplicate_surface():
    """중복을 덮어쓰면 파일 안 순서가 규칙이 된다 — 사람이 읽어서 알 수 없다."""
    with pytest.raises(ValueError, match="duplicate"):
        ledger_index(_judgements() + [_judgements()[0]])


def test_ledger_rejects_an_unknown_verdict():
    item = dict(_judgements()[0], verdict="MAYBE")
    with pytest.raises(ValueError, match="unknown verdict"):
        ledger_index([item])


def test_ledger_rejects_an_unknown_reason():
    item = dict(_judgements()[0], reason="느낌")
    with pytest.raises(ValueError, match="unknown reason"):
        ledger_index([item])


# ── canonical ↔ 코드 동기 ──────────────────────────────────────────────


def test_module_constants_match_canonical_table():
    """명절 규칙의 정본은 canonical §5.3 이고 모듈 상수는 사본이다.

    규칙은 기준 파일(바꾸면 사람 승인·반박자를 타는 곳)에 있는데 그것을 집행하는
    이 모듈은 잠금 밖이다. 어근을 조용히 좁히면 후보 모집단이 줄어 게이트가 아무
    일 없이 통과한다. **양방향으로 대조한다** — 좁혀도 넓혀도 FAIL 이라야, 표에
    없는 어근을 코드에만 더해 두는 반대 방향도 막힌다.
    """
    parsed = parse_canonical_holiday()
    assert tuple(parsed["roots"]) == HOLIDAY_ROOTS
    assert tuple(parsed["period_names"]) == PERIOD_NAMES
    assert tuple(parsed["day_heads"]) == DAY_HEADS
    assert tuple(parsed["span_heads"]) == SPAN_HEADS
    assert tuple(parsed["category_heads"]) == CATEGORY_HEADS


def test_every_category_head_is_reachable_by_some_net():
    """범주어도 후보로 올라와야 판정을 받는다.

    어근 목록이 유일한 길은 아니다 — `휴일` 은 접미 그물이 잡는다. 요구는
    "어근에 있어야 한다" 가 아니라 **어느 그물로든 후보가 돼야 한다** 이다.
    """
    for word in CATEGORY_HEADS:
        assert is_candidate(word), word


def test_canonical_parser_fails_loudly_when_the_table_loses_a_marker(tmp_path):
    """빈 목록은 '규칙이 실제로 비어 있다' 와 구별되지 않는다.

    어근 0 개짜리 후보 생성기는 후보를 하나도 안 올리므로 미배정도 0 이 되고,
    게이트가 정상처럼 초록불을 낸다. 그래서 조용한 0 이 아니라 예외여야 한다.
    """
    doc = tmp_path / "canonical.md"
    doc.write_text(
        "### 5.3 KO\n\n| 표면형 | 판정 | 근거 |\n|---|---|---|\n"
        "| `설날` | `EVT` | **하루 머리**: `날` · **기간 머리**: `시즌` |\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="roots"):
        parse_canonical_holiday(str(doc))


# ── 게이트가 못 보는 것 — 범위를 주장이 아니라 테스트로 고정한다 ────────


def test_gate_is_blind_outside_the_candidate_nets():
    """**이 게이트는 원장이 판정한 표면형 안에서만 조인다.**

    후보 그물에 안 걸린 `DAT`·`EVT` span 을 옮기거나 지우는 변경은 여기서 안
    걸린다 — 그건 재라벨 `apply` 의 자리-집합 단언이 맡을 몫이다. 이 한계를
    산문으로만 적어 두면 다음 사람이 초록불을 "무관한 이동까지 잡은 통과" 로
    읽고 더 넓은 방어를 안 만든다. 그래서 통과한다는 사실 자체를 고정한다.
    """
    rows = _relabel(_gold(), {"크리스마스", "설날"})
    rows.append(_row("8", "오늘 발표했다", [("DAT", 0, "오늘")]))
    rows.append(_row("9", "올해 실적이다", [("EVT", 0, "올해")]))  # 잘못된 라벨
    ledger = _ledger()
    ledger["sweep"]["inventory_sha256"] = inventory_sha256(rows)
    assert gate_failures(check_gate(rows, ledger)) == []


# ── 자리 대조 ──────────────────────────────────────────────────────────


def test_gate_blocks_a_relocated_site():
    """라벨 분포만 세면 자리를 옮긴 변경이 안 걸린다.

    `크리스마스` span 을 지우고 다른 행에 같은 개수로 넣으면 분포가 그대로다.
    원장 좌표는 재라벨 `apply` 가 소비할 입력이라, 낡은 채로 통과하면 어긋난
    자리에 삽입된다.
    """
    rows = json.loads(json.dumps(_gold()))
    rows[0]["entities"] = []
    rows.append(_row("8", "크리스마스에 눈이", [("DAT", 0, "크리스마스")]))
    ledger = _ledger()
    ledger["sweep"]["inventory_sha256"] = inventory_sha256(rows)
    failures = gate_failures(check_gate(rows, ledger))
    assert failures
    assert "크리스마스@1" in " ".join(failures)


# ── 완전성 앵커 ────────────────────────────────────────────────────────


def test_gate_blocks_a_new_surface_type():
    """gold 에 새 표면형이 생기면 그물이 못 잡은 이름일 수 있다.

    `unclassified: 0` 은 후보 생성기 자신의 자로 잰 0 이라, 어근·접미 어느
    그물에도 안 걸리는 이름은 미배정으로도 안 잡힌다. 지문이 그 경로를 막는다.
    """
    rows = _gold() + [_row("8", "한가위 연휴", [("DAT", 0, "한가위")])]
    failures = gate_failures(check_gate(rows, _ledger()))
    assert failures
    assert "inventory moved" in " ".join(failures)


def test_the_inventory_anchor_survives_the_relabel():
    """재라벨은 라벨만 바꾸고 표면형 집합은 안 바꾼다 — 한 값이 양쪽을 지킨다."""
    before = inventory_sha256(_gold())
    after = inventory_sha256(_relabel(_gold(), {"크리스마스", "설날"}))
    assert before == after


def test_the_inventory_covers_surfaces_no_net_catches():
    """앵커가 후보만 덮으면 그물 밖을 못 지킨다 — 거기가 정확히 사각이다."""
    rows = _gold() + [_row("8", "오늘 발표했다", [("DAT", 0, "오늘")])]
    assert "오늘" in surface_inventory(rows)
    assert "오늘" not in holiday_candidates(rows)


def test_ledger_requires_the_gold_fingerprint_and_the_sweep_anchor(tmp_path):
    """앵커가 선택 항목이면 그것이 빠진 원장이 조용히 통과한다."""
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"judgements": _judgements()}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_ledger(str(path))


# ── 표기 변이 회귀 ─────────────────────────────────────────────────────


def test_the_root_list_carries_the_variants_gold_actually_uses():
    """어근에 변이 둘을 넣고 셋째를 빠뜨리는 것이 실제로 일어난 실패다.

    `핼러윈`·`할로윈` 이 이미 있는데 gold 의 `핼로윈` 이 새어 나갔고, 순우리말
    `한가위`·외래 명절 `춘제`·`春節`·`광군제`·`라마단` 도 함께 빠졌다. 그 자리를
    다시 잃지 않도록 못 박는다.
    """
    for surface in ("핼로윈", "핼러윈", "할로윈", "한가위", "춘제", "春節",
                    "광군제", "라마단"):
        assert "root" in is_candidate(surface), surface


# ── 원장이 자기 검사를 끄지 못하게 ──────────────────────────────────────


def test_ledger_rejects_a_judgement_with_no_sites():
    """`sites` 를 빼면 자리 대조가 그 표면형을 통째로 건너뛴다.

    그런데 리포트의 빈 `stale` 목록은 "이상 없음" 과 "아무것도 안 봤음" 을
    구별하지 못한다 — 감사 대상이 스스로 검사를 끄는 길이다.
    """
    item = {k: v for k, v in _judgements()[0].items() if k != "sites"}
    with pytest.raises(ValueError, match="no sites"):
        ledger_index([item])


def test_ledger_rejects_duplicate_sites():
    """자리가 중복되면 실제 자리 하나가 조용히 미검증으로 남는다."""
    item = dict(_judgements()[0], count=2,
                labels={"DAT": 2}, sites=[["1", 0, 5], ["1", 0, 5]])
    with pytest.raises(ValueError, match="repeats a site"):
        ledger_index([item])


def test_ledger_rejects_a_count_that_disagrees_with_its_own_sites():
    item = dict(_judgements()[0], count=999)
    with pytest.raises(ValueError, match="disagrees with itself"):
        ledger_index([item])


def test_the_gate_reports_how_many_sites_it_verified():
    """빈 `stale` 이 무엇을 뜻하는지는 검증한 수와 나란히 놓아야 읽힌다."""
    report = check_gate(_gold(), _ledger())
    assert report["sites"]["verified"] == 7
    assert report["sites"]["stale"] == []


def test_the_inventory_anchor_covers_digit_bearing_surfaces():
    """후보 그물은 숫자 있는 표면형을 빼지만 앵커는 빼면 안 된다.

    빼면 `DAT`+`EVT` span 의 64.6%가 감시 밖에 남아, 어근에 없는 새 이름이
    숫자를 달고 들어오면 그물도 앵커도 안 건드린다.
    """
    rows = _gold() + [_row("8", "2026 광군절 세일", [("DAT", 0, "2026 광군절")])]
    assert "2026 광군절" in surface_inventory(rows)
    assert is_candidate("2026 광군절") == set()
    assert gate_failures(check_gate(rows, _ledger()))


# ── 재라벨 ─────────────────────────────────────────────────────────────


def _apply(rows=None, ledger=None):
    rows = _gold() if rows is None else rows
    ledger = _ledger(rows=rows) if ledger is None else ledger
    return apply_ledger(rows, ledger, ledger["gold_sha256"])


def test_apply_moves_exactly_the_ledger_sites():
    new_rows, prov = _apply()
    assert prov["moved"] == 2
    moved = {(r["id"], e["text"]) for r in new_rows for e in r["entities"]
             if e["label"] == "EVT"}
    assert moved == {("1", "크리스마스"), ("2", "설날"),
                     ("5", "강릉 단오제"), ("7", "설명회")}


def test_apply_refuses_a_gold_the_ledger_did_not_judge():
    """좌표를 다른 코퍼스에서 잰 원장으로는 시작하지 않는다."""
    with pytest.raises(SystemExit, match="different corpus"):
        apply_ledger(_gold(), _ledger(), "f" * 64)


def test_apply_refuses_when_a_planned_site_is_not_dat():
    """원장이 옮기라는 자리가 이미 다른 라벨이면 원장이 낡은 것이다."""
    rows = _relabel(_gold(), {"크리스마스"})
    with pytest.raises(SystemExit, match="not DAT"):
        _apply(rows, _ledger(rows=rows))


def test_apply_preserves_everything_but_the_label():
    """행 수·id·text·행별 엔티티 수와 나머지 여덟 타입이 그대로여야 한다."""
    rows = _gold() + [_row("8", "서울 강남구", [("LOC", 0, "서울")])]
    new_rows, _ = _apply(rows, _ledger(rows=rows))
    assert len(new_rows) == len(rows)
    for old, new in zip(rows, new_rows):
        assert old["id"] == new["id"] and old["text"] == new["text"]
        assert len(old["entities"]) == len(new["entities"])
        for a, b in zip(old["entities"], new["entities"]):
            assert {k: v for k, v in a.items() if k != "label"} == \
                   {k: v for k, v in b.items() if k != "label"}
    assert new_rows[-1]["entities"][0]["label"] == "LOC"


def test_apply_is_a_set_equation_not_a_count():
    """개수만 맞추는 적용은 옮기면서 지우는 변경을 통과시킨다.

    자리 집합으로 단언하면 `DAT_after == DAT_before − 원장` 이 깨져 걸린다.
    """
    new_rows, _ = _apply()
    before_dat = {(r["id"], e["start_char"]) for r in _gold()
                  for e in r["entities"] if e["label"] == "DAT"}
    after_dat = {(r["id"], e["start_char"]) for r in new_rows
                 for e in r["entities"] if e["label"] == "DAT"}
    assert before_dat - after_dat == {("1", 0), ("2", 0)}


# ── 커밋된 산출물 ↔ 실제 gold ──────────────────────────────────────────
#
# 위 테스트는 전부 합성 gold 로 돈다. 그것만으로는 **커밋된 원장·provenance 가
# 실제 gold 를 증언하는지**를 아무도 안 본다 — gold 는 gitignore 라 diff 에 안
# 남고, 그래서 산출물이 유일한 기록인데 그 기록을 대조하는 기계가 없으면 숫자를
# 적어 둔 것일 뿐이다. 아래 넷이 그 반대심문이다.

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[3]
_GOLD = _REPO / "data" / "klue" / "pii_all.jsonl"
_DATA = _REPO / "src" / "ner" / "labelers" / "ko" / "data"


def _live_gold():
    if not _GOLD.exists():
        warnings.warn(
            f"KO gold 대조를 건너뛴다 — {_GOLD} 가 없다. `data/` 는 gitignore 라"
            " 워크트리마다 없을 수 있고, 그때 이 검사는 돌지 않는다.",
            stacklevel=2)
        pytest.skip(f"gold 없음: {_GOLD}")
    return load_gold(str(_GOLD))


def _committed(name):
    return json.loads((_DATA / name).read_text(encoding="utf-8"))


def test_the_committed_ledger_still_describes_the_live_gold():
    """원장의 1,372 자리가 지금 gold 에 그대로 있고 판정이 라벨과 맞는지."""
    rows = _live_gold()
    ledger = load_ledger(str(_DATA / "evt_holiday_prereg.json"))
    report = check_gate(rows, ledger)
    assert gate_failures(report) == []
    assert report["evt"]["phase"] == "applied"
    assert report["sites"]["verified"] == 1372


def test_reversing_the_ledger_reproduces_the_pre_relabel_gold():
    """**`before` 지문을 저장소만으로 반증 가능하게 만드는 테스트다.**

    재라벨 전 gold 는 제자리에서 덮여 디스크에 없고 `data/` 는 gitignore 라,
    `evt_holiday_apply.json` 의 `gold_sha256.before` 는 그냥 적힌 값이 될 수
    있다. 그런데 재라벨은 원장 자리에 대한 전단사라 되돌리면 정확히 복원된다 —
    그 sha 가 산출물의 `before` 와 다르면 지금 gold 가 원장이 말하는 그 gold 가
    아니거나 산출물이 틀린 것이고, 둘 다 알아야 할 일이다.
    """
    rows = _live_gold()
    ledger = load_ledger(str(_DATA / "evt_holiday_prereg.json"))
    planned = planned_moves(ledger["judgements"])
    reverted = 0
    for row in rows:
        for ent in row["entities"]:
            if (row["id"], ent["start_char"], ent["end_char"]) in planned:
                assert ent["label"] == "EVT", ent
                ent["label"] = "DAT"
                reverted += 1
    assert reverted == len(planned)

    provenance = _committed("evt_holiday_apply.json")
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "reversed.jsonl"
        dump_gold(rows, str(path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == provenance["gold_sha256"]["before"]
    assert digest == ledger["gold_sha256"]


def test_the_committed_provenance_matches_the_live_gold():
    """provenance 의 수를 gold 에서 다시 세어 대조한다 — 적힌 값을 안 믿는다."""
    rows = _live_gold()
    provenance = _committed("evt_holiday_apply.json")
    counts = collections.Counter(e["label"] for r in rows for e in r["entities"])
    assert counts["DAT"] == provenance["label_totals"]["after"]["DAT"]
    assert counts["EVT"] == provenance["label_totals"]["after"]["EVT"]
    assert (provenance["label_totals"]["before"]["DAT"]
            - provenance["label_totals"]["after"]["DAT"]) == provenance["moved"]
    assert (provenance["label_totals"]["after"]["EVT"]
            - provenance["label_totals"]["before"]["EVT"]) == provenance["moved"]
    assert sum(provenance["by_reason"].values()) == provenance["moved"]
    assert (hashlib.sha256(_GOLD.read_bytes()).hexdigest()
            == provenance["gold_sha256"]["after"])


def test_the_committed_ledger_sweep_anchor_still_holds():
    """앵커가 지금 gold 와 어긋나면 표면형이 움직인 것이다 — 새 이름일 수 있다."""
    rows = _live_gold()
    ledger = load_ledger(str(_DATA / "evt_holiday_prereg.json"))
    assert ledger["sweep"]["inventory_sha256"] == inventory_sha256(rows)


# ── 기간 머리가 뒤따르는 자리 ──────────────────────────────────────────
#
# gold 의 경계가 같은 표현에서 갈린다 — `추석 연휴` 를 한 span 으로 단 행이 있고
# `추석` 만 달고 `연휴` 를 안 단 행이 있다. 재라벨 전에는 둘 다 `DAT` 라 경계만
# 달랐는데, 앞엣것만 `EVT` 로 옮기면 **경계 차이가 타입 차이로 바뀐다.** 이게
# 이 규칙이 닫으려던 바로 그 비일관이라, 원장이 그 자리를 따로 판정해야 한다.


def _gold_with_period_head():
    return _gold() + [_row("8", "추석 연휴를 앞두고", [("DAT", 0, "추석")])]


def _ledger_with_period_head(exception=False):
    rows = _gold_with_period_head()
    item = {"surface": "추석", "verdict": "EVT", "reason": "name",
            "count": 1, "labels": {"DAT": 1}, "sites": [["8", 0, 2]]}
    if exception:
        item["site_exceptions"] = [
            {"site": ["8", 0, 2], "verdict": "DAT", "reason": "span_head",
             "note": "바로 뒤에 기간 머리 `연휴` 가 붙는다"}]
    ledger = _ledger(rows=rows)
    ledger["judgements"] = _judgements() + [item]
    return rows, ledger


def test_the_detector_follows_coordination():
    """`설과 추석 연휴` 에서 `연휴` 는 두 접속항에 걸린다.

    바로 뒤만 보면 뒤 접속항(`추석`)만 걸러지고 앞 접속항(`설`)은 `EVT` 로
    올라가 **한 명사구 안에서 타입이 갈린다** — 행 사이 불일치보다 나쁘다.
    """
    assert following_span_head("설과 추석 연휴를 4일로", 1) == "연휴"
    assert following_span_head("설날과 추석 연휴, 그리고", 2) == "연휴"
    assert following_span_head("매년 설, 추석, 휴가철 때 고속도로", 4) == "때"
    assert following_span_head("매년 설, 추석, 휴가철 때 고속도로", 8) == "때"
    # 등위가 기간 머리로 안 끝나면 따라가지 않는다
    assert following_span_head("설과 추석에 고향에 간다", 1) == ""


def test_span_heads_are_wired_into_a_decision():
    """canonical 의 기간 머리 목록이 파싱만 되고 아무 판정에도 안 쓰이면,
    규칙은 기준 파일에 적혀 있는데 그것을 세는 코드가 없는 상태다."""
    assert following_span_head("추석 연휴를 앞두고", 2) == "연휴"
    assert following_span_head("크리스마스 시즌에 보면", 5) == "시즌"
    assert following_span_head("추석에 고향에 간다", 2) == ""
    assert all(following_span_head(f"설 {h}", 1) == h for h in SPAN_HEADS)


def test_the_gate_blocks_a_move_with_a_period_head_right_after_it():
    rows, ledger = _ledger_with_period_head(exception=False)
    failures = gate_failures(check_gate(rows, ledger))
    assert failures
    assert "period head follows" in " ".join(failures)
    assert "추석+연휴@8" in " ".join(failures)


def test_a_site_exception_releases_that_one_site():
    """표면형 판정은 그대로 두고 그 자리만 `DAT` 로 남기는 탈출구."""
    rows, ledger = _ledger_with_period_head(exception=True)
    assert gate_failures(check_gate(rows, ledger)) == []
    assert ("8", 0, 2) not in planned_moves(ledger["judgements"])


def test_apply_holds_back_an_excepted_site():
    rows, ledger = _ledger_with_period_head(exception=True)
    new_rows, prov = apply_ledger(rows, ledger, ledger["gold_sha256"])
    held = [e for r in new_rows if r["id"] == "8" for e in r["entities"]]
    assert held[0]["label"] == "DAT"
    assert prov["held_by_site_exception"] == 1
    assert sum(prov["by_reason"].values()) == prov["moved"]


def test_ledger_rejects_an_exception_for_a_site_it_does_not_list():
    """원장이 자기가 안 적은 자리를 면제하면 그 면제는 아무도 대조할 수 없다."""
    item = dict(_judgements()[0], site_exceptions=[
        {"site": ["99", 0, 5], "verdict": "DAT", "reason": "span_head"}])
    with pytest.raises(ValueError, match="excepts a site it does not list"):
        ledger_index([item])


def test_the_committed_ledger_excepts_every_period_head_site():
    """커밋된 원장의 예외가 실제 gold 에서 기계로 다시 찾은 자리와 같은지."""
    rows = _live_gold()
    ledger = load_ledger(str(_DATA / "evt_holiday_prereg.json"))
    by_id = {row["id"]: row for row in rows}
    excepted, found = set(), set()
    for item in ledger["judgements"]:
        if item["verdict"] != "EVT":
            continue
        for exc in item.get("site_exceptions", []):
            excepted.add(tuple(exc["site"][:3]))
        for site in item["sites"]:
            row = by_id.get(site[0])
            if row and following_span_head(row["text"], site[2]):
                found.add(tuple(site[:3]))
    assert excepted == found
    assert len(found) == 15


def test_every_exception_records_how_the_head_attaches():
    """`attachment` 가 탐지기의 실제 홉 여부와 맞는지 대조한다.

    note 만으로는 `바로 뒤` 와 `등위로 나눠 가짐` 이 구별되지 않아, 등위 축이
    원장 안에서 다시 안 보인다 — R6 이 연 자리가 기록에서 사라지는 셈이다.
    """
    rows = _live_gold()
    ledger = load_ledger(str(_DATA / "evt_holiday_prereg.json"))
    by_id = {row["id"]: row for row in rows}
    seen = collections.Counter()
    for item in ledger["judgements"]:
        for exc in item.get("site_exceptions", []):
            rid, _, end = exc["site"][:3]
            text = by_id[rid]["text"]
            head = following_span_head(text, end)
            assert head == exc["head"], exc
            adjacent = text[end:end + 12].lstrip().startswith(head)
            assert exc["attachment"] == ("adjacent" if adjacent else "coordination"), exc
            seen[exc["attachment"]] += 1
    assert seen == collections.Counter({"adjacent": 11, "coordination": 4})


def test_every_exception_note_quotes_text_that_exists():
    """note 는 축3 이 만들 span 을 지정한다 — 그 값이 코퍼스에 없으면 안 된다.

    `추석연휴`(무공백) 자리에 `추석 연휴` 라 적으면 다음 사람이 없는 경계를
    근거로 삼는다. note 를 읽는 코드가 없으므로 여기서 고정한다.
    """
    rows = _live_gold()
    ledger = load_ledger(str(_DATA / "evt_holiday_prereg.json"))
    by_id = {row["id"]: row for row in rows}
    checked = 0
    for item in ledger["judgements"]:
        for exc in item.get("site_exceptions", []):
            quoted = re.findall(r"`([^`]+)`", exc["note"])
            text = by_id[exc["site"][0]]["text"]
            for piece in quoted:
                assert piece in text, (exc["site"], piece)
                checked += 1
    assert checked >= 30


# ── 기간 머리가 아닌 머리 ──────────────────────────────────────────────
#
# 완전성을 `following_span_head()` 로 재면 **자기 탐지기의 시야로 자기 완전성을
# 재는** 꼴이라, 아직 자로 인정 안 한 머리 부류가 있어도 0 이 나온다. 실제로
# 범주 머리(`한글날 공휴일`)와 하루 머리(`제헌절날`)가 그렇게 안 보였다.


def test_the_wide_sweep_sees_heads_the_gate_does_not_force():
    assert following_head("추석 연휴를", 2) == ("연휴", "span")
    assert following_head("한글날 공휴일 되찾기", 3) == ("공휴일", "category")
    assert following_head("제헌절날 극장개봉", 3) == ("날", "day")
    assert following_head("추석에 고향에", 2) == ("", "")


def test_the_gate_blocks_an_unreviewed_non_period_head():
    """타입은 안 바뀌지만 경계는 바뀐다 — 안 본 채로 지나가지는 못한다."""
    rows = _gold() + [_row("8", "한글날 공휴일 되찾기", [("DAT", 0, "한글날")])]
    ledger = _ledger(rows=rows)
    ledger["judgements"] = _judgements() + [
        {"surface": "한글날", "verdict": "EVT", "reason": "name",
         "count": 1, "labels": {"DAT": 1}, "sites": [["8", 0, 3]]}]
    failures = gate_failures(check_gate(rows, ledger))
    assert failures
    assert "never reviewed" in " ".join(failures)
    ledger["reviewed_heads"] = [
        {"site": ["8", 0, 3], "surface": "한글날", "head": "공휴일",
         "kind": "category", "verdict": "EVT", "note": "되찾을 지위다"}]
    assert gate_failures(check_gate(rows, ledger)) == []


def test_the_committed_ledger_reviewed_every_non_period_head():
    """커밋된 원장의 검토 목록이 넓은 스윕이 찾은 자리와 같은지."""
    rows = _live_gold()
    ledger = load_ledger(str(_DATA / "evt_holiday_prereg.json"))
    by_id = {row["id"]: row for row in rows}
    found = set()
    for key in planned_moves(ledger["judgements"]):
        head, kind = following_head(by_id[key[0]]["text"], key[2])
        if head and kind != "span":
            found.add(key)
    assert {tuple(i["site"][:3]) for i in ledger["reviewed_heads"]} == found
    assert len(found) == 2


def test_the_committed_noregress_matches_a_recomputation():
    """산출물이 스스로 `identical: true` 라 적는 것 말고 다시 세어 본다."""
    art = _committed("evt_holiday_noregress.json")
    assert art["axis1"]["before"]["status_counts"] == art["axis1"]["after"]["status_counts"]
    assert art["axis1"]["identical"] is True
    rate_before, rate_after = art["r2"]["gate_unchanged"]["rate"]
    assert rate_before == rate_after
    assert art["r2"]["gate_unchanged"]["violations"][0] == art["r2"]["gate_unchanged"]["violations"][1]
    provenance = _committed("evt_holiday_apply.json")
    assert art["gold_sha256"] == provenance["gold_sha256"]
    evt_before, evt_after = art["r2"]["queue_moved"]["evt_total"]
    assert evt_after - evt_before == provenance["moved"]
    assert len(art["r2_candidates_this_relabel_created"]["items"]) == (
        art["r2"]["queue_moved"]["candidates_total"][1]
        - art["r2"]["queue_moved"]["candidates_total"][0])


# ── certified 면책 ↔ 이슈 문서 ────────────────────────────────────────
#
# 재라벨은 `EVT` 분모를 1,731 → 1,869 로 옮겼다. `certified/classifier/ko/**` 의
# `EVT`·`DAT` 지표는 전부 그 전 gold 에서 잰 값이라 재라벨 후 모델과 나란히
# 놓으면 안 되는데, **그 사실은 문서의 산문일 뿐이라 아무도 안 센다.** 아래 셋이
# 그 산문을 원장·산출물에 묶는다.

_ISSUE_DOC = _REPO / "docs" / "issues" / "issue-222-ko-holiday-evt.md"
_CERTIFIED_KO = _REPO / "certified" / "classifier" / "ko"
_DOC_ROW = re.compile(r"^\|(.+)\|\s*$", re.M)


def _doc_cells():
    """이슈 문서의 표 행을 칸 목록으로. 구분선(`---`)은 뺀다."""
    text = _ISSUE_DOC.read_text(encoding="utf-8")
    rows = []
    for line in _DOC_ROW.findall(text):
        cells = [c.strip() for c in line.split("|")]
        if any(set(c) <= set("-: ") and c for c in cells):
            continue
        rows.append(cells)
    return rows


def _doc_number(cell):
    return int(re.sub(r"[^\d]", "", cell))


def _doc_before_after():
    """§certified 면책 의 전후 표 — 지문 접두와 라벨 총계."""
    out = {}
    for cells in _doc_cells():
        head = cells[0].strip()
        label = re.fullmatch(r"`(DAT|EVT)` span", head)
        if head == "gold sha256" and len(cells) == 3:
            out["sha"] = tuple(c.strip("`… ") for c in cells[1:])
        elif label and len(cells) == 3:
            out[label.group(1)] = tuple(_doc_number(c) for c in cells[1:])
    assert set(out) == {"sha", "DAT", "EVT"}, out
    return out


def _doc_certified():
    """§certified 면책 의 원장 표 — 파일별 (`DAT`, `EVT`) support."""
    out = {}
    for cells in _doc_cells():
        name = cells[0].strip("`")
        if name.endswith(".json") and len(cells) == 3:
            out[name] = tuple(_doc_number(c) for c in cells[1:])
    assert out, "원장 support 표를 못 읽었다 — 파서가 읽을 자리가 사라졌다"
    return out


def _pooled_support():
    """문서 표가 싣는 자리 — pooled metric 파일의 `strict.per_entity`.

    표에 그 다섯만 적는 것은 그것이 리포트가 인용하는 headline 지표이기
    때문이다. **원장에는 다른 스키마로 support 를 담는 파일도 있고**(교차·
    비순환 산출물의 `full`·`recovery_free_subset`), 그쪽까지 표에 옮기면 문서가
    원장의 사본이 된다. 면책의 실체 — 재라벨 후 분모로 잰 값이 없다 — 는 표가
    아니라 아래 `_all_support()` 전량 스캔이 본다.
    """
    out = {}
    for path in sorted(_CERTIFIED_KO.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        per = data.get("strict", {}).get("per_entity") if isinstance(data, dict) else None
        if not isinstance(per, dict) or not {"DAT", "EVT"} <= set(per):
            continue
        out[str(path.relative_to(_CERTIFIED_KO))] = (
            per["DAT"]["support"], per["EVT"]["support"])
    return out


def _all_support():
    """원장 **전량**에서 (`DAT`, `EVT`) support 쌍을 스키마 무관하게 긁는다.

    `strict.per_entity` 만 보면 시야가 좁아 **재라벨 후 gold 로 잰 실험을 다른
    꼴로 승격하는 경로**가 열린다 — 실제로 비순환 산출물 3 개가 그 밖에서
    support 를 담고 있고 그중 하나(`9925`/`1714`)는 어느 표에도 없는 값이다.
    그래서 부재 주장은 파일 스키마를 묻지 않고 트리 전체를 훑어 확인한다.
    """
    pairs = {}

    def walk(node, path):
        if isinstance(node, dict):
            dat, evt = node.get("DAT"), node.get("EVT")
            if isinstance(dat, dict) and isinstance(evt, dict) \
                    and "support" in dat and "support" in evt:
                pairs[path] = (dat["support"], evt["support"])
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    for path in sorted(_CERTIFIED_KO.rglob("*.json")):
        walk(json.loads(path.read_text(encoding="utf-8")),
             str(path.relative_to(_CERTIFIED_KO)))
    return pairs


def test_the_issue_doc_matches_the_committed_apply_provenance():
    """면책 표의 지문·라벨 총계가 apply 산출물과 같다.

    문서가 옮긴 수를 손으로 적으면 틀린다 — 이 이슈에서 실제로 세 라운드 연속
    같은 칸이 틀렸다. 그래서 산출물을 정본으로 두고 문서를 그것에 묶는다.
    """
    doc = _doc_before_after()
    totals = _committed("evt_holiday_apply.json")["label_totals"]
    digests = _committed("evt_holiday_apply.json")["gold_sha256"]
    assert doc["DAT"] == (totals["before"]["DAT"], totals["after"]["DAT"])
    assert doc["EVT"] == (totals["before"]["EVT"], totals["after"]["EVT"])
    assert digests["before"].startswith(doc["sha"][0]), doc["sha"]
    assert digests["after"].startswith(doc["sha"][1]), doc["sha"]
    assert len(doc["sha"][0]) >= 16 and len(doc["sha"][1]) >= 16


def test_the_issue_doc_lists_every_pooled_certified_ko_support_and_no_other():
    """문서 표가 pooled 원장 파일과 **양방향으로** 같다.

    한 방향만 보면 새는 길이 남는다 — pooled 실험을 승격하면서 표에 안 적으면
    "적힌 것은 전부 옛 gold" 가 여전히 참이라 통과한다.

    **이름이 곧 사정거리다** — 이 검사는 pooled 파일만 보며, 다른 스키마로
    support 를 담는 원장 파일은 아래 부재 검사가 맡는다.
    """
    assert _doc_certified() == _pooled_support()


def test_no_certified_ko_run_was_measured_on_the_relabelled_gold():
    """**면책의 실체** — 재라벨 후 분모로 잰 원장 항목이 하나도 없다.

    없다는 것이 면책의 근거이며, 이 조건이 깨지는 날(재라벨 후 실험 승격)에는
    문서의 "전부 재라벨 전 gold 에서 잰 값" 이 거짓이 되므로 함께 실패해야 한다.
    support 쌍으로 보는 것은 gold 지문이 원장 metric 파일에 안 적혀 있어서다 —
    분모가 그 자리를 대신한다.

    **스캔이 실제로 pooled 밖 파일까지 닿는지 함께 단언한다** — 추출기가 눈이
    멀면 부재는 공짜로 참이 되고, 그게 이 검사가 막으려는 바로 그 결함이다.

    **넓이는 개수가 아니라 파일로 잰다.** pooled 파일은 저마다 `strict` 와
    `relaxed` 두 쌍을 내므로 "쌍이 pooled 항목보다 많다" 는 pooled 안쪽만
    훑어도 성립한다(10 > 5). 그 판으로는 비-pooled 파일을 통째로 안 봐도
    통과했다 — 넓이를 구속하려면 **pooled 밖 파일에서 나온 쌍이 실재하는지**를
    물어야 한다.
    """
    after = _committed("evt_holiday_apply.json")["label_totals"]["after"]
    target = (after["DAT"], after["EVT"])
    everywhere = _all_support()
    pooled = _pooled_support()
    outside = {path: pair for path, pair in everywhere.items()
               if path.split(".json")[0] + ".json" not in pooled}
    assert outside, everywhere
    for name, pair in pooled.items():
        assert everywhere.get(f"{name}.strict.per_entity") == pair, name
    clashes = {k: v for k, v in everywhere.items() if v == target}
    assert not clashes, clashes
