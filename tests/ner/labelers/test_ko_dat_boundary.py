"""KO `DAT` 경계 정합·무라벨 회수 하네스 테스트.

**통과가 증거가 되려면 잘못된 편집을 실제로 막아야 한다.** 이 이슈의 위험은
숫자가 아니라 자를 스스로 옮기는 것이다 — 조사 목록을 좁히면 무라벨 자리가
줄어 회수가 공짜로 끝나고, 표면형 인벤토리 지문을 편집 후 gold 에서 다시
구하면 canonical 이 어근 개방집합의 앵커로 삼은 검사가 항진명제가 된다. 그래서
아래 테스트 절반이 변이 시나리오다 — 규칙·목록·목표값을 하나씩 흔들어 게이트가
무는지 본다. 베이스라인이 초록인 것만으로는 아무것도 증명되지 않는다.
"""

import json
import pathlib
import subprocess
import sys
import warnings

import pytest

from ner.labelers.ko import ko_dat_audit as M
from ner.labelers.ko.ko_evt_holiday_audit import check_gate, gate_failures, load_ledger

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DATA = _ROOT / "src" / "ner" / "labelers" / "ko" / "data"
_GOLD = _ROOT / "data" / "klue" / "origin.jsonl"
_CANONICAL = _ROOT / "docs" / "manual" / "data" / "canonical-entity-schema.md"


def _live_gold():
    if not _GOLD.exists():
        warnings.warn(
            f"KO gold 대조를 건너뛴다 — {_GOLD} 가 없다. `data/` 는 gitignore 라"
            " 워크트리마다 없을 수 있고, 그때 이 검사는 돌지 않는다.",
            stacklevel=2)
        pytest.skip(f"gold 없음: {_GOLD}")
    return M.read_gold(str(_GOLD))


def _committed(name):
    return json.loads((_DATA / name).read_text(encoding="utf-8"))


def _manifest():
    return _committed("dat_edit_manifest.json")


def _canonical_copy(tmp_path, transform):
    text = _CANONICAL.read_text(encoding="utf-8")
    path = tmp_path / "canonical.md"
    path.write_text(transform(text), encoding="utf-8")
    return str(path)


# ── canonical ──────────────────────────────────────────────────────────

def test_canonical_declares_every_dat_clause_parameter():
    spec = M.parse_canonical_dat()
    assert spec["include_head"] == ("기간머리",)
    assert spec["exclude_attachment"] == ("등위",)
    assert sorted(spec["codes"]) == sorted(M.VERDICTS)
    assert spec["threshold"] == 0.5
    # 모집단 최소 길이도 문턱과 같은 손잡이라 기준 파일이 값을 쥔다. 모듈 상수만
    # 두면 좁혀도 canonical 이 안 움직여 그 축소를 볼 것이 없다.
    assert spec["min_surface_length"] == M.MIN_SURFACE_LENGTH


def test_canonical_mutation_raising_the_length_floor_shrinks_the_population(tmp_path):
    """변이 — 최소 길이를 올리면 모집단이 실제로 줄어든다."""
    rows = _live_gold()
    surfaces = M.ledger_surfaces(str(_DATA / "evt_holiday_prereg.json"))
    before = M.revert_apply(rows, _manifest())
    baseline = M.completeness_population(before, surfaces)
    path = _canonical_copy(
        tmp_path,
        lambda t: t.replace("**모집단 최소 길이**: `2`",
                            "**모집단 최소 길이**: `3`"))
    raised = M.parse_canonical_dat(path)["min_surface_length"]
    narrowed = M.completeness_population(before, surfaces, min_length=raised)
    assert len(narrowed) < len(baseline)
    assert baseline - narrowed


def test_canonical_no_longer_defers_the_boundary_to_axis_three():
    """유예 문장이 남으면 기준 파일 안에 '고쳐라' 와 '나중에' 가 나란히 선다."""
    text = _CANONICAL.read_text(encoding="utf-8")
    assert "경계 자체의 교정은" not in text
    assert "§5.3 `경계 조항` 이 정한다" in text


def test_canonical_mutation_deleting_the_clause_is_loud(tmp_path):
    """변이 ① — 조항을 지우면 파서가 조용한 0 이 아니라 예외를 낸다."""
    path = _canonical_copy(
        tmp_path, lambda t: t.replace("**경계 포함 머리**", "**삭제된 머리**"))
    with pytest.raises(ValueError, match="missing DAT clause markers"):
        M.parse_canonical_dat(path)


def test_canonical_mutation_flipping_the_head_class_changes_enforcement(tmp_path):
    """변이 ② — 조항 토큰을 뒤집으면 강제 대상이 실제로 달라진다.

    마커가 장식이면 값을 뒤집어도 게이트 결과가 같다. 그러면 규칙이 기준 파일에
    적혀 있어도 아무 판정에 안 쓰이는 것이라 이 기준은 미달이다.
    """
    rows = _live_gold()
    baseline = M.boundary_sites(rows, M.parse_canonical_dat())
    path = _canonical_copy(
        tmp_path,
        lambda t: t.replace("**경계 포함 머리**: `기간머리`",
                            "**경계 포함 머리**: `하루머리`"))
    with pytest.raises(ValueError, match="unsupported boundary head class"):
        M.boundary_sites(rows, M.parse_canonical_dat(path))
    assert baseline, "베이스라인이 비면 변이 시험이 공짜로 통과한다"


def test_canonical_mutation_dropping_the_exception_widens_enforcement(tmp_path):
    """변이 ⑤ — 등위 예외를 지우면 평면 BIO 가 못 만드는 자리까지 강제 대상이 된다."""
    rows = _live_gold()
    spec = M.parse_canonical_dat()
    enforced = [s for s in M.boundary_sites(rows, spec) if s["enforced"]]
    widened = dict(spec, exclude_attachment=())
    all_sites = [s for s in M.boundary_sites(rows, widened) if s["enforced"]]
    assert len(all_sites) > len(enforced)
    assert len(all_sites) - len(enforced) == len(
        _manifest()["boundary"]["excluded"])


# ── manifest ───────────────────────────────────────────────────────────

def test_manifest_pins_the_lists_that_decide_the_population():
    """조사 목록과 어절 경계 규칙이 매니페스트에 오늘 값으로 박혀 있다.

    이 둘은 모집단 생성기이자 완전성 측정기다. 좁히면 무라벨 자리가 줄어 회수가
    공짜로 끝나고 '판정이 갈리는 자리 수' 도 함께 줄어 완화 장치가 못 된다.
    """
    manifest = _manifest()
    assert manifest["josa_sha256"] == M.josa_sha256()
    assert manifest["word_char_pattern"] == M.WORD_CHAR.pattern
    assert manifest["ledger_surface_count"] == M.LEDGER_SURFACE_COUNT
    assert manifest["recovery_threshold"] == 0.5


def test_manifest_gold_is_the_generation_before_head():
    chain = [entry["sha256"] for entry in _committed("gold_lineage.json")["chain"]]
    manifest = _manifest()
    assert manifest["gold_sha256"] == chain[-2]


def test_manifest_boundary_matches_a_fresh_enumeration():
    """열거를 다시 돌려 매니페스트 자리 집합과 대조한다. 개수 일치는 통과가 아니다."""
    rows = _live_gold()
    manifest = _manifest()
    before = M.revert_apply(rows, manifest)
    fresh = M.boundary_sites(before, M.parse_canonical_dat())
    fresh_keys = {(s["row_id"], s["start"], s["end"], s["enforced"]) for s in fresh}
    pinned = {
        (s["row_id"], s["start"], s["end"], True)
        for s in manifest["boundary"]["enforced"]
    } | {
        (s["row_id"], s["start"], s["end"], False)
        for s in manifest["boundary"]["excluded"]
    }
    assert fresh_keys == pinned


def test_manifest_completeness_matches_a_fresh_enumeration():
    rows = _live_gold()
    manifest = _manifest()
    before = M.revert_apply(rows, manifest)
    surfaces = M.ledger_surfaces(str(_DATA / "evt_holiday_prereg.json"))
    fresh = M.completeness_survey(before, surfaces, manifest["recovery_threshold"])
    above = {item["surface"] for item in fresh if item["above_threshold"]}
    pinned = {item["surface"] for item in manifest["completeness"]}
    assert above == pinned


def test_manifest_mutation_narrowing_josa_shrinks_the_population():
    """변이 ⑥ — 조사 목록을 좁히면 무라벨 자리가 줄고 지문이 어긋난다."""
    narrowed = tuple(j for j in M.JOSA if j in ("", "은", "는", "이", "가"))
    assert M.josa_sha256(narrowed) != _manifest()["josa_sha256"]


def test_manifest_mutation_dropping_the_ledger_difference_is_caught():
    """변이 ⑨ — 차집합을 없애면 원장이 판정한 표면형이 완전성 모집단에 든다."""
    rows = _live_gold()
    surfaces = M.ledger_surfaces(str(_DATA / "evt_holiday_prereg.json"))
    proper = M.completeness_population(rows, surfaces)
    unguarded = M.completeness_population(rows, set())
    leaked = unguarded & surfaces
    assert leaked, "원장 표면형이 gold 에 없으면 이 변이가 공짜로 통과한다"
    assert not (proper & surfaces)


def test_ledger_surface_count_is_pinned_outside_the_difference():
    """빼고 나서 교집합을 묻는 것은 항진이라, 종수를 밖에서 못 박는다."""
    with pytest.raises(ValueError, match="expected"):
        original = M.LEDGER_SURFACE_COUNT
        M.LEDGER_SURFACE_COUNT = original + 1
        try:
            M.ledger_surfaces(str(_DATA / "evt_holiday_prereg.json"))
        finally:
            M.LEDGER_SURFACE_COUNT = original


# ── apply ──────────────────────────────────────────────────────────────

def test_apply_reproduces_the_committed_gold():
    rows = _live_gold()
    manifest = _manifest()
    before = M.revert_apply(rows, manifest)
    applied = M.apply_manifest(before, manifest)
    assert not M.verify_apply(before, applied, manifest)


def test_revert_reproduces_the_previous_generation_byte_for_byte(tmp_path):
    """gold 는 gitignore 라 이 왕복이 유일한 앵커다."""
    rows = _live_gold()
    manifest = _manifest()
    before = M.revert_apply(rows, manifest)
    path = tmp_path / "before.jsonl"
    M.write_gold(before, str(path))
    assert M.gold_sha256(str(path)) == manifest["gold_sha256"]


def test_apply_mutation_corrupting_a_target_string_is_caught():
    """변이 ⑦ — 목표 문자열을 바꾸면 gold 본문과 안 맞아 멈춘다."""
    rows = _live_gold()
    manifest = json.loads(json.dumps(_manifest()))
    before = M.revert_apply(rows, manifest)
    manifest["boundary"]["enforced"][0]["target"] = "없는문자열"
    with pytest.raises(ValueError, match="does not match the manifest target"):
        M.apply_manifest(before, manifest)


def test_apply_mutation_corrupting_the_projected_inventory_is_caught():
    """변이 ⑧ — 인벤토리 기대 지문을 바꾸면 편집 후 실측과 어긋난다."""
    rows = _live_gold()
    manifest = json.loads(json.dumps(_manifest()))
    before = M.revert_apply(rows, manifest)
    manifest["inventory_sha256_after_expected"] = "0" * 64
    problems = M.verify_apply(before, M.apply_manifest(before, manifest), manifest)
    assert any("inventory fingerprint" in p for p in problems)


def test_apply_refuses_to_create_adjacent_dat_pairs():
    """회수가 한 덩어리를 쪼개면 경계를 정해야 하는데 그 자는 이 조항 밖이다."""
    rows = _live_gold()
    manifest = json.loads(json.dumps(_manifest()))
    before = M.revert_apply(rows, manifest)
    victim = None
    for row in before:
        spans = sorted(row["entities"], key=lambda e: e["start_char"])
        for ent in spans:
            if ent["label"] == "DAT" and ent["start_char"] >= 3:
                victim = (row, ent)
                break
        if victim:
            break
    row, ent = victim
    start = ent["start_char"] - 2
    manifest["holiday_gaps"] = manifest["holiday_gaps"] + [{
        "row_id": row["id"], "start": start, "end": ent["start_char"] - 1,
        "text": row["text"][start:ent["start_char"] - 1],
    }]
    problems = M.verify_apply(
        before, M.apply_manifest(before, manifest), manifest)
    assert any("adjacent DAT pairs" in p for p in problems)


def test_baseline_has_no_false_positives():
    """변이 없이도 걸리면 위 시험들은 아무것도 증명하지 않는다."""
    rows = _live_gold()
    manifest = _manifest()
    before = M.revert_apply(rows, manifest)
    assert not M.verify_apply(before, M.apply_manifest(before, manifest), manifest)
    assert not M.check_invariants(rows)


# ── generation ─────────────────────────────────────────────────────────

def test_generation_ledger_is_a_pure_transform_of_the_committed_inputs():
    """새 원장에 손으로 넣은 값이 하나라도 있으면 여기서 걸린다."""
    rows = _live_gold()
    old = _committed("evt_holiday_prereg.json")
    manifest = _manifest()
    rebuilt = M.transform_ledger(old, manifest, str(_GOLD), rows)
    assert rebuilt == _committed("evt_holiday_prereg_g2.json")


def test_generation_ledger_carries_the_projected_inventory_not_a_recount():
    """편집 후 gold 에서 다시 구하면 앵커가 스스로를 증명하는 값이 된다."""
    manifest = _manifest()
    g2 = _committed("evt_holiday_prereg_g2.json")
    assert g2["sweep"]["inventory_sha256"] == manifest[
        "inventory_sha256_after_expected"]


def test_generation_the_old_ledger_must_fail_on_the_new_gold():
    """옛 원장이 새 gold 에서 통과하면 게이트가 느슨해진 것이다."""
    rows = _live_gold()
    old = load_ledger(str(_DATA / "evt_holiday_prereg.json"))
    assert gate_failures(check_gate(rows, old))


def test_generation_the_new_ledger_passes_on_the_new_gold():
    rows = _live_gold()
    new = load_ledger(str(_DATA / "evt_holiday_prereg_g2.json"))
    assert not gate_failures(check_gate(rows, new))


def test_generation_mutation_reverting_gold_breaks_the_new_ledger():
    """변이 ④ — 고치기 전 gold 를 물리면 새 원장이 무는지가 판별력이다."""
    rows = _live_gold()
    before = M.revert_apply(rows, _manifest())
    new = load_ledger(str(_DATA / "evt_holiday_prereg_g2.json"))
    assert gate_failures(check_gate(before, new))


# ── lineage and regression ─────────────────────────────────────────────

def test_lineage_and_regression_every_gold_fingerprint_is_in_the_chain():
    """지문을 정규식으로 긁지 않는다 — 그 디렉토리의 64-hex 는 부류가 넷이다."""
    chain = {entry["sha256"] for entry in _committed("gold_lineage.json")["chain"]}
    truncated = {sha[:16] for sha in chain}
    offenders = []

    def walk(node, path, source):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}", source)
        elif isinstance(node, list):
            for value in node:
                walk(value, f"{path}[]", source)
        elif isinstance(node, str):
            # `gold_sha256` 이 경로에 있을 때만 본다. 이 디렉토리의 64-hex 는
            # 부류가 넷이라(예측 덤프·canonical·원장·인벤토리) 정규식으로 긁으면
            # 서로 다른 물건을 한 사슬에 묶게 된다.
            if "gold_sha256" not in path or len(node) not in (16, 64):
                return
            if node in chain or node[:16] in truncated:
                return
            offenders.append(f"{source}{path} = {node[:16]}")

    for path in sorted(_DATA.glob("*.json")):
        if path.name == "gold_lineage.json":
            continue
        walk(json.loads(path.read_text(encoding="utf-8")), "", path.name)
    assert not offenders, offenders


def test_lineage_and_regression_rejecting_loaders_point_at_the_head():
    """gold 지문 불일치를 거부하는 원장은 현 세대를 가리켜야 한다."""
    head = _committed("gold_lineage.json")["head"]
    assert _committed("ko_locorg_ledger.json")["gold_sha256"] == head
    assert _committed("evt_holiday_prereg_g2.json")["gold_sha256"] == head


def test_lineage_and_regression_r2_queue_moved_as_the_manifest_expected():
    """회수가 EVT 회수 후보를 덮으면 여기서 걸린다."""
    manifest = _manifest()
    expected = manifest["r2_queue_expected_delta"]
    assert expected.get("total") == -2
    assert expected.get("크리스마스") == -2


def test_manifest_pins_the_minimum_surface_length():
    """한 글자 가드도 조사 목록과 같은 성질이라 못 박아야 한다.

    좁히면(예: 3) 모집단이 줄어 회수가 공짜로 끝나는데, 못 박히지 않으면 그 축소를
    볼 것이 없다. 실제로 이 가드는 여섯 표면형을 모집단에서 뺀다.
    """
    manifest = _manifest()
    assert manifest["min_surface_length"] == M.MIN_SURFACE_LENGTH
    rows = _live_gold()
    before = M.revert_apply(rows, manifest)
    surfaces = M.ledger_surfaces(str(_DATA / "evt_holiday_prereg.json"))
    assert (manifest["completeness_population"]["surfaces"]
            == len(M.completeness_population(before, surfaces)))


def test_manifest_records_the_below_threshold_surfaces():
    """§5.3 회수 조항이 '문턱 아래는 회수 대상이 아니며 목록만 남긴다' 고 한 그 목록."""
    manifest = _manifest()
    rows = _live_gold()
    before = M.revert_apply(rows, manifest)
    surfaces = M.ledger_surfaces(str(_DATA / "evt_holiday_prereg.json"))
    survey = M.completeness_survey(before, surfaces, manifest["recovery_threshold"])
    below = {item["surface"] for item in survey if not item["above_threshold"]}
    assert below == {
        item["surface"] for item in manifest["completeness_below_threshold"]}
    assert below, "문턱 아래가 비면 이 검사가 공짜로 통과한다"


def test_lineage_and_regression_axis_one_is_recomputed_not_quoted():
    """커밋된 무회귀 수를 gold 에서 다시 세어 대조한다 — 적힌 값을 안 믿는다."""
    from ner.labelers.ko.ko_evt_axis1_audit import (
        AXIS1_HEADS, _load_jsonl, check_axis1_gate)

    rows = _live_gold()
    manifest = _manifest()
    before = M.revert_apply(rows, manifest)
    provenance = _committed("dat_edit_apply.json")["noregress"]["axis1"]
    ledger = _load_jsonl(str(_DATA / "evt_axis1_judgements.jsonl"))
    for label, arm in (("before", before), ("after", rows)):
        report = check_axis1_gate(arm, AXIS1_HEADS, ledger)
        assert report["sites"] == provenance[label]["sites"], label
        assert report["status_counts"] == provenance[label]["status_counts"], label
    assert provenance["identical"]


def test_lineage_and_regression_label_totals_are_recomputed():
    """provenance 의 분모를 gold 에서 다시 세어 대조한다."""
    import collections as _collections

    rows = _live_gold()
    manifest = _manifest()
    before = M.revert_apply(rows, manifest)
    provenance = _committed("dat_edit_apply.json")
    for label, arm in (("before", before), ("after", rows)):
        counts = _collections.Counter(
            e["label"] for r in arm for e in r["entities"])
        assert counts["DAT"] == provenance["label_totals"][label]["DAT"], label
        assert counts["EVT"] == provenance["label_totals"][label]["EVT"], label
    _, inserts = M.planned_edits(manifest)
    assert (provenance["label_totals"]["after"]["DAT"]
            - provenance["label_totals"]["before"]["DAT"]) == len(inserts)


def test_scan_refuses_recovery_sites_the_enumerator_never_proposed(tmp_path):
    """`site_verdicts` 는 자유 좌표를 적을 수 있는 유일한 자리다.

    매니페스트 안의 배열끼리 맞추는 검사들은 여기서 전부 조용하다 — 되감기조차 같은
    위조본으로 정의되니 일관되게 통과한다. 그래서 회수 자리는 매니페스트가 아니라
    **gold 에서 방금 나온 열거**에 부분집합으로 건다.
    """
    manifest = json.loads(json.dumps(_manifest()))
    moved = None
    for item in manifest["completeness"]:
        if item["verdict"] != "PER_SITE":
            continue
        for verdict in item["site_verdicts"]:
            if verdict["verdict"] == "RECOVER":
                verdict["site"] = [verdict["site"][0], 0, len(item["surface"])]
                moved = item["surface"]
                break
        if moved:
            break
    assert moved, "PER_SITE 회수 자리가 없으면 이 변이가 공짜로 통과한다"
    path = tmp_path / "stray.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    _live_gold()
    result = subprocess.run(
        [sys.executable, "-m", "ner.labelers.ko.ko_dat_audit", "scan",
         "--gold", str(_GOLD), "--manifest", str(path)],
        capture_output=True, text=True, cwd=str(_ROOT))
    assert result.returncode == 1, result.stdout
    assert "never proposed" in result.stdout


def test_scan_refuses_a_manifest_that_cannot_revert_the_gold(tmp_path):
    """되감기가 매니페스트를 입력으로 쓰므로 대조가 자기일관이 될 수 있다.

    자리를 하나 지우면 그 자리가 안 되감겨 gold 에 라벨된 채 남고, 무라벨 열거에서도
    함께 빠져 두 집합이 사이좋게 줄어든다. 지문 확인이 그 순환을 끊는다.
    """
    manifest = json.loads(json.dumps(_manifest()))
    for item in manifest["completeness"]:
        if item["verdict"] == "RECOVER" and len(item["sites"]) > 1:
            item["sites"] = item["sites"][1:]
            break
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    _live_gold()
    result = subprocess.run(
        [sys.executable, "-m", "ner.labelers.ko.ko_dat_audit", "scan",
         "--gold", str(_GOLD), "--manifest", str(path)],
        capture_output=True, text=True, cwd=str(_ROOT))
    assert result.returncode == 1, result.stdout
    assert "disagree about what this edit was" in result.stdout


def test_scan_passes_on_the_committed_manifest():
    """베이스라인이 초록이 아니면 위 변이 시험은 아무것도 증명하지 않는다."""
    _live_gold()
    result = subprocess.run(
        [sys.executable, "-m", "ner.labelers.ko.ko_dat_audit", "scan",
         "--gold", str(_GOLD), "--manifest", str(_DATA / "dat_edit_manifest.json")],
        capture_output=True, text=True, cwd=str(_ROOT))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "scan passed" in result.stdout


def test_lineage_and_regression_r2_queue_is_recomputed():
    """R2 큐 이동을 gold 에서 다시 재어 매니페스트 기대값과 맞춘다."""
    from ner.labelers.ko.ko_evt_r2_audit import (
        _load_jsonl, _report_dict, run_audit)

    rows = _live_gold()
    manifest = _manifest()
    before = M.revert_apply(rows, manifest)
    ledger = _load_jsonl(str(_DATA / "evt_r2_judgements.jsonl"))
    reports = {
        label: _report_dict(run_audit(arm, ledger=ledger))
        for label, arm in (("before", before), ("after", rows))
    }
    totals = [reports[k]["candidates"]["total"] for k in ("before", "after")]
    provenance = _committed("dat_edit_apply.json")["noregress"]["r2"]
    assert totals == provenance["queue_total"]
    assert totals[1] - totals[0] == manifest["r2_queue_expected_delta"]["total"]
    for label in ("before", "after"):
        assert reports[label]["coverage"] == provenance["coverage"][label]
        assert reports[label]["violations"] == provenance["violations"][label]


def test_holiday_gaps_come_from_a_text_scan_not_a_hand_list():
    """명절 무라벨 자리는 표면형 목록으로 못 찾으니 원문을 어근으로 훑는다.

    라벨된 적 없는 자리라 gold 의 표면형에서 뽑는 후보 그물에는 안 걸린다. 손으로
    적으면 그 좌표를 검사할 것이 없으므로 열거기가 세고 매니페스트를 그것에 건다.
    """
    rows = _live_gold()
    manifest = _manifest()
    before = M.revert_apply(rows, manifest)
    found = {
        (site["row_id"], site["start"], site["end"])
        for site in M.holiday_gap_sites(before)
    }
    pinned = {
        (gap["row_id"], gap["start"], gap["end"])
        for gap in manifest["holiday_gaps"]
    }
    assert found == pinned
    assert found, "빈 열거면 이 대조가 공짜로 통과한다"


def test_scan_refuses_a_holiday_gap_the_scanner_never_found(tmp_path):
    """변이 — 열거기가 제안 못 하는 자리를 `holiday_gaps` 에 넣으면 문다.

    이 배열도 `site_verdicts` 와 같은 문이었다. 커밋 밖 판정 파일에서 좌표가 그대로
    실려 오고 삽입으로 바뀌므로, 대조가 없으면 어디든 span 을 심을 수 있다.
    """
    manifest = json.loads(json.dumps(_manifest()))
    rows = _live_gold()
    before = M.revert_apply(rows, manifest)
    victim = next(
        (r for r in before if "오늘날" in r["text"]), None)
    assert victim, "이 코퍼스에 `오늘날` 이 없으면 다른 조각으로 시험해야 한다"
    start = victim["text"].index("오늘날")
    manifest["holiday_gaps"] = manifest["holiday_gaps"] + [{
        "row_id": victim["id"], "start": start, "end": start + 2,
        "text": victim["text"][start:start + 2],
    }]
    path = tmp_path / "gap.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "ner.labelers.ko.ko_dat_audit", "scan",
         "--gold", str(_GOLD), "--manifest", str(path)],
        capture_output=True, text=True, cwd=str(_ROOT))
    assert result.returncode == 1, result.stdout
    assert "holiday gaps differ from the enumerator" in result.stdout
    assert "not on word boundaries" in result.stdout


def test_boundary_targets_are_rederived_from_the_previous_ledger():
    """목표 문자열은 저자가 짓지 않고 앞 세대 원장의 note 에서 옮긴다.

    `apply` 가 gold 본문과 맞춰 보므로 아무 문자열이나 들어가진 않지만, 같은 자리의
    *다른* 유효한 확장은 통과한다. 그 값들은 앞 세대가 숫자를 보기 전에 고정했고
    코퍼스 실재도 이미 단언돼 있으므로, 재도출과 대조하면 그 문이 닫힌다.
    """
    rows = _live_gold()
    manifest = _manifest()
    before = M.revert_apply(rows, manifest)
    spec = M.parse_canonical_dat()
    derived = {
        (s["row_id"], s["start"], s["end"]): s["target"]
        for s in M._load_boundary(
            before, spec, str(_DATA / "evt_holiday_prereg.json"))
        if s["enforced"]
    }
    pinned = {
        (s["row_id"], s["start"], s["end"]): s["target"]
        for s in manifest["boundary"]["enforced"]
    }
    assert pinned == derived
    assert pinned, "빈 목록이면 이 대조가 공짜로 통과한다"
