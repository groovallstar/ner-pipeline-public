"""Tests for ner.metrics.variance — fold std and comparison-validity gate."""
import json
import statistics

import pytest

from ner.metrics.variance import (
    check_comparable,
    compare,
    fold_std,
    leakage,
    leakage_dups,
    load_sigma_map,
    repro_std,
    write_sigma_repro,
)


def _fold_doc(config, per_entity_f1, overall_f1):
    """합성 fold metrics.json dict 를 만든다 (strict==relaxed 단순화)."""
    per = {e: {"f1": f1, "precision": f1, "recall": f1, "support": 100}
           for e, f1 in per_entity_f1.items()}
    overall = {"f1": overall_f1, "precision": overall_f1,
               "recall": overall_f1, "support": 1000}
    doc = dict(config)
    doc["overall_strict"] = overall
    doc["per_entity_strict"] = per
    doc["overall_relaxed"] = overall
    doc["per_entity_relaxed"] = per
    doc["overall"] = overall
    doc["per_entity"] = per
    return doc


def _pooled_doc(per_entity_f1, overall_f1, dups, n_folds, basis="group"):
    """합성 pooled_metrics.json dict 를 만든다.

    basis 기본값은 'group' — 그룹 키로 센 신뢰 가능한 근거다. 근거 자체를
    검증하는 테스트는 이 키를 지우거나 덮어써서 쓴다.
    """
    per = {e: {"f1": f1, "precision": f1, "recall": f1, "support": 100}
           for e, f1 in per_entity_f1.items()}
    block = {"overall": {"f1": overall_f1, "precision": overall_f1,
                         "recall": overall_f1, "support": 1000},
             "per_entity": per}
    return {"strict": block, "relaxed": block,
            "n_folds": n_folds, "cross_fold_orig_dups": dups,
            "cross_fold_group_dups": dups, "leak_check_basis": basis}


def _write_run(root, name, config, fold_f1_list, pooled_f1,
               pooled_overall=0.9, dups=0):
    """fold{N}/metrics.json + pooled_metrics.json 을 가진 실험 디렉토리 생성."""
    run = root / name
    for i, per in enumerate(fold_f1_list):
        fold_dir = run / f"fold{i}"
        fold_dir.mkdir(parents=True)
        overall = sum(per.values()) / len(per)
        (fold_dir / "metrics.json").write_text(
            json.dumps(_fold_doc(config, per, overall)), encoding="utf-8")
    (run / "pooled_metrics.json").write_text(
        json.dumps(_pooled_doc(pooled_f1, pooled_overall, dups,
                               len(fold_f1_list))),
        encoding="utf-8")
    return run


_CFG = {"lang": "ko", "data_path": "d.jsonl", "data_fingerprint": "fp_a",
        "kfold": 2, "group_key": "orig", "seed": 42, "stratify": True}


class TestFoldStd:
    def test_matches_hand_value(self, tmp_path):
        """ORG f1 [0.80, 0.90] → mean 0.85, 표본 std sqrt(0.005)=0.07071."""
        run = _write_run(tmp_path, "base", _CFG,
                         [{"ORG": 0.80}, {"ORG": 0.90}], {"ORG": 0.85})
        stats = fold_std(run)
        assert stats["ORG"]["n"] == 2
        assert abs(stats["ORG"]["mean"] - 0.85) < 1e-9
        assert abs(stats["ORG"]["std"] - 0.0707106781) < 1e-6

    def test_overall_series_present(self, tmp_path):
        run = _write_run(tmp_path, "base", _CFG,
                         [{"ORG": 0.80}, {"ORG": 0.90}], {"ORG": 0.85})
        assert "overall" in fold_std(run)

    def test_single_fold_std_none(self, tmp_path):
        run = _write_run(tmp_path, "base", _CFG,
                         [{"ORG": 0.80}], {"ORG": 0.80})
        assert fold_std(run)["ORG"]["std"] is None


class TestCheckComparable:
    def test_same_ruler_ok(self):
        ok, issues = check_comparable(dict(_CFG), dict(_CFG))
        assert ok and issues == []

    def test_group_key_mismatch(self):
        a = dict(_CFG, group_key=None)
        b = dict(_CFG, group_key="orig")
        ok, issues = check_comparable(a, b)
        assert not ok
        assert any("group_key" in s for s in issues)

    def test_data_path_not_in_ruler(self):
        """경로만 다르고 내용(지문)이 같으면 비교 가능 — 경로는 자가 아니다."""
        ok, issues = check_comparable(
            dict(_CFG), dict(_CFG, data_path="renamed.jsonl"))
        assert ok and issues == []

    def test_data_fingerprint_mismatch(self):
        """경로가 같아도 내용이 바뀌면(지문 불일치) 비교 거부."""
        ok, issues = check_comparable(
            dict(_CFG), dict(_CFG, data_fingerprint="fp_b"))
        assert not ok
        assert any("data_fingerprint" in s for s in issues)

    def test_seed_in_ruler(self):
        """seed 는 fold 멤버십을 정하므로 다르면 비교 거부."""
        ok, issues = check_comparable(dict(_CFG), dict(_CFG, seed=7))
        assert not ok
        assert any("seed" in s for s in issues)

    def test_stratify_in_ruler(self):
        """stratify 는 fold 멤버십을 바꾸므로 다르면 비교 거부."""
        ok, issues = check_comparable(dict(_CFG), dict(_CFG, stratify=False))
        assert not ok
        assert any("stratify" in s for s in issues)


class TestCompare:
    def test_refuses_mismatched_group_key(self, tmp_path):
        a = _write_run(tmp_path, "a", dict(_CFG, group_key=None),
                       [{"ORG": 0.80}, {"ORG": 0.90}], {"ORG": 0.85})
        b = _write_run(tmp_path, "b", dict(_CFG, group_key="orig"),
                       [{"ORG": 0.85}, {"ORG": 0.95}], {"ORG": 0.92})
        v = compare(a, b, "ORG")
        assert v["verdict"] == "INVALID"
        assert not v["comparable"]
        assert any("group_key" in s for s in v["comparability_issues"])

    def test_fails_on_leakage(self, tmp_path):
        a = _write_run(tmp_path, "a", _CFG,
                       [{"ORG": 0.80}, {"ORG": 0.90}], {"ORG": 0.85})
        b = _write_run(tmp_path, "b", _CFG,
                       [{"ORG": 0.85}, {"ORG": 0.95}], {"ORG": 0.95},
                       dups=3)
        v = compare(a, b, "ORG")
        assert v["verdict"] == "FAIL"
        assert not v["leakage_ok"]
        assert v["leakage"]["candidate_dups"] == 3

    def test_pass_clean_gain(self, tmp_path):
        a = _write_run(tmp_path, "a", _CFG,
                       [{"ORG": 0.80, "LOC": 0.80},
                        {"ORG": 0.82, "LOC": 0.82}],
                       {"ORG": 0.81, "LOC": 0.81})
        b = _write_run(tmp_path, "b", _CFG,
                       [{"ORG": 0.90, "LOC": 0.81},
                        {"ORG": 0.92, "LOC": 0.83}],
                       {"ORG": 0.91, "LOC": 0.82})
        v = compare(a, b, "ORG")
        assert v["target"]["status"] == "real_gain"
        assert v["regressions"] == []
        assert v["verdict"] == "PASS"

    def test_flags_collateral_regression(self, tmp_path):
        """ORG 는 진짜 상승이지만 LOC 가 밴드 밖 회귀 → FAIL."""
        a = _write_run(tmp_path, "a", _CFG,
                       [{"ORG": 0.80, "LOC": 0.80},
                        {"ORG": 0.82, "LOC": 0.82}],
                       {"ORG": 0.81, "LOC": 0.81})
        b = _write_run(tmp_path, "b", _CFG,
                       [{"ORG": 0.90, "LOC": 0.70},
                        {"ORG": 0.92, "LOC": 0.72}],
                       {"ORG": 0.91, "LOC": 0.71})
        v = compare(a, b, "ORG")
        assert v["target"]["status"] == "real_gain"
        assert v["verdict"] == "FAIL"
        assert any(r["entity"] == "LOC" for r in v["regressions"])

    def test_inconclusive_within_noise(self, tmp_path):
        """Δ 가 밴드 안이면 진짜 상승 아님 → INCONCLUSIVE."""
        a = _write_run(tmp_path, "a", _CFG,
                       [{"ORG": 0.80}, {"ORG": 0.90}], {"ORG": 0.85})
        b = _write_run(tmp_path, "b", _CFG,
                       [{"ORG": 0.83}, {"ORG": 0.93}], {"ORG": 0.88})
        v = compare(a, b, "ORG")
        assert v["target"]["status"] == "within_noise"
        assert v["verdict"] == "INCONCLUSIVE"

    def test_all_deltas_has_band_and_status(self, tmp_path):
        a = _write_run(tmp_path, "a", _CFG,
                       [{"ORG": 0.80}, {"ORG": 0.82}], {"ORG": 0.81})
        b = _write_run(tmp_path, "b", _CFG,
                       [{"ORG": 0.90}, {"ORG": 0.92}], {"ORG": 0.91})
        v = compare(a, b, "ORG")
        assert set(v["all_deltas"]["ORG"]) == {
            "delta", "sigma", "band", "band_source", "status",
            "fold_wins", "fold_losses", "fold_n", "consistent",
            "consistency_downgraded"}
        assert v["all_deltas"]["ORG"]["band_source"] == "sigma_fold"

    def test_sigma_repro_override_tightens_band(self, tmp_path):
        """σ_fold 밴드로는 within_noise 인 Δ 가 더 좁은 σ_repro 로는 진짜 상승."""
        a = _write_run(tmp_path, "a", _CFG,
                       [{"ORG": 0.80}, {"ORG": 0.90}], {"ORG": 0.85})
        b = _write_run(tmp_path, "b", _CFG,
                       [{"ORG": 0.85}, {"ORG": 0.95}], {"ORG": 0.90})
        # σ_fold=0.0707 → band 0.1414 → Δ 0.05 는 노이즈 안
        v_fold = compare(a, b, "ORG")
        assert v_fold["target"]["status"] == "within_noise"
        assert v_fold["target"]["band_source"] == "sigma_fold"
        # σ_repro=0.01 → band 0.02 → Δ 0.05 는 진짜 상승
        v_repro = compare(a, b, "ORG", sigma_override={"ORG": 0.01})
        assert v_repro["target"]["status"] == "real_gain"
        assert v_repro["target"]["band_source"] == "sigma_repro"
        assert v_repro["verdict"] == "PASS"


class TestConsistencyGate:
    """방향 일관성 게이트 — 조이기 전용. gain 을 내릴 뿐 절대 올리지 않는다."""

    def test_inconsistent_gain_downgraded_to_inconclusive(self, tmp_path):
        """pooled 는 밴드 밖 상승이지만 한 fold 가 끌어올린 경우 → INCONCLUSIVE.

        fold0 은 +0.30(승), fold1 은 −0.02(패). pooled Δ 는 밴드 밖이라
        magnitude 는 real_gain 이지만 승률 1/2 < 2/3 → within_noise 로 내려간다.
        """
        a = _write_run(tmp_path, "a", _CFG,
                       [{"ORG": 0.50}, {"ORG": 0.90}], {"ORG": 0.70})
        b = _write_run(tmp_path, "b", _CFG,
                       [{"ORG": 0.80}, {"ORG": 0.88}], {"ORG": 0.84})
        # 밴드를 좁혀(σ_repro) magnitude 를 real_gain 으로 만든 뒤 게이트를 본다.
        # fold0 Δ=+0.30(승), fold1 Δ=−0.02(패) → 승률 1/2 < 2/3.
        v = compare(a, b, "ORG", sigma_override={"ORG": 0.01})
        assert v["target"]["fold_wins"] == 1
        assert v["target"]["fold_losses"] == 1
        assert v["target"]["consistent"] is False
        assert v["target"]["consistency_downgraded"] is True
        assert v["target"]["status"] == "within_noise"
        assert v["verdict"] == "INCONCLUSIVE"

    def test_consistent_gain_survives(self, tmp_path):
        """모든 fold 에서 이기면 게이트를 통과한다."""
        a = _write_run(tmp_path, "a", _CFG,
                       [{"ORG": 0.80}, {"ORG": 0.82}], {"ORG": 0.81})
        b = _write_run(tmp_path, "b", _CFG,
                       [{"ORG": 0.90}, {"ORG": 0.92}], {"ORG": 0.91})
        v = compare(a, b, "ORG", sigma_override={"ORG": 0.01})
        assert v["target"]["consistent"] is True
        assert v["target"]["consistency_downgraded"] is False
        assert v["target"]["status"] == "real_gain"
        assert v["verdict"] == "PASS"

    def test_gate_never_upgrades(self, tmp_path):
        """일관된 방향이어도 magnitude 가 within_noise 면 gain 으로 안 올린다."""
        a = _write_run(tmp_path, "a", _CFG,
                       [{"ORG": 0.80}, {"ORG": 0.81}], {"ORG": 0.805})
        b = _write_run(tmp_path, "b", _CFG,
                       [{"ORG": 0.81}, {"ORG": 0.82}], {"ORG": 0.815})
        # 두 fold 다 +0.01 승(2/2 일관)이지만 pooled Δ=0.01 은 σ_fold 밴드 안
        v = compare(a, b, "ORG")
        assert v["target"]["fold_wins"] == 2
        assert v["target"]["status"] == "within_noise"
        assert v["verdict"] == "INCONCLUSIVE"

    def test_gate_never_suppresses_regression(self, tmp_path):
        """회귀는 소수 fold 에서 나타나도 계속 막는다(조이기 전용의 비대칭)."""
        a = _write_run(tmp_path, "a", _CFG,
                       [{"ORG": 0.90, "LOC": 0.80},
                        {"ORG": 0.90, "LOC": 0.80}],
                       {"ORG": 0.90, "LOC": 0.80})
        # ORG 일관 상승, LOC 는 fold0 만 큰 회귀(1/2) — 그래도 flag 되어야 한다
        b = _write_run(tmp_path, "b", _CFG,
                       [{"ORG": 0.95, "LOC": 0.40},
                        {"ORG": 0.95, "LOC": 0.81}],
                       {"ORG": 0.95, "LOC": 0.61})
        v = compare(a, b, "ORG", sigma_override={"ORG": 0.01, "LOC": 0.01})
        assert v["target"]["status"] == "real_gain"
        assert any(r["entity"] == "LOC" for r in v["regressions"])
        assert v["verdict"] == "FAIL"


class TestLeakageDups:
    def test_reads_counter(self, tmp_path):
        run = _write_run(tmp_path, "a", _CFG,
                         [{"ORG": 0.80}, {"ORG": 0.90}], {"ORG": 0.85},
                         dups=5)
        assert leakage_dups(run) == 5

    def test_missing_counter_raises(self, tmp_path):
        """카운터 키가 없으면 0 가정 없이 예외를 던진다(fail-loud)."""
        run = _write_run(tmp_path, "a", _CFG,
                         [{"ORG": 0.80}, {"ORG": 0.90}], {"ORG": 0.85})
        pooled = json.loads(
            (run / "pooled_metrics.json").read_text(encoding="utf-8"))
        del pooled["cross_fold_orig_dups"]
        del pooled["cross_fold_group_dups"]
        (run / "pooled_metrics.json").write_text(
            json.dumps(pooled), encoding="utf-8")
        with pytest.raises(ValueError, match="missing leak counter"):
            leakage_dups(run)


class TestFailLoud:
    def test_missing_ruler_field_not_comparable(self):
        """RULER 필드 키가 없으면 같은 자로 조용히 통과시키지 않는다.

        지문 없는 옛 산출물은 fail-loud INVALID 여야 한다.
        """
        a = {k: v for k, v in _CFG.items() if k != "data_fingerprint"}
        b = {k: v for k, v in _CFG.items() if k != "data_fingerprint"}
        ok, issues = check_comparable(a, b)
        assert not ok
        assert any("data_fingerprint" in s and "missing" in s for s in issues)

    def test_group_key_null_is_valid_value(self):
        """group_key=null 은 유효한 값 — 양쪽 null 이면 같은 자로 본다."""
        a = dict(_CFG, group_key=None)
        b = dict(_CFG, group_key=None)
        ok, issues = check_comparable(a, b)
        assert ok and issues == []


class TestReproStd:
    def _seed_runs(self, tmp_path, pooled_f1s):
        """pooled ORG F1 이 서로 다른 시드-반복 CV run 들을 만든다."""
        return [
            _write_run(tmp_path, f"seed{i}", _CFG,
                       [{"ORG": f1}, {"ORG": f1}], {"ORG": f1})
            for i, f1 in enumerate(pooled_f1s)
        ]

    def test_matches_hand_value(self, tmp_path):
        runs = self._seed_runs(tmp_path, [0.80, 0.82, 0.84])
        r = repro_std(runs)
        assert r["ORG"]["n"] == 3
        assert abs(r["ORG"]["mean"] - 0.82) < 1e-9
        assert abs(r["ORG"]["std"]
                   - statistics.stdev([0.80, 0.82, 0.84])) < 1e-9

    def test_needs_two_runs(self, tmp_path):
        runs = self._seed_runs(tmp_path, [0.80])
        with pytest.raises(ValueError, match="seed-repeat"):
            repro_std(runs)

    def test_write_and_load_sigma_map(self, tmp_path):
        runs = self._seed_runs(tmp_path, [0.80, 0.82, 0.84])
        out = tmp_path / "sigma_repro.json"
        payload = write_sigma_repro(runs, out)
        assert payload["n_runs"] == 3
        smap = load_sigma_map(out)
        assert abs(smap["ORG"]
                   - statistics.stdev([0.80, 0.82, 0.84])) < 1e-9


def _patch_pooled(run, **fields):
    """pooled_metrics.json 의 필드를 덮어쓴다."""
    path = run / "pooled_metrics.json"
    pooled = json.loads(path.read_text(encoding="utf-8"))
    pooled.update(fields)
    path.write_text(json.dumps(pooled), encoding="utf-8")


class TestLeakageVerification:
    """미측정·약한 근거는 크래시가 아니라 INVALID 다.

    정직하게 opt-out 한 run 이 죽고 거짓 0 을 낸 run 이 통과하면, 규칙이
    편법을 보상하게 된다.
    """

    def _pair(self, tmp_path):
        base = _write_run(tmp_path, "base", _CFG,
                          [{"ORG": 0.80}, {"ORG": 0.84}], {"ORG": 0.82})
        cand = _write_run(tmp_path, "cand", _CFG,
                          [{"ORG": 0.90}, {"ORG": 0.94}], {"ORG": 0.92})
        return base, cand

    def test_trusted_basis_zero_dups_passes(self, tmp_path):
        base, cand = self._pair(tmp_path)
        for run in (base, cand):
            _patch_pooled(run, cross_fold_group_dups=0,
                          leak_check_basis="group")
        result = compare(base, cand, "ORG")
        assert result["leakage_verified"] is True
        assert result["leakage_ok"] is True
        assert result["verdict"] != "INVALID"

    def test_unmeasured_leakage_is_invalid_not_crash(self, tmp_path):
        base, cand = self._pair(tmp_path)
        _patch_pooled(cand, cross_fold_group_dups=None,
                      leak_check_basis="none")
        result = compare(base, cand, "ORG")
        assert result["verdict"] == "INVALID"
        assert result["leakage_verified"] is False
        assert any("unmeasured" in i for i in result["comparability_issues"])

    def test_text_basis_zero_is_not_trusted(self, tmp_path):
        """문장 비교로 센 0 은 '이상 없음'이 아니라 '볼 수 없었음'이다."""
        base, cand = self._pair(tmp_path)
        _patch_pooled(cand, cross_fold_group_dups=0,
                      leak_check_basis="text")
        result = compare(base, cand, "ORG")
        assert result["verdict"] == "INVALID"
        assert result["leakage_verified"] is False
        assert any("untrustworthy" in i
                   for i in result["comparability_issues"])

    def test_real_leak_still_fails(self, tmp_path):
        base, cand = self._pair(tmp_path)
        for run in (base, cand):
            _patch_pooled(run, leak_check_basis="group")
        _patch_pooled(cand, cross_fold_group_dups=3)
        result = compare(base, cand, "ORG")
        assert result["verdict"] == "FAIL"
        assert result["leakage_verified"] is True
        assert result["leakage_ok"] is False

    def test_observed_leak_fails_even_on_weak_basis(self, tmp_path):
        """약한 근거는 누출을 놓칠 뿐 만들어내지 않는다 — 본 누출은 FAIL 이다.

        dups>0 이면 근거가 text 여도 실제로 관측된 것이므로, INVALID 로
        숨기지 않는다.
        """
        base, cand = self._pair(tmp_path)
        _patch_pooled(base, cross_fold_group_dups=0,
                      leak_check_basis="group")
        _patch_pooled(cand, cross_fold_group_dups=5,
                      leak_check_basis="text")
        result = compare(base, cand, "ORG")
        assert result["verdict"] == "FAIL"
        assert result["leakage_ok"] is False

    def test_legacy_pooled_without_basis_is_not_trusted(self, tmp_path):
        """근거 키가 없는 옛 산출물은 'unknown' — 무엇으로 셌는지 모른다.

        'orig' 로 간주하면 실제로는 text 기준이던 옛 run 을 신뢰하게 된다.
        """
        base, cand = self._pair(tmp_path)
        for run in (base, cand):
            path = run / "pooled_metrics.json"
            pooled = json.loads(path.read_text(encoding="utf-8"))
            del pooled["leak_check_basis"]
            path.write_text(json.dumps(pooled), encoding="utf-8")
        dups, basis = leakage(cand)
        assert dups == 0 and basis == "unknown"
        result = compare(base, cand, "ORG")
        assert result["verdict"] == "INVALID"
        assert result["leakage_verified"] is False
        assert any("untrustworthy" in i
                   for i in result["comparability_issues"])
