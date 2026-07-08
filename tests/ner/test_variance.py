"""Tests for ner.metrics.variance — fold std and comparison-validity gate."""
import json
import statistics

import pytest

from ner.metrics.variance import (
    check_comparable,
    compare,
    fold_std,
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


def _pooled_doc(per_entity_f1, overall_f1, dups, n_folds):
    """합성 pooled_metrics.json dict 를 만든다."""
    per = {e: {"f1": f1, "precision": f1, "recall": f1, "support": 100}
           for e, f1 in per_entity_f1.items()}
    block = {"overall": {"f1": overall_f1, "precision": overall_f1,
                         "recall": overall_f1, "support": 1000},
             "per_entity": per}
    return {"strict": block, "relaxed": block,
            "n_folds": n_folds, "cross_fold_orig_dups": dups}


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


_CFG = {"lang": "ko", "data_path": "d.jsonl", "kfold": 2,
        "group_key": "orig"}


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

    def test_data_path_mismatch(self):
        ok, issues = check_comparable(
            dict(_CFG), dict(_CFG, data_path="other.jsonl"))
        assert not ok
        assert any("data_path" in s for s in issues)


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
            "delta", "sigma", "band", "band_source", "status"}
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
        (run / "pooled_metrics.json").write_text(
            json.dumps(pooled), encoding="utf-8")
        with pytest.raises(ValueError, match="cross_fold_orig_dups"):
            leakage_dups(run)


class TestFailLoud:
    def test_missing_ruler_field_not_comparable(self):
        """RULER 필드 키가 없으면 같은 자로 조용히 통과시키지 않는다."""
        a = {k: v for k, v in _CFG.items() if k != "data_path"}
        b = {k: v for k, v in _CFG.items() if k != "data_path"}
        ok, issues = check_comparable(a, b)
        assert not ok
        assert any("data_path" in s and "missing" in s for s in issues)

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
