"""옛 실험 두 팔의 fold 대응 감사 테스트.

핵심은 **판정을 못 하는 상황에서 판정을 만들지 않는 것**이다. 이 감사는 겹침이
낮게 나오는 쪽이 가설이 예상한 답이라, 도구가 고장 나면 그 고장이 곧 "예상대로"
로 읽힌다. 회수 역적용·fold 겹침·판정 조건과 보고서의 지문·양성 대조 검증을
검사하며, 근거 없는 추측이 판정으로 굳지 않도록 한다.
"""

import pytest

from ner.labelers.ko.ko_split_audit import (
    OVERLAP_BROKEN_MAX,
    OVERLAP_MATCHED_MIN,
    VERDICT_INVALID,
    VERDICT_UNDECIDABLE,
    VERDICT_VALID,
    check_report,
    classify_overlap,
    decide,
    fold_overlap,
    strip_recovered,
)


def _row(row_id, ents):
    return {"id": row_id, "text": "x" * 40,
            "entities": [{"label": lab, "start_char": s, "end_char": e,
                          "text": "x" * (e - s)} for lab, s, e in ents]}


# --- 회수 역적용 -----------------------------------------------------------

def test_strip_removes_only_the_ledger_coordinates():
    rows = [_row("1", [("EVT", 0, 4), ("EVT", 10, 14), ("ORG", 20, 24)])]
    ledger = [{"row_index": 0, "start": 0, "end": 4, "verdict": "EVT"}]
    out, removed = strip_recovered(rows, ledger, "start", "end")
    assert removed == 1
    assert [(e["label"], e["start_char"]) for e in out[0]["entities"]] == [
        ("EVT", 10), ("ORG", 20)]


def test_strip_ignores_not_verdicts():
    rows = [_row("1", [("EVT", 0, 4)])]
    ledger = [{"row_index": 0, "start": 0, "end": 4, "verdict": "NOT"}]
    out, removed = strip_recovered(rows, ledger, "start", "end")
    assert removed == 0
    assert len(out[0]["entities"]) == 1


def test_strip_does_not_touch_other_labels_at_the_same_span():
    """같은 좌표라도 EVT 가 아니면 남는다 — 회수는 EVT 만 넣었다."""
    rows = [_row("1", [("ORG", 0, 4)])]
    ledger = [{"row_index": 0, "start": 0, "end": 4, "verdict": "EVT"}]
    out, removed = strip_recovered(rows, ledger, "start", "end")
    assert removed == 0
    assert out[0]["entities"][0]["label"] == "ORG"


def test_strip_uses_the_requested_offset_fields():
    """원장마다 좌표 필드 이름이 다르다 — 축1 회수는 insert_* 를 쓴다."""
    rows = [_row("1", [("EVT", 5, 9)])]
    ledger = [{"row_index": 0, "start": 0, "end": 9,
               "insert_start": 5, "insert_end": 9, "verdict": "EVT"}]
    out, removed = strip_recovered(rows, ledger, "insert_start", "insert_end")
    assert removed == 1
    assert out[0]["entities"] == []


# --- 겹침의 자 -------------------------------------------------------------

def test_overlap_is_intersection_over_base_fold_not_jaccard():
    """자가 바뀌면 같은 사건의 수가 문서마다 달라진다."""
    base = [{"a", "b", "c", "d"}]
    head = [{"a", "b", "e", "f", "g", "h"}]
    got = fold_overlap(base, head)
    assert got["mean_fold_overlap"] == 0.5      # 2/4, Jaccard 라면 0.25
    assert got["exact_folds"] == 0


def test_overlap_reports_identical_folds():
    base = [{"a", "b"}, {"c"}]
    got = fold_overlap(base, [{"a", "b"}, {"c"}])
    assert got["exact_folds"] == 2
    assert got["mean_fold_overlap"] == 1.0


# --- 문턱과 판정 -----------------------------------------------------------

def test_thresholds_were_fixed_before_measuring():
    """문턱이 실측값에 맞춰 움직이면 자를 결과에 맞춘 것이 된다."""
    assert OVERLAP_BROKEN_MAX == 0.20
    assert OVERLAP_MATCHED_MIN == 0.90


@pytest.mark.parametrize("overlap,expected", [
    (1.0, VERDICT_VALID),
    (0.9, VERDICT_VALID),
    (0.1012, VERDICT_INVALID),
    (0.0, VERDICT_INVALID),
    (0.5, VERDICT_UNDECIDABLE),      # 문턱 사이는 사람이 본다
])
def test_classify_overlap(overlap, expected):
    assert classify_overlap(overlap) == expected


def test_decide_refuses_to_choose_without_a_documented_basis():
    """두 시나리오가 갈리는데 근거가 없으면 판정이 아니라 추측이다."""
    scenarios = {"head_no_stratify": {"verdict": VERDICT_VALID},
                 "head_stratify": {"verdict": VERDICT_INVALID}}
    verdict, reason = decide(scenarios, head_basis=None)
    assert verdict == VERDICT_UNDECIDABLE
    assert "not recorded" in reason


def test_decide_uses_the_documented_scenario_when_there_is_one():
    scenarios = {"head_no_stratify": {"verdict": VERDICT_VALID},
                 "head_stratify": {"verdict": VERDICT_INVALID}}
    verdict, _ = decide(scenarios, head_basis="head_stratify")
    assert verdict == VERDICT_INVALID


def test_decide_can_conclude_without_a_basis_when_scenarios_agree():
    """설정이 답을 안 바꾸면 설정을 몰라도 답은 정해진다."""
    scenarios = {"head_no_stratify": {"verdict": VERDICT_INVALID},
                 "head_stratify": {"verdict": VERDICT_INVALID}}
    verdict, reason = decide(scenarios, head_basis=None)
    assert verdict == VERDICT_INVALID
    assert "agree" in reason


# --- 산출물 자체 검사 -------------------------------------------------------

def _report(verdict=VERDICT_UNDECIDABLE, head_basis=None, match=True,
            control_exact=10):
    return {
        "arms": {"base": {"basis": "doc:1"}, "head": {"basis": head_basis}},
        "gold_reconstruction": {"match": match, "mismatch": {} if match else
                                {"base": {"expected": "a", "got": "b"}}},
        "positive_control": {"run": {"exact_folds": control_exact,
                                     "n_folds": 10}},
        "scenarios": {"head_no_stratify": {"verdict": VERDICT_VALID},
                      "head_stratify": {"verdict": VERDICT_INVALID}},
        "verdict": verdict,
    }


def test_check_report_passes_a_clean_undecidable_report():
    assert check_report(_report()) == []


def test_check_report_blocks_a_guess_recorded_as_a_decision():
    problems = check_report(_report(verdict=VERDICT_INVALID))
    assert any("guess" in p for p in problems)


def test_check_report_allows_a_decision_when_the_basis_exists():
    assert check_report(
        _report(verdict=VERDICT_INVALID, head_basis="head_stratify")) == []


def test_check_report_catches_a_failed_gold_reconstruction():
    problems = check_report(_report(match=False))
    assert any("reconstruction mismatch" in p for p in problems)


def test_check_report_catches_a_failed_positive_control():
    """양성 대조가 깨지면 낮은 겹침은 고장과 구별되지 않는다."""
    problems = check_report(_report(control_exact=3))
    assert any("positive control failed" in p for p in problems)


def test_check_report_refuses_a_report_that_skipped_the_fingerprint_check():
    """지문 대조를 안 돌린 리포트는 복원이 맞았다고 주장할 수 없다."""
    report = _report()
    del report["gold_reconstruction"]
    problems = check_report(report)
    assert any("never verified" in p for p in problems)


def test_check_report_refuses_a_report_that_skipped_the_positive_control():
    """대조 없이는 낮은 겹침이 '어긋남' 인지 '고장' 인지 갈리지 않는다."""
    report = _report()
    del report["positive_control"]
    problems = check_report(report)
    assert any("positive control was never run" in p for p in problems)


def test_check_report_requires_a_basis_for_the_base_arm():
    """근거 없이 적은 설정은 기록이 아니라 가정이다.

    base 는 무조건 요구한다 — 문서로 확정되는 팔이다. head 는 근거 부재가 이
    감사의 전제라 위반이 아니고, 시나리오가 갈리는데 판정을 실을 때만 걸린다.
    """
    report = _report()
    report["arms"]["base"]["basis"] = None
    problems = check_report(report)
    assert any("base arm split setting" in p for p in problems)
