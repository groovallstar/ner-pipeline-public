"""KO EVT R2 감사·회수 하네스 테스트.

감사와 적용이 같은 판정을 쓰는지가 핵심이다 — 둘이 갈리면 감사가 통과시킨 gold 를
적용이 다르게 해석해 반영률이 조용히 안 오른다.
"""

import pytest

from ner.labelers.ko.ko_evt_r2_audit import (
    apply_decisions,
    find_bare_asymmetry,
    find_unlabeled_mentions,
    find_violations,
    followed_by_other_head,
    is_standalone_mention,
    run_audit,
    verify_invariants,
)


def _row(row_id, text, ents):
    return {
        "id": row_id,
        "text": text,
        "entities": [
            {"label": lab, "start_char": s, "end_char": e, "text": text[s:e]}
            for lab, s, e in ents
        ],
    }


# ── 형태소 경계 ────────────────────────────────────────────────────────


@pytest.mark.parametrize("text,surface,expected", [
    ("긴급 기자회견을 열었다", "기자회견", True),        # 조사 뒤
    ("기자회견 직후 배포됐다", "기자회견", True),         # 공백 뒤
    ("기자회견장에서 만났다", "기자회견", False),         # 파생 접미 '장'
    ("기자회견문을 냈다", "기자회견", False),            # 파생 접미 '문'
    ("제작발표회에 참석했다", "발표회", False),           # 앞이 한글 = 복합어
    ("시상식으로 꼽힌다", "시상식", True),               # '으로' 는 조사
    ("시상식인 이 행사는", "시상식", True),              # '인' 은 서술격
    ("2002년 한<일:;LC>월드컵에서", "월드컵", False),     # KLUE 마크업이 낱말을 가름
    ("영화 <광해>에서 시상식이 열렸다", "시상식", True),   # 제목 꺾쇠는 마크업 아님
])
def test_is_standalone_mention(text, surface, expected):
    assert is_standalone_mention(text, text.index(surface), surface) is expected


def test_followed_by_other_head():
    text = "올림픽 공원에서 열렸다"
    assert followed_by_other_head(text, len("올림픽")) == "공원"
    assert followed_by_other_head("올림픽을 앞두고", len("올림픽")) is None


# ── 회수 후보 ──────────────────────────────────────────────────────────


def test_find_unlabeled_mentions_skips_labeled_and_r1():
    rows = [
        _row("a", "박 대통령은 긴급 기자회견을 열었다", [("PER", 0, 1)]),
        _row("b", "올림픽 공원에서 만났다", []),          # R1 — 다른 개체 이름 일부
        _row("c", "기자회견을 마쳤다", [("EVT", 0, 4)]),   # 이미 라벨됨
        _row("d", "기자회견장에서 만났다", []),            # 파생 명사
    ]
    found = find_unlabeled_mentions(rows, ["기자회견", "올림픽"])
    assert [(c.row_id, c.surface) for c in found] == [("a", "기자회견")]


def test_run_audit_excludes_out_of_scope_procedure_by_default():
    rows = [
        _row("a", "사전투표를 실시했다", []),
        _row("b", "사전투표가 시작됐다", [("EVT", 0, 4)]),
    ]
    scoped = run_audit(rows)
    assert scoped.bucket_totals.get("B2b_proc") == 1
    assert [c.surface for c in scoped.candidates] == []

    widened = run_audit(rows, include_proc=True)
    assert [c.surface for c in widened.candidates] == ["사전투표"]


def test_run_audit_coverage_excludes_already_listed_forms():
    """R2 반영률 분모에 canonical 축2 등재분(`국정조사`)이 섞이면 안 된다."""
    rows = [
        _row("a", "국정조사를 마쳤다", [("EVT", 0, 4)]),
        _row("b", "기자회견을 열었다", []),
    ]
    rep = run_audit(rows)
    assert rep.coverage["R2"] == {"labeled": 0, "missing": 1, "rate": 0.0}
    assert rep.coverage["axis2_listed"]["labeled"] == 1


# ── 정합성 위반 ────────────────────────────────────────────────────────


def test_find_violations_axis2_generic_and_r3_stage():
    rows = [
        _row("a", "테러가 발생했다", [("EVT", 0, 2)]),
        _row("b", "결승전이 열렸다", [("EVT", 0, 3)]),
    ]
    kinds = {v.kind for v in find_violations(rows)}
    assert kinds == {"A1_axis2_generic", "A2_r3_stage"}
    assert all(v.fix == "drop" for v in find_violations(rows))


def test_find_violations_r1_other_entity():
    rows = [_row("a", "아시안게임 경기장을 지었다", [("EVT", 0, 5)])]
    vio, = find_violations(rows)
    assert vio.kind == "A4_r1_other_entity" and vio.fix == "drop"


def test_find_violations_axis3_trim_and_overlap_guard():
    # `브라질 월드컵` 이 EVT 어휘에 있으므로 같은 문장의 `월드컵` 은 트림이다
    rows = [
        _row("a", "브라질 월드컵이 열렸다", [("EVT", 0, 7)]),
        _row("b", "브라질 월드컵을 봤다", [("EVT", 4, 7)]),
    ]
    vio = [v for v in find_violations(rows) if v.kind == "A3_axis3_trim"]
    assert len(vio) == 1 and vio[0].replacement == (0, 7)

    # 확장이 라벨된 다른 타입을 삼키면 겹침 정책상 기각된다
    rows_clash = [
        _row("a", "브라질 월드컵이 열렸다", [("EVT", 0, 7)]),
        _row("b", "브라질 월드컵을 봤다", [("LOC", 0, 3), ("EVT", 4, 7)]),
    ]
    assert [v for v in find_violations(rows_clash) if v.kind == "A3_axis3_trim"] == []


# ── 적용 ───────────────────────────────────────────────────────────────


def test_apply_inserts_consensus_and_skips_hold():
    rows = [_row("a", "긴급 기자회견을 열었다", []),
            _row("b", "시상식을 열었다", [])]
    decisions = [
        {"row_index": 0, "start": 3, "end": 7, "surface": "기자회견", "verdict": "EVT"},
        {"row_index": 1, "start": 0, "end": 3, "surface": "시상식", "verdict": "HOLD"},
    ]
    new_rows, stats = apply_decisions(rows, decisions, [])
    assert stats["recovered"] == 1 and stats["skipped_hold"] == 1
    assert new_rows[0]["entities"][0] == {
        "label": "EVT", "start_char": 3, "end_char": 7, "text": "기자회견"}
    assert new_rows[1]["entities"] == []


def test_apply_drops_and_expands_violations():
    # 확장 대상은 EVT 어휘에 있어야 인식된다 — row "a" 가 `브라질 월드컵` 을 공급한다
    rows = [
        _row("a", "브라질 월드컵이 열렸다", [("EVT", 0, 7)]),
        _row("b", "브라질 월드컵을 봤다", [("EVT", 4, 7)]),
        _row("c", "테러가 발생했다", [("EVT", 0, 2)]),
    ]
    new_rows, stats = apply_decisions(rows, [], find_violations(rows))
    assert stats["violation_dropped"] == 1 and stats["violation_expanded"] == 1
    assert new_rows[2]["entities"] == []
    span, = new_rows[1]["entities"]
    assert (span["start_char"], span["end_char"], span["text"]) == (0, 7, "브라질 월드컵")


def test_apply_maps_by_row_id_not_position():
    """gold 행 순서가 달라져도 판정이 원래 문장에 붙어야 한다."""
    rows = [_row("z", "시상식을 열었다", []),
            _row("a", "긴급 기자회견을 열었다", [])]      # 판정 당시엔 index 0 이었다
    decisions = [{"row_index": 0, "row_id": "a", "start": 3, "end": 7,
                  "surface": "기자회견", "verdict": "EVT"}]
    new_rows, stats = apply_decisions(rows, decisions, [])
    assert stats["row_index_drift"] == 1 and stats["recovered"] == 1
    assert new_rows[0]["entities"] == []
    assert new_rows[1]["entities"][0]["text"] == "기자회견"


def test_apply_skips_decision_whose_surface_moved():
    rows = [_row("a", "완전히 다른 문장이다", [])]
    decisions = [{"row_index": 0, "row_id": "a", "start": 3, "end": 7,
                  "surface": "기자회견", "verdict": "EVT"}]
    new_rows, stats = apply_decisions(rows, decisions, [])
    assert stats["surface_mismatch_skipped"] == 1
    assert new_rows[0]["entities"] == []


def test_apply_refuses_to_clash_with_existing_span():
    rows = [_row("a", "기자회견을 열었다", [("PER", 0, 2)])]
    decisions = [{"row_index": 0, "start": 0, "end": 4, "surface": "기자회견",
                  "verdict": "EVT"}]
    new_rows, stats = apply_decisions(rows, decisions, [])
    assert stats["insert_clash"] == 1 and "recovered" not in stats
    assert [e["label"] for e in new_rows[0]["entities"]] == ["PER"]


# ── bare 비대칭 (보고 전용) ────────────────────────────────────────────


def test_find_bare_asymmetry_reports_modifier_spans_only():
    rows = [
        _row("a", "아카데미 시상식이 열렸다", [("EVT", 0, 8)]),      # 수식어 있음 → 보고
        _row("b", "인사청문회가 열렸다", [("EVT", 0, 5)]),           # 한 낱말 → 제외
        _row("c", "시상식이 열렸다", [("EVT", 0, 3)]),               # bare → 제외
    ]
    found = find_bare_asymmetry(rows)
    assert [(f["span"], f["modifier"]) for f in found] == [("아카데미 시상식", "아카데미")]


def test_bare_asymmetry_is_not_a_gate():
    """보고 전용이라 위반 카운트·게이트에 섞이면 안 된다."""
    rows = [_row("a", "아카데미 시상식이 열렸다", [("EVT", 0, 8)])]
    rep = run_audit(rows)
    assert len(rep.bare_asymmetry) == 1
    assert rep.violations == []


def test_judge_rubric_lists_every_r2_form():
    """판정 rubric 이 canonical R2 표와 어긋나면 그 어긋남이 gold 에 굳는다."""
    from ner.labelers.ko.ko_evt_r2_audit import JUDGE_TEMPLATE, R2_FORMS

    assert [f for f in R2_FORMS if f not in JUDGE_TEMPLATE] == []


# ── 불변식 ─────────────────────────────────────────────────────────────


def test_verify_invariants_detects_frozen_type_change():
    before = [_row("a", "박씨가 왔다", [("PER", 0, 1)])]
    after = [_row("a", "박씨가 왔다", [("LOC", 0, 1)])]
    assert verify_invariants(before, after)["frozen_types_identical"] is False


def test_verify_invariants_passes_on_evt_only_change():
    before = [_row("a", "기자회견을 열었다", [("PER", 6, 8)])]
    after = _row("a", "기자회견을 열었다", [("EVT", 0, 4), ("PER", 6, 8)])
    checks = verify_invariants(before, [after])
    assert checks["frozen_types_identical"] is True
    assert checks["span_text_mismatch"] == 0 and checks["entity_overlap"] == 0
    assert (checks["evt_before"], checks["evt_after"]) == (0, 1)
