"""KO EVT 축1-복합 head 열거·자리 스캔 테스트.

핵심은 **후보 집합을 저자가 고르지 않는다**는 것이다 — 열거가 gold 에서 결정적으로
나오지 않으면 표는 엄밀해 보이는데 여집합이 검토되지 않고, head 를 좁힐수록 그것을
분모로 쓰는 감사가 쉬워진다. 그래서 여기 테스트는 "무엇이 후보가 되나" 의 경계를
고정한다.
"""

import collections
import hashlib
import json
import pathlib

import pytest

from ner.labelers.ko.ko_evt_axis1_audit import (
    AXIS1_EXCLUDE,
    AXIS1_HEADS,
    CODE_ADOPTED,
    DEFAULT_MIN_PROPER_LEN,
    DEFAULT_MIN_PROPER_RATIO,
    PROPER_LABELS,
    CODE_COVERED,
    HeadCandidate,
    MIN_HEAD_LEN,
    apply_axis1_decisions,
    assign_codes,
    check_axis1_gate,
    diagnose_fp_blindspot,
    evt_span_words,
    find_boundary_residue,
    ledger_gold_sha,
    heads_sha256,
    parse_canonical_axis1,
    find_axis1_sites,
    proper_noun_lexicon,
    rescore_arms,
    sites_report,
    survey_head_candidates,
    survey_report,
)


def _cand(head, covered_by=(), span=3, last=3):
    return HeadCandidate(head=head, source="어절", span_count=span,
                         last_count=last, productive_types=1,
                         homomorph_types=0, evt_history_types=0,
                         covered_by=tuple(covered_by))


def _row(row_id, text, ents):
    return {
        "id": row_id,
        "text": text,
        "entities": [
            {"label": lab, "start_char": s, "end_char": e, "text": text[s:e]}
            for lab, s, e in ents
        ],
    }


def _evt_row(row_id, text, *spans):
    ents = []
    for surface in spans:
        start = text.index(surface)
        ents.append(("EVT", start, start + len(surface)))
    return _row(row_id, text, ents)


def test_span_words_counts_heads_buried_mid_span():
    """마지막 어절만 세면 더 긴 span 에 묻힌 head 가 존재 자체를 감춘다."""
    rows = [_evt_row(1, "보스턴 마라톤 대회 폭발 사건 이후", "보스턴 마라톤 대회 폭발 사건")]
    words = evt_span_words(rows)
    assert words["마라톤"] == {"span": 1, "last": 0}
    assert words["사건"] == {"span": 1, "last": 1}


def test_span_words_ignores_other_types():
    rows = [_row(1, "서울 시청 앞", [("LOC", 0, 2), ("ORG", 3, 5)])]
    assert evt_span_words(rows) == {}


def test_single_syllable_eojeol_is_not_a_head():
    """1 글자는 형태 매칭이 어휘 경계를 식별하지 못한다."""
    assert MIN_HEAD_LEN == 2
    rows = [_evt_row(1, "왕자의 난 이 있었다", "왕자의 난")]
    assert "난" not in evt_span_words(rows)


def test_min_count_threshold_is_recorded_and_binding():
    rows = [
        _evt_row(1, "세월호 참사 추모", "세월호 참사"),
        _evt_row(2, "대구 참사 기억", "대구 참사"),
    ]
    two = {c.head for c in survey_head_candidates(rows, min_count=2)}
    three = {c.head for c in survey_head_candidates(rows, min_count=3)}
    assert "참사" in two
    assert "참사" not in three
    assert survey_report(rows, min_count=2)["params"]["min_count"] == 2


def test_proper_noun_history_filters_the_modifier_not_the_head():
    """같은 span 의 수식부는 고유명 이력으로 빠지고 head 는 남는다.

    이 필터가 없으면 후보에 고유명이 절반 섞여 표를 읽을 수 없다.
    """
    rows = [
        _row(1, "보스턴 은 도시 다", [("LOC", 0, 3)]),
        _evt_row(2, "보스턴 마라톤 소식", "보스턴 마라톤"),
        _evt_row(3, "보스턴 마라톤 재개", "보스턴 마라톤"),
        _evt_row(4, "런던 마라톤 중계", "런던 마라톤"),
    ]
    heads = {c.head for c in survey_head_candidates(rows, min_count=3)}
    assert "마라톤" in heads
    assert "보스턴" not in heads


def test_candidate_must_be_a_standalone_corpus_noun():
    """코퍼스에 단독 명사로 안 나오는 형태는 head 가 아니다."""
    rows = [
        _evt_row(1, "소치올림픽 개최", "소치올림픽"),
        _evt_row(2, "소치올림픽 중계", "소치올림픽"),
        _evt_row(3, "소치올림픽 폐막", "소치올림픽"),
    ]
    heads = {c.head for c in survey_head_candidates(rows, min_count=3)}
    assert heads == {"소치올림픽"}          # 코퍼스 어절로 존재하는 것만
    assert "올림픽" not in heads            # 어절로 등장한 적이 없다


def test_covered_by_marks_redundant_longer_head():
    """더 짧은 후보가 이미 덮으면 별도 등재가 필요 없다."""
    rows = [
        _evt_row(1, "세월호 침몰사고 조사", "세월호 침몰사고"),
        _evt_row(2, "침몰사고 원인", "침몰사고"),
        _evt_row(3, "침몰사고 후속", "침몰사고"),
        _evt_row(4, "원전 사고 대응", "원전 사고"),
        _evt_row(5, "열차 사고 수습", "열차 사고"),
        _evt_row(6, "화재 사고 현장", "화재 사고"),
    ]
    by_head = {c.head: c for c in survey_head_candidates(rows, min_count=3)}
    assert by_head["침몰사고"].covered_by == ("사고",)
    assert by_head["사고"].covered_by == ()


def test_homomorph_evidence_separates_lexical_boundary():
    """동형 명사와 그중 EVT 이력을 함께 낸다 — 어휘경계 판정의 근거다.

    비율만으로는 어휘경계와 미회수가 안 갈리므로(모듈 docstring) 목록을 함께 낸다.
    """
    rows = [
        _evt_row(1, "외환위기 이후 위기 였다", "위기"),
        _evt_row(2, "금융 위기 여파", "위기"),
        _evt_row(3, "재정 위기 우려", "위기"),
        _row(4, "분위기 가 좋다 멸종위기 종 보호", []),
    ]
    cand = {c.head: c for c in survey_head_candidates(rows, min_count=3)}["위기"]
    assert "분위기" in cand.homomorph_samples
    assert cand.evt_history_types < cand.homomorph_types


def test_other_label_types_surface_type_clash():
    """규칙이 EVT 라 할 자리를 gold 가 다른 타입으로 쓰면 그것도 근거다."""
    rows = [
        _evt_row(1, "한국 전쟁 발발", "전쟁"),
        _evt_row(2, "걸프 전쟁 당시", "전쟁"),
        _evt_row(3, "남북 전쟁 이후", "전쟁"),
        _row(4, "사랑과전쟁 재방송", [("PROD", 0, 5)]),
    ]
    cand = {c.head: c for c in survey_head_candidates(rows, min_count=3)}["전쟁"]
    assert any("사랑과전쟁" in item for item in cand.other_label_types)


def test_report_records_parameters_that_change_the_table():
    """문턱을 바꾸면 표 전체가 바뀐다 — 값만 커밋하면 재현이 안 된다."""
    rows = [_evt_row(1, "용산참사 를 기억한다", "용산참사")]
    report = survey_report(rows, min_count=1)
    assert report["params"]["min_head_len"] == MIN_HEAD_LEN
    assert report["params"]["proper_labels"] == ["PER", "LOC", "ORG", "PROD"]
    assert report["evt_spans"] == 1


def test_global_lexicon_drops_short_and_low_ratio_surfaces():
    """1 글자 성씨와 저비율 표면형은 고유명 근거가 못 된다."""
    rows = [
        _row(1, "이 대통령 은 말했다", [("PER", 0, 1)]),
        _row(2, "이 사건 은 크다", []),
        _row(3, "이 자리 에서", []),
        _row(4, "보스턴 에서", [("LOC", 0, 3)]),
    ]
    lex = proper_noun_lexicon(rows, min_len=2, min_ratio=0.3)
    assert "이" not in lex
    assert "보스턴" in lex


def test_site_takes_bare_preceding_proper_noun():
    rows = [
        _row(1, "보스턴 테러 가 있었다", [("LOC", 0, 3)]),
    ]
    sites = find_axis1_sites(rows, ["테러"])
    assert [(s.surface, s.attachment, s.proper_basis) for s in sites] == [
        ("보스턴 테러", "선행어절", "동행라벨")
    ]


def test_site_rejects_preceding_proper_noun_with_particle():
    """`파리에서 테러` 는 복합명사가 아니라 문장 성분이다."""
    rows = [_row(1, "파리에서 테러 가 났다", [("LOC", 0, 2)])]
    assert find_axis1_sites(rows, ["테러"]) == []


def test_site_takes_word_internal_proper_noun():
    rows = [
        _row(1, "서울 은 크다", [("LOC", 0, 2)]),
        _row(2, "서울월드컵 이 열렸다", []),
    ]
    sites = find_axis1_sites(rows, ["월드컵"])
    assert [(s.surface, s.attachment) for s in sites] == [("서울월드컵", "어절내")]


def test_site_uses_global_history_when_row_has_no_label():
    """그 행이 고유명을 안 달았을 뿐인 자리가 빠지면 대표 사례를 놓친다."""
    rows = [
        _row(1, "후쿠시마 는 일본 이다", [("LOC", 0, 4)]),
        _row(2, "후쿠시마 사고 이후", []),
    ]
    sites = find_axis1_sites(rows, ["사고"])
    assert [(s.surface, s.proper_basis) for s in sites] == [
        ("후쿠시마 사고", "전역이력")
    ]


def test_gold_evt_overlap_is_measured_not_dropped():
    """겹친 자리를 버리면 흡수량이 안 세어져 회수량과 구별되지 않는다."""
    # 평면 BIO 라 EVT 와 고유명은 겹칠 수 없다 — 겹친 픽스처는 gold 에 없는 상태다
    rows = [
        _row(1, "보스턴 은 도시 다", [("LOC", 0, 3)]),
        _row(2, "보스턴 에 갔다", [("LOC", 0, 3)]),
        _row(3, "보스턴 테러 가 났다", [("EVT", 0, 6)]),
        _row(4, "보스턴 테러 참사 추모", [("EVT", 0, 11)]),
        _row(5, "보스턴 테러 이후", []),
    ]
    kinds = {s.row_index: s.gold_evt_overlap
             for s in find_axis1_sites(rows, ["테러"])}
    assert kinds[2] == "exact"
    assert kinds[3] == "inside_longer"
    assert kinds[4] == "none"


def test_longest_head_wins():
    """`민주화운동` 이 있는데 `운동` 으로 잘라 잡으면 경계가 짧아진다."""
    rows = [
        _row(1, "광주 는 도시 다", [("LOC", 0, 2)]),
        _row(2, "광주민주화운동 기념", []),
    ]
    sites = find_axis1_sites(rows, ["운동", "민주화운동"])
    assert [(s.surface, s.head) for s in sites] == [("광주민주화운동", "민주화운동")]


def test_sites_report_records_population_parameters():
    rows = [_row(1, "보스턴 테러 가 났다", [("LOC", 0, 3)])]
    report = sites_report(rows, ["테러"], min_len=2, min_ratio=0.3)
    assert report["params"]["heads"] == ["테러"]
    assert report["params"]["min_proper_ratio"] == 0.3
    assert report["population"] == 1


def test_suffix_generalization_catches_scattered_compounds():
    """`한국전쟁`·`걸프전쟁` 이 따로 세어지면 공통 head 가 문턱 아래로 흩어진다."""
    rows = [
        _evt_row(1, "한국전쟁 이후", "한국전쟁"),
        _evt_row(2, "걸프전쟁 당시", "걸프전쟁"),
        _evt_row(3, "남북전쟁 기록", "남북전쟁"),
        _row(4, "전쟁 은 참혹하다", []),
    ]
    by_head = {c.head: c for c in survey_head_candidates(rows, min_count=3,
                                                        min_types=3)}
    assert by_head["전쟁"].source == "접미"
    assert by_head["전쟁"].span_count == 0        # 어절로는 문턱 미달
    assert by_head["전쟁"].productive_types == 3


def test_stray_proper_label_does_not_kill_a_head():
    """작품명으로 한 번 쓰인 형태까지 죽이면 진짜 head 가 사라진다.

    고유명 판정은 자리 스캔과 **같은 어휘**(길이·비율 가드)를 쓴다 — 자가 둘이 되면
    한쪽이 통과시킨 것을 다른 쪽이 못 본다.
    """
    rows = [
        _evt_row(1, "한국전쟁 이후", "한국전쟁"),
        _evt_row(2, "걸프전쟁 당시", "걸프전쟁"),
        _evt_row(3, "남북전쟁 기록", "남북전쟁"),
        _row(4, "전쟁 재방송", [("PROD", 0, 2)]),
        _row(5, "전쟁 은 참혹하다 전쟁 이 끝났다 전쟁 기록", []),
    ]
    assert "전쟁" in {c.head for c in survey_head_candidates(rows, min_count=3)}


def test_proper_noun_is_excluded_even_when_frequent_in_spans():
    rows = [
        _row(1, "세월호 는 배 다", [("PROD", 0, 3)]),
        _row(2, "세월호 가 인양됐다", [("PROD", 0, 3)]),
        _evt_row(3, "세월호 참사 추모", "세월호 참사"),
        _evt_row(4, "세월호 참사 기억", "세월호 참사"),
        _evt_row(5, "세월호 참사 진상", "세월호 참사"),
    ]
    heads = {c.head for c in survey_head_candidates(rows, min_count=3)}
    assert "참사" in heads
    assert "세월호" not in heads


def test_covered_by_adopted_head_gets_covered_code():
    """채택된 짧은 head 가 덮으면 긴 형태는 별도 등재가 필요 없다."""
    cands = [
        _cand("사고", covered_by=()),
        _cand("침몰사고", covered_by=("사고",)),
    ]
    code = assign_codes(cands)
    assert code["사고"] == CODE_ADOPTED
    assert code["침몰사고"] == CODE_COVERED


def test_covered_by_excluded_head_inherits_that_clause():
    """`여론조사` 는 `조사` 와 같은 축2 소관이다 — 상속을 안 하면 코드 없이 남는다."""
    cands = [_cand("조사"), _cand("여론조사", covered_by=("조사",))]
    code = assign_codes(cands)
    assert code["조사"] == "타규칙(축2)"
    assert code["여론조사"] == "타규칙(축2)"


def test_inheritance_walks_more_than_one_step():
    cands = [
        _cand("선거"),
        _cand("보궐선거", covered_by=("선거",)),
        _cand("재보궐선거", covered_by=("보궐선거", "선거")),
    ]
    assert assign_codes(cands)["재보궐선거"] == "타규칙(축2)"


def test_unassigned_candidate_is_reported_not_dropped():
    """코드가 없는 후보가 조용히 사라지면 '전부에 붙였다' 가 반증 불가능해진다."""
    cands = [_cand("사고"), _cand("듣도보도못한말")]
    code = assign_codes(cands)
    assert "듣도보도못한말" not in code


def test_committed_ledger_covers_every_candidate_and_matches_constants():
    """상수를 고치고 원장을 다시 안 만들면 둘이 어긋난다 — 그때 실패해야 한다."""
    path = (pathlib.Path(__file__).resolve().parents[3]
            / "src/ner/labelers/ko/data/evt_axis1_head_codes.json")
    ledger = json.loads(path.read_text(encoding="utf-8"))
    assert ledger["unassigned"] == []
    assert ledger["assigned"] == ledger["candidates"]
    adopted = {h for h, v in ledger["codes"].items() if v["code"] == CODE_ADOPTED}
    assert adopted == set(AXIS1_HEADS)
    assert all(v["code"] for v in ledger["codes"].values())


def test_canonical_head_list_is_the_single_source():
    """규칙은 잠긴 canonical 에 있고 이 모듈은 잠금 밖이다 — 어긋나면 실패해야 한다."""
    parsed = parse_canonical_axis1()
    assert tuple(parsed["heads"]) == AXIS1_HEADS


def test_canonical_population_guards_match_module_constants():
    """head 만 잠그면 모자란다 — 고유명 타입·가드 임계도 모집단을 바꾼다."""
    parsed = parse_canonical_axis1()
    assert tuple(parsed["proper_labels"]) == PROPER_LABELS
    assert parsed["min_proper_len"] == DEFAULT_MIN_PROPER_LEN
    assert parsed["min_proper_ratio"] == DEFAULT_MIN_PROPER_RATIO


def test_canonical_exclude_codes_match_module():
    """canonical 이 선언한 사유코드 집합과 모듈이 집행하는 집합이 같아야 한다.

    `상위head포함` 만 예외다 — canonical 이 코드로 선언하되 멤버를 열거하지 않고,
    모듈이 `covered_by` 에서 기계적으로 배정한다.
    """
    parsed = parse_canonical_axis1()
    assert set(parsed["exclude_codes"]) == set(AXIS1_EXCLUDE) | {CODE_COVERED}


def test_sites_report_fingerprints_head_list_and_gold():
    """모집단이 왜 달라졌는지가 값으로 갈려야 한다 — head 가 움직였나 gold 가 움직였나."""
    rows = [_row(1, "보스턴 테러 가 났다", [("LOC", 0, 3)])]
    one = sites_report(rows, ["테러"])
    two = sites_report(rows, ["테러", "사고"])
    assert one["params"]["heads_sha256"] != two["params"]["heads_sha256"]
    assert one["params"]["gold_sha256"] is None          # 경로를 안 주면 없다
    assert heads_sha256(["사고", "테러"]) == heads_sha256(["테러", "사고"])


def _gate_rows():
    return [
        _row(1, "보스턴 은 도시 다", [("LOC", 0, 3)]),
        _row(2, "보스턴 테러 가 났다", []),
        _row(3, "보스턴 테러 이후", []),
    ]


def test_gate_fails_while_a_site_is_unclassified():
    report = check_axis1_gate(_gate_rows(), ["테러"])
    assert len(report["unclassified"]) == 2   # 라벨 행에는 head 가 없다
    assert report["recovered"] == 0


def test_ledger_is_read_per_site_not_per_surface():
    """같은 표면형이 행마다 다른 판정을 받는다 — 접으면 한 행이 다른 행을 사면한다."""
    rows = _gate_rows()
    sites = find_axis1_sites(rows, ["테러"])
    first, second = sites[0], sites[1]
    ledger = [{"row_index": first.row_index, "start": first.start,
               "end": first.end, "verdict": "EVT"}]
    report = check_axis1_gate(rows, ["테러"], ledger)
    assert report["recovered"] == 1
    assert [u["row_index"] for u in report["unclassified"]] == [second.row_index]


def test_type_clash_gets_its_own_bucket_instead_of_a_forced_not():
    """삽입 불가한 충돌에 출구가 없으면 게이트 통과 경로가 NOT 하나뿐이 된다."""
    rows = [
        _row(1, "세월호 는 배 다", [("PROD", 0, 3)]),
        _row(2, "세월호 참사 가족대책위원회 가 모였다", [("ORG", 0, 14)]),
    ]
    report = check_axis1_gate(rows, ["참사"])
    assert report["status_counts"].get("type_clash_recorded") == 1
    assert report["unclassified"] == []
    assert report["type_clash_recorded"][0]["labels"] == ["ORG"]


def test_site_inside_longer_evt_is_absorbed_not_recovered():
    rows = [
        _row(1, "보스턴 은 도시 다", [("LOC", 0, 3)]),
        _row(2, "보스턴 테러 참사 추모", [("EVT", 0, 11)]),
    ]
    report = check_axis1_gate(rows, ["테러"])
    assert report["status_counts"].get("covered_by_longer_evt") == 1
    assert report["recovered"] == 0


def test_boundary_condition_trims_only_the_labeled_proper_noun():
    """라벨된 인접 고유명은 삼키지 않고, head 까지 덮이면 삽입 자체가 불가능하다."""
    rows = [
        _row(1, "보스턴 은 도시 다", [("LOC", 0, 3)]),
        _row(2, "미국 보스턴 마라톤 중계", [("LOC", 0, 6)]),
        _row(3, "보스턴 마라톤 중계", []),
    ]
    sites = {s.row_index: s for s in find_axis1_sites(rows, ["마라톤"])}
    assert rows[1]["text"][sites[1].insert_start:sites[1].insert_end] == "마라톤"
    assert sites[1].boundary_reason.startswith("head 만")
    assert rows[2]["text"][sites[2].insert_start:sites[2].insert_end] == "보스턴 마라톤"
    assert sites[2].boundary_reason.startswith("고유명 포함")


def _pred(row, *spans, gold=()):
    return {"id": row["id"],
            "gold_spans": [{"type": t, "start": s, "end": e} for t, s, e in gold],
            "pred_spans": [{"type": t, "start": s, "end": e} for t, s, e in spans]}


def test_fp_shape_detector_does_not_consult_the_head_list():
    """목록으로 모양을 판정하면 목록이 못 본 head 는 사각 계산에서부터 빠진다."""
    rows = _residue_rows([_row(3, "보스턴 패션쇼 취재", [])])
    assert "패션쇼" not in AXIS1_HEADS
    report = diagnose_fp_blindspot(
        rows, [_pred(rows[2], ("EVT", 0, 7))], AXIS1_HEADS)
    assert report["stats"]["compound_fp"] == 1
    assert report["stats"]["absent_from_population"] == 1
    assert report["absent_by_head"] == {"패션쇼": 1}
    assert report["items"][0]["proper"] == "보스턴"


def test_fp_inside_the_population_is_not_a_blind_spot():
    rows = _residue_rows([_row(3, "보스턴 마라톤 취재", [])])
    report = diagnose_fp_blindspot(
        rows, [_pred(rows[2], ("EVT", 0, 7))], ["마라톤"])
    assert report["stats"]["in_population"] == 1
    assert report["items"] == []


def test_fp_matching_gold_exactly_is_not_counted():
    rows = _residue_rows([_row(3, "보스턴 패션쇼 취재", [])])
    report = diagnose_fp_blindspot(
        rows, [_pred(rows[2], ("EVT", 0, 7), gold=[("EVT", 0, 7)])], AXIS1_HEADS)
    assert report["stats"].get("evt_fp", 0) == 0


def test_fp_without_a_proper_noun_is_not_compound_shaped():
    rows = _residue_rows([_row(3, "정기 총회 개최", [])])
    report = diagnose_fp_blindspot(
        rows, [_pred(rows[2], ("EVT", 0, 5))], AXIS1_HEADS)
    assert report["stats"]["evt_fp"] == 1
    assert report["stats"].get("compound_fp", 0) == 0


def test_fp_head_that_is_a_word_fragment_is_split_out():
    """경계가 낱말을 자르면 head 가 조각이 된다 — 사각의 이름이 아니라 경계 오류다."""
    rows = _residue_rows([_row(3, "보스턴 패션쇼장 취재", [])])
    report = diagnose_fp_blindspot(
        rows, [_pred(rows[2], ("EVT", 0, 7))], AXIS1_HEADS)
    assert report["items"][0]["head"] == "패션쇼"
    assert not report["items"][0]["head_is_corpus_noun"]
    assert report["absent_head_not_a_corpus_noun"] == 1


def test_committed_blindspot_report_prereqisters_the_inputs():
    """head 팔 학습 **전에** 무엇이 확정돼 있었나 — 나중에 넓히면 해시가 어긋난다."""
    base = pathlib.Path(__file__).resolve().parents[3] / "src/ner/labelers/ko/data"
    report = json.loads(
        (base / "evt_axis1_fp_blindspot.json").read_text(encoding="utf-8"))
    prov = json.loads((base / "evt_axis1_apply.json").read_text(encoding="utf-8"))
    prereg = report["prereg"]
    assert prereg["heads_sha256"] == heads_sha256(AXIS1_HEADS)
    assert prereg["gold_sha256"] == prov["gold_sha256"]["after"]
    canonical = pathlib.Path(__file__).resolve().parents[3] / \
        "docs/manual/data/canonical-entity-schema.md"
    digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
    assert prereg["canonical_sha256"] == digest
    # 진단은 게이트가 아니라 보고다 — 항목마다 사람이 읽을 근거가 있어야 한다
    assert all(i["context"] and i["surface"] for i in report["items"])
    assert len(report["items"]) == report["stats"]["absent_from_population"]


def _arm(*folds):
    """[(fold 이름, {row id: [span]})] — span 은 (type, start, end)."""
    return [(f"fold{n}", {str(i): [{"type": t, "start": s, "end": e}
                                   for t, s, e in spans]
                          for i, spans in fold.items()})
            for n, fold in enumerate(folds)]


def _score_rows():
    return [_row(1, "보스턴 마라톤 중계", [("EVT", 0, 7)]),
            _row(2, "런던 올림픽 개막", [("EVT", 0, 6)])]


def test_rescore_of_identical_arms_has_zero_delta():
    rows = _score_rows()
    arm = _arm({1: [("EVT", 0, 7)]}, {2: [("EVT", 0, 6)]})
    report = rescore_arms(rows, arm, arm)
    assert report["rows_scored"] == 2
    assert all(v["delta"] == 0.0 for v in report["per_entity"].values())
    assert report["paired_evt"]["wins"] == report["paired_evt"]["losses"] == 0


def test_rescore_skips_the_rows_the_recovery_touched():
    """회수가 답을 넣어 준 자리를 빼야 '그 밖에서도 나아졌나' 가 나온다."""
    rows = _score_rows()
    base = _arm({1: []}, {2: [("EVT", 0, 6)]})
    head = _arm({1: [("EVT", 0, 7)]}, {2: [("EVT", 0, 6)]})
    full = rescore_arms(rows, base, head)
    subset = rescore_arms(rows, base, head, skip_ids=["1"])
    assert full["per_entity"]["EVT"]["delta"] > 0
    assert subset["rows_scored"] == 1
    assert subset["per_entity"]["EVT"]["delta"] == 0.0


def test_rescore_refuses_arms_that_cover_different_rows():
    rows = _score_rows()
    base = _arm({1: [("EVT", 0, 7)]}, {2: [("EVT", 0, 6)]})
    head = _arm({1: [("EVT", 0, 7)]}, {})
    with pytest.raises(SystemExit):
        rescore_arms(rows, base, head)


def test_rescore_refuses_predictions_for_rows_gold_does_not_have():
    """조용히 빼면 분모가 줄어든 채 그럴듯한 F1 이 나온다."""
    rows = _score_rows()
    arm = _arm({1: [("EVT", 0, 7)]}, {99: []})
    with pytest.raises(SystemExit):
        rescore_arms(rows, arm, arm)


def test_rescore_withholds_paired_delta_when_the_split_moved():
    """분할이 안 맞으면 paired 를 내지 않는다 — 짝지어 계산하면 서로 다른 문장
    집합을 비교한 수가 그럴듯한 모습으로 나온다. pooled 는 그대로 유효하다."""
    rows = _score_rows()
    base = _arm({1: [("EVT", 0, 7)]}, {2: [("EVT", 0, 6)]})
    head = _arm({2: [("EVT", 0, 6)]}, {1: [("EVT", 0, 7)]})
    report = rescore_arms(rows, base, head)
    assert report["paired_evt"]["available"] is False
    assert report["paired_evt"]["mean_fold_overlap"] == 0.0
    assert "mean" not in report["paired_evt"]
    # pooled 는 fold 배정과 무관하다 — 두 팔이 같은 행을 한 번씩 덮으면 성립한다
    assert report["rows_scored"] == 2
    assert report["per_entity"]["EVT"]["delta"] == 0.0


def test_rescore_refuses_a_row_that_lands_in_two_folds():
    """한 행이 두 fold 에 있으면 pooled 에서 조용히 한 번만 세어진다."""
    rows = _score_rows()
    arm = _arm({1: [("EVT", 0, 7)], 2: []}, {2: [("EVT", 0, 6)]})
    with pytest.raises(SystemExit):
        rescore_arms(rows, arm, arm)


def test_rescore_counts_paired_fold_wins_and_losses():
    rows = _score_rows()
    base = _arm({1: []}, {2: [("EVT", 0, 6)]})
    head = _arm({1: [("EVT", 0, 7)]}, {2: []})
    paired = rescore_arms(rows, base, head)["paired_evt"]
    assert paired["wins"] == 1 and paired["losses"] == 1
    assert set(paired["per_fold"]) == {"fold0", "fold1"}


def test_certified_noncircular_is_anchored_to_the_preregistered_inputs():
    """승격된 비순환 표가 사전등록한 자로 쟀나 — 아니면 사전등록이 장식이 된다."""
    root = pathlib.Path(__file__).resolve().parents[3]
    base = root / "src/ner/labelers/ko/data"
    prereg = json.loads((base / "evt_axis1_prereg.json").read_text(encoding="utf-8"))
    report = json.loads((
        root / "certified/classifier/ko/issue202-axis1-head/noncircular.json"
    ).read_text(encoding="utf-8"))
    assert report["gold_sha256"] == prereg["head_arm"]["gold_sha256"]
    assert report["sigma_sha256"] == prereg["base_arm"]["fold_sigma_sha256"]
    preserved = prereg["base_arm"]["preserved_pred_spans_sha256"]
    assert report["pred_sha256"]["base"] == [preserved[f"fold{i}"] for i in range(10)]
    ledger = [json.loads(line) for line
              in (base / "evt_axis1_judgements.jsonl").read_text(
                  encoding="utf-8").splitlines() if line.strip()]
    assert report["recovered_sites"] == sum(
        1 for r in ledger if r["verdict"] == "EVT")
    # 분할이 안 맞아 못 낸 통계는 빠지는 게 아니라 못 낸 이유가 남아야 한다
    for block in ("full", "recovery_free_subset"):
        paired = report[block]["paired_evt"]
        assert paired["available"] or (paired["reason"] and "mean" not in paired)


def test_prereg_pins_every_input_that_could_move_after_the_numbers_land():
    """head 팔 학습 전 상태의 사전등록 — 나중에 규칙을 손대면 여기서 어긋난다.

    "커밋 diff 로 본다" 는 집행이 아니다. 실행 순서는 diff 에 남지 않고 `results/`
    는 휘발이라, 지문을 다시 계산해 대조하는 것만이 순서를 되짚을 수 있다.
    """
    root = pathlib.Path(__file__).resolve().parents[3]
    base = root / "src/ner/labelers/ko/data"
    prereg = json.loads((base / "evt_axis1_prereg.json").read_text(encoding="utf-8"))

    def sha(path):
        return hashlib.sha256((root / path).read_bytes()).hexdigest()

    assert prereg["rule"]["canonical_sha256"] == \
        sha("docs/manual/data/canonical-entity-schema.md")
    assert prereg["rule"]["heads_sha256"] == heads_sha256(AXIS1_HEADS)
    assert prereg["rule"]["judgements_sha256"] == \
        sha("src/ner/labelers/ko/data/evt_axis1_judgements.jsonl")
    # σ 는 base 팔에서 뽑아 승격했다 — 그 파일이 바뀌면 노이즈 밴드가 바뀐다
    arm = prereg["base_arm"]
    assert arm["fold_sigma_sha256"] == \
        sha("certified/classifier/ko/issue202-axis1-base/fold_sigma.json")
    assert arm["pooled_metrics_sha256"] == \
        sha("certified/classifier/ko/issue202-axis1-base/pooled_metrics.json")
    # base 예측을 잃으면 비순환 재채점이 불가능해진다 — 보존본을 지문으로 묶는다
    for fold, digest in arm["preserved_pred_spans_sha256"].items():
        assert digest == sha(
            f"preserved/classifier/ko/issue202-axis1-base/{fold}/pred_spans.json")
    # 두 팔은 서로 다른 gold 를 본다 — 같으면 회수가 반영되지 않았다는 뜻이다
    prov = json.loads((base / "evt_axis1_apply.json").read_text(encoding="utf-8"))
    assert arm["gold_sha256"] == prov["gold_sha256"]["before"]
    assert prereg["head_arm"]["gold_sha256"] == prov["gold_sha256"]["after"]


def _residue_rows(extra=()):
    """`보스턴` 을 전역 고유명 어휘에 올려 두는 최소 코퍼스."""
    return [
        _row(1, "보스턴 은 도시 다", [("LOC", 0, 3)]),
        _row(2, "보스턴 은 항구 다", [("LOC", 0, 3)]),
    ] + list(extra)


def test_residue_counts_a_span_that_left_an_unlabeled_proper_noun_out():
    rows = _residue_rows([_row(3, "보스턴 마라톤 중계", [("EVT", 4, 7)])])
    items = find_boundary_residue(rows)
    assert [(i["kind"], i["candidate"], i["expected"]) for i in items] == [
        ("과축소", "보스턴", "보스턴 마라톤")]


def test_residue_does_not_flag_a_labeled_adjacent_proper_noun():
    """라벨돼 있으면 안 삼킨 것이 규칙대로다 — 위반이 아니라 조건절의 다른 갈래다."""
    rows = _residue_rows([_row(3, "보스턴 마라톤 중계", [("LOC", 0, 3), ("EVT", 4, 7)])])
    assert find_boundary_residue(rows) == []


def test_residue_ignores_a_preceding_eojeol_with_a_particle():
    rows = _residue_rows([_row(3, "보스턴에서 마라톤 중계", [("EVT", 6, 9)])])
    assert find_boundary_residue(rows) == []


def test_residue_flags_a_swallowed_labeled_proper_noun():
    """평면 BIO 가 표현할 수 없는 상태라 실측 0 이 전제다 — 깨지면 삽입이 만든 것이다."""
    rows = _residue_rows([
        _row(3, "보스턴 마라톤 중계", [("LOC", 0, 3), ("EVT", 0, 7)])])
    items = find_boundary_residue(rows)
    assert [(i["kind"], i["labels"]) for i in items] == [("고유명삼킴", ["LOC"])]


def test_residue_population_is_every_evt_span_not_only_axis1_heads():
    """모집단을 축1 head 로 좁히면 규칙이 못 보는 곳이 분모 밖으로 빠진다."""
    rows = _residue_rows([_row(3, "보스턴 정상회담 중계", [("EVT", 4, 8)])])
    assert "정상회담" not in AXIS1_HEADS
    assert [i["expected"] for i in find_boundary_residue(rows)] == ["보스턴 정상회담"]


def test_residue_catches_a_word_internal_proper_noun_prefix():
    rows = _residue_rows([_row(3, "보스턴마라톤 중계", [("EVT", 3, 6)])])
    assert [i["expected"] for i in find_boundary_residue(rows)] == ["보스턴마라톤"]


def test_committed_residue_report_was_measured_on_the_applied_gold():
    """계수기가 회수 **뒤** gold 를 봤나 — 앞을 보면 오삽입이 분모에 안 들어온다."""
    base = pathlib.Path(__file__).resolve().parents[3] / "src/ner/labelers/ko/data"
    residue = json.loads((base / "evt_axis1_residue.json").read_text(encoding="utf-8"))
    prov = json.loads((base / "evt_axis1_apply.json").read_text(encoding="utf-8"))
    assert residue["params"]["gold_sha256"] == prov["gold_sha256"]["after"]
    assert residue["evt_spans"] == prov["checks"]["evt_after"]
    assert residue["by_kind"].get("고유명삼킴", 0) == 0
    # 보고 자체가 게이트다 — 0 을 요구하지 않되 항목마다 근거가 남아야 한다
    assert len(residue["items"]) == residue["residue"]
    assert all(i["surface"] and i["expected"] != i["surface"]
               for i in residue["items"] if i["kind"] == "과축소")


def _apply_rows():
    """`미국 보스턴 마라톤` 은 LOC 가 앞을 덮어 head 만, `보스턴 마라톤` 은 통째로."""
    return [
        _row(1, "보스턴 은 도시 다", [("LOC", 0, 3)]),
        _row(2, "미국 보스턴 마라톤 중계", [("LOC", 0, 6)]),
        _row(3, "보스턴 마라톤 중계", []),
    ]


def _decision(rows, site, verdict="EVT", **over):
    text = rows[site.row_index]["text"]
    rec = {
        "row_index": site.row_index, "row_id": rows[site.row_index]["id"],
        "start": site.start, "end": site.end, "surface": site.surface,
        "insert_start": site.insert_start, "insert_end": site.insert_end,
        "insert_surface": text[site.insert_start:site.insert_end],
        "verdict": verdict,
    }
    rec.update(over)
    return rec


def test_apply_inserts_the_judged_boundary_and_leaves_other_types_alone():
    rows = _apply_rows()
    sites = {s.row_index: s for s in find_axis1_sites(rows, ["마라톤"])}
    ledger = [_decision(rows, sites[1]), _decision(rows, sites[2])]
    new_rows, stats = apply_axis1_decisions(rows, ledger, ["마라톤"])
    assert stats["recovered"] == 2
    inserted = {r["text"][e["start_char"]:e["end_char"]]
                for r in new_rows for e in r["entities"] if e["label"] == "EVT"}
    assert inserted == {"마라톤", "보스턴 마라톤"}
    # 불변 타입은 한 글자도 안 움직인다 — LOC 두 개가 그대로 있어야 한다
    assert [e for r in new_rows for e in r["entities"] if e["label"] == "LOC"] == \
           [e for r in rows for e in r["entities"] if e["label"] == "LOC"]


def test_apply_skips_not_verdicts():
    rows = _apply_rows()
    sites = {s.row_index: s for s in find_axis1_sites(rows, ["마라톤"])}
    new_rows, stats = apply_axis1_decisions(
        rows, [_decision(rows, sites[2], verdict="NOT")], ["마라톤"])
    assert stats == {"skipped_not": 1}
    assert not [e for r in new_rows for e in r["entities"] if e["label"] == "EVT"]


def test_apply_refuses_a_ledger_whose_insert_boundary_drifted():
    """원장이 승인한 경계와 현 규칙이 어긋나면 삽입하지 않고 센다.

    원장만 믿으면 경계 규칙을 고쳐도 원장이 조용히 낡고, 현 규칙만 믿으면 사람이
    승인하지 않은 경계가 소리 없이 들어간다.
    """
    rows = _apply_rows()
    sites = {s.row_index: s for s in find_axis1_sites(rows, ["마라톤"])}
    stale = _decision(rows, sites[1], insert_start=3, insert_surface="보스턴 마라톤")
    new_rows, stats = apply_axis1_decisions(rows, [stale], ["마라톤"])
    assert stats == {"insert_span_drift": 1}
    assert not [e for r in new_rows for e in r["entities"] if e["label"] == "EVT"]


def test_apply_reports_a_ledger_site_the_current_scan_no_longer_produces():
    rows = _apply_rows()
    sites = {s.row_index: s for s in find_axis1_sites(rows, ["마라톤"])}
    dropped = _decision(rows, sites[2])
    # head 목록에서 `마라톤` 이 빠지면 그 자리는 더 이상 열거되지 않는다
    _, stats = apply_axis1_decisions(rows, [dropped], ["올림픽"])
    assert stats == {"site_missing": 1}


def test_apply_resolves_rows_by_id_not_position():
    """행 순서가 달라져도 판정이 엉뚱한 문장에 박히면 안 된다."""
    rows = _apply_rows()
    sites = {s.row_index: s for s in find_axis1_sites(rows, ["마라톤"])}
    decision = _decision(rows, sites[2])
    shifted = [rows[2], rows[0], rows[1]]     # row_id 3 이 0 번으로 온다
    new_rows, stats = apply_axis1_decisions(shifted, [decision], ["마라톤"])
    assert stats["row_index_drift"] == 1 and stats["recovered"] == 1
    evt = [(r["id"], r["text"][e["start_char"]:e["end_char"]])
           for r in new_rows for e in r["entities"] if e["label"] == "EVT"]
    assert evt == [(3, "보스턴 마라톤")]


def test_apply_never_inserts_over_an_existing_span():
    rows = _apply_rows()
    sites = {s.row_index: s for s in find_axis1_sites(rows, ["마라톤"])}
    decision = _decision(rows, sites[2])
    occupied = rows[:2] + [dict(rows[2], entities=[
        {"label": "PROD", "start_char": decision["insert_start"],
         "end_char": decision["insert_end"], "text": decision["insert_surface"]}])]
    _, stats = apply_axis1_decisions(occupied, [decision], ["마라톤"])
    assert stats == {"insert_clash": 1}


def test_ledger_gold_sha_reports_every_distinct_fingerprint():
    """서로 다른 gold 를 본 판정이 한 원장에 섞이면 드러나야 한다."""
    assert ledger_gold_sha([{"gold_sha256_before": "aa"},
                            {"gold_sha256_before": "aa"}]) == ("aa",)
    assert len(ledger_gold_sha([{"gold_sha256_before": "aa"},
                                {"gold_sha256_before": "bb"}])) == 2


def test_committed_apply_provenance_matches_the_ledger():
    """gold 는 버전 관리 밖이라 이 원장이 회수의 유일한 감사 흔적이다."""
    base = pathlib.Path(__file__).resolve().parents[3] / "src/ner/labelers/ko/data"
    prov = json.loads((base / "evt_axis1_apply.json").read_text(encoding="utf-8"))
    ledger = [json.loads(line) for line
              in (base / "evt_axis1_judgements.jsonl").read_text(
                  encoding="utf-8").splitlines() if line.strip()]
    verdicts = collections.Counter(r["verdict"] for r in ledger)
    assert prov["stats"]["recovered"] == verdicts["EVT"]
    assert prov["stats"]["skipped_not"] == verdicts["NOT"]
    assert not [k for k in ("site_missing", "insert_span_drift",
                            "surface_mismatch", "insert_clash")
                if prov["stats"].get(k)]
    assert prov["checks"]["frozen_types_identical"]
    assert prov["checks"]["span_text_mismatch"] == 0
    assert prov["checks"]["entity_overlap"] == 0
    assert prov["checks"]["evt_after"] - prov["checks"]["evt_before"] == verdicts["EVT"]
    # 판정 당시의 gold 에 적용됐나 — 전 지문이 원장의 것과 같아야 한다
    assert prov["gold_sha256"]["before"] == ledger_gold_sha(ledger)[0]
    assert prov["gold_sha256"]["after"] != prov["gold_sha256"]["before"]
    # 적용 뒤에도 자리가 전부 분류돼 있나
    assert prov["gate_after"]["unclassified"] == 0
    assert prov["heads_sha256"] == heads_sha256(AXIS1_HEADS)


def test_committed_gate_report_is_closed_and_matches_the_ledger():
    """커밋된 게이트 산출물이 원장과 어긋나면 실패해야 한다 — 둘 다 잠금 밖이다."""
    base = pathlib.Path(__file__).resolve().parents[3] / "src/ner/labelers/ko/data"
    report = json.loads((base / "evt_axis1_gate.json").read_text(encoding="utf-8"))
    ledger = [json.loads(line) for line
              in (base / "evt_axis1_judgements.jsonl").read_text(
                  encoding="utf-8").splitlines() if line.strip()]
    assert report["unclassified"] == []
    # 이 보고는 **판정 시점**(적용 전) gold 의 종결이다. 적용 뒤에는 회수 14 건이
    # `already_evt` 로 접혀 같은 명령이 다른 표를 낸다 — 어느 gold 를 잰 표인지
    # 지문으로 못 박아 두지 않으면 낡음과 어긋남이 구별되지 않는다.
    assert report["params"]["gold_sha256"] == ledger_gold_sha(ledger)[0]
    verdicts = collections.Counter(r["verdict"] for r in ledger)
    assert report["recovered"] == verdicts["EVT"]
    assert report["status_counts"]["judged_not"] == verdicts["NOT"]
    # 판정은 사람이 확정한다 — 초안 모델은 판정 권한이 없다
    assert {r["judged_by"] for r in ledger} == {"human"}
    # 자리마다 사유가 있어야 한다. 없으면 규칙 일괄 적용과 구별되지 않는다
    assert all(r["verdict_reason"].strip() for r in ledger)
    assert len({(r["row_index"], r["start"], r["end"]) for r in ledger}) == len(ledger)
