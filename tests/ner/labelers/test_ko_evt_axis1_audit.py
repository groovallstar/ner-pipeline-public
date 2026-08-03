"""KO EVT 축1-복합 head 열거·자리 스캔 테스트.

핵심은 **후보 집합을 저자가 고르지 않는다**는 것이다 — 열거가 gold 에서 결정적으로
나오지 않으면 표는 엄밀해 보이는데 여집합이 검토되지 않고, head 를 좁힐수록 그것을
분모로 쓰는 감사가 쉬워진다. 그래서 여기 테스트는 "무엇이 후보가 되나" 의 경계를
고정한다.
"""

import json
import pathlib

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
    assign_codes,
    evt_span_words,
    heads_sha256,
    parse_canonical_axis1,
    find_axis1_sites,
    proper_noun_lexicon,
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
    rows = [
        _row(1, "보스턴 테러 가 났다", [("LOC", 0, 3), ("EVT", 0, 6)]),
        _row(2, "보스턴 테러 참사 추모", [("EVT", 0, 11)]),
        _row(3, "보스턴 테러 이후", []),
    ]
    kinds = {s.row_index: s.gold_evt_overlap
             for s in find_axis1_sites(rows, ["테러"])}
    assert kinds[0] == "exact"
    assert kinds[1] == "inside_longer"
    assert kinds[2] == "none"


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
