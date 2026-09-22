"""KO EVT R2 감사 하네스 테스트."""

import pytest

from ner.labelers.ko.ko_evt_r2_audit import (
    extract_noun_candidates,
    find_genitive_residue,
    span_evidence,
    R2_BARE_STANDALONE,
    R2_EXCLUDE,
    R2_HEADS,
    R2C_CEREMONY,
    check_homomorph,
    find_bare_asymmetry,
    find_head_homomorph,
    find_unlabeled_mentions,
    find_violations,
    followed_by_other_head,
    is_standalone_mention,
    parse_canonical_r2,
    r2_rule_of,
    run_audit,
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


# ── canonical ↔ 코드 동기 ──────────────────────────────────────────────


def test_module_constants_match_canonical_table():
    """R2 규칙의 정본은 canonical 표이고 모듈 상수는 사본이다.

    규칙은 기준 파일(바꾸면 사람 승인·반박자를 타는 곳)에 있는데 그 규칙을 세는
    감사 코드는 잠금 밖이다. 둘이 어긋나도 아무도 모르면, head 를 조용히 좁혀
    동형 게이트를 통과시킬 수 있다 — 자기 규칙을 자기 자로 재는 구조다.
    이 테스트가 그 경로를 막는다.
    """
    parsed = parse_canonical_r2()
    assert parsed["heads"] == R2_HEADS
    assert parsed["bare_standalone"] == R2_BARE_STANDALONE
    assert parsed["ceremony"] == R2C_CEREMONY
    assert parsed["exclude"] == R2_EXCLUDE


def test_canonical_parser_reads_heads_and_exclusions(tmp_path):
    """파서가 표의 어느 칸에서 무엇을 읽는지 고정한다."""
    doc = tmp_path / "canon.md"
    doc.write_text(
        "### 5.3 KO\n\n"
        "| 표면형 | 판정 | 근거 |\n|---|---|---|\n"
        "| 아래 head 로 끝나는 복합명사 (`국무회의`) | `EVT` | "
        "**R2 원칙** — 설명. **R2 head**: `회의`·`총회` |\n"
        "| `청문회`·`포럼` (단독) | `EVT` | R2 단독 — head 자체가 회의 형식 |\n"
        "| `위원회` | 비-entity | R2 제외 `상설조직` — 조직이다 |\n"
        "| `개막식` | `EVT` | R2c — 의식 |\n"
        "\n## 6. 다음 절\n",
        encoding="utf-8",
    )
    parsed = parse_canonical_r2(str(doc))
    # 표면형 칸의 괄호는 예시·한정어라 목록이 아니다 — `국무회의` 는 head 가 아니다
    assert parsed["heads"] == ("회의", "총회")
    assert parsed["bare_standalone"] == ("청문회", "포럼")
    assert parsed["exclude"] == {"상설조직": ("위원회",)}
    assert parsed["ceremony"] == ("개막식",)


def test_canonical_parser_raises_when_the_section_is_gone(tmp_path):
    """읽을 절이 없으면 빈 결과가 아니라 예외다.

    빈 결과는 "규칙이 실제로 비어 있다" 와 구별되지 않는다 — 그러면
    `canonical_rule_sha256()` 이 그 빈 규칙의 해시를 정상 값처럼 만들어 내고,
    파싱이 깨진 채 사전등록을 다시 뜨면 빈 규칙이 정본으로 굳는다.
    """
    doc = tmp_path / "canon.md"
    doc.write_text(
        "### 5.9 다른 절\n\n| 표면형 | 판정 | 근거 |\n|---|---|---|\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not found"):
        parse_canonical_r2(str(doc))


def test_canonical_parser_raises_when_the_table_is_gone(tmp_path):
    """절 제목만 남고 표가 사라진 경우도 같다 — 규칙을 못 읽은 것은 마찬가지다."""
    doc = tmp_path / "canon.md"
    doc.write_text(
        "### 5.3 KO\n\n산문만 남았다.\n\n## 6. 다음 절\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no table rows"):
        parse_canonical_r2(str(doc))


def test_canonical_parser_stops_at_the_next_sibling_section(tmp_path):
    """형제 절(`### 5.4`)이 생겨도 그 표의 행이 KO 규칙으로 섞이면 안 된다."""
    doc = tmp_path / "canon.md"
    doc.write_text(
        "### 5.3 KO\n\n"
        "| 표면형 | 판정 | 근거 |\n|---|---|---|\n"
        "| `청문회` (단독) | `EVT` | R2 단독 — head 자체가 회의 형식 |\n"
        "\n### 5.4 EN\n\n"
        "| 표면형 | 판정 | 근거 |\n|---|---|---|\n"
        "| `townhall` (단독) | `EVT` | R2 단독 — EN 절의 행 |\n",
        encoding="utf-8",
    )
    assert parse_canonical_r2(str(doc))["bare_standalone"] == ("청문회",)


def test_r2_rule_of_puts_exclusion_before_head_match():
    """제외가 형태 매칭에 덮이면 canonical 이 무력해진다.

    `군법회의` 는 `회의` head 로 끝나지만 사법 절차라 canonical 이 제외했다.
    순서가 뒤집히면 그 제외가 형태 매칭에 조용히 먹힌다.
    """
    assert r2_rule_of("군법회의") is None
    assert r2_rule_of("신체접촉") is None
    assert r2_rule_of("국무회의") == "R2원칙"
    assert r2_rule_of("청문회") == "R2단독"
    # head 자체 단독은 축2 generic 이라 R2 가 아니다
    assert r2_rule_of("회의") is None
    assert r2_rule_of("회담") is None


# ── head 동형 게이트 ───────────────────────────────────────────────────


def _homomorph_rows():
    return [
        _row("a", "국무회의가 열렸다", [("EVT", 0, 4)]),
        _row("b", "주교회의에서 결정했다", []),          # 동형·무라벨 → 미분류
        _row("c", "군법회의에서 선고했다", []),          # canonical 제외
        _row("d", "남북 당국회담이 열렸다", [("EVT", 0, 7)]),  # 겹침만
    ]


def test_noun_candidates_split_genitive_from_word_final_ui():
    """1 글자 조사 `의` 와 낱말 끝 `의` 를 가른다.

    무조건 떼면 `점검회의` 가 `점검회` 로 잘려 모집단에서 사라지고(누락 — 게이트가
    false-green 이 된다), 무조건 두면 `사회의`·`위원회의` 가 `회의` head 로 걸려
    사람이 판정할 목록만 부풀린다. 어간이 코퍼스에 이미 있으면 조사로 본다.
    """
    rows = [
        _row("a", "사회를 바꾸는 힘", []),        # 어간 `사회` 를 어휘에 넣는다
        _row("b", "우리 사회의 문제다", []),       # → 조사로 갈린다
        _row("c", "전날 밤 점검회의 결과를 설명했다", []),  # 어간 없음 → 낱말
    ]
    nouns = extract_noun_candidates(rows)
    assert nouns["사회"] >= 2 and nouns.get("사회의", 0) == 0
    assert nouns["점검회의"] == 1 and nouns.get("점검회", 0) == 0


def test_noun_candidates_survive_inline_quotes():
    """어절 안 인용부호 때문에 후보가 통째로 탈락하면 모집단이 조용히 준다."""
    rows = [_row("a", "각계 '원탁회의'는 오후에 열린다", [])]
    assert extract_noun_candidates(rows)["원탁회의"] == 1


def test_span_evidence_separates_compound_from_genitive():
    """`~의` 로 끝나는 span 은 직후가 갈라 준다.

    조사가 바로 붙으면 그 앞이 통째로 명사라 복합명사이고, 공백 뒤 명사구가 오면
    속격 독법이 열린다. 형태만으로는 `전체회의`(낱말)와 `귀족사회의`(속격)를
    구별할 수 없다.
    """
    t1 = "윤리특별위원회는 전체회의를 열어 논의했다"
    assert span_evidence(t1, t1.index("전체회의") + 4) == "조사직결"
    t2 = "영국 귀족사회의 완벽한 고증이 돋보인다"
    assert span_evidence(t2, t2.index("귀족사회의") + 5) == "없음"
    # 인용부호가 끼어도 조사를 봐야 한다 — 못 보면 근거 없음으로 밀려 목록이 부푼다
    t3 = "'우유 가치의 재발견을 위한 포럼'에서 발표했다"
    assert span_evidence(t3, t3.index("포럼") + 2) == "조사직결"


def test_span_evidence_rejects_incidental_verb_fragments():
    """개최 동사는 2 글자 이상 활용형으로 적어야 한다.

    짧은 대안을 두면 무관한 자리가 근거를 얻고, 그 자리는 속격 검사 대상에서
    빠진다 — 1 글자 `했` 이 "못했을" 을, bare `참석` 이 "참석자" 를 물었다.
    근거를 넉넉히 주는 실수는 조용히 통과시키는 쪽으로 실패한다.
    """
    assert span_evidence("추가회담 날짜도 잡지 못했을 가능성이", 4) == "없음"
    assert span_evidence("주교회의 참석자 3분의 2 이상의 찬성", 4) == "없음"
    # 진짜 개최 동사는 잡는다
    assert span_evidence("투자설명회 개최 불가 입장을 통보했다", 5) == "개최동사"
    assert span_evidence("간담회 열린 자리에서 밝혔다", 3) == "개최동사"


def test_genitive_residue_watches_what_the_gate_cannot_see():
    """동형 게이트는 gold 에 **들어간** 오류를 못 본다 — 그래서 따로 센다.

    속격 구성이 EVT 로 삽입되면 그 표면형은 gold EVT 이력을 얻고, 모집단은
    'EVT 이력 없는 표면형' 이라 그 순간 영구히 빠진다. 게이트를 몇 번 돌려도
    미분류 0 이 유지되므로 삽입된 쪽을 세지 않으면 아무도 못 본다.
    """
    rows = [_row("a", "영국 귀족사회의 완벽한 고증", [("EVT", 3, 8)]),
            _row("b", "위원회는 전체회의를 열었다", [("EVT", 5, 9)])]
    # 게이트는 이 오염을 못 본다 — 이미 EVT 이력이 생겨 모집단 밖이다
    assert "귀족사회의" not in {i["surface"]
                            for i in check_homomorph(rows, [])["items"]}
    # 속격 잔여 검사가 잡는다. 근거가 있는 `전체회의` 는 조용히 지나간다
    found = find_genitive_residue(rows)
    assert [g["surface"] for g in found] == ["귀족사회의"]


def test_genitive_residue_scopes_to_this_recovery():
    """gold 에 원래 있던 span 까지 세면 이번에 들어간 잔여가 묻힌다."""
    rows = [_row("a", "영국 귀족사회의 완벽한 고증", [("EVT", 3, 8)])]
    ledger = [{"row_id": "a", "start": 3, "end": 8,
               "surface": "귀족사회의", "verdict": "EVT"}]
    assert len(find_genitive_residue(rows, ledger)) == 1
    # 원장이 넣지 않은 자리는 이 변경의 책임 밖이다
    other = [{"row_id": "a", "start": 0, "end": 2,
              "surface": "영국", "verdict": "EVT"}]
    assert find_genitive_residue(rows, other) == []


def test_homomorph_population_is_gold_measured():
    """모집단이 '목록 안팎' 이 아니라 gold 실측이어야 반증 가능하다.

    포함 목록을 없앤 뒤 '등재/미등재로 갈리나' 를 물으면 그 대립 자체가 사라져
    0 이 재작성의 부산물로 참이 된다. gold 에서 같은 head 를 쓰는데 한쪽만 EVT 인
    자리를 세면 데이터가 0 을 반증할 수 있다.
    """
    items = {i.surface: i for i in find_head_homomorph(_homomorph_rows())}
    # 이미 EVT 로 라벨된 `국무회의` 는 모집단이 아니다
    assert "국무회의" not in items
    assert items["주교회의"].standalone_hits == 1
    # 더 긴 EVT span 안에 있으면 삽입할 자리가 없다
    assert items["당국회담"].standalone_hits == 0
    assert items["당국회담"].clash_hits == 1


def test_homomorph_gate_blocks_unclassified():
    """분류되지 않은 동형 표면형이 남으면 게이트가 막아야 한다."""
    result = check_homomorph(_homomorph_rows(), ledger=[])
    assert result["unclassified"] >= 1
    assert "주교회의" in [i["surface"] for i in result["items"]
                       if i["status"] == "unclassified"]


def test_homomorph_gate_accepts_three_classifications():
    """회수·명시 제외·NOT 판정 셋 중 하나면 통과한다."""
    rows = _homomorph_rows()
    ledger = [{"surface": "주교회의", "verdict": "NOT"}]
    result = check_homomorph(rows, ledger)
    status = {i["surface"]: i["status"] for i in result["items"]}
    assert result["unclassified"] == 0
    assert status["주교회의"] == "judged_not"
    assert status["군법회의"] == "excluded"              # canonical 사유코드
    assert status["당국회담"] == "covered_by_longer_evt"  # 축3 최대-span 안


def test_homomorph_type_clash_is_not_silently_passed():
    """규칙은 EVT 인데 gold 가 다른 타입이면 판정 없이 통과시키면 안 된다.

    삽입할 독립 자리가 없다는 이유로 묶어 통과시키면 이 충돌이 묻힌다. 라벨러
    프롬프트는 그 head 를 EVT 로 가르치므로 체계적 오류로 되돌아온다.
    """
    rows = [_row("a", "미국국회는 법안을 처리했다", [("ORG", 0, 4)])]
    result = check_homomorph(rows, ledger=[])
    item = next(i for i in result["items"] if i["surface"] == "미국국회")
    assert item["standalone_hits"] == 0      # 삽입할 자리는 없지만
    assert item["clash_types"] == ["ORG"]
    assert item["status"] == "unclassified"  # 통과시키지 않는다
    assert result["unclassified"] == 1


def test_audit_ledger_drops_judged_not_sites():
    """NOT 판정 자리를 빼지 않으면 판정할수록 반영률이 내려간다.

    규칙상 후보로 보이지만 사람이 아니라고 정한 자리다. 분모에 남겨두면 원장이
    쌓일수록 커버리지가 깎여, 판정을 성실히 할수록 게이트가 나빠진다.
    """
    rows = [_row("a", "다보스포럼 회장이 왔다", []),
            _row("b", "주교회의에서 결정했다", [])]
    before = run_audit(rows)
    assert {c.surface for c in before.candidates} == {"다보스포럼", "주교회의"}

    ledger = [{"row_id": "a", "start": 0, "end": 5,
               "surface": "다보스포럼", "verdict": "NOT"}]
    after = run_audit(rows, ledger=ledger)
    assert {c.surface for c in after.candidates} == {"주교회의"}


def test_audit_ledger_matches_sites_not_surfaces():
    """같은 표면형이라도 판정하지 않은 자리는 후보로 남아야 한다."""
    rows = [_row("a", "다보스포럼 회장이 왔다", []),
            _row("b", "다보스포럼이 열렸다", [])]
    ledger = [{"row_id": "a", "start": 0, "end": 5,
               "surface": "다보스포럼", "verdict": "NOT"}]
    rep = run_audit(rows, ledger=ledger)
    assert [(c.row_id, c.start) for c in rep.candidates] == [("b", 0)]
