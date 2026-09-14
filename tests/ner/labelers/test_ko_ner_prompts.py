"""한국어 NER 프롬프트의 규칙 정합성 테스트.

**few-shot 예시도 규칙의 일부다**(`docs/specs/coding-conventions.md`). 규칙 문장을
고치고 예시를 안 고치면 프롬프트가 같은 모양에 두 답을 가르치는데, 그건 어느 한쪽을
고르는 것보다 무조건 나쁘고 **어떤 기계 검사도 안 잡는다** — 게이트는 diff 의 글자만
읽고, 라벨러 출력은 LLM 을 불러야 보인다. 그래서 예시에서 기계로 확인할 수 있는 것을
여기서 고정한다.
"""

import json
import pathlib
import re
import tempfile

from ner.labelers.ko.canonical_rules import (
    canonical_axis1_heads,
    following_span_head,
    parse_canonical_holiday,
)
from ner.labelers.ko.ko_evt_r2_audit import CANONICAL_PATH, canonical_section_rows
from ner.labelers.ko.ner_prompts import SINGLE_PROMPT_TEMPLATE

_PAIR = re.compile(r"입력: (?P<text>.+)\n출력: (?P<spans>\[.*\])")
_QUOTED = re.compile(r'"([^"]+)"')


def _few_shots():
    """`{{` 이스케이프를 되돌려 (문장, [span]) 쌍으로 읽는다."""
    body = SINGLE_PROMPT_TEMPLATE.replace("{{", "{").replace("}}", "}")
    out = []
    for match in _PAIR.finditer(body):
        text = match.group("text")
        if text.strip() == "{sentence}":
            continue
        out.append((text, json.loads(match.group("spans"))))
    return out


def test_the_parser_reads_every_example_in_the_template():
    """파서가 빠뜨린 예시는 **영원히 검사 밖**이다 — 개수를 원문과 대조한다.

    `>= N` 같은 하한만 두면 파서가 예시를 빠뜨려도 통과하고, 그러면 안전망이 있다는
    기록만 남는다. **잡는 것은 파싱 누락뿐이다** — 세는 기준이 같은 원문이라 예시를
    템플릿에서 통째로 지우면 양쪽이 같이 줄어 안 걸린다.
    """
    written = SINGLE_PROMPT_TEMPLATE.count("입력: ") - 1     # `{sentence}` 자리 제외
    shots = _few_shots()
    assert len(shots) == written
    assert all(spans for _, spans in shots)


def _squeeze(text: str) -> str:
    """공백을 지운 좌표계.

    프롬프트는 복합 행정지명·날짜를 **일부러 붙여서** 낸다(`서울 삼성동` →
    `서울삼성동`, 절대 분리 금지 규칙). 그래서 원문 그대로의 부분문자열 검사는
    의도된 출력을 오탐한다. 공백만 지우면 그 규칙은 통과시키면서 "입력에 없는
    문자열" 은 그대로 걸린다.
    """
    return re.sub(r"\s+", "", text)


def test_every_few_shot_span_occurs_in_its_own_input():
    """예시 출력이 입력에 없는 문자열이면 모델에게 없는 자리를 가르치는 것이다."""
    for text, spans in _few_shots():
        squeezed = _squeeze(text)
        for span in spans:
            assert _squeeze(span["text"]) in squeezed, (span["text"], text)


def test_no_two_same_type_spans_are_separated_only_by_whitespace():
    """붙어 있는 같은 타입 span 둘은 쪼갤지 합칠지가 안 정해졌다는 신호다.

    실제로 이 검사가 잡은 것 — `천안함 침몰` + `추모식` 을 두 EVT 로 가르면서 세 줄
    위에서는 같은 모양(`소치올림픽 폐막식`)을 한 span 으로 가르쳤다. canonical 은
    고유명을 낀 것을 축3 최대-span 으로 정해 두었으므로 답은 하나여야 하고, 한
    프롬프트가 같은 모양에 두 답을 가르치는 것은 어느 한쪽을 고르는 것보다 나쁘다.
    """
    for text, spans in _few_shots():
        squeezed = _squeeze(text)
        placed = []
        for span in spans:
            body = _squeeze(span["text"])
            start = squeezed.index(body)
            placed.append((start, start + len(body), span["type"]))
        placed.sort()
        for (_, end, left), (start, _, right) in zip(placed, placed[1:]):
            assert not (left == right and start == end), (
                f"{left} spans touch in: {text}")


def _taught_heads(template: str) -> set:
    """프롬프트가 실제로 가르치는 축1 head 집합.

    `사건·재해 head(...)`·`대회·행사 head(...)` 두 묶음에서만 뽑는다. 따옴표를 통째로
    긁으면 같은 줄의 예시(`"세월호 참사"`)와 다른 규칙의 head(`"침몰 사고"`)가 섞이고,
    **모듈 상수에 있는 것만 남기고 거르면 초과분을 못 본다** — 프롬프트에만 있는
    head 는 걸러져 사라지므로 집합 비교가 한쪽 방향으로만 작동한다.
    """
    groups = re.findall(r"(?:사건·재해|대회·행사) head\(([^)]*)\)", template)
    return {head for group in groups for head in _QUOTED.findall(group)}


def test_the_template_teaches_exactly_the_axis1_head_list():
    """프롬프트가 가르치는 head 집합과 canonical §5.3 이 어긋나면 라벨러가 옛 기준으로
    답한다.

    존재 검사(`"사건 head" in template`)로는 부족하다 — 그 문자열은 PROD 절에도 있어서
    **축1 절을 통째로 지워도 통과한다.** 템플릿에서 head 집합을 실제로 뽑아 표와
    대조해야 좁힘·넓힘이 둘 다 걸린다.
    """
    assert _taught_heads(SINGLE_PROMPT_TEMPLATE) == set(canonical_axis1_heads())


def test_head_extraction_ignores_examples_and_other_rules():
    """뽑는 범위가 넓어지면 위 집합 비교가 의미를 잃는다 — 추출기 자체를 고정한다.

    같은 줄에 예시(`"세월호 참사"`)가 있고 다른 절에는 PROD 용 head(`"침몰 사고"`)가
    있다. 둘 중 하나라도 섞이면 집합이 안 맞아 비교가 늘 실패하거나, 반대로 필터를
    넣어 맞추면 초과분을 못 보게 된다.
    """
    taught = _taught_heads(SINGLE_PROMPT_TEMPLATE)
    assert "세월호 참사" not in taught
    assert "침몰 사고" not in taught
    assert "보스턴 마라톤" not in taught
    assert {"참사", "마라톤"} <= taught


def test_axis1_rule_names_the_particle_exclusion():
    """조사 붙은 선행 고유명 제외는 head 목록만큼 모집단을 바꾼다."""
    assert "앞 고유명에 조사가 붙으면" in SINGLE_PROMPT_TEMPLATE
    assert '"파리에서 테러"' in SINGLE_PROMPT_TEMPLATE


def test_particle_bearing_modifier_is_taught_as_out_of_span():
    """`파리에서 테러` 는 복합명사가 아니라 문장 성분이다 — 예시가 그걸 보여야 한다."""
    shots = dict(_few_shots())
    text = next(t for t in shots if "파리에서 테러" in t)
    types = {s["text"]: s["type"] for s in shots[text]}
    assert types["파리"] == "LOC"
    assert types["테러"] == "EVT"
    assert "파리에서 테러" not in types


# ── canonical §2.5 ↔ 프롬프트 LOC/ORG 동기 ────────────────────────────────
#
# 규칙은 기준 파일 §2.5 에 있고 그것을 실제로 집행하는 프롬프트는 잠금 밖이라,
# 둘이 어긋나도 게이트는 아무것도 못 본다. 특히 **비-entity 목록에서 `법원` 을
# 빼는 것과 ORG 목록에 `법원` 을 더하는 것은 같은 규칙 변경인데 모양이 다르다** —
# 한쪽만 검사하면 다른 쪽으로 새므로 두 집합을 양방향으로 대조한다.

KO_LOCORG_SECTION = "### 2.5"
_BACKTICK = re.compile(r"`([^`]+)`")
_ALLOWLIST_MARKER = re.compile(r"\*\*환유 정부건물명 allowlist \d+ 종\*\*:\s*(.+)")
_PROMPT_NONENTITY_LINE = re.compile(r"^ *\((?:1|2)\) (.+)$", re.M)
_PROMPT_ALLOWLIST = re.compile(r"환유 정부건물명\(([^)]*)\)")


def _ko_rows(path=CANONICAL_PATH):
    """§2.5 의 **3칸 판정표** 행만. 회색지대 표는 2칸이라 자연히 빠진다."""
    rows = []
    for line in canonical_section_rows(path, KO_LOCORG_SECTION):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0].startswith("---") or cells[0] == "규칙":
            continue
        rows.append(cells)
    return rows


def _canonical_examples(verdict, path=CANONICAL_PATH):
    """§2.5 판정표에서 그 판정을 받은 행의 예시 표면형."""
    out = set()
    for _rule, judgment, examples in _ko_rows(path):
        hit = ("비-entity" in judgment) if verdict == "non-entity" \
            else judgment == f"`{verdict}`"
        if hit:
            out |= set(_BACKTICK.findall(examples))
    return out


def _canonical_allowlist(text=None):
    if text is None:
        text = pathlib.Path(CANONICAL_PATH).read_text(encoding="utf-8")
    match = _ALLOWLIST_MARKER.search(text)
    assert match, "§2.5 의 allowlist 마커가 없다 — 파서가 읽을 자리가 사라졌다"
    return set(_BACKTICK.findall(match.group(1)))


def _prompt_nonentity(template=SINGLE_PROMPT_TEMPLATE):
    body = template.replace("{{", "{").replace("}}", "}")
    return {q for line in _PROMPT_NONENTITY_LINE.findall(body)
            for q in _QUOTED.findall(line)}


def _prompt_allowlist(template=SINGLE_PROMPT_TEMPLATE):
    match = _PROMPT_ALLOWLIST.search(template)
    assert match, "프롬프트의 환유 정부건물명 열거가 없다"
    return set(_QUOTED.findall(match.group(1)))


def test_prompt_nonentity_matches_canonical_section_25():
    """비-entity 예시 집합이 §2.5 와 정확히 같다 (좁힘·넓힘 양쪽)."""
    assert _prompt_nonentity() == _canonical_examples("non-entity")


def test_prompt_allowlist_matches_canonical_section_25():
    """환유 정부건물명 allowlist 가 §2.5 와 정확히 같다."""
    assert _prompt_allowlist() == _canonical_allowlist()


def test_no_canonical_nonentity_surface_is_taught_as_org():
    """`법원` 을 비-entity 에서 빼는 대신 **ORG 절에 더하는** 반대 방향 공격.

    집합 비교만 두면 이 경로로 샌다 — 비-entity 목록은 그대로인 채 ORG 열거만
    늘어나므로 위 두 테스트가 초록이다.
    """
    body = SINGLE_PROMPT_TEMPLATE.replace("{{", "{").replace("}}", "}")
    org_block = body[body.index("- ORG (기관명)"):body.index("- 비-entity")]
    taught_as_org = set(_QUOTED.findall(org_block))
    assert not (taught_as_org & _canonical_examples("non-entity"))


def test_few_shot_outputs_agree_with_canonical_section_25():
    """예시 출력이 §2.5 판정과 갈리지 않는다.

    `국세청은 삼성전자와 대법원에…` 한 줄이 drop 규칙의 실행형이라, 출력에
    `대법원` 을 더하는 커밋은 **어떤 열거도 안 건드리고** 규칙을 뒤집는다.
    """
    dropped = _canonical_examples("non-entity")
    org = _canonical_examples("ORG")
    pinned = _pinned_verdicts()
    for text, spans in _few_shots():
        types = {s["text"]: s["type"] for s in spans}
        for surface in dropped:
            assert surface not in types, (surface, text)
        for surface in org:
            if surface in text:
                assert types.get(surface) == "ORG", (surface, text)
        # 표 예시만 보면 산문이 못 박은 판정을 few-shot 이 뒤집어도 지나간다 —
        # `경찰`=ORG 하나가 gold 대비 FP 를 약 1,000 개 만든다.
        for surface, verdict in pinned.items():
            if surface not in text:
                continue
            if verdict == "비-entity":
                assert surface not in types, (surface, text)
            else:
                assert types.get(surface, verdict) == verdict, (surface, text)


def test_ko_locorg_extraction_ignores_the_other_tables():
    """뽑는 범위가 넓어지면 위 비교가 의미를 잃는다 — 추출기 자체를 고정한다.

    §2.5 에는 회색지대 표(2칸)가 함께 있고 프롬프트에는 LOC·ORG 절이 따로 있다.
    어느 쪽이 섞이면 집합이 늘 안 맞아 비교가 형식만 남는다.
    """
    dropped = _canonical_examples("non-entity")
    assert "국세청" not in dropped and "한강" not in dropped
    assert {"대법원", "삼성전자", "경부고속도로"} <= dropped
    assert "경찰서" not in dropped          # 회색지대 표(2칸)는 안 읽는다
    assert _canonical_examples("LOC") & {"한강", "백두산"}


def _sync_gap(canonical_text, template):
    """두 소스의 어긋남. 비어 있으면 동기 상태다."""
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "canonical.md"
        path.write_text(canonical_text, encoding="utf-8")
        return (_prompt_nonentity(template) ^ _canonical_examples("non-entity", str(path))) \
            | (_prompt_allowlist(template) ^ _canonical_allowlist(canonical_text))


def test_sync_breaks_in_both_directions():
    """한쪽만 고치면 실패한다 — **양방향을 실제로 태워** 확인한다.

    "규칙을 바꾸는 데 마찰이 생긴다" 가 이 이슈의 목적인데, 집합이 지금 같다는
    사실만으로는 그걸 못 잰다. 검사가 실제로 무는지는 어긋나게 만들어 봐야 안다.
    """
    canonical = pathlib.Path(CANONICAL_PATH).read_text(encoding="utf-8")
    assert not _sync_gap(canonical, SINGLE_PROMPT_TEMPLATE)      # 지금은 동기다

    # canonical 만 바꾼다 — 비-entity 행에서 예시 하나를 뺀다
    moved = canonical.replace("`대법원`, `수원지법`", "`수원지법`")
    assert moved != canonical
    assert _sync_gap(moved, SINGLE_PROMPT_TEMPLATE) == {"대법원"}

    # 프롬프트만 바꾼다 — allowlist 에 하나를 더한다
    widened = SINGLE_PROMPT_TEMPLATE.replace(
        '"크렘린", "펜타곤")도 ORG', '"크렘린", "펜타곤", "총리공관")도 ORG')
    assert widened != SINGLE_PROMPT_TEMPLATE
    assert _sync_gap(canonical, widened) == {"총리공관"}


# ── 범주 층 동기 ──────────────────────────────────────────────────────────
#
# 예시만 대조하면 **범주를 건드리는 변경이 통째로 샌다** — 비-entity 열거에서
# `병원` 을 빼거나 ORG 열거에 `법원` 을 더하는 것은 어느 예시도 안 건드린다.
# 모델이 일반화하는 것은 예시가 아니라 범주라, 그쪽이 규칙의 본체다.

_MARKER = r"\*\*{label} \d+ 종\*\*:\s*(.+)"
_TOKEN_SPLIT = re.compile(r"[·/()\s,.—:]+")


def _canonical_marker(label, text=None):
    if text is None:
        text = pathlib.Path(CANONICAL_PATH).read_text(encoding="utf-8")
    match = re.search(_MARKER.format(label=label), text)
    assert match, f"§2.5 의 `{label}` 마커가 없다 — 파서가 읽을 자리가 사라졌다"
    return set(_BACKTICK.findall(match.group(1)))


def _prompt_line(prefix, template=SINGLE_PROMPT_TEMPLATE):
    body = template.replace("{{", "{").replace("}}", "}")
    hits = [ln for ln in body.splitlines() if ln.strip().startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]


def _prompt_categories(tag, template=SINGLE_PROMPT_TEMPLATE):
    """비-entity (1)·(2) 줄의 **범주 명사**. 따옴표 예시는 걷어낸다."""
    line = re.sub(r'"[^"]*"', "", _prompt_line(tag, template))
    line = line.split("—", 1)[1] if "—" in line else line
    return {t for t in _TOKEN_SPLIT.split(line) if t}


def test_prompt_facility_categories_match_canonical():
    """시설·구조물 범주가 §2.5 와 정확히 같다 (`병원` 을 빼면 실패한다)."""
    assert _prompt_categories("(1)") == _canonical_marker("비-entity 시설·구조물 범주")


def test_prompt_organisation_categories_match_canonical():
    """조직 범주가 §2.5 와 정확히 같다 (`법원` 을 빼면 실패한다)."""
    assert _prompt_categories("(2)") == _canonical_marker("비-entity 조직 범주")


def _org_block(template=SINGLE_PROMPT_TEMPLATE):
    """ORG 절 **전체**. 한 줄만 보면 `주의:`·`예:` 줄로 새는 길이 남는다."""
    body = template.replace("{{", "{").replace("}}", "}")
    return body[body.index("- ORG (기관명)"):body.index("- 비-entity")]


def test_org_block_is_free_of_nonentity_terms():
    """**이슈 §왜 가 이름 붙인 공격** — 비-entity 에서 빼는 대신 ORG 에 더하기.

    집합 equality 는 이 경로를 못 막는다. 비-entity 목록은 그대로인 채 ORG 쪽만
    늘어나므로 양쪽 집합이 여전히 같다. 그래서 ORG 절의 **토큰**이 §2.5 의
    비-entity 범주·예시와 겹치는지 따로 본다 — 따옴표 없이 `법원/검찰` 로 더해도,
    `예:` 줄이나 새 `주의:` 줄에 심어도 잡힌다. 한 줄만 보면 옆줄로 새므로 절
    전체를 본다. `공기업` 토큰은 `기업` 과 달라 오탐하지 않는다.
    """
    block = re.sub(r'"[^"]*"', "", _org_block())
    tokens = {t for t in _TOKEN_SPLIT.split(block) if t}
    forbidden = (_canonical_marker("비-entity 시설·구조물 범주")
                 | _canonical_marker("비-entity 조직 범주")
                 | _canonical_examples("non-entity"))
    assert not (tokens & forbidden), tokens & forbidden


def _canonical_verdicts(label):
    """마커에서 `` `표면형`=판정 `` 쌍을 읽는다."""
    text = pathlib.Path(CANONICAL_PATH).read_text(encoding="utf-8")
    body = re.search(_MARKER.format(label=label), text)
    assert body, label
    pairs = re.findall(r"`([^`]+)`=(`?[^\s·(.]+`?)", body.group(1))
    return {surface: verdict.strip("`") for surface, verdict in pairs}


_PROMPT_PAIR = re.compile(r'"([^"]+)"=(\S+?)(?=[\s,.(]|$)')


def _prompt_verdicts(template=SINGLE_PROMPT_TEMPLATE):
    body = template.replace("{{", "{").replace("}}", "}")
    return dict(_PROMPT_PAIR.findall(body))


def _pinned_verdicts():
    """§2.5 가 못 박은 표면형 — 프롬프트와 **판정 쌍**으로 묶는 자리."""
    pinned = dict(_canonical_verdicts("단독 표면형"))
    pinned.update(_canonical_verdicts("시설 접미 예외"))
    pinned.update(_canonical_verdicts("접미 판정"))
    pinned["아우크스부르크"] = "LOC"        # §2.5 표의 지명 표면형 행
    return pinned


def test_prompt_pins_the_same_verdicts_as_canonical():
    """**존재가 아니라 판정으로 대조한다.**

    `"경찰"` 이라는 문자열이 남아 있는지만 보면 그 줄의 판정을 비-entity → ORG 로
    뒤집어도 통과한다. gold 는 `경찰` 1,077 등장 중 `ORG` 9 라, 그 뒤집기 하나가
    FP 를 약 1,000 개 만드는데도 게이트는 초록이다.
    """
    pinned = _pinned_verdicts()
    taught = _prompt_verdicts()
    assert set(pinned) <= set(taught), set(pinned) - set(taught)
    for surface, verdict in pinned.items():
        assert taught[surface] == verdict, (surface, verdict, taught[surface])


def test_prompt_declares_no_verdict_canonical_contradicts():
    """프롬프트가 §2.5 예시에 다른 판정을 새로 못 박는 것도 막는다."""
    taught = _prompt_verdicts()
    for verdict in ("ORG", "LOC"):
        for surface in _canonical_examples(verdict):
            if surface in taught:
                assert taught[surface] == verdict, (surface, taught[surface])
    for surface in _canonical_examples("non-entity"):
        assert taught.get(surface, "비-entity") == "비-entity", surface


# 조사는 **유한 폐쇄 품사**라 열거가 끝난다 — 목록 밖 하나(`까지`)가 남으면
# 그 하나로 규칙을 뒤집을 수 있다.
_PARTICLE = (r"(?:은|는|이라도|이나|이|가|을|를|도|의|에게|에서|에|와|과|만큼|만"
             r"|으로|로|까지|부터|조차|마저|밖에|처럼|라도|나|한테|보다)?")
_BOUNDARY = r"(?=[\s,.·()\"'/=—]|$)"
_VERDICT_WORDS = ("ORG", "LOC", "비-entity")
# 프롬프트가 스스로 붙인 판정 이름 — `- ORG (기관명):`·`- LOC (지명):`. 이걸 안
# 세면 `경찰은 기관명으로 뽑는다` 가 판정어 없이 규칙을 뒤집는다. **별도 낱말이
# 아니라 정규화여야 한다** — 별도로 세면 판정을 선언하는 그 줄 자체가 오탐한다.
_VERDICT_GLOSS = {"기관명": "ORG", "지명": "LOC"}


def _rule_lines(template=SINGLE_PROMPT_TEMPLATE):
    """규칙 영역을 **줄** 단위로. 판정 쌍은 걷어내고 few-shot 은 뺀다.

    문장(`.`)이 아니라 줄로 보는 것은 규칙이 줄 단위 불릿이기 때문이다 —
    `경찰은 예외다. 그것은 ORG 다` 처럼 한 줄 안에서 문장만 갈라도 공기가
    끊긴다. 줄 단위의 오탐 프로파일은 문장 단위와 같다(둘 다 위반 0).
    """
    body = template.replace("{{", "{").replace("}}", "}")
    rules = _PROMPT_PAIR.sub(" ", body[:body.index("입력: ")])
    for gloss, verdict in _VERDICT_GLOSS.items():
        rules = rules.replace(gloss, verdict)
    return [line for line in rules.splitlines() if line.strip()]


def _mentions(surface, sentence):
    """표면형이 **낱말로** 나타나는가 — 조사는 흡수하고 합성어는 제외한다.

    `경찰은`·`경찰이` 는 같은 낱말이지만 `경찰청`·`경찰서` 는 다른 낱말이다.
    정확 토큰 일치로는 앞엣것을 놓치고, 부분문자열로는 뒤엣것을 오탐한다.
    """
    return re.search(re.escape(surface) + _PARTICLE + _BOUNDARY, sentence) is not None


def _prose_pinned():
    """산문 검사용 pinned — 접미(`청`·`서`·`소`)는 뺀다.

    한 글자 접미는 낱말 경계로 못 가른다(`청` 은 `국세청`·`청와대` 안에 있다).
    접미는 쌍 대조가 맡고, 여기서는 표면형만 본다.
    """
    pinned = _pinned_verdicts()
    for suffix in _canonical_verdicts("접미 판정"):
        pinned.pop(suffix, None)
    return pinned


def test_pinned_surfaces_never_appear_beside_the_wrong_verdict():
    """쌍은 그대로 두고 **옆 산문으로 반대를 가르치는** 경로를 막는다.

    `"경찰"=비-entity` 를 남긴 채 `경찰은 수사 주체를 지시할 때 ORG 로 뽑는다` 를
    한 줄 더하면 쌍 대조는 초록이다. gold 는 `경찰` 1,077 등장 / `ORG` 9 라 그
    한 줄이 FP 를 약 1,000 개 만든다.

    **구역이 아니라 줄로 본다** — 절을 몇 개 열거하든 그 밖에 심으면 새고,
    LLM 은 프롬프트를 절 단위로 읽지 않는다. 같은 판정 아래의 언급(`정부` 가 ORG
    절 정의 문장에 있는 것)은 문장의 판정어가 pinned 와 같아 자연히 통과한다.
    """
    for line in _rule_lines():
        present = {word for word in _VERDICT_WORDS if word in line}
        for surface, verdict in _prose_pinned().items():
            if not _mentions(surface, line):
                continue
            wrong = present - {verdict}
            assert not wrong, (surface, verdict, wrong, line.strip())


def test_prompt_pins_nothing_a_category_forbids():
    """범주를 우회해 **개별 표면형에 판정을 못 박는** 경로를 막는다.

    `"세브란스병원"=ORG` 는 어느 열거도 안 건드리고 `병원`=비-entity 범주를
    무력화한다. 범주를 §2.5 로 올린 것이 요점이었으므로 초과 쌍도 범주와 대조한다.
    """
    forbidden = (_canonical_marker("비-entity 시설·구조물 범주")
                 | _canonical_marker("비-entity 조직 범주"))
    exempt = (_canonical_marker("환유 정부건물명 allowlist")
              | _canonical_examples("ORG") | _canonical_examples("LOC"))
    for surface, verdict in _prompt_verdicts().items():
        if verdict == "비-entity" or surface in exempt:
            continue
        for term in forbidden:
            assert not surface.endswith(term), (surface, verdict, term)


def test_category_marker_counts_match_their_lists():
    """마커의 `N 종` 이 목록 길이와 다르면 "옮긴 항목 수를 센다" 는 근거가 헐거워진다."""
    text = pathlib.Path(CANONICAL_PATH).read_text(encoding="utf-8")
    lists = ("비-entity 시설·구조물 범주", "비-entity 조직 범주",
             "환유 정부건물명 allowlist")
    pairs = ("단독 표면형", "시설 접미 예외", "접미 판정")
    for label in lists + pairs:
        match = re.search(_MARKER.format(label=label), text)
        assert match, label
        declared = int(re.search(r"(\d+) 종", match.group(0)).group(1))
        # 쌍 마커는 백틱이 표면형·판정 양쪽에 붙어 있어 표면형만 센다
        counted = len(_canonical_verdicts(label)) if label in pairs \
            else len(_canonical_marker(label, text))
        assert declared == counted, (label, declared, counted)


def test_a_discriminating_few_shot_survives():
    """판별 예시가 통째로 지워지는 것도 잡는다.

    few-shot 이 기준 3 의 집행 수단이 된 이상, "입력에 비-entity 표면형이 있는데
    출력에서 빠진" 예시가 **하나도 없는** 상태는 규칙을 가르칠 자리가 사라진 것이다.
    """
    dropped = _canonical_examples("non-entity")
    discriminating = [
        text for text, spans in _few_shots()
        if any(d in text for d in dropped)
        and not (dropped & {s["text"] for s in spans})
    ]
    assert discriminating, "비-entity 를 실제로 버리는 few-shot 이 없다"


# ── canonical §3.3·§5.3 ↔ 프롬프트 명절 판정 동기 ──────────────────────────
#
# gold 는 명절 이름 138 자리를 `DAT` 에서 `EVT` 로 옮겼는데 **프롬프트는 잠금
# 밖이다.** 둘이 갈리면 라벨러가 옛 기준으로 답하고 그 하락이 재라벨 탓인지
# 모델 탓인지 구별되지 않는다 — 어떤 기계 검사도 안 잡는 자리라 여기서 묶는다.
#
# 대조 상대는 **canonical 표**다. 사본을 두고 그것과 맞추면 두 사본이 나란히
# 틀려도 통과하는 고리가 생기므로, `canonical_rules` 는 사본 없이 표를 읽는다.

_HOLIDAY_LIST = {
    "day_heads": re.compile(r"하루 머리\(([^)]*)\)"),
    "span_heads": re.compile(r"기간 머리\(([^)]*)\)"),
    "category_heads": re.compile(r"범주 머리\(([^)]*)\)"),
    "period_names": re.compile(r"구간 이름\(([^)]*)\)"),
}


def _taught_holiday(key, template=SINGLE_PROMPT_TEMPLATE):
    """프롬프트가 실제로 열거하는 목록. **열거는 목록마다 한 자리여야 한다.**

    두 자리에 적으면 한쪽이 좁아져도 다른 쪽이 맞아 집합 비교가 통과한다 —
    LLM 은 둘 다 읽으므로 좁은 쪽이 실제 판정을 바꾼다. 그래서 개수를 여기서
    막고, 목록을 옮기려면 옮긴 자리 하나만 남겨야 한다.
    """
    body = template.replace("{{", "{").replace("}}", "}")
    hits = _HOLIDAY_LIST[key].findall(body)
    assert len(hits) == 1, (key, len(hits))
    return set(_QUOTED.findall(hits[0]))


def test_the_template_teaches_exactly_the_canonical_holiday_lists():
    """네 목록이 §5.3 과 정확히 같다 (좁힘·넓힘 양쪽).

    가르는 것은 이름이 아니라 **머리**다 — `크리스마스`(EVT)와
    `크리스마스 시즌`(DAT)을 나누는 것이 기간 머리 목록이라, 거기서 `즈음`
    하나가 빠지면 그 자리의 타입이 통째로 뒤집힌다.
    """
    parsed = parse_canonical_holiday()
    for key in _HOLIDAY_LIST:
        assert _taught_holiday(key) == set(parsed[key]), key


def test_holiday_list_extraction_ignores_the_surrounding_examples():
    """뽑는 범위가 넓어지면 위 집합 비교가 의미를 잃는다 — 추출기를 고정한다.

    같은 줄에 판정 예시(`"크리스마스 시즌"`)가 함께 있어, 따옴표를 줄째 긁으면
    집합이 늘 안 맞는다. 반대로 목록 밖을 걸러 맞추면 초과분을 못 본다.
    """
    span_heads = _taught_holiday("span_heads")
    assert "올해 크리스마스 시즌" not in span_heads
    assert "설 연휴" not in span_heads
    assert {"시즌", "연휴", "때"} <= span_heads
    assert "첫 날" in _taught_holiday("day_heads")     # 공백 있는 항목도 읽는다


def _holiday_gap(canonical_text, template):
    """두 소스의 어긋남. 비어 있으면 동기 상태다."""
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "canonical.md"
        path.write_text(canonical_text, encoding="utf-8")
        parsed = parse_canonical_holiday(str(path))
    gap = set()
    for key in _HOLIDAY_LIST:
        gap |= _taught_holiday(key, template) ^ set(parsed[key])
    return gap


def test_holiday_sync_breaks_in_both_directions():
    """한쪽만 고치면 실패한다 — **양방향을 실제로 태워** 확인한다.

    집합이 지금 같다는 사실만으로는 검사가 무는지 알 수 없다. canonical 을
    좁히는 쪽과 프롬프트를 넓히는 쪽을 각각 만들어 걸리는 것을 본다.
    """
    canonical = pathlib.Path(CANONICAL_PATH).read_text(encoding="utf-8")
    assert not _holiday_gap(canonical, SINGLE_PROMPT_TEMPLATE)      # 지금은 동기다

    narrowed = canonical.replace("·`즈음`", "")
    assert narrowed != canonical
    assert _holiday_gap(narrowed, SINGLE_PROMPT_TEMPLATE) == {"즈음"}

    widened = SINGLE_PROMPT_TEMPLATE.replace('"즈음")', '"즈음", "환절기")')
    assert widened != SINGLE_PROMPT_TEMPLATE
    assert _holiday_gap(canonical, widened) == {"환절기"}


def test_every_holiday_example_the_prompt_teaches_has_a_canonical_root():
    """프롬프트가 §5.3 이 모르는 이름을 EVT 로 가르치지 않는다.

    어근 목록은 gold 를 만들 때 쓴 **후보 그물**이라, 목록 밖 이름은 gold 에서
    `DAT` 로 남아 있다. 프롬프트만 `추분`·`백중` 을 EVT 로 가르치면 그
    예측은 전부 FP 가 되고, 어긋남을 볼 것이 없다.

    **잡는 것은 넓힘뿐이다** — 프롬프트의 예시는 열거가 아니라 표본이라
    어근을 덜 든 것은 결함이 아니다. 좁힘을 막는 것은 위 네 목록의 집합
    비교이고, 그쪽은 판정을 가르는 머리라 표본일 수 없다.
    """
    roots = parse_canonical_holiday()["roots"]
    for prefix in ("주의: 명절·기념일·절기 이름은 EVT", "주의: 하루 머리("):
        line = re.sub(r"하루 머리\([^)]*\)", "", _prompt_line(prefix))
        names = _QUOTED.findall(line)
        assert names, prefix
        for name in names:
            assert any(root in name for root in roots), (name, prefix)


DERIVED_EXAMPLE = "올해 크리스마스 시즌"


def test_the_holiday_contrast_pair_is_in_the_few_shots():
    """파생 `DAT` ↔ 맨이름 `EVT` 대조쌍이 예시로 있다.

    규칙 문장만으로는 **포함 우선순위**가 안 전해진다 — `올해 크리스마스 시즌`
    을 `DAT` 하나로 낼지, 그 안의 `크리스마스` 를 `EVT` 로 또 낼지가 문장에서는
    둘 다 읽히고 gold 는 평면 BIO 라 겹칠 수 없다. 그래서 파생 예시에 그 이름이
    **별도 span 으로 없다는 것**을 여기서 못 박는다.

    두 답이 한 예시에 같이 있으면 대조가 아니라 모순이므로 서로 다른 예시여야
    한다.
    """
    shots = _few_shots()
    derived = [(t, s) for t, s in shots
               if any(x["text"] == DERIVED_EXAMPLE for x in s)]
    assert len(derived) == 1, len(derived)
    text, spans = derived[0]
    assert [x["type"] for x in spans if x["text"] == DERIVED_EXAMPLE] == ["DAT"]
    assert "크리스마스" not in {x["text"] for x in spans}

    bare = [(t, s) for t, s in shots
            if any(x["text"] == "크리스마스" and x["type"] == "EVT" for x in s)]
    assert bare, "맨이름 `크리스마스`=EVT 예시가 없다"
    assert all(t != text for t, _ in bare)


def test_a_few_shot_teaches_the_period_head_shared_across_a_coordination():
    """`설과 추석 연휴` — 등위로 기간 머리를 나눠 갖는 자리.

    바로 뒤만 보면 `추석 연휴` 만 `DAT` 로 걸러지고 `설` 은 `EVT` 로 남아
    **한 명사구 안에서 타입이 갈린다.** gold 는 그 4 자리를 `DAT` 로 묶어 뒀으므로,
    프롬프트가 반대를 가르치면 그 자리마다 라벨러와 gold 가 어긋난다.
    """
    hit = [(t, s) for t, s in _few_shots() if "설과 추석 연휴" in t]
    assert hit, "등위 예시가 없다"
    _text, spans = hit[0]
    types = {x["text"]: x["type"] for x in spans}
    assert types.get("설") == "DAT", types
    assert types.get("추석 연휴") == "DAT", types


def test_few_shot_holiday_spans_obey_the_head_rule():
    """예시 출력이 §3.3 머리 원칙과 갈리지 않는다.

    위 두 검사는 지목한 예시만 본다 — **새 예시가 `추석`=DAT 를 가르쳐도
    지나간다.** 여기서는 명절 어근을 담은 모든 예시 span 을 판정 함수에
    태워, 자리마다 머리가 무엇을 부르는지로 기대 타입을 계산해 대조한다.
    머리를 보는 것은 `canonical_rules.following_span_head()` — gold 를 그렇게
    갈랐으므로 예시도 같은 자로 재야 한다.

    **span 을 붙여서 내는 예시는 건너뛴다**(`서울 삼성동`→`서울삼성동`) —
    원문에 그대로 없어 자리를 계산할 수 없다. 명절 예시에는 그런 자리가 없다.
    """
    parsed = parse_canonical_holiday()
    tails = tuple(parsed["span_heads"]) + tuple(parsed["category_heads"])
    seen = 0
    for text, spans in _few_shots():
        for span in spans:
            surface = span["text"]
            if not any(root in surface for root in parsed["roots"]):
                continue
            if surface not in text:
                continue
            seen += 1
            # 같은 표면형이 한 예시에 두 번 나오면 `index()` 가 첫 자리만 봐
            # 뒤엣것이 검사 밖으로 빠진다. 조용히 넘기지 않고 여기서 막는다 —
            # 그런 예시를 넣으려면 자리 계산을 먼저 고쳐야 한다.
            assert text.count(surface) == 1, (surface, text)
            end = text.index(surface) + len(surface)
            derived = surface.endswith(tails) or bool(following_span_head(text, end))
            expected = "DAT" if derived else "EVT"
            assert span["type"] == expected, (surface, expected, span["type"], text)
    assert seen >= 6, f"명절 어근을 담은 예시 span 이 {seen} 개뿐이다"
