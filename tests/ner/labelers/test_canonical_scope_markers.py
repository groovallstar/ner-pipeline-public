"""§3 `(공통)` 절의 언어 적용범위 마커 — 판정이 어느 언어에 걸리는지를 고정한다.

**왜 마커를 세는가.** canonical 은 `(공통)` 절과 언어별 절을 함께 두는데, `(공통)`
절이 `ORG` 를 판정하면서 적용 언어를 안 밝히면 읽는 사람은 그 판정이 자기 언어에도
걸린다고 읽는다. KO 는 §2.5 로 ORG 를 정부·행정·공공·정치 기관으로 좁혀 두었지만
§3 만 읽으면 `프리미어리그`·`KBS` 가 `ORG` 로 읽혔다 — 라벨러 프롬프트를 고치는
사람도 새 언어를 붙이는 사람도 그 문장을 근거로 삼을 수 있다.

**모집단은 §3 최상위 불릿이 아니라 §3 전체다.** §3 은 본문 불릿 뒤에 §3.1~§3.4
소절을 두고 그 안의 표·불릿도 `ORG` 를 판정한다. 최상위 불릿에서 멈추면 새 `ORG`
판정을 아무 소절에나 써서 마커 없이 통과시킬 수 있고, 그 무력화 비용은 0 이다.
그래서 절 경계는 다음 `## ` 까지이고 소절 제목은 모집단을 안 끊는다.

**대조 상대를 canonical 밖에 두는 이유.** 기대 목록을 canonical 에서 읽어 오면
canonical↔canonical 자기대조가 되어, 마커와 판정을 같은 커밋에서 함께 고치면 무엇을
적든 초록이다. 그래서 아래 상수는 문서에서 파싱하지 않고 여기에 손으로 적는다.

**모집단이 비면 예외를 던진다.** 파서가 읽는 절이 사라지거나 이름이 바뀌면 결과가
빈 목록이 되는데, 빈 목록은 "§3 에 ORG 판정이 실제로 없다" 와 구별되지 않는다.
규칙을 세는 쪽이 규칙을 못 읽었을 때는 조용한 0 이 아니라 시끄러운 실패여야 한다.
"""

import pathlib
import re

import pytest

from ner.labelers.ko.ko_evt_r2_audit import CANONICAL_PATH

SECTION3_HEADING = "## 3."

# §3 전체에서 `ORG` 를 판정하는 자리의 머리와 그것이 선언한 적용 스코프. 문서
# 순서대로 적는다. **이 목록은 canonical 에서 읽지 않는다** — 위 docstring 의
# 자기대조 사유. 한쪽만 고치면 붉어지는 것이 이 상수의 존재 이유다.
ORG_SCOPES = (
    ("정기 운영 리그·팀", "JA·VI·EN · KO 는 §2.5"),
    ("인프라", "JA·VI·EN · KO 는 §2.5"),
    ("약어 단독", "JA·VI·EN · KO 는 §2.5"),
    ("동음이의 가타카나 단독", "JA"),
    ("회사·사업체·레코드 레이블", "JA·VI·EN · KO 는 §2.5"),
    ("운영리그 vs 경기", "JA·VI·EN · KO 는 §2.5"),
)

_MARKER = re.compile(r"\*\*적용: ([^*]+)\*\*")
_HEAD_STOP = re.compile(r"\s*[(→—:]")
_ORG = "`ORG`"


def section3_units(text: str) -> list[str]:
    """§3 의 판정 단위 — 최상위 불릿과 표 행. 절이 없거나 단위가 없으면 예외.

    절의 끝은 다음 `## ` 다. `### ` 소절 제목에서 멈추지 않는 것은 §3.1~§3.4 의
    표·불릿도 `(공통)` 판정을 내리기 때문이다.
    """
    units: list[str] = []
    in_bullet = False
    found = False
    for line in text.splitlines():
        if line.startswith(SECTION3_HEADING):
            found = True
            continue
        if found and line.startswith("## "):
            break
        if not found:
            continue
        if line.startswith("- "):
            units.append(line[2:].strip())
            in_bullet = True
        elif line.startswith("  ") and in_bullet:
            units[-1] += " " + line.strip()
        elif line.startswith("|"):
            in_bullet = False
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 2 or cells[0].startswith("---") or cells[0] == "규칙":
                continue
            units.append(" | ".join(cells))
        else:
            in_bullet = False
    if not found:
        raise ValueError(
            f"canonical section {SECTION3_HEADING!r} not found; the scope-marker check "
            "reads its population from that section, and an empty result would be "
            "indistinguishable from a section with no ORG judgment"
        )
    if not units:
        raise ValueError(
            f"canonical section {SECTION3_HEADING!r} has no bullets or table rows; "
            "the scope-marker check reads its population from those units"
        )
    return units


def _head(unit: str) -> str:
    """판정 단위의 머리. 표 행이면 첫 칸, 불릿이면 판정 앞까지."""
    first = unit.split(" | ")[0]
    head = _HEAD_STOP.split(first, maxsplit=1)[0]
    return head.replace("*", "").replace("`", "").strip()


def org_scope_map(text: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """`ORG` 를 판정하는 단위의 (머리, 마커들) 을 문서 순서대로."""
    return tuple(
        (_head(unit), tuple(_MARKER.findall(unit)))
        for unit in section3_units(text)
        if _ORG in unit
    )


def _canonical_text() -> str:
    return pathlib.Path(CANONICAL_PATH).read_text(encoding="utf-8")


def test_every_org_judgment_in_section3_declares_a_language_scope():
    """`ORG` 를 판정하는 자리에 마커가 하나도 없으면 그 자리가 다시 열린다.

    canonical 머리말이 "판정하는 자리마다 마커를 단다" 를 전칭으로 선언하므로,
    마커 없는 자리가 하나라도 남으면 문서가 자기에 대해 거짓을 말하게 된다.
    """
    for head, markers in org_scope_map(_canonical_text()):
        assert markers, head


def test_org_scopes_match_the_independent_copy():
    """머리와 스코프가 위 상수와 정확히 같다 (좁힘·넓힘 양쪽)."""
    got = org_scope_map(_canonical_text())
    assert tuple((h, m[0] if m else None) for h, m in got) == ORG_SCOPES


def test_a_scope_that_names_ko_carries_a_pointer_not_a_bare_exclusion():
    """KO 를 빼기만 하면 그 언어에 규칙이 없어진다 — **무판정은 비-entity 가 아니다.**

    §3 의 약어 항이 정하는 것은 "시그널 부재 시 `ORG`" 라는 **기본값**이라, KO 를
    배제만 하면 지시체가 잡히는 약어(`EU`·`OECD`)까지 갈 곳을 잃는다. 그래서 KO 가
    언급된 마커는 KO 의 답이 어디 있는지를 절 번호로 가리켜야 한다.
    """
    for head, markers in org_scope_map(_canonical_text()):
        for marker in markers:
            # KO 가 아예 안 적힌 마커(`적용: JA`)는 그 판정에 KO 대응 자리가
            # 없다는 뜻이라 포인터를 요구하지 않는다. 그 사유는 사람이 본다.
            if "KO" not in marker:
                continue
            assert "§" in marker, (head, marker)


def test_the_parser_reaches_into_the_subsections():
    """소절에서 멈추면 §3.1~§3.4 의 `ORG` 판정이 통째로 검사 밖이 된다.

    최상위 불릿만 세면 새 `ORG` 판정을 아무 소절에나 써서 마커 없이 통과시킬 수
    있다 — 그 무력화 비용이 0 이라 여기서 못 박는다.
    """
    heads = [h for h, _ in org_scope_map(_canonical_text())]
    assert "회사·사업체·레코드 레이블" in heads   # §3.2 표 행
    assert "운영리그 vs 경기" in heads             # §3.3 보조 원칙 불릿


def test_the_parser_fails_loudly_when_the_section_is_gone():
    """조용한 0 이 아니라 시끄러운 실패여야 한다."""
    with pytest.raises(ValueError):
        section3_units("# 제목만 있는 문서\n\n본문\n")
    with pytest.raises(ValueError):
        section3_units("## 3. PROD vs EVT vs ORG 경계 (공통)\n\n산문만 있다\n")


def test_sync_breaks_in_both_directions():
    """canonical 만 고쳐도, 상수만 고쳐도 붉어진다.

    한 방향만 검사하면 반대쪽을 고쳐 통과시키는 길이 남는다. 두 assert 의 좌우는
    같은 shape 이라야 한다 — shape 이 다르면 비교가 언제나 참이라 검증력이 0 이다.
    """
    text = _canonical_text()
    projected = tuple((h, m[0] if m else None) for h, m in org_scope_map(text))
    assert projected == ORG_SCOPES

    # (a) canonical 에서 마커를 지우면 그 자리가 마커 없는 판정으로 남는다
    stripped = text.replace(" (**적용: JA·VI·EN · KO 는 §2.5**)", "", 1)
    after = org_scope_map(stripped)
    assert any(not markers for _, markers in after)
    assert tuple((h, m[0] if m else None) for h, m in after) != ORG_SCOPES

    # (b) 상수를 흔들면 실제 문서와 어긋난다
    mutated = ORG_SCOPES[:-1] + ((ORG_SCOPES[-1][0], "KO"),)
    assert projected != mutated


# §2.5 표면형이 §3 에 나올 때 비-entity 맥락인지 보는 창. 한국어 문장을 정확히
# 가르는 대신 앞뒤 글자수로 자른다 — 거친 그물이라는 것을 알고 쓴다.
_CONTEXT_BEFORE = 60
_CONTEXT_AFTER = 20


def test_section25_nonentity_surfaces_appear_only_in_nonentity_context():
    """§2.5 가 비-entity 로 정한 표면형이 §3 의 `ORG` 판정 자리에서 예시로 서지 않는다.

    **거친 그물이다** — 표면형 둘레 글자만 보고 "비-entity" 가 그 안에 있는지 묻는다.
    §2.5 예시는 대부분 한글이고 §3 예시는 `NHK`·`JR` 같은 로마자·일본어라 겹치는
    자리 자체가 드물어, 오늘 실제로 걸리는 것은 약어 항의 `KBS` 한 자리뿐이다. §3
    구멍을 실제로 막는 것은 이 검사가 아니라 위 마커 검사이며, 이 검사가 잡는 것은
    "나중에 누가 그 표면형을 `ORG` 예시로 올리는" 한 경로다.
    """
    from tests.ner.labelers.test_ko_ner_prompts import _canonical_examples

    dropped = _canonical_examples("non-entity")
    for unit in section3_units(_canonical_text()):
        if _ORG not in unit:
            continue
        for surface in dropped:
            start = 0
            while (hit := unit.find(surface, start)) != -1:
                window = unit[max(0, hit - _CONTEXT_BEFORE):hit + _CONTEXT_AFTER]
                assert "비-entity" in window, (surface, window)
                start = hit + len(surface)
