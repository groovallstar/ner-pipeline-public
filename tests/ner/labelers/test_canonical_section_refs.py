"""절 참조 무결성 — `§N`·`§N.M` 이 실재하는 canonical 절을 가리킨다.

**왜 검사하는가.** canonical 과 `src/` 는 판정 근거를 절 번호로 단다(`§2.5 로 더
좁다`, `# §2.3 국적·언어·계통 = 비-LOC`). 절이 옮겨지거나 이름이 바뀌면 그 참조는
조용히 아무 데도 안 가리키게 되는데, 없는 절을 가리키는 근거는 근거가 없는 것과
같으면서 있는 것처럼 보인다. 읽는 사람은 그 번호를 찾아보고 나서야 안다.

**오늘 이 검사가 잡는 것은 0 이다** — 현재 canonical 과 `src/` 의 참조는 전부 실재
절을 가리킨다. 회귀 그물이며, 절을 옮기는 다음 변경이 그 사실을 알게 하는 것이 이
파일의 일이다.

**모집단은 `.py` 로 좁히지 않는다** — `src/` 에서 절을 인용하는 것은 코드 주석만이
아니라 `CLAUDE.md`·원장 JSON 도 그렇다. 확장자로 좁히면 좁힌 만큼이 조용히 검사
밖으로 빠진다. `docs/issues/` 는 그때의 판을 적은 과거 스냅샷이라 모집단 밖이다.
"""

import pathlib
import re

import pytest

from ner.labelers.ko.ko_evt_r2_audit import CANONICAL_PATH

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
_SCANNED_SUFFIXES = (".py", ".md", ".json")

_HEADING = re.compile(r"^#{1,6}\s+(\d+(?:\.\d+)*)\.?\s")
_REFERENCE = re.compile(r"§\s?(\d+(?:\.\d+)*)")


def canonical_sections(text: str) -> set[str]:
    """canonical 이 실제로 가진 절 번호. 하나도 못 읽으면 예외."""
    found = {m.group(1) for line in text.splitlines() if (m := _HEADING.match(line))}
    if not found:
        raise ValueError(
            f"no numbered headings parsed from {CANONICAL_PATH}; an empty section set "
            "would make every reference look broken, or every check look vacuous"
        )
    return found


def _scanned_files() -> list[pathlib.Path]:
    files = [pathlib.Path(CANONICAL_PATH)]
    files += sorted(
        f for f in _SRC.rglob("*")
        if f.is_file() and f.suffix in _SCANNED_SUFFIXES and "__pycache__" not in f.parts
    )
    return files


def _references(path: pathlib.Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return set()
    return set(_REFERENCE.findall(text))


def test_the_scanner_actually_finds_references():
    """참조를 하나도 못 찾으면 통과는 검사 통과가 아니라 검사 부재다."""
    total = sum(len(_references(f)) for f in _scanned_files())
    assert total > 20, total


def test_every_section_reference_resolves_to_a_real_section():
    sections = canonical_sections(pathlib.Path(CANONICAL_PATH).read_text(encoding="utf-8"))
    broken = {
        f"{f.relative_to(_ROOT)}: §{ref}"
        for f in _scanned_files()
        for ref in _references(f)
        if ref not in sections
    }
    assert not broken, sorted(broken)


def test_the_section_parser_fails_loudly_when_headings_are_gone():
    with pytest.raises(ValueError):
        canonical_sections("본문만 있고 번호 붙은 제목이 없다\n")
