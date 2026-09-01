from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

INSTRUCTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("CLAUDE.md", "AGENTS.md"),
    ("docker/CLAUDE.md", "docker/AGENTS.md"),
    ("docker/server/CLAUDE.md", "docker/server/AGENTS.md"),
    ("docker/vllm/CLAUDE.md", "docker/vllm/AGENTS.md"),
    ("src/ner/CLAUDE.md", "src/ner/AGENTS.md"),
    ("src/ner/augmenters/CLAUDE.md", "src/ner/augmenters/AGENTS.md"),
    ("src/ner/classifier/CLAUDE.md", "src/ner/classifier/AGENTS.md"),
    ("src/ner/labelers/CLAUDE.md", "src/ner/labelers/AGENTS.md"),
    ("src/ner/labelers/ja/CLAUDE.md", "src/ner/labelers/ja/AGENTS.md"),
    ("src/ner/labelers/ko/CLAUDE.md", "src/ner/labelers/ko/AGENTS.md"),
    ("src/ner/labelers/vi/CLAUDE.md", "src/ner/labelers/vi/AGENTS.md"),
    ("src/ner/llm_eval/CLAUDE.md", "src/ner/llm_eval/AGENTS.md"),
    ("src/ner/scripts/CLAUDE.md", "src/ner/scripts/AGENTS.md"),
    ("src/ner/validity/CLAUDE.md", "src/ner/validity/AGENTS.md"),
    ("src/server/CLAUDE.md", "src/server/AGENTS.md"),
    ("tests/ner/CLAUDE.md", "tests/ner/AGENTS.md"),
)

FORBIDDEN_TARGET_TERMS = (
    ".claude/",
    "CLAUDE.md",
    ".omc/",
    ".omx/",
    "oh-my-codex",
    "refuter_gate.py",
    "AskUserQuestion",
    "/tdd",
)

ROOT_REQUIRED_TERMS = (
    "uv run",
    "PYTHONPATH",
    "certified/",
    "requesting-code-review",
    "verification-before-completion",
)

SKILL_DECISIONS: dict[str, str] = {
    "debug-triage": "통합",
    "explain-diff": "제거",
    "perf-measure": "제거",
    "tdd": "통합",
    "refuter": "통합",
    "lessons-digest": "보류",
}

PROJECT_SKILL_NAMES = (*SKILL_DECISIONS, "ner-debug-triage")
REMOVED_CODEX_HOOKS = (
    ".codex/hooks/commit_gate.py",
    ".codex/hooks/gate_core.py",
)


@pytest.mark.parametrize(("source", "target"), INSTRUCTION_PAIRS)
def test_every_claude_instruction_has_a_codex_target(source, target):
    assert (REPO_ROOT / source).is_file()
    assert (REPO_ROOT / target).is_file()


@pytest.mark.parametrize(("_source", "target"), INSTRUCTION_PAIRS)
def test_codex_instructions_do_not_reference_legacy_harness(_source, target):
    text = (REPO_ROOT / target).read_text()
    for term in FORBIDDEN_TARGET_TERMS:
        assert term not in text, (target, term)


def test_root_codex_instruction_keeps_required_project_policy():
    text = (REPO_ROOT / "AGENTS.md").read_text()
    for term in ROOT_REQUIRED_TERMS:
        assert term in text, term


def test_skill_decisions_match_inventory():
    text = (REPO_ROOT / "docs/superpowers/specs/codex-migration-inventory.md").read_text()
    rows_by_skill = {name: [] for name in SKILL_DECISIONS}
    for line in text.splitlines():
        if not line.startswith("| `"):
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        name = columns[0].strip("`")
        if name in rows_by_skill:
            rows_by_skill[name].append(columns)

    assert {name: len(rows) for name, rows in rows_by_skill.items()} == {
        name: 1 for name in SKILL_DECISIONS
    }
    inventory_decisions = {
        name: rows[0][1] for name, rows in rows_by_skill.items()
    }

    assert inventory_decisions == SKILL_DECISIONS


@pytest.mark.parametrize("name", PROJECT_SKILL_NAMES)
def test_legacy_skills_are_not_duplicated_in_codex(name):
    assert not (REPO_ROOT / ".agents" / "skills" / name).exists()


@pytest.mark.parametrize("path", REMOVED_CODEX_HOOKS)
def test_custom_codex_commit_gate_is_not_present(path):
    assert not (REPO_ROOT / path).exists()
