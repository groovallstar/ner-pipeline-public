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
    "debug-triage": "이관",
    "explain-diff": "이관",
    "perf-measure": "이관",
    "tdd": "통합",
    "refuter": "통합",
    "lessons-digest": "보류",
}


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
