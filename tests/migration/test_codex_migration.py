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
