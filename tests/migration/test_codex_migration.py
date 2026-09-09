from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

INSTRUCTION_TARGETS = (
    "AGENTS.md",
    "docker/AGENTS.md",
    "docker/server/AGENTS.md",
    "docker/vllm/AGENTS.md",
    "src/ner/AGENTS.md",
    "src/ner/augmenters/AGENTS.md",
    "src/ner/classifier/AGENTS.md",
    "src/ner/labelers/AGENTS.md",
    "src/ner/labelers/ja/AGENTS.md",
    "src/ner/labelers/ko/AGENTS.md",
    "src/ner/labelers/vi/AGENTS.md",
    "src/ner/llm_eval/AGENTS.md",
    "src/ner/scripts/AGENTS.md",
    "src/ner/validity/AGENTS.md",
    "src/server/AGENTS.md",
    "tests/ner/AGENTS.md",
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
    "전역 Codex 지침의 검증·완료 전 리뷰 절차",
)
RUNTIME_CHANGE_POLICY_CLAUSES = (
    "런타임 동작을 변경하는 작업은 main이 아닌 브랜치에서 진행한다.",
    "구현 전에 이슈 또는 승인된 spec에 문제, acceptance criteria, 필요한 테스트, "
    "문서 영향을 기록한다.",
)


PROJECT_SKILL_NAMES = (
    "debug-triage", "explain-diff", "perf-measure", "tdd", "refuter",
    "lessons-digest", "ner-debug-triage",
)
REMOVED_CODEX_HOOKS = (
    ".codex/hooks/commit_gate.py",
    ".codex/hooks/gate_core.py",
)


@pytest.mark.parametrize("target", INSTRUCTION_TARGETS)
def test_every_codex_instruction_is_preserved(target):
    assert (REPO_ROOT / target).is_file()


@pytest.mark.parametrize("target", INSTRUCTION_TARGETS)
def test_codex_instructions_do_not_reference_legacy_harness(target):
    text = (REPO_ROOT / target).read_text()
    for term in FORBIDDEN_TARGET_TERMS:
        assert term not in text, (target, term)


def test_root_codex_instruction_keeps_required_project_policy():
    text = (REPO_ROOT / "AGENTS.md").read_text()
    for term in ROOT_REQUIRED_TERMS:
        assert term in text, term
    normalized_text = " ".join(text.split())
    for clause in RUNTIME_CHANGE_POLICY_CLAUSES:
        assert clause in normalized_text, clause


def test_project_delegates_common_review_policy_without_plugin_dependency():
    assert not (REPO_ROOT / ".codex/agents/reviewer.toml").exists()
    text = " ".join((REPO_ROOT / "AGENTS.md").read_text().split())
    assert "전역 Codex 지침의 검증·완료 전 리뷰 절차" in text
    for obsolete in (
        "Superpowers", "superpowers:", "requesting-code-review",
        "verification-before-completion", "프로젝트 reviewer",
        "Findings", "Verification gaps", "Verdict", "git diff HEAD",
    ):
        assert obsolete not in text, obsolete


@pytest.mark.parametrize("name", PROJECT_SKILL_NAMES)
def test_legacy_skills_are_not_duplicated_in_codex(name):
    assert not (REPO_ROOT / ".agents" / "skills" / name).exists()


@pytest.mark.parametrize("path", REMOVED_CODEX_HOOKS)
def test_custom_codex_commit_gate_is_not_present(path):
    assert not (REPO_ROOT / path).exists()


def test_local_claude_configuration_is_removed():
    assert not (REPO_ROOT / ".claude").exists()
    assert not (REPO_ROOT / ".omc").exists()
    for target in INSTRUCTION_TARGETS:
        assert not (REPO_ROOT / target).with_name("CLAUDE.md").exists()
