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
RUNTIME_CHANGE_POLICY_CLAUSES = (
    "런타임 동작을 변경하는 작업은 main이 아닌 브랜치에서 진행한다.",
    "구현 전에 이슈 또는 승인된 spec에 문제, acceptance criteria, 필요한 테스트, "
    "문서 영향을 기록한다.",
)
REVIEWER_REQUIRED_TERMS = (
    "code-reviewer",
    "generic read-only subagent",
    "원래 요구사항 또는 승인된 spec",
    "git diff HEAD",
    "새로 실행한 검증 명령과 출력",
    "Findings",
    "Verification gaps",
    "Verdict",
    "파일을 수정하거나 승인 산출물을 만들지",
    "BLOCK",
    "PASS는 기계 검증을 대신하지 않는다",
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
CUSTOM_COMMIT_GATE_ITEM = "Codex custom commit hook"
CUSTOM_COMMIT_GATE_DECISION = "제거"
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
    normalized_text = " ".join(text.split())
    for clause in RUNTIME_CHANGE_POLICY_CLAUSES:
        assert clause in normalized_text, clause


def test_reviewer_uses_native_or_generic_read_only_subagent_contract():
    assert not (REPO_ROOT / ".codex/agents/reviewer.toml").exists()

    text = " ".join((REPO_ROOT / "AGENTS.md").read_text().split())
    for term in REVIEWER_REQUIRED_TERMS:
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


def test_custom_commit_gate_decision_matches_inventory():
    path = REPO_ROOT / "docs/superpowers/specs/codex-migration-inventory.md"
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        if columns and columns[0] == CUSTOM_COMMIT_GATE_ITEM:
            rows.append(columns)

    assert len(rows) == 1
    assert len(rows[0]) == 3
    assert rows[0][1] == CUSTOM_COMMIT_GATE_DECISION


def test_followup_tasks_do_not_reintroduce_removed_customizations():
    path = REPO_ROOT / "docs/superpowers/plans/2026-09-01-claude-to-codex-migration.md"
    text = path.read_text()
    task_5 = text.split("## Task 5:", 1)[1].split("## Task 6:", 1)[0]
    task_6 = text.split("## Task 6:", 1)[1]

    assert ".codex/agents/reviewer.toml" not in task_5
    assert "code-reviewer" in task_5
    assert "generic read-only subagent" in task_5
    assert ".codex/hooks" not in task_5
    assert "PreToolUse" not in task_5

    for required in (
        "custom commit hook 미채택",
        "사용자 수준 hook 등록을 하지 않는다는 결정",
        "automatic commit-time enforcement가 없고",
    ):
        assert required in task_6
    for removed_requirement in (
        "저장소 경로 한정 훅 등록 초안",
        "정상 명령과 차단 명령의 dry-run",
        "설정 제거를 통한 롤백",
        ".codex/hooks/commit_gate.py",
        "uv run ruff check .codex/hooks",
    ):
        assert removed_requirement not in task_6
