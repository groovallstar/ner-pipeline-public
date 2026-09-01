# Claude Code에서 Codex로의 병행 마이그레이션 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Claude Code 설정을 보존한 채 두 하네스의 실제 동작을 비교하고, 유지·대체·단순화·제거하기로 결정한 Codex용 자산만 추가한다.

**Architecture:** `CLAUDE.md`와 `.claude/**`는 병행 기간의 원본으로 유지하고, Codex 전용 파일을 `AGENTS.md`, `.agents/skills/`, `.codex/hooks/`, `.codex/agents/`에 새로 추가한다. 커밋 게이트는 Claude의 상태·Stop·refuter 결합부를 복제하지 않고 순수 검사만 독립 구현하며, 독립 반박 검토와 완료 검증은 Codex reviewer subagent와 Superpowers 절차로 수행한다. 사용자 수준 Codex 설정은 저장소 자산 검증이 끝난 뒤 별도 승인으로만 활성화한다.

**Tech Stack:** Python 3.13, pytest 9, Ruff, TOML (`tomllib`), Codex `AGENTS.md`·Hooks·custom agents·repository skills, Superpowers

**Spec:** `docs/superpowers/specs/2026-09-01-claude-to-codex-migration-design.md`

## Global Constraints

- 한 번에 한 Task만 실행하고, 해당 Task의 검증 결과를 사용자에게 제시한 뒤 다음 Task 승인을 기다린다.
- `.claude/**`, 모든 `CLAUDE.md`, `/home/rkim/.claude/**`, `/home/rkim/.codex/**`는 수정하거나 삭제하지 않는다.
- OMX 플러그인, `.omx/`, `.codex/state/`, Stop refuter 루프를 추가하지 않는다.
- NER 런타임 코드, 데이터, 모델, Docker 구성, 의존성, `certified/` 원장을 변경하지 않는다.
- Codex 전용 자산은 새 파일로 추가하며 Claude 전용 명령, 경로, 역할명을 복사하지 않는다.
- Claude와 다른 동작은 자동 실패로 처리하지 않고, 차이·영향·채택 판정을 이관표에 기록한다.
- Claude와 같은 절차량이나 reviewer 지적량을 Codex 품질 목표로 삼지 않는다.
- 동작을 구현하는 Task는 실패 테스트, 최소 구현, 대상 테스트, Ruff 순서로 검증한다.
- 각 Task는 reviewer 검토와 사용자 확인이 끝난 뒤에만 독립 커밋한다.
- 전역 Codex 훅·Rules 활성화는 Task 6의 산출물을 검토한 뒤 별도 작업으로 진행한다.

---

## 파일 구조

| 경로 | 책임 |
| --- | --- |
| `docs/superpowers/specs/codex-migration-inventory.md` | 원본 16개 지침과 6개 스킬의 대상·상태·검증 근거를 기록한다. |
| `tests/migration/test_codex_migration.py` | 파일 대응, 금지 표현, 원본 보존, TOML·스킬 구조를 정적으로 검증한다. |
| `AGENTS.md`와 하위 15개 `AGENTS.md` | 현재 디렉터리에 적용되는 Codex 프로젝트·모듈 지침을 제공한다. |
| `.agents/skills/ner-debug-triage/SKILL.md` | Superpowers 디버깅 절차에 NER별 재현·격리 검사를 덧붙인다. |
| `.agents/skills/explain-diff/SKILL.md` | 저장소 diff 설명 보고서의 형식과 범위를 제공한다. |
| `.agents/skills/perf-measure/SKILL.md` | NER 성능 변경의 측정·비교·회귀 방지 절차를 제공한다. |
| `.codex/hooks/gate_core.py` | `.omx`·refuter 상태 없이 diff 기반 결정적 검사를 수행한다. |
| `.codex/hooks/commit_gate.py` | Codex `PreToolUse` JSON을 파싱하고 커밋 생성 명령만 검사한다. |
| `.codex/agents/reviewer.toml` | 소스 수정 권한이 없는 독립 반박 reviewer 역할을 정의한다. |
| `docs/superpowers/specs/codex-activation-checklist.md` | 전역 설정에 적용할 훅·Rules 초안, 수동 검증, 롤백 절차를 기록한다. |
| `docs/superpowers/specs/codex-parallel-validation.md` | Claude와 Codex의 병행 검증 결과와 전환 미충족 항목을 기록한다. |

## Task 1: 이관표와 정적 계약 작성

**Files:**

- Modify: `docs/superpowers/specs/2026-09-01-claude-to-codex-migration-design.md`
- Create: `docs/superpowers/plans/2026-09-01-claude-to-codex-migration.md`
- Create: `docs/superpowers/specs/codex-migration-inventory.md`
- Create: `tests/migration/test_codex_migration.py`

**Interfaces:**

- Consumes: `CLAUDE.md` 16개, `.claude/skills/*/SKILL.md` 6개, 승인된 설계 문서
- Produces: `INSTRUCTION_PAIRS: tuple[tuple[str, str], ...]`, `SKILL_DECISIONS: dict[str, str]`라는 테스트 상수와 후속 Task가 갱신할 이관표

- [ ] **Step 1: 원본 지침 대응 테스트를 작성한다**

  `tests/migration/test_codex_migration.py`에 다음 대응표와 실패 테스트를 작성한다.

  ```python
  from pathlib import Path

  import pytest

  REPO_ROOT = Path(__file__).resolve().parents[2]

  INSTRUCTION_PAIRS = (
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

  @pytest.mark.parametrize(("source", "target"), INSTRUCTION_PAIRS)
  def test_every_claude_instruction_has_a_codex_target(source, target):
      assert (REPO_ROOT / source).is_file()
      assert (REPO_ROOT / target).is_file()
  ```

- [ ] **Step 2: 테스트가 대상 파일 누락으로 실패하는지 확인한다**

  Run: `uv run pytest tests/migration/test_codex_migration.py -q`

  Expected: 16개 매개변수 사례가 대응 `AGENTS.md` 부재로 FAIL한다.

- [ ] **Step 3: 이관표를 작성한다**

  문서에 지침 16개를 `원본 | Codex 대상 | Claude 동작 | Codex 후보 동작 | 관찰 또는 예상 차이 | 차이의 영향 | 판정 | 판정 근거·확정 시점 | 상태 | 검증` 열로 기록한다. 상태는 모두 `계획됨`으로 시작한다. 판정은 `동일 유지`, `동등 대체`, `의도적 단순화`, `제거`, `보류`만 사용하며 차이가 있다는 이유만으로 실패로 표시하지 않는다. 구현 전 판정은 예상값으로 기록하고 Task 2부터 6까지의 어떤 검증에서 확정할지 각 행에 명시한다. 스킬 6개는 다음 결정으로 고정한다.

  | 원본 스킬 | 결정 | Codex 대상 | 판정 |
  | --- | --- | --- | --- |
  | `debug-triage` | 이관 | `.agents/skills/ner-debug-triage/SKILL.md` | 동등 대체 |
  | `explain-diff` | 이관 | `.agents/skills/explain-diff/SKILL.md` | 동등 대체 |
  | `perf-measure` | 이관 | `.agents/skills/perf-measure/SKILL.md` | 동일 유지 |
  | `tdd` | 통합 | `superpowers:test-driven-development` | 동등 대체 |
  | `refuter` | 통합 | `superpowers:requesting-code-review`와 `superpowers:verification-before-completion` | 의도적 단순화 |
  | `lessons-digest` | 보류 | 상태 기반 자동 수집을 유지할지 제거할지 별도 판단 | 보류 |

  문서 끝에는 전역 설정, Rules, Hooks 활성화 상태를 `비활성`으로 기록한다.

- [ ] **Step 4: 이관표 자체를 검증한다**

  Run: `rg -n '^\| .*CLAUDE\.md.*AGENTS\.md' docs/superpowers/specs/codex-migration-inventory.md`

  Expected: 지침 대응 행 16개가 출력된다.

  Run: `rg -n '^\| `(debug-triage|explain-diff|perf-measure|tdd|refuter|lessons-digest)`' docs/superpowers/specs/codex-migration-inventory.md`

  Expected: 스킬 결정 행 6개가 출력된다.

- [ ] **Step 5: 변경 범위를 검증하고 사용자 검토를 요청한다**

  Run: `git diff --check`

  Run: `git diff --name-only`

  Expected: 이 Task의 설계, 계획, 이관표, 정적 계약 테스트 네 파일만 출력되며 `.claude`와 `CLAUDE.md`는 출력되지 않는다. 이관표와 실패 테스트 결과를 사용자에게 제시하고 멈춘다.

- [ ] **Step 6: 사용자 승인 후 Task 1을 커밋한다**

  ```bash
  git add docs/superpowers/specs/2026-09-01-claude-to-codex-migration-design.md \
    docs/superpowers/plans/2026-09-01-claude-to-codex-migration.md \
    docs/superpowers/specs/codex-migration-inventory.md \
    tests/migration/test_codex_migration.py
  git commit -m "docs: Codex 이관표 추가"
  ```

## Task 2: AGENTS.md 지침 계층 추가

**Files:**

- Create: `AGENTS.md`
- Create: `docker/AGENTS.md`
- Create: `docker/server/AGENTS.md`
- Create: `docker/vllm/AGENTS.md`
- Create: `src/ner/AGENTS.md`
- Create: `src/ner/augmenters/AGENTS.md`
- Create: `src/ner/classifier/AGENTS.md`
- Create: `src/ner/labelers/AGENTS.md`
- Create: `src/ner/labelers/ja/AGENTS.md`
- Create: `src/ner/labelers/ko/AGENTS.md`
- Create: `src/ner/labelers/vi/AGENTS.md`
- Create: `src/ner/llm_eval/AGENTS.md`
- Create: `src/ner/scripts/AGENTS.md`
- Create: `src/ner/validity/AGENTS.md`
- Create: `src/server/AGENTS.md`
- Create: `tests/ner/AGENTS.md`
- Modify: `tests/migration/test_codex_migration.py`
- Modify: `docs/superpowers/specs/codex-migration-inventory.md`

**Interfaces:**

- Consumes: Task 1의 `INSTRUCTION_PAIRS`
- Produces: 루트에서 현재 작업 디렉터리까지 합성되는 16개 Codex 지침 파일

- [ ] **Step 1: Codex 대상의 금지 표현과 필수 정책 테스트를 추가한다**

  ```python
  FORBIDDEN_TARGET_TERMS = (
      ".claude/",
      "CLAUDE.md",
      ".omc/",
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

  @pytest.mark.parametrize(("_source", "target"), INSTRUCTION_PAIRS)
  def test_codex_instructions_do_not_reference_legacy_harness(_source, target):
      text = (REPO_ROOT / target).read_text()
      for term in FORBIDDEN_TARGET_TERMS:
          assert term not in text, (target, term)

  def test_root_codex_instruction_keeps_required_project_policy():
      text = (REPO_ROOT / "AGENTS.md").read_text()
      for term in ROOT_REQUIRED_TERMS:
          assert term in text, term
  ```

- [ ] **Step 2: 테스트가 대상 파일 부재로 실패하는지 확인한다**

  Run: `uv run pytest tests/migration/test_codex_migration.py -q`

  Expected: `AGENTS.md` 파일 부재로 FAIL한다.

- [ ] **Step 3: 루트 AGENTS.md를 작성한다**

  루트 문서는 다음 순서로 작성한다: 프로젝트 범위, 개발 환경, 안전 경계, 작업 절차, 테스트·문서 검증, 독립 검토, 커밋 규칙. `PYTHONPATH`는 설정 금지 규칙으로 유지하고 실행 예시는 `uv run python ...`, `uv run pytest ...`, `uv run ruff check ...`를 사용한다. 기준 파일이나 주요 런타임 변경은 완료 전에 `requesting-code-review`로 읽기 전용 reviewer를 호출하고, 최종 보고 전에는 `verification-before-completion`을 사용하도록 명시한다.

- [ ] **Step 4: 하위 AGENTS.md 15개를 작성한다**

  각 원본에서 현재 디렉터리의 모듈 책임, 입력·출력 계약, 금지사항, 대상 테스트만 옮긴다. 루트와 중복되는 Git, 언어, 공통 테스트 규칙은 삭제한다. 원본의 경로가 실제 저장소와 다른 경우 `rg --files <디렉터리>`로 확인한 현재 경로를 사용하고, Claude·OMC 명령은 제거한다.

- [ ] **Step 5: 지침 계층 테스트를 통과시킨다**

  Run: `uv run pytest tests/migration/test_codex_migration.py -q`

  Expected: 지침 대응, 금지 표현, 루트 필수 정책 사례가 모두 PASS한다.

  Run: `rg -l '\.claude/|CLAUDE\.md|\.omc/|oh-my-codex|refuter_gate\.py|AskUserQuestion|/tdd' --glob 'AGENTS.md'`

  Expected: 출력이 없다.

- [ ] **Step 6: 이관표 상태를 갱신하고 원본 보존을 검증한다**

  지침 16개 행의 상태를 `작성됨`으로 바꾸고 검증 열에 `tests/migration/test_codex_migration.py`를 기록한다.

  Run: `git diff -- .claude CLAUDE.md ':(glob)**/CLAUDE.md'`

  Expected: 출력이 없다.

  Run: `git diff --check`

  Expected: 종료 코드 0이다. 결과를 사용자에게 제시하고 멈춘다.

- [ ] **Step 7: 사용자 승인 후 Task 2를 커밋한다**

  ```bash
  git add AGENTS.md docker/AGENTS.md docker/server/AGENTS.md docker/vllm/AGENTS.md \
    src/ner/AGENTS.md src/ner/augmenters/AGENTS.md src/ner/classifier/AGENTS.md \
    src/ner/labelers/AGENTS.md src/ner/labelers/ja/AGENTS.md \
    src/ner/labelers/ko/AGENTS.md src/ner/labelers/vi/AGENTS.md \
    src/ner/llm_eval/AGENTS.md src/ner/scripts/AGENTS.md \
    src/ner/validity/AGENTS.md src/server/AGENTS.md tests/ner/AGENTS.md \
    tests/migration/test_codex_migration.py \
    docs/superpowers/specs/codex-migration-inventory.md
  git commit -m "docs: Codex 지침 계층 추가"
  ```

## Task 3: 프로젝트 스킬 이관과 Superpowers 통합 기록

**Files:**

- Create: `.agents/skills/ner-debug-triage/SKILL.md`
- Create: `.agents/skills/explain-diff/SKILL.md`
- Create: `.agents/skills/perf-measure/SKILL.md`
- Modify: `tests/migration/test_codex_migration.py`
- Modify: `docs/superpowers/specs/codex-migration-inventory.md`

**Interfaces:**

- Consumes: Task 1의 스킬 결정표, 설치된 Superpowers 스킬 이름
- Produces: Codex가 발견할 수 있는 프로젝트 스킬 3개와 중복·보류 결정 3개

- [ ] **Step 1: 스킬 발견 계약 테스트를 작성한다**

  ```python
  import re

  import pytest

  MIGRATED_SKILLS = {
      "ner-debug-triage": "Systematic NER-specific debugging",
      "explain-diff": "rich explanation",
      "perf-measure": "Measure-first performance",
  }

  @pytest.mark.parametrize(("name", "description_fragment"), MIGRATED_SKILLS.items())
  def test_migrated_skill_has_valid_frontmatter(name, description_fragment):
      path = REPO_ROOT / ".agents" / "skills" / name / "SKILL.md"
      text = path.read_text()
      assert text.startswith("---\n")
      assert re.search(rf"(?m)^name: {re.escape(name)}$", text)
      assert description_fragment.lower() in text.lower()

  @pytest.mark.parametrize("name", ("tdd", "refuter", "lessons-digest"))
  def test_integrated_or_deferred_skills_are_not_duplicated(name):
      assert not (REPO_ROOT / ".agents" / "skills" / name).exists()
  ```

- [ ] **Step 2: 테스트가 스킬 파일 부재로 실패하는지 확인한다**

  Run: `uv run pytest tests/migration/test_codex_migration.py -q`

  Expected: 이관 대상 3개 사례가 `FileNotFoundError`로 FAIL한다.

- [ ] **Step 3: ner-debug-triage 스킬을 작성한다**

  frontmatter의 `name`은 `ner-debug-triage`, description은 `Systematic NER-specific debugging`을 포함한다. 본문은 먼저 `superpowers:systematic-debugging`을 사용하도록 요구하고, NER 추가 점검을 백엔드, 데이터 로더, 프롬프트, BIO 정렬, 평가 경계로 제한한다. 재현 명령은 `uv run pytest <node-id> -x -q`와 실제 모듈의 `uv run python -m ...` 형식을 사용하며 `PYTHONPATH`를 설정하지 않는다.

- [ ] **Step 4: explain-diff와 perf-measure 스킬을 작성한다**

  `explain-diff`는 변경 범위, 배경, 직관, 코드 흐름, 위험, 검증, 퀴즈와 정답을 포함하는 한국어 Markdown 보고서를 `/tmp/YYYY-MM-DD-explanation-<slug>.md`에 작성하도록 한다. `perf-measure`는 고정 데이터·모델·시드·샘플 수에서 baseline과 candidate의 wall time, 처리량, GPU 메모리, F1을 함께 기록하고 한 번에 한 변수만 변경하도록 한다.

- [ ] **Step 5: Claude 전용 의존성과 스킬 구조를 검증한다**

  Run: `uv run pytest tests/migration/test_codex_migration.py -q`

  Expected: 전체 PASS한다.

  Run: `rg -n 'PYTHONPATH=|\.claude/|CLAUDE\.md|\.omc/|AskUserQuestion|/tdd' .agents/skills`

  Expected: 출력이 없다.

- [ ] **Step 6: 이관표 상태를 갱신하고 사용자 검토를 요청한다**

  이관 대상 3개는 `작성됨`, 통합 대상 2개는 `Superpowers 통합`, lessons-digest는 `보류`로 갱신한다. 각 행의 검증 열에는 대응 스킬 또는 정적 테스트를 기록한다.

  Run: `git diff --check`

  Expected: 종료 코드 0이다. 스킬별 결정과 검증 결과를 사용자에게 제시하고 멈춘다.

- [ ] **Step 7: 사용자 승인 후 Task 3을 커밋한다**

  ```bash
  git add .agents/skills tests/migration/test_codex_migration.py \
    docs/superpowers/specs/codex-migration-inventory.md
  git commit -m "feat: Codex 프로젝트 스킬 추가"
  ```

## Task 4: 상태 없는 Codex PreToolUse 커밋 게이트 구현

**Files:**

- Create: `.codex/hooks/gate_core.py`
- Create: `.codex/hooks/commit_gate.py`
- Modify: `tests/hooks/test_gate.py`
- Modify: `docs/superpowers/specs/codex-migration-inventory.md`

**Interfaces:**

- Consumes: Codex hook JSON의 `tool_name: str`, `tool_input.command: str`, `cwd: str | None`
- Produces: `creates_a_commit(command: str | None) -> bool`, `evaluate(project_dir: str) -> GateResult`, `main() -> None`
- Produces: `GateResult`는 `NamedTuple`이며 `allowed: bool`, `check: str`, `reason: str`, `findings: tuple[str, ...]` 필드를 가진다.

- [ ] **Step 1: Codex 어댑터의 실패 테스트를 작성한다**

  기존 `_load_hook`을 경로 인수를 받도록 확장하고 `.codex/hooks/commit_gate.py`를 별도 모듈 이름으로 적재한다.

  ```python
  def _load_hook(name, root=".claude/hooks"):
      path = REPO_ROOT / root / f"{name}.py"
      spec = importlib.util.spec_from_file_location(f"{root}-{name}", path)
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)
      return module

  codex_commit_gate = _load_hook("commit_gate", ".codex/hooks")

  def _run_codex_hook(proj, payload):
      proc = subprocess.run(
          [sys.executable, REPO_ROOT / ".codex/hooks/commit_gate.py"],
          cwd=proj,
          input=json.dumps(payload),
          capture_output=True,
          text=True,
          check=True,
      )
      return json.loads(proc.stdout) if proc.stdout.strip() else {}

  def test_codex_hook_denies_a_ruler_commit(repo):
      _seed_ruler(repo)
      _write(repo, RULER_DOC, SCHEMA_BEFORE.replace("ORG", "ORG PROD"))
      _sh(repo, "git", "add", "-A")
      out = _run_codex_hook(repo, {
          "tool_name": "Bash",
          "tool_input": {"command": "git commit -m schema"},
          "cwd": repo,
      })
      decision = out["hookSpecificOutput"]
      assert decision["hookEventName"] == "PreToolUse"
      assert decision["permissionDecision"] == "deny"
      assert "independent reviewer" in decision["permissionDecisionReason"]
      assert ".omx" not in decision["permissionDecisionReason"]

  def test_codex_hook_allows_a_read_only_git_command(repo):
      out = _run_codex_hook(repo, {
          "tool_name": "Bash",
          "tool_input": {"command": "git status"},
          "cwd": repo,
      })
      assert out == {}

  def test_codex_hook_denies_certified_json_changes(repo):
      _write(repo, "certified/result.json", '{"score": 0.91}\n')
      _sh(repo, "git", "add", "-A")
      out = _run_codex_hook(repo, {
          "tool_name": "Bash",
          "tool_input": {"command": "git commit -m result"},
          "cwd": repo,
      })
      decision = out["hookSpecificOutput"]
      assert decision["permissionDecision"] == "deny"
      assert "certified/result.json" in decision["permissionDecisionReason"]
  ```

- [ ] **Step 2: 테스트가 Codex 훅 파일 부재로 실패하는지 확인한다**

  Run: `uv run pytest tests/hooks/test_gate.py -k 'codex_hook' -q`

  Expected: `.codex/hooks/commit_gate.py` 부재로 수집 단계에서 FAIL한다.

- [ ] **Step 3: 상태 없는 결정적 gate_core를 구현한다**

  `.claude/hooks/gate_core.py`에서 Git 실행, diff 수집, 기준 경로 판정, 변경 Python Ruff, 테스트 무결성, certified 수치 출처, 훅 자기 테스트 함수만 옮긴다. 여기에 staged 또는 unstaged diff의 `certified/*.json`과 `certified/**/*.json` 변경을 찾는 `check_protected_paths(project_dir: str) -> list[str]`를 추가한다. 다음 이름은 이관하지 않는다: `state_dir`, `allow_file`, `human_allowed`, `verdict_path`, `read_verdict`, `spawn_instructions`, `pass_notice`, refuter 판정·라운드·로그 함수.

  검사 순서는 다음 코드로 고정한다.

  ```python
  from typing import NamedTuple

  class GateResult(NamedTuple):
      allowed: bool
      check: str
      reason: str
      findings: tuple[str, ...]

  def evaluate(project_dir: str) -> GateResult:
      self_test = tuple(check_self_tests(project_dir))
      if self_test:
          return GateResult(False, "self-test", "Codex gate self-tests failed", self_test)

      protected = tuple(check_protected_paths(project_dir))
      if protected:
          return GateResult(
              False,
              "protected-path",
              "Protected certified JSON files changed; Codex must not commit these files.",
              protected,
          )

      ruff = tuple(check_ruff(project_dir))
      if ruff:
          return GateResult(False, "ruff", "Ruff failed for changed Python files", ruff)

      ruler = tuple(check_ruler_touched(project_dir))
      if ruler:
          return GateResult(
              False,
              "ruler-lock",
              "Protected criteria changed; obtain an independent reviewer report and let the user perform the commit outside this hook.",
              ruler,
          )

      hard = tuple(check_test_integrity(project_dir) + check_cited_metrics(project_dir))
      if hard:
          return GateResult(False, "hard", "Deterministic integrity checks failed", hard)

      return GateResult(True, "pass", "", ())
  ```

  `evaluate()` 내부의 Git, Ruff, pytest, diff 파싱을 포함한 결정적 검사 실행에서 예외가 발생하면 `GateResult(False, "internal-error", "Codex gate checks could not complete", (진단,))`로 변환한다. 검사 실행 오류는 진단을 포함해 커밋을 차단하며 정상 통과로 취급하지 않는다. `.codex/hooks/` 또는 `tests/hooks/`가 바뀐 경우에만 훅 자기 테스트를 실행한다.

- [ ] **Step 4: Codex PreToolUse 어댑터를 구현한다**

  `creates_a_commit`은 기존의 commit, cherry-pick, revert, am, rebase, merge 감지와 abort, quit, skip 예외를 보존한다. `main()`은 JSON 입력을 먼저 파싱하며 파싱 실패나 필수 필드 형식 오류는 `input-error` 진단을 포함한 deny로 fail-closed한다. 유효한 입력에서 Bash가 아닌 도구와 커밋 생성이 아닌 명령만 출력 없이 통과시킨다. 커밋 생성 명령은 빈 diff를 포함해 `evaluate()`를 호출하고, 검사 결과 또는 실행 예외가 허용을 명시하지 않으면 deny한다. 차단 결과는 다음 형식만 출력한다.

  ```python
  def deny(result):
      print(json.dumps({
          "hookSpecificOutput": {
              "hookEventName": "PreToolUse",
              "permissionDecision": "deny",
              "permissionDecisionReason": (
                  f"[{result.check}] {result.reason}\n"
                  + "\n".join(f"- {item}" for item in result.findings)
              ),
          }
      }, ensure_ascii=False))
  ```

- [ ] **Step 5: 상태 제거와 회귀 사례를 추가한다**

  Codex 훅 테스트에 정상 변경 통과, Ruff 오류 차단, 테스트 삭제 차단, protected certified JSON 변경 차단, certified 인용 수치 오류 차단, malformed JSON 차단, 필수 필드 형식 오류 차단, 결정적 검사 실행 예외 차단, Bash 외 도구 통과, 커밋 생성이 아닌 명령 통과를 추가한다. 오류 차단 사례는 `permissionDecision == "deny"`와 진단 문자열을 모두 검증한다. 기준 파일 차단 메시지에는 `human-allow`, `verdict`, `.omx`, `.codex/state`가 없어야 한다.

- [ ] **Step 6: 대상 테스트와 정적 검사를 실행한다**

  Run: `uv run pytest tests/hooks/test_gate.py -q`

  Expected: 기존 Claude 게이트와 새 Codex 어댑터 사례가 모두 PASS한다.

  Run: `uv run ruff check .codex/hooks tests/hooks/test_gate.py`

  Expected: 종료 코드 0이다.

  Run: `rg -n '\.omx|\.codex/state|human-allow|verdict|refuter_gate|Stop' .codex/hooks`

  Expected: 출력이 없다.

- [ ] **Step 7: 이관표를 갱신하고 사용자 검토를 요청한다**

  커밋 게이트 상태를 `구현됨, 비활성`으로 기록하고, 기준 파일 예외 커밋은 reviewer 보고 후 사용자가 훅 밖에서 수행한다는 제한을 명시한다.

  Run: `git diff --check`

  Run: `git diff -- .claude CLAUDE.md ':(glob)**/CLAUDE.md'`

  Expected: 두 명령 모두 출력이 없다. 테스트 결과와 의도적인 제한을 사용자에게 제시하고 멈춘다.

- [ ] **Step 8: 사용자 승인 후 Task 4를 커밋한다**

  ```bash
  git add .codex/hooks tests/hooks/test_gate.py \
    docs/superpowers/specs/codex-migration-inventory.md
  git commit -m "feat: Codex 커밋 게이트 추가"
  ```

## Task 5: 읽기 전용 reviewer 설정과 검토 계약 추가

**Files:**

- Create: `.codex/agents/reviewer.toml`
- Modify: `tests/migration/test_codex_migration.py`
- Modify: `AGENTS.md`
- Modify: `docs/superpowers/specs/codex-migration-inventory.md`

**Interfaces:**

- Consumes: 요구사항 문서 경로, 현재 diff, 실행한 검증 명령과 출력
- Produces: 심각도 순서의 `Findings`, 확인하지 못한 `Verification gaps`, 차단 여부인 `Verdict`를 반환하는 읽기 전용 reviewer 역할

- [ ] **Step 1: reviewer 설정 계약 테스트를 작성한다**

  ```python
  import tomllib

  def test_reviewer_agent_is_read_only_and_adversarial():
      path = REPO_ROOT / ".codex/agents/reviewer.toml"
      data = tomllib.loads(path.read_text())
      assert data["sandbox_mode"] == "read-only"
      instructions = data["developer_instructions"]
      for term in ("Findings", "Verification gaps", "Verdict", "Do not modify"):
          assert term in instructions
      assert "model" not in data
  ```

- [ ] **Step 2: 테스트가 reviewer 설정 부재로 실패하는지 확인한다**

  Run: `uv run pytest tests/migration/test_codex_migration.py -k reviewer -q`

  Expected: `.codex/agents/reviewer.toml` 부재로 FAIL한다.

- [ ] **Step 3: reviewer.toml을 작성한다**

  ```toml
  description = "Read-only adversarial reviewer for requirements, regressions, and false-green tests"
  sandbox_mode = "read-only"
  model_reasoning_effort = "high"
  developer_instructions = """
  Review the supplied requirements, diff, and fresh verification output adversarially.
  Look for unmet acceptance criteria, regressions, weakened assertions, unexecuted paths,
  unsafe changes to evaluation criteria, and claims unsupported by the evidence.
  Do not modify files, create approval artifacts, or broaden the requested scope.
  Return exactly three sections: Findings, Verification gaps, and Verdict.
  Findings are ordered by severity and include file references. Verification gaps list
  evidence that was required but unavailable. Verdict is BLOCK when any material defect
  or required evidence gap remains; otherwise it is PASS.
  """
  ```

- [ ] **Step 4: 루트 검토 절차를 reviewer 계약과 맞춘다**

  `AGENTS.md`의 독립 검토 절에 reviewer 입력 세 가지를 명시한다: 원래 요구사항 또는 승인된 spec, `git diff HEAD`, 이번 Task에서 새로 실행한 검증 출력. reviewer는 파일을 수정하지 않고 결과만 반환하며, BLOCK이면 수정 후 새 diff와 새 검증 출력으로 다시 요청한다. PASS는 기계 검증을 대신하지 않는다.

- [ ] **Step 5: 설정과 지침 테스트를 실행한다**

  Run: `uv run pytest tests/migration/test_codex_migration.py -q`

  Expected: 전체 PASS한다.

  Run: `rg -n '\.omx|refuter_gate|Stop hook|verdict.*json|human-allow' .codex/agents AGENTS.md`

  Expected: 출력이 없다.

- [ ] **Step 6: 실제 읽기 전용 reviewer smoke test를 수행한다**

  Codex 네이티브 subagent에 이 Task의 요구사항, `git diff HEAD`, 위 테스트 출력을 제공한다. reviewer가 `Findings`, `Verification gaps`, `Verdict` 세 절을 반환하고 작업 트리를 수정하지 않았는지 확인한다.

  Run before and after: `git status --short`

  Expected: reviewer 실행 전후 파일 목록이 동일하다.

- [ ] **Step 7: 이관표를 갱신하고 사용자 검토를 요청한다**

  reviewer 상태를 `구현됨, 프로젝트 설정`으로 기록하고 Stop 자동 실행이 아니라 주요 변경의 명시적 검토 절차임을 적는다.

  Run: `git diff --check`

  Expected: 종료 코드 0이다. reviewer 보고서와 작업 트리 불변 증거를 사용자에게 제시하고 멈춘다.

- [ ] **Step 8: 사용자 승인 후 Task 5를 커밋한다**

  ```bash
  git add .codex/agents/reviewer.toml AGENTS.md \
    tests/migration/test_codex_migration.py \
    docs/superpowers/specs/codex-migration-inventory.md
  git commit -m "feat: Codex reviewer 절차 추가"
  ```

## Task 6: 활성화 체크리스트와 병행 검증 기록 작성

**Files:**

- Create: `docs/superpowers/specs/codex-activation-checklist.md`
- Create: `docs/superpowers/specs/codex-parallel-validation.md`
- Modify: `docs/superpowers/specs/codex-migration-inventory.md`

**Interfaces:**

- Consumes: Task 1부터 5까지의 테스트 출력과 reviewer 보고서
- Produces: 전역 설정에 적용하기 전 사용자가 검토할 정확한 설정 초안, 롤백 명령, 병행 운영 판정

- [ ] **Step 1: 활성화 체크리스트를 작성한다**

  체크리스트에는 현재 `/home/rkim/.codex/config.toml`과 Rules를 읽기 전용으로 백업·비교하는 명령, 저장소 경로 한정 훅 등록 초안, 정상 명령과 차단 명령의 dry-run, 설정 제거를 통한 롤백을 기록한다. Rules는 명령 prefix 정책으로만 사용하며 경로 기반 파일 쓰기 보호를 보장한다고 기록하지 않는다. `apply_patch`를 포함한 파일 편집 도구가 PreToolUse 대상과 payload를 공식 문서 및 실제 dry-run으로 확인할 수 없으면, `certified/**/*.json` 보호가 커밋 시점 차단과 reviewer 검토까지만 제공된다는 차이를 기록하고, 그것을 수용할지 보강할지 별도로 판정한다. 실제 사용자 수준 파일을 수정하는 명령에는 `사용자 별도 승인 후 실행` 표식을 붙이고 이 Task에서는 실행하지 않는다.

- [ ] **Step 2: 병행 검증 문서를 작성한다**

  다음 행을 `검증 항목 | Claude 동작 | Codex 동작 | 관찰된 차이 | 채택 판정 | 근거` 열로 기록한다: 루트 지침, `src/ner`, `src/server`, `docker`, `tests/ner`, 프로젝트 스킬 발견, 정상 커밋, Ruff 위반, 테스트 무결성 위반, 기준 파일 변경, protected certified JSON 변경, certified 인용 수치, 파일 쓰기 전 차단 범위, 독립 reviewer, 완료 검증. 실제로 실행하지 않은 새 세션 항목은 `미실행`, 채택 판정은 `보류`로 유지한다. reviewer의 지적 개수나 표현 방식은 이 표에 넣지 않는다.

- [ ] **Step 3: 저장소 자동 검증을 모두 실행한다**

  Run: `uv run pytest tests/migration/test_codex_migration.py tests/hooks/test_gate.py -q`

  Expected: 전체 PASS한다.

  Run: `uv run ruff check .codex/hooks tests/hooks/test_gate.py tests/migration/test_codex_migration.py`

  Expected: 종료 코드 0이다.

  Run: `git diff -- .claude CLAUDE.md ':(glob)**/CLAUDE.md'`

  Expected: 출력이 없다.

- [ ] **Step 4: 새 Codex 세션이 필요한 수동 smoke test를 기록한다**

  저장소 루트, `src/ner`, `src/server`, `docker`, `tests/ner`에서 각각 새 세션을 시작해 적용 지침 요약을 요청하고 예상 규칙과 비교한다. `/skills`에서 세 프로젝트 스킬의 이름과 description을 확인한다. 결과와 실행 일시를 병행 검증 문서에 기록한다.

- [ ] **Step 5: verification-before-completion과 독립 reviewer를 실행한다**

  `superpowers:verification-before-completion`으로 Step 3의 새 출력을 확인한다. 이어서 읽기 전용 reviewer에 승인된 spec, 전체 diff, 자동·수동 검증 결과를 제공한다. reviewer는 선택한 Codex 설계 자체의 누락과 false-green을 검토하며 Claude와 동일한 절차를 요구하지 않는다. reviewer가 BLOCK이면 이 Task 안에서 문서 또는 누락 검증을 수정하고 Step 3부터 다시 실행한다.

- [ ] **Step 6: 이관표의 최종 상태를 갱신한다**

  자동 검증이 끝난 항목은 `검증됨`, 새 세션 smoke test가 남은 항목은 `수동 검증 필요`, 사용자 설정은 `비활성`으로 기록한다. Claude 제거와 기본 도구 전환은 모두 `범위 밖`으로 유지한다.

- [ ] **Step 7: 최종 변경 범위와 전환 보류 상태를 보고한다**

  Run: `git diff --check`

  Run: `git status --short`

  Expected: 예상한 Codex·문서·테스트 파일만 표시된다. 사용자에게 전체 검증 근거, 남은 수동 항목, 전역 활성화가 아직 수행되지 않았음을 보고하고 멈춘다.

- [ ] **Step 8: 사용자 승인 후 Task 6을 커밋한다**

  ```bash
  git add docs/superpowers/specs/codex-activation-checklist.md \
    docs/superpowers/specs/codex-parallel-validation.md \
    docs/superpowers/specs/codex-migration-inventory.md
  git commit -m "docs: Codex 병행 검증 절차 추가"
  ```

## 전역 활성화의 별도 승인 조건

Task 6까지 완료해도 사용자 수준 Codex 설정은 바뀌지 않는다. 다음 조건이 모두 충족된 뒤 사용자가 전역 활성화를 명시적으로 승인하면 별도 계획을 작성한다.

1. `tests/migration/test_codex_migration.py`와 `tests/hooks/test_gate.py`가 통과한다.
2. 주요 디렉터리별 새 Codex 세션 지침 smoke test가 통과한다.
3. 저장소 프로젝트 스킬 3개가 Codex에서 발견된다.
4. 정상 커밋과 네 종류의 위반 사례에서 Claude·Codex 동작 차이가 기록되고, 각 차이에 사용자가 수용한 판정이 있다.
5. reviewer가 읽기 전용으로 동작하고 최종 Verdict가 PASS이다.
6. `.claude/**`와 모든 `CLAUDE.md`의 diff가 비어 있다.
7. 사용자가 영향 범위와 롤백 절차를 검토한다.
