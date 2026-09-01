# Claude Code에서 Codex로의 병행 마이그레이션 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Claude Code 설정을 보존한 채 두 하네스의 실제 동작을 비교하고, 유지·대체·단순화·제거하기로 결정한 Codex용 자산만 추가한다.

**Architecture:** `CLAUDE.md`와 `.claude/**`는 병행 기간의 원본으로 유지하고, Codex 전용 파일을 `AGENTS.md`와 `.codex/agents/`에 추가한다. 프로젝트 스킬은 writing-skills authoring gate에서 임시 관찰하되 Task 3에서는 추가하지 않는다. custom commit gate는 arbitrary Bash indirection을 완전 판별할 수 없어 제거하며, 독립 반박 검토와 완료 검증은 Codex reviewer subagent와 Superpowers 절차로 수행한다. 사용자 수준 Codex 설정은 저장소 자산 검증이 끝난 뒤 별도 승인으로만 활성화한다.

**Tech Stack:** Python 3.13, pytest 9, Ruff, TOML (`tomllib`), Codex `AGENTS.md`·custom agents, Superpowers

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
- 전역 Codex Rules 활성화는 Task 6의 산출물을 검토한 뒤 별도 작업으로 진행하며 custom commit hook은 활성화 후보에서 제외한다.

---

## 파일 구조

| 경로 | 책임 |
| --- | --- |
| `docs/superpowers/specs/codex-migration-inventory.md` | 원본 16개 지침과 6개 스킬의 대상·상태·검증 근거를 기록한다. |
| `tests/migration/test_codex_migration.py` | 파일 대응, 금지 표현, 원본 보존, TOML·스킬 구조를 정적으로 검증한다. |
| `AGENTS.md`와 하위 15개 `AGENTS.md` | 현재 디렉터리에 적용되는 Codex 프로젝트·모듈 지침을 제공한다. |
| `.codex/agents/reviewer.toml` | 소스 수정 권한이 없는 독립 반박 reviewer 역할을 정의한다. |
| `docs/superpowers/specs/codex-activation-checklist.md` | custom commit hook 미채택, Rules 후보, 수동 검증과 잔여 차이를 기록한다. |
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
  | `debug-triage` | 통합 | `superpowers:systematic-debugging`과 `AGENTS.md` | 의도적 단순화 |
  | `explain-diff` | 제거 | Codex 기본 역량 | 제거 |
  | `perf-measure` | 제거 | Codex 기본 역량 | 제거 |
  | `tdd` | 통합 | `superpowers:test-driven-development` | 의도적 단순화 |
  | `refuter` | 통합 | `superpowers:requesting-code-review`와 `superpowers:verification-before-completion` | 의도적 단순화 |
  | `lessons-digest` | 보류 | 상태 기반 자동 수집을 유지할지 제거할지 별도 판단 | 보류 |

  문서 끝에는 전역 설정과 Rules를 `비활성`으로 기록하고, custom commit hook은
  Task 4에서 이관 여부를 최종 결정한다고 명시한다.

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

## Task 3: 프로젝트 스킬 결정 단순화

**Files:**

- Modify: `docs/superpowers/specs/2026-09-01-claude-to-codex-migration-design.md`
- Modify: `docs/superpowers/plans/2026-09-01-claude-to-codex-migration.md`
- Modify: `tests/migration/test_codex_migration.py`
- Modify: `docs/superpowers/specs/codex-migration-inventory.md`

**Interfaces:**

- Consumes: Task 1의 스킬 결정표, 설치된 Superpowers 스킬 이름, writing-skills의 임시 authoring 관찰
- Produces: 통합 3개·제거 2개·보류 1개의 결정과 중복 프로젝트 스킬 부재 계약

- [ ] **Step 1: 결정 일관성과 중복 자산 부재 테스트를 작성한다**

  ```python
  SKILL_DECISIONS = {
      "debug-triage": "통합",
      "explain-diff": "제거",
      "perf-measure": "제거",
      "tdd": "통합",
      "refuter": "통합",
      "lessons-digest": "보류",
  }

  @pytest.mark.parametrize("name", (*SKILL_DECISIONS, "ner-debug-triage"))
  def test_legacy_skills_are_not_duplicated_in_codex(name):
      assert not (REPO_ROOT / ".agents" / "skills" / name).exists()
  ```

- [ ] **Step 2: 테스트가 기존 이관표 결정 불일치로 실패하는지 확인한다**

  Run: `uv run pytest tests/migration/test_codex_migration.py -q`

  Expected: debug-triage, explain-diff, perf-measure의 기존 `이관` 값 때문에 FAIL한다.

- [ ] **Step 3: 임시 관찰과 여섯 결정을 이관표에 반영한다**

  임시 authoring 관찰은 새 프로젝트 스킬을 사용하지 않았다는 사실만 한 문단으로
  기록하고 장기 감사나 기능 동등성 증거로 취급하지 않는다. 설치된 Superpowers,
  중복 유지 비용과 사용자가 승인한 Codex-native 단순화를 근거로 통합 3개, 제거 2개,
  보류 1개를 기록한다. explain-diff의 목차, Mermaid, 객관식 선택지와 answer key는
  제거된 편의 계약이며 필요하면 일반 사용자 요청으로 생성하되 자동 보장하지 않는다.
  `.agents/skills/`는 만들지 않는다.

- [ ] **Step 4: 결정과 프로젝트 스킬 부재를 검증한다**

  Run: `uv run pytest tests/migration/test_codex_migration.py -q`

  Expected: 전체 PASS한다.

  Run: `test ! -d .agents/skills`

  Expected: 종료 코드 0이다.

- [ ] **Step 5: 변경 범위와 원본 보존을 검증한다**

  Run: `uv run ruff check tests/migration/test_codex_migration.py`

  Run: `rg -n 'T''BD|TO''DO|PLACE''HOLDER' docs/superpowers/specs/2026-09-01-claude-to-codex-migration-design.md docs/superpowers/plans/2026-09-01-claude-to-codex-migration.md docs/superpowers/specs/codex-migration-inventory.md tests/migration/test_codex_migration.py`

  Run: `git diff -- .claude CLAUDE.md ':(glob)**/CLAUDE.md'`

  Run: `git diff --check`

  Expected: 모든 명령이 성공하고 네 소유 파일 밖의 변경이 없다.

- [ ] **Step 6: Task 3을 커밋한다**

  ```bash
  git add docs/superpowers/specs/2026-09-01-claude-to-codex-migration-design.md \
    docs/superpowers/plans/2026-09-01-claude-to-codex-migration.md \
    docs/superpowers/specs/codex-migration-inventory.md \
    tests/migration/test_codex_migration.py
  git commit -m "docs: Codex 스킬 제거 근거 정리"
  ```

## Task 4: custom Codex 커밋 게이트 제거 결정

**Files:**

- Modify: `docs/superpowers/specs/2026-09-01-claude-to-codex-migration-design.md`
- Modify: `docs/superpowers/plans/2026-09-01-claude-to-codex-migration.md`
- Modify: `docs/superpowers/specs/codex-migration-inventory.md`
- Modify: `tests/migration/test_codex_migration.py`

**Decision:**

Bash 문자열 parser는 `$IFS`, 변수 executable, `eval`, nested shell, Python과
다른 wrapper의 indirect Git execution을 완전하게 판별할 수 없다. 부분 parser는
automatic enforcement라는 false security를 만들므로 custom Codex commit gate를
제거한다.

- [ ] **Step 1: 구현 커밋을 recoverable revert한다**

  Task 4 구현과 두 fix 커밋을 최신순으로 `git revert --no-edit`한다. 결과는
  `.codex/hooks/commit_gate.py`와 `.codex/hooks/gate_core.py`가 없고
  `tests/hooks/test_gate.py`가 Task 4 이전 상태와 같아야 한다.

- [ ] **Step 2: hook 비존재 계약을 추가한다**

  `tests/migration/test_codex_migration.py`에 두 custom hook 파일이 존재하지
  않는다는 정적 검사를 추가한다. Claude 원본 hook과 기존 `tests/hooks/`는
  변경하지 않는다.

- [ ] **Step 3: 설계와 이관표에 제거 영향을 기록한다**

  custom Codex commit gate 상태를 `제거`로 기록한다. Claude의 automatic
  commit-time deterministic enforcement가 사라지는 영향과 AGENTS 안전 규칙,
  explicit read-only reviewer, `verification-before-completion`, 사용자
  outside-hook commit policy라는 대체 경계를 명시한다.

- [ ] **Step 4: 후속 Task에서 hook 활성화 요구를 제거한다**

  Task 5는 reviewer 계약만 구현한다. Task 6 activation checklist는 custom commit
  hook 등록·dry-run·rollback을 요구하지 않고 미채택과 잔여 차이를 기록한다.
  병행 검증은 자동 커밋 차단 상실과 사람 절차 누락 가능성을 비교한다.

- [ ] **Step 5: 제거 상태를 검증한다**

  Run: `uv run pytest tests/migration/test_codex_migration.py -q`

  Run: `uv run pytest tests/hooks/test_gate.py -q`

  Run: `uv run ruff check tests/migration/test_codex_migration.py tests/hooks/test_gate.py`

  Run: `test ! -e .codex/hooks/commit_gate.py && test ! -e .codex/hooks/gate_core.py`

  Run: `git diff -- .claude CLAUDE.md ':(glob)**/CLAUDE.md'`

  Expected: migration 계약과 기존 Claude hook 회귀가 통과하고 custom Codex hook과
  Claude 원본 diff가 없다.

- [ ] **Step 6: 제거 결정을 커밋한다**

  ```bash
  git add docs/superpowers/specs/2026-09-01-claude-to-codex-migration-design.md \
    docs/superpowers/plans/2026-09-01-claude-to-codex-migration.md \
    docs/superpowers/specs/codex-migration-inventory.md \
    tests/migration/test_codex_migration.py
  git commit -m "docs: Codex 커밋 게이트 제거 결정"
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

  체크리스트에는 현재 `/home/rkim/.codex/config.toml`과 Rules를 읽기 전용으로 백업·비교하는 명령, custom commit hook 미채택, 사용자 수준 hook 등록을 하지 않는다는 결정, 수동 검증과 잔여 차이를 기록한다. Rules는 명령 prefix 정책으로만 사용하며 경로 기반 파일 쓰기나 commit-time deterministic enforcement를 보장한다고 기록하지 않는다. `certified/**/*.json`과 기준 파일 보호가 AGENTS, reviewer, 완료 검증, 사용자 outside-hook commit 정책에 의존한다는 차이를 기록하고 수용 여부를 별도로 판정한다. 실제 사용자 수준 파일을 수정하는 명령에는 `사용자 별도 승인 후 실행` 표식을 붙이고 이 Task에서는 실행하지 않는다.

- [ ] **Step 2: 병행 검증 문서를 작성한다**

  다음 행을 `검증 항목 | Claude 동작 | Codex 동작 | 관찰된 차이 | 채택 판정 | 근거` 열로 기록한다: 루트 지침, `src/ner`, `src/server`, `docker`, `tests/ner`, 프로젝트 스킬 결정, custom commit hook 미채택, 정상 커밋, Ruff 위반, 테스트 무결성 위반, 기준 파일 변경, protected certified JSON 변경, certified 인용 수치, 파일 쓰기 전 차단 범위, 독립 reviewer, 완료 검증. commit 관련 행은 Codex의 automatic commit-time enforcement가 없고 사람 절차에 의존한다는 잔여 차이를 명시한다. 실제로 실행하지 않은 새 세션 항목은 `미실행`, 채택 판정은 `보류`로 유지한다. reviewer의 지적 개수나 표현 방식은 이 표에 넣지 않는다.

- [ ] **Step 3: 저장소 자동 검증을 모두 실행한다**

  Run: `uv run pytest tests/migration/test_codex_migration.py tests/hooks/test_gate.py -q`

  Expected: 전체 PASS한다.

  Run: `uv run ruff check tests/hooks/test_gate.py tests/migration/test_codex_migration.py`

  Expected: 종료 코드 0이다.

  Run: `git diff -- .claude CLAUDE.md ':(glob)**/CLAUDE.md'`

  Expected: 출력이 없다.

- [ ] **Step 4: 새 Codex 세션이 필요한 수동 smoke test를 기록한다**

  저장소 루트, `src/ner`, `src/server`, `docker`, `tests/ner`에서 각각 새 세션을 시작해 적용 지침 요약을 요청하고 예상 규칙과 비교한다. 프로젝트 스킬이 추가되지 않았으며 debug-triage 요구가 `systematic-debugging`과 저장소 지침으로 처리되는지 확인한다. 실제 diff를 설명하는 Codex 응답도 관찰하되 Claude 스킬의 목차·diagram·객관식·answer key 전체를 재현하는 통과 gate로 삼지 않는다. 실제 동작 차이와 사용자의 수용 여부를 병행 검증 문서에 기록하고, 제거 결정을 유지할지 재검토한다.

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
3. 통합·제거·보류 결정이 이관표와 테스트 상수에서 일치하고 중복 프로젝트 스킬이 없다.
4. custom commit hook 미채택과 automatic commit-time enforcement 상실이 기록되고 사용자가 잔여 차이를 검토한다.
5. reviewer가 읽기 전용으로 동작하고 최종 Verdict가 PASS이다.
6. `.claude/**`와 모든 `CLAUDE.md`의 diff가 비어 있다.
7. 사용자가 Rules 영향 범위와 롤백 절차를 검토한다.
