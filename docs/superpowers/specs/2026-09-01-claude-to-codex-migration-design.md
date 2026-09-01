# Claude Code에서 Codex로의 병행 마이그레이션 설계

## 목적

이 저장소가 Claude Code와 Codex를 병행 사용하는 동안, 두 하네스의 실제 동작을 항목별로 비교하고 Codex에서 유지·대체·단순화·제거할 방식을 결정한다. 비교 결과가 다르다는 사실만으로 Claude 동작을 Codex에 복제하지 않는다. `.claude/`와 `CLAUDE.md` 파일은 전환 기준을 충족할 때까지 유지한다. 이 작업은 NER 파이프라인의 런타임 코드, 데이터, 모델, 인증 정보, `certified/` 원장을 변경하지 않는다.

## 배경과 조사 결과

현재 저장소에는 프로젝트 지침 `CLAUDE.md` 16개, Claude 전용 설정 `.claude/settings.json`, 훅 4개, 프로젝트 스킬 6개가 있다.

설계는 사용자가 지정한 [Claude Code에서 Codex로의 전환 가이드](https://goddaehee.tistory.com/602)와 OpenAI의 [다른 에이전트 설정 가져오기](https://learn.chatgpt.com/docs/import), [AGENTS.md 지침 계층](https://learn.chatgpt.com/docs/agent-configuration/agents-md), [훅](https://learn.chatgpt.com/docs/hooks), [저장소 스킬](https://learn.chatgpt.com/docs/build-skills) 문서를 기준으로 한다. 제품 문서의 동작은 변경될 수 있으므로 Codex 전역 설정을 실제로 활성화하는 단계에서 다시 확인한다.

| 원본 범주 | 수량 | 주요 내용 | Codex 처리 |
| --- | ---: | --- | --- |
| 프로젝트 지침 | 16 | 개발 환경, 모듈별 규칙, 데이터 안전, 이슈와 검증 절차 | 같은 위치의 `AGENTS.md`로 의미를 이관 |
| Claude 설정 | 1 | 제한 허용 명령, 민감 경로 쓰기 거부, PreToolUse와 Stop 훅 등록 | 권한 모델을 새로 설계하고 결정적 PreToolUse 검사만 이관 |
| 훅 진입점 | 3 | 커밋 게이트, 반박자 게이트, lessons 알림 | 커밋 게이트만 Codex 훅 어댑터와 회귀 테스트로 이관 |
| 훅 공통 로직 | 1 | diff 검사, 승인 상태, 반박자 상태 | 커밋 검증에 필요한 순수 로직만 재사용하고 반박자 상태는 이관하지 않음 |
| 프로젝트 스킬 | 6 | debug-triage, explain-diff, lessons-digest, perf-measure, refuter, tdd | 설치된 Superpowers, 중복 유지 비용과 사용자 승인에 따라 통합·제거·보류 |

Codex는 시작 시 전역 지침 다음에 저장소 루트부터 현재 작업 디렉터리까지의 `AGENTS.md`를 순서대로 합친다. 따라서 루트와 하위 디렉터리의 지침 계층을 모두 보존해야 모듈별 제약이 사라지지 않는다. Task 3의 임시 authoring 관찰은 새 프로젝트 스킬을 만들지 않는 결정을 보조했으며 장기 감사나 기능 동등성 증거로 유지하지 않는다.

## 범위

### 포함

- 루트 및 하위 15개 디렉터리의 `CLAUDE.md`를 검토하고 대응 `AGENTS.md`를 작성한다.
- 각 지침의 Claude 또는 OMC 전용 표현을 Codex 네이티브 기능 또는 Superpowers의 실제 동작으로 재설계한다.
- `.claude/hooks`의 커밋 게이트만 Codex `PreToolUse` 훅 호환 버전으로 만든다.
- 새 커밋 훅의 입력과 출력, 차단 동작, 기존 `tests/hooks/`의 회귀 검사를 검증한다.
- 6개 프로젝트 스킬을 Codex 기본 역량과 Superpowers에 대조하고 통합·제거·보류 결과를 기록한다.
- Superpowers의 코드 리뷰와 완료 검증 절차를 Codex용 하네스에 결합한다.
- 병행 기간 중 두 설정의 차이를 확인할 수 있는 이관표와 smoke test를 제공한다.

### 제외

- `.claude/**`, `CLAUDE.md`, `/home/rkim/.claude/**`의 삭제, 이름 변경 또는 자동 동기화.
- `/home/rkim/.codex/AGENTS.md`의 개인 취향 지침 재작성.
- NER 코드, 학습 데이터, 모델, Docker 구성, 의존성, `certified/` 결과의 변경.
- Claude의 권한 허용 목록을 Codex의 자동 승인으로 기계적으로 복제하는 작업.
- Claude Code 대화 기록, 플러그인, MCP 서버의 일괄 import.
- OMX 플러그인, OMX 런타임 상태, OMX 전용 워크플로의 설치 또는 이관.
- Claude Stop 훅의 자동 반박자 루프, `.codex/state/` 판정 파일, Stop 훅에서 subagent를 강제하는 커스텀 하네스.
- Claude Code와 동일한 절차 수, 자동화 복잡도, reviewer 지적량을 Codex의 품질 기준으로 삼는 작업.

## 설계 원칙

1. 원본 보존: Codex 결과물은 모두 새 파일로 추가한다. 원본 Claude 파일은 전환 승인 전까지 정본이다.
2. 동작 비교 후 결정: Claude와 Codex가 같은 입력에서 무엇을 수행·차단·요구하는지 먼저 기록하고, 차이의 영향을 평가한 뒤 유지·대체·단순화·제거 여부를 결정한다.
3. 안전 우선: Claude `permissions.deny`에 있던 사람 승인 파일과 `certified/**/*.json` 보호는 먼저 명시적으로 시험한다. 재현할 수 없는 보호는 이관 완료로 표기하지 않는다.
4. 단계적 활성화: 새 훅은 단위·통합 검증 후에만 Codex 설정에 연결한다. 처음부터 자동 차단을 켜지 않는다.
5. 단일 책임: 지침은 컨텍스트와 작업 절차를 설명하고, 지속 정책 차단은 Rules, 실행 시 검사는 Hooks, 반복 절차는 Skills에 둔다.
6. 플러그인 최소화: OMX는 사용하지 않는다. Codex의 `AGENTS.md`, Rules, Hooks, 네이티브 subagent와 Superpowers 절차 스킬만 사용한다.
7. 검토 분리: 반박 검토는 Stop 훅이 아니라 `requesting-code-review`가 요청한 읽기 전용 Codex reviewer subagent가 수행한다. 최종 완료 주장은 `verification-before-completion`의 새 검증 출력으로 뒷받침한다.

## 동작 비교와 판정 기준

각 항목은 Claude 동작, Codex 동작, 관찰된 차이, 채택 판정, 근거를 분리해서 기록한다. 기능 차이는 자동 실패가 아니며 다음 판정 중 하나를 사용한다.

| 판정 | 의미 |
| --- | --- |
| 동일 유지 | 절차와 결과를 그대로 유지할 명확한 이유가 있음 |
| 동등 대체 | 구현 방식은 다르지만 필요한 정책 결과를 충족함 |
| 의도적 단순화 | 절차나 상태를 줄여도 필요한 안전·품질 목적을 충족함 |
| 제거 | Codex에서는 필요하지 않거나 유지 비용이 목적보다 큼 |
| 보류 | 실제 동작 또는 영향 근거가 부족해 결정을 미룸 |

Claude reviewer와 Codex reviewer의 지적 수나 문체는 비교 대상이 아니다. reviewer 결과는 각 하네스의 내부 품질 검증에 사용하고, 마이그레이션 표에는 실제 기능·정책·부작용 차이만 기록한다. 안전 불변조건이나 사용자가 명시한 정책이 아니라면 Claude와 다른 동작을 Codex 결함으로 간주하지 않는다.

구현 전 판정은 원본 조사와 설계를 근거로 한 예상값이다. 이관표에는 차이의 영향과 판정 근거를 함께 적고, 어느 구현·smoke test에서 판정을 확정할지도 명시한다. 특히 자동 실행을 명시적 호출로 바꾸는 경우에는 호출 누락 가능성과 집행력 감소를 영향으로 기록하며, 실제 reviewer 및 병행 검증 전에는 확정 판정으로 취급하지 않는다.

## 작업 흐름 대체 방식

OMX의 별도 런타임·상태 관리·팀 오케스트레이션은 이관하지 않는다. 기능별 대체 수단은 다음과 같다.

| 필요 기능 | 채택 수단 | 적용 원칙 |
| --- | --- | --- |
| 요구사항 정리와 설계 | Superpowers `brainstorming` | 설계 승인 전에는 구현하지 않음 |
| 다단계 구현 계획 | Superpowers `writing-plans` | 설계 문서를 근거로 파일·테스트 단위 계획 작성 |
| 구현 | Superpowers TDD·계획 실행 스킬 | 실패 테스트와 최소 구현, 검증 순서 유지 |
| 독립적인 조사·검토 | Codex 네이티브 subagent | 공유 파일을 수정하지 않는 범위에서만 병렬 위임 |
| 디버깅 | Superpowers `systematic-debugging` | 재현과 원인 확인 후 수정 |
| 완료 검증 | Superpowers `verification-before-completion` | 테스트 출력 또는 명시적 검증 공백을 근거로 보고 |
| 독립적 반박 검토 | Superpowers `requesting-code-review` + 읽기 전용 Codex reviewer subagent | 구현자와 분리된 컨텍스트에서 요구사항·diff·테스트 위험 검토 |
| 장기 진행 기록 | 저장소의 설계·계획 문서와 Git 이력 | 별도 런타임 상태 디렉터리를 만들지 않음 |
| 지속 정책과 실행 검사 | Codex Rules와 Hooks | 자연어 지시와 분리하고 테스트로 검증 |

Superpowers는 Codex에서 제공되는 절차 스킬로 유지한다. 반면 프로젝트 `AGENTS.md`에는 Superpowers 내부 동작을 복제하지 않고, 이 저장소에서 지켜야 할 도메인·품질·안전 규칙만 기록한다.

## 대상 구조

완료된 저장소에는 다음 Codex 전용 자산이 추가된다.

| 경로 | 책임 |
| --- | --- |
| `AGENTS.md` | 최상위 프로젝트 개요, 개발 환경, 안전·검증·문서·커밋 규칙 |
| `docker/**/AGENTS.md`, `src/**/AGENTS.md`, `tests/ner/AGENTS.md` | 기존 하위 `CLAUDE.md`의 모듈별 지침을 Codex 계층에 제공 |
| `.codex/hooks/` | Codex PreToolUse 입력을 받아 커밋 검증 로직으로 연결하는 어댑터 |
| `.codex/agents/reviewer.toml` | 요구사항·회귀·테스트 위험을 읽기 전용으로 검토하는 Codex subagent 정의 |
| `tests/hooks/` | Claude와 Codex PreToolUse 입력에 대한 커밋 게이트 회귀 검사 |
| `docs/superpowers/specs/...` | 이 설계와 이관표 |
| `docs/superpowers/plans/...` | 승인된 구현 세부 단계와 검증 명령 |

Codex의 사용자 수준 설정은 `/home/rkim/.codex/config.toml`에 남는다. 이 저장소와 무관한 세션에 영향을 주지 않도록, 저장소 파일과 새 훅이 모두 검증되기 전에는 그 파일에 훅 등록이나 기본 권한 변경을 하지 않는다.

## 지침 이관 방식

각 `CLAUDE.md`는 같은 디렉터리의 `AGENTS.md`로 이관한다. 루트 문서는 다음 내용을 유지한다.

- NER 파이프라인의 개요, `uv` 개발 환경, `PYTHONPATH` 금지, 패키지 import 규칙
- 런타임 코드 변경 시 이슈·브랜치·수락 기준·테스트·문서가 필요한 조건
- 기준 파일과 `certified/`에 관한 안전 제약
- 코드·주석·로그 언어, 커밋 제목, 문서·이슈 규칙

다음은 복사하지 않고 Codex 방식으로 바꾼다.

| 원본 표현 | Codex 결과 |
| --- | --- |
| Claude 또는 OMC 고유 명령과 역할명 | 현재 사용 가능한 Codex 네이티브 기능 또는 Superpowers 절차만 명시 |
| Claude `permissions.allow` | Codex의 기존 sandbox와 approval 정책을 유지하고, 필요할 때 Rules로 최소 범위 규칙 추가 |
| Claude `permissions.deny` | 적용 가능한 Codex Rule 또는 PreToolUse 훅으로 명시적 차단을 구현하고 테스트 |
| `CLAUDE.md` 경로 참조 | 대응되는 `AGENTS.md` 경로로 변경 |
| Claude 전용 훅 설명 | Codex 훅의 이벤트 이름, 입력 형식, 차단 의미로 변경 |

지침에서 동일한 정책을 여러 곳에 중복하지 않는다. 전역 Codex와 Superpowers의 동작은 프로젝트 `AGENTS.md`에 복사하지 않고, 이 저장소에만 필요한 도메인과 품질 규칙만 둔다.

## 훅 및 권한 설계

### 커밋 게이트

`commit_gate.py`는 Codex `PreToolUse`의 `Bash` 이벤트에서 실행한다. 지원 범위는 직접 실행하는 단일 `git commit`과 literal path를 사용하는 `git -C <path> commit`이다. exact standalone abort·quit·skip 종료 명령만 비대상으로 통과한다. 그 밖의 history operation, wrapper, 환경 변수 assignment, `cd`, shell operator, target expansion 또는 모호한 Git 명령은 임의 shell semantics를 근사하지 않고 진단과 함께 차단한다. 사용자는 명령을 분리하거나 검토 후 훅 밖에서 history operation을 수행한다. 입력 파싱 또는 결정적 검사 실행이 실패해도 fail-closed하며, 차단 시에는 `hookSpecificOutput.permissionDecision = "deny"`와 이유를 반환한다.

기존 `gate_core.py`의 순수 diff 검사만 독립 이관하고 승인 파일이나 판정 상태는 재사용하지 않는다. certified root와 모든 catalog file의 canonical containment를 검사하며 symlink root·directory·file을 근거로 허용하지 않는다. Codex 고유 JSON은 어댑터에서 검증하고 core에는 검사할 확정 작업 디렉터리만 전달한다.

### 반박 검토와 lessons

Claude의 `refuter_gate.py`와 `lessons_nudge.py`는 Codex Stop 훅으로 이식하지 않는다. Stop 훅이 reviewer를 직접 생성하지 못하고 판정 상태·재진입 제어라는 별도 커스텀 하네스를 요구하기 때문이다.

반박 검토는 Superpowers `requesting-code-review`가 호출하는 읽기 전용 Codex reviewer subagent로 대체한다. 프로젝트 `AGENTS.md`는 기준 파일 변경과 주요 런타임 변경에서 이 절차를 요구한다. reviewer는 요구사항, 기준 diff, 변경된 테스트, 검증 출력을 입력으로 받아 결함·회귀·테스트 공백만 보고하며 구현을 수정하지 않는다. Stop 훅의 자동 호출을 제거하므로 명시적 호출을 빠뜨릴 수 있고 집행력이 낮아질 수 있다. 이 의도적 단순화 판정은 Task 5의 reviewer smoke test와 Task 6의 병행 검증에서 누락 방지 절차와 실제 호출 증거를 확인한 뒤 확정한다.

최종 완료 전에는 Superpowers `verification-before-completion`을 사용해 테스트·린트·문서·설정 검증의 새 출력을 확인한다. lessons-digest는 기존 Claude·OMC 상태 의존성이 있어 병행 기간에는 보류하고, 반복 지적의 사람 주도 승격 원칙만 문서 규칙으로 유지한다.

### 권한 정책

현재 Codex 전역 설정의 sandbox·approval 값은 변경하지 않는다. 새 보호의 우선순위는 다음과 같다.

1. 영구적인 명령 정책은 `/home/rkim/.codex/rules/`의 최소 `prefix_rule`로 표현한다.
2. 이 저장소의 현재 diff, 승인 지문, 테스트 상태에 따라 달라지는 판단은 PreToolUse 훅이 수행한다.
3. `apply_patch`를 통한 민감 경로 쓰기 여부는 별도의 PreToolUse 사례로 검사한다.
4. 셸을 포함한 모든 우회 경로를 완전히 차단할 수 없으면, 해당 한계를 이관표에 명시하고 사람 검토가 필요한 보호로 남긴다.

Codex Rules와 훅 설정은 사용자 수준 파일이므로, 실제 활성화 직전에는 파일 경로·명령 패턴·영향 범위를 다시 확인한다. 이 단계는 전역 동작을 바꾸므로 활성화 계획을 사용자에게 별도로 제시한다.

## 스킬 이관 방식

`.claude/skills/`의 6개 스킬은 `name`, `description`, 지시문, 참조 파일, 스크립트 의존성과 Codex에서의 유지 비용을 조사한다. 결과는 다음 세 상태 중 하나로 이관표에 기록한다.

| 상태 | 기준 | 조치 |
| --- | --- | --- |
| 통합 | Codex 또는 설치된 스킬을 채택하는 편이 프로젝트 스킬 중복보다 단순함 | 중복 파일을 만들지 않고 대응 수단과 의도적 차이를 기록 |
| 제거 | 사용자가 Claude 편의 계약의 동일 재현을 요구하지 않고 유지 비용이 효용보다 큼 | 중복 파일을 만들지 않고 필요할 때 일반 요청으로 수행 |
| 보류 | Claude 전용 명령·도구·권한의 영향이나 Codex에서의 필요성을 아직 결정하지 못함 | 원본은 유지하고 비교 근거와 재설계 또는 제거 후보를 기록 |

`debug-triage`는 별도 NER 스킬을 유지하지 않고 설치된 `systematic-debugging`과 `AGENTS.md`를 채택한다. `explain-diff`와 `perf-measure`는 Claude 편의 형식을 동일하게 보장하는 프로젝트 스킬의 유지 비용을 피하고 Codex 기본 역량에 일반 요청으로 맡긴다. `tdd`는 Superpowers `test-driven-development`로 통합하고, `refuter`는 `requesting-code-review`와 `verification-before-completion`으로 통합하되 자동 상태 루프 제거를 의도적 단순화로 기록한다. `lessons-digest`는 보류한다. 이는 기능 동등성 증명이 아니라 사용자가 승인한 Codex-native 단순화이며, 여섯 결정 모두 새 프로젝트 스킬을 만들지 않는다.

## 구현 단계와 전환 기준

### 단계 1: 문서 기반 이관

1. 이 설계를 정본으로 두고, 파일별 이관표를 추가한다.
2. 루트와 하위 지침의 Codex 초안을 작성한다.
3. 원본과 대상의 정책 항목을 비교하는 정적 검사를 추가한다.
4. 새 Codex 세션에서 루트, `src/ner`, `src/server`, `docker`, `tests/ner` 작업 디렉터리별 지침 로딩을 확인한다.

완료 조건은 모든 `CLAUDE.md`에 대응 `AGENTS.md`가 있고, Claude 전용 명령·경로가 대상 지침에 남지 않으며, 원본 파일의 Git diff가 비어 있는 것이다.

### 단계 2: 스킬 이관

1. 각 스킬의 의존성과 중복을 분류한다.
2. writing-skills authoring gate에서 프로젝트 스킬 없는 동작을 임시 관찰한다.
3. 통합·제거·보류 결정을 이관표에 기록하고 중복 프로젝트 스킬 부재를 정적 검사한다.

완료 조건은 이관표의 모든 스킬이 통합·제거·보류 중 하나로 결정되고, 여섯 legacy 이름과 `ner-debug-triage`의 프로젝트 스킬 디렉터리가 없는 것이다.

### 단계 3: 커밋 훅 포팅과 reviewer 설정

1. Codex 이벤트 JSON fixture와 어댑터 단위 테스트를 먼저 작성한다.
2. `PreToolUse` 커밋 차단을 dry-run으로 실행하여 direct commit 허용·차단,
   unsupported history operation, wrapper, 경로와 검사 오류 사례를 비교한다.
3. `.codex/agents/reviewer.toml`을 읽기 전용으로 작성하고, 요구사항과 diff를 입력한 검토 보고서 형식을 시험한다.
4. 사용자 수준 Codex 훅 설정 초안을 검토한 뒤, 이 저장소에서만 커밋 훅이 활성화되는지 확인한다.

완료 조건은 `tests/hooks/` 전체 통과, 정상 커밋 비차단, 린트·기준 파일·테스트 무결성 위반의 차단, reviewer가 소스 파일을 수정하지 않는 것이다.

### 단계 4: 병행 운영과 전환 결정

1. 실작업을 Claude와 Codex에서 각각 수행해 지침·스킬·게이트 결과를 비교하고,
   Codex의 direct commit-only 및 history operation 제한 차이를 기록한다.
2. 차이는 이관표에 원인과 결정을 기록한다.
3. 모든 항목에 관찰된 차이와 채택 판정이 있고, 유지하기로 한 안전·품질 정책이 검증된 경우에만 전환 후보로 표시한다.

Claude 자산의 제거는 이 설계 범위 밖이며, 병행 검증 기록을 사용자가 검토하고 전환 시점을 명시한 뒤 별도 작업으로 수행한다.

## 검증 전략

| 검증 대상 | 방법 | 성공 기준 |
| --- | --- | --- |
| 지침 계층 | 각 주요 하위 디렉터리에서 새 Codex 세션으로 지침 요약 요청 | 해당 경로의 프로젝트 규칙이 빠지지 않음 |
| 문서 변환 | `rg`로 Claude·OMC 전용 명령, 옛 파일 경로, 누락 대응 파일 검색 | 의도적으로 보존한 원본 외 대상 파일에 잔존하지 않음 |
| 스킬 결정 | 이관표와 테스트 상수 비교, 프로젝트 스킬 경로 부재 검사 | 여섯 결정이 일치하고 중복 프로젝트 스킬이 없음 |
| PreToolUse | JSON fixture와 임시 Git 저장소 | 위반 명령은 실행 전 차단, 정상 명령은 통과 |
| reviewer 절차 | 요구사항·diff·검증 출력을 준 읽기 전용 subagent 검토 | 구현자와 분리된 결함 보고를 받고, 코드 수정 없음 |
| 프로젝트 회귀 | `uv run pytest tests/hooks/ -v`, 변경 범위에 맞는 `ruff`와 pytest | 기존 게이트와 새 어댑터의 테스트가 모두 통과 |
| 원본 보존 | `git diff -- .claude CLAUDE.md '**/CLAUDE.md'` | 병행 기간 동안 원본 변경 없음 |

## 위험과 대응

| 위험 | 영향 | 대응 |
| --- | --- | --- |
| 지침을 두 번 관리하면서 내용이 갈라짐 | Claude와 Codex의 작업 결과가 달라짐 | 이관표에 원본·대상·검토 일자를 기록하고 변경 시 쌍으로 검토 |
| 훅 입력 형식 차이 | 차단 누락 또는 정상 작업 차단 | 어댑터 경계를 분리하고 실제 Codex JSON fixture로 회귀 테스트 |
| direct commit-only 제한 | 복합 명령과 history operation을 Codex 훅 안에서 실행할 수 없음 | 명령을 분리하고 Task 6에서 Claude 대비 차이를 검증한 뒤 필요한 작업은 사용자가 훅 밖에서 수행 |
| 전역 Codex 설정 변경 | 다른 저장소의 작업이 영향을 받음 | 저장소 자산 검증 후 최소 규칙만 별도 활성화 승인으로 적용 |
| 권한을 1:1 복사 | 과도한 권한 또는 보호 상실 | Rules, Hooks, sandbox, approval을 각 역할에 맞게 재설계 |
| Claude 전용 스킬 의존성 | Codex에서 실행 실패 | 의존성과 유지 비용을 검사하고 통합·제거·보류 상태를 명시 |
| 제거한 스킬의 편의 계약 누락 | 목차·diagram·문답 형식 같은 자동 편의가 사라짐 | 필요하면 일반 사용자 요청으로 생성하고 Task 6에서 실제 Codex 차이와 수용 여부를 기록 |

## 승인 후 다음 산출물

이 설계가 검토되면 `docs/superpowers/plans/2026-09-01-claude-to-codex-migration.md`에 단계 1부터 4까지의 실행 계획을 작성한다. 계획은 파일 단위 변경, 먼저 실패해야 하는 테스트, 실행 명령, 사용자 수준 Codex 설정을 활성화하기 전의 검토 지점을 포함한다.
