# Claude와 Codex 병행 검증 기록

## 판정 기준

이 기록은 두 하네스의 기능, 정책, 부작용 차이와 채택 판정만 비교한다. Claude와
Codex reviewer의 지적 수, 문체, 절차량은 parity metric이 아니다. `미실행`은
실제 새 세션 증거가 없다는 뜻이며 정적 검사를 근거로 PASS로 바꾸지 않는다.

| 검증 항목 | Claude 동작 | Codex 동작 | 관찰된 차이 | 채택 판정 | 근거 |
| --- | --- | --- | --- | --- | --- |
| 루트 지침 | `CLAUDE.md`가 도메인 규칙과 Claude·OMC 절차를 함께 제공 | `AGENTS.md`가 저장소 범위, 개발 환경, 안전, 검증, reviewer 계약을 제공 | 정적 내용 검사는 PASS. 새 루트 세션의 loader 동작은 미실행 | 보류 | migration 46 PASS, 새 세션 smoke 미실행 |
| `src/ner` 지침 | 루트와 `src/ner/CLAUDE.md`에서 NER 계약을 로드 | 루트와 `src/ner/AGENTS.md`에서 canonical label, 패키지, metric 경계를 로드 | 정적 대응은 확인. 새 세션의 계층 합성은 미실행 | 보류 | migration 지침 대응 검사 PASS, 새 세션 smoke 미실행 |
| `src/server` 지침 | 서버 API, 동시성, 준비성 계약을 로드 | 같은 제품 계약과 NER 큐·번역 즉시 429 차이를 로드 | 정적 대응은 확인. 새 세션 loader 동작은 미실행 | 보류 | migration 지침 대응 검사 PASS, 새 세션 smoke 미실행 |
| `docker` 지침 | Docker 공통·서비스별 배포 경계를 로드 | 상위와 하위 `AGENTS.md`로 GPU, 캐시, compose 책임을 합성 | 정적 대응은 확인. 새 세션 loader 동작은 미실행 | 보류 | migration 지침 대응 검사 PASS, 새 세션 smoke 미실행 |
| `tests/ner` 지침 | fixture, 테스트 무결성, 실행 규칙을 로드 | 같은 테스트 계약과 `uv run` 검증 명령을 로드 | 정적 대응은 확인. 새 세션 loader 동작은 미실행 | 보류 | migration 지침 대응 검사 PASS, 새 세션 smoke 미실행 |
| 프로젝트 스킬 결정 | 6개 Claude 프로젝트 스킬을 명시적으로 제공 | 새 프로젝트 스킬을 만들지 않고 debug와 TDD, refuter는 Superpowers에 통합하며 explain과 perf는 제거하고 lessons는 보류 | 이름 기반 발견성과 고정 편의 형식이 줄어듦. debug 통합과 actual-diff 새 세션 관찰은 미실행 | 후보 채택, 일부 보류 | migration 결정·부재 검사 PASS, 이관표 결정, 새 세션 smoke 미실행 |
| custom commit hook 미채택 | Bash PreToolUse commit gate를 등록 | known Bash indirection bypass 때문에 custom Codex commit hook을 완전 제거 | Codex에는 automatic commit-time enforcement가 없음 | 의도적 단순화 후보, 잔여 차이 수용 보류 | hook 구현 비존재 migration 검사, 설계와 이관표 |
| 정상 커밋 | 직접 감지한 커밋 명령에서 결정적 검사 후 통과 | 일반 Git 실행이며 Codex 전용 commit gate가 개입하지 않음 | 정상 경로의 추가 gate 비용은 없지만 커밋 전 검증 실행도 자동 보장되지 않음 | 사용자 절차 조건부 후보 | Claude hook 테스트 51 PASS, Codex hook 비존재 검사 |
| Ruff 위반 | 변경 Python의 Ruff 위반을 감지한 직접 커밋을 차단 | AGENTS가 Ruff를 요구하지만 커밋 시 자동 차단하지 않음 | automatic commit-time enforcement 상실. 사람 검증 누락 시 위반을 커밋할 수 있음 | 수용 여부 보류 | `tests/hooks/test_gate.py::test_ruff_failure_cannot_be_waived`, Codex hook 제거 결정 |
| 테스트 무결성 위반 | test 함수·assert 순삭제와 무조건 skip·xfail을 커밋 게이트에서 검사 | AGENTS, reviewer, 완료 검증이 위반을 찾도록 요구하나 자동 커밋 차단은 없음 | 사람 절차와 명시적 호출에 의존하며 reviewer와 verification은 gate와 동등하지 않음 | 수용 여부 보류 | Claude hook 무결성 회귀 테스트 PASS, 루트와 `tests/ner/AGENTS.md` |
| 기준 파일 변경 | ruler 변경에 승인과 refuter PASS를 요구 | 같은 기준으로 재평가하고 주요 변경 reviewer를 호출하도록 지시 | Codex는 커밋 직전 승인 지문과 verdict를 자동 강제하지 않음 | 수용 여부 보류 | Claude ruler-lock 테스트 PASS, 루트 안전·독립 검토 계약 |
| protected `certified` JSON 변경 | Claude Write/Edit permission deny가 JSON 쓰기를 막고 shell 경로는 열어 둠 | AGENTS, reviewer, 완료 검증, 사용자 outside-hook commit 정책에 의존 | Codex에는 동일한 경로 기반 Write/Edit 차단이 없고 Rules도 이를 보장하지 않음 | 수용 여부 보류 | `.claude/settings.json`, 이관 설계, 활성화 체크리스트 |
| certified 인용 수치 | 리포트·이슈 표의 새 수치를 선언한 certified 원장과 commit gate에서 대조 | 출처 명시를 AGENTS가 요구하며 reviewer와 완료 검증에서 확인 | Codex는 대조를 커밋 시 자동 실행하지 않음 | 수용 여부 보류 | Claude cited-metric 테스트 PASS, 루트 안전 계약 |
| 파일 쓰기 전 차단 범위 | permission deny가 지정된 Write/Edit 경로에 사전 적용되고 shell 쓰기는 범위 밖 | 현재 사용자 Rules 변경이 없고 저장소 지침은 도구 호출 전 경로 차단기가 아님 | Codex Rules는 명령 prefix 정책이며 경로 기반 파일 쓰기 보장이 아님 | 수용 여부 보류 | `.claude/settings.json`, OpenAI Rules 문서, 활성화 체크리스트 |
| 독립 reviewer | Stop refuter loop가 상태와 판정을 자동 결합 | `.codex/agents/reviewer.toml`을 명시적으로 호출하며 파일을 수정하지 않음 | 자동 Stop 호출과 상태 연속성이 없고 호출 누락 가능성이 있음 | 의도적 단순화 채택 | 정적 reviewer 계약 PASS, controller smoke PASS, worktree unchanged |
| 완료 검증 | Claude hook과 작업 절차가 일부 검사를 자동·수동 결합 | `verification-before-completion`으로 성공 주장마다 최신 명령 출력을 확인 | 명시적 호출이며 automatic commit gate를 대신하지 않음 | 의도적 단순화 채택 | 루트 `AGENTS.md`, 이번 Task의 최신 대상 검증 |

## 자동 검증 결과

- Task 5 시점 migration 테스트: `46 passed`.
- Task 4 시점 기존 Claude hook 테스트: `51 passed`.
- 프로젝트 reviewer: 정적 계약 PASS, controller smoke에서 정확한 세 절과 PASS를
  반환했고 작업 트리가 바뀌지 않았다.
- 원본 Claude 자산: `.claude/**`와 모든 `CLAUDE.md`의 diff가 없다.
- 전체 suite known gap: 이관과 무관한 EN 배포 package와 ledger fingerprint
  mismatch 2건이 남아 있다. 이 결과를 이번 이관의 PASS로 바꾸거나 숨기지 않는다.

## 미실행 수동 smoke

다음 항목은 모두 `미실행`이며 채택 판정은 `보류`이다.

1. 저장소 루트 새 Codex 세션에서 적용 지침 요약 확인
2. `src/ner`, `src/server`, `docker`, `tests/ner` 각각의 새 세션에서 상위·하위
   `AGENTS.md` 계층 합성 확인
3. 프로젝트 스킬이 추가되지 않았고 debug 요청이
   `superpowers:systematic-debugging`과 저장소 지침으로 처리되는지 확인
4. 실제 diff 설명에서 필요한 변경 요약을 제공하는지 확인

마지막 항목은 Claude `explain-diff`의 목차, diagram, 객관식 선택지, answer key를
모두 재현하는 gate가 아니다. 실제 기능과 사용자의 수용 여부만 기록한다.

## 전환 상태

Codex 자산은 활성화 후보이지만 전환은 보류한다. 새 세션 smoke와 사용자의 잔여
차이 수용, 별도 전역 활성화 승인이 남아 있다. Claude 자산 삭제와 기본 도구 전환은
범위 밖이며 이 문서가 이를 승인하지 않는다.
