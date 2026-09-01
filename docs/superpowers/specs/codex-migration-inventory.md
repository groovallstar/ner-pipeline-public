# Claude Code에서 Codex로의 병행 이관표

## 목적과 원칙

이 문서는 병행 운영 기간에 Claude Code와 Codex의 실제 동작, 차이, 채택 결정을 추적한다. 원본 `.claude/**`와 `CLAUDE.md` 파일은 현재 Claude 동작의 근거로 보존하며, Codex 대상은 새 파일로 추가한다. 차이는 자동 결함이 아니며 `동일 유지`, `동등 대체`, `의도적 단순화`, `제거`, `보류` 중 하나로 판정한다. Claude와 같은 절차량이나 reviewer 지적량은 이관 목표가 아니다.

## 프로젝트 지침

모든 항목은 대상 파일을 만들기 전이므로 `계획됨` 상태이다. 검증 열은 후속 작업에서 적용할 정적 계약을 가리킨다.

| 원본 | Codex 대상 | Claude 동작 | Codex 후보 동작 | 관찰 또는 예상 차이 | 판정 | 상태 | 검증 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `CLAUDE.md` | `AGENTS.md` | 프로젝트 규칙과 Claude·OMC 작업 절차를 함께 로드 | 도메인·안전 규칙은 로드하고 작업 절차는 Codex·Superpowers에 위임 | OMC 역할·명령과 자동 절차를 복제하지 않음 | 의도적 단순화 | 계획됨 | `tests/migration/test_codex_migration.py` 대상 존재 검사 |
| `docker/CLAUDE.md` | `docker/AGENTS.md` | Docker 공통 배포 경계와 운영 지침을 로드 | 같은 배포 경계를 Codex 계층에서 로드 | 지침 파일명과 로더만 다름 | 동등 대체 | 계획됨 | `tests/migration/test_codex_migration.py` 대상 존재 검사 |
| `docker/server/CLAUDE.md` | `docker/server/AGENTS.md` | REST API 컨테이너 설정·헬스체크·번역 의존성을 안내 | 같은 입력·준비성 계약을 현재 경로 기준으로 안내 | Claude 전용 실행 표현을 제거 | 동등 대체 | 계획됨 | `tests/migration/test_codex_migration.py` 대상 존재 검사 |
| `docker/vllm/CLAUDE.md` | `docker/vllm/AGENTS.md` | compose 역할과 GPU·모델별 운영 주의를 안내 | 같은 서비스 책임과 안전 범위를 안내 | 공통 Docker 설명의 중복을 줄임 | 의도적 단순화 | 계획됨 | `tests/migration/test_codex_migration.py` 대상 존재 검사 |
| `src/ner/CLAUDE.md` | `src/ner/AGENTS.md` | NER 모듈·라벨·import 규칙을 로드 | 같은 도메인 불변조건을 Codex 계층에서 로드 | 지침 파일명과 로더만 다름 | 동등 대체 | 계획됨 | `tests/migration/test_codex_migration.py` 대상 존재 검사 |
| `src/ner/augmenters/CLAUDE.md` | `src/ner/augmenters/AGENTS.md` | 증강 모드·gold 보존·대상 테스트를 안내 | 같은 입력·출력과 gold 보호를 안내 | 실행 예시를 `uv run`으로 정규화 | 동등 대체 | 계획됨 | `tests/migration/test_codex_migration.py` 대상 존재 검사 |
| `src/ner/classifier/CLAUDE.md` | `src/ner/classifier/AGENTS.md` | 학습 CLI·group split·metric·산출물 계약을 안내 | 같은 누출 방지와 평가 경계를 안내 | 현재 CLI·테스트 경로만 유지 | 의도적 단순화 | 계획됨 | `tests/migration/test_codex_migration.py` 대상 존재 검사 |
| `src/ner/labelers/CLAUDE.md` | `src/ner/labelers/AGENTS.md` | 공통 LLM 라벨링과 언어별 처리 규칙을 함께 안내 | 공통 계약만 상위에 두고 언어별 규칙은 하위 계층에서 로드 | 중복 설명을 줄임 | 의도적 단순화 | 계획됨 | `tests/migration/test_codex_migration.py` 대상 존재 검사 |
| `src/ner/labelers/ja/CLAUDE.md` | `src/ner/labelers/ja/AGENTS.md` | 일본어 offset·BIO·정규화 규칙을 안내 | 같은 일본어 경계 보존 규칙을 안내 | Claude 전용 작업 표현을 제거 | 동등 대체 | 계획됨 | `tests/migration/test_codex_migration.py` 대상 존재 검사 |
| `src/ner/labelers/ko/CLAUDE.md` | `src/ner/labelers/ko/AGENTS.md` | KLUE·canonical 엔티티·원문 span 규칙을 안내 | 같은 한국어 데이터·span 불변조건을 안내 | Claude 전용 작업 표현을 제거 | 동등 대체 | 계획됨 | `tests/migration/test_codex_migration.py` 대상 존재 검사 |
| `src/ner/labelers/vi/CLAUDE.md` | `src/ner/labelers/vi/AGENTS.md` | 베트남어 정규화·BIO·평가 연결을 안내 | 같은 언어별 경계와 평가 계약을 안내 | 공통 라벨러 설명을 중복하지 않음 | 의도적 단순화 | 계획됨 | `tests/migration/test_codex_migration.py` 대상 존재 검사 |
| `src/ner/llm_eval/CLAUDE.md` | `src/ner/llm_eval/AGENTS.md` | 프롬프트 실행·span 평가·산출물 계약을 안내 | 같은 평가 입력·출력과 재현 경로를 안내 | Claude 역할 표현을 제거 | 동등 대체 | 계획됨 | `tests/migration/test_codex_migration.py` 대상 존재 검사 |
| `src/ner/scripts/CLAUDE.md` | `src/ner/scripts/AGENTS.md` | 보조 스크립트와 wrapper 규약을 안내 | 실제 스크립트 경로와 패키지 실행만 안내 | 실행 형식을 `uv run`으로 단순화 | 의도적 단순화 | 계획됨 | `tests/migration/test_codex_migration.py` 대상 존재 검사 |
| `src/ner/validity/CLAUDE.md` | `src/ner/validity/AGENTS.md` | fold·분산·baseline 판정과 기준 보호를 안내 | 같은 측정 불변조건과 판정 경계를 안내 | 하네스 절차보다 측정 정책에 집중 | 의도적 단순화 | 계획됨 | `tests/migration/test_codex_migration.py` 대상 존재 검사 |
| `src/server/CLAUDE.md` | `src/server/AGENTS.md` | API·오류·백엔드·관측성 계약을 안내 | 같은 서비스 계약과 설정 검증을 안내 | Claude 전용 작업 표현을 제거 | 동등 대체 | 계획됨 | `tests/migration/test_codex_migration.py` 대상 존재 검사 |
| `tests/ner/CLAUDE.md` | `tests/ner/AGENTS.md` | 테스트 구조·fixture·안전 주의를 안내 | 같은 테스트 범위와 `uv run pytest` 실행법을 안내 | 공통 규칙을 루트에 위임 | 의도적 단순화 | 계획됨 | `tests/migration/test_codex_migration.py` 대상 존재 검사 |

## 프로젝트 스킬 결정

| 원본 스킬 | 결정 | Codex 대상 | Claude 동작 | Codex 후보 동작 | 관찰 또는 예상 차이 | 판정 | 상태 | 검증 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `debug-triage` | 이관 | `.agents/skills/ner-debug-triage/SKILL.md` | NER 전용 진단 체크리스트를 직접 제공 | `systematic-debugging`에 NER 점검만 추가 | 공통 디버깅 절차를 중복하지 않음 | 동등 대체 | 계획됨 | 후속 스킬 frontmatter 및 smoke test |
| `explain-diff` | 이관 | `.agents/skills/explain-diff/SKILL.md` | 한국어 변경 설명 보고서를 생성 | 같은 보고서 계약을 저장소 스킬로 제공 | 발견 경로만 바뀜 | 동일 유지 | 계획됨 | 후속 스킬 frontmatter 및 smoke test |
| `perf-measure` | 이관 | `.agents/skills/perf-measure/SKILL.md` | 고정 조건의 성능·품질 측정을 안내 | 같은 측정 절차를 저장소 스킬로 제공 | 발견 경로만 바뀜 | 동일 유지 | 계획됨 | 후속 스킬 frontmatter 및 smoke test |
| `tdd` | 통합 | `superpowers:test-driven-development` | 프로젝트 TDD 스킬이 RED·GREEN·REFACTOR를 안내 | 설치된 Superpowers TDD 절차를 사용 | 중복 프로젝트 스킬을 만들지 않음 | 동등 대체 | 계획됨 | 중복 프로젝트 스킬 부재 확인 |
| `refuter` | 통합 | `superpowers:requesting-code-review`와 `superpowers:verification-before-completion` | Stop 훅·상태 파일·반박자 루프를 결합 | 필요 시 읽기 전용 reviewer를 호출하고 최종 검증을 분리 | 자동 Stop 루프와 판정 상태를 제거 | 의도적 단순화 | 계획됨 | Superpowers 절차와 reviewer 계약 확인 |
| `lessons-digest` | 보류 | 상태 기반 자동 수집을 유지할지 제거할지 별도 판단 | Stop 알림과 refuter 상태에서 반복 지적을 수집 | 자동 수집을 유지할지 제거할지 결정하지 않음 | Claude·OMC 상태 의존성을 그대로 만들지 않음 | 보류 | 보류 | Codex 중복 스킬 부재 확인 |

## 활성화 상태

| 항목 | 상태 | 비고 |
| --- | --- | --- |
| Codex 전역 설정 | 비활성 | 저장소 자산 검증과 사용자 별도 승인 전에는 `/home/rkim/.codex/config.toml`을 변경하지 않음 |
| Codex Rules | 비활성 | 최소 명령 prefix 정책과 실제 영향 범위를 후속 활성화 체크리스트에서 검토 |
| Codex Hooks | 비활성 | 단위·통합 테스트와 dry-run이 끝난 뒤 저장소 범위 활성화를 별도로 승인 |
