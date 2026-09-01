# Claude Code에서 Codex로의 병행 이관표

## 목적과 원칙

이 문서는 병행 운영 기간에 Claude Code와 Codex의 실제 동작, 차이, 채택 결정을 추적한다. 원본 `.claude/**`와 `CLAUDE.md` 파일은 현재 Claude 동작의 근거로 보존하며, Codex 대상은 새 파일로 추가한다. 차이는 자동 결함이 아니며 `동일 유지`, `동등 대체`, `의도적 단순화`, `제거`, `보류` 중 하나로 판정한다. Claude와 같은 절차량이나 reviewer 지적량은 이관 목표가 아니다.

## 프로젝트 지침

프로젝트 지침 16개는 Codex 계층으로 작성되었다. 검증 열은 현재 적용하는 정적 계약을 가리킨다.

| 원본 | Codex 대상 | Claude 동작 | Codex 후보 동작 | 관찰 또는 예상 차이 | 차이의 영향 | 판정 | 판정 근거·확정 시점 | 상태 | 검증 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `CLAUDE.md` | `AGENTS.md` | 프로젝트 규칙과 Claude·OMC 작업 절차를 함께 로드 | 도메인·안전 규칙은 로드하고 작업 절차는 Codex·Superpowers에 위임 | OMC 역할·명령과 자동 절차를 복제하지 않음 | 자동 절차가 줄어 호출 누락 가능성이 생기지만 저장소 규칙은 유지됨 | 의도적 단순화 | 설계상 중복 런타임을 제거하는 예상 판정이며 Task 2 지침 검사와 Task 6 병행 검증 후 확정 | 작성됨 | `tests/migration/test_codex_migration.py` |
| `docker/CLAUDE.md` | `docker/AGENTS.md` | Docker 공통 배포 경계와 운영 지침을 로드 | 같은 배포 경계를 Codex 계층에서 로드 | 지침 파일명과 로더만 다름 | 운영 경계에는 영향이 없고 발견 방식만 달라짐 | 동등 대체 | 정책 결과가 같다는 예상 판정이며 Task 2 계층 검사와 Task 6 세션 검증 후 확정 | 작성됨 | `tests/migration/test_codex_migration.py` |
| `docker/server/CLAUDE.md` | `docker/server/AGENTS.md` | REST API 컨테이너 설정·헬스체크·번역 의존성을 안내 | 같은 입력·준비성 계약을 현재 경로 기준으로 안내 | Claude 전용 실행 표현을 제거 | 제품 기동 계약은 유지되고 에이전트별 표현만 사라짐 | 동등 대체 | 원본 계약 비교에 따른 예상 판정이며 Task 2 내용 검사와 Task 6 세션 검증 후 확정 | 작성됨 | `tests/migration/test_codex_migration.py` |
| `docker/vllm/CLAUDE.md` | `docker/vllm/AGENTS.md` | compose 역할과 GPU·모델별 운영 주의를 안내 | 같은 서비스 책임과 안전 범위를 안내 | 공통 Docker 설명의 중복을 줄임 | 세부 지침 탐색 시 상위 문서도 함께 읽어야 함 | 의도적 단순화 | 계층 합성을 전제로 한 예상 판정이며 Task 2 계층 검사와 Task 6 세션 검증 후 확정 | 작성됨 | `tests/migration/test_codex_migration.py` |
| `src/ner/CLAUDE.md` | `src/ner/AGENTS.md` | NER 모듈·라벨·import 규칙을 로드 | 같은 도메인 불변조건을 Codex 계층에서 로드 | 지침 파일명과 로더만 다름 | 모듈 작업 계약에는 영향이 없고 발견 방식만 달라짐 | 동등 대체 | 정책 결과가 같다는 예상 판정이며 Task 2 계층 검사와 Task 6 세션 검증 후 확정 | 작성됨 | `tests/migration/test_codex_migration.py` |
| `src/ner/augmenters/CLAUDE.md` | `src/ner/augmenters/AGENTS.md` | 증강 모드·gold 보존·대상 테스트를 안내 | 같은 입력·출력과 gold 보호를 안내 | 실행 예시를 `uv run`으로 정규화 | `PYTHONPATH` 오염을 피하면서 증강 안전 계약은 유지됨 | 동등 대체 | 저장소 패키징 규칙에 따른 예상 판정이며 Task 2 정적 검사와 Task 6 세션 검증 후 확정 | 작성됨 | `tests/migration/test_codex_migration.py` |
| `src/ner/classifier/CLAUDE.md` | `src/ner/classifier/AGENTS.md` | 학습 CLI·group split·metric·산출물 계약을 안내 | 같은 누출 방지와 평가 경계를 안내 | 현재 CLI·테스트 경로만 유지 | 오래된 설명은 줄지만 세부 예시 일부가 제공되지 않을 수 있음 | 의도적 단순화 | 핵심 측정 계약 보존을 근거로 한 예상 판정이며 Task 2 내용 검사와 Task 6 세션 검증 후 확정 | 작성됨 | `tests/migration/test_codex_migration.py` |
| `src/ner/labelers/CLAUDE.md` | `src/ner/labelers/AGENTS.md` | 공통 LLM 라벨링과 언어별 처리 규칙을 함께 안내 | 공통 계약만 상위에 두고 언어별 규칙은 하위 계층에서 로드 | 중복 설명을 줄임 | 단일 파일만 읽으면 언어별 세부 사항이 보이지 않지만 계층 합성 시 보존됨 | 의도적 단순화 | 중복 제거 설계에 따른 예상 판정이며 Task 2 계층 검사와 Task 6 세션 검증 후 확정 | 작성됨 | `tests/migration/test_codex_migration.py` |
| `src/ner/labelers/ja/CLAUDE.md` | `src/ner/labelers/ja/AGENTS.md` | 일본어 offset·BIO·정규화 규칙을 안내 | 같은 일본어 경계 보존 규칙을 안내 | Claude 전용 작업 표현을 제거 | 일본어 라벨 결과 계약에는 영향이 없음 | 동등 대체 | 도메인 불변조건 보존에 따른 예상 판정이며 Task 2 내용 검사와 Task 6 세션 검증 후 확정 | 작성됨 | `tests/migration/test_codex_migration.py` |
| `src/ner/labelers/ko/CLAUDE.md` | `src/ner/labelers/ko/AGENTS.md` | KLUE·canonical 엔티티·원문 span 규칙을 안내 | 같은 한국어 데이터·span 불변조건을 안내 | Claude 전용 작업 표현을 제거 | 한국어 gold와 span 계약에는 영향이 없음 | 동등 대체 | 도메인 불변조건 보존에 따른 예상 판정이며 Task 2 내용 검사와 Task 6 세션 검증 후 확정 | 작성됨 | `tests/migration/test_codex_migration.py` |
| `src/ner/labelers/vi/CLAUDE.md` | `src/ner/labelers/vi/AGENTS.md` | 베트남어 정규화·BIO·평가 연결을 안내 | 같은 언어별 경계와 평가 계약을 안내 | 공통 라벨러 설명을 중복하지 않음 | 상위 지침 의존성이 생기지만 베트남어 계약은 유지됨 | 의도적 단순화 | 계층 합성을 전제로 한 예상 판정이며 Task 2 계층 검사와 Task 6 세션 검증 후 확정 | 작성됨 | `tests/migration/test_codex_migration.py` |
| `src/ner/llm_eval/CLAUDE.md` | `src/ner/llm_eval/AGENTS.md` | 프롬프트 실행·span 평가·산출물 계약을 안내 | 같은 평가 입력·출력과 재현 경로를 안내 | Claude 역할 표현을 제거 | 평가 결과 계약에는 영향이 없음 | 동등 대체 | 평가 경계 보존에 따른 예상 판정이며 Task 2 내용 검사와 Task 6 세션 검증 후 확정 | 작성됨 | `tests/migration/test_codex_migration.py` |
| `src/ner/scripts/CLAUDE.md` | `src/ner/scripts/AGENTS.md` | 보조 스크립트와 wrapper 규약을 안내 | 실제 스크립트 경로와 패키지 실행만 안내 | 실행 형식을 `uv run`으로 단순화 | 임의 실행 방식은 줄고 재현 가능한 패키지 실행으로 제한됨 | 의도적 단순화 | 현재 패키징 규칙에 따른 예상 판정이며 Task 2 내용 검사와 Task 6 세션 검증 후 확정 | 작성됨 | `tests/migration/test_codex_migration.py` |
| `src/ner/validity/CLAUDE.md` | `src/ner/validity/AGENTS.md` | fold·분산·baseline 판정과 기준 보호를 안내 | 같은 측정 불변조건과 판정 경계를 안내 | 하네스 절차보다 측정 정책에 집중 | 자동 절차 설명은 줄지만 측정 자 보호 규칙은 유지됨 | 의도적 단순화 | 안전 불변조건 보존에 따른 예상 판정이며 Task 2 내용 검사와 Task 6 병행 검증 후 확정 | 작성됨 | `tests/migration/test_codex_migration.py` |
| `src/server/CLAUDE.md` | `src/server/AGENTS.md` | API·오류·백엔드·관측성 계약을 안내 | 같은 서비스 계약과 설정 검증을 안내 | Claude 전용 작업 표현을 제거 | API 런타임 계약에는 영향이 없음 | 동등 대체 | 서비스 경계 보존에 따른 예상 판정이며 Task 2 내용 검사와 Task 6 세션 검증 후 확정 | 작성됨 | `tests/migration/test_codex_migration.py` |
| `tests/ner/CLAUDE.md` | `tests/ner/AGENTS.md` | 테스트 구조·fixture·안전 주의를 안내 | 같은 테스트 범위와 `uv run pytest` 실행법을 안내 | 공통 규칙을 루트에 위임 | 하위 문서만 읽으면 공통 규칙이 보이지 않지만 계층 합성 시 유지됨 | 의도적 단순화 | 계층 합성을 전제로 한 예상 판정이며 Task 2 계층 검사와 Task 6 세션 검증 후 확정 | 작성됨 | `tests/migration/test_codex_migration.py` |

## 프로젝트 스킬 결정

### 프로젝트 스킬 없는 baseline 관찰

근거 원문은 `.superpowers/sdd/2026-09-01-claude-to-codex-migration/`의
`task-3-baseline-debug.md`, `task-3-baseline-diff.md`,
`task-3-baseline-perf.md`에 보존한다.

| baseline | 관찰 | 결정에 준 근거 |
| --- | --- | --- |
| debug-triage | `systematic-debugging`과 `AGENTS.md`만 적용한 응답이 원본 증거 보존, 단일 재현, BIO와 vLLM 경계별 관측, 프롬프트·adapter 2x2 격리, 회귀 검증 순서를 모두 제시했다. | NER 전용 스킬을 추가하지 않아도 진단 목적을 충족하므로 기존 절차에 통합한다. |
| explain-diff | 별도 프로젝트 스킬 없는 응답이 비전문가용 변경 설명에 필요한 입력 근거, 문서 구성, 전후·경계·비변경 예시, 품질 수치와 재현 근거의 검증 계약을 제시했다. | Codex 기본 역량이 요구 산출물을 충족하므로 중복 스킬을 제거한다. |
| perf-measure | 별도 프로젝트 스킬 없는 응답이 데이터·모델·환경 고정, 반복 측정, 처리량·지연·신뢰성·자원·품질 지표, 단일 변수 실험과 품질 게이트를 제시했다. | Codex 기본 역량이 측정 계획을 충족하므로 중복 스킬을 제거한다. |

| 원본 스킬 | 결정 | Codex 대상 | Claude 동작 | Codex 후보 동작 | 관찰 또는 예상 차이 | 차이의 영향 | 판정 | 판정 근거·확정 시점 | 상태 | 검증 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `debug-triage` | 통합 | `superpowers:systematic-debugging`과 `AGENTS.md` | NER 전용 진단 체크리스트를 직접 제공 | 공통 디버깅 절차와 저장소 지침을 함께 적용 | 별도 NER 스킬과 고정 체크리스트를 만들지 않음 | 진단 표현은 달라도 재현·증거 보존·원인 격리·회귀 검증이 유지됨 | 동등 대체 | 프로젝트 스킬 없는 debug baseline이 요구 절차를 충족함 | 통합 | baseline 보고서와 `tests/migration/test_codex_migration.py` |
| `explain-diff` | 제거 | Codex 기본 역량 | 한국어 변경 설명 보고서와 세부 표현 계약을 제공 | 요청과 diff 근거로 필요한 설명 문서를 직접 구성 | 고정 템플릿과 스킬 호출이 사라짐 | 요청별 구성은 달라질 수 있으나 근거·예시·검증 계약은 유지됨 | 제거 | 프로젝트 스킬 없는 diff baseline이 요구 산출물을 충족함 | 제거 | baseline 보고서와 `tests/migration/test_codex_migration.py` |
| `perf-measure` | 제거 | Codex 기본 역량 | 고정 조건의 성능·품질 측정을 안내 | 요청과 저장소 지침으로 비교 가능한 측정 계획을 직접 구성 | 고정 스킬 호출이 사라짐 | 계획 표현은 달라질 수 있으나 고정 조건·지표·품질 게이트는 유지됨 | 제거 | 프로젝트 스킬 없는 perf baseline이 요구 산출물을 충족함 | 제거 | baseline 보고서와 `tests/migration/test_codex_migration.py` |
| `tdd` | 통합 | `superpowers:test-driven-development` | 프로젝트 TDD 스킬이 RED·GREEN·REFACTOR를 안내 | 설치된 Superpowers TDD 절차를 사용 | 중복 프로젝트 스킬을 만들지 않음 | 적용 범위 표현은 달라도 테스트 우선 집행은 유지됨 | 동등 대체 | 설치된 Superpowers 절차가 동일 목적을 제공함 | 통합 | `tests/migration/test_codex_migration.py` |
| `refuter` | 통합 | `superpowers:requesting-code-review`와 `superpowers:verification-before-completion` | Stop 훅·상태 파일·반박자 루프를 자동 결합 | 필요 시 읽기 전용 reviewer를 명시적으로 호출하고 최종 검증을 분리 | 자동 Stop 루프와 판정 상태를 제거 | 명시적 호출 누락 가능성과 집행력 감소가 있으며 검토 이력의 자동 연속성도 사라짐 | 의도적 단순화 | 상태 없는 검토와 최신 검증 증거를 채택함 | 통합 | Task 5 reviewer 계약 및 Task 6 병행 검증 |
| `lessons-digest` | 보류 | 상태 기반 자동 수집을 유지할지 제거할지 별도 판단 | Stop 알림과 refuter 상태에서 반복 지적을 수집 | 자동 수집을 유지할지 제거할지 결정하지 않음 | Claude·OMC 상태 의존성을 그대로 만들지 않음 | 반복 지적의 자동 축적과 알림이 사라질 수 있음 | 보류 | 필요성과 대체 근거가 부족하므로 Task 6 병행 기록 검토 또는 별도 설계 승인 시 확정 | 보류 | Task 6 병행 검증 기록 |

## 활성화 상태

| 항목 | 상태 | 비고 |
| --- | --- | --- |
| Codex 전역 설정 | 비활성 | 저장소 자산 검증과 사용자 별도 승인 전에는 `/home/rkim/.codex/config.toml`을 변경하지 않음 |
| Codex Rules | 비활성 | 최소 명령 prefix 정책과 실제 영향 범위를 후속 활성화 체크리스트에서 검토 |
| Codex Hooks | 비활성 | 단위·통합 테스트와 dry-run이 끝난 뒤 저장소 범위 활성화를 별도로 승인 |
