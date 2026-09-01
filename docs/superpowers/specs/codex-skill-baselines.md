# Codex 프로젝트 스킬 baseline 감사 기록

## 목적

이 문서는 Claude 프로젝트 스킬을 Codex에 복제할 필요가 있는지 판단한 실행 조건과
관찰을 영구 보존한다. 휘발성 실행 산출물의 경로가 없어도 같은 입력과 조건을 다시
구성할 수 있도록 실행 봉투, 요청, 사전 수락 기준, 관찰 결과와 결정을 기록한다.

## 공통 실행 봉투

| 항목 | 고정값 |
| --- | --- |
| 모델 | `gpt-5.6-terra` |
| reasoning | `medium` |
| 컨텍스트 | `fork_turns=none`인 fresh context |
| 저장소 기준 | Task 2 완료 커밋 `dffdd2c`; 루트와 하위 `AGENTS.md`가 존재하는 worktree |
| 지침 | 저장소 루트에서 실행해 루트 `AGENTS.md`를 활성화하고, 입력에 포함된 하위 경로에는 해당 하위 `AGENTS.md` 계약을 적용 |
| 프로젝트 스킬 | `.agents/skills/`가 없는 상태; `debug-triage`, `explain-diff`, `perf-measure`를 포함한 프로젝트 스킬 미사용 |
| 변경 권한 | read-only, tracked 파일 수정 금지 |
| 위임 | subagent 사용 금지 |
| 탐색 | 저장소 추가 탐색 금지; 프롬프트에 제공한 입력만 사용 |

모든 baseline은 결과를 작성하기 전에 아래 수락 기준을 정했다. 산출물의 문체나
섹션 이름이 다른 것만으로 실패시키지 않고, 요청한 정책 결과와 검증 가능성을
충족하는지 판정했다.

## debug-triage baseline

### 요청 입력과 프롬프트 요지

입력 증상은 최근 프롬프트와 tokenizer adapter가 함께 바뀐 뒤 일본어 BIO offset
테스트가 실패하고, 별도 vLLM 경로에서는 entity span의 마지막 일본어 문자 하나가
누락된다는 것이었다. 알려진 재현 명령은
`uv run pytest tests/ner -k tag_aligner -x -q`였다.

프롬프트는 두 증상을 성급히 같은 원인으로 단정하거나 즉시 `+1` 보정을 하지 말고,
수정 전에 수행할 체계적인 진단 절차를 작성하도록 요청했다. debug baseline만 설치된
`superpowers:systematic-debugging`을 적용했으며 프로젝트 debug skill은 사용하지 않았다.

### 사전 수락 기준

- 최초 실패와 Unicode·token·span 증거를 보존한다.
- 단일 사례 최소 재현과 반복성 측정을 먼저 제시한다.
- BIO와 vLLM 경로를 분리하고 경계별 데이터 흐름을 관측한다.
- 최근 프롬프트와 adapter를 한 변수씩 비교하는 격리 실험을 제시한다.
- 원인이 좁혀진 뒤에만 회귀 테스트와 최소 수정을 허용한다.
- 저장소 규칙에 맞는 `uv run` 명령을 사용하고 `PYTHONPATH`를 주입하지 않는다.

### 관찰과 결정

응답은 원본 증거 보존, 단일 재현, Unicode 단위와 end convention 관측, BIO와
LLM 경계 분리, 프롬프트·adapter 2x2 실험, working case 비교, 회귀 검증 순서를
모두 제시했다. 따라서 NER 전용 체크리스트를 새로 복제하지 않고
`superpowers:systematic-debugging`과 `AGENTS.md`에 `debug-triage`를 통합하는
동등 대체를 채택했다.

## explain-diff 초기 계약 baseline

### 요청 입력과 프롬프트 요지

실제 diff가 없는 상태에서 NER labeler 변경을 처음 보는 한국어 비전문가가 변경의
이유, 동작 차이, 데이터와 품질 영향을 이해하고 검토할 수 있는 설명 문서의 산출물
계약을 설계하도록 요청했다. 근거가 없는 변경 내용이나 수치를 만들지 말고, 작성 전
필수 입력, 문서 구조, 예시, 검증과 완료 기준을 정하도록 했다.

### 사전 수락 기준

- 실제 diff가 없다는 제약을 인정하고 필요한 근거 입력을 먼저 열거한다.
- NER, span, BIO, F1을 모르는 독자를 위한 설명 순서를 정한다.
- 대표 성공, 경계, 비변경 사례를 포함하는 전후 비교 계약을 둔다.
- 품질 수치에는 동일 기준 재평가와 원본 metric 근거를 요구한다.
- 테스트가 보장하는 범위와 검증하지 않은 범위를 분리한다.

### 관찰과 결정

응답은 입력 근거 표, 독자와 목적, 아홉 개 문서 구성, 세 종류의 필수 예시,
검증 체크리스트와 완료 기준을 제시했다. 이는 실제 diff 설명의 준비 계약으로는
충분했지만, 실제 변경을 받아 생성한 산출물의 품질을 직접 증명하지는 않았다.
따라서 제거 후보의 1차 근거로만 사용하고 다음 actual diff baseline을 추가했다.

## explain-diff actual diff baseline

### 요청 입력과 프롬프트 요지

입력은 Task 2 review package의 실제 변경이다. 재현할 때는
`git diff c09967f..dffdd2c`로 루트와 하위 15개 `AGENTS.md`, migration inventory,
`tests/migration/test_codex_migration.py`의 Task 2 변경을 제공한다. 프롬프트는 이
diff만 근거로, NER 개념을 모르는 한국어 독자가 이해할 수 있는 self-contained
Markdown 설명을 작성하도록 요청했다. 배경, 핵심 직관, 실제 변화, 위험, 검증 범위와
이해 확인 질문 5개를 포함하고 실행하지 않은 검증은 성공으로 주장하지 않도록 했다.

### 사전 수락 기준

- 실제 diff의 범위와 비목표를 사실대로 설명한다.
- 배경, 직관, 변화, 위험, 검증을 self-contained 한국어 Markdown으로 제공한다.
- 계층형 `AGENTS.md`가 적용되는 방식을 비전문가가 이해할 수 있게 설명한다.
- 정적 migration 테스트가 보장하는 것과 보장하지 않는 것을 분리한다.
- 독자가 핵심 이해를 점검할 질문을 정확히 5개 제공한다.

### 관찰과 결정

응답은 제품 코드·모델·데이터·평가 수치를 바꾸지 않는 범위를 먼저 밝혔고, 경로별
지침 계층의 직관, 16개 지침과 migration 테스트 변화, 누락·상충 위험, 실행하지 않은
검증의 한계를 설명했다. 마지막에 이해 질문 5개를 제공해 기능 수락 기준을 모두
충족했다.

기존 `explain-diff` 표현 계약과 비교하면 목차, Mermaid 또는 다른 diagram,
객관식 선택지, answer key는 없었다. 이 요소들은 사실성·위험·검증 이해에 필수인
정책 결과가 아니며, 고정 템플릿과 다이어그램 문법을 별도 스킬로 유지할 비용이
더 크다고 판단해 의도적으로 제거한다. `explain-diff` 제거 결정은 유지하되,
이 표현 차이가 실제 사용성을 해치지 않는지는 Task 6 actual-diff smoke에서 확정한다.

## perf-measure baseline

### 요청 입력과 프롬프트 요지

입력 상황은 vLLM 기반 NER 평가 처리량을 높이려 하지만 기존 실행마다 샘플, 모델,
GPU 조건이 달라 수치를 직접 비교할 수 없다는 것이었다. 조정 대상은
`max-model-len`, `max-num-seqs`, client concurrency였다. 프롬프트는 품질을 희생하지
않는 비교 가능한 기준선과 단일 변수 실험 계획, 채택 조건과 재현 원장을 작성하도록
요청했다.

### 사전 수락 기준

- 데이터, 모델, 생성, 서버, 하드웨어, 환경, 클라이언트와 측정 경계를 고정한다.
- 워밍업과 최소 3회 반복을 분리하고 변동성을 보고한다.
- 처리량뿐 아니라 지연, 오류, 자원, NER 품질을 함께 측정한다.
- 세 조정 대상을 한 번에 하나씩 바꾸고 품질 회귀를 먼저 차단한다.
- 실행 ID, 원시 결과, 비교 기준과 미검증 조건을 재현 가능하게 남긴다.

### 관찰과 결정

응답은 고정 조건 표, 반복 기준선, 처리량·TTFT·tail latency·신뢰성·GPU 자원·품질
지표, 단일 변수 순서, 품질 게이트와 실행 원장을 모두 제시했다. Codex 기본 역량만으로
측정 계획의 정책 결과를 충족했으므로 `perf-measure` 프로젝트 스킬은 제거한다.

## 여섯 스킬의 채택 결정

| 원본 스킬 | 결정 | 영구 근거와 후속 확인 |
| --- | --- | --- |
| `debug-triage` | 통합 | debug baseline의 동등 대체 관찰 |
| `explain-diff` | 제거 | 초기 계약과 actual diff baseline; 표현 차이는 Task 6 actual-diff smoke에서 확정 |
| `perf-measure` | 제거 | perf baseline의 측정 정책 충족 관찰 |
| `tdd` | 통합 | 설치된 `superpowers:test-driven-development`가 RED·GREEN·REFACTOR 절차를 제공 |
| `refuter` | 통합 | `requesting-code-review`와 `verification-before-completion`으로 대체하며 자동 상태 루프 제거는 의도적 단순화 |
| `lessons-digest` | 보류 | 상태 기반 자동 수집의 필요성과 대체 근거가 부족해 Task 6 또는 별도 설계까지 보류 |
