# issue-260: Docker 스크립트 후속 수정

- Issue: https://github.com/groovallstar/ner-pipeline/issues/260
- 상태: 실패 전파 수정은 PR #281로 develop에 반영되었다. 아래 통합 작업과 독립 PR의 검증 기록을 함께 보존한다.

## 배경
Docker lifecycle 스크립트가 실패의 stderr와 종료 코드를 숨긴다.
- `docker/server/stop.sh`는 `down 2>/dev/null && echo ... || true`로 종료 실패에도 코드 0을 반환한다.
- `docker/vllm/stop.sh`도 같은 처리를 하고 마지막에 중지 완료 메시지를 출력한다.
- `docker/server/start.sh`는 기동 전 `down 2>/dev/null || true`로 정리 실패 뒤에도 up을 진행한다.

## 목적
Docker 작업 실패를 호출자가 종료 코드와 진단 출력으로 식별할 수 있게 한다.

## 성공 기준
- [x] server/vLLM stop에 Docker 실패를 주입하면 비정상 종료 코드와 진단 근거가 남는다.
- [x] vLLM 여러 프로젝트 중 일부 종료가 실패해도 전체 성공을 표시하지 않는다.
- [x] start의 사전 정리 실패를 무조건 숨기지 않고 후속 기동 여부를 명확히 처리한다.
- [x] 정상·실패 경로의 회귀 검사를 추가하고 관련 지침을 갱신한다.

## 범위
기존 start/stop의 실패 전파와 진단 처리에 한정한다. 기본 종료 대상 변경이나 기동 전 down 제거 자체는 승인된 구현 정책이 아니다.

## 검증
사전 조사에서 Docker 대역의 종료 코드 17이 스크립트 코드 0으로 바뀌는 것을 확인했다. 이번 등록 시 실패 은폐 구문을 정적으로 재확인했다. 실제 Docker 장애는 유발하지 않았다.
수정 시 Docker 대역 실패 주입과 `bash -n docker/server/*.sh docker/vllm/*.sh`를 실행하고 실제 서비스 검증과 구분한다.

### 구현 후 검증

- 수정 전 `uv run pytest tests/docker/test_lifecycle_failures.py -q`:
  11 failed, 3 passed. 실패 코드 17·23이 0으로 바뀌고 stderr가 사라지는
  경로를 재현했다. 정상 vLLM 검사 2건은 출력의 영문 전환 전이라 실패했다.
- 수정 후 `uv run pytest tests/docker/test_lifecycle_failures.py
  tests/server/test_lifecycle_scripts.py -q -rs`: 28 passed, skip 없음.
  Docker 대역으로 사전 down 실패 후 up 미호출, up 실패 전파, 정상 기동,
  기본·명시 프로젝트 종료, 일부·전체 실패와 첫 실패 코드 보존을 검사했다.
- `for script in docker/server/*.sh docker/vllm/*.sh; do bash -n "$script"
  || exit; done`: 통과. 모든 셸 파일을 개별 구문 검사했다.
- `uv run ruff check tests/docker/test_lifecycle_failures.py
  tests/server/test_lifecycle_scripts.py`: 통과.
- `git diff --check`: 통과.
- `uv run pytest tests/docker tests/server -q -rs
  --ignore=tests/server/test_inference_integration.py
  --ignore=tests/server/test_live_server.py`: 216 passed, skip 없음.
  Starlette TestClient의 httpx 사용 중단 예정 경고 1건이 발생했다.
- 실제 Docker 서비스의 기동·종료, GPU·모델 통합 경로는 실행하지 않았다.
- 완료 전 리뷰: Superpowers `requesting-code-review` 절차의 리뷰
  서브에이전트가 이번 작업의 최종 변경을 검토했다. Critical·Important·Minor
  지적이 모두 없으며 추가 수정 요청도 없었다.

## 변경 요약

- server start/stop은 down의 stderr와 실패 코드를 보존한다. start는 정리
  실패 메시지를 남기고 up 전에 종료한다.
- vLLM stop은 실패한 프로젝트를 진단하고 나머지 프로젝트도 종료한다.
  하나라도 실패하면 첫 실패 코드를 반환하며 전체 성공을 표시하지 않는다.
- `tests/docker/test_lifecycle_failures.py`와 서버·vLLM 하위 지침을 갱신했다.
  기존 서버 대상 선택 로직과 vLLM 기본 종료 대상은 유지했다.
- 서버 구현 맵의 Docker 지침 링크는 그대로 유효하다. API·엔티티 스키마
  변경이 없어 해당 문서는 갱신하지 않았다.

## 구현 단계

1. Docker CLI 대역으로 server start/stop과 vLLM stop의 실패 코드·stderr,
   후속 호출 여부를 검사하고 수정 전 실패를 확인한다.
2. server는 down 실패 코드를 그대로 반환하고 start의 up을 중단한다.
   vLLM은 모든 대상의 down을 시도하고 첫 실패 코드를 반환한다.
3. 정상·일부 실패·전체 실패를 검사하고 Docker 하위 지침을 갱신한다.
   API·엔티티 스키마와 서버 구현 맵은 변경 대상이 아니다.
4. 셸 구문, 대상 pytest, Ruff와 최종 변경 리뷰를 수행한다.

## 결정 로그
- 2026-09-14: 푸시·PR 요청에 따라 기존 작업 트리에서 이슈 260의 실패 처리만
  별도 작업 트리로 분리한다. 서버 대상 선택을 바꾸는 이슈 259는 포함하지
  않으며, 현재 develop의 `NER_SERVER_PROJECT`와 명시 인자 우선순위를 유지한다.
  사전 down도 유지하고 실패하면 up을 중단한다. vLLM은 모든 대상을 시도한 뒤
  첫 실패 코드를 반환한다. 성공 기준은 유지한다.


- 2026-09-09: 사용자 요청에 따라 하네스 정비에서 분리하여 이후 처리한다. 미구현 상태다.
- 2026-09-14: 이슈 진행 요청에 따라 기존 develop 작업 공간의 이슈 259 변경을
  보존하며 실패 처리만 수정한다. 기존의 실패 무시 대신 server start는 사전
  정리 실패 시 기동을 중단하고, vLLM stop은 남은 종료를 시도한 뒤 첫 실패
  코드를 반환한다. 대상 선택과 사전 down은 유지하며 미루는 성공 기준은 없다.
  이 결정은 로컬 기록에 반영했으며 GitHub Issue 본문은 수정하지 않았다.

## 독립 PR 검증

- 수정 전 `tests/docker/test_lifecycle_failures.py`: 11 failed, 3 passed.
  실패 종료 코드·stderr 은폐를 재현했다. 정상 vLLM 경로 2건은 영문 성공
  메시지 전환 전이어서 실패했다.
- 실제 Docker 서비스와 GPU·모델 통합 경로는 실행하지 않는다.

- 분리한 작업 트리는 기존 UV 환경을 재사용했다. 아래 명령에는
  `UV_PROJECT_ENVIRONMENT=/work/git/ner-pipeline/.venv`를 지정했으며
  `--no-sync`로 환경 재설치를 생략했다. 셸 실행 대상은 이 PR의 작업 트리다.
- `uv run --no-sync pytest tests/docker/test_lifecycle_failures.py -q`:
  14 passed, skip 없음.
- `uv run --no-sync pytest tests/docker tests/server -q -rs
  --ignore=tests/server/test_inference_integration.py
  --ignore=tests/server/test_live_server.py`: 202 passed, skip 없음.
  Starlette TestClient의 httpx 사용 중단 예정 경고 1건이 발생했다.
- `uv run --no-sync ruff check tests/docker/test_lifecycle_failures.py`: 통과.
- `for script in docker/server/*.sh docker/vllm/*.sh; do bash -n "$script"
  || exit; done`: 모든 셸 파일의 구문 검사 통과.
- `git diff --check`: 통과.
- 실모델 통합·live 파일은 수집에서 제외했으며, 실제 Docker daemon·서비스
  장애를 유발하지 않았다. Docker CLI 대역으로 실패 전파와 호출 순서를 검사했다.
