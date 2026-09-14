# issue-260: Docker 스크립트 후속 수정

- Issue: https://github.com/groovallstar/ner-pipeline/issues/260
- 상태: main 기반 구현·검증 완료. 커밋·PR 진행 중.
- 브랜치: `fix/issue-260-docker-lifecycle-failure`

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

### develop 작업 공간 검증 이력

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

### main 기반 PR 검증

- 기준: `74fd714ef46dbcb30f751a25d4bba0e284bacfa8`에서 분기한
  `fix/issue-260-docker-lifecycle-failure`이다. 위 develop 검증과 구분한다.
- 이식 전 대상 검사: 11 failed, 3 passed. 최종 대상 검사: 16 passed.
  main에는 이슈 259의 서버 스크립트 검사 파일이 없으므로 서버 stop의
  정상 경로 2건도 이번 회귀 검사에 포함했다.
- `uv run pytest tests/docker tests/server -q -rs
  --ignore=tests/server/test_inference_integration.py
  --ignore=tests/server/test_live_server.py`: 최종 204 passed, skip 없음.
  Starlette의 httpx 사용 중단 예정 경고 1건이 발생했다.
- `uv run ruff check tests/docker/test_lifecycle_failures.py`, 모든 Docker 셸의
  개별 `bash -n`, `git diff --check`: 통과.
- Superpowers 리뷰 서브에이전트의 main 기준 재리뷰에서 런타임·테스트 결함은
  없었다. 문서의 대상 검사 개수 지적 1건은 최종 재실행 결과인 16 passed로
  갱신했다.
- 실제 Docker·GPU·모델 통합 검사는 실행하지 않았다.

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
- 2026-09-09: 사용자 요청에 따라 하네스 정비에서 분리하여 이후 처리한다. 미구현 상태다.
- 2026-09-14: 이슈 진행 요청에 따라 기존 develop 작업 공간의 이슈 259 변경을
  보존하며 실패 처리만 수정한다. 기존의 실패 무시 대신 server start는 사전
  정리 실패 시 기동을 중단하고, vLLM stop은 남은 종료를 시도한 뒤 첫 실패
  코드를 반환한다. 대상 선택과 사전 down은 유지하며 미루는 성공 기준은 없다.
  이 결정은 로컬 기록에 반영했으며 GitHub Issue 본문은 수정하지 않았다.

- 2026-09-14: main 대상 커밋·PR 요청에 따라 최신 origin/main 기반의 별도
  작업 공간으로 실패 처리만 이식한다. 이전 develop 작업 공간에는 미커밋
  이슈 259가 섞여 있어 이를 PR에서 제외하기 위해 작업 기준을 변경했다.
  main의 기존 프로젝트 선택과 사전 down을 유지하며 요구 기능은 바꾸지 않는다.
  기존 검증·리뷰는 develop 기준의 이력이며 PR 최종 변경은 다시 검증·리뷰한다.
