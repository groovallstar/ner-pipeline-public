# issue-260: Docker 스크립트 후속 수정

- Issue: https://github.com/groovallstar/ner-pipeline/issues/260
- 상태: 등록 완료, 구현 대기.

## 배경
Docker lifecycle 스크립트가 실패의 stderr와 종료 코드를 숨긴다.
- `docker/server/stop.sh`는 `down 2>/dev/null && echo ... || true`로 종료 실패에도 코드 0을 반환한다.
- `docker/vllm/stop.sh`도 같은 처리를 하고 마지막에 중지 완료 메시지를 출력한다.
- `docker/server/start.sh`는 기동 전 `down 2>/dev/null || true`로 정리 실패 뒤에도 up을 진행한다.

## 목적
Docker 작업 실패를 호출자가 종료 코드와 진단 출력으로 식별할 수 있게 한다.

## 성공 기준
- [ ] server/vLLM stop에 Docker 실패를 주입하면 비정상 종료 코드와 진단 근거가 남는다.
- [ ] vLLM 여러 프로젝트 중 일부 종료가 실패해도 전체 성공을 표시하지 않는다.
- [ ] start의 사전 정리 실패를 무조건 숨기지 않고 후속 기동 여부를 명확히 처리한다.
- [ ] 정상·실패 경로의 회귀 검사를 추가하고 관련 지침을 갱신한다.

## 범위
기존 start/stop의 실패 전파와 진단 처리에 한정한다. 기본 종료 대상 변경이나 기동 전 down 제거 자체는 승인된 구현 정책이 아니다.

## 검증
사전 조사에서 Docker 대역의 종료 코드 17이 스크립트 코드 0으로 바뀌는 것을 확인했다. 이번 등록 시 실패 은폐 구문을 정적으로 재확인했다. 실제 Docker 장애는 유발하지 않았다.
수정 시 Docker 대역 실패 주입과 `bash -n docker/server/*.sh docker/vllm/*.sh`를 실행하고 실제 서비스 검증과 구분한다.

## 결정 로그
- 2026-09-09: 사용자 요청에 따라 하네스 정비에서 분리하여 이후 처리한다. 미구현 상태다.

관련 기록: `docs/issues/codex-harness-redefinition.md`.
