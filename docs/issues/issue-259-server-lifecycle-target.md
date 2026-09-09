# issue-259: Docker 스크립트 후속 수정

- Issue: https://github.com/groovallstar/ner-pipeline/issues/259
- 상태: 등록 완료, 구현 대기.

## 배경
NER 서버 스크립트가 같은 설정에서 서로 다른 대상을 선택하거나 잘못된 health 주소를 안내한다.
- `.env.example`은 `NER_SERVER_PROJECT`를 start/stop 공용 설정으로 안내하지만, 두 스크립트는 `.env`를 읽지 않고 셸 변수 또는 ner-server를 Compose의 `-p`에 전달한다.
- `docker/server/logs.sh`는 인자가 없으면 ner-server를 고정 선택하여 `NER_SERVER_CONTAINER` 설정을 반영하지 않는다.
- `start.sh`의 health 안내는 셸의 `NER_SERVER_PORT` 또는 8008을 사용하여 `.env`의 게시 포트와 달라질 수 있다.

## 목적
선택한 NER 서버 설정과 시작·종료·로그 대상 및 health 안내를 일치시킨다.

## 성공 기준
- [ ] 비기본 프로젝트·컨테이너·포트를 셸 환경과 `.env` 각각으로 지정했을 때 대상이 일치한다.
- [ ] 인자·환경변수·`.env` 우선순위를 정의하고 회귀 검사한다.
- [ ] 프로젝트와 컨테이너 이름이 달라도 로그·종료가 올바른 대상을 선택한다.
- [ ] health 안내가 게시 포트와 일치하며 설정 예시와 지침을 갱신한다.

## 범위
기존 NER lifecycle 스크립트와 설정 문서 및 회귀 검사에 한정한다. 기본값 제거나 worktree 소유 추적 신설은 확정된 요구가 아니다.

## 검증
사전 조사에서 Docker 대역으로 로그 대상 불일치를 확인했다. 임시 `.env`와 기존 start와 같은 `-p ner-server`를 적용한 Compose 설정 해석에서는 컨테이너·포트만 변경되고 프로젝트는 ner-server로 남았다. 이번 등록 시 코드도 정적으로 재확인했다. 실제 서비스 시작·종료는 수행하지 않았다.
수정 시 Docker 대역 검사, Compose 설정 해석, `bash -n docker/server/*.sh`를 실행하고 실제 서비스 검증과 구분한다.

## 결정 로그
- 2026-09-09: 사용자 요청에 따라 하네스 정비에서 분리하여 이후 처리한다. 미구현 상태다.
