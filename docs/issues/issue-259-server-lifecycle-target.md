# issue-259: Docker 스크립트 후속 수정

- Issue: https://github.com/groovallstar/ner-pipeline/issues/259
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-259-server-lifecycle-target` (기존 작업 공간에서 게시한다).
- 상태: 최신 develop과의 통합·검증·최종 리뷰를 마쳤으며 커밋·푸시·PR 게시를 진행한다.

## 배경
NER 서버 스크립트가 같은 설정에서 서로 다른 대상을 선택하거나 잘못된 health 주소를 안내한다.
- `.env.example`은 `NER_SERVER_PROJECT`를 start/stop 공용 설정으로 안내하지만, 두 스크립트는 `.env`를 읽지 않고 셸 변수 또는 ner-server를 Compose의 `-p`에 전달한다.
- `docker/server/logs.sh`는 인자가 없으면 ner-server를 고정 선택하여 `NER_SERVER_CONTAINER` 설정을 반영하지 않는다.
- `start.sh`의 health 안내는 셸의 `NER_SERVER_PORT` 또는 8008을 사용하여 `.env`의 게시 포트와 달라질 수 있다.

## 근본 원인
설정을 푸는 곳이 둘인데 서로 다른 출처를 본다. compose는 같은 디렉토리의 `.env`를
자동 로드하지만, 그 로드는 compose 파일 안의 `${...}` 치환에만 걸린다. 셸 스크립트는
`.env`를 읽지 않는다.

특히 `NER_SERVER_PROJECT`는 compose가 모르는 자체 변수다. compose 파일 어디에도
쓰이지 않아 스크립트가 `-p`로 날라줘야만 효력이 생기고, 그러려면 스크립트가 `.env`를
읽어야 한다. 해석기가 둘로 갈린 지점이 여기다.

## 목적
선택한 NER 서버 설정과 시작·종료·로그 대상 및 health 안내를 일치시킨다.

## 성공 기준
- [x] 비기본 프로젝트·컨테이너·포트를 셸 환경과 `.env` 각각으로 지정했을 때 대상이 일치한다.
- [x] 인자·환경변수·`.env` 우선순위를 정의하고 회귀 검사한다.
- [x] 프로젝트와 컨테이너 이름이 달라도 로그·종료가 올바른 대상을 선택한다.
- [x] health 안내가 게시 포트와 일치하며 설정 예시와 지침을 갱신한다.

## 범위
기존 NER lifecycle 스크립트와 설정 문서 및 회귀 검사에 한정한다. 기본값 제거나 worktree 소유 추적 신설은 확정된 요구가 아니다.

## 결정 로그
- 2026-09-09: 사용자 요청에 따라 하네스 정비에서 분리하여 이후 처리한다. 미구현 상태다.
- 2026-09-10: 스크립트가 `.env`를 직접 읽는 안을 버리고 compose를 유일한 해석기로
  삼는 안을 택했다. 앞 안은 compose와 조금이라도 다르게 읽는 두 번째 파서를 만드는
  일이라, 이 이슈가 고치려는 어긋남을 구조로 고정한다. 뒤 안은 세 값 중 둘을
  스크립트가 알 필요 자체를 없앤다.
- 2026-09-10: 프로젝트 이름을 자체 변수 `NER_SERVER_PROJECT`에서 compose 네이티브
  키 `COMPOSE_PROJECT_NAME`으로 옮겼다. 기존 키는 `.env`에 적어도 아무 효력이 없었으므로
  동작 회귀가 아니라 죽은 설정을 살린 것이다. 기본값 `ner-server`는 없애지 않고
  `docker-compose.yml`의 `name:`으로 옮겨 §범위의 "기본값 제거 아님"을 지켰다.

- 2026-09-14: 기존 미커밋 구현을 이어서 검증한다. 이전 기록의 "동작 회귀가 아니다"는
  `.env` 설정에만 해당한다. 셸의 `NER_SERVER_PROJECT`는 이전 스크립트에서 유효했으며,
  승인된 네이티브 키 전환에 따라 셸 설정도 `COMPOSE_PROJECT_NAME`으로 옮겨야 한다.
  요구사항과 범위는 유지한다. 원격 이슈의 과거 설명은 아직 정정하지 않았다.

## 구현 결과
| 값 | 이전 | 이후 |
|---|---|---|
| 프로젝트 이름 | 스크립트가 셸에서 읽어 `-p`로 전달 | `-p` 미전달. compose가 `COMPOSE_PROJECT_NAME`(셸 > `.env`) 또는 `name:`으로 정한다 |
| 컨테이너 이름 | `logs.sh`가 `ner-server` 고정 | `logs.sh`가 compose 서비스 이름으로 붙어 컨테이너 이름을 알 필요가 없다 |
| 게시 포트 | 셸 값 또는 8008 추측 | `resolve.sh`가 `docker compose config`에서 읽어 안내한다. 해석 실패 시 추측하지 않고 알린다 |

`stop.sh [project]` 인자는 유지했다. compose 밖에서 만든 프로젝트를 내릴 수단이고,
`-p`는 compose 자신의 최우선 입력이라 해석기를 늘리지 않는다.

## 검증

현재 작업에서 다음 검사를 다시 실행했다.

- `uv run pytest tests/server/test_lifecycle_scripts.py -q -rs` — 14 passed.
  Docker 스텁 호출 검사와 실제 Compose 설정 해석 검사가 모두 실행되었다.
- `uv run pytest tests/server -q -rs --ignore=tests/server/test_inference_integration.py --ignore=tests/server/test_live_server.py`
  — 202 passed. 기존 Starlette/httpx 사용 중단 경고가 1건 발생했다.
- `uv run ruff check tests/server/test_lifecycle_scripts.py` — 통과했다.
- `bash -n docker/server/*.sh docker/vllm/*.sh`와 `git diff --check` — 통과했다.
- 임시 배포 디렉터리에 비기본 `.env`를 두고 기본값 검사를 실행했다.
  수정 전에는 `ner-custom`과 기본값의 불일치로 실패했으며, 빈 환경 파일을
  명시한 수정 후에는 통과했다. 실제 배포 `.env`는 변경하지 않았다.
- 문서 영향: `docs/manual/rest-api-spec.md`의 로그 조회 설명을 갱신했다.
  서버 구현 맵은 기존 배포 지침 링크로 연결되며, canonical 데이터 스키마는
  변경하지 않아 갱신할 필요가 없다.
- 리뷰: Superpowers `requesting-code-review` 서브에이전트가 기본값 검사의
  로컬 `.env` 의존성과 기존 셸 변수의 이관 설명을 지적했다. 두 지적을 반영했으며
  최종 변경의 재리뷰에서 추가 지적 없이 승인되었다.
- **미검증**: 실제 서비스 기동·종료·로그 tail, GPU·실모델 경로는 수행하지 않았다.
  모델 통합 및 live 검사 파일은 기본 검사에서 명시적으로 제외했다.

## 후속 작업
- 기존 배포 호스트의 `.env`와 셸 설정에서 `NER_SERVER_PROJECT`를
  `COMPOSE_PROJECT_NAME`으로 바꿔야 한다. 기존 키는 `.env`에서는 무효였지만
  셸에서는 유효했다. 셸 설정을 이관하지 않으면 기본 프로젝트가 선택된다.

## 커밋·PR 준비

- 2026-09-15: 사용자가 현재 작업 공간의 미커밋 변경에 대해 커밋·푸시·PR 생성을 요청했다. 작업 공간은 유지하며 `feat/issue-259-server-lifecycle-target` 브랜치로 게시한다. 분기 기준과 PR 대상은 fetch로 갱신한 `origin/develop` (`d023ef2442b681d877538371c6dbc680f2977012`)이다. 로컬 `develop`은 원격보다 2커밋 뒤에 있으며 로컬 전용 커밋은 없다. 별도 worktree는 만들지 않는다. 앞선 이슈 278 구현 단계의 `develop` 직접 작업 기록은 보존하고 이번 게시 단계는 기본 `feat/` 절차를 따른다. 기존 변경 전체를 보존하고 원격에 이미 반영된 실패 처리와 통합한다. 미루는 목표는 없다.

- 최신 develop과 통합할 때 server start/stop·지침·이슈 260 기록의 충돌을 해결했다. 원격 실패 전파 구현과 로컬 대상 해석 구현을 함께 유지하고, 이슈 260의 독립 PR 검증·결정 기록도 보존했다. 원격 Docker 실패 테스트에는 환경 격리 한 줄이 추가되어 있어 이를 유지했다.
- 게시 전 대상 검사: `uv run pytest tests/docker/test_lifecycle_failures.py tests/server/test_lifecycle_scripts.py -q -rs`는 28 passed, skip 없음이다. `uv run ruff check src/server tests/server tests/docker`와 각 셸 파일의 `bash -n`, `git diff --check`가 통과했다.
- 확대 검사: `uv run pytest tests/docker tests/server -q -rs --ignore=tests/server/test_inference_integration.py --ignore=tests/server/test_live_server.py`는 216 passed, skip 없음이며 기존 Starlette/httpx 사용 중단 예정 경고 1건이 발생했다. 실제 서비스 기동·종료·로그 tail, GPU·실모델 검사는 이번에도 수행하지 않았다. 기존 보고서의 부하 수치는 재측정하지 않았다.
- 게시 전 리뷰: Superpowers `requesting-code-review` 절차의 독립 서브에이전트가 전체 최종 diff와 미추적 파일 4개를 검토했다. Critical·Important·Minor 지적은 모두 없으며 수정 요청도 없다. 리뷰어는 기존 검사 결과를 재사용하고 `git diff --check`와 별도 작업 디렉터리에서 실제 Compose 설정 해석을 확인했다. 실제 서비스·GPU·모델 경로와 과거 부하 수치는 미검증으로 유지한다.
- PR 생성 직전 `git fetch origin develop` 후 현재 브랜치가 `feat/issue-259-server-lifecycle-target`이고 기준이 `d023ef2442b681d877538371c6dbc680f2977012`임을 확인했다. `git log --oneline origin/develop..HEAD`에는 이번 작업의 지침·보고서·빈 파일 정리·서버 대상 수정 4커밋만 포함되며, `git diff --stat origin/develop...HEAD`는 17개 파일의 승인된 변경만 포함한다. 관계없는 커밋이나 대상 브랜치 예외는 없다. PR은 `--base develop --head feat/issue-259-server-lifecycle-target`으로 생성한다.
