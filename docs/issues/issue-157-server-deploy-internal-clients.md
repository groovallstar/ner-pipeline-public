# issue-157: NER REST API 배포 인프라 + 내부 소비자 예제

- Issue: https://github.com/groovallstar/ner_pipeline/issues/157
- PR: https://github.com/groovallstar/ner_pipeline/pull/158
- 브랜치: `feat/issue-157-server-deploy-internal-clients`
- 승인일: 2026-06-30

## 목적

ja·vi NER REST API 서버를 **내부망 별도 프로세스**가 HTTP 로 소비할 수 있도록
실제 배포 가능 상태로 만든다. 서버는 견고화·throughput 최적화(#147·#149·#154)를
마쳤으나 미배포·소비자 0 이었다 — 배포 인프라 + 내부 소비자 예제로 실효화한다.

## 범위

- 포함: `docker/server/` 배포(compose + 라이프사이클 + `.env.example`), 컨테이너
  준비성 healthcheck, 내부 소비자 예제(python `NERClient` + curl), 문서.
- 제외: 외부 공개·API-key 인증·계약 버저닝(내부 신뢰망 전제), `/metrics`·access
  log(별도 트랙), ko 서빙(별도 트랙), 동적 배칭(#154 제외 유지).

## 성공 기준 (AC)

- [x] **AC1 배포**: `docker/server/` compose(GPU reservation·`/data` 마운트)·
  `start/stop/logs.sh`·`.env.example`·이미지 태그 pin. 기동→ja·vi 로드→
  `/health`=`ok`.
- [x] **AC2 준비성**: healthcheck 가 `status:"ok"`(전 언어 로드)만 healthy,
  `start_period` 로 로드중 트래픽 차단. starting→healthy 전이.
- [x] **AC3 내부 소비자 예제**: `example_client.py`(python) + curl 예시가 단일·
  배치·미지원·400·413·429 를 실서버 대상 호출·검증(DEMO PASS).
- [x] **AC4 게이트**: `tests/server` green, ruff green, docker·server CLAUDE.md
  갱신.

## 현 상태 (fact)

- 빌드·기동 실증: `ner-server:0.1.0` 빌드 → ja(thresholds)·vi(raw) 로드 →
  `/health {"status":"ok"}` → docker health=healthy. 추론 스모크(단일·배치·
  미지원·400) 정상.
- `example_client` 자기검증: 단일·배치(순서 1:1)·unsupported·400·413·429 PASS
  (429 는 동시 64 workers×200req → 200:49 / 429:151 로 결정적 유발).
- 기존 서버 런타임 코드(app·inference·detect·concurrency) **무변경**(순수
  추가) — 회귀 0.

## 결정 로그 (append-only)

- 2026-06-30: 소비자를 **내부망 별도 프로세스**로 확정 → HTTP 경유 정당(같은
  코드베이스 `import` 아님). 외부 공개 계약 버저닝·API-key 인증은 범위 외.
- 2026-06-30: 로컬 develop 이 stale(#147)이라 origin/develop(a083703, #154
  포함) 기준으로 브랜치 재설정 — #154 코드(동시성·양성감지·bf16) 위 배포 보장.
- 2026-06-30: healthcheck grep 을 `'"status": ?"ok"'` 정규식으로(직렬화 공백
  유무 무관) 견고화 — Starlette 직렬화 변동 시 영구 unhealthy 방지.
- 2026-06-30: 서비스 GPU 1장 고정(compose `device_ids`·로컬 `CUDA_VISIBLE_
  DEVICES`, 기본 0) + 로컬 실행 경로(`run_local.sh`) 추가 — 컨테이너·로컬
  둘 다 GPU 0(uuid `72704dbf`) 단독 점유 실증.

## 변경 요약

- `docker/server/`: Dockerfile(CUDA 베이스 + uv sync), docker-compose.yml(GPU·
  `/data`·준비성 healthcheck·restart), start/stop/logs.sh, .env.example,
  CLAUDE.md.
- `.dockerignore`: 빌드 컨텍스트에서 `.venv`(5.9G)·산출물 제외.
- `src/server/scripts/example_client.py`: `NERClient` + 실서버 자기검증 데모.
- 문서: docker/CLAUDE.md(server 행), src/server/CLAUDE.md(배포 참조·미지원/에러
  curl·example_client).

## 검증

- 테스트: `uv run pytest tests/server/ -m "not live"` → 70 passed. ruff green.
- 실서버: `docker compose up` → `/health` healthy, `example_client` DEMO PASS.
- 반박자(결과-시점): diff_hash `9c0546608619` — **PASS**(회귀 0·AC1~4 충족).
  이후 nit 2건(준비성 표현·429 단언 주석)을 문서·주석에 반영.

## 관련 커밋

- 단일 커밋(본 이슈): docker/server 배포 + example_client + 문서·아카이브.
