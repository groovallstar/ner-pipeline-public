# issue-183: server 요청 로깅·거절 가시화 + request-id 미들웨어

- Issue: https://github.com/groovallstar/ner_pipeline/issues/183
- PR: https://github.com/groovallstar/ner_pipeline/pull/184
- 브랜치: `feat/issue-183-server-request-logging`
- 승인일: 2026-07-23

## 목적

ja·vi NER REST API 서버의 관측성 공백을 메운다. 기존 로깅은 **모델 로드
생명주기 + 500 예외**만 남기고, 정상 처리·부하 거절(429/503/413/401)·지연은
침묵했다(#157 에서 access log 를 "별도 트랙"으로 명시 연기). 운영 디버깅에
필요한 최소 관측성을 얹되 **로그량은 억제**한다 — 평상시 조용, 필요할 때 상세.

## 범위

- 포함: 거절 경로 로깅, 요청 로그 미들웨어(request-id·지연·lang·batch·
  entities), 로그 레벨 env 스위치, uvicorn 기본 액세스 로그 대체.
- 제외: 구조화(JSON)·파일 로깅, `/metrics`(Prometheus), 슬로우-리퀘스트
  임계값 로깅, `create_app` 임베드 로깅 초기화(전부 별도 트랙).

## 로그 정책 (합의: "기본 조용 + 레벨 스위치")

성공은 조용, 이상만 시끄럽게 — 로그를 신호로 유지한다.

- 성공(2xx) → **DEBUG** 한 줄. 기본 레벨 INFO 에선 침묵.
- 거절(400·401·413·422·429·503) → **WARNING** 한 줄(status·reason·path·rid).
- 미처리 예외(500) → 트레이스백 ERROR(rid 포함, `app._unhandled`). 요약 라인
  중복 생략.
- uvicorn 기본 액세스 로그 off(`access_log=False`) — 요청 라인은 미들웨어 단독.
- `NER_SERVER_LOG_LEVEL`(기본 INFO)로 DEBUG 상세를 켠다.

## 성공 기준 (AC)

- [x] **AC1 거절 로깅**: 429·503·413·401·400·422 응답이 WARNING 한 줄
  (status·reason·path·rid)로. 성공은 WARNING/INFO 로 남지 않음.
- [x] **AC2 요청 로그 미들웨어**: 모든 응답에 `X-Request-ID`(수신 에코·없으면
  8-hex). DEBUG 에서 성공은 latency_ms·lang·batch·entities·rid 한 줄.
- [x] **AC3 조용한 기본값**: 기본 INFO 에서 성공 무로깅(uvicorn 액세스 off).
  `NER_SERVER_LOG_LEVEL=DEBUG` 로 상세 on.
- [x] **AC4 무회귀·문서**: 기존 계약·전송 불변, `tests/server` green, ruff
  green, `src/server/CLAUDE.md`·`docker/server/{.env.example,compose}` 갱신.

## 설계 결정

- **미들웨어 위치 = 최외곽**: `add_middleware` 는 나중 등록이 바깥이라 `Request
  LogMiddleware` 를 `BodySizeLimitMiddleware` 뒤에 등록해, 전송 계층 413 을
  포함한 최종 상태·전체 지연을 관측하고 request-id 를 심는다.
- **핸들러→미들웨어 meta 전달 = scope 공유**: 순수 ASGI 미들웨어는 scope 를
  복제하지 않으므로, 핸들러가 `request.state.ner_meta`(= `scope['state']`)에
  lang·batch·entities 를 채우면 응답 완료 후 미들웨어가 그대로 읽는다. BaseHTTP
  Middleware 의 contextvar/state 전파 이슈를 피한다. 미기록(비-NER·예외)이면
  기본 요약만.
- **503 은 거절로 분류**: 모델 미로드(503)는 5xx 지만 서버 결함이 아니라
  거절이라, `status >= 500` 보다 사유 매핑(`_REASONS`)을 먼저 검사해
  `reason=model_unavailable` 로 남긴다. 500(진짜 결함)만 "failed".
- **레벨 정책으로 로그량 억제**: 성공을 DEBUG 로 두어 기본 INFO 운영에선
  시작 로그 + 경고/오류만 보이게 한다(요청당 라인 폭주 방지).

## 변경 요약

- `src/server/request_log.py`(신규): `RequestLogMiddleware`(순수 ASGI) +
  `_REASONS` 매핑 + `_request_id`/`_emit`.
- `src/server/app.py`: 미들웨어 등록, `ner` 핸들러 `request: Request` +
  `request.state.ner_meta` 기록, `_unhandled` rid 상관, `_lang_summary`.
- `src/server/config.py`·`__main__.py`: `log_level`(env `NER_SERVER_LOG_LEVEL`),
  `uvicorn.run(access_log=False)`.
- `docker/server/{.env.example,docker-compose.yml}`: `NER_SERVER_LOG_LEVEL` 전달.
- `src/server/CLAUDE.md`: 로깅·관측성 섹션 + env/파일 표.
- `tests/server/test_request_log.py`(신규): 11 tests.

## 검증

- 테스트: `uv run pytest tests/server/ -m "not live"` → **91 passed**(신규 11),
  ruff green.
- 스모크(육안): 성공 DEBUG(`status=200 latency_ms=.. lang=ja batch=1
  entities=1 rid=..`)·거절 WARNING(`reason=payload_too_large`/`model_unavailable`)
  ·`X-Request-ID` 에코가 설계와 일치. scope 공유 meta 전달 정상.

## 참고 — 로그 출력 경로

로그는 **stderr(콘솔) 스트리밍**(파일 미출력). 컨테이너는 `docker/server/
logs.sh`(= `docker logs`), 로컬은 실행 터미널. 파일·구조화 로깅은 별도 트랙.
