# issue-174: REST API 공격 표면 하드닝 — 바디 DoS·타이밍·에러 봉투

- Issue: https://github.com/groovallstar/ner_pipeline/issues/174
- PR: https://github.com/groovallstar/ner_pipeline/pull/175
- 브랜치: `feat/issue-174-restapi-attack-hardening`
- 승인일: 2026-07-14

## 목적

ja·vi NER REST API(`POST /v1/ner`·`GET /health`)에 적대적 입력 테스트를
수행해 결함을 찾고, 확인된 **전송·계약 계층** 3결함을 하드닝한다. 추론·모델
계층은 견고함이 확인됐다(아래 §현 상태).

## 범위

- 포함: 요청 바디 크기 상한(전송 가드), 상수시간 API 키 비교, 에러 봉투
  통일(422·500), 적대적 회귀 테스트, docker 배포 knob 노출.
- 제외: 인증 체계 자체 재설계(내부 신뢰망 전제 유지), rate-limit·access
  log(별도 트랙), 404/405 봉투 통일(선재 동작·범위 외), 모델·metric·gold
  변경(좌표계 불변 — feature-only).

## 성공 기준 (AC)

- [x] **AC1(DoS)**: 설정 바이트 상한 초과 바디를 전량 버퍼링 전에 413.
  Content-Length 조기 거부 + chunked 스트리밍 누적 거부(우회 차단).
  `NER_SERVER_MAX_BODY_BYTES`(기본 2MB), 인증 이전 단계.
- [x] **AC2(timing)**: API 키 비교를 `secrets.compare_digest`(bytes)로,
  401/200 동작 불변.
- [x] **AC3(envelope)**: `RequestValidationError`(422)·미처리 예외(500)를
  `{error:{status,message}}` 봉투로, 500 은 내부 메시지 미노출. 기존
  503·429·400·413 불변.
- [x] **AC4(tests)**: `tests/server/test_hardening.py` 8건(stub, 모델 없이).
- [x] **AC5**: 기존 `tests/server` green(80 passed), ruff clean.

## 현 상태 (fact)

- **바디 DoS(수정 전→후)**: 실제 uvicorn 에서 200MB 스트리밍 바디를 전량
  수신 후 413 → 수정 후 Content-Length 200MB 는 0.11s 조기 거부, chunked
  10MB 는 스트리밍 누적으로 413(봉투 정상). `API_KEY` on 이어도 초과 바디는
  401 이 아니라 413 이 먼저(인증 이전 단계).
- **에러 봉투**: 수정 전 Pydantic 실패 422 `{detail:[...]}`·모델 예외 500
  `Internal Server Error`(text/plain) → 수정 후 둘 다 `{error:{...}}`,
  500 에 내부 예외 문자열·트레이스백 미노출.
- **견고 확인(무변경)**: 실제 ja 모델 18종 적대적 입력(astral 이모지·
  surrogate·결합문자 폭탄·ZWJ·청킹 유발 장문) 크래시 0·offset 불변식
  (`text[start:end]==span.text`) 위반 0. 세마포어 취소폭풍(200 대기자)
  퍼밋 누수 0(Python 3.13). detect_lang 유니코드 크래시 0. 깊은 JSON·
  메서드·Content-Type 안전 처리. → 추론·모델 계층 결함 없음.

## 결정 로그 (append-only)

- 2026-07-14: 좌표계(정답·채점규칙·분할) 불변 확인 → spine 동결 불요,
  feature-only 하드닝으로 진행(기존 불변식 게이트에 건다).
- 2026-07-14: 바디 상한 강제를 Content-Length-only 대신 **스트리밍 누적**
  까지 확장 — chunked transfer-encoding 우회를 닫기 위함(라이브 uvicorn 으로
  chunked 10MB 413 실증). 상한 2MB 고정(합법 최대 요청 worst-case ~600KB 의
  여유).
- 2026-07-14: develop·create_eml_no_delete 가 타 워크트리 체크아웃 상태라
  현재 HEAD(996b30d, develop 의 조상·server 코드 동일)에서 분기 — 선재
  미커밋 docs 변경은 타 세션 소유라 커밋에서 제외.
- 2026-07-14: 결과-시점 반박자(Sonnet code-reviewer, 격리) PASS 후, 지적된
  비차단 관찰(새 knob 이 docker compose·.env.example 에 미노출)을 반영 —
  형제 knob 과 일관되게 `NER_SERVER_MAX_BODY_BYTES` 노출.

## 변경 요약

- `src/server/limits.py`(신규): `BodySizeLimitMiddleware`(순수 ASGI) —
  Content-Length 조기 거부 + `receive` 랩으로 스트리밍 바이트 누적 거부.
  초과 시 413 봉투를 직접 전송하고 앱에 `http.disconnect` 주입, 이후 앱
  응답은 `guarded_send` 로 삼켜 이중 전송을 막는다.
- `src/server/app.py`: 미들웨어 배선(라우팅·인증 앞단), 키 비교
  `secrets.compare_digest`(bytes), `RequestValidationError`(422)·generic
  `Exception`(500, 로그만·응답 비노출) 핸들러 추가.
- `src/server/config.py`: `max_body_bytes`(기본 2MB) + env 파싱.
- `tests/server/test_hardening.py`(신규): AC1~3 회귀 8건.
- docs: `src/server/CLAUDE.md`, `docker/server/{compose,.env.example}` knob.

## 검증

- 테스트: `uv run pytest tests/server/ -q` → **80 passed**(신규 8 + 기존
  72, ja 모델 통합 포함). `ruff check src/server/ tests/server/` clean.
- 라이브: stub uvicorn 대상 Content-Length 413(0.11s)·chunked 413·정상
  200 실증. 실제 ja 모델 18종 적대적 퍼징 크래시·offset 위반 0.
- 반박자: 결과-시점 격리 반박자(Sonnet) **PASS** — 미들웨어 이중응답/hang,
  Exception 핸들러 MRO 가로채기 두 우려를 소스 추적 + 재현으로 반증 실패.

## 관련 커밋

- `1749387`: feat(server) 바디 상한·상수시간 키·에러 봉투 + 회귀 8건
