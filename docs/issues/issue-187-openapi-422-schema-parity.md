# issue-187: OpenAPI 422 선언이 실제 에러 봉투와 불일치

- Issue: https://github.com/groovallstar/ner_pipeline/issues/187
- PR: https://github.com/groovallstar/ner_pipeline/pull/188
- 브랜치: `feat/issue-187-openapi-422-schema-parity`
- 승인일: 2026-07-23

## 목적

`/openapi.json` 의 `POST /v1/ner` 422 선언을 실제 응답 형태와 일치시킨다.
런타임 응답은 이미 옳으므로 **선언만** 고치는 작업이다.

## 범위

- 포함: `src/server/app.py` 의 422 `responses` 선언 + 봉투 Pydantic 모델,
  `tests/server/test_api.py` 계약 테스트
- 제외: 런타임 응답 자체. 400·413·429·500·503 의 동작·형식. 봉투에 필드별
  `detail` 을 되살리는 것(#174 의 의도적 결정이라 범위 밖)

## 현 상태 (fact)

#174 가 `RequestValidationError` 핸들러를 더해 422 응답을 구조화 봉투로
통일했으나, 라우트의 `responses` 선언을 함께 덮지 않아 OpenAPI 에는 FastAPI
기본 `HTTPValidationError` 가 남아 있었다.

```
실제 응답:   {"error":{"status":422,"message":"request validation failed (1 error(s))"}}
OpenAPI 선언: {"$ref":"#/components/schemas/HTTPValidationError"}
```

연동 가이드가 소비자에게 `GET /openapi.json` 을 기계 계약으로 안내하므로,
스키마에서 클라이언트를 생성·검증하는 쪽이 틀린 계약을 받는다.

## 결정 로그 (append-only)

- 2026-07-23: 봉투 모델(`ErrorResponse`)에 `detail` 필드를 두지 않기로 결정.
  #174 가 "필드 내부 구조는 노출하지 않고 개수만 요약한다"고 명시했으므로,
  선언은 실제 응답(`status`·`message`)을 그대로 반영해야 한다. 선언을 실제보다
  넓게 잡으면 없는 필드를 있는 것처럼 약속하게 된다.
- 2026-07-23: 계약 테스트를 `$ref` 이름 확인에서 멈추지 않고 **선언 필드 ↔
  실제 응답 필드 대조**까지 하도록 작성. 이름만 맞고 모양이 갈리면 통과하는
  허수 테스트가 되기 때문이다.

## 구현 단계

- [x] `ErrorBody`·`ErrorResponse` 모델 추가
- [x] 라우트 `responses={422: ErrorResponse}` 선언
- [x] `test_openapi_422_declares_error_envelope` 추가
- [x] 연동 가이드 §3.5 "신호" → "판별 문자" 문구 정리(별도 커밋)

---

## 구현 결과

라우트에 `responses={422: {'model': ErrorResponse, ...}}` 를 선언해 FastAPI
기본 `HTTPValidationError` 를 덮었다. 모델은 `_error()` 가 실제로 내는 형태
(`{error: {status, message}}`)를 그대로 옮긴 것이라 선언과 응답이 같은 모양이다.
런타임 코드 경로는 건드리지 않았다.

곁들여 연동 가이드 §3.5 의 "신호"(코드의 `signal` 을 그대로 옮긴 내부 어휘)를
"판별 문자"로 바꿨다 — 실제 판별 근거가 가나·vi 변별 부호라는 *문자*이므로
외부 소비자에게는 실체를 직접 쓰는 편이 정확하다.

## 검증

- 테스트: `uv run pytest tests/server/` → **84 passed**(신규 1개).
  `uv run ruff check src/server/ tests/server/` → All checks passed
- AC 대조 — 4개 기준 모두 충족:

| 기준 | 확인 | 결과 |
|---|---|---|
| OpenAPI 422 스키마 ↔ 실제 응답 일치 | `$ref` 가 `ErrorResponse`, `HTTPValidationError` 는 components 에서 사라짐 | PASS |
| 런타임 422 응답 무변화 | `{"error":{"status":422,"message":"request validation failed (1 error(s))"}}` 동일 | PASS |
| 400·413·429·500·503 무변화 | 기존 계약·하드닝 테스트 전수 통과 | PASS |
| 계약 테스트 추가 + 통과 | `test_openapi_422_declares_error_envelope` | PASS |
