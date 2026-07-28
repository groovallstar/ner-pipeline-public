# issue-191: 404·405 가 에러 봉투를 벗어남 — 라우트 미매칭 예외 미포착

- Issue: https://github.com/groovallstar/ner-pipeline/issues/191
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `fix/issue-191-route-error-envelope`
- 승인일: 2026-07-28

## 목적

서버가 내는 **모든** 에러 응답을 `{"error": {"status", "message"}}` 봉투 하나로
통일한다. 현재 404·405 만 FastAPI 기본 `{"detail": ...}` 로 새어, 소비자가 에러
파싱 경로를 두 갈래로 들고 있어야 한다.

## 범위

- 포함: 예외 핸들러 등록 클래스 교체, `_error()` 의 응답 헤더 통과, 회귀 테스트,
  `src/server/CLAUDE.md` 에러 계약 문단
- 제외: 미완성 바디 커넥션이 무기한 유지되는 건(아래 §범위 밖 참조)

## 현 상태 (fact)

수정 전 실측 — 같은 서버가 두 가지 에러 모양을 낸다.

| 요청 | 상태 | 바디 |
|---|---|---|
| `POST /v1/nonexistent` | 404 | `{"detail":"Not Found"}` |
| `GET /v1/ner` | 405 | `{"detail":"Method Not Allowed"}` |
| `POST /v1/ner {}` | 400 | `{"error":{"status":400,"message":"..."}}` |

원인은 예외 핸들러가 **자식 클래스에만** 걸려 있다는 것이다. 핸들러가 직접 던지는
400·401·413 은 `fastapi.HTTPException` 이라 잡히지만, 라우트 미매칭·메서드 불일치는
**부모인 `starlette.exceptions.HTTPException`** 으로 라우터가 던진다. Starlette 의
핸들러 조회는 던져진 예외의 MRO 를 거슬러 오르므로, 자식에 건 핸들러는 부모 인스턴스와
매칭되지 않고 FastAPI 기본 핸들러로 샌다. 부모에 걸면 자식은 서브클래스라 함께 잡힌다.

여기에 갈아끼울 때 걸리는 제약이 하나 더 있다. FastAPI 기본 핸들러는 `exc.headers` 를
응답에 실어주는데 기존 `_error()` 에는 헤더 인자가 없어, 그대로 교체하면 RFC 7231 이
405 에 요구하는 `Allow` 헤더가 사라진다.

## 성공 기준

- [x] `POST /v1/nonexistent` → 404 + 봉투, 응답에 `detail` 키 없음
- [x] `GET /v1/ner` → 405 + 봉투, `Allow: POST` 헤더 보존
- [x] 기존 봉투 경로(400·401·413·422·429·500·503) 무회귀
- [x] 위 3건을 stub registry 회귀 테스트로 고정(모델·GPU 불요)
- [x] `uv run pytest tests/server/` 전건 통과 + ruff clean
- [x] `src/server/CLAUDE.md` 에러 계약 문단에 404·405 반영

## 구현 결과

`src/server/app.py` 세 곳만 바뀐다.

1. `from starlette.exceptions import HTTPException as StarletteHTTPException`
2. `_error(status, message, headers=None)` — 예외가 실어 보낸 응답 헤더를
   `JSONResponse` 로 통과시킨다. 봉투로 감싸면서 프로토콜 헤더를 잃지 않게 하는 것이
   목적이라, 405 의 `Allow` 외에 앞으로 붙을 `WWW-Authenticate`·`Retry-After` 도
   같은 경로로 살아난다.
3. `@app.exception_handler(...)` 의 대상을 부모 클래스로 교체하고 `exc.headers` 전달.

나머지 핸들러(422·429·503·500)·라우트·미들웨어는 손대지 않았다.

## 검증

- 테스트: `uv run pytest tests/server/` → **118 passed**, `ruff check src/server/
  tests/server/` → clean
- 회귀 고정: `tests/server/test_hardening.py` 의 AC3(에러 봉투 일관성)에 2건 추가 —
  404 봉투(+`detail` 키 부재)와 405 봉투(+`Allow` 헤더 보존)
- 수정 후 실측:

  ```
  404: {"error":{"status":404,"message":"Not Found"}}
  405: {"error":{"status":405,"message":"Method Not Allowed"}}  allow= POST
  400: {"error":{"status":400,"message":"provide exactly one of 'text' or 'texts'"}}
  ```

## 범위 밖 — 미완성 바디 커넥션

같은 점검에서 확인된 별건이다. `Content-Length` 를 선언하고 바디를 일부만 보내면
서버가 커넥션을 끊지 않는다(20개를 붙잡아 30초 뒤 20/20 생존). uvicorn 은 바디 수신
타임아웃이 없고 — `timeout_keep_alive` 는 idle keep-alive 전용이다 — `docker/server/`
배포는 uvicorn 을 포트 매핑으로 직접 노출하므로 앞단에 끊어줄 프록시도 없다.

그럼에도 이번 범위에서 뺀 이유는 피해 범위가 좁아서다. `ConcurrencyGuard` 는 바디
파싱 뒤에 걸리므로 슬로우로리스가 추론 슬롯을 점유하지 못하고, 실제로 위 상태에서
정상 요청은 0.02s 에 200 을 받았다. 남는 비용은 fd·소켓 버퍼뿐이고 배포 전제가
내부망이라, 외부 노출이 생기는 시점에 앞단 프록시의 `client_body_timeout` 으로 받는
편이 맞다고 판단했다.

## 후속 작업

없음. 위 범위 밖 항목은 외부 노출 전환 시 재검토한다.
