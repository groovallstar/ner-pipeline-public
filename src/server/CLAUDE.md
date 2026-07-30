# src/server/ — ja·vi NER REST API 서비스

학습된 BERT 분류기(`/data/ner/{ja,vi}/model/`)를 감싸 HTTP 로 NER 추론을
제공한다. `ner.classifier` 의 data_utils(인코딩·디코드)와
confidence_threshold(임계값 fit·apply)에 의존하고, 학습·평가 모듈은
import 하지 않는다.

이 문서는 **모듈 오리엔테이션**(어느 파일이 무엇을 하고 왜 그렇게 갈랐나)이다.
소비자에게 주는 계약 문서는 따로 있다 — `docs/manual/rest-api-spec.md`(요청·응답
스키마·상태코드 명세)와 `docs/manual/rest-api-integration-guide.md`(연동 절차).
계약이 바뀌면 그 둘도 같이 고쳐야 한다.

## 기동

```bash
python -m server                 # 0.0.0.0:8008, /data/ner 로드
python -m server --host 127.0.0.1 --port 9000 --model-root /abs/root
bash src/server/scripts/run_local.sh --port 9000   # 로컬 GPU 0 고정 기동
```

설정은 `NER_SERVER_*` 환경변수: `MODEL_ROOT`(기본 `/data/ner`)·`MAX_LENGTH`
(256)·`MAX_CHARS`
(20000, 텍스트 1건)·`MAX_BATCH`(64, 배치 개수)·`MAX_TOTAL_CHARS`(100000,
배치 char 합산 — 요청당 작업량 가드)·`MAX_BODY_BYTES`(2MB, 요청 바디 바이트
상한 — 파싱 전 전송 계층 가드)·`MAX_CONCURRENCY`(8, 동시 추론 상한)·
`MAX_QUEUE`(32, 대기 큐 깊이)·`ACQUIRE_TIMEOUT_S`(10, 세마포어 대기 타임아웃
초)·`API_KEY`(미설정 시 인증 off)·`LOG_LEVEL`(INFO, `DEBUG` 로 요청별 상세
켬)·`LOG_FILE`(주간 회전 파일 로그 경로·일주일 보관, 기본
`/tmp/ner-server.log`·빈 값=stderr만)·`HOST`·`PORT`. 웹 데모 번역(additive,
기본 비활성): `TRANSLATE_ENABLED`(false)·`TRANSLATE_BASE_URL`
(`http://localhost:8081/v1`)·`TRANSLATE_MODEL`(활성 시 필수)·`TRANSLATE_API_KEY`
·`TRANSLATE_TIMEOUT_S`(30).

컨테이너 배포(내부망 별도 프로세스 소비자용)는 `docker/server/`(compose +
라이프사이클 + `.env.example`; 상세: `docker/server/CLAUDE.md`).

## API

| 엔드포인트 | 설명 |
|---|---|
| `POST /v1/ner` | 단일 `{text, lang?}` 또는 배치 `{texts:[...], lang?}`. `lang` 생략 시 텍스트별 자동감지. 신뢰도 임계값은 모델 로드 시 자동 적용 |
| `POST /v1/translate` | **웹 데모 전용**(OpenAPI 미노출) — `{text, lang, spans}`(spans=`/v1/ner` 결과)를 한국어로 번역. PII 는 마스킹-복원으로 원문 그대로 보존, 고유명사는 한글 음차. 기본 비활성→503(`NER_SERVER_TRANSLATE_*` 로 활성). `/v1/ner` 과 같은 `max_chars` 상한을 재사용하므로 초과→413, 미지원 lang·비정상 span→400 도 낸다. `/v1/ner` 계약과 독립 |
| `GET /v1/translate/status` | **웹 데모 전용**(OpenAPI 미노출) — `{enabled, available}`. `available` 은 토글 ON + vLLM 백엔드 liveness. UI 가 페이지 로드 시 1회 조회해 번역 버튼을 켜고, 미가용이면 계속 끈다(폴링 없음) |
| `GET /` | 내부 개발·데모용 웹 UI(자족적 HTML, 동일 출처로 `/v1/ner`·`/v1/translate` 호출·CORS 불필요). 인증·Swagger 미노출 |
| `GET /health` | 언어별 모델 로드 상태 + thresholds 존재 여부(인증 없음) |

응답 span 은 canonical `{label, start_char, end_char, text}`
(`.jsonl` 데이터 관례와 일치). 단일 → `{lang, entities}`, 배치 →
`{results:[{lang, entities}, ...]}`(입력 순서 1:1). `lang` 생략 시 자동감지가
ja·vi 신호를 못 찾으면 **`200 + {lang:"unsupported", entities:[]}`**(에러
아님, 모델 미호출) — 배치는 항목별 부분성공. 명시 `lang` 이 미지원이면 400
(클라이언트 계약). 에러는 구조화 `{error: {status, message}}` — 잘못된 요청
(lang·text/texts 택일)→400, 크기 한도(max_chars·max_batch·max_total_chars)
초과→413, 동시성 한도 초과(큐 만석·대기 타임아웃)→429, 모델 미로드→503,
API-key 불일치→401(헤더 이름은 `x-api-key`). 요청 바디가 `max_body_bytes`
초과면 파싱 전에 413(전송 계층 가드 — chunked 우회 포함). Pydantic 검증
실패→422, 미처리 예외→500 도 모두 동일 봉투로 감싸고 500 은 내부 메시지를
노출하지 않는다. 라우터가 내는 라우트 미매칭→404·메서드 불일치→405 도 같은
봉투다(405 는 `Allow` 헤더 유지) — 소비자는 에러 파싱 경로를 하나만 두면
된다. 봉투 통일은 예외 핸들러를 `starlette.exceptions.HTTPException`(부모)에
걸어 얻는다: `fastapi.HTTPException` 에만 걸면 라우터가 부모 클래스를 직접
던지는 404·405 가 FastAPI 기본 `{"detail": ...}` 로 샌다.

## 사용 예시 (curl)

```bash
# 단일(자동감지) — ja
curl -s -X POST localhost:8008/v1/ner \
  -H 'Content-Type: application/json' \
  -d '{"text":"織田信長は東京都千代田区に住んでいた。"}'
# → {"lang":"ja","entities":[
#      {"label":"PER","start_char":0,"end_char":4,"text":"織田信長"},
#      {"label":"LOC","start_char":5,"end_char":12,"text":"東京都千代田区"}]}

# 배치(혼합 언어, 텍스트별 감지)
curl -s -X POST localhost:8008/v1/ner \
  -H 'Content-Type: application/json' \
  -d '{"texts":["トヨタは日本の会社です。","Hà Nội là thủ đô."]}'
# → {"results":[{"lang":"ja","entities":[...]},{"lang":"vi","entities":[...]}]}

# 언어 명시(자동 감지 대신 직접 지정)
curl -s -X POST 'localhost:8008/v1/ner' \
  -H 'Content-Type: application/json' -d '{"text":"...","lang":"ja"}'

# 미지원 입력(영어 등) → 200 + 빈 결과(에러 아님, 모델 미호출)
curl -s -X POST localhost:8008/v1/ner \
  -H 'Content-Type: application/json' -d '{"text":"plain English"}'
# → {"lang":"unsupported","entities":[]}

# 계약 에러: 택일 위반 → 400 / 텍스트 크기 초과 → 413 (status 코드만 확인)
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8008/v1/ner \
  -H 'Content-Type: application/json' -d '{}'

# 헬스 / OpenAPI UI
curl -s localhost:8008/health   # {"status":"ok","langs":{...}}
# 브라우저: GET /docs
```

## 파일

| 파일 | 역할 |
|------|------|
| `config.py` | `ServerConfig` — env 설정·언어별 경로(`model_dir`/`thresholds_path`) |
| `detect.py` | `detect_lang` — ja·vi **양성 감지** + 미지원 명시. 가나→ja, vi-변별 코드포인트(horn·hook·dot 결합부호+đ)→vi, 그 외→`unsupported`(vi 폴백 안 함). `DETECTORS` (신호,언어) 레지스트리로 확장 — 스크립트 언어 추가는 한 줄. 근거: `docs/reports/language-detection-benchmark.md` |
| `chunking.py` | `split_for_length` — max_length 초과 입력을 문장 단위로 쪼개 `(substring, base_offset)` 반환(원문 char offset 보존) |
| `inference.py` | `LangModel`(모델·토크나이저·임계값 1회 로드·재사용; 단건 `predict`·cross-text 배치 `predict_many`)·`ModelRegistry`(언어별 보관·`predict_batch` 언어별 묶음, 미로드→`ModelUnavailable`→503). 추론은 fp32 전용(단건·배치 결정적), 입력은 NFC 정규화. 임계값은 `confidence_threshold` — graceful(파일 없으면 raw), canonical 변환 전 내부 span 에 적용 |
| `concurrency.py` | `ConcurrencyGuard`(async) — 전역 세마포어로 동시 in-flight ≤ `MAX_CONCURRENCY`, 대기 큐 `MAX_QUEUE`·타임아웃 `ACQUIRE_TIMEOUT_S` 로 bound, 초과 시 `Overloaded`→429 |
| `limits.py` | `BodySizeLimitMiddleware`(순수 ASGI) — 라우팅·인증 이전에 요청 바디를 `MAX_BODY_BYTES` 로 bound. Content-Length 조기 거부 + chunked 스트리밍 누적 거부(우회 차단), 초과 시 413 봉투. 전송 계층 메모리 고갈 가드 |
| `request_log.py` | `RequestLogMiddleware`(순수 ASGI, 최외곽) — 요청별 request-id 생성·`X-Request-ID` 에코, 지연·결과를 한 줄로. 성공 2xx→DEBUG(기본 침묵), 거절 4xx·503→WARNING(사유 태그). 핸들러가 `request.state.ner_meta`(lang·batch·entities)를 채워 성공 로그에 실린다 |
| `app.py` | `create_app(registry, config, translator)` — FastAPI 라우트(async)·Pydantic·인증·에러. 추론은 guard 안 `run_in_threadpool` 로 실행. registry 는 `predict`/`predict_batch`/`health` 를 가진 객체면 됨(실모델 또는 stub). `translator`(옵션)는 `/v1/translate` 용, None 이면 503. `GET /` 은 임포트 시 1회 읽은 `static/index.html` 을 그대로 반환 |
| `translate.py` | `LLMTranslator`·`build_translator` — 온프렘 LLM(OpenAI 호환) 마스킹-복원 번역. PII span 을 sentinel 로 가려 번역기에 미노출·복원 시 원문 그대로 보존(소실 시 부재, 훼손 없음), 고유명사 음차. sentinel 을 감싸는 괄호는 `SentinelFormat` 으로 갈아끼운다 — 표기가 번역기 토크나이저에 종속되기 때문이며 기본값은 종전 `【…】` 그대로다. `/v1/ner`·`inference.py` 무의존 additive |
| `static/index.html` | 내부 개발·데모용 웹 UI(자족적 HTML+vanilla JS, 빌드·신규 의존성 없음). 텍스트 입력 + 언어 셀렉터(auto/ja/vi) → 동일 출처 `/v1/ner` 호출 → 개체를 원문 위 라벨별 색상 하이라이트. 입력을 NFC 정규화해 offset 정합, code-point 슬라이스로 astral 문자 대응. **한국어 번역 보기**(온디맨드 버튼)도 여기 있다 — 페이지 로드 시 `/v1/translate/status` 를 1회 조회해 버튼을 켜거나 끄고(폴링 없음), 누르면 `/v1/translate` 를 호출한다 |
| `scripts/run_local.sh` | 호스트 로컬 기동 래퍼(GPU 0 고정, `--port` 전달) |
| `scripts/example_client.py` | 내부 소비자용 최소 레퍼런스 `NERClient` + 자기검증 (`python -m server.scripts.example_client`) |
| `scripts/throughput/bench.py` | 처리량·지연 측정 하네스 (근거: `docs/reports/server-inference-throughput.md`) |
| `__main__.py` | uvicorn 기동 진입점 + 로깅 구성(`_configure_logging` — stderr + 주간 회전 파일) |

## 추론 경로

`encode_row`(data_utils, 토크나이저 capability 분기) → model logits →
softmax → argmax+conf → `decode_bio_to_spans`(score=conf_mean) → canonical
변환 → 임계값 로드 시 자동 적용(graceful — 파일 없으면 raw). 긴 입력은 chunk 분할, 배치 요청(`texts`)은
언어별로 묶어 전 chunk 를 `[N, max_length]` 한 forward 로 추론하고(B=1 이면
단건과 동일) 글로벌 offset 으로 병합 — GPU 병렬로 장문/배치에서 가속된다.
추론은 fp32 전용이라 단건·배치가 같은 커널을 타 배치화가 결과를 바꾸지 않고,
입력은 NFC 로 정규화한다(NFD span 깨짐 방지).
추론은 전역 `ConcurrencyGuard` 안에서 실행돼 동시 부하를 bound 한다. 처리량·
정밀도 측정·동결은 `scripts/throughput/`·`docs/reports/server-inference-throughput.md`.

## 로깅·관측성

요청 로깅은 **평상시 조용, 필요할 때 상세**를 원칙으로 한다(로그량 억제,
`request_log.py`).

- **성공(2xx)** → DEBUG 한 줄(`request status=200 latency_ms=.. lang=..
  batch=.. entities=.. rid=..`). 기본 레벨 INFO 에선 침묵한다.
- **거절(400·401·413·422·429·503)** → WARNING 한 줄(`request rejected
  status=.. reason=.. path=.. rid=..`). 상시 남아 부하 셰딩·인증 실패가 보인다.
  503(모델 미로드)은 5xx 지만 서버 결함이 아니라 거절로 분류한다.
- **미처리 예외(500)** → 트레이스백을 ERROR 로(rid 포함, `app._unhandled`).
  요약 라인은 중복 방지로 생략(예외는 send 없이 전파돼 미들웨어 경로를 안 탄다).
- 응답에 `X-Request-ID` 헤더를 실어 로그 라인과 상관지을 수 있다(수신 헤더가
  있으면 에코, 없으면 8-hex 생성). **미처리 예외(500)는 예외다** — 헤더 주입이
  미들웨어의 `send` 래퍼 안에서 일어나는데 예외는 그 호출을 통과해 밖으로
  전파되므로, Starlette 바깥 층이 만든 500 응답에는 헤더가 붙지 않는다. 위
  "요약 라인 생략" 과 같은 원인이다 — 이때 상관짓는 단서는 ERROR 트레이스백에
  실린 rid 다.

레벨은 `NER_SERVER_LOG_LEVEL`(기본 INFO)로 조정한다 — 요청별 상세가 필요하면
`DEBUG`. uvicorn 기본 액세스 로그는 꺼서(`access_log=False`) 요청 라인을
`RequestLogMiddleware` 가 단독 소유한다(요청당 이중 로그 방지).

**출력** — 로그는 stderr 로 스트리밍하고(컨테이너는 `docker/server/logs.sh`
= `docker logs`, 로컬은 실행 터미널), 동시에 `NER_SERVER_LOG_FILE`(기본
`/tmp/ner-server.log`)에 파일로도 남긴다. 파일은 **매주 회전(월요일)해 직전
1주치만 보관**하고 오래된 파일은 자동 삭제한다(TimedRotating, `__main__.
_configure_logging`). 빈 값이면 stderr 만. 파일 열기 실패는 stderr 로깅을
유지한 채 경고만 낸다. 컨테이너 안 `/tmp` 는 컨테이너-로컬(재시작 시 휘발)이라
호스트에서 보려면 볼륨 마운트 경로로 `LOG_FILE` 을 바꾼다.

## 테스트·검증

- **pytest**: `uv run pytest tests/server/` — 계약·전송은 stub registry 로
  모델 없이 CI 가능, 모델 통합(offset 정합성·ja parity)은 `/data` 있을 때만
  (`pytest.skip` 가드). ja parity 는 `eval_*` 스크립트를 import 하지 않고
  `/data/ner/ja/metrics.json` 운영점과 대조한다.
- **소비자 예제·자기검증(python)**: `python -m server.scripts.example_client
  --base-url http://localhost:8008` — 내부 소비자가 서버를 호출하는 최소
  레퍼런스(`NERClient`). 단일·배치·미지원·계약 에러(400·413·429)를 실서버
  대상으로 호출·검증하고 PASS/FAIL 종료코드를 낸다.
- **실서버 pytest**: `tests/server/test_live_server.py`(`live` 마커) — 서버를
  서브프로세스로 띄워 httpx 로 검증. 모델 로드에 의존하므로 `/data` 없으면
  skip. `uv run pytest -m live` 로 따로 돌릴 수 있다.

## 번역 엔진 — sentinel 표기가 엔진에 종속된다

번역기 후보를 비교하던 하네스(`scripts/translate_bench/`)는 제거됐다. 번역은
NER 에 딸린 부가 기능이라 품질이 엔진 선정 기준이 아니고, 상시 돌 하네스를
유지할 값이 없었다. 과거 측정 경위·엔진 선정 근거는
`docs/reports/translate-engine-lightweight-benchmark.md` 에 남아 있다.

하네스와 함께 사라지지 않는 것이 하나 있다 — **sentinel 을 감싸는 괄호는
번역기의 토크나이저에 종속된다.** 그래서 `translate.py` 가 표기를
`SentinelFormat` 으로 받아 번역기가 고르게 한다.

- 기본 `【…】`(lenticular bracket)는 LLM 경로에서 생존율 100% 다.
- 전용 NMT(NLLB) 계열은 SentencePiece 어휘에 그 글자가 없어 **양쪽 괄호가
  `<unk>` 로 죽는다.** sentinel 본체는 멀쩡히 통과하는데 복원이 괄호째
  매칭하니 전량 실패로 집계된다 — 근거는
  `certified/translate_bench/nllb-1.3b-sentinel-ascii-2080ti/summary.json`
  (현행 표기 0/186, ASCII 괄호 186/186). 괄호만 ASCII 로 옮기면 nonce 를
  포함해 복원된다.

전용 NMT 를 백엔드로 고를 수 있게 하는 작업은 별건이며, 그때 함께 필요한 것이
둘 더 있다 — **문장 단위 분할**(문장 모델이라 통짜 입력의 뒷문장을 버린다)과
**반복 억제**(억제 없이 beam search 를 돌리면 한 어절을 `max_new_tokens` 까지
되풀이해 출력을 통째로 버리는 레코드가 나온다).
