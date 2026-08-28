# src/server/ — ja·ko·vi NER REST API 서비스

학습된 BERT 분류기(`/data/ner/{ja,ko,vi}/model/`)를 감싸 HTTP 로 NER 추론을
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
`/tmp/ner-server.log`·빈 값=stderr만)·`HOST`·`PORT`.

웹 데모 번역(additive, 기본 비활성)은 `TRANSLATE_ENABLED`(false)로 켜고, 켤 때
**백엔드를 반드시 고른다** — `TRANSLATE_BACKEND`(`llm`|`nllb`, **기본값 없음**)
·`TRANSLATE_MODEL`(두 백엔드 공용 "무슨 모델")·`TRANSLATE_MAX_CONCURRENCY`(2).
백엔드 전용 키는 `llm` 이 `TRANSLATE_BASE_URL`(필수)·`TRANSLATE_API_KEY`·
`TRANSLATE_TIMEOUT_S`(30), `nllb` 가 `TRANSLATE_DEVICE`(`cuda:0`)·
`TRANSLATE_NO_REPEAT_NGRAM`(미설정=자동 계산, 0=끔)이다. 왜 그렇게 갈랐는지와 미지정·오값·
반대편 키가 왜 기동 실패인지는 §번역 백엔드.

컨테이너 배포(내부망 별도 프로세스 소비자용)는 `docker/server/`(compose +
라이프사이클 + `.env.example`; 상세: `docker/server/CLAUDE.md`).

## API

| 엔드포인트 | 설명 |
|---|---|
| `POST /v1/ner` | 단일 `{text, lang?}` 또는 배치 `{texts:[...], lang?}`. `lang` 생략 시 텍스트별 자동감지. 신뢰도 임계값은 모델 로드 시 자동 적용 |
| `POST /v1/translate` | **웹 데모 전용**(OpenAPI 미노출) — `{text, lang, spans}`(spans=`/v1/ner` 결과)를 한국어로 번역. **lang 은 ja·vi 만 받는다**(§번역 대상 언어). PII 는 마스킹-복원으로 원문 그대로 보존, 고유명사는 한글 음차(백엔드가 `llm` 일 때 — `nllb` 는 프롬프트가 없어 지시할 수단이 없다). 기본 비활성→503(`NER_SERVER_TRANSLATE_*` 로 활성). `/v1/ner` 과 같은 `max_chars` 상한을 재사용하므로 초과→413, 미지원 lang·비정상 span→400, **번역 전용 동시성 상한 초과→429**. `/v1/ner` 계약과 독립 |
| `GET /v1/translate/status` | **웹 데모 전용**(OpenAPI 미노출) — `{enabled, available}`. `available` 은 토글 ON + 백엔드가 지금 번역할 수 있음이고, **무엇을 확인하는지는 백엔드가 정한다**(`llm`=원격 엔드포인트 liveness, `nllb`=로드 성공이 곧 가용이라 항상 true). UI 가 페이지 로드 시 1회 조회해 번역 버튼을 켜고, 미가용이면 계속 끈다(폴링 없음) |
| `GET /` | 내부 개발·데모용 웹 UI(자족적 HTML, 동일 출처로 `/v1/ner`·`/v1/translate` 호출·CORS 불필요). 인증·Swagger 미노출 |
| `GET /health` | 언어별 모델 로드 상태 + thresholds 존재 여부(인증 없음) |

응답 span 은 canonical `{label, start_char, end_char, text}`
(`.jsonl` 데이터 관례와 일치). 단일 → `{lang, entities}`, 배치 →
`{results:[{lang, entities}, ...]}`(입력 순서 1:1). `lang` 생략 시 자동감지가
ja·ko·vi 신호를 못 찾으면 **`200 + {lang:"unsupported", entities:[]}`**(에러
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
  -d '{"texts":["トヨタは日本の会社です。","삼성전자는 수원에 있다.","Hà Nội là thủ đô."]}'
# → {"results":[{"lang":"ja",...},{"lang":"ko",...},{"lang":"vi",...}]}

# 언어 명시(자동 감지 대신 직접 지정)
curl -s -X POST 'localhost:8008/v1/ner' \
  -H 'Content-Type: application/json' -d '{"text":"...","lang":"ja"}'

# 미지원 입력(영어·한자만 등) → 200 + 빈 결과(에러 아님, 모델 미호출)
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
| `detect.py` | `detect_lang` — ja·ko·vi **양성 감지** + 미지원 명시. 가나→ja, 한글(음절·자모)→ko, vi-변별 코드포인트(horn·hook·dot 결합부호+đ)→vi, 그 외→`unsupported`(vi 폴백 안 함). 한자만 있는 텍스트도 `unsupported` — ja·ko 가 한자를 공유해 어느 쪽도 가리키지 않으므로 *배타적* 스크립트만 신호로 쓴다. `DETECTORS` (신호,언어) 레지스트리로 확장 — 스크립트 언어 추가는 한 줄. 근거: `docs/reports/language-detection-benchmark.md` |
| `chunking.py` | `split_for_length` — max_length 초과 입력을 문장 단위로 쪼개 `(substring, base_offset)` 반환(원문 char offset 보존) |
| `inference.py` | `LangModel`(모델·토크나이저·임계값 1회 로드·재사용; 단건 `predict`·cross-text 배치 `predict_many`)·`ModelRegistry`(언어별 보관·`predict_batch` 언어별 묶음, 미로드→`ModelUnavailable`→503). 추론은 fp32 전용(단건·배치 결정적), 입력은 NFC 정규화. 임계값은 `confidence_threshold` — graceful(파일 없으면 raw), canonical 변환 전 내부 span 에 적용 |
| `concurrency.py` | `ConcurrencyGuard`(async) — 세마포어로 동시 in-flight ≤ `MAX_CONCURRENCY`, 대기 큐 `MAX_QUEUE`·타임아웃 `ACQUIRE_TIMEOUT_S` 로 bound, 초과 시 `Overloaded`→429. `app.py` 가 **둘을 만든다** — NER 용과 번역 전용(`TRANSLATE_MAX_CONCURRENCY`, 큐 없음) |
| `limits.py` | `BodySizeLimitMiddleware`(순수 ASGI) — 라우팅·인증 이전에 요청 바디를 `MAX_BODY_BYTES` 로 bound. Content-Length 조기 거부 + chunked 스트리밍 누적 거부(우회 차단), 초과 시 413 봉투. 전송 계층 메모리 고갈 가드 |
| `request_log.py` | `RequestLogMiddleware`(순수 ASGI, 최외곽) — 요청별 request-id 생성·`X-Request-ID` 에코, 지연·결과를 한 줄로. 성공 2xx→DEBUG(기본 침묵), 거절 4xx·503→WARNING(사유 태그). 핸들러가 `request.state.ner_meta`(lang·batch·entities)를 채워 성공 로그에 실린다 |
| `app.py` | `create_app(registry, config, translator)` — FastAPI 라우트(async)·Pydantic·인증·에러. 추론은 guard 안 `run_in_threadpool` 로 실행. registry 는 `predict`/`predict_batch`/`health` 를 가진 객체면 됨(실모델 또는 stub). `translator`(옵션)는 `/v1/translate` 용, None 이면 503. `GET /` 은 임포트 시 1회 읽은 `static/index.html` 을 그대로 반환 |
| `translate.py` | 마스킹-복원 + `LLMTranslator`(원격 OpenAI 호환) + `build_translator`(백엔드 선택·설정 검증). 번역 대상 언어 목록 `TRANSLATABLE_LANGS`(=`LANG_NAME` 의 키)도 여기 있다. PII span 을 sentinel 로 가려 번역기에 미노출·복원 시 원문 그대로 보존(소실 시 부재, 훼손 없음), 고유명사 음차. sentinel 을 감싸는 괄호는 `SentinelFormat` 으로 갈아끼운다 — 표기가 번역기 토크나이저에 종속되기 때문이며 기본값은 종전 `【…】` 그대로다. `/v1/ner`·`inference.py` 무의존 additive |
| `translate_nllb.py` | `NLLBTranslator` — 전용 NMT 를 **서버 프로세스 안에서** 돌리는 백엔드(같은 계약: `translate`·`available`). ASCII sentinel·문장 단위 분할·반복 억제가 엔진 속성으로 붙는다(§번역 백엔드). torch·transformers 를 끌어오므로 `build_translator` 가 `nllb` 를 고를 때만 임포트된다 |
| `static/index.html` | 내부 개발·데모용 웹 UI(자족적 HTML+vanilla JS, 빌드·신규 의존성 없음). 텍스트 입력 + 언어 셀렉터(auto/ja/ko/vi) → 동일 출처 `/v1/ner` 호출 → 개체를 원문 위 라벨별 색상 하이라이트. 입력을 NFC 정규화해 offset 정합, code-point 슬라이스로 astral 문자 대응. **한국어 번역 보기**(온디맨드 버튼)도 여기 있다 — 페이지 로드 시 `/v1/translate/status` 를 1회 조회해 버튼을 켜거나 끄고(폴링 없음), 누르면 `/v1/translate` 를 호출한다. 결과가 ko 면 버튼을 **감춘다** — 눌러도 400 이 될 버튼을 회색으로 남기면 "백엔드가 죽었나"로 읽힌다 |
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
- **번역 백엔드**: 계약·설정 검증·동시성은 stub 으로 모델 없이 돈다. `nllb`
  실모델 경로(ja·vi PII 5종 verbatim 보존)는 가중치가 필요해 env 로 연다 —
  `NER_SERVER_TEST_NLLB_MODEL=facebook/nllb-200-distilled-1.3B uv run pytest
  tests/server/test_translate_nllb.py`(미지정이면 skip, device 는
  `NER_SERVER_TEST_NLLB_DEVICE`·기본 `cuda:0`).
- **소비자 예제·자기검증(python)**: `python -m server.scripts.example_client
  --base-url http://localhost:8008` — 내부 소비자가 서버를 호출하는 최소
  레퍼런스(`NERClient`). 단일·배치·미지원·계약 에러(400·413·429)를 실서버
  대상으로 호출·검증하고 PASS/FAIL 종료코드를 낸다.
- **실서버 pytest**: `tests/server/test_live_server.py`(`live` 마커) — 서버를
  서브프로세스로 띄워 httpx 로 검증. 모델 로드에 의존하므로 `/data` 없으면
  skip. `uv run pytest -m live` 로 따로 돌릴 수 있다.

## 번역 대상 언어

번역이 받는 `lang` 은 **NER 이 받는 `lang` 보다 좁다.** `/v1/ner` 은
`SUPPORTED_LANGS`(ja·ko·vi)를 보지만 `/v1/translate` 는
`TRANSLATABLE_LANGS`(ja·vi)를 본다 — 이 엔드포인트는 "한국어로 번역"이라
한국어 원문은 옮길 곳이 없기 때문이다. `lang:"ko"` 는 400 이다.

목록을 따로 두지 않고 `LANG_NAME`(프롬프트에 박히는 언어 이름)의 키에서
유도한다. 이름이 곧 자격이라서다 — 프롬프트에 넣을 이름이 없는 언어는 번역할
수단이 없다. 목록을 별도로 두면 한쪽만 늘어나 조용히 어긋나고, 그때 나타나는
증상은 에러가 아니라 **원문이 그대로 '번역'으로 돌아오는 것**이다.

웹 UI 도 같은 목록을 두고 ko 결과에서는 번역 버튼을 감춘다.

## 번역 백엔드 — 기동 때 고르고, 기본값은 없다

여기는 설정 표면과 그렇게 가른 이유가 정본이다. 함수·알고리즘 수준의 구현
레퍼런스(마스킹-복원 계약·검증 분기·잠금 불변식·테스트 맵)는
`docs/manual/web-demo-translation.md` 에 있다.

번역을 켜면 `NER_SERVER_TRANSLATE_BACKEND` 가 구현을 정한다. **기본값을 두지
않은 것이 이 설계의 핵심이다** — 예전에는 `TRANSLATE_BASE_URL` 이
`http://localhost:8081/v1` 을 기본으로 가져, base_url 을 안 주고 번역을 켜도
서버가 뜨고 그 포트에 떠 있는 아무 모델이나 번역기로 썼다. 설정만 보고 무엇을
부르는지 알 수 없는 상태였다. 지금은 안 고르면 기동이 실패한다.

두 번째 이유는 배치다. 소형 GPU 온프렘(2080Ti 11GB 급, NER 추론과 VRAM 공유)
에서는 원격 LLM 을 따로 띄울 여유가 없을 수 있어, 가볍게 도는 전용 NMT 로
갈아탈 수단이 필요하다.

| | `llm` | `nllb` |
|---|---|---|
| 어디서 도나 | 원격 HTTP(vLLM/OpenAI 호환) | **서버 프로세스 안**(transformers) |
| 필수 키 | `MODEL`·`BASE_URL` | `MODEL`(HF 모델 ID) |
| 전용 키 | `API_KEY`·`TIMEOUT_S` | `DEVICE`·`NO_REPEAT_NGRAM` |
| `available` | 엔드포인트를 찔러 본 결과 | 로드 성공이 곧 가용(항상 true) |
| sentinel | `【…】`(생존율 100%) | `[…]`(ASCII — 아래) |
| 고유명사 음차 | 프롬프트로 지시 | 지시 수단 없음(프롬프트가 없는 모델) |

### 대칭이 아니라서 따라오는 것

인프로세스라 `nllb` 는 **NER 과 같은 GPU 를 쓴다.** "번역은 추론과 독립"이라는
종전 전제가 깨지므로 셋이 함께 움직인다.

- **동시성을 따로 묶는다.** `app.py` 가 번역 전용 `ConcurrencyGuard`(상한
  `TRANSLATE_MAX_CONCURRENCY`, 큐 없음)를 하나 더 만든다. 예산이 분리돼 번역
  폭주가 NER 슬롯을 잠식하지 않고 그 반대도 없다. 초과는 대기 없이 429 —
  번역은 온디맨드 버튼이라 기다리게 하는 것보다 돌려보내는 편이 낫다.
- **한 요청이 만드는 문장 배치도 묶는다.** 긴 입력(`max_chars` 2만 자)이 수백
  문장으로 갈릴 수 있어, 통째로 태우면 VRAM 피크가 입력 길이를 따라간다.
  `translate_nllb.py` 가 8문장씩 나눠 태워 피크를 상한에 가둔다(실측: 120문장
  입력에서 2795 MiB vs 무제한 5113 MiB — 대신 지연은 늘어난다).
- **`available` 의 정의를 백엔드가 쓴다.** 원격은 liveness 확인이 의미가
  있지만 인프로세스는 로드되면 언제나 가용이다(로드 실패면 서버가 안 뜬다).
- **번역기 상태가 요청 사이에 공유된다.** NLLB 토크나이저는 소스 언어 태그를
  인스턴스에 새겨 두고 인코딩할 때 읽는다 — 동시 요청이 겹치면 ja 가 vi 태그로
  인코딩돼 **에러 없이** 엉뚱한 번역이 나온다. 그래서 토크나이저를 만지는
  구간(인코딩·디코딩)을 하나의 잠금으로 직렬화하고 그 불변식을 테스트로
  고정한다. 오래 걸리는 생성만 잠금 밖이라 동시성 이득은 남는다.
- **요청 타임아웃이 없다.** `TIMEOUT_S` 는 원격 호출을 끊는 장치라 인프로세스
  생성에는 걸 데가 없다 — 아주 긴 입력(`max_chars` 상한 근처)은 그 슬롯을
  오래 잡는다. 번역 상한이 낮게(2) 잡혀 있어 여파는 "번역 요청이 429 로
  거절된다"에 그치고 `/v1/ner` 은 영향을 받지 않는다.

### sentinel 표기가 번역기 토크나이저에 종속된다

- 기본 `【…】`(lenticular bracket)는 LLM 경로에서 생존율 100% 다.
- 전용 NMT(NLLB) 계열은 SentencePiece 어휘에 그 글자가 없어 **양쪽 괄호가
  `<unk>` 로 죽는다.** sentinel 본체는 멀쩡히 통과하는데 복원이 괄호째
  매칭하니 전량 실패로 집계된다 — 근거는
  `certified/translate_bench/nllb-1.3b-sentinel-ascii-2080ti/summary.json`
  (현행 표기 0/186, ASCII 괄호 186/186). 괄호만 ASCII 로 옮기면 nonce 를
  포함해 복원된다.

**이 실패는 조용하다.** 마스킹-복원은 소실을 훼손이 아닌 '부재'로 처리하므로
에러가 나지 않는다 — 화면에서 전화번호·이메일이 그냥 사라지고 아무 신호가
없다. 그래서 표기 선택이 백엔드에 딸려 자동으로 바뀐다(`ASCII_SENTINEL`).

전용 NMT 라서 따라오는 나머지 둘도 엔진 속성으로 붙어 있다.

- **문장 단위 분할** — 문장 모델이라 통짜 입력의 뒷문장을 버린다.
- **반복 억제, 단 sentinel 이 복사될 만큼만** — 끄고 beam search 를 돌리면 한
  어절을 `max_new_tokens` 까지 되풀이해 출력을 통째로 버린다(실측: 같은 어절
  99회). 그런데 `no_repeat_ngram_size` 를 작게 잡으면 **sentinel 이 그 금지에
  먼저 걸린다** — 한 문장에 PII 가 둘이면 두 번째 sentinel 은 앞엣것과 토큰 열을
  공유하는데(공유 접두 8토큰) 그걸 못 쓰게 되니 모델이 글자를 바꿔 내놓고 복원이
  실패한다(실측: n=3·9 에서 5개 중 2개 소실, n=12 에서 0개). 그래서 억제 강도의
  하한을 **sentinel 토큰 길이 + 2** 로 계산한다 — nonce 가 프로세스마다 달라
  길이가 변하므로 상수가 아니라 실제 표기를 재서 정한다
  (`resolve_no_repeat_ngram`). 하한 이상이면 설정값을 그대로 쓰고, 더 작은
  양수는 하한으로 올리며 경고를 남긴다. `NO_REPEAT_NGRAM=0` 은 명시적 opt-out.

### 설정 검증 — 기동 시점에 실패시킨다

`build_translator` 가 넷을 본다: 백엔드 미지정 · `llm`·`nllb` 아닌 값 ·
공용 필수키(`MODEL`, `llm` 이면 `BASE_URL`) 누락 · **고른 백엔드에 안 맞는
키**(예: `nllb` + `BASE_URL`). 마지막 것을 실패로 두는 이유는 값이 조용히
무시되면 설정 파일이 실제 동작과 어긋난 채 남기 때문이다. 그래서 백엔드 전용
키는 `config.py` 에서 기본값 없이 `None` 으로 두고(미설정과 설정을 구별해야
검사가 성립한다) 실효 기본값은 그 값을 아는 쪽이 넣는다 — timeout·device 는
`translate.py`, 반복 억제 하한은 토크나이저를 쥔 `translate_nllb.py` 다.

과거 엔진 비교 하네스(`scripts/translate_bench/`)는 제거됐다 — 번역은 NER 에
딸린 부가 기능이라 품질이 엔진 선정 기준이 아니고, 상시 돌 하네스를 유지할 값이
없었다. 측정 경위는 `docs/reports/translate-engine-lightweight-benchmark.md`.
