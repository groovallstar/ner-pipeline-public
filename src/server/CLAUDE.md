# src/server/ — ja·ko·vi·en NER REST API 서비스

학습된 BERT 분류기(`/data/ner/{ja,ko,vi,en}/model/`)를 감싸 HTTP 로 NER 추론을
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
`TRANSLATE_MODEL`·`TRANSLATE_BASE_URL` 을 **반드시 준다**(기본값 없음 — 안 주면
기동 실패). 나머지는 `TRANSLATE_API_KEY`·`TRANSLATE_TIMEOUT_S`(30)·
`TRANSLATE_MAX_CONCURRENCY`(2)다. 왜 기본값을 두지 않는지는 §번역 설정 표면.

컨테이너 배포(내부망 별도 프로세스 소비자용)는 `docker/server/`(compose +
라이프사이클 + `.env.example`; 상세: `docker/server/CLAUDE.md`).

## API

| 엔드포인트 | 설명 |
|---|---|
| `POST /v1/ner` | 단일 `{text, lang?}` 또는 배치 `{texts:[...], lang?}`. `lang` 생략 시 텍스트별 자동감지. 신뢰도 임계값은 모델 로드 시 자동 적용 |
| `POST /v1/translate` | **웹 데모 전용**(OpenAPI 미노출) — `{text, lang, spans}`(spans=`/v1/ner` 결과)를 한국어로 번역. **lang 은 ja·vi·en 만 받는다**(§번역 대상 언어). PII 는 마스킹-복원으로 원문 그대로 보존, 고유명사는 한글 음차(프롬프트 지시). 기본 비활성→503(`NER_SERVER_TRANSLATE_*` 로 활성). `/v1/ner` 과 같은 `max_chars` 상한을 재사용하므로 초과→413, 미지원 lang·비정상 span→400, **번역 전용 동시성 상한 초과→429**. `/v1/ner` 계약과 독립 |
| `GET /v1/translate/status` | **웹 데모 전용**(OpenAPI 미노출) — `{enabled, available}`. `available` 은 토글 ON + 번역 엔드포인트 liveness(모델 목록 조회 1회) 다. UI 가 페이지 로드 시 1회 조회해 번역 버튼을 켜고, 미가용이면 계속 끈다(폴링 없음) |
| `GET /` | 내부 개발·데모용 웹 UI(자족적 HTML, 동일 출처로 `/v1/ner`·`/v1/translate` 호출·CORS 불필요). 인증·Swagger 미노출 |
| `GET /health` | 언어별 모델 로드 상태 + thresholds 존재 여부(인증 없음) |

응답 span 은 canonical `{label, start_char, end_char, text}`
(`.jsonl` 데이터 관례와 일치). 단일 → `{lang, entities}`, 배치 →
`{results:[{lang, entities}, ...]}`(입력 순서 1:1). `lang` 생략 시 자동감지가
배타적 스크립트 신호도 라틴 글자도 못 찾으면 **`200 + {lang:"unsupported",
entities:[]}`**(에러 아님, 모델 미호출) — 배치는 항목별 부분성공. 명시 `lang` 이 미지원이면 400
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

# 단일(자동감지) — en (라틴 폴백)
curl -s -X POST localhost:8008/v1/ner \
  -H 'Content-Type: application/json' \
  -d '{"text":"Barack Obama was born in Hawaii in 1961."}'
# → {"lang":"en","entities":[{"label":"PER",...},{"label":"LOC",...}]}

# 미지원 입력(한자만 등) → 200 + 빈 결과(에러 아님, 모델 미호출)
curl -s -X POST localhost:8008/v1/ner \
  -H 'Content-Type: application/json' -d '{"text":"東京都千代田区"}'
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
| `detect.py` | `detect_lang` — **두 단**이다. ① 배타적 스크립트 양성 감지: 가나→ja, 한글(음절·자모)→ko, vi-변별 코드포인트(horn·hook·dot 결합부호+đ)→vi. ② 셋이 모두 실패하면 라틴 글자가 있는지 보고 → en, 라틴도 없으면 → `unsupported`. 둘째 단은 감지가 아니라 폴백이라 대가를 진다 — 인도네시아어·로마자 일본어·무부호 베트남어가 en 으로 간다. 한자만 있는 텍스트는 라틴 글자도 없어 `unsupported` — ja·ko 가 한자를 공유해 어느 쪽도 가리키지 않으므로 *배타적* 스크립트만 신호로 쓴다. `DETECTORS` (신호,언어) 레지스트리로 확장 — 스크립트 언어 추가는 여전히 한 줄이다(폴백이 리스트 밖 루프 뒤에 있어 새 감지기가 언제나 앞에서 돈다). **언어를 추가할 때는 2 원소 열거 자리를 손으로 훑는다** — 열거 정합 검사의 하한이 3 원소라 그 자리들은 조용히 낡는다(`tests/server/test_docs_language_lists.py`). 근거: `docs/reports/language-detection-benchmark.md` |
| `chunking.py` | `split_for_length` — max_length 초과 입력을 문장 단위로 쪼개 `(substring, base_offset)` 반환(원문 char offset 보존) |
| `inference.py` | `LangModel`(모델·토크나이저·임계값 1회 로드·재사용; 단건 `predict`·cross-text 배치 `predict_many`)·`ModelRegistry`(언어별 보관·`predict_batch` 언어별 묶음, 미로드→`ModelUnavailable`→503). 추론은 fp32 전용(단건·배치 결정적), 입력은 NFC 정규화. 토크나이저 접근(`_tokenize`)은 잠금으로 직렬화 — 공유 인스턴스라 동시 요청이 `Already borrowed` 로 터진다(§토크나이저 접근의 직렬화). 임계값은 `confidence_threshold` — graceful(파일 없으면 raw), canonical 변환 전 내부 span 에 적용 |
| `concurrency.py` | `ConcurrencyGuard`(async) — 세마포어로 동시 in-flight ≤ `MAX_CONCURRENCY`, 대기 큐 `MAX_QUEUE`·타임아웃 `ACQUIRE_TIMEOUT_S` 로 bound, 초과 시 `Overloaded`→429. `app.py` 가 **둘을 만든다** — NER 용과 번역 전용(`TRANSLATE_MAX_CONCURRENCY`, 큐 없음) |
| `limits.py` | `BodySizeLimitMiddleware`(순수 ASGI) — 라우팅·인증 이전에 요청 바디를 `MAX_BODY_BYTES` 로 bound. Content-Length 조기 거부 + chunked 스트리밍 누적 거부(우회 차단), 초과 시 413 봉투. 전송 계층 메모리 고갈 가드 |
| `request_log.py` | `RequestLogMiddleware`(순수 ASGI, 최외곽) — 요청별 request-id 생성·`X-Request-ID` 에코, 지연·결과를 한 줄로. 성공 2xx→DEBUG(기본 침묵), 거절 4xx·503→WARNING(사유 태그). 핸들러가 `request.state.ner_meta`(lang·batch·entities)를 채워 성공 로그에 실린다 |
| `app.py` | `create_app(registry, config, translator)` — FastAPI 라우트(async)·Pydantic·인증·에러. 추론은 guard 안 `run_in_threadpool` 로 실행. registry 는 `predict`/`predict_batch`/`health` 를 가진 객체면 됨(실모델 또는 stub). `translator`(옵션)는 `/v1/translate` 용, None 이면 503. `GET /` 은 임포트 시 1회 읽은 `static/index.html` 을 그대로 반환 |
| `translate.py` | 마스킹-복원 + `LLMTranslator`(원격 OpenAI 호환) + `build_translator`(설정 검증). 번역 대상 언어 목록 `TRANSLATABLE_LANGS`(=`LANG_NAME` 의 키)도 여기 있다. PII span 을 sentinel `【PII{nonce}_{i}】` 로 가려 번역기에 미노출·복원 시 원문 그대로 보존(소실 시 부재, 훼손 없음), 고유명사 음차. `/v1/ner`·`inference.py` 무의존 additive |
| `static/index.html` | 내부 개발·데모용 웹 UI(자족적 HTML+vanilla JS, 빌드·신규 의존성 없음). 텍스트 입력 + 언어 셀렉터(auto/ja/ko/vi/en) → 동일 출처 `/v1/ner` 호출 → 개체를 원문 위 라벨별 색상 하이라이트. 입력을 NFC 정규화해 offset 정합, code-point 슬라이스로 astral 문자 대응. **한국어 번역 보기**(온디맨드 버튼)도 여기 있다 — 페이지 로드 시 `/v1/translate/status` 를 1회 조회해 버튼을 켜거나 끄고(폴링 없음), 누르면 `/v1/translate` 를 호출한다. 결과가 ko 면 버튼을 **감춘다** — 눌러도 400 이 될 버튼을 회색으로 남기면 "백엔드가 죽었나"로 읽힌다 |
| `scripts/run_local.sh` | 호스트 로컬 기동 래퍼(GPU 0 고정, `--port` 전달) |
| `scripts/example_client.py` | 내부 소비자용 최소 레퍼런스 `NERClient` + 자기검증 (`python -m server.scripts.example_client`) |
| `scripts/throughput/bench.py` | 처리량·지연 측정 하네스. `--concurrency N` 은 같은 작업량을 N 스레드로 나눠 서버의 실제 경로를 재현한다 — 잠금 경합처럼 동시 실행에서만 드러나는 비용은 N=1 에서 측정되지 않는다. **반복 수를 넉넉히 준다(reps 80)** — 단건 순차는 forward 가 7ms 안팎으로 짧고 간헐적이라 GPU clock 이 idle 에 머물고, 짧게 재면(reps 5~20) 같은 조건에서 ±15% 가 출렁인다. 80 이면 정상상태에 수렴해 ±1% 다 |
| `__main__.py` | uvicorn 기동 진입점 + 로깅 구성(`_configure_logging` — stderr + 주간 회전 파일) |

## 추론 경로

`encode_row`(data_utils, 토크나이저 capability 분기) → model logits →
softmax → argmax+conf → `decode_bio_to_spans`(score=conf_mean) → canonical
변환 → 임계값 로드 시 자동 적용(graceful — 파일 없으면 raw). 긴 입력은 chunk 분할, 배치 요청(`texts`)은
언어별로 묶어 전 chunk 를 `[N, max_length]` 한 forward 로 추론하고(B=1 이면
단건과 동일) 글로벌 offset 으로 병합 — GPU 병렬로 장문/배치에서 가속된다.
추론은 fp32 전용이라 단건·배치가 같은 커널을 타 배치화가 결과를 바꾸지 않고,
입력은 NFC 로 정규화한다(NFD span 깨짐 방지).
추론은 전역 `ConcurrencyGuard` 안에서 실행돼 동시 부하를 bound 한다. 처리량
측정은 `scripts/throughput/bench.py`.

### 토크나이저 접근의 직렬화

**언어당 토크나이저는 하나뿐인데 여러 스레드가 그것을 만진다.** `app.py` 가 추론을
`run_in_threadpool` 로 넘기므로 최대 `MAX_CONCURRENCY` 개 스레드가 같은 `LangModel`
에 동시에 들어온다. HF fast 토크나이저는 Rust 객체를 `RefCell` 로 감싸고 있고 인코딩이
그 상태를 바꾸는 호출(`no_truncation()`)을 지나기 때문에, 두 번째로 들어온 스레드가
`RuntimeError: Already borrowed` 로 터진다 — 잠그기 전 실측으로 **동시 2 요청만으로
ko·en 요청의 40~50% 가 500** 이었다.

그래서 `LangModel._tokenize` 가 잠금 하나로 분할과 인코딩을 함께 감싼다. 분할
(`chunking.split_for_length`)도 토큰 수를 세느라 같은 객체를 만지므로, 둘을 따로
잠그면 그 사이로 다른 스레드가 들어와 창이 다시 열린다.

| 구간 | 잠금 | 이유 |
|---|---|---|
| 분할 + 인코딩 (`_tokenize`) | ○ | 공유 토크나이저의 상태를 바꾼다 |
| forward (`_forward_feats`) | ✕ | GPU 구간이라 여기까지 잠그면 동시성 상한이 무의미해진다 |
| BIO 디코드 (`_decode_chunk`) | ✕ | 토크나이저를 안 만지고 스레드마다 자기 배열만 읽는다 |

**ja 의 slow 토크나이저도 함께 잠근다.** 이 경로는 `RefCell` 을 안 쓰지만 MeCab 의
스레드 안전이 검증된 바 없고, `is_fast` 로 갈라 두면 잠기는 언어와 아닌 언어가 조용히
어긋난다. 잠금은 `LangModel` 인스턴스마다 따로라 ja 를 잠가도 다른 언어의 처리량에는
닿지 않는다.

불변식은 `tests/server/test_tokenizer_lock.py`(stub, 모델 없이 상시)와
`test_inference_integration.py` 의 동시 추론 검사(실모델, `/data` 있을 때)가 고정한다.

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
  모델 없이 CI 가능, 모델 통합(`test_inference_integration.py`)은 `/data` 있을
  때만 (`pytest.skip` 가드). 통합은 **ja·ko·vi·en 을 같은 축으로** 태운다 —
  offset 정합·thresholds 부재 시 raw 동작·장문 청크 offset 과 후반 recall·
  청크 배치 parity·gold 엔티티 청크 경계 불가침·`predict_many` 등가·NFD 입력
  동일, 그리고 네 언어 인터리브 배치가 단건과 1:1. 계약 테스트는 stub 이라
  모델을 안 부르므로 토크나이저 분기(fast 는 ko 와 en 뿐이고 ja·vi 는 slow)·청킹
  경로(마침표로 끝나는 산문은 문장이 아니라 단어 경계로 쪼개진다)가 갈리는
  자리는 여기서만 잡힌다.
- **토크나이저 잠금**: `test_tokenizer_lock.py` 는 재진입하면 `Already
  borrowed` 를 내는 stub 으로 직렬화 불변식을 **모델 없이 상시** 고정한다
  (§토크나이저 접근의 직렬화). 겹침을 확률에 맡기지 않고 첫 진입 스레드가
  창을 열어 기다리므로, 잠금을 지우면 반드시 터진다 — 실모델 겹은 `/data`
  없으면 skip 이라 그것만으로는 CI 가 조용하다.
- **배포 parity**: 서버 predict 로 잰 F1 을 배포 `metrics.json` 과 대조해
  패키지가 자기가 나온 run 과 어긋나지 않는지 본다. `eval_*` 스크립트를
  import 하지 않고 기록된 값(데이터)과 맞춘다. ja 는 thresholds 가 있어
  abstention on/off 둘을, thresholds 없이 출하된 나머지는 raw 하나를
  본다(`overall_strict`).
- **번역**: 마스킹-복원 계약·설정 검증·동시성 모두 stub 으로 돈다 — 번역기가
  원격 호출 하나라 실모델 없이 전 경로가 상시 검사된다.
- **소비자 예제·자기검증(python)**: `python -m server.scripts.example_client
  --base-url http://localhost:8008` — 내부 소비자가 서버를 호출하는 최소
  레퍼런스(`NERClient`). 단일·배치·미지원·계약 에러(400·413·429)를 실서버
  대상으로 호출·검증하고 PASS/FAIL 종료코드를 낸다.
- **실서버 pytest**: `tests/server/test_live_server.py`(`live` 마커) — 서버를
  서브프로세스로 띄워 httpx 로 검증. 모델 로드에 의존하므로 `/data` 없으면
  skip. `uv run pytest -m live` 로 따로 돌릴 수 있다.

## 번역 대상 언어

번역이 받는 `lang` 은 **NER 이 받는 `lang` 보다 좁다.** `/v1/ner` 은
`SUPPORTED_LANGS`(ja·ko·vi·en)를 보지만 `/v1/translate` 는
`TRANSLATABLE_LANGS`(ja·vi·en)를 본다 — 이 엔드포인트는 "한국어로 번역"이라
한국어 원문은 옮길 곳이 없기 때문이다. `lang:"ko"` 는 400 이다.

목록을 따로 두지 않고 `LANG_NAME`(프롬프트에 박히는 언어 이름)의 키에서
유도한다. 이름이 곧 자격이라서다 — 프롬프트에 넣을 이름이 없는 언어는 번역할
수단이 없다. 목록을 별도로 두면 한쪽만 늘어나 조용히 어긋나고, 그때 나타나는
증상은 에러가 아니라 **원문이 그대로 '번역'으로 돌아오는 것**이다.

웹 UI 도 같은 목록을 두고 ko 결과에서는 번역 버튼을 감춘다.

## 번역 설정 표면

여기는 설정 표면과 그렇게 가른 이유가 정본이다. 함수·알고리즘 수준의 구현
레퍼런스(마스킹-복원 계약·검증 분기·테스트 맵)는
`docs/manual/web-demo-translation.md` 에 있다.

번역기는 원격 온프렘 LLM(vLLM 등 OpenAI 호환) 하나이고, `MODEL`·`BASE_URL` 을
**기본값 없이 필수로 둔 것이 이 설정의 핵심이다.** 예전에는
`TRANSLATE_BASE_URL` 이 `http://localhost:8081/v1` 을 기본으로 가져, 주소를 안
주고 번역을 켜도 서버가 뜨고 그 포트에 떠 있는 아무 모델이나 번역기로 썼다 —
설정만 보고 무엇을 부르는지 알 수 없는 상태였다. 지금은 둘 중 하나만 빠져도
기동이 실패한다.

| 키(`NER_SERVER_` 생략) | 무엇 |
|---|---|
| `TRANSLATE_ENABLED` | 번역 토글(기본 false). 끄면 `/v1/translate` 는 503 |
| `TRANSLATE_MODEL` | 활성 시 필수 — 그 엔드포인트가 서빙 중인 모델 이름 |
| `TRANSLATE_BASE_URL` | 활성 시 필수 — OpenAI 호환 `/v1` 주소 |
| `TRANSLATE_API_KEY` | 선택 — vLLM 은 불필요 |
| `TRANSLATE_TIMEOUT_S` | 선택(실효 기본 30) — 원격 호출을 끊는 상한 |
| `TRANSLATE_MAX_CONCURRENCY` | 번역 동시 in-flight 상한(기본 2, 초과 429) |

선택 키는 `config.py` 에서 기본값 없이 `None` 으로 두고 실효 기본값은
`translate.py` 가 넣는다 — 미설정과 설정을 값으로 구별해야 "안 준 것"이 보인다.

### 번역과 NER 의 예산 분리

`app.py` 가 번역 전용 `ConcurrencyGuard`(상한 `TRANSLATE_MAX_CONCURRENCY`, 큐
없음)를 하나 더 만든다. 번역 한 건은 원격 LLM 응답을 초 단위로 기다리므로, NER
과 예산을 함께 쓰면 그동안 추론 슬롯이 막힌다. 초과는 대기 없이 429 — 번역은
온디맨드 버튼이라 기다리게 하는 것보다 돌려보내는 편이 낫다.

### sentinel 표기의 고정

sentinel 은 `【PII{nonce}_{i}】` 하나로 고정이다. lenticular bracket 은 LLM
경로에서 생존율 100% 이고, 프로세스별 nonce 가 붙어 원문의 `【重要】` 같은 자연
표기와 섞이지 않는다. 프롬프트가 "이 표기를 그대로 두라"고 지시하므로 표기와
프롬프트는 함께 움직인다 — 한쪽만 바꾸면 복원이 조용히 실패한다(소실은 에러가
아니라 '부재'로 흡수돼 화면에 신호가 없다).

과거 엔진 비교 하네스(`scripts/translate_bench/`)는 제거됐다 — 번역은 NER 에
딸린 부가 기능이라 품질이 엔진 선정 기준이 아니고, 상시 돌 하네스를 유지할 값이
없었다. 그 벤치 리포트도 함께 폐기했다 — 하네스와 인프로세스 백엔드가 둘 다
사라져 재현할 코드가 없고, 남은 결론이 지금 고를 것을 바꾸지 않는다. 당시 경위는
`docs/issues/issue-197-translate-backend-option.md` 와
`issue-254-drop-inprocess-translate-backend.md` 에 남아 있다.
