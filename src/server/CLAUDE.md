# src/server/ — ja·vi NER REST API 서비스

학습된 BERT 분류기(`/data/ner/{ja,vi}/model/`)를 감싸 HTTP 로 NER 추론을
제공한다. `ner.classifier` 의 data_utils(인코딩·디코드)와
confidence_threshold(임계값 fit·apply)에 의존하고, 학습·평가 모듈은
import 하지 않는다.

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
초)·`API_KEY`(미설정 시 인증 off)·`HOST`·`PORT`.

컨테이너 배포(내부망 별도 프로세스 소비자용)는 `docker/server/`(compose +
라이프사이클 + `.env.example`; 상세: `docker/server/CLAUDE.md`).

## API

| 엔드포인트 | 설명 |
|---|---|
| `POST /v1/ner` | 단일 `{text, lang?}` 또는 배치 `{texts:[...], lang?}`. `lang` 생략 시 텍스트별 자동감지. 신뢰도 임계값은 모델 로드 시 자동 적용 |
| `GET /` | 내부 개발·데모용 웹 UI(자족적 HTML, 동일 출처로 `/v1/ner` 호출·CORS 불필요). 인증·Swagger 미노출 |
| `GET /health` | 언어별 모델 로드 상태 + thresholds 존재 여부(인증 없음) |

응답 span 은 canonical `{label, start_char, end_char, text}`
(`.jsonl` 데이터 관례와 일치). 단일 → `{lang, entities}`, 배치 →
`{results:[{lang, entities}, ...]}`(입력 순서 1:1). `lang` 생략 시 자동감지가
ja·vi 신호를 못 찾으면 **`200 + {lang:"unsupported", entities:[]}`**(에러
아님, 모델 미호출) — 배치는 항목별 부분성공. 명시 `lang` 이 미지원이면 400
(클라이언트 계약). 에러는 구조화 `{error: {status, message}}` — 잘못된 요청
(lang·text/texts 택일)→400, 크기 한도(max_chars·max_batch·max_total_chars)
초과→413, 모델 미로드→503, API-key 불일치→401. 요청 바디가 `max_body_bytes`
초과면 파싱 전에 413(전송 계층 가드 — chunked 우회 포함). Pydantic 검증
실패→422, 미처리 예외→500 도 모두 동일 봉투로 감싸고 500 은 내부 메시지를
노출하지 않는다.

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
| `app.py` | `create_app(registry, config)` — FastAPI 라우트(async)·Pydantic·인증·에러. 추론은 guard 안 `run_in_threadpool` 로 실행. registry 는 `predict`/`predict_batch`/`health` 를 가진 객체면 됨(실모델 또는 stub). `GET /` 은 임포트 시 1회 읽은 `static/index.html` 을 그대로 반환 |
| `static/index.html` | 내부 개발·데모용 웹 UI(자족적 HTML+vanilla JS, 빌드·신규 의존성 없음). 텍스트 입력 + 언어 셀렉터(auto/ja/vi) → 동일 출처 `/v1/ner` 호출 → 개체를 원문 위 라벨별 색상 하이라이트. 입력을 NFC 정규화해 offset 정합, code-point 슬라이스로 astral 문자 대응 |
| `__main__.py` | uvicorn 기동 진입점 |

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
