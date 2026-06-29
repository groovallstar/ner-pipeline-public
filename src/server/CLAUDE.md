# src/server/ — ja·vi NER REST API 서비스

학습된 BERT 분류기(`/data/ner/{ja,vi}/model/`)를 감싸 HTTP 로 NER 추론을
제공한다. `ner.classifier` 의 data_utils(인코딩·디코드)와
confidence_threshold(임계값 fit·apply)에 의존하고, 학습·평가 모듈은
import 하지 않는다.

## 기동

```bash
python -m server                 # 0.0.0.0:8000, /data/ner 로드
python -m server --host 127.0.0.1 --port 9000 --model-root /abs/root
```

설정은 `NER_SERVER_*` 환경변수: `MODEL_ROOT`(기본 `/data/ner`)·`MAX_LENGTH`
(256)·`MAX_CHARS`(20000, 텍스트 1건)·`MAX_BATCH`(64, 배치 개수)·
`MAX_TOTAL_CHARS`(100000, 배치 char 합산 — 요청당 작업량 가드)·`API_KEY`
(미설정 시 인증 off)·`HOST`·`PORT`.

## API

| 엔드포인트 | 설명 |
|---|---|
| `POST /v1/ner` | 단일 `{text, lang?}` 또는 배치 `{texts:[...], lang?}`. `?abstain=false` 로 임계값 무시. `lang` 생략 시 텍스트별 자동감지 |
| `GET /health` | 언어별 모델 로드 상태 + thresholds 존재 여부(인증 없음) |

응답 span 은 canonical `{label, start_char, end_char, text, score}`
(`.jsonl` 데이터 관례와 일치). 단일 → `{lang, entities}`, 배치 →
`{results:[{lang, entities}, ...]}`(입력 순서 1:1). 에러는 구조화 `{error:
{status, message}}` — 잘못된 요청(lang·text/texts 택일)→400, 크기 한도
(max_chars·max_batch·max_total_chars) 초과→413, 모델 미로드→503, API-key
불일치→401.

## 사용 예시 (curl)

```bash
# 단일(자동감지) — ja
curl -s -X POST localhost:8000/v1/ner \
  -H 'Content-Type: application/json' \
  -d '{"text":"織田信長は東京都千代田区に住んでいた。"}'
# → {"lang":"ja","entities":[
#      {"label":"PER","start_char":0,"end_char":4,"text":"織田信長","score":0.99},
#      {"label":"LOC","start_char":5,"end_char":12,"text":"東京都千代田区","score":0.99}]}

# 배치(혼합 언어, 텍스트별 감지)
curl -s -X POST localhost:8000/v1/ner \
  -H 'Content-Type: application/json' \
  -d '{"texts":["トヨタは日本の会社です。","Hà Nội là thủ đô."]}'
# → {"results":[{"lang":"ja","entities":[...]},{"lang":"vi","entities":[...]}]}

# 임계값 무시(raw)
curl -s -X POST 'localhost:8000/v1/ner?abstain=false' \
  -H 'Content-Type: application/json' -d '{"text":"...","lang":"ja"}'

# 헬스 / OpenAPI UI
curl -s localhost:8000/health   # {"status":"ok","langs":{...}}
# 브라우저: GET /docs
```

## 파일

| 파일 | 역할 |
|------|------|
| `config.py` | `ServerConfig` — env 설정·언어별 경로(`model_dir`/`thresholds_path`) |
| `detect.py` | `detect_lang` — 가나(히라가나·가타카나)→ja, 그 외→vi. vi 코퍼스의 한자 혼입은 가나가 없어 vi 로 분류 |
| `chunking.py` | `split_for_length` — max_length 초과 입력을 문장 단위로 쪼개 `(substring, base_offset)` 반환(원문 char offset 보존) |
| `inference.py` | `LangModel`(모델·토크나이저·임계값 1회 로드·재사용, char-offset span 추론)·`ModelRegistry`(언어별 보관, 미로드 언어→`ModelUnavailable`→503). 임계값은 `confidence_threshold.{load,apply}_thresholds` 사용 — graceful 로딩(파일 없으면 raw)은 서버에서 존재 확인, 적용은 canonical 변환 전 내부 span 에 |
| `app.py` | `create_app(registry, config)` — FastAPI 라우트·Pydantic 모델·인증·에러 핸들러. registry 는 `predict`/`health` 를 가진 객체면 됨(실모델 또는 테스트 stub) |
| `__main__.py` | uvicorn 기동 진입점 |

## 추론 경로

`encode_row`(data_utils, 토크나이저 capability 분기) → model logits →
softmax → argmax+conf → `decode_bio_to_spans`(score=conf_mean) → canonical
변환 → graceful abstention. 긴 입력은 chunk 분할 후 모든 chunk 를 `[K,
max_length]` 한 배치 forward 로 추론하고(chunk 1개면 배치 차원 1 = 단건과
동일) 글로벌 offset 으로 병합 — GPU 가 chunk 들을 병렬 처리해 장문/배치
요청에서 순차 대비 가속된다.

## 테스트·검증

- **pytest**: `uv run pytest tests/server/` — 계약·전송은 stub registry 로
  모델 없이 CI 가능, 모델 통합(offset 정합성·ja parity)은 `/data` 있을 때만
  (`pytest.skip` 가드). ja parity 는 `eval_*` 스크립트를 import 하지 않고
  `/data/ner/ja/metrics.json` 운영점과 대조한다.
- **실서버 스모크(셸)**: `bash src/server/scripts/smoke_test.sh [PORT]` —
  uvicorn 을 기동한 뒤 curl 로 `/health`·`/v1/ner`·에러 응답을 검증하고
  PASS/FAIL 종료코드를 낸다. in-process TestClient 가 못 보는 실제 포트
  바인딩·네트워크 경로를 확인하는 용도.
- **실서버 pytest**: `tests/server/test_live_server.py`(`live` 마커) — 서버를
  서브프로세스로 띄워 httpx 로 검증. 모델 로드에 의존하므로 `/data` 없으면
  skip. `uv run pytest -m live` 로 따로 돌릴 수 있다.
