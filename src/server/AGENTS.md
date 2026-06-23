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
(256)·`MAX_CHARS`(20000)·`MAX_BATCH`(64)·`API_KEY`(미설정 시 인증 off)·
`HOST`·`PORT`.

## API

| 엔드포인트 | 설명 |
|---|---|
| `POST /v1/ner` | 단일 `{text, lang?}` 또는 배치 `{texts:[...], lang?}`. `?abstain=false` 로 임계값 무시. `lang` 생략 시 텍스트별 자동감지 |
| `GET /health` | 언어별 모델 로드 상태 + thresholds 존재 여부(인증 없음) |

응답 span 은 canonical `{label, start_char, end_char, text, score}`
(`.jsonl` 데이터 관례와 일치). 단일 → `{lang, entities}`, 배치 →
`{results:[{lang, entities}, ...]}`(입력 순서 1:1). 에러는 구조화 `{error:
{status, message}}` — 잘못된 lang→400, 모델 미로드→503, API-key 불일치→401.

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
softmax → argmax+conf → `decode_bio_to_spans`(score=conf_mean) → 긴 입력은
chunk 별 추론 후 글로벌 offset 병합 → canonical 변환 → graceful abstention.

## 테스트

`tests/server/` — 계약·전송은 stub registry 로 모델 없이 CI 가능,
모델 통합(offset 정합성·ja parity)은 `/data` 있을 때만(`pytest.skip` 가드).
ja parity 는 `eval_*` 스크립트를 import 하지 않고 `/data/ner/ja/metrics.json`
의 운영점과 대조한다.
