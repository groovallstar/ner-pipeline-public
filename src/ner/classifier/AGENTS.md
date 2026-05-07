# src/ner/classifier/ — BERT 토큰 분류 파인튜닝

JA·VI canonical 10종 평면 (NER 5 + PII 5) 학습용 BERT NER 분류기. augmenters 가
produce 한 PII 주입 JSONL 을 입력으로 받아 BIO 21-class 모델을 학습하고 char-offset
span F1 을 측정한다.

## 책임 경계

- **포함**: 데이터 로딩(JSONL contract 소비) / wordpiece 정렬·BIO 변환 / HF Trainer 래퍼 / span 디코드 / 메트릭 산출
- **제외**: 데이터 증강, 라벨러, 평가용 LLM 백엔드, HF Hub 업로드, 추론 서빙
- **결합도**: 얕음 — augmenters 의 import 결합 없음. 파일(JSONL)만 소비한다. metrics 만 `src/ner/metrics` 에서 공용 import.

## 모듈

| 파일 | 역할 |
|---|---|
| `data_utils.py` | canonical 10종 라벨 맵 / JSONL 로더 / JA·VI tokenizer 분기 정렬 / BIO ↔ char-span 변환 |
| `train_eval.py` | HF Trainer 래퍼 (`fine_tune`) + best 모델 로드 후 평가 (`evaluate_model`) |
| `error_analysis.py` | baseline 모델의 test-set 오답 추출 + 카테고리 분류 (BOUNDARY / TYPE_MISMATCH / MISS / HALLUCINATION) + 사람 검수용 stratified 샘플. CLI: `python -m ner.classifier.error_analysis` |
| `__main__.py` | CLI: `python -m ner.classifier --lang {ja,vi}` |

## CLI

```bash
# JA 본 학습 (기본 hyperparameter)
python -m ner.classifier --lang ja

# VI 본 학습 (xlm-roberta-base 기본 모델)
python -m ner.classifier --lang vi --epochs 5 --batch-size 16

# 스모크 (100 train / 50 test / 1 epoch — CI·dev 검증용)
python -m ner.classifier --lang ja --smoke

# 모델·데이터 override
python -m ner.classifier --lang vi \
    --model-name vinai/phobert-base \
    --data data/wikiann_vi/pii_all.jsonl \
    --max-length 256 --lr 5e-5
```

기본값:
- `--lang ja`: 모델 `tohoku-nlp/bert-base-japanese-v3`, 데이터 `data/stockmark/pii_all.jsonl`
- `--lang vi`: 모델 `xlm-roberta-base`, 데이터 `data/wikiann_vi/pii_all.jsonl`
- `--valid-ratio 0.1`, `--test-ratio 0.1` (3-way split), `--seed 42`, `--max-length 256`, `--epochs 5`, `--batch-size 16`, `--lr 5e-5`
- 3-way 분할: train/valid/test = 80/10/10. valid 셋은 epoch best 모델 선택용 (`metric_for_best_model='eval_loss'`), test 셋은 최종 char-offset span F1 측정 단독. test 셋은 학습/모델 선택 어디에도 노출되지 않음.

## JSONL contract (입력 계약)

augmenters/pii, augmenters/wikiann_vi 가 출력하는 형식 그대로 소비.

```jsonl
{"text": "...",
 "entities": [{"label": "PER", "start_char": 0, "end_char": 3, "text": "..."}],
 "id": "..."}
```

- `label` ∈ canonical 10종 = `PER LOC ORG PROD EVT DAT EMAIL PHONE ID_NUM CREDIT_CARD`
- `start_char`, `end_char` — 문자 오프셋 [start, end), 반-개구간
- 검증: `data_utils.load_jsonl` 가 라벨이 canonical 10종 안에 있는지 확인하고, 위반 시 `ValueError`

## 라벨 셋

```
O                              # outside
B-PER, I-PER                   # NER 5종
B-LOC, I-LOC
B-ORG, I-ORG
B-PROD, I-PROD
B-EVT, I-EVT
B-DAT, I-DAT                   # PII 5종
B-EMAIL, I-EMAIL
B-PHONE, I-PHONE
B-ID_NUM, I-ID_NUM
B-CREDIT_CARD, I-CREDIT_CARD
```

총 21 labels (`O` + 10*B + 10*I). 단일 출처: `docs/manual/data/canonical-entity-schema.md`.

## 토크나이저 분기 (slow vs fast)

| 언어 | 토크나이저 | 정렬 방식 |
|---|---|---|
| JA | `BertJapaneseTokenizer` (slow, MeCab 의존) | `tokenize()` 결과를 `text.find(surface, pos)` 로 greedy 매칭. `##` 접두 subword 는 strip 후 매칭. UNK 시 0-length span. 의존성: `fugashi` + `unidic-lite` |
| VI | `XLMRobertaTokenizer` (fast) | `return_offsets_mapping=True` 직접 사용 |

slow tokenizer 는 `offset_mapping` 미지원 — `data_utils._encode_ja` 가 수동 정렬을 수행한다.

## 메트릭

- 주 메트릭: char-offset span F1 (`src/ner/metrics/span_metrics.compute_offset_span_f1`)
- 매칭: `(start, end, type)` 정확 매칭 (llm_eval 과 동일 정의 — 직접 비교 가능)
- per-entity: 10종 각각 micro-F1·precision·recall·support 산출

## 산출물

```
results/classifier/{ja,vi}/
├── best/                       # best 체크포인트 (HF model dir)
├── checkpoint-*/               # 중간 체크포인트 (save_total_limit=1 로 정리)
└── metrics.json                # 학습 설정 + overall + per-entity F1

docs/reports/japanese-bert-classifier-benchmark.md      # JA 리포트
docs/reports/vietnamese-bert-classifier-benchmark.md    # VI 리포트
```

## 합격선

deep-interview 결과: **overall span F1 ≥ 0.95** (시간·모델 크기 미고려).

`__main__.py` 마지막에 `PASS` / `FAIL` 출력.
미달 시 후속 이슈 발행 (사용자 승인 후 PR 진행).

## 테스트

```bash
python -m pytest tests/ner/classifier/ -q
```

- `test_data_utils.py` — 라벨 맵 / BIO 정렬 / span 디코드 / split 결정성 / class weight / curriculum mask
- `test_encode.py` — 실제 토크나이저(JA·VI)로 round-trip 검증
- `test_error_analysis.py` — span 오류 분류·집계·검수 샘플링 (10 테스트)

## 주의

- `PYTHONPATH` **설정·주입 금지** — uv editable install 이 자동 등록
- import: `from ner.classifier.xxx import Yyy` (src 접두어 없이)
- print/log/argparse help: 영문 / docstring·주석: 한국어 (코딩 컨벤션 `docs/specs/coding-conventions.md`)
