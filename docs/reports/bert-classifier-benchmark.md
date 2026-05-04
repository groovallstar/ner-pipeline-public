# BERT NER 분류기 벤치마크 (JA·VI canonical 10종 평면)

- 측정일: 2026-05-04
- 대상: JA `tohoku-nlp/bert-base-japanese-v3` / VI `xlm-roberta-base` 파인튜닝
- 데이터: JA `data/stockmark/pii_all.jsonl`, VI `data/wikiann_vi/pii_all.jsonl`
- 평가: char-offset span F1 (`src/ner/metrics/span_metrics.compute_offset_span_f1`), test 셋 단독

## 조건

| 항목 | JA | VI |
|---|---|---|
| 모델 | `tohoku-nlp/bert-base-japanese-v3` (110M) | `xlm-roberta-base` (278M) |
| 토크나이저 | `BertJapaneseTokenizer` (slow, MeCab) | `XLMRobertaTokenizer` (fast) |
| 데이터 | Stockmark NER + PII 주입 (5,307) | WikiANN-vi silver + PII 주입 (40,000) |
| 분할 | train 4,247 / valid 530 / test 530 | train 32,000 / valid 4,000 / test 4,000 |
| 분할 비율 | 80/10/10 (`seed=42`) | 80/10/10 (`seed=42`) |
| Hyperparameter | epochs=5, batch_size=16, lr=5e-5, max_length=256, fp16 | 동일 |
| 모델 선택 | `metric_for_best_model='eval_loss'` (valid 셋) | 동일 |
| 학습 시간 | 98.5 초 | 807.5 초 |
| 하드웨어 | NVIDIA RTX A6000 49GB | 동일 |

## 결과

### Overall (test 셋)

| 지표 | JA | VI |
|---|---|---|
| **F1** | **0.9058** | **0.8985** |
| Precision | 0.8653 | 0.8760 |
| Recall | 0.9501 | 0.9222 |
| Test Support | 1,745 spans | 8,778 spans |

### Per-entity

| Entity | JA F1 | JA Support | VI F1 | VI Support |
|---|---|---|---|---|
| PER | 0.9615 | 359 | 0.9272 | 2,072 |
| LOC | 0.8513 | 259 | 0.9182 | 2,004 |
| ORG | 0.8863 | 486 | 0.8593 | 748 |
| PROD | 0.8167 | 108 | 0.6434 | 208 |
| EVT | 0.8750 | 94 | 0.6457 | 61 |
| DAT | 0.9831 | 89 | 0.8868 | 730 |
| EMAIL | 0.9944 | 88 | 0.9960 | 739 |
| PHONE | 1.0000 | 101 | 0.8089 | 694 |
| ID_NUM | 0.8427 | 83 | 0.9045 | 758 |
| CREDIT_CARD | 0.9136 | 78 | 0.9050 | 764 |

### NER 5종 vs PII 5종 가중 평균

| 그룹 | JA | VI |
|---|---|---|
| NER 5종 (PER/LOC/ORG/PROD/EVT) | 0.886 | 0.890 |
| PII 5종 (DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD) | 0.948 | 0.901 |

## 재현

```bash
# 데이터 (생성은 augmenters 측, gitignore 됨)
ls data/stockmark/pii_all.jsonl
ls data/wikiann_vi/pii_all.jsonl

# 의존성
uv add fugashi unidic-lite   # JA tokenizer 의존성
uv sync

# JA 학습 + 평가 (산출물: results/classifier/ja/)
python -m ner.classifier --lang ja --epochs 5 --batch-size 16 --max-length 256

# VI 학습 + 평가 (산출물: results/classifier/vi/)
python -m ner.classifier --lang vi --epochs 5 --batch-size 16 --max-length 256

# 단위 + 토크나이저 round-trip 테스트
python -m pytest tests/ner/classifier/ -q
```

원시 메트릭: `results/classifier/{ja,vi}/metrics.json` (gitignore 대상).
