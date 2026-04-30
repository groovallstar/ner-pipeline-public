# WikiANN-vi + 합성 PII 벤치마크 리포트 (10종 canonical)

> 본 리포트는 이슈 #8 의 silver PII 데이터 산출물 품질 측정.
> WikiANN-vi 5종 NER (PER/LOC/ORG/PROD/EVT, #30 silver) + 합성 PII 5종
> (DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD) = **10종 평면 canonical** 위에서
> silver 라벨러 정당성을 측정한다.

**측정일**: 2026-04-30
**입력**: `data/wikiann_vi/{train,valid,test}.jsonl` (5종 canonical, #30 recall_strict silver)
**출력**: `data/wikiann_vi/test_pii_5gold_1000.jsonl` (test 1000 샘플 silver)
**Inject 모드**: `suffix` (규칙 기반 — 문장 끝에 PII 단서어 + 합성 값 부착, LLM 호출 없음)
**Verify 모델**: `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` (vLLM TP=1, GPU 2, 포트 8082)
**Verify policy**: `drop_span` — 미확인 span 만 제거, 레코드 보존

---

## 1. 산출 통계

### 1.1 전체 데이터셋 (40K records)

| Split | Records (kept) | Confirmed | Missed | Conflict | 신뢰율 |
|---|---:|---:|---:|---:|---:|
| train | 20,000 | 44,122 | 2,659 | 459 | 93.4% |
| valid | 10,000 | 21,826 | 1,452 | 200 | 93.0% |
| test | 10,000 | 21,976 | 1,336 | 255 | 93.3% |
| **계** | **40,000** | **87,924** | **5,447** | **914** | **93.3%** |

> kept ratio 100%: drop_span policy 라 모든 record 가 보존됨 (일부 span
> 손실 가능). 신뢰율 = confirmed / (confirmed + missed + conflict).
> 산출 시간: train 80 분 + (valid + test 병렬) 80 분 = 약 2 시간 40 분.

### 1.2 test 1000 샘플 (모델 벤치 대상)

| 지표 | 값 |
|---|---:|
| 입력 records | 1,000 |
| 산출 records (kept) | **1,000** (100.0%) |
| 총 entities | 2,300 (PII inject 후) |
| confirmed | **2,156** (93.7%) |
| missed | 132 (5.7%) |
| conflict | 12 (0.5%) |
| dropped | 0 |

> 본 1000 샘플은 §1.1 의 test 10K 의 첫 1000 부분집합 (id 기준 `0`~`999`
> 정렬, `data/wikiann_vi/test_canonical_5gold_1000.jsonl` 와 동일 id).
> 전체 test 신뢰율 (93.3%) 과 1000 샘플 신뢰율 (93.7%) 가 거의 일치 →
> 모델 F1 일반화 안전.

---

## 2. 라벨 분포 (verify 적용 후, test 1000)

| Label | confirmed | missed | conflict | 신뢰율 |
|---|---:|---:|---:|---:|
| PER | 546 | 45 | 1 | 92.2% |
| LOC | 460 | 46 | 7 | 89.7% |
| ORG | 201 | 23 | 4 | 88.2% |
| PROD | 51 | 3 | 0 | 94.4% |
| EVT | 17 | 2 | 0 | 89.5% |
| EMAIL | 165 | 1 | 0 | 99.4% |
| PHONE | 173 | 2 | 0 | 98.9% |
| DAT | 194 | 2 | 0 | 99.0% |
| ID_NUM | 175 | 2 | 0 | 98.9% |
| CREDIT_CARD | 174 | 6 | 0 | 96.7% |

**관찰**:
- **PII 5종 (98~99%)** > **NER 5종 (88~94%)** — PII 가 형식이 명확해
  verifier 신뢰도 높음.
- ORG 신뢰율 88% 이 5종 NER 중 최저. WikiANN 원본의 ORG 경계 모호함
  (시설=ORG vs 지명=LOC) 이 잔존 — `Lakewood`, `Cagliari`, `Vegas de
  Matute` 등 LOC↔ORG 충돌 12 conflict 모두 NER 5종 사이의 type 충돌.
- EVT support 17 — 절대량이 낮아 통계적 유의성 제한.

---

## 3. 검증 4종 결과 (test 1000)

| 검증 항목 | 결과 |
|---|---|
| 1. 라벨 셋 ⊂ 10종 (canonical) | **PASS** (외부 라벨 0/2300) |
| 2. Offset 정합성 | **PASS** (0 위반) |
| 3. PII 단서어 동반률 | **PASS** (suffix 모드 100%) |
| 4. 영문 라벨 leakage | **PASS** (0 records) |

> JA 측 #23 PR #25 에서 입증된 Rule 4·5 (단서어 부착 · 영문 라벨 누설
> 금지) 효과가 VI silver 에서도 동일하게 발현.
> 단서어 패턴: `Liên hệ`, `SĐT`, `Email`, `CMND`, `Thẻ`, `Địa chỉ` 등.

---

## 4. 모델별 PII 혼합 F1 (test 1000)

| # | 모델 | F1 | Precision | Recall | sec/sample | Total TPS | 총 시간 |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit | **0.9203** | 0.9194 | 0.9212 | 0.438 | 9,378 | 7:18 |
| 2 | cyankiwi/gemma-4-31B-it-AWQ-8bit | 0.9141 | 0.8589 | 0.9768 | 0.270 | 16,060 | 4:30 |
| 3 | openai:gpt-5.4-mini | 0.7818 | 0.6961 | 0.8915 | 0.263 | 2,637 | 4:23 |

**관찰**:
- vLLM 두 모델은 NER-only 벤치 대비 PII 추가로 **F1 ↑** (gemma 0.876→0.914,
  Qwen 0.812→0.920) — PII 형식이 명확해 손쉽게 검출.
- gpt-5.4-mini 는 **소폭 감소** (0.791→0.782) — 특히 LOC F1 0.836→0.488
  로 급락. 본 모델이 PII 단서어 옆 LOC 단어 (`P.14, ... Tan Binh, Da Nang`)
  를 ADDRESS=LOC 로 처리하는 데 어려움.

---

## 5. Per-Entity Breakdown (test 1000)

### 5.1 cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit

| Entity | F1 | Precision | Recall | Support |
|---|---:|---:|---:|---:|
| PER | 0.903 | 0.912 | 0.894 | 546 |
| LOC | 0.902 | 0.941 | 0.865 | 460 |
| ORG | 0.886 | 0.944 | 0.836 | 201 |
| PROD | 0.703 | 0.650 | 0.765 | 51 |
| EVT | 0.778 | 0.737 | 0.824 | 17 |
| EMAIL | 0.994 | 0.994 | 0.994 | 165 |
| PHONE | 0.991 | 0.989 | 0.994 | 173 |
| DAT | 0.886 | 0.795 | 1.000 | 194 |
| ID_NUM | 0.980 | 0.962 | 1.000 | 175 |
| CREDIT_CARD | 0.983 | 0.967 | 1.000 | 174 |

### 5.2 cyankiwi/gemma-4-31B-it-AWQ-8bit

| Entity | F1 | Precision | Recall | Support |
|---|---:|---:|---:|---:|
| PER | 0.932 | 0.885 | 0.984 | 546 |
| LOC | 0.921 | 0.866 | 0.983 | 460 |
| ORG | 0.885 | 0.821 | 0.960 | 201 |
| PROD | 0.577 | 0.412 | 0.961 | 51 |
| EVT | 0.765 | 0.765 | 0.765 | 17 |
| EMAIL | 0.939 | 1.000 | 0.885 | 165 |
| PHONE | 0.994 | 0.989 | 1.000 | 173 |
| DAT | 0.833 | 0.713 | 1.000 | 194 |
| ID_NUM | 0.989 | 0.978 | 1.000 | 175 |
| CREDIT_CARD | 0.983 | 0.967 | 1.000 | 174 |

### 5.3 openai:gpt-5.4-mini

| Entity | F1 | Precision | Recall | Support |
|---|---:|---:|---:|---:|
| PER | 0.914 | 0.863 | 0.971 | 546 |
| LOC | 0.488 | 0.386 | 0.663 | 460 |
| ORG | 0.866 | 0.807 | 0.935 | 201 |
| PROD | 0.388 | 0.249 | 0.882 | 51 |
| EVT | 0.682 | 0.556 | 0.882 | 17 |
| EMAIL | 0.997 | 0.994 | 1.000 | 165 |
| PHONE | 0.971 | 0.971 | 0.971 | 173 |
| DAT | 0.864 | 0.761 | 1.000 | 194 |
| ID_NUM | 0.989 | 0.978 | 1.000 | 175 |
| CREDIT_CARD | 0.864 | 0.958 | 0.787 | 174 |

---

## 6. 데이터 산출 경로

```
data/wikiann_vi/
├── train.jsonl                       # 입력: 20K (canonical 5-gold)
├── valid.jsonl                       # 입력: 10K
├── test.jsonl                        # 입력: 10K
├── test_canonical_5gold_1000.jsonl   # 입력 1000 샘플 부분집합 (벤치용)
├── test_pii_5gold_1000.jsonl         # 산출: 1000 샘플 PII silver
├── test_pii_5gold_1000.verify.json   # verify 리포트
└── test_pii_5gold_1000.stats.json    # inject 통계
```

전체 데이터셋 (train 20K + valid 10K + test 10K = 40K) PII 산출은
별도 백그라운드 작업 진행 중. 본 리포트는 test 1000 샘플로 한정.

---

## 7. 재현 명령

### PII inject + verify (1000 샘플)

```bash
source ./.env
uv run python -m augmenters.pii \
  --source jsonl --input data/wikiann_vi/test_canonical_5gold_1000.jsonl \
  --lang vi --output data/wikiann_vi/test_pii_5gold_1000.jsonl \
  --n-samples 1000 --pii-max 3 --mode suffix \
  --verify vllm \
  --verify-url http://localhost:8082/v1 \
  --verify-model cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --verify-concurrency 16
```

### 모델 벤치

```bash
uv run python -m llm_eval --lang vi \
  --local-file data/wikiann_vi/test_pii_5gold_1000.jsonl \
  --models vllm:cyankiwi/gemma-4-31B-it-AWQ-8bit \
  --vllm-url http://localhost:8081/v1 \
  --max-samples 1000 --no-bertscore \
  --output results/issue-8/pii_5gold/01-gemma-4-31B-AWQ-8bit.json
```

---

## 8. 한계

- **Inject 모드 = suffix**: 규칙 기반 PII 부착 — JA 의 `llm` 모드 대비
  자연도 ↓. 다만 verify 단계의 단서어 동반률 100% 보장은 동일.
- **Verifier 단일 모델**: Qwen3.6-35B 만 verify. 교차 검증을 위해 gemma
  를 verify 로 추가하면 conflict 분포 추가 데이터 확보 가능.
- **모델 라인업 3종**: JA 미러링 6종 중 3종 측정. Phase 2 (BF16/MoE)
  swap 후 라인업 확장 시 본 리포트 §4·§5 표 확장.
- **자체 평가 편향**: gemma·Qwen 은 silver 생성자라 자기일치 효과 일부
  포함. gpt-5.4-mini 만 외부 평가자.
