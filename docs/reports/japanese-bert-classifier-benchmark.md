# 일본어 BERT NER 분류기 벤치마크 (canonical 10종 평면)

- 측정일: 2026-05-04 ~ 2026-05-07 (Tier 1.B 재평가 + Tier 3 gold cleanup 포함)
- 대상: Stockmark NER + 합성 PII 주입 데이터에 BERT family 파인튜닝
- 데이터: `data/stockmark/pii_all.jsonl` (5,270 행, canonical 10종 = NER 5 + PII 5)
  - 옛 5,307 행 → Tier 3 검수 불가 34 sentences 제거 (3 collateral) →
    5,270 행. 본 시점부터 모든 측정값은 5,270-row 기준
- 평가: char-offset span F1 (`src/ner/metrics/span_metrics.compute_offset_span_f1`)
  + 보조: SemEval'13 Partial F1 (`compute_offset_span_f1_relaxed`)
- 출처: `docs/issues/issue-40-classifier-restore.md` §SOTA Sweep,
  `docs/issues/issue-45-ja-classifier-external-corpus.md` §종결,
  `docs/issues/issue-48-ja-classifier-tier2-tier3-followups.md` (진행 중)
- 베트남어 동일 평가 셋업의 분류기 결과는
  `docs/reports/vietnamese-bert-classifier-benchmark.md` 참조

## 요약

- **현 production 모델**: `tohoku-nlp/bert-base-japanese-v3` (110M).
  baseline strict F1 = 0.9058 (SOTA sweep 12 후보 + 변종 4단계 중 최선).
- **게이트 0.95 미달**. 28회 sweep + Tier 1.B 8 variant 재평가로 *데이터·평가
  양 측 천장* 동결.
- **DeBERTa-v3 family 학습 실패**: `ku-nlp/deberta-v3-base-japanese`,
  `microsoft/mdeberta-v3-base` 모두 F1=0 (majority-class collapse). 원인은
  하이퍼파라미터 미튜닝 (warmup·weight_decay·gradient_clip).
- **fast tokenizer fragmentation**: ModernBERT/mmBERT 가 PII 영숫자
  (`@`, 숫자열) 분해 실패로 EMAIL/ID_NUM/CREDIT_CARD F1 < 0.35.
- **Tier 3 완료**: gold cleanup (2,354 errors 전수 검수, 996건 net 보정 +
  18건 audit cleanup) → 보정 baseline strict F1 = **0.9249** /
  relaxed F1 = **0.9415**. 게이트 gap: strict 2.51pp / relaxed 0.85pp.
- **현재 진행**: Tier 2.B self-training 미시도.

## 조건

| 항목 | 값 |
|---|---|
| 데이터 | Stockmark JA Wikipedia NER + PII 주입 (5,307) |
| 분할 | train 4,247 / valid 530 / test 530 (80/10/10, `seed=42`) |
| 라벨 | `PER, LOC, ORG, PROD, EVT, DAT, EMAIL, PHONE, ID_NUM, CREDIT_CARD` |
| 학습 | epochs=5, batch_size=16, lr=5e-5, max_length=256, fp16 |
| 모델 선택 | `metric_for_best_model='eval_loss'` (valid) |
| 하드웨어 | NVIDIA RTX A6000 49GB, 단일 seed=42 |

> 변종 ablation (v2~v4) 은 도입 전 80/20 분할 기준이라 sweep 표와 직접
> 비교하지 않는다. 분할 변경 영향은
> `docs/issues/issue-40-classifier-restore.md` §"80/20 → 80/10/10 분할 변경
> 영향" 참조.

## SOTA 모델 Sweep — 7 후보 (overall)

| 모델 | F1 | Precision | Recall | Train time |
|---|---:|---:|---:|---:|
| **`tohoku-nlp/bert-base-japanese-v3`** (production) | **0.9058** | 0.8653 | 0.9501 | 99s |
| `tohoku-nlp/bert-large-japanese-v2` | 0.8894 | 0.8551 | 0.9266 | 676s |
| `jhu-clsp/mmBERT-base` | 0.7662 | 0.7508 | 0.7822 | 408s |
| `sbintuitions/modernbert-ja-130m` | 0.7649 | 0.7493 | 0.7811 | 259s |
| `llm-jp/llm-jp-modernbert-base` | 0.7220 | 0.6934 | 0.7530 | 357s |
| `ku-nlp/deberta-v3-base-japanese` (bf16, lr=2e-5) | 0.0000 | 0.0000 | 0.0000 | 360s — 학습 실패 |
| `microsoft/mdeberta-v3-base` (bf16, lr=2e-5) | 0.0000 | 0.0000 | 0.0000 | 391s — 학습 실패 |

원시 메트릭: `results/classifier/ja/sweep/<모델>/metrics.json`,
production 베이스라인은 `results/classifier/ja_sweep/baseline/metrics.json`.

### 베이스라인 per-entity (production: `bert-base-japanese-v3`)

| Entity | F1 | Precision | Recall | Support | 게이트 0.95 |
|---|---:|---:|---:|---:|:---:|
| PHONE | 1.0000 | 1.0000 | 1.0000 | 101 | ✅ |
| EMAIL | 0.9944 | — | — | 88 | ✅ |
| DAT | 0.9831 | — | — | 89 | ✅ |
| PER | 0.9615 | — | — | 359 | ✅ |
| CREDIT_CARD | 0.9136 | — | — | 78 | ❌ |
| ORG | 0.8863 | — | — | 486 | ❌ |
| EVT | 0.8750 | — | — | 94 | ❌ |
| ID_NUM | 0.8427 | — | — | 83 | ❌ |
| LOC | 0.8513 | — | — | 259 | ❌ |
| PROD | 0.8167 | — | — | 108 | ❌ |

> per-entity P/R 은 sweep 시점 기록에서 일부 누락. 다른 모델은 아래 표
> 참조 (전 모델 P/R/F1 풀세트).

### 모델별 per-entity F1

| Entity | base | large | mmBERT | modernbert-ja | llm-jp-modernbert |
|---|---:|---:|---:|---:|---:|
| PER | 0.9615 | 0.9507 | 0.8382 | 0.8698 | 0.9248 |
| LOC | 0.8513 | 0.8526 | 0.7519 | 0.7007 | 0.7942 |
| ORG | 0.8863 | 0.8751 | 0.7660 | 0.7520 | 0.8094 |
| PROD | 0.8167 | 0.7725 | 0.7426 | 0.5646 | 0.7243 |
| EVT | 0.8750 | 0.7979 | 0.7059 | 0.6701 | 0.7115 |
| DAT | 0.9831 | 0.9944 | 0.8268 | 0.8136 | 0.8603 |
| EMAIL | 0.9944 | 0.9040 | 0.5311 | 0.8023 | **0.1695** |
| PHONE | 1.0000 | 0.9950 | 0.6634 | 0.6567 | 0.5644 |
| ID_NUM | 0.8427 | 0.8621 | 0.7630 | 0.8966 | **0.1163** |
| CREDIT_CARD | 0.9136 | 0.8765 | 0.9342 | 0.8623 | **0.3106** |

(P/R 풀세트는 `results/classifier/ja/sweep/<모델>/metrics.json` 참조.)

### llm-jp-modernbert PII 붕괴 — P/R 상세

| Entity | F1 | Precision | Recall | Support |
|---|---:|---:|---:|---:|
| EMAIL | 0.1695 | 0.1685 | 0.1705 | 88 |
| ID_NUM | 0.1163 | 0.1124 | 0.1205 | 83 |
| CREDIT_CARD | 0.3106 | 0.3012 | 0.3205 | 78 |
| PHONE | 0.5644 | 0.5644 | 0.5644 | 101 |

P 와 R 이 거의 동일 — false positive 와 false negative 가 동시에 발생.
**boundary 오류가 아니라 토큰 분해 자체의 실패**.

## 학습 실패 모델 분석

### DeBERTa-v3 family (`ku-nlp/deberta-v3-base-japanese`, `microsoft/mdeberta-v3-base`)

| 항목 | 관측값 |
|---|---|
| 학습 loss | 정상 감소 (172 → 4.6) |
| eval_loss | 정상 감소 (70 → 3.7) |
| 평가 prediction | **모두 'O'** (majority-class collapse) |
| 모델 weight | NaN 없음 |
| classifier head 학습 신호 | 평균 ≈ 0.004 (미세하지만 받음) |
| 시도한 설정 | bf16 (fp16 unscale 에러 회피용), lr=2e-5 |

**원인 추정**: DeBERTa-v3 family 는 알려진 unstable family — `lr=2e-5 + bf16`
의 단순 설정으로는 token-classification head 가 majority-class 'O' 로 붕괴.
warmup_ratio (0.06~0.1), weight_decay (0.01), gradient_clip (1.0) 의 추가
튜닝이 필요. 본 sweep 은 모든 후보를 동일 프로토콜로 비교하기 위해 추가
튜닝을 적용하지 않았다.

JA·VI 양 언어에서 동일 패턴 재현 — 별도 hyperparameter sweep 이슈
(post-#40 핸드오프 §F3) 에서 분리 검증.

### Modern tokenizer fragmentation (`mmBERT-base`, `modernbert-ja-130m`, `llm-jp-modernbert-base`)

학습 자체는 성공하지만 PII 클래스가 collapse.

| 모델 | EMAIL F1 | ID_NUM F1 | CREDIT_CARD F1 | PHONE F1 |
|---|---:|---:|---:|---:|
| llm-jp-modernbert-base | 0.17 | 0.12 | 0.31 | 0.56 |
| modernbert-ja-130m | 0.80 | 0.90 | 0.86 | 0.66 |
| mmBERT-base | 0.53 | 0.76 | 0.93 | 0.66 |

**원인**: 다국어/현대형 BPE 가 일본어 PII 의 영숫자(`@`, 숫자열) 를
서브워드로 부정확 분해 → label-id 정렬이 깨지면서 PII 토큰이 학습되지
않음. base BERT (`BertJapaneseTokenizer`, slow + MeCab) 는 이 문제가 없다.
근거: `docs/issues/issue-40-classifier-restore.md:242`.

### 제외된 모델

- `studio-ousia/luke-japanese-large-lite` — `LukeTokenizer` 호환성 문제로
  현 `data_utils._encode_ja` 경로에 통합 불가. smoke run 결과만 보존
  (`results/classifier/ja/checkpoint-7/`, F1=0). 별도 어댑터 필요.

## 변종 ablation (v1~v4, 80/20 분할 기준)

> 누적 변종은 80/20 분할 시점 측정값. 80/10/10 분할로 갱신된 baseline
> (0.9058) 과 직접 비교하지 말 것.

| 변종 | 누적 변경 | F1 | Precision | Recall |
|---|---|---:|---:|---:|
| v1 baseline | 표준 CE / `eval_loss` best / base 모델 | 0.8952 | — | — |
| **v2** classwt + NER-best | NER B/I weight=2.0, PII B/I weight=0.5, `metric_for_best='ner_f1'` | **0.9134** | 0.8888 | 0.9394 |
| v3 + large | `bert-large-japanese-v2` | 0.9150 | 0.8966 | 0.9341 |
| v4 + curriculum | NER warmup 3 epoch + 21-class fine-tune | 0.9121 | 0.8875 | 0.9380 |
| **gate** | — | 0.9500 | — | — |

원시 메트릭: `results/classifier/ja/{v2,v3,v4}/metrics.json`.

핵심:
- **v2 가 ROI 최고** (+1.82pp). v3 (+0.16pp / 4-5x 시간) 비실용적.
- **PII 5종은 v1 부터 0.95+ 포화** — class weight down-weight 효과 미미.
- **NER 5종 +1~2pp 상승** — class weight 의 NER 측 효과는 부분 성립.
- **JA PROD v3 = 0.840 (+8.2pp vs v1)** — large + class weight 가
  lexical-poor 클래스에 단일 최대 개선.
- EVT 는 어떤 변종도 87% 천장 — 어휘·문맥 다양성 부족.

## Tier 1.B — boundary-relaxed evaluation (이슈 #45 재평가)

같은 baseline 으로 strict / SemEval'13 Partial F1 동시 산출:

| Variant | strict F1 | strict P | strict R | relaxed F1 | relaxed P | relaxed R | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | **0.9077** | 0.8792 | 0.9381 | **0.9177** | 0.8888 | 0.9484 | +0.0100 |
| evt4 | 0.8970 | 0.8593 | 0.9381 | 0.9074 | 0.8693 | 0.9490 | +0.0104 |
| evt_replace_v2 | 0.8950 | 0.8747 | 0.9163 | 0.9079 | 0.8873 | 0.9295 | +0.0129 |
| prod4_evt4 | 0.8890 | 0.8500 | 0.9318 | 0.9019 | 0.8623 | 0.9453 | +0.0128 |
| kwdlc_all5 | 0.8875 | 0.8630 | 0.9135 | 0.9020 | 0.8771 | 0.9284 | +0.0145 |
| evt_replace | 0.8841 | 0.8415 | 0.9312 | 0.8988 | 0.8555 | 0.9467 | +0.0147 |
| kwdlc_prod | 0.8832 | 0.8563 | 0.9117 | 0.8993 | 0.8719 | 0.9284 | +0.0161 |
| prod4 | 0.8823 | 0.8537 | 0.9129 | 0.8992 | 0.8700 | 0.9304 | +0.0169 |

**baseline 이 strict·relaxed 양쪽 1위**. augmentation variant 의 회귀가
boundary 손실이 아닌 *진짜 학습 손실*. 데이터 천장 가설 확정.

## NER 5종 vs PII 5종 가중 평균 (production baseline)

| 그룹 | F1 |
|---|---:|
| NER 5종 (PER/LOC/ORG/PROD/EVT) | 0.886 |
| PII 5종 (DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD) | 0.948 |

## 현재 진행 중 — 0.95 도달 시도

본 baseline 의 0.95 미달은 데이터·평가 양 측 천장으로 동결됐으나, 천장
자체를 분리·확장하기 위해 **이슈 #48** 이 다음 트랙으로 진행 중.

### Tier 3 — test-set error analysis (완료)

목적: baseline 의 오답 케이스 검수 → gold 라벨 오류율 측정.
**모델 천장 vs gold 천장 분리**.

본 트랙은 두 단계로 진행됐다 (v1 → audit cleanup → v2). 별도 리포트는
신설하지 않고 본 섹션에 누적 갱신.

#### 1단계 — 전수 검수 + 보정 baseline (v1, 중간)

- 신규 코드: `src/ner/classifier/error_analysis.py` (오답 추출 CLI),
  `tests/ner/classifier/test_error_analysis.py` (단위 테스트)
- 5,270-row pii_all.jsonl 위에서 baseline 재학습 → strict 0.8925,
  relaxed 0.9104 (옛 sweep 결과와 분포가 다르므로 본 측정값을 baseline 으로
  채택)
- 1,398 오류 문장 (test 209 + valid 223 + train 966) 전수 사람 검수
  (Gemini chat 보조, 한국어 단독 검토자 환경) — **2,354 errors**
- verdict vocabulary: `model_correct` / `gold_correct` / `both_wrong`
  / `ambiguous`
- gold 결함 비율: model_correct 35.0% + both_wrong 5.2% = **40.2%**
  gold 누락/오류. ambiguous 60건 별도 보관 (canonical schema 보강 후보)
- 보정 적용: 947건 (model_correct/both_wrong 통합) + chunk 단계 67건
  (ambiguous gold_error) = **1,014건**
- 1차 측정 (v1): strict F1 0.9252, relaxed F1 0.9398
- **회귀 발견**: DAT strict F1 0.9802 → 0.9259 (-0.0543) — chunk 단계
  검수 도구 (Gemini chat) 의 한글 부분 번역 → character offset shift 로
  span 좌표 손상

#### 2단계 — chunk 단계 보정 audit cleanup (v2, 최종)

라벨-문법 헤리스틱 (`/tmp/audit_entity_text_quality.py`) 으로 손상된
entity 18건 자동 제거 + 수동 확정 3건:

| 라벨 | 제거 수 | 패턴 |
|---|---:|---|
| DAT | 14 | hiragana 조사 leakage (に, の, る, けだ, えて 등) |
| LOC | 2 | 조사 'に' 시작 / 'の' 종료 leakage |
| CREDIT_CARD | 1 | 일본어 텍스트 leakage (`を管理番`) |
| ID_NUM | 1 | 일본어 텍스트 leakage (`の識別番号を持つ部`) |

순 보정 entity = 1,014 - 18 = **996** (in-place 적용, 백업 미보존).

#### v1 → v2 측정 비교

| 단계 | strict F1 | strict P | strict R | relaxed F1 | support |
|---|---:|---:|---:|---:|---:|
| baseline (보정 전) | 0.8925 | 0.8632 | 0.9240 | 0.9104 | 1,762 |
| v1 (chunk 포함) | 0.9252 | 0.9184 | 0.9320 | 0.9398 | 1,897 |
| **v2 (audit cleanup)** | **0.9249** | **0.9217** | 0.9281 | **0.9415** | 1,891 |
| Δ baseline → v2 | **+0.0324** | +0.0585 | +0.0041 | **+0.0311** | +129 |
| Δ v1 → v2 | -0.0003 | +0.0033 | -0.0039 | +0.0017 | -6 |

#### per-entity F1 (baseline → v2, strict / relaxed)

| Entity | b strict | v2 strict | Δ strict | b relaxed | v2 relaxed | Δ relaxed | sup b → v2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EMAIL | 0.9954 | 1.0000 | +0.0046 | 0.9954 | 1.0000 | +0.0046 | 108 → 109 |
| PHONE | 1.0000 | 1.0000 | 0 | 1.0000 | 1.0000 | 0 | 101 → 101 |
| DAT | 0.9802 | 0.9565 | -0.0237 | 0.9802 | 0.9614 | -0.0188 | 101 → 105 |
| ID_NUM | 0.8957 | 0.9724 | +0.0767 | 0.9018 | 0.9724 | +0.0705 | 77 → 89 |
| CREDIT_CARD | 0.8571 | 0.9610 | +0.1039 | 0.8701 | 0.9610 | +0.0909 | 72 → 78 |
| PER | 0.9447 | 0.9603 | +0.0156 | 0.9568 | 0.9722 | +0.0154 | 362 → 375 |
| LOC | 0.8015 | 0.8792 | +0.0777 | 0.8379 | 0.9027 | +0.0648 | 262 → 297 |
| ORG | 0.8675 | 0.8893 | +0.0218 | 0.8906 | 0.9170 | +0.0263 | 516 → 557 |
| PROD | 0.8304 | 0.8770 | +0.0466 | 0.8421 | 0.9037 | +0.0616 | 82 → 93 |
| EVT | 0.8471 | 0.9080 | +0.0610 | 0.8824 | 0.9253 | +0.0429 | 81 → 87 |

DAT 만 -0.0237 잔여 회귀 — 새로 추가된 4건 (`1957年`, `1986年`, `2002年`,
`2001年10月26日` partial) 을 모델이 잡지 못함. 학습 분포에 없는 raw
year-only 패턴 → 순수 모델 한계.

#### 0.95 게이트와 결론

| | F1 | gap to 0.95 |
|---|---:|---:|
| strict | 0.9249 | -0.0251 |
| relaxed | **0.9415** | **-0.0085** |

- strict ↔ relaxed gap = 1.66pp = boundary 정확도 한계
- 보정 baseline strict 0.9249 = **데이터 천장 + 잔여 모델 천장** 통합값
  - 데이터 천장: gold 결함 40.2% 보정 효과 (+3.24pp)
  - 잔여 모델 천장: -2.51pp 로 0.95 미달
- 0.95 도달은 보정 baseline 위에서 +2.5pp 가 추가 필요

#### 데이터 변경 사실 (재현 caveat)

- pii_all.jsonl 은 본 트랙 진행 중 in-place 변경됐다:
  - 행 수: 5,307 → **5,270** (검수 불가 34 sentences 직접 삭제, 3 collateral)
  - entities: net +996 보정 (1,014 보정 - 18 audit cleanup)
  - 백업 미보존 — 5,307-row 원본은 augmenters/pii 재실행으로 재생성만 가능
- 검수 입력 (review_full_*.adjudicated.jsonl, 1.jsonl~6.jsonl) 및
  보정·audit 스크립트 (`/tmp/apply_corrections.py`,
  `/tmp/apply_ambiguous_corrections.py`,
  `/tmp/audit_entity_text_quality.py`) 모두 일회성 — 재실행 불가
- 본 측정값은 **현 시점 pii_all.jsonl** 의 절대값이며, 옛 5,307-row
  측정값 (sweep 표 0.9058) 과 동일 분포 비교는 불가

### Tier 2.B — Self-training + adaptive thresholding (병렬)

목적: JA Wikipedia unlabeled corpus pseudo-label + Gaussian per-class
threshold 로 EVT/PROD 다양성 보강.

- 위험: 이슈 #45 의 LLM augmentation 회귀 패턴 (precision 회귀 >
  recall 회귀, false-positive 노이즈) 재현 시 채택 거부.
- cross-verifier (`augmenters/pii` 패턴 재사용) + 사람 spot check 200문장
  (≥ 90% 미달 시 데이터 사용 중단).

### 0.95 도달 가능성 판정 시점

- ✅ Tier 3 완료 — *gold 천장* 분리됨. 보정 baseline 동결값 = 본 리포트
  Tier 3 §"v1 → v2 측정 비교"
- ⏳ Tier 2.B 미시도 — 보정 baseline 대비 Δ 측정 보류
- 어느 트랙도 미달 시 → (A) 게이트 완화 / (B) 천장 동결 / (C) 추가 SOTA
  기법 분리 중 사용자 결정 (현 시점 잠정 (B) 가능: strict 0.9249 /
  relaxed 0.9415 production 후보)

상세: `docs/issues/issue-48-ja-classifier-tier2-tier3-followups.md`.

## 동결된 후속 트랙 (별도 이슈 후보)

이슈 #45 종결로 다음은 **현 데이터·모델 셋업에서 효과 없음** 으로 동결:

- F6-JA — 외부 코퍼스 통합 (SHINRA / KWDLC / UNER JA 등): 무료 + 네이티브
  EVT/PROD span 라벨 코퍼스 사실상 부재
- F3-JA — DeBERTa-v3 hyperparameter sweep (warmup·weight_decay·grad_clip):
  잠재력 미검증 상태로 *분리 이슈 후보* (본 sweep 에서 학습 실패만 확인)
- LUKE-japanese 통합: tokenizer 어댑터 신규 작성 비용

## 재현

```bash
# 데이터 (생성은 augmenters 측, gitignore 됨)
ls data/stockmark/pii_all.jsonl

# 의존성 (JA tokenizer)
uv add fugashi unidic-lite
uv sync

# Production baseline 학습 + 평가
python -m ner.classifier --lang ja \
  --epochs 5 --batch-size 16 --max-length 256

# strict + relaxed 동시 평가 (Tier 1.B)
python -m ner.classifier --lang ja \
  --epochs 5 --batch-size 16 --max-length 256 \
  --metric-mode both

# 단위 + 토크나이저 round-trip 테스트
python -m pytest tests/ner/classifier/ -q
```

## 산출물 위치

| 경로 | 내용 |
|---|---|
| `results/classifier/ja_sweep/baseline/metrics.json` | production 후보 (strict + relaxed) |
| `results/classifier/ja_sweep/baseline_corrected/metrics.json` | Tier 3 v2 (audit cleanup) 동결 — strict 0.9249, relaxed 0.9415 |
| `results/classifier/ja/sweep/<모델>/metrics.json` | SOTA sweep 6 후보 |
| `results/classifier/ja/{v2,v3,v4}/metrics.json` | 변종 ablation 메트릭 |
| `results/classifier/ja_sweep/baseline/error_analysis_{train,valid,test}.json` | 오답 케이스 dump (Tier 3 입력) |

`results/` 는 gitignore 대상 — 본 리포트의 표가 영구 인용 가능한 단일 출처.
