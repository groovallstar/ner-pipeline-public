# 일본어 BERT NER 분류기 벤치마크 (canonical 10종 평면)

- 측정일: 2026-05-04 ~ 2026-05-07 (Tier 1.B 재평가 + Tier 3 gold cleanup +
  이슈 #51 라운드 2)
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

- **현 production 모델**: `tohoku-nlp/bert-base-japanese-v3` (110M)
  + 데이터 보정 라운드 2 + boundary-aware loss (B=1.5, I=1.2)
  + **S7 환각 부정 예시 oversample (N=2, ambiguous 제외, 이슈 #58)**.
  최종 strict F1 = **0.9624** / relaxed F1 = **0.9697**.
  이전 production (v3+boundary, 이슈 #51, strict 0.9368) 대비 +2.56pp.
- **게이트 0.95 미달 일부 잔존**. 라운드 3 (S0 진단 #56 + S7 부정 예시
  #58) 으로 게이트 통과 5/10 → 7/10. 잔여 미달 3 클래스: LOC P -0.4pp /
  ORG R -0.8pp / PROD R -3.4pp.
- **DeBERTa-v3 family 학습 실패**: `ku-nlp/deberta-v3-base-japanese`,
  `microsoft/mdeberta-v3-base` 모두 F1=0 (majority-class collapse). 원인은
  하이퍼파라미터 미튜닝 (warmup·weight_decay·gradient_clip).
- **fast tokenizer fragmentation**: ModernBERT/mmBERT 가 PII 영숫자
  (`@`, 숫자열) 분해 실패로 EMAIL/ID_NUM/CREDIT_CARD F1 < 0.35.
- **Tier 3 완료**: gold cleanup (2,354 errors 전수 검수, 996건 net 보정 +
  18건 audit cleanup) → 보정 baseline strict F1 = 0.9249 / relaxed F1
  = 0.9415.
- **이슈 #51 라운드 2 완료**: test 오답 추가 검수 40 corrections + CRF
  + boundary loss + class weight ablation. 최선 = v3+boundary
  (strict 0.9368). NER 5종 strict ≥ 0.95 미달 → (B) 천장 동결.
- **PII 5종 → 0.94+** (EMAIL/PHONE 100%, DAT/ID_NUM/CC 0.94~1.00).
  **NER 5종 → 0.87~0.97** (PER 도달, ORG/LOC/PROD/EVT 미달).

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

### Tier 2.B — Self-training (이슈 #51 에서 미시도 결정)

JA Wikipedia unlabeled pseudo-label + Gaussian threshold + cross-verifier
트랙은 이슈 #45 의 LLM augmentation 8/8 회귀 패턴 재현 위험 + PROPOR
2026 의 -0.24~-1.81% 회귀 caveat 으로 *미시도 종결*. 이슈 #51 에서는
대신 라운드 2 (data 보정 + CRF + boundary) 를 진행했고 (B) 천장 동결
채택. Self-training 자체는 향후 별도 이슈 후보로 보존.

## 라운드 2 (이슈 #51 종결) — gold 추가 보정 + CRF + boundary loss

목적: NER 5종 strict P/R 모두 ≥ 0.95 게이트 도달 시도. 4 실험 묶음.

### 진행 흐름

1. **canonical schema 보강** — `docs/manual/data/canonical-entity-schema.md`
   §2.3 / §3 / §5.1 에 4영역 명문화: 경기장·서킷·도시공원, 城·城跡
   史跡, `[지명]+代表` 스포츠 대표팀, 부동산·산업 단지, 약어 단독 ORG
   우선 + 동음이의 가타카나 시그널 룰
2. **test 오답 재검수 라운드 2** — `baseline_corrected` 위에서 test 오답
   138 sentences (285 errors) 검수. 사람 spot check 사용자 결정 후
   **40 corrections** in-place 적용 (32 add + 7 replace + 1 replace_span)
3. **CRF head (E1)** — `pytorch-crf` 의 linear-chain CRF + Viterbi decode
4. **Boundary-aware loss (E3)** — B-/I-/O 토큰별 weight 차등
   (`boundary_weights_tensor`)

### 보정 데이터 변경 (in-place)

`pii_all.jsonl` 5,270 행 유지 + 24 sentences entities in-place 변경:

| 종류 | 수 |
|---|---:|
| add LOC (multi-occurrence + 누락) | 20 |
| add ORG | 4 |
| add PROD/PER/CC/ID_NUM/EVT | 7 |
| replace LOC→ORG (시설/공원/단지/서킷/대표팀) | 6 |
| replace ID_NUM→CC (16자리 generator IIN 매칭) | 1 |
| replace_span ORG (NHK boundary 확장) | 1 |

### 4 variants ablation (test split, baseline_corrected → v3)

| Variant | 변경 | strict F1 | strict P | strict R | relaxed F1 | gap to 0.95 |
|---|---|---:|---:|---:|---:|---:|
| v2 | (이전 baseline_corrected) | 0.9249 | 0.9217 | 0.9281 | 0.9415 | -2.51pp |
| **v3** | + data 보정 라운드 2 | 0.9346 | 0.9395 | 0.9298 | 0.9498 | -1.54pp |
| v3+CRF | + Viterbi BIO 일관성 | 0.9311 | 0.9308 | 0.9313 | 0.9469 | -1.89pp |
| **v3+boundary** | + B/I 가중치 (1.5/1.2) | **0.9368** | 0.9334 | 0.9402 | **0.9505** | **-1.32pp** |
| v3+combined | + NER class wt 2.0 + boundary | 0.9347 | 0.9242 | 0.9454 | 0.9480 | -1.53pp |

### per-entity (v3+boundary, production 후보)

| Entity | strict P | strict R | strict F1 | relaxed F1 | 게이트 0.95 |
|---|---:|---:|---:|---:|:---:|
| EMAIL | 1.0000 | 1.0000 | 1.0000 | 1.0000 | ✅ |
| PHONE | 1.0000 | 1.0000 | 1.0000 | 1.0000 | ✅ |
| PER | 0.9735 | 0.9761 | 0.9748 | 0.9815 | ✅ |
| DAT | 0.9901 | 0.9524 | 0.9709 | 0.9709 | ✅ |
| ID_NUM | 0.9263 | 0.9888 | 0.9565 | 0.9620 | P ❌ |
| CREDIT_CARD | 0.9500 | 0.9383 | 0.9441 | 0.9441 | R ❌ |
| LOC | 0.9369 | 0.9068 | 0.9216 | 0.9412 | ❌ |
| ORG | 0.8946 | 0.9152 | 0.9048 | 0.9266 | ❌ |
| EVT | 0.8681 | 0.8977 | 0.8827 | 0.9050 | ❌ |
| PROD | 0.8515 | 0.9053 | 0.8776 | 0.9082 | ❌ |

### 효과 분석

| 단계 | strict Δ | 단일 최대 효과 | 비고 |
|---|---:|---|---|
| data 보정 (v2→v3) | **+0.97pp** | LOC +2.93 / ORG +1.55 / PROD +1.43 | 전체 단일 최대 효과 |
| +CRF (v3→v3+CRF) | -0.35pp | LOC -2.25 / EVT -3.94 (회귀) | NER 회귀, PII 회복 |
| +boundary (v3→v3+boundary) | +0.22pp | LOC +1.31 / PER +0.67 | NER 5종 전체 안정 향상 |
| +combined (v3→v3+combined) | +0.01pp | LOC +2.23 / PROD +1.47 (BUT DAT -3.49) | NER class wt 가 PII 회귀 |

### 종결 결정 — (B) 천장 동결 → 라운드 3 (S0/S7) 으로 갱신

이슈 #51 시점 결정: NER 5종 strict P/R 모두 ≥ 0.95 게이트 미달 확정. 모든
시도 (data quality + CRF + boundary + class weight) 후에도 ORG/LOC/PROD/
EVT 0.85~0.94 천장. 사용자 결정 → **(B) v3+boundary 채택** (strict 0.9368
/ relaxed 0.9505).

라운드 3 (post-#51): #56 S0 진단으로 *환각이 단일 최대 leak (60건 중 ORG
28)* 확인 → #58 S7 환각 부정 예시 oversample 로 **production 갱신**
(v3+boundary → **v3+boundary+S7N2**, strict 0.9624). 상세 §라운드 3.

상세: `docs/issues/issue-51-ja-classifier-strict-095.md`,
`docs/issues/issue-56-ja-classifier-ceiling-diagnosis.md`,
`docs/issues/issue-58-ja-classifier-s7-negative-augment.md`.

## 라운드 3 (이슈 #56 + #58 종결) — 진단 기반 환각 부정 예시 oversample

라운드 2 종결 후 진단 기반 후속 사이클. 두 단계로 진행:

### 1) S0 진단 (이슈 #56)

`error_analysis.py --with-diagnosis` 신규 — confusion matrix + per-class
FP/FN top dump. v3+boundary 진단 결과:
- TYPE_MISMATCH ≤ 12% — 종류 혼동은 본질 아님
- 환각 60건 중 **ORG 28건 (47%) = 단일 최대 leak**
- 환각 + 경계 어긋남 = 잘못 짚음의 88%

### 2) S1 threshold + asymmetric focal 검증 (이슈 #57 not-planned)

S0 진단으로 PROD/EVT/ORG over-predict 확인 후, 사후 (post-hoc) threshold
tuning 으로 precision 회복 가능성 검증. valid 셋의 per-class confidence
분포에서 strict precision ≥ 0.95 충족 최저 threshold 탐색:

| calibration | PROD threshold | EVT | ORG | test 결과 |
|---|---:|---:|---:|---|
| valid-only (84 PROD pred) | 0.9882 | 0.7855 | 0.7344 | PROD recall **0.34** (붕괴) |
| train+valid (~850 PROD) | 0.6951 | 0.5895 | 0.6287 | overall F1 +0.41pp, ORG/EVT P ≥ 0.95 도달, PROD R 0.82 |

검증 결론:
- threshold 자체는 *예측을 거름* → recall 무조건 ↓ (또는 동등)
- 게이트 정의 (P AND R ≥ 0.95) 와 **구조적 부합 불가능**
- valid-only 표본 부족 (PROD valid ~28건) → threshold 과대추정 → test
  일반화 실패
- train+valid 합본 (~850 PROD) 도 게이트 미달 (PROD P 0.94 / R 0.82)

**결정**: 이슈 #57 **not-planned close**. P/R 동시 천장은 *데이터 측 처방*
으로만 도달 가능 확정 → S7 진행. asymmetric focal loss 2단계도 동일 한계
(학습 측 calibration) → 시도 안 함.

### 3) S7 환각 부정 예시 oversample (이슈 #58)

신규 도구 `python -m ner.augmenters.ja_negative` + classifier
`--data-extra-train-jsonl` 옵션 (leak-free 학습). 5 variant 비교:

| variant | seed | N | F1 strict | HALL | 게이트 |
|---|---|---:|---:|---:|---:|
| (production) v3+boundary | - | - | 0.9368 | 60 | 5/10 |
| v1 leak-free | 57 | 3 | 0.9597 | 32 | 6/10 |
| v2 ambig 제외 | 49 | 3 | 0.9552 | 22 | 6/10 |
| **v3 = S7N2 (채택)** | 49 | **2** | **0.9624** | 30 | **7/10** |
| v4 NER-only | 41 | 3 | 0.9572 | 31 | 5/10 |
| v5 PROD 제거 | 35 | 2 | 0.9525 | 32 | 4/10 |

### per-entity (v3+boundary+S7N2, **갱신된 production**)

| 종류 | strict P | ΔP vs prod | strict R | ΔR vs prod | 게이트 |
|---|---:|---:|---:|---:|---|
| PER | 0.9734 | +0.00 | 0.9708 | -0.53 | ✅ |
| LOC | 0.9459 | +0.91 | 0.9550 | +4.83 | ❌ P -0.4pp |
| ORG | 0.9501 | +5.55 | 0.9417 | +2.65 | ❌ R -0.8pp |
| PROD | 0.9775 | +12.60 | 0.9158 | +1.05 | ❌ R -3.4pp |
| EVT | 0.9659 | +9.78 | 0.9659 | +6.82 | ✅ |
| DAT | 0.9712 | -1.88 | 0.9619 | +0.95 | ✅ |
| EMAIL | 1.0000 | =0 | 1.0000 | =0 | ✅ |
| PHONE | 1.0000 | =0 | 1.0000 | =0 | ✅ |
| ID_NUM | 0.9886 | +6.23 | 0.9775 | -1.13 | ✅ |
| CREDIT_CARD | 0.9518 | +0.18 | 0.9753 | +3.70 | ✅ |
| **overall** | **0.9594** | **+2.61** | **0.9599** | **+1.97** | |

### 핵심 발견 (S7 검증)

- N=2 (보수적) 가 N=3 보다 우월 — recall 손실 완화 + 환각 효과 유지
- ambiguous 자동 제외 (학습 셋에 entity 로 등장한 surface) 가 회귀
  방지에 필수
- PROD seed 제거 (v5) 가 오히려 PROD 자체 악화 — 환각 surface 가 다른
  entity calibration 에도 기여하는 ripple 효과
- NER-only 분리 (PII 환각 제외, v4) 효과 없음 — PII 환각 surface 는 무해

### 다음 작업 + Fallback

잔여 미달 3 클래스 (LOC P -0.4pp / ORG R -0.8pp / PROD R -3.4pp) — 모두
작은 gap. v3+boundary+S7N2 재진단 결과 BOUNDARY 30건이 가장 큰 잔존
표적.

후속 후보:
- **S6** 경계 규칙 명문화 + 데이터 보정 (BOUNDARY 30 직격)
  - `docs/manual/data/canonical-entity-schema.md` 에 정책 추가:
    인용부호 (「」, 『』) 안 entity 범위 / 우편번호 prefix (`〒...`)
    LOC 포함 여부 / 영문 모델명·버전 번호 자르는 기준 / 복합 조직명
    (`国土交通省道路局`) 분리 정책
  - 학습 셋 일괄 보정 (자동 + 표본 검수)
  - 효과 목표: BOUNDARY 30 → < 15, 게이트 통과 8/10 이상
- **S2** Wikidata gazetteer (LOC 한정) — LOC P/R 추가 회복
- (옵션) **S4** targeted annotation — 라운드 2 에서 이미 40 corrections,
  추가 marginal
- (옵션) **S5** backbone swap (xlm-r-large / deberta-ja-v3-mc4) — 최후

Fallback (S6/S2 미달 시):
- **(A)** 게이트 완화 0.95 → 0.93 또는 NER/PII 분리 게이트
- **(C)** rule-based PII union + NER 단독 (EMAIL/PHONE regex 대체)

별도 트랙:
- VI: `handoff-post-issue-40-classifier-followups.md`

## 동결된 후속 트랙 (별도 이슈 후보)

이슈 #45 / #53 / #54 / #55 종결로 다음은 **현 데이터·모델 셋업에서 효과 없음** 으로 동결:

- F6-JA — 외부 코퍼스 통합 (SHINRA / KWDLC / UNER JA 등): 무료 + 네이티브
  EVT/PROD span 라벨 코퍼스 사실상 부재 (이슈 #45)
- F3-JA — DeBERTa-v3 hyperparameter sweep (warmup·weight_decay·grad_clip):
  잠재력 미검증 상태로 *분리 이슈 후보* (본 sweep 에서 학습 실패만 확인)
- E2-JA — NER 5종 oversampling (PROD/EVT 4x 단순 복제, 이슈 #53):
  overall strict F1 -0.06pp (사실상 동등) + EVT 양방향 +5.5pp 단일 성과 /
  PROD precision↔recall trade-off (P +7.87 / R -6.32) + 일반 NER (PER F1
  -1.09, LOC P -2.02) / PII (ID_NUM R -6.74, CREDIT_CARD F1 -0.51) 회귀
  가드 광범위 위반. CLI 미흡수
- F1-JA — ambiguous 라운드 3 검수 (이슈 #54): v3+boundary inference 의
  BOUNDARY/TYPE_MISMATCH 69 pairs 중 schema 4영역 룰로 in-place 보정 가능
  = **4건 / 1922 spans = 0.2%**. 회수 가능 entity 부족 실측 → 즉시 천장 동결
- E7-JA — GLiNER-Japanese span-based extraction (이슈 #55):
  `urchade/gliner_multi-v2.1` zero-shot overall strict F1 = **0.1134**
  (v3+boundary 대비 -82.34pp). canonical schema (ORG=시설 통합, PII 합성)
  vs GLiNER 일반 정의 mismatch 가 핵심. fine-tune 으로 +80pp 회복 비대칭 +
  학습 실패 패턴 위험. inference 38x 슬로우 (76 ms/sent)
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
| `results/classifier/ja_sweep/baseline/metrics.json` | 옛 baseline (5,307 행, 5,270 보정 전) |
| `results/classifier/ja_sweep/baseline_corrected/metrics.json` | Tier 3 v2 (audit cleanup) 동결 — strict 0.9249, relaxed 0.9415 |
| `results/classifier/ja_sweep/baseline_corrected/error_analysis.json` | 라운드 2 입력 오답 dump |
| `results/classifier/ja_sweep/baseline_v3/metrics.json` | 라운드 2 data 보정만 — strict 0.9346, relaxed 0.9498 |
| `results/classifier/ja_sweep/v3_crf/metrics.json` | v3 + CRF — strict 0.9311 |
| `results/classifier/ja_sweep/v3_boundary/metrics.json` | **v3 + boundary loss (production)** — strict 0.9368, relaxed 0.9505 |
| `results/classifier/ja_sweep/v3_combined/metrics.json` | v3 + NER class wt + boundary — strict 0.9347 |
| `results/classifier/ja/sweep/<모델>/metrics.json` | SOTA sweep 6 후보 |
| `results/classifier/ja/{v2,v3,v4}/metrics.json` | 변종 ablation 메트릭 (80/20 분할 시점) |
| `results/classifier/ja_sweep/baseline/error_analysis_{train,valid,test}.json` | 오답 케이스 dump (Tier 3 입력) |

`results/` 는 gitignore 대상 — 본 리포트의 표가 영구 인용 가능한 단일 출처.
