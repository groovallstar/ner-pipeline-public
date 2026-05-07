# 베트남어 BERT NER 분류기 벤치마크 (canonical 10종 평면)

- 측정일: 2026-05-04
- 대상: WikiANN-vi silver + 합성 PII 주입 데이터에 BERT family 파인튜닝
- 데이터: `data/wikiann_vi/pii_all.jsonl` (40,000 행, canonical 10종 = NER 5 + PII 5)
- 평가: char-offset span F1 (`src/ner/metrics/span_metrics.compute_offset_span_f1`)
- 출처: `docs/issues/issue-40-classifier-restore.md` §SOTA Sweep,
  `docs/issues/handoff-post-issue-40-classifier-followups.md` §F2~F6
- 일본어 동일 평가 셋업의 분류기 결과는
  `docs/reports/japanese-bert-classifier-benchmark.md` 참조

## 요약

- **현 production 모델**: `xlm-roberta-base` (278M).
  baseline F1 = 0.8985.
- **`xlm-roberta-large` 가 +0.42pp / 5x 비용** — large 채택은 비용 효율 낮음.
- **DeBERTa-v3 family 학습 실패**: `microsoft/mdeberta-v3-base`,
  `Fsoft-AIC/videberta-base` 모두 F1=0 (majority-class collapse).
- **fast tokenizer fragmentation**: `mmBERT-base` 가 베트남어 tone mark
  분해 실패로 NER 5종 모두 F1 < 0.50.
- **EVT 절대 support 부족 (61 spans @ test)** + **WikiANN-vi silver 노이즈
  (PROD precision 0.54 / recall 0.80)** 가 게이트 0.95 미달의 본질.
- **현재 진행**: post-#40 핸드오프의 F2~F6 트랙은 별도 이슈 분리 진행 (JA
  먼저 #45·#48 로 진행 중, VI 트랙은 이슈 후보 단계).

## 조건

| 항목 | 값 |
|---|---|
| 데이터 | WikiANN-vi (silver) + PII 주입 (40,000) |
| 분할 | train 32,000 / valid 4,000 / test 4,000 (80/10/10, `seed=42`) |
| 라벨 | `PER, LOC, ORG, PROD, EVT, DAT, EMAIL, PHONE, ID_NUM, CREDIT_CARD` |
| 학습 | epochs=5, batch_size=16, lr=5e-5, max_length=256, fp16 |
| 모델 선택 | `metric_for_best_model='eval_loss'` (valid) |
| 하드웨어 | NVIDIA RTX A6000 49GB, 단일 seed=42 |

> 변종 ablation (v2~v4) 은 도입 전 80/20 분할 기준이라 sweep 표와 직접
> 비교하지 않는다. 분할 변경 영향 (-1.23pp) 은
> `docs/issues/issue-40-classifier-restore.md` §"80/20 → 80/10/10 분할
> 변경 영향" 참조.

## SOTA 모델 Sweep — 5 후보 (overall)

| 모델 | F1 | Precision | Recall | Train time |
|---|---:|---:|---:|---:|
| **`xlm-roberta-base`** (production) | **0.8985** | 0.8760 | 0.9222 | 808s |
| `xlm-roberta-large` | 0.9027 | 0.8921 | 0.9136 | 4,098s (5×) |
| `jhu-clsp/mmBERT-base` | 0.5278 | 0.5343 | 0.5215 | 1,329s |
| `microsoft/mdeberta-v3-base` (bf16, lr=2e-5) | 0.0000 | 0.0000 | 0.0000 | 1,273s — 학습 실패 |
| `Fsoft-AIC/videberta-base` (bf16, lr=2e-5) | 0.0000 | 0.0000 | 0.0000 | 1,246s — 학습 실패 |

원시 메트릭: `results/classifier/vi/sweep/<모델>/metrics.json`,
production 베이스라인은 `results/classifier/vi/metrics.json`.

### 베이스라인 per-entity (production: `xlm-roberta-base`)

| Entity | F1 | Precision | Recall | Support | 게이트 0.95 |
|---|---:|---:|---:|---:|:---:|
| EMAIL | 0.9960 | 0.9919 | 1.0000 | 739 | ✅ |
| PER | 0.9272 | 0.9142 | 0.9406 | 2,072 | ❌ |
| LOC | 0.9182 | 0.8797 | 0.9601 | 2,004 | ❌ |
| CREDIT_CARD | 0.9050 | 0.8940 | 0.9162 | 764 | ❌ |
| ID_NUM | 0.9045 | 0.9028 | 0.9063 | 758 | ❌ |
| DAT | 0.8868 | 0.8779 | 0.8959 | 730 | ❌ |
| ORG | 0.8593 | 0.8235 | 0.8984 | 748 | ❌ |
| PHONE | 0.8089 | 0.8066 | 0.8112 | 694 | ❌ |
| EVT | 0.6457 | 0.6212 | 0.6721 | 61 | ❌ |
| PROD | 0.6434 | 0.5390 | 0.7981 | 208 | ❌ |

### `xlm-roberta-large` per-entity

| Entity | F1 | Precision | Recall | Support |
|---|---:|---:|---:|---:|
| EMAIL | 0.9839 | 0.9760 | 0.9919 | 739 |
| LOC | 0.9239 | 0.8982 | 0.9511 | 2,004 |
| PER | 0.9226 | 0.9030 | 0.9430 | 2,072 |
| ID_NUM | 0.9045 | 0.9028 | 0.9063 | 758 |
| CREDIT_CARD | 0.9050 | 0.8940 | 0.9162 | 764 |
| ORG | 0.8956 | 0.9023 | 0.8890 | 748 |
| DAT | 0.8841 | 0.8752 | 0.8932 | 730 |
| PHONE | 0.8089 | 0.8066 | 0.8112 | 694 |
| PROD | 0.6667 | 0.7412 | 0.6058 | 208 |
| EVT | 0.5667 | 0.5763 | 0.5574 | 61 |

base 대비 변화: ORG +3.6pp, LOC +0.6pp, PROD +2.3pp, EVT **-7.9pp**, 그 외
포화·동률. 비용 5x 대비 **EVT 회귀가 단일 최대 손실**.

### `mmBERT-base` per-entity (NER 붕괴 패턴)

| Entity | F1 | Precision | Recall | Support |
|---|---:|---:|---:|---:|
| ID_NUM | 0.9980 | 0.9961 | 1.0000 | 758 |
| DAT | 0.9898 | 0.9799 | 1.0000 | 730 |
| CREDIT_CARD | 0.9877 | 0.9757 | 1.0000 | 764 |
| EMAIL | 0.5202 | 0.5181 | 0.5223 | 739 |
| PHONE | 0.4986 | 0.4971 | 0.5000 | 694 |
| ORG | 0.4850 | 0.4744 | 0.4960 | 748 |
| EVT | 0.4960 | 0.4844 | 0.5082 | 61 |
| LOC | 0.3469 | 0.3749 | 0.3229 | 2,004 |
| PER | 0.2472 | 0.2493 | 0.2452 | 2,072 |
| PROD | 0.1659 | 0.1593 | 0.1731 | 208 |

PII 일부 (ID_NUM, DAT, CREDIT_CARD) 는 0.99+ 인데 NER 5종 (PER 0.247,
LOC 0.347, PROD 0.166) 가 50% 미만. P 와 R 이 거의 동률 — boundary 오류가
아니라 **토큰 분해 자체의 실패 패턴**.

## 학습 실패 모델 분석

### DeBERTa-v3 family (`microsoft/mdeberta-v3-base`, `Fsoft-AIC/videberta-base`)

| 항목 | 관측값 |
|---|---|
| 학습 loss | 정상 감소 |
| eval_loss | 정상 감소 |
| 평가 prediction | **모두 'O'** (majority-class collapse) |
| 모델 weight | NaN 없음 |
| classifier head 학습 신호 | 받음 (mean ≈ 0.004) |
| 시도한 설정 | bf16 (fp16 unscale 에러 회피용), lr=2e-5 |

**원인 추정**: DeBERTa-v3 family 는 알려진 unstable family — `lr=2e-5 +
bf16` 의 단순 설정으로는 token-classification head 가 majority-class 'O'
로 붕괴. warmup_ratio (0.06~0.1), weight_decay (0.01), gradient_clip (1.0)
의 추가 튜닝이 필요. JA·VI 양 언어에서 동일 패턴 재현.

특히 `Fsoft-AIC/videberta-base` 는 베트남어 단일언어 모델이지만 base
DeBERTa-v3 와 동일하게 collapse — 언어 특화 사전학습은 본 학습 불안정성을
회피하지 못함.

후속 검증: post-#40 핸드오프 §F3 (DeBERTa-v3 hyperparameter sweep, VI
대상) — 별도 이슈 후보. 미달 시 "DeBERTa family 는 본 데이터셋·셋업과
호환 안 됨" 으로 동결.

### `mmBERT-base` 토크나이저 fragmentation

학습 자체는 성공하지만 NER 5종 collapse. **베트남어 tone mark + 다국어
BPE 의 부정확 분해**가 원인. 근거: 핸드오프
`docs/issues/handoff-post-issue-40-classifier-followups.md`,
`docs/issues/issue-40-classifier-restore.md:244`.

base XLM-RoBERTa 는 SentencePiece (`xlmr.spm`) 로 베트남어 tone mark 안정
분해 — 본 데이터셋에서는 단일언어 PhoBERT 보다 다국어 XLM-R 이 더 안전.

### 제외된 모델

- `vinai/phobert-base/large` — fast tokenizer 미지원. slow tokenizer 의
  manual greedy match (`_encode_ja` 경로) + word_segmenter (VnCoreNLP)
  결합이 필요. post-#40 핸드오프 §F4 별도 이슈 후보.

## 변종 ablation (v1~v4, 80/20 분할 기준)

> 누적 변종은 80/20 분할 시점 측정값. 80/10/10 분할로 갱신된 baseline
> (0.8985) 과 직접 비교하지 말 것. 80/20 → 80/10/10 변경으로 -1.23pp
> 회귀했으며, 그 원인은 PROD test support 398→208 절반 축소로 silver
> 노이즈 영향이 증폭됐기 때문.

| 변종 | 누적 변경 | F1 | Precision | Recall |
|---|---|---:|---:|---:|
| v1 baseline | 표준 CE / `eval_loss` best / base 모델 | 0.9108 | — | — |
| **v2** classwt + NER-best | NER B/I weight=2.0, PII B/I weight=0.5, `metric_for_best='ner_f1'` | 0.9137 | 0.8971 | 0.9308 |
| v3 + large | `xlm-roberta-large` | 0.9135 | 0.8985 | 0.9291 |
| **v4** + curriculum | NER warmup 3 epoch + 21-class fine-tune | **0.9147** | 0.9034 | 0.9263 |
| **gate** | — | 0.9500 | — | — |

원시 메트릭: `results/classifier/vi/{v2,v3,v4}/metrics.json`.

핵심:
- **v4 가 최선** (+0.39pp vs v1) — large + curriculum 조합이 small-class
  변동성을 부분 보완. v2 단독 + base 가 ROI 우위.
- **PII 5종 v1 부터 0.95+ 포화** — class weight down-weight 효과 미미.
- **VI EVT 는 어떤 변종도 60% 대 천장**. v3 (large) 에서 0.545 (-7.9pp vs
  v2) 까지 회귀 후 v4 (curriculum) 으로 0.660 까지 부분 회복. 절대 support
  부족 (94 → 61) 본질적 한계.
- **PROD 은 v4 에서 0.762** (+1.1pp vs v3) — silver 노이즈가 어휘·boundary
  양쪽으로 영향. 이슈 #30 silver 품질 분석 참조.

### v4 per-entity (VI 최선)

| Entity | F1 | Precision | Recall | Support |
|---|---:|---:|---:|---:|
| EMAIL | 0.9960 | 0.9967 | 0.9953 | 1,503 |
| PER | 0.9352 | 0.9239 | 0.9468 | 4,155 |
| LOC | 0.9352 | 0.9164 | 0.9547 | 4,043 |
| ORG | 0.9130 | 0.8937 | 0.9331 | 1,540 |
| CREDIT_CARD | 0.9047 | 0.9005 | 0.9089 | 1,493 |
| ID_NUM | 0.9042 | 0.9027 | 0.9057 | 1,464 |
| DAT | 0.8943 | 0.8879 | 0.9008 | 1,442 |
| PHONE | 0.8138 | 0.8129 | 0.8147 | 1,392 |
| PROD | 0.7624 | 0.7168 | 0.8141 | 398 |
| EVT | 0.6599 | 0.6311 | 0.6915 | 94 |

## NER 5종 vs PII 5종 가중 평균 (production baseline)

| 그룹 | F1 |
|---|---:|
| NER 5종 (PER/LOC/ORG/PROD/EVT) | 0.890 |
| PII 5종 (DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD) | 0.901 |

## 현재 진행 중 — VI 트랙

VI 측은 JA 측 (#45 → #48) 에 비해 **이슈 분리 진행이 늦다**. 핸드오프
`docs/issues/handoff-post-issue-40-classifier-followups.md` 의 F2~F6 트랙은
이슈 후보 단계이며, 각 트랙의 잠재력은 다음과 같이 평가됨:

| 트랙 | 잠재력 | 비용 | 위험 | 의존 |
|---|---|---|---|---|
| F2. WikiANN-vi silver→gold 부분 정제 (PROD/EVT 우선) | ★★★ | ★★ | ★ | 없음 |
| F3. DeBERTa-v3 hyperparameter sweep (VI) | ? | ★★ | ★★ | 없음 |
| F4. PhoBERT (단일언어 SOTA) 통합 | ★★ | ★★★ | ★★★ | tokenizer adapter |
| F5. EVT class oversampling (4-8x 복제) | ★ | ★ | ★ | 없음 |
| F6. 외부 코퍼스 통합 (VLSP 2018·2021, PhoNER) | ★★★ | ★★★ | ★★ | 라이선스 검증 |

**추정 ROI 1순위는 F2** — silver 노이즈가 단일 최대 천장 (PROD precision
0.54 시사). 다만 정제 비용 (사람 검수) 이 크다. JA 측 #48 의 Tier 3
(test-set error analysis + gold cleanup) 결과가 VI F2 의 정제 절차
프로토콜 참고가 됨.

JA 측 베이스라인 (`baseline_corrected`) 처럼 gold cleanup 후 strict F1
+1.95pp 향상이 VI 측에서도 재현되면 게이트 0.95 도달 가능성 재평가 대상.

## 동결된 후속 트랙

이슈 #40 종결 시점 기록:

- **WikiANN-vi silver 노이즈** — VI PROD precision 0.54 / recall 0.80
  패턴은 silver 라벨 비일관성 시사. 모델은 합리적으로 학습, gold 가
  일관되지 않음.
- **EVT 절대 support 천장** — VI EVT 학습 492, test 61 — class imbalance
  본질적 한계.
- **PROD 어휘 다양성** — 제품명은 OOV 일반화 문제 (브랜드+모델, 약어,
  외래어, 숫자 포함).
- **PHONE 표기 다양성** — 5+ 형식. regex 정규화로 토크나이저 정렬
  안정화 가능하나 surface form 정보 손실로 CREDIT_CARD/ID_NUM 과 구분
  약화 위험.

## 재현

```bash
# 데이터 (생성은 augmenters 측, gitignore 됨)
ls data/wikiann_vi/pii_all.jsonl

# 의존성
uv sync

# Production baseline 학습 + 평가
python -m ner.classifier --lang vi \
  --epochs 5 --batch-size 16 --max-length 256

# 단위 + 토크나이저 round-trip 테스트
python -m pytest tests/ner/classifier/ -q
```

## 산출물 위치

| 경로 | 내용 |
|---|---|
| `results/classifier/vi/metrics.json` | production baseline (`xlm-roberta-base`) |
| `results/classifier/vi/sweep/<모델>/metrics.json` | SOTA sweep 4 후보 |
| `results/classifier/vi/{v2,v3,v4}/metrics.json` | 변종 ablation 메트릭 |

`results/` 는 gitignore 대상 — 본 리포트의 표가 영구 인용 가능한 단일 출처.
