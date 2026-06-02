# 일본어 BERT NER 분류기 — 실험 히스토리 2편 (층화 K-fold 프로토콜)

측정 프로토콜 개편 (#69) 이후의 일본어 NER classifier
(`src/ner/classifier/`, `--lang ja`) 실험 기록. 1편
(`japanese-bert-classifier-history.md`, Phase 0~8, 동결) 과는 측정
프로토콜이 달라 **수치를 직접 비교할 수 없다** — 1편은 80/10/10 단일
분할 (test 527 문장, Phase 7~8 은 extra 중복 포함), 본 문서는 층화
10-fold 교차 검증 pooled (test 5,270 문장 전체, 중복 자동 검증).

- 요약본: `japanese-bert-classifier-benchmark.md`
- 운영 규칙: 새 실험이 종결되면 본 문서에 섹션을 추가한다. 실험 명칭은
  자기서술형으로 붙인다 (라운드·Phase 번호 같은 불투명 카운터 금지)

## 측정 프로토콜 — 층화 10-fold 교차 검증

- 층화 기준: 희소 entity (PROD/EVT) 보유 여부 → fold 간 균등 분포
- fold k = test, fold (k+1)%10 = valid, 나머지 8개 fold = train
  → **train 4,216 / valid 527 / test 527** (1편의 80/10/10 과 동일 크기)
- fold 0~9 로 10회 학습 후 전체 test 예측을 합쳐 **pooled micro-average
  F1** 산출. 모든 문장이 정확히 1회 test 에 등장 → 유효 test = 전체
  5,270 문장. 측정 노이즈: overall ±0.63pp → ±0.28pp, PROD ±4.10pp →
  ±1.28pp, EVT ±3.69pp → ±1.21pp

공통 조건: `tohoku-nlp/bert-base-japanese-v3`, epochs=5, BS 16, LR 5e-5,
max_len 256, fp16, boundary-aware loss (B=1.5/I=1.2),
`metric_for_best_model='eval_loss'`, 분할 seed 42.
RTX A6000, fold 당 학습 ~97초.

```bash
# 10-fold 학습 + pooled 평가
for fold in 0 1 2 3 4 5 6 7 8 9; do
  CUDA_VISIBLE_DEVICES=0 uv run python -m ner.classifier --lang ja \
    --data data/stockmark/pii_all_phonediv.jsonl \
    --boundary-b-weight 1.5 --boundary-i-weight 1.2 \
    --kfold 10 --fold-index ${fold} \
    --output-dir results/classifier/ja_sweep/<실험명>/fold${fold} \
    --seed 42 --metric-mode strict
done
uv run python -m ner.classifier.kfold_pool \
  --fold-dirs results/classifier/ja_sweep/<실험명>/fold{0..9}
```

## 배경 — 왜 프로토콜을 바꿨나 (#69)

1. **단일 분할 측정은 노이즈가 분산을 지배.** base 단독 5-seed 재분할
   실측 (seed 42~46): strict F1 = 0.9249 / 0.9292 / 0.9095 / 0.9181 /
   0.9198 — median 0.9198, std ±0.0075, 폭 1.97pp. per-entity 는 PROD
   10.3pp / EVT 7.5pp 폭. 분산 분해 결과 seed 간 분산의 **72~100% 가
   표본추출 노이즈** (test 527 문장, 희소 entity ~94 span) — 데이터
   편향이 아니라 측정 프로토콜 문제로 판명.

2. **extra train 데이터의 train/test 중복 발견.** production (S8) 의
   extra (`pii_extra_s8_prod_domain_N5.jsonl`) 에 test 문장의 30~35% 가
   원문·동일 라벨 그대로 포함 — `ja.negative` 후보 선정이 전체 행 대상
   + `ja.prod_seed` 생성 seed (42) / 학습 seed (45) 불일치가 원인.
   production checkpoint (S8 seed 45) 를 중복/비중복으로 나눠 측정:

   | test 부분 | 문장 | strict F1 |
   |---|---:|---:|
   | 전체 (1편 보고값) | 527 | 0.9644 |
   | 중복 부분 (extra 에 원문 존재) | 187 | 0.9947 (EVT/ORG/PER 전부 1.0 = 암기) |
   | **비중복 부분** | 340 | **0.9478** (게이트 미달) |

   → 1편 Phase 7~8 수치는 ~+1.7pp 부풀려진 값. 중복 수정은 별도 이슈로.

## 실험 — base 단독 10-fold 측정 (#69)

extra 없는 base 데이터 (`pii_all_phonediv.jsonl` 단독) 의 청정 기준선
(clean baseline) 확정 + 층화 10-fold 프로토콜 첫 실측.

### pooled 결과 (전체 5,270 문장, 18,349 span)

| 메트릭 | strict | relaxed (SemEval'13 Partial) |
|---|---:|---:|
| **F1** | **0.9195** | 0.9309 |
| Precision | 0.8982 | 0.9092 |
| Recall | 0.9420 | 0.9536 |

fold 간 분포: mean 0.9196 / std ±0.0084 / 범위 [0.9056, 0.9355]
(fold 0~9: 0.9232 / 0.9196 / 0.9143 / 0.9056 / 0.9276 / 0.9195 /
0.9105 / 0.9193 / 0.9355 / 0.9207). 개별 fold 값은 여전히 노이즈가
크므로 **pooled 값이 본 프로토콜의 측정값**이다.

per-entity (strict, pooled):

| entity | F1 | P | R | support |
|---|---:|---:|---:|---:|
| EMAIL | 0.9985 | 0.9970 | 1.0000 | 1,013 |
| PHONE | 0.9933 | 0.9907 | 0.9959 | 964 |
| PER | 0.9692 | 0.9554 | 0.9834 | 3,726 |
| DAT | 0.9582 | 0.9545 | 0.9620 | 1,025 |
| ID_NUM | 0.9487 | 0.9353 | 0.9624 | 932 |
| CREDIT_CARD | 0.9244 | 0.8922 | 0.9590 | 854 |
| LOC | 0.9010 | 0.8749 | 0.9287 | 2,832 |
| ORG | 0.8893 | 0.8665 | 0.9133 | 5,133 |
| EVT | 0.8494 | 0.8055 | 0.8984 | 876 |
| PROD | 0.7943 | 0.7493 | 0.8451 | 994 |

### 해석

1. **base 단독 청정 기준선 = strict F1 0.9195** (게이트 -3.05pp).
   5-seed median 0.9198 과 일치 — 수준 교차 확인.
2. **노이즈 감소 목표 달성.** ±1.5pp 이상의 per-entity 변화는 이제
   노이즈가 아닌 실제 효과로 판별 가능.
3. **전체 오류의 최대 기여자는 ORG** (support 5,133 = 전체의 28%,
   F1 0.8893). PROD 는 F1 이 가장 낮지만 (0.7943) support 가 5.4% 라
   전체 기여는 ORG 보다 작다. 게이트 도달이 목표면 ORG 가 1순위.
4. **1편 production 0.9644 와의 격차** = extra 진짜 기여 (+2.5~3pp,
   비중복 부분 0.9478 기준) + 중복 인플레이션 (~+1.7pp) + 프로토콜
   차이. extra 를 중복 없이 재생성해 본 프로토콜로 재측정하는 것이
   후속 작업.

### 산출물

`results/classifier/ja_sweep/kfold10_phonediv_noextra/` —
fold{0..9}/metrics.json·test_predictions.json + pooled_metrics.json
(`results/` 는 gitignore 대상 — 수치의 영구 인용 출처는 본 문서)
