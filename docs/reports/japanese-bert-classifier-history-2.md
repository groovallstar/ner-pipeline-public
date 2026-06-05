# 일본어 BERT NER 분류기 — 실험 히스토리 2편

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

## 실험 — ORG 정밀도: gold 품질 천장 확인 (#71 + 누락 회복 후속)

청정 기준선 최대 오류 기여자 ORG (support 5,133 = 28%, **P 0.8665 <
R 0.9133 = 과예측**) 를 진단·처방했다. 결론: ORG 잔여 오류는 모델 한계가
아니라 **gold 품질 한계** — "환각" FP 의 다수가 실은 gold 누락 (모델이
맞음) 이라, 측정 ORG 정밀도는 gold 노이즈로 깎인 하한이다.

### 진단 — ORG FP 722 / FN 445 전수 분해 (pooled, 재추론 0)

`error_analysis.py --from-predictions` 로 fold 예측을 합쳐 분해:

| 경로 | ORG FP | ORG FN | 비고 |
|---|---:|---:|---|
| clean 환각 (gold-ORG 0회) | 391 | — | 노이즈 + gold 누락 혼재 (아래 재분해) |
| BOUNDARY (접미 절단) | 176 | 176 | `中国軍`→`中国`, 99% 토큰 표현 가능 |
| TYPE_MISMATCH | 79 | 68 | ORG↔LOC/PROD |
| ambiguous (gold-ORG 도 됨) | 76 | 20 | `海軍`·`民主党` 문맥의존 |

경계 176 은 모델이 접미 토큰 (`軍`/`院`) 을 O 로 떨군 추론 문제,
ambiguous 96 은 문맥의존 정상 라벨이라 일괄 보정 대상 아님.

### 처방 1·2 — 모델·초기 데이터 측 모두 노이즈 내

- **boundary loss I-weight** (I=1.5/2.0/2.5 ×10-fold): 최선 ORG F1 0.8916,
  경계 −7% 뿐 — 전역 P↔R 맞교환 → 폐기.
- **LLM 2-모델 census**: 명명 조직 89 span 추가 (5,133→5,222), F1
  0.8893→0.8903. OVER 의심 53 은 대부분 진짜 ORG (gold 정확 역증명, 제거
  0). 데이터 품질↑, F1 무효.

### clean 환각의 정체 — 76% 가 모델이 맞거나 거의 맞음 (핵심)

census 후에도 남은 clean 환각 (현 gold 기준 344건 / 327 surface) 을
스키마로 전수 판정하니 "모델 오류" 가 아니라 정반대 4종의 혼합이었다:

| 무리 | 고유 surface | 비중 | 정체 |
|---|---:|---:|---|
| GOLD 누락 (모델 맞음) | ~120 | ~37% | 진짜 명명 조직 누락 (`三井住友銀行`·`東京地裁`·`国務院`) |
| TYPE 혼동 (ORG 아님) | ~80 | ~24% | LOC (노선·성·공원·묘지)·PER·PROD |
| FRAGMENT (경계 누수) | ~50 | ~15% | 더 큰 조직의 일부만 (`厚生労働`→省) |
| 진짜 과예측 (모델 틀림) | ~75 | ~23% | 일반명사·스키마 제외 (`政権`·`帝国`·`宗派`) |

GOLD누락 + TYPE + FRAGMENT = **~76% 가 모델이 진짜 엔티티를 찾아냄** →
ORG "천장" 은 모델이 아니라 gold 품질 문제, 순수 모델 과예측은 ~23% 뿐
(고유 surface 기준, 회색지대 ±5pp).

### 누락 회복 +78 — 데이터·측정 정확성 보정 (F1 레버 아님)

GOLD누락 중 *범주적으로 명백한* 명명 조직 69 surface / 78 span 을 스키마
근거로 일관 라벨 (전 occurrence) 로 gold 에 추가 (5,222→5,300). 안티순환:
회색 metonym (도시명=구단)·PROD·국명+軍/艦隊 서술형·시설 *종류* 는 제외.

- **재학습 0 리스코어** (모델 +78 미학습 = 순수 gold 효과): ORG F1
  0.8932→0.9001 (**+0.69pp**). 78 중 72 가 FP→TP (P +1.32pp), 6 은 신규
  FN — FP/FN 자연 균형 (순환 아님).
- **보정 gold 10-fold 재학습** (아래 표): ORG **P +1.77pp** 회복,
  R −0.94pp (신규 FN), F1 +0.47pp.

| gold | overall F1 | ORG F1 | ORG P | ORG R | sup |
|---|---:|---:|---:|---:|---:|
| base | 0.9195 | 0.8893 | 0.8665 | 0.9133 | 5,133 |
| census +89 | 0.9180 | 0.8903 | 0.8646 | 0.9175 | 5,222 |
| **+누락 회복 +78** | 0.9232 | **0.8950** | **0.8823** | 0.9081 | 5,300 |

추가분이 전체 span 의 0.4% 라 overall 변화는 run 변동과 섞인다 — 귀속
가능한 신호는 **ORG 정밀도 회복** (모델의 ORG 예측이 보정 gold 로 검증됨).
데이터·측정 정확성은 회복했으나 게이트 레버는 아니다.

### 결론

`results/classifier/ja_sweep/kfold10_orgfix/` (보정 데이터 10-fold) +
`.../kfold10_phonediv_noextra/diag_org/` (ORG 진단) +
`.../org_census/{gemma,qwen}_preds.json·gaps.jsonl·over.jsonl`
(`results/`·`data/` 는 gitignore — 수치 영구 출처는 본 문서)

## 실험 — PROD 천장 진단 + 전수 독립 census (#73)

청정 baseline 최하위 클래스 PROD (F1 0.7895, P 0.7656 / R 0.8149,
support 994) 의 잔여 오류가 gold 천장인지, gold 누락(gap) 보정으로
회복 가능한지를 **model-neutral** 로 측정했다.

### PROD pooled 진단 (재추론 없이)

- FP 248 = HALLUCINATION 171 / BOUNDARY 61 / TYPE 16
- FN 184 = MISS 88 / BOUNDARY 61 / TYPE 35
- 모델은 PROD 과예측(P<R), leak 은 정밀도 쪽. TYPE 혼동 PROD↔ORG 지배.

### gold gap census — 두 방식의 대조

독립 판정자 3종(평가 BERT 와 무관): gemma-4-31B + Qwen3.6-35B +
Claude(canonical schema). gold 는 Stockmark 인간 주석이라 독립.

| census | 탐색 범위 | gap | PROD F1 | Δ |
|---|---|---|---|---|
| model-FP (편향) | 모델 HALL FP 171 | 25 | 0.8049 | +1.54pp |
| **model-neutral** | 5,270 전수 gemma∩qwen | **49** | **0.7933** | **+0.38pp** |

model-FP 의 +1.54pp 중 ~1.2pp 는 **모델-편향**(모델이 이미 맞춘 자리만
gold 화 = FP→TP 직접 전환). 전수 독립으로 편향 제거 시 **+0.38pp =
노이즈 바닥(±1.28pp) 미달**. 서명: HALL FP 171→151(↓) 이나 MISS
88→96(↑) — 독립 탐색 gap 은 모델이 못 잡아 새 FN 이 되어 F1 불변.

### 전수 census 판정 (GAP 149)

ACCEPT 61 / REJECT 41 / **BORDERLINE 47**. BORDERLINE = 군함·전차·함포·
법령·프로젝트·전시·랭킹 등 **canonical schema 미규정 범주** — ACCEPT 와
맞먹는 규모이며 gold 비일관·PROD 천장의 구조적 원인. OVER(gold=PROD,
LLM 부정) 20 중 진짜 over-label 은 법령 3건뿐 → gold 는 누락 한 방향.

### 결론

PROD 도 ORG 와 동일하게 **gold 천장** — gold gap 을 model-neutral 로
보정해도 held-out F1 이 노이즈 위로 안 올라간다(+0.38pp). 잔여는
long-tail 1회성 MISS + PROD↔ORG/EVT 본질 모호성 + schema 미규정
회색지대. 측정 결과 **미측정 증강 도구 `prod_seed`/`negative` 폐기**
(PROD 신호 추가가 census2 에서 HALL 재생성 → recall oversample 은
precision leak 악화로 무효임을 직접 측정).

이후 ORG gold 수정(누적 5300 = #69 5133 + #71 census 89 + 추가 78)까지
반영한 현재 gold(ORG 5300 / PROD 1043) 전체 재측정: overall strict
**0.9167** / ORG **0.8932** / PROD **0.7826**. gold 수정(ORG +167 /
PROD +49)에도 per-entity 변동은 노이즈 바닥 내 → 천장 재확인 (구 목표
0.95 대비 −3.3pp). 현재 gold 는 gitignore 로컬, 영구 재현 기준은 #69 의
0.9195.

### 산출물

`.../kfold10_phonediv_census`(model-FP)·`…_census2`(model-neutral) +
`.../diag_prod/{fullcorpus_prod_labels,census_fullcorpus_candidates,
census_fullcorpus_verdicts}` (`results/`·`data/` gitignore — gold 49건
보정은 로컬 한정·미커밋, 영구 수치 출처는 본 문서의 994 기준)
