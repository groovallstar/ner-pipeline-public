# 일본어 BERT NER 분류기 — 엔티티별 성능 진단·보정

측정 프로토콜 개편 (#69) 으로 측정 노이즈를 낮춘 뒤, 엔티티별 (ORG·
PROD·CREDIT_CARD·EVT·LOC) 잔여 오류의 정체(gold 천장 vs 수정 가능
버그)를 진단·보정한 기록. 1편 (`japanese-bert-classifier-history.md`,
Phase 0~8, 동결, 시계열 히스토리) 의 후속이며, 1편과는 측정 프로토콜이
달라 **수치를 직접 비교할 수 없다** — 1편은 80/10/10 단일 분할 (test
527 문장, Phase 7~8 은 extra 중복 포함), 본 문서는 층화 10-fold 교차
검증 pooled (test 5,270 문장 전체, 중복 자동 검증).

- 요약본: `japanese-bert-classifier-benchmark.md`
- 운영 규칙: 새 실험이 종결되면 본 문서에 섹션을 추가한다. 실험 명칭은
  자기서술형으로 붙인다 (라운드·Phase 번호 같은 불투명 카운터 금지)

## 측정 프로토콜 — 층화 10-fold 교차 검증

- 층화 기준: 희소 entity (PROD/EVT) 보유 여부 → fold 간 균등 분포
- fold k = test, fold (k+1)%10 = valid, 나머지 8개 fold = train
  → **train 4,216 / valid 527 / test 527** (1편의 80/10/10 과 동일 크기)
- fold 0~9 로 10회 학습 후 전체 test 예측을 합쳐 **pooled micro-average
  F1** (전 fold test 예측을 합쳐 한 번에 계산) 산출. 모든 문장이 정확히
  1회 test 에 등장 → 유효 test = 전체 5,270 문장. 측정 노이즈 (이항 SE,
  pooling 시 √10 축소): overall ±0.63pp → ±0.20pp, PROD ±4.10pp →
  ±1.28pp, EVT ±3.69pp → ±1.21pp

공통 조건: `tohoku-nlp/bert-base-japanese-v3`, epochs=5, BS 16, LR 5e-5,
max_len 256, fp16, boundary-aware loss (B/I 토큰에 가중치를 더 주는
토큰분류 손실; B=1.5/I=1.2),
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
   | **비중복 부분** | 340 | **0.9478** |

   → 1편 Phase 7~8 수치는 ~+1.7pp 부풀려진 값. 중복 수정은 별도 이슈로.

## 실험 — base 단독 10-fold 측정 (#69)

extra 없는 base 데이터 (`pii_all_phonediv.jsonl` 단독, `phonediv` =
PHONE 형식 다양화) 의 청정 기준선 (clean baseline) 확정 + 층화 10-fold
프로토콜 첫 실측.

### pooled 결과 (전체 5,270 문장, 18,349 span)

strict = (시작,끝,종류) 완전 일치, relaxed = SemEval'13 Partial (경계가
살짝 어긋난 건 부분 점수).

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

1. **base 단독 청정 기준선 = strict F1 0.9195**. 5-seed median 0.9198
   과 일치 — 독립 프로토콜 간 교차 확인.
2. **노이즈 감소 목표 달성.** ±1.5pp 이상의 per-entity 변화는 이제
   노이즈가 아닌 실제 효과로 판별 가능.
3. **전체 오류의 최대 기여자는 ORG** (support 5,133 = 전체의 28%,
   F1 0.8893). PROD 는 F1 이 가장 낮지만 (0.7943) support 가 5.4% 라
   전체 기여는 ORG 보다 작다. overall 향상 여지는 ORG 가 1순위.
4. **1편 production 0.9644 와의 격차** = extra 진짜 기여 (+2.5~3pp,
   비중복 부분 0.9478 기준) + 중복 인플레이션 (~+1.7pp) + 프로토콜
   차이. extra 를 중복 없이 재생성해 본 프로토콜로 재측정하는 것이
   후속 작업.

### 산출물

`results/classifier/ja_sweep/kfold10_phonediv_noextra/` —
fold{0..9}/metrics.json·test_predictions.json + pooled_metrics.json
(`results/` 는 gitignore 대상 — 수치의 영구 인용 출처는 본 문서)

## 실험 — ORG 정밀도: gold 품질 천장 확인 (#71 + 누락 회복 후속)

> `gold 천장` = gold 라벨 품질이 만드는 성능 상한 — 모델을 더 키워도
> 잘못·누락된 정답 때문에 측정 F1 이 그 위로 안 올라가는 지점.

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
- **LLM census** (후보 span 을 독립 판정자가 재판정; gemma-4-31B +
  Qwen3.6-35B 후보 탐지 + Claude canonical schema 판정): 명명 조직 89
  span 추가 (5,133→5,222), F1 0.8893→0.8903. OVER 의심 53 은 대부분 진짜
  ORG (gold 정확 역증명, 제거 0). 데이터 품질↑, F1 무효.

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
근거로 일관 라벨 (전 occurrence) 로 gold 에 추가 (5,222→5,300). 모델
편향 회피: 회색 metonym (환유, 도시명=구단)·PROD·국명+軍/艦隊 서술형·
시설 *종류* 는 제외.

- **재학습 0 리스코어** (모델 +78 미학습 = 순수 gold 효과): ORG F1
  0.8932→0.9001 (**+0.69pp**). 78 중 72 가 FP→TP (P +1.32pp), 6 은 신규
  FN — FP/FN 자연 균형 (모델 편향 아님).
- **보정 gold 10-fold 재학습** (아래 표): ORG **P +1.77pp** 회복,
  R −0.94pp (신규 FN), F1 +0.47pp.

| gold | overall F1 | ORG F1 | ORG P | ORG R | sup |
|---|---:|---:|---:|---:|---:|
| base | 0.9195 | 0.8893 | 0.8665 | 0.9133 | 5,133 |
| census +89 | 0.9180 | 0.8903 | 0.8646 | 0.9175 | 5,222 |
| +누락 회복 +78 | 0.9232 | 0.8950 | 0.8823 | 0.9081 | 5,300 |
| **현재 (+ #73 PROD +49)** | **0.9167** | **0.8932** | 0.8707 | 0.9170 | 5,300 |

위 3행은 PROD 994 기준 ORG gold 진행(누락 회복 효과 격리), **마지막 행이
현재 권위 baseline** — ORG 5300 + PROD 1043 결합 gold (§PROD #73 과 동일
측정, overall 0.9167 / ORG 0.8932 / PROD 0.7826). PROD +49 추가로 ORG F1
0.8950→0.8932 (−0.18pp, 노이즈). 추가분이 전체 span 의 0.4% 라 overall
변화는 run 변동과 섞이며, 귀속 가능한 신호는 **ORG 정밀도 회복** (모델의
ORG 예측이 보정 gold 로 검증됨). 데이터·측정 정확성 회복이지 overall F1
레버는 아니다.

### 결론

`results/classifier/ja_sweep/kfold10_orgfix/` (보정 데이터 10-fold) +
`.../kfold10_phonediv_noextra/diag_org/` (ORG 진단) +
`.../org_census/{gemma,qwen}_preds.json·gaps.jsonl·over.jsonl`
(`results/`·`data/` 는 gitignore — 수치 영구 출처는 본 문서)

## 실험 — PROD 천장 진단 + 전수 독립 census (#73)

청정 baseline 최하위 클래스 PROD (본 census 캠페인 noextra 재pool
F1 0.7895, P 0.7656 / R 0.8149, support 994; #69 동결표의 0.7943 과는
학습 비결정성 run 변동 ±1.28pp 내) 의 잔여 오류가 gold 천장인지, gold
누락(gap) 보정으로 회복 가능한지를 **model-neutral** (모델 예측과 무관한
대칭 탐색) 로 측정했다. 이하 census Δ 는 모두 캠페인 base 0.7895 기준.

### PROD pooled 진단 (재추론 없이)

- FP 248 = HALLUCINATION 171 / BOUNDARY 61 / TYPE 16
- FN 184 = MISS 88 / BOUNDARY 61 / TYPE 35
- 모델은 PROD 과예측(P<R), leak 은 정밀도 쪽. TYPE 혼동 PROD↔ORG 지배.

### gold gap census — 두 방식의 대조

독립 판정자 3종(평가 BERT 와 무관): gemma-4-31B + Qwen3.6-35B (후보
탐지) + Claude (canonical schema 판정). gold 는 Stockmark 인간 주석이라
독립. model-FP = 모델 FP 자리만 탐색(편향), model-neutral = 코퍼스 전수
대칭 탐색(무편향). 탐지는 gemma∩qwen 교집합, 채택은 Claude schema 판정.

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
PROD +49)에도 per-entity 변동은 노이즈 바닥 내 → 천장 재확인. 현재
gold 는 gitignore 로컬, 영구 재현 기준은 #69 의 0.9195.

### 산출물

`.../kfold10_phonediv_census`(model-FP)·`…_census2`(model-neutral) +
`.../diag_prod/{fullcorpus_prod_labels,census_fullcorpus_candidates,
census_fullcorpus_verdicts}` (`results/`·`data/` gitignore — gold 49건
보정은 로컬 한정·미커밋, 영구 수치 출처는 본 문서의 994 기준)

### 후속 — schema 규정으로 회색지대 정정 (#84 · ADR 0001 · #85)

위 BORDERLINE 47(canonical 미규정 회색지대)이 PROD 천장의 구조적
원인이라, data 수급이 아니라 **schema 규정**으로 전환했다.

- **#84 4범주**(제조 artifact→PROD / 법령→비-entity / 전시→EVT /
  프로젝트→EVT): gold `prodschema`, PROD 1043→1042. 10-fold pooled
  PROD F1 0.8004→0.8090 (ΔP +1.73 / ΔR −0.11 비대칭 P-레버, 노이즈
  대역 내).
- **ADR 0001 서비스 제외**: 재감사에서 남은 비일관 주범이 "서비스"로
  드러나 PROD 를 positive 재규정(서비스·온라인 운영물·기술 표준 제외).
  오라벨 재배치(회사→`ORG`, 계획·전시·시리즈→`EVT`, 상·훈장·규격→
  비-entity). gold `prodclean` 승격, PROD 1042→**943**(이탈 99 = 재라벨
  20 + 제거 79). JA 프롬프트 positive 통일.

| 지표 | baseline(loccensus) | prodschema | **prodclean** |
|---|---|---|---|
| PROD F1·P·R | 0.8004·0.7725·0.8303 | 0.8090·0.7898·0.8292 | **0.8168·0.7913·0.8441** |
| ORG F1 | 0.8982 | 0.8913 | 0.8965 |
| EVT F1 | 0.8593 | 0.8366 | 0.8353 |
| overall F1 | 0.9256 | 0.9222 | 0.9251 |

PROD F1 상승은 **가장 비일관·난해했던 79 스팬 제거의 기계적 효과**
(support 943≠1043) — baseline 과 비교 불가, 성능 레버 아닌 gold 일관성
보정(본 절 천장 결론과 정합). overall 은 baseline 수준 회복, ORG 회복.
**EVT −2.4pp(vs baseline) single-run 은 실재/노이즈 미판정 → #85 에서
multi-seed(s42·43·44) 재확인.** 상세 `docs/decisions/0001-prod-service-exclusion.md`.

## 실험 — CREDIT_CARD 정밀도: LLM injector 환각 규명 (#78)

ORG·PROD 와 달리 CREDIT_CARD 는 **gold 천장이 아니라 고칠 수 있는
파이프라인 버그**였다. 합성 PII 인데 P 가 낮은(0.8922) 이상치를 진단해
LLM injector 환각을 규명하고, 결정론적 relabel + 재학습으로 +6.6pp 회복.

### 진단 — pooled FP/FN (재추론 0)

청정 base(noextra) fold 예측에서 CREDIT_CARD FP 80 / FN 43 분해:

| FP 80 경로 | 건수 | FN 43 경로 | 건수 |
|---|---:|---|---:|
| 무겹침(환각) | 50 | 순수 MISS | 29 |
| 경계 겹침 | 21 | 경계 겹침 | 12 |
| TYPE 혼동 | 9 | TYPE 혼동 | 2 |

환각 FP 50 중 34 가 **풀카드**(`3470-6936-8716-7867`·`3486070609001385`).
경계 21+12 는 분할/절단 서명(`5525-…-6271`→`…-5760`+`71`,
`3477|266921017989`).

### 원인 — LLM injector 가 무라벨 카드숫자를 창작

본문 내 16자리 카드포맷 표면 **961 중 854 만 CREDIT_CARD, 107 무라벨(O)**.
무라벨 107 분해: 같은 행 중복 0 / 타 행 재사용 27 / **어디에도 라벨 없음
(LLM 창작) 80**. `識別番号`·`管理番号`·`という番号`·수학 문맥 등 일반
"번호" 틀. `extract_spans` 는 주입값을 string-match 로만 라벨하므로,
LLM 이 자연 삽입 중 창작한 여분 카드숫자는 `pii_values` 에 없어 무라벨로
코퍼스에 박힌다. 같은 형식인데 854 CC / 107 O = **학습 불가능한 표면
충돌** → P 가 854/961≈0.888 로 기계적 천장. canonical schema
(`CREDIT_CARD=13~19자리 카드번호, 공백·하이픈 허용`) 기준 107 은 schema
위반 무라벨.

### 처방 — 결정론적 relabel + 측정 (부호 반전이 핵심)

무라벨 카드포맷 16자리 → CREDIT_CARD 일괄(107, 854→961, 백업·offset
결함 0·신규 overlap 0). 두 측정의 **부호 반전**이 진단을 정량 증명한다:

| 측정 | CREDIT_CARD F1 | 비고 |
|---|---|---|
| 리스코어(재학습 0) | 0.9295 → 0.9125 **(−1.70pp)** | 오도 — 옛 모델은 모순 gold 학습 |
| **보정 gold 10-fold 재학습** | 0.9295 → **0.9907 (+6.1pp)** | 진짜 값 |

(before 0.9295 = 본 실험 noextra 10-fold 실측; #69 공표값 0.9244/P 0.8922
와는 run 변동 ~0.5pp. 어느 기준이든 delta +6pp 대로 견고.)

리스코어가 내려간 건(R −7pp) 옛 모델이 *모순 gold(854 CC / 107 O)* 로
학습돼 새로 라벨된 위치를 못 맞춰서다. 이 fix 의 본질은 측정 정정이
아니라 **학습 모순 제거** — 일관 라벨로 재학습하니 모델이 "카드포맷→CC"
를 구분자까지 깨끗이 학습해 EMAIL/PHONE 처럼 포화. 경계분할 잔차는 모델
한계가 아니라 모순 학습의 산물이었다.

### 평가지표 — 보정 gold 10-fold pooled (현재 권위, CC 961)

결합 gold(ORG 5300 / PROD 1043 / CC 961), n=5,270, support 18,672.
strict = (시작,끝,종류) 완전 일치, relaxed = SemEval'13 Partial.

| entity | strict F1 | P | R | relaxed F1 | support |
|---|---:|---:|---:|---:|---:|
| **overall** | **0.9255** | 0.9128 | 0.9385 | **0.9376** | 18,672 |
| EMAIL | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1,013 |
| PHONE | 0.9912 | 0.9876 | 0.9948 | 0.9912 | 964 |
| CREDIT_CARD | **0.9907** | 0.9866 | 0.9948 | 0.9912 | 961 |
| PER | 0.9664 | 0.9540 | 0.9791 | 0.9713 | 3,726 |
| ID_NUM | 0.9613 | 0.9507 | 0.9721 | 0.9660 | 932 |
| DAT | 0.9585 | 0.9719 | 0.9454 | 0.9654 | 1,025 |
| ORG | 0.8994 | 0.8876 | 0.9115 | 0.9181 | 5,300 |
| LOC | 0.8987 | 0.8757 | 0.9230 | 0.9147 | 2,832 |
| EVT | 0.8454 | 0.8056 | 0.8893 | 0.8703 | 876 |
| PROD | 0.8009 | 0.7883 | 0.8140 | 0.8278 | 1,043 |

상위 6종(EMAIL·PHONE·CREDIT_CARD·PER·ID_NUM·DAT) 포화권(F1≥0.958),
잔여 헤드룸은 NER 4종(ORG·LOC·EVT·PROD)에 집중. ORG·PROD·EVT 는 gold
천장 확정(#71·#73·#80), LOC 미진단.

### 결론

CREDIT_CARD 의 P 천장은 ORG/PROD 식 인간 주석 모호성이 아니라 **LLM
injector 환각**(라벨 안 된 카드숫자 107 주입). 판정 census 없이 schema
기준 결정론적 relabel 로 닫혔고, 재학습 +6.1pp(CC)·+0.88pp(overall)
회복. overall 0.9167→0.9255 는 노이즈 바닥(±0.20pp)의 약 4배 = ORG/PROD
census 단발 시도(<+0.4pp)보다 큰 단일 이득.

> 근본 원인은 코드에 잔존 — gold 보정은 로컬·미커밋(gitignore)이라
> 코퍼스 재생성 시 107 재유입. injector 출력 후처리 하드닝(무라벨
> 카드/전화/ID 포맷 digit열 reject)은 별도 이슈.

### 산출물

`.../kfold10_phonediv_ccfix/`(보정 gold 10-fold)·`.../diag_prod/`(CC FP/FN
진단 재사용) — `results/`·보정 gold 모두 gitignore(로컬). 영구 재현
기준은 #69 base 0.9195(CC 854).

## 실험 — EVT 차하위 진단 + HALLUCINATION census (#80)

### EVT pooled 3-갈래 (ccfix 예측, 재추론 0)

strict TP 779 / FP 188 / FN 97 (F1 0.8454 / P 0.8056 / R 0.8893).
BOUNDARY 46·46 / HALLUC 117·MISS 34 / TYPE(EVT↔ORG) 25·17.
confusion 은 EVT↔ORG 양방향 24건 지배.

### 경계 레버 가설 반증 (핵심)

이슈 선두 가설("gold extent 비일관 = 학습 모순")을 직접 측정으로 반증.
gold 일관: `オリンピック` bare=0/mod=35·`選手権` 0/82·`大会` 0/67·
`選挙` 0/47·`戦争` 0/37(bare 혼재는 ワールドカップ·ダービー 3건뿐).
경계의 실제 정체 = ① 연도 prefix 의 DAT 경합(EVT-내부 연도 표면 16/24
가 standalone DAT, 264회:38회) ② suffix 이질 24종 = 모델 compositional
(구성성 일반화) 한계. strict↔relaxed +2.5pp 는 학습 모순 아닌 진짜 난이도.

### HALLUCINATION census — 두 방식의 대조 (#73 프로토콜)

독립 판정자 3종 gemma-4-31B + Qwen3.6-35B (후보 탐지, 교집합) + Claude
(canonical schema 판정) — 평가 BERT·gold 와 무관:

| census | 탐색 | 적용 gap | EVT F1 | Δ vs 0.8454 | EVT P | EVT R |
|---|---|---:|---:|---:|---:|---:|
| model-FP (편향) | 모델 FP 117 → ACCEPT 17 | 17 | 0.8559 | +1.05pp | 0.8175 | 0.8981 |
| **model-neutral (전수)** | 5,270 전수 → ACCEPT 92 | **92** | **0.8495** | **+0.41pp** | 0.8208 | 0.8802 |

model-FP: 117 중 둘 다 EVT 21 / 둘 다 아님 72(62% = 진짜 leak). accept
가 모델이 이미 맞춘 자리라 FP→TP 직접 전환 = 상한. model-neutral: 전수
GAP 124 → schema ACCEPT 92(reject 7=질병·법령·그룹·정치구상), gold
876→968, 모델 무관 대칭 탐색 = 무편향.

### 결론 — EVT = gold 천장 (model-neutral 확정)

model-FP +1.05pp 중 ~0.64pp 가 모델 편향. 전수 독립으로 제거 시 **+0.41pp
= 노이즈 바닥(±1.21pp) 미달**(EVT per-seed ±3.69pp 는 한참 위) → **EVT =
gold 천장 model-neutral 확정**. 서명: 무편향 보정에서 **R 0.8893→0.8802(↓)**
— 독립 탐색 gap 92 의 대부분이 모델이 못 잡아 새 FN, F1 안 오름(P 만
+1.52). #73 PROD(편향 +1.54→무편향 +0.38) 와 동일 패턴. 세 갈래 모두
천장/회색: 경계 레버 반증·HALLUC 62% leak·TYPE EVT↔ORG 미규정.

### 산출물

`.../kfold10_phonediv_ccfix/diag_evt/`(진단·model-FP·model-neutral
census·verdicts)·`.../kfold10_phonediv_evtcensus{,_neutral}/`(보정
재학습) — `results/`·무편향 보정 gold(EVT 968, 백업 `.preevtcensus`=876)
모두 gitignore(로컬). 현재 권위 overall 0.9268/EVT 0.8495. 영구 재현
기준은 #69 base 0.9195 유지.

## 실험 — LOC 최대레버 진단 + 무편향 census 재학습 (#82)

### LOC pooled 3-갈래 (ccfix 예측, 재추론 0)

strict TP 2614 / FP 371 / FN 218 (F1 0.8987 / P 0.8757 / R 0.9230).
HALLUC 247·MISS 77 / BOUNDARY 94·94 / TYPE(LOC↔ORG) 30·47. HALLUC
분해: attributive 국가 82 / 〒·주소 blob 23 / 일회성 고유 지명 142.
confusion 은 LOC↔ORG 양방향 66(gold LOC→ORG 40·역 26) 지배. BOUNDARY
는 모델 under-extent 72%(행정·주소 복합체 분할).

### attributive gold 모순 — #80 과 정반대 (핵심)

#80 EVT 는 gold extent 가 완전 일관(bare=0)이라 레버 반증. LOC 는
attributive 국가명에서 **gold 자기모순을 직접 확인**: `日本国内` gold有
3·無 3, `日本の実業家` 8·3 = 50/50; referential `日本では` 18:0 은 일관.
attributive nationality 의 본질적 모호성 = 정의 가능한 gold 모순 클래스.
canonical §2.3 에 규칙화(지명+人/語/系/製·국적 수식 명사구 = 비-LOC).

### 무편향 census → gold-fix → 재학습 (권위 실측)

model-FP(247 HALL_FP): 둘 다 LOC 64 / 모델오류 101(41%). 전수 census
GAP 180·OVER 85. 보정→10-fold 재학습 (baseline evtcensus_neutral LOC
0.8964, 바닥 이항 SE ±0.6pp — EVT ±1.21pp 의 √(876/2832) 축소):

| gold-fix | LOC F1 | P | R | Δ vs 0.8964 | 바닥 |
|---|---:|---:|---:|---:|---|
| **refonly +67 (구체 지명만)** | **0.8995** | 0.8743 | 0.9262 | **+0.31pp** | 미달 |
| maximal +129 (attributive 포함) | 0.9043 | 0.8829 | 0.9267 | +0.79pp | 위 |

### 결론 — LOC = gold 천장 (4연속, 정정 가능 결함 존재)

**방어 가능한 referential-only 보정(東京·京都·パリ 등 구체 지명 67)은
+0.31pp < 바닥 = 천장** — ORG·PROD·EVT 와 동일. 바닥 위(+0.79)는
attributive 국가명(`日本の`)을 LOC 로 인정하는 tag-all 에만 의존(maximal
초과분 +0.48pp 가 전부 attributive=모델이 이미 찍은 FP→TP). refonly 는
P 0.8810→0.8743(↓), 추가가 대부분 모델 미예측 자리라 R 만 +1.38pp.

단 **EVT(완전 무결)와 질적 차이**: GAP 67(구체 누락) + OVER 85
(attributive 과태깅 + 합성 주소 경계) = 대칭 gold 결함 존재. 천장이되
**gold 품질 정정 여지가 있는 천장**. refonly +67 채택(품질 정정, #80
EVT 968 선례와 일관). 현재 권위 overall 0.9256/LOC 0.8995(support
2899). "schema 결정 먼저"·"referential-only 변종" 이 정확히 attributive
인공물 함정을 드러냄.

### 산출물

`.../kfold10_phonediv_ccfix/diag_loc/`(진단·model-FP·전수 census·
candidates)·`.../kfold10_phonediv_loccensus_{neutral,refonly}/`(보정
재학습) — `results/`·보정 gold(LOC 2899, 백업 `.preloccensus`=2832) 모두
gitignore(로컬). 영구 재현 기준은 #69 base 0.9195 유지.
