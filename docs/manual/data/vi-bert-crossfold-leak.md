# 베트남어 BERT 벤치마크 cross-fold 데이터 누출 — 발견·진단·수정

WikiANN-vi 기반 BERT 분류기 벤치마크에서 **원문 cross-fold 누출**(같은
원문 문장이 train·test fold 에 동시에 걸쳐 모델이 일반화 대신 암기로 점수를
올리는 현상)을 발견·정량·영구 차단한 과정의 방법론 메뉴얼이다. "왜 흔한
dedup 으로는 못 막았나"와 "어떤 측정 설계로 누출 효과만 분리했나"가 핵심이다.

- 근본 코드: `src/ner/classifier/data_utils.py`
  (`split_kfold_stratified(group_key=)`),
  `src/ner/classifier/kfold_pool.py` (원문 단위 가드),
  `src/ner/classifier/__main__.py` (`--group-key`)
- 운영 가이드: `src/ner/classifier/AGENTS.md` §누출-free 분할
- 영구 아카이브: `docs/issues/issue-124-vi-ner-crossfold-leak.md`
- 정량 출처: `docs/reports/vietnamese-bert-classifier-{benchmark,history}.md`
- 관련 스키마: `docs/manual/data/canonical-entity-schema.md`

용어
- **cross-fold 누출(data leakage)**: 평가용 test fold 의 정보가 학습 데이터에
  새어 들어가, 측정 점수가 실제 일반화 성능보다 낙관적으로 부풀려지는 결함.
- **group K-fold**: 같은 그룹(여기선 같은 원문) 파생 샘플을 통째로 한 fold 에
  배정해 그룹이 분할선을 넘지 못하게 하는 교차검증 분할 방식.
- **pooled micro span F1**: fold 별 F1 평균이 아니라, 전체 fold 의 test 문장을
  하나의 코퍼스로 합쳐 char-offset span 단위로 계산한 micro-average F1.

## 한 눈에 (TL;DR)

| 축 | 무엇 |
|---|---|
| **발견** | 자연 코퍼스(주입 전 원문 `orig` 필드 보유)로 실제 split 을 재현 → test 행의 **30.2%**, NER 엔티티의 **27.7%** 가 원문이 train/valid 에도 존재 |
| **진단** | 근본 원인 = 분할이 **행 단위 round-robin** 이라 같은 원문 파생 행들이 fold 로 흩어짐. #103 의 full-text dedup 은 행마다 다른 주입 PII 가 원문 중복을 **가려서** 무력했다 |
| **수정** | 원문 단위 **group-kfold**(`--group-key orig`) + `kfold_pool` 원문 가드 + pytest 의 **3중 차단**. 코퍼스 고정·분할만 교체해 누출 효과만 분리 측정 |
| **영향** | NER 5종 절대 F1 이 **+1.1~1.4pp**(EVT/ORG/PROD **+3~5pp**) 부풀려져 있었음. PII 5종은 불변(누출 0). #103/#108/#112 과거 측정의 NER 절대값을 인플레로 정정 |

## 1. 무대 — 왜 이 구조에서 누출이 생기는가

합성 PII 주입(`python -m ner.augmenters.pii`)은 **한 원문에서 여러 행을
파생**시킨다 — 같은 문장에 행마다 다른 개인정보(이메일·전화·ID·카드)를 넣는다.
NER 라벨은 원문에서 그대로 오므로 파생 행들은 NER span 을 공유한다.

```
원본 WikiANN-vi 원문  S = "Hà Nội là thủ đô của Việt Nam."   (NER: Hà Nội=LOC, Việt Nam=LOC)
        │  PII 주입 — 행마다 다른 값
        ├─ 행1: S + "Email: an@x.vn"        ← 전체 텍스트 유니크
        ├─ 행2: S + "SĐT: 0901 234 567"     ← 전체 텍스트 유니크
        └─ 행3: S + "CCCD: 0790 1234 5678"  ← 전체 텍스트 유니크
```

여기에 누출은 두 겹으로 쌓여 있었다.

- **(A) WikiANN-vi 내재 중복**: raw `unimelb-nlp/wikiann`(vi) 자체에
  test∩train = 1,778 문장, train 내부 중복 19.7% 가 박혀 있다.
- **(B) 원문 cross-fold 흩어짐**: PII 주입으로 파생된 행들이 분할 단계에서
  서로 다른 fold 로 흩어진다. 이게 #124 가 잡은 **진짜 잔존 누출**이다.

## 2. 발견 — 어떻게 드러났나

### 2.1 1차 누출과 "거짓 안심" (#103)

#103 은 (A)를 인지하고 **원문 키 dedup**(주입 PII 를 뺀 원문이 같은 행을
하나만 남김)으로 40,000 → **38,371 행**(−1,629)을 만들고, "어떤 split·fold
조합에서도 cross-split 중복 0"으로 재검증했다. 실증으로 CafeBERT 가 leaky
single-split 0.9504 → clean fold0 0.9259 로 떨어지는 것까지 확인했다.

그러나 이 "cross-split 중복 0"은 **거짓 안심**이었다. 주입 PII 가 원문 중복을
가렸기 때문이다:

```
full-text dedup 의 맹점
  행1/행2/행3 은 "서로 다른 문자열"(꼬리 PII 가 달라서)
        │
        ▼
  dedup 은 1개도 안 지운다  →  원문 S 의 중복이 그대로 잔존
  'cross-split 중복 0' 재검증도 행마다 유니크한 *주입 텍스트* 기준
        │
        ▼
  같은 원문 S 가 fold0(train)·fold3(test) 로 흩어짐  =  cross-fold 누출
  모델이 train 에서 S 의 NER 라벨을 암기 → test 에서 재현 → F1 부풀림
```

핵심 수치: 원본 WikiANN-vi(40,000 행)의 **유니크 원문은 29,343 개뿐**인데
#103 dedup 은 1,629 행만 제거 → 38,371 행에 원문 중복 ~9,000 행이 그대로
남았다. dedup 이 *주입 후* 전체 텍스트 기준으로 동작했고, 재검증도 같은
기준이라 **원문 단위 누출이 통과**됐다.

### 2.2 진짜 누출 정량 (#124, #116 에서 스핀오프)

#116(VI PII 자연 주입)에서 만든 자연 코퍼스 `pii_all.jsonl`(37,706 행)은
행마다 **주입 전 원문을 담은 `orig` 필드**를 보유한다. 이 `orig` 를 기준으로
실제 split 을 재현하니 행 단위 분할에서:

- test 행의 **30.2%(11,392)** 가 원문이 train/valid 에도 존재
- NER 엔티티의 **27.7%(15,229/54,895)** 가 동일 누출 영향권
- PII 는 행별 랜덤값이라 train·test 값이 겹치지 않음 → **누출 0**

## 3. 진단 — 근본 원인과 측정 설계

### 3.1 근본 원인

분할 함수 `split_kfold_stratified` 가 **행 단위 라운드로빈**이었다. 한 원문에서
파생된 여러 행이 fold 로 흩어지므로, 어떤 reshape(#108·#112 re-silver, #116 PII
재주입)을 거쳐도 잠복 누출이 **새 형태로 재발**했다. dedup 은 증상(겉보기 중복)을
지웠을 뿐 분할 단위(원인)를 못 고쳤다.

```
원본 40,000 행 ── 유니크 원문 29,343 개  (행:원문 ≈ 1.36 : 1)
        │
   #103 full-text dedup (−1,629)
        ▼
   38,371 행 ── 원문 중복 ~9,000 행 잔존  ←★ 여기가 누출의 씨앗
```

### 3.2 측정 설계 — 교란 제거

원안(goal 1: "원문 dedup 29,343 재학습 vs 누출 baseline 37,706")은 변수
**2개**(누출 제거 + 학습데이터 22% 손실)를 동시에 바꿔, 인플레 폭을 누출
단독에 귀속할 수 없는 교란(confound)이 있다. 그래서 통제 설계로 바꿨다:

- **코퍼스를 37,706 행으로 고정** → 데이터량 변수 제거
- **분할만** leaked(행 단위) ↔ grouped(원문 group-kfold) 로 교체
  → 누출 효과만 단독 분리
- 둘 다 전수 1회 test, `n_sentences=37,706` 동일

```
[leaked] 행 단위 round-robin (group_key=None)
  S 파생 행들이 fold 경계를 넘어 흩어진다
      행1→fold0(train)   행2→fold2(train)   행3→fold3(TEST)
      └──────── 같은 원문 S 가 train·test 양쪽 ────────┘     ❌ 누출

[grouped] 원문 단위 group-kfold (group_key='orig')
  S 파생 행 전체가 한 unit 으로 묶여 통째로 한 fold 로
      {행1, 행2, 행3} ─────────────────────→ fold3 (TEST 전용)
      train 에는 원문 S 가 전혀 없음                           ✅ 누출-free
```

## 4. 수정 — group-kfold 3중 차단

### 4.1 ① split 단계 — 구조적 차단

`data_utils.split_kfold_stratified(group_key=)` — 같은 `row[group_key]`
(예: `orig`) 값을 공유하는 행을 한 **unit** 으로 묶어 통째로 한 fold 에
배정한다. 층화(PROD/EVT 보유 여부)·라운드로빈·결정성은 행 단위와 동일하되,
층화 기준은 unit 안 strat_labels 의 **합집합**이다. 분류기 CLI `--group-key`
로 노출하고 `test_predictions.json` 에 `orig` 를 기록한다.

> **백워드 호환(BC)**: `group_key=None`(기본)이면 unit = 행 1개라 기존 행 단위
> 분할과 **bit-for-bit 동일**(같은 seed → 동일 결과). 누출-free 는 opt-in.

### 4.2 ② pooling 단계 — 사후 검증 가드

`kfold_pool.pool_fold_predictions(require_no_leak=True)` — fold 별
`test_predictions.json` 의 `orig` 를 본다. 같은 fold 안 원문 중복은 group
정상이라 허용하고, **fold 가 갈리면 누출**로 본다(`cross_fold_orig_dups++`
→ `ValueError`). 누출 baseline 을 일부러 잴 때만 `--allow-cross-fold-leak`
로 카운트만 받는다. `orig` 없는 레거시 코퍼스는 `text` 전체 문장 중복으로
fallback(`id` 는 비고유라 검증 기준 미사용).

### 4.3 ③ pytest — 회귀 차단

`test_data_utils.py`(group 분할 cross-fold 원문중복 0·그룹 불가분·전수 1회
test·`group_key=None` 동일성 5개) + `test_kfold_pool.py`(cross-fold raise·
카운트 3개).

```
① split 단계      split_kfold_stratified(group_key='orig')
  (구조적 차단)    같은 orig 행을 한 unit 으로 묶어 한 fold 배정 → 누출 불가능
        │
        ▼
② pooling 단계    kfold_pool.pool_fold_predictions(require_no_leak=True)
  (사후 검증)      test_predictions.json 의 orig 가 두 fold 에 걸치면 ValueError
        │
        ▼
③ pytest (CI)     test_data_utils(5) · test_kfold_pool(3)
  (회귀 차단)      cross-fold 원문중복 0 강제 + group_key=None bit-for-bit 동일성
```

## 5. 정량 결과 (영구 인용)

자연 코퍼스 37,706 행 **고정**, 분할만 leaked(행 단위 stratified 5-fold) ↔
grouped(원문 group-kfold)로 변경. 둘 다 전수 1회 test, pooled micro
char-offset span F1.

| 모델 | overall | NER 5종 micro | PII 5종 micro |
|---|---:|---:|---:|
| phobert leaked → grouped | 0.9506 → 0.9437 (**+0.69**) | 0.9198 → 0.9086 (**+1.12**) | 0.9985 → 0.9983 (+0.02) |
| xlm-r leaked → grouped | 0.9517 → 0.9430 (**+0.87**) | 0.9226 → 0.9083 (**+1.43**) | 0.9972 → 0.9974 (−0.02) |

per-entity strict F1 (leaked → grouped, 인플레 pp):

| ent | phobert (Δpp) | xlm-r (Δpp) | support |
|---|---|---|---:|
| EVT | 0.8372 → 0.7973 (**+3.98**) | 0.8242 → 0.7713 (**+5.29**) | 559 |
| ORG | 0.8882 → 0.8647 (**+2.35**) | 0.8901 → 0.8557 (**+3.44**) | 8,033 |
| PROD | 0.7362 → 0.7076 (**+2.86**) | 0.7116 → 0.6811 (**+3.05**) | 2,314 |
| PER | 0.9235 → 0.9145 (+0.90) | 0.9382 → 0.9254 (+1.28) | 21,848 |
| LOC | 0.9506 → 0.9443 (+0.63) | 0.9457 → 0.9394 (+0.63) | 22,141 |
| PII 5종 | 0.998~1.000 (±0.05) | 0.997~1.000 (±0.06) | ~35,500 |

해석
- **NER 5종 절대값 인플레 확정**: micro +1.1~1.4pp, 저support·고난도 엔티티
  (EVT/ORG/PROD)에 집중. 암기는 희소·어려운 엔티티에서 가장 크게 작동한다.
- **PII 5종 불변(±0.05pp)** — 누출 0 주장 검증(행별 랜덤값이라 train·test 값
  비중복).
- **가드 실측**: grouped `cross_fold_orig_dups=0`, leaked=6,973
  (`kfold_pool` 가 직접 카운트).
- **상대 비교·델타는 유효**: phobert ≈ xlm-r, #108/#112 re-silver 이득은 같은
  누출이 양쪽에 동일 적재돼 보존된다.

출처: `results/classifier/vi/leak_audit/{leaked,grouped}/<model>/fold{0..4}`
+ `pooled_metrics.json` + `leak_inflation_summary.json` (results 는 gitignore
이므로 위 리포트 표가 영구 인용 출처).

## 6. 영구 방지 + 과거 정정

- **영구 방지**: `--group-key orig` 를 VI 분류기 평가 프로토콜로 고정. split
  단계 그룹 분할 + `kfold_pool` 원문 가드 + pytest 가 어떤 reshape 에도 재발을
  구조적으로 막는다.
- **과거 정정**: `vietnamese-bert-classifier-{benchmark,history}.md` 에
  §원문 누출 정정(#124)을 추가해 #103/#108/#112 의 NER 5종 절대값을 인플레로
  표기. suffix v1/v2/v3 코퍼스 파일은 단일 `pii_all.jsonl` 정책으로 폐기돼
  정확 재측정 불가 → 자연 코퍼스 측정의 **대표 추정**(suffix v3 누출률 27.9%
  로 규모 유사). PII·phase 간 델타·상대비교는 유효.

## 7. 재현 / 운영

```bash
# 누출-free group-kfold: 같은 원문 파생 행을 한 fold 로 묶어 cross-fold 차단
for fold in 0 1 2 3 4; do
  python -m ner.classifier --lang vi \
    --data-jsonl results/.../pii_all.jsonl \
    --kfold 5 --fold-index ${fold} --group-key orig \
    --output-dir results/.../grouped/<model>/fold${fold}
done

# pooled 평가 — 원문 단위 cross-fold 누출이 있으면 ValueError 로 중단
python -m ner.classifier.kfold_pool \
  --fold-dirs results/.../grouped/<model>/fold{0,1,2,3,4} \
  --output results/.../grouped/<model>/pooled_metrics.json

# 누출 baseline 을 일부러 측정할 때만 (행 단위 분할 + 가드 카운트 모드)
#   --group-key 생략(행 단위) + kfold_pool --allow-cross-fold-leak
```

## 8. 교훈 (일반화)

- **파생 텍스트로 dedup 하면 원천 누출을 가린다.** 한 원천에서 행마다 다른
  꼬리(PII·타임스탬프·ID)를 붙여 파생시키면 전체 텍스트는 유니크해져
  full-text dedup·중복검사가 전부 통과한다 — 누출은 살아 있는데.
- **분할은 "불변 키" 단위로.** 증상(겉보기 중복)을 지우지 말고, 누출이
  발생하는 **단위 자체**(원문 그룹)를 분할 단위로 삼아야 구조적으로 막힌다.
- **측정은 변수 1개만 바꿔라.** dedup+재학습은 누출 제거와 데이터 손실을 함께
  바꿔 교란된다. 코퍼스 고정·분할만 교체해야 인플레를 누출 단독에 귀속할 수
  있다.
- **이중 가드 + 회귀 테스트.** split 단계 차단(예방)과 pooling 단계 검증(탐지)은
  직교한다. 둘 다 두고 pytest 로 박아야 reshape 마다 재발하지 않는다.
