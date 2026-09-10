# 영어 BERT NER 분류기 백본 벤치마크 (canonical 10종 평면)

- 측정일: 2026-08-21
- 대상: OntoNotes5 유래 canonical 10종 평면 데이터에 인코더 백본 파인튜닝 —
  **baseline 수치 확립**(임계값 게이트 아님)
- 데이터: `data/ontonotes_en/pii_all.jsonl` (76,378 행, canonical 10종 = NER 5 + PII 5)
- 평가: char-offset span F1 (strict), **단일 split**(test 0.1 / valid 0.1, `seed=42`,
  `--group-key orig`)
- 순위 판정: 후보마다 **seed 3개(42·43·44) median**
- 출처: GitHub 이슈 #215
- 한국어·일본어·베트남어 동일 셋업:
  `docs/reports/{korean,japanese,vietnamese}-bert-classifier-benchmark.md`

## 이 표의 수치가 속한 gold 세대

**바로 아래 §현행 gold 의 `roberta-base` baseline 을 뺀 이 리포트의 모든 수치는
gold 세대 1 로 잰 것이며 현행 코퍼스로는 재현되지 않는다.** 측정 뒤 정답이 두 번
바뀌었다.

| 언제 | 무엇이 바뀌었나 | 이 표에 미치는 영향 |
|---|---|---|
| 2026-08-26 (#225) | 엔티티 밖 문장부호를 엔티티 안으로 라벨하던 결함을 없앴다 (전수 80,288건) | 같은 예측이 다르게 채점된다. `dataset_fingerprint` 는 안 바뀌므로 비교 유효성 게이트가 이 경계를 못 잡는다 |
| 2026-08-31 (#236·#239) | 원본 `FAC` 를 전량 `ORG` 로 태우던 것을 표면별로 `ORG`·`LOC`·삭제로 갈랐다 | `LOC` +274 · `ORG` −289 span. `dataset_fingerprint` 가 `39cb0f9e9c16f265` → `258166d6972c8144` 로 바뀌어 새 런과의 비교는 자동 거부된다 |

데이터 경로도 그 사이 `pii_all.jsonl` 에서 `origin.jsonl` 로 개명됐다(#238).
아래 본문은 측정 당시 표기를 그대로 둔다 — 조건을 사후에 고쳐 적으면 이 표가
어느 조건에서 나왔는지가 흐려지기 때문이다.

현행 코퍼스의 지문은 `258166d6972c8144` 이고, 파일별 sha256 과 `FAC` 재라벨로
손댄 span 은 `src/ner/augmenters/ontonotes_en/data/fac_disk_migration_ledger.json`
에 있다. 그 원장과 디스크의 `origin.jsonl` 이 어긋나지 않는지는
`tests/ner/augmenters/ontonotes_en/test_fac_disk_migration_ledger.py` 가 본다 —
원장이 함께 적는 주입 split 의 지문은 그 파일들이 디스크에서 내려가 대조할
상대가 없다.

## 현행 gold 의 `roberta-base` baseline

아래 백본 비교는 gold 세대 1 로 잰 것이라 현행 코퍼스에서 그대로 쓸 수 없다.
그래서 #244 가 **`roberta-base` 한 종만** 세대 3 gold 로 같은 레시피(epochs 5 ·
lr 5e-5 · max_length 256 · bf16 · batch 16 · `--group-key orig` · 분할 seed 42)
로 다시 3-seed 돌려 현행 baseline 을 세웠다. 세 run 모두
`data_fingerprint = 258166d6972c8144` 다.


| 백본 | seed | **NER-5 median** | std | min | max | macro | (참고)PII5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `roberta-base` | 42·43·44 | **0.8956** | 0.0077 | 0.8884 | 0.9039 | 0.8089 | 0.9476 |


| 타입 | PER | LOC | ORG | PROD | EVT |
|---|---:|---:|---:|---:|---:|
| median strict F1 | 0.9486 | 0.9145 | 0.8574 | 0.5794 | 0.7615 |

**이 값과 아래 세대 1 표를 대소로 읽지 않는다.** gold 가 두 번 바뀌어 재는 자가
다르므로, 두 수치는 같은 축 위에 있지 않다. 세대 3 에서 백본 순위가 어떻게
되는지는 #244 가 `roberta-base` 만 재서 측정하지 않았고, `roberta-base` 가
현행 gold 에서도 최선이라는 것은 증거 없는 승계 가정으로 남아 있다.

천장이 여전히 **PROD·EVT** 에 있다는 그림은 세대 1 과 같다.

## 요약

- **baseline = `roberta-base`** — NER-5 strict micro-F1 median **0.8795**,
  macro **0.8106**. 채택을 정한 것은 macro 와 저빈도 타입이고, 근거 사슬은
  §채택 결론에 있다.
- **micro 단독으로는 1위를 못 가른다.** 상위 세 후보가 0.23pp 안에 몰려
  있는데 같은 후보의 seed 표준편차가 0.40~0.51pp 다. 순위 간격보다 한 후보
  안의 흔들림이 크므로, micro 만 보고 우열을 말할 수 없다.
- **단일 seed 였다면 다른 결론을 적었다.** seed 42 만 보면 순위가
  `bert-base-cased` > `xlm-roberta-base` > `ModernBERT-base` > `roberta-base`
  인데, median 으로는 `xlm-roberta-base` 가 최하위로 내려가고 `roberta-base`
  가 올라온다. ko·ja·vi 벤치는 단일 seed 였으므로 같은 함정이 남아 있다.
- **다국어의 페널티가 PER 한 곳에 몰린다.** ko 에서 `xlm-roberta-base` 는
  NER-5 다섯 항목 **전부** 단일언어에 졌는데, en 에서는 PER 만 크게 지고
  (0.920, 최하위) LOC 은 2위(0.888)다. 영어가 다국어 모델 사전학습의 큰 축이라
  용량 페널티가 전면적이지 않다.
- **`microsoft/deberta-v3-base` 제외** — 표준 레시피 학습 불가(아래).
  ko·vi 에 이어 **세 번째 언어**에서 같은 실패다.
- 천장은 **PROD(~0.64)·EVT(~0.75)** — 저빈도(test PROD 212 · EVT 107 span)라
  백본이 아니라 데이터·증강 레버리지다. 게이트 수치는 미설정(baseline 측정 목적).

## 조건

| 항목 | 값 |
|---|---|
| 데이터 | `data/ontonotes_en/pii_all.jsonl` 76,378 행 (OntoNotes5 유래 + 합성 PII 주입) |
| 분할 | **단일 split** — valid 0.1 / test 0.1, `seed=42`, `--group-key orig` |
| 라벨 | `PER, LOC, ORG, PROD, EVT, DAT, EMAIL, PHONE, ID_NUM, CREDIT_CARD` |
| 학습 | epochs=5, lr=5e-5, max_length=256, **bf16**, batch 16, class-weight·curriculum 미사용 |
| 모델 선택 | `metric_for_best_model='eval_loss'` (valid) |
| seed | `--train-seed 42·43·44` (분할 seed 는 42 고정) |
| 하드웨어 | NVIDIA RTX A6000 49GB |
| 데이터 지문 | `39cb0f9e9c16f265` (전 run 동일) |

**모델별 하이퍼파라미터 sweep 은 하지 않았다.** 이 벤치가 재는 것은 백본이지
튜닝 솜씨가 아니라, 한 후보만 조건을 손봐주면 비교가 깨진다. ko·vi 벤치도 같은
원칙을 썼다.

### 공식 3분할을 쓰지 않은 이유

OntoNotes5 는 train/valid/test 가 이미 갈려 있고 #209 산출물도 split 별 파일로
나온다. 그런데도 세 파일을 합쳐 재분할한 것은 **ko #122 와 조건을 맞추기
위해서**다. 공식 분할을 그대로 받으려면 분할 코드(`data_utils.py`)에 새 경로를
뚫어야 하는데, 그 파일은 비교 유효성의 자라 함부로 바꾸지 않는다.

대가로 **외부 논문·리더보드의 OntoNotes 수치와 직접 비교할 수 없다.** 다만 이
대가는 분할 때문만은 아니다 — 18종을 canonical 6종으로 매핑한 시점에 라벨
공간이 이미 달라져 published OntoNotes NER F1 과 나란히 놓을 수 없다.

### 병합이 `split` 필드를 떨어뜨리는 이유

세 파일을 그대로 이어붙이면 `split` 이 한 파일 안에서 세 값을 갖는다.
`validate_group_key` 는 값이 적으면서 선언한 키의 그룹을 가르지 않는 필드를
"더 강한 그룹 키"로 보고 거부하는데, 같은 `orig` 의 행은 언제나 같은 split 에
있어(경계를 하나도 가르지 않는다) 정확히 그 모양이 된다. 그대로 두면
`--group-key orig` 가 통째로 막힌다.

재분할이 전제라 원 split 은 학습에 쓰이지 않으며, 어느 split 이었는지는
`id`·`orig` 접두사(`en-train-`/`en-valid-`/`en-test-`)에 남아 정보도 잃지 않는다.

## 비교 기준 — canonical NER-5 (PII 제외)

PII 5종(`DAT`·`EMAIL`·`PHONE`·`ID_NUM`·`CREDIT_CARD`)은 **합성 주입값**이라
패턴이 인위적으로 깨끗하다 — EMAIL/PHONE/ID_NUM/CREDIT_CARD 는 F1 0.94~0.98,
DAT 만 ~0.85(날짜 표현 다양성). 백본 변별력이 작아 **선정 기준에서 제외**하고
참고로만 보고한다. 백본 비교·baseline 선정은
**NER-5(PER/LOC/ORG/PROD/EVT)** 로 한다.

다만 en 에서 이 전제의 반례를 하나 얻었다 — §관측된 이상.

## NER-5 결과 (3-seed median, strict)

| 모델 | 계열 | 파라미터 | **median** | std | min | max | macro | (참고)PII5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`roberta-base`** | mono | 124M | **0.8795** | 0.0051 | 0.8734 | 0.8836 | **0.8106** | 0.9462 |
| `bert-base-cased` | mono | 108M | 0.8791 | **0.0040** | 0.8751 | 0.8831 | 0.7899 | 0.9405 |
| `answerdotai/ModernBERT-base` | mono | 150M | 0.8772 | 0.0048 | 0.8692 | 0.8778 | 0.8001 | 0.9370 |
| `google/electra-base-discriminator` | mono | 109M | 0.8656 | 0.0109 | 0.8475 | 0.8673 | 0.7718 | 0.9475 |
| `xlm-roberta-base` | **multi** | 278M | 0.8610 | 0.0129 | 0.8577 | 0.8815 | 0.7835 | 0.9386 |
| ~~`microsoft/deberta-v3-base`~~ | mono | 184M | — | — | — | — | **학습불가** | — |

### per-entity strict F1 (NER-5, median)

| 모델 | PER | LOC | ORG | PROD | EVT |
|---|---:|---:|---:|---:|---:|
| **`roberta-base`** | 0.947 | 0.882 | 0.837 | **0.640** | 0.743 |
| `bert-base-cased` | 0.944 | **0.893** | **0.843** | 0.562 | 0.723 |
| `ModernBERT-base` | 0.943 | 0.881 | 0.823 | 0.607 | **0.752** |
| `electra-base` | **0.946** | 0.876 | 0.806 | 0.513 | 0.702 |
| `xlm-roberta-base` (multi) | 0.920 | 0.888 | 0.817 | 0.560 | 0.742 |

**PROD 가 순위를 가른다.** 1위와 최하위의 차가 12.7pp(0.640 vs 0.513)로 다섯
타입 중 가장 크다. PER 은 2.7pp 안에 다섯 후보가 모여 변별력이 없다.

### baseline `roberta-base` 전체 per-entity P/R/F1 (10종)

median seed(44)의 값이다. overall-10 F1 0.9221 (P 0.9049 / R 0.9399).

| Entity | support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| PER | 2,071 | 0.9474 | 0.9474 | 0.9474 |
| LOC | 2,262 | 0.8737 | 0.8899 | 0.8817 |
| ORG | 1,836 | 0.8225 | 0.8431 | 0.8327 |
| PROD | 212 | 0.7289 | 0.5708 | 0.6402 |
| EVT | 107 | 0.7547 | 0.7477 | 0.7512 |
| DAT | 1,408 | 0.8115 | 0.8991 | 0.8531 |
| EMAIL | 2,385 | 0.9695 | 0.9987 | 0.9839 |
| PHONE | 2,253 | 0.9331 | 0.9902 | 0.9608 |
| ID_NUM | 2,266 | 0.9178 | 0.9801 | 0.9479 |
| CREDIT_CARD | 2,292 | 0.9353 | 0.9773 | 0.9558 |

**PROD 는 recall 이 문제다** — precision 0.729 인데 recall 0.571 로, 찾은 것은
대체로 맞지만 절반 가까이 놓친다. OntoNotes 의 `PRODUCT` 와 `WORK_OF_ART` 를
한 칸에 담은 타입이라 표면형이 넓고 test 가 212 span 뿐이다.

## seed 하나로는 순위를 못 정한다

### 무엇이 흔들리나

| 모델 | seed 42 | seed 43 | seed 44 | median | 폭 |
|---|---:|---:|---:|---:|---:|
| `roberta-base` | 0.8734 | 0.8836 | 0.8795 | 0.8795 | 1.02pp |
| `bert-base-cased` | 0.8831 | 0.8791 | 0.8751 | 0.8791 | 0.80pp |
| `ModernBERT-base` | 0.8778 | 0.8772 | 0.8692 | 0.8772 | 0.86pp |
| `electra-base` | 0.8656 | 0.8673 | 0.8475 | 0.8656 | 1.98pp |
| `xlm-roberta-base` | 0.8815 | 0.8610 | 0.8577 | 0.8610 | **2.38pp** |

**같은 후보 안의 폭이 후보 간 간격보다 크다.** median 1위와 3위의 차는
0.23pp 인데, `xlm-roberta-base` 한 후보의 폭만 2.38pp 다.

### 결론이 실제로 뒤집힌다

seed 42 만 쓴 표와 median 표의 순위를 나란히 놓으면:

| 순위 | seed 42 단독 | 3-seed median |
|---|---|---|
| 1 | `bert-base-cased` (0.8831) | `roberta-base` (0.8795) |
| 2 | `xlm-roberta-base` (0.8815) | `bert-base-cased` (0.8791) |
| 3 | `ModernBERT-base` (0.8778) | `ModernBERT-base` (0.8772) |
| 4 | `roberta-base` (0.8734) | `electra-base` (0.8656) |
| 5 | `electra-base` (0.8656) | `xlm-roberta-base` (0.8610) |

1위와 4위가 자리를 바꾸고, 2위가 최하위로 내려간다. **단일 seed 벤치는
백본의 실력이 아니라 추첨 결과를 적을 수 있다.**

이 설계는 ja 리포트의 자기비판에서 나왔다 — 출하 체크포인트의 PROD R 0.9362
는 5-seed 분산 [0.857, 0.937](std 0.033) 안의 단일 draw 였다. ko·ja·vi 백본
벤치는 모두 단일 seed 였으므로 같은 함정이 남아 있다.

### 왜 평균이 아니라 median 인가

`xlm-roberta-base` 의 seed 42 값(0.8815)은 다른 두 seed(0.8610·0.8577)와 2pp
넘게 떨어져 있다. 평균을 쓰면 이 한 값이 0.8667 까지 끌어올려 3위권으로
보이지만, median 은 0.8610 으로 그 draw 를 순위에 반영하지 않는다.

## 다국어 페널티가 PER 한 곳에 몰린다

ko #122 에서 `xlm-roberta-base` 는 NER-5 **다섯 항목 전부** 단일언어 3종에
졌다(−7~8pt). en 에서는 양상이 다르다.

| 타입 | xlm-r | 단일언어 최고 | 차이 | xlm-r 순위 |
|---|---:|---:|---:|---|
| PER | 0.920 | 0.947 | −2.7pp | **5위(최하위)** |
| LOC | 0.888 | 0.893 | −0.5pp | 2위 |
| ORG | 0.817 | 0.843 | −2.6pp | 4위 |
| PROD | 0.560 | 0.640 | −8.0pp | 3위 |
| EVT | 0.742 | 0.752 | −1.0pp | 3위 |

**LOC 은 사실상 동률이고 PER 만 명확히 뒤진다.** 영어가 다국어 모델 사전학습
코퍼스의 큰 축이라 ko 처럼 전면적인 용량 페널티(curse of multilinguality)를
치르지 않는다. 그럼에도 median 최하위인 것은 절대 성능보다 **seed 안정성**
때문이다 — 폭 2.38pp 로 후보 중 가장 크다.

즉 en 에서 다국어 백본을 피할 근거는 "성능이 낮아서"가 아니라 "같은 조건에서
결과가 가장 많이 흔들려서"다.

## `microsoft/deberta-v3-base` 제외 (표준 레시피 학습 불가)

```
step  500 : loss 288.7, grad_norm nan
step 1000~: loss 0,     grad_norm nan   (5 epoch 내내)
eval      : eval_loss nan → 전 타입 F1 0 (예측이 전부 O)
```

**`loss: 0` 은 학습이 잘 된 것이 아니다** — 가중치가 NaN 으로 오염된 뒤의
부산물이라, 진행 표시줄만 보면 정상으로 읽힌다. 500 스텝 안에 발산했고 나머지
5 epoch(약 84분)은 그대로 낭비됐다.

토크나이저는 드롭인이라(`is_fast=True`, offset 정상) 정렬 문제가 아니다.
모델별 sweep(warmup·정밀도 분리 등)은 본 벤치의 범위("백본만 비교") 밖이라
제외한다.

### 세 언어에서 같은 실패 — 정밀도 변수가 좁혀진다

| 벤치 | 정밀도 | LR | 결과 |
|---|---|---|---|
| ko #122 | bf16 | 5e-5 | 첫 스텝부터 loss 폭증(240→337) + grad_norm NaN → 발산 |
| ko #122 | bf16 | 1e-5 | 발산은 멈추나 eval all-O 붕괴, F1 0 |
| vi | bf16 | 5e-5 / 1e-5 / 2e-5 | 발산 또는 all-O 붕괴 (`videberta` 도 동일) |
| **en (본 벤치)** | **bf16** | **5e-5** | **발산 → NaN 고착 → F1 0** |

ko·vi 리포트는 한계로 "테스트한 설정이 모두 bf16 이었고 정밀도(fp16/fp32)
변수는 미분리"를 적었다. en 도 bf16 확정 조건이므로 **세 언어 모두 bf16 에서
실패**한 것이 되고, fp32 는 여전히 아무도 시도하지 않았다. 정밀도가 원인인지
아닌지는 이 벤치들로 갈리지 않는다.

## 관측된 이상 — 합성 PII 도 seed 를 탄다

`roberta-base` seed 43 에서 EMAIL 만 F1 0.731 로 무너졌다.

| seed | EMAIL P | EMAIL R | EMAIL F1 |
|---|---:|---:|---:|
| 42 | 0.9715 | 0.9996 | 0.9853 |
| **43** | **0.6401** | 0.8507 | **0.7305** |
| 44 | 0.9695 | 0.9987 | 0.9839 |

recall 0.851 은 정상 범위인데 precision 이 0.640 이라 **과잉 예측** 형태다 —
예측 수가 gold 2,385 보다 약 790 개 많다. 같은 run 의 NER-5 타입은 모두 정상
범위였고(오히려 그 후보의 최고 micro), EMAIL 은 선정 기준 밖이라 순위에는
영향이 없다.

ko·ja·vi 리포트는 "합성 PII 는 패턴이 깨끗해 포화하므로 변별력이 없다"고
적었는데, 이 사례는 **포화가 seed 마다 보장되지는 않는다**는 것을 보인다.
PII 를 선정 기준에서 빼는 판단 자체는 유지한다 — 한 seed 의 국소 붕괴는 백본
우열을 말해주지 않기 때문이다. 다만 "PII 는 늘 1.0 근처"라는 전제는 en 에서
반례를 얻었고, 배포 전 PII 성능을 단일 run 으로 확인하면 이런 붕괴를 놓친다.

## 산출물이 자기 로그와 맞는지 확인한다

15회 전부 **시각 정합 검사**를 통과했다(gap 50~124초).

로그의 첫 시각 + `train_time_sec` 은 `metrics.json` 이 쓰인 시각과 맞아야
한다 — 그 사이에는 평가와 체크포인트 저장만 있으므로 차이는 작은 양수여야
한다. 이 검사를 두는 이유는 **두 프로세스가 같은 출력 디렉토리를 쓰면 값만
조용히 바뀌어 눈으로는 안 보이기** 때문이다. 실제로 본 벤치 진행 중 같은
워크트리에서 두 세션이 같은 명령을 각자 실행하는 사고가 있었고, 이 검사가
오염된 run 두 건을 잡아냈다(`electra_base_seed43` gap −1,846초 ·
`xlmr_base_seed44` 중복 이력). 두 건은 폐기하고 다시 돌린 값이 위 표에 실려
있다.

구현은 `ner.classifier.backbone_summary --check` 였다. **이 도구는 이후
제거됐다** — 백본 비교가 `roberta-base` 채택으로 끝나 상시 유지할 값이
없었다. 아래 재현 절차의 집계 단계도 그래서 지금은 돌지 않는다.

## 학습시간 — 1 epoch GPU 독점 재측

| 후보 | 파라미터 | 1 epoch 학습시간 | 초당 샘플 |
|---|---|---|---|
| `bert-base-cased` | 108M | 252초 | 244 |
| `roberta-base` | 125M | 254초 | 242 |
| `google/electra-base-discriminator` | 110M | 254초 | 242 |
| `xlm-roberta-base` | 278M | 290초 | 212 |
| `answerdotai/ModernBERT-base` | 149M | 437초 | 140 |

측정 조건은 위 §조건과 같고 `--epochs 1` 만 다르다. GPU 한 장(A6000 0번)을
한 프로세스가 독점한 상태에서 다섯 후보를 **하나씩 차례로** 돌렸고, 값은
`metrics.json` 의 `train_time_sec` — 학습 루프만이며 토크나이즈·평가·저장은
빠져 있다. `microsoft/deberta-v3-base` 는 학습이 안 돼 제외했다. 이 실행들의
F1 은 1 epoch 짜리라 순위 판정에 쓰지 않았고 어디에도 인용하지 않는다.

**병렬 wall-clock 은 시간을 재는 데 쓸 수 없다.** 폐기한 병렬 실행에서는
`roberta-base` 가 `bert-base-cased` 의 2.1배로 보였는데, 독점 재측에서는 252초
대 254초로 차이가 1% 안이다. 앞의 배수는 모델의 성질이 아니라 그때 GPU 를
누가 함께 쓰고 있었는지의 기록이었다.

**파라미터 수로 학습시간을 갈음할 수 없다.** 가장 큰 `xlm-roberta-base`(278M)가
가장 느리지 않고, 그 절반인 `ModernBERT-base`(149M)가 가장 느리다. xlm-r 의
여분 파라미터는 대부분 25만 어휘의 임베딩 표에 있는데, 임베딩은 층 계산이
아니라 행 조회라 파라미터가 늘어도 스텝 시간이 그만큼 늘지 않는다.
`ModernBERT-base` 가 느린 원인은 이 실행에서 확인하지 않았다 — 로그에 attention
백엔드 관련 경고는 남지 않았다.

**서빙 비용의 대리 지표로만 읽는다.** 학습 한 스텝은 순전파와 역전파를 함께
돌지만 추론은 순전파만 하므로, 이 표의 배수가 추론 지연에 그대로 옮겨가지
않는다. baseline 선정은 이 표를 쓰지 않았다 — 갈림은 macro 와 PROD 에서 났다.

## 채택 결론

**세대 1 백본 비교가 `roberta-base` 를 고른 근거는 macro F1 과 PROD F1 둘이다.**
micro 와 seed 안정성은 근거가 아니다. 아래 값은 모두 이 문서의 세대 1 표에서
옮긴 것이고, 그 표의 원장은 #244 가 지워 대조 상대가 없다(§산출물 위치).

| 지표 | `roberta-base` | 2위 | 채택 근거 |
|---|---:|---:|---|
| micro median | 0.8795 | 0.8791 (`bert-base-cased`) | ✕ 후보를 못 가른다 |
| **macro median** | **0.8106** | 0.8001 (`ModernBERT-base`) | ○ |
| **PROD** | **0.640** | 0.607 (`ModernBERT-base`) | ○ |
| seed std | 0.0051 | 0.0040 (`bert-base-cased`) | ✕ 다섯 중 3위 |

**micro 를 근거로 쓸 수 없는 것은 후보 사이의 간격이 한 후보 안의 흔들림보다
좁기 때문이다.** 상위 세 후보가 0.23pp 안에 들어 있는데, 모델은 그대로 두고
seed 만 바꿔 돌린 값의 표준편차가 0.40~0.51pp 다. 눈금 간격이 자의 떨림보다
좁으면 그 자로 잰 순위는 실력이 아니라 추첨 결과다(§seed 하나로는 순위를 못
정한다).

**갈린 자리는 저빈도 타입 하나로 모인다.** macro 와 PROD 는 따로 센 두 근거처럼
보이지만 실은 겹친다 — 2위 `ModernBERT-base` 와의 타입별 median 차를 다섯 개
모두 더하면 4.3pp 인데 그중 3.3pp 가 PROD 한 타입에서 나온다. macro 가 micro 와
다른 답을 내는 이유도 같다. PROD·EVT 는 합쳐서 test support 의 4.9%(319/6,488
span)뿐이라 span 을 한 통에 붓는 micro 에서는 묻히고, 타입마다 5분의 1 씩 주는
macro 에서는 40%를 차지한다. 즉 채택 근거는 **저빈도 타입을 덜 놓친다** 한
문장으로 줄어든다.

**seed 안정성은 채택 근거가 아니다.** `roberta-base` 의 std 0.0051 은 다섯 후보
중 3위이고, seed 폭 1.02pp 도 `bert-base-cased`(0.80pp)·`ModernBERT-base`(0.86pp)
보다 크다. 흔들림이 뚜렷하게 큰 것은 `electra-base`(1.98pp)와
`xlm-roberta-base`(2.38pp) 둘뿐이라, 안정성은 이 둘을 떨어뜨리는 데만 쓰였고
1·2·3위를 가르는 데는 쓰이지 않았다.

**이 근거는 세대 3 에서 다시 확인되지 않았다.** #244 가 현행 gold 로 다시 돌린
것은 `roberta-base` 한 종뿐이라, 채택을 정했던 macro·PROD 우위가 현행 코퍼스에서도
유지되는지는 재지 않았다. 그래서 이후 en 분류기 변경의 회귀 비교는 이 표가 아니라
§현행 gold 의 `roberta-base` baseline 값과 나란히 놓는다.

**올려야 할 곳은 백본이 아니라 PROD·EVT 의 데이터·증강이다.** 세대 1 에서 1위와
최하위의 micro 차가 1.85pp 인 반면 PROD 한 타입의 후보 간 차가 12.7pp 였고, 세대
3 에서도 가장 낮은 타입은 PROD(0.5794)다.

## 한계

- **단일 split** — 분할은 하나이고 seed 만 셋이다. 따라서 std 가 재는 것은
  *학습 초기화·셔플*의 흔들림이지 *분할*의 흔들림이 아니다. 다른 분할에서
  순위가 유지되는지는 확인하지 않았다.
- **외부 수치와 비교 불가** — 공식 test 를 안 쓰고, 18종을 6종으로 매핑해
  라벨 공간도 다르다. 이 표의 절대값은 published OntoNotes NER F1 과 나란히
  놓을 수 없다.
- **학습시간은 후보당 1회 1 epoch** — 반복이 없어 이 값들의 흔들림 폭을
  모른다. 상위 세 후보의 252~254초 차이는 그 폭 안일 수 있으므로 서로
  구분되는 값으로 읽지 않는다. `ModernBERT-base` 가 1.7배 느린 것처럼 폭이
  큰 차이만 실체가 있다.
- **large 급 미포함** — 영문 인코더 SOTA 는 large 급에 있지만 ko·ja·vi 와
  파라미터 층위를 맞추기 위해 base 급으로 한정했다.
- **기성 OntoNotes 파인튜닝 모델과 미비교** — `tner/roberta-large-ontonotes5`
  등은 라벨 공간이 18종이라 canonical 매핑이 필요하고 PII 5종을 못 낸다.

## 재현

```bash
ls data/ontonotes_en/pii/   # #209 산출물 (생성은 augmenters, gitignore)
uv sync

# 1) split 병합 — split 필드를 떨어뜨린다
python -m ner.augmenters.ontonotes_en.merge_splits \
    --input-dir data/ontonotes_en/pii \
    --output data/ontonotes_en/origin.jsonl

# 2) 후보 1종 × seed 1개 (6종 × seed 42·43·44 = 18회, 실제 15회 + deberta 1회)
CUDA_VISIBLE_DEVICES=0 python -m ner.classifier \
    --lang en --model-name roberta-base --group-key orig \
    --precision bf16 --train-seed 42 --epochs 5 --lr 5e-5 \
    --batch-size 16 --seed 42 \
    --output-dir results/classifier/en_bench/roberta_base_seed42

# 3) 무결성 검사 + median 집계 (당시 실행 경위 — 도구는 제거돼 지금은 돌지 않는다)
python -m ner.classifier.backbone_summary \
    --runs-dir results/classifier/en_bench --pattern '*_seed4*' --check \
    --output results/classifier/en_bench/median_summary.json

# 4) 학습시간 재측 — 5종을 하나씩, GPU 독점 상태에서 1 epoch
#    시작 전 nvidia-smi --query-compute-apps 로 다른 프로세스가 없음을 확인한다
CUDA_VISIBLE_DEVICES=0 python -m ner.classifier \
    --lang en --model-name roberta-base --group-key orig \
    --precision bf16 --train-seed 42 --epochs 1 --lr 5e-5 \
    --batch-size 16 --max-length 256 --seed 42 \
    --output-dir results/classifier/en_bench/time_1ep/roberta_base
```

**동시에 여러 실행을 띄우지 않는다.** 한 GPU 에 나눠 담으면 F1 은 그대로지만
학습시간이 오염되고, 같은 출력 디렉토리에 두 프로세스가 붙으면 F1 까지
조용히 바뀐다.

## 산출물 위치

| 무엇 | 경로 |
|---|---|
| 원장(인용 근거) | **없다** — 세대 1 원장 `certified/classifier/en/backbone-bench/` 는 #244 가 지웠고, 남아 있던 `certified/` 도 이후 통째로 폐기했다. 아래 백본 비교표는 대조 상대가 없는 기록이다 |
| 현행 baseline 수치의 출처 | §현행 gold 의 `roberta-base` baseline. 그 원본 JSON 은 [삭제 전 Git 기록](https://github.com/groovallstar/ner-pipeline/tree/a60cb813d0be19a1ccb414c660c83c6407bb50a5/certified/classifier/en/roberta-3seed)에 있다 |
| scratch | `results/classifier/en_bench/` (gitignore·휘발) |
| 학습시간 재측 | `results/classifier/en_bench/time_1ep/` (gitignore·휘발) |
| 병합 도구 | `src/ner/augmenters/ontonotes_en/merge_splits.py` |
| 집계·검사 도구 | **없다** — `src/ner/classifier/backbone_summary.py` 는 백본 비교 종결 후 제거됐다 |
| 이슈 문서 | `docs/issues/issue-215-en-backbone-benchmark.md` |
