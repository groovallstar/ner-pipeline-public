# issue-215: 영문(en) BERT 백본 NER 벤치마크 — base 급 6종 baseline 확립

- Issue: https://github.com/groovallstar/ner-pipeline/issues/215
- 브랜치: `feat/issue-215-en-backbone-benchmark`
- 승인일: 2026-08-21

## 배경 (왜)

**영문 데이터는 있는데 그것을 학습할 백본이 정해져 있지 않았다.** #209 가
OntoNotes5 를 canonical 10종 평면으로 옮겨 `data/ontonotes_en/pii/` 까지
만들었지만 그 이슈는 "모델은 만들지 않는다"를 범위에 명시했고, `classifier`
는 `--lang {ja,vi,ko}` 만 받았다.

ko·ja·vi 는 각각 백본 벤치마크를 거쳐 출발점을 못 박았다(ko #122 →
`koelectra-base-v3`). en 만 그 출발점이 없어서, 아무 모델이나 골라 학습하면
그 수치가 좋은 것인지 나쁜 것인지 견줄 자리가 없었다.

## 목적

영문 canonical 10종 데이터에 base 급 인코더 백본 6종을 같은 레시피로
파인튜닝해 ko·ja·vi 와 같은 자(char-offset strict span F1)로 재고, 영문
baseline 백본을 확정한다. 임계값 게이트가 아니라 **baseline 수치 확립**이다.

## 범위

- 포함: `--lang en` 배선 · split 병합 · base 급 6종 비교 · 3-seed median 판정
  · 리포트 · `certified/` 승격
- 제외: large 급 백본 · 기성 OntoNotes 파인튜닝 모델 비교 · 공식 분할 경로
  신설 · 모델별 하이퍼파라미터 sweep · 서빙 배선 · 게이트 수치 설정

## 성공 기준

- [x] `--lang en --smoke --group-key orig` 가 끝까지 돈다 — 스모크 로그 + pytest
- [x] 병합본 76,378 행 · span 자기정합 불일치 0 · `orig` group-key 통과 —
      pytest (`test_merged_corpus.py`, 데이터 없으면 skipif)
- [x] 6종 결과표 완성, 학습 불가 후보는 실패 양상·시도 LR 명시 제외
- [x] 1위를 2pp 안 후보 3-seed median 으로 확정, std 병기
- [ ] **학습시간은 1 epoch 순차 재측값만 인용** — 미완(§미해결 질문)
- [ ] 리포트 작성 — 미완. 원장 승격은 완료

## 현 상태 (fact)

### 데이터 — 병합본

세 split 파일을 하나로 합쳐 기존 방식대로 재분할한다. ko #122 와 조건을 맞추기
위해서이고, 이렇게 하면 분할 코드(`data_utils.py` — 기준 파일)를 안 건드려
2단계 게이트를 타지 않는다. 대가로 외부 논문·리더보드의 OntoNotes 수치와 직접
비교할 수 없다.

| 항목 | 값 |
|---|---|
| 경로 | `data/ontonotes_en/pii_all.jsonl` |
| 행 | 76,378 (train 59,675 + valid 8,493 + test 8,210) |
| `orig` 그룹 | 70,641 |
| span | 167,871 (NER 6종 75,172 + PII 4종 92,699) |
| 지문 | `39cb0f9e9c16f265` |
| 분할 | 단일 split, valid 0.1 / test 0.1, `seed=42`, `--group-key orig` |

**병합할 때 `split` 필드를 떨어뜨린다.** 세 파일을 그대로 이어붙이면 그 필드가
한 파일 안에서 세 값을 갖는데, `validate_group_key` 는 값이 적으면서 선언한
키의 그룹을 가르지 않는 필드를 "더 강한 그룹 키"로 보고 거부한다. 같은 `orig`
의 행은 언제나 같은 split 에 있어(경계를 하나도 가르지 않는다) 정확히 그
모양이 되고, `--group-key orig` 가 통째로 막힌다. 재분할이 전제라 원 split 은
학습에 쓰이지 않으며, 어느 split 이었는지는 `id`·`orig` 접두사에 남는다.

### 학습 조건

epochs=5, lr=5e-5, batch 16, max_length=256, **bf16**,
`metric_for_best_model='eval_loss'`, `--train-seed {42,43,44}`. 후보 간 고정이며
모델별 sweep 은 하지 않았다 — 이 벤치가 재는 것은 백본이지 튜닝 솜씨가 아니라,
한 후보만 조건을 손봐주면 비교가 깨진다.

### 결과 — 3-seed median

<!-- certified: classifier/en/backbone-bench -->

| 후보 | 파라미터 | median | std | min | max | macro | (참고)PII5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`roberta-base`** | 124M | **0.8795** | 0.0051 | 0.8734 | 0.8836 | **0.8106** | 0.9462 |
| `bert-base-cased` | 108M | 0.8791 | 0.0040 | 0.8751 | 0.8831 | 0.7899 | 0.9405 |
| `answerdotai/ModernBERT-base` | 150M | 0.8772 | 0.0048 | 0.8692 | 0.8778 | 0.8001 | 0.9370 |
| `google/electra-base-discriminator` | 109M | 0.8656 | 0.0109 | 0.8475 | 0.8673 | 0.7718 | 0.9475 |
| `xlm-roberta-base` | 278M | 0.8610 | 0.0129 | 0.8577 | 0.8815 | 0.7835 | 0.9386 |
| ~~`microsoft/deberta-v3-base`~~ | 184M | — | — | — | — | — | **학습 불가** |

타입별 median strict F1 (NER-5):

<!-- certified: classifier/en/backbone-bench -->

| 후보 | PER | LOC | ORG | PROD | EVT |
|---|---:|---:|---:|---:|---:|
| `roberta-base` | 0.947 | 0.882 | 0.837 | **0.640** | 0.743 |
| `bert-base-cased` | 0.944 | **0.893** | **0.843** | 0.562 | 0.723 |
| `ModernBERT-base` | 0.943 | 0.881 | 0.823 | 0.607 | **0.752** |
| `electra-base` | **0.946** | 0.876 | 0.806 | 0.513 | 0.702 |
| `xlm-roberta-base` | 0.920 | 0.888 | 0.817 | 0.560 | 0.742 |

### micro 로는 1위를 못 가른다

상위 세 후보가 0.23pp 안에 몰려 있는데 같은 후보의 seed 표준편차가
0.40~0.51pp 다. **순위 간격보다 한 후보 안의 흔들림이 크다.** 그래서 micro
단독으로는 누가 위인지 말할 수 없고, 갈림은 다른 축에서 난다 —
`roberta-base` 가 macro 최고(0.8106) · PROD 최고(0.640) · std 최소(0.0051)라
저빈도 타입에서 앞서고 seed 안정성도 좋다. baseline 을 이 근거로 고른다.

**단일 seed 였다면 다른 결론을 적었다.** seed 42 만 보면 순위가
`bert-base-cased`(0.8831) > `xlm-roberta-base`(0.8815) > `ModernBERT`(0.8778)
> `roberta-base`(0.8734) 였는데, median 으로는 `xlm-roberta-base` 가 최하위로
내려가고 `roberta-base` 가 올라온다. xlm-r 의 seed 폭이 2.38pp(0.8577~0.8815)
로 후보 중 가장 컸기 때문이다.

### DeBERTa-v3 학습 불가 — 세 번째 언어에서 같은 실패

`microsoft/deberta-v3-base` 는 표준 레시피(lr 5e-5 · bf16)에서 학습되지 않았다.

```
step  500 : loss 288.7, grad_norm nan
step 1000~: loss 0,     grad_norm nan   (5 epoch 내내)
eval      : eval_loss nan → 전 타입 F1 0
```

`loss: 0` 은 학습이 잘 된 것이 아니라 가중치가 NaN 으로 오염된 뒤의
부산물이다. ko(#122)·vi 두 벤치에서 DeBERTa-v3 계열이 같은 양상으로 실패했고
en 이 세 번째다. 모델별 sweep 은 이 벤치의 범위 밖이라 사유를 적고 제외한다.

두 선행 리포트는 한계로 "정밀도(fp16/fp32) 변수 미분리"를 적었는데, en 은
bf16 확정 조건에서 발산했으므로 **세 언어 모두 bf16 에서 실패**한 것이 된다.
fp32 는 아직 아무도 시도하지 않았다.

### 관측된 이상 — PII 도 seed 를 탄다

`roberta-base` seed 43 에서 EMAIL 만 F1 0.731 로 무너졌다(같은 후보의 다른
seed 는 0.985·0.984). recall 0.851 은 정상인데 precision 이 0.640 이라 **과잉
예측** 형태이고, 예측 수가 gold 2,385 보다 약 790 개 많다. NER-5 선정 기준
밖이라 순위에는 영향이 없다.

ko·ja·vi 리포트는 "합성 PII 는 패턴이 깨끗해 포화하므로 변별력이 없다"고
적었는데, 이 사례는 **포화가 seed 마다 보장되지는 않는다**는 것을 보인다.
선정 기준에서 빼는 판단 자체는 유지하되(한 seed 의 국소 붕괴가 백본 우열을
말해주지 않는다), "PII 는 늘 1.0 근처"라는 전제는 en 에서 반례를 얻었다.

### 산출물 무결성

15회 전부 **시각 정합 검사**를 통과했다(gap 50~124초). 이 검사는 로그의 첫
시각 + `train_time_sec` 이 `metrics.json` 이 쓰인 시각과 맞는지 보는 것으로,
그 사이에는 평가와 체크포인트 저장만 있으므로 차이는 작은 양수여야 한다.
두 프로세스가 같은 출력 디렉토리를 쓰면 값만 조용히 바뀌어 눈으로는 안
보이는데, 이 검사가 실제로 오염된 run 두 건을 잡아냈다(§결정 로그 2026-08-21).

## 결정 로그

- **2026-08-21**: 분할을 공식 3분할이 아니라 병합 후 재분할로 정함. 공식
  분할을 쓰려면 `data_utils.py` 에 새 경로를 뚫어야 하는데 그 파일은 기준
  파일이라 2단계 게이트를 타고, ko #122 와 조건도 갈린다. 대가로 외부 수치와
  직접 비교는 포기.
- **2026-08-21**: 병합 시 `split` 필드를 떨어뜨리기로 함. 남기면
  `validate_group_key` 가 그것을 "더 강한 그룹 키"로 잡아 `--group-key orig`
  를 막는다. 설계 시점에는 예상하지 못했고 실제로 합쳐본 뒤 드러났다.
- **2026-08-21**: 후보 범위를 base 급으로 한정. 영문 인코더 SOTA 는 large
  급에 있지만 ko·ja·vi 가 모두 base 급이라 네 언어 리포트를 같은 파라미터
  층위에서 비교하는 쪽을 택했다.
- **2026-08-21**: 순위를 단일 seed 가 아니라 3-seed median 으로 정하기로 함
  (정의 시점 결정). ja 리포트가 자기비판하듯 출하 체크포인트의 PROD R 0.9362
  는 5-seed 분산 안의 단일 draw 였고, 백본 간 차가 1~2pp 로 붙으면 단일 seed
  순위는 실력이 아니라 추첨을 적는 것이 된다. **결과적으로 이 결정이 결론을
  바꿨다** — 1차 단일 seed 1·2위가 median 에서 4위·5위로 내려갔다.
- **2026-08-21**: 학습시간 지표를 병렬 실행 wall-clock 이 아니라 1 epoch 순차
  독점 재측으로 정함. 여러 후보가 한 GPU 를 나눠 쓰면 그 숫자는 모델의 성질이
  아니라 스케줄링의 부산물이 된다.
- **2026-08-21**: 같은 워크트리에서 claude 세션 둘이 같은 명령을 각자 실행해
  같은 출력 디렉토리에 두 프로세스가 쓰는 사고가 발생. `electra_base_seed43`
  (gap −1,846초)과 `xlmr_base_seed44`(중복 이력)를 폐기하고 재실행했다. 시간
  재측은 오염이 확인돼 전량 폐기했고 미완으로 남긴다.

## 미해결 질문

- **학습시간 재측이 미완이다.** 세션 중복 사고로 측정 조건(GPU 독점 + 조용한
  호스트)이 깨져 값을 전부 버렸다. 재개하려면 호스트에 학습 프로세스가 하나도
  없는 것을 확인한 뒤 5종을 하나씩 순차로 돌린다(§재현). 이 값이 없어도
  baseline 선정은 성립한다 — 갈림은 macro·PROD 에서 났고 시간은 서빙 비용
  정보일 뿐이다.
- **`DEFAULT_MODEL['en']` 은 아직 잠정값(`roberta-base`)이다.** 벤치 결과가
  마침 같은 모델을 가리키므로 값 자체는 맞지만, 잠정이라는 주석은 리포트
  작성 후 걷어야 한다.
- **리포트(`docs/reports/english-bert-classifier-benchmark.md`)가 미작성이다.**

## 재현

```bash
# 1) 데이터 병합 (data/ 는 gitignore — 재생성 경로가 이 CLI 다)
python -m ner.augmenters.ontonotes_en.merge_splits \
    --input-dir data/ontonotes_en/pii \
    --output data/ontonotes_en/pii_all.jsonl

# 2) 후보 1종 × seed 1개 (후보 6종 × seed 42·43·44 반복)
CUDA_VISIBLE_DEVICES=0 python -m ner.classifier \
    --lang en --model-name roberta-base --group-key orig \
    --precision bf16 --train-seed 42 \
    --output-dir results/classifier/en_bench/roberta_base_seed42

# 3) 무결성 검사 + median 집계
python -m ner.classifier.backbone_summary \
    --runs-dir results/classifier/en_bench --pattern '*_seed4*' --check \
    --output results/classifier/en_bench/median_summary.json

# 4) 학습시간 재측 (미완) — 호스트에 학습 프로세스가 없는지 먼저 확인한다
ps -eo cmd | grep -c '[n]er.classifier'   # 0 이어야 한다
CUDA_VISIBLE_DEVICES=0 python -m ner.classifier \
    --lang en --model-name roberta-base --group-key orig \
    --precision bf16 --train-seed 42 --epochs 1 \
    --output-dir results/classifier/en_bench/timing/roberta_base
```

**동시에 여러 실행을 띄우지 않는다.** 한 GPU 에 나눠 담으면 F1 은 그대로지만
학습시간이 오염되고, 같은 출력 디렉토리에 두 프로세스가 붙으면 F1 까지
조용히 바뀐다.

## 산출물 위치

| 무엇 | 경로 |
|---|---|
| 원장(인용 근거) | `certified/classifier/en/backbone-bench/` — run 16개 `metrics.json` + `median_summary.json` |
| scratch | `results/classifier/en_bench/` (gitignore·휘발) |
| 병합 도구 | `src/ner/augmenters/ontonotes_en/merge_splits.py` |
| 집계·검사 도구 | `src/ner/classifier/backbone_summary.py` |

## 관련 커밋

- `c1b7a58` feat(classifier): 영문 en 배선 + split 병합 도구
