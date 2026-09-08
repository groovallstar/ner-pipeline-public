# 한국어 BERT NER 분류기 백본 벤치마크 (canonical 10종 평면)

- 측정일: 2026-06-17
- 대상: KLUE 유래 canonical 10종 평면 데이터에 인코더 백본 파인튜닝 — **baseline 수치 확립**(임계값 게이트 아님)
- 데이터: `data/klue/pii_all.jsonl` (25,989 행, canonical 10종 = NER 5 + PII 5)
- 평가: char-offset span F1 (strict), **단일 split**(test 0.1 / valid 0.1, `seed=42`)
- 출처: GitHub 이슈 #122
- 일본어·베트남어 동일 셋업: `docs/reports/{japanese,vietnamese}-bert-classifier-benchmark.md`

## 요약

- **baseline = `monologg/koelectra-base-v3-discriminator`** — NER-5 strict micro-F1
  **0.8581** / macro **0.7773**, NER-5 **전 항목 1위 + 최단 학습시간(7.3분)**.
- **PII 제외가 필수였다**: overall-10에선 3 단일언어가 0.90~0.91로 촘촘했지만,
  합성 PII를 빼고 **NER-5 macro로 보면 0.70~0.78로 벌어진다**. 주입 PII(F1≈1.0)가
  백본 차이를 가렸다 → 백본 선정은 **canonical NER-5(PER/LOC/ORG/PROD/EVT)** 로 한다.
- **단일언어 > 다국어 확정**: `xlm-roberta-base`(다국어)가 NER-5 **전 항목에서**
  단일언어 3종에 패배 — 영어 혼용 잦은 ORG/PER/PROD에서조차 −7~8pt. 한국어
  임베드 영어는 단일언어 모델이 이미 잘 처리(EMAIL F1≈1.0)하고, 다국어는 그 이점
  없이 한국어 용량 페널티(curse of multilinguality)만 치름. KLUE 논문
  (`klue-roberta > mBERT/XLM-R`)과 일치.
- **`microsoft/mdeberta-v3-base` 제외** — 표준 레시피 학습 불가(아래). VI 동일 거동.
- **KLUE 사전학습 계열(`klue/*`) 배제** — 평가셋이 KLUE 유래라 백본 사전학습
  코퍼스가 평가 원천(news·MODU)과 겹치는 도메인 누출 회피.
- 천장은 **EVT(~0.58)·PROD(~0.71)** — 저빈도·의미 모호. 백본이 아니라 데이터·증강
  레버리지. 게이트 수치는 미설정(baseline 측정 목적).

## 조건

| 항목 | 값 |
|---|---|
| 데이터 | `data/klue/pii_all.jsonl` 25,989 행 (KLUE 유래 + 합성 PII 주입) |
| 분할 | **단일 split** — train 20,793 / valid 2,598 / test 2,598 (`seed=42`) |
| 라벨 | `PER, LOC, ORG, PROD, EVT, DAT, EMAIL, PHONE, ID_NUM, CREDIT_CARD` |
| 학습 | epochs=5, lr=5e-5, max_length=256, **bf16**, batch 16, class-weight·curriculum 미사용 |
| 모델 선택 | `metric_for_best_model='eval_loss'` (valid) |
| 하드웨어 | NVIDIA RTX A6000 49GB, `CUDA_VISIBLE_DEVICES=0` |

## 비교 기준: canonical NER-5 (PII 제외)

PII 5종(`DAT·EMAIL·PHONE·ID_NUM·CREDIT_CARD`)은 **합성 주입값**이라 패턴이 인위적
으로 깨끗하다 — EMAIL/PHONE/ID_NUM/CREDIT_CARD는 F1 0.97~1.0으로 포화(DAT만 ~0.86,
날짜 표현 다양성). 백본 변별력이 없어 **선정 기준에서 제외**하고 참고로만 보고한다.
백본 비교·baseline 선정은 **NER-5(PER/LOC/ORG/PROD/EVT)** 로 한다.

## NER-5 결과 (단일 split, strict)

| 모델 | 계열 | micro-P | micro-R | **micro-F1** | macro-F1 | (참고)PII5 | (참고)ovr10 | 학습 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`koelectra-base-v3-discriminator`** | mono | 0.8524 | 0.8638 | **0.8581** | **0.7773** | 0.9665 | 0.9135 | 7.3분 |
| `kakaobank/kf-deberta-base` | mono | 0.8560 | 0.8350 | 0.8453 | 0.7532 | 0.9658 | 0.9073 | 14.1분 |
| `snunlp/KR-ELECTRA-discriminator` | mono | 0.8468 | 0.8164 | 0.8313 | 0.7234 | 0.9651 | 0.9004 | 7.3분 |
| `xlm-roberta-base` | **multi** | 0.7845 | 0.7830 | 0.7838 | 0.7017 | 0.9494 | 0.8688 | 8.5분 |
| ~~`microsoft/mdeberta-v3-base`~~ | multi | — | — | **학습불가** | — | — | — | — |

### per-entity strict F1 (NER-5)

| 모델 | PER | LOC | ORG | PROD | EVT |
|---|---:|---:|---:|---:|---:|
| **koelectra-base-v3** | **0.9274** | **0.8509** | **0.8202** | **0.7069** | **0.5809** |
| kf-deberta-base | 0.9205 | 0.8177 | 0.8134 | 0.6781 | 0.5364 |
| kr-electra | 0.9068 | 0.8301 | 0.8111 | 0.5378 | 0.5310 |
| xlm-roberta-base (multi) | 0.8602 | 0.7618 | 0.7438 | 0.6051 | 0.5375 |

다국어(xlm-r)는 NER-5 5개 항목 **전부** 단일언어 3종보다 낮다.

### baseline `koelectra-base-v3` 전체 per-entity P/R/F1 (10종)

| Entity | support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| PER | 1,924 | 0.9215 | 0.9335 | 0.9274 |
| LOC | 780 | 0.8764 | 0.8269 | 0.8509 |
| ORG | 1,040 | 0.8073 | 0.8337 | 0.8202 |
| PROD | 332 | 0.6916 | 0.7229 | 0.7069 |
| EVT | 123 | 0.5302 | 0.6423 | 0.5809 |
| DAT | 1,029 | 0.8452 | 0.8756 | 0.8601 |
| EMAIL | 850 | 1.0000 | 1.0000 | 1.0000 |
| PHONE | 835 | 1.0000 | 1.0000 | 1.0000 |
| ID_NUM | 850 | 1.0000 | 0.9988 | 0.9994 |
| CREDIT_CARD | 834 | 0.9988 | 0.9988 | 0.9988 |

## 단일언어 > 다국어 (curse of multilinguality)

영어 혼용이 많은 한국어에서 다국어 백본이 유리하리라는 직관은 데이터로 기각됐다.
xlm-r은 영어 ASCII가 핵심인 EMAIL조차 단일언어가 이미 F1≈1.0으로 처리하는 영역에서
이점이 없고, ORG(0.744 vs 0.81~0.82)·PER(0.860 vs 0.91~0.93)·PROD(0.605 vs
0.68~0.71) 등 영어-명명 개체가 섞이는 곳에서도 단일언어에 −7~8pt 뒤진다. 다국어는
고정 용량을 100+ 언어에 분산해 한국어 표현력이 희석되는 페널티가, 영어 처리 이점
보다 크다. (KLUE 논문도 `klue-roberta > mBERT/XLM-R` 보고.)

## `microsoft/mdeberta-v3-base` 제외 (표준 레시피 학습 불가)

- lr 5e-5: 첫 스텝부터 loss 폭증(240→337) + `grad_norm` NaN → **발산**.
- lr 1e-5: 발산은 멈추나(loss ~1.25 안정) eval가 **all-O 붕괴, F1=0** — **비학습**.

발산은 저LR로 막히지만 그 뒤 학습 자체가 안 되는 패턴은 **VI 벤치
(`vietnamese-bert-classifier-benchmark.md`)의 mDeBERTa-v3·videberta 거동과 동일**
하다. 모델별 sweep(warmup·정밀도 분리 등)은 본 벤치 범위("백본만 비교") 밖이라
제외. 토크나이저는 드롭인(`is_fast`+offset 정상)이라 정렬 문제는 아니다.

## 한계

- **단일 split** — VI(5-fold pooled)와 달리 분산 추정이 없다. NER-5 순위(koelectra >
  kf-deberta > kr-electra)는 항목 전반에서 일관돼 신뢰되나, 절대 수치는 fold 변동
  폭만큼 흔들릴 수 있다. baseline 1종 5-fold 안정성 확인은 후속(이슈 #122 선택 항목).
- **절대 F1은 낙관적** — 단일언어 백본의 사전학습 코퍼스(뉴스·웹·위키)가 KLUE 원천
  도메인과 겹친다(라벨은 아님). 전 단일언어 공통이라 *상대 비교는 유효*.

## 재현

```bash
ls data/klue/origin.jsonl   # 25,989 (생성은 augmenters, gitignore)
uv sync
python -m ner.classifier --lang ko \
  --model-name monologg/koelectra-base-v3-discriminator \
  --precision bf16 --epochs 5 --lr 5e-5 --batch-size 16 --seed 42 \
  --output-dir results/classifier/ko/koelectra-base-v3
# NER-5 집계는 metrics.json 의 per_entity 에서 PER/LOC/ORG/PROD/EVT 만 micro 합산
```

## 산출물 위치

| 경로 | 내용 |
|---|---|
| `results/classifier/ko/<모델>/metrics.json` | 모델별 overall+per-entity P/R/F1 |

`results/`는 gitignore — 본 리포트 표가 영구 인용 단일 출처.
