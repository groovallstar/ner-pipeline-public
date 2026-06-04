# 일본어 BERT NER 분류기 — Phase 0~8 실험 히스토리 상세

본 문서는 일본어 NER classifier (`src/ner/classifier/`, `--lang ja`) 의
실험 히스토리를 **Phase 0→8 누적으로, 각 단계의 가설·시도·결과를 실측
수치와 함께** 정리한 상세 기록이다. 각 단계는 "이전 단계에서 무엇이
한계로 드러났나 → 본 단계에서 무엇을 바꿨나 → 결과(수치) → 다음으로
이어진 가설" 흐름으로 서술한다.

전체 흐름의 특징은 **한 번에 도약하는 단일 기법이 아닌, 매 라운드 진단
결과를 다음 라운드 가설로 이어붙여 점진적으로 끌어올리는 사이클**이다.
처음 0.9058 에서 시작해 데이터 정제 → 손실 함수 → 부정 예시 → 도메인
seed 까지 7~8단계에 걸쳐 strict F1 0.9644 (Phase 8 의 S8) 에 도달했다.

> ⚠️ **그러나 이 0.9644 를 "도달한 성능" 으로 읽어선 안 된다.** Phase
> 0~8 전체가 80/10/10 *단일 분할* (test 527 문장) 위에서만 측정됐는데,
> 사후 재분할 검증 (#69) 에서 이 측정 프로토콜 자체에 두 가지 구조적
> 결함이 드러났다 (상세: 하단 §본 문서 동결):
> 1. **측정 노이즈가 개선폭을 삼킨다** — 희소 entity (~94 span) 표본
>    에서 seed 간 분산의 72~100% 가 표본추출 노이즈라, ±4pp 이내 변화는
>    진짜 개선인지 seed 운인지 가릴 수 없다. 후반 Phase 의 1pp 미만
>    개선 다수가 *측정 가능 범위 밖* 이었다.
> 2. **Phase 7~8 의 extra train 이 test 문장을 누설했다** — 추가 학습
>    데이터에 test 문장의 30~35% 가 원문 그대로 섞여, 누설 구간 F1
>    0.9947 (암기) vs 비누설 구간 0.9478 로 갈렸다. 즉 0.9644 는 약
>    +1.7pp 부풀려진 값이고, *누설을 걷어낸 실측 0.9478 은 게이트 0.95
>    에 미달* 한다.
>
> 따라서 본 문서의 수치는 *같은 프로토콜 안의 상대 비교* 로만 유효하다.
> 0.9644 는 production 일반화 성능이 아니라 이 측정 방식이 만들어낸
> *상한 착시* 에 가깝다. 누설을 걷어낸 비중복 실측 (0.9478, 게이트
> 미달) 이 진짜 일반화에 더 가깝고, extra 없는 base 청정 기준선 (층화
> K-fold) 은 strict F1 **0.9195** (게이트 -3.05pp) 다. 0.9644 와의
> 격차는 *모두 거품이 아니라* extra 의 진짜 기여 (+2.5~3pp) 와 누설
> 인플레이션 (~+1.7pp) 이 섞인 값이므로, 부풀린 몫을 걷어내면 게이트
> 아래로 내려간다. 그래서 이후 실험은 새 기법이 아니라 *측정 신뢰성
> 회복* 을 먼저 했고, 2편 (`japanese-bert-classifier-history-2.md`) 으로
> 이어진다.

- **요약본** (현 production 결론·F1 추이·핵심 교훈·산출물·재현 명령):
  `japanese-bert-classifier-benchmark.md`
- 본 문서는 정리되어 제거된 옛 산출물 (변종 metrics.json 등) 수치의
  **영구 인용 단일 출처**다 — Phase 상세 표를 삭제·요약하지 말 것
- 운영 규칙: 본 문서는 Phase 0~8 에서 **동결**되었다 (하단 §본 문서
  동결 참조). 새 실험은 측정 프로토콜이 달라
  `japanese-bert-classifier-history-2.md` (2편) 에 기록한다
- 라운드↔Phase 매핑: 라운드 2 = Phase 4 (#51) / 라운드 3 = Phase 5~7
  (#56·#57·#58) / 라운드 4 = Phase 8 (#61)

---

## Phase 0 — 데이터 스키마 정착 (#21, #27)

> 분류기 학습 *이전*의 데이터셋 생성 단계라 BERT 측 측정 수치는 없다.
> 이후 모든 classifier 실험의 입력 단일 출처를 만든 사전 작업.

### 무엇을 했나

원본 Stockmark NER 의 일본어 8종 라벨을 canonical 10종 평면 (NER 5 +
PII 5) 으로 통합. 시설명 (역·공항·병원·학교·점포·박물관·종교시설) 을
LOC → ORG 로 재배치하여 LOC 는 지명·주소만 보유하도록 정리. PII 5종
(DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD) 은 augmenter 가 합성 주입.

### 왜

classifier 학습 전 단계로 라벨 정의가 라벨러·증강기와 정합해야 한다.
시설명을 LOC 에 두면 "도시명 vs 그 안의 역" 같은 경계 모호함이
classifier 학습 신호를 흐리게 만든다.

### 결과

- `docs/manual/data/canonical-entity-schema.md` 정착
- 5,307 행 PII 합성 데이터셋 (`data/stockmark/pii_all.jsonl`) 생성
- 라벨러 (LLM 단독 NER) 8모델 재벤치 → 시설=ORG 정의에서도
  Filtered F1 평균 +0.005 ~ +0.014 향상 (회귀 없음). 라벨러 수치 원본은
  `docs/reports/japanese-ner-benchmark.md` / `japanese-ner-pii-benchmark.md`

이 데이터셋이 이후 모든 classifier 실험의 입력 단일 출처가 된다.

---

## Phase 1 — 첫 학습 + SOTA 모델 sweep (#40)

### 직전 상태

Phase 0 에서 canonical 10종 평면 데이터셋이 정착. 어떤 BERT 계열 모델이
이 데이터에서 가장 잘 학습되는지, 그리고 baseline 측정값이 어느 수준
인지 먼저 확정해야 다음 가설을 세울 수 있다.

### 본 단계에서 시도

- 80/20 → 80/10/10 분할 도입. valid 셋이 epoch best 선택을 담당해 test
  셋이 학습 어디에도 노출 안 됨
- SOTA 모델 sweep 7 후보 + 변종 4 (class weight / large / curriculum)
  총 18회 학습

### 결과 — SOTA sweep 7 후보 (overall)

| 모델 | F1 | Precision | Recall | Train time |
|---|---:|---:|---:|---:|
| **`tohoku-nlp/bert-base-japanese-v3`** (baseline / production backbone) | **0.9058** | 0.8653 | 0.9501 | 99s |
| `tohoku-nlp/bert-large-japanese-v2` | 0.8894 | 0.8551 | 0.9266 | 676s |
| `jhu-clsp/mmBERT-base` | 0.7662 | 0.7508 | 0.7822 | 408s |
| `sbintuitions/modernbert-ja-130m` | 0.7649 | 0.7493 | 0.7811 | 259s |
| `llm-jp/llm-jp-modernbert-base` | 0.7220 | 0.6934 | 0.7530 | 357s |
| `ku-nlp/deberta-v3-base-japanese` (bf16, lr=2e-5) | 0.0000 | 0.0000 | 0.0000 | 360s — 학습 실패 |
| `microsoft/mdeberta-v3-base` (bf16, lr=2e-5) | 0.0000 | 0.0000 | 0.0000 | 391s — 학습 실패 |

> 원시 sweep metrics (`results/classifier/ja/sweep/<모델>/`,
> `ja_sweep/baseline/`) 는 실험 종결 후 정리되어 더 이상 없다. 본 표가
> 영구 인용 출처 (§산출물 위치).

#### 베이스라인 per-entity (`bert-base-japanese-v3`)

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

#### 모델별 per-entity F1

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

#### llm-jp-modernbert PII 붕괴 — P/R 상세

| Entity | F1 | Precision | Recall | Support |
|---|---:|---:|---:|---:|
| EMAIL | 0.1695 | 0.1685 | 0.1705 | 88 |
| ID_NUM | 0.1163 | 0.1124 | 0.1205 | 83 |
| CREDIT_CARD | 0.3106 | 0.3012 | 0.3205 | 78 |
| PHONE | 0.5644 | 0.5644 | 0.5644 | 101 |

P 와 R 이 거의 동일 — false positive 와 false negative 가 동시에 발생.
**boundary 오류가 아니라 토큰 분해 자체의 실패**.

### 학습 실패 모델 분석

#### DeBERTa-v3 family (`ku-nlp/deberta-v3-base-japanese`, `microsoft/mdeberta-v3-base`)

| 항목 | 관측값 |
|---|---|
| 학습 loss | 정상 감소 (172 → 4.6) |
| eval_loss | 정상 감소 (70 → 3.7) |
| 평가 prediction | **모두 'O'** (majority-class collapse) |
| 모델 weight | NaN 없음 |
| classifier head 학습 신호 | 평균 ≈ 0.004 (미세하지만 받음) |
| 시도한 설정 | bf16 (fp16 unscale 에러 회피용), lr=2e-5 |

**원인 추정**: DeBERTa-v3 family 는 알려진 unstable family —
`lr=2e-5 + bf16` 의 단순 설정으로는 token-classification head 가
majority-class 'O' 로 붕괴. warmup_ratio (0.06~0.1), weight_decay (0.01),
gradient_clip (1.0) 의 추가 튜닝이 필요. 본 sweep 은 모든 후보를 동일
프로토콜로 비교하기 위해 추가 튜닝을 적용하지 않았다.

JA·VI 양 언어에서 동일 패턴 재현 — 별도 hyperparameter sweep 이슈
(post-#40 핸드오프 §F3) 에서 분리 검증.

#### Modern tokenizer fragmentation (`mmBERT-base`, `modernbert-ja-130m`, `llm-jp-modernbert-base`)

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

#### 제외된 모델

- `studio-ousia/luke-japanese-large-lite` — `LukeTokenizer` 호환성 문제로
  현 `data_utils._encode_ja` 경로에 통합 불가. smoke run 결과만 보존
  (F1=0). 별도 어댑터 필요.

### 핵심 발견

- **베이스라인이 최선**. 다른 후보 모두 회귀 또는 학습 실패
- ModernBERT/mmBERT 는 영숫자 PII (`@`, 숫자열) 토큰 분해가 부정확 →
  EMAIL/ID_NUM/CC F1 < 0.35
- DeBERTa-v3 family 는 `lr=2e-5 + bf16` 만으로는 학습이 풀리지 않음 —
  **모든 토큰을 "entity 아님 (O)" 으로 예측**해버려 사실상 아무것도 못
  맞히는 상태로 수렴 (다수 클래스 붕괴 패턴)
- PII 5종 (PHONE/EMAIL/DAT 등) 은 baseline 부터 이미 충분히 높음. NER
  5종이 병목

### 참고 — 변종 ablation (v1~v4, 80/20 분할 시점)

> 누적 변종은 80/20 분할 시점 측정값. 80/10/10 분할로 갱신된 baseline
> (0.9058) 과 직접 비교하지 말 것.

| 변종 | 누적 변경 | F1 | Precision | Recall |
|---|---|---:|---:|---:|
| v1 baseline | 표준 CE / `eval_loss` best / base 모델 | 0.8952 | — | — |
| **v2** classwt + NER-best | NER B/I weight=2.0, PII B/I weight=0.5, `metric_for_best='ner_f1'` | **0.9134** | 0.8888 | 0.9394 |
| v3 + large | `bert-large-japanese-v2` | 0.9150 | 0.8966 | 0.9341 |
| v4 + curriculum | NER warmup 3 epoch + 21-class fine-tune | 0.9121 | 0.8875 | 0.9380 |
| **gate** | — | 0.9500 | — | — |

핵심:
- **v2 가 ROI 최고** (+1.82pp). v3 (+0.16pp / 4-5x 시간) 비실용적.
- **PII 5종은 v1 부터 0.95+ 포화** — class weight down-weight 효과 미미.
- **NER 5종 +1~2pp 상승** — class weight 의 NER 측 효과는 부분 성립.
- **JA PROD v3 = 0.840 (+8.2pp vs v1)** — large + class weight 가
  lexical-poor 클래스에 단일 최대 개선.
- EVT 는 어떤 변종도 87% 천장 — 어휘·문맥 다양성 부족.

### NER 5종 vs PII 5종 가중 평균 (baseline)

| 그룹 | F1 |
|---|---:|
| NER 5종 (PER/LOC/ORG/PROD/EVT) | 0.886 |
| PII 5종 (DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD) | 0.948 |

### 잔여 한계 → 다음 단계 가설

- PER 0.96, PHONE/EMAIL/DAT 0.98~1.00
- ORG 0.886 / EVT 0.875 / LOC 0.851 / PROD 0.817
- 모델 측 변경으로는 거의 움직이지 않음 → **데이터 측 다양성 부족이
  주된 원인일 것** 이라는 가설로 다음 단계 진입

---

## Phase 2 — 외부 코퍼스 / 증강 시도 (#45)

### 직전 가설

NER 5종 (특히 EVT/PROD) 의 어휘·문맥 다양성 부족. 외부 무료 JA NER
코퍼스 통합 또는 LLM 기반 증강으로 다양성을 채우면 개선될 것.

### 본 단계에서 시도 (28회 학습)

1. **무료 코퍼스 사전조사** — SHINRA (article-level, span NER 아님),
   KWDLC (EVT 0건 + ARTIFACT type drift), UNER JA (EVT/PROD 미보유) →
   네이티브 span 라벨 코퍼스 사실상 부재
2. **단순 oversample** (PROD/EVT 4x 복제) — overall -0.02 ~ -0.30pp
3. **LLM entity replacement** (Qwen 으로 EVT surface 치환 + Gemma
   검증) — v1 -0.49pp / v2 -0.07pp
4. **KWDLC gold 통합** (PROD/all5) — PROD F1 -9.96pp (ARTIFACT
   정의 mismatch 직격, 아래 풀이)
5. **Tier 1.B boundary-relaxed evaluation** — strict F1 (위치·종류 모두
   완전 일치만 정답) 과 함께 *경계가 한두 글자 어긋나도 부분 점수* 를
   주는 relaxed F1 (SemEval'13 Partial) 을 동시 측정. 회귀가 *boundary
   정확도 손실* 인지 *진짜 학습 손실* 인지 분리하는 평가 장치.
   결과: 8 augmentation variant 모두 strict·relaxed 양쪽 회귀 →
   boundary 문제 아님 = 진짜 학습 손실

### 결과 — overall Δ (#45 원측정)

| 시도 | overall Δ | 비고 |
|---|---:|---|
| baseline rerun | (=0.9077) | — |
| 단순 oversample | -0.02~-0.30 | 같은 행 복제는 효과 없음 |
| LLM entity replacement | -0.49~-0.07 | augmentation 노이즈 |
| KWDLC PROD | -1.28 | type drift |
| KWDLC all5 | -0.79 | PROD -5.49 직격 |

핵심 발견:
- **augmentation 8회 모두 strict·relaxed 양쪽 회귀** = boundary 손실이
  아닌 진짜 학습 손실. *외부 데이터로 다양성을 채우는 단순 경로는
  실효성 없음* 으로 확정
- KWDLC ARTIFACT → PROD type drift 가 silver 노이즈보다 더 치명적.
  "gold = 안전" 가설 반증 (아래 풀이)
- PROPOR 2026 의 "LLM augmentation can reduce NER F1 by 0.24~1.81%"
  본 실험에서 그대로 재현

#### Tier 1.B — boundary-relaxed 8 variant (절대 strict/relaxed)

> 위 Δ표는 #45 원측정(overall), 본 표는 동일 baseline 으로 strict /
> SemEval'13 Partial F1 을 동시 산출한 재평가다.

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

#### KWDLC 정의 mismatch 풀이 — "왜 gold 인데 망가졌나"

같은 라벨 이름 (`ARTIFACT` → `PROD`) 이 서로 다른 걸 가리키고 있어서
PROD 학습이 망가졌다.

| | 우리 PROD | KWDLC ARTIFACT |
|---|---|---|
| 정의 | 판매되는 상품 (商品) | 사람이 만든 모든 물건 |
| 예시 | iPhone, Galaxy, 코카콜라 | 「進撃の巨人」, 「みんなのうた」, 「Beatles」 |

"ARTIFACT → PROD" 매핑으로 같이 학습 → 모델이 "작품도 PROD 다" 라고
잘못 배움 → test 의 진짜 PROD (상품) 를 못 맞혀 **PROD F1 -9.96pp**.
정의가 어긋난 그 한 클래스만 표적처럼 깨졌다는 의미로 *직격*.

교훈: gold 라벨이라도 *정의 자체의 어긋남* 은 silver 노이즈보다 더
치명적. 외부 코퍼스 통합 트랙 동결 결정의 근거.

**최종 production 데이터**: KWDLC 데이터·코드 모두 실험 종결 후 제거.
현 production 에 KWDLC 일체 포함되지 않음 (`data/kwdlc_ja/` 부재,
production data 내 KWDLC entity 0건, `src/` 내 'kwdlc' 잔재 0건).

### 잔여 한계 → 다음 단계 가설

- 외부 데이터 통합이 회귀를 부르는 이유는 *type 정의 mismatch*
- 본 데이터의 오답을 다시 들여다보면, **gold 라벨 자체가 일관되지
  않은 부분이 있을 가능성** → Phase 3 (Tier 3 gold cleanup) 로 전환

---

## Phase 3 — Tier 3 gold cleanup (#48)

### 직전 가설

외부 데이터 추가가 회귀를 부르므로 본 데이터 자체의 정합성을 먼저
정리. 모델 오답이 진짜 모델 오류인지 gold 라벨 오류인지 분리하면
모델 측 영향과 데이터 측 영향을 깔끔하게 갈라낼 수 있다.

### 본 단계에서 시도

- 신규 CLI `python -m ner.classifier.error_analysis` (오답 추출),
  `tests/ner/classifier/test_error_analysis.py` (단위 테스트)
- **5,307-row → 5,270-row**: 검수 불가 34 sentences 직접 삭제 (+ 3
  collateral)
- baseline 의 1,398 오류 문장 (test 209 + valid 223 + train 966) 전수
  사람 검수 (Gemini chat 보조, 한국어 단독 검토자 환경) → **2,354 errors**
  - verdict 어휘: `model_correct` / `gold_correct` / `both_wrong` /
    `ambiguous`
  - gold 결함 비율: model_correct 35.0% + both_wrong 5.2% = **40.2%**
    gold 누락/오류. ambiguous 60건 별도 보관 (canonical schema 보강 후보)
- 보정 적용 947건 (model_correct/both_wrong) + chunk 단계 67건
  (ambiguous gold_error) = **1,014건** in-place 적용
- 라벨-문법 휴리스틱 audit cleanup 으로 손상 entity 18건 제거 → 순
  보정 +996 entity

> **회귀 발견 (v1 중간)**: chunk 단계 보정 직후 DAT strict F1
> 0.9802 → 0.9259 (-0.0543) — 검수 도구 (Gemini chat) 의 한글 부분
> 번역이 character offset 을 shift 시켜 span 좌표가 손상. 아래 audit
> cleanup 으로 회복.

#### audit cleanup 상세 (v2, 최종)

라벨-문법 휴리스틱으로 손상된 entity 18건 자동 제거 + 수동 확정 3건:

| 라벨 | 제거 수 | 패턴 |
|---|---:|---|
| DAT | 14 | hiragana 조사 leakage (に, の, る, けだ, えて 등) |
| LOC | 2 | 조사 'に' 시작 / 'の' 종료 leakage |
| CREDIT_CARD | 1 | 일본어 텍스트 leakage (`を管理番`) |
| ID_NUM | 1 | 일본어 텍스트 leakage (`の識別番号を持つ部`) |

순 보정 entity = 1,014 - 18 = **996** (in-place 적용, 백업 미보존).

### 결과 (v1 → v2 audit cleanup)

| 단계 | strict F1 | strict P | strict R | relaxed F1 | support |
|---|---:|---:|---:|---:|---:|
| baseline (보정 전) | 0.8925 | 0.8632 | 0.9240 | 0.9104 | 1,762 |
| v1 (chunk 포함) | 0.9252 | 0.9184 | 0.9320 | 0.9398 | 1,897 |
| **v2 (audit cleanup, 동결)** | **0.9249** | **0.9217** | 0.9281 | **0.9415** | 1,891 |
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

#### 결론 (#48 보정 효과)

- 보정 baseline strict 0.9249 / relaxed 0.9415
- strict ↔ relaxed gap = 1.66pp = boundary 정확도 한계
- gold 결함 40.2% 보정분 = +3.24pp (데이터 정합성 개선), 잔여는 모델
  측 한계

#### 데이터 변경 사실 (재현 caveat)

- pii_all.jsonl 은 본 트랙 진행 중 in-place 변경됐다:
  - 행 수: 5,307 → **5,270** (검수 불가 34 sentences 직접 삭제, 3 collateral)
  - entities: net +996 보정 (1,014 보정 - 18 audit cleanup)
  - 백업 미보존 — 5,307-row 원본은 augmenters/pii 재실행으로 재생성만 가능
- 검수 입력·보정·audit 스크립트 모두 일회성 — 재실행 불가
- 본 측정값은 **현 시점 데이터** 의 절대값이며, 옛 5,307-row 측정값
  (sweep 표 0.9058) 과 동일 분포 비교는 불가

### 잔여 한계 → 다음 단계 가설

- 데이터 보정으로 잡힌 부분과 별개로, 모델이 **BOUNDARY (경계 어긋남)**
  종류의 오류를 구조적으로 반복하고 있을 가능성
- 추가로 사람 검수 라운드 2 를 한 번 더 + 모델 측 손실 함수도 동시
  손봐서 양쪽 다 잡아본다 → Phase 4 진입

---

## Phase 4 — gold 보정 라운드 2 + CRF + boundary loss (#51, 라운드 2)

### 직전 가설

v2 보정 baseline 위에서 test 오답을 다시 들여다보고, BOUNDARY 오류
(전체 오류의 ~40%) 를 모델 측 손실 함수에서 직접 다룬다.

### 본 단계에서 시도

데이터 측 2가지 + 모델 측 2가지를 한 묶음으로 진행.

1. **라벨 정의서 보강 (데이터 측)** — 어떤 entity 를 어느 라벨로 둘지
   모호했던 4영역을 룰로 못 박음 (`canonical-entity-schema.md` §2.3 /
   §3 / §5.1).
   - 시설명 (역·공항·박물관 등) 을 LOC 로 볼지 ORG 로 볼지
   - `チェコ代表` 같은 *국가명 + 代表* (스포츠 대표팀) 의 라벨
   - 부동산·산업 단지 같은 복합 시설의 라벨
   - 가타카나 동음이의어 (회사명과 일반명사가 겹치는 경우)

2. **틀린 문장 라운드 2 재검수 (데이터 측)** — Phase 3 보정 후에도
   baseline 이 여전히 틀리는 138 문장 (285 errors) 을 다시 검수.
   Claude 가 1차 후보 작성 → 사용자가 샘플링으로 확인 → 최종 **40건
   라벨 정정** in-place 적용 (32건 entity 추가 + 7건 라벨 교체 +
   1건 위치만 교체). 데이터 행 수는 5,270 유지.

> **사전 지식: BIO 태깅이란** — NER 모델은 문장의 각 토큰 (대략 글자
> 단위) 마다 태그를 하나씩 붙인다. `B-` = entity 의 *시작* 토큰,
> `I-` = entity 의 *이어지는* 토큰, `O` = entity 아님. 예를 들어
> `東京都に住む` → 東 `B-LOC` / 京 `I-LOC` / 都 `I-LOC` / に `O` /
> 住 `O` / む `O`. 즉 "東京都" 세 글자가 하나의 LOC 로 묶인다.
> 시도 3·4 는 이 태그를 *더 잘 붙이게* 만드는 두 가지 다른 접근이다.

3. **CRF head 추가 (모델 측)** — *문제*: 기본 모델은 토큰마다 태그를
   **따로따로** 고른다. 옆 토큰에 무슨 태그가 붙었는지 안 보기 때문에
   "B-PER (사람 시작) 다음에 갑자기 I-LOC (지명 계속)" 처럼 *문법적
   으로 불가능한 조합* 이 나올 수 있다 (한 entity 중간에 종류가 바뀔
   수는 없다).
   *해법*: 모델 맨 끝에 CRF (Conditional Random Field) 층을 붙인다.
   CRF 는 "어떤 태그 다음에 어떤 태그가 올 수 있는지" 의 규칙 (전이
   점수) 을 함께 학습해서, 토큰을 하나씩 보지 않고 **문장 전체에서
   가장 앞뒤가 맞는 태그 묶음** 을 고른다. 이 "가장 말이 되는 묶음" 을
   효율적으로 찾는 게 Viterbi 알고리즘.
   *기대*: 경계가 어긋나는 (BOUNDARY) 오류를 구조적으로 줄임. (단,
   실제로는 일부 NER 클래스에서 오히려 회귀 → 채택 보류, 결과 표 참조)

4. **경계 가중치 손실 / boundary-aware loss (모델 측)** — *배경*:
   학습은 "모델이 틀린 정도 (= 손실)" 를 줄이는 방향으로 진행되는데,
   보통 모든 토큰의 틀림을 똑같은 무게로 센다.
   *문제*: 우리 오류의 큰 비중이 "경계 어긋남" (entity 의 시작·끝을
   한두 글자 잘못 잡는 것) 인데, 정작 그 경계 토큰을 특별히 더
   신경 쓰지 않는다.
   *해법*: 토큰의 *역할별* 로 틀렸을 때의 벌점 (손실 가중치) 을 차등
   부여 — entity 시작 (`B-`) = 1.5배, 이어지는 토큰 (`I-`) = 1.2배,
   entity 아님 (`O`) = 1.0배. 경계 토큰을 틀리면 더 큰 벌점을 받으니
   모델이 *어디서 entity 가 시작하고 끝나는지* 를 더 집중해서 학습
   하게 된다. CRF 와 달리 태그 순서 규칙을 강제하지는 않고, *학습
   강조점만* 경계로 옮기는 가벼운 방식이라 본 단계에서 최종 채택.

#### 보정 데이터 변경 (in-place)

`pii_all.jsonl` 5,270 행 유지 + 24 sentences entities in-place 변경:

| 종류 | 수 |
|---|---:|
| add LOC (multi-occurrence + 누락) | 20 |
| add ORG | 4 |
| add PROD/PER/CC/ID_NUM/EVT | 7 |
| replace LOC→ORG (시설/공원/단지/서킷/대표팀) | 6 |
| replace ID_NUM→CC (16자리 generator IIN 매칭) | 1 |
| replace_span ORG (NHK boundary 확장) | 1 |

### 결과 (4 variants)

| Variant | strict F1 | strict P | strict R | relaxed F1 | 학습 시간 |
|---|---:|---:|---:|---:|---:|
| v2 (이전 baseline_corrected) | 0.9249 | 0.9217 | 0.9281 | 0.9415 | 95s |
| **v3 (data 보정 only)** | **0.9346** | 0.9395 | 0.9298 | **0.9498** | 95s |
| v3+CRF | 0.9311 | 0.9308 | 0.9313 | 0.9469 | 340s (+3.6x) |
| **v3+boundary** ⭐ | **0.9368** | 0.9334 | 0.9402 | **0.9505** | 100s |
| v3+combined (NER cw + boundary) | 0.9347 | 0.9242 | 0.9454 | 0.9480 | 99s |

#### per-entity (v3+boundary, production 후보)

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

#### 효과 분석

| 단계 | strict Δ | 단일 최대 효과 | 비고 |
|---|---:|---|---|
| data 보정 (v2→v3) | **+0.97pp** | LOC +2.93 / ORG +1.55 / PROD +1.43 | 전체 단일 최대 효과 |
| +CRF (v3→v3+CRF) | -0.35pp | LOC -2.25 / EVT -3.94 (회귀) | NER 회귀, PII 회복 |
| +boundary (v3→v3+boundary) | +0.22pp | LOC +1.31 / PER +0.67 | NER 5종 전체 안정 향상 |
| +combined (v3→v3+combined) | +0.01pp | LOC +2.23 / PROD +1.47 (BUT DAT -3.49) | NER class wt 가 PII 회귀 |

> ⚠️ **사후 재평가 (#69) 관점**: 본 표의 +boundary (+0.22pp)·+combined
> (+0.01pp) 같은 1pp 미만 차이는 단일 분할 측정 노이즈 (±4pp) 안에
> 완전히 묻힌다. 당시엔 "향상" 으로 보고 v3+boundary 를 채택했지만, 이
> 프로토콜로는 진짜 효과인지 seed 운인지 가릴 근거가 없었다 (§본 문서
> 동결).

### 종결 결정 — (B) 천장 동결

이슈 #51 시점: data quality + CRF + boundary + class weight 시도 후에도
ORG/LOC/PROD/EVT 는 strict 0.85~0.94 천장. 사용자 결정 →
**(B) v3+boundary 채택** (strict 0.9368 / relaxed 0.9505).

> **Tier 2.B Self-training 미시도**: JA Wikipedia unlabeled pseudo-label
> + Gaussian threshold + cross-verifier 트랙은 Phase 2 의 augmentation
> 8/8 회귀 패턴 재현 위험 + PROPOR 2026 의 -0.24~-1.81% 회귀 caveat 으로
> *미시도 종결*. 향후 별도 이슈 후보로 보존.

### 잔여 한계 → 다음 단계 가설

v3+boundary 채택 (strict 0.9368 / relaxed 0.9505) 후에도 LOC/ORG/PROD/
EVT 가 0.85~0.94 구간. 이 단계까지는 **"어떤 오류가 얼마나 남았나"를
정확히 보지 않고** data 정제·손실 함수 변경 같은 broad fix 만 했다.
다음은 **잔여 오류의 패턴을 정밀 진단해 표적 시도** 로 전환.

---

## Phase 5 — 잔여 오류 정밀 진단 (#56, 라운드 3 / S0)

### 직전 가설

추가 개선을 위해 broad fix 가 아닌 표적 처방이 필요. test 오답을
5종 카테고리로 분해해 단일 최대 leak 표적을 찾는다.

### 본 단계에서 시도

- `error_analysis.py` 확장: `build_confusion_matrix`,
  `top_errors_by_type`, `--with-diagnosis` 추가 (BC 유지)
- test 527 문장 inference + 5종 오답 카테고리 분류:
  - EXACT (정답 일치)
  - BOUNDARY (종류 맞음, 위치 다름)
  - TYPE_MISMATCH (위치 겹침, 종류 다름)
  - MISS (정답 있는데 모델 미반응)
  - HALLUCINATION (정답 없는데 모델이 entity 부여)

### 결과 — 오답 5종 분류 + confusion matrix

먼저 test 527 문장의 전체 오류를 5종으로 집계 (정답 1,922 / 예측
1,936 / 정답 일치 EXACT 1,807):

| 구분 | 합계 | 환각 | 경계 어긋남 | 누락 | 종류 혼동 |
|---|---:|---:|---:|---:|---:|
| 놓침 (FN) | 115 | — | 55 (48%) | 46 (40%) | 14 (12%) |
| 잘못 짚음 (FP) | 129 | 60 (47%) | 55 (43%) | — | 14 (11%) |

> FN 은 정답을 *못 맞힌* 것이라 환각이 없고, FP 는 *없는 걸 만든* 것
> 이라 누락이 없다 (정의상 빈칸). 본 집계는 NER 5종 + PII 5종 전체 +
> 경계 어긋남까지 포함한다.

이 중 **종류 혼동** 만 떼어 "무엇을 무엇으로 헷갈렸나" 를 보는 게 아래
confusion matrix (NER 5종 only, *경계 어긋남·PII 는 제외*). 행 = 정답,
열 = 예측, ∅ = 없음:

| 정답＼예측 | PER | LOC | ORG | PROD | EVT | ∅ |
|---|---:|---:|---:|---:|---:|---:|
| PER | 373 | - | 2 | - | - | 2 |
| LOC | 1 | 294 | 4 | - | - | 12 |
| ORG | 1 | - | 545 | 2 | 1 | 17 |
| PROD | - | - | - | 92 | - | 3 |
| EVT | - | - | - | 1 | 83 | 4 |
| **∅ (환각)** | 3 | 7 | **28** | 6 | 7 | - |

> 위 두 표의 범위 차이: confusion matrix 의 ∅행 (NER 5종 환각) 을
> 더하면 51, ∅열 (NER 5종 누락) 을 더하면 38 이다 (표 직접 합산).
> FP 표의 환각 60 · FN 표의 누락 46 은 여기에 PII 5종 몫이 더해진
> 전체값이라 더 크다. 경계 어긋남 55 는 위치만 어긋난 오류라 type
> 행렬 (confusion matrix) 에는 아예 표시되지 않는다.

핵심 발견:
- 잘못 짚음 (FP) 129 = **환각 60 (47%) + 경계 어긋남 55 (43%) + 종류
  혼동 14 (11%)**. 환각 + 경계 어긋남 = 잘못 짚음의 88%
- 종류 혼동 ≤ 12% → **"ORG·PROD 헷갈림"이 본질이 아님**
- **ORG 환각 28건 (전체 환각의 47%) = 단일 최대 leak 표적**
- LOC 놓침 29 = 누락 (MISS) 12 + 경계 어긋남 (BOUNDARY) 12 + 종류
  혼동 (TYPE_MISMATCH) 5. 위 confusion matrix 의 LOC 행에서 직접
  보이는 건 누락 12 (∅ 열) 와 종류 혼동 5 (→ORG 4 + →PER 1) 뿐 —
  경계 어긋남 12 는 type 행렬에 안 잡히는 별도 카테고리다. 종류 혼동
  5건 중 4건이 외국 지명을 ORG 로 오분류 (`モーデン` 등)

### 진단으로 얻은 우선순위

진단 결과를 기반으로 후속 시나리오 재정렬:
- 종류 분리 학습 (S3) ROI 낮음으로 강등
- 신규 **S7 (환각 부정 예시 oversample)** 와 **S6 (경계 규칙 명문화)**
  도출
- 다음 작업 = **S1 (사후 calibration) + S7 (학습 데이터 보강)** 병렬

> ⚠️ **사후 재평가 (#69) 관점**: 이 진단 (오류 5종 분류·confusion
> matrix·"ORG 환각 28건 최대 leak") 은 전부 *그 단일 test 527 문장*
> 산물이다. 표본이 바뀌면 어느 entity 가 최대 leak 인지도 달라질 수
> 있어, 뒤이은 표적 처방 (S7·S8) 이 그 분할에 과적합됐을 위험을 배제할
> 수 없다. 다만 "ORG 가 최대 기여 오류" 라는 결론 자체는 층화 K-fold
> pooled (ORG support 28% / F1 0.8893) 에서 재확인됐다 (2편 §해석).

---

## Phase 6 — 사후 calibration → 학습 측 처방으로 전환 (#57, 라운드 3 / S1)

### 직전 가설

S0 진단으로 PROD/EVT/ORG 가 over-predict (precision 낮음) 확인 →
학습 없이 사후 threshold 만 올려도 정밀도 회복되는지 즉시 측정.

### 본 단계에서 시도

valid 셋의 per-class confidence 분포에서 strict precision ≥ 0.95 충족
최저 threshold 탐색. valid-only vs train+valid 두 calibration 비교.

### 결과

| calibration | PROD threshold | EVT | ORG | test 결과 |
|---|---:|---:|---:|---|
| valid-only (84 PROD pred) | 0.9882 | 0.7855 | 0.7344 | PROD recall **0.34** (붕괴) |
| train+valid (~850 PROD) | 0.6951 | 0.5895 | 0.6287 | overall F1 +0.41pp, ORG/EVT P ≥ 0.95 도달, PROD R 0.82 |

핵심 발견:
- threshold 는 *예측을 거름* → recall 무조건 ↓ (또는 동등). precision↔
  recall 트레이드오프는 사후 보정으로 해소 불가
- 게이트 정의 (P AND R ≥ 0.95) 와 **구조적 부합 불가능**
- valid-only 표본 부족 (PROD valid ~28건) → threshold 과대추정 → test
  일반화 실패

**결정**: #57 미시도 종결. P/R 동시 천장은 *데이터 측 처방* 으로만 도달
가능 확정 → S7 진행. asymmetric focal loss 2단계도 동일 한계 (학습 측
calibration) → 시도 안 함. **P/R 동시 개선은 학습 데이터 자체에 부정·
도메인 신호 추가로만 가능** → Phase 7 진입.

---

## Phase 7 — "이건 entity 아님" 예시를 학습에 추가 (#58, 라운드 3 / S7)

> 정식 명칭: 환각 부정 예시 oversample (= S7). production 을
> v3+boundary → **v3+boundary+S7N2** (strict 0.9368 → 0.9624) 로 갱신.

### 한 줄 요약

모델이 자꾸 entity 라고 *우기는* 표현들을, 학습 데이터에서 "이건
entity 아님" 으로 콕 집어 더 많이 보여줘서 환각을 줄였다. 그 결과
ORG/PROD/EVT 의 정밀도가 크게 올랐다.

### 직전 가설

Phase 5 진단에서 가장 큰 문제는 **환각** (정답에 없는데 모델이
entity 라고 만들어낸 것) 이었고, 그 중 ORG 환각 28건이 최대였다.
모델은 짧은 영어 약어 (`PC`, `GM`), 단일 한자 (`米`, `局`), 흔한
일반 명사 (`工場`, `海軍`) 를 자꾸 entity 로 착각한다. 학습 데이터가
"이런 건 entity 가 아니다" 라는 신호를 충분히 주지 않았기 때문 →
그 신호를 일부러 주입하면 환각이 줄 것이다.

### 본 단계에서 시도

- 신규 모듈 `ja_negative/` (이후 `ja/negative/` 로 이전): 환각 seed
  추출 → ambiguous 제외 → 후보 문장 찾기 → N배 복제 출력
- classifier 에 `--data-extra-train-jsonl` 옵션: 추가 문장을 train
  에만 합치는 leak-free 경로
- 5가지 버전을 비교 (무엇을·얼마나 넣을지 조합 바꿔가며)

### 결과 (5 버전 비교)

| 버전 | 무엇이 다른가 | seed | N | strict F1 | 환각 잔존 | 게이트 10종 |
|---|---|---|---:|---:|---:|---:|
| (기준) v3+boundary | 부정 예시 없음 | - | - | 0.9368 | 60 | 5 |
| v1 | 환각 표현 전부 | 57 | 3 | 0.9597 | 32 | 6 |
| v2 | ambiguous 제외 | 49 | 3 | 0.9552 | 22 | 6 |
| **v3 (채택, =S7N2)** | ambiguous 제외 | 49 | **2** | **0.9624** | 30 | **7** |
| v4 | NER 환각만 (PII 제외) | 41 | 3 | 0.9572 | 31 | 5 |
| v5 | PROD 환각 빼고 | 35 | 2 | 0.9525 | 32 | 4 |

> "환각 잔존" = 60건이던 환각이 학습 후 몇 건 남았나. "게이트 10종" =
> 10개 라벨 중 목표선에 도달한 개수. v3 가 F1·도달 둘 다 최고라 채택
> (= S7 의 N2 버전 → S7N2).

#### per-entity (v3+boundary+S7N2, 갱신된 production)

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

> ⚠️ **사후 재평가 (#69) 관점**: 본 단계의 negative extra 는 후보
> 문장을 *전체 행 (test 포함)* 에서 골랐다. extra 자체는 설계대로
> train 에만 합쳐졌지만, 그 문장 중 일부가 test 문장의 *복제본* 이라
> 결국 같은 문장이 train·test 양쪽에 존재 = 누설. "leak-free" 라던
> 가정이 후보 선정 단계에서 깨져 있었다. 이 누설이 Phase 8 까지 누적돼
> 0.9624·0.9644 를 약 +1.7pp 부풀렸다 (§본 문서 동결).

### 핵심 발견 (S7 검증)

- **N=2 가 N=3 보다 낫다** — 너무 많이 넣으면 진짜 entity 도 회피.
  적게 넣어도 환각은 충분히 잡힌다
- **ambiguous 제외가 필수** — 빼지 않으면 진짜 entity 까지 못 잡는 회귀
- **PROD 환각을 일부러 빼봤더니 (v5) PROD 가 오히려 나빠짐** =
  ripple (파급) 효과. 한 라벨용 부정 예시가 *다른 라벨 판단에도* 도움을
  준다는 뜻 → 환각 표현은 통째로 넣는 게 낫다
- **PII 환각만 빼봐도 (v4) 차이 없음** — PII 환각 표현은 어차피 무해

효과 (채택본 v3): ORG 정밀도 **+5.5pp** / PROD **+12.6pp** / EVT
**+9.8pp** / ID_NUM +6.3pp / LOC 재현율 +4.8pp.

### 잔여 한계 → 다음 단계 가설

환각 (잘못 만들어내는 문제) 은 크게 잡혔지만, PROD recall 87/95
(0.916) 가 여전히 부족하다. 이건 *있는 PROD 를 놓치는* 반대쪽 문제라
부정 예시로는 못 고친다. PROD 를 어떤 상황에서 놓치는지 들여다봐야 →
Phase 8.

---

## Phase 8 — PROD 가 놓치는 도메인을 콕 집어 더 보여주기 (#61, 라운드 4 / S8)

> 정식 명칭: PROD 도메인 seed oversample + PHONE 다양화 (= S8).
> production 을 S7N2 → **S8 (strict 0.9624 → 0.9644)** 로 갱신.

### 한 줄 요약

Phase 7 로 환각 (있지도 않은 entity 를 만들어내는 문제) 은 잡았지만,
모델이 *진짜 있는* PROD 를 자꾸 놓쳤다. 알고 보니 놓치는 PROD 가
전부 특정 분야 (법률·책·음식·교통카드·음악) 에 몰려 있었고, 학습
데이터에 그 분야 예시가 거의 없었다. 그래서 그 분야 문장만 골라
더 보여줬더니 PROD 를 덜 놓치게 됐다. 덤으로 전화번호 생성기도
실제처럼 다양화했다.

### 직전 가설

PROD recall 87/95 (0.916) — *진짜 있는 PROD 의 8건을 놓치는* 문제가
남았다. 그 8건의 정체를 보니:

| 놓친 PROD | 분야 |
|---|---|
| `連邦制定法`, `外国人地方参政権付与法案` | 법률 |
| `米子商工案内` | 책·문헌 |
| `玉すだれ`, `食べ比べセット` | 음식 상품 |
| `ICOCA` | 교통카드 |
| `たたえられよ、サラエヴォ` | 음악·영상 작품 |
| `CAP gauge` | 영문 모델명 |

공통점: 이 분야들이 학습 데이터에 거의 없다 (각 1~5건). 모델이
*배운 적 없는 분야* 라 못 잡는 것. → 그 분야 예시를 채워주면 잡을
것이다. random PROD-positive oversample 은 surface 분포가 famous
media·software 에 편중되어 test FN 과 substring overlap **0/8** →
천장 그대로.

### 본 단계에서 시도

- 신규 모듈 `src/ner/augmenters/ja/prod_seed/`: `DOMAIN_PATTERNS`
  (law/book/food/transit_card/music_work 5종) + `select_domain_seed_
  indices` (train+valid PROD-pos 정밀 선별, leak-free) +
  `select_long_seed_indices` (LONG surface 보조 풀, famous media 제외)
  + CLI (`--base-extra` 로 S7N2 negative extra prepend, `--oversample N`
  seed-major 출력). 분야 문장 59개를 N=5 복제, Phase 7 의 부정 예시와
  한 파일로 합쳐 주입
- `ja_negative/` → `ja/negative/` 로 옮겨 `ja/` 한 폴더로 통합
- `pii/generators/ja.py` PHONE 생성기 4 카테고리 + 구분자 변형으로 확장
- **공정 비교 장치**: non-determinism 때문에 baseline 과 S8 을
  *각각 5 seed* 로 학습해 중앙값으로 맞붙임

### 학습 non-determinism 발견

byte-identical jsonl + 동일 seed=42 + 동일 hyperparameter 인데도 PROD R
이 0.937 → 0.910 → 0.895 식으로 흔들림. CUDA/cuDNN non-determinism
영향. **1회 학습 결과로 게이트 판정 불가** → multi-seed (5 seeds) median
비교 채택.

### 결과 — 5 seed 중앙값으로 공정 비교 (둘 다 phonediv data)

| setup | F1 median | F1 std | PROD R median | PROD R std | PROD R range | PROD P median |
|---|---:|---:|---:|---:|---|---:|
| (기준) S7N2 baseline (neg only) | 0.9499 | 0.0085 | 0.8901 | **0.0788** | [0.747, 0.947] | 0.9000 |
| **S8 prod_seed (neg + domain N=5)** | **0.9634** | 0.0062 | **0.9053** | **0.0332** | [0.857, 0.937] | 0.8889 |

S8 이 **F1 +1.4pp / PROD R +1.5pp** 우월한데, 그보다 더 중요한 건
**PROD R 분산 -58%** (0.079 → 0.033) — *점수가 훨씬 덜 흔들린다.*

#### per-seed PROD R 비교 (같은 seed 끼리 baseline vs S8)

| seed | 기준(S7N2) | S8 | 차이 |
|---:|---:|---:|---:|
| 42 | 0.9263 | 0.9368 | +0.011 |
| 43 | 0.9468 | 0.8936 | -0.053 |
| 44 | 0.8901 | 0.8571 | -0.033 |
| **45** | 0.8511 | **0.9362** | **+0.085** ⭐ |
| 46 | **0.7474** | 0.9053 | **+0.158** |

핵심은 *바닥* 이 올라간 것. 기준선은 운 나쁜 seed 에서 0.747
(95개 중 71개만 맞힘) 까지 추락했는데, S8 의 최저점은 0.857 로
받쳐졌다. = 분야 예시가 "운 나쁜 학습" 의 바닥을 메워준다.

#### per-entity median 변화 (S8 - baseline)

| Entity | base P | S8 P | ΔP | base R | S8 R | ΔR |
|---|---:|---:|---:|---:|---:|---:|
| PER | 0.965 | 0.972 | +0.7 | 0.978 | 0.979 | +0.1 |
| LOC | 0.938 | 0.953 | +1.5 | 0.970 | 0.963 | -0.7 |
| ORG | 0.922 | 0.941 | +1.9 | 0.948 | 0.951 | +0.3 |
| PROD | 0.900 | 0.889 | -1.1 | 0.890 | 0.905 | +1.5 |
| EVT | 0.924 | 0.921 | -0.3 | 0.922 | 0.921 | -0.1 |
| DAT | 0.980 | 0.990 | +1.0 | 0.980 | 0.970 | -1.0 |
| ID_NUM | 0.977 | 0.988 | +1.1 | 0.981 | 0.988 | +0.7 |
| CREDIT_CARD | 0.900 | 0.988 | **+8.8** | 0.976 | 0.968 | -0.8 |
| EMAIL/PHONE | 1.000 | 1.000 | 0 | 1.000 | 1.000 | 0 |

→ 거의 모든 entity 에서 미세 개선. PROD P 만 -1.1pp 작은 회귀.
**CREDIT_CARD P +8.8pp** 가 가장 큰 부수 효과 (PHONE 다양화의 ripple
가능성 추정 — 별도 검증 필요).

### Production 채택

**`results/classifier/ja_sweep/s8_prod_domain_N5_seed45/best/`**
- F1 = 0.9644, PROD R = 0.9362. *overall F1* 기준 최선 seed 로 선택
  (PROD R 기준이면 seed 42 가 더 높다)
- best seed 단일값은 분산 안의 한 draw 이고, 안정 도달선은 *중앙값*
  PROD R 0.9053 (§데이터 천장 확정)

### 데이터 천장 확정

- S8 median PROD R 0.9053 (std 0.033), S7N2 baseline median 0.8901
  (std 0.071) — S8 가 median +1.5pp · 분산 -58%
- 본 데이터·모델 셋업에서 PROD R 의 안정 도달선 (median) 은 0.9053.
  best seed 의 0.9362 는 분산 안의 단일 draw

> ⚠️ **사후 재평가 (#69) 관점**: 여기서 확정한 headline 0.9644 와 천장
> 수치는 Phase 7~8 의 train/test 누설에 가장 크게 오염된 값이다. 같은
> production checkpoint (S8 seed 45) 를 누설 기준으로 쪼개면 — 누설
> 구간 187 문장 F1 0.9947 (EVT/ORG/PER 전부 1.0 = 암기), 비누설 340
> 문장 0.9478 (게이트 미달). 즉 0.9644 의 천장은 *암기로 부풀려진 상한*
> 이고, 누설·노이즈를 제거한 청정 기준선은 strict F1 0.9195 다 (§본
> 문서 동결, 2편 §해석).

---

## 본 문서 동결 — 측정 프로토콜 개편 (#69)

Phase 0~8 의 모든 측정은 80/10/10 단일 분할 (test 527 문장) 기준이다.
재분할 검증 실험 (#69) 에서 이 프로토콜의 구조적 한계가 확인되었다:

1. **측정 노이즈가 분산을 지배** — test 527 문장 (희소 entity ~94 span)
   에서 seed 간 분산의 72~100% 가 표본추출 노이즈. ±4pp 이내의
   개선/회귀를 단일 분할 측정으로 판별할 수 없다
2. **extra train 데이터의 train/test 중복** — Phase 7~8 extra 에 test
   문장의 30~35% 가 원문 그대로 포함되어 측정값이 부풀려졌다
   (중복 부분 F1 0.9947 vs 비중복 부분 0.9478 = 게이트 미달)

### 결함 1 뒷받침 수치 — 분할 재추출 노이즈

`--seed` 는 *데이터 분할* 만 바꾼다 (학습 seed 는
`TrainingArguments(seed=42)` 로 고정 — `train_eval.py`). base 단독으로
분할 seed 만 42~46 으로 돌려 5회 재측정한 결과:

| 분할 seed | 42 | 43 | 44 | 45 | 46 | median | std | 폭 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall strict F1 | 0.9249 | 0.9292 | 0.9095 | 0.9181 | 0.9198 | 0.9198 | ±0.0075 | 1.97pp |

*어느 527 문장이 test 냐* 만 바뀌어도 overall 이 1.97pp, 희소 entity 는
**PROD 10.3pp / EVT 7.5pp** 폭으로 흔들린다. 분산 분해 결과 이 변동의
**72~100% 가 표본추출 노이즈** (희소 entity ~94 span) — 데이터 편향이
아니라 *분할 운(運)* 이다. 여기에 학습 seed 를 고정해도 CUDA/cuDNN
non-determinism 이 더해져, *같은 분할을 재학습만 해도* PROD R 이
0.937 → 0.910 → 0.895 로 흔들린다 (Phase 8) — 두 번째 노이즈 바닥.

→ 후반 Phase 의 개선폭 (Phase 4 +boundary +0.22pp / +combined +0.01pp
급) 은 이 노이즈 (overall 1.97pp, PROD 10.3pp) 보다 작아, 단일 분할
측정으로는 *개선인지 분할 운인지* 가릴 수 없다.

### 이 결함이 전체 실험에 의미하는 것

두 결함은 "마지막 숫자 하나가 약간 틀렸다" 가 아니라 **Phase 0~8 서사
전체의 해석을 흔든다.**

- **후반 미세 개선의 상당수가 측정 불가 영역이었다.** Phase 4 의
  +boundary (+0.22pp)·+combined (+0.01pp) 같은 1pp 미만 차이는 ±4pp
  노이즈 안에 완전히 묻힌다. 채택은 했지만 *진짜 효과인지 seed 운인지
  구분할 근거가 없었다.* byte-identical 데이터·동일 seed 인데도 Phase
  8 의 PROD R 이 0.937→0.895 로 흔들린 관측이 그 방증이다.
- **headline 0.9644 는 누설로 부풀려진 값이다.** 가장 큰 상승 구간
  (Phase 7~8, 0.9368 → 0.9644) 에 바로 test 누설이 끼었다. 누설을
  걷어낸 비중복 실측 0.9478 은 production 게이트 (0.95) 에 *미달* —
  "게이트를 넘겼다" 던 Phase 7~8 의 결론 자체가 측정 아티팩트였다.
- **진단→처방 사이클의 토대도 같은 분할 위에 있었다.** Phase 5 의
  오류 5종 분류·confusion matrix·"ORG 환각 28건 최대 leak" 진단은 모두
  그 단일 test 527 문장 산물이다. 표적 처방 (S7·S8) 이 그 분할에
  과적합됐을 위험이 있다 (단, ORG 가 최대 기여 오류라는 결론은 K-fold
  pooled 에서 재확인됨 — 2편 §해석).

### 왜 측정 프로토콜을 갈아엎어야 했나

위 문제는 hyperparameter 를 더 만지거나 데이터를 더 보정해서 풀리지
않는다 — *측정 도구 자체가 ±4pp 를 못 가르고 누설로 점수를 부풀리는*
한, 어떤 개선도 "진짜 올랐다" 고 말할 수 없기 때문이다. 그래서 다음
단계는 새 기법이 아니라 **측정의 신뢰성 회복** 이어야 했다:

- 단일 분할 → **층화 K-fold 교차 검증**: 전체 5,270 문장을 돌아가며
  test 로 써서 표본추출 노이즈를 평균낸다 (overall ±0.63pp → ±0.28pp,
  PROD ±4.10pp → ±1.28pp 로 감소)
- extra train 의 **누설 제거** 후 비중복 기준으로 재측정

이 전환으로 얻은 청정 기준선 (base 단독, extra 없음·누설/노이즈 제거)
은 strict F1 **0.9195** (게이트 -3.05pp) 다. headline 0.9644 와의 약
4.5pp 간극은 *전부 거품이 아니라* extra 의 진짜 기여 (+2.5~3pp, 비중복
0.9478 기준) + 누설 인플레이션 (~+1.7pp) + 프로토콜 차이로 분해된다 —
즉 부풀린 몫만 걷어내도 production 일반화는 게이트 아래다. *본 문서의
핵심 교훈은 "0.9058 → 0.9644 로 올렸다" 가 아니라, 그 개선의 상당
부분이 측정 방식의 결함에 가려져 검증 불가능했다는 것 — 그래서 개선
보다 먼저 신뢰할 수 있는 측정자를 세우는 일이 필요했다는 것이다.*

따라서 본 문서의 수치는 *같은 프로토콜 안에서의 상대 비교*로만 유효하다.
이후 실험은 층화 K-fold 교차 검증 프로토콜로 전환하여
`japanese-bert-classifier-history-2.md` (2편) 에 기록하며,
본 문서에는 새 Phase 를 추가하지 않는다 (Phase 0~8 동결).
