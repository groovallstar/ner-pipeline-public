# 베트남어 NER silver 데이터 품질 검증 리포트

> WikiANN-vi 인간 주석 3-gold(PER/LOC/ORG)와 LLM 라벨러 5-종 silver
> (PER/LOC/ORG/PROD/EVT)를 비교해 silver 의 production 학습 활용 정당성을
> 정량 측정한다. 동시에 cross-model agreement(Gemma+Qwen kappa)와
> Wikidata 앵커 외부 검증으로 PROD/EVT(직접 gold 없음)의 신뢰도를 추정한다.

**대상 데이터셋**: HuggingFace `unimelb-nlp/wikiann/vi` train/validation/test (각 20K/10K/10K)
**라벨러**: `cyankiwi/gemma-4-31B-it-AWQ-8bit` (primary, 8081) + `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` (validator, 8082, MoE 활성 ~3B)
**평가지표**: 문자 offset span F1 (`metrics/span_metrics.compute_offset_span_f1`)
**라벨 스키마**: PER · LOC · ORG · PROD · EVT (canonical 5종)

---

## 0. 측정 구조 및 한계 (먼저 읽을 것)

### 0.1 측정 대상

```
[입력 문장: WikiANN-vi 40K]
        │
        ├─ LLM 라벨러 (5-종 PER/LOC/ORG/PROD/EVT) ──→ silver
        │
        └─ WikiANN-vi 인간 주석 BIO (3-종 PER/LOC/ORG) ──→ gold
                                                       ↓
                       비교: silver 에서 PER/LOC/ORG span 만 필터링 후
                            gold 와 strict span match (start, end, type)
```

### 0.2 본 측정의 핵심 한계

| 한계 | 영향 받는 type | 의미 |
|---|---|---|
| **gold 가 3종만 보유** | PROD, EVT | 직접 F1 측정 불가능 — 간접 신호로만 추정 |
| **gold 와 silver 의 LOC/ORG 정의 불일치** | LOC, ORG | gold 는 시설=LOC, silver 는 시설=ORG. type mismatch 가 F1 ↓ 의 일부 원인 |
| **gold 자체가 자동 생성 silver** | 모든 type | WikiANN 은 인간 검수 일부만 들어간 자동 라벨링. 절대 정답이 아님 |
| **PER 만 정의가 일치** | PER | 본 측정에서 silver 라벨러의 절대 성능을 직접 측정 가능한 유일한 type |

### 0.3 따라서 본 리포트의 결론은 type 별로 신뢰 수준이 다르다

- **PER F1**: 라벨러 절대 성능의 직접 지표 — 신뢰 가능
- **LOC/ORG F1**: 라벨러 성능 + 스키마 차이가 섞인 값 — *모델 간 상대 비교*만 의미 있음
- **PROD/EVT**: 직접 F1 불가 — cross-model agreement + Wikidata anchor 으로 간접 추정

---

## 1. 파이프라인 구조

### 1.1 라벨링 환경

| 항목 | 값 |
|---|---|
| 데이터셋 | `unimelb-nlp/wikiann` config=`vi`, splits=`train`/`validation`/`test` |
| 전체 크기 | train 20,000 + validation 10,000 + test 10,000 = 40,000 |
| 모델 A (primary) | `cyankiwi/gemma-4-31B-it-AWQ-8bit` @ :8081 (Dense 31B, AWQ-8) |
| 모델 B (validator) | `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` @ :8082 (MoE 활성 ~3B, AWQ-4) |
| 프롬프트 | `src/augmenters/wikiann_vi/prompts.py` SINGLE (~5K chars 포함 PROD/EVT 6규칙) |
| 요청 제어 | concurrency=16, temperature=0, max_tokens=2048, timeout=120s |
| max_model_len | 8192 |
| thinking | vLLM 컨테이너 기본 `enable_thinking=false` (둘 다) |
| 앵커 소스 | `vi.wikipedia.org/w/api.php` (pageprops) + `www.wikidata.org/w/api.php` (P31) |

CLI:
- 재라벨: `python -m augmenters.wikiann_vi`
- merge (정책별): `python -m augmenters.wikiann_vi.merge_confidence`
- Cross-model kappa: `python -m augmenters.wikiann_vi.kappa`
- Wikidata 앵커: `python -m augmenters.wikiann_vi.wikidata_anchor`
- silver vs gold span F1: `python -m llm_eval.vi_silver_quality`

### 1.2 두 라벨러 → 단일 silver 병합 (confidence 카테고리)

`merge_confidence.categorize_spans` 가 두 모델 출력을 비교해 4 카테고리로 분류:

| confidence | source | 조건 | 의미 |
|---|---|---|---|
| `high` | `both` | A·B 같은 offset + 같은 type | 양쪽 합의 (최고 신뢰) |
| `conflict` | `both_disagree` | 같은 offset, type 다름 | offset 동의, type 충돌 |
| `medium_recall` | `gemma_only` | A만 라벨 | Gemma 단독 (recall 보강) |
| `medium_prec` | `qwen_only` | B만 라벨 | Qwen 단독 (precision 보강) |

5 정책 (학습 데이터 선택 시 적용):

| 정책 | PER/LOC/ORG | PROD/EVT | test 데이터량 |
|---|---|---|---:|
| `recall` | high + medium_recall | high + medium_recall | 11,160 |
| `precision` | high + medium_prec | high + medium_prec | (적게) |
| `high_only` | high 만 | high 만 | (가장 적게) |
| `full` | 전체 (conflict 포함) | 전체 | 12,300+ |
| **`recall_strict`** | **high + medium_recall** | **high 만** | **10,669** |

### 1.3 최종 산출물 (`data/wikiann_vi/`, gitignore)

production 학습 데이터셋 (Stockmark 와 동일 포맷):

```
data/wikiann_vi/
├── train.jsonl              # 20,000 records, 21,379 entities
├── valid.jsonl              # 10,000 records, 10,580 entities
├── test.jsonl               # 10,000 records, 10,669 entities
└── wikidata_cache.json      # Q-ID + P31 API 캐시 (재실행 시 30~60분 절약)
```

레코드 스키마:

```json
{
  "id": "0",
  "text": "Đồng bằng sông Cửu Long",
  "entities": [
    {"label": "LOC", "start_char": 0, "end_char": 23, "text": "Đồng bằng sông Cửu Long"}
  ]
}
```

`recall_strict` 정책 통과한 entity 만 포함. drop 된 span 위치는 BIO 변환 시 `O` 로 처리. 다른 정책 (recall, precision, high_only) 또는 per-model 평가가 필요하면 §12 재현 명령으로 재생성.

---

## 2. 요약 — silver F1 vs WikiANN-vi 3-gold

### 2.1 단일 모델 silver

| silver | split | overall F1 | P | R | PER | LOC | ORG |
|---|---|---:|---:|---:|---:|---:|---:|
| **gemma** | test | 0.5652 | 0.5980 | 0.5359 | **0.749** | 0.384 | 0.552 |
| gemma | val | 0.5652 | 0.5978 | 0.5360 | **0.752** | 0.379 | 0.567 |
| gemma | train | 0.5601 | 0.5901 | 0.5330 | **0.757** | 0.373 | 0.551 |
| **qwen** | test | 0.5553 | 0.6364 | 0.4925 | **0.784** | 0.343 | 0.510 |
| qwen | val | 0.5574 | 0.6408 | 0.4933 | **0.791** | 0.338 | 0.530 |
| qwen | train | 0.5515 | 0.6318 | 0.4893 | **0.799** | 0.331 | 0.511 |

### 2.2 merge 정책 적용 silver (recall_strict)

| silver | split | F1 | PER | LOC | ORG |
|---|---|---:|---:|---:|---:|
| recall_strict | test | 0.5652 | 0.750 | 0.382 | 0.553 |
| recall_strict | val | 0.5651 | 0.752 | 0.378 | 0.567 |
| recall_strict | train | 0.5601 | 0.759 | 0.371 | 0.551 |

→ recall_strict 의 PER/LOC/ORG F1 이 gemma 와 거의 동일한 이유: PROD/EVT high-only 필터는 PROD/EVT span 만 줄이므로 PER/LOC/ORG 평가에 영향 없음.

### 2.3 type 별 F1 분석

| type | 정의 일치성 | F1 의 의미 | 권장 해석 |
|---|---|---|---|
| **PER** | gold = silver = "인물명" | 라벨러 절대 성능 | Qwen 0.78~0.80, Gemma 0.75~0.76 — 둘 다 production 임계 (0.85) 미달 |
| **LOC** | gold ⊃ silver (gold 가 시설 포함) | F1 ↓ 의 일부는 스키마 차이 | 절대 비교 무의미. 모델 간 비교: 두 모델 비슷 (0.34~0.38) |
| **ORG** | gold ⊂ silver (silver 가 시설 흡수) | precision 0.51~0.78 vs recall 0.39~0.46 → silver 가 광범위 추출 | 정의 차이로 R ↓ 가 자연. ORG 라벨러 보강의 정당화는 다른 메트릭으로 |

---

## 3. silver 분포 (5-종, 3-split)

Gemma 라벨링 결과 기준 (라벨링 성공률 ~86%, 에러 0건):

| split | 총 spans | PER | LOC | ORG | PROD | EVT |
|---|---:|---:|---:|---:|---:|---:|
| test | 11,307 | 4,134 | 2,823 | 2,659 | 1,308 | 230 |
| validation | 11,177 | 4,030 | 2,879 | 2,562 | 1,297 | 209 |
| train | 22,606 | 8,288 | 5,777 | 5,331 | 2,584 | 469 |

→ PROD/EVT 도 학습 데이터로 충분한 빈도 확보 (train PROD 2,584, EVT 469). 별도 합성 데이터 보충 불필요 (§9 참조).

---

## 4. cross-model agreement — PROD/EVT 신뢰의 간접 신호

직접 gold 가 없는 PROD/EVT 의 신뢰는 **두 독립 LLM (Gemma + Qwen) 의 합의율**로 추정한다. 합의율이 높을수록 무작위·환각 가능성이 낮음을 시사하지만, *공통 편향*은 잡지 못한다 (§10.2 참조).

### 4.1 silver 합의율 (per-type, 3-split 평균)

| type | 합의율 (%) | 해석 |
|---|---:|---|
| **LOC** | 77.5% | 두 모델이 지명 정의에 일관 |
| **ORG** | 74.7% | 시설 흡수 정의에 일관 |
| **PER** | 73.3% | 인물명 일관성 양호 |
| **EVT** | 59.4% | 신규 type, 합의 절반 수준 |
| **PROD** | 52.7% | 신규 type, 합의 절반 수준 (가장 낮음) |

### 4.2 Cohen kappa (3-split)

| split | kappa | po | pe |
|---|---:|---:|---:|
| test | **0.6461** | 0.7289 | 0.2341 |
| validation | **0.6419** | 0.7252 | 0.2326 |
| train | **0.6545** | 0.7349 | 0.2326 |

평균 kappa **0.6475** — Landis & Koch 기준 "substantial agreement" 영역(0.61~0.80)의 하단. moderate 와 substantial 경계.

### 4.3 coverage vs type 이중 구조

| 지표 | test | val | train |
|---|---:|---:|---:|
| Gemma span 수 | 11,307 | 11,177 | 22,606 |
| Qwen span 수 | 9,116 | 8,946 | 18,152 |
| 동일 offset 교집합 | ~9,000 | ~8,800 | ~17,800 |
| 동일 offset 타입 일치율 | ~97% | ~97% | ~97% |

**함의**: 두 모델이 *같은 span 을 entity 로 본 경우* 타입 선택은 ~97% 합의. kappa 가 0.65 에 머무는 주원인은 "무엇을 entity 로 볼지" 의 coverage 차이 — Qwen 이 보수적으로 빈 배열 반환하는 빈도가 높음.

---

## 5. Wikidata anchor — 외부 검증

silver 가 추출한 entity 의 표면형을 vi.wikipedia.org → Wikidata 로 조회해, P31(instance of) 이 가리키는 canonical type 과 silver 의 type 이 일치하는지 비교한다 (`augmenters/wikiann_vi/wikidata_anchor.py`).

### 5.1 결과 (per-split)

| split | total entities | mapped (Wikidata 매핑 가능) | overall agreement |
|---|---:|---:|---:|
| test | 11,307 | 6,904 (61.1%) | **96.07%** |
| validation | 11,177 | 6,776 (60.6%) | **96.37%** |
| train | 22,606 | 13,730 (60.7%) | **96.43%** |

### 5.2 per-type agreement

| type | mapped count (test/val/train) | agreement |
|---|---|---:|
| **PER** | 3,104 / 2,980 / 6,063 | **98.8~99.0%** |
| **LOC** | 2,061 / 2,064 / 4,184 | **98.2~98.8%** |
| **PROD** | 487 / 483 / 1,029 | **93.6~95.2%** |
| **EVT** | 76 / 57 / 158 | 84.8~89.5% |
| **ORG** | 1,176 / 1,192 / 2,296 | 86.2~87.7% |

→ silver 의 PROD/EVT 라벨이 (Wikipedia 에 등재된 entity 인 경우) **PROD 93~95%, EVT 85~89% 외부 검증 통과**. 이게 본 리포트의 PROD/EVT 신뢰 근거 중 가장 강한 신호.

### 5.3 매핑 테이블 구성 (~225 Q-ID 큐레이션)

`WIKIDATA_TO_CANONICAL` 은 5종 스펙의 대표 Q-ID + 실측 빈도 상위 unmapped Q-ID 를 기반으로 약 225종 Q-ID 큐레이션. 대상은 실측 빈도가 높고 진짜 엔티티인 것에 한정했으며, `Q4167410`(disambiguation), `Q16521`(taxon) 등 메타·비엔티티는 의도적으로 미매핑.

PROD/EVT 주요 매핑:
- PROD: `Q7366`(song), `Q838948`(work of art), `Q15709879`(anime), `Q1004`(comic book), `Q2188189`(musical work), `Q1107654`(OS), `Q24856`(film series), `Q1339864`(video game franchise), `Q11424`(film), `Q482994`(album), `Q21198342`(manga series) 등
- EVT: `Q40231`(election), `Q3306904`(peace agreement), `Q1768295`(uprising), `Q56019`(military campaign), `Q11514315`(historical period), `Q500834`(tournament instance), `Q198`(war), `Q131569`(treaty) 등

### 5.4 주요 불일치 패턴 (정성 분석)

**카테고리 A — LLM 오류, 앵커 정정**:
- `Becamex Bình Dương` (축구 클럽): LLM `CORP/PROD` → 앵커 `ORG` ✓
- `Hải Lăng Vương` (베트남 왕족): LLM `PROD` → 앵커 `PER` ✓

**카테고리 B — 앵커 오류, LLM 정정**:
- `Thư viện Quốc gia Pháp` (도서관): LLM `ORG` (스펙 올바름) → 앵커 publisher (P31 잘못 선택)
- `Her Morning Elegance` (노래): LLM `PROD` (올바름) → 앵커 `PER` (Wikipedia 리다이렉트가 가수 페이지로)

**카테고리 C — 진짜 모호**:
- `đế quốc La Mã Thần thánh` (신성 로마 제국): LLM `ORG/POL` vs 앵커 `LOC`. 역사 국가·정치체의 이중 본성

---

## 6. PROD/EVT 합성 보충 불필요

5종 silver 집계 결과:
- **PROD**: train 2,584 / val 1,297 / test 1,308 (학습 데이터로 충분)
- **EVT**: train 469 / val 209 / test 230 (희소하지만 최소 실용 수준)

→ 합성 보충 불필요. 재어노테이션만으로 두 타입 모두 확보. EVT 가 다른 type 대비 1/10 수준이지만, (a) Wikidata anchor agreement 85~89% 로 품질 관리 가능하고, (b) 합성 주입의 부작용(문맥 부자연성·학습 편향)을 감수할 만큼 부족하지는 않다.

---

## 7. WikiANN silver 오라벨 교정 부작용

LLM 재라벨 과정에서 WikiANN 의 silver 오라벨을 일부 교정하는 유익한 부작용 관찰:

- `Ốc móng tay` (조개류 요리): WikiANN gold = LOC → Gemma/Qwen 모두 skip (음식 = entity 아님)
- `Lãm` (인명): WikiANN gold = non-entity → Gemma PER 신규 탐지

다만 교정 규모를 정량화하려면 진짜 gold 수준의 수작업 평가가 필요 — 본 프로젝트 제약으로 수작업 검증 불가하므로 정성 언급에 한정. 본 리포트 §0 한계의 일부.

---

## 8. 한계 (Limitations) ⚠️

### 8.1 본 평가의 본질적 한계

1. **gold 자체가 silver 임**. WikiANN 은 자동 생성 (cross-lingual 링크) + 부분 인간 검수. 본 측정의 모든 절대 F1 값은 "silver vs silver" 임. 인간 inter-annotator agreement 가 없어 ceiling 을 모름.

2. **PROD/EVT 의 직접 F1 측정 불가**. gold 가 3종만이라 본 리포트의 PROD/EVT 신뢰 결론은 모두 *간접 신호* (cross-model agreement + Wikidata anchor) 기반. 직접 측정하려면 PROD/EVT 인간 주석 데이터가 필요.

3. **LOC/ORG F1 의 스키마 차이 분리 불가**. silver 의 LOC F1 0.34~0.38 은 (a) 라벨러 진짜 오류 + (b) gold 시설=LOC 를 silver 가 ORG 로 보낸 정의 차이가 합쳐진 값. 두 효과를 분리하려면 gold 를 5-종 정의에 맞게 변환 (수동) 하거나 silver 에서 ORG 시설을 LOC 로 fold 하는 schema-collapsed 비교가 필요 (본 작업 범위 외).

### 8.2 cross-model agreement 의 한계

1. **공통 편향 미감지**. Gemma 와 Qwen 이 같은 사전학습 코퍼스 편향을 공유하면, 둘이 *동시에 틀린 답에서 일치* 할 수 있다. 합의율은 정답률의 상한이 아니다.

2. **PER 합의율 73% 인데 PER F1 0.75~0.80**. 합의율과 F1 사이 단조 관계가 깨질 수 있음 — 합의된 span 도 gold 와 어긋날 수 있고, 비합의 span 중 한쪽이 gold 와 일치하는 경우도 많음. PROD/EVT 합의율 50~60% 가 "정답률 50~60%" 를 의미하지는 않는다 (실제 정답률은 더 높을 수도 낮을 수도 있음).

### 8.3 Wikidata anchor 의 한계

1. **mapped coverage 60%**. 베트남어 Wikipedia 에 페이지 없는 entity (지방 무명 인물·로컬 시설·소수 작품) 는 검증 불가. 매핑 안 된 40% 의 신뢰는 cross-model agreement 만으로 추정.

2. **agreement 96% 의 분모**. Wikidata 매핑된 6,904 개 (test) 중에서의 96% 임. 매핑 안 된 4,403 개의 신뢰는 모름.

3. **Wikidata redirect/disambiguation 노이즈**. "Galaxy" 가 Galaxy 우주 vs 삼성 Galaxy 로 redirect 될 수 있음. agreement 분자에 잘못 일치한 케이스가 섞여 있을 수 있음 (qualitative 샘플링 미실시).

4. **`Q11514315` (historical period) 는 EVT 가 아닐 수 있음**. 매핑 후 약 60건이 hit 하지만 "thời nhà Lý" 같은 시대명은 EVT(1회성 사건) 정의와 다름 — 후속 검토 필요.

### 8.4 Qwen 보수적 coverage

Qwen 의 entity 미탐지(skip) 비율이 높아 Gemma-only span 이 다수 (test 기준 ~3,200건). 이게 kappa 가 "moderate" 구간(0.61~0.65) 에 머무는 주원인. 동일 offset 타입 일치율(~97%) 은 매우 높으므로 *타입 스펙 자체는 견고*.

### 8.5 WikiANN-vi 도메인 한계

WikiANN-vi 는 Wikipedia 기사 첫 문장을 문장 단위로 분리한 데이터. 베트남어 자연 문장의 일반 분포를 대표하지 않을 가능성. silver 학습 모델이 다른 도메인(뉴스·SNS·법률 등)에 일반화될지는 별도 검증 필요.

---

## 9. 결론 — silver 의 production 학습 활용 정당성

### 9.1 type 별 신뢰 등급

| type | 직접 증거 | 간접 증거 | 신뢰 등급 |
|---|---|---|---|
| **PER** | gold F1 0.75~0.80 (Qwen 우위) | kappa 0.65, anchor 99% | **B (사용 가능)** — 0.85 임계 미달이지만 cross-model 합의 + anchor 검증 강함 |
| **LOC** | F1 0.34~0.38 (스키마 차이 포함) | kappa 0.65, anchor 98% | **B (사용 가능)** — 절대 F1 은 무의미, anchor 검증 매우 강함 |
| **ORG** | F1 0.51~0.55 (시설 흡수 정의) | kappa 0.65, anchor 86~88% | **B (사용 가능)** — anchor agreement 가 다른 type 보다 낮음, 시설 경계 모호 잔존 |
| **PROD** | gold 없음 | 합의율 53%, anchor 94% | **C (제한적 사용)** — 합의율 낮지만 anchor mapping 한 entity 는 매우 높은 정확도. **recall_strict 적용 권장** |
| **EVT** | gold 없음 | 합의율 59%, anchor 87% | **C (제한적 사용)** — PROD 와 유사 |

### 9.2 학습 데이터 정책 권장안

본 리포트 결과 기반:

1. **5-종 silver 학습 정책**: `recall_strict` 사용
   - PER/LOC/ORG: high + medium_recall (Gemma 측 광범위 커버리지)
   - PROD/EVT: high 만 (양쪽 합의)
   - 데이터량: train 21,379 spans (recall 22,314 대비 -4.2%)

2. **PROD/EVT 학습 신중 모드** (선택지)
   - 학습 시 PROD/EVT 의 weight 를 낮추거나
   - PROD/EVT 만 따로 모델로 학습 후 ensemble

3. **5-종 silver 만으로 production 모델 학습은 부족**
   - 절대 F1 가 0.55~0.57 (PER/LOC/ORG 평균) — production 임계 (0.85+) 한참 미달
   - 다만 본 측정의 한계 (§10) 를 감안하면 실제 production 성능은 더 높을 가능성이 큼
   - 권장: silver 로 사전학습 → 소량 인간 검수 라벨로 fine-tune

### 9.3 운영 제안 (주·보조 라벨러 분리)

- **Gemma = primary 재라벨러**. recall 넓고 라벨링 성공률 ~86% 로 높음
- **Qwen = validator**. MoE 활성 ~3B 로 raw 디코딩 속도는 빠르나 entity 인식에 보수적이라 성공률 ~64%. 둘 다 라벨한 span = "고신뢰 silver"
- Gemma-only span 은 "확장 silver" 로 분리 보관 가능. 학습 시 포함/제외 ablation 권장

### 9.4 후속 작업 제안

- **schema-collapsed 비교**: silver 의 ORG 시설 후보 (접두어 `Bệnh viện`, `Sân bay`, `Trường`, `Bảo tàng` 등) 를 LOC 로 되돌려 gold 와 다시 비교 → LOC/ORG 의 진짜 라벨러 능력 분리
- **인간 샘플링**: PROD/EVT 각 50건 무작위 샘플 → 사용자 ✓/✗ 판정 → 직접 정확도 산출 (anchor 보정)
- **`Q11514315` (historical period) 매핑 검토**: EVT 가 아닌 ORG 또는 LOC 가 적절한지 사례 검토
- **3rd LLM tiebreaker** (PROD/EVT conflict 만): GPT-5-mini 또는 Gemma 26B 에 conflict 만 보내 다수결
- **BERT 파인튜닝 + held-out 평가**: silver 로 학습한 모델을 별도 도메인 (뉴스·법률) 에 적용해 일반화 측정

---

## 10. 재현 명령

```bash
# 1) silver 재라벨링 (각 split, ~25분 test/val + ~50분 train per model)
for SPLIT in test validation train; do
  N=10000; [[ $SPLIT == train ]] && N=20000
  python -m augmenters.wikiann_vi --split $SPLIT --max-samples $N \
    --concurrency 16 --base-url http://localhost:8081/v1 \
    --model cyankiwi/gemma-4-31B-it-AWQ-8bit \
    --output data/wikiann_vi/gemma_${SPLIT}.jsonl
  python -m augmenters.wikiann_vi --split $SPLIT --max-samples $N \
    --concurrency 16 --base-url http://localhost:8082/v1 \
    --model cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
    --output data/wikiann_vi/qwen_${SPLIT}.jsonl
done

# 2) merge (recall_strict 권장), kappa, anchor
for SPLIT in test validation train; do
  python -m augmenters.wikiann_vi.merge_confidence \
    --gemma data/wikiann_vi/gemma_${SPLIT}.jsonl \
    --qwen  data/wikiann_vi/qwen_${SPLIT}.jsonl \
    --policy recall_strict \
    --output data/wikiann_vi/vi_wikiann_recall_strict_${SPLIT}.jsonl
  python -m augmenters.wikiann_vi.kappa \
    --a data/wikiann_vi/gemma_${SPLIT}.jsonl \
    --b data/wikiann_vi/qwen_${SPLIT}.jsonl \
    --json-out data/wikiann_vi/kappa_${SPLIT}.json
  python -m augmenters.wikiann_vi.wikidata_anchor \
    --input data/wikiann_vi/gemma_${SPLIT}.jsonl \
    --cache data/wikiann_vi/wikidata_cache.json \
    --json-out data/wikiann_vi/wikidata_anchor_${SPLIT}.json
done

# 3) silver vs gold span F1
for PREFIX in gemma qwen vi_wikiann_recall_strict; do
  python -m llm_eval.vi_silver_quality \
    --silver-dir data/wikiann_vi --silver-prefix $PREFIX \
    --splits test validation train \
    --output results/issue-30/silver_quality_${PREFIX}.json
done

# 4) 최종 정리 — recall_strict 만 train/valid/test 로 보존
cd data/wikiann_vi
mv vi_wikiann_recall_strict_train.jsonl train.jsonl
mv vi_wikiann_recall_strict_validation.jsonl valid.jsonl
mv vi_wikiann_recall_strict_test.jsonl test.jsonl
rm -f gemma_*.jsonl qwen_*.jsonl vi_wikiann_recall_*.jsonl
rm -f kappa_*.json wikidata_anchor_*.json
# wikidata_cache.json 은 보존 (재실행 시 네트워크 절약)
```

## 11. 참조

- 이슈: #30 (`feat/issue-30-vi-silver-quality-vs-3gold`), 선행 #10·#21·#23·#27
- 코드:
  - silver 평가: `src/llm_eval/wikiann_vi_gold.py`, `src/llm_eval/vi_silver_quality.py`
  - merge 정책: `src/augmenters/wikiann_vi/merge_confidence.py`
  - 프롬프트: `src/augmenters/wikiann_vi/prompts.py`
  - anchor: `src/augmenters/wikiann_vi/wikidata_anchor.py`
  - kappa: `src/augmenters/wikiann_vi/kappa.py`
  - 라벨링 CLI: `src/augmenters/wikiann_vi/__main__.py`, `src/augmenters/wikiann_vi/relabel_8type.py`
- 라벨 정의: `docs/manual/data/japanese-canonical-entity-schema.md`, `docs/manual/data/vietnamese-ner.md`
- 이슈 md: `docs/issues/issue-30-vi-silver-quality-vs-3gold.md`
- 테스트: `tests/augmenters/wikiann_vi/`, `tests/llm_eval/`
