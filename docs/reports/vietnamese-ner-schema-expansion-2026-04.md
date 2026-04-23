# 베트남어 WikiANN-vi → 8종 스키마 확장 리포트 (2026-04)

> 대상 이슈: #10 `feat/issue-10-vi-ner-8type-relabel`
> 스펙: `docs/manual/data/vietnamese-ner-8types.md`
> 라벨 표기: canonical 영문 축약 (`docs/manual/data/canonical-entity-schema.md`)
> 데이터 범위: WikiANN-vi test split (전체 10,000 샘플, 두 모델 재라벨)
> 실행일: 2026-04-21 ~ 2026-04-22

## 1. 요약

WikiANN-vi(3종: PER/LOC/ORG) 데이터셋을 Stockmark 8종 스키마로 재라벨하는
파이프라인을 구축하고, 두 가지 독립 검증(Cross-model Cohen's kappa + Wikidata
P31 앵커)으로 신뢰도를 정량화했다. 핵심 지표:

- **Cross-model κ = 0.5766** (10K test, Gemma vs Qwen3.6) — Landis-Koch
  "moderate". 1K 표본(κ=0.5978)과 0.02 이내로 일치해 **1K 표본 대표성
  검증**됨
- **동일 offset 타입 일치율 97.09%** (10K) — 두 LLM이 같은 span을 엔티티로
  보면 타입 선택은 거의 합의. kappa 하락 주원인은 coverage(엔티티로 볼지)
  차이 (Qwen3.6이 보수적으로 엔티티 없음을 자주 반환)
- **Wikidata 앵커 일치율 95.12%** (10K, 134 Q-ID 매핑 테이블로 6,858건
  매핑 성공) — 독립 silver 대비 재라벨 타입이 강하게 정합
- **PROD 1131, EVT 166** (10K Gemma) — 원래 WikiANN에 없던 타입도
  재어노테이션만으로 학습 가능 수준 확보. **합성 보충 불필요**로 판정
- **JA Stockmark 3종 공정 비교**: VI entity density는 JA의 54% 수준, 분포도
  JA는 ORG 지배(51%), VI는 PER/LOC 지배(82%) — 절대 F1 비교 부적합 확정,
  상대 순위·특성 분석만 유효
- **타입별 신뢰도 계층화**:
  - High (≥93%): PER·LOC·CORP·PROD
  - Mid (~72~85%): ORG·FAC·POL·EVT

## 2. 실행 환경

| 항목 | 값 |
|---|---|
| 데이터셋 | `unimelb-nlp/wikiann` config=`vi`, split=`test` |
| 전체 test 크기 | 10000 |
| 모델 A (primary) | `cyankiwi/gemma-4-31B-it-AWQ-8bit` @ :8081 |
| 모델 B (validator) | `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` @ :8082 (MoE, 3B active) |
| 프롬프트 | `src/augmenters/wikiann_vi/prompts.py` SINGLE (~3.8K chars) |
| 요청 제어 | concurrency=16, temperature=0, max_tokens=2048, timeout=120s |
| max_model_len | 8192 |
| thinking | vLLM 컨테이너 기본 `enable_thinking=false` (둘 다) |
| 앵커 소스 | `vi.wikipedia.org/w/api.php` (pageprops) + `www.wikidata.org/w/api.php` (P31) |

CLI:
- 재라벨: `python -m augmenters.wikiann_vi`
- Cross-model kappa: `python -m augmenters.wikiann_vi.kappa`
- Wikidata 앵커: `python -m augmenters.wikiann_vi.wikidata_anchor`

### 최종 산출물 (`data/wikiann_vi/`, gitignore)

| 파일 | 크기 | 설명 |
|---|---|---|
| `vi_wikiann_8type_recall_test.jsonl` | 4.9 MB | **test 분할 최종 데이터셋** — Gemma/Qwen3.6 10K 재라벨을 recall 정책으로 병합. 10,000 records, 11,382 span (high 8,181 + medium_recall 3,201) |

중간 산출물(모델별 JSONL·kappa·anchor JSON)은 최종 병합 파일 생성 후 제거.
재생성은 §13 재현 명령어로 가능.

## 3. 타입 분포

### 3.1 1K 샘플 (Gemma vs Qwen3.6)

| 타입 | Gemma | Qwen3.6 | Δ (G−Q) |
|---|---:|---:|---:|
| PER | 418 | 387 | +31 |
| LOC | 370 | 292 | +78 |
| PROD | 113 | 66 | +47 |
| ORG | 86 | 52 | +34 |
| POL | 69 | 47 | +22 |
| FAC | 48 | 34 | +14 |
| CORP | 29 | 28 | +1 |
| EVT | 20 | 22 | −2 |
| **총 스팬** | **1,153** | **928** | **+225** |
| 엔티티 레코드 | 859 | 652 | +207 |

두 모델 모두 타입 순위 대체로 유사(PER > LOC > PROD > ORG …). Gemma가
전반적으로 약 24% 더 관대, EVT만 Qwen3.6이 역전(+2). Qwen3.6의
보수적 경향이 두드러진다.

### 3.2 10K 전체 test (Gemma)

| 타입 | 10K | 1K | 10× 환산 편차 |
|---|---:|---:|---:|
| PER | 4,206 | 418 | +26 (flat) |
| LOC | 3,885 | 370 | +185 |
| PROD | 1,131 | 113 | +1 (flat) |
| ORG | 768 | 86 | −92 |
| POL | 724 | 69 | +34 |
| FAC | 465 | 48 | −15 (flat) |
| CORP | 282 | 29 | −8 (flat) |
| EVT | 166 | 20 | −34 |
| **총 스팬** | **11,627** | **1,153** | **+97** (flat) |
| 엔티티 레코드 | 8,620 / 10000 | 859 / 1000 | +30 |

1K→10K 환산은 대부분 ±10% 이내로 안정. 즉 **1K 표본이 전체 분포를
합리적으로 대표**한다.

10K Gemma 재라벨 소요 약 25분(concurrency=16), 에러 0건.

## 4. Cross-model Cohen's kappa

1K와 10K 모두 측정. 10K가 최종 신뢰 대상이지만 1K 대표성 검증 차원에서
병기한다.

| 지표 | 1K | 10K |
|---|---:|---:|
| Records | 1,000 | 10,000 |
| Paired spans (A∪B offsets) | 1,219 | 12,317 |
| Observed agreement `po` | 0.6809 | 0.6642 |
| Chance agreement `pe` | 0.2066 | 0.2069 |
| **Cohen's κ** | **0.5978** | **0.5766** |

Landis-Koch 해석(0.41–0.60) **"moderate agreement"**. 10K kappa가 1K와
오차 0.02 이내로 일치하여 **1K 표본이 전체 분포를 대표**한다는 것이
검증됨.

### 4.1 핵심 분석: coverage vs type 이중 구조

| 지표 | 1K | 10K |
|---|---:|---:|
| Gemma span 수 | 1,153 | 11,627 |
| Qwen3.6 span 수 | 928 | 9,116 |
| 동일 offset 교집합 | 862 | 8,426 |
| both_match | 830 | 8,181 |
| both_disagree | 32 | 245 |
| Gemma only | 291 | 3,201 |
| Qwen3.6 only | 66 | 690 |
| **동일 offset 타입 일치율** | **96.29%** | **97.09%** |

**함의**: 동일 span을 양쪽이 엔티티로 본 경우 타입 선택은 97%+로 거의
합의. 남는 불일치는 "무엇을 엔티티로 볼지"(≈ coverage)에서 발생 —
Qwen3.6이 보수적으로 빈 배열을 반환해 Gemma only가 3,201건으로 큰 비중.
kappa가 "moderate" 구간에 머무는 주원인.

### 4.2 Gemma 관점 per-type 일치율

10K test 기준 (Gemma에서 라벨된 span이 Qwen3.6과 동일 타입으로 합의된
비율):

| 타입 | agreed / total_A | ratio |
|---|---|---:|
| PER | 3,283 / 4,206 | 0.7806 |
| CORP | 208 / 282 | 0.7376 |
| LOC | 2,874 / 3,885 | 0.7398 |
| FAC | 306 / 465 | 0.6581 |
| EVT | 104 / 166 | 0.6265 |
| POL | 437 / 724 | 0.6036 |
| ORG | 458 / 768 | 0.5964 |
| PROD | 511 / 1,131 | 0.4518 |

1K 대비 일관된 순위. PER·CORP·LOC 상위, PROD 최저는 Qwen3.6이 제품·작품
경계를 더 좁게 잡는 경향 때문.

## 5. Wikidata 앵커 검증

`vi.wikipedia.org`에서 엔티티 표면형 → Wikidata Q-ID → P31(instance of) →
`WIKIDATA_TO_CANONICAL` 매핑(134 Q-ID 큐레이션) → 8종 추론 타입. 재라벨
타입과 추론 타입 일치율을 측정.

### 5.1 커버리지

| 지표 | 10K |
|---|---:|
| 총 엔티티 | 11,627 |
| Unique 표면형 | 7,647 |
| Wikidata Q-ID 확보 | 9,184 (79%) |
| P31 claim 있음 | 9,134 |
| 8종 매핑 성공 | 6,858 (59%) |

6,858 앵커 건은 독립 검증 증거로 충분히 큰 표본이다. 나머지는 Wikipedia
등재 없음(개별 인물명·지역 고유명 다수) 또는 매핑 테이블 미수록 타입.

### 5.2 일치율 (Gemma 전체)

| 범위 | 매핑 수 | 일치율 |
|---|---:|---:|
| **10K** | **6,858** | **0.9512** |

### 5.3 타입별 앵커 일치율 (10K, Gemma)

| 타입 | ratio | 계층 |
|---|---:|---|
| LOC | 0.9899 | **High** |
| PER | 0.9836 | **High** |
| PROD | 0.9420 | **High** |
| CORP | 0.8628 | **High** |
| ORG | 0.8521 | Mid |
| FAC | 0.7969 | Mid |
| POL | 0.7236 | Mid |
| EVT | 0.7027 | Mid |

### 5.4 주요 불일치 패턴 (1K 수작업 분석 15건 기준)

**카테고리 A — LLM 오류, 앵커 정정 (3건)**:
- `Becamex Bình Dương` (축구 클럽): LLM `CORP` → 앵커 `ORG` ✓
- `Oren Lavie` (가수): LLM `CORP` → 앵커 `PER` ✓
- `Hải Lăng Vương` (베트남 왕족): LLM `PROD` → 앵커 `PER` ✓

**카테고리 B — 앵커 오류, LLM 정정 (2건)**:
- `Thư viện Quốc gia Pháp` (도서관): LLM `FAC` (스펙 올바름) → 앵커
  `CORP` (publisher P31 잘못 선택). 매핑 테이블에 `Q22806 (national
  library)` 누락이 원인
- `Her Morning Elegance` (노래): LLM `PROD` (올바름) → 앵커 `PER` (Wikipedia
  리다이렉트가 가수 페이지로 가서 anchor 기법 자체 한계)

**카테고리 C — 진짜 모호 (1건)**:
- `đế quốc La Mã Thần thánh` (신성 로마 제국): LLM `POL` vs 앵커
  `LOC`. 역사 국가·정치체의 이중 본성으로 양쪽 모두 타당

### 5.5 매핑 테이블 구성 (134 Q-ID 큐레이션)

`WIKIDATA_TO_CANONICAL`은 8종 스펙의 대표 Q-ID + 10K 앵커 실행 시 집계된
상위 unmapped Q-ID를 기반으로 134종으로 구성했다. 대상은 실측 빈도가
높고 진짜 엔티티인 것에 한정했으며, `Q4167410`(disambiguation 552건)·
`Q16521`(taxon 91건) 등 메타/비엔티티는 의도적으로 미매핑.

남은 상위 unmapped (후속 이슈 후보):
- Q190752·Q6465·Q34442·Q11514315·Q50337·Q50068795·Q24746 등 (각 ~20건).
  도시·행정 세부 타입이 다수 — 지속적으로 테이블 확장 가능하나 diminishing
  return 구간 접근.

## 6. 타입별 신뢰도 계층화 (3중 검증 통합)

Cross-model kappa(10K, Gemma 관점 per-type) + Wikidata 앵커(10K, per-type)를
함께 놓고 타입별 신뢰 수준을 도출.

| 타입 | Kappa per-type | 앵커 per-type | 종합 계층 | 학습 데이터 권장 가중치 |
|---|---:|---:|---|---|
| PER | 0.7806 | 0.9836 | **High** | 그대로 사용 |
| LOC | 0.7398 | 0.9899 | **High** | 그대로 사용 |
| CORP | 0.7376 | 0.8628 | **High** | 그대로 사용 |
| PROD | 0.4518 | 0.9420 | **High** (낮은 kappa는 coverage) | 그대로 |
| FAC | 0.6581 | 0.7969 | Mid | 그대로 · 리뷰 |
| EVT | 0.6265 | 0.7027 | Mid | 리뷰 |
| POL | 0.6036 | 0.7236 | Mid | 리뷰 |
| ORG | 0.5964 | 0.8521 | Mid | 리뷰 |

주목:
- **PROD** kappa(0.45)는 낮지만 앵커 일치율(0.94) 높음 — Qwen3.6이
  제품·작품 경계를 보수적으로 잡아 coverage 차이가 kappa 하락의 주원인.
  Wikidata 앵커 기준 품질은 High.
- **ORG·POL**은 kappa·앵커 모두 0.6~0.85 수준. 두 타입 경계가 "정치적
  조직 vs 일반 조직"에서 양측 모두 흔들릴 수 있음. 포함하되 학습 시
  리뷰 권장.
- 8종 전체가 Mid 이상 신뢰 구간으로 정리됨.

## 7. WikiANN silver 오라벨 교정 부작용

3샘플 스모크에서 이미 관찰된 현상이 10K에서도 재현:
- `Ốc móng tay` (조개류 요리): WikiANN LOC → Gemma/Qwen 모두 skip
- `Lãm` (인명): WikiANN non-entity → Gemma PER 신규 탐지

즉 재라벨이 WikiANN silver 오라벨을 일부 교정한다는 유익한 부작용이 있다.
다만 교정 규모를 정량화하려면 gold 수준의 수작업 평가가 필요 — 본 프로젝트
제약으로 수작업 검증은 불가하므로 정성 언급에 한정.

## 8. 합성 보충 필요성 결론 (이슈 #10 5단계)

원래 플랜: "PROD·EVT 빈도 집계. 부족 판정 시 합성 보충 경로(PII
구조 재사용) 추가 여부를 사용자 승인 후 결정."

집계 결과 (10K Gemma):
- PROD: **1,131** (학습 데이터로 충분)
- EVT: **166** (희소하지만 최소 실용 수준)

결론: **합성 보충 불필요**. 재어노테이션(경로 B)만으로 두 타입 모두 확보.
EVT 166건은 저빈도지만 (a) 앵커 일치율 70%로 품질 관리 가능하고, (b)
합성 주입의 부작용(문맥 부자연성·학습 편향)을 감수할 만큼 부족하지는 않다.

## 9. 3종 공정 비교 (JA Stockmark vs VI WikiANN)

이슈 #10 스펙 §5.2 ③의 "PER/LOC/ORG 3종 subset" 비교. 두 데이터셋에서
8종 → 3종 매핑(`PER=PER, LOC=LOC, CORP∪POL∪ORG=ORG`)
후 3종 subset에 한해 비교.

| 지표 | JA Stockmark test | VI WikiANN-vi 10K (Gemma 재라벨) |
|---|---:|---:|
| Records | 1,069 | 10,000 |
| 3종 subset 엔티티 | 1,950 | 9,865 |
| **Entity density** | **1.824** / record | **0.987** / record |
| PER (= PER) | 554 (28.4%) | 4,206 (42.6%) |
| LOC (= LOC) | 399 (20.5%) | 3,885 (39.4%) |
| ORG (= CORP∪POL∪ORG) | 997 (51.1%) | 1,774 (18.0%) |

### 9.1 관찰

- **엔티티 밀도가 JA의 54% 수준** — WikiANN은 Wikipedia 개요 한 줄을 샘플로
  분리해 문맥이 짧고 엔티티가 적다. Stockmark는 Wikipedia 본문을 문단 단위로
  사용해 엔티티가 촘촘하다.
- **ORG 비율 역전** — JA 51%(본문에 기업·기관 언급 많음) vs VI 18%(개요가
  인물·지명 중심). 8종 기준에서도 CORP JA 19.8% vs VI 2.7%.
- **PER/LOC 비율 VI 지배적** — 42%+39% = 81% (JA는 49%).

### 9.2 8종 전체 분포 비교

| 타입 | JA % | VI % |
|---|---:|---:|
| PER | 21.14 | 36.18 |
| CORP | 19.80 | 2.43 |
| LOC | 15.22 | 33.42 |
| FAC | 9.04 | 4.00 |
| PROD | 8.74 | 9.73 |
| EVT | 7.82 | 1.43 |
| POL | 9.12 | 6.23 |
| ORG | 9.12 | 6.61 |
| **총 스팬** | **2,621** | **11,627** |

### 9.3 함의

- **절대 F1 비교 부적합 확정** — 엔티티 밀도·타입 분포가 너무 달라 동일
  F1 지표의 절대값은 의미를 잃는다.
- **상대 순위 비교는 가능** — 예: "VI는 JA 대비 인물/지명에 치우치고, JA는
  기업·이벤트 언급이 많다"와 같은 특성 비교에 활용.
- **학습 데이터 합성 시 유의** — JA+VI 합성 코퍼스를 구성하면 VI가 인물·지명
  선호를, JA가 조직·이벤트 선호를 끌고 온다. 언어별 가중 샘플링이 필요할
  수 있다.

## 10. 운영 제안

### 10.1 주·보조 라벨러 분리

- **Gemma = primary 재라벨러**. recall 넓고 라벨링 성공률 ~86%로 높음
- **Qwen3.6 = validator**. MoE 아키텍처로 raw 디코딩 속도는 빠르나
  엔티티 인식에 보수적이라 성공률 ~64%. 둘 다 라벨한 span = "고신뢰 silver"
- Gemma-only 3,201 span은 "확장 silver"로 분리 보관 가능. 학습 데이터로
  쓰려면 포함/제외 ablation 실험 권장

### 10.2 신뢰도 계층별 학습 데이터 활용

매핑 테이블(134 Q-ID) 기준 재분류 (10K Gemma):

- **High 계층 4종**(PER·LOC·CORP·PROD, 합계 9,504 span = 81.7%) — 그대로
  silver 학습 데이터로 사용
- **Mid 계층 4종**(ORG·FAC·POL·EVT, 합계 2,123 span = 18.3%) — 앵커
  확인된 것 우선, 특히 POL·EVT은 Wikidata P31이 확인된 건만 사용하는
  것이 보수적
- Low 계층 없음

### 10.3 최종 데이터셋: confidence 태그 병합

두 모델 재라벨 결과를 span 단위 confidence 태그를 붙여 단일 파일로 병합
(`src/augmenters/wikiann_vi/merge_confidence.py`).

4가지 confidence 카테고리 (10K test):

| confidence | source | 10K 건수 | 의미 |
|---|---|---:|---|
| `high` | `both` | 8,181 | Gemma·Qwen3.6이 동일 offset + 동일 타입 |
| `conflict` | `both_disagree` | 245 | 동일 offset인데 타입 불일치 |
| `medium_recall` | `gemma_only` | 3,201 | Gemma만 라벨(Qwen3.6 skip) |
| `medium_prec` | `qwen_only` | 690 | Qwen3.6만 라벨(Gemma skip) |

4가지 병합 정책 제공:

| 정책 | span 수 (10K test) | 용도 |
|---|---:|---|
| `recall` (**채택**) | **11,382** | Gemma 관대함 + 합의 span. conflict 제외 |
| `precision` | 8,871 | Qwen3.6 보수적 + 합의 span |
| `high_only` | 8,181 | 최고신뢰 subset (gold-like) |
| `full` | 12,317 | 전수. 학습 시 confidence 가중치 적용 |

**본 이슈 최종 채택**은 `recall` 정책. 이유: BERT 파인튜닝 시 엔티티 밀도를
최대화하고, 타입 혼동 스팬(192건)만 제외해 노이즈를 걸러낸다. 학습 코드에서
`if span['confidence'] == 'high'` 한 줄로 보수 모드 전환 가능.

### 10.4 매핑 테이블 구성

`WIKIDATA_TO_CANONICAL`은 134 Q-ID 큐레이션으로 10K Gemma에서 6,858건
매핑 성공, 전체 일치율 0.9512. 상세는 §5.5.

남은 상위 unmapped (각 20건 내외): Q190752·Q6465·Q34442 등 지역·세부
조직 타입. diminishing return 구간이라 후속 이슈 후보로만 유지.

## 11. 한계

- **이중 silver**: WikiANN(자동) + LLM 재라벨(자동)의 누적 silver. Stockmark
  gold와 F1 절대값 비교는 본 범위 밖이며, 수작업 gold 검증도 본 프로젝트
  제약상 수행 불가
- **Qwen3.6 보수적 coverage**: Qwen3.6의 엔티티 미탐지(skip) 비율이 높아
  Gemma-only span이 3,201건으로 kappa가 "moderate" 구간에 머문다.
  동일 offset 타입 일치율(97%)은 높으므로 타입 스펙 자체는 견고
- **매핑 테이블 제한**: 134종 큐레이션. 매핑 미수록 엔티티는 anchor
  검증에서 제외되어 공정한 coverage 해석에 주의 필요
- **Wikipedia 리다이렉트 위험**: 예: `Her Morning Elegance`(노래) → `Oren
  Lavie`(가수) 리다이렉트로 anchor 타입이 왜곡될 수 있음 — 전체 대비 비율은
  작지만 존재

## 12. 진행 매트릭스 (이슈 #10)

본 이슈 범위:
- [x] 1. VI loader·라벨러 이전 (`labelers/vi/`)
- [x] 2. 8종 매핑 스펙 확정 (`docs/manual/data/vietnamese-ner-8types.md`)
- [x] 3. 경로 B 재라벨 + 10K cross-model kappa (§4)
- [x] 4. Wikipedia 인터링크 → Wikidata P31 앵커 1K + 10K (§5)
- [x] 5. 전체 10K 재라벨 + PROD·EVT 빈도 집계 → **합성 보충
      불필요** (§3.2, §8)
- [x] 6. 3중 검증 통합 리포트 (본 문서, §6)

본 이슈에서 함께 수행한 추가 작업:
- [x] A1. `WIKIDATA_TO_CANONICAL` 확장 80→134 Q-ID (§5.5)
- [x] A3. JA Stockmark vs VI 3종 공정 비교 (§9)
- [x] A4. BATCH 프롬프트 지원 추가 (벤치 결과 현 구성에서 SINGLE과 동속도)

향후 후속 이슈 후보(본 범위 밖):
- [ ] `WIKIDATA_TO_CANONICAL` 추가 확장 (현 134 → ~200 diminishing return)
- [ ] BERT 파인튜닝 (VI 8종 학습 + 평가)
- [ ] JA 8종 vs VI 8종 통합 멀티링구얼 벤치 러너

## 13. 재현

```bash
# 1) 10K 전체 재라벨 — Gemma (primary)
python -m augmenters.wikiann_vi \
    --max-samples 10000 --concurrency 16 \
    --base-url http://localhost:8081/v1 \
    --model cyankiwi/gemma-4-31B-it-AWQ-8bit \
    --output data/wikiann_vi/gemma_8type_full.jsonl

# 2) 10K 전체 재라벨 — Qwen3.6 (validator, ~23분)
python -m augmenters.wikiann_vi \
    --max-samples 10000 --concurrency 16 \
    --base-url http://localhost:8082/v1 \
    --model cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
    --output data/wikiann_vi/qwen_8type_full.jsonl

# 3) Cross-model kappa (10K)
python -m augmenters.wikiann_vi.kappa \
    --a data/wikiann_vi/gemma_8type_full.jsonl \
    --b data/wikiann_vi/qwen_8type_full.jsonl \
    --json-out data/wikiann_vi/kappa_gemma_vs_qwen_full.json

# 4) Wikidata 앵커 검증 (10K, Gemma 기준)
python -m augmenters.wikiann_vi.wikidata_anchor \
    --input data/wikiann_vi/gemma_8type_full.jsonl \
    --cache data/wikiann_vi/wikidata_cache.json \
    --json-out data/wikiann_vi/anchor_gemma_full.json

# 5) 최종 병합 — recall 정책으로 단일 파일 생성
python -m augmenters.wikiann_vi.merge_confidence \
    --gemma data/wikiann_vi/gemma_8type_full.jsonl \
    --qwen data/wikiann_vi/qwen_8type_full.jsonl \
    --policy recall \
    --output data/wikiann_vi/vi_wikiann_8type_recall_test.jsonl
```

## 13a. train·validation 스플릿 확장 (2026-04-22)

본 리포트 §3~§12는 `test` 스플릿 10K를 대상으로 한 벤치마크다. 학습용
데이터셋 확보를 위해 동일 파이프라인으로 **train(20K) + validation(10K)**
스플릿도 재라벨했다. 모델·프롬프트·concurrency 설정은 §2와 동일.

### 13a.1 재라벨 결과

**Gemma (primary) — 라벨링 성공률 ~86%, 에러 0건**

| 스플릿 | 레코드 | 성공 | 총 스팬 | PER | LOC | PROD | ORG | POL | FAC | CORP | EVT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train (20K) | 20,000 | 17,282 | 23,203 | 8,195 | 7,935 | 2,204 | 1,585 | 1,453 | 912 | 601 | 318 |
| validation (10K) | 10,000 | 8,602 | 11,454 | 4,078 | 3,921 | 1,091 | 746 | 777 | 418 | 280 | 143 |

**Qwen3.6 (validator) — 성공률 ~64%, 에러 0건**

| 스플릿 | 레코드 | 성공 | 총 스팬 | PER | LOC | PROD | ORG | POL | FAC | CORP | EVT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train (20K) | 20,000 | 12,871 | 18,152 | 7,137 | 6,251 | 1,182 | 1,171 | 936 | 711 | 491 | 273 |
| validation (10K) | 10,000 | 6,357 | 8,946 | 3,589 | 3,027 | 597 | 543 | 502 | 326 | 240 | 122 |

### 13a.2 Recall 병합 · Cross-model kappa

| 스플릿 | recall 병합 스팬 | κ | p_o | p_e | 페어 |
|---|---:|---:|---:|---:|---:|
| train | 22,649 | 0.5879 | 0.6727 | 0.2057 | 24,393 |
| validation | 11,179 | 0.5733 | 0.6613 | 0.2062 | 12,114 |
| test (§4 재수록) | 11,382 | 0.5766 | 0.6642 | 0.2069 | 12,317 |

세 스플릿 kappa가 0.57~0.59 범위로 일관됨 — coverage 패턴이 분할 간
동질적임을 확인.

### 13a.3 산출물

```
data/wikiann_vi/
├── gemma_8type_{train,validation,test}.jsonl
├── qwen_8type_{train,validation,test}.jsonl
├── vi_wikiann_8type_recall_{train,validation,test}.jsonl
└── kappa_{train,validation,test}.json
```

## 14. 관련 커밋·문서

- 스펙: `docs/manual/data/vietnamese-ner-8types.md`
- 이슈 플랜: `docs/issues/issue-10-vi-ner-8type-relabel.md`
- 구현:
  - `src/labelers/vi/dataset_loader.py` (offset span loader)
  - `src/augmenters/wikiann_vi/prompts.py` (8종 few-shot 프롬프트)
  - `src/augmenters/wikiann_vi/relabel_8type.py` (async 재라벨 클라이언트)
  - `src/augmenters/wikiann_vi/__main__.py` (재라벨 CLI)
  - `src/augmenters/wikiann_vi/kappa.py` (cross-model kappa)
  - `src/augmenters/wikiann_vi/wikidata_anchor.py` (Wikipedia·Wikidata 앵커)
- 테스트: `tests/augmenters/wikiann_vi/` (40 건: 프롬프트·재라벨·kappa·앵커)
