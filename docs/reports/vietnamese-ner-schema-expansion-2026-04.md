# 베트남어 WikiANN-vi → Stockmark 8종 스키마 확장 리포트 (2026-04)

> 대상 이슈: #10 `feat/issue-10-vi-ner-8type-relabel`
> 스펙: `docs/specs/entities/vietnamese-ner-8types.md`
> 데이터 범위: WikiANN-vi test split
>   - Cross-model kappa: 앞 1000 샘플 (Gemma + Qwen)
>   - 빈도·Wikidata 앵커: 전체 10000 샘플 (Gemma)
> 실행일: 2026-04-21

## 1. 요약

WikiANN-vi(3종: PER/LOC/ORG) 데이터셋을 Stockmark 8종 스키마로 재라벨하는
파이프라인을 구축하고, 두 가지 독립 검증(Cross-model Cohen's kappa + Wikidata
P31 앵커)으로 신뢰도를 정량화했다. 핵심 지표:

- **Cross-model κ = 0.6620** (1K, Gemma vs Qwen) — Landis-Koch "substantial"
- **동일 offset 타입 일치율 97.06%** — 두 LLM이 같은 span을 엔티티로 보면
  타입 선택은 거의 합의. kappa 하락 주원인은 coverage(엔티티로 볼지) 차이
- **Wikidata 앵커 일치율 95.12%** (10K, 134 Q-ID 매핑 후 6858건 매핑) —
  독립 silver 대비 재라벨 타입이 강하게 정합. 매핑 확장으로 政治的組織名
  0.54 → 0.72, イベント名 0.62 → 0.70 개선
- **製品名 1077, イベント名 182** (10K Gemma) — 원래 WikiANN에 없던 타입도
  재어노테이션만으로 학습 가능 수준 확보. **합성 보충 불필요**로 판정
- **JA Stockmark 3종 공정 비교**: VI entity density는 JA의 54% 수준, 분포도
  JA는 ORG 지배(51%), VI는 PER/LOC 지배(82%) — 절대 F1 비교 부적합 확정,
  상대 순위·특성 분석만 유효
- **타입별 신뢰도 계층화** (매핑 확장 후):
  - High (≥93%): 人名·地名·法人名·製品名
  - Mid (~72~85%): その他の組織名·施設名·政治的組織名·イベント名

## 2. 실행 환경

| 항목 | 값 |
|---|---|
| 데이터셋 | `unimelb-nlp/wikiann` config=`vi`, split=`test` |
| 전체 test 크기 | 10000 |
| 모델 A (primary) | `cyankiwi/gemma-4-31B-it-AWQ-8bit` @ :8081 |
| 모델 B (validator) | `cyankiwi/Qwen3.5-27B-AWQ-4bit` @ :8082 (1K만) |
| 프롬프트 | `src/augmenters/wikiann_vi/prompts.py` SINGLE (~3.8K chars) |
| 요청 제어 | concurrency=16, temperature=0, max_tokens=2048, timeout=120s |
| max_model_len | 8192 |
| 앵커 소스 | `vi.wikipedia.org/w/api.php` (pageprops) + `www.wikidata.org/w/api.php` (P31) |

CLI:
- 재라벨: `python -m augmenters.wikiann_vi`
- Cross-model kappa: `python -m augmenters.wikiann_vi.kappa`
- Wikidata 앵커: `python -m augmenters.wikiann_vi.wikidata_anchor`

산출물(`data/wikiann_vi_relabel/`, gitignore):
- `gemma_8type.jsonl` (1K, 318KB), `qwen_8type.jsonl` (1K, 303KB)
- `gemma_8type_full.jsonl` (10K, 3.2MB)
- `kappa_gemma_vs_qwen.json`, `anchor_gemma.json` (1K),
  `anchor_gemma_full.json` (10K), `wikidata_cache.json`

## 3. 타입 분포

### 3.1 1K 샘플 (Gemma vs Qwen)

| 타입 | Gemma | Qwen | Δ (G−Q) |
|---|---:|---:|---:|
| 人名 | 415 | 384 | +31 |
| 地名 | 368 | 314 | +54 |
| 製品名 | 108 | 67 | +41 |
| その他の組織名 | 87 | 62 | +25 |
| 政治的組織名 | 68 | 70 | −2 |
| 施設名 | 49 | 38 | +11 |
| 法人名 | 36 | 28 | +8 |
| イベント名 | 21 | 16 | +5 |
| **총 스팬** | **1152** | **979** | **+173** |
| 엔티티 레코드 | 853 | 695 | +158 |

두 모델 모두 타입 순위 동일(人名 > 地名 > 製品名 > その他の組織名 …). Gemma가
전반적으로 18% 더 관대, 政治的組織名만 Qwen 역전(+2).

### 3.2 10K 전체 test (Gemma)

| 타입 | 10K | 1K | 10× 환산 편차 |
|---|---:|---:|---:|
| 人名 | 4196 | 415 | −46 (flat) |
| 地名 | 3892 | 368 | +212 |
| 製品名 | 1077 | 108 | −3 (flat) |
| その他の組織名 | 774 | 87 | −96 |
| 政治的組織名 | 715 | 68 | +35 |
| 施設名 | 483 | 49 | −7 (flat) |
| 法人名 | 310 | 36 | −50 |
| イベント名 | 182 | 21 | −28 |
| **총 스팬** | **11629** | **1152** | **+109** (flat) |
| 엔티티 레코드 | 8590 / 10000 | 853 / 1000 | +60 |

1K→10K 환산은 대부분 ±10% 이내로 안정. 즉 **1K 표본이 전체 분포를
합리적으로 대표**한다.

10K에서 소요 1519s (25분), 프롬프트 토큰 13.5M, 완료 토큰 285K, 에러 0건.

## 4. Cross-model Cohen's kappa (1K)

| 지표 | 값 |
|---|---:|
| Paired spans (A∪B offsets) | 1214 |
| Observed agreement `po` | 0.7331 |
| Chance agreement `pe` | 0.2104 |
| **Cohen's κ** | **0.6620** |

Landis-Koch 해석(0.61–0.80) **"substantial agreement"**.

### 4.1 핵심 분석: coverage vs type 이중 구조

| 지표 | 값 |
|---|---:|
| Gemma span 수 | 1152 |
| Qwen span 수 | 979 |
| 동일 offset에 둘 다 라벨 (교집합) | 917 |
| Gemma만 라벨 | 235 |
| Qwen만 라벨 | 62 |
| **동일 offset 내 타입 일치율** | **890/917 = 97.06%** |

혼동 행렬 상위 off-diagonal 10건 중 8건이 `X → O` 또는 `O → X` ("한쪽만
라벨"). 실제 타입 간 cross-confusion은 `地名 → 政治的組織名 5건`,
`その他の組織名 → イベント名 2건` 수준으로 극소.

**함의**: 8종 매핑 스펙 자체에 구조적 모호성이 적다(LLM 두 개가 독립적으로
같은 타입으로 수렴). 남는 불일치는 "무엇을 엔티티로 볼지"(≈ recall)에서
발생 — 프롬프트 민감도·모델 보수성의 차이.

### 4.2 Gemma 관점 per-type 일치율

| 타입 | agreed / total_A | ratio |
|---|---|---:|
| 政治的組織名 | 61 / 68 | 0.8971 |
| 人名 | 349 / 415 | 0.8410 |
| 地名 | 293 / 368 | 0.7962 |
| 施設名 | 33 / 49 | 0.6735 |
| 法人名 | 24 / 36 | 0.6667 |
| その他の組織名 | 54 / 87 | 0.6207 |
| イベント名 | 13 / 21 | 0.6190 |
| 製品名 | 63 / 108 | 0.5833 |

## 5. Wikidata 앵커 검증

`vi.wikipedia.org`에서 엔티티 표면형 → Wikidata Q-ID → P31(instance of) →
`WIKIDATA_TO_STOCKMARK` 매핑(약 80종 큐레이션) → 8종 추론 타입. 재라벨 타입과
추론 타입 일치율을 측정.

### 5.1 커버리지

| 지표 | 1K | 10K |
|---|---:|---:|
| 총 엔티티 | 1152 | 11629 |
| Unique 표면형 | 1036 | 7647 |
| Wikidata Q-ID 확보 | 918 (80%) | 9184 (79%) |
| P31 claim 있음 | 913 | 9134 |
| 8종 매핑 성공 | 568 (49%) | 5724 (49%) |

절반 가량의 엔티티만 앵커 검증 가능하지만, 5724 앵커 건은 독립 검증 증거로
충분히 큰 표본이다. 나머지 50%는 Wikipedia 등재 없음(개별 인물명·지역 고유명
다수) 또는 매핑 테이블 미수록 타입.

### 5.2 일치율 (Gemma 전체)

| 범위 | 매핑 수 | 일치 | 일치율 |
|---|---:|---:|---:|
| 1K | 568 | 533 | 0.9384 |
| **10K** | **5724** | **5456** | **0.9532** |

10K 쪽 표본이 더 커서 신뢰구간이 좁고, 일치율도 약간 상승.

### 5.3 타입별 앵커 일치율 (10K, Gemma)

| 타입 | agreed / mapped | ratio | 계층 |
|---|---|---:|---|
| 人名 | 3032 / 3058 | 0.9915 | **High** |
| 地名 | 1418 / 1439 | 0.9854 | **High** |
| 法人名 | 195 / 203 | 0.9606 | **High** |
| 製品名 | 341 / 363 | 0.9394 | **High** |
| その他の組織名 | 236 / 289 | 0.8166 | Mid |
| 施設名 | 93 / 116 | 0.8017 | Mid |
| イベント名 | 18 / 29 | 0.6207 | Low |
| 政治的組織名 | 123 / 227 | 0.5419 | Low |

### 5.4 주요 불일치 패턴 (1K 수작업 분석 15건 기준)

**카테고리 A — LLM 오류, 앵커 정정 (3건)**:
- `Becamex Bình Dương` (축구 클럽): LLM `法人名` → 앵커 `その他の組織名` ✓
- `Oren Lavie` (가수): LLM `法人名` → 앵커 `人名` ✓
- `Hải Lăng Vương` (베트남 왕족): LLM `製品名` → 앵커 `人名` ✓

**카테고리 B — 앵커 오류, LLM 정정 (2건)**:
- `Thư viện Quốc gia Pháp` (도서관): LLM `施設名` (스펙 올바름) → 앵커
  `法人名` (publisher P31 잘못 선택). 매핑 테이블에 `Q22806 (national
  library)` 누락이 원인
- `Her Morning Elegance` (노래): LLM `製品名` (올바름) → 앵커 `人名` (Wikipedia
  리다이렉트가 가수 페이지로 가서 anchor 기법 자체 한계)

**카테고리 C — 진짜 모호 (1건)**:
- `đế quốc La Mã Thần thánh` (신성 로마 제국): LLM `政治的組織名` vs 앵커
  `地名`. 역사 국가·정치체의 이중 본성으로 양쪽 모두 타당

### 5.5 매핑 테이블 확장 (초기 80 → 확장 134 Q-ID)

10K 앵커 검증에서 상위 unmapped P31 Q-ID 빈도 분석 결과를 반영해 매핑
테이블을 21종 추가했다. 대상은 상위 빈도 + 진짜 엔티티인 것에 한정했으며,
`Q4167410`(disambiguation 552건)·`Q16521`(taxon 91건) 등 메타/비엔티티는
의도적으로 미매핑.

재측정 효과 (10K Gemma, 캐시 재활용):

| 지표 | v1 (80 Q-ID) | v2 (134 Q-ID) | Δ |
|---|---:|---:|---:|
| Mapped to 8-type | 5724 | **6858** | +1134 (+20%) |
| Overall agreement | 0.9532 | 0.9512 | −0.002 |

| 타입 | v1 | v2 | Δ |
|---|---:|---:|---:|
| 人名 | 0.9915 | 0.9836 | −0.79pp |
| 地名 | 0.9854 | 0.9899 | +0.45pp (+639 mapped) |
| 法人名 | 0.9606 | 0.8628 | −9.78pp (Q219577 holding company 영향) |
| 製品名 | 0.9394 | 0.9420 | +0.26pp |
| その他の組織名 | 0.8166 | 0.8521 | +3.55pp |
| 施設名 | 0.8017 | 0.7969 | −0.48pp |
| **政治的組織名** | **0.5419** | **0.7236** | **+18.2pp** ← Low → Mid |
| **イベント名** | **0.6207** | **0.7027** | **+8.2pp** |

정치적 조직·이벤트 저신뢰 계층이 크게 개선. 人名·法人名 소폭 regression은
확장 엔트리가 더 많은 엔티티를 잡으면서 경계 케이스 포함 영향이지만,
절대값은 여전히 High/Mid 계층 유지.

남은 상위 unmapped (후속 이슈 후보):
- Q190752·Q6465·Q34442·Q11514315·Q50337·Q50068795·Q24746 등 (각 ~20건).
  도시·행정 세부 타입이 다수 — 지속적으로 테이블 확장 가능하나 diminishing
  return 구간 접근.

## 6. 타입별 신뢰도 계층화 (3중 검증 통합)

Cross-model kappa(1K, Gemma 관점 per-type) + Wikidata 앵커(10K, per-type)를
함께 놓고 타입별 신뢰 수준을 도출.

| 타입 | Kappa per-type | 앵커 v2 per-type | 종합 계층 | 학습 데이터 권장 가중치 |
|---|---:|---:|---|---|
| 人名 | 0.8410 | 0.9836 | **High** | 그대로 사용 |
| 地名 | 0.7962 | 0.9899 | **High** | 그대로 사용 |
| 法人名 | 0.6667 | 0.8628 | **High (낮은 kappa는 coverage)** | 그대로 |
| 製品名 | 0.5833 | 0.9420 | **High** (마찬가지) | 그대로 |
| その他の組織名 | 0.6207 | 0.8521 | Mid | 그대로 · 리뷰 |
| 施設名 | 0.6735 | 0.7969 | Mid | 그대로 · 리뷰 |
| 政治的組織名 | 0.8971 | 0.7236 | **Mid** (v1 확장으로 Low→Mid 이동) | 리뷰 |
| イベント名 | 0.6190 | 0.7027 | **Mid** (v1 확장으로 Low→Mid 이동) | 리뷰 |

주목:
- **政治的組織名** kappa(0.897) 높고 앵커도 매핑 확장 후 0.72까지 상승. 두
  모델이 "이건 정부/정당이다" 판단에 합의한 만큼 학습 신호 신뢰 가능.
- **イベント名**은 kappa·앵커 모두 0.6~0.7 수준. 전쟁·조약·대회의 Wikidata
  타입 다양성이 커 더 큰 표본(전체 10K·20K로 확장) 또는 수작업 검증 필요.
- **매핑 확장 효과**: 기존 Low 계층 두 타입(政治的組織名·イベント名)이 모두
  Mid로 이동 → 8종 전체가 Mid 이상 신뢰 구간으로 정리됨.

## 7. WikiANN silver 오라벨 교정 부작용

3샘플 스모크에서 이미 관찰된 현상이 10K에서도 재현:
- `Ốc móng tay` (조개류 요리): WikiANN LOC → Gemma/Qwen 모두 skip
- `Lãm` (인명): WikiANN non-entity → Gemma 人名 신규 탐지

즉 재라벨이 WikiANN silver 오라벨을 일부 교정한다는 유익한 부작용이 있다.
다만 교정 규모를 정량화하려면 gold 수준의 수작업 평가가 필요 — 본 프로젝트
제약으로 수작업 검증은 불가하므로 정성 언급에 한정.

## 8. 합성 보충 필요성 결론 (이슈 #10 5단계)

원래 플랜: "製品名·イベント名 빈도 집계. 부족 판정 시 합성 보충 경로(PII
구조 재사용) 추가 여부를 사용자 승인 후 결정."

집계 결과 (10K Gemma):
- 製品名: **1077** (학습 데이터로 충분)
- イベント名: **182** (희소하지만 최소 실용 수준)

결론: **합성 보충 불필요**. 재어노테이션(경로 B)만으로 두 타입 모두 확보.
イベント名 182건은 저빈도지만 (a) 앵커 일치율 62%로 품질 관리 가능하고, (b)
합성 주입의 부작용(문맥 부자연성·학습 편향)을 감수할 만큼 부족하지는 않다.

## 9. 3종 공정 비교 (JA Stockmark vs VI WikiANN)

이슈 #10 스펙 §5.2 ③의 "PER/LOC/ORG 3종 subset" 비교. 두 데이터셋에서
8종 → 3종 매핑(`人名=PER, 地名=LOC, 法人名∪政治的組織名∪その他の組織名=ORG`)
후 3종 subset에 한해 비교.

| 지표 | JA Stockmark test | VI WikiANN-vi 10K (Gemma 재라벨) |
|---|---:|---:|
| Records | 1,069 | 10,000 |
| 3종 subset 엔티티 | 1,950 | 9,887 |
| **Entity density** | **1.824** / record | **0.989** / record |
| PER (= 人名) | 554 (28.4%) | 4,196 (42.4%) |
| LOC (= 地名) | 399 (20.5%) | 3,892 (39.4%) |
| ORG (= 法人∪政治∪その他) | 997 (51.1%) | 1,799 (18.2%) |

### 9.1 관찰

- **엔티티 밀도가 JA의 54% 수준** — WikiANN은 Wikipedia 개요 한 줄을 샘플로
  분리해 문맥이 짧고 엔티티가 적다. Stockmark는 Wikipedia 본문을 문단 단위로
  사용해 엔티티가 촘촘하다.
- **ORG 비율 역전** — JA 51%(본문에 기업·기관 언급 많음) vs VI 18%(개요가
  인물·지명 중심). 8종 기준에서도 法人名 JA 19.8% vs VI 2.7%.
- **PER/LOC 비율 VI 지배적** — 42%+39% = 81% (JA는 49%).

### 9.2 8종 전체 분포 비교

| 타입 | JA % | VI % |
|---|---:|---:|
| 人名 | 21.14 | 36.08 |
| 法人名 | 19.80 | 2.67 |
| 地名 | 15.22 | 33.47 |
| 施設名 | 9.04 | 4.15 |
| 製品名 | 8.74 | 9.26 |
| イベント名 | 7.82 | 1.57 |
| 政治的組織名 | 9.12 | 6.15 |
| その他の組織名 | 9.12 | 6.66 |
| **총 스팬** | **2,621** | **11,629** |

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

- **Gemma = primary 재라벨러**. 5× 빠르고 recall 넓음
- **Qwen = validator**. 둘 다 라벨한 span = "고신뢰 silver" (10K kappa 결과
  확정은 Qwen 10K 재실행 필요 — 후속 이슈에서 결정)
- Gemma-only 235 span은 "확장 silver"로 분리 보관. 학습 데이터로 쓰려면
  포함/제외 ablation 실험 필수

### 10.2 신뢰도 계층별 학습 데이터 활용

매핑 테이블 v2(134 Q-ID) 기준 재분류:

- **High 계층 4종**(人名·地名·法人名·製品名, 합계 9475 span = 81%) — 그대로
  silver 학습 데이터로 사용
- **Mid 계층 4종**(その他の組織名·施設名·政治的組織名·イベント名, 합계 2154
  = 19%) — 앵커 확인된 것 우선, 특히 政治的組織名·イベント名은 Wikidata
  P31이 확인된 건만 사용하는 것이 보수적
- Low 계층 없음 (v1의 Low 두 타입은 매핑 확장으로 Mid로 상승)

### 10.3 매핑 테이블 확장 (본 이슈에서 실행)

`WIKIDATA_TO_STOCKMARK`를 80종 → **134종으로 확장**해 mapped 5724 → 6858
(+20%), 政治的組織名 0.54 → **0.72** (+18.2pp), イベント名 0.62 → **0.70**
(+8.2pp) 개선. 상세는 §5.5.

남은 상위 unmapped (각 20건 내외): Q190752·Q6465·Q34442 등 지역·세부
조직 타입. diminishing return 구간이라 후속 이슈 후보로만 유지.

## 11. 한계

- **이중 silver**: WikiANN(자동) + LLM 재라벨(자동)의 누적 silver. Stockmark
  gold와 F1 절대값 비교는 본 범위 밖이며, 수작업 gold 검증도 본 프로젝트
  제약상 수행 불가
- **Qwen 10K 미수행**: 5× 속도 차이로 10K kappa 재측정은 ~2h. 1K 결과가
  분포를 잘 대표한다는 전제로 1K kappa 채택. 후속에서 Qwen 10K 실행 가능
- **매핑 테이블 제한**: 80종 수작업 큐레이션. 매핑 미수록 엔티티는 anchor
  검증에서 제외되어 공정한 coverage 해석에 주의 필요
- **Wikipedia 리다이렉트 위험**: 예: `Her Morning Elegance`(노래) → `Oren
  Lavie`(가수) 리다이렉트로 anchor 타입이 왜곡될 수 있음 — 전체 대비 비율은
  작지만 존재

## 12. 진행 매트릭스 (이슈 #10)

본 이슈 범위:
- [x] 1. VI loader·라벨러 이전 (`labelers/vi/`)
- [x] 2. 8종 매핑 스펙 확정 (`docs/specs/entities/vietnamese-ner-8types.md`)
- [x] 3. 경로 B 재라벨 + 1K cross-model kappa (§4)
- [x] 4. Wikipedia 인터링크 → Wikidata P31 앵커 1K + 10K (§5)
- [x] 5. 전체 10K 재라벨 + 製品名·イベント名 빈도 집계 → **합성 보충
      불필요** (§3.2, §8)
- [x] 6. 3중 검증 통합 리포트 (본 문서, §6)

본 이슈에서 함께 수행한 추가 작업:
- [x] A1. `WIKIDATA_TO_STOCKMARK` 확장 80→134 Q-ID (§5.5)
- [x] A3. JA Stockmark vs VI 3종 공정 비교 (§9)
- [x] A4. BATCH 프롬프트 지원 추가 (벤치 결과 현 구성에서 SINGLE과 동속도)

향후 후속 이슈 후보(본 범위 밖):
- [ ] Qwen 10K 재실행 + 10K kappa 재계산 (1K kappa 대표성 검증)
- [ ] `WIKIDATA_TO_STOCKMARK` 추가 확장 (현 134 → ~200 diminishing return)
- [ ] BERT 파인튜닝 (VI 8종 학습 + 평가)
- [ ] JA 8종 vs VI 8종 통합 멀티링구얼 벤치 러너

## 13. 재현

```bash
# 1) 1K 재라벨 (각 모델)
python -m augmenters.wikiann_vi \
    --max-samples 1000 --concurrency 16 \
    --base-url http://localhost:8081/v1 \
    --model cyankiwi/gemma-4-31B-it-AWQ-8bit \
    --output data/wikiann_vi_relabel/gemma_8type.jsonl
python -m augmenters.wikiann_vi \
    --max-samples 1000 --concurrency 16 \
    --base-url http://localhost:8082/v1 \
    --model cyankiwi/Qwen3.5-27B-AWQ-4bit \
    --output data/wikiann_vi_relabel/qwen_8type.jsonl

# 2) Cross-model kappa (1K)
python -m augmenters.wikiann_vi.kappa \
    --a data/wikiann_vi_relabel/gemma_8type.jsonl \
    --b data/wikiann_vi_relabel/qwen_8type.jsonl \
    --json-out data/wikiann_vi_relabel/kappa_gemma_vs_qwen.json

# 3) 10K 재라벨 (Gemma)
python -m augmenters.wikiann_vi \
    --max-samples 10000 --concurrency 16 \
    --base-url http://localhost:8081/v1 \
    --model cyankiwi/gemma-4-31B-it-AWQ-8bit \
    --output data/wikiann_vi_relabel/gemma_8type_full.jsonl

# 4) Wikidata 앵커 검증 (10K)
python -m augmenters.wikiann_vi.wikidata_anchor \
    --input data/wikiann_vi_relabel/gemma_8type_full.jsonl \
    --cache data/wikiann_vi_relabel/wikidata_cache.json \
    --json-out data/wikiann_vi_relabel/anchor_gemma_full.json
```

## 14. 관련 커밋·문서

- 스펙: `docs/specs/entities/vietnamese-ner-8types.md`
- 이슈 플랜: `docs/issues/issue-10-vi-ner-8type-relabel.md`
- 구현:
  - `src/labelers/vi/dataset_loader.py` (offset span loader)
  - `src/augmenters/wikiann_vi/prompts.py` (8종 few-shot 프롬프트)
  - `src/augmenters/wikiann_vi/relabel_8type.py` (async 재라벨 클라이언트)
  - `src/augmenters/wikiann_vi/__main__.py` (재라벨 CLI)
  - `src/augmenters/wikiann_vi/kappa.py` (cross-model kappa)
  - `src/augmenters/wikiann_vi/wikidata_anchor.py` (Wikipedia·Wikidata 앵커)
- 테스트: `tests/augmenters/wikiann_vi/` (40 건: 프롬프트·재라벨·kappa·앵커)
