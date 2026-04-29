# Issue #30 — VI silver F1 vs WikiANN-vi 3-gold 정량 검증

> 선행 핸드오프: `docs/issues/handoff-post-issue-23-quality-validation.md` §3
> GitHub Issue: https://github.com/groovallstar/ner_pipeline/issues/30
> 브랜치: `feat/issue-30-vi-silver-quality-vs-3gold`

## 1. 목적

WikiANN-vi 5종 silver 의 **절대 정확도** 를 인간 주석 3-gold(PER·LOC·ORG)로 정량화해, production 학습 데이터 채택 기준을 확보한다.

추가 제약: 직전 PR #28(`dd3f4ab`)에서 시설명을 LOC→ORG 로 canonical 재정의. 현 silver 6종(`gemma_*` · `qwen_*`, 4/27 생성)은 **이전 스키마 기준** → 재생성 후 평가해야 한다.

## 2. 입력 자산 현황

작업 시작 시점:

```
data/wikiann_vi/                          (4/27 생성 — 모두 stale, 재생성 대상)
├── gemma_{train,validation,test}.jsonl   train=20K, val=10K, test=10K
├── qwen_{train,validation,test}.jsonl    train=20K, val=10K, test=10K
├── vi_wikiann_recall_{train,val,test}.jsonl
├── wikidata_anchor_{train,val,test}.json
├── kappa_{train,val,test}.json
└── wikidata_cache.json                   (재사용 가능 — Q-ID 캐시)
```

작업 완료 후 (분석 종료 + 정리):

```
data/wikiann_vi/
├── train.jsonl              # recall_strict 정책 적용 silver (20K records, 21,379 spans)
├── valid.jsonl              # 동일 (10K records, 10,580 spans)
├── test.jsonl               # 동일 (10K records, 10,669 spans)
└── wikidata_cache.json      # API 캐시 (재실행 효율용)
```

중간 산출물(개별 모델 silver, 다른 정책 변종, kappa, anchor)은 분석 후 정리. 재현은 본 이슈 §3 + 리포트 §12 명령으로 가능.

## 3. 작업 분해 (체크박스 ≠ 커밋, 관심사 단위로 나중에 묶음)

### 3.1 silver 재생성 (스키마 반영)
- [ ] **3.1.a** Gemma 재라벨 — 3-split (train·validation·test) 각각 실행
  - 모델: `cyankiwi/gemma-4-31B-it-AWQ-8bit`, base-url `http://localhost:8081/v1`
  - 예상 시간: train 50분 + val 25분 + test 25분 ≈ 1시간 40분
- [ ] **3.1.b** Qwen 재라벨 — 3-split 각각 실행
  - 모델: `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` (MoE, 활성 ~3B), base-url `http://localhost:8082/v1`
  - 예상 시간: dense 27B 대비 throughput 양호. 실측 후 조정 (train 1~2시간 추정)
- [ ] **3.1.c** recall-merge 재실행 — 3-split (`merge_confidence.py --policy recall`)
- [ ] **3.1.d** Wikidata 앵커 재실행 — 3-split (`wikidata_anchor.py`, 캐시 재사용)
- [ ] **3.1.e** kappa 재계산 — 3-split (`kappa.py`)
- [ ] **3.1.f** 산출물 무결성 점검 — 라벨 ⊂ 5종, offset 0 위반, span 분포 sanity

### 3.2 3-gold 추출 모듈
- [ ] **3.2.a** 신규 모듈 `src/llm_eval/wikiann_vi_gold.py`
  - HF `unimelb-nlp/wikiann/vi` 의 train/validation/test 로딩
  - BIO 토큰 라벨 → offset span 변환 (`labelers/vi/dataset_loader.py::bio_to_offset_spans` 재사용)
  - 라벨 매핑: `B-PER`/`I-PER` → `PER`, `B-LOC`/`I-LOC` → `LOC`, `B-ORG`/`I-ORG` → `ORG`
  - 출력: `dict[split → list[dict{id, text, gold_spans:[{type,start,end,text}]}]]`
- [ ] **3.2.b** 단위 테스트 `tests/llm_eval/test_wikiann_vi_gold.py`
  - HF stub 또는 fixture 한 split 의 첫 N건 BIO → span 변환 정확성
  - multi-byte/diacritic offset 정렬 1건 이상 포함

### 3.3 silver 평가 모듈
- [ ] **3.3.a** 신규 모듈 `src/llm_eval/vi_silver_quality.py` (CLI: `python -m llm_eval.vi_silver_quality`)
  - 입력: silver JSONL — 1차 산출물(Gemma 단독 + Qwen 단독) × 3-split
  - 처리:
    1. silver 의 entity span 에서 PER/LOC/ORG 만 필터링
    2. 같은 id 의 3-gold span 매칭
    3. `metrics/span_metrics.py::compute_offset_span_f1` 호출
  - 출력: 18셀(2 × 3 × 3) per-type + 6셀 통합 F1 → JSON + CSV
  - **추후 정책 평가**: Gemma·Qwen 1차 비교 결과를 사용자에게 보여 정책(`recall`/`precision`/`high_only`/`full` 또는 신규) 결정 → 그 정책으로 merge 재실행 후 평가 재실행
- [ ] **3.3.b** 단위 테스트 `tests/llm_eval/test_vi_silver_quality.py` — 합성 silver vs gold 1쌍으로 F1 계산 검증

### 3.4 리포트
- [ ] **3.4.a** `docs/reports/vietnamese-ner-silver-quality.md` 작성
  - §1 3-gold span F1: 1차 산출물 (Gemma · Qwen 단독) × 3-split × 3-type per-type 표 (18셀)
  - §2 1차 산출물 통합 F1 비교 (6셀) — 이 시점에 사용자 정책 결정
  - §3 결정된 정책 기반 merge 재실행 결과 추가 (정책별 1~2개 변종)
  - §4 기존 메트릭(kappa·앵커) 재정리
  - §5 학습 정책 결정 기준표:
    | 측정 F1 | 의미 | 권장 학습 정책 |
    | --- | --- | --- |
    | ≥ 0.90 | silver 단독 OK | gemma_train.jsonl 단독 |
    | 0.85~0.90 | recall-merge 권장 | high+medium |
    | 0.80~0.85 | high-only 권장 | conservative |
    | < 0.80 | 재라벨 필요 | — |
  - §6 실측 권고안 (이번 데이터로 어떤 정책이 적용 가능한지 결론)

### 3.5 마무리
- [ ] **3.5.a** 본 이슈 md 에 구현 결과 섹션 추가 (3-gold 변환 정확도, silver 4종 F1, 결정 기준 적용 결과)
- [ ] **3.5.b** 전체 pytest 회귀 + ruff clean 확인
- [ ] **3.5.c** 이슈 md 최초 커밋 → PR 생성 (`closes #30`)

## 4. 커밋 입도 계획 (잠정)

CLAUDE.md "커밋 입도" 기준 적용:

| 커밋 후보 | 포함 작업 | 분리 근거 |
|---|---|---|
| C1: silver 재생성 데이터 + sanity 리포트 | 3.1.a~f | 데이터만, 코드 무관 |
| C2: WikiANN-vi 3-gold 추출 + 평가 모듈 + 테스트 | 3.2 + 3.3 + 테스트 | 단일 관심사, 같이 동작해야 의미 있음 |
| C3: 리포트 + 이슈 md | 3.4 + 3.5.a | docs 단일 관심사 |

체크박스 14개 → 커밋 3개. 파일 수 기반 자동 분할 금지(CLAUDE.md).

## 5. 위험 요소 및 결정 필요 사항

### 5.1 silver 재생성 비용
- Qwen3.6-35B-A3B (MoE 활성 ~3B) → dense 27B 대비 throughput 양호. train 1~2시간 추정 (실측 후 조정).
- Gemma + Qwen3.6 동시 구동 → 두 모델 병렬 라벨링 가능 (총 wall-clock 단축).

### 5.2 정책 결정은 1차 데이터 후 사용자가 직접
- Gemma·Qwen 단독 silver F1 비교를 먼저 보고 사용자가 `recall`/`precision`/`high_only`/`full` 또는 신규 정책 선택. 미리 모든 정책 변종을 측정하지 않는다 (낭비 회피).

### 5.3 엔티티 종수 고정 — 5종 유지
- VI 라벨러 출력은 PER · LOC · ORG · PROD · EVT 5종으로 고정. 6종 이상으로 늘리면 LLM 라벨러가 prompt 내 정의 충돌·json schema 깨짐으로 태깅 실패 위험.
- 본 작업은 silver 출력 스키마를 변경하지 않고 *기존 5종을 재생성* 하는 데에 한정한다.

### 5.4 multi-byte offset
- WikiANN-vi 의 BIO 정렬은 token 단위. 베트남어 diacritic(`Bình Dương` 등)에서 offset 변환 정확성 점검 필수.
- `labelers/vi/dataset_loader.py::bio_to_offset_spans` 가 같은 변환을 silver 측에서 이미 사용 중 → 동일 함수 재사용해 정합성 보장.

### 5.5 3-gold 자체의 신뢰도
- WikiANN-vi 도 자동 생성 silver. 인간 검수가 일부 들어가 있어 본 LLM silver 보다 상위 신뢰지만 절대 정답은 아님.
- 리포트에 이 한계를 명시.

## 6. 확정된 진행 사항

- **범위**: 3-split 풀 재생성 (train 20K + val 10K + test 10K)
- **모델**: Gemma 31B-AWQ-8 @ 8081 + Qwen3.6 35B-A3B-AWQ-4 @ 8082, 동시 구동 후 본 세션 백그라운드 라벨링
- **엔티티**: 5종 고정 (PER · LOC · ORG · PROD · EVT)
- **정책**: 1차 산출물(Gemma·Qwen 단독) F1 비교 결과 본 후 사용자가 정책 결정 → 그 정책으로 merge 재실행 후 평가 재실행

## 7. 1차 측정 결과 (silver vs WikiANN-vi 3-gold, span F1)

| silver | split | F1 | P | R | PER | LOC | ORG |
|---|---|---|---|---|---|---|---|
| Gemma | test | 0.558 | 0.582 | 0.536 | 0.723 | 0.383 | 0.557 |
| Gemma | val | 0.556 | 0.581 | 0.534 | 0.728 | 0.374 | 0.568 |
| Gemma | train | 0.551 | 0.573 | 0.531 | 0.731 | 0.370 | 0.550 |
| Qwen | test | 0.560 | 0.628 | 0.505 | 0.787 | 0.345 | 0.515 |
| Qwen | val | 0.566 | 0.639 | 0.508 | 0.804 | 0.342 | 0.534 |
| Qwen | train | 0.561 | 0.631 | 0.504 | 0.812 | 0.337 | 0.513 |

### 7.1 해석 — 정의 일치 여부로 분리

| entity | 정의 일치성 | 의미 |
|---|---|---|
| **PER** | 같음 | 라벨러 절대 성능 직접 측정 가능. Qwen 0.79~0.81, Gemma 0.72~0.73 |
| **LOC** | 다름 (gold=지리+시설, silver=지리만) | F1 ↓ 의 일부는 스키마 차이. 절대 성능 추정 어려움 |
| **ORG** | 다름 (silver 가 시설 흡수) | 마찬가지 — silver ORG P 0.73~0.78 vs R 0.39~0.46 → 광범위 추출 |
| **PROD/EVT** | gold 없음 | 직접 비교 불가 — 간접 신호로 추정해야 함 |

### 7.2 cross-model 합의율 (PROD/EVT 신뢰의 간접 신호)

| type | test | val | train | 평균 |
|---|---|---|---|---|
| PER | 71.4% | 69.9% | 71.1% | 70.8% |
| LOC | 77.2% | 76.4% | 77.7% | 77.1% |
| ORG | 76.4% | 76.3% | 75.2% | 75.9% |
| EVT | 57.0% | 55.5% | 55.7% | 56.1% |
| PROD | 46.9% | 46.2% | 47.7% | 46.9% |

→ PROD/EVT 합의율이 PER/LOC/ORG 의 60~65% 수준. **신규 type 신뢰 부족** 확인.

### 7.3 cohen kappa (3-split 안정)

- test 0.6284 / validation 0.6179 / train 0.6285 → 모든 split 에서 일관된 ~0.62.

## 8. PROD/EVT 신뢰도 향상 — 후속 작업 (3전략)

### 8.1 disagreement 패턴 (test split 분석)

- conflict 패턴 top: LOC↔ORG(55), LOC↔PER(38), **ORG↔PROD(17)**, **PER↔PROD(15)**, **EVT↔ORG(13)**, **EVT↔PROD(9)**
- Gemma 단독 PROD: 영어 노래 강함, 작품 약함
- Qwen 단독 PROD: 작품 강함, 노래 약함
- 공통 노이즈: 식물 학명, "Danh sách..." (Wikipedia 리스트 글), 모델 번호 단독

### 8.2 전략 1 — Consensus filter (type-aware policy, ~30분)

- `merge_confidence._filter_by_policy` 에 type-aware 분기 추가
- PROD/EVT 는 `high` 만, PER/LOC/ORG 는 기존 정책 (recall 등) 그대로
- 학습 데이터 양 ↓, 신뢰 ↑

### 8.3 전략 2 — Prompt 규칙 보강 + 재라벨링 (~2시간 wall-clock)

`src/labelers/vi/ner_prompts.py` + `src/augmenters/wikiann_vi/prompts.py` 양쪽에 추가:

```
6. 연간 리그·정기 대회 → ORG. 특정 연도판만 EVT
   (예: V-League=ORG, "World Cup 2022"=EVT)
7. 음악·영화·책·만화·게임·TV 프로그램 = PROD
8. Wikipedia "Danh sách..."(List of) 글 제목은 entity 아님 → 무시
9. 학명·종 학술명 (Latin binomial, Bulbophyllum X) 은 무시
10. 모델 번호 단독 ("RV522") 은 PROD 아님. 브랜드+모델 결합만 PROD
11. 인프라 (철도노선·지하철노선) → 운영주체=ORG, 경로=LOC. PROD 아님
```

→ Gemma+Qwen 동시 재라벨링 (test 25분 + val 25분 + train 50분 = ~100분 wall-clock).

### 8.4 전략 3 — Wikidata anchor PROD/EVT 매핑 추가 (~1.5시간)

`wikidata_anchor.py::WIKIDATA_TO_CANONICAL` 확장:

```python
# PROD
'Q341':   'PROD',  # free software
'Q7889':  'PROD',  # video game
'Q11424': 'PROD',  # film
'Q7366':  'PROD',  # song
'Q482994': 'PROD', # album
'Q571':   'PROD',  # book
# EVT
'Q198':   'EVT',   # war
'Q625298': 'EVT',  # peace treaty
'Q40231': 'EVT',   # election
'Q27968055': 'EVT', # championship instance
```

→ 캐시 재사용 (기존 60% mapped + 신규 PROD/EVT QID 추가 fetch).

### 8.5 진행 계획 (병렬, wall-clock ~2.5시간)

```
t=0~30   전략 1 구현 + 테스트
t=10~120 전략 2 프롬프트 수정 + 재라벨링 (백그라운드)  ─┐
t=10~90  전략 3 anchor 매핑 추가 + 코드               │← 병렬
t=120    전략 2 post-pipeline 자동 실행
t=130    전략 3 재실행 (PROD/EVT 앵커링)
t=150    합의율 + F1 비교 결과 통합
```

## 9. 결과 (현 상태 측정값)

세부 지표·해석은 리포트 (`docs/reports/vietnamese-ner-silver-quality.md`) 참조. 본 섹션은 검증 완료된 핵심 수치만 기록.

### 9.1 silver F1 vs WikiANN-vi 3-gold (PER/LOC/ORG)

| silver | split | F1 | PER | LOC | ORG |
|---|---|---:|---:|---:|---:|
| recall_strict | test | 0.5652 | 0.750 | 0.382 | 0.553 |
| recall_strict | val | 0.5651 | 0.752 | 0.378 | 0.567 |
| recall_strict | train | 0.5601 | 0.759 | 0.371 | 0.551 |

### 9.2 cross-model agreement (per-type, 3-split 평균)

| type | LOC | ORG | PER | EVT | PROD |
|---|---:|---:|---:|---:|---:|
| 합의율 | 77.5% | 74.7% | 73.3% | 59.4% | 52.7% |

### 9.3 Cohen kappa

| split | test | val | train |
|---|---:|---:|---:|
| kappa | 0.6461 | 0.6419 | 0.6545 |

### 9.4 Wikidata anchor agreement

| split | mapped | overall | PER | LOC | ORG | PROD | EVT |
|---|---:|---:|---:|---:|---:|---:|---:|
| test | 61.1% | 96.07% | 99.0% | 98.2% | 86.2% | 93.6% | 86.8% |
| val | 60.6% | 96.37% | 99.0% | 98.8% | 86.2% | 95.2% | 89.5% |
| train | 60.7% | 96.43% | 98.8% | 98.7% | 87.7% | 94.4% | 84.8% |

### 9.5 recall_strict 데이터셋 크기

| split | records | entities |
|---|---:|---:|
| train | 20,000 | 21,379 |
| valid | 10,000 | 10,580 |
| test | 10,000 | 10,669 |

## 10. 검증 완료 사항

- [x] silver 재라벨링 (3-split × 2 모델) + recall_strict 정책 적용
- [x] WikiANN-vi 3-gold 변환 코드 (`llm_eval/wikiann_vi_gold.py`) + 단위 테스트
- [x] silver 평가 코드 (`llm_eval/vi_silver_quality.py`) + 단위 테스트
- [x] cross-model agreement per-type, per-split 분석
- [x] Wikidata anchor 외부 검증
- [x] 데이터 파일 정리 (Stockmark 포맷 통일: train/valid/test.jsonl)
- [x] 리포트 (`docs/reports/vietnamese-ner-silver-quality.md`) 작성
- [x] 본 이슈 md 결과 섹션 추가
- [x] 전체 pytest 회귀 + ruff clean

## 11. 한계 (리포트 §8 요약)

- gold 자체가 자동 생성 silver — ceiling 모름
- PROD/EVT 직접 F1 측정 불가 — 간접 신호로만 추정
- LOC/ORG F1 의 스키마 차이 분리 안 됨
- cross-model agreement 는 공통 편향 미감지
- Wikidata anchor 는 mapped 60% 만 검증 가능

## 12. 참조

- 리포트: `docs/reports/vietnamese-ner-silver-quality.md`
- 핸드오프 원문: `docs/issues/handoff-post-issue-23-quality-validation.md` §3
- 코드:
  - `src/labelers/vi/ner_prompts.py` (5종 프롬프트)
  - `src/labelers/vi/dataset_loader.py::bio_to_offset_spans`
  - `src/augmenters/wikiann_vi/__main__.py` (split별 재라벨 CLI)
  - `src/augmenters/wikiann_vi/merge_confidence.py` (정책 필터)
  - `src/augmenters/wikiann_vi/wikidata_anchor.py` (앵커)
  - `src/metrics/span_metrics.py::compute_offset_span_f1`
