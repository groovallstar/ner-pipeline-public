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

중간 산출물(개별 모델 silver, 다른 정책 변종, kappa, anchor)은 분석 후 정리. 재현은 본 이슈 §3 + 리포트 §10 명령으로 가능.

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

## 7. 1차 측정 결과

수치 표·해석은 모두 리포트로 이전(`docs/reports/vietnamese-ner-silver-quality.md` §2·§4). 본 이슈에는 1차 비교가 후속 작업(§8) 의 출발점이었다는 사실만 남긴다.

요지:
- PER 만 정의가 같아 절대 측정 가능 (Gemma 0.72~0.76, Qwen 0.78~0.81)
- LOC/ORG 는 silver↔gold 정의 불일치로 절대값 무의미
- PROD/EVT 는 직접 측정 불가 → 합의율·anchor 로 간접 추정 필요

## 8. PROD/EVT 신뢰도 향상 — 후속 작업 (3전략)

1차 측정에서 PROD/EVT 합의율이 47~57% 로 PER/LOC/ORG 대비 60~65% 수준에 머물러 신뢰 보강 필요. 세 전략을 병렬 실행:

- **전략 1 — type-aware 정책**: `merge_confidence._filter_by_policy` 에 분기 추가. PROD/EVT 만 `high` 강제, PER/LOC/ORG 는 `recall` 유지 → `recall_strict` 정책 신설.
- **전략 2 — 프롬프트 규칙 보강 후 재라벨링**: 리그/대회·작품·리스트글·학명·모델번호·인프라 6개 규칙을 `src/labelers/vi/ner_prompts.py` + `src/augmenters/wikiann_vi/prompts.py` 에 추가. Gemma+Qwen 동시 재라벨링.
- **전략 3 — Wikidata anchor PROD/EVT 매핑 확장**: `wikidata_anchor.py::WIKIDATA_TO_CANONICAL` 에 PROD(film/song/album/video game/book/free software) + EVT(war/treaty/election/championship instance) Q-ID 추가, 캐시 재사용.

세 전략 산출물의 최종 측정치는 리포트 §4·§5 참조.

## 9. 결과

검증 완료 수치는 모두 리포트로 단일화:

- silver F1 vs WikiANN-vi 3-gold → 리포트 §2
- cross-model 합의율 + Cohen kappa → 리포트 §4
- Wikidata anchor agreement → 리포트 §5
- `recall_strict` 데이터셋 크기 → 리포트 §1.3

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

## 11. 한계

리포트 §8 참조.

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
