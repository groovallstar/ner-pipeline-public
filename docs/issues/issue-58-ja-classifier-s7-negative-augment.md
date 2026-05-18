# issue-58: JA classifier S7 — 환각 부정 예시 추가 학습

- Issue: https://github.com/groovallstar/ner_pipeline/issues/58
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-58-ja-classifier-s7-negative-augment`
- 승인일: 2026-05-18

## 목적

v3+boundary 의 PROD/EVT/ORG precision 미달 단일 최대 leak — 환각 60건 중
ORG 28 (47%) — 을 데이터 측에서 해소. 환각 surface 패턴을 학습 셋에서
*entity 아님* 신호로 강화 → 모델 calibrate.

## 범위

- 포함: 부정 예시 추출/보강 데이터셋 생성 / 재학습 / 결과 측정·재진단
- 제외: 경계 규칙 (S6), LOC gazetteer (S2), backbone 교체 (S5)

## 성공 기준

- [ ] PROD/EVT/ORG strict precision ≥ 0.95 AND recall ≥ 0.93
- [ ] ORG 환각 28 → < 10 / 전체 환각 60 → < 30
- [ ] PER/LOC/PII 회귀 없음

## 위험·의존성

- 부정 보강 비율 민감 — 너무 많으면 모델이 entity 회피
- ambiguous surface (`徳川`, `海軍`, `工場`) — 학습 시 *해당 surface 가
  entity 로 라벨된 적 없는* 문장에서만 강화 (schema 충돌 최소화)
- 학습 시간 ~10~30분 (CRF/boundary 옵션 유지)

## 현 상태 (fact, #56 진단 인용)

- 기준선: strict 0.9368 / relaxed 0.9505
- 환각 60: ORG 28, LOC 7, EVT 7, PROD 6, 기타 12
- 패턴: 단일 한자, 영어 약어, 짧은 일반 명사
- S1 (#57) not-planned: threshold 는 P/R trade-off — 게이트 부적합 확정

## 결정 로그 (append-only)

- 2026-05-18: #57 (S1 threshold) not-planned 후 S7 진입. P/R 동시 천장
  가능 메커니즘 (데이터 측) 으로 결정.
- 2026-05-18: 1차 시도 — train oversampling (Q1=a), #56 dump 그대로
  (Q2=i), N=3 (Q3), 문장 단위 ambiguous 판단 (Q4). leak 수용.
- 2026-05-18: leak 포함 1차 F1 0.9784 가 ~80% leak 으로 부풀려짐 확인.
  classifier CLI `--data-extra-train-jsonl` 옵션 추가 → leak-free 평가
  도입.
- 2026-05-18: v1 (60 seed) leak-free → CREDIT_CARD P / PROD R 회귀.
  ambiguous seed 8개 자동 분류 → v2 (49 seed) → 회귀 패턴 ripple.
- 2026-05-18: N=2 보수적 (v3) → 7/10 게이트 통과, overall F1 0.9624 —
  최선. NER-only (v4) / PROD seed 제거 (v5) 둘 다 v3 보다 악화.
- 2026-05-18: **v3 (N=2, ambiguous 제외) production 후보로 채택**.
  임시 스크립트 → `src/ner/augmenters/ja_negative/` 정식 모듈 promote.

## 구현 단계 (완료)

- [x] 부정 예시 추출기 정식 모듈 (`src/ner/augmenters/ja_negative/`)
- [x] CLI (`python -m ner.augmenters.ja_negative`) + 단위 테스트 7종
- [x] classifier CLI 에 `--data-extra-train-jsonl` 옵션 (BC 유지)
- [x] 보강 데이터셋 생성 (5 variants: N3-all / N3-no-amb / N2-no-amb /
      N3-NER-only / N2-no-PROD)
- [x] classifier leak-free 학습 + per-entity strict P/R
- [x] confusion matrix 재진단
- [x] benchmark report §라운드 3 통합 — S0/S1/S7 모든 실험 영구 기록
      + production 후보 갱신 + 다음 작업·Fallback 정의

---

## 변경 요약

- 신규 모듈 `src/ner/augmenters/ja_negative/` — 환각 부정 예시
  oversampler (extract_seeds / filter_ambiguous_seeds /
  find_candidate_indices / oversample_to_jsonl) + CLI (BC 유지)
- 신규 단위 테스트 7종 (extract / filter / candidate / oversample
  combos)
- `src/ner/classifier/__main__.py` 에 `--data-extra-train-jsonl` 옵션
  추가 (~10줄, BC 유지) — train 에만 합치고 valid/test 는 원본 split
  유지하여 leak-free 평가
- benchmark report §라운드 3 (S7) 추가 — production 후보 갱신
  (v3+boundary → **v3+boundary+S7N2** strict 0.9368 → 0.9624)

## 검증

- 테스트: `uv run python -m pytest tests/ner/augmenters/ja_negative/
  tests/ner/classifier/ -q` → 47 passed
- Lint: `uv run ruff check src/ner/augmenters/ja_negative/
  src/ner/classifier/__main__.py` → All checks passed
- Smoke: `python -m ner.augmenters.ja_negative --help` 정상 출력
- 학습 산출물: `results/classifier/ja_sweep/s7_neg_n2_leakfree/` (v3
  채택)

## 진단 결과 — production vs S7 (v3)

5 variant 비교:

| 모델 | seed | N | F1 strict | HALL | 게이트 | 메모 |
|---|---|---|---:|---:|---:|---|
| production v3+boundary | - | - | 0.9368 | 60 | 5/10 | 기준선 |
| v1 (leak 포함, N=3 all) | 57 | 3 | 0.9784 | 8 | - | leak 부풀림 |
| v1 (leak-free, N=3 all) | 57 | 3 | 0.9597 | 32 | 6/10 | CREDIT_CARD/PROD 회귀 |
| v2 (N=3 ambig 제외) | 49 | 3 | 0.9552 | 22 | 6/10 | LOC/ID_NUM 회귀 |
| **v3 (N=2 ambig 제외)** | 49 | 2 | **0.9624** | 30 | **7/10** | 최선 — 채택 |
| v4 (N=3 NER-only) | 41 | 3 | 0.9572 | 31 | 5/10 | CREDIT_CARD 회귀 |
| v5 (N=2 PROD 제거) | 35 | 2 | 0.9525 | 32 | 4/10 | 가설 wrong |

v3 채택 — per-entity strict (production 대비 Δ):

| 종류 | P before→after | ΔP | R before→after | ΔR | 게이트 |
|---|---|---:|---|---:|---|
| PER | 0.974→0.974 | +0.0 | 0.976→0.971 | -0.5 | ✅ |
| LOC | 0.937→0.946 | +0.9 | 0.907→0.955 | +4.8 | ❌ P -0.4pp |
| ORG | 0.895→0.950 | +5.5 | 0.915→0.942 | +2.7 | ❌ R -0.8pp |
| PROD | 0.852→0.978 | +12.6 | 0.905→0.916 | +1.1 | ❌ R -3.4pp |
| EVT | 0.868→0.966 | +9.8 | 0.898→0.966 | +6.8 | ✅ |
| DAT | 0.990→0.971 | -1.9 | 0.952→0.962 | +1.0 | ✅ |
| EMAIL | 1.0→1.0 | =0 | 1.0→1.0 | =0 | ✅ |
| PHONE | 1.0→1.0 | =0 | 1.0→1.0 | =0 | ✅ |
| ID_NUM | 0.926→0.989 | +6.3 | 0.989→0.978 | -1.1 | ✅ |
| CREDIT_CARD | 0.950→0.952 | +0.2 | 0.938→0.975 | +3.7 | ✅ |
| **overall** | **0.9333→0.9594** | **+2.6** | **0.9402→0.9599** | **+2.0** | |

핵심 발견:
1. PROD seed 6개 제거 (v5) 가 오히려 PROD 자체 악화 → 환각 surface 가
   다른 entity calibration 에도 기여하는 ripple 효과 확인
2. N=2 가 N=3 보다 우월 — 보수적 oversample 이 회귀 완화하면서 환각
   효과 유지
3. ambiguous 자동 제외 (학습 셋에 entity 로 등장한 surface) 가 회귀
   방지에 필수
4. NER-only 분리 (PII 환각 제외) 효과 없음 — PII 환각 surface 는 무해

## 관련 커밋

- `<hash>`: <!-- commit 직후 채움 -->

## 후속 작업

S7 채택 후 잔여 미달 3 클래스 (LOC P -0.4pp / ORG R -0.8pp / PROD R
-3.4pp) 회복 후보:
- **S6** 경계 규칙 명문화 + 데이터 보정 (경계 어긋남 30 직격)
- (옵션) 게이트 정합 — NER/PII 분리 게이트 또는 0.95 → 0.93 완화
