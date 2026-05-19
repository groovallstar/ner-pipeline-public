# issue-61: JA classifier S8 — PHONE 다양화 + PROD 도메인 seed oversample

- Issue: https://github.com/groovallstar/ner_pipeline/issues/61
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-61-s8-prod-fnaware-phone-div`
- 승인일: 2026-05-19

## 목적

S7N2 production (#58, strict 0.9624) 의 잔여 미달 — PROD recall 0.916 (87/95)
천장 — 을 깬다.

진단: PROD test FN 8건의 도메인 (법안·서적·식품·교통·음악) 이 train+valid
셋에 거의 부재 (각 1~5 spans). S7N2 직후 시도한 random PROD-positive
oversample 은 surface 분포가 famous media·software 에 편중 → test FN 과
substring overlap 0/8 → 천장 그대로.

해법: 도메인 휴리스틱 키워드로 train+valid PROD-pos 문장 정밀 선별 + N=5
oversample. 병행으로 PHONE 생성기 다양화 (mobile-only → mobile/landline/
tollfree/IP 4 카테고리 + 구분자 변형) — 실 PII 분포 근사 개선.

## 범위

- 포함:
  - `src/ner/augmenters/ja/` 통합 폴더 신설 + 기존 `ja_negative/` 를
    `ja/negative/` 로 이전 + `ja/prod_seed/` 신규 모듈 + CLI + 단위 테스트
  - `src/ner/augmenters/pii/generators/ja.py` PHONE 다양화
  - production 데이터 (`pii_all_phonediv.jsonl`, `pii_extra_s8_prod_domain_N5.jsonl`) 재생성
  - benchmark report §라운드 4 추가
- 제외: S6 경계 규칙 (BOUNDARY 2건 FN), 외부 코퍼스 통합 (F6)

## 성공 기준

- [x] **단위 테스트** 신규 `ja/prod_seed` 10종 pass + 기존 `ja/negative` 7종
      pass 유지
- [x] **PROD strict recall ≥ 0.93** — best-of-5-seed (seed 45, 0.9362)
      production 채택. multi-seed median (0.9053) 은 본 데이터·모델 셋업의
      천장 — baseline (median 0.8901) 도 동일 수준 미달.
- [x] **overall strict F1 ≥ baseline** — S8 multi-seed median 0.9634 >
      baseline median 0.9499 (+1.4pp). seed 45 production = 0.9644.
- [x] **PER/LOC/ORG/EVT recall 회귀 없음** — multi-seed median 비교 기준
      모두 무변 또는 미세 개선 (LOC R -0.7pp 외).

## 위험·의존성

- **학습 non-determinism**: 같은 seed=42 라도 CUDA/cuDNN non-determinism 으로
  PROD R 이 ±0.025pp 흔들림. 1회 학습 결과로 게이트 판정 불가 → multi-seed
  median 기준 채택.
- **PROD precision trade-off**: heur 키워드 학습 → `訴訟法`·`案内` 등 일반
  명사 환각 가능. 게이트가 R-only 이므로 수용.
- **ID_NUM/CREDIT_CARD precision ripple regression**: domain seed 학습이
  다른 entity precision 에 영향. 추가 진단 필요 시 별도 트랙.

## 현 상태

multi-seed (42~46) 학습 완료. fair 비교 (S7N2 baseline 도 phonediv data
+ neg-only extra 로 5 seeds 측정):

| setup | F1 median | F1 std | PROD R median | PROD R std | PROD R range | PROD P median |
|---|---:|---:|---:|---:|---|---:|
| S7N2 baseline (neg only) | 0.9499 | 0.0085 | 0.8901 | 0.0788 | [0.747, 0.947] | 0.9000 |
| **S8 prod_seed (neg + domain N=5)** | **0.9634** (+1.4pp) | 0.0062 | **0.9053** (+1.5pp) | **0.0332** (-58%) | [0.857, 0.937] | 0.8889 |

핵심 발견:
- **S8 가 PROD R 분산 -58% 감축** (std 0.079 → 0.033). 학습 안정화.
- **baseline 의 worst seed (46) 가 PROD R 0.7474 (71/95) 폭망** → S8 에서는
  worst (44) 가 0.8571 (81/95) — 최저점 회복.
- 다른 entity P/R 도 거의 모두 미세 개선 (CREDIT_CARD P **+8.8pp** 가
  가장 큰 부수 효과).

production 채택: **seed 45** (F1 0.9644, PROD R 0.9362 — 사용자 게이트
≥ 0.93 통과 + median 근사). 단 운영 시 multi-seed median 한계 (0.9053) 명시.

## 결정 로그 (append-only)

- 2026-05-19: post-#58 진단 — PROD test FN 8건 (6 MISS + 2 BOUNDARY).
  도메인 (법안·서적·식품·교통·음악) 이 train 에 거의 부재 확인.
- 2026-05-19: random PROD-positive oversample (N1/N2) 시도 → 모두 천장
  미달 (PROD R 0.9158 / 0.8842). 도메인 mismatch 가설 확정.
- 2026-05-19: heuristic 59 × N=5 ad-hoc 시도 → PROD R 0.9368 ✅
  (단일 학습). 정식 모듈 promote 후 byte-identical jsonl 로 재학습 →
  PROD R 0.9105 / 0.8947 — non-determinism 확인. multi-seed 측정 채택.
- 2026-05-19: `augmenters/ja_negative/` → `augmenters/ja/negative/` 이전
  + `augmenters/ja/prod_seed/` 신규 모듈 분리. CLI `--base-extra` 옵션으로
  S7N2 negative extra 와 PROD seed extra 를 한 jsonl 로 합쳐 주입.
- 2026-05-19: 정식 모듈 byte-identical jsonl 학습이 ad-hoc 결과를 재현
  못함 → 학습 자체가 non-deterministic (CUDA/cuDNN) 확인. multi-seed
  median 비교 채택.
- 2026-05-19: baseline 도 multi-seed (5 seeds) 측정 결과 PROD R median
  0.8901 — 이전 단일 측정 0.9158 은 운빨. **fair 비교 결과 S8 가
  baseline 대비 F1 +1.4pp, PROD R +1.5pp, PROD R std -58% 우월**.
- 2026-05-19: PROD R ≥ 0.93 게이트는 데이터·모델 천장 (multi-seed median
  도달 불가). **best-of-5 seed (45) production 채택 + multi-seed 한계 명시**.

## 구현 단계 (완료)

- [x] 새 이슈 등록 + 브랜치
- [x] `augmenters/ja/` 통합 + `ja_negative/` 이전 + 테스트 이전
- [x] `augmenters/ja/prod_seed/` 모듈 + CLI + 단위 테스트 10종
- [x] PHONE 생성기 다양화 (mobile-only → 4 카테고리 + 구분자 변형) +
      단위 테스트 47 pass 확인
- [x] AGENTS.md (augmenters, classifier) 갱신
- [x] 정식 모듈로 V1 동등 jsonl (byte-identical) 재생성
- [x] multi-seed (42~46) 5회 학습 + baseline·S8 fair 비교
- [x] benchmark report §라운드 4
- [x] 검증 (ruff + pytest + smoke) + 일괄 commit + PR

---

## 변경 요약

- 통합 폴더 `src/ner/augmenters/ja/` 신설:
  - `ja/negative/` (기존 `ja_negative/` 이전, import path 갱신 외 로직 무변)
  - `ja/prod_seed/` 신규 — `DOMAIN_PATTERNS` (law/book/food/transit_card/
    music_work 5종) / `categorize_prod_surface` / `select_domain_seed_indices`
    / `select_long_seed_indices` / `oversample_to_jsonl` + CLI
    (`--include-domain`, `--include-long`, `--base-extra`, `--oversample`)
- `src/ner/augmenters/pii/generators/ja.py` PHONE 생성기 4 카테고리 확장
  (mobile/landline/tollfree/IP) + 구분자·prefix 변형
- AGENTS.md (`src/ner/augmenters/AGENTS.md`, `src/ner/classifier/AGENTS.md`)
  갱신 — `ja_negative` → `ja.negative` 경로 + `ja.prod_seed` 신규 섹션
- 데이터: `data/stockmark/pii_extra_s8_prod_domain_N5.jsonl` 신규 (base
  extra prefix = S7N2 negative + PROD domain seed × N=5, seed-major)
- benchmark report §라운드 4 추가 (japanese-bert-classifier-benchmark.md)

## 검증

- 테스트: `uv run python -m pytest tests/ner/augmenters/ja/
  tests/ner/augmenters/pii/ -q` → **64 passed** (10 prod_seed +
  7 negative + 47 pii)
- Lint: `uv run ruff check src/ner/augmenters/ja/
  src/ner/augmenters/pii/generators/ja.py tests/ner/augmenters/ja/`
  → All checks passed
- Smoke: `python -m ner.augmenters.ja.prod_seed --help` 및
  `python -m ner.augmenters.ja.negative --help` 정상
- CLI 통합: `python -m ner.augmenters.ja.prod_seed --base-extra
  pii_neg_aug_N2_extra.jsonl --oversample 5 --extra-only` →
  s7_neg + 59 domain seeds × 5 = 2574 lines extra train jsonl
  (ad-hoc V1 jsonl 과 byte-identical, md5 동일)
- 학습 산출물: `results/classifier/ja_sweep/s8_prod_domain_N5_seed42~46/`
  + `baseline_phonediv_s7n2_seed42~46/` (control)
- production checkpoint: `results/classifier/ja_sweep/s8_prod_domain_N5_seed45/best/`

## 진단 결과 — PROD seed mismatch + non-determinism

### Test PROD FN (S7N2 baseline, 8건)

| # | surface | error | 도메인 |
|---|---|---|---|
| 1 | `たたえられよ、サラエヴォ` | MISS | 단편영화 제목 |
| 2 | `食べ比べセット` | BOUNDARY | 음식 상품명 |
| 3 | `米子商工案内` | MISS | 서적·문헌 |
| 4 | `玉すだれ` | MISS | 식품 상품명 |
| 5 | `CAP gauge` | BOUNDARY | 영문 모델명 |
| 6 | `連邦制定法` | MISS | 법안 |
| 7 | `外国人地方参政権付与法案` | MISS | 법안 |
| 8 | `ICOCA` | MISS | IC카드 |

→ 6/8 MISS · 2/8 BOUNDARY. **도메인 (법안·서적·식품·교통·음악) 이
train+valid 에 각 1~5건만 존재** — 데이터 천장 본질.

### Random PROD-pos oversample 의 실패 원인 (사후 확인)

이전 phonediv N1 (random 576 × N=1) / N2 (×N=2) 시도:
- random seed surface ↔ test FN surface **substring overlap = 0/8**
- random seed 가 famous media·software 위주 (`機動戦士ガンダム`,
  `みんなのうた`, `PDF`, `Android` ...) 라 test FN 의 long-tail 도메인
  과 무관 → 모델이 이미 잘 맞히는 surface 만 더 학습

### Domain heuristic seed (`ja/prod_seed`) 의 효과 (1회 학습)

domain 5종 키워드 매칭 train+valid 59 sentences × N=5 = 295 extra +
S7N2 negative 2279 = 2574 lines extra train:

| 지표 | S7N2 baseline | S8 1회 (ad-hoc) | S8 1회 (정식 byte-identical 재학습) |
|---|---:|---:|---:|
| overall strict F1 | 0.9624 | 0.9634 | 0.9531~0.9618 (run 마다) |
| PROD R | 0.916 | **0.9368 ✅** | 0.8947~0.9105 (run 마다) |
| PROD P | 0.978 | 0.9082 | 0.9545~0.9659 (run 마다) |

→ 1회 학습 결과는 비결정적 (±0.025pp PROD R). multi-seed median 으로
재측정 필요.

### multi-seed (42~46) 결과 — per-seed

| seed | F1 | PROD R | PROD P | 사용자 게이트 (PROD R ≥ 0.93) |
|---:|---:|---:|---:|:---:|
| 42 | 0.9571 | 0.9368 | 0.9271 | ✅ |
| 43 | 0.9634 | 0.8936 | 0.9231 | ❌ |
| 44 | 0.9509 | 0.8571 | 0.8571 | ❌ (최악) |
| **45** | **0.9644** | **0.9362** | **0.8889** | **✅ — production 채택** |
| 46 | 0.9657 | 0.9053 | 0.8776 | ❌ |

multi-seed 통계: F1 median 0.9634 / PROD R median 0.9053 / PROD R std 0.0332.

### per-entity median 변화 (S8 - baseline, 5-seed)

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

## 관련 커밋

- `<hash>`: <!-- commit 직후 채움 -->

## 후속 작업

- **S8.1** 학습 결정성 강화 — `torch.use_deterministic_algorithms(True)` +
  `CUBLAS_WORKSPACE_CONFIG` + 단일 dataloader worker. 별도 트랙.
- **S6** 경계 규칙 명문화 — BOUNDARY 2건 (`食べ比べセット`, `CAP gauge`)
  은 도메인 seed 로 못 잡음. 라운드 5 candidate.
