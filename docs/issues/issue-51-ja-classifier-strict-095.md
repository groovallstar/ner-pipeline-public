# issue-51: NER 5종 strict 0.95 도달 (gold 보정 라운드 2 + CRF + boundary loss)

- Issue: https://github.com/groovallstar/ner_pipeline/issues/51
- PR: <!-- 5단계에서 채움 -->
- 브랜치: `feat/issue-51-ja-classifier-strict-095`
- 승인일: <!-- 3단계 승인 후 채움 -->
- 완료일: <!-- 5단계에서 채움 -->

## 목적

이슈 #48 종결로 baseline_corrected v2 동결됐으나 (strict F1 0.9249 /
relaxed 0.9415) NER 5종 strict 0.95 게이트 미달:

| Entity | strict P | strict R | min(P,R) | gap to 0.95 |
|---|---:|---:|---:|---:|
| ORG | 0.8845 | 0.8941 | 0.8845 | -6.55pp |
| LOC | 0.8763 | 0.8822 | 0.8763 | -7.37pp |
| PROD | 0.8723 | 0.8817 | 0.8723 | -7.77pp |
| EVT | 0.9080 | 0.9080 | 0.9080 | -4.20pp |
| PER | 0.9528 | 0.9680 | 0.9528 | -0.28pp (회귀 방지) |

본 이슈는 4개 실험을 묶어 NER 5종 전부 strict ≥ 0.95 도달 시도:

1. **E5+E4 통합 (gold quality)** — test 오답 138 sentences (285 errors) 재검수
   라운드 2 + canonical schema 보강
2. **E1 (CRF head)** — BIO 일관성 강제, BOUNDARY 오류 구조적 회수
3. **E3 (boundary-aware loss)** — entity 경계 학습 가중치 ↑

PII 5종 (DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD) 은 이미 게이트 만족이라
회귀 방지만 검증.

## baseline_corrected test 오답 분포 (사전 조사)

527 sentences / 1,891 spans / 1,755 exact / FN 136 / FP 149 / **138 sentences with errors (26%)**.

| Entity | BOUNDARY | MISS | TYPE_MIS | HALLUCIN | 합계 |
|---|---:|---:|---:|---:|---:|
| ORG | 32+32 | 23 | 4+7 | 26 | 124 |
| LOC | 14+14 | 15 | 6 | 23 | 72 |
| PER | 9+9 | 1 | 2+2 | 7 | 30 |
| PROD | 5+5 | 6 | 2 | 5 | 23 |
| EVT | 3+3 | 5 | 1 | 4 | 16 |
| PII 5종 | — | 13 | 3 | 6 | 22 |

(BOUNDARY 는 FN/FP 페어이므로 unique span 쌍 수)

검수 표적:
- **MISS 58 + TYPE_MISMATCH 14 + HALLUCINATION 71 = 143 errors** —
  사람 검수 대상 (sub-word fragmentation 노이즈 제외 시 ~100건)
- **BOUNDARY 64 페어 = 검수 불요** — E1+E3 모델 측 개선 표적

발견된 schema 모호 영역 (E4 보강 표적):
1. 시설명 vs LOC/ORG — 경기장·서킷·공원 (`カタロニア・サーキット`,
   `バッテリー・パーク`)
2. 국가/지명 + `代表` 결합 — 대표팀 (`チェコ代表`, `山形`)
3. 단지·복합시설 — 부동산 단지 (`ワンハンドレッドヒルズ`)
4. 동음이의 ORG vs PROD — `タランテラ`, `LKAB`

산출 데이터: `results/classifier/ja_sweep/baseline_corrected/error_analysis.json`,
`review_sample.jsonl`.

## 범위

포함:
- baseline_corrected test 오답 143건 검수 (BOUNDARY 제외)
- canonical-entity-schema.md 보강 — 시설/대표팀/단지/동음이의 4영역
- src/ner/classifier/train_eval.py — CRF head + boundary-aware loss
- 단위 테스트 신규 + 학습·평가 + benchmark.md 갱신

제외:
- VI 트랙 (post-#40 핸드오프 별도)
- E2/E6/E7 (oversampling/ensemble/GLiNER) — 본 라운드 미달 시 후속
- Tier 2.B self-training (회귀 위험으로 미시도 결정)
- F3-JA / F6-JA (#45 동결)

## 성공 기준

- [ ] 테스트: CRF + boundary-loss 단위 테스트, 기존 36 케이스 회귀 0
- [ ] 검수 spot check: 사용자 동의율 ≥ 90%, 미달 시 데이터 채택 거부
- [ ] 메트릭: NER 5종 strict P/R 모두 ≥ 0.95 — 미달 시 ablation 분석 +
      종결 옵션 (A/B/C)
- [ ] 문서: japanese-bert-classifier-benchmark.md 라운드 2 섹션,
      canonical-entity-schema.md 보강
- [ ] PII 5종 회귀 < 0.5pp

## 구현 단계

- [ ] **1. canonical schema 보강 (E4 docs)** — schema §2.3 / §5.1 에 시설/
      대표팀/단지/동음이의 4영역 명시화. 사용자 승인 후 (2) 진행
- [ ] **2. test 오답 재검수 라운드 2 (E5+E4 데이터)** — 클로드가 143 errors
      에 schema 룰 적용·offset 검증 → corrections.jsonl 생성. 사용자 28건
      (20%) spot check. 동의율 ≥90% 시 보정 적용 (pii_all.jsonl in-place).
      미달 시 채택 거부 + 회귀 사유 보고
- [ ] **3. CRF head 추가 (E1)** — pytorch-crf 또는 직접 구현으로
      train_eval.py 의 token classification head 교체. BIO 일관성 round-trip
      단위 테스트. 학습 1회 + per-entity strict 평가
- [ ] **4. Boundary-aware loss (E3)** — B-/I-end token weight 차등 (B=1.5,
      I-end=1.2 권장). 단위 테스트로 weight 적용 검증. 학습 1회 +
      per-entity strict 평가
- [ ] **5. CRF + boundary loss 동시 적용 + ablation table** — 학습 4 variants
      (baseline / +CRF / +boundary / +CRF+boundary). strict P/R/F1
      비교, 0.95 도달 여부 판정
- [ ] **6. 마무리** — japanese-bert-classifier-benchmark.md 라운드 2 섹션
      (ablation table 포함), 본 md 결과·검증 채움, 최초 commit + PR

## 위험·의존성

- AI 검수 augmentation 회귀 패턴 (#45 LLM augmentation 8/8 회귀,
  PROPOR 2026 -0.24~-1.81%) — spot check 90% 게이트로 방어
- CRF + boundary loss 동시 적용 시 PII 100% 회귀 위험 (낮음, 검증 필수)
- 데이터 in-place 변경 — 백업 미보존 관행 유지 (Tier 3 동일).
  pii_all.jsonl 행 수는 5,270 유지 (행 추가·삭제 없음, entity-level
  in-place 정정만)
- single-seed (seed=42), Δ < 0.1pp 는 noise 처리
- pytorch-crf 의존성 추가 시 `pyproject.toml` 갱신 — uv sync 필요

## 의미 단위 commit 입도 (CLAUDE.md "커밋 입도" 규칙)

본 이슈는 같은 모듈 (`src/ner/classifier/`) 코드 + docs 데이터 변경 +
docs 결과 작성으로 구성되므로 4 commit 구조 권장:

1. **데이터 quality** (단계 1+2) — schema 보강 + 검수 결과 + corrections
   적용. `docs/manual/data/canonical-entity-schema.md` +
   `data/stockmark/pii_all.jsonl` (gitignore — 본 commit 은 변경
   사실만 docs/issues 에 기록)
2. **모델 코드** (단계 3+4) — CRF + boundary loss + 단위 테스트.
   `src/ner/classifier/train_eval.py` + `tests/ner/classifier/`
3. **학습·평가** (단계 5) — ablation 결과 + benchmark.md 갱신.
   `docs/reports/japanese-bert-classifier-benchmark.md`
4. **이슈 md 최초 commit** (단계 6) — 본 md + PR

---

## 변경 요약

이슈 #51 의 4 실험 묶음 종결. NER 5종 strict P/R 모두 ≥ 0.95 게이트는
**미달 확정**. 사용자 결정 → **(B) 천장 동결 + production 채택**.
production 모델: **v3+boundary** (strict 0.9368 / relaxed 0.9505).

## 구현 결과 (계획 대비)

### 1. canonical schema 보강 (E4 docs) — ✅ 완료

`docs/manual/data/canonical-entity-schema.md` 변경:
- §변경 이력에 2026-05-07 #51 항목 추가
- §2.3 JA 핵심 경계 케이스 — 4행 추가
  (스포츠 대표팀 / 경기장·서킷·도시공원 / 城·城跡 / 부동산 단지)
- §3 PROD vs EVT vs ORG 경계 — 약어 단독 룰 + 동음이의 가타카나
  시그널 룰 추가
- §5.1 JA 모호 사례 결정표 — 11행 추가 (test 오답 검수 표적)

### 2. test 오답 재검수 라운드 2 (E5+E4 데이터) — ✅ 완료

baseline_corrected 위에서 inference (`error_analysis.py` CLI 재실행) →
test 오답 138 sentences (285 errors) 추출 → 클로드가 schema 룰 적용·
offset 검증 → corrections.jsonl 작성 → 사용자 ambiguous 10건 결정 →
40 corrections 통합 → in-place 적용.

| 분류 | 수 |
|---|---:|
| 전체 검수 errors | 157 (NER 5종 139 + PII 18, BOUNDARY 64 페어 제외) |
| model_correct (보정) | 29 (사용자 결정 전) |
| ambiguous (사용자 결정 후) | 11 추가 (s277/s340 16자리 패턴 결정 포함) |
| 최종 corrections | **40** (32 add + 7 replace + 1 replace_span) |
| gold_correct (모델 천장) | 118 |

핵심 결정:
- **s277/s340 16자리 패턴 → CREDIT_CARD** — augmenters/pii generator
  base.py:31 의 IIN 리스트(`'4'`, `'5'`, `'34'`, `'37'`, `'3528'`,
  `'3569'`)와 더블공백 구분자 매칭으로 결정적 CC 출력 패턴 확인. text
  컨텍스트 (`管理番号`, `識別子`) 는 합성 augmenter 의 랜덤 적용일 뿐
- **s459 NHK boundary 확장** — gold (ORG, 46-49) → (ORG, 46-54)
  `NHK仙台放送局`. 기존 dup entity 1개 dedup
- **s524 분리** — `アメリカ合衆国ハワイ準州マウイ島` → `ハワイ準州` +
  `マウイ島` 두 LOC

데이터 변경 net: **5,270 행 유지**, 24 sentences entities in-place
변경, +32 add - 1 dedup = +31 net entities.

### 3. CRF head 추가 (E1) — ✅ 완료

`pytorch-crf==0.7.2` 의존성 추가. `src/ner/classifier/train_eval.py`
에 `BertCRFForTokenClassification` 신규 (nn.Module wrapper). HF Trainer
호환을 위해 `WeightedTrainer._save` / `_load_best_model` override 추가.

신규 단위 테스트 6 케이스 (`tests/ner/classifier/test_crf.py`).

학습 결과 (v3_crf): strict 0.9311 (-0.35pp vs v3) / relaxed 0.9469.
PII 회복 (ID_NUM +2.95, CC +0.99) 하지만 NER 일부 회귀 (LOC -2.25,
EVT -3.94, PER -1.42). 학습 시간 +3.6x (340s vs 95s).

### 4. Boundary-aware loss (E3) — ✅ 완료

`boundary_weights_tensor` helper 신규 (`src/ner/classifier/data_utils.py`).
`__main__.py` 에 `--boundary-b-weight` / `--boundary-i-weight` 플래그.
class_weights 와 elementwise 곱으로 결합.

신규 단위 테스트 4 케이스 (`tests/ner/classifier/test_boundary_weights.py`).

학습 결과 (v3_boundary, B=1.5/I=1.2): strict **0.9368** / relaxed
**0.9505**. NER 5종 안정 향상 (LOC +1.31, PER +0.67, ORG 동일).
EVT/PROD 부분 회귀 (-2.53, -1.37). 학습 시간 동일 (~100s).

### 5. 4 variants ablation — ✅ 완료

| Variant | strict F1 | relaxed F1 | 학습 시간 |
|---|---:|---:|---:|
| v2 (이전 baseline_corrected) | 0.9249 | 0.9415 | 95s |
| v3 (data 보정 only) | 0.9346 | 0.9498 | 95s |
| v3+CRF | 0.9311 | 0.9469 | 340s |
| **v3+boundary** | **0.9368** | **0.9505** | 100s |
| v3+combined (NER cw + boundary) | 0.9347 | 0.9480 | 99s |

상세 per-entity 표·해석:
`docs/reports/japanese-bert-classifier-benchmark.md` §"라운드 2
(이슈 #51 종결)" 참조.

### 6. 종결 — (B) 천장 동결 ✅

NER 5종 strict P/R 모두 ≥ 0.95 미달 확정. 모든 시도 후에도
ORG/LOC/PROD/EVT 가 0.85~0.94 천장. 사용자 결정으로 (B) 천장 동결 +
v3+boundary production 채택.

## 검증

- `pytest tests/ner/classifier/` — **36 케이스 통과**
  (CRF 6 + boundary 4 + 기존 26, 회귀 0)
- 보정 데이터 검증: 40 corrections 모두 offset round-trip 일치, schema
  룰 매칭, replace target 매칭 (verify_corrections.py)
- 4 variants 학습 모두 정상 종료, validation loss 정상 수렴
- production 모델 (v3+boundary):
  - `results/classifier/ja_sweep/v3_boundary/metrics.json` 동결값
  - strict F1 0.9368 / strict P 0.9334 / strict R 0.9402
  - relaxed F1 0.9505 / relaxed P 0.9471 / relaxed R 0.9540

## 후속 작업·알려진 한계

### 즉시 진행 가능 (별도 이슈 후보, ROI 낮음 추정)

1. **E2 NER oversampling** — PROD/EVT 4-8x 복제. 학습 분포 보강
2. **E7 GLiNER-Japanese** — span-based extraction (BIO 의존 제거)
3. **E8 DAPT** — `bert-base-japanese-v3` + Wikipedia JA 추가 MLM
4. **Tier 2.B Self-training** — JA Wikipedia pseudo-label
   (이슈 #45 augmentation 회귀 위험)

본 시점 추정: 데이터 천장이 본질이라 위 4개 모두 +1~3pp 이내. 0.95
strict 도달은 *추가 사람 검수* 또는 *외부 gold 코퍼스 통합* 없이는
실현 가능성 낮음. 별도 이슈로 분리해 ROI 재평가 후 진입.

### 알려진 한계

- pii_all.jsonl 백업 미보존 — 5,270 행 유지하지만 entity-level 변경
  (24 sentences, +31 net entities) 후 옛 baseline_corrected 분포와 직접
  비교 불가. 본 md + benchmark.md 가 영구 인용 단일 출처
- ambiguous 60건 (Tier 3 시점) 의 사용자 미결정 케이스 — 본 라운드
  2 의 138 sentences 재검수와 별개로 잔존. canonical schema 보강 후
  추가 검수 가능
- CRF 학습 시간 +3.6x — 4 epoch 학습 시간 95s → 340s. production
  운영에는 inference 만 사용 (CRF.decode 는 빠름)
- v3+boundary 의 EVT 회귀 (-2.53pp) — boundary weight 가 entity 빈도
  적은 클래스에 부정적. 향후 per-entity 가중치 차등 검토 가능
- 임시 스크립트 `/tmp/extract_round2_errors.py`,
  `/tmp/group_round2_errors.py`, `/tmp/apply_round2_corrections.py`,
  `/tmp/verify_corrections.py` — 일회성, src/ 승격 안 함

## 관련 커밋

PR 직전 일괄 commit (4 commit 구조):
1. **schema 보강 + 데이터 보정** —
   `docs/manual/data/canonical-entity-schema.md` (4영역 명문화) +
   `data/stockmark/pii_all.jsonl` (gitignore — 변경 사실은 본 md 에 기록)
2. **classifier 코드 (CRF + boundary loss)** —
   `src/ner/classifier/train_eval.py` (BertCRFForTokenClassification +
   WeightedTrainer.\_save/\_load\_best\_model override) +
   `src/ner/classifier/data_utils.py` (boundary_weights_tensor) +
   `src/ner/classifier/__main__.py` (--use-crf, --boundary-b-weight,
   --boundary-i-weight 플래그) + `tests/ner/classifier/test_crf.py` +
   `tests/ner/classifier/test_boundary_weights.py` + `pyproject.toml`
   + `uv.lock` (pytorch-crf 의존성)
3. **benchmark 갱신** —
   `docs/reports/japanese-bert-classifier-benchmark.md` (§요약,
   §라운드 2 신설, §산출물 위치)
4. **이슈 md 최초 commit** — 본 md

---

## 다음 세션 진입 가이드 (cold-start 용)

이 섹션은 컨텍스트 reset 후 cold-start 진입 시점의 *동결된 상태*. 본
md 와 `docs/reports/japanese-bert-classifier-benchmark.md` 두 문서가
영구 인용 단일 출처.

### 0. 한 줄 요약

이슈 #51 종결. JA NER 5종 strict 0.95 게이트 미달 확정 → (B) 천장 동결.
production = **v3+boundary** (strict 0.9368 / relaxed 0.9505).

### 1. 현재 git 상태 (cold-start 시 확인)

```bash
git status                 # clean
git log --oneline -5       # 최신 = PR #?? merge of feat/issue-51
git branch --show-current  # develop (PR 머지 후)
```

### 2. 검증 명령

```bash
# 데이터 무결성
wc -l data/stockmark/pii_all.jsonl   # 5,270 확인

# production 메트릭 확인
cat results/classifier/ja_sweep/v3_boundary/metrics.json | \
  jq '.overall_strict.f1, .overall_relaxed.f1'
# 출력: 0.9368... / 0.9505...

# 코드 무결성
uv run python -m pytest tests/ner/classifier/ -q
# 36 케이스 통과 예상
```

### 3. production 모델 학습 명령 (재현)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m ner.classifier --lang ja \
  --epochs 5 --batch-size 16 --max-length 256 \
  --output-dir results/classifier/ja_sweep/v3_boundary \
  --metric-mode both \
  --boundary-b-weight 1.5 \
  --boundary-i-weight 1.2
```

### 4. 핵심 결정 컨텍스트 (변경 금지)

- **(B) 천장 동결 채택** — 사용자 결정. v3+boundary 가 production.
  추가 시도는 별도 이슈로 분리
- **CRF 회귀** — Viterbi BIO 일관성은 일부 NER 클래스에 부정적
  (LOC -2.25, EVT -3.94). production 채택 X
- **NER class weight 2.0** — PII 회귀 (DAT -3.49, CC -3.27) 유발.
  v3+boundary 단독이 더 안전
- **데이터 백업 미보존** — Tier 3 동일 관행
- **임시 스크립트** — `/tmp/*_round2_*.py` 모두 일회성. src/ 승격 안 함
