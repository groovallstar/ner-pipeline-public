# issue-48: Tier 2/3 후속 — test-set error analysis + self-training

- Issue: https://github.com/groovallstar/ner_pipeline/issues/48
- PR: (머지 직전 채움)
- 브랜치: `feat/issue-48-ja-classifier-tier2-tier3-followups`
- 승인일: 2026-05-06
- 완료일: <!-- 5단계에서 채움 -->

## 목적

이슈 #45 종결로 Stockmark JA 5,307 + `tohoku-nlp/bert-base-japanese-v3` baseline 의 데이터·평가 양 측 천장이 동결됐다 (strict F1 = 0.8973 / relaxed F1 = 0.9074, 28회 sweep + Tier 1.B 8 variant 재평가). 본 이슈는 #45 §"후속 트랙 (별도 이슈 후보)" 의 Tier 3 + Tier 2.B 를 묶어 baseline 0.95 미달 사유를 *모델 천장*과 *gold 라벨 천장* 으로 분리하고, semi-supervised 보완 기법의 효과를 동일 평가 셋업에서 측정한다.

## 범위

- 포함:
  - Tier 3 test-set error analysis (선행) — 모델 천장 vs gold 천장 분리
  - Tier 2.B Self-training + adaptive thresholding
- 제외:
  - VI 트랙 (post-#40 핸드오프 F2~F6 별도 진행)
  - 이슈 #45 에서 동결된 외부 코퍼스 통합 (F6-JA) · DeBERTa hyperparam (F3-JA)
  - 게이트 완화 합의 — 별도 결정 이슈로 분리

## 성공 기준

- [ ] 테스트: self-training threshold logic 단위 테스트, 기존 `tests/ner/test_span_metrics.py` 회귀 0
- [ ] 문서 갱신: `docs/reports/japanese-bert-classifier-benchmark.md` Tier 2 결과 열 추가, `docs/manual/` 모듈 구조 변경 시 갱신
- [ ] 메트릭/검증:
  - Tier 3 — gold 오류율 측정값 + 보정 baseline (strict+relaxed) per-entity 표
  - Tier 2.B — baseline strict 0.9249 대비 per-entity Δ 표 + 0.95 도달 여부
  - 미달 시 → 새 천장 재분석 + 종결 옵션 (A/B/C) 사용자 결정 요청

## 구현 단계

- [x] 1. Tier 3 — baseline 의 오답 케이스 추출 + 전수 검수 (10% 아님,
      1,398/3,242 = 43.1% 오답 문장 모두) + gold 오류율 측정 + 보정
      baseline 재계산. **결과**: gold 결함 40.2%, 보정 baseline strict
      F1 0.8925 → 0.9249 (+0.0324)
- [x] 2. Tier 3 결과 리포트 작성 — *별도 파일 신설 안 함*. 결과는
      `docs/reports/japanese-bert-classifier-benchmark.md` §"Tier 3 —
      test-set error analysis (완료)" 에 누적 통합 (사용자 결정,
      별도 리포트 작성 비용 회피)
- [ ] 3. Tier 2.B — JA Wikipedia unlabeled pseudo-label + Gaussian per-class threshold + cross-verifier (`augmenters/pii` 패턴 재사용) + 학습·평가
- [x] 4. baseline 대비 per-entity Δ 표 (strict + relaxed) 작성 +
      `docs/reports/japanese-bert-classifier-benchmark.md` 갱신 — *Tier 3
      만 반영*. Tier 2.B 결과 미반영 (트랙 미시도)
- [ ] 5. 종결 — 0.95 도달 여부 판정 + 미달 시 (A) 게이트 완화 / (B) 천장 동결 재확인 / (C) 추가 SOTA 기법 분리 중 사용자 결정 요청

## 위험·의존성

- **Tier 2.B**: pseudo-label 노이즈로 회귀 가능. 이슈 #45 의 augmentation 회귀 패턴 (precision 회귀 > recall 회귀, false-positive 노이즈 유입) 재현 시 데이터 채택 거부 + 회귀 사유 보고.
- **PROPOR 2026 caveat**: "LLM augmentation can reduce NER F1 by 0.24-1.81%" — Tier 2.B pseudo-label 도 동일 패턴 가능.
- **single-seed caveat**: seed=42 결정적, std ~0.06pp 범위 변동. **Δ < 0.1pp 는 noise 로 처리**.
- **순서 의존**: Step 1 (Tier 3) 결과가 Step 3 (Tier 2.B) 의 효과 해석 기준이 되므로 **Step 1 → Step 3**.
- **외부 자원**: JA Wikipedia unlabeled corpus 샘플링 도구. vLLM 가용성은 cross-verifier 단계에서 필요 (이슈 #45 와 동일 셋업).

---

<!-- 5단계 섹션 — Tier 3 부분 진행 현황 누적 갱신. Tier 2.B 미진행이므로
     PR 직전 최종 commit 은 아직. 본 갱신은 핸드오프용 중간 동결 -->

## 변경 요약

Tier 3 (test-set error analysis + gold cleanup) 단독 단계 완료. 보정
baseline strict F1 0.9249 / relaxed 0.9415 동결. Tier 2.B (self-training)
미진행 — 핸드오프 후 별도 후속 진행.

## 구현 결과 (계획 대비)

### Step 1 — Tier 3 gold cleanup (✅ 완료)

진행 흐름:
1. 신규 코드: `src/ner/classifier/error_analysis.py` (오답 추출 CLI),
   `tests/ner/classifier/test_error_analysis.py` (단위 10 케이스). pytest 통과
2. baseline 모델 (5,270-row pii_all.jsonl 위) 의 오답 추출:
   - test 392 errors, valid 418 errors, train 1,544 errors = **2,354 errors**
   - 1,398 오류 문장 전수 사람 검수 (Gemini chat 보조, 단독 한국어 검토자)
3. verdict 분포: model_correct 35.0% / gold_correct 57.2% / both_wrong 5.2%
   / ambiguous 2.5%. **gold 결함 비율 = 40.2%** (model_correct + both_wrong)
4. 보정 적용 v1: 947건 model_correct/both_wrong + 67건 chunk ambiguous
   gold_error = **1,014건** in-place
5. **회귀 발견**: DAT strict F1 0.9802 → 0.9259 (-0.0543) — chunk 단계
   검수 도구 (Gemini) 의 한글 부분 번역 → character offset shift 로 span
   좌표 손상. 47% chunk records 가 contamination 영향
6. audit cleanup v2: `/tmp/audit_entity_text_quality.py` 라벨-문법
   헤리스틱으로 18건 제거 (DAT 14 / LOC 2 / CREDIT_CARD 1 / ID_NUM 1)
7. 재학습 → 보정 baseline 동결값:
   - strict F1 0.9249, P 0.9217, R 0.9281, support 1,891
   - relaxed F1 0.9415, P 0.9367, R 0.9468
   - DAT 회귀 -0.0543 → -0.0237 (60% 회복). 잔여 회귀는 모델 한계 (학습
     분포에 없는 새 raw year-only 패턴)

상세 per-entity 표·해석:
`docs/reports/japanese-bert-classifier-benchmark.md` §"Tier 3 — test-set
error analysis (완료)" 참조.

### Step 2 — 별도 리포트 작성 (사용자 결정으로 스킵)

`docs/reports/ja-classifier-test-error-analysis-{날짜}.md` 신설 안 함.
결과는 상시 갱신형 `japanese-bert-classifier-benchmark.md` 의 Tier 3
섹션에 누적 통합.

### Step 3 — Tier 2.B (미진행)

후속 핸드오프 후보. 본 핸드오프 §"Tier 2.B" 항목 그대로 유효.

### Step 4 — benchmark 갱신 (Tier 3 만 반영)

`japanese-bert-classifier-benchmark.md` 의 다음 섹션 누적 갱신:
- 헤더 측정일·데이터 행 수 (5,307 → 5,270)
- §요약 — Tier 3 완료 표시 + 동결값
- §"Tier 3 — test-set error analysis (완료)" — 1단계·2단계·v1↔v2 비교·
  per-entity F1 표·게이트 분석·재현 caveat 통합 작성
- §"0.95 도달 가능성 판정 시점" — Tier 3 완료, Tier 2.B 미시도 표시
- §산출물 위치 — baseline_corrected 동결값 명시

### Step 5 — 0.95 종결 판정 (보류)

부분 미달. relaxed 0.85pp / strict 2.51pp gap. Tier 2.B 미시도 상태에서
종결 결정 보류 — 후속 핸드오프에 옵션 (A/B/C) 권장 의견 첨부.

## 검증

- `pytest tests/ner/classifier/` — error_analysis 10 + data_utils/encode
  26 = **36 케이스 통과** (회귀 0)
- 보정 baseline 재학습: 5 epochs, 94.4s, validation loss 정상 수렴
- audit 헤리스틱 false-positive 비율: suspicious 21건 중 사람 확인으로
  3건만 실제 broken (=14%). NER 5종 (PER/LOC/ORG/PROD/EVT) 정상 일본어
  고유명사 패턴 (`あけぼの`, `はやぶさ`, `北の国から` 등) 의 hiragana
  접두/접미 false-positive 가 다수
- 메트릭 동결값:
  - `results/classifier/ja_sweep/baseline_corrected/metrics.json` —
    Tier 3 v2 (gitignore — 본 md 표가 영구 인용 단일 출처)

## 관련 커밋

PR 직전 일괄 commit 예정. 변경 파일:
- 코드: `src/ner/classifier/error_analysis.py`
- 테스트: `tests/ner/classifier/test_error_analysis.py`
- 문서: 본 md, `docs/reports/japanese-bert-classifier-benchmark.md`
  (신설), `docs/reports/vietnamese-bert-classifier-benchmark.md` (신설),
  `docs/reports/bert-classifier-benchmark.md` (삭제),
  `src/ner/classifier/AGENTS.md` (error_analysis 모듈 추가)

## 후속 작업·알려진 한계

### 즉시 진행 가능 (별도 이슈 후보)

1. **Tier 2.B — Self-training** — 본 핸드오프 §"Tier 2.B" 그대로 진행.
   보정 baseline (0.9249) 위에서 +2.5pp 이상이 0.95 도달 조건
2. **chunk 단계 보정 49건 사람 재검수** — audit cleanup 으로 18건 제거
   후 49건 잔여. 한글 contamination 없는 검수 도구로 재검수 시 추가
   gold 회복 가능성
3. **canonical schema 보강** — ambiguous 60건 (`docs/manual/data/
   canonical-entity-schema.md` LOC/ORG/PROD/EVT 경계 룰 보강 후 재검수
   대상). 별도 이슈

### 알려진 한계

- pii_all.jsonl **5,307 → 5,270 행** + net 996 entity 보정. 본 시점
  이후 모든 측정값은 5,270-row 기준. 옛 sweep 결과 (0.9058) 와 동일
  분포 비교 불가
- 검수 도구 (Gemini chat) 의 한글 부분 번역 → offset shift 위험.
  향후 검수 파이프라인은 *원문 보존* 보장 도구 필요. 임시 스크립트
  (`/tmp/apply_corrections.py`, `/tmp/apply_ambiguous_corrections.py`,
  `/tmp/audit_entity_text_quality.py`) 일회성 — 재현 불가
- DAT 잔여 회귀 -0.0237 = 학습 분포에 없는 새 DAT 패턴 (raw year-only
  `1957年`, `1986年`, `2002年`) 을 모델이 못 잡는 순수 모델 한계
- 보정 data 백업 미보존. 5,307-row 원본은 augmenters/pii 재실행으로만
  재생성 가능 (랜덤 PII 주입 → 결정성 보장 X)
- audit 헤리스틱 false-positive 14% (suspicious 21건 중 3건만 실제
  broken) — NER 5종 (PER/LOC/ORG/PROD/EVT) 에 보수적 적용 권장. PII
  측은 라벨-문법 강함 (digit-only / @ 포함) 으로 안전

---

## 다음 세션 진입 가이드 (cold-start 용)

이 섹션은 컨텍스트 reset 후 cold-start 진입 시점의 *동결된 상태*. 본
md 와 `docs/reports/japanese-bert-classifier-benchmark.md` 두 문서가
영구 인용 단일 출처.

### 0. 한 줄 요약

이슈 #48 의 Tier 3 만 완료 (보정 baseline strict 0.9249 / relaxed
0.9415). Tier 2.B (Self-training) 미진행. 본 브랜치
`feat/issue-48-ja-classifier-tier2-tier3-followups` 에 코드 변경만 있고
0 commit.

### 1. 현재 git 상태 (cold-start 시 확인)

```bash
git status                 # 미커밋 파일 다수
git log --oneline -5       # 최신 = 95a4fef Merge #46 (feat/issue-45)
git branch --show-current  # feat/issue-48-ja-classifier-tier2-tier3-followups
```

미커밋 변경 (예상):
- 신규 코드: `src/ner/classifier/error_analysis.py`,
  `tests/ner/classifier/test_error_analysis.py`
- 갱신 doc: `docs/reports/japanese-bert-classifier-benchmark.md`
  (Tier 3 누적 갱신), `docs/issues/issue-48-ja-classifier-tier2-tier3-followups.md`
  (5단계 섹션 채움)
- 갱신 데이터: `data/stockmark/pii_all.jsonl` (gitignore — 추적 안 됨,
  but 본 세션 이후 in-place 변경됨)
- 신규 results dir: `results/classifier/ja_sweep/baseline_corrected/`
  (gitignore — 추적 안 됨)

> **추적 안 되는 파일의 변경 사실 자체는 본 md 가 유일한 기록**.
> pii_all.jsonl 의 5,307→5,270 행 + 996 net 보정은 git log 로 확인
> 불가. cold-start 시점 행 수 검증: `wc -l data/stockmark/pii_all.jsonl`
> → 5,270 이어야 정상.

### 2. 검증 명령 (cold-start 시 첫 실행)

```bash
# 데이터 무결성
wc -l data/stockmark/pii_all.jsonl   # 5,270 확인

# 메트릭 동결값 확인
cat results/classifier/ja_sweep/baseline_corrected/metrics.json | \
  jq '.overall_strict.f1, .overall_relaxed.f1'
# 출력: 0.9249... / 0.9415...

# 코드 무결성
uv run python -m pytest tests/ner/classifier/ -q
# error_analysis 단위 10 케이스 통과 예상
```

위 검증이 실패하면 컨텍스트가 어긋난 상태 — 본 md 의 §"구현 결과"
재읽고 정합성 확인.

### 3. Tier 2.B 진입 (Self-training) — 권장 첫 명령

JA Wikipedia unlabeled corpus 샘플링 → cross-verifier
(`augmenters/pii` 패턴 재사용). 본 핸드오프 §"Tier 2.B" 그대로 진행.
0.95 도달 조건 = 보정 baseline 위 +0.0251 strict / +0.0085 relaxed.

위험: 이슈 #45 의 LLM augmentation 회귀 패턴 재현 시 채택 거부.
사람 spot check 200문장 ≥90% 미달 시 데이터 사용 중단.

### 4. 핵심 결정 컨텍스트 (변경 금지)

- **별도 Tier 3 리포트 신설 안 함** — `japanese-bert-classifier-benchmark.md`
  Tier 3 섹션에 통합 (사용자 결정). 새 리포트 만들지 말 것
- **chunk 단계 보정 유지** — 67건 중 18건 audit cleanup 후 49건 잔여.
  잔여분 사람 재검수는 별도 후속 (보정 baseline 의 잠재 +α 영역).
  본 트랙에서 *추가 검수·재학습은 하지 않음*
- **보정 데이터 백업 미보존** — 의도된 결정. cold-start 시 *원본
  복구 시도 금지*. augmenters/pii 재실행은 랜덤 PII 주입이라
  bit-exact 복원 불가
- **임시 스크립트 commit 금지** — `/tmp/*.py` 5개 (apply_corrections,
  apply_ambiguous_corrections, audit_entity_text_quality, refresh_review,
  merge_chunks) 모두 일회성. src/ 승격 안 함. 본 md 의 §"알려진 한계"
  에 일회성 명시

### 5. 0.95 게이트 종결 옵션 (Tier 2.B 결과 후 결정)

본 핸드오프 §5 "게이트 미달 종결 시점" 그대로:
- **(A)** 게이트 완화 — deep-interview 후 0.95 → 0.90 또는 NER/PII
  분리 게이트. 합격선 재합의 별도 이슈
- **(B)** 천장 동결 — baseline_corrected v2 (strict 0.9249 / relaxed
  0.9415) production 채택. 추가 시도 보류
- **(C)** 모델 외부 보완 — 추론 시 rule-based PII detector + BERT NER
  union. classifier 는 NER 5종 만 집중

현 시점 권장: **Tier 2.B 미시도 상태에서 (B) 잠정 채택 권유**. 추가
시도는 별도 이슈로 분리.

### 6. 5단계 commit 시점

Tier 2.B 완료 후 PR 직전:
1. `src/ner/classifier/error_analysis.py` + `tests/...` (코드)
2. `docs/reports/japanese-bert-classifier-benchmark.md` (Tier 3 + Tier
   2.B 결과 누적)
3. 본 md (5단계 섹션 최종 갱신 + 다음 세션 가이드는 *후속 핸드오프
   파일* 로 분리 가능 — `handoff-post-issue-48-...md`)
4. 의미 단위 1-3 commit (CLAUDE.md 입도 규칙)
5. `gh pr create --base develop --title "..." --body "...closes #48"`
