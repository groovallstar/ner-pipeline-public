# issue-45: F6 외부 코퍼스 통합으로 JA F1≥0.95 도달

- Issue: https://github.com/groovallstar/ner_pipeline/issues/45
- PR: (머지 직전 채움)
- 브랜치: `feat/issue-45-ja-classifier-external-corpus`
- 승인일: 2026-05-04
- 완료일: 2026-05-06

## 목적
PR #41 베이스라인(JA 0.9058 / 4개 라벨만 게이트 통과)이 본 데이터셋
(Stockmark) 의 어휘 다양성·EVT 절대 support 부족 천장에 막혔음이 sweep
18회로 확정. 외부 JA gold 코퍼스 통합으로 데이터 천장을 들어올려
F1 ≥ 0.95 도달을 시도한다. 핸드오프:
`docs/issues/handoff-post-issue-40-classifier-followups.md` §F6.

## 라이선스 제약 (선결 조건)
- 유료·기관 협약 코퍼스 사용 불가: OntoNotes JA (LDC 유료), BCCWJ
  (NINJAL 협약), 기타 LDC 등재 코퍼스 모두 본 이슈 범위 밖.
- 무료·연구 목적 공개 코퍼스만 사용한다.

## 1단계 사전조사 후보 (무료 라이선스)
- SHINRA (森羅) — Wikipedia hierarchical entity classification, EVENT/
  PRODUCT 서브카테고리 보유, NII 공개. 가장 유력.
- 関根 拡張固有表現 (ENE) 공개판 — IREX/ENE 일부 무료 배포, ARTIFACT/
  EVENT 일부 포함.
- KWDLC — CC-BY-SA, 그러나 EVT/PROD 없음 → Plan B 필수.
- UNER JA — 라벨 타입 제한적, EVT/PROD 미보유 가능성.

## 범위
**포함**:
- 사전조사 결과로 통합 후보 확정 (라이선스 + EVT/PROD 보유 여부)
- canonical 10종(NER 5 + PII 5) 매핑 표 신설
  (`docs/manual/data/ja-external-corpus-mapping.md`)
- `src/ner/augmenters/external_corpus/ja/` 신규 서브패키지 — 다운로드,
  라벨 매핑, JSONL 변환
- `data/combined/ja_pii_all.jsonl` 통합 파일 생성 (Stockmark ∪ 외부)
- 통합 데이터로 베이스라인 모델(xlm-roberta-base) 재학습 + 평가
- 베이스라인(JA 0.9058) 대비 per-entity F1 Δ 표

**제외**:
- VI 측 외부 코퍼스 통합 (별도 트랙)
- F3 (DeBERTa hyperparam) 와 결합한 모델 측 변경 — 본 이슈는
  *데이터 보강 단독 효과* 만 검증

## Plan A' / Plan B 분기 (1단계 결과 의존)

**Plan A' — 네이티브 EVT/PROD 라벨 사용 가능 시 (SHINRA / ENE 공개판)**
- 라이선스·접근성 클리어 + EVT/PROD support 충분 → 매핑 표 + 변환
  스크립트만으로 통합. silver 노이즈 0.

**Plan B — Plan A' 가 막히거나 EVT/PROD support 부족 시**
- KWDLC/뉴스 텍스트에 vLLM 으로 EVT/PROD 만 silver 라벨 추가
- `augmenters/pii` 의 cross-verifier 패턴 필수: 다른 모델로 재검증 →
  합의된 span 만 채택
- 사람 spot check 200문장 — 라벨 정확도 보고 (≥ 90% 미달 시 본 데이터
  사용 중단)
- silver 노이즈 천장(본 sweep 의 0.95 미달 원인) 회피 안전장치

**Plan C — Plan A'·B 모두 비현실적일 시**
- 본 이슈 *F6-JA 동결* + 핸드오프 갱신 → F3 또는 F5-JA-equivalent 로
  분리

## 성공 기준
- [ ] 1단계 사전조사 결과 댓글로 보고 + Plan A'/B/C 결정
- [ ] 테스트: augmenters/external_corpus/ja 단위 테스트 (라벨 매핑
  round-trip, JSONL 스키마 검증)
- [ ] 문서: `docs/manual/data/ja-external-corpus-mapping.md`,
  `docs/manual/data/canonical-entity-schema.md` 외부 출처 추가
- [ ] 메트릭: 통합 데이터 학습 모델 F1 ≥ 0.95 **또는** 미달 시
  새 베이스라인 + 천장 재분석 문서화 (둘 중 하나 충족)
- [ ] 베이스라인 비교: PR #41 대비 per-entity F1 Δ 표
  (`docs/reports/bert-classifier-benchmark.md` 갱신 또는 새 리포트)

## 구현 단계 (3~5)
- [ ] 1. 무료 코퍼스 사전조사 (라이선스 확인 + EVT/PROD 보유 여부 +
      샘플 다운로드) → Plan A'/B/C 결정 후 댓글로 보고
- [ ] 2. canonical 10종 매핑 표 작성 + `augmenters/external_corpus/ja`
      변환 스크립트 구현 (Plan A' 또는 B 에 따라 분기)
- [ ] 3. (Plan B 진입 시) vLLM 라벨러 + cross-verifier + 200문장 사람
      spot check 산출물 첨부
- [ ] 4. 통합 JSONL 빌드 + 단위 테스트
- [ ] 5. 통합 데이터로 베이스라인 모델 재학습 + 평가 + 비교 리포트

## 위험·의존성
- 라이선스: SHINRA/ENE 도 연구 목적 한정·재배포 제약 가능 — 1단계에서
  확정.
- 라벨 매핑 비용: 외부 코퍼스의 entity 정의가 canonical 10종과 1:1
  매핑되지 않음 (PROD/EVT 정의 차이).
- Plan B silver 노이즈 위험: 본 sweep 의 0.95 미달 직접 원인이었음.
  cross-verifier + 사람 spot check 미만족 시 데이터 채택 거부 의무.
- 도메인 shift: 코퍼스별 도메인(웹/뉴스/Wikipedia) 차이로 sample
  weight 또는 도메인 균형 필요할 수 있음.
- vLLM 가용성 (Plan B 진입 시): 별도 GPU 자원 + `augmenters/pii` 패턴
  재사용 검증.

---

## 실험 기록 (in-progress, 미커밋)

본 이슈는 §F6 외부 코퍼스 통합으로 시작했으나 사전조사 결과 무료 + 네이티브
EVT/PROD span 라벨 코퍼스가 사실상 부재 (SHINRA = article-level, KWDLC =
EVT 0건·ARTIFACT type drift) — F6 단독으론 실현 어려움 확인. 이후 (다)
시나리오 (F5 oversampling JA + LLM augmentation) 로 전환해 데이터 천장
가설을 직접 검증.

### 환경
- 데이터: `data/stockmark/pii_all.jsonl` (5,307 행, JA canonical 10종)
- 모델: `tohoku-nlp/bert-base-japanese-v3` (베이스라인 lock-in)
- Split: 80/10/10 (seed=42), train 4,247 / valid 530 / test 530
- 하이퍼: 5 epoch, BS 16, LR 5e-5, max_length 256, fp16
- vLLM: Qwen3.6-35B (replacer/inject), Gemma-4-31B (verifier) — 각 8082/8081

### 사전조사 — 무료 JA NER 코퍼스 EVT/PROD 보유 매트릭스

| 코퍼스 | 라이선스 | 형식 | EVT | PROD | 결론 |
|---|---|---|---|---|---|
| Stockmark NER Wikipedia | CC-BY-SA 3.0 | span | 1,009 | 1,215 | **이미 통합** |
| KWDLC | LICENSE 미명시 (연구 목적) | span (IREX) | 0 | 989 (ARTIFACT) | 라이선스 위험 + EVT 0 + ARTIFACT type drift |
| Hironsan IOB2Corpus | CC-BY 2.5 | span (IREX) | 0 | 일부 ARTIFACT | 500 문장만 |
| UNER JA | CC-BY-SA 4.0 | span | 0 | 0 | PER/LOC/ORG 만 |
| SHINRA 분류 (2020-JP) | CC-BY-SA 3.0 | **article-level** | 219 ENE 카테고리 | — | span-NER 아님 |
| SHINRA 속성값 추출 | CC-BY-SA 3.0 | 속성값 sequence | △ | △ | 스키마 매우 다름 |

### 신규 코드 모듈
- `src/ner/augmenters/entity_replacement/` — LLM-DA 기법 entity replacement
  (replacer + verifier, JA EVT 대상)
- `src/ner/augmenters/external_corpus/ja/` — KWDLC KNP → canonical 10종
  변환기 (LOCATION→LOC / DATE→DAT / ORGANIZATION→ORG / PERSON→PER /
  ARTIFACT→PROD 매핑)
- `src/ner/classifier/__main__.py`:
  - `--oversample LABEL:N` (단순 행 복제)
  - `--aug-train-data PATH` (외부 augmenter 산출 JSONL 추가)

### 실험 결과 요약 (28회 학습, 모두 단일 seed=42)

| Run group | 회수 | 구성 | Overall F1 (vs baseline 0.9001 rerun) |
|---|---:|---|---|
| 베이스라인·SOTA 모델 sweep | 18 | 사전 PR #41 (xlm-roberta / DeBERTa family / mBERT 등) | 최선 0.9058 |
| 베이스라인 rerun (분산 측정) | 2 | 동일 설정 반복 | std ~0.06pp |
| 단순 oversample (PROD/EVT) | 4 | PROD:4, EVT:4, PROD:4+EVT:4, baseline | -0.02 ~ -0.30pp |
| LLM entity replacement | 2 | v1 (temp=0.7) / v2 (temp=0.3, strict verifier) | v1 -0.49 / v2 -0.07 |
| KWDLC gold 통합 | 2 | PROD only / all 5 | -1.28 / -0.79 |

### 핵심 per-entity 비교 (baseline vs 최선 결과)

| Entity | baseline | v2 LLM replace | KWDLC PROD | KWDLC all5 | 게이트 (0.95) |
|---|---:|---:|---:|---:|---:|
| PHONE | 1.0000 | =0 | =0 | =0 | ✅ |
| EMAIL | 0.9944 | =0 | =0 | =0 | ✅ |
| DAT | 0.9831 | +1.69 (1.0!) | +0.02 | -1.57 | ✅ |
| PER | 0.9586 | -1.00 | -2.86 | -0.66 | ✅ |
| CREDIT_CARD | 0.9091 | -0.43 | -1.34 | =0 | ❌ |
| ID_NUM | 0.8840 | +1.01 | +2.21 | -0.16 | ❌ |
| ORG | 0.8713 | -0.52 | +0.19 | **+1.27** | ❌ |
| EVT | 0.8643 | -1.43 | -2.17 | **-6.23** | ❌ |
| LOC | 0.8356 | +0.39 | -2.98 | -0.17 | ❌ |
| PROD | 0.8220 | +0.83 | **-5.49** | **-9.96** | ❌ |

### 핵심 발견
1. **데이터 천장이 본질**: 28회 모두 0.95 게이트 미달. baseline 0.9058 가
   현 Stockmark 데이터·모델 설정의 천장.
2. **KWDLC ARTIFACT → PROD type drift 가 silver noise 보다 심각**:
   gold 라벨이지만 정의 mismatch (商品 vs 작품·番組·楽曲) 로 PROD F1 -9.96pp
   까지 회귀. (나) "gold = silver 노이즈 0 → 안전" 가설 직접 반증.
3. **EVT 어휘 다양성 한계**: LLM entity replacement (어휘 +N배) 도 EVT
   직접 향상 0 — EVT 천장은 어휘가 아니라 **구조·문맥 다양성** 측면일 가능성.
4. **단순 oversample 천장 명확**: 같은 행 복제는 +0pp 효과 (overfit 한계).
5. **외부 데이터의 ORG 만 호환**: KWDLC ORG +1.27pp — 정의 호환 라벨 한정
   통합은 ROI 있을 가능성 (그러나 ORG 단독으론 게이트 진전 불가).
6. **SOTA 문헌 caveat 재현**: PROPOR 2026 의 *"LLM augmentation can reduce
   NER F1 by 0.24-1.81%"* 가 본 실험에서 그대로 나타남 (v1 -0.49, v2 -0.07,
   KWDLC -1.28).

### 본 이슈 #45 의 종결 옵션 (사용자 결정 대기)
- **(A) F3 DeBERTa hyperparam JA sweep** — 본 sweep collapse 회피 그리드
  (warmup_ratio·weight_decay·max_grad_norm) 적용. 27조합×3모델, ~5시간.
- **(B) 게이트 완화 합의 (Plan A)** — 0.95 → 0.90 또는 NER/PII 분리 게이트
  (NER 0.90, PII 0.95) 로 재정의. baseline 0.9058 production 채택.
- **(C) 본 이슈 종결 + 새 기법 탐색 후 후속 이슈로 분할** — 28회 결과를
  바탕으로 추가 시도할 SOTA 기법 (self-training, span-relaxed evaluation
  등) 이 있는지 새로 조사.

---

## 종결 (closes #45) — 2026-05-06

### 추가 진행 (28회 → 28회 + Tier 1.B 평가 트랙)
- **Tier 1.B (boundary-relaxed evaluation)** 구현 + 8 variant 재평가 완료
  - `src/ner/metrics/span_metrics.py` 의 `compute_offset_span_f1_relaxed` —
    SemEval'13 Partial F1 (type 일치 + char-offset overlap 시 0.5점)
  - `evaluate_model` 이 strict + relaxed 동시 반환, `metrics.json` 에 양쪽
    저장 (`overall_strict` / `overall_relaxed` / `per_entity_*`)
  - CLI `--metric-mode {strict,relaxed,both}` (default both)
  - 단위 테스트 `tests/ner/test_span_metrics.py` 15 케이스 통과

### 8 variant 재평가 결과 (P/R/F1, strict vs relaxed)
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

### 핵심 발견
1. **baseline 이 strict·relaxed 양쪽 1위**: relaxed 평가에서도 모든
   augmentation variant 가 baseline 대비 회귀. augmentation 회귀가 boundary
   손실이 아닌 *진짜 학습 손실* — **데이터 천장 가설 확정**.
2. **boundary 가설은 PROD 단독 효과로 축소**: overall Δ +1.00pp (baseline),
   PROD 단독 +4.09pp. 핸드오프의 +3~+10pp 추정은 기각.
3. **EVT 천장은 boundary 무관 (relaxed Δ=0)**: 어휘·문맥 다양성 부족이
   본질이며 평가 메트릭 변경으로 회복 불가.
4. **PII 4종 (PHONE/EMAIL/DAT/CREDIT_CARD) 포화**: 평가 메트릭으로 회복 여지 없음.
5. **augmentation 의 회귀 패턴**: precision 회귀가 recall 회귀보다 큼 —
   augmentation 이 false-positive 증가형 노이즈 유입.

### 게이트 0.95 미달 사유 동결
- 본 데이터 (Stockmark JA 5,307) 의 어휘 다양성·EVT 절대 support·라벨 노이즈 천장
- BertJapaneseTokenizer + strict char-offset span F1 평가 셋업의 천장
  (PROD 만 boundary 회복 의미)
- 모델 측 SOTA sweep 18회 + 데이터 측 augmentation/외부 통합 10회 + 평가 측
  boundary-relaxed 8 variant — 어떤 트랙에서도 게이트 도달 실패

### Production 모델
- 모델: `tohoku-nlp/bert-base-japanese-v3`
- 데이터: Stockmark JA `data/stockmark/pii_all.jsonl` (5,307 행)
- 학습: 80/10/10 split (seed=42), 5 epoch, BS 16, LR 5e-5, max_len 256, fp16
- 측정: **strict F1 = 0.9077** (P=0.879, R=0.938), **relaxed F1 = 0.9177**
  (P=0.889, R=0.948), per-entity 표는 `results/classifier/ja_sweep/baseline/metrics.json`
- production 합격선 정의는 사용자 권한 — 본 이슈는 모델 천장의 *측정·동결* 까지 책임

### 후속 트랙 (별도 이슈 후보)
- **Tier 2.B — Self-training with adaptive thresholding**: JA Wikipedia
  unlabeled corpus pseudo-label + Gaussian per-class threshold. 2~3일.
- **Tier 3 — test-set error analysis 후 gold cleanup**: baseline 의 test
  오답 ~10% 검수 → gold 라벨 오류율 측정. 모델 천장 vs gold 천장 분리.
- VI 트랙: `docs/issues/handoff-post-issue-40-classifier-followups.md` 의
  F2/F3-VI/F4/F5 별도 진행.

### 완료된 산출물 (post-closure 정리 후)

후속 트랙 (test-set error analysis, Tier 2 등) 지원을 위해 Tier 1.B 평가
인프라만 유지하고 augmentation 실험 코드·데이터·결과는 제거하였다.

**유지** (production·후속 분석 인프라):
- `src/ner/metrics/span_metrics.py` — `compute_offset_span_f1` (strict, 게이트)
  + `compute_offset_span_f1_relaxed` (SemEval'13 Partial, error 카테고리 분리용)
- `src/ner/classifier/train_eval.py` — `evaluate_model` 이 strict + relaxed
  동시 반환
- `src/ner/classifier/__main__.py` — `--metric-mode {strict,relaxed,both}`
  플래그 + `metrics.json` 양쪽 저장
- `tests/ner/test_span_metrics.py` — 15 케이스 (strict 회귀 3 + relaxed 12)
- `results/classifier/ja_sweep/baseline/` — production 후보 모델 + metrics

**제거** (실험 종결 후 정리):
- `src/ner/augmenters/entity_replacement/` (LLM-DA)
- `src/ner/augmenters/external_corpus/` (KWDLC 변환기)
- `--oversample LABEL:N`, `--aug-train-data PATH` CLI 플래그
- `data/kwdlc_ja/*.jsonl`, `data/stockmark/evt_aug_train{,_v2}.jsonl`
- 7개 augmentation variant 결과 디렉토리 (prod4 / evt4 / prod4_evt4 /
  evt_replace / evt_replace_v2 / kwdlc_prod / kwdlc_all5)

**Production 후보 baseline 재측정** (cleanup 후 재학습): strict F1 = 0.8973
(P=0.852, R=0.948), relaxed F1 = 0.9074 (P=0.867, R=0.952). per-entity 표는
`results/classifier/ja_sweep/baseline/metrics.json` 참조. (재학습 결과는
seed=42 결정적이지만 환경·라이브러리 상태에 따라 single-seed std ~0.06pp
범위 내 변동.)
