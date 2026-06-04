# issue-71: JA classifier ORG 정밀도 회복 — 청정 K-fold 진단 + 표적 처방

- Issue: https://github.com/groovallstar/ner_pipeline/issues/71
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-71-ja-org-precision-recovery`
- 승인일: 2026-06-04

## 목적

청정 K-fold 기준선(base 단독, overall strict 0.9195)에서 **전체 오류의
최대 기여자 ORG**(support 5,133 = 28%, F1 0.8893, P 0.8665 / R 0.9133)
의 FP/FN 경로를 pooled 예측(5,270문장)으로 정밀 진단하고, 그 근거로
ORG 표적 처방 1개를 설계·구현해 10-fold 로 재측정한다.

## 왜 진단이 선행하는가 (핵심 전제)

- ORG 는 **정밀도 한계**(P 0.8665 < R 0.9133 = 과예측). leak 은
  HALLUCINATION + BOUNDARY + TYPE_MISMATCH 중 어디인지에 따라 처방이
  완전히 갈린다.
- #56 의 ORG 진단(환각 28건 = 최대 leak, 47%)은 **구 프로토콜**
  (v3+boundary, 단일 527문장, S7 negative 적용 *전*) 기반이다. 현 청정
  기준선은 base 단독(S7/S8 미적용)이라 그 패턴이 유지된다는 보장이 없다
  → **재진단 없이는 처방을 정할 수 없다.**

## 정직한 한계 (게이트와의 관계)

ORG 가 완벽(F1→1.0)해져도 overall 기여는 ~+1.7pp(28% 가중) 수준 →
overall 0.95 게이트 단독 도달은 아니다. 본 이슈는 *게이트를 향한 최대
단일 레버*이지 게이트 도달 이슈가 아니다. (PROD 0.7943 / EVT 0.8494 는
support 가 작아 별도 이슈.)

## 단계 (다섯)

### 1. 진단 도구 확장 — `error_analysis.py` pooled-예측 입력 경로

현 `error_analysis.py` 는 단일 모델·단일 split 에 `run_inference` 를
돌려 진단한다. K-fold 는 이미 fold 별 `test_predictions.json`
(`{text, gold_spans, pred_spans}`, span = `{type, start, end}`)을
저장하므로 **재추론 없이** 그대로 진단할 수 있다.

- 추론 이후 파이프라인(classify → aggregate → confusion → top-N →
  파일 쓰기)을 `_analyze_sentences(sentences, ...)` 헬퍼로 추출
  (기존 `run_error_analysis` 는 이를 호출하도록 리팩터 — 동작 불변).
- 신규 `run_pooled_error_analysis(fold_dirs, output_dir, ...)`:
  fold 별 `test_predictions.json` 을 합쳐 `sentences` 로 변환 후
  `_analyze_sentences` 호출. fold 간 중복 text 검증(kfold_pool 과 동일
  규칙) 재사용.
- CLI: `--from-predictions --fold-dirs ...` 모드 추가
  (`--model-path`/`--tokenizer-name` 과 상호배타, BC 유지).
- 단위 테스트: pooled 입력 → sentence_results 개수·aggregate 손계산
  일치 + 중복 text ValueError. (`tests/ner/classifier/test_error_analysis.py` 확장)

### 2. ORG 진단 실측 (재학습 없음)

```bash
uv run python -m ner.classifier.error_analysis --from-predictions \
  --lang ja \
  --fold-dirs results/classifier/ja_sweep/kfold10_phonediv_noextra/fold{0..9} \
  --output-dir results/classifier/ja_sweep/kfold10_phonediv_noextra/diag_org \
  --with-diagnosis --diagnosis-fp-types ORG --diagnosis-fn-types ORG \
  --diagnosis-top-n 80
```

산출물: `confusion_matrix.md` + `fp_top_ORG.md` + `fn_top_ORG.md`
(5,133 ORG span 전수 기준). 정량화 목표:
- ORG FP 를 HALL / BOUNDARY / TYPE_MISMATCH 비율로 분해
- HALL 이면 surface 군집(단일 한자·영어 약어·일반명사·시설/역명 등)
- BOUNDARY 면 어긋남 패턴(인용부호·복합 조직명·영문 모델명 등)
- TYPE_MISMATCH 면 혼동 상대 type(ORG↔LOC/PROD)

### 3. 처방 설계 — **진단 게이트 후 결정** (중간 승인 1회)

진단 비율에 따라 분기 (사용자에게 진단 결과 제시 후 1개 확정):

| 진단 우세 경로 | 후보 처방 | 비용 |
|---|---|---|
| HALLUCINATION | ORG 표적 negative oversample (#58 S7 방식) **단, fold-aware 생성 필수** | ★★★ (augmenter fold-aware 개조 + 10 fold 재학습) |
| BOUNDARY | boundary loss weight 재튜닝(`--boundary-b/i-weight` 기존 knob) 또는 경계 정책 데이터 보정 | ★ (학습-side, extra 무) |
| TYPE_MISMATCH | ORG↔상대 type 대비 보강 | ★★ |

> **fold-aware 경고**: 처방이 extra 데이터(negative 등)면, #69 가 적발한
> train/test 중복 인플레이션 재발을 막기 위해 후보를 **각 fold 의 train
> 행에서만** 선별해야 한다. `ja.negative` 는 현재 전체 행 대상이므로
> fold-aware 모드 추가가 본 단계에 포함된다(이 경로의 비용이 가장 큼).
> 학습-side 처방(boundary 재튜닝)이면 extra·fold-aware 불필요.

### 4. 10-fold 재측정

확정 처방을 적용해 fold 0~9 학습(GPU 0, ~50분) → `kfold_pool` 로 pooled
산출 → 기준선과 per-entity diff. 판정:
- **실효 조건**: ORG F1 > 0.8893 + 1.5pp (= > 0.9043, 노이즈 바닥 초과)
- **회귀 가드**: overall ≥ 0.9195 − noise, 타 entity 1.5pp 이상 하락 없음

### 5. 문서 + 마무리

히스토리 2편(`japanese-bert-classifier-history-2.md`)에 진단·처방·재측정
실험 섹션 추가(자기서술형 명칭), 본 이슈 md 에 결과·검증 추가 →
정합성 확인 후 최초 커밋 → PR(`closes #71`).

## 커밋 입도 (예정)

- 진단 도구 확장 + 테스트 = 1 커밋 (`classifier/`, refs #71)
- 처방 구현 = 1 커밋 (처방이 augmenter fold-aware 면 `augmenters/ja/`
  별도 모듈이므로 분리; boundary 재튜닝이면 코드 변경 없을 수 있음)
- 문서(히스토리 2편 + 이슈 md) = 1 커밋 (`docs:`)

## 위험·의존성

- 진단은 재학습 0 (기존 fold 예측 재사용) — 빠르고 확정적.
- 재측정은 GPU 10회 학습. 타 세션 `results/` 정리와 충돌 주의(#69 전례).
- 처방이 negative 경로면 fold-aware 생성 정확성이 결과 타당성을 좌우.

## 결정 로그 (append-only)

- 2026-06-04: 등록. scope = 진단+처방+재측정 통합(사용자 선택).
  처방은 진단 게이트 후 중간 승인 1회로 확정하는 구조.
- 2026-06-04: 계획 승인 — 진단 우선 + 3단계 중간 승인 게이트로 진행.
- 2026-06-04: 진단 완료(pooled 5,270문장). ORG FP 722 = clean 환각 391
  (54%) / 경계 176 / TYPE_MISMATCH 79 / ambiguous 환각 76. 경계 176건은
  99%가 토큰 경계로 표현 가능 → 토크나이저 아닌 모델 추론 문제로 확인.
- 2026-06-04: 처방 B(boundary I-weight 브래킷 I=1.5/2.0/2.5 × 10-fold)
  측정 → 전부 노이즈 바닥(±1.2pp) 내, 경계 버킷 최선 −13/176(7%)만 감소.
  P↔R 맞교환(교훈 #4)일 뿐 net 무효 → **B 폐기**(산출물 51GB 삭제).
- 2026-06-04: 피벗 결정 — 진단이 가리키는 근본 원인은 gold 품질/일관성.
  스키마(`canonical-entity-schema.md`)상 政府·軍·의회·정당·위원회·국제
  기관·건물·약어회사 = ORG → clean 환각 상당수가 gold 누락(모델이 맞음).
  처방을 **스키마 기반 gold 일관성 보정**으로 전환(사용자 선택).
  안티-순환 원칙: 라벨 변경은 *스키마 근거*로만, 모델 예측은 후보 발굴
  용도로만 사용.
- 2026-06-04: 보수적 ~50건 보정은 노이즈 바닥 내(측정 불가) 확인 →
  사용자 결정으로 **ORG gold 전수 조사(LLM 보조)**로 확대. 방법: 독립
  vLLM 2개(gemma:8081 / qwen:8082)로 5,270문장 ORG 라벨링 → 두 모델
  overlap 합의 vs gold diff 로 GAP(누락)/OVER(과라벨) 후보 추출 →
  Claude 가 스키마로 판정(회색지대는 본 후 정책 결정). 게이트 도달은
  목표 아님. 라벨링은 vLLM(대량 throughput), 판정은 Claude(BERT 와
  독립 — 순환 아님).

## 미해결 질문 (open question) — 해소

- ORG leak 우세 경로 → clean 환각 391 + 경계 176. 경계는 모델 추론
  문제(99% 토큰 표현 가능), 환각은 노이즈+gold 누락 혼재.
- 처방: boundary loss(무효) → gold census(데이터 품질↑·F1 무효).
  negative oversample 경로는 진단상 ambiguous 필터에 막혀 미채택.

---

## 변경 요약

- `error_analysis.py`: 추론 이후 파이프라인을 `_analyze_sentences` 로
  추출하고, K-fold pooled 예측을 재추론 없이 진단하는
  `--from-predictions --fold-dirs` 경로(`load_pooled_predictions` +
  `run_pooled_error_analysis`) 추가. 단위 테스트 3종 추가. BC 유지.
- 데이터: LLM 보조 census 로 찾은 명명 ORG 누락 89건을
  `data/stockmark/pii_all_phonediv.jsonl` 에 병합(ORG 5,133 → 5,222).
- 문서: 히스토리 2편에 ORG 조사 섹션, classifier AGENTS.md 갱신.

## 검증

- 테스트: `pytest tests/ner/classifier/` → **53 passed** (신규 3 포함,
  회귀 없음). ruff 클린.
- 진단 도구: `--from-predictions` 로 pooled 5,270문장 ORG FP/FN 전수
  분해 산출(confusion matrix + surface dump) 정상.
- 처방 B(boundary I-weight ×3 10-fold): 전부 노이즈 바닥 내 → 미채택.
- 처방 census(2-모델 합의 +89): 보정 후 overall 0.9180 / ORG 0.8903
  — 노이즈 바닥 내(데이터 품질만 개선).

## 결과

세 각도(모델 boundary loss / 보수 보정 / LLM 전수 census) 모두 ORG F1 을
노이즈 위로 못 올림 → **ORG 는 이 데이터셋 gold 천장에 도달**. 잔여
오류는 문맥의존(`海軍` 특정조직 vs 일반명사) + long-tail. 게이트(0.95)는
ORG 단독으로 불가, PROD·EVT 별도 과제. 상세·수치 영구 출처:
`docs/reports/japanese-bert-classifier-history-2.md` §ORG 정밀도 회복.

산출물: 재사용 도구 `error_analysis.py --from-predictions` + 89건 gold
교정(데이터 반영) + 진단/census 산출물(`results/` gitignore).

## 관련 커밋

- (작성 예정): feat(classifier) — error_analysis pooled-예측 진단 경로 + 테스트
- (작성 예정): docs(reports) — 히스토리 2편 ORG 조사 섹션 + AGENTS.md + 이슈 md
