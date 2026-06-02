# issue-69: JA classifier 층화 K-fold 교차 검증 평가 프로토콜 도입

- Issue: https://github.com/groovallstar/ner_pipeline/issues/69
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-69-ja-stratified-kfold-eval`
- 승인일: 2026-06-02

## 목적

단일 80/10/10 분할(test 527문장)의 측정 노이즈를 fold 당 동일 학습 비용으로
절반 이하로 축소 — 층화 K-fold pooled 평가(test 5,270문장)로 strict F1 산출
(PROD 측정 노이즈 ±4.1pp → ±1.3pp 달성).

## 범위

- 포함: data_utils 층화 K-fold 분할 / classifier CLI fold 옵션·test 예측 저장 /
  pooled 평가 CLI / 히스토리 2편 신설 + 요약본 링크 / classifier AGENTS.md
- 제외: extra train 데이터의 train/test 중복 문장 처리(별도 이슈),
  요약본 production 결론 재기준선(2편 측정 결과 확인 후 별도 결정)

## 성공 기준

- [x] 테스트: 분할 무결성(모든 문장이 정확히 한 fold의 test에 등장),
      층화 균등성(PROD/EVT 포함 문장이 fold 간 균등 분포), 기존 테스트 회귀 없음
- [x] 문서 갱신: `docs/reports/japanese-bert-classifier-history-2.md` 신설,
      요약본 링크 추가, `src/ner/classifier/AGENTS.md` 갱신
- [x] 메트릭: base 단독 10-fold pooled strict F1 산출 (0.9195),
      per-entity 유효 support 10배 확인 (PROD 94 → 994)

## 위험·의존성

- GPU 5회 학습 필요 (~30분). 타 세션의 `results/` 정리 작업과 충돌 주의
  (재분할 실험 중 완료된 결과 디렉토리가 삭제된 사례 있음)
- 학습 자체의 CUDA 비결정성은 본 이슈로 제거되지 않음 — pooled 측정은 분할
  표본 노이즈만 줄인다

## 현 상태 (fact)

- **10-fold pooled strict F1 = 0.9195** (P 0.8982 / R 0.9420, 18,349 span) —
  base 단독 청정 기준선 확정. 5-seed 재분할 median 0.9198 과 교차 확인
- 측정 노이즈: overall ±0.63pp → ±0.28pp, PROD ±4.10pp → ±1.28pp,
  EVT ±3.69pp → ±1.21pp (목표 "절반 이하" 달성)
- production(S8) test 문장의 30~35%가 extra train 데이터에 원문 그대로 존재:
  중복 부분 F1 0.9947 vs 비중복 부분 0.9478 (처리는 범위 외, 별도 이슈)
- 수치 상세·해석의 영구 인용 출처:
  `docs/reports/japanese-bert-classifier-history-2.md`

## 결정 로그 (append-only)

- 2026-06-02: 단순 재분할 sweep 반복 대신 K-fold 도입으로 방향 확정 —
  분산의 지배 요인이 분할 편향이 아니라 측정 노이즈로 판명
- 2026-06-02: 히스토리 문서는 기존(Phase 0~8) 동결, 신규 실험은 2편으로 분리 —
  측정 프로토콜이 달라 기존 수치와 직접 비교 불가
- 2026-06-02: 2편 파일명은 `japanese-bert-classifier-history-2.md` (숫자 접미사)
  로 확정. 1편 하단 동결 섹션 + 상단 운영 규칙 수정 완료 (커밋은 5단계에서)
- 2026-06-02: 5-fold → **10-fold** 로 변경 — 5-fold 는 train 이 60% 로 줄어
  기존 80/10/10 프로토콜과 비교 불가. 10-fold 는 train 4,216 / valid 527 /
  test 527 로 기존과 정확히 일치 (학습 10회, ~50분)
- 2026-06-02: 데이터셋 id 비고유 발견 (5,270행 중 고유 id 5,166개) —
  pooled 무결성 검사를 id 기준 → text 기준으로 변경 (text 는 전부 고유)

## 미해결 질문 (open question)

- K-fold 결과 확인 후: 요약본 production 결론(0.9644)을 청정 측정 기준으로
  재기준선할 것인가
- extra 데이터 중복 처리 방식: fold별 재생성 vs 후보를 train 행으로 제한

## 구현 단계 (완료)

- [x] 1. data_utils: 층화 K-fold 분할 함수 + 단위 테스트
- [x] 2. classifier CLI: `--kfold N --fold-index K` 옵션 + test 예측 span JSON 저장 + 단위 테스트
- [x] 3. pooled 평가 CLI: fold별 예측 통합 → strict/relaxed F1 + per-entity 표 + 단위 테스트
- [x] 4. 실측: base 단독 10-fold 학습·평가 (GPU) → pooled 결과 산출
- [x] 5. 문서: 히스토리 2편 신설(재분할 검증 + K-fold 실측 기록), 1편 동결 섹션,
      요약본 링크, AGENTS.md 갱신

---

## 변경 요약

`split_kfold_stratified`(PROD/EVT 층화, fold 라운드로빈 배정)를 data_utils 에
추가하고, classifier CLI 에 `--kfold/--fold-index` 모드와 fold 별
`test_predictions.json` 저장을 연결했다. 신규 `kfold_pool` CLI 가 fold 예측을
합쳐 pooled micro-average F1 을 산출한다 (fold 간 중복 text 자동 검증).
첫 실측으로 base 단독 10-fold 를 학습해 청정 기준선 0.9195 를 확정하고,
히스토리 문서를 1편(동결)·2편(K-fold 프로토콜)으로 분리했다.

## 검증

- 테스트: `.venv/bin/python -m pytest tests/ner/classifier/ tests/ner/augmenters/ja/ -q`
  → **67 passed** (kfold 신규 10개 포함, 기존 회귀 없음)
- CLI: `python -m ner.classifier --help` / `python -m ner.classifier.kfold_pool --help` → 정상
- 린트: 신규·수정 코드 ruff 클린 (잔여 8건은 기존 코드의 E741/F401 — 범위 외)
- 코드 리뷰: 별도 reviewer 패스 — K-fold 보장(전 행 1회 test 등장, seed 고정 시
  배정 불변) 수학적·실증적 확인, MAJOR 1건(n_folds<3 가드)·MINOR 2건 수정 반영
- 실데이터 무결성: 5,270행 전부 정확히 1회 test 등장 (행 인덱스 기준), 분할
  크기 train 4,214~4,220 / valid 524~528 / test 524~528 (80/10/10 일치)
- 메트릭: pooled strict F1 0.9195 — 상세는 히스토리 2편 §실험

## 관련 커밋

- `0bb0292`: feat(classifier) — 층화 K-fold 분할·CLI·pooled 평가 + 테스트
- (본 문서 포함 커밋): docs(reports) — 히스토리 1편 동결·2편 신설·요약본 갱신

## 후속 작업

- extra train 데이터의 train/test 중복 수정 (후보 선정을 train 행으로 제한
  또는 fold 별 재생성) — 별도 이슈
- 수정된 extra 로 production 레시피를 10-fold 재측정 → 요약본 production
  결론(0.9644) 재기준선 여부 결정 — 별도 이슈
