# issue-92: JA classifier abstention 운영점 — per-class 신뢰도 임계값 통합

- Issue: https://github.com/groovallstar/ner_pipeline/issues/92
- PR: https://github.com/groovallstar/ner_pipeline/pull/93
- 브랜치: `feat/issue-92-ja-classifier-abstention-thresholds`
- 승인일: 2026-06-10

## 목적

검증 세트에서 fit한 per-class 신뢰도 임계값을 저장·추론 적용해 overall
P·R·F1 ≥ 0.93 운영점을 재현 가능하게 통합. opt-in·default OFF (BC 보존).

## 범위

- 포함: scored span decode / threshold fit·apply·save·load / CLI 배선 / 테스트·docs
- 제외: 모델 천장(0.9265) 자체 향상(레버 B/C), VI 적용, 재학습 자동화

## 성공 기준

- [ ] 테스트: scored decode·apply·fit 단위 테스트 (`tests/ner/classifier/`)
- [ ] 문서 갱신: `classifier/AGENTS.md` + `docs/manual` + 본 doc 결과 섹션
- [ ] 메트릭: 단일 모델 fit(valid)→apply(test) E2E가 probe 범위
  (P≈0.94 / R≈0.93 / F1≈0.936) 재현. 플래그 미지정 시 baseline 불변

## 위험·의존성

- 임계값은 모델 종속 — 재학습 시 재-fit 필수(고원 좁음). 하드코딩 금지.
- micro R≥0.93은 포화 PII가 떠받침 — NER-4 per-class recall은 희생됨(문서화).
- conf 포착이 평가 경로 추가 연산 — fp16 NaN 회피 위해 평가는 float32 유지.

## 현 상태 (fact)

- held-out 검증(throwaway probe, `results/.../reach_threshold_sweep/`):
  baseline F1=0.9265±0.0014, val-thr held-out P=0.9397/R=0.9316/F1=0.9358,
  5/5 seed ≥0.93. 신뢰도 정의 = conf_mean.
- 통합 지점: `data_utils.decode_bio_to_spans`(span score 없음),
  `train_eval.evaluate_model`(argmax만, softmax 버림),
  `__main__`(test_predictions.json에 score 미저장).

## 결정 로그 (append-only)

- 2026-06-10: 레버 A(abstention) 단독으로 held-out 0.93 확인 → 모델 개선
  아닌 운영점으로 통합. 임계값 하드코딩 대신 valid-fit·저장 경로 채택
  (모델 종속·재현성).

## 미해결 질문 (open question)

- thresholds.json 스키마에 conf_key·target·fit-set 메타 포함 범위?
- error_analysis / kfold_pool 가 score 키를 무시하고 BC 유지되는지 확인 필요.

## 구현 단계 (가변)

- [x] 1. `decode_bio_to_spans(..., confs=None)` + `evaluate_model(capture_scores=)`
- [x] 2. `apply_thresholds` (순수 함수, span 필터)
- [x] 3. `fit_thresholds` greedy + save/load (`abstention.py`)
- [x] 4. CLI `--fit-abstain` / `--abstain-thresholds`, predictions score, metrics 블록
- [x] 5. E2E 재현 검증
- [x] 6. docs (AGENTS.md)

> 설계 조정: 임계값 적용을 `evaluate_model` 내부가 아니라 순수 함수
> `apply_thresholds` 로 분리(테스트성·추론 서빙 재사용). `evaluate_model` 은
> `capture_scores` 만 추가, 적용·fit·리포트는 `__main__`/`abstention` 이 조립.

---

## 변경 요약

- `data_utils.decode_bio_to_spans(confs=)` — 토큰 신뢰도 입력 시 span 에
  conf_mean `score` 부착(미입력 시 BC). `train_eval.evaluate_model(capture_scores=)`
  — softmax 신뢰도 포착(non-CRF).
- `abstention.py`(신규) — `fit_thresholds`(valid greedy, P·R≥target) /
  `apply_thresholds` / `save·load_thresholds`. NER 4종만 대상.
- `__main__` — `--fit-abstain`(valid fit→`thresholds.json` 저장→test 적용
  리포트) / `--abstain-thresholds PATH`. opt-in·default OFF, `overall` 키는
  baseline 보존, 운영점은 `metrics.json` `abstention` 블록.

## 검증

- 테스트: `pytest tests/ner/classifier/` 59 pass (신규 `test_abstention.py` 6:
  scored decode·apply·fit·save/load + BC). 변경 파일 `ruff check` 클린.
- 메트릭(동결 gold `pii_all_phonediv.jsonl`, 10-fold pooled, production 경로):
  baseline P=0.9121/R=0.9429/F1=0.9273 → 운영점 **P=0.9420/R=0.9302/F1=0.9361
  (ALL≥0.93)**. 실험 held-out(0.9397/0.9316/0.9358) 재현. R 마진 얇음
  (일부 fold 단독 R<0.93, pooled 충족).

## 관련 커밋

- `732342d`: feat(classifier) — abstention fit·apply·CLI 통합 + 테스트·AGENTS

## 후속 작업

- 천장(0.9265) 자체 향상은 별도 — 레버 B(경계 디코딩)·C(model-FP gold 보정).
- `kfold_pool` 에 임계값 적용 pooled 운영점 옵션(현재는 raw pool).
