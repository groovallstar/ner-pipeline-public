# issue-128: 한국어 PROD 오류 진단 + silver junk 정리·재학습

- Issue: https://github.com/groovallstar/ner_pipeline/issues/128
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-128-ko-prod-diagnosis`
- 승인일: 2026-06-18

## 목적

koelectra-base-v3 의 PROD F1 0.690(NER-5 2nd 최저) 정체를 FP+FN 전수 진단으로
4버킷 분해하고, 지배 원인인 **silver gold junk(비-제품 과태깅)** 를 정리해
재학습으로 개선한다.

## 범위

- 포함: `data/klue` PROD gold 오류 진단·재검증·정리, 재학습·재측정, 리포트 PROD 절.
- 제외: 모델 코드·하이퍼(#122 동일), PER/LOC/DAT(사람 gold) 라벨, gold 누락
  recall 보강(후속).

## 성공 기준

- [x] baseline PROD F1 0.690(P0.659/R0.723) 재현 + FP/FN distinct 189 4버킷 진단.
- [x] gemma 전체 3253 PROD 재검증(canonical rubric) → drop 123(junk)/retype 31,
  **gold 95% 재확인**. 표본 16건 수기 94% 정확.
- [x] cleaned gold 재학습 → **PROD F1 0.724**(test +0.006 / 학습 +0.028 분해),
  distinct 189→149, HALLUCINATION 54→31. joint gate NER-5 micro/macro 무회귀.
- [x] 에러 진단·교정을 리포트·이슈에 기록.
- [x] `pytest tests/ner` green · ruff clean · refuter PASS.

## 현 상태 (fact)

- 오류 4버킷(distinct 189): HALLUCINATION 54(실제품 gold 누락 ~26 + garbage ~28)
  / PROD↔PER eponymy 38 / MISS 37 / BOUNDARY 27 / PROD↔ORG·LOC·EVT·DAT 33.
- gemma 재검증: kept 3099 / dropped 123(junk 3.8%) / retyped 31. PROD train
  2580→2458, test 332→317.
- 교정 결과(strict, cleaned test): baseline 0.696 → clean **0.724**(학습 +0.028,
  PROD 단일런 노이즈 ~±0.03 수준이라 modest). 채널: HALLUCINATION 54→31,
  PROD↔ORG 18→10. joint gate: clean NER-5 micro 0.860 / macro 0.790 / PER 0.925.
- 결론: silver junk 가 PROD 과발화를 학습시킨 게 천장의 한 축 — 정리 시
  precision 0.66→0.72. 잔여: PROD↔PER eponymy(구조적)·실작품 recall·경계.

## 결정 로그 (append-only)

- 2026-06-18: 진단을 support/learning-curve 로 착수했으나 사용자 지시로 **오류
  4버킷 처리(교정)** 중심 재초점 — support 실험(learning curve·train-seed) 폐기.
- 2026-06-19: 측정 무결성 위해 test-span 만 고치지 않고 **전체 데이터셋 LLM
  재검증** 후 재학습. test-change(+0.006)와 학습효과(+0.028)를 baseline-on-clean
  -test 로 분리. junk drop 은 PROD 를 줄이나 과발화 감소로 precision 순증.

## 변경 요약

- 신규 `src/ner/scripts/entity_revalidate.py` — silver 라벨 LLM 재검증
  (keep/drop/retype, canonical rubric, resumable 캐시 + retry).
- `data/klue/pii_all.prodclean.jsonl` — PROD junk 정리 gold(로컬, gitignore).
- `docs/reports/korean-bert-classifier-per-entity-diagnosis.md` PROD 절 —
  4버킷 진단 · gemma 재검증 · 정리 후 재학습 결과.

## 검증

- 테스트: `pytest tests/ner` → 352 passed. ruff: 변경 .py clean.
- 재검증: 전체 3253 PROD(failed 0), 표본 16건 수기 94% 정확.
- before/after: `koelectra-base-v3`(0.690) vs `koelectra_prod_clean`(0.724),
  control `base_on_cleantest`(0.696). error_analysis distinct 189→149.
- refuter: 측정 무결성(리포트 ↔ metrics.json·error_analysis) 독립 재계산 PASS.

## 관련 커밋

- `<hash>`: feat(scripts) entity_revalidate + docs PROD 오류 진단·정리
