# issue-103: 베트남어 BERT classifier 캐노니컬 벤치마크 재수행

- Issue: https://github.com/groovallstar/ner_pipeline/issues/103
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-103-vi-classifier-canonical-benchmark`
- 베이스: 6363017 (origin/develop tip)
- 승인일: 2026-06-12

## 목적

VI BERT classifier 전 모델을 **단일 split·기본 설정**으로 재학습·평가해 비교
가능한 캐노니컬 표 하나로 정리한다. 신규 SOTA 인코더(CafeBERT, PhoBERT)를
편입하고, 과거 0.0 붕괴의 원인을 규명한다.

## 범위

- 포함: 캐노니컬 모델 재학습+metrics 재생성 / fast-tok offset 정렬 수정 /
  PhoBERT 단어분절 통합 / 비교표 리포트 갱신.
- 제외: LLM NER·PII 벤치 / 하이퍼파라미터 sweep("모델만") / F2·F5·F6 트랙 /
  OOD test셋(사용자 결정) / DeBERTa-V3(표준 레시피 학습 불가, 아래).

## 성공 기준

- [x] 테스트: offset-trim·PhoBERT round-trip + `pytest tests/ner/classifier/` green, ruff clean
- [x] 문서 갱신: `docs/reports/vietnamese-bert-classifier-benchmark.md` 캐노니컬 5-fold 표
- [x] 메트릭/검증: 5모델 5-fold pooled metrics + 누출 제거 검증(cross-split 0)
- [ ] refuter 게이트 PASS → PR(`closes #103`)

## 현 상태 (fact)

- 데이터: `pii_all.jsonl` **원문 기준 중복 제거 38,371**(PII 제외 동일 원문을 하나로,
  누출 제거). 원본 split 파일
  (train/valid/test, pii_*) 제거(백업 `/tmp/wikiann_vi_backup_preDedup`).
- 평가: **stratified 5-fold pooled** micro-avg(당초 80/10/10에서 변경 — 상위
  모델이 단일 split 변동에 묻혀 순위 불가했기 때문).
- 최종 모델셋(5): phobert-base-v2(0.9461) ≈ xlm-r-base(0.9459) > mmbert(0.9395)
  > cafebert(0.9367) > xlm-r-large(0.8364†, fold0 붕괴 포함). 상세는 리포트.
- DeBERTa-V3(mdeberta·videberta) 제외: 발산(warmup/LR로 해소 가능) + 비학습
  (all-O, warmup·LR 무관, bf16 정밀도 미분리) 두 문제.
- 인터뷰 스펙: `.omc/specs/deep-interview-vi-classifier-canonical-benchmark.md`.

## 결정 로그 (append-only)

- 2026-06-12: 범위 = classifier 벤치만. 재현 tolerance 게이트 폐기. "설정 말고
  모델만"(통합 작업은 허용).
- 2026-06-15: **누출 발견** — raw WikiANN test∩train 1,778·train 20% 자기중복을
  PII 주입이 가림. test 4.17%(글자 일치)/6.33%(원문 일치) 누출 → 원문 기준
  중복 제거로 40K→38,371, train·test 간 동일 문장 0 검증.
- 2026-06-15: split 80/10/10 → **stratified 5-fold**(상위 모델 단일-split 변동에
  묻혀 순위 불가).
- 2026-06-15: offset-trim 수정이 **mmBERT 구제**(0.53→0.94)·전 fast-tok 정렬
  개선. PhoBERT는 **pyvi 단어분절** 통합으로 편입(최상위).
- 2026-06-15: DeBERTa-V3 — warmup 추가해도 비학습(all-O) 지속 → warmup 되돌리고
  **제외**(사전 합의 폴백). 정밀도 추가조사는 사용자 결정으로 안 함.
- 2026-06-15: xlm-r-large fold0 all-O 붕괴 → **재실행 없이 투명 보고**(seed 고정
  재현성·cherry-pick 회피).

## 변경 요약

- `classifier/data_utils.py`: `_trim_offset`(선행 공백+후행 `.`/`,`) 추가해
  `_encode_vi` fast-tok offset 정렬 교정; `_encode_phobert` + `_is_phobert`
  (pyvi 단어분절·단어 char-span 정렬) 추가; `--legacy-no-offset-trim` 진단 경로.
- `classifier/__main__.py`: `--legacy-no-offset-trim` 플래그, metrics에 `offset_trim`.
- `tests/ner/classifier/test_encode.py`: DeBERTa 다단어·숫자형 PII trailing-punct,
  PhoBERT 단어분절 round-trip 회귀 가드 3종(+기존 E741 정리).
- `pyproject.toml`: `pyvi` 의존성.
- `docs/reports/vietnamese-bert-classifier-benchmark.md`: 캐노니컬 5-fold로 전면 갱신.
- 구현맵 동기화: `classifier/AGENTS.md`(3-way 토크나이저 분기), `src/ner/CLAUDE.md`,
  루트 `CLAUDE.md`의 data_utils 설명.

## 검증

- `pytest tests/ner/classifier/` 52 passed, `ruff check` clean.
- 누출 제거: 중복 제거 후 5-fold pooling이 train·test 간 동일 문장 0(ValueError 없음).
- 5모델×5fold = 25런 전부 rc=0. pooled 메트릭은 리포트 표.
- refuter 게이트: <!-- 실행 후 채움 -->

## 관련 커밋
<!-- 커밋 후 채움 -->
