# issue-90: JA ID_NUM injector 환각 무라벨 relabel + injector 코드 근본수정 (CC #78 동형)

- Issue: https://github.com/groovallstar/ner_pipeline/issues/90
- PR: (작성 예정)
- 브랜치: `feat/issue-90-ja-idnum-injector-relabel`
- 승인일: 2026-06-09

## 목적

#78(CREDIT_CARD)에서 별도 이슈로 남긴 injector 포맷-충돌 하드닝의 ID_NUM
판 + 근본 코드 수정. ID_NUM(F1 0.9646 / **P 0.9491**)이 EMAIL·PHONE·CC
포화권 대비 P 가 낮은 잔여 이상치 — gold 천장인지 CC 식 파이프라인 버그인지.

## 범위

- 포함: ID_NUM pooled 포맷-충돌 감사, 결정론 relabel, 10-fold 재학습 측정,
  **injector 코드 하드닝(#78이 미룬 근본 fix — CC·ID_NUM 공용)**
- 제외: PHONE(무라벨 3, 노이즈), 코퍼스 재생성(로컬 gold 보정만)

## 현 상태 (fact)

- 원인 = #78 CC 와 동형 **LLM injector 환각**(gold 천장 아님). 마이넘버
  포맷(12자리) 표면 965 중 **932 ID_NUM / 33 무라벨(O)** = 일관성 96.6%
  (CC 수정전 88.9% 대비 양호하나 동일 버그). 무라벨 33 프레임: 管理番号
  10 / 番号 2 / ID 1 / 기타 20 — 라벨된 932 도 **管理番号 390·ID番号 315**
  로 같은 프레임 → **포맷+프레임 결정적**(학습가능, CC형). LOC attributive
  (#89 폐기, 문맥의존 천장)와 질적으로 다름.
- 처방 1 = 결정론 relabel(무라벨 마이넘버포맷 33 → ID_NUM, 932→965).
  백업 `.preidnum`, 타 라벨 불변.
- 처방 2(근본) = injector `extract_spans` 에 하드닝 step 추가
  (`harden_pii_format_collisions`): 주입 후 무라벨 카드(13~19자리)·
  마이넘버(12자리) 포맷열을 해당 PII 타입으로 일관 relabel. 카드(긴
  포맷)를 먼저 매칭해 12자리 오인 방지, 기존 span 무겹침. → 코퍼스
  재생성 시 CC·ID_NUM 무라벨 재유입 차단(#78 open question 해소).
- 측정: 보정 gold 10-fold pooled 재학습 — ID_NUM 0.9646→**0.9793**
  (F1 +1.47 / P 0.9491→**0.9773** +2.82pp / R 0.9807→0.9813), FP
  49→**22**(−27). CC 패턴 재현(P↑·FP↓, TP +33=relabel 정확히). overall
  0.9231 은 run-to-run 비결정성 대역 내 — ID_NUM 33/18,790 라벨 변동이라
  overall 인과 기여 ~+0.1pp 로 노이즈에 묻힘(ORG/PROD seed 변동이 지배).

## 결정 로그 (append-only)

- 2026-06-09: CC #78 의 "별도 이슈(무라벨 카드/전화/ID 포맷 reject)" 를 본
  이슈로 수행. **reject 대신 relabel** 채택 — 측정상 relabel 방향이
  +6.1pp(CC)/+2.82pp(ID) 회복을 검증(reject 는 데이터 손실).
- 2026-06-09: ID_NUM 이 LOC attributive(#89, 문맥 천장)와 다름을 프레임
  공존(管理番号 라벨 390·무라벨 10)으로 확인 — 포맷결정적이라 학습 가능.

## 성공 기준

- [x] 진단: 마이넘버포맷 965 중 33 무라벨, 라벨/무라벨 프레임 공존 확인
- [x] 원인: injector 환각(CC #78 동형) 확정 — 포맷결정적, 천장 아님
- [x] 처방·측정: relabel 재학습 ID_NUM P +2.82pp / FP 49→22, CC 패턴 재현
- [x] 근본수정: `harden_pii_format_collisions` + 테스트 7개 (pii 54 pass)
- [x] 문서: 본 doc + per-entity-diagnosis §ID_NUM

## 변경 요약

- 코드: `src/ner/augmenters/pii/llm_injector.py` — `harden_pii_format_collisions`
  추가, `extract_spans` step 3 호출. 무라벨 CC/ID_NUM 포맷열 일관 relabel.
- 테스트: `tests/ner/augmenters/pii/test_llm_injector.py` — 하드닝 7개
  (solid/grouped/prefix오인방지/기존span보존/짧은숫자/extract_spans통합).
- gold: 보정은 로컬 한정(`data/` gitignore). injector 하드닝으로 재생성
  시 자동 반영.

## 검증

- 측정: 보정 gold 10-fold pooled(`kfold10_phonediv_idnumfix`).
  ID_NUM 0.9646→0.9793 / P 0.9491→0.9773 / FP 49→22. 상세·전체 P/R/F1:
  per-entity-diagnosis §ID_NUM(#90).
- 테스트: `pytest tests/ner/augmenters/pii/` 54 passed, ruff 통과.
- 산출물 `results/`·보정 gold gitignore(로컬). 영구 재현 기준은 #69 base
  0.9195(ID_NUM 932 기준).

## 관련 커밋

- (작성 예정)
