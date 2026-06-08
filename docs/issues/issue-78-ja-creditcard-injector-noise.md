# issue-78: JA classifier CREDIT_CARD 정밀도 천장 진단 — LLM injector 환각 카드숫자 규명 + gold 보정·재학습

- Issue: https://github.com/groovallstar/ner_pipeline/issues/78
- PR: https://github.com/groovallstar/ner_pipeline/pull/79
- 브랜치: `feat/issue-78-ja-creditcard-injector-noise`
- 승인일: 2026-06-05

## 목적

합성 PII 인데도 정밀도가 낮은 CREDIT_CARD(F1 0.9244 / **P 0.8922** /
R 0.959) 의 P 저하 원인 규명 — gold 천장(ORG/PROD)인지 고칠 수 있는
파이프라인 버그인지. 같은 합성 PII EMAIL(0.9985)·PHONE(0.9933) 은
포화권인데 CREDIT_CARD 만 P 가 7pp 낮은 이상치.

## 범위

- 포함: pooled FP/FN 진단(재추론 0), 원인 규명, gold 보정, 재학습 측정
- 제외: LLM injector 코드 하드닝(근본 fix·별도 이슈), 타 PII 점검

## 현 상태 (fact)

- 원인 = **LLM injector 환각**(gold 천장 아님). 본문 16자리 카드포맷
  961 중 854 만 CC 라벨 / 107 무라벨(O). 107 = LLM injector 가 자연
  삽입 중 주입값 외로 창작(80)·재사용(27)한 카드숫자로, `pii_values`
  에 없어 string-match 라벨이 안 붙음(`extract_spans`). 같은 형식인데
  854 CC / 107 O → 학습 불가능한 표면 충돌 → P 가 854/961≈0.888 로
  기계적 천장. canonical schema(`CREDIT_CARD=13~19자리 카드번호`) 기준
  107 은 schema 위반 무라벨.
- 처방 = 결정론적 relabel(카드포맷 16자리 무라벨 → CREDIT_CARD, 107건,
  854→961). 백업 `.bak`, offset 결함 0, 신규 overlap 0.
- 측정: **리스코어(재학습 0) −1.70pp** ↔ **재학습 +6.6pp** 부호 반전 →
  이 fix 는 측정 정정이 아니라 *학습 모순 제거*(보정 gold 재학습 필수).
  보정 후 CREDIT_CARD 0.9295→**0.9907**(+6.1pp, P 0.9866/R 0.9948),
  overall 0.9167→**0.9255**(+0.88pp, 노이즈 ±0.28pp 의 3배).

## 결정 로그 (append-only)

- 2026-06-05: 초기 가설 "깔끔한 생성기 버그→0.99" → 진단 후 "ORG/PROD식
  gold 천장 절반" 으로 하향 → 재학습 0.9907 로 둘 다 빗나감. 경계분할
  잔차는 모델 한계가 아니라 모순 학습 산물이었음(일관 라벨로 소거).
- 2026-06-05: 리스코어 −1.70pp 를 결과로 오인하지 않음 — 옛 모델은
  모순 gold 로 학습돼 새 라벨을 예측 못 함. 재학습이 정답 측정.

## 성공 기준

- [x] 진단: P 0.8922 = HALL FP 50 중 34 풀카드가 무라벨 카드포맷
- [x] 원인: 파이프라인 버그(LLM injector 환각) 확정 — gold 천장 아님
- [x] 처방·측정: 재학습 +6.1pp(CC) / +0.88pp(overall), 노이즈 위
- [x] 문서: 본 doc + per-entity-diagnosis §CREDIT_CARD + 평가지표 표

## 미해결 질문 (open question)

- LLM injector 하드닝(출력 후처리에 무라벨 카드/전화/ID 포맷 digit열
  reject) 미수행 → gold 보정은 로컬·미커밋(gitignore)이라 코퍼스
  재생성 시 107 재유입. 별도 이슈로 분리 예정.

---

## 변경 요약

코드 변경 없음(진단 도구·학습 CLI 재사용). gold 보정은 로컬 한정
(`data/` gitignore). 문서만 커밋: 본 이슈 doc + per-entity-diagnosis §CREDIT_CARD
실험 섹션(진단·원인·처방·전체 P/R/F1 표).

## 검증

- 진단·측정: `error_analysis` 인라인 + 보정 gold 10-fold pooled
  (`kfold10_phonediv_ccfix`). CREDIT_CARD 0.9295→**0.9907**, overall
  0.9167→**0.9255**. 상세 표·전체 P/R/F1: per-entity-diagnosis §CREDIT_CARD(#78).
- 산출물 `results/`·보정 gold 모두 gitignore(로컬). 영구 재현 기준은
  #69 의 base 0.9195(CC 854 기준).

## 관련 커밋

- `docs(issues,reports): CREDIT_CARD LLM injector 환각 규명 + gold 보정
  재학습` (PR #79)
