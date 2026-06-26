# issue-149: 지원언어(ja·vi) 양성 감지 + 미지원 명시 + 응답 언어 표기

- Issue: https://github.com/groovallstar/ner_pipeline/issues/149
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-149-supported-lang-detect`
- 승인일: 2026-06-26

## 목적

REST API 입력은 어떤 언어든 올 수 있으나 지원은 ja·vi 뿐이다. 기존
`detect_lang` 은 "가나→ja, 그 외 전부 vi"라 미지원 언어를 vi 로 조용히
오처리하고 응답 `lang` 도 거짓이 됐다. ja·vi 를 **양성 감지**하고 둘 다
아니면 **unsupported 로 명시**(vi 폴백 제거), 응답에 언어를 표기하며, 신규
언어 추가가 국소 변경이 되도록 확장 가능하게 한다.

## 범위

- 포함: 감지 재설계(`detect.py`), 미지원·응답 언어(`app.py`), gold 다국어
  평가셋 + 감지기 벤치마크, 확장 레지스트리, 테스트·문서.
- 제외: 한 텍스트 내 ja·vi 세그먼트 라우팅, 3번째 언어 *모델* 추가(감지만
  확장), 단일 다국어 모델 통합(`area:classifier`).

## 성공 기준 (검증됨)

- [x] gold 다국어 평가셋(14버킷) — 언어별 건수·출처·비율 고정·문서화
      (FLORES-200 dev + 파생 4종, seed 42, 1,398문장).
- [x] 평가 = 혼동행렬(detect 출력 × gold expected) 클래스별 P/R. coverage율
      금지(규칙 재진술 방지).
- [x] 후보 2종 벤치(같은 셋): 손규칙 / fastText-LID, 둘 다 가나 override.
- [x] falsifiable 목표: ja-kana recall 100%, pan-Latin vi false-accept **0**,
      vi-부호 recall 100%, ko/en/zh/무부호-vi/romaji-ja → unsupported 100%.
      kanji-only ja recall 0% 보고(한계).
- [x] 미지원 = `200 + {lang:"unsupported"}`(en 단정 금지), 배치 per-item +
      순서 1:1 계약 테스트.
- [x] 무부호 vi(không dấu) = 수용된 한계로 문서화.
- [x] 확장성: `DETECTORS` 레지스트리 한 줄 등록(테스트로 입증).
- [x] ruff + 테스트 green(65 passed), `server/CLAUDE.md` 갱신.

## 결정 로그 (append-only)

- 2026-06-26: 미지원 응답 = `200 + {lang:"unsupported", entities:[]}`(에러
  아님). 배치 항목별. en 단정 금지 — ja·vi 신호 *부재* 를 감지.
- 2026-06-26: 무부호 vi(không dấu) = 수용된 한계 → unsupported(vi 복구 안 함).
- 2026-06-26: vi 감지는 vi-변별 코드포인트(horn·hook·dot+đ)만 — 범-라틴
  false-accept 방지(정의-시점 반박자 C5).
- 2026-06-26: 감지 방식은 사전 고정 않고 gold 벤치로 결정 — 손규칙 vs
  fastText-LID(가나는 양쪽 override 유지).
- 2026-06-26: coverage 율 폐기 → gold 혼동행렬로 전환(coverage 는 규칙
  재진술·gameable 이라는 정의-시점 반박자 지적 해소).
- 2026-06-26: **손규칙 채택**. 두 후보 모두 하드 목표 충족하나 손규칙이 모든
  측정 지표에서 ≥ fastText 이고 không-dấu(100% unsupported)에서 결정에
  정합(fastText 7% vi 복구는 결정 위반·비일관). 무의존·결정적 우선, LID 의
  라틴-스크립트 언어 확장이점은 현 범위 밖이라 미발현. 상세: 벤치 리포트.

## 구현 결과

- `detect.py`: `detect_lang(text)→ja|vi|unsupported`. `DETECTORS` (신호,
  언어) 순서 레지스트리(가나→ja, vi-mark→vi). `has_vi_mark` 는 NFD 후
  horn(U+031B)·hook(U+0309)·dot(U+0323)·đ 검출 — 범-라틴 부호 불검출.
  `default='vi'` 폴백 제거.
- `app.py`: `_infer_one` 으로 단일·배치 통일 — 자동감지 unsupported 면 모델
  미호출·빈 결과. 명시 미지원 lang 은 400 유지(클라이언트 계약).
- `server.scripts.lang_detect`: gold 빌더·후보·혼동행렬·벤치 하네스. 손규칙
  후보 = 프로덕션 `detect_lang`(벤치가 사본 아닌 출하 코드를 시험). gold·
  매니페스트 패키지 동봉(커밋), 원시 결과는 `results/`(gitignore).

## 검증

- 테스트: `uv run pytest tests/server/` → **65 passed**(+22). 신규: detect
  양성·미지원·distractor(pan-Latin/romaji/không-dấu/ascii)·NFD 분해입력·
  배치 per-item 부분성공·확장성(한글 detector 한 줄 등록). ruff green.
- 메트릭(요약): 손규칙 ja-kana 100% / vi-부호 100% / pan-Latin vi
  false-accept 0 / ko·en·zh·romaji·không-dấu → unsupported 100% /
  kanji-only-ja 0%(보고 전용 한계). 상세 혼동행렬·gold 설계·한계 3종:
  `docs/reports/language-detection-benchmark.md`, 원시
  `results/lang_detect_bench.json`.
