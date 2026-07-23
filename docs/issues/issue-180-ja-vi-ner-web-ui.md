# issue-180: 일본어·베트남어 NER 추론 데모 웹 페이지

- Issue: https://github.com/groovallstar/ner_pipeline/issues/180
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-180-ja-vi-ner-web-ui`
- 승인일: 2026-07-22

## 목적                          [필수]

기존 ja·vi NER REST API(`src/server`) 위에 **내부 개발·데모용 단일 웹 페이지**를
얹는다. 개발자가 텍스트를 붙여넣고 추론하면 기존 `POST /v1/ner`(동일 출처)을
호출해 canonical 개체를 원문 위 라벨별 색상 하이라이트로 보여준다. 요구사항은
deep-interview(4라운드, 최종 ambiguity 15%)로 확정했다.

## 범위                          [권장]

- 포함: 기존 FastAPI 앱에 `GET /` 라우트 추가로 자족적 HTML+vanilla JS 서빙,
  단일 텍스트, 하이라이트-only, 언어 자동감지 기본 + `auto/ja/vi` 수동 셀렉터
- 제외: 배치 입력 UI · 엔티티 표 · 원본 JSON 뷰 · score 표시 · 인증·CORS·공개노출
  · SPA·빌드 파이프라인 · 신규 무거운 의존성 (**API 계약 불변**)

## 성공 기준(AC)                 [필수]

- [x] 테스트: `GET /` 200·`text/html`·핵심 요소(textarea·언어 셀렉터·`/v1/ner`)
  검증 2건 추가(`tests/server/test_api.py`), 기존 21건 무회귀 → **23 passed**
- [x] 문서 갱신: `docs/manual/rest-api-spec.md`(엔드포인트 목록 + `GET /` 절),
  `src/server/CLAUDE.md`(API·파일 표), 루트 `CLAUDE.md`(디렉토리 트리)
- [x] 기능: 자동감지·수동 지정·unsupported 처리·NFC offset 정합·10종 색상 하이라이트

## 결정 로그 (append-only)       [불변식]

- 2026-07-22: 서빙은 **FastAPI 단일 HTML 서빙** 채택(별도 SPA·정적 호스팅 대신)
  — 동일 출처라 CORS·빌드·신규 의존성이 없고 `python -m server` 하나로 API+UI 제공.
- 2026-07-22: 언어는 **자동감지 기본 + 수동 토글**(자동감지-only 대신) — `detect.py`가
  무부호 vi·romaji ja 를 `unsupported`로 분류하므로 수동 지정 경로가 데모에서 필수.
- 2026-07-22: 기능은 **하이라이트-only**(표·JSON·배치·score 제외) — 내부 데모 목적에
  최소 표면. 특히 score 는 `Span` 모델에 없어 표시하려면 서버 스키마 수정이 필요한데,
  하이라이트-only 결정으로 **서버 계약을 건드리지 않는다**.

## 구현 결과                     [권장]

- `src/server/static/index.html` (신규) — 자족적 데모 페이지(inline CSS/JS).
  - 입력을 `normalize('NFC')`로 맞춰 서버 offset(NFC 기준)과 정합.
  - 하이라이트는 `Array.from(text)` **code-point 슬라이스** — UTF-16 단위가 아니라
    파이썬 str 인덱스와 동일 단위라 astral 문자(이모지·CJK 확장)에서도 경계 정합.
  - canonical 10종 색상 맵 + 범례, unsupported·HTTP 에러·빈 입력 안내.
- `src/server/app.py` — 임포트 시 `static/index.html`을 1회 읽어 `_UI_HTML`에 캐시,
  `GET /`(`HTMLResponse`, `include_in_schema=False`, 인증 없음)로 반환.

## 검증                          [필수]

- **단위/계약**: `uv run pytest tests/server/test_api.py -q` → 23 passed. ruff 통과.
- **JS 문법**: `node --check`(script 추출) rc=0.
- **실서버 스모크**(`/data/ner` 모델 로드, 127.0.0.1:8055):
  - `GET /` → 200·`text/html`·textarea·`/v1/ner` 확인.
  - ja 자동감지 `織田信長は東京都千代田区に…` → `PER:織田信長`, `LOC:東京都千代田区`.
  - vi 자동감지 `Hà Nội là thủ đô của Việt Nam.` → `LOC:Hà Nội`, `LOC:Việt Nam`.
  - 수동 `lang=vi` 무부호 `Ha Noi` → `LOC:Ha Noi`(자동감지 우회 경로 실증).
  - 영어 → `{lang:"unsupported", entities:[]}`(안내 문구 경로).
  - **offset 불변식**: 페이지의 code-point 슬라이스 로직으로 세그먼트를 재조립한 결과가
    원문과 일치(`reconstruct == original: true`), 모든 엔티티 슬라이스가 `.text`와 일치.
