# issue-189: JA·VI 데모 UI에 한국어 번역 글로스

> GitHub: https://github.com/groovallstar/ner_pipeline/issues/189
> 브랜치: `feat/issue-189-web-ja-vi-ko-gloss` (origin/develop 베이스)
> 딥인터뷰 스펙 원본: 2026-09-10 폐기

## 목적 (Goal)

웹 데모 UI(`GET /` → `src/server/static/index.html`)에서 JA/VI 입력 문장의 **한국어 글로스(뜻풀이)** 를 NER 하이라이트와 함께 표시한다. 기계 소비자용 `/v1/ner` JSON API와 추론 코드는 **무접촉(diff 0)**, 순수 additive.

**성공 기준**
- Phase 1: 손수 제작 평가셋에서 루브릭(①sentinel 생존 ②음차 식별 ③뜻 전달) VI ≥80% 통과 → go
- Phase 2: PII 원문 보존 테스트 통과 + `/v1/ner`·`inference.py` diff 0 + 데모 페이지 글로스 표시 + graceful degradation + 기존 `tests/server` 회귀 통과

## 현 상태 (fact)

- 서버: FastAPI, **top-level `server` 패키지**(`from server.xxx`, `python -m server`). 엔드포인트 `/v1/ner`·`/health`·`/`(데모 UI). 지원 언어 **ja/vi만**.
- 데모 UI: `src/server/static/index.html`, 클라이언트에서 `fetch("/v1/ner")`만 호출.
- 번역 코드·의존성 **없음**. LLM 클라이언트는 오프라인 labeler 전용(`ner.labelers.base_vllm_labeler`=`localhost:8081/v1`, `base_openai_labeler`).
- 응답 `Span{label,start_char,end_char,text}`. 캐노니컬 10종(PII 5: DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD).
- refuter 게이트 ruler-lock 대상 아님(gold/metric/split 무접촉).

## 제약

- **로컬/온프렘 전용** — 외부 클라우드 번역 API 불가(입력 원문 PII → 제3자 전송 금지)
- **PII 원문 그대로 보존** — 마스킹-복원(PII는 번역기 미투입, 구조적 보장)
- **PII 탐지 = `/v1/ner` span 재사용** — 데모 페이지가 span 전달, PII 보존 = NER recall 종속 감수
- **고유명사(PER/LOC/ORG/PROD/EVT) 한글 음차** — 마스킹 아님, 번역기 통과
- 한국어 글로스는 별도 문자열 — 원본 offset 유지, 재계산 없음

## 접근 (Path — 가변)

2단계, Phase 1 go가 Phase 2를 게이트.

**Phase 1 — 벤치마크 (go/no-go)**
1. 손수 제작 평가셋(JA·VI, PII·고유명사 촘촘히, ~20–30문장)
2. 마스킹-복원 파이프라인 시제 + 후보 엔진(NLLB-200-distilled-600M 계열 vs 온프렘 vLLM) JA→KO·VI→KO
3. 루브릭 채점 + 문장당 속도 측정
4. 산출: go/no-go + 엔진 선정 + UX 트리거(자동/온디맨드) 결정

**Phase 2 — 구현 (go 이후)**
5. 독립 번역 모듈: `(원문, lang, PII spans) → 마스킹 → 음차 번역 → 복원 → ko`
6. 새 additive 엔드포인트(예: `POST /v1/translate`), 기존 무접촉
7. 데모 페이지 병렬 호출·글로스 표시 + graceful degradation
8. 토글 env(`NER_SERVER_TRANSLATE_*`)·모델 경로·docker 배선 + 테스트

## 결정 로그 (append-only)

- 2026-07-24: 엔진 **로컬 전용** 확정 — 외부 API는 PII 제3자 전송이라 불가.
- 2026-07-24: 합격선 = gist 아니라 **고유명사·PII 보존**까지. → 순진한 번역 불가, **마스킹-복원 강제**.
- 2026-07-24: PII 탐지는 새 정규식 대신 **NER span 재사용**(①안) — PII 보존 = NER recall 종속 감수.
- 2026-07-24: 고유명사는 **한글 음차**(원문 유지 아님).
- 2026-07-24: **벤치 먼저** — 최고 불확실성은 VI→KO 로컬 품질. UX 트리거는 벤치 속도 결과로 결정(deferral).
- 2026-07-24: 평가셋은 손수 제작 대신 **웹 공개셋 소싱 + 긴 문장 포함**(보유 데이터가 짧음). **NTREX-128**(뉴스, CC BY-SA 4.0) JA/VI 원문 + KO 레퍼런스에서 길이 층화 28문장, 절반에 PII 5종 suffix 주입(`ner.augmenters.pii` 재사용, gold span 보장). KO 레퍼런스에 gold 음차 포함 → 음차 채점 근거.
- 2026-07-24: (한계) PII suffix 주입이라 sentinel 위치가 문장 끝 편향. 문중 PII 스트레스는 Phase 1 결과 보고 보강 판단.

- 2026-07-24: **Phase 1 판정 = GO, 엔진 = 온프렘 vLLM(Gemma-4-31B).** NLLB 탈락(sentinel 5종 재시험 최선 33% — NMT 지시 불가 + 후미 절 드롭으로 하드 PII 보존 불가). UX 트리거 = **온디맨드 버튼**(Gemma 2.7~7s는 자동엔 느림).
- 2026-07-24: (설계 정련) LLM도 sentinel 보존은 수학적 보장 아님 → 복원 단계에서 **sentinel 생존 검증 + graceful degrade**. 보장 정의 = "PII 보존-또는-부재, 훼손·유출 없음".
- 2026-07-24: (refuter round1 FAIL→수정) sentinel `【N】`이 일본어 자연 표기(`【重要】`·`【9】`)와 충돌 — 삭제·PII 오배치 위험 → sentinel 접두 분리. 겹치는/범위밖 PII span은 마스킹 전 **거절(400)**. openai 직접 의존 명시, 백엔드 실패 로깅 추가.
- 2026-07-24: (refuter round2 PASS + MEDIUM 후속) 리터럴 `【PII0】` 입력 시 잔여 충돌 가능(트리거 ~0이나 존재) → sentinel에 **프로세스별 랜덤 nonce**(`【PII{nonce}_{i}】`) 추가로 완전 차단. 클라이언트가 형식 예측 불가. 실 vLLM 생존 3/3 유지 확인.

## Phase 1 결과 (벤치마크)

평가셋: NTREX-128 JA/VI 28문장(18~332자), 절반 PII 5종 suffix 주입(gold span). 하네스: 마스킹-복원(`【N】`) + 자동지표. (스크립트·데이터: 세션 scratchpad — Phase 2 착수 시 repo로 승격)

| 엔진 | 지연/문장 | PII 보존 | sentinel 생존 | 음차 | 뜻전달 |
|---|---|---|---|---|---|
| **Gemma-4-31B (vLLM)** | ~2.7s (긴 문장 7s) | **24/24** | **24/24** | 식별 가능 | 정확 |
| NLLB-600M (로컬) | ~0.4s | 0/24 | 최선 형식 33% | 준수 | 준수 |

- **엔진 = Gemma/온프렘 vLLM** — VI→KO(핵심 리스크)까지 음차·뜻전달 우수. 기존 배포 vLLM(`cyankiwi/gemma-4-31B-it-AWQ-8bit` @ `:8081`) 재사용 → 신규 모델 배포 0.
- **NLLB 탈락** — 로컬·고속이나 placeholder 최선 33%, 하드 PII 보존 불가(구조적).
- **UX = 온디맨드 버튼**.
- **캐비엇**: N=28(24 PII span) 소규모 신호; suffix 주입이 NLLB에 불리(Gemma는 견고); Gemma는 기존 vLLM 용량 공유 → Phase 2에서 동시성·타임아웃 고려.

## Phase 2 결과 (구현)

additive 신규 모듈·엔드포인트만 추가, `/v1/ner`·`inference.py`·NER 파이프라인 **diff 0**.

- `src/server/translate.py` — `LLMTranslator`(마스킹-복원 + 음차 프롬프트)·`build_translator`·`_mask`/`_restore`. PII를 `【PII{nonce}_{i}】` sentinel(프로세스 랜덤 nonce)로 가려 번역기 미노출, 복원 시 원문 그대로. sentinel 소실 시 부재 처리(훼손·유출 없음), 잔여는 nonce 형식만 제거(자연 `【N】`·리터럴 유사 텍스트 보존). 겹침·범위밖 span 거절.
- `src/server/app.py` — `POST /v1/translate`(`{text, lang, spans}` → `{lang, translation}`) + `GET /v1/translate/status`(`{enabled, available}`) 추가. 둘 다 **`include_in_schema=False`**(외부 OpenAPI 명세엔 `/v1/ner`만 남음). translator 미주입/lang 미지원/겹침 span/백엔드 실패 = 503/400/400/503. `/v1/ner`·`/`·`/health` 핸들러 무변경.
- `src/server/config.py` — `NER_SERVER_TRANSLATE_{ENABLED,BASE_URL,MODEL,API_KEY,TIMEOUT_S}`. 기본 비활성.
- `src/server/__main__.py` — `build_translator(config)` 배선.
- `src/server/static/index.html` — '한국어 번역 보기' 온디맨드 버튼 + 글로스. **페이지 로드 시 `/v1/translate/status` 1회 조회**(폴링 없음) — vLLM 가용 시에만 추론 후 버튼 활성, 미가용이면 계속 비활성 + 안내. 실패 시 NER은 그대로 두고 글로스만 생략(graceful).
- `docker/server/{.env.example,docker-compose.yml}` — 번역 env 배선(+ 컨테이너 내 localhost 주의).
- `tests/server/test_translate.py` — 마스킹-복원 PII 보존·부재·엔드포인트 계약 10종.

**검증**
- `tests/server/` 97 → 116 통과(신규 test_translate.py 19: 마스킹-복원·자연 `【N】` 보존·PII 오배치 없음·리터럴 sentinel 유사텍스트 보존·겹침/범위 거절·엔드포인트 계약·가용성 status 3종), 기존 회귀 0. ruff clean.
- refuter round1 FAIL(sentinel 충돌·겹침) → 수정 → round2 PASS(+MEDIUM nonce 후속 반영). 실 vLLM 생존 재검증.
- **`/v1/ner` 외부 계약 불변 재확인**: NER 경로 의존 파일(inference/detect/chunking/concurrency/limits/request_log) diff 0, app.py 유일 수정은 `create_app` 시그니처(translator 옵션), 실서버 OpenAPI 노출 경로 = `['/v1/ner']` 뿐, `/v1/ner` 응답 스키마 동일.
- 번역 가용성: 실 vLLM 가동 시 `/v1/translate/status`=available:true, 죽은 백엔드는 1.5s 내 false(버튼 비활성).
- 실 vLLM(Gemma-4-31B @ :8081) 라이브: JA 문중 PII `08060197475`·VI 카드 `4111...` 원문 보존, 음차 정확(조란 자에프·응우옌 반 A·하노이·마케도니아). 문중 PII도 정상(Phase 1 suffix 한계 보완 확인).

## 미해결 질문 (후속)

- vLLM 용량 공유 시 번역 요청 동시성 가드·타임아웃 튜닝(현재 per-call timeout만)
- 평가셋·벤치 하네스 repo 승격 여부(현재 세션 scratchpad)
