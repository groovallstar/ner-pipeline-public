# issue-141: ja·vi NER REST API 서비스 신설

- Issue: https://github.com/groovallstar/ner_pipeline/issues/141
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-141-ja-vi-ner-rest-api`
- 승인일: 2026-06-23

## 목적

학습된 BERT 분류기(`/data/ner/{ja,vi}/model/`)를 감싸 ja/vi NER 추론을
제공하는 FastAPI REST API 서비스(`src/ner/server/`)를 신설한다. 단일·배치
엔드포인트로 char-offset 엔티티 span 을 반환하며, abstention 임계값은
graceful 로딩(파일 있으면 적용·없으면 raw). LLM 라벨러 경로 미사용.

설계 근거: `.omc/specs/deep-interview-ja-vi-ner-rest-api.md` (deep-interview,
ambiguity 12%).

## 범위

- 포함: `src/ner/server/`(FastAPI 앱·추론 코어·언어감지·임계값·청킹),
  단일+배치 엔드포인트, 언어 자동감지, 긴 입력 auto-chunk, 경량 운영
  기본값(env API-key·입력 한도·구조화 에러), scale-ready 설계.
- 제외: vi thresholds fit(별도 이슈, `area:classifier`), 배포·서빙
  인프라(Docker·오토스케일), LLM 백엔드, 프론트엔드(추후 루트 `frontend/`).

## 성공 기준 (검증의 AC로 흡수)

§검증 참조.

## 현 상태 (fact)

- 추론 코어는 안정적 `classifier/data_utils`(build_label_maps·encode_row·
  decode_bio_to_spans)만 의존. `train_eval`·`abstention/confidence_threshold`
  등 동시 리팩터 중인 모듈은 import 하지 않는다(서버 자족 임계값 구현).
- 모델: ja=tohoku bert-japanese(MeCab/fugashi), vi=PhoBERT
  `vinai/phobert-base-v2`(pyvi). 두 모델 startup 로드.
- 임계값: ja `thresholds.json` 존재(ORG/LOC/EVT/PROD) → 적용. vi 부재 →
  raw. 파일 도착 시 코드 변경 없이 자동 적용.

## 결정 로그 (append-only)

- 2026-06-23: 백엔드 = BERT 분류기(LLM 라벨러 아님). 모델 `/data/ner`.
- 2026-06-23: 통합 단일 엔드포인트 + 언어 자동감지(텍스트별). router·
  마이크로서비스는 배포 토폴로지라 보류 — in-process 모델 선택으로 충분.
- 2026-06-23: span 필드 = 프로젝트 canonical `{label,start_char,end_char,
  text,score}` (.jsonl 관례와 일치, 파이프라인 되먹임 가능).
- 2026-06-23: vi thresholds fit 은 별도 세션·이슈로 분리 → 서버는 graceful
  로딩으로 디커플. 동시 진행 `abstention→confidence_threshold` 리네임과
  코드로 안 엮이게 자족 임계값 모듈을 서버에 둔다.
- 2026-06-23: 폴더명 `server/`(앱+생명주기 포함, `python -m ner.server`).

## 미해결 질문 (open question)

- vi parity: vi `thresholds.json` 생성(별도 이슈) 후에야 vi operating-point
  parity 검증 가능. 현재는 graceful-raw 동작만 검증.

## 구현 결과

`src/ner/server/` 8 파일: `config`(env 설정)·`detect`(가나→ja, 그 외 vi)·
`thresholds`(자족 graceful 로딩·적용)·`chunking`(문장분할 offset 보존)·
`inference`(LangModel 1회 로드·재사용, ModelRegistry, 추론 루프 직접 구현)·
`app`(FastAPI `/v1/ner`·`/health`, 인증·구조화 에러)·`__main__`(uvicorn).
`pyproject.toml`에 fastapi·uvicorn 추가. 모듈 가이드 `server/AGENTS.md`.

## 검증

- 테스트: `uv run pytest tests/ner/server/` → **38 passed**(32 계약·단위
  stub + 6 실모델 통합). 전체 스위트 회귀 없음(444 passed/0 skipped). ruff
  clean.
- AC 충족:
  - [x] 단일·배치 스키마(canonical span)·순서 1:1 — `test_api`
  - [x] 언어 자동감지(한자혼입 vi→vi, 명시 우회) — `test_detect`·`test_api`
  - [x] 에러·인증(400/503/401)·`/health`·무상태 — `test_api`
  - [x] abstention 토글 graceful — `test_thresholds`·`test_api`
  - [x] offset 정확성 `text[s:e]==surface` (ja/vi 실모델) — 통합테스트
  - [x] auto-chunk offset 보정(실모델 장문, >1 청크) — 통합테스트
  - [x] **ja parity**: operating F1 0.9493 / baseline 0.9369 가
    `/data/ner/ja/metrics.json` 과 1e-6 일치(리팩터 스크립트 비의존,
    기록 데이터 대조) — `test_ja_parity_*`
- 결과시점 refuter: PASS (`.omc/state/refuter/48829907e8d1.json`). parity
  비-tautology 확인. 후속 cleanup 반영(response_model 배선, 임계값/dedup
  순서 주석).
- 알려진 한계: chunk 경로의 *엔티티 recall* 은 ground-truth 대비 미검증
  (offset 정합성만 검증). 장문 entity 가 청크 경계에서 누락될 경우 recall
  하락으로 드러나며 silent 오류는 아님. 필요 시 후속 보강.

## 후속 작업

- vi thresholds fit (별도 이슈, `area:classifier`) → 완료 시 vi parity 추가.
- `src/ner/CLAUDE.md` server/ 행 추가 — 동시 리팩터 착지 후.
