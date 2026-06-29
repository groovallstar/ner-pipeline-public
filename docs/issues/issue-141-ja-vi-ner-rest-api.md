# issue-141: ja·vi NER REST API 서비스 신설

- Issue: https://github.com/groovallstar/ner_pipeline/issues/141
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-141-ja-vi-ner-rest-api`
- 승인일: 2026-06-23

## 목적

학습된 BERT 분류기(`/data/ner/{ja,vi}/model/`)를 감싸 ja/vi NER 추론을
제공하는 FastAPI REST API 서비스(`src/server/`)를 신설한다. 단일·배치
엔드포인트로 char-offset 엔티티 span 을 반환하며, abstention 임계값은
graceful 로딩(파일 있으면 적용·없으면 raw). LLM 라벨러 경로 미사용.

설계 근거: `.omc/specs/deep-interview-ja-vi-ner-rest-api.md` (deep-interview,
ambiguity 12%).

## 범위

- 포함: `src/server/`(FastAPI 앱·추론 코어·언어감지·임계값·청킹),
  단일+배치 엔드포인트, 언어 자동감지, 긴 입력 auto-chunk, 경량 운영
  기본값(env API-key·입력 한도·구조화 에러), scale-ready 설계.
- 제외: vi thresholds fit(별도 이슈, `area:classifier`), 배포·서빙
  인프라(Docker·오토스케일), LLM 백엔드, 프론트엔드(추후 루트 `frontend/`).

## 성공 기준 (검증의 AC로 흡수)

§검증 참조.

## 현 상태 (fact)

- 추론 코어는 `classifier/data_utils`(build_label_maps·encode_row·
  decode_bio_to_spans)와 `classifier/confidence_threshold`(load·apply)에
  의존. `train_eval` 등 학습·평가 모듈은 import 하지 않는다.
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
- 2026-06-23: 폴더명 `server/`(앱+생명주기 포함, `python -m server`).
- 2026-06-23: confidence_threshold 리네임(#144) develop 머지 → 서버 자족
  `thresholds.py` 제거하고 `ner.classifier.confidence_threshold` 사용으로
  전환(중복 제거). 임계값은 canonical 변환 전 내부 span 에 적용 — 학습-시점
  eval 경로와 동일해 parity 유지(파리티 테스트로 회귀 확인).
- 2026-06-23: server 를 `src/ner/server/` → `src/server/` 로 승격(별도
  top-level 패키지, pyproject 2-패키지). 서비스 레이어를 `ner` 라이브러리와
  표현상 분리 — import `from server.x`, CLI `python -m server`. 런타임은 여전히
  `ner.classifier` 의존(조직상 분리, 결합도 변화 없음).
- 2026-06-23: 모듈 문서를 `server/AGENTS.md` → `server/CLAUDE.md` 로 통합
  (디렉토리 진입 시 자동 로드 + 사용 예시·검증 보강). 실서버 검증 추가 —
  셸 스모크 스크립트 + `live` 마커 pytest(서브프로세스 uvicorn, httpx).

## 미해결 질문 (open question)

- vi parity: vi `thresholds.json` 생성(별도 이슈) 후에야 vi operating-point
  parity 검증 가능. 현재는 graceful-raw 동작만 검증.

## 구현 결과

`src/server/` 7 파일: `config`(env 설정)·`detect`(가나→ja, 그 외 vi)·
`chunking`(문장분할 offset 보존)·`inference`(LangModel 1회 로드·재사용,
ModelRegistry, 추론 루프 직접 구현, 임계값은 classifier.confidence_threshold
사용)·`app`(FastAPI `/v1/ner`·`/health`, 인증·구조화 에러)·`__main__`
(uvicorn). `pyproject.toml`에 fastapi·uvicorn 추가. 모듈 가이드
`server/CLAUDE.md`.

## 검증

- 테스트: `uv run pytest tests/server/` → **35 passed**(26 계약·단위 stub +
  6 실모델 통합 + 3 실서버 live). 셸 스모크 `src/server/scripts/smoke_test.sh`
  PASS. ruff clean.
- AC 충족:
  - [x] 단일·배치 스키마(canonical span)·순서 1:1 — `test_api`
  - [x] 언어 자동감지(한자혼입 vi→vi, 명시 우회) — `test_detect`·`test_api`
  - [x] 에러·인증(400/503/401)·`/health`·무상태 — `test_api`
  - [x] abstention 토글 graceful — `test_api`(분기)·통합테스트(vi raw)
  - [x] offset 정확성 `text[s:e]==surface` (ja/vi 실모델) — 통합테스트
  - [x] auto-chunk offset 보정(실모델 장문, >1 청크) — 통합테스트
  - [x] **ja parity**: operating F1 0.9493 / baseline 0.9369 가
    `/data/ner/ja/metrics.json` 과 1e-6 일치(리팩터 스크립트 비의존,
    기록 데이터 대조) — `test_ja_parity_*`
- 결과시점 refuter: PASS (`.omc/state/refuter/48829907e8d1.json`). parity
  비-tautology 확인. 후속 cleanup 반영(response_model 배선, 임계값/dedup
  순서 주석).
- chunk recall (후속 검증·종결): 실 ja·vi test gold 엔티티 전수가 청크
  경계를 가로지르지 않음을 확인하고(엔티티는 문장 내부, 청크 경계는
  문장·단어 경계에만 생성), 장문에서 후반 청크 엔티티가 정상 회수됨을
  검증 — `test_*_gold_entities_never_straddle_chunk_boundary`·
  `test_*_long_input_chunk_offsets` 로 회귀 가드. 잔여(이론): over-budget
  단일 문장의 단어 분할은 다단어 엔티티를 자를 수 있으나, 실 test
  데이터엔 단일 row 가 한도를 넘는 경우가 없어 미발생.

## 후속 작업

- vi thresholds fit (별도 이슈, `area:classifier`) → 완료 시 vi parity 추가.
