# 최신 작업 트리의 Codex 브랜치 반영

## 배경

`/work/git/ner-pipeline`의 `develop` 작업 트리가 최신 제품 코드이며, 현재
`ner-pipe-codex` 브랜치에는 Codex 마이그레이션과 서버 검증 개선이 별도로 있다.
사용자는 원본의 최신 내용을 반영하고 최종적으로 Codex만 사용하도록 요청했다.

## 목적

원본의 커밋 및 미커밋 제품 변경을 반영하면서 Codex 운영 지침과 검증 개선을 보존한다.

## 성공 기준

- 원본 HEAD `6c7a64f`와 미커밋 app/detect/test_detect/rest-api-spec 및 새
  `language-detection.md`를 반영한다. 원본 작업 트리는 수정하지 않는다.
- `.claude/`, `.omc/`, `CLAUDE.md`, 훅 테스트는 가져오지 않는다.
- `AGENTS.md`, Codex 기록, 기존 live 하네스의 준비성·로그·인증 처리를 보존한다.
- 새 지원 언어·번역 설정·제거된 모듈에 맞춰 지침과 매뉴얼을 갱신한다.
- 출처 검사는 Claude 훅 없이 실행하며 결과 JSON의 값은 변경하지 않는다.

## 결정 로그

- 2026-09-09: 사용자가 원본 작업 트리를 최신 제품 기준으로 지정했다. 공통 조상
  `352b445`를 기준으로 3-way 병합하며 현재에만 있는 Codex 변경을 보존한다.
  원본의 폐기 모듈·대응 테스트 삭제는 제품 정리로 반영하며 실패 은폐로 삭제하지 않는다.
- 2026-09-09: 이전 Codex 정리에서는 certified 원장을 제거했다. 최신 제품 문서와
  새 출처 검사가 해당 자료를 사용하므로 원본의 현재 JSON을 그대로 복원한다.
  Claude 커밋 게이트와 수치 대조 운영은 복원하지 않으며, 출처 경로 검사는
  독립 pytest로 유지한다. 원본을 사용한 새 벤치마크나 재평가는 수행하지 않는다.
- 2026-09-09: 원본의 서버 모듈 설명은 도구 설정과 구분하여 기존
  `docs/manual/server-implementation.md`에 반영한다. 실행 절차는 Codex 지침을 따른다.

## 검증

- `uv run --frozen pytest tests/test_certified_declarations.py tests/migration tests/server -q -rs --ignore=tests/server/test_inference_integration.py --ignore=tests/server/test_live_server.py`: 211 passed.
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' uv run --frozen pytest tests/ner -m 'not integration' -q -rs --ignore=tests/ner/classifier/test_encode.py`: 836 passed, 20 skipped, 2 deselected.
  데이터 부재에 따른 20 skip과 KO gold 대조를 생략한 경고 경로는 미검증이다.
  모델 자립 검사 2개와 실제 tokenizer 검사 파일, 서버 모델 통합·live 경로도 미검증이다.
- `uv run --frozen ruff check src tests`, `git diff --check`,
  `bash -n docker/server/*.sh docker/vllm/*.sh`, `uv lock --check`: 통과했다.
- 원본의 비설정 파일 누락은 0개이며, 결과 JSON 27개는 원본과 바이트가 동일하다.
  Codex 고유 코드·검증과 도구 독립화가 필요한 파일은 의도적으로 원본과 다르다.
- 초기 검사에서 Claude 훅 import 수집 실패, 기존 live 하네스의 영어 모델 누락,
  제거된 sentinel API 참조를 확인하고 수정한 뒤 위 검사를 재실행했다.
- 최종 리뷰 방식은 `codex review --uncommitted`이며 결과는 완료 보고에 명시한다.
모델 다운로드·GPU 사용·외부 서비스 기동은 이번 동기화 검증에 포함하지 않는다.
