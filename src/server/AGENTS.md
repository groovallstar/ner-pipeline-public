# NER REST API 서비스 지침

## 책임과 API 계약

`src/server`는 JA·KO·VI·EN 모델을 로드해 FastAPI REST API와 웹 데모를 제공한다.
입력은 단일 `text` 또는 배치 `texts` 중 하나이며, 출력 entity는 label, 원문
char offset, text를 보존한다.

- `/v1/ner`는 ja·ko·vi·en을 지원한다. 미지원 언어는 모델을 호출하지 않고
  `lang: "unsupported"`, 빈 entity 목록을 반환한다.
- 단일·배치 택일 위반은 400, 크기 초과는 413으로 응답한다.
- `/v1/ner`는 대기 큐가 가득 차거나 semaphore acquire timeout이 발생하면
  429로 응답한다.
- 번역 guard는 대기 큐가 없으며 전용 동시성 슬롯이 가득 차면 즉시 429로
  응답한다.
- `/health`는 모든 언어 모델의 준비 상태를 정직하게 보고한다.
- `/v1/translate`는 ja·vi·en만 받으며 원격 LLM의 모델과 base URL을 명시해야 한다.

## 런타임 경계

- startup에서 설정을 검증하고 모델을 모두 로드한 뒤 포트를 연다.
- inference는 언어별 tokenizer와 chunking 경로를 유지하고, batch 결과 순서를
  입력과 1:1로 보존한다.
- gold entity를 가로지르는 chunk 경계를 만들지 않는다.
- NER과 번역의 concurrency budget을 분리한다.
- 언어별 NER tokenizer의 분할·인코딩 구간은 하나의 잠금으로 보호한다.
- 로그에는 요청 ID, 상태, 지연을 남기되 원문 PII를 노출하지 않는다.

## 금지사항

- 번역 모델이나 원격 base URL에 암묵적 기본값을 추가하지 않는다.
- ja·vi·en보다 넓은 NER 언어 목록을 번역 endpoint에 그대로 재사용하지 않는다.
- 모델 준비 전 health를 정상으로 보고하지 않는다.
- 배치 최적화 때문에 단건과 다른 entity 결과를 만들지 않는다.

## 검증

- 모델 없는 기본 검사는 다음 명령으로 실행한다. 모델 통합·live 파일은
  수집에서 제외한다.
  ```bash
  uv run pytest tests/server -q -rs --ignore=tests/server/test_inference_integration.py --ignore=tests/server/test_live_server.py
  ```
- 전체 검사는 `uv run pytest tests/server -q -rs`로 실행한다. 로컬 모델이 있으면
  실모델을 로드하고 live 서버를 기동하므로 실행 전에 모델·GPU와 설정을 확인한다.
- 로컬 자원 없는 live·모델 통합 테스트의 skip 사유와 명시적으로 제외한
  경로를 일반 PASS와 구분한다. `-m 'not live'`만으로 모델 없는 검사가 되지는 않는다.
- CLI 배선은 `uv run python -m server.scripts.example_client --help`와
  `uv run python -m server --help`로 확인한다. 이 검사는 실서버 기동이나
  NER·번역 사용자 흐름의 성공을 보장하지 않는다.

- live 검사만 실행할 때는 `uv run pytest tests/server/test_live_server.py -q -rs`를
  사용한다. `NER_SERVER_MODEL_ROOT`의 ja·vi·ko·en 모델 디렉터리를 먼저 확인하며,
  누락은 skip, 기동 실패·준비성 타임아웃은 실패다. 실패 메시지의 `startup.log`
  경로에서 진단한다. 로그는 pytest 임시 디렉터리의 수명 동안만 보관된다.
- NER→번역 실서버 흐름까지 요구한 작업은 backend와 자원을 준비한 뒤
  `NER_SERVER_TEST_LIVE_TRANSLATE=1 uv run pytest tests/server/test_live_server.py -q -rs`
  를 실행한다. 기존 `NER_SERVER_TRANSLATE_*` 설정을 사용하고 ja·vi의 PHONE
  추출, 한국어 번역과 추출된 PII 원문 보존을 확인한다. 번역 비활성·backend
  미가용은 실패이며, 모델 부재로 skip되면 전체 흐름은 미검증이다.
- `test_live_harness.py`는 모델 없이 사전 조건·준비성·로그·프로세스 회수를,
  `test_translate.py`는 NER→번역 HTTP 배선과 외부 호출 경계의 PII 마스킹을
  검증한다. 실모델 번역 품질이나 브라우저 UI 동작을 검증하는 것은 아니다.
