# NER REST API 서비스 지침

## 책임과 API 계약

`src/server`는 JA·KO·VI 모델을 로드해 FastAPI REST API와 웹 데모를 제공한다.
입력은 단일 `text` 또는 배치 `texts` 중 하나이며, 출력 entity는 label, 원문
char offset, text를 보존한다.

- `/v1/ner`는 ja·ko·vi를 지원한다. 미지원 언어는 모델을 호출하지 않고
  `lang: "unsupported"`, 빈 entity 목록을 반환한다.
- 단일·배치 택일 위반은 400, 크기 초과는 413으로 응답한다.
- `/v1/ner`는 대기 큐가 가득 차거나 semaphore acquire timeout이 발생하면
  429로 응답한다.
- 번역 guard는 대기 큐가 없으며 전용 동시성 슬롯이 가득 차면 즉시 429로
  응답한다.
- `/health`는 모든 언어 모델의 준비 상태를 정직하게 보고한다.
- `/v1/translate`는 ja·vi만 받으며 backend를 명시해야 한다.

## 런타임 경계

- startup에서 설정을 검증하고 모델을 모두 로드한 뒤 포트를 연다.
- inference는 언어별 tokenizer와 chunking 경로를 유지하고, batch 결과 순서를
  입력과 1:1로 보존한다.
- gold entity를 가로지르는 chunk 경계를 만들지 않는다.
- NER과 번역의 concurrency budget을 분리한다.
- NLLB tokenizer의 source language 상태 변경 구간은 잠금으로 보호한다.
- 로그에는 요청 ID, 상태, 지연을 남기되 원문 PII를 노출하지 않는다.

## 금지사항

- 번역 backend나 원격 base URL에 암묵적 기본값을 추가하지 않는다.
- ja·vi보다 넓은 NER 언어 목록을 번역 endpoint에 그대로 재사용하지 않는다.
- 모델 준비 전 health를 정상으로 보고하지 않는다.
- 배치 최적화 때문에 단건과 다른 entity 결과를 만들지 않는다.

## 검증

- `uv run pytest tests/server -q`
- 로컬 자원 없는 NLLB·live 테스트의 skip 사유를 확인하고 일반 PASS와 구분한다.
- 소비자 smoke test는 `uv run python -m server.scripts.example_client --help`,
  기동 smoke test는 `uv run python -m server --help`로 배선을 확인한다.
