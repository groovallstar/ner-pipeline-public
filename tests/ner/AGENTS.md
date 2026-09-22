# NER 테스트 지침

## 책임과 범위

이 디렉터리는 augmenters, classifier, labelers, llm_eval, metrics, validity,
cross-package scripts의 단위·통합 회귀 테스트를 소유한다. 서버 테스트는
`tests/server/`에 둔다.

## 테스트 계약

- 가장 가까운 모듈 디렉터리 구조를 따르고 실제 공개 경계와 산출물을 검증한다.
- LLM API는 요청 경계에서 mock하되 JSON parsing, 정렬, 병합, metric 로직은
  실제 구현을 실행한다.
- fixture 기대값은 손으로 유도하고 구현 helper로 다시 계산하지 않는다.
- gold, ledger, metric 결과는 count만 보지 말고 row, text, offset, label,
  set equality와 fingerprint 등 손상 가능한 계약을 직접 검증한다.
- 조건부 skip은 GPU라는 추상 조건이 아니라 실제 로컬 모델·tokenizer·선택적
  패키지 부재에만 건다.

## 금지사항

- 네트워크나 실제 LLM backend를 단위 테스트의 전제로 만들지 않는다.
- snapshot을 구현 변경 결과로 무심코 덮어쓰지 않는다.
- 한국어 라벨러 테스트가 별도 `ko/` 테스트 디렉터리에 있다고 가정하지 않는다.

## 실행

- 전체: `uv run pytest tests/ner -q`
- lint: `uv run ruff check tests/ner`
- 실제 모델이 필요한 live 경로는 일반 테스트와 분리하고 미검증 사유를 보고한다.
