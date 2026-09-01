# 공용 라벨러 지침

## 책임과 인터페이스

이 패키지는 데이터셋 loader, OpenAI 호환 LLM 라벨러, BIO 정렬, span 매칭을
제공한다. LLM 라벨러의 공통 인터페이스는 `label(text)`, `label_spans(text)`,
`label_records(records)`이다. 로컬 `HFNERLabeler`는 별도 baseline 인터페이스이다.

## 계약

- JSON 응답은 `labeler_base.parse_json_response()`로만 파싱한다.
- 문장 분리와 span-to-BIO는 `llm_helpers.py`, 문자 span 매칭은
  `span_matcher.py`, BIO 정규화는 `tag_aligner.py`의 공용 구현을 재사용한다.
- LLM 요청은 `AsyncOpenAI` 한 클라이언트로 OpenAI 호환 backend에 보낸다.
- 언어별 prompt는 품질 계약이므로 동작 근거 없이 단순화하지 않는다.

## 금지사항

- 언어 하위 패키지에 공용 JSON parser, sentence splitter, span matcher를
  복제하지 않는다.
- 겹침, 조사·접미사, 정규화 실패를 조용히 버리지 않는다.
- 단위 테스트에서 실제 LLM 서버를 호출하지 않는다.

## 검증

- `uv run pytest tests/ner/test_base_labelers.py tests/ner/test_llm_helpers.py tests/ner/test_span_matcher.py tests/ner/test_bio_dataset.py -q`
- 언어별 변경은 해당 하위 지침의 테스트도 실행한다.
