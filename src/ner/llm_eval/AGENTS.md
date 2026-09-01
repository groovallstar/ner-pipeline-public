# LLM 평가 지침

## 책임과 평가 경로

이 패키지는 benchmark 실행, metric 계산, report, 오류 분석을 담당한다. KO는
BIO eval mode, JA·VI는 character-offset span eval mode를 사용한다.

## 계약

- KO는 `span_match`를 주 metric으로 보고 seqeval과 BIO에서 변환한 span F1을
  함께 낸다.
- JA·VI의 `{text, type}` 예측은 공용 `match_spans()`로 오프셋을 붙인 뒤
  `compute_offset_span_f1`에 전달한다.
- strict와 relaxed 결과를 구분하고 언어별 tag normalization map을 유지한다.
- report의 새 수치는 재현 가능한 run 산출물과 연결한다.

## 금지사항

- KO BIO 평가와 JA·VI offset 평가를 혼용하지 않는다.
- span matching 전처리를 건너뛴 예측을 offset metric에 넣지 않는다.
- 오류 분석 분류와 headline metric의 의미를 임의로 합치지 않는다.

## 검증

- `uv run pytest tests/ner/llm_eval tests/ner/test_span_evaluator.py tests/ner/test_span_f1.py tests/ner/test_span_metrics.py -q`
- CLI 배선은 `uv run python -m ner.llm_eval --help`로 확인한다.
- 실제 benchmark는 별도로 실행 중인 backend가 있을 때만 검증한다.
