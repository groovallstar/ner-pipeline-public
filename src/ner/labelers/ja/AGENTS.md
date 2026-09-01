# 일본어 라벨러 지침

## 책임과 입출력

일본어 라벨러는 canonical Stockmark JSONL을 읽고 canonical 10종의
character-offset span을 생성한다. loader는 `data/stockmark/{train,test}.jsonl`만
읽으며 네트워크에서 데이터를 받거나 다시 split하지 않는다.

## 계약

- 예측 span은 공용 `ner.labelers.span_matcher.match_spans()`로 원문 오프셋에
  맞춘다. 겹침은 긴 match 우선과 consumed range로 처리한다.
- 일본어 문장 분리는 `。！？`와 ASCII 문장부호를 모두 인식한다.
- particle·honorific 제거 fallback은 실제 entity 문자를 잘못 자를 수 있으므로
  관련 회귀 테스트 없이 넓히지 않는다.
- prompt와 출력에서 canonical 10종, 원문 문자, entity 경계를 유지한다.

## 금지사항

- loader에 HuggingFace fetch 또는 label mapping을 추가하지 않는다.
- 생성 시 고정된 train/test dump를 소비자에서 재분할하지 않는다.
- 일본어 전용 span matcher를 다시 만들지 않는다.

## 검증

- `uv run pytest tests/ner/labelers/ja tests/ner/test_span_matcher.py -q`
- prompt 예산 변경은 `uv run pytest tests/ner/labelers/test_prompt_token_budget.py -q`
