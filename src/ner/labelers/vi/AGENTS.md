# 베트남어 라벨러 지침

## 책임과 입출력

베트남어 라벨러는 WikiANN VI 기반 canonical JSONL을 읽고 canonical 10종의
character-offset span을 생성한다. 기본 split dump의 `entities`를 읽어
`gold_spans`로 정규화한다.

## 계약

- 기본 dump는 `data/wikiann_vi/{train,valid,test}.jsonl`에서만 읽으며 파일이
  없으면 즉시 `FileNotFoundError`를 낸다.
- 원본 WikiANN 3종 평가가 아니라 canonical 5종 gold를 기준으로 비교한다.
- diacritics, 다중 단어 entity 단일화, 직함 제외 규칙을 prompt에서 보존한다.
- 공용 span matcher의 경계를 유지한다.

## 금지사항

- loader에 네트워크 접근을 추가하지 않는다.
- 시설 LOC인 원본 WikiANN 기준과 시설 ORG인 canonical 기준을 혼용하지 않는다.
- 베트남어 전용 span matcher를 복제하지 않는다.

## 검증

- `uv run pytest tests/ner/labelers/vi -q`
- LLM 호출 단위 테스트는 `AsyncOpenAI` 요청 경계를 mock한다.
