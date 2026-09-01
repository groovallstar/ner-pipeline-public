# 한국어 라벨러 지침

## 책임과 스키마

한국어 라벨러는 KLUE 기반 `PER LOC ORG DAT`에 LLM 재라벨 `PROD EVT`를 더한다.
LOC/ORG는 narrow-ORG 규칙을 사용한다. 정부·행정·공공·정치 기관만 ORG이며,
시설과 민간 조직은 entity로 추출하지 않는다. JA·VI의 ORG 외연과 다르다.

## 불변조건

- prompt의 한국어 경계, 조사 제외, 복합 entity, PROD/EVT, 명절·기간 규칙을
  근거 없이 단순화하지 않는다.
- prompt는 `SINGLE_PROMPT_TEMPLATE` 한 벌만 유지하고 canonical 규칙과 양방향
  동기 테스트를 유지한다.
- `spans_to_bio()`의 exact 후 substring 2-pass 정렬과 공용 sentence splitter를
  재사용한다.
- gold audit는 모델 예측으로 candidate 모집단을 정의하지 않는다. ledger,
  gold hash, set equality, 다른 entity type 보존 검사를 유지한다.
- `data/*.json`과 `*.jsonl`은 감사 provenance이다. 수치만 맞추려고 재생성하거나
  판정을 자동 선택하지 않는다.

## 금지사항

- canonical 문서와 prompt에 서로 다른 LOC/ORG 또는 EVT 규칙을 두지 않는다.
- count equality만으로 relabel 안전성을 증명하지 않는다.
- recovery가 이미 성공할 model false positive만 골라 gold에 넣지 않는다.
- flat BIO에서 겹치는 entity를 만들지 않는다.

## 검증

- `uv run pytest tests/ner/labelers/test_ko_ner_prompts.py tests/ner/labelers/test_prompt_token_budget.py -q`
- audit 변경은 대응하는 `tests/ner/labelers/test_ko_*audit.py`와
  `tests/ner/labelers/test_ko_locorg_ledger.py`를 실행한다.
- 공용 정렬 변경은 `uv run pytest tests/ner/test_base_labelers.py tests/ner/test_llm_helpers.py -q`
