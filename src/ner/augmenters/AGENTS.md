# 데이터 증강 지침

## 책임과 계약

이 패키지는 PII 합성 주입을 담당한다. 출력 JSONL은 `text`, `entities`, `id`를
제공하며 entity는 canonical label과 `[start_char, end_char)` 오프셋을 갖는다.

- `pii/`의 suffix 모드는 결정적이고, llm 모드는 자연 삽입 후 원본 entity를
  다시 찾아 보존한다. seed와 주입 밀도 계약을 유지한다.

## 금지사항

- 원본 gold entity를 새 텍스트에서 찾지 못했는데 조용히 유지했다고 보고하지 않는다.
- locale별 PII 형식과 email domain을 무관한 언어에 섞지 않는다.
- 평가용 loader에 네트워크 fetch 책임을 추가하지 않는다.
- 증강 모듈에서 classifier를 import하지 않는다. 파일 계약으로만 연결한다.

## 검증

- PII: `uv run pytest tests/ner/augmenters/pii -q`
- 통합 변경은 `uv run pytest tests/ner/augmenters tests/ner/classifier/test_data_utils.py -q`
