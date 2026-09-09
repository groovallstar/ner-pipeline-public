# BERT 토큰 분류기 지침

## 책임과 입력 계약

이 패키지는 canonical JSONL을 소비해 BIO 21-class 모델을 학습하고 strict와
relaxed char-offset span F1을 산출한다. 증강, LLM 라벨링, HF Hub 업로드,
추론 서빙은 이 패키지의 책임이 아니다.

- 입력 label은 canonical 10종이어야 하며 오프셋은 `[start_char, end_char)`이다.
- 모든 3-way와 K-fold 학습은 `--group-key`를 명시한다. JA는 `id`, VI와
  EN은 `orig`, 형제 행이 없는 KO는 `id` 또는 `none`을 사용한다.
- `none`은 누출 미측정 선언이며 누출 0으로 바꾸지 않는다.
- fast, PhoBERT, JA slow tokenizer 정렬 경로를 각각 보존한다.
- metric은 `ner.metrics.span_metrics`를 재사용하고 평가 모델은 float32로 로드한다.

## 출력 계약

- run은 `metrics.json`, best checkpoint, 필요 시 `thresholds.json`과 fold별
  `test_predictions.json`을 남긴다.
- pooled 결과는 그룹 중복 수와 `leak_check_basis`, 비교 기준 필드를 기록한다.
- cross-fold 그룹 누출은 명시적 진단 opt-out이 아니면 `ValueError`로 중단한다.

## 금지사항

- 행 단위 무작위 split로 형제 행을 train/test에 나누지 않는다.
- 고유한 행 ID를 보호 효과가 있는 group key로 가장하지 않는다.
- BIO token 점수와 char-offset span F1을 같은 metric으로 취급하지 않는다.
- 증강 패키지 구현을 직접 import하지 않는다.

## 검증

- `uv run pytest tests/ner/classifier -q`
- tokenizer 정렬 변경은 `tests/ner/classifier/test_encode.py`와
  `uv run python src/ner/scripts/audit_offset_alignment.py --help`도 확인한다.
