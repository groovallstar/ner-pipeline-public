# Span Evaluator

LLM NER 라벨링 결과를 gold 데이터와 비교하여 span 수준 메트릭을 산출하는 평가 모듈.

- 소스: `src/ner/llm_eval/span_evaluator.py`
- 테스트: `tests/ner/test_span_evaluator.py`

## 배경

`bio_dataset.load()`로 gold 데이터를 로드하고 LLM 라벨러로 span을 추출한 뒤, 두 결과를 비교하여 precision/recall/F1을 계산한다. 기존 `BenchmarkRunner`는 레거시 `DatasetLoader` 포맷에 의존하고 BERT 평가 경로와 혼재되어 있어, LLM span 평가만 독립적으로 수행하기 어려웠다.

## 공개 API

### `evaluate(gold_records, labeler, *, max_samples=None, show_progress=True, collect_diffs=False) -> dict`

`bio_dataset.load()` 반환 레코드를 gold로, `labeler.label_spans(sentence)` 결과를 pred로 받아 span 수준 메트릭을 계산한다.

```python
from ner.labelers.bio_dataset import load
from ner.labelers.ko.vllm_ner_labeler import VllmNERLabeler
from ner.llm_eval.span_evaluator import evaluate

gold = load("klue", "validation", max_samples=500)
labeler = VllmNERLabeler(
    base_url="http://localhost:8081/v1",
    model="Qwen/Qwen3.5-27B",
    concurrency=32,
)
result = evaluate(gold, labeler)
```

#### 파라미터

| 이름 | 타입 | 설명 |
|------|------|------|
| `gold_records` | `list[dict]` | `bio_dataset.load()` 반환값. 각 레코드에 `sentence`, `spans` 필드 필요 |
| `labeler` | any | `label_spans(text) -> list[{"text", "type"}]` 메서드를 가진 객체 |
| `max_samples` | `int \| None` | 평가할 최대 레코드 수 제한 |
| `show_progress` | `bool` | tqdm 진행 표시 (기본값: `True`) |
| `collect_diffs` | `bool` | `True`이면 반환값에 레코드별 diff 분석 `"diffs"` 키 추가 (기본값: `False`) |

#### 반환값

```python
{
    "exact": {
        "overall": {"f1": 0.7345, "precision": 0.7743, "recall": 0.6986},
        "per_entity": {
            "PS": {"f1": 0.8365, "precision": 0.9463, "recall": 0.7494, "support": 447},
            "LC": {"f1": 0.5090, ...},
            ...
        },
    },
    "relaxed": {
        "overall": {"f1": 0.8094, "precision": 0.8533, "recall": 0.7698},
        "per_entity": { ... },
    },
    "counts": {
        "gold_spans": 1390,
        "pred_spans": 1254,
        "exact_matches": 971,
        "relaxed_matches": 1070,
        "errors": 15,
    },
    "latency": {
        "total_seconds": 568.02,
    },
    # collect_diffs=True 시에만 포함
    "diffs": [ ... ],  # 레코드별 diff 분석 리스트
}
```

## 메트릭 설명

### Exact Match

gold span과 pred span의 **텍스트와 타입이 모두 동일**할 때 매칭. 텍스트 비교 전에 공백을 제거하여 정규화한다.

| gold (KLUE 음절 BIO) | pred (LLM) | 정규화 후 | 판정 |
|-----------------------|------------|-----------|------|
| `"지난19일"` | `"지난 19일"` | `"지난19일"` = `"지난19일"` | exact match |
| `"경찰"` | `"경찰청"` | `"경찰"` ≠ `"경찰청"` | not exact |

### Relaxed Match

타입이 동일하고, 한쪽 텍스트가 다른 쪽을 **포함**할 때 매칭. 경계(boundary) 차이를 허용한다.

| gold | pred | 판정 |
|------|------|------|
| `"박"` (PS) | `"박씨"` (PS) | relaxed match (`"박"` ⊂ `"박씨"`) |
| `"서울"` (LC) | `"서울시"` (LC) | relaxed match (`"서울"` ⊂ `"서울시"`) |
| `"경찰"` (OG) | `"경찰"` (PS) | not match (타입 불일치) |

### Per-entity 메트릭

엔티티 타입별(PS, LC, OG, DT, TI) precision, recall, F1을 독립 산출한다. KLUE NER gold 태그셋은 PS/LC/OG/DT/TI 5종이며 QT는 포함되지 않는다.

## 에러 처리

`labeler.label_spans()`가 특정 레코드에서 예외를 발생시키면:
- 해당 레코드를 건너뛰고 `counts.errors`를 증가
- 나머지 레코드는 정상 평가 계속
- 에러 원인은 `logging.warning`으로 기록

## 데이터 흐름

```
bio_dataset.load(spec_name, split, max_samples)
    ↓
[{id, tokens, bio_tags, spans, sentence}, ...]
    ↓
evaluate(gold_records, labeler)
    ├─ record["sentence"] → labeler.label_spans() → pred spans
    ├─ record["spans"] → gold spans
    └─ MetricsCalculator.compute_span_match(gold_all, pred_all)
        ├─ exact: 공백 제거 후 text+type 동일
        └─ relaxed: 공백 제거 후 type 동일 + 텍스트 포함 관계
            ↓
        {exact, relaxed, counts, latency}
```

## 테스트 케이스

실행: `pytest tests/ner/test_span_evaluator.py -v`

| 테스트 | 검증 내용 |
|--------|-----------|
| `test_empty_gold_records` | 빈 입력 → 모든 메트릭 0.0 |
| `test_perfect_match` | gold = pred → exact F1 = 1.0 |
| `test_partial_match` | gold 3건 중 2건 정확 → precision 1.0, recall 0.67 |
| `test_space_normalization` | `"지난19일"` vs `"지난 19일"` → exact match |
| `test_relaxed_match` | `"박"` vs `"박씨"` → exact 0.0, relaxed 1.0 |
| `test_labeler_error` | 1건 에러 → skip + errors=1, 나머지 정상 |
| `test_max_samples` | 100건 중 10건만 평가 |
| `test_return_structure` | 반환값에 exact, relaxed, counts, latency 키 존재 |
| `test_per_entity_metrics` | OG F1=1.0, PS F1=0.0 독립 계산 |
