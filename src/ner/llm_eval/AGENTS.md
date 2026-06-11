# llm_eval

## Purpose
벤치마크 오케스트레이션·메트릭 계산 패키지. NER 벤치마크 CLI, seqeval/span-level
메트릭, 포맷 리포트 생성, 문장별 오류 분석을 제공한다.
한국어(BIO 기반)·일본어·베트남어(character-offset span 기반) 평가 경로를 지원한다.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | 패키지 마커 — 직접 모듈 임포트 사용 (BenchmarkRunner, ReportGenerator 등) |
| `__main__.py` | CLI: `python -m ner.llm_eval --lang {ko,ja,vi}` |
| `benchmark_runner.py` | `BenchmarkRunner` — KO(BIO)·JA·VI(offset-span) 공용 러너, eval_mode 스위치 |
| `report.py` | `ReportGenerator` — span-match·seqeval 테이블 + per-entity 분석 (KO·JA·VI) |
| `error_analysis.py` | 문장별 오류 분류 CLI: TYPE_MISMATCH, BOUNDARY_SUBSET, MISS, HALLUCINATION |
| `span_evaluator.py` | gold 레코드 + 라벨러로 span-match F1 계산 유틸 |
| `span_evaluator_cli.py` | 예측 JSONL 소비 → char-offset span F1 비교 테이블 CLI |
| `vi_silver_quality.py` | VI silver vs gold 비교 |
| `wikiann_vi_gold.py` | WikiANN-vi gold 데이터 로딩 유틸 |

## For AI Agents

### Working In This Directory
- 평가 경로: KO는 `BenchmarkRunner`의 BIO eval_mode, JA·VI는 동일 러너의
  offset-span eval_mode 사용 — 혼용 금지
- KO 3개 메트릭: `span_match`(주), seqeval(부), `span_f1`(character-offset from BIO)
- JA·VI: offset-span F1은 `ner.metrics.span_metrics.compute_offset_span_f1` 사용
- 태그 정규화(`_TAG_NORMALIZE_MAP`)는 `labelers/tag_aligner.py` 참조
- `error_analysis.py`는 `llm_eval.__main__`에서 import — 지연 import로 동작

### Testing Requirements
- `TagAligner.spans_to_syllable_bio()` — 공백 토큰 포함 KLUE 음절 토큰으로 테스트
- `compute_span_match()` relaxed matching 테스트
- 벤치마크는 실행 중인 LLM 서버 또는 OpenAI API 키 필요

### Common Patterns
- CLI: `python -m ner.llm_eval --lang ko|ja|vi --models "backend:model" --max-samples N`
- 모델 스펙: `backend:model_name` (예: `vllm:Qwen/Qwen3.5-27B`, `openai:gpt-4o-mini`)
- 최신 벤치마크 수치: `docs/reports/japanese-ner-benchmark.md` 등 참조

## Dependencies

### Internal
- `ner.labelers.tag_aligner` (TagAligner, normalize_tags, extract_spans_from_bio)
- `ner.labelers.hf_ner_labeler` (HFNERLabeler — BERT 베이스라인)
- `ner.labelers.dataset_loader`, `ner.labelers.ko.*`, `ner.labelers.ja.*`, `ner.labelers.vi.*`

### External
- `seqeval`, `tqdm`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
