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
| `vi_silver_quality.py` | VI silver vs gold 비교 |
| `wikiann_vi_gold.py` | WikiANN-vi gold 데이터 로딩 유틸 |

## For AI Agents

### Working In This Directory
- 평가 경로: KO는 `BenchmarkRunner`의 BIO eval_mode, JA·VI는 동일 러너의
  offset-span eval_mode 사용 — 혼용 금지
- KO 3개 메트릭: `span_match`(주), seqeval(부), `span_f1`(character-offset from BIO)
- JA·VI: LLM 은 오프셋 없는 `{text, type}` 을 돌려주므로, `ner.labelers.span_matcher.match_spans()`
  로 문자 오프셋을 먼저 붙인 뒤 `ner.metrics.span_metrics.compute_offset_span_f1` 에 넣는다 —
  이 전처리를 건너뛰면 채점 자체가 성립하지 않는다
- 태그 정규화 맵은 언어별로 갈린다 — `_TAG_NORMALIZE_MAP_{KO,JA,VI}` 를 `_TAG_NORMALIZE_MAPS`
  가 묶고 lang 키로 고른다(미등록 lang 은 KO 맵 폴백). `labelers/tag_aligner.py` 참조

### Testing Requirements
- `TagAligner.spans_to_syllable_bio()` — 공백 토큰 포함 KLUE 음절 토큰으로 테스트
- `compute_span_match()` relaxed matching 테스트
- 벤치마크는 실행 중인 vLLM 서버 필요

### Common Patterns
- CLI: `python -m ner.llm_eval --lang ko|ja|vi --models "backend:model" --max-samples N`
- 모델 스펙: `backend:model_name` (예: `vllm:Qwen/Qwen3.5-27B`, `hf:soddokayo/klue-roberta-large-klue-ner`)
- 최신 벤치마크 수치: `docs/reports/japanese-ner-benchmark.md` 등 참조

## Dependencies

### Internal
- `ner.labelers.tag_aligner` (TagAligner, normalize_tags, extract_spans_from_bio)
- `ner.labelers.span_matcher` (match_spans — JA·VI 예측 텍스트 → 문자 오프셋)
- `ner.labelers.hf_ner_labeler` (HFNERLabeler — BERT 베이스라인)
- `ner.labelers.dataset_loader`, `ner.labelers.ko.*`, `ner.labelers.ja.*`, `ner.labelers.vi.*`
- `ner.metrics.bio_metrics` (MetricsCalculator — KO 3개 메트릭), `ner.metrics.span_metrics` (JA·VI strict·relaxed offset span F1)

### External
- `seqeval`, `tqdm`, `datasets` (`wikiann_vi_gold.py` 의 `load_dataset`·`ClassLabel`)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
