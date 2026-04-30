<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-08 | Updated: 2026-04-08 -->

# llm_eval

## Purpose
Benchmark orchestration and metrics computation package. Provides CLI entrypoints for running NER benchmarks against gold datasets, computing seqeval/span-level/BERTScore metrics, generating formatted reports, and performing per-sentence error analysis. Supports both Korean (BIO-based) and Japanese (character-offset-based) evaluation paths.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Lazy-import public API: TagAligner (re-export from labelers), MetricsCalculator, BenchmarkRunner, ReportGenerator |
| `__main__.py` | CLI entrypoint (`python -m llm_eval`): parses model specs, routes to Korean or Japanese benchmark runners |
| `benchmark_runner.py` | `BenchmarkRunner` for Korean NER — orchestrates labeling, tag alignment, and three metric levels |
| `ja_benchmark_runner.py` | `JaBenchmarkRunner` for Japanese NER — offset-span pipeline with `JaReportGenerator` |
| `metrics.py` | `MetricsCalculator`: seqeval, span_f1, span_match (exact+relaxed), BERTScore |
| `ja_metrics.py` | `compute_offset_span_f1()` for Japanese offset-span evaluation |
| `report.py` | `ReportGenerator` for Korean: span-match and seqeval tables with per-entity breakdowns |
| `error_analysis.py` | Per-sentence error classification CLI: TYPE_MISMATCH, BOUNDARY_SUBSET, MISS, HALLUCINATION |

## For AI Agents

### Working In This Directory
- Two distinct evaluation paths: Korean (BIO via `BenchmarkRunner`) and Japanese (offset-span via `JaBenchmarkRunner`). Do not mix
- Korean computes three metrics: `span_match` (primary), seqeval (secondary), `span_f1` (character-offset from BIO)
- Japanese computes only offset-span F1 via `ja_metrics.py`
- Tag normalization (`_TAG_NORMALIZE_MAP` in `labelers/tag_aligner.py`) maps HF model tags to KLUE format
- `compute_bertscore()` pads entity lists to equal length — known approximation
- `error_analysis.py` imports from `llm_eval.__main__` — circular-ish dependency that works via lazy imports

### Testing Requirements
- Test `TagAligner.spans_to_syllable_bio()` with KLUE syllable tokens containing space tokens (TagAligner is in `labelers/tag_aligner.py`)
- Test `compute_span_match()` relaxed matching
- Benchmarks require running LLM servers or OpenAI API key

### Common Patterns
- CLI: `python -m llm_eval --lang ja|ko --models "backend:model" --max-samples N`
- Model spec format: `backend:model_name` (e.g., `vllm:Qwen/Qwen3.5-27B`, `openai:gpt-4o-mini`)

## Dependencies

### Internal
- `labelers.tag_aligner` (TagAligner, normalize_tags, extract_spans_from_bio)
- `labelers.hf_ner_labeler` (HFNERLabeler for BERT baseline)
- `labelers.dataset_loader`, `labelers.ko.*`, `labelers.ja.*` (via factory in `__main__.py`)

### External
- `seqeval`, `bert_score`, `tqdm`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
