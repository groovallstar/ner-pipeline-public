<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-08 | Updated: 2026-04-08 -->

# labelers

## Purpose
Core NER labeling package. Provides a HuggingFace dataset loader and LLM-based labelers that call Ollama, vLLM, or OpenAI backends to extract named entities from text. Each labeler converts raw LLM JSON span output into BIO-tagged token sequences. Contains language-specific subpackages for Korean and Japanese.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Public API: exports DatasetLoader, NERRecord, OllamaNERLabeler |
| `labeler_base.py` | Shared `parse_json_response()` for extracting JSON spans from LLM output (handles think-tags, markdown fences, wrapped dicts) |
| `dataset_loader.py` | `DatasetLoader` for HuggingFace NER datasets (KLUE, KMounLP) with ClassLabel conversion and JSONL fallback |
| `tag_aligner.py` | `TagAligner`: BIO tag alignment to gold token boundaries, KLUE syllable tokenization, tag normalization (PER→PS, LOC→LC), `extract_spans_from_bio()` |
| `hf_ner_labeler.py` | `HFNERLabeler` wrapping HuggingFace pipeline for BERT-based NER baseline |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `ko/` | Korean NER labelers — KLUE 6-entity (PS, LC, OG, DT, TI, QT) (see `ko/AGENTS.md`) |
| `ja/` | Japanese NER labelers — Stockmark 8-entity with span matcher and enhanced labeler (see `ja/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- All labelers share a common interface: `label(text)`, `label_spans(text)`, `label_records(records)`
- `parse_json_response()` in `labeler_base.py` is the canonical JSON extraction utility — use it rather than writing new JSON parsing
- `_split_sentences()` and `_spans_to_bio()` are duplicated across all labeler files — when modifying, update all copies
- Each labeler tracks token usage via `total_prompt_tokens` and `total_completion_tokens`
- `DatasetLoader` has a JSONL fallback path at `/data/ner/` for Python 3.13 compatibility

### Testing Requirements
- Unit tests should mock LLM API calls
- Test `parse_json_response()` with edge cases: think-tags, nested dicts, markdown-fenced JSON
- Integration tests require running LLM servers

### Common Patterns
- Prompt engineering is the primary quality lever — prompts are highly tuned per language
- Labelers use `openai.AsyncOpenAI` (vllm/openai) and `langchain_ollama.ChatOllama` (ollama)
- Default models: Ollama=`qwen3.5:27b`, vLLM=`Qwen/Qwen3.5-27B`, OpenAI=`gpt-5-mini`

## Dependencies

### Internal
- None (leaf package for labeling; `tag_aligner.py` is consumed by `llm_eval/` for benchmark evaluation)

### External
- `openai`, `langchain_ollama`, `langchain_core`, `datasets`, `transformers`, `torch`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
