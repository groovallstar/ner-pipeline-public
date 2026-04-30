<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-08 | Updated: 2026-04-08 -->

# ja

## Purpose
Japanese NER labelers targeting the Stockmark NER Wikipedia dataset with 8 entity types (person, corporate, location, facility, product, event, political org, other org). Two backend implementations (vLLM, OpenAI) plus a Japanese-specific dataset loader and character-offset span matcher.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Re-exports prompts, JapaneseDatasetLoader, and match_spans |
| `ner_prompts.py` | Japanese NER prompts with 8 entity types and disambiguation rules (e.g., railroad=corporate, military=political, sports league=other org) |
| `dataset_loader.py` | `JapaneseDatasetLoader` — reads canonical Stockmark JSONL dumps (`data/stockmark/{train,test}.jsonl`). No HF fetch, no label mapping — fails fast if dumps are missing (issue #21) |
| `span_matcher.py` | `match_spans()` — converts LLM text spans to character offsets using longest-first matching with overlap prevention and Japanese particle stripping |
| `vllm_ner_labeler.py` | Japanese vLLM labeler — sentence splitting includes Japanese punctuation `。！？` |
| `openai_ner_labeler.py` | Japanese OpenAI labeler |

## For AI Agents

### Working In This Directory
- Japanese evaluation uses character-offset spans (not BIO tags). `span_matcher.py` is the critical alignment module
- `span_matcher.py` handles overlapping entities via consumed-range tracking — longest match wins
- `_strip_particles()` fallback strips trailing Japanese particles/honorific suffixes — can incorrectly strip legitimate entity characters
- `JapaneseDatasetLoader` loads pre-dumped canonical JSONL. Deterministic train/test split (seed=42, test_size=0.2) is baked into the dump at generation time — consumers should not re-split
- `DEFAULT_ENTITY_TYPES = ["PER", "LOC", "ORG", "PROD", "EVT", "EMAIL", "PHONE", "DAT", "ID_NUM", "CREDIT_CARD"]` — canonical 10종 평면 목록(이슈 #21). 상세: `docs/manual/data/canonical-entity-schema.md`
- canonical 5종 Stockmark 덤프는 augmenters 마이그레이션에서 HF 원본 일본어 8종 → 영문 5종으로 변환·저장된 결과이며, 로더는 그 결과만 읽는다(변환 책임 없음)

### Testing Requirements
- Test `match_spans()` with overlapping entities, particle-containing entities, and whitespace-collapsed matches
- Dataset loader requires network access or cached HuggingFace data

### Common Patterns
- All labelers import from `ner_prompts.py`
- Sentence splitting regex includes both ASCII and Japanese punctuation

## Dependencies

### Internal
- `labelers.ja.ner_prompts`, `labelers.ja.span_matcher`, `labelers.ja.dataset_loader`

### External
- `openai`, `langchain_core`, `datasets`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
