# ja

## Purpose
일본어 NER 라벨러. Stockmark NER Wikipedia 데이터셋 기반, canonical 10종 평면
(NER 5종 + PII 5종) 출력. vLLM·OpenAI 백엔드 2종 구현 + 일본어 전용 데이터셋
로더 + character-offset span matcher 포함.
라벨 스키마 단일 출처: `docs/manual/data/canonical-entity-schema.md`

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Re-exports prompts, JapaneseDatasetLoader, match_spans, OpenAINERLabeler, VllmNERLabeler |
| `ner_prompts.py` | Japanese NER prompts — canonical 10종 평면 엔티티 타입 정의 및 disambiguation 규칙 |
| `dataset_loader.py` | `JapaneseDatasetLoader` — reads canonical Stockmark JSONL dumps (`data/stockmark/{train,test}.jsonl`). No HF fetch, no label mapping — fails fast if dumps are missing |
| `span_matcher.py` | `match_spans()` — converts LLM text spans to character offsets using longest-first matching with overlap prevention and Japanese particle stripping |
| `vllm_ner_labeler.py` | Japanese vLLM labeler — sentence splitting includes Japanese punctuation `。！？` |
| `openai_ner_labeler.py` | Japanese OpenAI labeler |

## For AI Agents

### Working In This Directory
- Japanese evaluation uses character-offset spans (not BIO tags). `span_matcher.py` is the critical alignment module
- `span_matcher.py` handles overlapping entities via consumed-range tracking — longest match wins
- `_strip_particles()` fallback strips trailing Japanese particles/honorific suffixes — can incorrectly strip legitimate entity characters
- `JapaneseDatasetLoader` loads pre-dumped canonical JSONL. Deterministic train/test split (seed=42, test_size=0.2) is baked into the dump at generation time — consumers should not re-split
- `DEFAULT_ENTITY_TYPES = ["PER", "LOC", "ORG", "PROD", "EVT", "EMAIL", "PHONE", "DAT", "ID_NUM", "CREDIT_CARD"]` — canonical 10종 평면 목록. 상세: `docs/manual/data/canonical-entity-schema.md`
- canonical 5종 Stockmark 덤프는 augmenters 마이그레이션에서 HF 원본 일본어 8종 → 영문 5종으로 변환·저장된 결과이며, 로더는 그 결과만 읽는다(변환 책임 없음)

### Testing Requirements
- Test `match_spans()` with overlapping entities, particle-containing entities, and whitespace-collapsed matches
- Dataset loader requires the local JSONL dump (`data/stockmark/{train,test}.jsonl`); raises `FileNotFoundError` immediately if absent — no HuggingFace or network access

### Common Patterns
- All labelers import from `ner_prompts.py`
- Sentence splitting regex includes both ASCII and Japanese punctuation

## Dependencies

### Internal
- `ner.labelers.ja.ner_prompts`, `ner.labelers.ja.span_matcher`, `ner.labelers.ja.dataset_loader`

### External
- `openai`, `datasets`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
