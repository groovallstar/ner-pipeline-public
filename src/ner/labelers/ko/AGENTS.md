# ko

## Purpose
Korean NER labelers aligned to the canonical schema with the NER 5 types (PER=person, LOC=location, ORG=organization, PROD=product/work, EVT=event) plus DAT=date. PER/LOC/ORG/DAT are renamed from KLUE (PS/LC/OG/DT) with KLUE's TI/QT dropped and still follow KLUE NER guidelines (particle exclusion, syllable boundaries, compound merging). PROD/EVT follow the canonical gray-zone rules (schema §3.1-§3.3: operating leagues=ORG vs specific-edition competitions=EVT; laws/eras/abstract-disputes are non-entities). The PII types remain unfilled (later increments). Two backend implementations (vLLM, OpenAI) share identical prompt templates and span-to-BIO conversion logic.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Re-exports all prompts, helpers, and labeler classes |
| `ner_prompts.py` | Korean NER prompt templates (6 types: PER/LOC/ORG/DAT/PROD/EVT) with few-shot examples and entity type definitions — Korean-specific rules (particle exclusion, compound place names) plus PROD/EVT canonical gray-zone rules |
| `vllm_ner_labeler.py` | `VllmNERLabeler` — async with configurable concurrency (default 32) via AsyncOpenAI against vLLM API |
| `openai_ner_labeler.py` | `OpenAINERLabeler` — token-budget-based batching (max_tokens_per_batch=1000) with chat format |

## For AI Agents

### Working In This Directory
- Prompt engineering is the primary quality lever — `ner_prompts.py` contains highly tuned Korean linguistic rules. Do not simplify
- `_spans_to_bio()` uses 2-pass matching: exact token match first, then substring containment (handles Korean particles like "서울에서" matching "서울")
- `split_sentences()` splits on `.!?` followed by whitespace, minimum 10-char buffer — defined once in `llm_helpers.py`, called via the shared base classes
- vLLM sends all sentences concurrently; OpenAI uses token-budget batching

### Testing Requirements
- Test span-to-BIO alignment for Korean compound entities with particles
- Mock `AsyncOpenAI.chat.completions.create()`

### Common Patterns
- Both labelers import prompts from `ner_prompts.py`
- `DEFAULT_ENTITY_TYPES = ["PER", "LOC", "ORG", "DAT", "PROD", "EVT"]`

## Dependencies

### Internal
- `ner.labelers.ko.ner_prompts`

### External
- `openai` (vllm + openai backends)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
