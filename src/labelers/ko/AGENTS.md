<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-08 | Updated: 2026-04-08 -->

# ko

## Purpose
Korean NER labelers targeting KLUE NER annotation guidelines with 6 entity types (PS=person, LC=location, OG=organization, DT=date, TI=time, QT=quantity). Three backend implementations (Ollama, vLLM, OpenAI) share identical prompt templates and span-to-BIO conversion logic.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Re-exports all prompts, helpers, and labeler classes |
| `ner_prompts.py` | Korean NER prompt templates with few-shot examples and entity type definitions including Korean-specific rules (particle exclusion, compound place names) |
| `ollama_ner_labeler.py` | `OllamaNERLabeler` — synchronous batch processing via LangChain ChatOllama, temperature=0 |
| `vllm_ner_labeler.py` | `VllmNERLabeler` — async with configurable concurrency (default 32) via AsyncOpenAI against vLLM API |
| `openai_ner_labeler.py` | `OpenAINERLabeler` — token-budget-based batching (max_tokens_per_batch=1000) with chat format |

## For AI Agents

### Working In This Directory
- Prompt engineering is the primary quality lever — `ner_prompts.py` contains highly tuned Korean linguistic rules. Do not simplify
- `_spans_to_bio()` uses 2-pass matching: exact token match first, then substring containment (handles Korean particles like "서울에서" matching "서울")
- `_split_sentences()` splits on `.!?` followed by whitespace, minimum 10-char buffer — duplicated in all three files
- vLLM sends all sentences concurrently; Ollama batches sequentially; OpenAI uses token-budget batching

### Testing Requirements
- Test span-to-BIO alignment for Korean compound entities with particles
- Mock `ChatOllama.invoke()`, `AsyncOpenAI.chat.completions.create()`

### Common Patterns
- All three labelers import prompts from `ner_prompts.py`
- `DEFAULT_ENTITY_TYPES = ["PS", "LC", "OG", "DT", "TI", "QT"]`

## Dependencies

### Internal
- `labelers.ko.ner_prompts`

### External
- `openai` (vllm + openai backends), `langchain_ollama` + `langchain_core` (ollama backend)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
