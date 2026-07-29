# ko

## Purpose
Korean NER labelers aligned to the canonical schema with the NER 5 types (PER=person, LOC=location, ORG=organization, PROD=product/work, EVT=event) plus DAT=date. PER/DAT are renamed from KLUE (PS/DT) with KLUE's TI/QT dropped and still follow KLUE NER guidelines (particle exclusion, syllable boundaries, compound merging).

**LOC/ORG do NOT follow KLUE guidelines — they follow the narrow-ORG redefinition.** `ORG` covers government/administrative/public/political bodies only (ministries, agencies, commissions, parties, labor unions, state enterprises, international bodies, legislatures, plus metonymic government-building names). Man-made facilities and venues (stations, airports, hospitals, universities, stadiums, museums) and non-governmental organizations (private companies, broadcasters/press, sports clubs and leagues, military units, courts/prosecution, religious bodies) are **not extracted at all** (non-entity). `LOC` likewise covers only administrative divisions, natural place names, and addresses — building and facility names are dropped. Single source for this rubric: `docs/manual/data/canonical-entity-schema.md` §KO and `docs/manual/data/korean-entity-labeling-rules.md`.

Consequently **KO's LOC/ORG extension differs from JA·VI** — JA·VI canonical absorbs every man-made facility into ORG, while KO discards those same facilities. Do not compare ORG numbers across languages without accounting for this.

PROD/EVT: continuously operating leagues and clubs ("프리미어리그", "메이저리그") are private sports organizations, so they are non-entity rather than ORG; only a specific edition of a competition or a specific match is EVT. Laws, era labels, multi-year states/processes (냉전, 산업혁명), and abstract disputes are non-entity too. The PII types remain unfilled (later increments).

Two backend implementations (vLLM, OpenAI) encode the same rules in **two different templates** — vLLM uses `SINGLE_PROMPT_TEMPLATE` (many few-shot examples), OpenAI uses `SYSTEM_PROMPT` + `USER_PROMPT_TEMPLATE` (condensed, no examples). Editing the rules on one side only makes the two backends disagree. span-to-BIO conversion logic is shared.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Re-exports all prompts, helpers, and labeler classes |
| `ner_prompts.py` | Korean NER prompt templates (6 types: PER/LOC/ORG/DAT/PROD/EVT) — Korean-specific rules (particle exclusion, compound place names) + the **narrow-ORG rubric** (government/political bodies only as ORG, with facilities and non-governmental organizations explicitly enumerated as non-entity) + PROD/EVT boundaries. Two templates: `SINGLE_PROMPT_TEMPLATE` (vLLM, few-shot) / `SYSTEM_PROMPT`+`USER_PROMPT_TEMPLATE` (OpenAI, condensed) |
| `vllm_ner_labeler.py` | `VllmNERLabeler` — async with configurable concurrency (default 32) via AsyncOpenAI against vLLM API |
| `openai_ner_labeler.py` | `OpenAINERLabeler` — token-budget-based batching (max_tokens_per_batch=1000) with chat format |
| `klue_to_canonical_gold.py` | KLUE BIO → canonical char-span gold JSONL conversion (CLI) |
| `ko_prod_evt_relabel.py` | PROD/EVT LLM relabel incremental pipeline (CLI: relabel\|merge) |

## For AI Agents

### Working In This Directory
- Prompt engineering is the primary quality lever — `ner_prompts.py` contains highly tuned Korean linguistic rules. Do not simplify
- `spans_to_bio()` (public, defined in `llm_helpers.py` — shared across ko/ja/vi, not ko-local) uses 2-pass matching: exact token match first, then substring containment (handles Korean particles like "서울에서" matching "서울")
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
- `ner.labelers.base_vllm_labeler` / `ner.labelers.base_openai_labeler` — both labelers subclass these
- `ner.labelers.span_matcher` — used by `ko_prod_evt_relabel.py` for relabel span matching
- `ner.labelers.dataset_loader` — used by `klue_to_canonical_gold.py` to load KLUE source

### External
- `openai` (vllm + openai backends)
- `datasets` — no direct import; pulled in transitively through `dataset_loader`, so it is required when running `klue_to_canonical_gold.py`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
