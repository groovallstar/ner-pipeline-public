<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-08 | Updated: 2026-04-08 -->

# ref

## Purpose
Reference/utility scripts for synthetic PII data generation. Contains template-based generators that create annotated sentences with entity spans for Japanese and Vietnamese, used to train and benchmark PII NER models.

## Key Files

| File | Description |
|------|-------------|
| `pii_gen.py` | Original PII generator: 7 types (NAME, PHONE, ADDRESS, DOB, ID_NUMBER, EMAIL, CREDIT_CARD), Japanese+Vietnamese templates. Imported by `classifier/pii_benchmark.py` |
| `pii_gen_new.py` | Enhanced v2: adds BIO tag generation, optional spaCy tokenization, CoNLL-style output, `SyntheticSample.lang` field |

## For AI Agents

### Working In This Directory
- `pii_gen.py` is the version imported by `classifier/pii_benchmark.py` — modify this one for production changes
- Both generators use `text.find(value)` for annotation — fails if same PII value appears twice in a template (known limitation)
- `pii_gen_new.py` attempts spaCy model loading but falls back to regex tokenization
- ~12 templates per language covering formal, conversational, call-center, and noise formats

### Testing Requirements
- Test `generate_synthetic_sentence()` to ensure all entity offsets are correctly computed (no -1 from find failures)
- `pii_gen_new.py` requires spaCy for full functionality

## Dependencies

### Internal
- None (standalone utility)

### External
- `spacy` (optional, for `pii_gen_new.py`)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
