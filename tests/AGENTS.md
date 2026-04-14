<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-08 | Updated: 2026-04-08 -->

# tests

## Purpose
pytest-based unit tests for core NER pipeline components: span matching logic (Japanese) and span-level F1 metric computation (Korean KLUE).

## Key Files

| File | Description |
|------|-------------|
| `CLAUDE.md` | Test guide: run command, framework, strategy, caveats |
| `test_span_matcher.py` | 6 tests for `match_spans()`: perfect match, duplicates, substring overlap, not found, empty, multi-type |
| `test_span_f1.py` | 5 tests for `MetricsCalculator.compute_span_f1()`: perfect match, partial mismatch, wrong type, empty, multi-sentence |

## For AI Agents

### Working In This Directory
- Run: `PYTHONPATH=src pytest tests/ -v`
- `test_span_matcher.py` imports from `labelers.ja.span_matcher`
- `test_span_f1.py` uses hardcoded `sys.path.insert(0, "/work/git/ner_pipeline/src")` — only works in dev container
- Test coverage is minimal: no tests for labelers, dataset loading, or classifier

### Testing Requirements
- Lint: `ruff check`
- All tests should pass before committing

## Dependencies

### Internal
- `metrics.bio_metrics` (MetricsCalculator)
- `labelers.ja.span_matcher`

### External
- `pytest`, `ruff`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
