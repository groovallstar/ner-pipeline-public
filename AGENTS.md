<!-- Generated: 2026-04-08 | Updated: 2026-04-08 -->

# ner_pipeline

## Purpose
Multilingual Named Entity Recognition (NER) pipeline built on LangChain with multi-backend LLM support (vLLM, Ollama, OpenAI). Supports Korean (KLUE NER, 6 entity types) and Japanese (Stockmark NER Wikipedia, 8 entity types). Includes BERT fine-tuning for comparison, benchmark evaluation, and Docker-based GPU inference infrastructure.

## Key Files

| File | Description |
|------|-------------|
| `CLAUDE.md` | AI agent project instructions, benchmark commands, skill routing |
| `README.md` | Project overview, installation, usage, benchmark results |
| `pyproject.toml` | Python package definition, all dependencies, UV config with PyTorch CUDA index |
| `HANDOFF.md` | Session handoff document for continuity between work sessions |
| `uv.lock` | UV lockfile for reproducible dependency resolution |
| `.gitignore` | Excludes venvs, caches, model weights, IDE files |
| `.claudeignore` | Excludes large binaries and model weights from AI context |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `src/` | Source code: labelers, evaluators, classifier, reference utils (see `src/AGENTS.md`) |
| `docker/` | Docker service configs for dev, Ollama, and vLLM (see `docker/AGENTS.md`) |
| `tests/` | pytest unit tests (see `tests/AGENTS.md`) |
| `results/` | Benchmark result JSONs, reports, and model checkpoints (see `results/AGENTS.md`) |
| `data/` | Dataset files (see `data/AGENTS.md`) |
| `docs/` | Reference documentation (see `docs/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Set `PYTHONPATH=/work/git/ner_pipeline/src/` before running Python
- Package manager is UV (`uv pip install`), not pip
- Development happens inside Docker containers with GPU access
- Python version: 3.13
- Imports use absolute paths from `src/`: `from labelers.xxx import Xxx`
- Active development on `develop` branch; `main` is the PR target

### Testing Requirements
- Run: `PYTHONPATH=src pytest tests/ -v`
- Lint: `ruff check`
- Benchmarks: `PYTHONPATH=src python -m llm_eval --lang ja|ko --models "backend:model" --max-samples N`

### Common Patterns
- LLM labelers share a common interface: `label(text)`, `label_spans(text)`, `label_records(records)`
- Korean uses BIO tag evaluation; Japanese uses character-offset span evaluation
- Three LLM backends: Ollama (single GPU), vLLM (multi-GPU tensor parallel), OpenAI API

## Dependencies

### External
- LangChain ecosystem (langchain, langchain-openai, langchain-ollama)
- transformers, datasets, evaluate, seqeval, bert-score
- torch (CUDA 13.0), accelerate
- numpy, pandas, scikit-learn

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
