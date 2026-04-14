<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-08 | Updated: 2026-04-08 -->

# docker

## Purpose
Orchestrates three Docker service layers: a GPU-enabled development container (dev/), an Ollama LLM inference server (ollama/), and a vLLM high-performance inference server (vllm/). Each subdirectory is self-contained with its own docker-compose.yml, config, and lifecycle scripts.

## Key Files

| File | Description |
|------|-------------|
| `CLAUDE.md` | Docker service guide: mount points, environment variables, usage commands |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `dev/` | Development container: PyTorch+CUDA, UV, Claude Code (see `dev/AGENTS.md`) |
| `ollama/` | Ollama server on GPU 0, port 11434 (see `ollama/AGENTS.md`) |
| `vllm/` | vLLM server on GPUs 1,2 with tensor parallelism, port 8081 (see `vllm/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Read `CLAUDE.md` first for mount topology and env var overview
- GPU allocation is fixed: Ollama=GPU 0, vLLM=GPUs 1,2, dev=all GPUs
- Container names: `ner_pipeline_dev`, `ollama-server`, `vllm-server`
- All lifecycle scripts use `cd "$(dirname "$0")"` — run from their subdirectories
- Host paths `/work` and `/data` must exist

### Common Patterns
- Each service has start.sh, stop.sh, logs.sh lifecycle scripts
- Config via `.env` files in `config/` subdirectories

## Dependencies

### External
- Docker Engine with Compose v2
- NVIDIA Container Toolkit

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
