<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-08 | Updated: 2026-04-08 -->

# docker

## Purpose
Orchestrates two Docker service layers: a GPU-enabled development container (dev/) and a vLLM high-performance inference server (vllm/). Each subdirectory is self-contained with its own docker-compose.yml, config, and lifecycle scripts.

## Key Files

| File | Description |
|------|-------------|
| `CLAUDE.md` | Docker service guide: mount points, environment variables, usage commands |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `dev/` | Development container: PyTorch+CUDA, UV, Claude Code (see `dev/AGENTS.md`) |
| `vllm/` | vLLM inference server, tensor parallelism, port 8081 (see `vllm/AGENTS.md`) |
| `ollama/` | Ollama inference backend (config only) |

## For AI Agents

### Working In This Directory
- Read `CLAUDE.md` first for mount topology and env var overview
- GPU allocation is fixed: vLLM=GPUs 1,2, dev=all GPUs
- Container names: `ner_pipeline_dev`, `vllm-server`
- All lifecycle scripts use `cd "$(dirname "$0")"` — run from their subdirectories
- Host paths `/work` and `/data` must exist

### Common Patterns
- vllm/ has lifecycle scripts: `start-*.sh` (model-specific), `stop.sh`, `logs.sh`
- dev/ has only `init_with_claude_extension.sh` (first-run setup); no start/stop/logs scripts
- Env vars: host CLIs auto-load repo-root `.env` (gitignored) via `_load_env`; vllm/ uses its own gitignored `config/.env` (model selection, written by `load-and-bench.sh`)

## Dependencies

### External
- Docker Engine with Compose v2
- NVIDIA Container Toolkit

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
