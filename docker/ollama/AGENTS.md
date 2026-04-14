<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-08 | Updated: 2026-04-08 -->

# ollama

## Purpose
Runs an Ollama inference server (v0.19.0) on GPU 0 for local LLM inference. Models cached on host at `/data/.ollama/`. Flash attention enabled by default.

## Key Files

| File | Description |
|------|-------------|
| `docker-compose.yml` | Ollama service: port 11434, GPU reservation, model volume |
| `config/.env` | Server config: CUDA_VISIBLE_DEVICES=0, KEEP_ALIVE=1h, FLASH_ATTENTION=1 |
| `start.sh` | Start with env-file |
| `stop.sh` | Stop container |
| `logs.sh` | Tail logs |

## For AI Agents

### Working In This Directory
- Uses GPU 0 exclusively — do not change without checking vLLM allocation (1,2)
- KEEP_ALIVE=1h means models unload after 1h idle; set to -1 for benchmarks
- Models must be pulled after server start: `docker exec ollama-server ollama pull <model>`

## Dependencies

### External
- Docker with NVIDIA Container Toolkit
- Host path `/data/.ollama/`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
