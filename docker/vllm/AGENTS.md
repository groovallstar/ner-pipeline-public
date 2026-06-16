<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-08 | Updated: 2026-04-08 -->

# vllm

## Purpose
Runs a vLLM OpenAI-compatible inference server (v0.23.0) with tensor parallelism across GPUs 1,2 for high-throughput LLM inference. Includes model-specific start scripts (all default to port 8081; compose default is 8000) and automated benchmark orchestration.

## Key Files

| File | Description |
|------|-------------|
| `docker-compose.yml` | vLLM service: configurable model/GPU/memory/dtype/port via env vars, IPC host, HF cache mount |
| `config/.env` | Env file for model selection (populated by start scripts) |
| `start-*.sh` | Model-specific start scripts (set `VLLM_PORT=8081`, GPUs, tensor parallelism) |
| `load-and-bench.sh` | Single-model pipeline: download, write config/.env, restart, benchmark |
| `stop.sh` | Stop vLLM container |
| `logs.sh` | Tail vLLM logs |

## For AI Agents

### Working In This Directory
- Start scripts export env vars then call docker compose — they do NOT modify config/.env (exception: `load-and-bench.sh` uses `sed -i`)
- `load-and-bench.sh` has hardcoded Docker network IP (`172.22.0.2:8081`) — breaks if network topology changes
- `--default-chat-template-kwargs '{"enable_thinking": false}'` in docker-compose.yml disables Qwen3's reasoning mode
- Readiness check: poll `curl -s $URL/models | grep $MODEL_NAME`

### Testing Requirements
- `load-and-bench.sh` runs `ner.llm_eval.error_analysis` on 50 samples by default

## Dependencies

### External
- Docker with NVIDIA Container Toolkit (2 GPUs required)
- Host paths: `/data`, `/work/.huggingface`

### Internal
- `ner.llm_eval` (benchmark CLI: `python -m ner.llm_eval`)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
