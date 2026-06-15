<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-08 | Updated: 2026-06-15 -->

# vllm

## Purpose
Runs vLLM OpenAI-compatible inference servers for NER silver labeling. The production setup launches two single-GPU containers concurrently — a gemma labeler (GPU1, port 8081) and a Qwen verifier/replacer (GPU2, port 8082), each tensor-parallel=1. Also provides single-model benchmark orchestration (`load-and-bench.sh`). Compose port default is 8000; each start script overrides it via `VLLM_PORT`.

## Key Files

| File | Description |
|------|-------------|
| `docker-compose.yml` | vLLM service: configurable model/GPU/memory/dtype/port via env vars, IPC host, HF cache mount |
| `start-gemma4-31b-awq-8bit.sh` | Production labeler — `cyankiwi/gemma-4-31B-it-AWQ-8bit`, GPU1, port 8081, container `vllm-gemma4-31b-awq8` (image `vllm/vllm-openai:gemma4`) |
| `start-qwen3.6-35b-a3b-awq.sh` | Production verifier/replacer — `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`, GPU2, port 8082, container `vllm-qwen36-35b-a3b-awq4` |
| `config/.env` | Model-selection env file — written by `load-and-bench.sh` (`sed`); production start scripts export env vars directly |
| `load-and-bench.sh` | Benchmark pipeline: download a model, swap config/.env, restart the default `vllm` project (`vllm-server`), run `error_analysis` on 50 samples |
| `stop.sh` / `logs.sh` | Stop / tail the default `vllm` compose project only (load-and-bench's `vllm-server`) — NOT the named production containers |

## For AI Agents

### Working In This Directory
- Start scripts export env vars then call docker compose — they do NOT modify config/.env (exception: `load-and-bench.sh` uses `sed -i`)
- Production start scripts launch under `docker compose -p <container-name>` (own project) with an explicit `container_name`; manage them with `docker rm -f <name>` / `docker logs -f <name>` — `stop.sh`/`logs.sh` (default `vllm` project) do NOT reach them
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
