<!-- Generated: 2026-04-08 | Updated: 2026-06-16 -->

# vllm

## Purpose
Runs vLLM OpenAI-compatible inference servers (v0.23.0) for high-throughput
LLM inference. Each model runs with tensor-parallel size 1 on a dedicated GPU
via its own start script (gemma on GPU1:8081, qwen on GPU2:8082; compose
default port is 8000).

## Key Files

| File | Description |
|------|-------------|
| `docker-compose.yml` | vLLM service: configurable model/GPU/memory/dtype/port via env vars, IPC host, HF cache mount |
| `start-gemma4-31b-awq-8bit.sh` | Start gemma-4-31B-it-AWQ-8bit (GPU1, port 8081) |
| `start-qwen3.6-35b-a3b-awq.sh` | Start Qwen3.6-35B-A3B-AWQ-4bit (GPU2, port 8082) |
| `start-gemma2-9b.sh` | Start gemma-2-9b-it-w4a16 for web-demo translation (GPU1, port 8081; GPU/port/memory overridable via env). **Ampere+ only** — bf16-only model (fp16 rejected, fp32 fails in Marlin kernel), so it will not start on Turing |
| `stop.sh` | Stop vLLM container |
| `logs.sh` | Tail vLLM logs |

## For AI Agents

### Working In This Directory
- Start scripts export env vars then call `docker compose -p <project> up` — all `${...}` values are interpolated from the shell environment
- `--default-chat-template-kwargs '{"enable_thinking": false}'` in docker-compose.yml disables Qwen3's reasoning mode
- Readiness check: poll `curl -s $URL/models | grep $MODEL_NAME`

## Dependencies

### External
- Docker with NVIDIA Container Toolkit
- Host paths: `/data`, `/work/.huggingface`

### Internal
- `ner.llm_eval` (benchmark CLI: `python -m ner.llm_eval`)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
