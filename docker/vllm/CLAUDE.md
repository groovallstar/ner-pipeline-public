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
| `stop.sh` | Stop vLLM containers — with no argument it brings down `vllm-gemma`+`vllm-qwen` only (`PROJECTS="${*:-vllm-gemma vllm-qwen}"`). **`vllm-gemma2-9b` is NOT in that default list** — stop it by name (`stop.sh vllm-gemma2-9b`) |
| `logs.sh` | Tail vLLM logs — defaults to `vllm-gemma`; pass another container as an argument (`logs.sh vllm-qwen`) |

## For AI Agents

### Working In This Directory
- Start scripts export env vars then call `docker compose -p <project> up` — all `${...}` values are interpolated from the shell environment
- All three start scripts share a **single compose file** with no per-model branching — they differ
  only in the env vars they export and the `-p <project>` they pass. So
  `--default-chat-template-kwargs '{"enable_thinking": false}'` in `command:` reaches every launch,
  not just Qwen3's, which is its intended target; the gemma models simply ignore it.
  Making the flag Qwen-only would require splitting the compose file
- Readiness check: poll `curl -s $URL/models | grep $MODEL_NAME`

## Dependencies

### External
- Docker with NVIDIA Container Toolkit
- Host paths: `/data`, `/work/.huggingface`

### Internal
- `ner.llm_eval` (benchmark CLI: `python -m ner.llm_eval`)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
