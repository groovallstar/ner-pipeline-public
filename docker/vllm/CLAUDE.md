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
| `stop.sh` | Stop vLLM containers — 인자 없으면 `vllm-gemma`+`vllm-qwen` **둘 다** 내린다 (`PROJECTS="${*:-vllm-gemma vllm-qwen}"`). 한쪽만 내리려면 이름을 인자로 준다 |
| `logs.sh` | Tail vLLM logs — 기본 대상은 `vllm-gemma`, 인자로 다른 컨테이너 지정 (`logs.sh vllm-qwen`) |

## For AI Agents

### Working In This Directory
- Start scripts export env vars then call `docker compose -p <project> up` — all `${...}` values are interpolated from the shell environment
- Both models share a **single compose file** with no per-model branching, so
  `--default-chat-template-kwargs '{"enable_thinking": false}'` in `command:` is passed to the
  gemma launch as well — its intended target is Qwen3's reasoning mode, and gemma just ignores it.
  Making the flag Qwen-only would require splitting the compose file
- Readiness check: poll `curl -s $URL/models | grep $MODEL_NAME`

## Dependencies

### External
- Docker with NVIDIA Container Toolkit
- Host paths: `/data`, `/work/.huggingface`

### Internal
- `ner.llm_eval` (benchmark CLI: `python -m ner.llm_eval`)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
