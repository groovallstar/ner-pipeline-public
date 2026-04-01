# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

This project runs inside a Docker container with GPU support. All development should be done inside the container.

### Starting the Container

```bash
# First-time setup (builds image from scratch)
cd docker/dev && bash init.sh

# Subsequent starts
cd docker/dev && bash start.sh
```

The container runs SSH on port 11111 (host) → 2222 (container). Connect with `ssh root@localhost -p 11111` (password: `root`).

### Python Package Manager

This project uses **UV** (not pip or poetry). Install packages with:

```bash
uv pip install <package>
```

### Environment

- `PYTHONPATH` is set to `/work/git/ner_pipeline/src/` — all source modules live under `src/`
- HuggingFace model cache: `/work/.huggingface/`
- Ollama model cache: `/data/.ollama/`

## Architecture

The project is a **Named Entity Recognition pipeline** built on LangChain with multi-backend LLM support.

**Key dependency groups:**

| Layer | Libraries |
|---|---|
| LLM backends | LangChain + OpenAI, Ollama (local), HuggingFace |
| Vector store / RAG | Milvus (`pymilvus`, `langchain-milvus`), `sentence_transformers` |
| NLP/ML | `transformers`, `datasets`, `accelerate`, `evaluate`, `seqeval`, `bert-score` |
| Orchestration | LangGraph |
| Data | NumPy, Pandas, scikit-learn |

Source code lives under `src/collectors/` (the only current module path). New modules should be added under `src/`.

## Docker Details

The compose file at `docker/dev/docker-compose.yml` mounts:
- `/work` → workspace (project code)
- `/data` → datasets and model weights
- `/var/run/docker.sock` → nested Docker access

GPU support is enabled via NVIDIA device reservations. The base image is `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`.

### Claude Code Settings Sync

After updating Claude Code settings inside the container, sync them to the host with:

```bash
bash docker/dev/copy_claude_setting.sh
```
