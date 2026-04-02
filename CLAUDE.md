# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

This project runs inside a Docker container with GPU support. All development should be done inside the container.

### Starting the Container

```bash
# First-time setup (builds image from scratch)
cd docker/dev && bash init.sh
```

VS Code devcontainer (`.devcontainer/devcontainer.json`) 또는 `docker exec`로 접속한다. SSH 포트는 열려 있지 않다.

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

GPU support is enabled via NVIDIA device reservations. The base image is `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel`.

### Claude Code Settings Sync

After updating Claude Code settings inside the container, sync them to the host with:

```bash
bash docker/dev/copy_claude_setting.sh
```

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health
