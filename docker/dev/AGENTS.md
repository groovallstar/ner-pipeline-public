<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-08 | Updated: 2026-04-08 -->

# dev

## Purpose
Builds and runs the primary development container based on `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel`. Includes UV for Python dependency management, Node.js LTS, Docker CLI (nested access), Claude Code CLI, and oh-my-claudecode.

## Key Files

| File | Description |
|------|-------------|
| `dockerfile` | Multi-stage build: apt packages, UV, uv sync, host UID/GID mapping, Node.js, Claude Code |
| `docker-compose.yml` | Service definition: Docker socket mount, GPU reservation, privileged mode, `sleep infinity` |
| `init_with_claude_extension.sh` | First-run setup: builds image, authenticates Claude Code, copies auth to host |

## For AI Agents

### Working In This Directory
- The Dockerfile copies `pyproject.toml`, `uv.lock`, `.python-version` at build time — these are the authoritative dependency definitions
- Container runs as non-root user matching host UID/GID (build args)
- `.claude` directory mounted from `${HOME}/dev/.claude` on the host
- `init_with_claude_extension.sh` is interactive — cannot be run headlessly

## Dependencies

### External
- Base: `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel`
- UV, Node.js LTS, npm packages: `@fission-ai/openspec`, `oh-my-claude-sisyphus`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
