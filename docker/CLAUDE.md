# docker/ — Docker 서비스 가이드

## 개발 컨테이너 (docker/dev/)

```bash
# 최초 설정
cd docker/dev && bash init_with_claude_extension.sh
```

- VS Code devcontainer 또는 `docker exec`로 접속
- SSH 포트 미개방
- 베이스 이미지: `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel`

### 마운트

| 호스트 | 컨테이너 | 용도 |
|--------|----------|------|
| /var/run/docker.sock | /var/run/docker.sock | 중첩 Docker 접근 |
| ${HOME}/dev/.claude | ~/.claude | Claude Code 인증 공유 |

### Claude Code 설정 동기화

`init_with_claude_extension.sh` 최초 실행 시 컨테이너 내 인증 파일을
`docker cp` 로 호스트 `${HOME}/dev/.claude/` 에 복사한다.
이후 컨테이너는 해당 경로를 bind-mount로 공유하므로 별도 동기화 불필요.

## vLLM 서비스 (docker/vllm/)

각 모델을 tensor_parallel=1로 전용 GPU에 띄운다 (gemma: GPU1, qwen: GPU2).
포트: gemma 8081, qwen 8082; compose 자체 기본값은 8000 (`${VLLM_PORT:-8000}`).

```bash
# gemma-4-31B-it-AWQ-8bit 시작 (GPU1, 8081)
bash docker/vllm/start-gemma4-31b-awq-8bit.sh

# Qwen3.6-35B-A3B-AWQ-4bit 시작 (GPU2, 8082)
bash docker/vllm/start-qwen3.6-35b-a3b-awq.sh

# 중지
bash docker/vllm/stop.sh

# 로그
bash docker/vllm/logs.sh
```

## 이미지 Pull 규칙

- `:latest` 태그 사용 금지. 항상 명시된 버전 태그로 pin 한다 (예: `python:3.12.7-slim`, `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel`).
- 최신 이미지를 받아야 할 때는 Docker Hub/GHCR에서 **최신 안정 버전 태그를 확인한 뒤 그 버전을 명시**한다.
- 이유: `:latest`는 빌드 재현성을 깨고, 무인지 업그레이드로 회귀·환경 불일치를 유발한다.

## 환경 변수

- `HF_HOME=~/.huggingface/` (dev 컨테이너 내 사용자 홈 기준)
- `OPENAI_API_KEY` — 레포 루트 `.env`에서 로드 (CLI `_load_env`)
- `CUDA_VISIBLE_DEVICES` — vLLM 시작 스크립트에서 자동 설정
