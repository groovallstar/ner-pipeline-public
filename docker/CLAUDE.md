# docker/ — Docker 서비스 가이드

vLLM 추론 서버(`vllm/`)·ja·vi NER REST API 서버(`server/`) 두 서비스 계층을
묶는다. 각 하위 디렉토리는 자체 `docker-compose.yml`·설정·라이프사이클
스크립트로 자족한다. 개발은 호스트에서 직접 하며 개발 컨테이너는 두지 않는다.

| 하위 | 용도 |
|------|------|
| `vllm/` | NER 라벨링용 vLLM 추론 서버 — 2모델·포트 8081/8082 (상세: `vllm/CLAUDE.md`) |
| `server/` | ja·vi NER REST API 추론 서버 — `/data` 모델 마운트·포트 8008 (상세: `server/CLAUDE.md`) |

## vLLM 서비스 (docker/vllm/)

각 모델을 tensor_parallel=1로 전용 GPU에 띄운다 (gemma: GPU1, qwen: GPU2).
포트: gemma 8081, qwen 8082; compose 자체 기본값은 8000 (`${VLLM_PORT:-8000}`).

```bash
# gemma-4-31B-it-AWQ-8bit 시작 (GPU1, 8081)
bash docker/vllm/start-gemma4-31b-awq-8bit.sh

# Qwen3.6-35B-A3B-AWQ-4bit 시작 (GPU2, 8082)
bash docker/vllm/start-qwen3.6-35b-a3b-awq.sh

# 중지 (기본: vllm-gemma vllm-qwen 모두)
bash docker/vllm/stop.sh

# 로그 (기본: vllm-gemma; 예: logs.sh vllm-qwen)
bash docker/vllm/logs.sh
```

## 이미지 Pull 규칙

- `:latest` 태그 사용 금지. 항상 명시된 버전 태그로 pin 한다 (예: `python:3.12.7-slim`, `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel`).
- 최신 이미지를 받아야 할 때는 Docker Hub/GHCR에서 **최신 안정 버전 태그를 확인한 뒤 그 버전을 명시**한다.
- 이유: `:latest`는 빌드 재현성을 깨고, 무인지 업그레이드로 회귀·환경 불일치를 유발한다.

## 환경 변수

- `HF_HOME=~/.huggingface/` (호스트 사용자 홈 기준; ner-server 는 `/data/ner/_hf_cache`)
- `OPENAI_API_KEY` — 레포 루트 `.env`에서 로드 (CLI `_load_env`)
- `CUDA_VISIBLE_DEVICES` — vLLM 시작 스크립트에서 자동 설정

## 의존성·운영 주의

- 의존성: Docker Engine + Compose v2, NVIDIA Container Toolkit
- GPU 배정: vLLM 은 gemma(labeler)=GPU1, qwen(verifier)=GPU2 각 tensor-parallel=1; ner-server 는 1장 고정(기본 GPU0, NER_SERVER_GPU 로 교체)
- 호스트 경로 `/work`·`/data` 가 존재해야 한다
- 컨테이너명: `vllm-gemma`+`vllm-qwen`(운영, 각자 `-p <name>` compose 프로젝트); `ner-server`(server)
- 모든 라이프사이클 스크립트는 `cd "$(dirname "$0")"` 를 쓰므로 각 하위 디렉토리에서 실행한다
