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

두 가지 워크플로가 공존한다.

### 1) 프로덕션 라벨링 — 2-모델 동시 구동

NER silver 라벨링은 GPU 2장에 단일-GPU 컨테이너 2개를 띄운다 (각 `tensor_parallel=1`).
compose 자체 포트 기본값은 8000 (`${VLLM_PORT:-8000}`)이며, 각 start 스크립트가
`VLLM_PORT`를 명시해 덮어쓴다.

| 역할 | 모델 | 스크립트 | 컨테이너 | 포트/GPU |
|------|------|----------|----------|----------|
| 1차 라벨러·PII 주입 | `cyankiwi/gemma-4-31B-it-AWQ-8bit` | `start-gemma4-31b-awq-8bit.sh` | `vllm-gemma4-31b-awq8` | 8081 / GPU1 |
| 검증·replacer | `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | `start-qwen3.6-35b-a3b-awq.sh` | `vllm-qwen36-35b-a3b-awq4` | 8082 / GPU2 |

```bash
# 기동 (각 스크립트가 자기 컨테이너를 -p 프로젝트로 정리 후 재기동)
bash docker/vllm/start-gemma4-31b-awq-8bit.sh
bash docker/vllm/start-qwen3.6-35b-a3b-awq.sh

# 로그·중지는 컨테이너 이름으로 직접 다룬다
docker logs -f vllm-gemma4-31b-awq8
docker rm -f vllm-gemma4-31b-awq8 vllm-qwen36-35b-a3b-awq4
```

> 명명 컨테이너는 `-p <이름>` 프로젝트로 뜨므로, 기본 compose 프로젝트(`vllm`)만
> 다루는 `stop.sh`·`logs.sh`로는 제어되지 않는다.

### 2) 벤치마크 오케스트레이션 — load-and-bench.sh

임의 모델을 받아 교체·평가하는 단일-모델 파이프라인. 기본 compose 프로젝트(`vllm`,
컨테이너 `vllm-server`, 포트 8081)를 쓰며 `stop.sh`·`logs.sh`가 여기에 대응한다.

```bash
bash docker/vllm/load-and-bench.sh <모델명> [출력파일]  # 다운로드→로드→error_analysis 50샘플
bash docker/vllm/stop.sh                                # vllm-server 중지
bash docker/vllm/logs.sh                                # vllm-server 로그
```

## 이미지 Pull 규칙

- `:latest` 태그 사용 금지. 항상 명시된 버전 태그로 pin 한다 (예: `python:3.12.7-slim`, `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel`).
- 최신 이미지를 받아야 할 때는 Docker Hub/GHCR에서 **최신 안정 버전 태그를 확인한 뒤 그 버전을 명시**한다.
- 이유: `:latest`는 빌드 재현성을 깨고, 무인지 업그레이드로 회귀·환경 불일치를 유발한다.

## 환경 변수

- `HF_HOME=~/.huggingface/` (dev 컨테이너 내 사용자 홈 기준)
- `OPENAI_API_KEY` — 레포 루트 `.env`에서 로드 (CLI `_load_env`)
- `CUDA_VISIBLE_DEVICES` — vLLM 시작 스크립트에서 자동 설정
