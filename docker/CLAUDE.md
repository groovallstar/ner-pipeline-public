# docker/ — Docker 서비스 가이드

## 개발 컨테이너 (docker/dev/)

```bash
# 최초 설정
cd docker/dev && bash init.sh
```

- VS Code devcontainer 또는 `docker exec`로 접속
- SSH 포트 미개방
- 베이스 이미지: `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel`

### 마운트

| 호스트 | 컨테이너 | 용도 |
|--------|----------|------|
| /work | /work | 프로젝트 코드 |
| /data | /data | 데이터셋, 모델 가중치 |
| /var/run/docker.sock | /var/run/docker.sock | 중첩 Docker 접근 |

### Claude Code 설정 동기화

컨테이너 내 설정 변경 후 호스트로 동기화:

```bash
bash docker/dev/copy_claude_setting.sh
```

## vLLM 서비스 (docker/vllm/)

GPU 1,2 사용, tensor_parallel=2, 포트 8081.

```bash
# 27B 모델 시작
bash docker/vllm/start-27b.sh

# 35B-A3B 모델 시작
bash docker/vllm/start-35b-a3b.sh

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

- `HF_DATASETS_CACHE=/work/.huggingface/`
- `OPENAI_API_KEY` — `.env` 파일에서 로드
- `CUDA_VISIBLE_DEVICES` — vLLM 시작 스크립트에서 자동 설정
