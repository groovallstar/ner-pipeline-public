# syntax=docker/dockerfile:1
# ja·vi NER REST API 추론 서버 이미지 — dev 와 동일 CUDA 베이스에서 uv 로
# 프로젝트(ner·server)를 설치하고 `python -m server` 로 기동한다. 모델은
# 이미지에 넣지 않고 런타임에 호스트 /data 를 마운트해 읽는다.
FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel

SHELL ["/bin/bash", "-c"]

# 빌드 도구(일부 토크나이저 빌드)·curl(healthcheck)·git 최소 설치
RUN <<EOF
set -e
apt-get update
apt-get install -y --no-install-recommends curl git build-essential
rm -rf /var/lib/apt/lists/*
EOF

# uv 설치(전역 경로)
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /app

# 의존성·프로젝트 설치: 메타+소스 COPY 후 uv sync(.venv 생성, 라이브러리
# ner 와 server 패키지를 editable 로 등록). 빌드 입력은 .dockerignore 로 정리.
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src/ ./src/
RUN uv sync --frozen --system-certs

# 추론 기본값(런타임은 NER_SERVER_* env 로 덮어쓴다)
ENV NER_SERVER_MODEL_ROOT=/data/ner \
    NER_SERVER_HOST=0.0.0.0 \
    NER_SERVER_PORT=8000

EXPOSE 8000

# uv run 으로 .venv 활성화 + 서버 기동. host/port 는 env(from_env)로 결정.
ENTRYPOINT ["uv", "run", "--frozen", "python", "-m", "server"]
