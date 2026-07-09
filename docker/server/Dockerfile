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

# 의존성·프로젝트를 2단계로 설치해 레이어 캐시를 최대화한다. uv 다운로드
# 캐시는 build 간 마운트로 재사용해 재빌드 시 재다운로드도 피한다(캐시 경계를
# 넘으므로 hardlink 대신 copy).
ENV UV_LINK_MODE=copy

# 1) 의존성만 설치(프로젝트 소스 제외) — lock·메타에만 의존하는 레이어라
#    src/ 코드가 바뀌어도 무효화되지 않는다(무거운 torch 등 재설치 회피).
COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --system-certs

# 2) 소스 COPY 후 프로젝트(ner·server)만 editable 로 설치 — deps 는 위에서
#    이미 깔려 빠르게 끝난다. 코드 변경은 이 레이어부터만 재실행된다.
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --system-certs

# 추론 기본값(런타임은 NER_SERVER_* env 로 덮어쓴다)
ENV NER_SERVER_MODEL_ROOT=/data/ner \
    NER_SERVER_HOST=0.0.0.0 \
    NER_SERVER_PORT=8008

EXPOSE 8008

# uv run 으로 .venv 활성화 + 서버 기동. host/port 는 env(from_env)로 결정.
ENTRYPOINT ["uv", "run", "--frozen", "python", "-m", "server"]
