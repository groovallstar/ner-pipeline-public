#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export DOCKER_UID=$(id -u)
export DOCKER_GID=$(id -g)

CLAUDE_HOST_DIR="${HOME}/dev/.claude"
CONTAINER_NAME="ner_pipeline_dev"
CONTAINER_USER="${USER:-devuser}"
CONTAINER_HOME="/home/${CONTAINER_USER}"

# 1. 이미지 빌드
echo "=== 이미지 빌드 ==="
docker compose -f docker-compose.yml build

# 2. .claude 마운트 없이 컨테이너 시작 (인증을 위해)
echo "=== 컨테이너 시작 (마운트 없이) ==="
docker compose -f docker-compose.yml -f docker-compose.init.yml up -d

# 3. 사용자에게 인증 안내
echo ""
echo "============================================"
echo " Claude Code 인증이 필요합니다."
echo " 아래 명령어로 컨테이너에 접속 후 claude를 실행하세요:"
echo ""
echo "   docker exec -it -u ${CONTAINER_USER} ${CONTAINER_NAME} bash"
echo "   claude"
echo ""
echo " 인증이 완료되면 이 터미널로 돌아와서 Enter를 누르세요."
echo "============================================"
echo ""
read -p "인증 완료 후 Enter를 누르세요..."

# 4. 컨테이너에서 호스트로 인증 파일 복사
echo "=== 인증 파일을 호스트로 복사 ==="
mkdir -p "${CLAUDE_HOST_DIR}"

docker cp "${CONTAINER_NAME}:${CONTAINER_HOME}/.claude/." "${CLAUDE_HOST_DIR}/"

# ~/.claude.json이 있으면 함께 복사
if docker exec ${CONTAINER_NAME} test -f "${CONTAINER_HOME}/.claude.json"; then
    docker cp "${CONTAINER_NAME}:${CONTAINER_HOME}/.claude.json" "${CLAUDE_HOST_DIR}/../.claude.json"
    echo "  - .claude.json 복사 완료"
fi

echo "  - .claude/ 복사 완료 → ${CLAUDE_HOST_DIR}"

# 5. 마운트 포함하여 컨테이너 재시작
echo "=== 컨테이너 재시작 (마운트 포함) ==="
docker compose -f docker-compose.yml down
docker compose -f docker-compose.yml up -d

echo ""
echo "=== 완료! ==="
echo "컨테이너가 ${CLAUDE_HOST_DIR} 마운트와 함께 실행 중입니다."
