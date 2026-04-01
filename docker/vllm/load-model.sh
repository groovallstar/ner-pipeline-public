#!/bin/bash
# 모델을 교체하고 vLLM 서버를 재시작합니다.
# 사용법: bash load-model.sh <모델명> [포트]
# 예시:  bash load-model.sh Qwen/Qwen2.5-7B-Instruct
#        bash load-model.sh meta-llama/Llama-3.1-8B-Instruct 8001

set -e

if [ -z "$1" ]; then
  echo "사용법: $0 <모델명> [포트]"
  echo "예시:  $0 Qwen/Qwen2.5-7B-Instruct"
  echo "       $0 Qwen/Qwen2.5-7B-Instruct 8001"
  exit 1
fi

MODEL="$1"
PORT="${2:-}"

# .env 파일에서 VLLM_MODEL 업데이트
sed -i "s|^VLLM_MODEL=.*|VLLM_MODEL=${MODEL}|" config/.env

# 포트 인자가 있으면 .env도 업데이트
if [ -n "$PORT" ]; then
  sed -i "s|^VLLM_PORT=.*|VLLM_PORT=${PORT}|" config/.env
fi

echo "모델: ${MODEL}"
[ -n "$PORT" ] && echo "포트: ${PORT}"
echo ""
echo "컨테이너 재시작 중..."
docker compose --env-file config/.env -f docker-compose.yml down
docker compose --env-file config/.env -f docker-compose.yml up -d

echo ""
echo "로그 확인: bash logs.sh"
