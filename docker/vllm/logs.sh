#!/bin/bash
# vLLM 컨테이너 로그 tail
# 사용법: logs.sh [컨테이너명] (기본: vllm-gemma4-31b-awq-8bit)

docker logs -f "${1:-vllm-gemma4-31b-awq-8bit}"
