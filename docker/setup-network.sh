#!/bin/bash
NETWORK_NAME="llm-network"

if docker network inspect "$NETWORK_NAME" &>/dev/null; then
  echo "Network '$NETWORK_NAME' already exists."
else
  docker network create "$NETWORK_NAME"
  echo "Network '$NETWORK_NAME' created."
fi
