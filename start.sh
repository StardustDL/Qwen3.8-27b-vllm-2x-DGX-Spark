#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

: "${WORKER_HOST:?WORKER_HOST must be set in .env}"

API_URL="http://${MASTER_ADDR}:8000/v1/models"

cd "$SCRIPT_DIR"

echo "Starting worker on ${WORKER_HOST}..."
scp "$SCRIPT_DIR/.env" "${WORKER_HOST}:${SCRIPT_DIR}/.env"
scp "$SCRIPT_DIR/docker-compose.yml" "${WORKER_HOST}:${SCRIPT_DIR}/docker-compose.yml"
scp "$SCRIPT_DIR/start.sh" "${WORKER_HOST}:${SCRIPT_DIR}/start.sh"
scp "$SCRIPT_DIR/stop.sh" "${WORKER_HOST}:${SCRIPT_DIR}/stop.sh"
ssh "${WORKER_HOST}" "cd '${SCRIPT_DIR}' && source .env && export NODE_RANK=1 && export HEADLESS=1 && docker compose up -d"

echo "Starting head on spark1..."
docker compose up -d

echo "Waiting for vLLM API..."
for _ in $(seq 1 80); do
  if curl -fsS --max-time 5 "$API_URL" >/dev/null; then
    echo "Qwen3.8 27B is running: $API_URL"
    docker compose ps
    ssh "${WORKER_HOST}" "cd '${SCRIPT_DIR}' && docker compose ps"
    exit 0
  fi
  sleep 15
done

CONTAINER_NAME=qwen38-27b-vllm-2x-dgx-spark-vllm-1

echo "Timed out waiting for API. Recent spark1 logs:"
docker logs --tail=120 $CONTAINER_NAME 2>&1 || true
echo "Recent spark2 logs:"
ssh "${WORKER_HOST}" "docker logs --tail=120 $CONTAINER_NAME 2>&1" || true
exit 1
