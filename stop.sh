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

cd "$SCRIPT_DIR"

echo "Stopping head on spark1..."
docker compose down || true

echo "Stopping worker on ${WORKER_HOST}..."
ssh "${WORKER_HOST}" "cd '${SCRIPT_DIR}' && docker compose down" || true

echo "Qwen3.8 27B stopped."
