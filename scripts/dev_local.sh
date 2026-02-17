#!/usr/bin/env bash
set -euo pipefail

uvicorn slocum_api.main:app --reload --app-dir apps/api --port 8000 &
API_PID=$!

npm --prefix apps/web run dev -- --host 127.0.0.1 --port 5173 &
WEB_PID=$!

cleanup() {
  kill "$API_PID" "$WEB_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

wait "$API_PID" "$WEB_PID"
