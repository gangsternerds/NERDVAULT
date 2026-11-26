#!/usr/bin/env bash
# smoke-test.sh: quick 3-check verification of a running openset-llm server (/health, /v1/models, /v1/chat/completions).
#
#   bash scripts/smoke-test.sh                       # checks http://127.0.0.1:8080
#   bash scripts/smoke-test.sh http://127.0.0.1:9000 # custom URL
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8080}"
BASE_URL="${BASE_URL%/}"

case "${1:-}" in -h|--help) sed -n '2,5p' "$0"; exit 0 ;; esac
command -v curl >/dev/null 2>&1 || { echo "error: curl is required" >&2; exit 1; }

echo "==> 1/3 Checking $BASE_URL/health"
curl -fsS "$BASE_URL/health" >/dev/null
echo "    OK"

echo "==> 2/3 Checking $BASE_URL/v1/models"
curl -fsS "$BASE_URL/v1/models" | grep -q '"data"'
echo "    OK"

echo "==> 3/3 Running short completion on $BASE_URL/v1/chat/completions"
START_MS=$(date +%s%3N 2>/dev/null || echo 0)
RESP=$(curl -fsS "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"openset-llm","max_tokens":16,"temperature":0,"messages":[{"role":"user","content":"Reply with the single word: ready"}]}')
END_MS=$(date +%s%3N 2>/dev/null || echo 0)

echo "$RESP" | grep -q '"choices"' || { echo "error: unexpected response: $RESP" >&2; exit 1; }
if [ "$START_MS" != 0 ] && [ "$END_MS" != 0 ]; then
  echo "    OK ($((END_MS - START_MS)) ms)"
else
  echo "    OK"
fi
echo "All smoke checks passed for $BASE_URL"
