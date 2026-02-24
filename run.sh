#!/usr/bin/env bash
#
# Convenience launcher for Coding Agent.
# Usage: ./run.sh [--list-models | -l]   (use -l to list available LLM models and exit)
#

set -euo pipefail

# Load .env if present (LLM_URL, MODEL, DB_*, etc.)
if [ -f "$(dirname "$0")/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$(dirname "$0")/.env"
  set +a
fi

# ---- Configuration ------------------------------------------------------------
# Path to your repository
REPO_PATH="${REPO_PATH:-$(pwd)}"

# LLM settings (from .env or default)
LLM_URL="${LLM_URL:-http://localhost:11434}"
MODEL="${MODEL:-codellama}"
SOURCE_IP="${SOURCE_IP:-}"  # Optional: bind to local IP to bypass VPN (e.g. SOURCE_IP=10.152.50.103)

# Git settings
BASE_BRANCH="${BASE_BRANCH:-main}"
BRANCH_PREFIX="${BRANCH_PREFIX:-agent/}"

# Agent settings
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
VERBOSE="${VERBOSE:-false}"

# Path to coding_agent.py
AGENT_SCRIPT="${AGENT_SCRIPT:-$(dirname "$0")/coding_agent.py}"

# ---- List models (--list-models / -l) -----------------------------------------
if [ "${1:-}" = "--list-models" ] || [ "${1:-}" = "-l" ]; then
  echo "Listing models at $LLM_URL"
  echo ""
  python3 -c "
import urllib.request
import json
import sys

url = sys.argv[1].rstrip('/')

def try_openai_compat():
    '''LM Studio and other OpenAI-compatible servers: GET /v1/models'''
    try:
        r = urllib.request.urlopen(url + '/v1/models', timeout=10)
        d = json.load(r)
        if isinstance(d.get('data'), list):
            for m in d['data']:
                print(m.get('id', str(m)))
            return 'openai'
    except Exception:
        pass
    return None

def try_ollama():
    '''Ollama: GET /api/tags'''
    try:
        r = urllib.request.urlopen(url + '/api/tags', timeout=10)
        d = json.load(r)
        if isinstance(d.get('models'), list):
            for m in d['models']:
                print(m.get('name', m.get('model', m)))
            return 'ollama'
    except Exception:
        pass
    return None

# Try OpenAI-compat first so we never hit /api/tags on LM Studio (it returns 200 + error)
kind = try_openai_compat()
if kind is None:
    kind = try_ollama()

if kind is None:
    print('Could not list models: server did not respond as Ollama (/api/tags) or OpenAI-compat (/v1/models).', file=sys.stderr)
    print('Check LLM_URL and that the server is running.', file=sys.stderr)
    sys.exit(1)

print(f'Detected: {kind}', file=sys.stderr)
" "$LLM_URL"
  exit 0
fi

# ---- Build command ------------------------------------------------------------
CMD="python \"$AGENT_SCRIPT\" \
  --repo \"$REPO_PATH\" \
  --llm-url \"$LLM_URL\" \
  --model \"$MODEL\" \
  --base-branch \"$BASE_BRANCH\" \
  --branch-prefix \"$BRANCH_PREFIX\" \
  --max-iterations $MAX_ITERATIONS"

if [ -n "${SOURCE_IP:-}" ]; then
  CMD="$CMD --source-ip \"$SOURCE_IP\""
fi

if [ "$VERBOSE" = "true" ]; then
  CMD="$CMD --verbose"
fi

# ---- Run --------------------------------------------------------------------
echo "Starting Coding Agent..."
echo "Repository: $REPO_PATH"
echo "Model: $MODEL"
[ -n "${SOURCE_IP:-}" ] && echo "Source IP: $SOURCE_IP"
echo ""

eval $CMD
