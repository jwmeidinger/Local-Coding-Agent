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
MAX_PROMPT_CHARS="${MAX_PROMPT_CHARS:-80000}"
MAX_TOOL_RESULT_CHARS="${MAX_TOOL_RESULT_CHARS:-3000}"
MAX_CONSECUTIVE_ERRORS="${MAX_CONSECUTIVE_ERRORS:-2}"
VERBOSE="${VERBOSE:-false}"

# Build verification
BUILD_COMMAND="${BUILD_COMMAND:-}"        # e.g. "pnpm run typecheck", "tsc --noEmit", "make check"
NO_VERIFY="${NO_VERIFY:-false}"          # Set to "true" to disable auto-verify after edits

# Enable verbose mode with -v flag
if [ "${1:-}" = "-v" ] || [ "${1:-}" = "--verbose" ]; then
  VERBOSE="true"
  shift
fi

# Path to coding_agent.py
AGENT_SCRIPT="${AGENT_SCRIPT:-$(dirname "$0")/coding_agent.py}"

# Python interpreter selection:
# 1) PYTHON_BIN override
# 2) project-local virtualenv
# 3) python3 from PATH
# 4) python from PATH
if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON="$PYTHON_BIN"
elif [ -x "$(dirname "$0")/.venv/bin/python" ]; then
  PYTHON="$(dirname "$0")/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
fi

# ---- List models (--list-models / -l) -----------------------------------------
if [ "${1:-}" = "--list-models" ] || [ "${1:-}" = "-l" ]; then
  echo "Listing models at $LLM_URL"
  [ -n "${SOURCE_IP:-}" ] && echo "Binding to source IP: $SOURCE_IP"
  echo ""
  "$PYTHON" -c "
import urllib.request
import json
import socket
import sys

url = sys.argv[1].rstrip('/')
source_ip = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else ''

# ---- Monkey-patch socket.create_connection to bind to source IP ----
# This forces ALL urllib (and http.client) traffic through the specified
# network interface, bypassing VPN default routes.
if source_ip:
    _orig_create_connection = socket.create_connection

    def _bound_create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
                                  source_address=None, **kwargs):
        # Force source_address to our chosen IP if not already set
        if source_address is None:
            source_address = (source_ip, 0)
        return _orig_create_connection(address, timeout, source_address, **kwargs)

    socket.create_connection = _bound_create_connection
    print(f'Bound outgoing connections to {source_ip}', file=sys.stderr)

def try_openai_compat():
    '''LM Studio and other OpenAI-compatible servers: GET /v1/models'''
    try:
        r = urllib.request.urlopen(url + '/v1/models', timeout=10)
        d = json.load(r)
        if isinstance(d.get('data'), list):
            for m in d['data']:
                print(m.get('id', str(m)))
            return 'openai'
    except Exception as e:
        print(f'OpenAI-compat probe failed: {e}', file=sys.stderr)
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
    except Exception as e:
        print(f'Ollama probe failed: {e}', file=sys.stderr)
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
" "$LLM_URL" "${SOURCE_IP:-}"
  exit 0
fi

# ---- Build command ------------------------------------------------------------
CMD=(
  "$PYTHON" "$AGENT_SCRIPT"
  --repo "$REPO_PATH"
  --llm-url "$LLM_URL"
  --model "$MODEL"
  --base-branch "$BASE_BRANCH"
  --branch-prefix "$BRANCH_PREFIX"
  --max-iterations "$MAX_ITERATIONS"
  --max-prompt-chars "$MAX_PROMPT_CHARS"
  --max-tool-result-chars "$MAX_TOOL_RESULT_CHARS"
  --max-consecutive-errors "$MAX_CONSECUTIVE_ERRORS"
)

if [ -n "${SOURCE_IP:-}" ]; then
  CMD+=(--source-ip "$SOURCE_IP")
fi

if [ -n "${BUILD_COMMAND:-}" ]; then
  CMD+=(--build-command "$BUILD_COMMAND")
fi

if [ "$NO_VERIFY" = "true" ]; then
  CMD+=(--no-verify)
fi

if [ "$VERBOSE" = "true" ]; then
  CMD+=(--verbose)
fi

# ---- Pass through extra arguments (fixes --index, --search, etc.) -------------
# This makes ./run.sh --index actually reach coding_agent.py and index the correct repo
CMD+=("$@")
# -------------------------------------------------------------------------------

# ---- Run --------------------------------------------------------------------
echo "Starting Coding Agent..."
echo "Repository: $REPO_PATH"
echo "Model: $MODEL"
[ -n "${SOURCE_IP:-}" ] && echo "Source IP: $SOURCE_IP"
[ -n "${BUILD_COMMAND:-}" ] && echo "Build command: $BUILD_COMMAND"
[ "$NO_VERIFY" = "true" ] && echo "Auto-verify: DISABLED"
[ "$VERBOSE" = "true" ] && echo "Verbose mode: ENABLED"
echo ""
echo "Log files:"
echo "  - .coding-agent/agent.log   (main agent log)"
echo "  - .coding-agent/llm.log    (LLM requests/responses)"
echo ""

"${CMD[@]}"