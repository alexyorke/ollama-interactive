#!/usr/bin/env bash
set -euo pipefail

session="${1:-ollama-code}"
model="${2:-${OLLAMA_CODE_MODEL:-batiai/gemma4-26b:iq4}}"
workspace="${3:-$(pwd)}"
python_bin="${PYTHON_BIN:-python3}"
ollama_host="${OLLAMA_HOST:-127.0.0.1:11435}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required." >&2
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama is required." >&2
  exit 1
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "$python_bin is required." >&2
  exit 1
fi

tmux has-session -t "$session" 2>/dev/null || tmux new-session -d -s "$session" \
  "cd \"$workspace\" && export OLLAMA_HOST=\"$ollama_host\" && $python_bin -m ollama_code.cli --model \"$model\""

exec tmux attach -t "$session"
