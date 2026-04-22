#!/usr/bin/env bash
set -euo pipefail

workspace="${OLLAMA_CODE_WORKSPACE:-/workspace}"
model="${OLLAMA_CODE_MODEL:-batiai/gemma4-26b:iq4}"

mkdir -p "$workspace"

if [[ $# -gt 0 ]]; then
  case "$1" in
    bash|sh|/bin/bash|/bin/sh|python|python3)
      exec "$@"
      ;;
    ollama-code)
      shift
      ;;
  esac
fi

exec ollama-code --cwd "$workspace" --model "$model" "$@"
