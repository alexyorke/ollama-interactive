#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if [[ -x "$repo_root/.venv/bin/python" ]]; then
  exec "$repo_root/.venv/bin/python" -m ollama_code.cli "$@"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 -m ollama_code.cli "$@"
fi

exec python -m ollama_code.cli "$@"
