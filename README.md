# Ollama Code

`ollama-code` is a small local coding CLI that feels closer to tools like aider or OpenCode, but runs against your own Ollama model.

It gives the model a guarded tool loop for:

- listing files
- reading files
- searching the workspace
- writing or replacing file content
- running shell commands
- running nested sub-agents for scoped subtasks

By default it asks before edits or shell commands. You can switch to `--approval auto` for hands-off runs.

## Install

```bash
python -m pip install -e .
```

Inside WSL:

```bash
python3 -m pip install -e .
```

## Run

Interactive REPL:

```bash
export OLLAMA_HOST=127.0.0.1:11435
ollama-code
```

One-shot prompt:

```bash
export OLLAMA_HOST=127.0.0.1:11435
ollama-code --approval auto "Inspect this repo and explain what it does."
```

If your Ollama daemon is not on the default host, set `OLLAMA_HOST`:

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
```

## Docker

Build the image:

```bash
docker build -t ollama-code:local .
```

Run the unit test suite inside the image:

```bash
docker run --rm --entrypoint python ollama-code:local -m unittest discover -s tests -v
```

Run the REPL in Docker with the current checkout mounted as the workspace:

```bash
docker compose run --rm ollama-code
```

Run a one-shot prompt:

```bash
docker compose run --rm ollama-code --approval auto "Inspect this repo and summarize it."
```

By default `compose.yaml` points the container at `http://host.docker.internal:11435`, which matches the tested user-space WSL Ollama daemon on this machine. Override `OLLAMA_CODE_OLLAMA_HOST` if your daemon is somewhere else:

```bash
OLLAMA_CODE_OLLAMA_HOST=http://host.docker.internal:11434 docker compose run --rm ollama-code
```

If you want a shell inside the container instead of launching the CLI, pass `bash`:

```bash
docker compose run --rm ollama-code bash
```

## WSL + tmux

From WSL in the repo root:

```bash
./scripts/start-wsl-tmux.sh
```

From PowerShell:

```powershell
.\scripts\start-wsl-tmux.ps1
```

## Live Validation

Run the real-model smoke matrix from WSL:

```bash
export OLLAMA_HOST=127.0.0.1:11435
python3 scripts/live_matrix.py
```

If your Ollama daemon is on a non-default host or port, set `OLLAMA_HOST` first. Example for a user-space WSL daemon on `127.0.0.1:11435`:

```bash
export OLLAMA_HOST=127.0.0.1:11435
python3 scripts/live_matrix.py --models batiai/gemma4-26b:iq4 gemma4
```

For stricter transcript-verified end-to-end checks against one model:

```bash
export OLLAMA_HOST=127.0.0.1:11435
python3 scripts/e2e_suite.py
```

Containerized end-to-end check against the host Ollama daemon:

```bash
OLLAMA_CODE_OLLAMA_HOST=http://host.docker.internal:11435 docker compose run --rm --entrypoint python ollama-code scripts/e2e_suite.py
```

## GitHub Actions

GitHub Actions runs the portable checks on hosted runners:

- install the package
- run the unit test suite on Ubuntu and Windows across Python 3.10, 3.11, and 3.12
- build the Docker image and run the unit test suite inside the container
- build the wheel and source distribution

Tagged releases with `v*` also publish the built `dist/` artifacts to a GitHub release.

Example:

```bash
git tag v0.1.0
git push origin v0.1.0
```

## Slash Commands

- `/help`
- `/status`
- `/models`
- `/model <name>`
- `/approval ask|auto|read-only`
- `/reset`
- `/save [path]`
- `/tools`
- `/quit`

## Notes

- The CLI is designed to run inside the current workspace root.
- File mutations are limited to that workspace.
- Nested agents can be started through the `run_agent` tool, with a configurable depth cap.
- On this machine, the tested default model is `batiai/gemma4-26b:iq4` on the user-space WSL Ollama daemon at `127.0.0.1:11435`.
- The Windows-side Ollama install still has no usable local models for this project, so running inside WSL is the simplest path.
