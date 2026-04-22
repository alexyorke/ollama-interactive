# Ollama Code

`ollama-code` is a small local coding CLI that feels closer to tools like aider or OpenCode, but runs against your own Ollama model.

It gives the model a guarded tool loop for:

- listing files
- reading files
- searching the workspace
- writing or replacing file content
- running shell commands
- running a configured test command
- inspecting git status and diffs
- creating guarded git commits
- running nested sub-agents for scoped subtasks

By default it asks before edits or shell commands. You can switch to `--approval auto` for hands-off runs.
Sessions are also auto-saved locally under `.ollama-code/sessions`, so you can continue or resume prior work.

## Install

```bash
python -m pip install -e .
```

Inside WSL:

```bash
python3 -m pip install -e .
```

## Config File

Create `.ollama-code/config.json` in your workspace to set the local Ollama endpoint and default model without repeating flags or env vars. This path is already ignored by git.

```json
{
  "host": "http://127.0.0.1:11435",
  "model": "batiai/gemma4-26b:iq4"
}
```

You can also point at a custom file:

```bash
ollama-code --config ~/my-ollama.json
```

Precedence:

- `--host` and `--model` override everything else
- when resuming a saved session, the saved model wins unless `--model` is provided
- `OLLAMA_HOST` and `OLLAMA_CODE_MODEL` override the config file for one-off runs
- otherwise the CLI falls back to `.ollama-code/config.json`, then the built-in defaults

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

Continue the most recent local session in the current workspace:

```bash
export OLLAMA_HOST=127.0.0.1:11435
ollama-code --continue
```

Run with a default test command so the model can use `run_test` and `/test` directly:

```bash
export OLLAMA_HOST=127.0.0.1:11435
ollama-code --test-cmd "python3 -m unittest discover -s tests -v"
```

Resume a specific saved transcript:

```bash
export OLLAMA_HOST=127.0.0.1:11435
ollama-code --resume .ollama-code/sessions/20260421-120000-abcdef.json
```

If your Ollama daemon is not on the default host, set `OLLAMA_HOST`:

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
```

You can also set the default test command with `OLLAMA_CODE_TEST_CMD`.

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

Continue the latest saved session in Docker:

```bash
docker compose run --rm ollama-code --continue
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
- `/sessions [limit]`
- `/load <path>`
- `/git`
- `/diff [--cached] [path]`
- `/commit <message>`
- `/test [command]`
- `/tools`
- `/quit`

## Notes

- The CLI is designed to run inside the current workspace root.
- File mutations are limited to that workspace.
- Session history is auto-saved under `.ollama-code/sessions` in the workspace.
- You can configure a default test runner with `--test-cmd` or `OLLAMA_CODE_TEST_CMD`, and the model can invoke it through `run_test` or `/test`.
- Nested agents can be started through the `run_agent` tool, with a configurable depth cap.
- On this machine, the tested default model is `batiai/gemma4-26b:iq4` on the user-space WSL Ollama daemon at `127.0.0.1:11435`.
- The Windows-side Ollama install still has no usable local models for this project, so running inside WSL is the simplest path.
