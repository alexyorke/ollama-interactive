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
- running default-on tool-step assumption audits plus risky final verification

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

You do not need to activate a virtualenv every time just to run the CLI from this repo. These launchers will prefer `.venv` if it exists and otherwise fall back to your normal Python on `PATH`:

```powershell
.\start.ps1
```

```bash
./start.sh
```

You can also run it directly from the checkout without installing a console script:

```bash
python -m ollama_code.cli
```

## Config File

Create `.ollama-code/config.json` in your workspace to keep the app defaults in one place instead of repeating flags or env vars. This path is already ignored by git.

```json
{
  "host": "http://127.0.0.1:11434",
  "model": "batiai/qwen3.6-35b:iq4",
  "approval": "ask",
  "debate": true,
  "max_tool_rounds": 100,
  "max_agent_depth": 2,
  "timeout": 300,
  "test_cmd": "python -m unittest -v"
}
```

You can also point at a custom file:

```bash
ollama-code --config ~/my-ollama.json
```

Precedence:

- CLI flags override everything else
- when resuming a saved session, the saved model wins unless `--model` is provided
- `OLLAMA_HOST`, `OLLAMA_CODE_MODEL`, `OLLAMA_CODE_TEST_CMD`, and `OLLAMA_CODE_DEBATE` override the config file for one-off runs
- otherwise the CLI falls back to `.ollama-code/config.json`, then the built-in defaults

## Run

Interactive REPL:

```bash
export OLLAMA_HOST=127.0.0.1:11434
ollama-code
```

One-shot prompt:

```bash
export OLLAMA_HOST=127.0.0.1:11434
ollama-code --approval auto "Inspect this repo and explain what it does."
```

Continue the most recent local session in the current workspace:

```bash
export OLLAMA_HOST=127.0.0.1:11434
ollama-code --continue
```

Run with a default test command so the model can use `run_test` and `/test` directly:

```bash
export OLLAMA_HOST=127.0.0.1:11434
ollama-code --test-cmd "python3 -m unittest discover -s tests -v"
```

Debate mode is on by default. It now means tool-step assumption auditing plus grounded verification for risky final answers. Low-risk replies skip verification automatically. Disable it entirely for the fastest or most literal runs:

```bash
export OLLAMA_HOST=127.0.0.1:11434
ollama-code --debate off
```

Resume a specific saved transcript:

```bash
export OLLAMA_HOST=127.0.0.1:11434
ollama-code --resume .ollama-code/sessions/20260421-120000-abcdef.json
```

In the interactive REPL, press `Esc` to interrupt the current model request or tool subprocess.

If your Ollama daemon is not on the default host, set `OLLAMA_HOST`:

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
```

You can also set the default test command with `OLLAMA_CODE_TEST_CMD`.
You can disable assumption auditing and grounded verification with `OLLAMA_CODE_DEBATE=off`.

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

By default `compose.yaml` points the container at `http://host.docker.internal:11434`, which matches the tested WSL Ollama daemon on this machine. Override `OLLAMA_CODE_OLLAMA_HOST` if your daemon is somewhere else:

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
export OLLAMA_HOST=127.0.0.1:11434
python3 scripts/live_matrix.py
```

The default serial smoke matrix tries the installed baseline models first, then optional Granite and Gemma eval targets:

- `gemma3:4b`
- `qwen3:8b`
- `granite4.1:8b`
- `gemma4:e4b`

If your Ollama daemon is on a non-default host or port, set `OLLAMA_HOST` first. Example for the tested WSL daemon on `127.0.0.1:11434`:

```bash
export OLLAMA_HOST=127.0.0.1:11434
python3 scripts/live_matrix.py --models gemma3:4b qwen3:8b
```

For stricter transcript-verified end-to-end checks against one model:

```bash
export OLLAMA_HOST=127.0.0.1:11434
python3 scripts/e2e_suite.py
```

For serial verification on/off A/B runs that record latency, tool-call sequences, assumption-audit retries, verifier retries, and pass/fail:

```bash
export OLLAMA_HOST=127.0.0.1:11434
python3 scripts/verification_eval.py --strict-on
```

Containerized end-to-end check against the host Ollama daemon:

```bash
OLLAMA_CODE_OLLAMA_HOST=http://host.docker.internal:11434 docker compose run --rm --entrypoint python ollama-code scripts/e2e_suite.py
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
- `/debate on|off`
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
- Debate mode now means two controller checks on the same model with `think=false`: a pre-tool assumption auditor on tool turns, plus grounded verification on risky final replies. The auditor can accept or force up to two corrective retries before a tool runs. The verifier can accept a risky final or force up to two corrective retries. Low-risk finals still skip verification, cached read-only tool repeats skip the auditor, and exact tool-error requests can return directly from the tool result.
- Explicit forbidden-tool constraints such as `do not use read_file` are enforced before tool execution.
- Tool-heavy turns run with thinking disabled by default to cut latency and token use. Simple non-tool turns still use the normal Ollama thinking path.
- Repeated read-only tool calls in one user turn are cached, and only compact tool-result summaries are fed back into the model. Full raw tool results still stay in the transcript and event log.
- You can configure a default test runner with `--test-cmd` or `OLLAMA_CODE_TEST_CMD`, and the model can invoke it through `run_test` or `/test`.
- Nested agents can be started through the `run_agent` tool, with a configurable depth cap.
- The recommended serial eval order is `gemma3:4b`, `qwen3:8b`, `granite4.1:8b`, then `gemma4:e4b`.
- On this machine, the tested serial live baselines are `gemma3:4b` and `qwen3:8b` on the WSL Ollama daemon at `127.0.0.1:11434`. If you do not pass `--model` and the built-in preferred default is not installed, the CLI falls back to the first available preferred local model and reports that choice.
- The Windows-side Ollama install still has no usable local models for this project, so running inside WSL is the simplest path.
