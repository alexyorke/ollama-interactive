# Ollama Code

`ollama-code` is a small local coding CLI that feels closer to tools like aider or OpenCode, but runs against your own Ollama model.

It gives the model a guarded tool loop for:

- listing files
- reading files
- searching the workspace
- searching code symbols, showing compact code outlines, and reading exact function/class bodies
- writing or replacing file content
- running shell commands
- running a configured test command
- inspecting git status and diffs
- creating guarded git commits
- running nested sub-agents for scoped subtasks
- running default-on tool-step assumption audits plus claim-aware risky final verification and evidence-backed rewrite
- running artifact reconciliation after failed tests or validator-like tools

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
  "verifier_model": "granite4.1:8b",
  "approval": "ask",
  "debate": true,
  "reconcile": "auto",
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
- `OLLAMA_HOST`, `OLLAMA_CODE_MODEL`, `OLLAMA_CODE_VERIFIER_MODEL`, `OLLAMA_CODE_TEST_CMD`, `OLLAMA_CODE_DEBATE`, `OLLAMA_CODE_RECONCILE`, and `OLLAMA_CODE_NUM_CTX` override the config file for one-off runs
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

Artifact reconciliation defaults to `auto`. It runs a cheap serial critic after failed tests/validator-like tools to force a focused repair plan before the agent keeps going. Use `off` for fastest literal runs or `on` for stricter failed-edit handling too:

```bash
export OLLAMA_HOST=127.0.0.1:11434
ollama-code --reconcile on
```

Use a stronger serial verifier/rewrite model while keeping the main working model smaller:

```bash
export OLLAMA_HOST=127.0.0.1:11434
ollama-code --model gemma3:4b --verifier-model granite4.1:8b
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
You can set failed-artifact reconciliation with `OLLAMA_CODE_RECONCILE=off|on|auto`.
You can override the verifier/rewrite model with `OLLAMA_CODE_VERIFIER_MODEL`.
The client sends an adaptive `num_ctx` option for normal compact turns so large-context models do not allocate 40K-131K context for tiny prompts. Set `OLLAMA_CODE_NUM_CTX=off` to use the model default, or set an integer such as `8192` to force a fixed context.

Profile local Ollama speed with raw load/prompt/generation counters:

```bash
python scripts/ollama_perf_probe.py --models gemma3:4b qwen3:8b granite4.1:8b --output scratch/perf/ollama.json
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

For serial verification on/off A/B runs that record latency, tool-call sequences, assumption-audit retries, verifier retries, verifier rewrites, and pass/fail:

```bash
export OLLAMA_HOST=127.0.0.1:11434
python3 scripts/verification_eval.py --strict-on
```

To test a stronger verifier model serially:

```bash
export OLLAMA_HOST=127.0.0.1:11434
python3 scripts/verification_eval.py --models gemma3:4b --verifier-model granite4.1:8b --strict-on
```

For serial token-efficiency A/B runs, write raw ignored JSON under `scratch/` and compare against a prior run:

```bash
export OLLAMA_HOST=127.0.0.1:11434
python3 scripts/token_efficiency_eval.py --output scratch/token-efficiency/after.json --compare scratch/token-efficiency/baseline.json --strict-accuracy
```

For realistic coding-task accuracy plus token-profile checks:

```bash
export OLLAMA_HOST=127.0.0.1:11434
python3 scripts/coding_benchmark_eval.py --suite local-small --models gemma3:4b qwen3:8b granite4.1:8b --modes off on --strict-accuracy --strict-budget
```

See [docs/coding-benchmarks.md](docs/coding-benchmarks.md) for the local suites, recorded metrics, and optional external benchmark preflights.

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
- `/reconcile off|on|auto`
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
- Debate mode now means two controller checks with `think=false`: a pre-tool assumption auditor on tool turns, plus claim-aware grounded verification on risky final replies. The auditor can accept or force up to two corrective retries before a tool runs. The verifier now receives extracted candidate claims and a compact evidence table, can return structured claim corrections, and can trigger one evidence-backed rewrite before the controller falls back to another primary-model retry or fails closed. Low-risk finals still skip verification, cached read-only tool repeats skip the auditor, and exact tool-error requests can return directly from the tool result.
- Reconciliation mode is separate from debate. `auto` runs only after failed tests/validator-like tools that matter to the request, `on` also checks failed edits/shell/subagents, and `off` disables it.
- `verifier_model` is optional. When set, it is used for final verification and evidence-backed rewrite only; the primary model still handles the normal tool loop and tool-step assumption audits.
- Explicit forbidden-tool constraints such as `do not use read_file` are enforced before tool execution.
- Tool-heavy turns run with thinking disabled by default to cut latency and token use. Simple non-tool turns still use the normal Ollama thinking path.
- Repeated read-only tool calls in one user turn are cached, and only compact tool-result summaries are fed back into the model. Full raw tool results still stay in the transcript and event log.
- Code navigation tools (`search_symbols`, `code_outline`, `read_symbol`) let the model inspect relevant functions/classes instead of reading full files. `replace_symbol` can mechanically replace one function/class/method by symbol range and rejects Python replacements that would break file syntax. Python uses AST ranges; other common code files use a lightweight definition fallback.
- Token profiling is recorded in `llm_call` events, including prompt chars by role and largest prompt messages, so evals can identify input-token waste.
- You can configure a default test runner with `--test-cmd` or `OLLAMA_CODE_TEST_CMD`, and the model can invoke it through `run_test` or `/test`.
- Nested agents can be started through the `run_agent` tool, with a configurable depth cap.
- The recommended serial eval order is `gemma3:4b`, `qwen3:8b`, `granite4.1:8b`, then `gemma4:e4b`.
- On this machine, the tested serial live baselines are `gemma3:4b` and `qwen3:8b` on the WSL Ollama daemon at `127.0.0.1:11434`. If you do not pass `--model` and the built-in preferred default is not installed, the CLI falls back to the first available preferred local model and reports that choice.
- The Windows-side Ollama install still has no usable local models for this project, so running inside WSL is the simplest path.
