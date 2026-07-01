# Ollama Code

`ollama-code` is a small local coding CLI that feels closer to tools like aider or OpenCode, but runs against your own Ollama model.

It gives the model a guarded tool loop for:

- listing files
- reading files
- searching the workspace with `rg`, optional `fd`, and a repo-local background SQLite/FTS index
- searching code symbols, showing compact code outlines, and reading exact function/class bodies
- indexing and searching repo-local verified function cards for reusable Python utilities
- writing or replacing file content
- running shell commands
- running a configured test command
- inspecting git status and diffs
- creating guarded git commits
- tracking in-session todos for complex multi-step work
- running nested sub-agents for scoped subtasks
- running default-on tool-step assumption audits plus claim-aware risky final verification and evidence-backed rewrite
- running artifact reconciliation after failed tests or validator-like tools
- using optional local integrations such as ast-grep, Semgrep, LSP-style navigation, MCP stdio servers, Playwright smoke checks, and security scanners when installed

By default it asks before edits or shell commands. You can switch to `--approval auto` for hands-off runs.
Sessions are also auto-saved locally under `.ollama-code/sessions`, so you can continue or resume prior work.

## Install

Runtime requirements:

- Python 3.10+
- Ollama running somewhere reachable by `OLLAMA_HOST`
- `git` for git-aware status/diff/commit helpers
- `ripgrep` (`rg`) recommended for fast search; the CLI has slower fallbacks when it is missing

The Python package intentionally has no required third-party Python dependencies. Optional adapters such as tree-sitter, `ast-grep`, `rg`, `fd`, `jq`, `yq`, `uv`, Ruff, pytest helpers, mypy, Pyright/basedpyright, deptry, Vulture, Hypothesis, Python profilers, TypeScript/ESLint/Prettier/Biome, `difftastic`, actionlint, ShellCheck, Hadolint, OSV-Scanner, Semgrep/Opengrep, SCIP indexers, OPA, Inspect AI, `sqlite-vec`, Comby, Phoenix, Ctags, Mergiraf, Playwright, and language servers are detected at runtime and fail closed with install guidance when missing.

Fresh Ubuntu/Debian quickstart:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv git ripgrep ca-certificates
git clone https://github.com/alexyorke/ollama-interactive.git
cd ollama-interactive
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
ollama-code --help
ollama-code --doctor
```

If Ollama is not installed or no model is pulled yet, `--doctor` will report that and print the default pull hint. Install Ollama separately, start the daemon, then pull the default model:

```bash
ollama pull granite4.1:8b
```

Vagrant is not needed by the CLI itself. To test inside a Vagrant VM, install Vagrant plus a provider such as VirtualBox or Hyper-V on the host first, then run the same Linux quickstart inside the guest.

Editable install in an already prepared Python environment:

```bash
python -m pip install -e .
```

Optional local tooling can be installed as needed. `--doctor` reports installed/missing integrations, `/tools missing` shows install hints, and `/tools install <tool-id>` prompts before running the exact installer command. Nothing is installed silently, including in `--approval auto` mode.

The recommended Python-side tooling bundle can be installed manually with:

```bash
python -m pip install -e ".[tools]"
```

External CLIs such as `ast-grep`, `rg`, `fd`, `jq`, `yq`, TypeScript/ESLint/Prettier/Biome, ShellCheck, Hadolint, `difftastic`, OPA, and OSV-Scanner remain external package-manager installs. Use `/tools install --recommended` in an interactive session to review and approve available install commands for the current platform.

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
python -m ollama_code
```

## Config File

Create `.ollama-code/config.json` in your workspace to keep the app defaults in one place instead of repeating flags or env vars. This path is already ignored by git.

```json
{
  "host": "http://127.0.0.1:11434",
  "model": "granite4.1:8b",
  "verifier_model": "granite4.1:8b",
  "approval": "ask",
  "debate": true,
  "reconcile": "auto",
  "max_tool_rounds": 100,
  "max_agent_depth": 2,
  "timeout": 300,
  "test_cmd": "python -m unittest discover -s tests -v",
  "tools": {
    "default_enabled": true,
    "disabled": []
  },
  "indexer": {
    "enabled": true,
    "watch": true,
    "poll_interval_ms": 5000
  }
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
- if the built-in default is not installed, runtime fallback only chooses a known preferred local model; pass `--model` for custom `hf.co/...` or vendor tags

## Run

Interactive REPL:

```bash
export OLLAMA_HOST=127.0.0.1:11434
ollama-code
```

The built-in default model is `granite4.1:8b`. Install it once if needed:

```bash
ollama pull granite4.1:8b
```

Check first-use setup before asking it to edit code:

```bash
python -m ollama_code --doctor
```

Run the fast local readiness tier before broader edits or benchmarks:

```bash
python scripts/local_validation.py --tier smoke
```

`smoke` and `agent` automatically use `pytest` with a bounded `xdist` worker count when those optional packages are installed; otherwise they fall back to serial `unittest`.
The JSON summary records the resolved runner, resolved worker mode, completed tiers, and any tiers skipped after a failure so the local gate reflects what actually ran.
For `pytest` runs it also records a `coverage_summary` block proving whether the repo-owned validation plan covered the discovered test targets exactly once, with zero duplicates and zero uncovered files.

Run the fuller local validation stack before merging larger controller or tooling changes:

```bash
python scripts/local_validation.py --tier full
```

When `pytest` is installed, `full` uses the same bounded-worker `pytest` path for the broad final repo pass and skips the `smoke` and `agent` targets it already ran, instead of rerunning them inside a broad `pytest tests` command. Environments without `pytest` still fall back to the older `unittest` path.
Treat `coverage_summary.full_plan_covers_all_discovered_targets=true` as the readiness invariant for the `full` tier. If that flag is false, `scripts/local_validation.py` exits nonzero even if the subprocesses themselves were green, because the validation plan has drifted.
The console output now also prints the slowest validation step and a compact coverage line so contributors do not need to open the JSON artifact just to confirm timing and partition completeness.

If local test runs feel slow and you want an apples-to-apples baseline, compare the preferred path against raw `unittest` discovery:

```bash
python scripts/local_validation.py --tier full --compare-unittest-baseline
```

That writes the normal JSON summary plus an opt-in `baseline_compare` block so you can see whether the slowdown is runner choice or a real suite regression.

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
ollama-code --model granite4.1:8b --verifier-model granite4.1:8b
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
The client sends `temperature=0` by default for reproducible local runs. It also sends an adaptive `num_ctx` option for normal compact turns so large-context models do not allocate 40K-131K context for tiny prompts. Set `OLLAMA_CODE_NUM_CTX=off` to use the model default, or set an integer such as `8192` to force a fixed context.

The background indexer is on by default. It keeps `.ollama-code/index` warm for `file_search`, `repo_index_search`, `indexed_search`, `fts_search`, and `verified_function_search`; it never executes project code and can be disabled with `"indexer": {"enabled": false}` or `--no-indexer`.

The verified function library stores Python cards in `.ollama-code/index/verified_functions.sqlite`. Retrieval finds candidate utilities only; trust is labeled as `verified`, `probable`, or `unverified` from source hashes, purity hints, extracted examples, probes, and focused tests.

Profile local Ollama speed with raw load/prompt/generation counters:

```bash
python scripts/ollama_perf_probe.py --models granite4.1:8b gemma4:e4b qwen3:8b --output scratch/perf/ollama.json
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

### Nightly improvement report

Always-on self-hosting is not part of the core CLI path. The legacy watchdog/task files are archived under `archive/experimental-self-host/` for reference only.

Use the gated report harness to measure product changes before proposing implementation work:

```bash
python scripts/nightly_self_improvement_report.py --models granite4.1:8b --strict-accuracy --strict-budget
```

For a cheap local baseline that skips model-driven evals but still records tool speed, SDK retrieval, question quality, dataset-catalog status, and compare-path behavior:

```bash
python scripts/nightly_self_improvement_report.py --skip-llm --generated-files 100 --strict-accuracy
```

The report writes `scratch/nightly-self-improvement/<timestamp>/report.json` with pass/fail deltas, token totals, tool latency, slowest tools, clarification-question quality, local trajectory analysis, dataset-catalog status, and suggested implementation targets. When `--compare` is omitted it auto-selects the latest prior report from either `scratch/nightly-self-improvement/` or the legacy `.ollama-code/self-improvement-runs/` location. It also records `runtime.llm_skip_reason` and `runtime.trajectory_skip_reason` so skipped sections are explicit in the JSON.

Trajectory profile, error, and evidence commands only run when local datasets exist under `scratch/external/datasets/` (or a custom `--trajectory-data-root`). The harness does not edit, merge, push, or run a background worker.

Bootstrap the local trajectory corpus with:

```bash
python scripts/trajectory_dataset_fetch.py --datasets nebius-swe-agent-trajectories
```

The fetcher downloads only the supported public Parquet globs for each requested dataset, writes a per-dataset manifest at `scratch/external/datasets/<dataset>/.ollama-interactive-manifest.json`, and records a summary at `scratch/external/datasets/trajectory-dataset-fetch.json`. Treat those local files and the generated JSON under `scratch/` as regenerated evidence, not versioned source data.

The supported fetch set includes Agent Race, Trace Commons, Thoughtworks, SWE-agent, OpenHands, SWE-smith, SWE-Hero, Open-SWE, CoderForge SWE-bench Verified, CC-Bench, and TerminalBench trajectory corpora. The catalog also tracks manual-review datasets such as `NJU-LINK/CodeTraceBench`, `AlienKevin/SWE-ZERO-12M-trajectories`, `badlogicgames/pi-mono`, `thomasmustier/pi-mono-sessions`, `thomasmustier/pi-nes-sessions`, `nmuendler/share-codex`, `peteromallet/my-personal-codex-data`, `misterkerns/my-personal-claude-code-data`, `ultralazr/claude-code-traces`, and `Glint-Research/Fable-5-traces`, plus gated sets such as `SWE-chat`. Use `python scripts/trajectory_dataset_catalog.py` to see which corpora are local, public-missing, or gated, and see `docs/trajectory-profiling.md` for current evidence artifact paths.

For the web-discovered real-user session corpora, regenerate the bounded local review with:

```bash
python scripts/web_discovered_agent_dataset_analysis.py
```

The current web-discovered pass pulls bounded local samples from the Pi-family public traces such as `pi-mono`, `pi-mono-sessions`, `pi-sessions-viewer`, `gradio-pi-sessions`, and `Prayagmatic/agent-traces`, plus `share-codex`, `peteromallet/my-personal-codex-data`, `misterkerns/my-personal-claude-code-data`, raw Codex Desktop session exports such as `nielsr/add-sam-3-lite-text-agent-traces`, and larger parquet-backed slices such as `nlile/misc-merged-claude-code-traces-v1`, then summarizes task families, tool usage, prompt examples, and where available token totals. The newer DataClaw exports are useful because they expose direct `input_tokens`, `output_tokens`, and `tool_uses` counts for real coding-agent sessions, which makes context blow-up per tool loop measurable instead of anecdotal. The merged Claude Code parquet adds a larger public real-user slice, but only after filtering helper rows and reading `<tool_use>` blocks from `assistant_response` instead of mistaking `tools_json` for executed calls. The raw Codex Desktop exports are useful for measuring first-turn instruction bloat because they preserve the developer wrapper and executed `function_call` events directly from `.codex/sessions`.

### Python SDK Retrieval

For Python API/stdlib questions, the CLI exposes an installed-SDK index instead of relying on model memory:

```bash
python scripts/python_sdk_search_eval.py --strict-accuracy
```

The `python_sdk_search` tool builds `.ollama-code/index/python_sdk.sqlite` from the current interpreter's stdlib signatures and docstrings. It uses SQLite FTS by default and can optionally rerank the top lexical candidates with cached or on-demand local Ollama embeddings, avoiding a full stdlib embedding precompute:

```bash
python scripts/python_sdk_search_eval.py --use-embeddings --embedding-model qwen3-embedding:8b
```

To make CLI tool calls use SDK embeddings automatically, set:

```bash
OLLAMA_CODE_SDK_EMBED_MODEL=qwen3-embedding:8b
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

- `granite4.1:8b`
- `gemma4:e4b`
- `qwen3:8b`

If your Ollama daemon is on a non-default host or port, set `OLLAMA_HOST` first. Example for the tested WSL daemon on `127.0.0.1:11434`:

```bash
export OLLAMA_HOST=127.0.0.1:11434
python3 scripts/live_matrix.py --models granite4.1:8b gemma4:e4b qwen3:8b
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
python3 scripts/verification_eval.py --models gemma4:e4b --verifier-model granite4.1:8b --strict-on
```

For serial token-efficiency A/B runs, write raw ignored JSON under `scratch/` and compare against a prior run:

```bash
export OLLAMA_HOST=127.0.0.1:11434
python3 scripts/token_efficiency_eval.py --output scratch/token-efficiency/after.json --compare scratch/token-efficiency/baseline.json --strict-accuracy
```

For realistic coding-task accuracy plus token-profile checks:

```bash
export OLLAMA_HOST=127.0.0.1:11434
python3 scripts/coding_benchmark_eval.py --suite local-small --models granite4.1:8b gemma4:e4b qwen3:8b --modes off on --strict-accuracy --strict-budget
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
- `/doctor`
- `/reset`
- `/todos [clear]`
- `/save [path]`
- `/sessions [limit]`
- `/load <path>`
- `/git`
- `/diff [--cached] [path]`
- `/commit <message>`
- `/test [command]`
- `/tools`
- `/tools missing`
- `/tools install <tool-id>|--recommended`
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
- `inspect_library_source` lets the model inspect installed Python library functions/classes by import path, with signature/doc/disassembly fallback for builtins or C extensions where source is unavailable.
- `systems_lens` gives the model compact systems questions for broad debugging, design, refactor, and performance tasks: boundary, observer/metric, categories, state/scale, feedback, delays, stocks/flows, coupling, model limits, and intervention effects. It guides planning but does not count as file evidence before edits.
- Token profiling is recorded in `llm_call` events, including prompt chars by role and largest prompt messages, so evals can identify input-token waste.
- You can configure a default test runner with `--test-cmd` or `OLLAMA_CODE_TEST_CMD`, and the model can invoke it through `run_test` or `/test`.
- Nested agents can be started through the `run_agent` tool, with a configurable depth cap.
- `todo_read` and `todo_write` give the model a Claude Code-style in-session checklist for complex tasks. Todo state is saved in session transcripts, does not touch workspace files, and is shown to the model only when the current request benefits from it.
- The recommended default coding model is `granite4.1:8b`; install it with `ollama pull granite4.1:8b`.
- On the latest July 1, 2026 serial live gate, `granite4.1:8b`, `gemma4:e4b`, and `qwen3:8b` all passed `local-small`, and Granite stayed the default because it used the fewest benchmark tokens (`2049` vs `2436` and `2532`).
- `scratch/live-model-gate/live-model-gate-summary.json` is the canonical release artifact for a specific source state. `scripts/live_model_gate.py` mirrors the latest run into that fixed path even when detailed run outputs live under a different scratch directory, and it records `git_commit`, `git_dirty`, `benchmark_suite`, `selected_default_model`, `selection_reason`, and per-model gate rows so the chosen default does not need to be reconstructed from separate benchmark files or guessed against a different checkout.
- The recommended serial eval order is `granite4.1:8b`, `gemma4:e4b`, then `qwen3:8b`.
- If you do not pass `--model` and the default Granite tag is not installed, the CLI falls back only to a known preferred local model and prints the pull command. Custom `hf.co/...` or vendor tags are never selected implicitly; pass `--model` for those.
