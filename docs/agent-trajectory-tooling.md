# Agent Trajectory Tooling Notes

This note records the evidence used to expand the optional OSS tool registry. The goal is not to install every tool by default. The CLI should detect useful tools, explain the failure mode they address, and prompt before installing anything.

## Evidence Reviewed

- Local `scratch/external/datasets/trajectory-profile-all-20260516.json` over all locally available rows: 80,036 SWE-agent rows, 67,074 OpenHands rows, and 102,078 SWE-smith rows.
- Local `scratch/external/datasets/trajectory-error-profile-full-20260516.json` over the same all-row corpus.
- Local `.ollama-code/sessions` summary: 209 usable sessions, 5,940 LLM calls, and tool failures led by `run_shell` timeouts, invalid test commands, missing dependencies, and test assertions.
- Public dataset cards and papers for SWE-chat, SWE-rebench/OpenHands trajectories, SWE-rebench V2, and AgentLens.

## Recurring Failure Modes

- Context-only loops: repeated search/read actions before edit or validation.
- Edits without validation in some scaffolds.
- Test assertion failures that need compact, structured diagnosis.
- Syntax errors after edits.
- Missing commands, missing imports, invalid shell args, and path errors.
- Long-running or hung tests and benchmarks.
- Multilingual/config-heavy repos where Python-only validation is insufficient.

## Tooling Added To The Optional Registry

- Search/data plumbing: `ripgrep`, `fd`, `jq`, `yq`.
- Environment/dependency setup: `uv`, `deptry`, `pipdeptree`.
- Python test diagnosis: `pytest`, `pytest-json-report`, `pytest-timeout`, `pytest-xdist`, `pytest-cov`, `coverage.py`, `pytest-testmon`.
- Python correctness/profiling: Ruff, mypy, Pyright, basedpyright, Vulture, Hypothesis, `py-spy`, Scalene, `hyperfine`.
- Repo-native validation: `pre-commit`, tox, nox.
- JS/TS/config validation: TypeScript, ESLint, Prettier, Biome, Stylelint, Taplo, `yamllint`, `check-jsonschema`, `markdownlint-cli2`, `codespell`.
- Language-specific validation: `golangci-lint`, `cargo-nextest`, SQLFluff.
- Security/dependency scanning: OSV-Scanner, `pip-audit`, Gitleaks, Trivy, Grype.
- Previously planned structural/safety/eval tools remain: tree-sitter, ast-grep, difftastic, actionlint, ShellCheck, Hadolint, Semgrep/Opengrep, SCIP, OPA, Inspect AI, sqlite-vec, Comby, Phoenix, Ctags, Mergiraf, and GitHub CLI.

## Tool Awareness Shape

The model-facing tool prompt is grouped by capability family: planning, navigation, structural, validation, editing, verified-functions, security, git, runtime, and tooling. This keeps exact tool signatures available without presenting an undifferentiated flat list.

## Product Rule

These are optional integrations. `ollama-code --doctor`, `/tools missing`, and `/tools install <tool-id>` expose them. The agent must not silently mutate the user environment; installation requires an explicit interactive confirmation and uses the exact command shown to the user. Inspect AI stays in a separate optional `eval` extra because current Inspect releases can require dependency versions that conflict with local runtime stacks such as LiteLLM. Semgrep stays in a separate optional `security` extra for the same reason: current Semgrep releases can pull `jsonschema` versions that conflict with local LiteLLM pins.

Python-based CLI tooling should prefer isolated installs. The registry uses `mode=isolated-venv` for conflict-prone tools and installs them under `.ollama-code/tool-envs/<tool-id>` through `python -m ollama_code.tool_dependencies install-venv ...`. Shared `pip --user` hints remain fallback-only for cases where the user explicitly wants to manage the host environment themselves. Container hints use `mode=docker` and are intended for heavy tools such as Semgrep where a Docker image is safer than mutating the app interpreter.

Remote Docker uses Docker's SSH transport, not an exposed Docker TCP socket. Set `OLLAMA_CODE_DOCKER_HOST=ssh://car-detection-server` or any other SSH Docker host alias before launching `ollama-code`. Container-backed tools copy the local target into a temporary remote container with `docker cp`, run the tool there, and remove the container, which avoids the broken pattern of mounting a local Windows path into a remote Linux daemon.
