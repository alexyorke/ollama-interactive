# Trajectory Profiling

`scripts/trajectory_profile.py` profiles downloaded coding-agent trajectory datasets to find workflow patterns that waste tokens and turns.

Supported local datasets under ignored `scratch/external/datasets/`:

- `nebius-swe-agent-trajectories`
- `nebius-swe-rebench-openhands-trajectories`
- `swe-smith-trajectories`

`SWE-chat` is intentionally not assumed to be present because the Hugging Face dataset is gated.

## Usage

Profile a manageable sample from every supported dataset:

```bash
python scripts/trajectory_profile.py --max-rows 2000 --output scratch/external/datasets/trajectory-profile.json
```

Profile one dataset only:

```bash
python scripts/trajectory_profile.py --datasets nebius-swe-rebench-openhands-trajectories --max-rows 5000
```

## Output

The script writes raw JSON and prints a concise summary. Per dataset it reports:

- average tool calls per trajectory
- tool-category mix (`read`, `search`, `edit`, `test`, `shell`, `git`, `other`)
- rows with long read/search loops before progress
- rows with edits but no later test
- rows that edit before enough context
- average and median tool steps before first edit
- top tools and repeated-tool loop signatures
- structured recommendations for controller/tooling changes
- a merged portfolio recommendation list across all profiled datasets

## Why it exists

This profiler is meant to answer questions like:

- Do agents burn too many steps on repeated reads and searches before editing?
- Do they mutate files before gathering enough context?
- How often do they fail to validate edits with tests?
- Which tools dominate trajectories, and which loops should be short-circuited or made deterministic?

Those outputs are intended to drive generic CLI improvements such as:

- symbol-first navigation
- compact context planning
- post-edit validation gates
- failure compression
- loop caps and controller intervention

## Recommendation Layer

Each recommendation now includes:

- `id`: stable key for grouping across datasets
- `priority`: `high`, `medium`, or `low`
- `change_type`: `controller`, `guard`, `tooling`, or `analysis`
- `trigger`: the metric threshold that fired
- `rationale`: why the profiler thinks this change matters
- `expected_effect`: what should improve if the change works
- `experiments`: concrete A/B or benchmark ideas to validate it

The JSON also includes `portfolio_recommendations`, which merges repeated recommendation IDs across datasets so the most general improvements rise to the top.

The first controller implementation is the `trajectory-guards` feature profile. It uses generic signals only: repeated context tools, missing grounding before mutation, missing validation after edits, and repeated failed-test output.

## Error Profile

`scripts/trajectory_error_profile.py` scans only tool results and observations where the dataset exposes them. It avoids task descriptions so benchmark text does not inflate error counts.

```bash
python scripts/trajectory_error_profile.py --output scratch/external/datasets/trajectory-error-profile.json
```

It classifies result failures into generic buckets:

- `test_assertion`
- `missing_dependency`
- `import_error`
- `syntax_error`
- `invalid_args`
- `command_not_found`
- `path_missing`
- `cwd_git`
- `timeout`
- `permission`
- `patch_apply`

Mapped prevention policies:

- `diagnose_test_failure` now emits `error_class`, `missing_dependency`, `next_tool`, expected/actual hints, and likely target files.
- `run_shell` preflights common command families (`git`, Python test runners, `pytest`, `npm`/`pnpm`/`yarn` test commands, `ruff`, `mypy`, `pyright`, `tsc`) and rejects dangerous or malformed recognized commands before execution.
- `run_shell` runs `bash -n -c <command>` when bash is available and the command is not clearly PowerShell, rejecting shell syntax errors before execution.
- Unknown commands that pass shell syntax checking are still rejected before execution when the first executable cannot be resolved and is not a shell/cmd builtin.
- Valid recognized commands run as argv with `shell=False`; unknown commands keep the legacy shell path.
- `trajectory-guards` blocks a third identical tool/error-class failure and forces path discovery, syntax repair, dependency fail-closed, or another non-repeating next step.
- Path and cwd failures include nearest existing path suggestions when available.

Latest full local run over downloaded supported datasets:

```text
nebius-swe-agent-trajectories: rows=80036 results=2035588 errors=572350
  test_assertion=166074 syntax_error=146301 command_not_found=72121 invalid_args=46324 path_missing=43823 import_error=43224 timeout=35951 missing_dependency=18135
nebius-swe-rebench-openhands-trajectories: rows=67074 results=4249707 errors=373045
  test_assertion=170744 timeout=74787 invalid_args=34645 missing_dependency=30681 import_error=28785 syntax_error=15757 path_missing=15201 command_not_found=1564
swe-smith-trajectories: rows=102078 results=269242 errors=21980
  test_assertion=9044 import_error=4593 timeout=2341 syntax_error=2141 path_missing=1838 invalid_args=1720 missing_dependency=176 command_not_found=81
portfolio: diagnose-test-failure=345862 syntax-repair-gate=164199 validated-cli-preflight=156455 dependency-or-import-guard=125594 bounded-command-validation=113079 path-repair-guard=60869
```

These counts are directional because public trajectory formats differ, but the top fixes are stable across datasets: better test-failure diagnosis, syntax repair gates, validated CLI execution, dependency/import fail-closed behavior, bounded command handling, and path repair.

Shell-specific full-run counts:

```text
nebius-swe-agent-trajectories:
  other_shell_error=236418 command_not_found=32827 missing_file_or_dir=18475 unrecognized_argument=7232 bash_unexpected_token=6281 bash_unexpected_eof=915
nebius-swe-rebench-openhands-trajectories:
  missing_file_or_dir=12592 bash_unexpected_token=6413 other_shell_error=4689 unrecognized_argument=1811 command_not_found=1187 permission_denied=386
swe-smith-trajectories:
  missing_file_or_dir=1323 other_shell_error=738 unrecognized_argument=287 command_not_found=77 bash_unexpected_token=27 permission_denied=25
```

Mapped shell prevention:

- `bash_unexpected_token`, `bash_unexpected_eof`, and `unmatched_quote`: reject with `bash -n`/quote parsing before execution.
- `command_not_found`: reject missing unknown first executables before execution.
- `missing_file_or_dir`: return nearest path suggestions and block repeated same-path failures under `trajectory-guards`.
- `unrecognized_argument`: keep family-specific argv allowlists for common validators and test runners.
- `permission_denied`: fail closed instead of retrying or changing permissions automatically.
- common observed mistake: Python code pasted directly into bash (`def ...`, `print(module.attr)`) is caught by `bash -n`; the model should use `python -c`, `run_function_probe`, or `run_test` instead.

Validation after implementing the generic guards:

```text
python -m unittest discover -v
  301 passed, 3 skipped
scripts/token_efficiency_eval.py --strict-accuracy
  all default cases passed
scripts/verification_eval.py --strict-on
  all strict-on cases passed
scripts/e2e_suite.py --model gemma3:4b
  all scenarios passed
scripts/coding_benchmark_eval.py --suite local-small --models granite4.1:8b --feature-profiles trajectory-guards --strict-accuracy
  8/8 passed
```

On the local-small Granite run, `trajectory-guards` reduced the live model-token total on key non-zero-LLM cases versus the same-session baseline run: `issue_fix_hidden_tests` 6044 -> 4625 tokens, `multi_file_refactor` 18859 fail-closed -> 11458 pass, and `multi_turn_session_task` 4824 -> 4809 tokens. A separate `all` profile run was non-deterministic on `multi_file_refactor`, so promote `trajectory-guards` independently before expanding defaults.
