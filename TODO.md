# TODO

Current roadmap:

- [ ] Keep `scripts/nightly_self_improvement_report.py` green and use `scratch/nightly-self-improvement/*/report.json` as the primary measurement artifact.
- [ ] Reduce the slowest no-LLM probe cost, starting with `lint_typecheck`, then re-measure with the nightly report.
- [ ] Add at least one local trajectory dataset under `scratch/external/datasets/` so trajectory profile, error, and evidence reports become active inputs instead of catalog-only metadata.
- [ ] Keep `local-small` benchmark coverage green for `gemma4:e4b`, with explicit attention on multi-file refactor behavior whenever controller logic changes.
- [ ] Preserve optional-tool fail-closed behavior and install guidance for `actionlint`, `ShellCheck`, `tree-sitter`, `ast-grep`, and `Semgrep`-compatible scans.
