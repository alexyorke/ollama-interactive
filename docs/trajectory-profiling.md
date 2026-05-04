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
