# Superstate Graphs

An end-to-end research prototype for turning recurring decisions in an agent's
experience into reusable task-construction targets.

Pipeline: natural rollouts → prefix-only history records → GEPA-refined
superstates → observed transition graph → cross-task paths → executable tasks.

The formation objective is coherent decision sharing and useful cross-task
composition. Terminal rewards are attached after formation for descriptive
superstate outcome statistics, not used to assign histories or optimize a value
function. Pooled variance is an encounter-level statistic; it does not prove
each individual history has the same continuation difficulty.

## Initial experiment

- Environment: Snowflake-Labs/data-eng-bench, DuckDB shared retail warehouse.
- Source revision: `a3278ad102829a6084dde086244a0ef665a8011c`.
- Rollout learner and frozen classifier: Qwen3.5-9B.
- GEPA reflective proposer: Qwen3.5-35B-A3B-FP8.
- Initial target: a small real rollout collection, several evaluated prompt
  revisions, a graph with cross-task witnesses, and 2–3 constructed task examples.
- This is a feasibility study, not a benchmark leaderboard submission or a
  demonstrated training improvement.

## Setup

```bash
uv sync --extra dev
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Snowflake-Labs/data-eng-bench.git vendor/data-eng-bench
git -C vendor/data-eng-bench checkout a3278ad102829a6084dde086244a0ef665a8011c
uv run python scripts/fetch_database.py
```

Code, prompts, configuration, and concise reports are versioned. Credentials,
benchmark copies, databases, raw trajectories, and large artifacts remain outside
Git. Downloaded benchmark material retains its upstream license.

See `docs/EXPERIMENT.md` for the execution plan and interpretation boundaries.
