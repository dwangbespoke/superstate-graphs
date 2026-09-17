# September 16 feasibility study

This is the earlier, small-corpus experiment. Its Qwen-generated rollouts and
results are separate from the full Sonnet 4.5 corpus experiment.
All commands below run from the repository root.

An end-to-end research prototype for turning recurring decisions in an agent's
experience into reusable task-construction targets.

Pipeline: natural rollouts → prefix-only history records → GEPA-evaluated
superstates → observed transition graph → cross-task paths → executable tasks.

The formation objective is coherent decision sharing and useful cross-task
composition. Terminal rewards are attached after formation for descriptive
superstate outcome statistics, not used to assign histories or optimize a value
function. Pooled variance is an encounter-level statistic; it does not prove
each individual history has the same continuation difficulty.

The September 16 pilot is documented in [the results note](OVERNIGHT_RESULTS.md)
and [interactive report](../reports/overnight-poc.html). It includes two executable
critic-assisted examples, with no observed high-variance groups or GEPA validation
gain. Failed automatic generation attempts remain recorded.

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

See [EXPERIMENT.md](EXPERIMENT.md) for the execution plan and interpretation boundaries.

## Run the proof of concept

```bash
uv run python -m superstate_graphs.prepare
SG_MODEL_ROLE=learner uv run modal run scripts/modal_models.py --duration-seconds 10800
# In a second terminal after the runtime endpoint file appears:
uv run python scripts/run_rollouts.py --name pilot-v2
# Bring up the reflector when rollouts are available:
SG_MODEL_ROLE=reflector uv run modal run scripts/modal_models.py --duration-seconds 7200
uv run python scripts/fetch_runtime_database.py
uv run python -m superstate_graphs.analyze \
  --rollouts results/rollouts/pilot-v2 --output results/analysis_v1 \
  --exclude-task dbt-consolidate --max-metric-calls 144
uv run python scripts/generate_dbt_examples.py --analysis results/analysis_v1 \
  --count 3 --max-paths 6 --judge-uncached
# Supported initial-decision fallback, explicitly not sequential graph traversal:
uv run python scripts/export_shared_start_paths.py --analysis results/analysis_v1
uv run python scripts/generate_dbt_examples.py --analysis results/analysis_v1 \
  --paths results/analysis_v1/shared_start_paths.json --count 3 --max-paths 3 --resume
uv run python scripts/build_report.py --analysis results/analysis_v1
uv run python scripts/export_review_bundle.py --analysis results/analysis_v1
```

Endpoint files in `results/runtime/` contain temporary credentials and are ignored
by Git. Stop a server by creating `results/runtime/learner.stop` or
`results/runtime/reflector.stop`; remove that marker explicitly before restarting.

`python scripts/fetch_runtime_database.py` exports the pristine materialized
benchmark database for local task construction. Mutable dbt tasks preserve project
configuration decisions; the optional `generate_examples.py` backend supports
read-only SQL targets. Formats and limitations are documented in `docs/TASK_FORMAT.md`.

The recorded pilot contains six tasks and twelve rollouts. The core analysis
excludes `dbt-consolidate`, which uses separate CSV inputs and a separate database,
leaving five tasks and ten rollouts. This world-scope correction and the observed
failures are documented in `docs/PILOT_AUDIT.md`. All nine graded core outcomes
are failures; one is ungraded. Task construction therefore uses supported groups
as a fallback in this pilot, with no claim of observed high-variance discovery.
