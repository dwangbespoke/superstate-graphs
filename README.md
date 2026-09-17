# Superstate Graphs

Experience-grounded graphs of local decision situations, learned from complete
agent histories with GEPA. The graph supplies explicit directed operation
contracts for downstream task construction. Terminal outcomes are attached only
after formation to estimate the reward variance within each superstate.

The full experiment uses **1,030 archived Sonnet 4.5 trajectories from 103 Data
Eng Bench task families**, with ten trajectories per task. The lossless corpus
contains **37,532 histories and 36,502 action–observation groups**. Every history
includes the original query and a complete prefix; commands sharing one aggregate
observation advance the history once. No embedding, summary, or sampled
representative replaces a history during final assignment.

## Method

1. Split original tasks into 73 training, 15 Pareto-selection, and 15 test tasks.
2. Induce an initial state and edge specification from grounded training examples.
3. Use GEPA to jointly revise both specifications. One rollout is one evaluation
   example; each minibatch contains six rollouts from different original tasks.
   Children must retain all previously classified training history IDs.
4. Compare the selected frozen candidate with the seed on the untouched test
   tasks, then classify the entire corpus. Add explicit definitions for residual
   uncovered histories in a separately labeled transductive completion stage.
5. Preserve every observed transition, audit proposed reusable contracts, estimate
   outcome variance, and construct small executable tasks from supported paths.

Routing uses **Qwen3.5-35B-A3B-FP8**; graph discovery, reflection, semantic judging,
and task proposals use **Qwen3.5-122B-A10B-FP8**. Model revisions, full request
identities, accepted graph versions, and operational continuation receipts are
recorded. This is a graph-specific application of GEPA with historical retention
and optional graph reconciliation; it is not a claim to have invented GEPA.

- [Full method and pseudocode](docs/FULL_CORPUS_METHOD.md)
- [Model serving and inference provenance](docs/MODEL_SERVING.md)
- [Research claims and limitations](docs/RESEARCH_CLAIMS.md)
- [Qualitative review of the selected frozen graph](docs/FROZEN_GRAPH_REVIEW.md)
- [Related work](docs/RELATED_WORK.md)
- [Documented reflection-evidence repair](docs/REFLECTION_EVIDENCE_CORRECTION.md)

## Frozen specification evaluation

GEPA produced 17 proposals within the time-bounded run; three revisions entered
its archive alongside the seed. No crossover proposal occurred. The selected
candidate has 28 states and 34 edges. Its state specification is unchanged from
the seed; selection changed only the edge specification. Held-out routing
assignments are also identical. Differences in membership or coherence scores
therefore reflect separate full-graph judge responses, not revised assignments.

The frozen comparison uses the same 150 held-out rollouts from 15 original tasks:

| LLM proxy metric | Seed | Selected |
|---|---:|---:|
| Fixed composite score | 0.4249 | 0.4429 |
| Supported transition fraction | 12.35% | 14.46% |

The composite difference is **+0.0180**, with a paired original-task bootstrap
95% interval of **[+0.0055, +0.0302]**. This interval conditions on the fixed
specifications and one cached set of LLM judgments; it excludes generation and
optimization uncertainty. The gain is a graph-proxy result. It does not establish
an improved state partition, universal applicability, or learner improvement.
The [qualitative review](docs/FROZEN_GRAPH_REVIEW.md) records redundant additions
and concrete contract conflicts. All-corpus completion and reconstructed contracts
are separate from this frozen test comparison.

## Reproduction

Python 3.12 and an authenticated Modal account are required for inference.
The historical source trajectories are supplied separately; the current public
benchmark is not a replacement for those archived instructions and observations.

```bash
uv sync --extra dev
uv run python -m superstate_graphs.full_corpus \
  --dataset-root ../data/data_eng_bench_sonnet45 --output results/full_graph/corpus
```

Launch the router and teacher as described in [model serving](docs/MODEL_SERVING.md),
wait for their authenticated readiness markers, then run:

```bash
uv run python -m superstate_graphs.full_graph \
  --dataset ../data/data_eng_bench_sonnet45 \
  --output results/full_graph/run_v1 \
  --runtime results/runtime/graph.json \
  --teacher-runtime results/runtime/graph-teacher.json \
  --proposals 100 --minibatch 6 --optimization-hours 2
```

Repeat `--runtime` and `--teacher-runtime` to use multiple identical replicas.
The recorded production fleet has eight routers and two teachers. The proposal
count is a ceiling, and the optimization deadline is checked between iterations.
Existing checkpoints resume only when their corpus, candidate, inference, and
optimizer identities match. Raw inputs, inference caches, runtime credentials,
and local work products are ignored by Git.

```bash
uv run pytest -q
uv run python scripts/verify_frozen_graph_comparison.py \
  --run results/full_graph/run_v1 --corpus results/full_graph/corpus \
  --output results/full_graph/run_v1/verification/frozen_heldout_verification.json
uv run python scripts/build_full_graph_report.py
uv run python scripts/export_full_graph_artifacts.py --include-tasks
```

Export requires an exact completed history and transition census; it also validates
the selected candidate against GEPA's accepted archive and verifies executable
task receipts. See the [publication handoff](docs/FULL_CORPUS_METHOD.md#report-and-publication-handoff)
for artifact locations and integrity checks.

## Interpretation

Observed transitions establish that a progression occurred in the corpus. They
do not establish that its operation applies to every member of the source state.
Reusable contracts receive separate sampled semantic checks, with counterexamples
and unknown judgments retained. These checks use a separate prompt but the same
Qwen model family. Successful local task execution is additional, narrower evidence.

Reward variance is descriptive. Repeated visits from a trajectory are correlated,
so the analysis reports history-weighted, unique-trajectory, and equal-task
statistics, with task-level bootstrap intervals. A high pooled variance does not
establish identical long-term decision problems or training value. No downstream
post-training lift has been measured.

## Earlier feasibility study

The September 16 pilot used a separate, small Qwen-generated rollout collection.
Its [reproduction instructions](docs/PILOT_REPRODUCTION.md),
[results note](docs/OVERNIGHT_RESULTS.md), and [interactive report](reports/overnight-poc.html)
remain available. Its measurements must not be combined with the full-corpus run.
