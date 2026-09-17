# Full-corpus superstate graph formation

## Research question

Can an LLM learn a compact, interpretable graph of reusable local decision
situations from recorded agent trajectories, with explicit operation contracts
that support coherent cross-task composition?

The formation objective is local semantic coherence and reusable transitions.
It is not long-term behavioral equivalence, reward prediction, or a claim that
members have identical continuation difficulty. Task construction and eventual
post-training are downstream experiments.

## Corpus and history boundaries

The input is the audited historical Horizon/Sonnet 4.5 corpus mapped to 103
Data Eng Bench task families: ten trajectories per task, 1,030 total. The exact
delivered historical task query is preserved; current public task instructions
are not substituted. The source corpus has 1,029 verifier outcomes and one
explicitly reviewed failure assignment, retained with its provenance.

There are **36,502 completed action-observation groups and 37,532 histories**,
including one initial query history per rollout. Commands sharing an aggregate
terminal response form one group. Splitting those commands would invent
intermediate observations. Malformed actions followed by parser-error responses
also create a history. Every full prefix is provided to the router; no summary,
embedding, truncation, or representative-history assignment replaces it.

The lossless index stores each transcript once, with prefix offsets and SHA-256
hashes. Model-facing histories contain policy-visible content only. Terminal
rewards, verifier feedback, split labels, and trajectory IDs are not router input.

## Split and evaluation unit

A seeded, reward-independent task-level split assigns 73 original tasks to
training, 15 to adaptive Pareto selection, and 15 to test. All ten rollouts from
each original task stay together:

| Split | Tasks | Rollouts | Histories |
|---|---:|---:|---:|
| Training | 73 | 730 | 27,152 |
| Pareto | 15 | 150 | 5,432 |
| Test | 15 | 150 | 4,948 |

The split prevents exact original-task leakage; it does not establish semantic
independence between related task variants. One complete rollout is one GEPA
example. All prefixes in an evaluated rollout are assigned independently using
only that prefix and the state specification.

## Candidate and optimizer

We use the installed, pinned GEPA package (`gepa==0.1.4`) with its adapter API.
Each candidate has two jointly editable text components: `state_spec` (routing
instructions and state definitions) and `edge_spec` (explicit directed operations).
The routing model never receives the edge table. An edge-only edit can therefore
reuse state assignments exactly; a state-specification change creates a new
assignment cache. Model, prompt, and corpus provenance are part of cache identity.

Routing uses pinned Qwen3.5-35B-A3B-FP8 replicas with reasoning enabled and
an evidence-first response schema. Discovery, reflection, and
semantic judging use Qwen3.5-122B-A10B-FP8 with reasoning enabled. The complete
raw prefix is retained, followed by a repetition of the actual last observation
and the exact parser-supplied completed-group count to anchor the current situation.
Initial production
grounding checks found that schema-constrained non-reasoning outputs could quote
real text while confusing a requested deliverable with an observed outcome.
Rejected initialization outputs are preserved and excluded from the seed graph.

An initial codebook is induced from three explicit source-history/next-observation
pairs in one training rollout per training task. Full source prefixes are supplied;
the program assigns the actual history IDs and checks source, target, and observation
quotes verbatim against the correct sections. Invalid proposals are repaired or
retained as rejected evidence. These descriptions help propose a codebook;
subsequent routing still uses complete histories. Reflection evaluates minibatches of six rollouts from distinct original
tasks and can add, split, merge, remove, or rewrite nodes and edges together.

The frozen rollout evaluator scores prefix membership, recognition of recorded
transitions by existing edges, node coherence, advertised outgoing-edge
applicability, and description complexity. The fixed scalar is:

```
0.25 * supported_history_fraction
+ 0.50 * supported_transition_fraction
+ 0.15 * node_coherence
+ 0.10 * outgoing_edge_applicability
- 0.03 * specification_characters / 100000
```

Semantic verdicts are LLM proxy judgments. They are not manually supplied cluster
labels, executable verification, or mathematical correctness certificates. The
evaluator reports evidence and proposed distinctions for reflection; rewards do
not enter this process. All full Pareto rollouts are evaluated for each accepted
candidate. The final candidate maximizes mean Pareto score.

The run allows up to 100 proposal attempts, with a two-hour optimization
deadline checked between iterations so time remains for exhaustive deployment
and analysis. Actual attempts, accepted candidates, stopping reason, inference
counts, and score changes must be reported from artifacts; a configured ceiling
is not a completed number of updates.

## Historical retention and crossover

The implemented control flow is:

```text
Algorithm 1: Evaluate a Recorded Rollout
Input: frozen candidate C, one complete recorded rollout r
  Independently route every complete prefix using C.state_spec and the full prefix.
  Match each observed action–observation group against existing C.edge_spec edges.
  Obtain fixed semantic membership, transition, coherence, and applicability verdicts.
  Return the fixed scalar score, all identity-indexed verdicts, and textual evidence.

Algorithm 2: Evolve a Graph with Historical Retention
  Induce a grounded seed from training data; evaluate it on all Pareto rollouts.
  Until the proposal or time budget is reached:
    Select a parent using GEPA's per-example Pareto coverage strategy.
    Sample six training rollouts from distinct original tasks.
    Evaluate the parent and collect training-only reflective evidence.
    Propose an atomic joint state/edge patch, optionally reconciling a second parent.
    Evaluate the child on the same minibatch; apply the improvement screen.
    Reevaluate the union of parents' previously encountered training rollouts.
    Reject if any previously non-null classified history becomes unassigned.
    Otherwise evaluate every Pareto rollout and add the child to GEPA's archive.
  Select the archived candidate with the highest mean Pareto score.

Algorithm 3: Complete and Audit the Corpus Graph
  Evaluate the selected frozen candidate on test tasks and save those scores.
  Route every corpus prefix; create explicit additional definitions for residual nulls.
  Retain every observed adjacent-prefix transition with exact witness identities.
  Propose bounded operation contracts and independently audit sampled applicability.
  Estimate descriptive reward variances with trajectory/task dependence preserved.
  Construct grounded task examples and execute those with supported local fixtures.
```

One rollout evaluation is one GEPA task. The minibatch average aggregates those
per-rollout scores; it is not a second scoring rule. Textual feedback supplies
evidence to the proposer while numerical scores govern selection and acceptance.

Each candidate tracks every training rollout evaluated under that candidate.
After the cheap minibatch screen, a child is evaluated on the union of its
parents' previously evaluated training rollouts. Every history assigned a non-null
state by either parent must remain assigned a non-null state, by history identity
rather than count. Labels may change, including reassignment after node splitting,
merging, or removal. This gate applies even if the evaluator previously judged the
assignment unsupported. Semantic membership and transition judgments remain in the
fixed score; their earlier Boolean verdicts are not additional hard constraints.

This is an **evolving observed-training ledger**. It does not claim to reroute all
730 training rollouts after every proposal. All 1,030 rollouts are exhaustively
routed with the selected system in the final census. Rejected proposals and their
lost-coverage counterexamples are retained for later training-only reflection.

Every fifth proposal can attempt semantic crossover when complementary
nonancestor Pareto candidates share ancestry. A joint LLM rewrite reconciles
definitions, IDs, and edges. A small Pareto subset screens it against both
parents; no Pareto transcript or textual critique enters reflection. Accepted
crossovers retain both parents' non-null training-history coverage and record both
parents in GEPA's ancestry. If no pair exists, the step remains reflection.

Before the first production proposal, the retention gate was corrected from an
unexercised design that would have hard-retained positive membership and transition
judgments. The declared correction matches the classification-coverage requirement;
it was not selected from proposal or heldout outcomes. The fixed evaluator, score,
minibatch/crossover screens, and Pareto selection were unchanged. No empirical
comparison or superiority over the earlier design is claimed.

## Exhaustive final graph and held-out interpretation

First, the chosen optimized specification and the initial seed receive frozen
evaluations on the same test rollouts. Their paired score differences are summarized
by original task, with a task-cluster bootstrap interval; neither test feedback
nor test scores affect prompt updates or candidate selection.
Then every one of the 37,532 corpus histories is routed. Residual uncovered
situations can receive explicit additional definitions in a separately labeled
**all-corpus, transductive completion stage**. Frozen earlier routing stages retain
their assignments; only null assignments reach new stages. Every residual
history still receives its own full-prefix LLM call. No generic `other` state or
silent forced label completes the census.

The final graph preserves all observed adjacent-prefix witnesses. Observed
quotient edges are distinguished from proposed reusable operation contracts.
An observed A-to-B transition does not establish that every history in A admits
the advertised operation. Independent sampled source-applicability checks mark
support, contradictions, and unknowns separately; finite LLM checks cannot prove
the universal-source/existential-target contract for unseen histories.

Test scores describe the frozen pre-completion system. They must never be
presented as held-out performance of the graph subsequently expanded using all
corpus splits.

## Independent checks and outcome statistics

Post-formation checks use separate frozen prompts and blinded evidence:
cross-task within-state coherence, between-state redundancy, and edge
applicability on source histories that did not take the proposed edge. Exact
evidence references are checked programmatically. Same-model judges are labeled
as such; prompt separation is not model independence.

Per-state outcome summaries include raw history-weighted variance,
trajectory-deduplicated variance, equal-task-weighted variance, task/trajectory
counts, task-cluster bootstrap uncertainty, and sensitivity excluding the one
manually assigned outcome. Repeated histories from the same rollout are not
independent reward samples. For binary outcomes, population variance is p(1-p).
These are pooled descriptive statistics, not estimates proving equal conditional
value or stochastic difficulty at every member history.

Grounded path-based task examples preserve transition witnesses and entity
requirements. A generated task and a model feasibility review are clearly labeled
as drafts/proxy checks until an environment actually executes and verifies them.
No post-training lift is claimed by this graph-construction run.

## Reproduction

```bash
uv sync --extra dev
uv run python -m superstate_graphs.full_corpus \
  --dataset-root ../data/data_eng_bench_sonnet45 --output results/full_graph/corpus

# Start bounded authenticated Modal servers; see MODEL_SERVING.md.
uv run python -m superstate_graphs.full_graph \
  --dataset ../data/data_eng_bench_sonnet45 \
  --output results/full_graph/run_v1 \
  --runtime results/runtime/graph.json \
  --runtime results/runtime/graph-two.json \
  --runtime results/runtime/graph-three.json \
  --runtime results/runtime/graph-four.json \
  --runtime results/runtime/graph-five.json \
  --runtime results/runtime/graph-six.json \
  --runtime results/runtime/graph-seven.json \
  --runtime results/runtime/graph-eight.json \
  --teacher-runtime results/runtime/graph-teacher.json \
  --teacher-runtime results/runtime/graph-teacher-two.json \
  --proposals 100 --minibatch 6 --optimization-hours 2
```

Successful model responses, per-candidate assignments/evaluations, GEPA
checkpoints, graph versions, and analysis outputs are durable. Rerunning uses
content-addressed caches and the optimizer checkpoint. Runtime credentials and
raw data are ignored by Git. Source code, methods, tests, and concise reports are
versioned. The earlier September 16 feasibility study remains separately
documented and is not evidence for the full-corpus run.

## Report and publication handoff

The report builder can run during inference; missing artifacts remain explicitly
partial. It never starts inference or changes the research run:

```bash
uv run python scripts/build_full_graph_report.py \
  --run results/full_graph/run_v1 --corpus results/full_graph/corpus \
  --output reports/full-corpus-2026-09-17
```

After the run writes `completion.json`, publish the full research artifacts to a
new output directory. Include the executed tasks explicitly:

```bash
uv run python scripts/export_full_graph_artifacts.py \
  --run results/full_graph/run_v1 --corpus results/full_graph/corpus \
  --output reports/full-corpus-2026-09-17/artifacts --include-tasks
```

The exporter refuses incomplete results or replacement of an existing export.
It verifies all 103 original task IDs, 1,030 rollout IDs, 37,532 history
assignments, and 36,502 transition witnesses against the immutable corpus
indexes. Every witness must match its recorded prefix endpoints and their final
superstates. Variance must cover every occupied state with exact membership
counts; the exporter recomputes the three descriptive reward weightings and
manual-grade sensitivity from immutable terminal outcomes. Unoccupied state
definitions have no empirical reward estimate.

The compressed JSONL artifacts retain every membership and witness ID, while
the compact HTML report displays only a labeled subset of witness references.
The export includes graph definitions, routing stages, full splits, variance,
paired frozen held-out aggregates, and a SHA-256 manifest. Raw transcripts,
assignment evidence, judge prompts, and runtime credentials remain local.

Each exported executable example includes its instruction, DuckDB fixture,
reference queries, expected result, execution receipt, and validated links to
the exact graph edges and recorded source witnesses. Verify a submitted query:

```bash
uv run python -m superstate_graphs.graph_task_examples verify \
  --task-dir reports/full-corpus-2026-09-17/artifacts/executable_tasks/PATH_ID \
  --submission answer.sql
```

Execution means two independently prompted reference queries ran and agreed on
a synthetic fixture. It does not establish learner success or execution of the
composed original histories. The report separately records whether the requested
number of examples was reached. A completed artifact workflow may still report
exhausted supported paths, failed semantic requests, or contradicted edges.

Only the reviewed report/export directory should be staged for publication.
The repository ignores `*.duckdb`, so include the sanitized exported fixtures
explicitly when at least one executed example exists:

```bash
git add reports/full-corpus-2026-09-17
git add -f reports/full-corpus-2026-09-17/artifacts/executable_tasks/*/fixture.duckdb
```

Do not stage the unrestricted `results/` directory. Retain the exported manifest
with its files so a downloaded copy can be checked against the local final export.

### Usage across operational restarts

The report and `usage_accounting.json` distinguish cumulative retained-cache
evidence from process-local counters. Cache usage is deduplicated by request key
across the current and archived `llm_cache` directories. This includes earlier
initialization and rejected candidates that share those caches; cache metadata
cannot reliably attribute every request to the selected graph. A successful
cached request means a completed JSON response, not scientific acceptance.

Two overlapping views are shown: the last successful response for each key, and
all saved response attempts in each successful request chain, including output
retries. The final response is included once in the latter view. Older records
without an attempt ledger contribute only their saved final response. Conflicting
copies of one key are flagged and excluded from token totals.
Transport-error entries in a saved attempt ledger are counted separately; they
are not returned model responses and add no reported tokens. This count covers
only failures recorded within retained successful request chains, not every
transport failure experienced during the experiment.

Current/final and archived process counters are displayed separately and never
added to the cache totals or summed across snapshots. This prevents both counting
retained requests twice after a restart and presenting a restarted process's
counters as the whole experiment. Calls without a retained successful cache
record, overwritten/deleted records, and diagnostics outside the run directory
remain incompletely measured. Endpoint-reported token totals are not measured
hardware work or invoices; no exact spend is inferred.
