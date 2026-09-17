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

An initial codebook is induced from one full training rollout per training task.
These summaries help propose a codebook; subsequent routing still uses complete
histories. Reflection evaluates minibatches of six rollouts from distinct original
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

The run allows up to 100 proposal attempts, with a four-hour optimization
deadline checked between iterations so time remains for exhaustive deployment
and analysis. Actual attempts, accepted candidates, stopping reason, inference
counts, and score changes must be reported from artifacts; a configured ceiling
is not a completed number of updates.

## Historical retention and crossover

Each candidate tracks every training rollout evaluated under that candidate.
After the cheap minibatch screen, a child is evaluated on the union of its
parents' previously evaluated training rollouts. Supported history and recorded
transition sets must be retained by identity, not just by count. Labels may change.

This is an **evolving observed-training ledger**. It does not claim to reroute all
730 training rollouts after every proposal. All 1,030 rollouts are exhaustively
routed with the selected system in the final census. Rejected proposals and their
lost-coverage counterexamples are retained for later training-only reflection.

Every fifth proposal can attempt semantic crossover when complementary
nonancestor Pareto candidates share ancestry. A joint LLM rewrite reconciles
definitions, IDs, and edges. A small Pareto subset screens it against both
parents; no Pareto transcript or textual critique enters reflection. Accepted
crossovers retain both parents' supported training evidence and record both
parents in GEPA's ancestry. If no pair exists, the step remains reflection.

## Exhaustive final graph and held-out interpretation

First, the chosen optimized specification receives its frozen test evaluation.
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
  --proposals 100 --minibatch 6 --optimization-hours 4
```

Successful model responses, per-candidate assignments/evaluations, GEPA
checkpoints, graph versions, and analysis outputs are durable. Rerunning uses
content-addressed caches and the optimizer checkpoint. Runtime credentials and
raw data are ignored by Git. Source code, methods, tests, and concise reports are
versioned. The earlier September 16 feasibility study remains separately
documented and is not evidence for the full-corpus run.
