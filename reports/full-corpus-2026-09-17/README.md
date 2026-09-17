# Full-corpus superstate graph

**Status: COMPLETE.** Artifact and census completion; not a statement that all semantic checks passed.

Snapshot: 2026-09-17T19:28:51.105310+00:00. Router: Qwen/Qwen3.5-35B-A3B-FP8. Teacher: Qwen/Qwen3.5-122B-A10B-FP8.

| Measurement | Observed value |
|---|---|
| Original tasks | 103 |
| Rollouts | 1,030 |
| Complete history prefixes | 37,532 |
| Recorded transitions | 36,502 |
| GEPA proposed revisions | 17 |
| GEPA accepted revisions | 3 |
| Initial Pareto mean score | 0.4258 |
| Best Pareto mean score | 0.4327 |
| Frozen held-out mean score | 0.4429 |
| Assignment records | 37,532 |
| Missing history IDs | 0 |
| Explicitly unassigned histories | 0 |
| Superstates | 39 |
| Edges in available graph | 550 |
| Edges eligible under sampled applicability checks | 1 |
| Audit result records (including request errors) | 1,776 |
| Audit checks without request errors | 1,776 |
| Planned audit checks without result records | 0 |
| Independent audit failed requests | 0 |
| Locally executed consistent synthetic tasks | 0 |
| Executable task target reached | No |
| Executable construction errors | 0 |
| Executable stage failed requests (when recorded) | Not available |

## Optimizer continuation disclosure

1 optimizer continuation receipt(s) declare a method change during optimization. Boundaries, source hashes, and retained-checkpoint hashes are recorded below.

- After proposal 7; resumed 2026-09-17T15:34:36.612597+00:00: Corrected an off-by-one in training reflection evidence: membership h\_k now receives producing group k-1; transition k retains group k. Invalid references are recorded rather than clamped. Previous candidates and measured scores are retained with the archived source and checkpoint. Receipt declares unchanged evaluation identity: Yes; evidence-only repair: Yes.

The reporter validates receipt field types, SHA-256 syntax, and UTC timestamps; the receipt's claims of unchanged evaluation identity and evidence-only repair are declarations by the run operator, not independently established by this report.

Full source and archived-checkpoint SHA-256 values are retained in `report.json` and, when declared, the final export's `optimizer_continuations.json`.

## Frozen seed versus selected graph

The selected state specification, including routing instructions, is byte-for-byte unchanged from the seed. GEPA selection changed only the edge specification.

Seed versus selected frozen specification, before all-corpus completion and edge reconstruction. Prompt-text differences alone do not establish semantic or behavioral changes.

Both fixed graphs were evaluated on the same 150 rollouts from 15 held-out tasks, before all-corpus completion. These are LLM semantic proxy scores; they are not learner performance.

| Metric | Seed mean | Selected mean | Paired task mean change | 95% task-cluster interval |
|---|---:|---:|---:|---|
| score | 0.4249 | 0.4429 | 0.0180 | 0.0055 to 0.0302 |
| history\_coverage | 0.8814 | 0.8885 | 0.0071 | -0.0135 to 0.0288 |
| transition\_coverage | 0.1235 | 0.1446 | 0.0211 | 0.0072 to 0.0340 |
| coherence | 0.6300 | 0.6499 | 0.0199 | -0.0117 to 0.0524 |
| outgoing\_applicability | 0.5336 | 0.5633 | 0.0297 | -0.0109 to 0.0669 |
| complexity | 0.1689 | 0.1796 | 0.0108 | 0.0108 to 0.0108 |

Paired original-task cluster bootstrap: 10,000 draws. Positive change favors the selected graph except complexity, where lower is better. Intervals condition on fixed graphs and cached model judgments; they exclude optimizer selection and model-generation uncertainty.

## Graph construction stages

| Stage | States | Edges | Evaluation scope |
|---|---:|---:|---|
| Initial seed specification | 28 | 30 | Initialization before GEPA |
| Selected frozen specification | 28 | 34 | GEPA selection and frozen held-out scores |
| Final reconstructed witnessed graph | 39 | 550 | Full-corpus assignment, reconstructed contracts, and sampled independent audits |

Selected frozen specification only; not the later reconstructed witnessed graph. Observed endpoint-pair edges receive newly proposed contracts after assignment; they are not the same edge specification scored by GEPA.

Displayed graph: All-corpus witnessed graph after frozen evaluation.

Available graph source: `graph.json`. Assignment source: `assignments_all.json`.

## States with the most assigned histories

Reward mean and variance below count each visiting trajectory once. Variance is a population descriptive estimate.

| State | Histories | Rollouts | Tasks | Mean reward | Reward variance |
|---|---:|---:|---:|---:|---:|
| model\_execution\_successful: dbt model execution successful | 9,355 | 993 | 103 | 0.2669 | 0.1956 |
| model\_files\_created: Model SQL files written to disk | 6,367 | 1,016 | 102 | 0.2598 | 0.1923 |
| existing\_project\_explored: Existing dbt project structure explored | 3,962 | 757 | 90 | 0.2576 | 0.1912 |
| data\_verified: Model data verified | 2,710 | 782 | 101 | 0.2673 | 0.1958 |
| model\_execution\_in\_progress: dbt run parsing in progress | 2,365 | 808 | 102 | 0.2649 | 0.1947 |
| model\_execution\_failed: dbt model execution failed | 1,683 | 479 | 90 | 0.2777 | 0.2006 |
| initial\_task\_context: Initial task context at /app | 1,444 | 1,030 | 103 | 0.2592 | 0.1920 |
| directory\_structure\_discovered: Directory structure discovered | 1,434 | 621 | 100 | 0.2448 | 0.1849 |
| model\_compilation\_started: dbt compilation initiated | 985 | 429 | 89 | 0.2821 | 0.2025 |
| model\_rebuilt\_with\_fix: Model rebuilt after fix | 800 | 356 | 87 | 0.2360 | 0.1803 |
| profiles\_configured: Database connection configured | 778 | 340 | 47 | 0.2647 | 0.1946 |
| dbt\_project\_missing: Required dbt\_project directory missing | 769 | 301 | 44 | 0.2492 | 0.1871 |
| duckdb\_cli\_unavailable: DuckDB CLI tool unavailable | 702 | 568 | 87 | 0.2553 | 0.1901 |
| dbt\_project\_initialized: dbt project initialized | 627 | 328 | 72 | 0.2409 | 0.1828 |
| dbt\_deps\_installed: dbt dependencies installed | 609 | 467 | 81 | 0.2655 | 0.1950 |
| dbt\_deps\_needed: dbt dependencies not installed | 552 | 537 | 72 | 0.2700 | 0.1971 |
| sql\_syntax\_error: SQL syntax or logic error | 454 | 238 | 71 | 0.2563 | 0.1906 |
| project\_file\_structure\_verified: Project file structure confirmed | 334 | 209 | 73 | 0.2488 | 0.1869 |
| data\_verification\_failed: Data verification failed | 322 | 214 | 85 | 0.2430 | 0.1839 |
| data\_verification\_attempted: Data verification method attempted | 286 | 167 | 65 | 0.2216 | 0.1725 |
| schema\_naming\_issue: Schema naming configuration issue | 229 | 128 | 28 | 0.2656 | 0.1951 |
| dbt\_debug\_verified: dbt configuration and connection verified | 202 | 180 | 40 | 0.2222 | 0.1728 |
| test\_execution\_started: dbt test execution initiated | 148 | 74 | 16 | 0.4054 | 0.2411 |
| source\_definitions\_discovered: Source definition files found | 145 | 82 | 27 | 0.1951 | 0.1570 |
| column\_type\_error: Column type casting error | 76 | 52 | 24 | 0.3846 | 0.2367 |

## Independent audits

| Check type | Supported | Contradicted | Unknown (including errors) | Request errors | Non-error unknown | Not run |
|---|---:|---:|---:|---:|---:|---:|
| within\_state\_coherence | 14 | 16 | 71 | 0 | 71 | 0 |
| boundary\_and\_redundancy | 3 | 5 | 22 | 0 | 22 | 0 |
| edge\_source\_applicability | 115 | 459 | 1,071 | 0 | 1,071 | 0 |

## Selected task specifications

| Draft | Feasibility review | Executed | Benchmark validated |
|---|---|---|---|
| Debug dbt Customer Dimension Model with Incorrect Metrics | contradicted | No | No |
| Create Deferred Revenue Recognition Schedule dbt Model | contradicted | No | No |
| Advanced Customer Cross-Sell Insights Analytics Mart | contradicted | No | No |
| dbt Project Initialization Workflow | contradicted | No | No |
| Price Elasticity Analysis with Temporal Data Constraints | contradicted | No | No |
| dbt Project Source Discovery and Dependency Installation | contradicted | No | No |
| Create Customer Activity Tracking Model with Package Dependencies | contradicted | No | No |
| Build Cart Abandonment Recovery Scoring Model with dbt | contradicted | No | No |

## Locally executed synthetic tasks

These are separate from the reviewed specification drafts above. Execution means two independently prompted reference SQL queries ran on a synthetic fixture and agreed; it does not mean a learner passed the task or the source benchmark was reproduced.

Stage status: supported\_paths\_exhausted. Verified local receipts: 0. Requested: 3.

| Task | Reference executed | Independent query executed | Results agree |
|---|---|---|---|

## Inference usage accounting

All retained successful JSON request records under this run's llm_cache directories, including earlier initialization and rejected candidates sharing those caches.

Unique successful request keys: 65,761. Duplicate cache copies excluded: 0. Conflicting keys excluded from token totals: 0.
Recorded transport failures in retained successful request chains: 332.

| Cache accounting scope | Recorded responses | Prompt tokens | Completion tokens | Responses missing usage |
|---|---:|---:|---:|---:|
| Final successful responses | 65,761 | 1,830,743,282 | 47,553,204 | 0 |
| All recorded attempts in successful request chains | 65,909 | 1,836,929,821 | 47,957,301 | 0 |

Process snapshots below overlap the cache totals above and are not added to them.

| Snapshot | Role | Returned responses | Cache hits | Prompt tokens | Completion tokens | Retries |
|---|---|---:|---:|---:|---:|---:|
| completion.json:llm\_usage | router | 42,230 | 0 | 1,087,104,481 | 26,387,595 | 4 |
| operational\_restarts/2026-09-17-classified-history-retention/llm\_usage.json | router | 5,425 | 1 | 158,698,070 | 3,519,308 | 406 |
| operational\_restarts/2026-09-17-global-json-whitespace/llm\_usage.json | router | 77 | 364 | 5,678,350 | 78,655 | 5 |
| operational\_restarts/2026-09-17-reflection-evidence-boundaries/llm\_usage.json | router | 7,733 | 59 | 220,759,385 | 4,829,982 | 2 |
| operational\_restarts/2026-09-17-resampled-output-retries/llm\_usage.json | router | 13 | 117 | 927,804 | 35,963 | 4 |
| operational\_restarts/2026-09-17-small-batch-scheduling/llm\_usage.json | router | 5,483 | 0 | 163,314,602 | 3,537,278 | 69 |
| completion.json:teacher\_usage | teacher | 3,070 | 0 | 141,595,561 | 5,895,374 | 4 |
| operational\_restarts/2026-09-17-classified-history-retention/teacher\_usage.json | teacher | 142 | 0 | 7,271,051 | 590,583 | 0 |
| operational\_restarts/2026-09-17-global-json-whitespace/teacher\_usage.json | teacher | 7 | 0 | 598,873 | 30,467 | 0 |
| operational\_restarts/2026-09-17-reflection-evidence-boundaries/teacher\_usage.json | teacher | 406 | 0 | 21,566,393 | 1,682,573 | 2 |
| operational\_restarts/2026-09-17-resampled-output-retries/teacher\_usage.json | teacher | 3 | 0 | 250,734 | 15,223 | 0 |
| operational\_restarts/2026-09-17-small-batch-scheduling/teacher\_usage.json | teacher | 146 | 0 | 7,591,466 | 616,318 | 1 |

- Final-response totals count each usable cache key once. Recorded-attempt totals replace, rather than add to, final-response totals and include saved output retries.
- Recorded transport failures are counted separately within retained successful request chains. They are not returned model responses and contribute no reported tokens; unclassified attempt entries are also excluded from response totals.
- Process counters overlap cached records and are shown separately without summing across snapshots or roles. Completion counters take precedence over duplicate current usage files.
- A process's requests counter counts returned API responses, including invalid-output responses; it is not a complete count of attempted network requests.
- Calls that never produced a retained cache record, deleted or overwritten records, diagnostics outside this run, and missing token metadata are not fully represented.
- Initialization versus selected-candidate usage cannot be separated reliably from cache metadata. Successful means a cached JSON response, not a scientifically accepted graph decision.
- Endpoint-reported prompt and completion tokens are not measured hardware work or billing. No exact spend is inferred; a live snapshot can lag concurrent requests.


## Interpretation and limits

- The corpus contains archived Horizon/Sonnet 4.5 trajectories mapped to Data Eng Bench task families; these are not executions of the current public benchmark.
- Task IDs are disjoint across training, Pareto, and test splits. Semantic family disjointness has not been established. Pareto scores are adaptively selected validation scores.
- Frozen held-out evaluation precedes all-corpus graph completion. Final completion and graph audits are transductive and are not a second held-out generalization measurement.
- Semantic scores and source applicability audits are LLM proxy judgments. The independent audit uses a separate frozen procedure, not a different model family. Sampled support does not establish universal applicability or executable graph paths.
- Terminal reward variance is descriptive. Repeated histories from one rollout share its outcome; trajectory-deduplicated and task-balanced estimates are reported separately. No claim of causal difficulty, identical-history variance, or post-training lift is made.
- The dataset includes one zero assigned by manual transcript review; the variance artifact contains a sensitivity analysis excluding manual rewards.
- Generated task specifications have LLM feasibility reviews only unless an explicit execution receipt is present. No benchmark execution or training benefit is inferred from a draft.

The searchable [HTML report](report.html) contains every available node and edge, plus an adjacency matrix with exact witness counts and a sampled-traversable filter. [report.json](report.json) contains allowlisted report data and artifact hashes. Edge witness references are limited to three labeled examples per edge; the full local graph remains unchanged.

## Snapshot warnings

- Executable-task construction exhausted eligible paths before reaching its target.
