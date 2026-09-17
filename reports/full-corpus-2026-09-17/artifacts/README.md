# Complete superstate graph research artifacts

This export contains **103 original task IDs, 1,030 rollouts, 37,532 complete history assignments, and 36,502 recorded transition witnesses**. Every membership and witness ID is retained; no witness subsampling is used here. Original transcript contents are not redistributed.

| File | Content |
|---|---|
| nodes.jsonl.gz | All state definitions and complete member-history ID lists |
| edges.jsonl.gz | All directed operation contracts and complete witness-ID lists |
| sampled_supported_graph.json | Ready-to-use directed graph with all nodes and only edges whose finite sampled eligibility was independently recomputed from audit records |
| assignments.jsonl.gz | Every history-to-state mapping, prefix hashes, split, and terminal outcome metadata |
| witnesses.jsonl.gz | Every transition, with exact history/state endpoints and rollout identity |
| rollouts.jsonl.gz | Complete rollout metadata and reward provenance, excluding messages |
| selected_candidate.json; routing_stages.json | Exact selected specification and ordered completion stages |
| seed_candidate.json; accepted_candidate_archive.json | Validated baseline and every recorded GEPA candidate specification, parent indices, and aggregate Pareto scores; no reflection evidence |
| splits.json | Full original-task and rollout split membership |
| state_reward_variance.json | Full outcome moments, weightings, intervals, and manual-grade sensitivity |
| evaluation_summary.json; provenance.json | Observed evaluation summaries and source fingerprints |
| usage_accounting.json | Deduplicated successful cached-request usage and separate overlapping process snapshots; not billing |
| optimizer_continuations.json (when declared) | Allowlisted method-repair receipts, boundaries, source/checkpoint hashes, and declared invariants |
| heldout_comparison.json (when available) | Paired seed-versus-selected held-out aggregates, task-cluster intervals, and common rollout IDs; no inference evidence |
| manifest.json | Exported-file SHA-256 hashes and record counts |

JSONL compression uses gzip with a fixed timestamp. Read with Python's `gzip.open(path, 'rt')`; each line is one JSON object. The `history_id` and `transition_id` columns provide lossless joins. The first routing stage is the optimized candidate; later stages classify only histories left unassigned by earlier stages. Final graph completion is transductive.

For path proposals, load `sampled_supported_graph.json` and traverse its `edges` by `source` and `target`. It retains all nodes, including isolated nodes, and can have zero eligible edges. Every included edge has at least one supported source check and no contradicted, unknown, missing planned check, or proposer rejection. This is finite sampled LLM support, not universal source applicability or proof that a composed path executes. The complete observed graph and witness ledger remain in the unfiltered JSONL files.

The accepted-candidate archive preserves the final GEPA result's original indices, parent rows, and aggregate Pareto-validation scores. Its candidate zero equals the published seed, and best_idx equals the published selected candidate. It does not include rejected proposals or imply that all archived versions remain Pareto-optimal. Candidate indices are not proposal-attempt numbers; continuation boundaries do not directly index the candidate archive. Any declared method continuation remains disclosed separately in optimizer_continuations.json.

The source is the audited historical Horizon/Sonnet 4.5 counterpart corpus, not a rerun of the current public benchmark. One terminal zero comes from documented manual trace review. Formation excludes rewards; descriptive reward statistics are attached afterward. Trajectory-deduplicated and task-balanced statistics address repeated visits and uneven task representation.

Held-out scores precede all-corpus completion. Independent audits use a separate frozen LLM procedure, not an independent model family. Traversable flags mean sampled support for drafting, not universal applicability or guaranteed executable compositions. No post-training lift is claimed.

Optional locally executed synthetic tasks included: **0**. Their SQL agreement checks are not learner evaluations or official benchmark verification. See per-task receipts and instructions when present.
