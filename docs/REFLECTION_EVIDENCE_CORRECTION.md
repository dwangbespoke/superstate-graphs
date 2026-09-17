# Reflection evidence boundary correction — 2026-09-17

This correction was identified by a read-only diagnostic of proposals 0002–0006 and their training evidence. No Pareto or test feedback was inspected for the diagnostic. The defect concerns evidence attached to the proposal model, not the history boundaries used by the router or evaluator.

## Confirmed defect and correction

The evaluator defines history `h_k` as the query plus completed groups 1 through k. `render_step(k)` returns the next recorded transition, `h_k -> h_{k+1}`. Previously, `make_reflective_dataset` attached `render_step(k)` to every feedback item, including a membership critique of `h_k`. Thus a critique could be accompanied by evidence from after the history it concerned. Invalid indices were also silently clamped to another transition.

The corrected retrieval rule is:

| Feedback kind | Meaning of feedback index k | Attached evidence |
|---|---|---|
| `membership` | History `h_k` | Producing transition `k−1`; query only at `h_0` |
| `transition` | Transition `h_k -> h_{k+1}` | Transition `k` |
| `edge`, `redundancy` | Source-history boundary `h_k` for retrieval | Producing transition `k−1`; query only at `h_0` |

The last row is a conservative retrieval convention because these feedback kinds do not declare a separate index type. It does not change the judge's outputs or scores. Every attached record explicitly labels the feedback kind/index, history ID, and transition index. Unsupported kinds and invalid indices now produce an `invalid_feedback_reference` record with no attached group. Feedback text remains available verbatim. The existing first-eight-item retrieval limit is unchanged.

## Training witnesses

- `c21e45d0-b12e-416c-8ac2-f0806b59f1e5:h0023`, task `dbt-fix-product-metrics`, proposal 0004 training minibatch. The membership critique explicitly concerns successful compilation at h23. Transition `t0022` returns the compiled SQL. The former attachment, `t0023`, instead starts `dbt run -s fct_product_metrics`, a subsequent action. Saved evaluation: `results/full_graph/run_v1/evaluations/cfb648265455441d68fc7de948e9da1426564e1412674cdfdb44d9a34f7f50ff/c21e45d0-b12e-416c-8ac2-f0806b59f1e5.json`.
- `65736a71-2684-4368-880b-331ceaa4e21d:h0006`, task `dbt-customer-lifecycle-journey`, same training minibatch. Feedback criticizes assigning `model_files_created` after only a macro was created. Producing transition `t0005` writes and reads that macro. The former attachment, `t0006`, contains the subsequent attempted staging-model write and its JSON parsing error. Saved evaluation: `results/full_graph/run_v1/evaluations/4f58605de31d149bfd7be307d444f208637a163daaaf944f51733444302d024b/65736a71-2684-4368-880b-331ceaa4e21d.json`.

## Scope and provenance

Only `GraphGEPAAdapter.make_reflective_dataset`, its tests, and this document changed. `JUDGE_SYSTEM`, routing, scoring, schemas, decoding, caches, acceptance, and selection are unchanged. Existing exact routing/evaluation caches and recorded population scores remain valid for their original inputs. The changed reflection input can change later proposals; continuation must record an explicit optimizer provenance boundary. This document does not itself migrate any checkpoint or identity.

SHA-256 for `src/superstate_graphs/graph_evolution.py`:

- Before: `d15c8bdef53acc7762cb36716e86f468050b4e3ced758f30cfc70a8aa766ca8e`
- After: `3d0691a0cc72235d4a78d2c194eda05e509ea683fd3af70863b403346b44eb02`

Validation: 52 graph-evolution tests passed; Ruff and `git diff --check` passed. New cases verify initial and terminal history boundaries, producing versus following observations, transition indexing, conservative edge/redundancy retrieval, and invalid references without clamping.

## Separate structural proposal failures

The diagnostic found no evidence of wrong-parent graph serialization or partial component updates. Proposals 0002, 0003, and 0005 all target the saved seed. Their rejection reasons are concrete model patch mistakes: 0002 deletes nonexistent edge `discover_source_definitions`; 0003 deletes states but leaves seven inherited edges referencing them; 0005 supplies edge IDs `model_fails_type_cast` and `model_fails_syntax` as state deletions. The patch instructions explicitly retain omitted records and require all incident references to be resolved. Rejection reasons are included in subsequent reflection context. The indexing defect is not established as the cause of these structural failures.
