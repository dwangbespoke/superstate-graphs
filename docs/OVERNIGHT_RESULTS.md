# Overnight POC — September 16, 2026

The run produced real rollouts, evaluated GEPA prompt variants, an observed graph,
and two executable task examples. It also exposed substantial limitations. This
is a feasibility result, not evidence for the full research hypothesis.

## Artifacts and outcomes

| Stage | Observed result |
| --- | --- |
| Collection | 12 Qwen3.5-9B rollouts over six data-eng-bench tasks |
| Shared-world corpus | Five tasks, ten rollouts, 60 sampled prefix histories; one auxiliary CSV-world task excluded |
| Learner outcomes | Nine graded failures and one unknown context-limit outcome in the core corpus |
| Formation | Five evaluated prompt versions, including the seed; 44 frozen compatibility probes |
| GEPA selection | All versions tied at 4/12 development-validation score; seed retained |
| Graph | 13 occupied superstates, 49 witnessed transition records |
| Construction proposals | 12 intermediate cross-task junction proposals; three supported shared-start branch compositions |
| Automatic construction | All three bounded Qwen shared-start attempts failed, each allowing three repairs |
| Assisted examples | Two Codex-critic repairs passed finite Modal dbt execution and result checks |

Open [the interactive report](../reports/overnight-poc.html) to inspect group
definitions, member prefixes, witnesses, prompt variants, proposed compositions,
task instructions, and execution receipts. Compact review artifacts are in
[reports/poc-2026-09-16](../reports/poc-2026-09-16/). Full raw rollouts and model
caches remain in local `results/`; warehouse files remain in `data/`.

## Executable examples

1. **Geographic daily analytics** combines state-level customer geography with
   daily order metrics. It requires a fresh dbt profile and materialized model,
   exact observed-group semantics, and a data-anchored reporting window.
   The solution produces **34 rows**.
2. **Acquisition-channel cohorts** combines customer cohort retention with channel
   revenue reporting. It defines first-sale acquisition channels, mature cohorts,
   exact-day retention, and 90-day revenue attribution. The solution produces
   **725 rows**. A separate Python calculation additionally checked cohort
   membership and retention counts during critic repair.

For both examples, the pristine and untouched starter states do not pass. The
oracle successfully runs dbt, creates a `BASE TABLE`, and matches the frozen
reference result's columns and row multiplicities on the pinned warehouse.
Each receipt records the task specification hash, warehouse hash, commands, and
sandbox termination. Oracle files and expected results are outside the learner's
image context.

These are **Codex-assisted repairs of failed Qwen proposals**. Original proposals,
deterministic scaffold changes, critic repair notes, and rejected attempts are
retained. They must not be counted as unassisted generation successes. Some
immutable proposal-stage notes predate runtime validation; the later receipt for
the matching specification hash establishes execution status.

## What remains unestablished

- No positive pooled outcome variance appeared. Construction used supported
  situations as a fallback, so this run does not test variance-guided selection.
- The supported construction targets are initial backend-discovery decisions.
  They are simpler than the intended difficult intermediate situations.
- The two source segments are branches from a shared initial decision. Their
  new combined goals were authored during construction. This does not establish
  sequential graph-path composition or successful reuse of complete skills.
- Passing the task verifier does not establish that a learner actually inspected
  `DB_TYPE`; a correctly guessed profile can pass. Target engagement and generated
  task difficulty still need fresh learner rollouts.
- GEPA produced genuine candidate changes but no selected validation improvement.
  Its frozen LLM compatibility labels are proxy evidence, not execution labels or
  proof that reward distributions are homogeneous inside each superstate.
- No learner training or learning improvement was measured. The development task
  split is not a held-out benchmark result.

The practical next experiment should obtain a mixture of successful and failed
natural rollouts, then test whether generated tasks preserve a specific recurring
intermediate decision. The current artifact flow and mutable dbt task format can
support that experiment; the present pilot does not settle it.

## Runtime and validation notes

The learner/extractor/classifier was Qwen3.5-9B; the GEPA reflector, compatibility
judge, and initial task constructor were Qwen3.5-35B-A3B-FP8. Model revisions,
benchmark revision, and decoding settings are recorded in the experiment files.
Later GEPA proposals encountered temporary HTTP 503 failures. Five successfully
evaluated versions were retained; failed proposals were not counted as revisions.

Twenty-one focused constructor/runtime tests and Ruff passed. Both actual task
validation sandboxes terminated, and both GPU model servers were stopped.
Token counters in the analysis summary describe that invocation, not total
overnight usage across previous protocol revisions and construction.
