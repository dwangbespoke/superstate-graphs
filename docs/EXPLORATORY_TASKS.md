# Conditional witness-specific executable examples

`scripts/prepare_exploratory_task_examples.py` prepares a separate supplementary
experiment after the original run has finished. It asks whether particular
observed transition witnesses can be instantiated as a coherent synthetic SQL
task. It does not establish reusable edge applicability. Its outputs never change
the original graph, independent audits, completion record, or executable-example
summary. The primary report and export continue to use their original eligibility
rules.

This experiment is conditional: implementation and offline tests do not mean it
was run. Actual activation and outcomes must be established from its separate
selection and execution receipts.

## Selection and interpretation

A candidate uses two distinct observed edges with concrete original witnesses
from at least two tasks. Each edge must have a positive proposer plausibility
verdict, completed source audits, and valid witness endpoints and memberships.
Missing or failed requests are excluded. Every raw or citation-validated
contradiction is excluded, including a raw contradiction downgraded to unknown.

Recorded unknown applicability is allowed. When a supported/unknown raw judgment
has invalid citations, its validated verdict remains unknown and receives the
explicit label `unresolved_applicability_due_to_citation_errors`. Selection
receipts preserve both verdicts and the citation-error count. Such a judgment is
neither supporting evidence nor a validated counterexample.

Qualifying saved paths are considered first, prioritizing plausible draft
reviews. A contradicted primary draft review excludes its entire ordered edge
sequence, including resampling under another path ID or alternate witnesses.
Primary-stage attempts are also skipped, including another sampled witness choice
for the same ordered edge sequence. The optional `--resample-filtered-paths` flag
fills remaining candidates using `select_grounded_paths` with seed **20260919**, length
two, and cross-task witnesses. Sampling uses an in-memory copy of the exploratory
edge pool; no graph flags are saved. Every selected path records its selection
origin. At most 24 candidates are attempted, stopping after at most three successes.

## Read-only selection

From the repository root, with the project's environment installed:

```sh
.venv/bin/python scripts/prepare_exploratory_task_examples.py \
  --run results/full_graph/run_v1 \
  --corpus results/full_graph/corpus \
  --resample-filtered-paths
```

Omit the resampling flag to inspect only previously saved paths. Selection prints
safe IDs, verdict labels, counts, and source hashes. It does not open a model
runtime or create an output directory. Both `completion.json: status=complete`
and a finalized original executable stage are required. No pending audit or
partial run is promoted to completion by this script.

## Explicit activation and resume

After reviewing the actual completed primary outcome and selection receipt:

```sh
.venv/bin/python scripts/prepare_exploratory_task_examples.py \
  --run results/full_graph/run_v1 \
  --corpus results/full_graph/corpus \
  --resample-filtered-paths \
  --execute \
  --runtime /absolute/private/path/to/teacher-runtime.json \
  --output results/full_graph/run_v1_exploratory_tasks_v1 \
  --max-paths 24 --target 3
```

The output directory must be separate from the original run and corpus. Runtime
configuration is read only during explicit execution and is never printed.
Source hashes are checked before execution and after each attempt. Checkpoints
bind the selection, model identity, script, constructor implementation, and
saved result/artifact hashes. Repeating the same command resumes its recorded
attempt ledger. Changed inputs require a new output directory. A stopped attempt
whose result was saved before checkpointing can use the constructor's own
validated cache on resume. A recorded construction error counts as an attempt;
this script does not silently repeat unsuccessful attempts.

Each attempt calls the existing `construct_executable_example` procedure using
complete source prefixes and observed transition extensions. Its independent
review must judge the generated task faithful and produce SQL without seeing the
generator's reference SQL. Both queries then run locally on the synthetic
DuckDB fixture with external access disabled. A successful example requires a
faithful review, two executed queries, and matching results. This is evidence
about that task and fixture; it does not prove an arbitrary path executable,
universal edges, benchmark reproduction, learner performance, or training lift.

`exploratory_task_summary.json` and `selection.json` retain the explicit
`witness_specific_exploratory_not_sampled_supported` label. They remain outside
the primary report's supported-path and executable-example counts. Selection
after seeing the primary audit is exploratory and cannot estimate population
task-generation quality.

## Publication boundary

Only after the supplementary experiment has finalized, create a separate,
allowlisted package without model calls:

```sh
.venv/bin/python scripts/publish_exploratory_task_examples.py \
  --experiment results/full_graph/run_v1_exploratory_tasks_v1 \
  --run results/full_graph/run_v1 \
  --corpus results/full_graph/corpus \
  --output reports/full-corpus-2026-09-17/exploratory-tasks
```

The publisher recomputes the selection against the original source hashes,
verifies the finalized attempt ledger and implementation/model identity, and
checks result and artifact receipts. Successful tasks must retain consistent
faithfulness-review and local-execution records. It scans both file bytes and
decoded database cells for recognizable credentials or transcript framing.
Sources are read-only; an existing output directory is never overwritten. Zero
successful examples still produces an honest safe summary, including every
attempt's status count. No raw failure messages are published.

### Optional semantic-review receipt

Add `--manual-review /absolute/path/to/manual-review.json` to the publication
command when a separate manual review is available. This does not change the
fixed experiment's query-agreement statuses, success counts, or stopping target.
The package adds `review.json` and an explicit warning to each example's README,
records semantic verdict counts separately, and binds the review source in the
manifest. A failed review is labeled **FAILED RESEARCH CASE — NOT LEARNER-READY**.
Its original queries and expected results remain available as research evidence,
including the documented defects. A missing review is `not_reviewed`, never a
pass. `no_defect_identified` refers only to the supplied review's scope and is
not benchmark validation or a learner-readiness certificate.

The strict input schema is:

```json
{
  "format": "exploratory-task-manual-review-v1",
  "reviews": [{
    "path_id": "an_actual_exported_example_id",
    "verdict": "failed",
    "findings": ["Concise, source-safe account of a checked semantic defect."],
    "review_method": "Describe the arithmetic, instruction, or verifier checks performed.",
    "reviewer_kind": "coding_agent",
    "reviewed_at_utc": "2026-09-17T19:30:00+00:00",
    "reviewed_artifact_sha256": {
      "instruction.md": "actual_sha256",
      "fixture.duckdb": "actual_sha256",
      "oracle.sql": "actual_sha256",
      "independent.sql": "actual_sha256",
      "expected_result.json": "actual_sha256"
    }
  }]
}
```

This is a schema illustration, not an experimental result. All fields shown are
required; extra fields are rejected. Verdicts are `failed`, `unresolved`, or
`no_defect_identified`; reviewer kinds are `coding_agent` or `human`. Findings
must be nonempty, source-safe strings. The publisher rejects duplicate or
unexported path IDs and stale hashes for any of the five bound task artifacts.
It records the supplied review rather than inferring semantic validity from SQL
agreement. Human review and coding-agent review remain explicitly distinguished.

Before publicly distributing the package, inspect generated instructions,
titles, SQL literals, expected outputs, and synthetic fixture cells for private
or copied source material. Automated scans cannot prove semantic privacy; the
manifest records this limitation and a privacy-review requirement. This local
export does not claim that such human inspection has happened.

Do not copy the supplementary directory wholesale into a public report. Model
caches, generator responses, source evidence, raw reviews, and runtime files are
private research artifacts. A separate allowlisted supplementary package can
include successful examples' synthetic instructions, DuckDB fixtures, reference
and independent SQL, expected outputs, checked local-execution receipts, safe
edge/witness IDs, safe selection/summary receipts, and file hashes. Allowlist task
metadata and inspect generated text for source transcripts or credentials before
publication. Regenerate example README links as relative paths.

Before any learner use, inspect the semantic-review status; failed research cases
are not learner-ready. Learner inputs, if a task is later made suitable, are only
`instruction.md` and `fixture.duckdb`; keep SQL oracles and expected results hidden.
Published examples must retain their exploratory evidence
label and raw-versus-validated audit provenance. Do not pass them through the
primary exporter, weaken its supported-edge checks, or insert them into the
original `executable_task_summary.json`.
