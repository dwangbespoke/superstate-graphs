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

Do not copy the supplementary directory wholesale into a public report. Model
caches, generator responses, source evidence, raw reviews, and runtime files are
private research artifacts. A separate allowlisted supplementary package can
include successful examples' synthetic instructions, DuckDB fixtures, reference
and independent SQL, expected outputs, checked local-execution receipts, safe
edge/witness IDs, safe selection/summary receipts, and file hashes. Allowlist task
metadata and inspect generated text for source transcripts or credentials before
publication. Regenerate example README links as relative paths.

Give a learner only `instruction.md` and `fixture.duckdb`; keep the SQL oracles and
expected result hidden. Published examples must retain their exploratory evidence
label and raw-versus-validated audit provenance. Do not pass them through the
primary exporter, weaken its supported-edge checks, or insert them into the
original `executable_task_summary.json`.
