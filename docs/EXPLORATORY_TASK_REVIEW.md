# Supplementary task review: execution and semantic correctness

This post-formation coding-agent review covers two early, purposively selected
failed attempts and both examples that later passed the supplementary execution
gate. It is separate from the [primary graph and task review](FINAL_EDGE_AND_TASK_REVIEW.md).
The [supplementary experiment](EXPLORATORY_TASKS.md) exhausted 16 selected paths;
two received `locally_executed_consistent`. Both retain that recorded status and
their model review's `faithful` verdict. The separate manual semantic verdict is
`failed` for both: **these are failed research cases, not learner-ready tasks**.

The review inspected instructions, actual fixture tables, both SQL solutions,
expected rows, and concrete source transitions. It used independent Python
calendar/quantity arithmetic and the unchanged original verifier. No source
task, SQL oracle, fixture, model verdict, or recorded result was repaired. The
findings do not estimate population accuracy or establish learner performance.
The separately prompted SQL solutions and faithfulness reviewer use the same
model family; their agreement is not independent gold correctness.

## Two early execution failures

| Path | Recorded review | Recorded execution | Additional finding |
| --- | --- | --- | --- |
| `path_0206be1787df8b3a1890` | `faithful` | `execution_failed` | Incorrect date-arithmetic instruction; date-dependent answer. |
| `path_0788c0148d2793f7966a` | `faithful` | `execution_failed` | Source-discovery challenge removed despite the positive verdict. |

The first task asks for customer recency. Its reference query fails because its
`DATE()` call is unavailable in the actual DuckDB 1.2.2 runtime. Separately, the
instruction says subtracting two dates produces an interval. An isolated check
in that runtime returned `BIGINT` with value `1` for the difference between two
`DATE` literals one day apart. The fixture supplies date strings and asks for
conversion to dates, so the instruction misstates its own intended case. Both
queries also use `CURRENT_DATE`; without a fixed as-of date, a runnable version's
answer would change over time. No successful answer was produced or repaired.

The second reference query fails because a later CTE uses `CAMPAIGN_ID` that an
earlier CTE did not project. Its path, `E_573ad8b1cd98` followed by `E_b405a305f817`,
concerns finding a source-definition file and reading its YAML. The generated
task instead supplies named campaign tables and asks for engagement metrics.
The reviewer explicitly acknowledges that source discovery was not preserved
while returning `faithful`. A syntax repair alone would not resolve that
contradiction or restore the local challenge.

## Executed example: weekly sales

`path_a264cce3ccec364d078e` contains 25 orders and a separate three-row table of
excluded orders. Both SQL solutions execute and agree, but share this expression:

```sql
DATE_TRUNC('week', ordered_at) - INTERVAL '1 day'
```

It moves Sunday orders into the preceding week. The actual fixture contains
orders on January 7, 8, and 10 with revenues 150, 200, and 175. A Sunday-start
January 7 week therefore has **three orders, two customers, and revenue 525**.
The stored answer instead creates a December 31 row for the January 7 order
with revenue 150, then places revenue 675 in the January 7 row. Independent
calendar grouping gives 13 weeks; the stored answer contains 14.

The original verifier accepts the saved oracle and rejects an empty query with
the same output schema. It also rejects a diagnostic query that changes only
the Sunday boundary by adding one day before truncation. Thus the verifier
works as an answer comparator while enforcing a shared semantic error. The
diagnostic was not saved over either original solution.

The learner instruction demands 13 columns in a specified order and growth
classification thresholds, but supplies neither the column list nor the
thresholds. They exist only in hidden task metadata/SQL. The fixture also leaves
filtering untested: every queried order is `COMPLETED`; the invalid statuses are
in a separate table that neither solution reads. No prior week has zero revenue.

Path fidelity is also limited. `E_945befbdb7a9` uses witness
`5d59e5dd-bc53-414e-abda-eddea5ff27ae:h0017` to `h0018`, which waits for an already
running `dbt debug`. `E_fed5351a5c1e` uses
`00b9900d-8f0b-4bf5-b5a6-00fcf8fc7cd7:h0010` to `h0011`, which starts `dbt build`
and observes parsing. The generated SELECT takes a downstream weekly-sales
problem from the source task while removing those configuration/execution
operations. Execution does not demonstrate that their composition was preserved.

## Executed example: fulfillment analysis

`path_c4e060c4230cbe9cc658` contains 15 orders and 20 order lines. Independent
Python arithmetic reproduces all 11 stored rows **conditional on the hidden
oracle rules**. For example, `ORD001` has three positive-quantity items and takes
one day to ship and two more to arrive. This arithmetic agreement does not make
the learner task fully specified: its instruction omits the ten required output
columns/order, the seven-day on-time cutoff, and fulfillment-tier thresholds.

There is also a concrete instruction/output conflict. `ORD006` lacks a shipping
date. The instruction says such orders should be excluded from on-time
calculations, but both solutions classify it as `false`. A diagnostic that
returns `NULL` for that excluded classification is rejected. The instruction
does not explain an alternative representation that resolves the conflict.

The unchanged verifier produced these control results:

| Submitted query variant | Result |
| --- | --- |
| Saved oracle | Passed |
| Empty result, same schema | Rejected |
| Include negative quantities in item totals | Rejected |
| Omit status trimming | Passed |
| Omit invalid-status exclusion entirely | Passed |

The last two variants pass because every invalid-status order is already
undelivered, while whitespace appears only on valid statuses. Those requirements
therefore lack a distinguishing fixture case even though the model review calls
all preserved challenges testable.

The concrete path witnesses install dependencies with `dbt deps`
(`E_1901c1138853`, `58b7c16c-15e6-47be-87df-1751080b6bef:h0013` to `h0014`) and run
selected models with mixed successes/errors (`E_f1c64ef4904a`,
`a9584652-9940-41ce-818b-19ffb5cd9a8b:h0026` to `h0027`). A standalone fulfillment
SELECT removes dependency installation and partial-execution diagnosis. Its
successful execution provides no evidence that those local operations survived.

## Publication and interpretation

The two executed examples may be published as explicitly labeled failure cases,
with their frozen execution statuses and separate failed semantic reviews.
Inspection of their proposed public instructions, SQL, expected outputs, and all
actual fixture tables found generic synthetic identifiers and no apparent
credentials, contact details, local user paths, or raw transcripts. This was a
scoped coding-agent inspection, not a human review or a privacy guarantee. Raw
source prefixes and private review receipts are excluded from publication.

Future task construction needs a learner-visible specification check, calendar
boundary checks independent of model-generated SQL, distinguishing fixture rows,
and a review that verifies the selected local operations. These observations
were not fed back into formation or used to change this experiment's outcomes.
