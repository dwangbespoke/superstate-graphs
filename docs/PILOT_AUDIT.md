# Pilot outcome audit — September 16

An adversarial critic inspected two completed easy-task failures while the
collection continued. No shared harness or warehouse problem was found.

- `dbt-daily-order-summary` successfully built a view in `main`, while its
  instruction requires `daily_analytics` and explicitly asks for that schema in
  `profiles.yml`. It voluntarily declared completion at 27 turns. The missing
  required-schema output is a learner error.
- `dbt-customer-geographic` ignored the requested `/app/dbt_project`, worked in
  the reference project, deleted model directories during unsuccessful repairs,
  and exhausted the 30-turn budget. Its oracle run passed all nine tests.
- The first `dbt-fix-daily-cohorts` run exceeded context length. Its usable prefixes
  remain eligible for graph construction; its terminal outcome stays unknown.

The bounded learner protocol is unchanged. These observations are not enough to
conclude that superstates have useful outcome variance or that generated tasks
will improve training.

A construction limitation deserves explicit review: the current generated-task
format is read-only SQL, so it can reproduce relational decisions but cannot
faithfully reproduce every dbt project/configuration failure. Review actual
selected targets against generated examples rather than assuming SQL execution
proves preservation of the targeted decision.

## Shared-world scope correction

The six-task collection contains five retail-warehouse tasks and one auxiliary
advertising-CSV task. `dbt-consolidate` specifies `/app/data/googleads.csv`,
`metaads.csv`, and `tiktokads.csv`, and builds `/app/consolidate.duckdb`. Its two
rollouts are retained as auxiliary data but excluded before core graph formation,
task splitting, probe creation, and GEPA. The exclusion follows the task's input
world, irrespective of its outcomes.

The same task explicitly asks for installing dbt-utils through `dbt deps`, so our
network restriction materially changes its setup. Do not interpret its failures
as unqualified evidence about learner capability. None of the other five tasks
requires that external installation. The core has 10 rollouts: nine graded zeros
and one context-limit error with unknown reward.

The CAC instruction also contains an upstream reference-path ambiguity between
`/app/dbt_transforms` and backend-specific project directories; retain this caveat
when interpreting those traces.

## Prefix-compression audit

A critic compared three actual annotations against their recorded prefixes. No
future-action or reward leakage was found, but two important evidence losses were
identified:

- Daily order summary `h026` promoted the learner's assertion about
  `daily_analytics` into a fact, despite visible evidence of a `main` output and a
  profile without the required schema.
- Channel revenue `h023` omitted the observed `sale_key AS sales_id` mapping and
  described that mapping as still unknown. This confuses missing information with
  observed information the learner has failed to use.
- The initial geographic annotation described too much of a future workflow as
  the current decision. This is a smaller precision issue.

The extraction layer therefore needs to distinguish task requirements, tool
observations, and learner beliefs; preserve conflicts and explicit mappings; and
carry source-indexed evidence snippets alongside summaries. These annotations
remain a model-generated approximation of the history, not certified sufficient
statistics.
# Frozen-judge protocol correction

The first frozen bank labeled all 68 sampled pairs contradicted. An adversarial
review found a concrete mismatch with the intended abstraction: two initial
histories both needed to identify `$DB_TYPE`, yet the judge rejected them because
their eventual output models and target schemas differed. Ten initial/initial
pairs were present, so candidate retrieval had not simply missed this shared
decision. Other negatives, such as initialization versus final validation, were
legitimate.

This unsuccessful downstream run is preserved in
`results/analysis_v1/previous_task_identity_judge`. The revised protocol evaluates
local decision compatibility from the prefixes alone, then evaluates a witnessed
operation under explicit artifact bindings. Different final goals remain
constraints to respect, rather than an automatic reason to reject a shared
local decision. Local transfer evidence must not be presented as proof that a
whole observed segment or graph path can be transplanted.

The same run exposed a GEPA adapter integration error: a missing
`propose_new_texts` attribute prevented reflective mutation. Its single evaluated
seed is not a prompt evolution. The repair includes a cheap stub-based test of
the actual installed optimizer's reflection path before paid optimization resumes.
