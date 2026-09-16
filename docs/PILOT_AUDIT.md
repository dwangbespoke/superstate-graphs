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
