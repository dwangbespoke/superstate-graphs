# Overnight POC review artifacts

These are compact outputs from real learner rollouts and prompt optimization.
Raw trajectories, databases, and model caches remain in the local `results/` and
`data/` directories. The HTML report one level up contains interactive inspection.

Formation metrics use frozen LLM judgments, not independent execution labels.
Generated SQL tasks have executable oracle consistency checks. Mutable dbt tasks
have finite-sandbox starter/oracle checks against the pristine warehouse. Those checks do
not establish faithful recreation of each source decision or training benefit.

The source benchmark and warehouse are from Snowflake-Labs/data-eng-bench at the
revision in ../../docs/EXPERIMENT.md; upstream licensing continues to apply.
