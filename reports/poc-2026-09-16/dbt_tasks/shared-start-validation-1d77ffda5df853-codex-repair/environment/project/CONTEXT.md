# Supplied starting context

The initial terminal working directory is /app.

Both /app/dbt_models_duckdb and /app/dbt_models_snowflake exist as reference projects.

The new task-owned project is /app/sg_project; its backend profile and analytical model are incomplete.

The reference projects are not the submission location; the verifier builds and checks /app/sg_project.

Inspect DB_TYPE before writing backend-dependent configuration. Configure the task-owned project using the connection environment for the observed backend.


# Supplied starting context

The initial terminal working directory is /app.

Both /app/dbt_models_duckdb and /app/dbt_models_snowflake exist as reference projects.

The new task-owned project is /app/sg_project; its backend profile and analytical model are incomplete.

The reference projects are not the submission location; the verifier builds and checks /app/sg_project.
