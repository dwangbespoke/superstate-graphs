Create a dbt project at /app/sg_project that materializes model sg_result_6a8ba1bfe2 as table sg_6a8ba1bfe2.sg_result_6a8ba1bfe2.

Before writing configuration or SQL, run `echo $DB_TYPE` to discover the active database backend. Inspect the corresponding connection environment variables; if the backend is DuckDB, inspect `DUCKDB_PATH` (documented fallback: /app/database/retail.duckdb). Do not change DB_TYPE. Reference projects at /app/dbt_models_duckdb/ and /app/dbt_models_snowflake/ may be inspected but must not be modified.

Use project name sg_project_6a8ba1bfe2 and profile name sg_profile_6a8ba1bfe2. Set the target schema sg_6a8ba1bfe2 in profiles.yml. Implement models/sg_result_6a8ba1bfe2.sql and run `dbt run --project-dir /app/sg_project --profiles-dir /app/sg_project` successfully. Materialize a table; row order is not graded.

Use ORDERS.ORDERS and CUSTOMER.CUSTOMER_ADDRESSES. A qualifying order has STATUS in ('COMPLETED', 'DELIVERED', 'SHIPPED'). Define the reporting end date as the maximum CAST(ORDERED_AT AS DATE) across all qualifying orders, before any address join. Include the 365 calendar dates from end_date minus 364 days through end_date, inclusive. This window is determined from the data, not the current clock.

Keep addresses with IS_DEFAULT_SHIPPING = true and non-NULL STATE_PROVINCE, and join qualifying orders to those addresses on CUSTOMER_ID. Emit exactly the observed (state_province, order_date) groups with at least one joined qualifying order in the reporting window. Do not construct a full state/date grid or emit zero-order groups. The supplied warehouse has at most one default shipping address per customer; inspect the source schema as needed.

Return these columns in exactly this order:
- state_province: the default shipping address's STATE_PROVINCE.
- order_date: CAST(ORDERED_AT AS DATE).
- order_count: count of distinct ORDER_ID in the group.
- total_revenue: sum of GRAND_TOTAL, rounded to 2 decimal places.
- avg_order_value: arithmetic mean of GRAND_TOTAL over joined orders, rounded to 2 decimal places.
- revenue_per_customer: the unrounded group revenue sum divided by the number of distinct CUSTOMER_ID with a qualifying order in this same group, then rounded to 2 decimal places.
- orders_per_customer: distinct order count divided by that same distinct customer count using fractional division, rounded to 2 decimal places.

No output column may be NULL. The supplied data yields nonnegative metrics and a nonempty result under 1000 rows. Do not use an arbitrary LIMIT, hardcoded result rows, or a hardcoded reporting end date. Read /app/sg_project/CONTEXT.md for the supplied starting context. Do not modify source warehouse tables.

Before editing the project, read /app/sg_project/CONTEXT.md for the supplied starting context.

