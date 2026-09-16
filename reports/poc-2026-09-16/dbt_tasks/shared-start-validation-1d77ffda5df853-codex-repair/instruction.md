You start in /app. Read /app/sg_project/CONTEXT.md. Before writing backend-dependent configuration, run `echo $DB_TYPE` and inspect the connection environment for that backend. Do not infer the active backend from the mere existence of either reference project. Work only in /app/sg_project and keep the source warehouse unchanged.

Configure profile sg_profile_991dec701d for the active backend. Implement model sg_result_991dec701d.sql and materialize its output as a TABLE in schema sg_991dec701d. The verifier runs dbt against /app/sg_project with profiles from that same directory; changing the reference projects is not a submission.

Build an acquisition-channel cohort report with the following precise definitions:

1. Use ANALYTICS.FACT_SALES joined to ANALYTICS.DIM_DATE by DATE_KEY. Retain only rows with a matching date and non-null CUSTOMER_KEY. These are the eligible dated sales; do not infer a calendar date from an unmatched DATE_KEY. Treat each FACT_SALES row as a sale-line record and use CUSTOMER_KEY as the customer identity without filtering through DIM_CUSTOMER.
2. Each customer's cohort_date is their earliest eligible sale date. Their acquisition channel is CHANNEL_KEY from the first sale, ordered by sale date, then TIME_KEY ascending NULLS LAST, ORDER_ID ascending NULLS LAST, then SALE_KEY ascending. Assign each customer exactly once. Keep only customers whose cohort_date is at least 90 calendar days before the maximum eligible sale date across the entire warehouse; this removes right-censored cohorts.
3. Emit exactly one row for each nonempty (cohort_date, acquisition channel) group. month_start is the first calendar day of cohort_date's month. channel_key is the acquisition channel; obtain channel_code from ANALYTICS.DIM_CHANNEL, using 'UNKNOWN' if the channel is unmatched or its code is null.
4. cohort_size counts distinct assigned customers. day_0_customers, day_7_customers, day_30_customers and day_90_customers count assigned customers with at least one eligible sale on exactly that many calendar days after their own cohort_date. These are exact-day retention counts, not cumulative windows or month buckets. Each corresponding retention is its count divided by cohort_size.
5. For the same assigned customers, include every eligible sale from cohort_date through cohort_date + 90 days, inclusive, regardless of the sale's later channel. Attribute those sales to the customer's acquisition channel. order_count counts distinct ORDER_ID values in the group. revenue sums TOTAL_AMOUNT as recorded (do not subtract discounts or taxes again); profit sums PROFIT_AMOUNT; discount_amount sums DISCOUNT_AMOUNT.
6. margin_rate = profit / revenue; avg_order_value = revenue / order_count; discount_rate = discount_amount / revenue. Preserve legitimate zero values. A zero denominator yields NULL. Round revenue, profit, discount_amount and avg_order_value to two decimal places and rates to six. Compute ratios from unrounded sums.

Output exactly these columns in this order:
cohort_date, month_start, channel_key, channel_code, cohort_size, order_count, day_0_customers, day_7_customers, day_30_customers, day_90_customers, revenue, profit, discount_amount, margin_rate, avg_order_value, discount_rate, day_0_retention, day_7_retention, day_30_retention, day_90_retention.
Dates must be DATE values. Monetary outputs must be DECIMAL(18,2), and rates DECIMAL(18,6); counts are integers. Row ordering does not matter. Run `dbt run --project-dir /app/sg_project --profiles-dir /app/sg_project` successfully. No network or package installation is needed.

Before editing the project, read /app/sg_project/CONTEXT.md for the supplied starting context.

