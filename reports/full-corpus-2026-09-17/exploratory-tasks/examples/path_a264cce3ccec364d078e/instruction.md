Build a DuckDB query that calculates weekly sales metrics with week-over-week growth analysis. The data contains orders from January to June 2024. You must:

1. Aggregate orders by week where weeks start on SUNDAY (not Monday)
2. Calculate week-over-week revenue growth percentage
3. Classify growth status based on percentage thresholds
4. Calculate rolling 4-week average (only when 4 weeks of data exist)
5. Track cumulative revenue and flag the best week seen so far

Key challenge: DuckDB's date_trunc('week', ...) returns Monday. You must adjust to get Sunday-based weeks. The first week should have NULL for growth metrics since there's no previous week to compare.

Your query should produce exactly one row per week with all 13 columns in the specified order.
