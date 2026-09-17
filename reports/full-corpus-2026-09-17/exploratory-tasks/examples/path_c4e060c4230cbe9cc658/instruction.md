Analyze POS order fulfillment data to calculate delivery metrics while properly handling data quality issues. The ORDERS table contains orders with various statuses and fulfillment states. Some orders have NULL values for shipped_at, delivered_at, or cancelled_at. Some orders have invalid status values with leading/trailing whitespace. Some orders have missing or zero quantities in ORDER_LINES.

Your task is to create a single SELECT query that:
1. Filters to only 2024 orders (ordered_at >= '2024-01-01' AND ordered_at < '2025-01-01')
2. Excludes orders where TRIM(status) is in ('CANCELLED', 'RETURNED', 'FAILED')
3. Calculates fulfillment metrics for each order that has been delivered
4. Handles NULL values appropriately in calculations
5. Returns one row per delivered order with the specified columns

The query must correctly handle edge cases where:
- shipped_at is NULL but delivered_at is not (should be excluded from on-time calculations)
- delivered_at is NULL (order not yet delivered)
- status has whitespace that needs trimming
- quantities are zero or negative (should be excluded from item counts)
