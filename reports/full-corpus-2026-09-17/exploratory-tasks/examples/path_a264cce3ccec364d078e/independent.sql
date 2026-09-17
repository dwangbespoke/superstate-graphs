WITH valid_orders AS (
    SELECT 
        order_id,
        customer_id,
        ordered_at,
        grand_total
    FROM orders
    WHERE status NOT IN ('CANCELLED', 'RETURNED', 'FAILED')
      AND ordered_at >= '2024-01-01'
      AND ordered_at < '2024-07-01'
),
weekly_agg AS (
    SELECT 
        (DATE_TRUNC('week', ordered_at) - INTERVAL '1 day')::DATE AS week_start,
        COUNT(*) AS order_count,
        COUNT(DISTINCT customer_id) AS unique_customers,
        ROUND(SUM(grand_total), 2) AS total_revenue
    FROM valid_orders
    GROUP BY 1
),
with_metrics AS (
    SELECT 
        week_start,
        DENSE_RANK() OVER (ORDER BY week_start) AS week_number,
        order_count,
        unique_customers,
        total_revenue,
        ROUND(total_revenue / order_count, 2) AS avg_order_value,
        LAG(total_revenue, 1) OVER (ORDER BY week_start) AS prev_week_revenue,
        total_revenue - LAG(total_revenue, 1) OVER (ORDER BY week_start) AS revenue_change,
        CASE 
            WHEN LAG(total_revenue, 1) OVER (ORDER BY week_start) IS NULL THEN NULL
            WHEN LAG(total_revenue, 1) OVER (ORDER BY week_start) = 0 THEN NULL
            ELSE ROUND(((total_revenue - LAG(total_revenue, 1) OVER (ORDER BY week_start)) 
                / LAG(total_revenue, 1) OVER (ORDER BY week_start)) * 100, 2)
        END AS revenue_growth_pct,
        ROUND(SUM(total_revenue) OVER (ORDER BY week_start ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS cumulative_revenue,
        CASE 
            WHEN total_revenue = MAX(total_revenue) OVER (ORDER BY week_start ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            THEN 'Y' ELSE 'N' END AS is_best_week
    FROM weekly_agg
)
SELECT 
    week_start,
    week_number,
    order_count,
    unique_customers,
    total_revenue,
    avg_order_value,
    prev_week_revenue,
    revenue_change,
    revenue_growth_pct,
    CASE 
        WHEN revenue_growth_pct IS NULL THEN NULL
        WHEN revenue_growth_pct >= 50 THEN 'Strong Growth'
        WHEN revenue_growth_pct > 0 THEN 'Growing'
        WHEN revenue_growth_pct = 0 THEN 'Stable'
        WHEN revenue_growth_pct > -50 THEN 'Declining'
        ELSE 'Sharp Decline'
    END AS growth_status,
    CASE WHEN COUNT(*) OVER (ORDER BY week_start ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) = 4 
         THEN ROUND(AVG(total_revenue) OVER (ORDER BY week_start ROWS BETWEEN 3 PRECEDING AND CURRENT ROW), 2)
         ELSE NULL END AS rolling_4wk_avg_revenue,
    cumulative_revenue,
    is_best_week
FROM with_metrics
ORDER BY week_start
