WITH order_lines_agg AS (
    SELECT
        order_id,
        SUM(CASE WHEN quantity_ordered > 0 THEN quantity_ordered ELSE 0 END) AS total_items
    FROM ORDER_LINES
    GROUP BY order_id
),
valid_orders AS (
    SELECT
        o.order_id,
        CAST(o.ordered_at AS DATE) AS order_date,
        o.order_source,
        o.grand_total AS total_revenue,
        o.shipped_at,
        o.delivered_at,
        o.cancelled_at,
        ola.total_items
    FROM ORDERS o
    LEFT JOIN order_lines_agg ola ON o.order_id = ola.order_id
    WHERE o.ordered_at >= '2024-01-01'
      AND o.ordered_at < '2025-01-01'
      AND TRIM(o.status) NOT IN ('CANCELLED', 'RETURNED', 'FAILED')
      AND o.delivered_at IS NOT NULL
)
SELECT
    order_id,
    order_date,
    order_source,
    COALESCE(total_items, 0) AS total_items,
    total_revenue,
    CASE 
        WHEN shipped_at IS NOT NULL THEN DATEDIFF('day', order_date, shipped_at)
        ELSE NULL
    END AS order_to_ship_days,
    CASE 
        WHEN shipped_at IS NOT NULL AND delivered_at IS NOT NULL 
        THEN DATEDIFF('day', shipped_at, delivered_at)
        ELSE NULL
    END AS ship_to_delivery_days,
    DATEDIFF('day', order_date, delivered_at) AS total_fulfillment_days,
    CASE 
        WHEN DATEDIFF('day', order_date, delivered_at) <= 7 THEN TRUE
        ELSE FALSE
    END AS is_on_time,
    CASE 
        WHEN DATEDIFF('day', order_date, delivered_at) <= 3 THEN 'EXCELLENT'
        WHEN DATEDIFF('day', order_date, delivered_at) <= 5 THEN 'GOOD'
        WHEN DATEDIFF('day', order_date, delivered_at) <= 7 THEN 'ACCEPTABLE'
        ELSE 'SLOW'
    END AS fulfillment_tier
FROM valid_orders
ORDER BY order_id
