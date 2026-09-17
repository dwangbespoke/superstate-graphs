WITH valid_orders AS (
    SELECT
        order_id,
        order_source,
        ordered_at,
        grand_total,
        shipped_at,
        delivered_at,
        cancelled_at
    FROM ORDERS
    WHERE ordered_at >= '2024-01-01'
      AND ordered_at < '2025-01-01'
      AND TRIM(status) NOT IN ('CANCELLED', 'RETURNED', 'FAILED')
      AND delivered_at IS NOT NULL
),
order_items AS (
    SELECT
        order_id,
        SUM(CASE WHEN quantity_ordered > 0 THEN quantity_ordered ELSE 0 END) AS total_items
    FROM ORDER_LINES
    GROUP BY order_id
)
SELECT
    vo.order_id,
    vo.ordered_at AS order_date,
    vo.order_source,
    COALESCE(oi.total_items, 0) AS total_items,
    vo.grand_total AS total_revenue,
    CASE
        WHEN vo.shipped_at IS NOT NULL
        THEN vo.shipped_at - vo.ordered_at
        ELSE NULL
    END AS order_to_ship_days,
    CASE
        WHEN vo.delivered_at IS NOT NULL AND vo.shipped_at IS NOT NULL
        THEN vo.delivered_at - vo.shipped_at
        ELSE NULL
    END AS ship_to_delivery_days,
    vo.delivered_at - vo.ordered_at AS total_fulfillment_days,
    CASE
        WHEN vo.delivered_at - vo.ordered_at <= 7 THEN TRUE
        ELSE FALSE
    END AS is_on_time,
    CASE
        WHEN vo.delivered_at - vo.ordered_at <= 3 THEN 'EXCELLENT'
        WHEN vo.delivered_at - vo.ordered_at <= 5 THEN 'GOOD'
        WHEN vo.delivered_at - vo.ordered_at <= 7 THEN 'ACCEPTABLE'
        ELSE 'SLOW'
    END AS fulfillment_tier
FROM valid_orders vo
LEFT JOIN order_items oi ON vo.order_id = oi.order_id
ORDER BY vo.order_id
