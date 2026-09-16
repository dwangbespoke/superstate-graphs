WITH eligible_orders AS (
  SELECT ORDER_ID, CUSTOMER_ID, GRAND_TOTAL,
         CAST(ORDERED_AT AS DATE) AS order_date
  FROM ORDERS.ORDERS
  WHERE STATUS IN ('COMPLETED', 'DELIVERED', 'SHIPPED')
), report_window AS (
  SELECT MAX(order_date) AS last_date FROM eligible_orders
), default_addresses AS (
  SELECT CUSTOMER_ID, STATE_PROVINCE
  FROM CUSTOMER.CUSTOMER_ADDRESSES
  WHERE IS_DEFAULT_SHIPPING = true AND STATE_PROVINCE IS NOT NULL
)
SELECT
  da.STATE_PROVINCE AS state_province,
  eo.order_date,
  COUNT(DISTINCT eo.ORDER_ID) AS order_count,
  ROUND(SUM(eo.GRAND_TOTAL), 2) AS total_revenue,
  ROUND(AVG(eo.GRAND_TOTAL), 2) AS avg_order_value,
  ROUND(SUM(eo.GRAND_TOTAL) / COUNT(DISTINCT eo.CUSTOMER_ID), 2) AS revenue_per_customer,
  ROUND(COUNT(DISTINCT eo.ORDER_ID) * 1.0 / COUNT(DISTINCT eo.CUSTOMER_ID), 2) AS orders_per_customer
FROM eligible_orders eo
JOIN default_addresses da ON da.CUSTOMER_ID = eo.CUSTOMER_ID
CROSS JOIN report_window w
WHERE eo.order_date BETWEEN w.last_date - 364 AND w.last_date
GROUP BY da.STATE_PROVINCE, eo.order_date
