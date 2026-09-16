WITH dated_sales AS (
    SELECT f.*, d.FULL_DATE AS sale_date
    FROM ANALYTICS.FACT_SALES f
    JOIN ANALYTICS.DIM_DATE d ON d.DATE_KEY = f.DATE_KEY
    WHERE f.CUSTOMER_KEY IS NOT NULL
), ranked_first_sales AS (
    SELECT CUSTOMER_KEY, CHANNEL_KEY, sale_date,
           ROW_NUMBER() OVER (
               PARTITION BY CUSTOMER_KEY
               ORDER BY sale_date, TIME_KEY NULLS LAST, ORDER_ID NULLS LAST, SALE_KEY
           ) AS first_sale_rank
    FROM dated_sales
), customer_cohorts AS (
    SELECT CUSTOMER_KEY, CHANNEL_KEY AS acquisition_channel_key, sale_date AS cohort_date
    FROM ranked_first_sales
    WHERE first_sale_rank = 1
      AND sale_date <= (SELECT MAX(sale_date) FROM dated_sales) - INTERVAL '90 days'
), cohort_activity AS (
    SELECT c.CUSTOMER_KEY, c.acquisition_channel_key, c.cohort_date,
           s.sale_date, s.ORDER_ID, s.TOTAL_AMOUNT, s.PROFIT_AMOUNT, s.DISCOUNT_AMOUNT
    FROM customer_cohorts c
    JOIN dated_sales s ON s.CUSTOMER_KEY = c.CUSTOMER_KEY
                     AND s.sale_date BETWEEN c.cohort_date AND c.cohort_date + INTERVAL '90 days'
), cohort_metrics AS (
    SELECT cohort_date, acquisition_channel_key,
           COUNT(DISTINCT CUSTOMER_KEY) AS cohort_size,
           COUNT(DISTINCT ORDER_ID) AS order_count,
           COUNT(DISTINCT CASE WHEN sale_date = cohort_date THEN CUSTOMER_KEY END) AS day_0_customers,
           COUNT(DISTINCT CASE WHEN sale_date = cohort_date + INTERVAL '7 days' THEN CUSTOMER_KEY END) AS day_7_customers,
           COUNT(DISTINCT CASE WHEN sale_date = cohort_date + INTERVAL '30 days' THEN CUSTOMER_KEY END) AS day_30_customers,
           COUNT(DISTINCT CASE WHEN sale_date = cohort_date + INTERVAL '90 days' THEN CUSTOMER_KEY END) AS day_90_customers,
           SUM(TOTAL_AMOUNT) AS revenue,
           SUM(PROFIT_AMOUNT) AS profit,
           SUM(DISCOUNT_AMOUNT) AS discount_amount
    FROM cohort_activity
    GROUP BY cohort_date, acquisition_channel_key
)
SELECT m.cohort_date,
       CAST(DATE_TRUNC('month', m.cohort_date) AS DATE) AS month_start,
       m.acquisition_channel_key AS channel_key,
       COALESCE(c.CHANNEL_CODE, 'UNKNOWN') AS channel_code,
       m.cohort_size, m.order_count,
       m.day_0_customers, m.day_7_customers, m.day_30_customers, m.day_90_customers,
       CAST(ROUND(m.revenue, 2) AS DECIMAL(18,2)) AS revenue,
       CAST(ROUND(m.profit, 2) AS DECIMAL(18,2)) AS profit,
       CAST(ROUND(m.discount_amount, 2) AS DECIMAL(18,2)) AS discount_amount,
       CAST(ROUND(m.profit / NULLIF(m.revenue, 0), 6) AS DECIMAL(18,6)) AS margin_rate,
       CAST(ROUND(m.revenue / NULLIF(m.order_count, 0), 2) AS DECIMAL(18,2)) AS avg_order_value,
       CAST(ROUND(m.discount_amount / NULLIF(m.revenue, 0), 6) AS DECIMAL(18,6)) AS discount_rate,
       CAST(ROUND(m.day_0_customers * 1.0 / NULLIF(m.cohort_size, 0), 6) AS DECIMAL(18,6)) AS day_0_retention,
       CAST(ROUND(m.day_7_customers * 1.0 / NULLIF(m.cohort_size, 0), 6) AS DECIMAL(18,6)) AS day_7_retention,
       CAST(ROUND(m.day_30_customers * 1.0 / NULLIF(m.cohort_size, 0), 6) AS DECIMAL(18,6)) AS day_30_retention,
       CAST(ROUND(m.day_90_customers * 1.0 / NULLIF(m.cohort_size, 0), 6) AS DECIMAL(18,6)) AS day_90_retention
FROM cohort_metrics m
LEFT JOIN ANALYTICS.DIM_CHANNEL c ON c.CHANNEL_KEY = m.acquisition_channel_key
