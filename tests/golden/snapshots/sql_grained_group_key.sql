SELECT
  "order_items.created_at[month]" AS "month",
  COUNT(DISTINCT "users.id") AS "buyers"
FROM ref_1
GROUP BY
  "order_items.created_at[month]"
LIMIT 50000
