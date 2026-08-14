SELECT
  "users.state",
  SUM("order_items.sale_price") AS "revenue",
  COUNT("order_items.id") AS "n"
FROM ref_1
GROUP BY
  "users.state"
LIMIT 50000
