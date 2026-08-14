SELECT
  "users.state",
  SUM("order_items.sale_price") AS "revenue"
FROM ref_1
GROUP BY
  "users.state"
ORDER BY
  "revenue" DESC
LIMIT 10
OFFSET 5
