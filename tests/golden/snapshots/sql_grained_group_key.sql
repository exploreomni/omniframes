SELECT
  ${order_items.created_at[month]},
  COUNT(DISTINCT ${users.id}) AS of_expr_1
FROM ${order_items}
GROUP BY
  1
LIMIT 50000
