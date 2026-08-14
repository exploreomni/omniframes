SELECT
  ${users.state},
  SUM(${order_items.sale_price}) AS of_expr_1,
  COUNT(${order_items.id}) AS of_expr_2
FROM ${order_items}
WHERE
  ${order_items.status} = 'complete'
GROUP BY
  1
LIMIT 50000
