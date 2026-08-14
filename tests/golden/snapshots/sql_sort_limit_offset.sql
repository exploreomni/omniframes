SELECT
  ${users.state},
  SUM(${order_items.sale_price}) AS of_expr_1
FROM ${order_items}
GROUP BY
  1
ORDER BY
  2 DESC
LIMIT 10
OFFSET 5
