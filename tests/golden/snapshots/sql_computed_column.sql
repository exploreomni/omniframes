SELECT
  ${users.state},
  ${order_items.sale_price},
  ${order_items.quantity},
  ${order_items.sale_price} / ${order_items.quantity} AS of_expr_1
FROM ${order_items}
LIMIT 25
