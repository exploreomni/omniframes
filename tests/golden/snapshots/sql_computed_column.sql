SELECT
  "users.state",
  "order_items.sale_price",
  "order_items.quantity",
  "order_items.sale_price" / "order_items.quantity" AS "unit_price"
FROM ref_1
LIMIT 25
