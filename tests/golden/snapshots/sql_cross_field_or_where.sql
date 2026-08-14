SELECT
  "order_items.id",
  "users.state",
  "users.age"
FROM ref_1
WHERE
  (
    "users.state" = 'California' OR "users.age" > 60
  )
LIMIT 50000
