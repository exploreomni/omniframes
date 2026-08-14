SELECT
  "order_items.id",
  "users.state",
  "users.name"
FROM ref_1
WHERE
  (
    LOWER("users.state") LIKE LOWER('%cal%') ESCAPE '!'
    OR "users.name" LIKE 'A!%%' ESCAPE '!'
  )
LIMIT 50000
