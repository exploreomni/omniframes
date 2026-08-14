SELECT
  ${order_items.id},
  ${order_items.created_at},
  ${order_items.returned},
  ${users.age}
FROM ${order_items}
WHERE
  (
    (
      ${order_items.created_at} >= CAST('2025-07-01' AS DATE)
      AND ${order_items.created_at} < CAST('2026-07-01' AS DATE)
    )
    OR ${order_items.returned} = TRUE
    OR (
      ${users.age} >= 18 AND ${users.age} <= 21
    )
  )
LIMIT 50000
