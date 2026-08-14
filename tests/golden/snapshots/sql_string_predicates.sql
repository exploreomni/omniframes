SELECT
  ${order_items.id},
  ${users.state},
  ${users.name}
FROM ${order_items}
WHERE
  (
    LOWER(${users.state}) LIKE LOWER('%cal%') ESCAPE '!'
    OR ${users.name} LIKE 'A!%%' ESCAPE '!'
  )
LIMIT 50000
