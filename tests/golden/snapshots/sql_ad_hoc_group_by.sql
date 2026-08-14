SELECT
  "users.state",
  COUNT(DISTINCT "users.id") AS "buyers"
FROM ref_1
GROUP BY
  "users.state"
LIMIT 50000
