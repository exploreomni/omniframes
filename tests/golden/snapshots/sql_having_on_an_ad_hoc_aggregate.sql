SELECT
  "users.state",
  COUNT(DISTINCT "users.id") AS "buyers"
FROM ref_1
GROUP BY
  "users.state"
HAVING
  COUNT(DISTINCT "users.id") > 25
LIMIT 50000
