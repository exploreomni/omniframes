# Omniframes

A PySpark-style Python DataFrame library for [Omni](https://omni.co). You write dataframe code;
it compiles into **governed semantic queries** and pushes as much compute as possible into Omni's
SQL execution layer.

!!! warning "Status: pre-release (`0.1.0.dev`)"
    APIs may still change, and the package is **not yet published to PyPI** — install from a
    checkout for now. See the [quickstart](quickstart.md).

```python
import omniframes as of
from omniframes import functions as F

session = (
    of.OmniSession.builder.host("acme.omni.co")
    .api_key_from_env()  # reads OMNI_API_KEY
    .get_or_create()
)

orders = session.read.topic("ecommerce", "order_items")

monthly = (
    orders.filter((F.col("users.state") == "California") & ~F.col("order_items.returned"))
    .group_by(F.col("order_items.created_at").grain("month").alias("month"))
    .agg(F.measure("order_items.total_sale_price").alias("revenue"))
    .sort(F.col("month").desc())
    .limit(12)
)

print(monthly.explain())  # shows exactly what runs remotely vs. locally
df = monthly.to_pandas()
```

That whole frame compiles to **one** governed query:

```text
== Physical plan ==
Remote [tier 1 · semantic → POST /api/v1/query/run]
  topic: order_items   model: ecommerce
  fields: [order_items.created_at[month], order_items.total_sale_price]
  group by: [order_items.created_at[month]]
  measures: [order_items.total_sale_price]
  filters: users.state = 'California' AND NOT order_items.returned
  sort: order_items.created_at[month] DESC   limit: 12   version: 9
  aliases: order_items.created_at[month] -> month, order_items.total_sale_price -> revenue
Local [pandas]
  (none — fully pushed down)
```

## Why

Notebooks and semantic layers usually disagree. A notebook wants raw rows and Python; a semantic
layer wants its own definitions of *revenue*, *active customer* and *fiscal quarter* to be the
only ones anybody uses. Pulling rows into pandas to compute `sum(sale_price)` yourself throws the
governance away — and melts the laptop on the way.

Omniframes takes the other side of that trade. `F.measure("order_items.total_sale_price")` is
**Omni's** definition of revenue, computed by Omni, with the model's join paths and row-level
security applied. Around it you still get the dataframe API: filters, grains, joins, UDFs,
`to_pandas()`. What Omni can express, Omni computes. What it cannot, omniframes computes here —
and says so, out loud, every time.

## The three tiers

At action time the planner compiles your plan to the **highest tier that can express it**,
maximizing the remote portion. `explain()` always shows where the line fell.

<div class="grid cards" markdown>

-   **Tier 1 — semantic query**

    ---

    Fully governed: model measures, topic join paths, row-level security. Dimensions plus
    measures *is* the group-by, filters become typed wire filters, and a filter on a measure is a
    genuine server-side `HAVING`.

-   **Tier 2 — OmniSQL job**

    ---

    One statement written against the **model itself** (`FROM ${topic}`, `${view.field}`,
    `${view.measure}`), planned as a governed job. Picks up ad-hoc aggregations, cross-field
    `OR`, computed columns and `HAVING` — the rows never leave the warehouse.

-   **Tier 3 — local execution**

    ---

    Arrow compute in this process, over the largest remote prefix omniframes could push down.
    UDFs, `map_pandas`, cross-frame joins and unions live here.

</div>

A frame compiles to the highest tier that can express it, and the split is always on the screen.
This mixed aggregation — one governed measure, one ad-hoc `COUNT(DISTINCT …)` — is a single
statement: the measure ref expands to its governed SQL server-side, beside the ad-hoc aggregate.

```text
== Physical plan ==
Remote [tier 2 · sql → POST /api/v1/query/run]
  topic: order_items   model: ecommerce
  sql:
    SELECT
      ${users.state},
      ${order_items.total_sale_price},
      COUNT(DISTINCT ${users.id}) AS of_expr_1
    FROM ${order_items}
    GROUP BY
      1
    LIMIT 50000
Local [pandas]
  (none — fully pushed down)
```

The governed measure is never emulated here — its definition lives in the model, and the
statement asks Omni for it by name. Put a UDF above that frame and `explain()` grows a
`Local [arrow compute]` section naming every operator that runs in this process, and the remote
query underneath shrinks to exactly what could still be pushed down.

## Design commitments

- **Lazy and immutable.** Every DataFrame is a logical plan. Nothing executes until an action
  (`collect`, `to_pandas`, `to_arrow`, `show`, `count`, `first`, `schema`).
- **Always explicit.** The library never sends an implicit limit, never silently falls back to
  local execution, and never truncates a result without a `TruncationWarning`.
- **Omni-native, PySpark-inspired.** Selecting dimensions plus a measure *is* the group-by;
  `group_by().agg()` is familiar sugar over the same semantics. This is **not** a PySpark
  drop-in — see the [mental model](mental-model.md).
- **Keys stay secret.** The API key lives inside the transport and nowhere else: it never
  appears in a repr, a log, an error message or a fixture (and there is a test for it).

## Where next

- [Quickstart](quickstart.md) — install, authenticate, run your first query.
- [Mental model](mental-model.md) — measures vs. aggregations, what runs where, limits,
  aliases, `between()` and NULLs.
- [API reference](api.md) — the full public surface.
- [Offline testing](offline-testing.md) — the wire-faithful fake and the bench dataset.

## Related projects

[`omni-python-sdk`](https://github.com/exploreomni/omni-python-sdk) is the official low-level
Python wrapper for the Omni API. Omniframes is a higher-level DataFrame front end; reach for the
SDK when you want direct endpoint access.
