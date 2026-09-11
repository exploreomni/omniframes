import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Omniframes — the guided tour

    **Dataframe code in, governed semantic queries out.** This notebook walks the whole feature
    surface of [omniframes](https://github.com/exploreomni/omniframes) — read, aggregate, `HAVING`,
    mixed aggregation, UDFs, raw SQL, cross-frame joins, totals, writers — and prints
    `explain()` after every step, so you can always see *which tier* did the work:

    | tier | what it is | where it runs |
    |---|---|---|
    | **1 · semantic** | a governed Omni query: model measures, topic join paths, row-level security | Omni |
    | **2 · sql** | warehouse SQL over an embedded *governed* sub-query | the warehouse |
    | **3 · local** | Arrow compute over the largest remote prefix omniframes could push down | this process |

    ## Two backends, one notebook

    The setup cell below picks its backend from the environment:

    * **Offline (the default).** With no credentials set, it runs against `FakeOmniAPI` — an
      in-process, wire-faithful fake of the Omni query API, executing real DuckDB queries over the
      deterministic bench dataset checked into `tests/data/bench/`. No network, no key, real answers.
    * **Live.** Set `OMNI_BASE_URL` and `OMNI_API_KEY` and every cell below runs against your org,
      unchanged. Point it at a model with `OMNI_MODEL` / `OMNI_TOPIC` (defaults: `bench_ecommerce` /
      `order_items`), and — for the raw-SQL cells only — name the warehouse schema holding the bench
      tables with `OMNI_BENCH_SCHEMA` (e.g. `OMNIFRAMES_BENCH.`, trailing dot included).

    ```bash
    export OMNI_BASE_URL="https://acme.omniapp.co"
    export OMNI_API_KEY="…"                 # never paste a key into a cell
    export OMNI_BENCH_SCHEMA="OMNIFRAMES_BENCH."
    ```

    That parity is the point: the offline fake serves exactly the model documented in
    `docs/bench_omni_model.md`, so an expectation that holds here holds live.
    """)
    return


@app.cell
def _():
    """Build a session — against a live org if credentials are set, else against the fake."""

    import os
    import sys
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv()

    import omniframes as of
    from omniframes import functions as F

    MODEL = os.environ.get("OMNI_MODEL", "Postgres")
    TOPIC = os.environ.get("OMNI_TOPIC", "order_items")
    SCHEMA = os.environ.get("OMNI_BENCH_SCHEMA", "")  # raw-SQL cells only; "" offline
    LIVE = bool(os.environ.get("OMNI_BASE_URL") and os.environ.get("OMNI_API_KEY"))

    if LIVE:
        # The key is read from OMNI_API_KEY and lives inside the transport — it never appears in a
        # repr, a log line or an error message, and it must never appear in a notebook cell either.
        session = of.OmniSession.builder.base_url_from_env().api_key_from_env().get_or_create()
        backend = f"live org at {os.environ['OMNI_BASE_URL']}"
    else:
        import httpx

        from omniframes.transport import HttpTransport

        # tests/ is a dev-only package (it ships with the repository, not with the wheel), so put
        # the checkout root on sys.path.  Run this notebook from inside the repository.
        here = Path.cwd().resolve()
        root = next(
            (
                candidate
                for candidate in (here, *here.parents)
                if (candidate / "pyproject.toml").exists()
                and (candidate / "tests" / "fakes").is_dir()
            ),
            None,
        )
        if root is None:
            raise RuntimeError(
                f"no omniframes checkout found at or above {here}. Either run this notebook from "
                "inside the repository, or set OMNI_BASE_URL and OMNI_API_KEY to use a live org."
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from tests.fakes import DEFAULT_TOKEN, FakeOmniAPI

        BASE_URL = "https://bench.omniapp.co"
        handler = FakeOmniAPI(model_name=MODEL)
        client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
        session = (
            of.OmniSession.builder.base_url(BASE_URL)
            .transport(HttpTransport(base_url=BASE_URL, api_key=DEFAULT_TOKEN, client=client))
            .get_or_create()
        )
        backend = "in-process FakeOmniAPI (no network, no credentials)"

    print(f"omniframes {of.__version__} → {backend}")
    print(f"model={MODEL!r}  topic={TOPIC!r}")

    # Building a session performs no I/O at all; the whoami preflight runs lazily, once, before the
    # first call that needs the network.
    orders = session.read.topic(MODEL, TOPIC)

    REVENUE = "order_items.sale_price_sum"  # governed measure: SUM(sale_price)
    ORDERS = "order_items.count"  # governed measure: COUNT(*)
    return F, MODEL, ORDERS, Path, REVENUE, SCHEMA, orders, session


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1 · Read a topic

    A **topic** carries the model's join paths, so fields from any joined view (`users.*`,
    `products.*`) are selectable without saying how they join. Everything below is lazy: nothing
    touches the network until `.show()`.
    """)
    return


@app.cell
def _(orders):
    orders.select("order_items.id", "users.state").limit(100).show()
    return


@app.cell
def _(F, orders):
    recent = (
        orders.select("order_items.id", "users.state", "order_items.status")
        .filter(F.col("order_items.status").isin("Complete", "Shipped"))
        .sort(F.col("order_items.id").desc())
        .limit(5)
    )
    recent.show()
    return (recent,)


@app.cell
def _(recent):
    print(recent.explain())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One governed query, `(none — fully pushed down)`: the projection, the `IN` filter, the sort and
    the limit all rode the wire. Note `limit: 5` — omniframes **always** sends an explicit limit
    (`50000` by default), so nothing ever streams unbounded by accident.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1b · A plain filter, pushed down

    No aggregation, no join — just `.filter()` on a raw scan. The predicate still travels as the
    query's `WHERE`, not a local `pandas` mask: `explain()` is the proof, not a claim.
    """)
    return


@app.cell(hide_code=True)
def _(F, orders):
    california = orders.select("order_items.id", "users.state", "order_items.status").filter(
        F.col("users.state") == "California"
    )
    california.show(5)
    return (california,)


@app.cell(hide_code=True)
def _(california):
    print(california.explain())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One governed query, `(none — fully pushed down)`: the `WHERE users.state = 'California'`
    predicate rode the wire with the projection and the default limit. Nothing about this filter ran
    in this process.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2 · Aggregate with governed measures

    `F.measure(...)` references a measure **defined in the Omni model**. Omni computes it, under its
    own definition, with the model's joins and security applied — omniframes never emulates one
    locally.

    `group_by().agg()` is sugar: in Omni, selecting dimensions alongside measures *is* the group-by,
    and `orders.select("users.state", F.measure(REVENUE))` compiles to the identical query.

    `.grain("month")` buckets a timestamp server-side; `.alias(...)` renames client-side (the wire
    has no aliasing) and can be used in the `sort` that follows.
    """)
    return


@app.cell
def _(F, ORDERS, REVENUE, orders):
    month = F.col("order_items.created_at").grain("month").alias("month")

    monthly = (
        orders.group_by(month, F.col("users.state").alias("state"))
        .agg(F.measure(REVENUE).alias("revenue"), F.measure(ORDERS).alias("orders"))
        .filter(F.col("users.state").isin("California", "New York"))
        .sort(F.col("month").desc(), F.col("state"))
        .limit(6)
    )
    monthly.show()
    return (monthly,)


@app.cell
def _(monthly):
    print(monthly.explain())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Still tier 1. The `aliases:` line is the whole aliasing story: `order_items.created_at[month]`
    goes on the wire, `month` comes back to you.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3 · Filter *after* aggregating — a real `HAVING`

    A predicate on a governed measure is not a client-side pass. It compiles to a measure-keyed wire
    filter, which the server applies after the `GROUP BY` — a genuine `HAVING`. It works whether the
    measure is selected or not, and whether you write it before or after the `group_by()`.
    """)
    return


@app.cell
def _(F, REVENUE, orders):
    big_states = (
        orders.group_by(F.col("users.state").alias("state"))
        .agg(F.measure(REVENUE).alias("revenue"))
        .filter(F.measure(REVENUE) > 600_000)
        .sort(F.col("revenue").desc())
    )
    big_states.show()
    return (big_states,)


@app.cell
def _(big_states):
    print(big_states.explain())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `having:` line is the server's, not ours. Three states cleared $600k; 423 were grouped.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4 · Mixed aggregation — one governed measure, one ad-hoc aggregate

    Mix a **governed measure** with an **ad-hoc aggregation** (`F.count_distinct`, which has no
    server-side definition) in one `agg()`. A governed measure is a legal select item in OmniSQL
    (`\${view.measure}` expands server-side), so it sits beside `COUNT(DISTINCT ...)` in the same
    statement — the common case is **one tier-2 request**, not a split.

    The old two-step plan (tier-1 measure + a raw tier-2/local scan, aligned with a local join) still
    exists, but only as the fallback for the rare shape tier 2 declines to render. `explain()` always
    says which path ran.
    """)
    return


@app.cell
def _(F, REVENUE, orders):
    mixed = orders.group_by(F.col("users.state").alias("state")).agg(
        F.measure(REVENUE).alias("revenue"),
        F.count_distinct("users.id").alias("buyers"),
    )
    mixed.sort(F.col("buyers").desc()).show(5)
    return (mixed,)


@app.cell
def _(mixed):
    print(mixed.explain())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One tier-2 statement: the governed `\${order_items.sale_price_sum}` token sits beside
    `COUNT(DISTINCT \${users.id})` in the same `SELECT`, both under one `GROUP BY`. No align-join,
    no second remote step — the split only appears when tier 2 declines the shape.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5 · A UDF — the honest local fallback

    `F.udf(fn)` wraps any Python function and applies it row by row. Nothing about a UDF is
    expressible on the wire, so **everything from the UDF up runs here** — and `explain()` says so
    before you pay for it. The query underneath is still pushed down as far as it goes.
    """)
    return


@app.cell
def _(F, REVENUE, orders):
    def region(state):
        """Map a state onto a coarse region — the sort of thing no semantic model has."""
        if state is None:
            return "unknown"
        if state in {"California", "Oregon", "Washington"}:
            return "west"
        if state in {"New York", "New Jersey", "Massachusetts"}:
            return "east"
        return "other"

    regions = (
        orders.group_by(F.col("users.state").alias("state"))
        .agg(F.measure(REVENUE).alias("revenue"))
        .with_column("region", F.udf(region)("state"))
        .sort(F.col("revenue").desc())
        .limit(6)
    )
    regions.show()
    return (regions,)


@app.cell
def _(regions):
    print(regions.explain())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Local [arrow compute]` names the function (`region(state)`), the sort and the limit that had to
    follow it down. That is the trade the plan is asking you to accept: 423 aggregated rows come back
    and Python touches each one. Had the UDF sat over a raw scan instead, the remote step would say
    `limit: 50000` — same honesty, much bigger bill.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6 · Raw SQL as an entry point

    `session.read.sql(...)` sends **your** statement on the model's connection (needs `QUERY_SQL`).
    It always travels with `rewriteSql: false` — the marker without which the server silently ignores
    the SQL and answers a question nobody asked.

    The SQL is **opaque**: omniframes did not write it and will never re-derive it, so every
    operation on top runs in the local engine over its result. The plan says exactly that.
    """)
    return


@app.cell
def _(F, MODEL, SCHEMA, session):
    CATEGORY_SQL = f"""
    SELECT ii.product_category AS category,
           COUNT(*)            AS items,
           SUM(oi.sale_price)  AS revenue
    FROM {SCHEMA}order_items oi
    LEFT JOIN {SCHEMA}inventory_items ii ON ii.id = oi.inventory_item_id
    GROUP BY 1
    """.strip()

    categories = (
        session.read.sql(MODEL, CATEGORY_SQL)
        .select("category", "items", "revenue")
        .filter(F.col("category").is_not_null())
        .sort(F.col("revenue").desc())
        .limit(5)
    )
    categories.show()
    return (categories,)


@app.cell
def _(categories):
    print(categories.explain())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `rewriteSql: false` is on the plan line, and the projection, filter, sort and limit are all under
    `Local [arrow compute]` — none of them was pushed into somebody else's SQL.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 7 · Cross-frame join — with SQL NULL semantics

    Two frames, two independent queries (they may even be different tiers, as here: a governed
    aggregate joined to a raw-SQL job), combined in this process — the query API takes one query at a
    time.

    The join follows **SQL**: a NULL key matches nothing, not even another NULL. If the governed side
    carries a NULL-state group (buyers with no state, or order items whose `user_id` matches no user
    at all — the topic's joins are LEFT joins, so this can happen), an `inner` join drops it and a
    `left` join keeps it, unmatched. If there is no such group — a clean org enforcing referential
    integrity, as below — `inner` and `left` agree, which is the same rule producing no visible
    effect rather than a different one.
    """)
    return


@app.cell
def _(F, MODEL, REVENUE, SCHEMA, orders, session):
    RESIDENTS_SQL = f"""
    SELECT u.state  AS "users.state",
           COUNT(*) AS residents
    FROM {SCHEMA}users u
    WHERE u.state IS NOT NULL
    GROUP BY 1
    """.strip()

    states_sql = session.read.sql(MODEL, RESIDENTS_SQL)
    revenue_by_state = orders.group_by("users.state").agg(F.measure(REVENUE).alias("revenue"))

    inner = revenue_by_state.join(states_sql, "users.state", "inner")
    left = revenue_by_state.join(states_sql, "users.state", "left")

    inner.sort(F.col("revenue").desc()).show(5)
    print(
        f"grouped states: {revenue_by_state.count()}   inner: {inner.count()}   left: {left.count()}"
    )
    return (inner,)


@app.cell
def _(inner):
    print(inner.explain())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This org has no NULL-state group to lose: `grouped`, `inner` and `left` all print the same count.
    Nothing was silently reconciled here either — there was simply nothing to reconcile. Point a NULL
    key at a real orphan (the offline bench dataset has one) and the counts diverge exactly as the SQL
    standard says they must.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 8 · Totals

    `with_totals()` asks Omni for its own grand-total row and appends it with a derived `row_type`
    column. It **re-aggregates over every row the query touched** — post-filter, pre-limit — so it is
    not the sum of the values above it. That is the entire reason to ask the server for it rather
    than summing the page yourself.
    """)
    return


@app.cell
def _(F, ORDERS, REVENUE, orders):
    totals = (
        orders.group_by(F.col("order_items.status").alias("status"))
        .agg(F.measure(REVENUE).alias("revenue"), F.measure(ORDERS).alias("orders"))
        .sort(F.col("revenue").desc())
        .with_totals()
    )
    totals.show()
    return (totals,)


@app.cell
def _(totals):
    print(totals.explain())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `totals: column_totals [::total::]` on the wire; `row_type` derived on this side (the run
    endpoint has no such column — it flags totals rows with reserved indicator columns that never
    reach you).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 9 · Write it out

    `df.write` runs the query and writes the **normalized, aliased** result — Parquet keeps the
    decimal columns exact, which is why a frame of money should not be a CSV.
    """)
    return


@app.cell
def _(Path, mixed):
    import tempfile

    import pyarrow.parquet as pq

    out = Path(tempfile.mkdtemp(prefix="omniframes-demo-")) / "revenue_by_state.parquet"
    mixed.write.parquet(out)

    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    print(pq.read_table(out).schema)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `df.write.csv(path)` is the other writer; both take the small set of options that matter.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 10 · Hand off to pandas

    `to_pandas()` is the exit. (`to_arrow()` / `collect()` give you the Arrow table zero-copy, and
    `to_polars()` is one `pip install 'omniframes[polars]'` away.)
    """)
    return


@app.cell
def _(F, mixed):
    frame = mixed.sort(F.col("revenue").desc()).limit(8).to_pandas()

    print(frame.dtypes.to_string())
    frame
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 11 · A computed column that rides the wire

    `with_column()` is the same call you saw put a Python UDF into the plan in section 5 — and this time
    the work does *not* come home. An arithmetic expression over fields and measures is something tier 2
    can render, so it is compiled into the `SELECT` beside the governed measures rather than evaluated
    here.

    The base frame is the one section 12 clusters: revenue, items and distinct buyers per market, with
    the small markets already dropped by a `HAVING`.
    """)
    return


@app.cell
def _(F, ORDERS, REVENUE, orders):
    market_features = (
        orders.group_by(F.col("users.state").alias("state"))
        .agg(
            F.measure(REVENUE).alias("revenue"),
            F.measure(ORDERS).alias("items"),
            F.count_distinct("users.id").alias("buyers"),
        )
        .filter(F.measure(REVENUE) > 100_000)
        .sort(F.col("revenue").desc())
    )
    market_features.show(5)
    return (market_features,)


@app.cell
def _(market_features):
    print(market_features.explain())
    return


@app.cell
def _(F, market_features):
    market_economics = market_features.with_column(
        "aov", F.col("revenue") / F.col("items")
    ).with_column(
        # ``* 1.0`` keeps this out of the warehouse's integer division — both operands are counts.
        "items_per_buyer",
        F.col("items") * 1.0 / F.col("buyers"),
    )
    market_economics.show(5)
    return (market_economics,)


@app.cell
def _(market_economics):
    print(market_economics.explain())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Both columns are inside the one remote statement — `${order_items.sale_price_sum} /
    ${order_items.count}` and `${order_items.count} * 1.0 / COUNT(DISTINCT ${users.id})` — and the local
    plan is still `(none — fully pushed down)`. Nothing evaluated a ratio in this process.

    A computed column is a first-class field once it exists, so a filter on one pushes down too. This is
    where `explain(analyze=True)` earns its keep: it spends a `planOnly` round trip to get back the
    **warehouse SQL** Omni actually compiled, so you can see what the expression became.
    """)
    return


@app.cell
def _(F, market_economics):
    premium_markets = market_economics.filter(F.col("aov") > 46)

    print(premium_markets.explain(analyze=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There it is in the warehouse's own SQL: the predicate on `aov` became
    `HAVING COALESCE(SUM(...), 0) / COUNT(*) > 46`, sitting next to the measure `HAVING` from the base
    frame. Two filters written a page apart, in different vocabularies, compiled into one clause.

    So `with_column` is not a tier — it is an expression, and the tier depends on what you put in it.
    `F.udf(...)` is unrenderable and drags everything above it local (section 5); arithmetic over
    measures is renderable and travels. `explain()` is how you tell the two apart before you pay for the
    difference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 12 · Clustering — the local tier, on purpose

    Everything so far pushed work *down*. This section does the opposite, deliberately: k-means has no
    wire representation at all, so the split is the whole design. Omni does the part that touches every
    row — grouping 400+ markets, applying the governed revenue measure, dropping the small ones, working
    out each market's basket economics — and hands back a few dozen rows. The clustering runs here, on
    that summary, in `numpy`.

    The features are exactly the `market_economics` columns the warehouse just computed, plus
    `log10(revenue)` so market *scale* enters as one well-behaved axis instead of letting California
    dominate every distance. Standardising is not optional: dollars, items and items-per-buyer live on
    wildly different scales, and k-means only knows Euclidean distance.

    That is the shape of every honest ML step on a governed warehouse: aggregate remotely, model
    locally, and keep the boundary visible in `explain()` rather than hidden behind a helper.
    """)
    return


@app.cell
def _(market_economics):
    import numpy as np
    import pandas as pd

    def kmeans(X, k, *, seed=0, iters=100):
        """Lloyd's algorithm with a k-means++ seed — deterministic for a fixed seed.

        Returns the labels and the centroids: section 13 fits on a sample and assigns everything.
        """
        rng = np.random.default_rng(seed)
        centers = [X[rng.integers(len(X))]]
        for _ in range(k - 1):
            d2 = ((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1).min(1)
            centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])

        C = np.array(centers)
        for _ in range(iters):
            labels = assign(X, C)
            moved = np.array(
                [X[labels == j].mean(0) if (labels == j).any() else C[j] for j in range(k)]
            )
            if np.allclose(moved, C):
                break
            C = moved
        return assign(X, C), C

    def assign(X, C):
        """Nearest centroid for every row, in chunks so the distance block stays small."""
        return np.concatenate(
            [
                ((X[i : i + 50_000, None, :] - C[None]) ** 2).sum(-1).argmin(1)
                for i in range(0, len(X), 50_000)
            ]
        )

    clustered = market_economics.to_pandas()

    # The wire keeps money and ratios exact (``Decimal``); k-means wants machine floats.
    _features = pd.DataFrame(
        {
            "scale": np.log10(clustered["revenue"].astype(float)),
            "aov": clustered["aov"].astype(float),
            "items_per_buyer": clustered["items_per_buyer"].astype(float),
        }
    )
    _z = (_features - _features.mean()) / _features.std(ddof=0)
    clustered["cluster"], _ = kmeans(_z.to_numpy(), 3, seed=0)

    print(f"{len(clustered)} markets → {clustered['cluster'].nunique()} clusters")
    clustered
    return assign, clustered, kmeans, np, pd


@app.cell
def _(clustered):
    segments = (
        clustered.groupby("cluster")
        .agg(
            markets=("state", "size"),
            revenue=("revenue", "mean"),
            aov=("aov", lambda s: s.astype(float).mean()),
            items_per_buyer=("items_per_buyer", lambda s: s.astype(float).mean()),
            examples=("state", lambda s: ", ".join(s.head(3))),
        )
        .round(2)
    )
    segments
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Three segments fall out of a table this process never had to scan: a small head of very large
    markets, and the rest split by basket economics rather than size — average order value and items per
    buyer. Read that split honestly. Separation is only ever as strong as the features you fed it, and
    on a demo dataset a dollar of average order value is most of what distinguishes two of these groups.
    k-means will always return `k` clusters; whether they mean anything is a question about the data, not
    about the algorithm.

    Change `k`, change the features, re-run — the remote half is unchanged and cheap, because the
    expensive part already happened server-side. The labels are ordinary `pandas` from here: join them
    back, write them out with `df.write`, or feed them to whatever model you actually came for.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 13 · The same algorithm on individual order lines

    Section 12 clustered 26 markets — a table small enough that where the work happened barely
    mattered. This one clusters the **order lines themselves**: every completed, shipped or in-flight
    item, one row each, joined to its product and its buyer. Omni does the joins, the status filter and
    the two computed columns; what crosses the wire is a few hundred thousand rows, and that is the
    whole point of looking at the plan first.

    Note `limit(None)`. omniframes sends an explicit limit on every query — 50 000 by default — and
    warns when a result comes back exactly that size, because a silently truncated frame is the
    expensive kind of wrong. Asking for everything is allowed; it just has to be *asked* for.
    """)
    return


@app.cell
def _(F, orders):
    line_items = (
        orders.select(
            F.col("order_items.sale_price").alias("sale_price"),
            F.col("products.retail_price").alias("retail_price"),
            F.col("products.department").alias("department"),
            F.col("products.category").alias("category"),
            # The buyer, as dimensions — not as a person.  The topic also carries users.email and
            # users.full_name; a clustering frame has no business pulling either into this process.
            F.col("users.age").alias("age"),
            F.col("users.gender").alias("gender"),
            F.col("users.traffic_source").alias("traffic_source"),
            F.col("users.country").alias("country"),
            F.col("users.created_at").grain("month").alias("signup_month"),
        )
        .filter(F.col("order_items.status").isin("Complete", "Shipped", "Processing"))
        .with_column("margin", F.col("order_items.sale_price") - F.col("products.cost"))
        .limit(None)
    )
    line_items.show(5)
    return (line_items,)


@app.cell
def _(line_items):
    print(line_items.explain())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One tier-2 statement again: the three-view join comes from the topic, the status filter is a `WHERE`,
    and `margin` is computed in the warehouse — the frame that arrives is already the feature table.
    Fetching it is the expensive line in this notebook, so it gets its own cell and its own clock.
    """)
    return


@app.cell
def _(line_items):
    import time

    _started = time.perf_counter()
    lines_pd = line_items.to_pandas()
    _elapsed = time.perf_counter() - _started

    print(f"{len(lines_pd):,} order lines in {_elapsed:.1f}s")
    lines_pd.head()
    return (lines_pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The feature set is one order line, described from both ends. **What was sold**: price (log-scaled —
    order values are heavy-tailed, and one $999 line should not define an axis) and how much of it was
    margin. **Who bought it**: age, how long they had been a customer when this data was cut, how they
    found the shop, and where they are. `signup_month` arrives as a real timestamp because
    `.grain("month")` bucketed it server-side, the same call section 2 used on order dates.

    The buyer dimensions are categorical, so they are one-hot encoded — and then deliberately *not*
    standardised. Z-scoring a rare level (`traffic_source = Email`, 3% of lines) inflates it into the
    loudest signal in the space; left as 0/1 each level contributes at most one unit of distance while
    the numeric axes are standardised to comparable spread. k-means on mixed data is an approximation
    whatever you do — k-prototypes exists for exactly this reason — so the compromise is better made in
    the open than buried in a scaler.

    At this size the fit gets one more concession: centroids are fitted on a seeded sample and then
    every row is assigned to its nearest one. The ordinary mini-batch bargain — it changes the runtime,
    not the picture.
    """)
    return


@app.cell
def _(assign, kmeans, lines_pd, np, pd):
    NUMERIC = ["log_price", "margin_rate", "age", "tenure_months"]

    _price = lines_pd["sale_price"].astype(float)
    line_features = pd.DataFrame(
        {
            "log_price": np.log10(_price),
            "margin_rate": lines_pd["margin"].astype(float) / _price,
            "age": lines_pd["age"].astype(float),
            # Tenure as of the newest signup in the data, so the axis does not drift with the clock.
            "tenure_months": (lines_pd["signup_month"].max() - lines_pd["signup_month"]).dt.days
            / 30.44,
        }
    ).join(
        pd.get_dummies(
            lines_pd[["gender", "traffic_source", "country"]], prefix_sep="=", dtype=float
        )
    )

    _scaled = line_features.copy()
    _scaled[NUMERIC] = (_scaled[NUMERIC] - _scaled[NUMERIC].mean()) / _scaled[NUMERIC].std(ddof=0)
    line_matrix = _scaled.to_numpy()

    _sample = np.random.default_rng(0).choice(len(line_matrix), size=25_000, replace=False)
    _, line_centers = kmeans(line_matrix[_sample], 4, seed=0)

    line_clusters = lines_pd.assign(cluster=assign(line_matrix, line_centers))
    print(f"{len(line_features):,} lines × {line_features.shape[1]} features")
    print(line_clusters["cluster"].value_counts().sort_index().to_string())
    line_clusters.head()
    return line_centers, line_clusters, line_features, line_matrix


@app.cell
def _(line_clusters, line_features):
    line_segments = (
        line_clusters.assign(
            margin_rate=line_features["margin_rate"],
            tenure_months=line_features["tenure_months"],
            female=line_features["gender=Female"],
            usa=line_features["country=USA"],
            search=line_features["traffic_source=Search"],
        )
        .groupby("cluster")
        .agg(
            lines=("age", "size"),
            sale_price=("sale_price", "mean"),
            margin_rate=("margin_rate", "mean"),
            age=("age", "mean"),
            tenure_months=("tenure_months", "mean"),
            female_share=("female", "mean"),
            usa_share=("usa", "mean"),
            search_share=("search", "mean"),
            top_categories=("category", lambda s: ", ".join(s.value_counts().head(2).index)),
        )
        .round(2)
    )
    line_segments
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Four kinds of order line, and the buyer dimensions are doing visible work: one segment is defined by
    tenure (customers of a month or two, still buying at full price), another by age, another by a
    low-price, female-skewed accessories basket. None of that was a category the model was asked to
    define in advance.

    Two of the added dimensions barely move: `country` and `traffic_source` sit at roughly their
    population mix inside every cluster. That is a finding, not a failure — the encoding worked, and
    this shop's traffic mix simply does not vary by what people buy. A feature that separates nothing is
    worth knowing about before it ends up in a slide.

    The bill is the other lesson. Section 12 aggregated first and clustered 26 rows in milliseconds;
    this section moved a few hundred thousand rows because the question genuinely needed them, and the
    plan said so before a byte crossed the wire. That is the division of labour: Omni owns the joins,
    the governance and the arithmetic SQL can express; this process owns what it cannot — and pays,
    visibly, for every row it asked to see.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 14 · Seeing it — PCA, and an honest picture

    Thirteen features is four more dimensions than anyone can look at. **PCA** fixes that the cheap
    way: rotate the space so the first axis carries as much variance as it can, the second as much of
    what is left, and plot those two. It is a `numpy` one-liner — `np.linalg.svd` on the centered
    matrix — and it needs no library this notebook did not already have.

    Read the loadings before the picture. They say what the axes *mean*; without them a PCA scatter is
    a decorative blob.
    """)
    return


@app.cell
def _(line_centers, line_features, line_matrix, np, pd):
    _center = line_matrix.mean(0)
    _centered = line_matrix - _center
    _, _singular, _axes = np.linalg.svd(_centered, full_matrices=False)

    pca_axes = _axes[:2]
    pca_explained = (_singular**2 / (_singular**2).sum())[:2]
    line_xy = _centered @ pca_axes.T
    centroid_xy = (line_centers - _center) @ pca_axes.T

    print(
        f"PC1 {pca_explained[0]:.0%} + PC2 {pca_explained[1]:.0%} = "
        f"{pca_explained.sum():.0%} of the variance in 13 features"
    )

    pca_loadings = pd.DataFrame(
        pca_axes.T, index=line_features.columns, columns=["PC1", "PC2"]
    ).round(2)
    pca_loadings
    return centroid_xy, line_xy, pca_explained, pca_loadings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `PC1` is the **basket** axis: price loads on it one way, margin rate the other — cheap,
    high-margin accessories at one end, expensive, thinner-margin outerwear at the other. `PC2` is the
    **buyer** axis: age and tenure load on it together. Every one-hot column loads at roughly zero,
    which is the same finding the segment table gave, now visible as geometry.

    Two axes hold 40% of the variance of thirteen features — say that out loud rather than cropping it
    out. What follows is a projection with more than half the variation flattened away, so it shows
    where clusters sit relative to each other, not how cleanly they separate.

    One picture, four panels: each cluster against the same grey backdrop of every other line. Four
    overplotted colours on one pair of axes would be a blob — and at four categorical hues a scatter
    cannot clear the colourblind-separation floor for every pair anyway, so small multiples are both
    the more readable and the more accessible choice. The table above is the same data in text.
    """)
    return


@app.cell
def _(
    centroid_xy,
    line_centers,
    line_clusters,
    line_xy,
    mo,
    np,
    pca_explained,
    pca_loadings,
):
    def _dots(points, box, extent):
        """Many points as one path — 'M x y h .01' dots, far lighter than one <circle> each."""
        (x0, y0, w, h) = box
        (xlo, xhi, ylo, yhi) = extent
        sx, sy = w / (xhi - xlo), h / (yhi - ylo)
        return "".join(
            f"M{x0 + (px - xlo) * sx:.1f} {y0 + h - (py - ylo) * sy:.1f}h.01"
            for px, py in points
            if xlo <= px <= xhi and ylo <= py <= yhi
        )

    def _pca_figure(xy, labels, centers, loadings, explained, *, sample=12_000, seed=0):
        rng = np.random.default_rng(seed)
        take = rng.choice(len(xy), size=min(sample, len(xy)), replace=False)
        shown, shown_labels = xy[take], labels[take]

        pad = 0.06
        (xlo, xhi), (ylo, yhi) = (np.percentile(xy[:, i], [0.5, 99.5]) for i in (0, 1))
        xlo, xhi = xlo - pad * (xhi - xlo), xhi + pad * (xhi - xlo)
        ylo, yhi = ylo - pad * (yhi - ylo), yhi + pad * (yhi - ylo)
        extent = (xlo, xhi, ylo, yhi)

        size, gap, left, top = 196, 14, 40, 24
        panels = []
        for k in range(len(centers)):
            x0 = left + k * (size + gap)
            box = (x0, top, size, size)
            share = (labels == k).mean()
            cx = x0 + (centroid_xy[k, 0] - xlo) / (xhi - xlo) * size
            cy = top + size - (centroid_xy[k, 1] - ylo) / (yhi - ylo) * size
            zx = x0 + (0 - xlo) / (xhi - xlo) * size
            zy = top + size - (0 - ylo) / (yhi - ylo) * size
            panels.append(
                f'<text class="title" x="{x0}" y="{top - 9}">cluster {k}'
                f'<tspan class="sub"> · {(labels == k).sum():,} lines · {share:.0%}</tspan></text>'
                f'<rect class="frame" x="{x0}" y="{top}" width="{size}" height="{size}" rx="3"/>'
                f'<line class="zero" x1="{zx}" y1="{top}" x2="{zx}" y2="{top + size}"/>'
                f'<line class="zero" x1="{x0}" y1="{zy}" x2="{x0 + size}" y2="{zy}"/>'
                f'<path class="ctx" d="{_dots(shown[shown_labels != k], box, extent)}"/>'
                f'<path class="pts" d="{_dots(shown[shown_labels == k], box, extent)}"/>'
                f'<circle class="hub" cx="{cx:.1f}" cy="{cy:.1f}" r="5">'
                f"<title>cluster {k} centroid</title></circle>"
            )

        def axis(pc, share):
            drivers = " · ".join(loadings[pc].abs().nlargest(2).index)
            return f"{pc} — {share:.0%} of variance ({drivers})"

        width = left + 4 * size + 3 * gap + 8
        height = top + size + 28
        return mo.Html(f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"
         width="100%" role="img"
         aria-label="PCA projection of order lines, one panel per cluster">
      <style>
        .surface {{ fill: #fcfcfb; }}
        .frame {{ fill: none; stroke: #e6e5e0; stroke-width: 1; }}
        .zero {{ stroke: #eceae5; stroke-width: 1; }}
        .ctx {{ stroke: #d6d5cf; stroke-width: 2; stroke-linecap: round; opacity: .55; }}
        .pts {{ stroke: #2a78d6; stroke-width: 2; stroke-linecap: round; opacity: .5; }}
        .hub {{ fill: #2a78d6; stroke: #fcfcfb; stroke-width: 2; }}
        .title {{ fill: #0b0b0b; font: 600 11px system-ui, sans-serif; }}
        .sub, .axis {{ fill: #52514e; font: 400 11px system-ui, sans-serif; }}
        @media (prefers-color-scheme: dark) {{
          .surface {{ fill: #1a1a19; }}
          .frame {{ stroke: #34332f; }} .zero {{ stroke: #2a2926; }}
          .ctx {{ stroke: #4a4945; }} .pts {{ stroke: #3987e5; }}
          .hub {{ fill: #3987e5; stroke: #1a1a19; }}
          .title {{ fill: #ffffff; }} .sub, .axis {{ fill: #c3c2b7; }}
        }}
      </style>
      <rect class="surface" x="0" y="0" width="{width}" height="{height}" rx="4"/>
      {"".join(panels)}
      <text class="axis" x="{left}" y="{height - 12}">{axis("PC1", explained[0])}</text>
      <text class="axis" transform="translate(13 {top + size}) rotate(-90)">{axis("PC2", explained[1])}</text>
    </svg>
    """)

    pca_plot = _pca_figure(
        line_xy, line_clusters["cluster"].to_numpy(), line_centers, pca_loadings, pca_explained
    )
    pca_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The geometry matches the table, which is the point of drawing it: cluster 3 sits hard right on the
    basket axis (cheap, high-margin accessories), cluster 2 sits high on the buyer axis (the older
    cohort), and clusters 0 and 1 split vertically — same baskets, different tenure. The centroids are
    where k-means actually put them, projected through the same rotation, not decorative dots.

    What the picture cannot tell you is whether the clusters are *separated*, because 60% of the
    variance was thrown away to make it. The panels show neighbourhoods that overlap at every edge —
    honest for k-means, which partitions space whether or not the data has gaps in it. Treat the figure
    as a map of where the segments sit relative to each other, and the segment table as the description.

    None of this needed a plotting library: `np.linalg.svd` for the rotation, an f-string for the SVG.
    If you want interactivity instead, `pip install altair` and `mo.ui.altair_chart` gives you
    brushing and linked selection over the same two columns.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Where next

    * **`docs/quickstart.md`** — install, auth, first query.
    * **`docs/mental-model.md`** — measures vs. aggregations, limits and truncation, alias semantics,
      `between()`, and SQL NULL semantics, in one honest page.
    * **`docs/offline-testing.md`** — the fake, the bench dataset and the six test lanes.
    * **`docs/DESIGN.md`**, **`docs/HYBRID.md`**, **`docs/SQLTIER.md`** — the design behind the tiers.

    This notebook is executed headlessly against the fake by
    `tests/e2e/test_demo_notebook.py` on every CI run, so it cannot rot unnoticed. **Commit it with
    its outputs cleared.**
    """)
    return


if __name__ == "__main__":
    app.run()
