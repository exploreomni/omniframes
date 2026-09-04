# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Before making changes, read and follow `AGENTS.md`. Its repository-wide requirements, including
Conventional Commits and mandatory coding-agent attribution, apply alongside this guide.

## What this is

Omniframes: a PySpark-style Python DataFrame library for Omni. Users write dataframe code; it
compiles into governed semantic queries and pushes as much compute as possible into Omni's SQL
execution layer. Open source (Apache-2.0), PyPI-bound, lazy/immutable, with a three-tier compile
chain (semantic → SQL job → local pandas).

**Authoritative in-repo docs — read these before changing compiler or transport code:**
- `docs/CONTRACT_NOTES.md` — the wire contract for the Omni query API, sourced from monorepo
  server code (`~/omni/omni` @ `86569cd50c6`), NOT the public OpenAPI spec (which is wrong in
  several places). Includes the LIVE-VALIDATE register of unconfirmed behaviors.
- `docs/DESIGN.md` — architecture, tier semantics (incl. the DAG splitter and mixed-aggregation
  decomposition), and settled API decisions (limit policy, client-side aliasing, grain-filter
  rule, no-I/O session building). Don't relitigate settled decisions casually.

Framing: **Omni-native with PySpark-inspired ergonomics**, not a PySpark drop-in. Snake_case
primary; thin camelCase aliases exist for muscle memory.

## Commands

- `uv sync --all-extras` — install/refresh the environment
- `uv run pytest` — full suite (offline; there is no live pytest lane)
- `uv run pytest tests/unit/test_x.py::test_name` — single test
- `uv run python scripts/live_smoke.py` — the live-org probe (needs credentials)
- `uv run marimo edit examples/demo.py` — launch the marimo demo notebook (needs
  OMNI_BASE_URL/OMNI_API_KEY for live cells)
- `uv run ruff format && uv run ruff check --fix` — format + lint (run after every edit)
- `uv run mypy` — strict type check
- Milestone gate: `uv run ruff format --check && uv run ruff check && uv run mypy && uv run pytest -m "not live"`

## Architecture (src/omniframes/)

- `session.py`, `catalog.py`, `dataframe.py`, `column.py`, `functions.py` — user API.
  Everything immutable + lazy; actions trigger compilation.
- `plan/` — logical plan nodes (Scan/Project/Filter/Aggregate/Sort/Limit/Join/Union/WithColumn/MapPandas).
- `compile/` — `querymodel.py` (typed wire contract port), `semantic.py` (tier 1),
  `sqlgen.py` (tier 2, SQLGlot), `splitter.py` (DAG pushdown frontier), `local.py`
  (pandas operator interpreter), `explain.py`.
- `transport/` — `QueryTransport` protocol; `http.py` (httpx, NDJSON wait-loop, error mapping),
  `ndjson.py`, `arrow.py`, `normalize.py`. Nothing above the transport may assume HTTP.
- `io/` — writers, pandas/arrow/polars conversion.

Key semantics (details in docs/DESIGN.md): governed measures always execute remotely; ad-hoc
aggregations go tier 2 or tier 3 (never tier-1 calculations — `original_formula` does NOT work
on `/query/run`); the library always sends an explicit `limit` (default 50 000 + truncation
warning); aliases are client-side renames; `summary.fields` from a `planOnly` run is the only
schema authority.

## Testing

Five lanes under `tests/`: `unit/`, `golden/` (compiler snapshots as checked-in JSON),
`wire/` (HttpTransport vs fixture NDJSON bytes), `differential/` (pushdown vs pure-pandas
equivalence; `dropna=False` semantics), `e2e/`. Live-org checks are `scripts/live_smoke.py`, a
standalone script — `pytest -m live` collects nothing.
`tests/fakes/` holds FakeOmniAPI — an in-process httpx.MockTransport fake with exact NDJSON
framing over the bench dataset (DuckDB-executed; dev dep only). The fake implements only what
current milestones exercise — grow it with the milestone that needs it.

## Conventions

- Do not commit unless asked. Never touch `~/omni/omni` (read-only reference checkout).
- New wire behavior must cite a monorepo source location in `docs/CONTRACT_NOTES.md`; unverified
  behavior gets a `LIVE-VALIDATE` entry there, not a guess.
- API keys must never appear in reprs, logs, errors, docs, or fixtures (tested).
- Open questions that need a human call: PyPI publish timing/name claim, repo publication.
