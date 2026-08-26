# Changelog

## 0.1.0 (unreleased)

Initial release.

- Idempotent GETs now recover from WAF 429s within a configurable cumulative wait budget;
  `SessionBuilder.rate_limit_wait(...)` configures it without changing POST retry behavior.

- Lazy, immutable PySpark-style `DataFrame` API over Omni's semantic layer
  (`OmniSession`, `read.topic` / `read.view` / `read.sql` / `read.saved_query`, `session.ask`).
- Three-tier compile chain with a DAG splitter: governed semantic queries (tier 1), one
  generated **OmniSQL** statement planned as a governed model job (tier 2), local Arrow
  execution (tier 3) — with `explain()` showing exactly what runs where. A tier-2 statement
  refers to the model directly (`FROM ${topic}`, `${view.field}`, `${view.measure}`), so an
  `agg()` mixing governed measures with ad-hoc aggregations is a single request.
- Removed before the first release, with the tier-2 mechanism they belonged to: the
  `SessionBuilder.sql_dialect(...)` knob and the refusal to push a filter value containing a
  backslash. Omni parses the statement and re-renders it in the warehouse's own dialect, so
  neither had anything left to do (docs/SQLTIER.md §6).
- A formatted time grain arrives as a timestamp, not as the model's display string: Omni returns
  both, and omniframes keeps the raw value under the plain column name (docs/SQLTIER.md §5).
- Governed measures, time grains, measure filters (HAVING), column totals, cross-frame joins
  and unions with SQL NULL semantics, ad-hoc aggregations, UDFs via `map_pandas`/`F.udf`.
- Offline-first test bed: a wire-faithful fake of the Omni query API over a deterministic
  bench dataset, plus golden, differential and e2e test lanes, and `scripts/live_smoke.py` for
  checks that need a real org.
- Documentation site (MkDocs + Material + mkdocstrings): landing page, quickstart, a
  measure-first mental-model page (tiers, limits, aliases, `between()` and NULL semantics), a
  generated API reference, an offline-testing guide for contributors, and the design notes.
  `uv run mkdocs build --strict` is a CI job; every code sample in the README and the
  user-facing docs is executed against the fake.
- `examples/demo.ipynb` — a guided tour of the whole feature surface with `explain()` after
  every step, running either against the in-process fake (default, no credentials) or against a
  live org (`OMNI_BASE_URL` / `OMNI_API_KEY`, model and topic from `OMNI_MODEL` / `OMNI_TOPIC`).
  Executed headlessly in CI by `tests/e2e/test_demo_notebook.py`.
- Packaging validated end to end: `uv build`, `twine check`, and a clean-venv wheel install that
  runs a fake-backed query (`py.typed` ships in the wheel).
