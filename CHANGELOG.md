# Changelog

## 0.1.0 (unreleased)

Initial release.

- Lazy, immutable PySpark-style `DataFrame` API over Omni's semantic layer
  (`OmniSession`, `read.topic` / `read.view` / `read.sql` / `read.saved_query`, `session.ask`).
- Three-tier compile chain with a DAG splitter: governed semantic queries (tier 1),
  warehouse-executed SQL over embedded semantic sub-queries (tier 2), local Arrow execution
  (tier 3) — with `explain()` showing exactly what runs where.
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
