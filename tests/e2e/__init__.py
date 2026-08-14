"""End-to-end lane: the whole stack (session → compiler → transport → normalize) at once.

Where the unit lanes test one layer and the golden lane tests the payload, this lane runs real
dataframe code against :class:`~tests.fakes.FakeOmniAPI` — real NDJSON framing, real Arrow, real
DuckDB answers over the bench dataset — and checks both the numbers that come back and the exact
envelopes that went out.
"""
