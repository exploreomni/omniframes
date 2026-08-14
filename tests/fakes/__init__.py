"""In-process fakes for the omniframes test suite.

:class:`FakeOmniAPI` is the keystone of the fixtures-first testing model (docs/DESIGN.md §5): an
``httpx.MockTransport`` handler that serves the Omni Query API with exact wire fidelity
(``docs/CONTRACT_NOTES.md``) over the checked-in bench dataset (``docs/BENCH_DATASET.md``),
executing queries through DuckDB so the answers are real::

    handler = FakeOmniAPI()
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://bench.example.omni.co",
    )

The model it serves is mirrored, YAML-side, in ``docs/bench_omni_model.md`` so the live lane can
reuse offline expectations.
"""

from __future__ import annotations

from tests.fakes.bench_model import (
    AVERAGE_SCALE,
    BENCH_BASE_VIEW,
    BENCH_DATA_DIR,
    BENCH_MODEL_ID,
    BENCH_MODEL_NAME,
    BENCH_TOPIC,
    BENCH_TOPIC_NAME,
    DEFAULT_PERMISSIONS,
    FakeField,
    FakeModel,
    FakeRelationship,
    FakeTopic,
    FakeView,
)
from tests.fakes.documents import (
    BENCH_DOCUMENT_ID,
    DEFAULT_DOCUMENTS,
    DOCUMENT_WITHOUT_DASHBOARD,
    GENERATED_QUERIES,
    NO_QUERY_GENERATED,
    RUN_QUERY_REFUSAL,
    GeneratedQuery,
    SavedQuery,
    bench_query,
)
from tests.fakes.engine import (
    COLUMN_TOTAL_INDICATOR,
    GRAND_TOTAL_INDICATOR,
    GRAND_TOTAL_KEY,
    TIME_GRAINS,
    TOTAL_INDICATOR_COLUMN,
    BenchEngine,
    ColumnTotals,
    Grain,
    PlanFailure,
    PlannedQuery,
    ResolvedField,
    arrow_data_type,
)
from tests.fakes.fake_omni import (
    DEFAULT_TOKEN,
    NDJSON_CONTENT_TYPE,
    REDACTED_ERROR_MESSAGE,
    WORKBOOK_URL_HEADER,
    FakeOmniAPI,
    RecordedRequest,
)
from tests.fakes.sqljobs import (
    SUMM_SIDECAR_SUFFIX,
    QueryReference,
    SqlJob,
    is_raw_sql_job,
    sidecar_name,
    synthesize_fields,
)

__all__ = [
    "AVERAGE_SCALE",
    "BENCH_BASE_VIEW",
    "BENCH_DATA_DIR",
    "BENCH_DOCUMENT_ID",
    "BENCH_MODEL_ID",
    "BENCH_MODEL_NAME",
    "BENCH_TOPIC",
    "BENCH_TOPIC_NAME",
    "COLUMN_TOTAL_INDICATOR",
    "DEFAULT_DOCUMENTS",
    "DEFAULT_PERMISSIONS",
    "DEFAULT_TOKEN",
    "DOCUMENT_WITHOUT_DASHBOARD",
    "GENERATED_QUERIES",
    "GRAND_TOTAL_INDICATOR",
    "GRAND_TOTAL_KEY",
    "NDJSON_CONTENT_TYPE",
    "NO_QUERY_GENERATED",
    "REDACTED_ERROR_MESSAGE",
    "RUN_QUERY_REFUSAL",
    "SUMM_SIDECAR_SUFFIX",
    "TIME_GRAINS",
    "TOTAL_INDICATOR_COLUMN",
    "WORKBOOK_URL_HEADER",
    "BenchEngine",
    "ColumnTotals",
    "FakeField",
    "FakeModel",
    "FakeOmniAPI",
    "FakeRelationship",
    "FakeTopic",
    "FakeView",
    "GeneratedQuery",
    "Grain",
    "PlanFailure",
    "PlannedQuery",
    "QueryReference",
    "RecordedRequest",
    "ResolvedField",
    "SavedQuery",
    "SqlJob",
    "arrow_data_type",
    "bench_query",
    "is_raw_sql_job",
    "sidecar_name",
    "synthesize_fields",
]
