"""Positive and negative authorization contracts for each live principal."""

from __future__ import annotations

from typing import Any

import pytest

from omniframes import OmniSession, QueryError, TruncationWarning
from omniframes import functions as F

pytestmark = pytest.mark.live

_MODEL_MEASURE = "wwi_sales_fact.invoice_count"
_RAW_FIELD = "wwi_sales_fact.wwi_invoice_id"


def _assert_forbidden(error: QueryError, *, message: str, api_key: str) -> None:
    # Do this before comparing the message: should scrubbing ever regress, the assertion output
    # contains only a boolean and cannot echo the secret into a CI log.
    leaked = api_key in str(error)
    assert leaked is False
    assert error.error_type == "FORBIDDEN"
    assert str(error) == message


def test_topic_planning_is_available_but_sql_visibility_follows_the_role(
    omni_session: OmniSession,
    live_settings: Any,
) -> None:
    frame = omni_session.read.topic(live_settings.model_id, "wwi_sales").select(
        F.measure(_MODEL_MEASURE)
    )

    assert frame.schema.names == (_MODEL_MEASURE,)
    explanation = frame.explain(analyze=True)
    if live_settings.principal == "querier":
        assert "SELECT" in explanation
        assert "VIEW_SQL" not in explanation
    else:
        assert "SELECT" not in explanation
        assert "SQL redacted" in explanation
        assert "VIEW_SQL" in explanation


def test_bare_view_access_follows_query_full_model_permission(
    omni_session: OmniSession,
    live_settings: Any,
    wwi_contract: dict[str, Any],
) -> None:
    frame = (
        omni_session.read.view(live_settings.model_id, "wwi_sales_fact").select(_RAW_FIELD).limit(1)
    )
    if live_settings.principal == "querier":
        with pytest.warns(TruncationWarning):
            result = frame.collect()
        assert result.num_rows == 1
    else:
        with pytest.raises(QueryError) as caught:
            frame.collect()
        _assert_forbidden(
            caught.value,
            message=wwi_contract["forbidden_messages"]["view"],
            api_key=live_settings.api_key,
        )


def test_raw_sql_access_follows_query_sql_permission(
    omni_session: OmniSession,
    live_settings: Any,
    wwi_contract: dict[str, Any],
) -> None:
    fixture = wwi_contract["fixture"]
    statement = f'''SELECT
  fixture_name,
  fixture_version,
  content_sha256,
  schema_sha256
FROM "{fixture["schema"]}"."fixture_metadata"'''
    frame = omni_session.read.sql(live_settings.model_id, statement)
    if live_settings.principal == "querier":
        assert frame.collect().to_pylist() == [
            {
                "fixture_name": fixture["name"],
                "fixture_version": fixture["version"],
                "content_sha256": fixture["content_sha256"],
                "schema_sha256": fixture["schema_sha256"],
            }
        ]
    else:
        with pytest.raises(QueryError) as caught:
            frame.collect()
        _assert_forbidden(
            caught.value,
            message=wwi_contract["forbidden_messages"]["sql"],
            api_key=live_settings.api_key,
        )


def test_ad_hoc_topic_aggregation_obeys_the_manual_sql_boundary(
    omni_session: OmniSession,
    live_settings: Any,
    wwi_contract: dict[str, Any],
) -> None:
    # Governed measures stay in the topic query API. An ad-hoc aggregate uses Omniframes'
    # tier-2 OmniSQL path, which Omni deliberately treats as manually written SQL.
    frame = (
        omni_session.read.topic(live_settings.model_id, "wwi_sales")
        .group_by(_RAW_FIELD)
        .agg(F.sum("wwi_sales_fact.total_including_tax").alias("ad_hoc_sales"))
        .limit(1)
    )
    if live_settings.principal == "querier":
        result = frame.collect()
        assert result.num_rows == 1
    else:
        with pytest.raises(QueryError) as caught:
            frame.collect()
        _assert_forbidden(
            caught.value,
            message=wwi_contract["forbidden_messages"]["sql"],
            api_key=live_settings.api_key,
        )
