"""Known-answer and join-integrity queries over the six WWI topics."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from omniframes import OmniSession
from omniframes import functions as F

pytestmark = pytest.mark.live

_TOPICS = (
    "wwi_sales",
    "wwi_orders",
    "wwi_purchasing",
    "wwi_inventory",
    "wwi_stock_movements",
    "wwi_financial_transactions",
)


@pytest.mark.parametrize("topic_name", _TOPICS)
def test_modeled_measures_match_independently_computed_goldens(
    topic_name: str,
    omni_session: OmniSession,
    live_settings: Any,
    wwi_contract: dict[str, Any],
) -> None:
    expected = wwi_contract["topics"][topic_name]["measures"]
    table = (
        omni_session.read.topic(live_settings.model_id, topic_name)
        .select(*(F.measure(name) for name in expected))
        .collect()
    )

    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert set(row) == set(expected)
    for field_name, expected_value in expected.items():
        assert row[field_name] is not None
        assert Decimal(str(row[field_name])) == Decimal(expected_value), field_name


@pytest.mark.parametrize("topic_name", _TOPICS)
def test_every_relationship_preserves_an_additive_fact_measure(
    topic_name: str,
    omni_session: OmniSession,
    live_settings: Any,
    wwi_contract: dict[str, Any],
) -> None:
    topic_contract = wwi_contract["topics"][topic_name]
    measure = topic_contract["join_invariant_measure"]
    expected_total = Decimal(topic_contract["measures"][measure])

    for dimension, expected_groups in topic_contract["join_probes"].items():
        table = (
            omni_session.read.topic(live_settings.model_id, topic_name)
            .select(dimension, F.measure(measure))
            .collect()
        )

        assert table.num_rows == expected_groups, dimension
        grouped_total = sum(
            (Decimal(str(row[measure])) for row in table.to_pylist()),
            start=Decimal(0),
        )
        assert grouped_total == expected_total, dimension
