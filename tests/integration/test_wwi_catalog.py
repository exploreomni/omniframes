"""Catalog and relationship-shape contracts for the WWI model."""

from __future__ import annotations

from typing import Any

import pytest

from omniframes import OmniSession

pytestmark = pytest.mark.live


def test_model_and_topic_catalog_match_the_pinned_contract(
    omni_session: OmniSession,
    live_settings: Any,
    wwi_contract: dict[str, Any],
) -> None:
    model = omni_session.catalog.model(live_settings.model_id)
    assert model.id == live_settings.model_id
    assert model.name == wwi_contract["model_name"]

    summaries = {topic.name: topic for topic in omni_session.catalog.topics(live_settings.model_id)}
    expected_topics = wwi_contract["topics"]
    assert set(summaries) == set(expected_topics)

    for topic_name, expected in expected_topics.items():
        summary = summaries[topic_name]
        assert summary.label == expected["label"]
        assert summary.group_label == "Wide World Importers"
        assert summary.base_view_name == expected["base_view"]
        assert summary.hidden is False


@pytest.mark.parametrize(
    "topic_name",
    [
        "wwi_sales",
        "wwi_orders",
        "wwi_purchasing",
        "wwi_inventory",
        "wwi_stock_movements",
        "wwi_financial_transactions",
    ],
)
def test_topic_detail_exposes_the_expected_relationships_and_measures(
    topic_name: str,
    omni_session: OmniSession,
    live_settings: Any,
    wwi_contract: dict[str, Any],
) -> None:
    expected = wwi_contract["topics"][topic_name]
    topic = omni_session.catalog.topic(live_settings.model_id, topic_name)

    assert topic.base_view_name == expected["base_view"]
    assert [(edge.left_view_name, edge.right_view_name) for edge in topic.relationships] == [
        (expected["base_view"], right_view) for right_view in expected["relationships"]
    ]
    assert {edge.join_type for edge in topic.relationships} == {"ALWAYS_LEFT"}
    assert {edge.relationship_type for edge in topic.relationships} == {"MANY_TO_ONE"}

    visible_measures = {measure.name for view in topic.views for measure in view.measures}
    assert set(expected["measures"]) <= visible_measures
