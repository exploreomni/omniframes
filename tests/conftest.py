"""Repository-wide pytest command-line options."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    live = parser.getgroup("live Omni")
    live.addoption(
        "--live",
        action="store_true",
        default=False,
        help="run tests that make read-only requests to a live Omni organization",
    )
    live.addoption(
        "--principal",
        choices=("querier", "restricted"),
        help="user PAT identity for the live suite",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Require an explicit opt-in before a test can contact a live Omni org."""
    if config.getoption("--live"):
        return

    skip_live = pytest.mark.skip(reason="pass --live to run tests against Omni")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
