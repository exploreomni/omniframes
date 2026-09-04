"""Fixtures for the opt-in WWI live integration suite."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from omniframes import OmniSession

Principal = Literal["querier", "restricted"]

_CONTRACT_PATH = Path(__file__).with_name("wwi_contract.json")
_PAT_ENV = {
    "querier": "OMNI_QUERIER_PAT",
    "restricted": "OMNI_RESTRICTED_QUERIER_PAT",
}
_MEMBERSHIP_ENV = {
    "querier": "OMNI_QUERIER_MEMBERSHIP_ID",
    "restricted": "OMNI_RESTRICTED_MEMBERSHIP_ID",
}


@dataclass(frozen=True)
class LiveSettings:
    principal: Principal
    base_url: str
    model_id: str
    api_key: str = field(repr=False, compare=False)
    expected_membership_id: str | None = field(default=None, repr=False)


@pytest.fixture(scope="session")
def wwi_contract() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(_CONTRACT_PATH.read_text("utf-8")))


@pytest.fixture(scope="session")
def live_settings(pytestconfig: pytest.Config, wwi_contract: dict[str, Any]) -> LiveSettings:
    if not pytestconfig.getoption("--live"):
        pytest.skip("pass --live to run tests against Omni")

    principal = pytestconfig.getoption("--principal")
    if principal not in _PAT_ENV:
        raise pytest.UsageError("--live requires --principal querier or --principal restricted")

    # Local runs use the principal-specific variables from .env.integration. CI maps exactly
    # one GitHub secret into the generic variable, so the other identity never enters that job.
    pat_env = _PAT_ENV[principal]
    api_key = os.environ.get(pat_env) or os.environ.get("OMNI_API_KEY")
    if not api_key:
        raise pytest.UsageError(
            f"{pat_env} (local) or OMNI_API_KEY (CI) must contain that user's PAT"
        )

    base_url = os.environ.get("OMNI_BASE_URL") or str(wwi_contract["base_url"])
    model_id = os.environ.get("OMNI_WWI_MODEL_ID") or str(wwi_contract["model_id"])
    expected_membership_id = os.environ.get(_MEMBERSHIP_ENV[principal]) or os.environ.get(
        "OMNI_EXPECTED_MEMBERSHIP_ID"
    )
    return LiveSettings(
        principal=principal,
        base_url=base_url,
        model_id=model_id,
        api_key=api_key,
        expected_membership_id=expected_membership_id,
    )


@pytest.fixture(scope="session")
def omni_session(live_settings: LiveSettings) -> Iterator[OmniSession]:
    session = (
        OmniSession.builder.base_url(live_settings.base_url)
        .api_key(live_settings.api_key)
        .cache("SkipCache")
        .get_or_create()
    )
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="session")
def whoami(omni_session: OmniSession) -> dict[str, Any]:
    return omni_session.verify()
