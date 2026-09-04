"""Authentication and effective-role contracts for the two user PATs."""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.live


def test_pat_resolves_to_the_expected_member_and_permissions(
    live_settings: Any,
    whoami: dict[str, Any],
    wwi_contract: dict[str, Any],
) -> None:
    assert whoami["keyScope"] == "user", "the live suite must use a user PAT, not an org key"
    assert whoami["orgRole"] == "MEMBER", "integration identities must not be administrators"

    roles_by_model = whoami.get("rolesByModel", {})
    assert live_settings.model_id in roles_by_model, "the PAT cannot see the configured WWI model"
    actual = roles_by_model[live_settings.model_id]
    expected = wwi_contract["principals"][live_settings.principal]

    assert actual["roleName"] == expected["role_name"]
    assert actual["baseRole"] == expected["role_name"]

    permissions = set(actual["permissions"])
    assert set(expected["required_permissions"]) <= permissions
    assert set(expected["forbidden_permissions"]).isdisjoint(permissions)

    if live_settings.expected_membership_id:
        assert whoami["membershipId"] == live_settings.expected_membership_id
