"""Tests for coarse Python and hosted-runtime identification."""

from __future__ import annotations

import sys

import pytest

from omniframes import __version__
from omniframes._user_agent import USER_AGENT, _format_version, build_user_agent


def user_agent(environ: dict[str, str]) -> str:
    return build_user_agent(
        library_version="1.2.3",
        python_version="3.12.4",
        implementation="cpython",
        implementation_version="3.12.4",
        environ=environ,
        path_exists=lambda _path: False,
    )


def test_user_agent_identifies_the_library_and_python_vm() -> None:
    assert user_agent({}) == "omniframes/1.2.3 python/3.12.4 cpython/3.12.4"


def test_process_user_agent_carries_the_public_library_version() -> None:
    python_version = _format_version(
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
        sys.version_info.releaselevel,
        sys.version_info.serial,
    )
    implementation_version = _format_version(
        sys.implementation.version.major,
        sys.implementation.version.minor,
        sys.implementation.version.micro,
        sys.implementation.version.releaselevel,
        sys.implementation.version.serial,
    )

    assert USER_AGENT.startswith(f"omniframes/{__version__} ")
    assert f"python/{python_version}" in USER_AGENT.split()
    assert f"{sys.implementation.name}/{implementation_version}" in USER_AGENT.split()


@pytest.mark.parametrize(
    "marker",
    [
        "DATABRICKS_RUNTIME_VERSION",
        "DATABRICKS_ENV_VERSION",
    ],
)
def test_databricks_value_markers_are_detected(marker: str) -> None:
    assert user_agent({marker: "marker-value"}).endswith(" runtime/databricks")


@pytest.mark.parametrize(
    "marker", ["IS_IN_DB_MODEL_SERVING_ENV", "IS_IN_DATABRICKS_MODEL_SERVING_ENV"]
)
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_databricks_boolean_markers_are_detected(marker: str, value: str) -> None:
    assert user_agent({marker: value}).endswith(" runtime/databricks")


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_false_databricks_boolean_markers_are_ignored(value: str) -> None:
    assert " runtime/" not in user_agent({"IS_IN_DB_MODEL_SERVING_ENV": value})


def test_databricks_connection_settings_do_not_claim_to_be_the_runtime() -> None:
    assert " runtime/" not in user_agent(
        {
            "DATABRICKS_APP_PORT": "8000",
            "DATABRICKS_HOST": "https://workspace.example",
            "DATABRICKS_TOKEN": "secret",
        }
    )


def test_deployed_databricks_app_markers_are_detected_together() -> None:
    environ = {
        "DATABRICKS_APP_NAME": "example",
        "DATABRICKS_APP_PORT": "8000",
        "DATABRICKS_WORKSPACE_ID": "123",
    }

    assert user_agent(environ).endswith(" runtime/databricks")


def test_databricks_runtime_file_is_a_fallback_signal() -> None:
    result = build_user_agent(
        environ={}, path_exists=lambda path: path == "/databricks/DBR_VERSION"
    )

    assert result.endswith(" runtime/databricks")


@pytest.mark.parametrize(
    "marker", ["COLAB_RELEASE_TAG", "COLAB_BACKEND_VERSION", "COLAB_JUPYTER_TRANSPORT"]
)
def test_google_colab_markers_are_detected(marker: str) -> None:
    assert user_agent({marker: "marker-value"}).endswith(" runtime/google-colab")


def test_google_colab_enterprise_is_distinguished() -> None:
    environ = {"VERTEX_PRODUCT": "COLAB_ENTERPRISE", "COLAB_RELEASE_TAG": "release-2026"}

    assert user_agent(environ).endswith(" runtime/google-colab-enterprise")


def test_google_colab_runtime_file_is_a_fallback_signal() -> None:
    result = build_user_agent(environ={}, path_exists=lambda path: path == "/var/colab/hostname")

    assert result.endswith(" runtime/google-colab")


def test_databricks_takes_precedence_over_colab_when_markers_conflict() -> None:
    environ = {"DATABRICKS_RUNTIME_VERSION": "15.4", "COLAB_RELEASE_TAG": "release-2026"}

    assert user_agent(environ).endswith(" runtime/databricks")


def test_raw_runtime_values_are_never_copied_into_the_header() -> None:
    raw_value = "15.4\r\nX-Workspace-ID: private"

    result = user_agent({"DATABRICKS_RUNTIME_VERSION": raw_value})

    assert raw_value not in result
    assert "\r" not in result
    assert "\n" not in result
    assert result.endswith(" runtime/databricks")


@pytest.mark.parametrize(
    "marker",
    [
        "DATABRICKS_RUNTIME_VERSION",
        "DATABRICKS_ENV_VERSION",
        "COLAB_RELEASE_TAG",
        "COLAB_BACKEND_VERSION",
        "COLAB_JUPYTER_TRANSPORT",
    ],
)
@pytest.mark.parametrize("value", ["", "  "])
def test_empty_value_markers_are_ignored(marker: str, value: str) -> None:
    assert " runtime/" not in user_agent({marker: value})


def test_prerelease_interpreter_versions_are_not_reported_as_final() -> None:
    assert _format_version(3, 15, 0, "candidate", 2) == "3.15.0rc2"
    assert _format_version(3, 15, 0, "beta", 1) == "3.15.0b1"


def test_dynamic_product_tokens_are_sanitized_and_bounded() -> None:
    result = build_user_agent(
        library_version="1.2.3\r\nX-Evil: yes",
        python_version="3.12.4",
        implementation="Custom VM",
        implementation_version="v" * 100,
        environ={},
        path_exists=lambda _path: False,
    )

    assert result == (f"omniframes/1.2.3-X-Evil-yes python/3.12.4 custom-vm/{'v' * 64}")


def test_non_ascii_product_values_still_make_an_ascii_header() -> None:
    result = build_user_agent(
        library_version="1.2.3+naïve",
        python_version="3.12.4",
        implementation="Pyπ",
        implementation_version="1.0",
        environ={},
        path_exists=lambda _path: False,
    )

    assert result.isascii()


def test_unreadable_runtime_marker_paths_do_not_break_user_agent_creation() -> None:
    def denied(_path: str) -> bool:
        raise PermissionError

    result = build_user_agent(environ={}, path_exists=denied)

    assert " runtime/" not in result
