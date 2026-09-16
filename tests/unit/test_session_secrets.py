"""Notebook credential providers, exercised without SDKs or live notebook services."""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, call

import httpx
import pytest
from tests.fakes import DEFAULT_TOKEN, FakeOmniAPI

from omniframes import OmniSession
from omniframes import session as session_module
from omniframes.errors import CompileError
from omniframes.session import API_KEY_ENV, BASE_URL_ENV
from omniframes.transport import HttpTransport

BASE_URL = "https://bench.omniapp.co"
PROVIDERS = ("colab", "databricks", "snowflake", "snowflake-legacy")
NOTEBOOK_MODULES = (
    "google.colab",
    "google.colab.userdata",
    "databricks.sdk.runtime",
    "snowflake.snowpark.secrets",
    "streamlit",
)


class SecretNotFoundError(Exception):
    pass


class NotebookAccessError(Exception):
    pass


class TimeoutException(Exception):
    pass


def install_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "environ", {})
    for name in NOTEBOOK_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
    install_module(monkeypatch, "IPython", get_ipython=Mock(return_value=None))


@pytest.fixture
def colab(monkeypatch: pytest.MonkeyPatch) -> Mock:
    get = Mock(side_effect={BASE_URL_ENV: BASE_URL, API_KEY_ENV: DEFAULT_TOKEN}.__getitem__)
    install_module(monkeypatch, "google.colab")
    install_module(
        monkeypatch,
        "google.colab.userdata",
        get=get,
        SecretNotFoundError=SecretNotFoundError,
        NotebookAccessError=NotebookAccessError,
        TimeoutException=TimeoutException,
    )
    return get


@pytest.fixture
def databricks(monkeypatch: pytest.MonkeyPatch) -> Mock:
    get = Mock(
        side_effect=lambda scope, key: {
            BASE_URL_ENV: BASE_URL,
            API_KEY_ENV: DEFAULT_TOKEN,
        }[key]
    )
    dbutils = SimpleNamespace(secrets=SimpleNamespace(get=get))
    shell = SimpleNamespace(user_ns={"dbutils": dbutils})
    install_module(monkeypatch, "IPython", get_ipython=Mock(return_value=shell))
    return get


@pytest.fixture
def snowflake(monkeypatch: pytest.MonkeyPatch) -> Mock:
    get = Mock(side_effect={BASE_URL_ENV: BASE_URL, API_KEY_ENV: DEFAULT_TOKEN}.__getitem__)
    install_module(monkeypatch, "snowflake.snowpark.secrets", get_generic_secret_string=get)
    return get


@pytest.fixture
def snowflake_legacy(monkeypatch: pytest.MonkeyPatch) -> Mock:
    get = Mock(side_effect={BASE_URL_ENV: BASE_URL, API_KEY_ENV: DEFAULT_TOKEN}.__getitem__)

    class Secrets:
        def __getitem__(self, key: str) -> object:
            return get(key)

    install_module(monkeypatch, "streamlit", secrets=Secrets())
    return get


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeOmniAPI]:
    fake = FakeOmniAPI()
    with httpx.Client(transport=httpx.MockTransport(fake)) as client:

        def transport(**options: Any) -> HttpTransport:
            return HttpTransport(**options, client=client)

        monkeypatch.setattr(session_module, "HttpTransport", transport)
        yield fake
    fake.close()


def reader_for(request: pytest.FixtureRequest, provider: str) -> Mock:
    reader: Mock = request.getfixturevalue(provider.replace("-", "_"))
    return reader


def assert_no_secret_in_error(error: BaseException) -> None:
    assert DEFAULT_TOKEN not in str(error)
    assert DEFAULT_TOKEN not in repr(error)
    assert DEFAULT_TOKEN not in "".join(traceback.format_exception(error))


@pytest.mark.parametrize("provider", PROVIDERS)
def test_secret_configuration_is_lazy(
    provider: str, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = reader_for(request, provider)
    importer = Mock(side_effect=AssertionError("must not import a provider during configuration"))
    monkeypatch.setattr(importlib, "import_module", importer)
    getter = sys.modules["IPython"].get_ipython
    getter.side_effect = AssertionError("must not detect a runtime during configuration")

    builder = OmniSession.builder
    assert builder.secrets(provider, scope="omni" if provider == "databricks" else None) is builder

    reader.assert_not_called()
    importer.assert_not_called()
    getter.assert_not_called()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_each_provider_authenticates_without_eager_omni_calls(
    provider: str, request: pytest.FixtureRequest, api: FakeOmniAPI
) -> None:
    reader = reader_for(request, provider)
    builder = OmniSession.builder.secrets(
        provider, scope="omni" if provider == "databricks" else None
    )

    with builder.get_or_create() as session:
        assert api.requests == []
        assert BASE_URL in repr(session)
        assert DEFAULT_TOKEN not in repr(session)
        assert DEFAULT_TOKEN not in repr(builder)
        assert API_KEY_ENV not in os.environ
        assert BASE_URL_ENV not in os.environ
        session.verify()

    assert api.paths == ["GET /api/v1/whoami"]
    if provider == "databricks":
        assert reader.call_args_list == [
            call(scope="omni", key=BASE_URL_ENV),
            call(scope="omni", key=API_KEY_ENV),
        ]
    else:
        assert reader.call_args_list == [call(BASE_URL_ENV), call(API_KEY_ENV)]


@pytest.mark.parametrize("source", ["explicit", "environment", "helpers"])
@pytest.mark.parametrize("provider", ["auto", *PROVIDERS])
def test_complete_credentials_skip_all_runtime_detection_and_provider_imports(
    source: str, provider: str, api: FakeOmniAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(BASE_URL_ENV, BASE_URL)
    monkeypatch.setenv(API_KEY_ENV, DEFAULT_TOKEN)
    builder = OmniSession.builder.secrets(provider)
    if source == "explicit":
        builder.host(BASE_URL).api_key(DEFAULT_TOKEN)
        monkeypatch.setenv(BASE_URL_ENV, "https://wrong.omniapp.co")
        monkeypatch.setenv(API_KEY_ENV, "wrong-test-token")
    elif source == "helpers":
        builder.base_url_from_env().api_key_from_env()
    importer = Mock(side_effect=AssertionError("must not import a notebook library"))
    monkeypatch.setattr(importlib, "import_module", importer)
    getter = sys.modules["IPython"].get_ipython
    getter.side_effect = AssertionError("must not detect a notebook runtime")

    with builder.get_or_create() as session:
        session.verify()

    importer.assert_not_called()
    getter.assert_not_called()


@pytest.mark.parametrize("source", ["explicit", "environment"])
@pytest.mark.parametrize("configured", [BASE_URL_ENV, API_KEY_ENV])
@pytest.mark.parametrize("provider", PROVIDERS)
def test_only_missing_credentials_use_the_selected_provider(
    source: str,
    configured: str,
    provider: str,
    request: pytest.FixtureRequest,
    api: FakeOmniAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = reader_for(request, provider)
    builder = OmniSession.builder.secrets(
        provider, scope="omni" if provider == "databricks" else None
    )
    value = BASE_URL if configured == BASE_URL_ENV else DEFAULT_TOKEN
    if source == "environment":
        monkeypatch.setenv(configured, value)
    elif configured == BASE_URL_ENV:
        builder.host(value)
    else:
        builder.api_key(value)

    with builder.get_or_create() as session:
        session.verify()

    missing = API_KEY_ENV if configured == BASE_URL_ENV else BASE_URL_ENV
    if provider == "databricks":
        reader.assert_called_once_with(scope="omni", key=missing)
    else:
        reader.assert_called_once_with(missing)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_custom_secret_names_and_paths(
    provider: str, request: pytest.FixtureRequest, api: FakeOmniAPI
) -> None:
    reader = reader_for(request, provider)
    host_name = "app/config/omni_host" if provider == "snowflake" else "omni_host"
    key_name = "app/config/omni_key" if provider == "snowflake" else "omni_key"
    values = {host_name: BASE_URL, key_name: DEFAULT_TOKEN}
    if provider == "databricks":
        reader.side_effect = lambda scope, key: values[key]
    else:
        reader.side_effect = values.__getitem__

    with OmniSession.builder.secrets(
        provider,
        scope="production-omni" if provider == "databricks" else None,
        base_url_name=host_name,
        api_key_name=key_name,
    ).get_or_create() as session:
        session.verify()

    if provider == "databricks":
        assert reader.call_args_list == [
            call(scope="production-omni", key=host_name),
            call(scope="production-omni", key=key_name),
        ]
    else:
        assert reader.call_args_list == [call(host_name), call(key_name)]


@pytest.mark.parametrize("helper", ["api_key_from_env", "base_url_from_env"])
@pytest.mark.parametrize("custom_names", [False, True])
def test_from_env_helpers_never_read_notebook_secrets(
    helper: str, custom_names: bool, colab: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = API_KEY_ENV if helper == "api_key_from_env" else BASE_URL_ENV
    if custom_names:
        name = "MY_OMNI_SETTING"
    monkeypatch.setenv(name, "")
    importer = Mock(side_effect=AssertionError("from_env must only read the environment"))
    monkeypatch.setattr(importlib, "import_module", importer)

    with pytest.raises(CompileError, match=name):
        getattr(OmniSession.builder, helper)(name)

    importer.assert_not_called()
    colab.assert_not_called()


def test_from_env_helpers_support_custom_environment_names(
    api: FakeOmniAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_OMNI_HOST", BASE_URL)
    monkeypatch.setenv("MY_OMNI_TOKEN", DEFAULT_TOKEN)
    with (
        OmniSession.builder.base_url_from_env("MY_OMNI_HOST")
        .api_key_from_env("MY_OMNI_TOKEN")
        .get_or_create() as session
    ):
        session.verify()


@pytest.mark.parametrize("provider", ["auto", *PROVIDERS])
def test_injected_transport_skips_all_notebook_access(
    provider: str, api: FakeOmniAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    with httpx.Client(transport=httpx.MockTransport(api)) as client:
        transport = HttpTransport(BASE_URL, DEFAULT_TOKEN, client=client)
        builder = OmniSession.builder.transport(transport).secrets(provider)
        importer = Mock(side_effect=AssertionError("must not import a notebook library"))
        monkeypatch.setattr(importlib, "import_module", importer)
        getter = sys.modules["IPython"].get_ipython
        getter.side_effect = AssertionError("must not detect a notebook runtime")

        with builder.get_or_create() as session:
            assert api.requests == []
            session.verify()

    importer.assert_not_called()
    getter.assert_not_called()


def test_colab_automatic_detection_uses_the_loaded_module(colab: Mock, api: FakeOmniAPI) -> None:
    with OmniSession.builder.get_or_create() as session:
        session.verify()
    assert colab.call_count == 2


def test_explicit_colab_selection_does_not_require_automatic_runtime_hint(
    colab: Mock, api: FakeOmniAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with OmniSession.builder.secrets("colab").get_or_create() as session:
        session.verify()
    assert colab.call_count == 2


def test_colab_enterprise_does_not_automatically_use_consumer_userdata(
    colab: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERTEX_PRODUCT", "COLAB_ENTERPRISE")
    with pytest.raises(CompileError, match=API_KEY_ENV):
        OmniSession.builder.host(BASE_URL).get_or_create()
    colab.assert_not_called()


def test_databricks_automatic_detection_uses_existing_notebook_dbutils(
    databricks: Mock, api: FakeOmniAPI
) -> None:
    with OmniSession.builder.secrets(scope="omni").get_or_create() as session:
        session.verify()
    assert databricks.call_count == 2


def test_databricks_uses_ipython_user_global_namespace_when_needed(
    databricks: Mock, api: FakeOmniAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    getter = sys.modules["IPython"].get_ipython
    dbutils = getter.return_value.user_ns["dbutils"]
    getter.return_value = SimpleNamespace(
        user_ns={}, ns_table={"user_global": {"dbutils": dbutils}}
    )
    with OmniSession.builder.secrets(scope="omni").get_or_create() as session:
        session.verify()
    assert databricks.call_count == 2


@pytest.mark.parametrize("provider", ["auto", "databricks"])
def test_databricks_requires_a_scope_when_secrets_are_needed(
    provider: str, databricks: Mock
) -> None:
    with pytest.raises(CompileError, match="scope"):
        OmniSession.builder.host(BASE_URL).secrets(provider).get_or_create()
    databricks.assert_not_called()


def test_databricks_does_not_construct_remote_sdk_dbutils(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk_reader = Mock(side_effect=AssertionError("remote SDK credentials must not be used"))
    install_module(
        monkeypatch,
        "databricks.sdk.runtime",
        dbutils=SimpleNamespace(secrets=SimpleNamespace(get=sdk_reader)),
    )
    with pytest.raises(CompileError, match="unavailable"):
        OmniSession.builder.host(BASE_URL).secrets("databricks", scope="omni").get_or_create()
    sdk_reader.assert_not_called()


@pytest.mark.parametrize(
    ("marker", "value"),
    [
        ("COLAB_RELEASE_TAG", "release"),
        ("COLAB_BACKEND_VERSION", "backend"),
        ("COLAB_JUPYTER_IP", "127.0.0.1"),
        ("DATABRICKS_RUNTIME_VERSION", "17.0"),
        ("SNOWFLAKE_ACCOUNT", "account"),
        ("SNOWFLAKE_CONTAINER_SERVICES_SECRET_PATH_PREFIX", "/secrets"),
    ],
)
def test_telemetry_markers_do_not_select_secret_providers(
    marker: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(marker, value)
    importer = Mock(side_effect=AssertionError("environment marker must not import a provider"))
    monkeypatch.setattr(importlib, "import_module", importer)
    with pytest.raises(CompileError, match=API_KEY_ENV):
        OmniSession.builder.host(BASE_URL).get_or_create()
    importer.assert_not_called()


def test_loaded_snowflake_packages_do_not_select_a_notebook_flavor(
    snowflake: Mock, snowflake_legacy: Mock
) -> None:
    with pytest.raises(CompileError, match=API_KEY_ENV):
        OmniSession.builder.host(BASE_URL).get_or_create()
    snowflake.assert_not_called()
    snowflake_legacy.assert_not_called()


def test_ambiguous_runtime_requires_explicit_provider(colab: Mock, databricks: Mock) -> None:
    with pytest.raises(CompileError, match=r"multiple.*select one"):
        OmniSession.builder.secrets(scope="omni").get_or_create()
    colab.assert_not_called()
    databricks.assert_not_called()


def test_explicit_provider_resolves_runtime_ambiguity(
    colab: Mock, databricks: Mock, api: FakeOmniAPI
) -> None:
    with OmniSession.builder.secrets("colab").get_or_create() as session:
        session.verify()
    assert colab.call_count == 2
    databricks.assert_not_called()


def test_provider_is_selected_once_for_both_credentials(colab: Mock, api: FakeOmniAPI) -> None:
    other_reader = Mock(side_effect=AssertionError("must keep the originally selected provider"))

    def read(name: str) -> str:
        dbutils = SimpleNamespace(secrets=SimpleNamespace(get=other_reader))
        sys.modules["IPython"].get_ipython.return_value = SimpleNamespace(
            user_ns={"dbutils": dbutils}
        )
        return {BASE_URL_ENV: BASE_URL, API_KEY_ENV: DEFAULT_TOKEN}[name]

    colab.side_effect = read
    with OmniSession.builder.get_or_create() as session:
        session.verify()
    assert colab.call_count == 2
    other_reader.assert_not_called()


def test_secrets_can_be_disabled_even_when_a_notebook_is_detected(
    colab: Mock, databricks: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    importer = Mock(side_effect=AssertionError("disabled secrets must not import providers"))
    monkeypatch.setattr(importlib, "import_module", importer)
    with pytest.raises(CompileError, match=API_KEY_ENV):
        OmniSession.builder.host(BASE_URL).secrets(None).get_or_create()
    colab.assert_not_called()
    databricks.assert_not_called()
    importer.assert_not_called()


@pytest.mark.parametrize("provider", ["", "unknown", "COLAB", "google-colab"])
def test_invalid_provider_is_rejected_during_configuration(provider: str) -> None:
    with pytest.raises(CompileError, match="provider"):
        OmniSession.builder.secrets(provider)


@pytest.mark.parametrize("name", ["", " ", "\t\n"])
@pytest.mark.parametrize("option", ["api_key_name", "base_url_name", "scope"])
def test_blank_secret_configuration_is_rejected(option: str, name: str) -> None:
    with pytest.raises(CompileError):
        OmniSession.builder.secrets("databricks", **{option: name})


@pytest.mark.parametrize("provider", ["colab", "snowflake", "snowflake-legacy", None])
def test_scope_is_only_supported_for_databricks_or_auto(provider: str | None) -> None:
    with pytest.raises(CompileError, match="scope"):
        OmniSession.builder.secrets(provider, scope="omni")


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (SecretNotFoundError, "missing|not found|does not exist"),
        (NotebookAccessError, "denied|access"),
        (TimeoutException, "timed out|timeout|Colab UI"),
        (RuntimeError, "could not|unavailable|failed"),
    ],
)
def test_colab_errors_are_actionable_and_sanitized(
    error: type[Exception], message: str, colab: Mock
) -> None:
    colab.side_effect = error(DEFAULT_TOKEN)
    with pytest.raises(CompileError, match=message) as caught:
        OmniSession.builder.host(BASE_URL).get_or_create()
    assert API_KEY_ENV in str(caught.value)
    assert_no_secret_in_error(caught.value)


def test_colab_without_exception_classes_still_sanitizes_failures(
    colab: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    userdata = sys.modules["google.colab.userdata"]
    for name in ("SecretNotFoundError", "NotebookAccessError", "TimeoutException"):
        monkeypatch.delattr(userdata, name)
    colab.side_effect = RuntimeError(DEFAULT_TOKEN)
    with pytest.raises(CompileError) as caught:
        OmniSession.builder.host(BASE_URL).get_or_create()
    assert_no_secret_in_error(caught.value)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("value", ["", " \t", None, 123, {"password": DEFAULT_TOKEN}])
def test_invalid_secret_values_are_rejected_without_echoing_them(
    provider: str, value: object, request: pytest.FixtureRequest
) -> None:
    reader = reader_for(request, provider)
    reader.side_effect = None
    reader.return_value = value
    with pytest.raises(CompileError, match="non-empty string") as caught:
        (
            OmniSession.builder.host(BASE_URL)
            .secrets(provider, scope="omni" if provider == "databricks" else None)
            .get_or_create()
        )
    assert_no_secret_in_error(caught.value)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_unexpected_provider_exceptions_hide_the_entire_exception_chain(
    provider: str, request: pytest.FixtureRequest
) -> None:
    reader = reader_for(request, provider)

    def fail(*args: object, **kwargs: object) -> str:
        try:
            raise ValueError(DEFAULT_TOKEN)
        except ValueError as error:
            raise RuntimeError(DEFAULT_TOKEN) from error

    reader.side_effect = fail
    with pytest.raises(CompileError) as caught:
        (
            OmniSession.builder.host(BASE_URL)
            .secrets(provider, scope="omni" if provider == "databricks" else None)
            .get_or_create()
        )
    assert_no_secret_in_error(caught.value)


@pytest.mark.parametrize("provider", ["colab", "snowflake", "snowflake-legacy"])
def test_unavailable_provider_has_sanitized_error(provider: str) -> None:
    with pytest.raises(CompileError, match="unavailable") as caught:
        OmniSession.builder.host(BASE_URL).secrets(provider).get_or_create()
    assert "environment variables" in str(caught.value)
    assert_no_secret_in_error(caught.value)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (NotImplementedError, "unavailable"),
        (ValueError, "not found|missing|not accessible|not authorized"),
        (PermissionError, "denied|access"),
        (TimeoutError, "timed out|timeout"),
    ],
)
def test_snowflake_distinguishes_unsupported_runtime_from_inaccessible_secret(
    error: type[Exception], message: str, snowflake: Mock
) -> None:
    snowflake.side_effect = error(DEFAULT_TOKEN)
    with pytest.raises(CompileError, match=message) as caught:
        OmniSession.builder.host(BASE_URL).secrets("snowflake").get_or_create()
    assert_no_secret_in_error(caught.value)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (KeyError, "not found|missing|does not exist"),
        (FileNotFoundError, "unavailable"),
        (PermissionError, "denied|access"),
        (TimeoutError, "timed out|timeout"),
    ],
)
def test_snowflake_legacy_errors_are_actionable(
    error: type[Exception], message: str, snowflake_legacy: Mock
) -> None:
    snowflake_legacy.side_effect = error(DEFAULT_TOKEN)
    with pytest.raises(CompileError, match=message) as caught:
        OmniSession.builder.host(BASE_URL).secrets("snowflake-legacy").get_or_create()
    assert_no_secret_in_error(caught.value)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (PermissionError, "denied|access"),
        (TimeoutError, "timed out|timeout"),
        (RuntimeError, "could not|unavailable|failed"),
    ],
)
def test_databricks_errors_are_sanitized(
    error: type[Exception], message: str, databricks: Mock
) -> None:
    databricks.side_effect = error(DEFAULT_TOKEN)
    with pytest.raises(CompileError, match=message) as caught:
        OmniSession.builder.host(BASE_URL).secrets("databricks", scope="omni").get_or_create()
    assert_no_secret_in_error(caught.value)


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("RESOURCE_DOES_NOT_EXIST", "not found"),
        ("PERMISSION_DENIED", "denied"),
        ("UNAUTHENTICATED", "authentication.*unavailable"),
    ],
)
def test_databricks_structured_error_codes_are_classified_without_echoing_text(
    code: str, message: str, databricks: Mock
) -> None:
    class ProviderError(Exception):
        error_code = code

    databricks.side_effect = ProviderError(DEFAULT_TOKEN)
    with pytest.raises(CompileError, match=message) as caught:
        OmniSession.builder.host(BASE_URL).secrets("databricks", scope="omni").get_or_create()
    assert_no_secret_in_error(caught.value)


def test_local_python_without_ipython_does_not_import_notebook_libraries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "IPython", None)
    importer = Mock(side_effect=AssertionError("local configuration must not import notebook SDKs"))
    monkeypatch.setattr(importlib, "import_module", importer)
    with pytest.raises(CompileError, match=API_KEY_ENV):
        OmniSession.builder.host(BASE_URL).get_or_create()
    importer.assert_not_called()


def test_a_failed_selected_provider_never_probes_other_providers(
    colab: Mock, databricks: Mock, snowflake: Mock, snowflake_legacy: Mock
) -> None:
    colab.side_effect = SecretNotFoundError(DEFAULT_TOKEN)
    with pytest.raises(CompileError, match="not found"):
        OmniSession.builder.host(BASE_URL).secrets("colab").get_or_create()
    databricks.assert_not_called()
    snowflake.assert_not_called()
    snowflake_legacy.assert_not_called()


def test_exception_while_discovering_ipython_is_sanitized() -> None:
    sys.modules["IPython"].get_ipython.side_effect = RuntimeError(DEFAULT_TOKEN)
    with pytest.raises(CompileError, match=API_KEY_ENV) as caught:
        OmniSession.builder.host(BASE_URL).get_or_create()
    assert_no_secret_in_error(caught.value)


@pytest.mark.parametrize("dbutils", [None, 123, object(), SimpleNamespace(secrets="unrelated")])
def test_unrelated_ipython_dbutils_variable_does_not_select_databricks(dbutils: object) -> None:
    sys.modules["IPython"].get_ipython.return_value = SimpleNamespace(user_ns={"dbutils": dbutils})
    with pytest.raises(CompileError, match=API_KEY_ENV):
        OmniSession.builder.host(BASE_URL).get_or_create()


def test_databricks_user_global_fallback_ignores_unrelated_user_ns_variable(
    databricks: Mock, api: FakeOmniAPI
) -> None:
    getter = sys.modules["IPython"].get_ipython
    dbutils = getter.return_value.user_ns["dbutils"]
    getter.return_value = SimpleNamespace(
        user_ns={"dbutils": 123}, ns_table={"user_global": {"dbutils": dbutils}}
    )
    with OmniSession.builder.secrets(scope="omni").get_or_create() as session:
        session.verify()
    assert databricks.call_count == 2
