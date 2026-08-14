"""Omniframes: a PySpark-style DataFrame library for Omni.

You write dataframe code; it compiles into governed semantic queries and pushes
as much compute as possible into Omni's SQL execution layer::

    from omniframes import OmniSession, functions as F

    session = OmniSession.builder.host("acme.omni.co").api_key_from_env().get_or_create()
    df = session.read.topic("bench_ecommerce", "order_items")
    (
        df.select("users.state", "order_items.status")
        .filter(F.col("users.state") == "California")
        .limit(10)
        .to_pandas()
    )
"""

__version__ = "0.1.0.dev0"

# `__version__` is assigned before these imports on purpose: the transport stamps it into its
# User-Agent with `from omniframes import __version__`, and importing the submodules first
# would reach that line while this module is still empty.
from omniframes import functions
from omniframes.column import Column
from omniframes.dataframe import DataFrame, GroupedData
from omniframes.errors import (
    AuthError,
    CompileError,
    FeatureFlagError,
    ModelPermissionError,
    OmniframesError,
    QueryError,
    QueryTimeoutError,
    TransportError,
    TruncationWarning,
)
from omniframes.session import OmniSession
from omniframes.types import OmniDataType, OmniField, OmniSchema

__all__ = [
    "AuthError",
    "Column",
    "CompileError",
    "DataFrame",
    "FeatureFlagError",
    "GroupedData",
    "ModelPermissionError",
    "OmniDataType",
    "OmniField",
    "OmniSchema",
    "OmniSession",
    "OmniframesError",
    "QueryError",
    "QueryTimeoutError",
    "TransportError",
    "TruncationWarning",
    "__version__",
    "functions",
]
