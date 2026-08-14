"""Typed port of the Omni query-API wire contract.

Ported from the Omni monorepo @ ``86569cd50c6`` (`services/query-manager` Kotlin
serialization model + `packages/bi-app/app/omni-query` and `packages/types` on the client
side). ``docs/CONTRACT_NOTES.md`` in this repo is the source of truth for everything below;
section references in the docstrings point at it:

* §2.1 — the ``POST /api/v1/query/run`` request envelope → :class:`RunRequest`
* §3   — the query object → :class:`Query`
* §3.1 — the filter arms → :class:`Filter` and its subclasses
* §3.2 — time grains → :data:`GRAINS` / :func:`is_valid_grain`
* §3.4 — raw-SQL jobs → :meth:`Query.for_sql`
* §3.5 — ``staticQueryReferences`` → :attr:`Query.static_query_references`
* §3.6 — the parsed-OmniSQL path (tier 2) → :meth:`Query.for_omnisql`

Every type here is frozen and does exactly two things: hold a validated value and render
itself with ``to_wire()`` into a JSON-ready ``dict`` using the exact key spelling the server
expects. The envelope is camelCase throughout; the query object is a mix — ``modelId``,
``userEditedSQL``, ``rewriteSql``, ``sqlSortsEnabled`` and ``staticQueryReferences`` are
camelCase, everything else is snake_case (Jackson/kotlinx on the Kotlin planner side, which
**silently drops unknown keys** — a misspelling never errors, so the spelling here is
load-bearing).

Nothing in this module performs I/O.
"""

from __future__ import annotations

import abc
import enum
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, ClassVar, Final, Literal, TypeAlias

from omniframes.errors import CompileError

__all__ = [
    "DATE_GRAINS",
    "DEFAULT_FETCH_LIMIT",
    "DEFAULT_SERVER_LIMIT",
    "DURATION_GRAINS",
    "GRAINS",
    "GRAND_TOTAL_KEY",
    "HIGH_LIMIT_THRESHOLD",
    "QUERY_VERSION",
    "UNSET",
    "BooleanFilter",
    "CachePolicy",
    "Calculation",
    "CompositeFilter",
    "DateFilter",
    "DateFilterKind",
    "Filter",
    "FilterConjunction",
    "FilterType",
    "NullFilter",
    "NullSort",
    "NumberFilter",
    "NumberFilterKind",
    "NumberValue",
    "Query",
    "QueryFilter",
    "ResultType",
    "RunRequest",
    "Sort",
    "StringFilter",
    "StringFilterKind",
    "Unset",
    "UserAttributeFilter",
    "WireDict",
    "is_valid_grain",
]

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

#: ``version`` value every query carries (``QUERY_VERSION_CALC_PUSHDOWN``, CONTRACT_NOTES §3).
QUERY_VERSION: Final = 9

#: Limit applied when the user never called ``.limit()`` (docs/DESIGN.md §3).
DEFAULT_FETCH_LIMIT: Final = 50_000

#: What the SERVER applies when a query object arrives with no ``limit`` key at all
#: (``createQuery``'s ``DEFAULT_ROW_LIMIT``, CONTRACT_NOTES §3).  Omniframes never writes such a
#: query, but a stored or AI-generated blob it was handed may well be missing the key — and an
#: absent key is 1000 rows, not "unlimited".
DEFAULT_SERVER_LIMIT: Final = 1_000

#: Above this the server switches into high-limit override mode (CONTRACT_NOTES §2.1).
HIGH_LIMIT_THRESHOLD: Final = 50_000

#: Grand-total key accepted by ``column_totals`` / ``row_totals`` (CONTRACT_NOTES §2.7/§3).
GRAND_TOTAL_KEY: Final = "::total::"

_AGGREGATION: Final = "aggregation"


class _Unset(enum.Enum):
    """Sentinel type separating "never set" from an explicit ``None``.

    An ``enum`` (rather than ``object()``) so that ``Literal[_Unset.UNSET]`` narrows under
    mypy: ``limit`` genuinely has three states and each one puts something different on the
    wire.
    """

    UNSET = "UNSET"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "UNSET"


#: The "never set" sentinel — see :attr:`Query.limit`.
UNSET: Final = _Unset.UNSET

#: Type of the :data:`UNSET` sentinel, for annotating tri-state fields.
Unset: TypeAlias = Literal[_Unset.UNSET]

#: What a caller may hand to :class:`NumberFilter`; always serialized as a string.
NumberValue: TypeAlias = "int | float | Decimal | str"

WireDict: TypeAlias = "dict[str, Any]"


# --------------------------------------------------------------------------------------
# Enums (CONTRACT_NOTES §2.1, §3.1)
# --------------------------------------------------------------------------------------


class CachePolicy(enum.Enum):
    """``cache`` on the envelope. The OpenAPI values (``disabled``/``normal``/…) 400."""

    STANDARD = "Standard"
    SKIP_REQUERY = "SkipRequery"
    SKIP_CACHE = "SkipCache"
    SKIP_CACHE_AND_REBUILD_EXTRACTS = "SkipCacheAndRebuildExtracts"


class ResultType(enum.Enum):
    """``resultType`` on the envelope — switches to single-document mode (§2.5)."""

    CSV = "csv"
    JSON = "json"
    XLSX = "xlsx"


class NullSort(enum.Enum):
    """``sorts[*].null_sort``."""

    LAST = "LAST"
    FIRST = "FIRST"
    DIALECT_DEFAULT = "DIALECT_DEFAULT"
    OMNI_DEFAULT = "OMNI_DEFAULT"


class FilterType(enum.Enum):
    """The ``type`` discriminator of a filter entry."""

    STRING = "string"
    NUMBER = "number"
    DATE = "date"
    BOOLEAN = "boolean"
    NULL = "null"
    COMPOSITE = "composite"
    QUERY = "query"
    USER_ATTRIBUTE = "user_attribute"


class StringFilterKind(enum.Enum):
    """``kind`` for ``type: "string"``. EQUALS with many values becomes IN; others OR."""

    CONTAINS = "CONTAINS"
    ENDS_WITH = "ENDS_WITH"
    STARTS_WITH = "STARTS_WITH"
    EQUALS = "EQUALS"
    IS_EMPTY = "IS_EMPTY"
    SQL_LIKE = "SQL_LIKE"


class NumberFilterKind(enum.Enum):
    """``kind`` for ``type: "number"``. BETWEEN is ``[lower, upper]``, upper exclusive."""

    LESS_THAN = "LESS_THAN"
    GREATER_THAN = "GREATER_THAN"
    EQUALS = "EQUALS"
    BETWEEN = "BETWEEN"


class DateFilterKind(enum.Enum):
    """``kind`` for ``type: "date"``."""

    BETWEEN = "BETWEEN"
    ON_OR_AFTER = "ON_OR_AFTER"
    BEFORE = "BEFORE"
    TIME_FOR_INTERVAL_DURATION = "TIME_FOR_INTERVAL_DURATION"
    TIME_FOR_UNIT_DURATION = "TIME_FOR_UNIT_DURATION"
    QUERY_OFFSET = "QUERY_OFFSET"
    IS_ON_DAY_OF_WEEK = "IS_ON_DAY_OF_WEEK"
    IS_ON_DAY_OF_MONTH = "IS_ON_DAY_OF_MONTH"
    IS_ON_DAY_OF_QUARTER = "IS_ON_DAY_OF_QUARTER"
    IS_ON_DAY_OF_YEAR = "IS_ON_DAY_OF_YEAR"
    IS_IN_MONTH_OF_YEAR = "IS_IN_MONTH_OF_YEAR"
    IS_IN_QUARTER_OF_YEAR = "IS_IN_QUARTER_OF_YEAR"
    IS_IN_WEEK_OF_YEAR = "IS_IN_WEEK_OF_YEAR"
    IS_AT_HOUR_OF_DAY = "IS_AT_HOUR_OF_DAY"


class FilterConjunction(enum.Enum):
    """``conjunction`` for ``type: "composite"``."""

    AND = "AND"
    OR = "OR"


# --------------------------------------------------------------------------------------
# Time grains (CONTRACT_NOTES §3.2)
# --------------------------------------------------------------------------------------

#: Grains available on date/timestamp dimensions.
DATE_GRAINS: Final[frozenset[str]] = frozenset(
    {
        "year",
        "quarter",
        "month",
        "week",
        "date",
        "hour",
        "minute",
        "second",
        "millisecond",
        "quarter_of_year",
        "week_of_year",
        "day_of_week_name",
        "day_of_week_num",
        "month_name",
        "month_num",
        "hour_of_day",
        "day_of_month",
        "day_of_year",
        "day_of_quarter",
        "fiscal_year",
        "fiscal_quarter",
        "epoch",
        "time_of_day",
    }
)

#: Grains available on duration dimensions.
DURATION_GRAINS: Final[frozenset[str]] = frozenset(
    {
        "seconds",
        "minutes",
        "hours",
        "days",
        "weeks",
        "months",
        "quarters",
        "years",
    }
)

#: Every grain Omni accepts in ``field[grain]``. Canonical spelling is lowercase; the server
#: matches case-insensitively. An invalid grain does NOT error — the field silently lands in
#: ``summary.missing_fields``, which is why this list exists client-side.
GRAINS: Final[frozenset[str]] = DATE_GRAINS | DURATION_GRAINS


def is_valid_grain(grain: str) -> bool:
    """Return whether ``grain`` is a grain Omni understands (case-insensitive)."""
    return grain.lower() in GRAINS


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _require_uuid(value: str, where: str) -> None:
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        raise CompileError(f"{where} must be a UUID string, got {value!r}") from None


def _coerce_number(value: NumberValue, kind: NumberFilterKind) -> str:
    """Render one number-filter value as the string the raw wire path requires (§3.1)."""
    if isinstance(value, bool):
        raise CompileError(
            f"number filter ({kind.value}) values must be numeric; got the boolean {value!r}. "
            "Use BooleanFilter for boolean fields."
        )
    if isinstance(value, str):
        text = value
    elif isinstance(value, int | float | Decimal):
        text = str(value)
    else:
        raise CompileError(
            f"number filter ({kind.value}) values must be int, float, Decimal or str; "
            f"got {type(value).__name__}"
        )
    if text.strip().lower() == "null":
        raise CompileError(
            '"null" is not a valid number filter value; use NullFilter to filter on NULL.'
        )
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError):
        raise CompileError(
            f"number filter ({kind.value}) value {value!r} is not a number"
        ) from None
    if not parsed.is_finite():
        raise CompileError(
            f"number filter ({kind.value}) value {value!r} is not finite; the warehouse has no "
            "literal for it"
        )
    return text


# --------------------------------------------------------------------------------------
# Filters (CONTRACT_NOTES §3.1)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Filter(abc.ABC):
    """One entry of ``query.filters``, keyed by field name (filters AND across fields).

    The three keys declared here are common to every arm and are keyword-only so that each
    subclass can keep its own positional signature.
    """

    is_negative: bool | None = field(default=None, kw_only=True)
    cancel_query_filter: bool | None = field(default=None, kw_only=True)
    ignore_if_unjoinable: bool | None = field(default=None, kw_only=True)

    filter_type: ClassVar[FilterType]

    @abc.abstractmethod
    def to_wire(self) -> WireDict:
        """Return the JSON-ready payload for this filter."""

    # Arms with nothing to check (boolean, null) inherit this no-op body deliberately.
    def validate(self) -> None:  # noqa: B027
        """Raise :class:`CompileError` if this filter cannot be compiled. Default: no-op."""

    def _base_wire(self) -> WireDict:
        wire: WireDict = {"type": self.filter_type.value}
        if self.is_negative is not None:
            wire["is_negative"] = self.is_negative
        if self.cancel_query_filter is not None:
            wire["cancel_query_filter"] = self.cancel_query_filter
        if self.ignore_if_unjoinable is not None:
            wire["ignore_if_unjoinable"] = self.ignore_if_unjoinable
        return wire


@dataclass(frozen=True)
class StringFilter(Filter):
    """``type: "string"`` — CONTAINS / ENDS_WITH / STARTS_WITH / EQUALS / IS_EMPTY / SQL_LIKE."""

    kind: StringFilterKind
    values: Sequence[str] = ()
    case_insensitive: bool | None = None

    filter_type = FilterType.STRING

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))

    def to_wire(self) -> WireDict:
        wire = self._base_wire()
        wire["kind"] = self.kind.value
        wire["values"] = list(self.values)
        if self.case_insensitive is not None:
            wire["case_insensitive"] = self.case_insensitive
        return wire

    def validate(self) -> None:
        if self.kind is StringFilterKind.IS_EMPTY:
            if self.values:
                raise CompileError("string filter IS_EMPTY takes no values")
        elif not self.values:
            raise CompileError(f"string filter {self.kind.value} needs at least one value")


@dataclass(frozen=True)
class NumberFilter(Filter):
    """``type: "number"``.

    Values go on the wire as **strings** — the raw API path has no number→string transform,
    so ``NumberFilter(NumberFilterKind.EQUALS, [42])`` serializes ``"values": ["42"]``.
    """

    kind: NumberFilterKind
    values: Sequence[NumberValue] = ()
    is_inclusive: bool | None = None

    filter_type = FilterType.NUMBER

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "values", tuple(_coerce_number(value, self.kind) for value in self.values)
        )

    def to_wire(self) -> WireDict:
        wire = self._base_wire()
        wire["kind"] = self.kind.value
        wire["values"] = [str(value) for value in self.values]
        if self.is_inclusive is not None:
            wire["is_inclusive"] = self.is_inclusive
        return wire

    def validate(self) -> None:
        if self.kind is NumberFilterKind.BETWEEN:
            if len(self.values) != 2:
                raise CompileError(
                    "number filter BETWEEN needs exactly two values [lower, upper] "
                    f"(upper exclusive); got {len(self.values)}"
                )
        elif not self.values:
            raise CompileError(f"number filter {self.kind.value} needs at least one value")


@dataclass(frozen=True)
class DateFilter(Filter):
    """``type: "date"`` — see CONTRACT_NOTES §3.1 for the literal grammar of the sides."""

    kind: DateFilterKind
    left_side: str | None = None
    right_side: str | None = None

    filter_type = FilterType.DATE

    _NEEDS_LEFT: ClassVar[frozenset[DateFilterKind]] = frozenset(
        {
            DateFilterKind.BETWEEN,
            DateFilterKind.ON_OR_AFTER,
            DateFilterKind.TIME_FOR_INTERVAL_DURATION,
            DateFilterKind.TIME_FOR_UNIT_DURATION,
        }
    )
    _NEEDS_RIGHT: ClassVar[frozenset[DateFilterKind]] = frozenset(
        {
            DateFilterKind.BETWEEN,
            DateFilterKind.BEFORE,
            DateFilterKind.TIME_FOR_INTERVAL_DURATION,
        }
    )

    def to_wire(self) -> WireDict:
        wire = self._base_wire()
        wire["kind"] = self.kind.value
        if self.left_side is not None:
            wire["left_side"] = self.left_side
        if self.right_side is not None:
            wire["right_side"] = self.right_side
        return wire

    def validate(self) -> None:
        if self.kind in self._NEEDS_LEFT and not self.left_side:
            raise CompileError(f"date filter {self.kind.value} requires left_side")
        if self.kind in self._NEEDS_RIGHT and not self.right_side:
            raise CompileError(f"date filter {self.kind.value} requires right_side")


@dataclass(frozen=True)
class BooleanFilter(Filter):
    """``type: "boolean"``.

    ``is_negative=False`` means "is true", ``is_negative=True`` means "is false", and leaving
    it unset is a no-op placeholder the server ignores.
    """

    treat_nulls_as_false: bool | None = None

    filter_type = FilterType.BOOLEAN

    def to_wire(self) -> WireDict:
        wire = self._base_wire()
        if self.treat_nulls_as_false is not None:
            wire["treat_nulls_as_false"] = self.treat_nulls_as_false
        return wire


@dataclass(frozen=True)
class NullFilter(Filter):
    """``type: "null"`` — IS NULL, or IS NOT NULL with ``is_negative=True``."""

    filter_type = FilterType.NULL

    def to_wire(self) -> WireDict:
        return self._base_wire()


@dataclass(frozen=True)
class CompositeFilter(Filter):
    """``type: "composite"`` — AND/OR of other filters on the SAME field, recursively.

    There is no depth cap on ``/query/run`` (the cap of 4 applies to document writes only).
    """

    conjunction: FilterConjunction
    filters: Sequence[Filter] = ()

    filter_type = FilterType.COMPOSITE

    def __post_init__(self) -> None:
        object.__setattr__(self, "filters", tuple(self.filters))

    def to_wire(self) -> WireDict:
        wire = self._base_wire()
        wire["conjunction"] = self.conjunction.value
        wire["filters"] = [child.to_wire() for child in self.filters]
        return wire

    def validate(self) -> None:
        if not self.filters:
            raise CompileError("composite filter needs at least one child filter")
        for child in self.filters:
            child.validate()


@dataclass(frozen=True)
class QueryFilter(Filter):
    """``type: "query"`` — filter a field by the results of a referenced query.

    ``query_id`` is a key of :attr:`Query.static_query_references` (§3.5).
    """

    field_name: str
    query_id: str
    disregard_limit: bool | None = None

    filter_type = FilterType.QUERY

    def to_wire(self) -> WireDict:
        wire = self._base_wire()
        wire["field_name"] = self.field_name
        wire["query_id"] = self.query_id
        if self.disregard_limit is not None:
            wire["disregard_limit"] = self.disregard_limit
        return wire

    def validate(self) -> None:
        if not self.field_name:
            raise CompileError("query filter requires a field_name")
        if not self.query_id:
            raise CompileError("query filter requires a query_id")


@dataclass(frozen=True)
class UserAttributeFilter(Filter):
    """``type: "user_attribute"`` — compare the field against a user attribute's value."""

    user_attribute_name: str

    filter_type = FilterType.USER_ATTRIBUTE

    def to_wire(self) -> WireDict:
        wire = self._base_wire()
        wire["user_attribute_name"] = self.user_attribute_name
        return wire

    def validate(self) -> None:
        if not self.user_attribute_name:
            raise CompileError("user_attribute filter requires a user_attribute_name")


# --------------------------------------------------------------------------------------
# Sorts and calculations
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sort:
    """One entry of ``query.sorts``.

    ``column_name`` is the EXACT field name as selected, brackets included
    (``order_items.created_at[month]``).
    """

    column_name: str
    sort_descending: bool = False
    null_sort: NullSort = NullSort.OMNI_DEFAULT

    def to_wire(self) -> WireDict:
        return {
            "column_name": self.column_name,
            "sort_descending": self.sort_descending,
            "null_sort": self.null_sort.value,
        }

    def validate(self) -> None:
        if not self.column_name:
            raise CompileError("sort requires a column_name")


@dataclass(frozen=True)
class Calculation:
    """``query.calculations`` entry — a placeholder for wire completeness (§3.3).

    Omniframes does **not** emit calculations in 0.1: ``sql_expression`` must be a serialized
    parse tree (``original_formula`` is not parsed on this endpoint), and calcs are post-limit
    row operations rather than in-DB aggregation. Ad-hoc expressions ride tier 2/3 instead.
    ``sql_expression`` is passed through verbatim and is not deep-copied.
    """

    calc_name: str
    sql_expression: Mapping[str, Any] = field(default_factory=dict)

    def to_wire(self) -> WireDict:
        return {"calc_name": self.calc_name, "sql_expression": dict(self.sql_expression)}

    def validate(self) -> None:
        if not self.calc_name:
            raise CompileError("calculation requires a calc_name")
        if not self.sql_expression:
            raise CompileError(
                f"calculation {self.calc_name!r} requires a sql_expression: the query API does "
                "not parse original_formula"
            )


# --------------------------------------------------------------------------------------
# The query object (CONTRACT_NOTES §3)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Query:
    """The ``query`` object of a run request.

    The server fills in defaults for absent keys; omniframes instead sends explicit values for
    everything it controls, so :meth:`to_wire` always emits ``limit`` and ``version`` along
    with the core collections. Truly optional keys (``join_paths_from_topic_name``,
    ``rewriteSql``, ``sqlSortsEnabled``, ``staticQueryReferences``) are omitted while unset.

    ``userEditedSQL`` has two mutually exclusive readings, and the server picks between them
    from the ``rewriteSql`` key alone (CONTRACT_NOTES §3.5): ``false`` runs the text verbatim
    on the warehouse (:meth:`for_sql`), while an **absent** key parses it as OmniSQL and plans
    a governed model job (:meth:`for_omnisql`, tier 2).  :attr:`omnisql` records which one was
    intended so :meth:`validate` can refuse the third, unintended reading — see §3.6.

    ``limit`` is a trichotomy:

    * :data:`UNSET` (default) → :data:`DEFAULT_FETCH_LIMIT` goes on the wire;
    * ``None`` → ``"limit": null``, i.e. unlimited (rejected by the server together with
      non-empty ``pivots`` unless ``resultType`` is set);
    * ``int`` → that value, where anything above :data:`HIGH_LIMIT_THRESHOLD` puts the server
      into high-limit mode.

    Sequence/mapping arguments are copied at construction, so the instance is immutable as
    long as callers do not mutate objects nested inside a ``Calculation.sql_expression``.
    """

    model_id: str
    fields: Sequence[str] = ()
    table: str = ""
    join_paths_from_topic_name: str | None = None
    filters: Mapping[str, Filter] = field(default_factory=dict)
    sorts: Sequence[Sort] = ()
    limit: int | Unset | None = UNSET
    offset: int = 0
    pivots: Sequence[str] = ()
    calculations: Sequence[Calculation] = ()
    fill_fields: Sequence[str] = ()
    column_totals: Sequence[str] = ()
    row_totals: Sequence[str] = ()
    user_edited_sql: str = ""
    rewrite_sql: bool | None = None
    sql_sorts_enabled: bool | None = None
    static_query_references: Mapping[str, Query] = field(default_factory=dict)
    #: NOT a wire key: ``userEditedSQL`` is compiled OmniSQL, to be parsed against the model
    #: (``rewriteSql`` absent — CONTRACT_NOTES §3.6).  The wire cannot express the distinction
    #: between "no rewriteSql because OmniSQL" and "no rewriteSql because nobody set it", so it
    #: is carried here and checked by :meth:`validate`.
    omnisql: bool = False
    default_group_by: bool = True
    version: int = QUERY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "sorts", tuple(self.sorts))
        object.__setattr__(self, "pivots", tuple(self.pivots))
        object.__setattr__(self, "calculations", tuple(self.calculations))
        object.__setattr__(self, "fill_fields", tuple(self.fill_fields))
        object.__setattr__(self, "column_totals", tuple(self.column_totals))
        object.__setattr__(self, "row_totals", tuple(self.row_totals))
        object.__setattr__(self, "filters", dict(self.filters))
        object.__setattr__(self, "static_query_references", dict(self.static_query_references))

    # -- construction helpers ----------------------------------------------------------

    @classmethod
    def for_sql(
        cls,
        model_id: str,
        sql: str,
        *,
        sql_sorts_enabled: bool = True,
        **kwargs: Any,
    ) -> Query:
        """Build a raw-SQL job (§3.4).

        ``rewriteSql`` is forced to ``False`` — with the default ``true`` the server parses
        ``userEditedSQL`` as OmniSQL and plans a governed model job instead of running the
        text verbatim (CONTRACT_NOTES §3.5). ``sqlSortsEnabled``
        controls whether ``sorts``/``calculations``/``column_totals`` are applied on top of
        the SQL result; when false the server strips them.
        """
        return cls(
            model_id=model_id,
            user_edited_sql=sql,
            rewrite_sql=False,
            sql_sorts_enabled=sql_sorts_enabled,
            **kwargs,
        )

    @classmethod
    def for_omnisql(
        cls,
        model_id: str,
        sql: str,
        *,
        limit: int | Unset | None = UNSET,
        offset: int = 0,
    ) -> Query:
        """Build a tier-2 OmniSQL job — one statement the server parses against the model (§3.6).

        ``rewriteSql`` stays **absent** (not ``False``): that is what selects the parsed path,
        where ``${view.field}`` / ``${topic}`` refs resolve, measures expand to their governed
        SQL and row-level policies apply.  No ``staticQueryReferences``, no ``sqlSortsEnabled``
        — the statement is the whole plan, and the signature has no seam for either.

        ``limit``/``offset`` mirror what the statement's own ``LIMIT``/``OFFSET`` say, but they
        are **client bookkeeping only**: the server ignores the query object's limit on this
        path, so the always-explicit-limit invariant lives in the SQL text (docs/SQLTIER.md §2).
        Mirroring them keeps ``effective_limit`` — and with it the truncation warning — honest.
        """
        return cls(
            model_id=model_id,
            user_edited_sql=sql,
            rewrite_sql=None,
            omnisql=True,
            limit=limit,
            offset=offset,
        )

    # -- limit trichotomy --------------------------------------------------------------

    @property
    def limit_is_unset(self) -> bool:
        """True when no limit was ever chosen (the default fetch limit will be sent)."""
        return self.limit is UNSET

    @property
    def effective_limit(self) -> int | None:
        """The value :meth:`to_wire` puts in ``limit``; ``None`` means unlimited."""
        if self.limit is UNSET:
            return DEFAULT_FETCH_LIMIT
        return self.limit

    @property
    def is_high_limit(self) -> bool:
        """True when the applied limit pushes the server into high-limit mode."""
        limit = self.effective_limit
        return limit is None or limit > HIGH_LIMIT_THRESHOLD

    # -- serialization -----------------------------------------------------------------

    def to_wire(self) -> WireDict:
        """Return the JSON-ready query object. Call :meth:`validate` first."""
        wire: WireDict = {
            "modelId": self.model_id,
            "table": self.table,
            "fields": list(self.fields),
            "filters": {name: flt.to_wire() for name, flt in self.filters.items()},
            "sorts": [sort.to_wire() for sort in self.sorts],
            "limit": self.effective_limit,
            "offset": self.offset,
            "pivots": list(self.pivots),
            "calculations": [calc.to_wire() for calc in self.calculations],
            "fill_fields": list(self.fill_fields),
            "column_totals": {key: {"type": _AGGREGATION} for key in self.column_totals},
            "row_totals": {key: {"type": _AGGREGATION} for key in self.row_totals},
            "userEditedSQL": self.user_edited_sql,
            "default_group_by": self.default_group_by,
            "version": self.version,
        }
        if self.join_paths_from_topic_name is not None:
            wire["join_paths_from_topic_name"] = self.join_paths_from_topic_name
        if self.rewrite_sql is not None:
            wire["rewriteSql"] = self.rewrite_sql
        if self.sql_sorts_enabled is not None:
            wire["sqlSortsEnabled"] = self.sql_sorts_enabled
        if self.static_query_references:
            wire["staticQueryReferences"] = {
                key: ref.to_reference_wire() for key, ref in self.static_query_references.items()
            }
        return wire

    def to_reference_wire(self) -> WireDict:
        """Serialize this query as a ``staticQueryReferences`` entry (§3.5).

        Identical to :meth:`to_wire` plus the extra **snake_case** ``model_id`` the referenced
        query carries alongside ``modelId``.
        """
        wire = self.to_wire()
        wire["model_id"] = self.model_id
        return wire

    # -- validation --------------------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`CompileError` for anything the server would reject or silently drop."""
        _require_uuid(self.model_id, "query.modelId")

        is_sql_job = bool(self.user_edited_sql)
        if not self.fields and not is_sql_job:
            raise CompileError("query.fields must not be empty")

        if isinstance(self.limit, bool):
            raise CompileError("query.limit must be a positive integer or None, not a bool")
        if isinstance(self.limit, int) and self.limit <= 0:
            raise CompileError(
                f"query.limit must be a positive integer or None (unlimited); got {self.limit}"
            )
        if self.offset < 0:
            raise CompileError(f"query.offset must not be negative; got {self.offset}")
        if self.version != QUERY_VERSION:
            raise CompileError(
                f"query.version must be {QUERY_VERSION} (QUERY_VERSION_CALC_PUSHDOWN); "
                f"got {self.version}"
            )

        if not self.table and not self.join_paths_from_topic_name and not is_sql_job:
            raise CompileError(
                "query needs a table (base view) or join_paths_from_topic_name (topic)"
            )

        missing_pivots = [name for name in self.pivots if name not in self.fields]
        if missing_pivots:
            raise CompileError(
                f"pivot fields must also appear in query.fields; missing: {missing_pivots}"
            )

        self._validate_sql_path(is_sql_job)

        for name, flt in self.filters.items():
            if not name:
                raise CompileError("query.filters keys must be non-empty field names")
            flt.validate()
        self._validate_query_filter_references()

        for sort in self.sorts:
            sort.validate()

        for calc in self.calculations:
            calc.validate()
            if calc.calc_name not in self.fields:
                raise CompileError(
                    f"calculation {calc.calc_name!r} must also appear in query.fields"
                )

        for key, reference in self.static_query_references.items():
            if not key:
                raise CompileError("staticQueryReferences keys must be non-empty")
            reference.validate()

    def _validate_sql_path(self, is_sql_job: bool) -> None:
        """The XOR between the two ``userEditedSQL`` readings (CONTRACT_NOTES §3.5/§3.6).

        Exactly one of them must be chosen explicitly: ``rewriteSql: false`` (verbatim text,
        :meth:`for_sql`) or the :attr:`omnisql` flag with ``rewriteSql`` absent (parsed OmniSQL,
        :meth:`for_omnisql`).  Neither is the silent failure: the server takes the absent key
        as "parse it", so text meant for the warehouse comes back as a governed model job — or
        fails to bind — with nothing on the wire saying so.
        """
        if not is_sql_job:
            if self.omnisql:
                raise CompileError(
                    "the omnisql flag marks userEditedSQL as compiled OmniSQL, but this query "
                    "carries no SQL"
                )
            return

        if not self.omnisql:
            if self.rewrite_sql is not False:
                raise CompileError(
                    "userEditedSQL requires rewriteSql=False (verbatim SQL, Query.for_sql) or "
                    "the omnisql flag with rewriteSql unset (compiled OmniSQL, "
                    "Query.for_omnisql); with neither, the server parses the text as OmniSQL "
                    "and plans a governed model job"
                )
            return

        if self.rewrite_sql is not None:
            raise CompileError(
                "an OmniSQL query must leave rewriteSql unset: the absent key is what selects "
                f"the parsed path, and rewriteSql={self.rewrite_sql!r} sends the statement to "
                "the warehouse verbatim, ${...} refs and all (use Query.for_sql for verbatim "
                "SQL)"
            )
        if self.sql_sorts_enabled is not None:
            raise CompileError(
                "sqlSortsEnabled belongs to verbatim SQL jobs; an OmniSQL statement carries its "
                "own ORDER BY (CONTRACT_NOTES §3.6)"
            )
        if self.static_query_references:
            raise CompileError(
                "an OmniSQL statement cannot reference staticQueryReferences: a refKey is not a "
                "table, and the ${...} refs bind against the model instead (CONTRACT_NOTES §3.5)"
            )

    def _validate_query_filter_references(self) -> None:
        known = set(self.static_query_references)
        for name, flt in self.filters.items():
            for query_id in _query_filter_ids(flt):
                if query_id not in known:
                    raise CompileError(
                        f"filter on {name!r} references query_id {query_id!r}, which is not a "
                        "key of staticQueryReferences"
                    )


def _query_filter_ids(flt: Filter) -> list[str]:
    """Collect the ``query_id``s used by a filter, descending through composites."""
    if isinstance(flt, QueryFilter):
        return [flt.query_id]
    if isinstance(flt, CompositeFilter):
        return [query_id for child in flt.filters for query_id in _query_filter_ids(child)]
    return []


# --------------------------------------------------------------------------------------
# The request envelope (CONTRACT_NOTES §2.1)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRequest:
    """The body of ``POST /api/v1/query/run``.

    Optional keys are omitted while unset, which lets the server apply its own defaults
    (``cache`` defaults to ``SkipRequery``). ``branchId`` is top-level ONLY — nested inside
    ``query`` it is a hard 400, which is why :class:`Query` has no branch field at all.
    """

    query: Query
    branch_id: str | None = None
    cache: CachePolicy | None = None
    result_type: ResultType | None = None
    format_results: bool | None = None
    plan_only: bool | None = None
    timezone: str | None = None
    user_id: str | None = None
    workbook_url: bool | None = None

    def to_wire(self) -> WireDict:
        """Return the JSON-ready request body. Call :meth:`validate` first."""
        wire: WireDict = {"query": self.query.to_wire()}
        if self.branch_id is not None:
            wire["branchId"] = self.branch_id
        if self.cache is not None:
            wire["cache"] = self.cache.value
        if self.result_type is not None:
            wire["resultType"] = self.result_type.value
        if self.format_results is not None:
            wire["formatResults"] = self.format_results
        if self.plan_only is not None:
            wire["planOnly"] = self.plan_only
        if self.timezone is not None:
            wire["timezone"] = self.timezone
        if self.user_id is not None:
            wire["userId"] = self.user_id
        if self.workbook_url is not None:
            wire["workbookUrl"] = self.workbook_url
        return wire

    def validate(self, *, user_id_query_param: str | None = None) -> None:
        """Apply the server's 400-refinements client-side, raising :class:`CompileError`.

        Pass ``user_id_query_param`` when the transport also intends to send ``?userId=`` —
        supplying it in both places is a 400.
        """
        self.query.validate()

        if self.plan_only and self.result_type is not None:
            raise CompileError(
                "planOnly and resultType cannot both be provided (the server rejects this "
                "combination with 400)"
            )
        if self.plan_only and self.workbook_url:
            raise CompileError(
                "planOnly and workbookUrl cannot both be provided (the server rejects this "
                "combination with 400)"
            )
        if self.workbook_url and self.query.static_query_references:
            raise CompileError(
                "workbookUrl is not supported for queries with staticQueryReferences"
            )
        if self.format_results is not None and self.result_type is None:
            raise CompileError("formatResults cannot be provided without resultType")
        if self.query.effective_limit is None and self.query.pivots and self.result_type is None:
            raise CompileError(
                "Unlimited limit (null) cannot be used with pivoted queries unless resultType "
                "is set"
            )
        if self.user_id is not None and user_id_query_param is not None:
            raise CompileError(
                "userId may be provided in either the request body or as a query parameter, "
                "but not both"
            )

        if self.branch_id is not None:
            _require_uuid(self.branch_id, "branchId")
        if self.user_id is not None:
            _require_uuid(self.user_id, "userId")
        if self.timezone is not None and not self.timezone:
            raise CompileError("timezone must be a non-empty IANA identifier")
