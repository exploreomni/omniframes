"""Exception hierarchy for omniframes.

Everything raised to users derives from :class:`OmniframesError`; warnings derive from
:class:`OmniframesWarning`. Transport code maps the three server error envelopes (see
docs/CONTRACT_NOTES.md §1) onto these types — HTTP details never leak above the transport.
"""

from __future__ import annotations

__all__ = [
    "AuthError",
    "CompileError",
    "FeatureFlagError",
    "ModelPermissionError",
    "OmniframesError",
    "OmniframesWarning",
    "QueryError",
    "QueryTimeoutError",
    "TransportError",
    "TruncationWarning",
]


class OmniframesError(Exception):
    """Base class for all omniframes errors."""


class TransportError(OmniframesError):
    """A network/protocol-level failure talking to the Omni API.

    ``status`` carries the HTTP status the failure came from, when there was one — ``None`` for
    a request that never got an answer at all.  It is the *only* HTTP detail that crosses the
    transport seam, and it does so as an attribute rather than as text: two endpoints change
    their advice based on the status (CONTRACT_NOTES §4), and recovering it by re-reading the
    message would make the wording load-bearing.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class AuthError(TransportError):
    """The bearer token was rejected (invalid, expired, or malformed header)."""


class FeatureFlagError(TransportError):
    """The organization does not have the `query-api` feature flag enabled.

    The remedy is administrative: an org admin must enable the Query API for the
    organization (Omni settings), after which the same key works unchanged.
    """


class ModelPermissionError(TransportError):
    """The key's user lacks a required model permission (e.g. QUERY_TOPICS, QUERY_FULL_MODEL)."""

    def __init__(
        self, message: str, *, permission: str | None = None, status: int | None = None
    ) -> None:
        super().__init__(message, status=status)
        self.permission = permission


class QueryError(OmniframesError):
    """A submitted job reached a terminal error state (in-band job error line).

    ``statement`` carries the OmniSQL text the server rejected, for the one failure that is
    omniframes' own fault rather than the user's: a tier-2 statement the model no longer binds
    (docs/SQLTIER.md §8).  It stays ``None`` for every other job error, including a raw-SQL job,
    whose SQL the user wrote and already has.
    """

    def __init__(
        self,
        message: str,
        *,
        error_type: str | None = None,
        job_id: str | None = None,
        statement: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.job_id = job_id
        self.statement = statement


class QueryTimeoutError(OmniframesError):
    """The client-side deadline elapsed while jobs were still running."""

    def __init__(self, message: str, *, remaining_job_ids: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.remaining_job_ids = remaining_job_ids


class CompileError(OmniframesError):
    """The logical plan cannot be compiled (bad field/alias/grain, invalid combination)."""


class OmniframesWarning(UserWarning):
    """Base class for omniframes warnings."""


class TruncationWarning(OmniframesWarning):
    """Returned row count equals the applied limit — the result is likely truncated."""
