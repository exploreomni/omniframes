"""NDJSON protocol parsing against checked-in wire bodies (CONTRACT_NOTES §2.2)."""

from __future__ import annotations

import pytest

from omniframes.errors import TransportError
from omniframes.transport.ndjson import (
    FAILURE_STATUSES,
    REDACTED_ERROR_MESSAGE,
    TERMINAL_STATUSES,
    Footer,
    Header,
    JobLine,
    JobStatus,
    ParsedResponse,
    StreamAccumulator,
    TrailingError,
    parse_line,
    parse_response,
)
from tests.wire import FIXTURES_DIR, read_fixture
from tests.wire.build_fixtures import (
    JOB_ERROR,
    JOB_EXECUTING,
    JOB_FAST,
    JOB_HAPPY,
    JOB_KILLED,
    JOB_SLOW,
    JOB_UNKNOWN_STATUS,
    JOB_UNPLANNED,
    build_fixtures,
)


def parsed(name: str) -> ParsedResponse:
    return parse_response(read_fixture(name))


# --------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------


def test_every_builder_scenario_is_checked_in():
    on_disk = {path.name for path in FIXTURES_DIR.glob("*.ndjson")}

    assert on_disk == set(build_fixtures())


def test_the_fixture_builder_is_deterministic():
    assert build_fixtures() == build_fixtures()


def test_only_the_upstream_failure_tail_is_unterminated():
    for name in build_fixtures():
        body = read_fixture(name)
        has_trailing_error = parse_response(body).trailing_error is not None
        assert body.endswith(b"\n") == (not has_trailing_error), name


# --------------------------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------------------------


def test_happy_response_has_header_job_and_footer():
    response = parsed("happy_single_job.ndjson")

    assert [type(line) for line in response.lines] == [Header, JobLine, Footer]
    assert isinstance(response.header, Header)
    assert response.header.jobs_submitted == {JOB_HAPPY: "cri-happy"}
    assert response.header.job_ids == (JOB_HAPPY,)

    (job,) = response.jobs
    assert job.job_id == JOB_HAPPY
    assert job.status is JobStatus.COMPLETE
    assert job.client_result_id == "cri-happy"
    assert job.result is not None
    assert job.summary is not None
    assert job.stream_stats == {"server_stream": 42}
    assert job.is_terminal
    assert not job.is_failure
    assert job.failure_reason is None


def test_footer_timed_out_is_a_string_on_the_wire():
    done = parsed("happy_single_job.ndjson").footer
    running = parsed("unknown_status.ndjson").footer

    assert done is not None
    assert done.timed_out_raw == "false"
    assert done.timed_out is False
    assert done.remaining_job_ids == ()

    assert running is not None
    assert running.timed_out_raw == "true"
    assert running.timed_out is True
    assert running.remaining_job_ids == (JOB_EXECUTING, JOB_UNKNOWN_STATUS)


def test_accepts_str_and_bytes_bodies():
    body = read_fixture("happy_single_job.ndjson")

    assert parse_response(body) == parse_response(body.decode("utf-8"))


def test_wait_response_has_no_header_line():
    response = parsed("wait_cycle_wait.ndjson")

    assert response.header is None
    assert len(response.jobs) == 1
    assert response.footer is not None


def test_expired_wait_slice_returns_only_a_footer():
    response = parsed("wait_cycle_wait_timeout.ndjson")

    assert response.header is None
    assert response.jobs == ()
    assert response.remaining_job_ids == (JOB_SLOW,)
    assert response.timed_out is True


# --------------------------------------------------------------------------------------------
# Status: open enum
# --------------------------------------------------------------------------------------------


def test_unknown_status_is_preserved_and_never_terminal():
    executing, unknown = parsed("unknown_status.ndjson").jobs

    assert executing.status is JobStatus.EXECUTING
    assert executing.status.is_known
    assert not executing.is_terminal

    assert unknown.status.value == "WARP_SPEED_ENGAGED"
    assert unknown.status == "WARP_SPEED_ENGAGED"
    assert not unknown.status.is_known
    assert not unknown.is_terminal
    assert not unknown.is_failure


def test_unknown_statuses_are_singletons():
    assert JobStatus("WARP_SPEED_ENGAGED") is JobStatus("WARP_SPEED_ENGAGED")
    assert JobStatus.from_wire("WARP_SPEED_ENGAGED") is JobStatus("WARP_SPEED_ENGAGED")
    assert JobStatus("WARP_SPEED_ENGAGED") not in list(JobStatus)


def test_only_complete_error_failed_are_terminal():
    assert set(TERMINAL_STATUSES) == {JobStatus.COMPLETE, JobStatus.ERROR, JobStatus.FAILED}
    assert set(FAILURE_STATUSES) == {JobStatus.ERROR, JobStatus.FAILED}
    for status in (JobStatus.ADDED, JobStatus.PLANNING, JobStatus.PLANNING_COMPLETE):
        assert not status.is_terminal
        assert not status.is_failure
    assert JobStatus.COMPLETE.is_terminal
    assert not JobStatus.COMPLETE.is_failure
    assert JobStatus.ERROR.is_failure
    assert JobStatus.FAILED.is_failure


def test_planned_line_carries_schema_but_no_result():
    (job,) = parsed("plan_only.ndjson").jobs

    assert job.status is JobStatus.PLANNED
    assert not job.is_terminal  # the footer, not the status, ends a planOnly call
    assert job.result is None
    assert job.cache_metadata is None
    assert job.summary is not None
    assert "users.state" in job.summary["fields"]


# --------------------------------------------------------------------------------------------
# Error lines
# --------------------------------------------------------------------------------------------


def test_error_lines_carry_type_message_and_kill_reason():
    response = parsed("error_line.ndjson")
    errored, killed = response.jobs

    assert errored.job_id == JOB_ERROR
    assert errored.status is JobStatus.ERROR
    assert errored.error_type == "QUERY"
    assert errored.error_message is not None
    assert "totl_sale_price" in errored.error_message
    assert errored.kill_reason is None
    assert errored.is_failure
    assert errored.failure_reason == errored.error_message

    assert killed.job_id == JOB_KILLED
    assert killed.status is JobStatus.FAILED
    assert killed.error_type == "KILL"
    assert killed.kill_reason == "TIMEOUT"
    assert killed.is_failure


def test_client_result_id_may_be_the_literal_string_null():
    errored, _ = parsed("error_line.ndjson").jobs

    assert errored.client_result_id == "null"
    assert errored.client_result_id_or_none is None


def test_missing_client_result_id_is_none():
    (job,) = parsed("exotic_types.ndjson").jobs

    assert job.client_result_id is None
    assert job.client_result_id_or_none is None


def test_redacted_error_message_is_detected():
    (job,) = parsed("error_line_redacted.ndjson").jobs

    assert job.is_redacted
    assert job.error_message == REDACTED_ERROR_MESSAGE
    assert job.display_sql == ""
    assert job.summary is not None
    assert job.summary["fields"]["users.state"]["sql"] == ""
    assert not job.failed_to_plan


def test_complete_but_failed_to_plan_is_a_failure():
    (job,) = parsed("complete_failed_to_plan.ndjson").jobs

    assert job.status is JobStatus.COMPLETE
    assert not job.has_failure_status
    assert job.failed_to_plan
    assert job.is_failure
    assert job.failure_reason is not None
    assert "Failed to plan query" in job.failure_reason


def test_requery_line_without_result_is_refused():
    (job,) = parsed("requery_without_result.ndjson").jobs

    assert job.status is JobStatus.COMPLETE
    assert job.requery_sql is not None
    assert job.result is None
    assert job.needs_client_materialization
    assert job.is_failure
    assert job.failure_reason is not None
    assert "materialization" in job.failure_reason
    assert job.raw["column_name_mapping"] == {"c0": "users.state"}


def test_failure_reason_falls_back_to_the_status():
    job = JobLine(job_id="j1", status=JobStatus.FAILED)

    assert job.failure_reason == "job j1 finished with status FAILED"


# --------------------------------------------------------------------------------------------
# Tolerated trailing error
# --------------------------------------------------------------------------------------------


def test_unterminated_trailing_error_line_is_tolerated():
    body = read_fixture("trailing_error.ndjson")
    assert not body.endswith(b"\n")

    response = parse_response(body)

    assert response.footer is None
    assert len(response.jobs) == 1
    trailing = response.trailing_error
    assert isinstance(trailing, TrailingError)
    assert trailing.is_json
    assert trailing.message == "Upstream request failed"
    assert trailing.reason == "ECONNRESET"
    assert trailing.detail == "Upstream request failed: ECONNRESET"


def test_truncated_trailing_error_line_is_tolerated():
    response = parsed("trailing_error_truncated.ndjson")

    trailing = response.trailing_error
    assert isinstance(trailing, TrailingError)
    assert not trailing.is_json
    assert trailing.message is None
    assert trailing.text == '{"message":"Upstream request fail'
    assert trailing.detail == trailing.text


def test_malformed_line_mid_stream_raises():
    body = b'{"jobs_submitted":{"j1":null}}\n{"job_id": broken\n{"remaining_job_ids":[],"timed_out":"false"}\n'

    with pytest.raises(TransportError, match="malformed JSON"):
        parse_response(body)


def test_unrecognized_line_shape_raises():
    with pytest.raises(TransportError, match="unrecognized NDJSON line"):
        parse_line('{"surprise":1}')


def test_non_object_line_raises():
    with pytest.raises(TransportError, match="not a JSON object"):
        parse_line("[1, 2, 3]")


def test_blank_body_parses_to_nothing():
    response = parse_response("\n\n")

    assert response.lines == ()
    assert response.header is None
    assert response.remaining_job_ids == ()
    assert response.timed_out is False


# --------------------------------------------------------------------------------------------
# StreamAccumulator
# --------------------------------------------------------------------------------------------


def test_accumulator_merges_run_and_wait_responses():
    accumulator = StreamAccumulator()

    def snapshot() -> tuple[set[str], tuple[str, ...], bool]:
        return set(accumulator.jobs_by_id), accumulator.remaining_job_ids, accumulator.is_done

    accumulator.add_response(read_fixture("wait_cycle_run.ndjson"))
    after_run = snapshot()

    accumulator.add_response(read_fixture("wait_cycle_wait_timeout.ndjson"))
    after_expired_slice = snapshot()

    accumulator.add_response(read_fixture("wait_cycle_wait.ndjson"))
    after_wait = snapshot()

    assert accumulator.submitted_job_ids == (JOB_FAST, JOB_SLOW)
    assert after_run == ({JOB_FAST}, (JOB_SLOW,), False)
    # An expired wait slice carries only a footer: nothing resolved, the loop keeps going.
    assert after_expired_slice == ({JOB_FAST}, (JOB_SLOW,), False)
    assert after_wait == ({JOB_FAST, JOB_SLOW}, (), True)
    assert accumulator.response_count == 3
    assert accumulator.failures == ()
    assert accumulator.job(JOB_SLOW).status is JobStatus.COMPLETE


def test_accumulator_tracks_the_string_timed_out_flag_across_the_cycle():
    accumulator = StreamAccumulator()

    accumulator.add_response(read_fixture("wait_cycle_run.ndjson"))
    while_running = accumulator.timed_out
    accumulator.add_response(read_fixture("wait_cycle_wait.ndjson"))
    when_done = accumulator.timed_out

    assert (while_running, when_done) == (True, False)


def test_accumulator_keeps_the_first_header_only():
    accumulator = StreamAccumulator()
    accumulator.add_response(read_fixture("wait_cycle_run.ndjson"))
    accumulator.add_response(read_fixture("happy_single_job.ndjson"))

    assert accumulator.header is not None
    assert accumulator.submitted_job_ids == (JOB_FAST, JOB_SLOW)


def test_accumulator_keeps_job_lines_exactly_once():
    accumulator = StreamAccumulator()
    accumulator.add_response(read_fixture("wait_cycle_wait.ndjson"))
    accumulator.add_response(read_fixture("wait_cycle_wait.ndjson"))

    assert len(accumulator.jobs) == 1
    assert accumulator.duplicate_job_ids == (JOB_SLOW,)


def test_accumulator_upgrades_a_non_terminal_job_line_to_the_terminal_one():
    """An EXECUTING line in the run response must not shadow the COMPLETE line that follows.

    §2.2 documents ``status`` as an OPEN enum in which ADDED/EXECUTING/PLANNING exist and count
    as non-terminal, and ``pending_job_ids`` keeps polling exactly such a job — so the terminal
    line genuinely arrives in a later slice.  First-wins would throw the answer away and the
    transport would report a job that "carried no result payload".
    """
    accumulator = StreamAccumulator()
    accumulator.add_parsed(
        ParsedResponse(
            header=Header({JOB_SLOW: None}),
            jobs=(JobLine(job_id=JOB_SLOW, status=JobStatus.EXECUTING),),
            footer=Footer(remaining_job_ids=(JOB_SLOW,), timed_out=True),
        )
    )
    assert not accumulator.is_done

    accumulator.add_parsed(
        ParsedResponse(
            jobs=(JobLine(job_id=JOB_SLOW, status=JobStatus.COMPLETE, result="QVJST1cx"),),
            footer=Footer(remaining_job_ids=(), timed_out=False),
        )
    )

    assert accumulator.job(JOB_SLOW).status is JobStatus.COMPLETE
    assert accumulator.job(JOB_SLOW).result == "QVJST1cx"
    assert accumulator.duplicate_job_ids == (JOB_SLOW,)
    assert accumulator.is_done


def test_accumulator_never_downgrades_a_terminal_job_line():
    """The reverse order keeps the answer too — a late intermediate line cannot erase it."""
    accumulator = StreamAccumulator()
    accumulator.add_parsed(
        ParsedResponse(
            jobs=(JobLine(job_id=JOB_SLOW, status=JobStatus.COMPLETE, result="QVJST1cx"),)
        )
    )
    accumulator.add_parsed(
        ParsedResponse(jobs=(JobLine(job_id=JOB_SLOW, status=JobStatus.EXECUTING),))
    )

    assert accumulator.job(JOB_SLOW).status is JobStatus.COMPLETE
    assert accumulator.job(JOB_SLOW).result == "QVJST1cx"


def test_accumulator_footer_is_replaced_each_response():
    accumulator = StreamAccumulator()
    accumulator.add_response(read_fixture("wait_cycle_run.ndjson"))
    assert accumulator.footer is not None
    assert accumulator.footer.remaining_job_ids == (JOB_SLOW,)

    accumulator.add_response(read_fixture("wait_cycle_wait.ndjson"))
    assert accumulator.footer is not None
    assert not accumulator.footer.remaining_job_ids


def test_accumulator_never_waits_on_an_already_terminal_job():
    accumulator = StreamAccumulator()
    accumulator.add_response(read_fixture("wait_cycle_run.ndjson"))
    # A footer that still lists a job we already resolved must not keep the loop alive.
    accumulator.add_response(f'{{"remaining_job_ids":["{JOB_FAST}"],"timed_out":"true"}}\n')

    assert accumulator.remaining_job_ids == (JOB_FAST,)
    assert accumulator.pending_job_ids == ()
    assert accumulator.is_done


def test_accumulator_collects_failures_and_trailing_errors():
    accumulator = StreamAccumulator()
    accumulator.add_response(read_fixture("error_line.ndjson"))
    accumulator.add_response(read_fixture("complete_failed_to_plan.ndjson"))
    accumulator.add_response(read_fixture("trailing_error.ndjson"))

    assert {job.job_id for job in accumulator.failures} == {JOB_ERROR, JOB_KILLED, JOB_UNPLANNED}
    assert accumulator.trailing_error is not None
    assert accumulator.trailing_error.reason == "ECONNRESET"
    assert "pending=" in repr(accumulator)
