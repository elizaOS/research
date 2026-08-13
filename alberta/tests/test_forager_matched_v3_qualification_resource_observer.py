"""Tests for the bounded score-blind matched-v3 resource observer."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import os
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_qualification_plan_v2 as plan_v2,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_qualification_resource_observer as observer,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii") + b"\n"


def _ceilings() -> tuple[tuple[str, int], ...]:
    values = {name: 10_000_000_000 for name in plan_v2.RESOURCE_CEILING_FIELDS}
    values["max_environment_interactions"] = 499_712
    values["max_thread_count"] = 16
    values["max_attempt_count"] = 2
    values["max_failure_count"] = 1
    return tuple((name, values[name]) for name in plan_v2.RESOURCE_CEILING_FIELDS)


def _request(
    *,
    target_pid: int = 4321,
    reader_kind: str = "injected_test_reader_v1",
) -> observer.MatchedV3QualificationResourceObservationRequest:
    return observer.MatchedV3QualificationResourceObservationRequest(
        candidate_id="random_policy",
        qualification_case_id="matched-v3-q-random-policy-00",
        qualification_case_manifest_sha256=_sha("case"),
        qualification_plan_sha256=_sha("plan"),
        resource_requirement_body_sha256=_sha("requirement"),
        executor_identity_sha256=_sha("executor"),
        observation_nonce_sha256=_sha("nonce"),
        target_pid=target_pid,
        target_process_start_time_ticks=987_654,
        cgroup_v2_path="/sys/fs/cgroup/alberta-q/case-00",
        expected_cgroup_device=29,
        expected_cgroup_inode=303,
        reader_kind=reader_kind,
        monotonic_clock_id=observer.MONOTONIC_CLOCK_ID,
        declared_ceilings=_ceilings(),
        output_byte_ceilings=(
            ("maximum_result_bytes", 2048),
            ("maximum_stdout_bytes", 1024),
            ("maximum_stderr_bytes", 1024),
        ),
        attempt_ordinal=0,
    )


def _sample(
    phase: str,
    *,
    cpu_usec: int,
    populated: bool,
    member_pids: tuple[int, ...],
    target_proc_present: bool,
    target_pid: int = 4321,
    process_start_time_ticks: int | None = None,
    target_proc_cgroup_exact: bool | None = None,
    cgroup_device: int = 29,
    cgroup_inode: int = 303,
    memory_current_bytes: int | None = 4000,
    memory_peak_bytes: int | None = 5000,
    pids_current: int | None = 1,
    pids_peak: int | None = 3,
    oom_kill_count: int | None = 0,
    unsupported_fields: tuple[str, ...] = (),
) -> observer.ResourceReaderSample:
    if process_start_time_ticks is None and target_proc_present:
        process_start_time_ticks = 987_654
    if target_proc_cgroup_exact is None and target_proc_present:
        target_proc_cgroup_exact = True
    return observer.ResourceReaderSample(
        phase=phase,
        reader_kind="injected_test_reader_v1",
        target_pid=target_pid,
        target_proc_present=target_proc_present,
        target_process_start_time_ticks=process_start_time_ticks,
        target_proc_cgroup_exact=target_proc_cgroup_exact,
        cgroup_v2_path="/sys/fs/cgroup/alberta-q/case-00",
        cgroup_device=cgroup_device,
        cgroup_inode=cgroup_inode,
        cgroup_cpu_usage_usec=cpu_usec,
        memory_current_bytes=memory_current_bytes,
        memory_peak_bytes=memory_peak_bytes,
        pids_current=pids_current,
        pids_peak=pids_peak,
        oom_kill_count=oom_kill_count,
        cgroup_populated=populated,
        cgroup_member_pids=member_pids,
        unsupported_fields=unsupported_fields,
    )


class _FakeReader:
    reader_kind = "injected_test_reader_v1"

    def __init__(self, samples: tuple[observer.ResourceReaderSample, ...]) -> None:
        self._samples = list(samples)
        self.calls: list[tuple[str, int, int, str]] = []

    def sample(
        self,
        *,
        phase: str,
        target_pid: int,
        expected_start_time_ticks: int,
        cgroup_v2_path: str,
    ) -> observer.ResourceReaderSample:
        self.calls.append(
            (phase, target_pid, expected_start_time_ticks, cgroup_v2_path)
        )
        if not self._samples:
            raise AssertionError("unexpected reader call")
        return self._samples.pop(0)


class _Clock:
    def __init__(self, *values: int) -> None:
        self._values = list(values)

    def __call__(self) -> int:
        if not self._values:
            raise AssertionError("unexpected clock call")
        return self._values.pop(0)


def _finish_report(
    request: observer.MatchedV3QualificationResourceObservationRequest,
    *,
    target_pid: int | None = None,
    stdout_size_bytes: int = 100,
    stderr_size_bytes: int = 10,
    result_size_bytes: int = 1000,
    timed_out: bool = False,
    output_counts_complete: bool = True,
) -> observer.MatchedV3QualificationResourceFinishReport:
    return observer.MatchedV3QualificationResourceFinishReport(
        request_sha256=observer.matched_v3_qualification_resource_observation_request_sha256(
            request
        ),
        executor_finish_receipt_sha256=_sha("executor-finish"),
        candidate_id=request.candidate_id,
        qualification_case_id=request.qualification_case_id,
        target_pid=request.target_pid if target_pid is None else target_pid,
        returncode=-9 if timed_out else 0,
        timed_out=timed_out,
        process_group_termination_requested=True,
        process_group_termination_succeeded=True,
        direct_child_waited=True,
        direct_child_reaped=True,
        descendant_cleanup_reported_complete=True,
        cleanup_deadline_expired=False,
        output_counts_complete=output_counts_complete,
        stdout_size_bytes=stdout_size_bytes,
        stderr_size_bytes=stderr_size_bytes,
        result_size_bytes=result_size_bytes,
    )


def _issue_and_finish(
    *,
    request: observer.MatchedV3QualificationResourceObservationRequest | None = None,
    start: observer.ResourceReaderSample | None = None,
    finish: observer.ResourceReaderSample | None = None,
    clock: Callable[[], int] | None = None,
    report: observer.MatchedV3QualificationResourceFinishReport | None = None,
) -> observer.MatchedV3QualificationResourceObservation:
    exact_request = _request() if request is None else request
    exact_start = (
        _sample(
            "start",
            cpu_usec=100,
            populated=True,
            member_pids=(exact_request.target_pid,),
            target_proc_present=True,
            target_pid=exact_request.target_pid,
        )
        if start is None
        else start
    )
    exact_finish = (
        _sample(
            "finish",
            cpu_usec=350,
            populated=False,
            member_pids=(),
            target_proc_present=False,
            target_pid=exact_request.target_pid,
            memory_current_bytes=0,
            pids_current=0,
        )
        if finish is None
        else finish
    )
    reader = _FakeReader((exact_start, exact_finish))
    exact_clock = _Clock(1_000_000, 3_000_000) if clock is None else clock
    capability = observer.issue_matched_v3_qualification_resource_observation_capability(
        request=exact_request,
        reader=reader,
        monotonic_ns=exact_clock,
    )
    return observer.finish_matched_v3_qualification_resource_observation(
        capability=capability,
        finish_report=_finish_report(exact_request) if report is None else report,
    )


def test_descriptor_is_frozen_strict_and_nonauthorizing() -> None:
    descriptor = observer.matched_v3_qualification_resource_observer_descriptor()
    raw = observer.canonical_matched_v3_qualification_resource_observer_descriptor_bytes()

    assert descriptor["status"] == "implemented_uninvoked_non_authorizing"
    assert descriptor["claims"] == {
        "acceptance_evaluated": False,
        "execution_authorized": False,
        "full_28_field_resource_accounting_complete": False,
        "peak_memory_ceiling_enforced": False,
        "qualification_granted": False,
        "runtime_qualified": False,
    }
    assert descriptor["prohibitions"]["performance_payload_input"] is True
    assert descriptor["capabilities"]["network"] is False
    assert descriptor["capabilities"]["process_launcher"] is False
    assert descriptor["production_reader"]["allowlisted_files"] == [
        "cgroup.events",
        "cgroup.procs",
        "cpu.stat",
        "memory.current",
        "memory.events",
        "memory.peak",
        "pids.current",
        "pids.peak",
        "/proc/<exact-pid>/cgroup",
        "/proc/<exact-pid>/stat",
    ]
    assert descriptor["measurement_semantics"]["current_and_peak_distinguished"] is True
    assert (
        descriptor["measurement_semantics"]["continuous_cgroup_membership_attested"]
        is False
    )
    assert descriptor["qualification_binding"]["plan_schema_version"] == (
        plan_v2.QUALIFICATION_PLAN_V2_SCHEMA_VERSION
    )
    assert descriptor["qualification_binding"]["candidate_order"] == list(
        plan_v2.MATCHED_V3_QUALIFICATION_V2_CANDIDATE_IDS
    )
    assert descriptor["qualification_binding"]["resource_ceiling_fields"] == list(
        plan_v2.RESOURCE_CEILING_FIELDS
    )
    assert observer.parse_matched_v3_qualification_resource_observer_descriptor(raw) == descriptor
    assert raw.endswith(b"\n")
    assert observer.matched_v3_qualification_resource_observer_descriptor_sha256() == (
        hashlib.sha256(raw).hexdigest()
    )
    assert observer.PINNED_QUALIFICATION_RESOURCE_OBSERVER_DESCRIPTOR_SHA256 == (
        "e424201576200d05f5da31822cb59a5a61ef06ee29ec267cb20727e8e2e6bfb7"
    )


def test_request_roundtrip_and_exact_resource_order() -> None:
    request = _request()
    raw = observer.canonical_matched_v3_qualification_resource_observation_request_bytes(request)
    assert observer.parse_matched_v3_qualification_resource_observation_request(raw) == request
    assert observer.replay_matched_v3_qualification_resource_observation_request(
        raw,
        expected_request_sha256=hashlib.sha256(raw).hexdigest(),
    ) == request
    assert request.declared_ceilings == _ceilings()


def test_request_accepts_exact_plan_horizon_and_rejects_one_interaction_short() -> None:
    request = _request()
    values = dict(request.declared_ceilings)
    assert values["max_environment_interactions"] == 499_712
    values["max_environment_interactions"] = 499_711
    with pytest.raises(
        observer.ForagerMatchedV3QualificationResourceObserverError,
        match="cannot cover the matched-v3 horizon",
    ):
        dataclasses.replace(
            request,
            declared_ceilings=tuple(
                (name, values[name]) for name in plan_v2.RESOURCE_CEILING_FIELDS
            ),
        )


def test_finish_report_roundtrip() -> None:
    request = _request()
    report = _finish_report(request)
    raw = observer.canonical_matched_v3_qualification_resource_finish_report_bytes(report)
    assert observer.parse_matched_v3_qualification_resource_finish_report(raw) == report


def test_interval_observation_is_exact_but_full_accounting_stays_blocked() -> None:
    observation = _issue_and_finish()
    receipt = observation.receipt()

    assert receipt["interval_counters"] == {
        "cpu_time_ns": 250_000,
        "finish_monotonic_ns": 3_000_000,
        "start_monotonic_ns": 1_000_000,
        "wall_time_ns": 2_000_000,
    }
    assert receipt["resource_ceiling_checks"]["max_cpu_time_ns"]["within_ceiling"] is True
    assert receipt["resource_ceiling_checks"]["max_wall_time_ns"]["within_ceiling"] is True
    assert receipt["resource_ceiling_checks"]["max_peak_rss_bytes"]["supported"] is False
    assert receipt["resource_ceiling_checks"]["max_thread_count"]["supported"] is False
    assert receipt["qualification_projection"] == {
        "all_resource_observations_within_predeclared_integer_ceilings": False,
        "full_28_field_resource_observation_emitted": False,
        "horizon_accounting_exact": False,
        "reward_membership_structural_only": False,
    }
    assert "read_only_memory_peak_freshness_or_reset_not_proven" in receipt["blockers"]
    assert "whole_process_launch_boundary_not_observed" in receipt["blockers"]
    assert observation.receipt_sha256 == hashlib.sha256(
        observation.canonical_receipt_bytes
    ).hexdigest()


def test_raw_memory_and_task_peaks_are_retained_without_false_enforcement() -> None:
    receipt = _issue_and_finish().receipt()
    samples = receipt["samples"]
    assert samples["start"]["memory_peak_bytes"] == 5000
    assert samples["start"]["memory_current_bytes"] == 4000
    assert samples["finish"]["memory_peak_bytes"] == 5000
    assert samples["finish"]["memory_current_bytes"] == 0
    assert samples["finish"]["pids_peak"] == 3
    assert receipt["resource_ceiling_checks"]["max_peak_rss_bytes"]["reason"] == (
        "cgroup_memory_peak_reset_or_freshness_not_proven"
    )


def test_unsupported_optional_cgroup_files_are_explicit_blockers() -> None:
    unsupported = (
        "memory.current",
        "memory.peak",
        "pids.current",
        "pids.peak",
        "memory.events:oom_kill",
    )
    start = _sample(
        "start",
        cpu_usec=100,
        populated=True,
        member_pids=(4321,),
        target_proc_present=True,
        memory_current_bytes=None,
        memory_peak_bytes=None,
        pids_current=None,
        pids_peak=None,
        oom_kill_count=None,
        unsupported_fields=unsupported,
    )
    finish = _sample(
        "finish",
        cpu_usec=200,
        populated=False,
        member_pids=(),
        target_proc_present=False,
        memory_current_bytes=None,
        memory_peak_bytes=None,
        pids_current=None,
        pids_peak=None,
        oom_kill_count=None,
        unsupported_fields=unsupported,
    )
    receipt = _issue_and_finish(start=start, finish=finish).receipt()
    assert receipt["samples"]["start"]["unsupported_fields"] == list(unsupported)
    assert "optional_cgroup_measurements_unsupported" in receipt["blockers"]


@pytest.mark.parametrize(
    ("field", "start_value", "finish_value", "message"),
    [
        ("cpu", 101, 100, "CPU counter rolled back"),
        ("memory_peak", 101, 100, "memory peak counter rolled back"),
        ("pids_peak", 4, 3, "task peak counter rolled back"),
        ("oom", 1, 0, "OOM-kill counter rolled back"),
    ],
)
def test_monotonic_counter_rollback_fails_closed(
    field: str,
    start_value: int,
    finish_value: int,
    message: str,
) -> None:
    kwargs_start: dict[str, Any] = {}
    kwargs_finish: dict[str, Any] = {}
    if field == "memory_peak":
        kwargs_start["memory_peak_bytes"] = start_value
        kwargs_finish["memory_peak_bytes"] = finish_value
    elif field == "pids_peak":
        kwargs_start["pids_peak"] = start_value
        kwargs_finish["pids_peak"] = finish_value
    elif field == "oom":
        kwargs_start["oom_kill_count"] = start_value
        kwargs_finish["oom_kill_count"] = finish_value
    start_cpu = start_value if field == "cpu" else 100
    finish_cpu = finish_value if field == "cpu" else 200
    start = _sample(
        "start",
        cpu_usec=start_cpu,
        populated=True,
        member_pids=(4321,),
        target_proc_present=True,
        **kwargs_start,
    )
    finish = _sample(
        "finish",
        cpu_usec=finish_cpu,
        populated=False,
        member_pids=(),
        target_proc_present=False,
        **kwargs_finish,
    )
    with pytest.raises(observer.ForagerMatchedV3QualificationResourceObserverError, match=message):
        _issue_and_finish(start=start, finish=finish)


def test_clock_rollback_and_counter_overflow_fail_closed() -> None:
    with pytest.raises(
        observer.ForagerMatchedV3QualificationResourceObserverError,
        match="monotonic clock rolled back",
    ):
        _issue_and_finish(clock=_Clock(20, 19))

    finish = _sample(
        "finish",
        cpu_usec=(2**63 - 1) // 1000 + 101,
        populated=False,
        member_pids=(),
        target_proc_present=False,
        memory_current_bytes=0,
        pids_current=0,
    )
    with pytest.raises(
        observer.ForagerMatchedV3QualificationResourceObserverError,
        match="CPU nanosecond conversion overflowed",
    ):
        _issue_and_finish(finish=finish)


@pytest.mark.parametrize(
    ("finish", "message"),
    [
        (
            _sample(
                "finish",
                cpu_usec=200,
                populated=False,
                member_pids=(),
                target_proc_present=False,
                cgroup_inode=304,
            ),
            "cgroup identity drifted",
        ),
        (
            _sample(
                "finish",
                cpu_usec=200,
                populated=True,
                member_pids=(4321,),
                target_proc_present=True,
                process_start_time_ticks=987_655,
            ),
            "target PID was reused",
        ),
        (
            _sample(
                "finish",
                cpu_usec=200,
                populated=False,
                member_pids=(),
                target_proc_present=False,
                target_pid=4322,
            ),
            "sample target PID differs",
        ),
    ],
)
def test_identity_drift_pid_reuse_and_cross_target_pid_fail_closed(
    finish: observer.ResourceReaderSample,
    message: str,
) -> None:
    with pytest.raises(observer.ForagerMatchedV3QualificationResourceObserverError, match=message):
        _issue_and_finish(finish=finish)


def test_cgroup_migration_detected_when_target_exists_at_finish() -> None:
    finish = _sample(
        "finish",
        cpu_usec=200,
        populated=True,
        member_pids=(4321,),
        target_proc_present=True,
        target_proc_cgroup_exact=False,
    )
    with pytest.raises(
        observer.ForagerMatchedV3QualificationResourceObserverError,
        match="target process cgroup membership differs",
    ):
        _issue_and_finish(finish=finish)


@pytest.mark.parametrize(
    ("start", "message"),
    [
        (
            _sample(
                "start",
                cpu_usec=100,
                populated=True,
                member_pids=(4321, 4322),
                target_proc_present=True,
            ),
            "dedicated singleton cgroup",
        ),
        (
            _sample(
                "start",
                cpu_usec=100,
                populated=True,
                member_pids=(4321,),
                target_proc_present=True,
                target_proc_cgroup_exact=False,
            ),
            "target process cgroup membership differs",
        ),
        (
            _sample(
                "start",
                cpu_usec=100,
                populated=True,
                member_pids=(4321,),
                target_proc_present=False,
            ),
            "target process is absent",
        ),
    ],
)
def test_start_boundary_requires_exact_process_and_dedicated_cgroup(
    start: observer.ResourceReaderSample,
    message: str,
) -> None:
    with pytest.raises(observer.ForagerMatchedV3QualificationResourceObserverError, match=message):
        _issue_and_finish(start=start)


def test_incomplete_cleanup_is_recorded_not_upgraded() -> None:
    finish = _sample(
        "finish",
        cpu_usec=200,
        populated=True,
        member_pids=(4444,),
        target_proc_present=False,
    )
    receipt = _issue_and_finish(finish=finish).receipt()
    assert receipt["process_state"]["cgroup_cleanup_observed_complete"] is False
    assert receipt["process_state"]["cleanup_exact"] is False
    assert "cgroup_still_populated_after_finish" in receipt["blockers"]


def test_output_counts_are_bounded_without_reading_output_content() -> None:
    request = _request()
    report = _finish_report(request, stdout_size_bytes=1025)
    receipt = _issue_and_finish(request=request, report=report).receipt()
    assert receipt["output_counts"]["stdout"]["within_ceiling"] is False
    assert receipt["output_counts"]["stdout"]["observed_size_bytes"] == 1025
    assert "output_byte_ceiling_exceeded" in receipt["blockers"]
    finish_signature = inspect.signature(
        observer.finish_matched_v3_qualification_resource_observation
    )
    assert set(finish_signature.parameters) == {"capability", "finish_report"}


def test_timeout_and_incomplete_executor_report_are_fail_closed_facts() -> None:
    request = _request()
    report = _finish_report(request, timed_out=True, output_counts_complete=False)
    receipt = _issue_and_finish(request=request, report=report).receipt()
    assert receipt["process_state"]["timed_out"] is True
    assert receipt["process_state"]["cleanup_exact"] is False
    assert "executor_output_counts_incomplete" in receipt["blockers"]
    assert "execution_timed_out" in receipt["blockers"]


def test_capability_is_single_use_noncopyable_nonserializable_and_process_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    reader = _FakeReader(
        (
            _sample(
                "start",
                cpu_usec=100,
                populated=True,
                member_pids=(4321,),
                target_proc_present=True,
            ),
            _sample(
                "finish",
                cpu_usec=200,
                populated=False,
                member_pids=(),
                target_proc_present=False,
            ),
        )
    )
    capability = observer.issue_matched_v3_qualification_resource_observation_capability(
        request=request,
        reader=reader,
        monotonic_ns=_Clock(1, 2),
    )
    with pytest.raises(TypeError):
        copy.copy(capability)
    with pytest.raises(TypeError):
        copy.deepcopy(capability)
    with pytest.raises(TypeError):
        pickle.dumps(capability)
    real_pid = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: real_pid + 1)
    with pytest.raises(
        observer.ForagerMatchedV3QualificationResourceObserverError,
        match="different observer process",
    ):
        observer.finish_matched_v3_qualification_resource_observation(
            capability=capability,
            finish_report=_finish_report(request),
        )
    monkeypatch.setattr(os, "getpid", lambda: real_pid)
    observation = observer.finish_matched_v3_qualification_resource_observation(
        capability=capability,
        finish_report=_finish_report(request),
    )
    assert observation.receipt()["request"]["target_pid"] == 4321
    with pytest.raises(
        observer.ForagerMatchedV3QualificationResourceObserverError,
        match="unknown or already consumed",
    ):
        observer.finish_matched_v3_qualification_resource_observation(
            capability=capability,
            finish_report=_finish_report(request),
        )


def test_finish_failure_consumes_capability() -> None:
    request = _request()
    reader = _FakeReader(
        (
            _sample(
                "start",
                cpu_usec=100,
                populated=True,
                member_pids=(4321,),
                target_proc_present=True,
            ),
            _sample(
                "finish",
                cpu_usec=99,
                populated=False,
                member_pids=(),
                target_proc_present=False,
            ),
        )
    )
    capability = observer.issue_matched_v3_qualification_resource_observation_capability(
        request=request,
        reader=reader,
        monotonic_ns=_Clock(1, 2),
    )
    with pytest.raises(observer.ForagerMatchedV3QualificationResourceObserverError):
        observer.finish_matched_v3_qualification_resource_observation(
            capability=capability,
            finish_report=_finish_report(request),
        )
    with pytest.raises(
        observer.ForagerMatchedV3QualificationResourceObserverError,
        match="unknown or already consumed",
    ):
        observer.finish_matched_v3_qualification_resource_observation(
            capability=capability,
            finish_report=_finish_report(request),
        )


def test_finish_report_and_reader_identity_mismatches_fail_closed() -> None:
    request = _request()
    bad_report = _finish_report(request, target_pid=9999)
    with pytest.raises(
        observer.ForagerMatchedV3QualificationResourceObserverError,
        match="finish report target PID differs",
    ):
        _issue_and_finish(request=request, report=bad_report)

    bad_reader = _FakeReader(
        (
            _sample(
                "start",
                cpu_usec=100,
                populated=True,
                member_pids=(4321,),
                target_proc_present=True,
            ),
        )
    )
    bad_reader.reader_kind = "different_reader_v1"
    with pytest.raises(
        observer.ForagerMatchedV3QualificationResourceObserverError,
        match="reader kind differs",
    ):
        observer.issue_matched_v3_qualification_resource_observation_capability(
            request=request,
            reader=bad_reader,
            monotonic_ns=_Clock(1),
        )


def test_optional_support_state_cannot_change_between_samples() -> None:
    finish = _sample(
        "finish",
        cpu_usec=200,
        populated=False,
        member_pids=(),
        target_proc_present=False,
        memory_peak_bytes=None,
        unsupported_fields=("memory.peak",),
    )
    with pytest.raises(
        observer.ForagerMatchedV3QualificationResourceObserverError,
        match="optional resource support changed",
    ):
        _issue_and_finish(finish=finish)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_key",
        "bool_integer_alias",
        "float",
        "duplicate_key",
        "noncanonical",
        "digest_mismatch",
    ],
)
def test_strict_request_parser_rejects_aliases_and_noncanonical_bytes(mutation: str) -> None:
    raw = observer.canonical_matched_v3_qualification_resource_observation_request_bytes(
        _request()
    )
    value = json.loads(raw)
    expected_sha = hashlib.sha256(raw).hexdigest()
    if mutation == "unknown_key":
        value["unknown"] = 1
        changed = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    elif mutation == "bool_integer_alias":
        value["target_pid"] = True
        changed = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    elif mutation == "float":
        changed = raw.replace(b'"target_pid":4321', b'"target_pid":4321.0')
    elif mutation == "duplicate_key":
        changed = raw.replace(b"{", b'{"schema_version":"duplicate",', 1)
    elif mutation == "noncanonical":
        changed = b" " + raw
    else:
        changed = raw
        expected_sha = _sha("wrong")
    with pytest.raises(observer.ForagerMatchedV3QualificationResourceObserverError):
        observer.replay_matched_v3_qualification_resource_observation_request(
            changed,
            expected_request_sha256=expected_sha,
        )


def test_request_authority_boolean_integer_alias_is_rejected() -> None:
    value = json.loads(
        observer.canonical_matched_v3_qualification_resource_observation_request_bytes(
            _request()
        )
    )
    value["authority"]["execution_authorized"] = 0
    raw = _canonical(value)
    with pytest.raises(observer.ForagerMatchedV3QualificationResourceObserverError):
        observer.replay_matched_v3_qualification_resource_observation_request(
            raw,
            expected_request_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_dataclasses_reject_aliasing_order_and_impossible_finish_state() -> None:
    request = _request()
    with pytest.raises(observer.ForagerMatchedV3QualificationResourceObserverError):
        observer.MatchedV3QualificationResourceObservationRequest(
            **{
                **{
                    name: getattr(request, name)
                    for name in request.__dataclass_fields__
                },
                "declared_ceilings": tuple(reversed(request.declared_ceilings)),
            }
        )
    with pytest.raises(
        observer.ForagerMatchedV3QualificationResourceObserverError,
        match="successful termination requires a request",
    ):
        dataclasses.replace(
            _finish_report(request),
            process_group_termination_requested=False,
        )


def test_production_reader_rejects_paths_outside_exact_cgroup_v2_root() -> None:
    reader = observer.LinuxCgroupV2ResourceReader()
    with pytest.raises(
        observer.ForagerMatchedV3QualificationResourceObserverError,
        match="beneath /sys/fs/cgroup",
    ):
        reader.sample(
            phase="start",
            target_pid=1,
            expected_start_time_ticks=1,
            cgroup_v2_path="/tmp/not-a-cgroup",
        )
    with pytest.raises(observer.ForagerMatchedV3QualificationResourceObserverError):
        reader.sample(
            phase="start",
            target_pid=1,
            expected_start_time_ticks=1,
            cgroup_v2_path="/sys/fs/cgroup/../escape",
        )


def test_production_reader_and_clock_identity_cannot_be_spoofed_without_live_reads() -> None:
    request = _request(reader_kind=observer.LINUX_CGROUP_V2_READER_KIND)
    disguised = _FakeReader(())
    disguised.reader_kind = observer.LINUX_CGROUP_V2_READER_KIND
    with pytest.raises(
        observer.ForagerMatchedV3QualificationResourceObserverError,
        match="exact production reader type",
    ):
        observer.issue_matched_v3_qualification_resource_observation_capability(
            request=request,
            reader=disguised,
            monotonic_ns=_Clock(1),
        )
    with pytest.raises(
        observer.ForagerMatchedV3QualificationResourceObserverError,
        match="exact time.monotonic_ns",
    ):
        observer.issue_matched_v3_qualification_resource_observation_capability(
            request=request,
            reader=observer.LinuxCgroupV2ResourceReader(),
            monotonic_ns=_Clock(1),
        )


def test_public_api_has_no_default_reader_clock_or_execution_inputs() -> None:
    issue = inspect.signature(
        observer.issue_matched_v3_qualification_resource_observation_capability
    )
    assert set(issue.parameters) == {"request", "reader", "monotonic_ns"}
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in issue.parameters.values()
    )
    assert "reward" not in " ".join(issue.parameters).lower()
    assert "score" not in " ".join(issue.parameters).lower()
    assert "ranking" not in " ".join(issue.parameters).lower()


def test_source_has_no_network_process_launch_or_payload_decoder_surface() -> None:
    source = Path(observer.__file__).read_text(encoding="utf-8")
    assert "from alberta_framework" not in source
    assert "import alberta_framework" not in source
    assert "import socket" not in source
    assert "import urllib" not in source
    assert "import requests" not in source
    assert "import subprocess" not in source
    assert "Popen(" not in source
    assert "reward_trace: bytes" not in source
    assert "reward_trace=" not in source
    assert "cumulative_reward" not in source
    assert "candidate_ranking" not in source


def test_strict_parser_normalizes_excessive_nesting_to_contract_error() -> None:
    raw = (b"[" * 2000) + (b"]" * 2000) + b"\n"
    with pytest.raises(observer.ForagerMatchedV3QualificationResourceObserverError):
        observer.parse_matched_v3_qualification_resource_observation_request(raw)


def test_receipt_parser_rejects_digest_or_canonicalization_drift() -> None:
    observation = _issue_and_finish()
    with pytest.raises(observer.ForagerMatchedV3QualificationResourceObserverError):
        observer.parse_matched_v3_qualification_resource_observation_receipt(
            observation.canonical_receipt_bytes,
            expected_receipt_sha256=_sha("wrong"),
        )
    with pytest.raises(observer.ForagerMatchedV3QualificationResourceObserverError):
        observer.parse_matched_v3_qualification_resource_observation_receipt(
            b" " + observation.canonical_receipt_bytes,
            expected_receipt_sha256=hashlib.sha256(
                b" " + observation.canonical_receipt_bytes
            ).hexdigest(),
        )


@pytest.mark.parametrize(
    "mutation",
    ["finish_candidate", "finish_pid", "sample_cgroup", "sample_phase", "sample_pid_reuse"],
)
def test_receipt_replay_revalidates_all_cross_record_identity_links(mutation: str) -> None:
    value = json.loads(_issue_and_finish().canonical_receipt_bytes)
    if mutation == "finish_candidate":
        value["finish_report"]["candidate_id"] = "search_oracle"
    elif mutation == "finish_pid":
        value["finish_report"]["target_pid"] = 9999
    elif mutation == "sample_cgroup":
        value["samples"]["finish"]["cgroup_inode"] = 9999
    elif mutation == "sample_phase":
        value["samples"]["finish"]["phase"] = "start"
    else:
        value["samples"]["finish"]["target_proc_present"] = True
        value["samples"]["finish"]["target_process_start_time_ticks"] = 987_655
        value["samples"]["finish"]["target_proc_cgroup_exact"] = True
    raw = _canonical(value)
    with pytest.raises(observer.ForagerMatchedV3QualificationResourceObserverError):
        observer.parse_matched_v3_qualification_resource_observation_receipt(
            raw,
            expected_receipt_sha256=hashlib.sha256(raw).hexdigest(),
        )
