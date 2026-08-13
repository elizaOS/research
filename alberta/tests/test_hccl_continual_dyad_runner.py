"""Bounded orchestration contracts for the production HCCL life runner."""

from __future__ import annotations

import hashlib
import json
from typing import cast

import numpy as np
import pytest

from alberta_framework.core.hccl_continual_dyad_runner import (
    HCCL_CONTINUAL_DYAD_LIFE_TRACE_SCHEMA,
    HCCL_CONTINUAL_DYAD_RUNNER_CONFIG_SCHEMA,
    HCCL_CONTINUAL_DYAD_RUNNER_EVIDENCE_LEVEL,
    HCCLContinualDyadLifeError,
    HCCLContinualDyadLifeTrace,
    HCCLContinualDyadRunner,
    HCCLContinualDyadRunnerConfig,
    _collect_bounded_life,
    _CommittedEvent,
    _expected_regime_ids,
    _profile_for_steps,
    validate_hccl_continual_dyad_life_trace,
)
from alberta_framework.evaluation.hccl_causal_core_endpoints import (
    HCCLCausalCoreCompleteTrace,
    validate_hccl_causal_core_complete_trace,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
    HCCL_CAUSAL_CORE_L2_PROFILE,
    HCCL_CAUSAL_CORE_L3_PROFILE,
    HCCL_CAUSAL_CORE_SMOKE_PROFILE,
    hccl_causal_core_schedule_for_profile,
)

pytestmark = [pytest.mark.unit, pytest.mark.development]

_CANONICAL_STEPS = 8_998
_SMOKE_STEPS = 420
_CORE_L2_STEPS = 71_984
_CORE_L3_STEPS = 1_007_776


def _clock_rows(steps: int) -> tuple[np.ndarray, np.ndarray]:
    pre = np.zeros((steps, 2), dtype=np.uint32)
    post = np.zeros((steps, 2), dtype=np.uint32)
    pre[:, 1] = np.arange(steps, dtype=np.uint32)
    post[:, 1] = np.arange(1, steps + 1, dtype=np.uint32)
    return pre, post


def _valid_trace(profile: str) -> HCCLContinualDyadLifeTrace:
    steps = {
        HCCL_CAUSAL_CORE_CANONICAL_PROFILE: _CANONICAL_STEPS,
        HCCL_CAUSAL_CORE_SMOKE_PROFILE: _SMOKE_STEPS,
        HCCL_CAUSAL_CORE_L2_PROFILE: _CORE_L2_STEPS,
        HCCL_CAUSAL_CORE_L3_PROFILE: _CORE_L3_STEPS,
    }[profile]
    regimes = _expected_regime_ids(profile)
    pre, post = _clock_rows(steps)
    row = np.asarray((0.125, 0.25, 0.5, 0.75), dtype=np.float32)
    matrix = np.broadcast_to(row, (steps, 4)).copy()
    task = matrix[np.arange(steps), regimes].copy()
    net = np.broadcast_to(task[:, None], (steps, 2)).copy()
    return HCCLContinualDyadLifeTrace(
        schedule_profile=profile,
        regime_ids=regimes,
        transaction_committed=np.ones((steps,), dtype=np.bool_),
        pre_step_words=pre,
        post_step_words=post,
        task_scores=task,
        net_rewards=net,
        all_regime_score_matrix=matrix,
    )


class _ToyExecutor:
    """Pure host seam; it models only collector-visible committed records."""

    def __init__(self, profile: str, *, bad_regime_at: int | None = None) -> None:
        self.profile = profile
        self.bad_regime_at = bad_regime_at
        self.calls: list[int] = []

    @property
    def final_state(self) -> object:
        return object()

    def execute_event(self, step_index: int) -> _CommittedEvent:
        self.calls.append(step_index)
        regime = int(_expected_regime_ids(self.profile)[step_index])
        if self.bad_regime_at == step_index:
            regime = (regime + 1) % 4
        scores = np.asarray((0.125, 0.25, 0.5, 0.75), dtype=np.float32)
        task = np.float32(scores[regime])
        return _CommittedEvent(
            regime_id=regime,
            committed=True,
            pre_step_words=np.asarray((0, step_index), dtype=np.uint32),
            post_step_words=np.asarray((0, step_index + 1), dtype=np.uint32),
            task_score=task,
            net_rewards=np.full((2,), task, dtype=np.float32),
            all_regime_scores=scores,
        )


class _MalformedExecutor(_ToyExecutor):
    def execute_event(self, step_index: int) -> _CommittedEvent:
        self.calls.append(step_index)
        return cast(_CommittedEvent, object())


def test_runner_configs_are_exact_profile_specific_and_roundtrip() -> None:
    cases = (
        (
            HCCLContinualDyadRunnerConfig(),
            HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
            _CANONICAL_STEPS,
            True,
            False,
            False,
        ),
        (
            HCCLContinualDyadRunnerConfig.mechanics_smoke(),
            HCCL_CAUSAL_CORE_SMOKE_PROFILE,
            _SMOKE_STEPS,
            False,
            True,
            False,
        ),
        (
            HCCLContinualDyadRunnerConfig.core_l2(),
            HCCL_CAUSAL_CORE_L2_PROFILE,
            _CORE_L2_STEPS,
            False,
            False,
            True,
        ),
        (
            HCCLContinualDyadRunnerConfig.core_l3(),
            HCCL_CAUSAL_CORE_L3_PROFILE,
            _CORE_L3_STEPS,
            False,
            False,
            True,
        ),
    )
    for config, profile, steps, endpoint_compatible, smoke_only, longevity in cases:
        assert config.schedule_profile == profile
        assert config.total_steps == steps
        assert config.canonical_endpoint_compatible is endpoint_compatible
        assert config.mechanics_smoke_only is smoke_only
        assert config.longevity_life is longevity
        payload = json.loads(json.dumps(config.to_config()))
        assert HCCLContinualDyadRunnerConfig.from_config(payload) == config
        assert payload["schema"] == HCCL_CONTINUAL_DYAD_RUNNER_CONFIG_SCHEMA
        assert payload["trace_schema"] == HCCL_CONTINUAL_DYAD_LIFE_TRACE_SCHEMA
        assert payload["evidence_level"] == HCCL_CONTINUAL_DYAD_RUNNER_EVIDENCE_LEVEL
        assert payload["total_steps"] == steps
        assert payload["fresh_factory_initialization_per_run"] is True
        assert payload["complete_fixed_life_only"] is True
        assert payload["partial_life_supported"] is False
        assert payload["bounded_development_life_execution_authorized"] is True
        assert payload["factory_initialization_contract_itself_authorizes_execution"] is False
        assert payload["reset_callback_count"] == 0
        assert payload["boundary_callback_count"] == 0
        assert payload["evaluator_columns_computed_after_atomic_adoption"] is True
        assert payload["evaluator_labels_exposed_to_learner"] is False
        assert payload["counterfactual_score_columns_exposed_to_learner"] is False
        assert payload["canonical_endpoint_compatible"] is endpoint_compatible
        assert payload["mechanics_smoke_only"] is smoke_only
        assert payload["benchmark_execution_authorized"] is False
        assert payload["output_writes_authorized"] is False
        assert payload["artifact_authorized"] is False
        assert payload["evidence_authorized"] is False
        assert payload["promotion_authorized"] is False

    altered = json.loads(json.dumps(HCCLContinualDyadRunnerConfig().to_config()))
    altered["total_steps"] = _CANONICAL_STEPS - 1
    with pytest.raises(ValueError, match="noncanonical"):
        HCCLContinualDyadRunnerConfig.from_config(altered)
    with pytest.raises(TypeError, match="exact dict"):
        HCCLContinualDyadRunnerConfig.from_config(cast(dict[str, object], []))

    crossed = json.loads(json.dumps(HCCLContinualDyadRunnerConfig.core_l2().to_config()))
    crossed["schedule_profile"] = HCCL_CAUSAL_CORE_L3_PROFILE
    with pytest.raises(ValueError, match="noncanonical"):
        HCCLContinualDyadRunnerConfig.from_config(crossed)


def test_existing_canonical_and_smoke_runner_manifests_remain_byte_identical() -> None:
    expected = {
        "canonical": (
            HCCLContinualDyadRunnerConfig(),
            2_526,
            "9f5edf05ab23fe10140ec99f636bc7a3c03315818891de73cb92d8b533488cab",
        ),
        "smoke": (
            HCCLContinualDyadRunnerConfig.mechanics_smoke(),
            2_533,
            "cdc8f591b00d485a03a674c57643bdcdfab81dc48142f65bb7049cdbb50a4d83",
        ),
    }
    for config, expected_length, expected_sha256 in expected.values():
        payload = json.dumps(
            config.to_config(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        assert (len(payload), hashlib.sha256(payload).hexdigest()) == (
            expected_length,
            expected_sha256,
        )


def test_runner_constructs_every_production_factory_without_executing_a_life() -> None:
    configs = (
        HCCLContinualDyadRunnerConfig(),
        HCCLContinualDyadRunnerConfig.mechanics_smoke(),
        HCCLContinualDyadRunnerConfig.core_l2(),
        HCCLContinualDyadRunnerConfig.core_l3(),
    )
    for config in configs:
        runner = HCCLContinualDyadRunner(config)
        assert runner.config is config
        assert runner.to_config() == config.to_config()


def test_executor_profile_resolution_accepts_only_four_exact_lifetimes() -> None:
    assert _profile_for_steps(_SMOKE_STEPS) == HCCL_CAUSAL_CORE_SMOKE_PROFILE
    assert _profile_for_steps(_CANONICAL_STEPS) == HCCL_CAUSAL_CORE_CANONICAL_PROFILE
    assert _profile_for_steps(_CORE_L2_STEPS) == HCCL_CAUSAL_CORE_L2_PROFILE
    assert _profile_for_steps(_CORE_L3_STEPS) == HCCL_CAUSAL_CORE_L3_PROFILE
    for invalid in (True, _CANONICAL_STEPS - 1, float(_CORE_L2_STEPS)):
        with pytest.raises(ValueError, match="exact bounded life"):
            _profile_for_steps(cast(int, invalid))


@pytest.mark.parametrize(
    ("profile", "steps"),
    (
        (HCCL_CAUSAL_CORE_L2_PROFILE, _CORE_L2_STEPS),
        (HCCL_CAUSAL_CORE_L3_PROFILE, _CORE_L3_STEPS),
    ),
)
def test_long_life_regime_vectors_match_every_world_owned_occurrence(
    profile: str,
    steps: int,
) -> None:
    regime_ids = _expected_regime_ids(profile)
    schedule = hccl_causal_core_schedule_for_profile(profile)
    assert regime_ids.shape == (steps,)
    assert regime_ids.dtype == np.dtype(np.int32)
    assert schedule[0] == ("A", 0, 769)
    assert schedule[-1][2] == steps
    for name, start, end in schedule:
        expected_id = ("A", "B", "C", "D").index(name)
        assert int(regime_ids[start]) == expected_id
        assert int(regime_ids[end - 1]) == expected_id
    np.testing.assert_array_equal(
        np.flatnonzero(regime_ids == 3),
        np.arange(2_395, 3_252, dtype=np.int64),
    )


def test_ci_cheap_smoke_orchestration_is_exactly_bounded_and_frozen() -> None:
    config = HCCLContinualDyadRunnerConfig.mechanics_smoke()
    executor = _ToyExecutor(config.schedule_profile)
    trace = _collect_bounded_life(config, executor)

    assert executor.calls == list(range(_SMOKE_STEPS))
    assert trace.schedule_profile == HCCL_CAUSAL_CORE_SMOKE_PROFILE
    assert trace.total_steps == _SMOKE_STEPS
    assert trace.schema == HCCL_CONTINUAL_DYAD_LIFE_TRACE_SCHEMA
    assert bool(np.all(trace.transaction_committed))
    assert trace.reset_callback_count == 0
    assert trace.boundary_callback_count == 0
    assert trace.learner_received_evaluator_regime_ids is False
    assert trace.learner_received_counterfactual_scores is False
    for value in (
        trace.regime_ids,
        trace.transaction_committed,
        trace.pre_step_words,
        trace.post_step_words,
        trace.task_scores,
        trace.net_rewards,
        trace.all_regime_score_matrix,
    ):
        assert value.flags.writeable is False
    with pytest.raises(ValueError, match="not a canonical endpoint"):
        trace.to_canonical_endpoint_trace()


def test_collector_aborts_at_first_schedule_mismatch_and_returns_no_partial_trace() -> None:
    config = HCCLContinualDyadRunnerConfig.mechanics_smoke()
    executor = _ToyExecutor(config.schedule_profile, bad_regime_at=17)
    with pytest.raises(HCCLContinualDyadLifeError) as captured:
        _collect_bounded_life(config, executor)
    assert captured.value.step_index == 17
    assert captured.value.stage == "trace-collection"
    assert executor.calls == list(range(18))
    assert not hasattr(captured.value, "trace")


def test_collector_fails_closed_on_malformed_executor_record() -> None:
    config = HCCLContinualDyadRunnerConfig.mechanics_smoke()
    executor = _MalformedExecutor(config.schedule_profile)
    with pytest.raises(HCCLContinualDyadLifeError) as captured:
        _collect_bounded_life(config, executor)
    assert captured.value.step_index == 0
    assert captured.value.stage == "event-executor"
    assert executor.calls == [0]


def test_event_and_trace_contracts_reject_rollback_clock_and_score_tamper() -> None:
    scores = np.asarray((0.125, 0.25, 0.5, 0.75), dtype=np.float32)
    with pytest.raises(ValueError, match="exact True"):
        _CommittedEvent(
            regime_id=0,
            committed=False,
            pre_step_words=np.asarray((0, 0), dtype=np.uint32),
            post_step_words=np.asarray((0, 0), dtype=np.uint32),
            task_score=np.float32(scores[0]),
            net_rewards=np.full((2,), scores[0], dtype=np.float32),
            all_regime_scores=scores,
        )

    valid = _valid_trace(HCCL_CAUSAL_CORE_SMOKE_PROFILE)
    bad_post = np.array(valid.post_step_words, copy=True)
    bad_post[10, 1] -= np.uint32(1)
    with pytest.raises(ValueError, match="committed clocks"):
        HCCLContinualDyadLifeTrace(
            schedule_profile=valid.schedule_profile,
            regime_ids=valid.regime_ids,
            transaction_committed=valid.transaction_committed,
            pre_step_words=valid.pre_step_words,
            post_step_words=bad_post,
            task_scores=valid.task_scores,
            net_rewards=valid.net_rewards,
            all_regime_score_matrix=valid.all_regime_score_matrix,
        )
    bad_matrix = np.array(valid.all_regime_score_matrix, copy=True)
    bad_matrix[10, valid.regime_ids[10]] += np.float32(0.125)
    with pytest.raises(ValueError, match="selected evaluator columns"):
        HCCLContinualDyadLifeTrace(
            schedule_profile=valid.schedule_profile,
            regime_ids=valid.regime_ids,
            transaction_committed=valid.transaction_committed,
            pre_step_words=valid.pre_step_words,
            post_step_words=valid.post_step_words,
            task_scores=valid.task_scores,
            net_rewards=valid.net_rewards,
            all_regime_score_matrix=bad_matrix,
        )


def test_canonical_life_converts_without_loss_to_endpoint_trace() -> None:
    life = _valid_trace(HCCL_CAUSAL_CORE_CANONICAL_PROFILE)
    endpoint = life.to_canonical_endpoint_trace()
    assert type(endpoint) is HCCLCausalCoreCompleteTrace
    assert validate_hccl_causal_core_complete_trace(endpoint) is endpoint
    np.testing.assert_array_equal(endpoint.regime_ids, life.regime_ids)
    np.testing.assert_array_equal(endpoint.transaction_committed, life.transaction_committed)
    np.testing.assert_array_equal(endpoint.pre_step_words, life.pre_step_words)
    np.testing.assert_array_equal(endpoint.post_step_words, life.post_step_words)
    np.testing.assert_array_equal(endpoint.task_scores, life.task_scores)
    np.testing.assert_array_equal(endpoint.net_rewards, life.net_rewards)
    np.testing.assert_array_equal(
        endpoint.all_regime_score_matrix, life.all_regime_score_matrix
    )
    assert endpoint.reset_callback_count == 0
    assert endpoint.boundary_callback_count == 0
    assert endpoint.learner_received_evaluator_regime_ids is False
    assert endpoint.learner_received_counterfactual_scores is False


def test_core_l2_trace_is_valid_but_cannot_masquerade_as_core_l1_endpoint() -> None:
    life = _valid_trace(HCCL_CAUSAL_CORE_L2_PROFILE)
    assert validate_hccl_continual_dyad_life_trace(life) is life
    assert life.total_steps == _CORE_L2_STEPS
    assert life.longevity_life is True
    assert life.canonical_endpoint_compatible is False
    with pytest.raises(ValueError, match="not a canonical endpoint"):
        life.to_canonical_endpoint_trace()


def test_core_l3_trace_shape_is_strict_without_allocating_or_running_the_life() -> None:
    empty_i32 = np.empty((0,), dtype=np.int32)
    with pytest.raises(ValueError, match=r"regime_ids must have shape \(1007776,\)"):
        HCCLContinualDyadLifeTrace(
            schedule_profile=HCCL_CAUSAL_CORE_L3_PROFILE,
            regime_ids=empty_i32,
            transaction_committed=np.empty((0,), dtype=np.bool_),
            pre_step_words=np.empty((0, 2), dtype=np.uint32),
            post_step_words=np.empty((0, 2), dtype=np.uint32),
            task_scores=np.empty((0,), dtype=np.float32),
            net_rewards=np.empty((0, 2), dtype=np.float32),
            all_regime_score_matrix=np.empty((0, 4), dtype=np.float32),
        )


def test_public_trace_revalidation_detects_post_construction_thaw() -> None:
    trace = _valid_trace(HCCL_CAUSAL_CORE_SMOKE_PROFILE)
    assert validate_hccl_continual_dyad_life_trace(trace) is trace
    trace.task_scores.flags.writeable = True
    with pytest.raises(ValueError, match="read-only and C-contiguous"):
        validate_hccl_continual_dyad_life_trace(trace)
