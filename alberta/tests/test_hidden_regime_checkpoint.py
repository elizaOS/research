"""Development-only exact checkpoint/resume audits for the hidden-regime runner."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from copy import deepcopy

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.slot_signaling_agent import (
    DURABLE_WRITE_SELECTIVE,
    DURABLE_WRITE_WRITABLE,
    REPLACEMENT_TARGET_EVIDENCE,
    REPLACEMENT_TARGET_LRU,
    SlotSignalingConfig,
)
from alberta_framework.evaluation.hidden_regime_checkpoint import (
    CHECKPOINT_EXECUTION_SCOPE,
    CHECKPOINT_INTEGRITY_SCOPE,
    HIDDEN_REGIME_CHECKPOINT_SCHEMA,
    HIDDEN_REGIME_TRACE_CHUNK_SCHEMA,
    HiddenRegimeCheckpoint,
    HiddenRegimeCheckpointError,
    HiddenRegimeTraceChunk,
    concatenate_hidden_regime_trace_chunks,
    initialize_hidden_regime_checkpoint,
    parse_hidden_regime_checkpoint,
    resume_hidden_regime_to_completion,
    run_hidden_regime_chunk,
    validate_hidden_regime_checkpoint,
    validate_hidden_regime_terminal_checkpoint,
)
from alberta_framework.evaluation.hidden_regime_signaling_development import (
    SELECTIVE_FULL,
    SELECTIVE_LRU,
    SHUFFLED_CHANNEL,
    WRITABLE_EVIDENCE,
    WRITABLE_LRU,
    HiddenRegimeDevelopmentConfig,
    HiddenRegimePrimitiveTrace,
    HiddenRegimeRunResult,
    HiddenRegimeSeedPair,
    run_hidden_regime_condition,
)
from alberta_framework.streams.hidden_regime_signaling import (
    DEFAULT_REGIME_PERMUTATIONS,
    DEFAULT_SEGMENT_REGIMES,
    HIDDEN_REGIME_CALIBRATION_MANIFESTS,
    HIDDEN_REGIME_STRUCTURAL_MANIFESTS,
    HiddenRegimeWorldConfig,
    build_hidden_regime_repeating_phase_drift_world,
)

pytestmark = pytest.mark.development

_EXECUTION_SEED = HiddenRegimeSeedPair(
    namespace="hidden-regime-manual-checkpoint-resume-ci-v1",
    index=33,
    world_seed=1033,
    learner_seed=2033,
)


@pytest.fixture(scope="module")
def lifecycle_config() -> HiddenRegimeDevelopmentConfig:
    """Consumed manual configuration that reaches every lifecycle event in 388 steps."""

    segment_lengths = tuple(4 if regime == 4 else 24 for regime in DEFAULT_SEGMENT_REGIMES)
    return HiddenRegimeDevelopmentConfig(
        world=HiddenRegimeWorldConfig(
            segment_lengths=segment_lengths,
            segment_regimes=DEFAULT_SEGMENT_REGIMES,
            regime_permutations=DEFAULT_REGIME_PERMUTATIONS,
            repeat_schedule=False,
        ),
        learner=SlotSignalingConfig(
            learning_rate=0.5,
            epsilon=0.1,
            relevance_rate=0.5,
            lease_length=4,
            confirmation_steps=1,
            durable_retrieval_threshold=0.1,
            candidate_confirmation_threshold=0.2,
            candidate_confirmation_leases=1,
        ),
        metric_window=4,
    )


@pytest.fixture(scope="module")
def direct_one_shot(
    lifecycle_config: HiddenRegimeDevelopmentConfig,
) -> HiddenRegimeRunResult:
    return run_hidden_regime_condition(
        SELECTIVE_FULL,
        seed_pair=_EXECUTION_SEED,
        config=lifecycle_config,
    )


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resign(payload: dict[str, object]) -> None:
    body = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    payload["integrity_sha256"] = _canonical_digest(body)


def _roundtrip(checkpoint: HiddenRegimeCheckpoint) -> HiddenRegimeCheckpoint:
    return parse_hidden_regime_checkpoint(json.loads(checkpoint.to_json()))


def _assert_trace_equal(
    actual: HiddenRegimePrimitiveTrace,
    expected: HiddenRegimePrimitiveTrace,
) -> None:
    for field in dataclasses.fields(HiddenRegimePrimitiveTrace):
        assert np.array_equal(
            np.asarray(getattr(actual, field.name)),
            np.asarray(getattr(expected, field.name)),
        ), field.name


def _assert_learner_state_equal(actual: object, expected: object) -> None:
    for role_name in ("helper", "beneficiary"):
        actual_role = getattr(actual, role_name)
        expected_role = getattr(expected, role_name)
        for field in dataclasses.fields(actual_role):
            actual_value = getattr(actual_role, field.name)
            expected_value = getattr(expected_role, field.name)
            if field.name == "key":
                actual_value = jr.key_data(actual_value)
                expected_value = jr.key_data(expected_value)
            assert np.array_equal(np.asarray(actual_value), np.asarray(expected_value)), (
                role_name,
                field.name,
            )


def _lifecycle_cut_points(trace: HiddenRegimePrimitiveTrace) -> dict[str, int]:
    """Locate cuts from observed events rather than assuming their step numbers."""

    lease_offset_post = np.asarray(trace.helper_lease_offset_post)
    boundary_steps = np.flatnonzero(np.asarray(trace.helper_lease_boundary))
    regime_starts = np.flatnonzero(np.diff(np.asarray(trace.segment_index))) + 1
    retest_steps = np.flatnonzero(np.asarray(trace.helper_scratch_retest_started))
    commit_steps = np.flatnonzero(np.asarray(trace.helper_committed_slot) >= 0)
    replacement_steps = np.flatnonzero(np.asarray(trace.helper_retired_slot) >= 0)
    assert boundary_steps.size
    assert regime_starts.size
    assert retest_steps.size
    assert commit_steps.size >= 2
    assert replacement_steps.size
    inside_candidates = np.flatnonzero(
        np.logical_and(lease_offset_post > 0, lease_offset_post < 3)
    )
    assert inside_candidates.size
    return {
        "inside_lease": int(inside_candidates[0]) + 1,
        "lease_boundary": int(boundary_steps[0]) + 1,
        "regime_boundary": int(regime_starts[0]),
        "scratch_retest": int(retest_steps[0]) + 1,
        "commit": int(commit_steps[1]) + 1,
        "replacement": int(replacement_steps[0]) + 1,
    }


@pytest.fixture(scope="module")
def direct_lifecycle_chunks(
    lifecycle_config: HiddenRegimeDevelopmentConfig,
    direct_one_shot: HiddenRegimeRunResult,
) -> tuple[tuple[HiddenRegimeTraceChunk, ...], dict[str, int]]:
    cuts = _lifecycle_cut_points(direct_one_shot.trace)
    boundaries = sorted({0, *cuts.values(), lifecycle_config.num_steps})
    assert boundaries[0] == 0
    assert boundaries[-1] == lifecycle_config.num_steps
    checkpoint = initialize_hidden_regime_checkpoint(
        SELECTIVE_FULL,
        seed_pair=_EXECUTION_SEED,
        config=lifecycle_config,
    )
    chunks: list[HiddenRegimeTraceChunk] = []
    for start, end in zip(boundaries, boundaries[1:]):
        assert checkpoint.completed_steps == start
        checkpoint = _roundtrip(checkpoint)
        chunk = run_hidden_regime_chunk(checkpoint, end - start)
        roundtripped_end = _roundtrip(chunk.end_checkpoint)
        chunk = dataclasses.replace(chunk, end_checkpoint=roundtripped_end)
        chunks.append(chunk)
        checkpoint = roundtripped_end
    return tuple(chunks), cuts


def test_checkpoint_payload_is_complete_exact_and_has_no_runtime_oracle_state(
    lifecycle_config: HiddenRegimeDevelopmentConfig,
) -> None:
    checkpoint = initialize_hidden_regime_checkpoint(
        SELECTIVE_FULL,
        seed_pair=_EXECUTION_SEED,
        config=lifecycle_config,
    )
    assert validate_hidden_regime_checkpoint(checkpoint) == ()
    payload = checkpoint.to_dict()
    assert payload["schema_version"] == HIDDEN_REGIME_CHECKPOINT_SCHEMA
    assert payload["development_only"] is True
    assert payload["scientific_promotion_allowed"] is False
    assert payload["artifact_written"] is False
    assert payload["execution_equivalence_scope"] == CHECKPOINT_EXECUTION_SCOPE
    assert payload["integrity_scope"] == CHECKPOINT_INTEGRITY_SCOPE
    assert "not a signature" in CHECKPOINT_INTEGRITY_SCOPE
    assert "does not prove that nonzero learner values arose" in CHECKPOINT_INTEGRITY_SCOPE
    assert "contiguous trace chain or replay" in CHECKPOINT_INTEGRITY_SCOPE
    assert payload["config"] == lifecycle_config.to_dict()
    assert payload["config_sha256"] == _canonical_digest(lifecycle_config.to_dict())
    binding = payload["condition_binding"]
    assert isinstance(binding, dict)
    assert binding == {
        "condition": SELECTIVE_FULL,
        "channel": "direct",
        "channel_code": 0,
        "helper_write_enabled": True,
        "beneficiary_write_enabled": True,
        "effective_durable_write_policy": DURABLE_WRITE_SELECTIVE,
        "effective_replacement_target_policy": REPLACEMENT_TARGET_EVIDENCE,
    }
    world_state = payload["world_state"]
    assert isinstance(world_state, dict)
    assert set(world_state) == {
        "cue_key_data",
        "channel_key_data",
        "cue",
        "step_count",
        "schedule_position",
    }
    learner_state = payload["learner_state"]
    assert isinstance(learner_state, dict)
    for role in ("helper", "beneficiary"):
        role_state = learner_state[role]
        assert isinstance(role_state, dict)
        assert np.asarray(role_state["values_float32_bits"]).shape == (4, 3, 3)
        assert np.asarray(role_state["relevance_mean_float32_bits"]).shape == (4,)
        assert np.asarray(role_state["relevance_mass_float32_bits"]).shape == (4,)
        assert len(role_state["policy_key_data"]) == 2
    runtime_state_text = json.dumps(
        {"world_state": world_state, "learner_state": learner_state},
        sort_keys=True,
    )
    for forbidden in ("oracle", "target", "performance", "summary", "regime_id", "segment"):
        assert forbidden not in runtime_state_text
    assert _roundtrip(checkpoint).to_dict() == checkpoint.to_dict()


def test_json_roundtripped_lifecycle_chunks_equal_one_shot_bit_for_bit(
    lifecycle_config: HiddenRegimeDevelopmentConfig,
    direct_one_shot: HiddenRegimeRunResult,
    direct_lifecycle_chunks: tuple[tuple[HiddenRegimeTraceChunk, ...], dict[str, int]],
) -> None:
    chunks, cuts = direct_lifecycle_chunks
    assert set(cuts) == {
        "inside_lease",
        "lease_boundary",
        "regime_boundary",
        "scratch_retest",
        "commit",
        "replacement",
    }
    assert len(set(cuts.values())) == len(cuts)
    chunked_trace = concatenate_hidden_regime_trace_chunks(chunks)
    _assert_trace_equal(chunked_trace, direct_one_shot.trace)
    assert not np.any(np.asarray(chunked_trace.world_terminated))
    assert np.array_equal(
        np.asarray(chunked_trace.world_discount).view(np.uint32),
        np.full(
            (lifecycle_config.num_steps,),
            np.float32(1.0).view(np.uint32),
            dtype=np.uint32,
        ),
    )
    final_checkpoint = chunks[-1].end_checkpoint
    assert validate_hidden_regime_terminal_checkpoint(final_checkpoint) == ()
    _assert_learner_state_equal(final_checkpoint.learner_state, direct_one_shot.final_state)

    final_world = final_checkpoint.world_state
    trace = direct_one_shot.trace
    assert int(np.asarray(final_world.cue)) == int(np.asarray(trace.world_cue_post)[-1])
    assert int(np.asarray(final_world.step_count)) == lifecycle_config.num_steps
    assert int(np.asarray(final_world.schedule_position)) == lifecycle_config.num_steps - 1
    assert np.array_equal(
        np.asarray(jr.key_data(final_world.cue_key)),
        np.asarray(trace.world_cue_key_data_post)[-1],
    )
    assert np.array_equal(
        np.asarray(jr.key_data(final_world.channel_key)),
        np.asarray(trace.world_channel_key_data_post)[-1],
    )


def test_resume_api_completes_from_a_json_roundtripped_midlife_checkpoint(
    lifecycle_config: HiddenRegimeDevelopmentConfig,
    direct_one_shot: HiddenRegimeRunResult,
) -> None:
    initial = initialize_hidden_regime_checkpoint(
        SELECTIVE_FULL,
        seed_pair=_EXECUTION_SEED,
        config=lifecycle_config,
    )
    prefix = run_hidden_regime_chunk(initial, 37)
    resumed = resume_hidden_regime_to_completion(
        _roundtrip(prefix.end_checkpoint),
        max_chunk_steps=73,
    )
    complete = concatenate_hidden_regime_trace_chunks((prefix, *resumed.chunks))
    _assert_trace_equal(complete, direct_one_shot.trace)
    _assert_learner_state_equal(resumed.final_checkpoint.learner_state, direct_one_shot.final_state)
    assert validate_hidden_regime_terminal_checkpoint(resumed.final_checkpoint) == ()
    with pytest.raises(HiddenRegimeCheckpointError, match="already terminal"):
        resume_hidden_regime_to_completion(
            resumed.final_checkpoint,
            max_chunk_steps=1,
        )


def test_shuffled_channel_checkpoint_chunks_are_bit_exact(
    lifecycle_config: HiddenRegimeDevelopmentConfig,
) -> None:
    one_shot = run_hidden_regime_condition(
        SHUFFLED_CHANNEL,
        seed_pair=_EXECUTION_SEED,
        config=lifecycle_config,
    )
    checkpoint = initialize_hidden_regime_checkpoint(
        SHUFFLED_CHANNEL,
        seed_pair=_EXECUTION_SEED,
        config=lifecycle_config,
    )
    chunks: list[HiddenRegimeTraceChunk] = []
    for end in (5, 137, lifecycle_config.num_steps):
        chunk = run_hidden_regime_chunk(_roundtrip(checkpoint), end - checkpoint.completed_steps)
        chunks.append(chunk)
        checkpoint = chunk.end_checkpoint
    _assert_trace_equal(concatenate_hidden_regime_trace_chunks(chunks), one_shot.trace)
    _assert_learner_state_equal(checkpoint.learner_state, one_shot.final_state)
    assert checkpoint.binding.channel == "shuffled"
    assert checkpoint.binding.channel_code == 2


@pytest.mark.parametrize(
    ("condition", "write_policy", "replacement_policy"),
    (
        (SELECTIVE_FULL, DURABLE_WRITE_SELECTIVE, REPLACEMENT_TARGET_EVIDENCE),
        (WRITABLE_EVIDENCE, DURABLE_WRITE_WRITABLE, REPLACEMENT_TARGET_EVIDENCE),
        (SELECTIVE_LRU, DURABLE_WRITE_SELECTIVE, REPLACEMENT_TARGET_LRU),
        (WRITABLE_LRU, DURABLE_WRITE_WRITABLE, REPLACEMENT_TARGET_LRU),
    ),
)
def test_factorial_condition_binding_survives_roundtrip_and_execution(
    lifecycle_config: HiddenRegimeDevelopmentConfig,
    condition: str,
    write_policy: str,
    replacement_policy: str,
) -> None:
    checkpoint = initialize_hidden_regime_checkpoint(
        condition,  # type: ignore[arg-type]
        seed_pair=_EXECUTION_SEED,
        config=lifecycle_config,
    )
    checkpoint = _roundtrip(checkpoint)
    assert checkpoint.binding.effective_durable_write_policy == write_policy
    assert checkpoint.binding.effective_replacement_target_policy == replacement_policy
    chunk = run_hidden_regime_chunk(checkpoint, 1)
    assert chunk.end_checkpoint.binding == checkpoint.binding
    assert bool(np.asarray(chunk.trace.helper_write_enabled)[0])
    assert bool(np.asarray(chunk.trace.beneficiary_write_enabled)[0])


@pytest.mark.parametrize(
    "mutation",
    (
        "schema",
        "schema_v1",
        "condition",
        "config",
        "config_digest",
        "completed_steps",
        "world_state",
        "learner_state",
        "integrity",
    ),
)
def test_checkpoint_tampering_fails_closed(
    lifecycle_config: HiddenRegimeDevelopmentConfig,
    mutation: str,
) -> None:
    checkpoint = initialize_hidden_regime_checkpoint(
        SELECTIVE_FULL,
        seed_pair=_EXECUTION_SEED,
        config=lifecycle_config,
    )
    payload = deepcopy(checkpoint.to_dict())
    if mutation == "schema":
        payload["schema_version"] = "alberta.hidden-regime-signaling.checkpoint.v999"
    elif mutation == "schema_v1":
        payload["schema_version"] = "alberta.hidden-regime-signaling.checkpoint.v1"
    elif mutation == "condition":
        binding = payload["condition_binding"]
        assert isinstance(binding, dict)
        binding["channel"] = "shuffled"
        _resign(payload)
    elif mutation == "config":
        config = payload["config"]
        assert isinstance(config, dict)
        config["metric_window"] = 5
    elif mutation == "config_digest":
        payload["config_sha256"] = "0" * 64
        _resign(payload)
    elif mutation == "completed_steps":
        payload["completed_steps"] = 1
        _resign(payload)
    elif mutation == "world_state":
        world = payload["world_state"]
        assert isinstance(world, dict)
        world["cue"] = (int(world["cue"]) + 1) % 3
        _resign(payload)
    elif mutation == "learner_state":
        learners = payload["learner_state"]
        assert isinstance(learners, dict)
        helper = learners["helper"]
        assert isinstance(helper, dict)
        values = helper["values_float32_bits"]
        assert isinstance(values, list)
        values[0][0][0] = 0x80000000
        _resign(payload)
    elif mutation == "integrity":
        payload["integrity_sha256"] = "f" * 64
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(mutation)
    with pytest.raises(HiddenRegimeCheckpointError):
        parse_hidden_regime_checkpoint(payload)


def test_chunk_schema_digest_order_gap_overlap_and_endpoint_tampering_fail_closed(
    direct_lifecycle_chunks: tuple[tuple[HiddenRegimeTraceChunk, ...], dict[str, int]],
) -> None:
    chunks, _ = direct_lifecycle_chunks
    assert all(chunk.schema_version == HIDDEN_REGIME_TRACE_CHUNK_SCHEMA for chunk in chunks)
    with pytest.raises(HiddenRegimeCheckpointError, match="schema"):
        concatenate_hidden_regime_trace_chunks(
            (dataclasses.replace(chunks[0], schema_version="trace-chunk-v999"),)
        )
    with pytest.raises(HiddenRegimeCheckpointError, match="schema"):
        concatenate_hidden_regime_trace_chunks(
            (
                dataclasses.replace(
                    chunks[0],
                    schema_version="alberta.hidden-regime-signaling.trace-chunk.v1",
                ),
            )
        )
    with pytest.raises(HiddenRegimeCheckpointError, match="trace_sha256"):
        concatenate_hidden_regime_trace_chunks(
            (dataclasses.replace(chunks[0], trace_sha256="0" * 64),)
        )
    with pytest.raises(HiddenRegimeCheckpointError, match="overlap"):
        concatenate_hidden_regime_trace_chunks((chunks[1], chunks[0]))
    with pytest.raises(HiddenRegimeCheckpointError, match="gap"):
        concatenate_hidden_regime_trace_chunks((chunks[0], chunks[2]))
    with pytest.raises(HiddenRegimeCheckpointError, match="overlap"):
        concatenate_hidden_regime_trace_chunks((chunks[0], chunks[0]))

    changed_reward = dataclasses.replace(
        chunks[0].trace,
        reward=chunks[0].trace.reward.at[0].set(jnp.float32(1.0) - chunks[0].trace.reward[0]),
    )
    with pytest.raises(HiddenRegimeCheckpointError, match="trace_sha256"):
        concatenate_hidden_regime_trace_chunks(
            (dataclasses.replace(chunks[0], trace=changed_reward),)
        )

    changed_terminated = dataclasses.replace(
        chunks[0].trace,
        world_terminated=chunks[0].trace.world_terminated.at[0].set(True),
    )
    resigned_terminated = dataclasses.replace(
        chunks[0],
        trace=changed_terminated,
        trace_sha256=_canonical_digest(changed_terminated.to_dict()),
    )
    with pytest.raises(HiddenRegimeCheckpointError, match="world_terminated"):
        concatenate_hidden_regime_trace_chunks((resigned_terminated,))

    changed_discount = dataclasses.replace(
        chunks[0].trace,
        world_discount=chunks[0].trace.world_discount.at[0].set(jnp.float32(0.5)),
    )
    resigned_discount = dataclasses.replace(
        chunks[0],
        trace=changed_discount,
        trace_sha256=_canonical_digest(changed_discount.to_dict()),
    )
    with pytest.raises(HiddenRegimeCheckpointError, match="world_discount"):
        concatenate_hidden_regime_trace_chunks((resigned_discount,))


def test_chunk_bounds_initial_terminal_and_protected_worlds_fail_closed(
    lifecycle_config: HiddenRegimeDevelopmentConfig,
) -> None:
    initial = initialize_hidden_regime_checkpoint(
        SELECTIVE_FULL,
        seed_pair=_EXECUTION_SEED,
        config=lifecycle_config,
    )
    assert validate_hidden_regime_checkpoint(initial) == ()
    assert validate_hidden_regime_terminal_checkpoint(initial) == (
        "checkpoint is not at the finite terminal execution bound",
    )
    for invalid in (True, 0, -1):
        with pytest.raises(HiddenRegimeCheckpointError, match="positive"):
            run_hidden_regime_chunk(initial, invalid)
    with pytest.raises(HiddenRegimeCheckpointError, match="remaining"):
        run_hidden_regime_chunk(initial, lifecycle_config.num_steps + 1)
    with pytest.raises(HiddenRegimeCheckpointError, match="nonempty"):
        concatenate_hidden_regime_trace_chunks(())

    for manifest in HIDDEN_REGIME_STRUCTURAL_MANIFESTS.values():
        protected_config = HiddenRegimeDevelopmentConfig(
            world=manifest.to_world_config(repeat_schedule=False),
            learner=lifecycle_config.learner,
            metric_window=4,
        )
        with pytest.raises(HiddenRegimeCheckpointError, match="protected"):
            initialize_hidden_regime_checkpoint(
                SELECTIVE_FULL,
                seed_pair=_EXECUTION_SEED,
                config=protected_config,
            )

        for extension in range(1, 16):
            derived = build_hidden_regime_repeating_phase_drift_world(
                manifest,
                final_regime_extension_steps=extension,
            )
            with pytest.raises(ValueError, match="nonrepeating"):
                HiddenRegimeDevelopmentConfig(
                    world=derived.world,
                    learner=lifecycle_config.learner,
                    metric_window=4,
                )
            finite_shape_only = dataclasses.replace(derived.world, repeat_schedule=False)
            protected_phase_config = HiddenRegimeDevelopmentConfig(
                world=finite_shape_only,
                learner=lifecycle_config.learner,
                metric_window=4,
            )
            with pytest.raises(HiddenRegimeCheckpointError, match="protected"):
                initialize_hidden_regime_checkpoint(
                    SELECTIVE_FULL,
                    seed_pair=_EXECUTION_SEED,
                    config=protected_phase_config,
                )

    for manifest in HIDDEN_REGIME_CALIBRATION_MANIFESTS.values():
        calibration_config = HiddenRegimeDevelopmentConfig(
            world=manifest.to_world_config(repeat_schedule=False),
            learner=lifecycle_config.learner,
            metric_window=4,
        )
        with pytest.raises(HiddenRegimeCheckpointError, match="managed calibration"):
            initialize_hidden_regime_checkpoint(
                SELECTIVE_FULL,
                seed_pair=_EXECUTION_SEED,
                config=calibration_config,
            )
