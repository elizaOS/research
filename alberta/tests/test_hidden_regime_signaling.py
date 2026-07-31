"""Unit tests for the continuing ternary hidden-regime signaling world."""

import dataclasses
import inspect

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.streams.hidden_regime_signaling import (
    CALIBRATION_ONLY_PARTITION,
    CALIBRATION_ONLY_PERFORMANCE_STATUS,
    CONSTANT_ONE_TERNARY_CHANNEL,
    CONSTANT_TWO_TERNARY_CHANNEL,
    CONSTANT_ZERO_TERNARY_CHANNEL,
    DEFAULT_REGIME_PERMUTATIONS,
    DEFAULT_SEGMENT_LENGTHS,
    DEFAULT_SEGMENT_REGIMES,
    DIRECT_TERNARY_CHANNEL,
    HIDDEN_REGIME_CALIBRATION_A_V1,
    HIDDEN_REGIME_CALIBRATION_B_V1,
    HIDDEN_REGIME_CALIBRATION_C_V1,
    HIDDEN_REGIME_CALIBRATION_MANIFESTS,
    HIDDEN_REGIME_MANIFEST_USE_LEDGER,
    HIDDEN_REGIME_STRUCTURAL_A_V1,
    HIDDEN_REGIME_STRUCTURAL_B_V1,
    HIDDEN_REGIME_STRUCTURAL_C_V1,
    HIDDEN_REGIME_STRUCTURAL_MANIFESTS,
    PROTECTED_CANDIDATE_LEARNER_OUTCOMES_EXECUTED,
    PROTECTED_CANDIDATE_PARTITION,
    PROTECTED_CANDIDATE_PERFORMANCE_STATUS,
    SHUFFLED_TERNARY_CHANNEL,
    STRUCTURAL_GENERALIZATION_BASELINE_SEGMENT_LENGTHS,
    HiddenRegimeScheduleManifest,
    HiddenRegimeSignalingWorld,
    HiddenRegimeWorldConfig,
    HiddenRegimeWorldKeys,
    build_hidden_regime_repeating_phase_drift_world,
    hidden_regime_calibration_manifest,
    hidden_regime_calibration_world_config,
    hidden_regime_manifest_use,
    hidden_regime_structural_manifest,
    hidden_regime_world_config_for_manifest,
    hidden_regime_world_keys,
)

pytestmark = pytest.mark.unit


def _tiny_config(*, repeat: bool = False) -> HiddenRegimeWorldConfig:
    return HiddenRegimeWorldConfig(
        segment_lengths=(2, 1, 2),
        segment_regimes=(0, 1, 0),
        regime_permutations=((0, 1, 2), (1, 2, 0)),
        repeat_schedule=repeat,
    )


_EXPECTED_STRUCTURAL_MANIFESTS = {
    "hidden-regime-structural-a-v1": {
        "segment_regimes": (0, 1, 2, 0, 1, 2, 0, 1, 3, 0, 3, 1, 0, 4, 1, 3, 0),
        "segment_lengths": (
            1026,
            1155,
            1034,
            905,
            1141,
            886,
            1032,
            1031,
            1028,
            1149,
            1033,
            1158,
            890,
            16,
            891,
            1011,
            1142,
        ),
        "length_perturbations_by_regime": (
            (2, 9, 8, -3, -6, -10),
            (3, -11, 7, 6, -5),
            (10, -10),
            (4, 9, -13),
            (0,),
        ),
        "cue_symbol_relabeling": (1, 2, 0),
        "action_symbol_relabeling": (2, 0, 1),
        "change_point_residues": (2, 5, 15, 8, 13, 3, 11, 2, 6, 3, 12, 2, 12, 12, 7, 10),
    },
    "hidden-regime-structural-b-v1": {
        "segment_regimes": (1, 0, 2, 1, 0, 2, 1, 0, 3, 1, 3, 0, 4, 3, 0, 1, 0),
        "segment_lengths": (
            1153,
            1032,
            1029,
            1145,
            888,
            891,
            1033,
            1026,
            1020,
            1163,
            1023,
            1155,
            16,
            1029,
            896,
            882,
            1147,
        ),
        "length_perturbations_by_regime": (
            (8, -8, 2, 3, 0, -5),
            (1, -7, 9, 11, -14),
            (5, -5),
            (-4, -1, 5),
            (0,),
        ),
        "cue_symbol_relabeling": (2, 0, 1),
        "action_symbol_relabeling": (1, 0, 2),
        "change_point_residues": (1, 9, 14, 7, 15, 10, 3, 5, 1, 12, 11, 14, 14, 3, 3, 5),
    },
    "hidden-regime-structural-c-v1": {
        "segment_regimes": (0, 2, 1, 0, 2, 1, 0, 3, 1, 0, 3, 1, 4, 0, 1, 3, 0),
        "segment_lengths": (
            1028,
            1029,
            1149,
            889,
            891,
            1144,
            1025,
            1034,
            1031,
            1156,
            1022,
            1144,
            16,
            899,
            908,
            1016,
            1147,
        ),
        "length_perturbations_by_regime": (
            (4, -7, 1, 4, 3, -5),
            (-3, -8, 7, -8, 12),
            (5, -5),
            (10, -2, -8),
            (0,),
        ),
        "cue_symbol_relabeling": (0, 2, 1),
        "action_symbol_relabeling": (2, 1, 0),
        "change_point_residues": (4, 9, 6, 15, 10, 2, 3, 13, 4, 8, 6, 14, 14, 1, 13, 5),
    },
}

_EXPECTED_CALIBRATION_MANIFESTS = {
    "hidden-regime-calibration-a-v1": {
        "segment_regimes": (2, 0, 1, 2, 1, 0, 0, 1, 3, 0, 1, 3, 0, 4, 3, 1, 0),
        "segment_lengths": (
            1021,
            1019,
            1166,
            899,
            1139,
            881,
            1018,
            1021,
            1028,
            1161,
            1146,
            1035,
            902,
            16,
            1009,
            904,
            1163,
        ),
        "length_perturbations_by_regime": (
            (-5, -15, -6, 9, 6, 11),
            (14, -13, -3, -6, 8),
            (-3, 3),
            (4, 11, -15),
            (0,),
        ),
        "cue_symbol_relabeling": (0, 1, 2),
        "action_symbol_relabeling": (1, 2, 0),
        "change_point_residues": (13, 8, 6, 9, 12, 13, 7, 4, 8, 1, 11, 6, 12, 12, 13, 5),
    },
    "hidden-regime-calibration-b-v1": {
        "segment_regimes": (1, 2, 0, 0, 1, 2, 1, 0, 3, 1, 0, 3, 4, 0, 1, 3, 0),
        "segment_lengths": (
            1163,
            1030,
            1033,
            898,
            1142,
            890,
            1026,
            1018,
            1036,
            1153,
            1137,
            1021,
            16,
            908,
            892,
            1015,
            1150,
        ),
        "length_perturbations_by_regime": (
            (9, 2, -6, -15, 12, -2),
            (11, -10, 2, 1, -4),
            (6, -6),
            (12, -3, -9),
            (0,),
        ),
        "cue_symbol_relabeling": (1, 0, 2),
        "action_symbol_relabeling": (0, 1, 2),
        "change_point_residues": (11, 1, 10, 12, 2, 12, 14, 8, 4, 5, 6, 3, 3, 15, 11, 2),
    },
    "hidden-regime-calibration-c-v1": {
        "segment_regimes": (2, 1, 0, 1, 2, 0, 1, 0, 3, 0, 3, 1, 0, 4, 0, 3, 1),
        "segment_lengths": (
            1023,
            1155,
            1009,
            1167,
            897,
            898,
            1027,
            1030,
            1030,
            1151,
            1027,
            1145,
            908,
            16,
            1148,
            1015,
            882,
        ),
        "length_perturbations_by_regime": (
            (-15, 2, 6, -1, 12, -4),
            (3, 15, 3, -7, -14),
            (-1, 1),
            (6, 3, -9),
            (0,),
        ),
        "cue_symbol_relabeling": (2, 1, 0),
        "action_symbol_relabeling": (2, 0, 1),
        "change_point_residues": (15, 2, 3, 2, 3, 5, 8, 14, 4, 3, 6, 15, 11, 11, 7, 14),
    },
}


def _lengths_for_order_and_perturbations(
    order: tuple[int, ...],
    perturbations: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    baseline_by_regime: list[list[int]] = [[] for _ in DEFAULT_REGIME_PERMUTATIONS]
    for regime, length in zip(
        DEFAULT_SEGMENT_REGIMES,
        STRUCTURAL_GENERALIZATION_BASELINE_SEGMENT_LENGTHS,
        strict=True,
    ):
        baseline_by_regime[regime].append(length)
    cursors = [0] * len(DEFAULT_REGIME_PERMUTATIONS)
    lengths: list[int] = []
    for regime in order:
        ordinal = cursors[regime]
        lengths.append(baseline_by_regime[regime][ordinal] + perturbations[regime][ordinal])
        cursors[regime] += 1
    return tuple(lengths)


def test_config_is_strict_finite_and_explicitly_development_only() -> None:
    invalid = (
        {"segment_lengths": ()},
        {"segment_lengths": [1]},
        {"segment_lengths": (True,)},
        {"segment_lengths": (0,)},
        {"segment_lengths": (1,), "segment_regimes": (0, 1)},
        {"segment_lengths": (1,), "segment_regimes": (5,)},
        {"regime_permutations": ()},
        {"regime_permutations": ((0, 1, 1),)},
        {"regime_permutations": ((0, 1),)},
        {
            "regime_permutations": (
                (False, 1, 2),
                *DEFAULT_REGIME_PERMUTATIONS[1:],
            )
        },
        {
            "regime_permutations": (
                (0.0, 1, 2),
                *DEFAULT_REGIME_PERMUTATIONS[1:],
            )
        },
        {"repeat_schedule": 1},
    )
    for kwargs in invalid:
        with pytest.raises(ValueError):
            HiddenRegimeWorldConfig(**kwargs)  # type: ignore[arg-type]
    payload = HiddenRegimeWorldConfig().to_dict()
    assert payload["development_only"] is True
    assert payload["scientific_promotion_allowed"] is False
    assert payload["total_schedule_steps"] == sum(payload["segment_lengths"])


def test_ordinary_observation_and_world_api_cannot_reveal_oracle_facts() -> None:
    world = HiddenRegimeSignalingWorld(_tiny_config())
    state = world.init(hidden_regime_world_keys(jr.key(4)))
    observation_fields = {field.name for field in dataclasses.fields(world.observe(state))}
    assert observation_fields == {"helper_cue"}
    for method in (world.observe, world.deliver, world.step, world.step_with_delivery):
        parameters = inspect.signature(method).parameters
        assert "regime" not in parameters
        assert "target" not in parameters
        assert "segment" not in parameters
    transition, _ = world.step(state, jnp.int32(0), jnp.int32(0))
    assert {field.name for field in dataclasses.fields(transition.oracle)} == {
        "step_count",
        "segment_index",
        "segment_step",
        "regime_id",
        "target",
    }
    assert "oracle" not in observation_fields


def test_evaluator_schedule_changes_hidden_permutation_without_reset() -> None:
    config = _tiny_config()
    world = HiddenRegimeSignalingWorld(config)
    state = world.init(hidden_regime_world_keys(jr.key(9)))
    segments: list[int] = []
    segment_steps: list[int] = []
    regimes: list[int] = []
    for _ in range(8):
        transition, state = world.step(state, jnp.int32(0), jnp.int32(0))
        oracle = transition.oracle
        segments.append(int(oracle.segment_index))
        segment_steps.append(int(oracle.segment_step))
        regimes.append(int(oracle.regime_id))
        cue = int(transition.observation.helper_cue)
        permutation = config.regime_permutations[int(oracle.regime_id)]
        assert int(oracle.target) == permutation[cue]
        assert not bool(transition.terminated)
        assert float(transition.discount) == 1.0
    assert segments == [0, 0, 1, 2, 2, 2, 2, 2]
    assert segment_steps == [0, 1, 0, 0, 1, 1, 1, 1]
    assert regimes == [0, 0, 1, 0, 0, 0, 0, 0]
    assert int(state.step_count) == 8
    assert int(state.schedule_position) == 4


def test_repeating_schedule_recurs_without_resetting_global_step() -> None:
    world = HiddenRegimeSignalingWorld(_tiny_config(repeat=True))
    state = world.init(hidden_regime_world_keys(jr.key(3)))
    segments: list[int] = []
    regimes: list[int] = []
    for _ in range(12):
        transition, state = world.step(state, jnp.int32(0), jnp.int32(0))
        segments.append(int(transition.oracle.segment_index))
        regimes.append(int(transition.oracle.regime_id))
    assert segments == [0, 0, 1, 2, 2, 0, 0, 1, 2, 2, 0, 0]
    assert regimes == [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    assert int(state.step_count) == 12
    assert int(state.schedule_position) == 2


def test_global_step_saturates_while_repeating_schedule_cursor_keeps_advancing() -> None:
    world = HiddenRegimeSignalingWorld(_tiny_config(repeat=True))
    initial = world.init(hidden_regime_world_keys(jr.key(31)))
    maximum = np.iinfo(np.int32).max
    state = dataclasses.replace(
        initial,
        step_count=jnp.asarray(maximum - 1, dtype=jnp.int32),
        schedule_position=jnp.asarray(4, dtype=jnp.int32),
    )
    final_segment, state = world.step(state, jnp.int32(0), jnp.int32(0))
    assert int(final_segment.oracle.step_count) == maximum - 1
    assert int(final_segment.oracle.segment_index) == 2
    assert int(state.step_count) == maximum
    assert int(state.schedule_position) == 0

    first_segment, state = world.step(state, jnp.int32(0), jnp.int32(0))
    assert int(first_segment.oracle.step_count) == maximum
    assert int(first_segment.oracle.segment_index) == 0
    assert int(state.step_count) == maximum
    assert int(state.schedule_position) == 1


def test_global_step_and_held_schedule_both_saturate_without_wrapping() -> None:
    world = HiddenRegimeSignalingWorld(_tiny_config())
    initial = world.init(hidden_regime_world_keys(jr.key(32)))
    maximum = np.iinfo(np.int32).max
    state = dataclasses.replace(
        initial,
        step_count=jnp.asarray(maximum, dtype=jnp.int32),
        schedule_position=jnp.asarray(4, dtype=jnp.int32),
    )
    transition, next_state = jax.jit(world.step)(
        state,
        jnp.int32(0),
        jnp.int32(0),
    )
    assert int(transition.oracle.step_count) == maximum
    assert int(transition.oracle.segment_index) == 2
    assert int(next_state.step_count) == maximum
    assert int(next_state.schedule_position) == 4


def test_reward_uses_hidden_target_but_neither_message_nor_delivery_directly() -> None:
    world = HiddenRegimeSignalingWorld(_tiny_config())
    state = world.init(hidden_regime_world_keys(jr.key(15)))
    target = world.target_of(state.step_count, state.cue)
    correct, _ = world.step_with_delivery(
        state,
        jnp.int32(2),
        jnp.int32(1),
        target,
    )
    wrong, _ = world.step_with_delivery(
        state,
        jnp.int32(0),
        jnp.int32(2),
        (target + 1) % 3,
    )
    assert float(correct.reward) == 1.0
    assert float(wrong.reward) == 0.0
    assert int(correct.oracle.target) == int(wrong.oracle.target)


def test_ternary_channel_interventions_have_exact_causal_semantics() -> None:
    world = HiddenRegimeSignalingWorld(_tiny_config())
    state = world.init(hidden_regime_world_keys(jr.key(11)))
    for message in range(3):
        assert int(world.deliver(state, jnp.int32(message), DIRECT_TERNARY_CHANNEL)) == message
        assert int(world.deliver(state, jnp.int32(message), CONSTANT_ZERO_TERNARY_CHANNEL)) == 0
        assert int(world.deliver(state, jnp.int32(message), CONSTANT_ONE_TERNARY_CHANNEL)) == 1
        assert int(world.deliver(state, jnp.int32(message), CONSTANT_TWO_TERNARY_CHANNEL)) == 2
    draws = [
        int(world.deliver(state, jnp.int32(message), SHUFFLED_TERNARY_CHANNEL))
        for message in range(3)
    ]
    assert draws[0] == draws[1] == draws[2]
    with pytest.raises(ValueError, match="unknown hidden-regime channel"):
        world.deliver(state, jnp.int32(0), "unknown")  # type: ignore[arg-type]


def test_named_channel_stream_is_deterministic_and_independent_of_cues() -> None:
    world = HiddenRegimeSignalingWorld(_tiny_config(repeat=True))
    common_channel = jr.key(919)
    state_a = world.init(HiddenRegimeWorldKeys(cue=jr.key(1), channel=common_channel))
    state_b = world.init(HiddenRegimeWorldKeys(cue=jr.key(2), channel=common_channel))
    cues_a: list[int] = []
    cues_b: list[int] = []
    deliveries: list[int] = []
    for _ in range(48):
        delivered_a = world.deliver(state_a, jnp.int32(0), SHUFFLED_TERNARY_CHANNEL)
        delivered_b = world.deliver(state_b, jnp.int32(2), SHUFFLED_TERNARY_CHANNEL)
        np.testing.assert_array_equal(delivered_a, delivered_b)
        cues_a.append(int(state_a.cue))
        cues_b.append(int(state_b.cue))
        deliveries.append(int(delivered_a))
        _, state_a = world.step(
            state_a,
            jnp.int32(0),
            jnp.int32(0),
            SHUFFLED_TERNARY_CHANNEL,
        )
        _, state_b = world.step(
            state_b,
            jnp.int32(2),
            jnp.int32(0),
            SHUFFLED_TERNARY_CHANNEL,
        )
    assert cues_a != cues_b
    assert set(deliveries) == {0, 1, 2}


def test_channel_choice_and_message_never_perturb_future_cues() -> None:
    world = HiddenRegimeSignalingWorld(_tiny_config(repeat=True))
    keys = hidden_regime_world_keys(jr.key(27))
    direct = world.init(keys)
    shuffled = world.init(keys)
    for _ in range(32):
        np.testing.assert_array_equal(direct.cue, shuffled.cue)
        _, direct = world.step(
            direct,
            jnp.int32(0),
            jnp.int32(0),
            DIRECT_TERNARY_CHANNEL,
        )
        _, shuffled = world.step(
            shuffled,
            jnp.int32(2),
            jnp.int32(0),
            SHUFFLED_TERNARY_CHANNEL,
        )


def test_world_is_jittable_scannable_fixed_shape_and_finite() -> None:
    world = HiddenRegimeSignalingWorld(_tiny_config(repeat=True))
    state = world.init(hidden_regime_world_keys(jr.key(42)))

    @jax.jit
    def run(initial_state):
        def body(carry, action):
            transition, next_state = world.step(carry, action, action)
            output = jnp.stack(
                (
                    transition.reward,
                    transition.discount,
                    transition.oracle.segment_index.astype(jnp.float32),
                    transition.oracle.regime_id.astype(jnp.float32),
                    transition.oracle.target.astype(jnp.float32),
                )
            )
            return next_state, output

        actions = jnp.arange(40, dtype=jnp.int32) % 3
        return jax.lax.scan(body, initial_state, actions)

    final_state, outputs = run(state)
    assert outputs.shape == (40, 5)
    assert bool(jnp.all(jnp.isfinite(outputs)))
    assert final_state.cue.shape == ()
    assert final_state.step_count.shape == ()
    assert final_state.schedule_position.shape == ()
    assert jr.key_data(final_state.cue_key).shape == (2,)
    assert jr.key_data(final_state.channel_key).shape == (2,)


def test_named_key_derivation_is_stable_and_physically_separate() -> None:
    keys_a = hidden_regime_world_keys(jr.key(8))
    keys_b = hidden_regime_world_keys(jr.key(8))
    np.testing.assert_array_equal(jr.key_data(keys_a.cue), jr.key_data(keys_b.cue))
    np.testing.assert_array_equal(jr.key_data(keys_a.channel), jr.key_data(keys_b.channel))
    assert not np.array_equal(jr.key_data(keys_a.cue), jr.key_data(keys_a.channel))


def test_structural_manifests_have_frozen_exact_definitions() -> None:
    expected_names = tuple(_EXPECTED_STRUCTURAL_MANIFESTS)
    assert tuple(HIDDEN_REGIME_STRUCTURAL_MANIFESTS) == expected_names
    assert HIDDEN_REGIME_STRUCTURAL_MANIFESTS[expected_names[0]] is HIDDEN_REGIME_STRUCTURAL_A_V1
    assert HIDDEN_REGIME_STRUCTURAL_MANIFESTS[expected_names[1]] is HIDDEN_REGIME_STRUCTURAL_B_V1
    assert HIDDEN_REGIME_STRUCTURAL_MANIFESTS[expected_names[2]] is HIDDEN_REGIME_STRUCTURAL_C_V1

    for name, expected in _EXPECTED_STRUCTURAL_MANIFESTS.items():
        manifest = hidden_regime_structural_manifest(name)
        assert manifest.name == name
        for field, value in expected.items():
            assert getattr(manifest, field) == value
        payload = manifest.to_dict()
        assert payload["schema"] == "hidden-regime-structural-manifest-v1"
        assert payload["name"] == name
        assert payload["use_partition"] == PROTECTED_CANDIDATE_PARTITION
        assert payload["candidate_only"] is True
        assert payload["protected_candidate"] is True
        assert payload["calibration_use_allowed"] is False
        assert payload["learner_outcome_status"] == PROTECTED_CANDIDATE_PERFORMANCE_STATUS
        assert payload["learner_outcomes_executed"] is False
        assert payload["scientific_promotion_allowed"] is False
        assert payload["protected_seed_namespace"] is None


def test_calibration_manifests_have_separate_frozen_exact_definitions() -> None:
    expected_names = tuple(_EXPECTED_CALIBRATION_MANIFESTS)
    assert tuple(HIDDEN_REGIME_CALIBRATION_MANIFESTS) == expected_names
    assert HIDDEN_REGIME_CALIBRATION_MANIFESTS[expected_names[0]] is (
        HIDDEN_REGIME_CALIBRATION_A_V1
    )
    assert HIDDEN_REGIME_CALIBRATION_MANIFESTS[expected_names[1]] is (
        HIDDEN_REGIME_CALIBRATION_B_V1
    )
    assert HIDDEN_REGIME_CALIBRATION_MANIFESTS[expected_names[2]] is (
        HIDDEN_REGIME_CALIBRATION_C_V1
    )

    for name, expected in _EXPECTED_CALIBRATION_MANIFESTS.items():
        manifest = hidden_regime_calibration_manifest(name)
        assert manifest.name == name
        assert manifest.use_partition == CALIBRATION_ONLY_PARTITION
        for field, value in expected.items():
            assert getattr(manifest, field) == value
        payload = manifest.to_dict()
        assert payload["schema"] == "hidden-regime-calibration-manifest-v1"
        assert payload["use_partition"] == CALIBRATION_ONLY_PARTITION
        assert payload["candidate_only"] is False
        assert payload["protected_candidate"] is False
        assert payload["calibration_use_allowed"] is True
        assert payload["learner_outcome_status"] == CALIBRATION_ONLY_PERFORMANCE_STATUS
        assert payload["learner_outcomes_executed"] is None
        assert payload["scientific_promotion_allowed"] is False
        assert payload["protected_seed_namespace"] is None


def test_manifest_partitions_have_disjoint_registries_resolvers_and_immutable_use_ledger() -> None:
    structural_names = tuple(HIDDEN_REGIME_STRUCTURAL_MANIFESTS)
    calibration_names = tuple(HIDDEN_REGIME_CALIBRATION_MANIFESTS)
    assert not (set(structural_names) & set(calibration_names))
    assert tuple(HIDDEN_REGIME_MANIFEST_USE_LEDGER) == structural_names + calibration_names
    assert PROTECTED_CANDIDATE_LEARNER_OUTCOMES_EXECUTED is False

    for name in structural_names:
        manifest = hidden_regime_structural_manifest(name)
        use = hidden_regime_manifest_use(name)
        assert manifest.use_partition == PROTECTED_CANDIDATE_PARTITION
        assert use.use_partition == PROTECTED_CANDIDATE_PARTITION
        assert use.learner_outcome_status == PROTECTED_CANDIDATE_PERFORMANCE_STATUS
        assert use.learner_outcomes_executed is False
        assert use.calibration_use_allowed is False
        assert use.protected_evaluation_candidate is True
        assert use.scientific_promotion_allowed is False
        with pytest.raises(ValueError, match="calibration manifest"):
            hidden_regime_calibration_manifest(name)
        with pytest.raises(ValueError, match="calibration manifest"):
            hidden_regime_calibration_world_config(name)

    for name in calibration_names:
        manifest = hidden_regime_calibration_manifest(name)
        use = hidden_regime_manifest_use(name)
        assert manifest.use_partition == CALIBRATION_ONLY_PARTITION
        assert use.use_partition == CALIBRATION_ONLY_PARTITION
        assert use.learner_outcome_status == CALIBRATION_ONLY_PERFORMANCE_STATUS
        assert use.learner_outcomes_executed is None
        assert use.calibration_use_allowed is True
        assert use.protected_evaluation_candidate is False
        assert use.scientific_promotion_allowed is False
        with pytest.raises(ValueError, match="structural manifest"):
            hidden_regime_structural_manifest(name)
        with pytest.raises(ValueError, match="structural manifest"):
            hidden_regime_world_config_for_manifest(name)

    with pytest.raises(TypeError):
        HIDDEN_REGIME_CALIBRATION_MANIFESTS["new"] = HIDDEN_REGIME_CALIBRATION_A_V1  # type: ignore[index]
    with pytest.raises(TypeError):
        HIDDEN_REGIME_MANIFEST_USE_LEDGER["new"] = hidden_regime_manifest_use(  # type: ignore[index]
            structural_names[0]
        )
    with pytest.raises(dataclasses.FrozenInstanceError):
        hidden_regime_manifest_use(structural_names[0]).learner_outcomes_executed = True  # type: ignore[misc]


def test_structural_manifest_registry_and_values_are_immutable() -> None:
    with pytest.raises(TypeError):
        HIDDEN_REGIME_STRUCTURAL_MANIFESTS["new"] = HIDDEN_REGIME_STRUCTURAL_A_V1  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        HIDDEN_REGIME_STRUCTURAL_A_V1.name = "changed"  # type: ignore[misc]
    assert isinstance(HIDDEN_REGIME_STRUCTURAL_A_V1.segment_lengths, tuple)
    assert isinstance(HIDDEN_REGIME_STRUCTURAL_A_V1.length_perturbations_by_regime, tuple)
    assert all(
        isinstance(row, tuple)
        for row in HIDDEN_REGIME_STRUCTURAL_A_V1.length_perturbations_by_regime
    )


@pytest.mark.parametrize("manifest", tuple(HIDDEN_REGIME_STRUCTURAL_MANIFESTS.values()))
def test_structural_manifest_preserves_exposure_and_lifecycle_invariants(
    manifest: HiddenRegimeScheduleManifest,
) -> None:
    expected_counts = (6, 5, 2, 3, 1)
    expected_exposures = (6144, 5376, 1920, 3072, 16)
    counts = tuple(manifest.segment_regimes.count(regime) for regime in range(5))
    exposures = tuple(
        sum(
            length
            for length, actual_regime in zip(
                manifest.segment_lengths,
                manifest.segment_regimes,
                strict=True,
            )
            if actual_regime == regime
        )
        for regime in range(5)
    )
    assert len(manifest.segment_lengths) == 17
    assert sum(manifest.segment_lengths) == 16_528
    assert counts == expected_counts
    assert exposures == expected_exposures
    assert tuple(sum(row) for row in manifest.length_perturbations_by_regime) == (0,) * 5
    assert manifest.segment_lengths == _lengths_for_order_and_perturbations(
        manifest.segment_regimes,
        manifest.length_perturbations_by_regime,
    )

    d_index = manifest.segment_regimes.index(4)
    assert manifest.segment_regimes.count(4) == 1
    assert manifest.segment_lengths[d_index] == 16
    first_c_new = manifest.segment_regimes.index(3)
    bank_fill_prefix = manifest.segment_regimes[:first_c_new]
    assert all(bank_fill_prefix.count(regime) >= 2 for regime in (0, 1, 2))
    assert {0, 1, 3}.issubset(manifest.segment_regimes[d_index + 1 :])

    assert (
        manifest.change_point_residues
        == _EXPECTED_STRUCTURAL_MANIFESTS[manifest.name]["change_point_residues"]
    )
    assert all(residue != 0 for residue in manifest.change_point_residues)
    assert len(set(manifest.change_point_residues)) >= 8


@pytest.mark.parametrize("manifest", tuple(HIDDEN_REGIME_CALIBRATION_MANIFESTS.values()))
def test_calibration_manifest_preserves_the_same_tight_off_grid_grammar(
    manifest: HiddenRegimeScheduleManifest,
) -> None:
    counts = tuple(manifest.segment_regimes.count(regime) for regime in range(5))
    exposures = tuple(
        sum(
            length
            for length, actual_regime in zip(
                manifest.segment_lengths,
                manifest.segment_regimes,
                strict=True,
            )
            if actual_regime == regime
        )
        for regime in range(5)
    )
    assert counts == (6, 5, 2, 3, 1)
    assert exposures == (6144, 5376, 1920, 3072, 16)
    assert sum(manifest.segment_lengths) == 16_528
    assert tuple(sum(row) for row in manifest.length_perturbations_by_regime) == (0,) * 5
    assert manifest.segment_lengths == _lengths_for_order_and_perturbations(
        manifest.segment_regimes,
        manifest.length_perturbations_by_regime,
    )
    d_index = manifest.segment_regimes.index(4)
    first_c_new = manifest.segment_regimes.index(3)
    assert manifest.segment_lengths[d_index] == 16
    assert manifest.segment_regimes[:d_index].count(3) == 2
    assert manifest.segment_regimes[d_index + 1 :].count(3) == 1
    assert all(manifest.segment_regimes[:first_c_new].count(regime) >= 2 for regime in (0, 1, 2))
    assert {0, 1, 3}.issubset(manifest.segment_regimes[d_index + 1 :])
    assert (
        manifest.change_point_residues
        == _EXPECTED_CALIBRATION_MANIFESTS[manifest.name]["change_point_residues"]
    )
    assert all(residue != 0 for residue in manifest.change_point_residues)
    assert len(set(manifest.change_point_residues)) >= 8


def test_structural_manifest_orders_and_symbol_families_are_distinct() -> None:
    manifests = tuple(HIDDEN_REGIME_STRUCTURAL_MANIFESTS.values())
    assert len({manifest.segment_regimes for manifest in manifests}) == len(manifests)
    assert len({manifest.segment_lengths for manifest in manifests}) == len(manifests)
    assert len({manifest.cue_symbol_relabeling for manifest in manifests}) == len(manifests)
    assert len({manifest.action_symbol_relabeling for manifest in manifests}) == len(manifests)
    assert len({manifest.regime_permutations for manifest in manifests}) == len(manifests)


def test_calibration_and_protected_manifest_families_are_structurally_disjoint() -> None:
    structural = tuple(HIDDEN_REGIME_STRUCTURAL_MANIFESTS.values())
    calibration = tuple(HIDDEN_REGIME_CALIBRATION_MANIFESTS.values())
    all_manifests = structural + calibration
    assert len({manifest.segment_regimes for manifest in all_manifests}) == len(all_manifests)
    assert len({manifest.segment_lengths for manifest in all_manifests}) == len(all_manifests)
    assert len(
        {
            (manifest.cue_symbol_relabeling, manifest.action_symbol_relabeling)
            for manifest in all_manifests
        }
    ) == len(all_manifests)
    assert len({manifest.regime_permutations for manifest in all_manifests}) == len(all_manifests)
    assert not (
        {manifest.length_perturbations_by_regime for manifest in structural}
        & {manifest.length_perturbations_by_regime for manifest in calibration}
    )


@pytest.mark.parametrize("manifest", tuple(HIDDEN_REGIME_STRUCTURAL_MANIFESTS.values()))
def test_structural_symbol_relabeling_is_exact_conjugation(
    manifest: HiddenRegimeScheduleManifest,
) -> None:
    inverse_cue = [0] * 3
    for canonical_cue, visible_cue in enumerate(manifest.cue_symbol_relabeling):
        inverse_cue[visible_cue] = canonical_cue
    expected = tuple(
        tuple(
            manifest.action_symbol_relabeling[
                DEFAULT_REGIME_PERMUTATIONS[regime][inverse_cue[visible_cue]]
            ]
            for visible_cue in range(3)
        )
        for regime in range(5)
    )
    assert manifest.regime_permutations == expected
    config = hidden_regime_world_config_for_manifest(manifest.name)
    assert config.segment_lengths == manifest.segment_lengths
    assert config.segment_regimes == manifest.segment_regimes
    assert config.regime_permutations == expected
    assert config.repeat_schedule is False
    assert "manifest" not in config.to_dict()


@pytest.mark.parametrize("manifest", tuple(HIDDEN_REGIME_CALIBRATION_MANIFESTS.values()))
def test_calibration_symbol_relabeling_and_world_builder_are_exact_and_nonpromoting(
    manifest: HiddenRegimeScheduleManifest,
) -> None:
    inverse_cue = [0] * 3
    for canonical_cue, visible_cue in enumerate(manifest.cue_symbol_relabeling):
        inverse_cue[visible_cue] = canonical_cue
    expected = tuple(
        tuple(
            manifest.action_symbol_relabeling[
                DEFAULT_REGIME_PERMUTATIONS[regime][inverse_cue[visible_cue]]
            ]
            for visible_cue in range(3)
        )
        for regime in range(5)
    )
    assert manifest.regime_permutations == expected
    config = hidden_regime_calibration_world_config(manifest.name)
    assert config.segment_lengths == manifest.segment_lengths
    assert config.segment_regimes == manifest.segment_regimes
    assert config.regime_permutations == expected
    assert config.repeat_schedule is False
    assert config.to_dict()["scientific_promotion_allowed"] is False
    assert "manifest" not in config.to_dict()


@pytest.mark.parametrize("manifest", tuple(HIDDEN_REGIME_STRUCTURAL_MANIFESTS.values()))
def test_structural_world_is_deterministic_and_does_not_leak_manifest_or_oracle(
    manifest: HiddenRegimeScheduleManifest,
) -> None:
    config = hidden_regime_world_config_for_manifest(manifest.name, repeat_schedule=True)
    world_a = HiddenRegimeSignalingWorld(config)
    world_b = HiddenRegimeSignalingWorld(config)
    state_a = world_a.init(hidden_regime_world_keys(jr.key(6031)))
    state_b = world_b.init(hidden_regime_world_keys(jr.key(6031)))
    observation_fields = {field.name for field in dataclasses.fields(world_a.observe(state_a))}
    state_fields = {field.name for field in dataclasses.fields(state_a)}
    assert observation_fields == {"helper_cue"}
    assert state_fields == {"cue_key", "channel_key", "cue", "step_count", "schedule_position"}
    assert not ({"manifest", "regime", "target", "segment"} & observation_fields)
    assert not ({"manifest", "regime", "target", "segment"} & state_fields)

    for step in range(96):
        message = jnp.int32(step % 3)
        action = jnp.int32((step + 1) % 3)
        transition_a, state_a = world_a.step(state_a, message, action)
        transition_b, state_b = world_b.step(state_b, message, action)
        jax.tree.map(np.testing.assert_array_equal, transition_a, transition_b)
        np.testing.assert_array_equal(jr.key_data(state_a.cue_key), jr.key_data(state_b.cue_key))
        np.testing.assert_array_equal(
            jr.key_data(state_a.channel_key),
            jr.key_data(state_b.channel_key),
        )
        np.testing.assert_array_equal(state_a.cue, state_b.cue)
        np.testing.assert_array_equal(state_a.step_count, state_b.step_count)
        np.testing.assert_array_equal(state_a.schedule_position, state_b.schedule_position)


@pytest.mark.parametrize("manifest", tuple(HIDDEN_REGIME_STRUCTURAL_MANIFESTS.values()))
def test_structural_world_targets_use_relabeling_without_changing_observation_shape(
    manifest: HiddenRegimeScheduleManifest,
) -> None:
    world = HiddenRegimeSignalingWorld(hidden_regime_world_config_for_manifest(manifest.name))
    initial = world.init(hidden_regime_world_keys(jr.key(902)))
    starts = np.cumsum((0, *manifest.segment_lengths[:-1]))
    for segment_index, (position, regime) in enumerate(
        zip(starts, manifest.segment_regimes, strict=True)
    ):
        for cue in range(3):
            state = dataclasses.replace(
                initial,
                cue=jnp.int32(cue),
                step_count=jnp.int32(position),
                schedule_position=jnp.int32(position),
            )
            observation = world.observe(state)
            transition, _ = world.step(state, jnp.int32(0), jnp.int32(0))
            assert int(observation.helper_cue) == cue
            assert int(transition.oracle.segment_index) == segment_index
            assert int(transition.oracle.regime_id) == regime
            assert int(transition.oracle.target) == manifest.regime_permutations[regime][cue]


@pytest.mark.parametrize(
    "name",
    ("", "hidden-regime-structural-a", " hidden-regime-structural-a-v1", 1),
)
def test_structural_manifest_names_fail_closed(name: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        hidden_regime_structural_manifest(name)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name",
    ("", "hidden-regime-calibration-a", " hidden-regime-calibration-a-v1", 1),
)
def test_calibration_manifest_and_use_ledger_names_fail_closed(name: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        hidden_regime_calibration_manifest(name)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        hidden_regime_manifest_use(name)  # type: ignore[arg-type]


def test_manifest_name_and_use_partition_cannot_be_crossed() -> None:
    with pytest.raises(ValueError, match="match its explicit use partition"):
        dataclasses.replace(
            HIDDEN_REGIME_STRUCTURAL_A_V1,
            use_partition=CALIBRATION_ONLY_PARTITION,
        )
    with pytest.raises(ValueError, match="match its explicit use partition"):
        dataclasses.replace(
            HIDDEN_REGIME_CALIBRATION_A_V1,
            use_partition=PROTECTED_CANDIDATE_PARTITION,
        )
    with pytest.raises(ValueError, match="use_partition"):
        dataclasses.replace(
            HIDDEN_REGIME_STRUCTURAL_A_V1,
            use_partition="unknown",  # type: ignore[arg-type]
        )


def test_structural_manifest_rejects_invalid_total_delta_and_boundary_phase() -> None:
    manifest = HIDDEN_REGIME_STRUCTURAL_A_V1
    changed_lengths = (manifest.segment_lengths[0] + 1, *manifest.segment_lengths[1:])
    with pytest.raises(ValueError):
        dataclasses.replace(manifest, segment_lengths=changed_lengths)

    changed_perturbations = (
        (3, *manifest.length_perturbations_by_regime[0][1:]),
        *manifest.length_perturbations_by_regime[1:],
    )
    with pytest.raises(ValueError):
        dataclasses.replace(manifest, length_perturbations_by_regime=changed_perturbations)

    aligned_perturbations = (
        (0, 11, 8, -3, -6, -10),
        *manifest.length_perturbations_by_regime[1:],
    )
    aligned_lengths = _lengths_for_order_and_perturbations(
        manifest.segment_regimes,
        aligned_perturbations,
    )
    with pytest.raises(ValueError, match="nonzero lease phases"):
        dataclasses.replace(
            manifest,
            segment_lengths=aligned_lengths,
            length_perturbations_by_regime=aligned_perturbations,
        )

    seven_phase_perturbations = (
        (1, 0, 1, -2, 1, -1),
        (2, -1, 2, -1, -2),
        (1, -1),
        (2, 2, -4),
        (0,),
    )
    seven_phase_lengths = _lengths_for_order_and_perturbations(
        manifest.segment_regimes,
        seven_phase_perturbations,
    )
    with pytest.raises(ValueError, match="at least eight distinct"):
        dataclasses.replace(
            manifest,
            segment_lengths=seven_phase_lengths,
            length_perturbations_by_regime=seven_phase_perturbations,
        )


def test_structural_manifest_rejects_missing_bank_fill_or_post_transient_recurrence() -> None:
    manifest = HIDDEN_REGIME_STRUCTURAL_A_V1
    no_bank_fill = (0, 1, 3, 2, 0, 1, 2, 0, 1, 0, 3, 1, 0, 4, 1, 3, 0)
    with pytest.raises(ValueError, match="bank-fill"):
        dataclasses.replace(
            manifest,
            segment_regimes=no_bank_fill,
            segment_lengths=_lengths_for_order_and_perturbations(
                no_bank_fill,
                manifest.length_perturbations_by_regime,
            ),
        )

    d_last = (*tuple(regime for regime in manifest.segment_regimes if regime != 4), 4)
    with pytest.raises(ValueError, match="C-new occurrences|post-transient"):
        dataclasses.replace(
            manifest,
            segment_regimes=d_last,
            segment_lengths=_lengths_for_order_and_perturbations(
                d_last,
                manifest.length_perturbations_by_regime,
            ),
        )

    d_before_replacement = (0, 0, 2, 2, 1, 0, 1, 1, 4, 1, 3, 3, 0, 0, 3, 1, 0)
    with pytest.raises(ValueError, match="two C-new occurrences before D"):
        dataclasses.replace(
            manifest,
            segment_regimes=d_before_replacement,
            segment_lengths=_lengths_for_order_and_perturbations(
                d_before_replacement,
                manifest.length_perturbations_by_regime,
            ),
        )


def test_structural_manifest_rejects_invalid_symbol_relabeling() -> None:
    manifest = HIDDEN_REGIME_STRUCTURAL_A_V1
    with pytest.raises(ValueError, match="cue_symbol_relabeling"):
        dataclasses.replace(manifest, cue_symbol_relabeling=(0, 0, 2))
    with pytest.raises(ValueError, match="action_symbol_relabeling"):
        dataclasses.replace(manifest, action_symbol_relabeling=(0, 1, 3))
    with pytest.raises(ValueError, match="nonidentity"):
        dataclasses.replace(
            manifest,
            cue_symbol_relabeling=(0, 1, 2),
            action_symbol_relabeling=(0, 1, 2),
        )


@pytest.mark.parametrize(
    ("manifest", "expected_final_regime"),
    (
        (HIDDEN_REGIME_STRUCTURAL_A_V1, 0),
        (HIDDEN_REGIME_CALIBRATION_C_V1, 1),
    ),
)
@pytest.mark.parametrize(
    ("extension", "expected_period"),
    ((1, 16), (7, 16), (8, 2), (15, 16)),
)
def test_repeating_phase_drift_world_is_explicit_changed_and_phase_shifting(
    manifest: HiddenRegimeScheduleManifest,
    expected_final_regime: int,
    extension: int,
    expected_period: int,
) -> None:
    source_before = manifest.to_dict()
    use_before = hidden_regime_manifest_use(manifest.name).to_dict()
    derived = build_hidden_regime_repeating_phase_drift_world(
        manifest,
        final_regime_extension_steps=extension,
    )
    source_after = manifest.to_dict()
    use_after = hidden_regime_manifest_use(manifest.name).to_dict()
    config = derived.world
    metadata = derived.metadata

    assert source_after == source_before
    assert use_after == use_before
    assert not isinstance(derived, HiddenRegimeWorldConfig)
    assert config.repeat_schedule is True
    assert config.segment_lengths[:-1] == manifest.segment_lengths[:-1]
    assert config.segment_lengths[-1] == manifest.segment_lengths[-1] + extension
    assert config.segment_regimes == manifest.segment_regimes
    assert config.regime_permutations == manifest.regime_permutations
    assert config.total_schedule_steps == 16_528 + extension
    assert config.total_schedule_steps % 16 == extension

    assert metadata.source_manifest_name == manifest.name
    assert metadata.source_use_partition == manifest.use_partition
    assert metadata.final_regime_id == expected_final_regime
    assert metadata.final_regime_extension_steps == extension
    assert metadata.base_total_schedule_steps == 16_528
    assert metadata.repeated_total_schedule_steps == 16_528 + extension
    assert metadata.cycle_phase_drift == extension
    assert metadata.phase_period_cycles == expected_period
    assert len(set(metadata.cycle_start_residues)) == expected_period
    assert all(
        (right - left) % 16 == extension
        for left, right in zip(
            metadata.cycle_start_residues,
            (*metadata.cycle_start_residues[1:], 0),
            strict=True,
        )
    )
    expected_exposure = list(metadata.base_exposure_by_regime)
    expected_exposure[expected_final_regime] += extension
    assert metadata.repeated_exposure_by_regime == tuple(expected_exposure)

    payload = derived.to_dict()
    assert payload["schema"] == "hidden-regime-repeating-phase-drift-world-v1"
    assert payload["derived_schedule_not_finite_manifest"] is True
    metadata_payload = payload["metadata"]
    assert isinstance(metadata_payload, dict)
    assert metadata_payload["finite_manifest_equivalent"] is False
    assert metadata_payload["source_manifest_unchanged"] is True
    assert metadata_payload["construction_is_outcome_free"] is True
    assert metadata_payload["builder_authorizes_learner_execution"] is False
    assert metadata_payload["source_calibration_use_allowed"] is (
        manifest.use_partition == CALIBRATION_ONLY_PARTITION
    )
    assert metadata_payload["protected_source_performance_must_remain_unexecuted"] is (
        manifest.use_partition == PROTECTED_CANDIDATE_PARTITION
    )
    assert metadata_payload["scientific_promotion_allowed"] is False


def test_repeating_phase_drift_wraps_and_does_not_add_learner_visible_fields() -> None:
    manifest = HIDDEN_REGIME_CALIBRATION_A_V1
    derived = build_hidden_regime_repeating_phase_drift_world(
        manifest,
        final_regime_extension_steps=5,
    )
    world = HiddenRegimeSignalingWorld(derived.world)
    initial = world.init(hidden_regime_world_keys(jr.key(604)))
    state = dataclasses.replace(
        initial,
        schedule_position=jnp.int32(derived.world.total_schedule_steps - 1),
    )
    transition, next_state = jax.jit(world.step)(state, jnp.int32(0), jnp.int32(0))
    assert int(transition.oracle.segment_index) == len(manifest.segment_lengths) - 1
    assert int(next_state.schedule_position) == 0
    assert {field.name for field in dataclasses.fields(world.observe(next_state))} == {"helper_cue"}
    assert {field.name for field in dataclasses.fields(next_state)} == {
        "cue_key",
        "channel_key",
        "cue",
        "step_count",
        "schedule_position",
    }
    world_payload = derived.world.to_dict()
    assert "manifest" not in world_payload
    assert "source_manifest_name" not in world_payload
    assert "phase_drift" not in world_payload
    assert "extension" not in world_payload


@pytest.mark.parametrize("manifest", (None, "hidden-regime-structural-a-v1"))
def test_repeating_phase_drift_requires_an_explicit_manifest_object(manifest: object) -> None:
    with pytest.raises(TypeError, match="explicit schedule manifest"):
        build_hidden_regime_repeating_phase_drift_world(
            manifest,  # type: ignore[arg-type]
            final_regime_extension_steps=1,
        )


@pytest.mark.parametrize("extension", (True, 1.0, "1", None))
def test_repeating_phase_drift_rejects_nonstrict_extension_types(extension: object) -> None:
    with pytest.raises(TypeError, match="strict integer"):
        build_hidden_regime_repeating_phase_drift_world(
            HIDDEN_REGIME_STRUCTURAL_A_V1,
            final_regime_extension_steps=extension,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("extension", (-1, 0, 16, 17))
def test_repeating_phase_drift_rejects_nonphase_shifting_extensions(extension: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 15"):
        build_hidden_regime_repeating_phase_drift_world(
            HIDDEN_REGIME_STRUCTURAL_A_V1,
            final_regime_extension_steps=extension,
        )


def test_repeating_public_schedule_helpers_require_authoritative_state() -> None:
    world = HiddenRegimeSignalingWorld(_tiny_config(repeat=True))
    initial = world.init(hidden_regime_world_keys(jr.key(601)))
    maximum = np.iinfo(np.int32).max
    state = dataclasses.replace(
        initial,
        cue=jnp.int32(2),
        step_count=jnp.int32(maximum),
        schedule_position=jnp.int32(1),
    )
    assert int(world.schedule_position_of(state)) == 1
    assert int(world.segment_index_of(state)) == 0
    assert int(world.segment_step_of(state)) == 1
    assert int(world.regime_of(state)) == 0
    assert int(world.target_of(state, state.cue)) == 2
    for helper in (
        world.schedule_position_of,
        world.segment_index_of,
        world.segment_step_of,
        world.regime_of,
    ):
        with pytest.raises(ValueError, match="authoritative HiddenRegimeWorldState"):
            helper(state.step_count)
    with pytest.raises(ValueError, match="authoritative HiddenRegimeWorldState"):
        world.target_of(state.step_count, state.cue)


def test_default_nonrepeating_config_and_counter_helpers_remain_backward_compatible() -> None:
    config = HiddenRegimeWorldConfig()
    assert config.segment_lengths == DEFAULT_SEGMENT_LENGTHS
    assert config.segment_regimes == DEFAULT_SEGMENT_REGIMES
    assert config.regime_permutations == DEFAULT_REGIME_PERMUTATIONS
    world = HiddenRegimeSignalingWorld(config)
    state = world.init(hidden_regime_world_keys(jr.key(602)))
    assert int(world.schedule_position_of(state.step_count)) == 0
    assert int(world.segment_index_of(state.step_count)) == 0
    assert int(world.segment_step_of(state.step_count)) == 0
    assert int(world.regime_of(state.step_count)) == 0
    assert int(world.target_of(state.step_count, state.cue)) == int(
        world.target_of(state, state.cue)
    )
