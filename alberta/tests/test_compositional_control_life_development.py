"""Unit contracts for the silent-task compositional control life."""

from __future__ import annotations

import copy
from typing import cast

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.compositional_features import (
    OP_PRODUCT,
    CompositionalFeatureLearner,
    CompositionalFeatureState,
)
from alberta_framework.evaluation.compositional_control_life_development import (
    ACTIVE_SLOTS,
    CANDIDATE_SLOTS,
    CONTROL_LIFE_ARMS,
    DEFAULT_PHASE_LENGTHS,
    PHASE_ORDER,
    RAW_DIM,
    RAW_PAIR_NAMES,
    CompositionalControlLifeProtocol,
    build_default_protocol,
    compositional_control_state_nbytes_formula,
    learner_config_for_arm,
    product_signature_counts,
)
from alberta_framework.evaluation.generated_class_lifecycle_scrub import (
    persistent_compositional_state_nbytes,
)

pytestmark = pytest.mark.unit


def _genesis(arm_name: str) -> CompositionalFeatureState:
    learner = CompositionalFeatureLearner.from_config(
        learner_config_for_arm(arm_name)
    )
    return learner.init(feature_dim=RAW_DIM, key=jr.key(17))


def test_protocol_is_exact_silent_nonpromoting_and_round_trips_strictly() -> None:
    protocol = build_default_protocol()

    assert protocol.phase_lengths == DEFAULT_PHASE_LENGTHS
    assert protocol.total_steps == 8_998
    assert PHASE_ORDER == ("A", "B", "A", "D", "A", "C", "A", "B", "C", "A")
    config = protocol.to_config()
    assert config["learner_observation_fields"] == ["raw_rademacher_values"]
    assert config["learner_feedback_fields"] == ["selected_action_reward"]
    assert config["resets_allowed"] is False
    assert config["scientific_promotion_allowed"] is False
    assert CompositionalControlLifeProtocol.from_config(config) == protocol

    malformed = copy.deepcopy(config)
    malformed["phase_order"] = list(reversed(PHASE_ORDER))
    with pytest.raises(ValueError, match="reconstruct exactly"):
        CompositionalControlLifeProtocol.from_config(malformed)

    malformed = copy.deepcopy(config)
    malformed["threshold"] = 0.0
    with pytest.raises(ValueError, match="fields"):
        CompositionalControlLifeProtocol.from_config(malformed)


def test_arm_matrix_uses_only_declared_novelty_ancestor_and_depth_ablations() -> None:
    names = tuple(arm.name for arm in CONTROL_LIFE_ARMS)
    assert names == (
        "myopic_full",
        "explore_ancestor",
        "dovetail_coverage_ancestor",
        "dovetail_coverage_ancestor_headroom",
        "dovetail_coverage_ancestor_headroom_leftpack",
        "explore_ancestor_readout_blocked",
        "explore_ancestor_no_slow",
        "depth1_ceiling",
    )

    configs = {name: learner_config_for_arm(name) for name in names}
    assert configs["myopic_full"]["candidate_novelty_admission_bonus"] == 0.0
    assert configs["myopic_full"]["ancestor_utility_backup_decay"] == 0.0
    assert configs["explore_ancestor"]["candidate_novelty_admission_bonus"] == 1.0
    assert configs["explore_ancestor"]["ancestor_utility_backup_decay"] == 0.95
    assert configs["dovetail_coverage_ancestor"]["generation_strategy"] == (
        "dovetail_product_coverage"
    )
    assert configs["dovetail_coverage_ancestor"]["operation_prior"] is None
    assert configs["dovetail_coverage_ancestor"]["topology_headroom_reserve"] is False
    assert configs["dovetail_coverage_ancestor_headroom"][
        "topology_headroom_reserve"
    ] is True
    assert {
        key
        for key in configs["dovetail_coverage_ancestor"]
        if configs["dovetail_coverage_ancestor"][key]
        != configs["dovetail_coverage_ancestor_headroom"][key]
    } == {"topology_headroom_reserve"}
    assert configs["dovetail_coverage_ancestor_headroom_leftpack"][
        "topology_left_pack_destinations"
    ] is True
    assert {
        key
        for key in configs["dovetail_coverage_ancestor_headroom"]
        if configs["dovetail_coverage_ancestor_headroom"][key]
        != configs["dovetail_coverage_ancestor_headroom_leftpack"][key]
    } == {"topology_left_pack_destinations"}
    assert configs["explore_ancestor_readout_blocked"] == configs["explore_ancestor"]
    assert configs["explore_ancestor_no_slow"]["retention_slow_utility_decay"] == 0.0
    assert configs["depth1_ceiling"]["max_depth"] == 1
    assert all(config["n_features"] == ACTIVE_SLOTS for config in configs.values())
    assert all(config["candidate_count"] == CANDIDATE_SLOTS for config in configs.values())
    assert all(config["n_tasks"] == 2 for config in configs.values())


def test_robust_genesis_has_fixed_budget_and_no_target_scaffold() -> None:
    expected_nbytes = compositional_control_state_nbytes_formula(
        active_slots=ACTIVE_SLOTS,
        candidate_slots=CANDIDATE_SLOTS,
        action_heads=2,
    )
    assert expected_nbytes == 2_072

    for arm in CONTROL_LIFE_ARMS:
        state = _genesis(arm.name)
        assert state.output_weights.shape == (2, ACTIVE_SLOTS)
        assert state.candidate_output_weights.shape == (2, CANDIDATE_SLOTS)
        assert persistent_compositional_state_nbytes(state) == expected_nbytes
        counts = product_signature_counts(state)
        active_counts = cast(dict[str, int], counts["active"])
        candidate_counts = cast(dict[str, int], counts["candidate"])
        active_pairs = cast(dict[str, int], counts["raw_pair_active"])
        candidate_pairs = cast(dict[str, int], counts["raw_pair_candidate"])
        assert set(active_counts.values()) == {0}
        assert set(candidate_counts.values()) == {0}
        assert tuple(active_pairs) == RAW_PAIR_NAMES
        assert {name for name, count in active_pairs.items() if count} == {
            "p01",
            "p02",
            "p03",
            "p04",
            "p05",
        }
        assert not any(candidate_pairs.values())


def test_product_signature_audit_is_factorization_independent() -> None:
    state = _genesis("explore_ancestor")
    left_associated = cast(
        CompositionalFeatureState,
        state.replace(  # type: ignore[attr-defined]
            ops=state.ops.at[6].set(OP_PRODUCT).at[7].set(OP_PRODUCT),
            parent_a=state.parent_a.at[6].set(4).at[7].set(1),
            parent_b=state.parent_b.at[6].set(5).at[7].set(6),
            depth=state.depth.at[6].set(1).at[7].set(2),
        ),
    )
    alternative = cast(
        CompositionalFeatureState,
        state.replace(  # type: ignore[attr-defined]
            ops=state.ops.at[6].set(OP_PRODUCT).at[7].set(OP_PRODUCT),
            parent_a=state.parent_a.at[6].set(1).at[7].set(6),
            parent_b=state.parent_b.at[6].set(4).at[7].set(5),
            depth=state.depth.at[6].set(1).at[7].set(2),
        ),
    )

    left_counts = cast(
        dict[str, int], product_signature_counts(left_associated)["active"]
    )
    alternative_counts = cast(
        dict[str, int], product_signature_counts(alternative)["active"]
    )
    assert left_counts["A"] == 1
    assert alternative_counts["A"] == 1
    assert left_counts["shared_p45"] == 1
    assert alternative_counts["shared_p45"] == 0
    assert jnp.array_equal(left_associated.ops, alternative.ops)
