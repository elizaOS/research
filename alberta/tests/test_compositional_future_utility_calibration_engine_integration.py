"""Production-trace integration for the identity-free calibration engine."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_features import CompositionalFeatureLearner
from alberta_framework.evaluation import (
    _compositional_future_utility_calibration_engine as engine,
)
from alberta_framework.evaluation import compositional_control_life_development as control

pytestmark = pytest.mark.integration


def test_production_trace_closes_exact_cadence_shapes_and_curation_algebra() -> None:
    protocol = control.build_short_test_protocol()
    source = control.build_bound_compositional_control_life_source(
        protocol,
        observation_key=jr.key(101),
        exploration_key=jr.key(102),
        random_action_key=jr.key(103),
        learner_key=jr.key(104),
    )
    arm = engine.FutureUtilityArmSpec(
        name="current_mix0_decay095_none",
        role="production integration reference",
        mix=0.0,
        trace_decay=0.95,
        normalization="none",
    )
    learner = CompositionalFeatureLearner.from_config(
        engine.build_future_utility_learner_config(
            control.learner_config_for_arm(
                "dovetail_coverage_ancestor_headroom_leftpack"
            ),
            arm,
        )
    )
    execution = control.execute_compositional_control_life_arm(
        protocol,
        learner,
        source.learner_key,
        source.observations,
        source.phase_indices,
        source.exploration_mask,
        source.random_actions,
        composed_readout_enabled=True,
    )
    geometry = engine.FutureUtilityEndpointGeometry(
        phase_order=control.PHASE_ORDER,
        phase_lengths=protocol.phase_lengths,
        target_names=("A", "B", "C"),
        curation_interval=control.CURATION_INTERVAL,
    )
    totals_array = np.sum(
        np.asarray(execution.events.curation_counts, dtype=np.int64),
        axis=0,
    )
    totals = {
        name: int(value)
        for name, value in zip(control.CURATION_COUNT_NAMES, totals_array, strict=True)
    }

    shapes = engine.validate_future_utility_trace_shapes(geometry, execution.events)
    audit = engine.future_utility_cadence_audit_from_events(
        geometry,
        execution.events,
        pinned_due_mask=source.curation_due_mask,
    )
    closure = engine.validate_future_utility_curation_count_closure(audit, totals)
    experience = engine.validate_future_utility_experience_semantics(
        geometry,
        execution.events,
        observations=source.observations,
        phase_indices=source.phase_indices,
        exploration_mask=source.exploration_mask,
        random_actions=source.random_actions,
        phase_target_raw_indices=(
            (1, 4, 5),
            (2, 4, 5),
            (1, 4, 5),
            (1, 2, 3),
            (1, 4, 5),
            (3, 4, 5),
            (1, 4, 5),
            (2, 4, 5),
            (3, 4, 5),
            (1, 4, 5),
        ),
        action_reward_multipliers=(-1.0, 1.0),
        composed_readout_enabled=True,
    )

    assert shapes["decision_margin_passed"] == [protocol.total_steps]
    assert shapes["decision_candidate_margin_eligible"] == [
        protocol.total_steps,
        control.CANDIDATE_SLOTS,
        control.ACTIVE_SLOTS,
    ]
    assert audit.due_opportunity_count == int(
        jnp.count_nonzero(source.curation_due_mask)
    )
    assert closure["all_checked_counts_close"] is True
    assert experience["all_experience_semantics_match"] is True
    assert totals["proposal"] == (
        totals["promotion"] + totals["ordinary_candidate_refresh"]
    )
    assert totals["post_promotion_candidate_refresh"] == totals["promotion"]
    assert all(
        partition.off_opportunity_count == 0
        for partition in audit.mutation_partitions.values()
    )

    tampered_reward = np.array(execution.events.executed_reward, copy=True)
    tampered_reward[0] += 2.0
    tampered_events = execution.events._replace(
        executed_reward=jnp.asarray(tampered_reward)
    )
    with pytest.raises(RuntimeError, match="reward"):
        engine.validate_future_utility_experience_semantics(
            geometry,
            tampered_events,
            observations=source.observations,
            phase_indices=source.phase_indices,
            exploration_mask=source.exploration_mask,
            random_actions=source.random_actions,
            phase_target_raw_indices=(
                (1, 4, 5),
                (2, 4, 5),
                (1, 4, 5),
                (1, 2, 3),
                (1, 4, 5),
                (3, 4, 5),
                (1, 4, 5),
                (2, 4, 5),
                (3, 4, 5),
                (1, 4, 5),
            ),
            action_reward_multipliers=(-1.0, 1.0),
            composed_readout_enabled=True,
        )
