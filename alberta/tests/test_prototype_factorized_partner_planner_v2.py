# mypy: disable-error-code="attr-defined,no-untyped-call,no-untyped-def"
"""Focused contracts for source-update-first routed partner planning v2."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.hccl_feature_consumer_route import (
    HCCL_FEATURE_CONTEXT_START,
    HCCL_FEATURE_PAIR_START,
    HCCLFeatureConsumerRoute,
    HCCLFeatureConsumerRouteResult,
)
from alberta_framework.core.prototype_factorized_partner_planner_v2 import (
    GROUNDED_OUTPUT_ORDER,
    NET_REWARD_OUTPUT_INDEX,
    PrototypeFactorizedPartnerPlannerV2,
    PrototypeFactorizedPartnerPlannerV2Config,
)

_PAIR_START = HCCL_FEATURE_PAIR_START
_SOURCE_PAIRS = ((0, 1), (0, 2), (1, 3))
_DESTINATION_PAIRS = ((1, 3), (0, 1), (2, 3))


def _pairs(*live: tuple[int, int]) -> jax.Array:
    values = np.full((12, 2), -1, dtype=np.int32)
    for index, descriptor in enumerate(live):
        values[index] = descriptor
    return jnp.asarray(values, dtype=jnp.int32)


def _admissions(*slots: int) -> jax.Array:
    values = np.zeros((12,), dtype=np.bool_)
    values[list(slots)] = True
    return jnp.asarray(values, dtype=jnp.bool_)


def _semantic_representation(
    physical: jax.Array,
    pairs: tuple[tuple[int, int], ...],
    *,
    sign: float,
) -> jax.Array:
    values = jnp.zeros((35,), dtype=jnp.float32)
    values = values.at[:16].set(physical)
    values = values.at[16:19].set(
        jnp.asarray((0.17 * sign, -0.09 * sign, 0.0), dtype=jnp.float32)
    )
    values = values.at[19:23].set(
        jnp.asarray((0.03, -0.04, 0.07, -0.02), dtype=jnp.float32) * sign
    )
    for local, (left, right) in enumerate(pairs):
        values = values.at[_PAIR_START + local].set(
            values[left] * values[right]
        )
    return values


def _representations() -> jax.Array:
    return jnp.stack(
        (
            _semantic_representation(
                jnp.linspace(0.05, 0.20, 16, dtype=jnp.float32),
                _SOURCE_PAIRS,
                sign=1.0,
            ),
            _semantic_representation(
                jnp.linspace(-0.21, -0.06, 16, dtype=jnp.float32),
                _SOURCE_PAIRS,
                sign=-1.0,
            ),
        )
    )


def _destination_representations() -> jax.Array:
    return jnp.stack(
        (
            _semantic_representation(
                jnp.linspace(-0.10, 0.05, 16, dtype=jnp.float32),
                _DESTINATION_PAIRS,
                sign=1.0,
            ),
            _semantic_representation(
                jnp.linspace(0.31, 0.16, 16, dtype=jnp.float32),
                _DESTINATION_PAIRS,
                sign=-1.0,
            ),
        )
    )


def _assert_tree_bit_exact(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree  # type: ignore[operator]
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            left_array = jr.key_data(left_array)
            right_array = jr.key_data(right_array)
        np.testing.assert_array_equal(left_array, right_array)


def _fixture():
    owners = (HCCLFeatureConsumerRoute(agent_index=0), HCCLFeatureConsumerRoute(agent_index=1))
    ledgers = tuple(
        owner.init(
            context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
            pair_descriptors=_pairs(*_SOURCE_PAIRS),
        )
        for owner in owners
    )
    planner = PrototypeFactorizedPartnerPlannerV2(
        PrototypeFactorizedPartnerPlannerV2Config()
    )
    source_features = _representations()
    state = planner.init(
        jr.key(19),
        ledger_agent_0=ledgers[0],
        ledger_agent_1=ledgers[1],
        representations=source_features,
    )
    routes = tuple(
        owner.prepare_successor(
            ledger,
            destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
            context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
            context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
            pair_descriptors=_pairs(*_DESTINATION_PAIRS),
            pair_admission_mask=_admissions(2),
        )
        for owner, ledger in zip(owners, ledgers, strict=True)
    )
    return planner, state, source_features, routes


def _step(
    planner: PrototypeFactorizedPartnerPlannerV2,
    state: object,
    source_features: jax.Array,
    routes: tuple[HCCLFeatureConsumerRouteResult, HCCLFeatureConsumerRouteResult],
    destination_features: jax.Array | None = None,
):
    task_score = jnp.asarray(1.25, dtype=jnp.float32)
    safety = jnp.asarray((0.10, 0.20), dtype=jnp.float32)
    message = jnp.asarray((0.05, 0.07), dtype=jnp.float32)
    net = task_score - message - safety
    return planner.observe_route_and_plan(
        state,  # type: ignore[arg-type]
        route_result_agent_0=routes[0],
        route_result_agent_1=routes[1],
        source_representations=source_features,
        destination_representations=(
            _destination_representations()
            if destination_features is None
            else destination_features
        ),
        executed_actions=jnp.asarray((0, 1), dtype=jnp.int32),
        next_physical_observations=jnp.stack(
            (
                jnp.linspace(-0.2, 0.1, 16, dtype=jnp.float32),
                jnp.linspace(0.3, 0.0, 16, dtype=jnp.float32),
            )
        ),
        task_score=task_score,
        safety_costs=safety,
        message_charges=message,
        net_rewards=net,
        discount=jnp.asarray(0.97, dtype=jnp.float32),
    )


@pytest.mark.unit
def test_source_updates_route_survivors_and_zero_newborns_before_planning() -> None:
    planner, state, source_features, routes = _fixture()
    result = _step(planner, state, source_features, routes)

    assert bool(result.transaction_applied)
    assert bool(planner.state_valid(result.state))
    assert bool(result.receipt.source_update.phase_valid)
    assert bool(result.receipt.feature_route.phase_valid)
    assert bool(result.receipt.plan.phase_valid)
    assert bool(jnp.all(result.receipt.source_update.behavior_update_applied))
    assert bool(jnp.all(result.receipt.source_update.grounded_update_applied))

    partner_actions = jnp.asarray((1, 0), dtype=jnp.int32)
    expected_behavior = planner.behavior_model.update(
        state.agent_0.behavior,
        source_features[0],
        partner_actions[0],
    ).state
    targets_0 = result.receipt.source_update.grounded_targets[0]
    expected_grounded = planner.grounded_world_model.update(
        state.agent_0.grounded,
        source_features[0],
        jnp.asarray(0, dtype=jnp.int32),
        partner_actions[0],
        targets_0[:19],
        targets_0[NET_REWARD_OUTPUT_INDEX],
        targets_0[-1],
    ).state

    destination = result.candidate_state.agent_0
    np.testing.assert_array_equal(
        destination.behavior.weights[:, _PAIR_START],
        expected_behavior.weights[:, _PAIR_START + 2],
    )
    np.testing.assert_array_equal(
        destination.grounded.weights[:, :, _PAIR_START],
        expected_grounded.weights[:, :, _PAIR_START + 2],
    )
    np.testing.assert_array_equal(
        destination.behavior.weights[:, _PAIR_START + 2],
        0.0,
    )
    np.testing.assert_array_equal(
        destination.grounded.weights[:, :, _PAIR_START + 2],
        0.0,
    )
    np.testing.assert_array_equal(destination.behavior.weights[:, -1], 0.0)
    np.testing.assert_array_equal(destination.grounded.weights[:, :, -1], 0.0)
    np.testing.assert_array_equal(destination.behavior.bias, expected_behavior.bias)
    np.testing.assert_array_equal(destination.grounded.bias, expected_grounded.bias)
    np.testing.assert_array_equal(destination.behavior.step_words, (0, 1))
    np.testing.assert_array_equal(destination.grounded.update_words, (0, 1))
    np.testing.assert_array_equal(destination.ledger.source_clock_words, (0, 1))
    np.testing.assert_array_equal(
        jr.key_data(destination.behavior.rng_key),
        jr.key_data(state.agent_0.behavior.rng_key),
    )


@pytest.mark.unit
def test_fixed_output_order_one_belief_four_cells_and_no_second_update() -> None:
    planner, state, source_features, routes = _fixture()
    result = _step(planner, state, source_features, routes)

    assert GROUNDED_OUTPUT_ORDER == tuple(
        [f"physical_{index}" for index in range(16)]
        + ["task_score", "safety_cost", "message_charge", "net_reward", "discount"]
    )
    targets = result.receipt.source_update.grounded_targets
    np.testing.assert_array_equal(targets[:, 16], 1.25)
    np.testing.assert_array_equal(
        targets[:, 17],
        jnp.asarray((0.10, 0.20), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        targets[:, 18],
        jnp.asarray((0.05, 0.07), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        targets[:, 19],
        jnp.asarray(1.25, dtype=jnp.float32)
        - jnp.asarray((0.05, 0.07), dtype=jnp.float32)
        - jnp.asarray((0.10, 0.20), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        targets[:, 20],
        jnp.asarray(0.97, dtype=jnp.float32),
    )

    plan = result.receipt.plan
    assert plan.world_raw_predictions.shape == (2, 2, 2, 21)
    np.testing.assert_array_equal(plan.joint_cells_evaluated_per_agent, (4, 4))
    assert bool(jnp.all(plan.identical_partner_belief_across_own_rows))
    for agent in range(2):
        np.testing.assert_array_equal(
            plan.partner_belief_by_own_action[agent, 0],
            plan.partner_belief[agent],
        )
        np.testing.assert_array_equal(
            plan.partner_belief_by_own_action[agent, 1],
            plan.partner_belief[agent],
        )
    np.testing.assert_array_equal(result.prepared_actions, plan.proposed_actions)
    np.testing.assert_array_equal(
        result.state.agent_0.behavior.step_words,
        result.receipt.source_update.behavior_post_step_words[0],
    )
    np.testing.assert_array_equal(
        result.state.agent_0.grounded.update_words,
        result.receipt.source_update.grounded_post_update_words[0],
    )
    assert bool(jnp.all(plan.model_clocks_unchanged_by_planning))
    assert int(result.work.behavior_model_updates) == 2
    assert int(result.work.grounded_model_updates) == 2
    assert int(result.work.grounded_joint_cell_evaluations) == 8
    assert int(result.work.second_model_updates) == 0
    assert int(result.work.post_init_rng_draws) == 0


@pytest.mark.unit
def test_invalid_or_stale_route_fails_closed_to_bit_exact_source() -> None:
    planner, state, source_features, routes = _fixture()
    bad_route = routes[0].replace(
        witness=routes[0].witness.replace(
            route_map=routes[0].witness.route_map.replace(
                source_slots=routes[0].witness.route_map.source_slots.at[0].set(-1)
            )
        )
    )
    invalid = _step(planner, state, source_features, (bad_route, routes[1]))
    assert not bool(invalid.transaction_applied)
    assert not bool(invalid.receipt.feature_route.route_result_integrity_valid[0])
    _assert_tree_bit_exact(invalid.state, state)

    accepted = _step(planner, state, source_features, routes)
    stale = _step(
        planner,
        accepted.state,
        _destination_representations(),
        routes,
    )
    assert not bool(stale.transaction_applied)
    assert not bool(jnp.all(stale.receipt.feature_route.route_result_integrity_valid))
    _assert_tree_bit_exact(stale.state, accepted.state)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutator",
    [
        lambda state: state.replace(
            agent_0=state.agent_0.replace(
                behavior=state.agent_0.behavior.replace(
                    weights=state.agent_0.behavior.weights.at[0, 0].add(0.25)
                )
            )
        ),
        lambda state: state.replace(
            agent_0=state.agent_0.replace(
                grounded=state.agent_0.grounded.replace(
                    bias=state.agent_0.grounded.bias.at[0, 0].add(0.25)
                )
            )
        ),
        lambda state: state.replace(
            agent_0=state.agent_0.replace(
                grounded=state.agent_0.grounded.replace(
                    update_words=jnp.asarray((0, 1), dtype=jnp.uint32)
                )
            )
        ),
    ],
)
def test_source_column_bias_or_clock_tamper_fails_closed(
    mutator: Callable[[object], object],
) -> None:
    planner, state, source_features, routes = _fixture()
    tampered = planner._seal_state(mutator(state))
    result = _step(planner, tampered, source_features, routes)

    assert not bool(result.receipt.source_state_valid)
    assert not bool(result.transaction_applied)
    _assert_tree_bit_exact(result.state, tampered)


@pytest.mark.unit
@pytest.mark.parametrize(
    "slot",
    [
        pytest.param(_PAIR_START, id="survivor-pair"),
        pytest.param(_PAIR_START + 2, id="newborn-pair"),
        pytest.param(_PAIR_START + 11, id="inactive-pair"),
        pytest.param(HCCL_FEATURE_CONTEXT_START + 2, id="inactive-context"),
    ],
)
def test_destination_representation_semantic_tamper_fails_closed(slot: int) -> None:
    planner, state, source_features, routes = _fixture()
    destination = _destination_representations().at[0, slot].add(
        jnp.asarray(0.125, dtype=jnp.float32)
    )
    result = _step(
        planner,
        state,
        source_features,
        routes,
        destination,
    )

    assert not bool(result.receipt.feature_route.destination_representation_matches_ledger[0])
    assert not bool(result.receipt.event_inputs_valid)
    assert not bool(result.transaction_applied)
    _assert_tree_bit_exact(result.state, state)


@pytest.mark.unit
def test_source_representation_is_checked_even_in_a_resealed_coherent_cache() -> None:
    planner, state, source_features, routes = _fixture()
    malformed = source_features.at[0, _PAIR_START].add(
        jnp.asarray(0.125, dtype=jnp.float32)
    )
    agent = state.agent_0
    malicious_cache = planner._build_cache(
        agent.ledger,
        agent.behavior,
        agent.grounded,
        malformed[0],
    )
    malicious = planner._seal_state(
        state.replace(agent_0=agent.replace(cache=malicious_cache))
    )

    assert not bool(planner.state_valid(malicious))
    result = _step(planner, malicious, malformed, routes)
    assert not bool(result.receipt.source_update.source_representation_matches_ledger[0])
    assert not bool(result.receipt.source_state_valid)
    assert not bool(result.transaction_applied)
    _assert_tree_bit_exact(result.state, malicious)


@pytest.mark.unit
@pytest.mark.parametrize("forge", ["belief", "raw_prediction"])
def test_resealed_coherent_cache_forgery_is_recomputed_and_rejected(forge: str) -> None:
    planner, state, source_features, routes = _fixture()
    cache = state.agent_0.cache
    if forge == "belief":
        belief = jnp.asarray((0.75, 0.25), dtype=jnp.float32)
        expected = cache.world_raw_predictions[:, :, NET_REWARD_OUTPUT_INDEX] @ belief
        forged_cache = cache.replace(
            partner_belief=belief,
            partner_belief_by_own_action=jnp.broadcast_to(belief, (2, 2)),
            expected_net_rewards=expected,
            prepared_action=jnp.argmax(expected).astype(jnp.int32),
        )
    else:
        forged_cache = cache.replace(
            world_raw_predictions=cache.world_raw_predictions.at[0, 0, 0].add(
                jnp.asarray(0.125, dtype=jnp.float32)
            )
        )
    forged = planner._seal_state(
        state.replace(agent_0=state.agent_0.replace(cache=forged_cache))
    )

    assert not bool(planner.state_valid(forged))
    result = _step(planner, forged, source_features, routes)
    assert not bool(result.receipt.source_state_valid)
    assert not bool(result.transaction_applied)
    _assert_tree_bit_exact(result.state, forged)


@pytest.mark.unit
def test_serialized_float_fields_reject_python_integer_aliases() -> None:
    config = PrototypeFactorizedPartnerPlannerV2Config()
    payload = config.to_config()
    payload["behavior_step_size"] = 1

    with pytest.raises(ValueError, match="exact float"):
        PrototypeFactorizedPartnerPlannerV2Config.from_config(payload)


@pytest.mark.unit
def test_work_receipt_matches_spied_update_and_inference_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner, state, source_features, routes = _fixture()
    counts = {
        "behavior_update": 0,
        "grounded_update": 0,
        "behavior_predict": 0,
        "grounded_predict": 0,
    }
    behavior_update = planner.behavior_model.update
    grounded_update = planner.grounded_world_model.update
    behavior_predict = planner.behavior_model.predict_probabilities
    grounded_predict = planner.grounded_world_model.predict

    def spy_behavior_update(*args: object, **kwargs: object) -> object:
        counts["behavior_update"] += 1
        return behavior_update(*args, **kwargs)

    def spy_grounded_update(*args: object, **kwargs: object) -> object:
        counts["grounded_update"] += 1
        return grounded_update(*args, **kwargs)

    def spy_behavior_predict(*args: object, **kwargs: object) -> object:
        counts["behavior_predict"] += 1
        return behavior_predict(*args, **kwargs)

    def spy_grounded_predict(*args: object, **kwargs: object) -> object:
        counts["grounded_predict"] += 1
        return grounded_predict(*args, **kwargs)

    monkeypatch.setattr(planner.behavior_model, "update", spy_behavior_update)
    monkeypatch.setattr(planner.grounded_world_model, "update", spy_grounded_update)
    monkeypatch.setattr(
        planner.behavior_model,
        "predict_probabilities",
        spy_behavior_predict,
    )
    monkeypatch.setattr(planner.grounded_world_model, "predict", spy_grounded_predict)

    result = _step(planner, state, source_features, routes)
    work = result.work
    assert bool(result.transaction_applied)
    assert counts["behavior_update"] == int(work.behavior_model_updates)
    assert counts["grounded_update"] == int(work.grounded_model_updates)
    assert counts["behavior_predict"] == int(
        work.behavior_probability_evaluations
        + work.cache_validation_behavior_probability_evaluations
    )
    assert counts["grounded_predict"] == int(
        work.grounded_joint_cell_evaluations
        + work.cache_validation_grounded_joint_cell_evaluations
    )
