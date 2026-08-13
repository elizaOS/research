# mypy: disable-error-code="attr-defined,call-arg"
"""L0 contracts for proposal-only ensemble short rollouts."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.ensemble_short_rollouts import (
    ENSEMBLE_SHORT_ROLLOUT_CONFIG_SCHEMA,
    ENSEMBLE_SHORT_ROLLOUT_EVIDENCE_LEVEL,
    ENSEMBLE_SHORT_ROLLOUT_MECHANISM_STATUS,
    ENSEMBLE_SHORT_ROLLOUT_SCIENTIFIC_PROMOTION_ALLOWED,
    EnsembleShortRolloutConfig,
    EnsembleShortRolloutPlanner,
    EnsembleShortRolloutState,
    RealStateRolloutAnchor,
    RolloutPolicyValueAuthority,
    load_ensemble_short_rollout_checkpoint,
    save_ensemble_short_rollout_checkpoint,
)
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleConfig,
    WorldModelEnsembleState,
)

pytestmark = pytest.mark.unit

OBSERVATION = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
REVISION_ONE = jnp.asarray([0, 1], dtype=jnp.uint32)


@pytest.fixture(autouse=True)
def _run_contract_tests_eagerly() -> Iterator[None]:
    """Leave explicit compilation to the integration parity tests."""

    with jax.disable_jit():
        yield
    jax.clear_caches()  # type: ignore[no-untyped-call]


def _ensemble() -> WorldModelEnsemble:
    model = ActionConditionedWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        gamma=0.95,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
        use_layer_norm=False,
        error_decay=0.8,
    )
    signals = LearningSignalEstimatorConfig(
        ensemble_size=2,
        target_dim=4,
        progress_warmup_steps=2,
        change_calibration_steps=2,
        max_input_magnitude=1_000.0,
        max_predicted_variance=10_000.0,
        max_observed_loss=10_000.0,
    )
    return WorldModelEnsemble(
        WorldModelEnsembleConfig(
            model=model,
            signal_estimator=signals,
            ensemble_size=2,
            bootstrap_probability=0.5,
            residual_variance_warmup_steps=1,
            residual_variance_floor=1.0e-6,
        )
    )


def _set_action_outputs(
    ensemble: WorldModelEnsemble,
    state: WorldModelEnsembleState,
    outputs: tuple[
        tuple[tuple[float, float, float, float], tuple[float, float, float, float]],
        ...,
    ],
) -> WorldModelEnsembleState:
    """Install exact action-conditioned linear outputs without clock changes."""

    assert len(outputs) == ensemble.config.ensemble_size
    members = []
    for member, per_action in zip(state.member_states, outputs, strict=True):
        learner = member.learner_state
        weights = []
        biases = []
        for head_index in range(4):
            weight = jnp.zeros_like(learner.head_params.weights[head_index])
            weight = weight.at[0, 2].set(per_action[0][head_index])
            weight = weight.at[0, 3].set(per_action[1][head_index])
            weights.append(weight)
            biases.append(jnp.zeros((1,), dtype=jnp.float32))
        heads = learner.head_params.replace(
            weights=tuple(weights),
            biases=tuple(biases),
        )
        members.append(member.replace(learner_state=learner.replace(head_params=heads)))
    result = cast(WorldModelEnsembleState, state.replace(member_states=tuple(members)))
    assert bool(ensemble.state_valid(result))
    return result


def _constant_outputs(
    *,
    reward: float = 1.0,
    continuation: float = 0.5,
) -> tuple[
    tuple[tuple[float, float, float, float], tuple[float, float, float, float]],
    ...,
]:
    value = (0.0, 0.0, reward, continuation)
    return ((value, value), (value, value))


def _config(**overrides: object) -> EnsembleShortRolloutConfig:
    defaults: dict[str, object] = {
        "rollout_horizon": 3,
        "rollout_budget": 2,
        "require_residual_proxy_ready": False,
        "max_epistemic_disagreement": 100.0,
        "max_residual_variance": 100.0,
        "max_proposal_calls": 10,
        "max_rollout_attempts": 20,
        "max_imagined_steps": 60,
    }
    defaults.update(overrides)
    return EnsembleShortRolloutConfig(**defaults)  # type: ignore[arg-type]


def _authority(
    planner: EnsembleShortRolloutPlanner,
    model_state: WorldModelEnsembleState,
    *,
    revision: int = 1,
    policy_bias: tuple[float, float] = (20.0, -20.0),
    value_bias: float = 8.0,
    support: tuple[int, int] = (10, 10),
) -> RolloutPolicyValueAuthority:
    words = jnp.asarray([0, revision], dtype=jnp.uint32)
    return planner.bind_authority(
        policy_weights=jnp.zeros((2, 2), dtype=jnp.float32),
        policy_bias=jnp.asarray(policy_bias, dtype=jnp.float32),
        value_weights=jnp.zeros((2,), dtype=jnp.float32),
        value_bias=jnp.asarray(value_bias, dtype=jnp.float32),
        action_support_counts=jnp.asarray(support, dtype=jnp.int32),
        source_revision_words=words,
        model_state=model_state,
        policy_revision_words=words,
        value_revision_words=words,
    )


def _system(
    *,
    config: EnsembleShortRolloutConfig | None = None,
    outputs: tuple[
        tuple[tuple[float, float, float, float], tuple[float, float, float, float]],
        ...,
    ]
    | None = None,
) -> tuple[
    EnsembleShortRolloutPlanner,
    WorldModelEnsembleState,
    RolloutPolicyValueAuthority,
    EnsembleShortRolloutState,
    RealStateRolloutAnchor,
]:
    ensemble = _ensemble()
    model_state = _set_action_outputs(
        ensemble,
        ensemble.init(jr.key(1, impl="threefry2x32")),
        outputs or _constant_outputs(),
    )
    planner = EnsembleShortRolloutPlanner(ensemble, config or _config())
    authority = _authority(planner, model_state)
    state = planner.init(
        jr.key(2, impl="threefry2x32"),
        model_state,
        authority,
    )
    anchor = planner.bind_real_anchor(
        OBSERVATION,
        jnp.asarray([0, 1], dtype=jnp.uint32),
        authority,
    )
    return planner, model_state, authority, state, anchor


def _materialize_keys(tree: object) -> object:
    def materialize(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(materialize, tree)


def _assert_tree_equal(left: object, right: object) -> None:
    chex.assert_trees_all_equal(_materialize_keys(left), _materialize_keys(right))


def test_config_construction_and_resource_scope_are_strict_l0_contracts() -> None:
    config = _config(selection_mode="uncertainty_directed", rollout_budget=3)
    payload = config.to_config()
    assert payload["schema"] == ENSEMBLE_SHORT_ROLLOUT_CONFIG_SCHEMA
    assert payload["mechanism_status"] == ENSEMBLE_SHORT_ROLLOUT_MECHANISM_STATUS
    assert payload["evidence_level"] == ENSEMBLE_SHORT_ROLLOUT_EVIDENCE_LEVEL == "L0"
    assert payload["control_benefit_assessed"] is False
    assert payload["scientific_promotion_allowed"] is False
    assert ENSEMBLE_SHORT_ROLLOUT_SCIENTIFIC_PROMOTION_ALLOWED is False
    assert EnsembleShortRolloutConfig.from_config(payload) == config
    with pytest.raises(ValueError, match="fields"):
        EnsembleShortRolloutConfig.from_config({**payload, "extra": True})
    with pytest.raises(ValueError, match="selection_mode"):
        _config(selection_mode="random")
    with pytest.raises(ValueError, match="max_imagined_steps"):
        _config(max_imagined_steps=2)

    planner, model_state, authority, state, _ = _system(config=config)
    restored = EnsembleShortRolloutPlanner.from_config(planner.to_config())
    assert restored.to_config() == planner.to_config()
    budget = planner.resource_budget
    state_bytes = sum(
        np.asarray(leaf).nbytes for leaf in jax.tree.leaves(_materialize_keys(state))
    )
    assert state_bytes == budget.persistent_state_bytes
    assert budget.persistent_bytes_scope.startswith("planner-owned-array-leaves-only")
    assert budget.proposal_bytes_scope.endswith("not-dispatched")
    assert budget.temporary_bytes_scope.startswith("not-measured")
    assert budget.max_ensemble_prediction_calls_per_call == 18
    assert budget.max_member_predictions_per_call == 36
    assert budget.max_rng_draws_per_call == 0
    assert budget.model_state_owned == 0
    assert budget.policy_value_state_owned == 0
    assert budget.actor_or_critic_updates_per_call == 0
    assert budget.dispatch_authority == 0
    with pytest.raises(TypeError, match="typed scalar threefry"):
        planner.init(jr.key_data(jr.key(2)), model_state, authority)


def test_analytical_multistep_return_uses_horizon_bootstrap_and_keeps_owners_read_only() -> None:
    planner, model_state, authority, state, anchor = _system()
    model_before = _materialize_keys(model_state)
    authority_before = _materialize_keys(authority)

    result = planner.propose(state, model_state, authority, anchor)

    assert bool(result.diagnostics.transaction_applied)
    assert bool(jnp.all(result.proposals.path_accepted))
    assert bool(jnp.all(result.proposals.transition_valid))
    assert result.proposals.return_targets[0].tolist() == pytest.approx(
        [2.75, 3.5, 5.0]
    )
    assert float(result.proposals.bootstrap_values[0]) == pytest.approx(8.0)
    assert float(result.proposals.root_returns[0]) == pytest.approx(2.75)
    assert int(result.state.proposal_call_count_words[1]) == 1
    assert int(result.state.rollout_attempt_count_words[1]) == 2
    assert int(result.state.accepted_rollout_count_words[1]) == 2
    assert int(result.state.imagined_step_count_words[1]) == 6
    assert bool(planner.state_valid(result.state))
    _assert_tree_equal(model_state, model_before)
    _assert_tree_equal(authority, authority_before)
    chex.assert_trees_all_equal(
        result.proposals.source_revision_words[0],
        anchor.source_revision_words,
    )
    chex.assert_trees_all_equal(
        result.proposals.model_revision_words[0],
        model_state.event_count_words,
    )
    assert int(result.proposals.anchor_integrity_tags[0]) != 0


def test_learned_terminal_continuation_stops_padding_and_forbids_bootstrap() -> None:
    planner, model_state, authority, state, anchor = _system(
        config=_config(max_abs_value=10.0),
        outputs=_constant_outputs(reward=3.0, continuation=0.0)
    )
    terminal_authority = _authority(planner, model_state, value_bias=999.0)
    state = planner.init(
        jr.key(2, impl="threefry2x32"),
        model_state,
        terminal_authority,
    )
    anchor = planner.bind_real_anchor(
        OBSERVATION,
        jnp.asarray([0, 1], dtype=jnp.uint32),
        terminal_authority,
    )

    result = planner.propose(state, model_state, terminal_authority, anchor)

    assert result.proposals.transition_valid[0].tolist() == [True, False, False]
    assert result.proposals.terminated[0].tolist() == [True, False, False]
    assert result.proposals.continuations[0].tolist() == [0.0, 0.0, 0.0]
    assert float(result.proposals.bootstrap_values[0]) == 0.0
    assert float(result.proposals.root_returns[0]) == pytest.approx(3.0)
    assert float(result.proposals.return_targets[0, 0]) == pytest.approx(3.0)
    # The out-of-bound successor value (999 > configured 10) is deliberately
    # irrelevant on a learned terminal transition: terminal return acceptance
    # depends only on reward and continuation zero, never on a bootstrap.
    assert bool(result.proposals.path_accepted[0])
    assert not bool(result.diagnostics.path_failed[0])
    assert int(result.state.imagined_step_count_words[1]) == 2


def test_member_termination_disagreement_rejects_complete_paths() -> None:
    action0 = (0.0, 0.0, 1.0, 0.0)
    action1 = (0.0, 0.0, 1.0, 0.0)
    continuing0 = (0.0, 0.0, 1.0, 0.5)
    continuing1 = (0.0, 0.0, 1.0, 0.5)
    planner, model_state, authority, state, anchor = _system(
        outputs=((action0, action1), (continuing0, continuing1))
    )

    result = planner.propose(state, model_state, authority, anchor)

    assert bool(result.diagnostics.transaction_applied)
    assert not bool(result.diagnostics.termination_agreement[0, 0])
    assert bool(result.diagnostics.path_failed[0])
    assert not bool(jnp.any(result.proposals.transition_valid))
    assert int(result.state.accepted_rollout_count_words[1]) == 0
    assert int(result.state.rejected_rollout_count_words[1]) == 2


def test_uncertainty_directed_selection_chooses_highest_admitted_disagreement() -> None:
    action0 = (0.0, 0.0, 1.0, 0.5)
    member0_action1 = (0.0, 0.0, 1.0, 0.5)
    member1_action1 = (0.0, 0.0, 5.0, 0.5)
    config = _config(selection_mode="uncertainty_directed")
    planner, model_state, _, _, _ = _system(
        config=config,
        outputs=((action0, member0_action1), (action0, member1_action1)),
    )
    authority = _authority(planner, model_state, policy_bias=(0.0, 0.0))
    state = planner.init(jr.key(9, impl="threefry2x32"), model_state, authority)
    anchor = planner.bind_real_anchor(
        OBSERVATION,
        jnp.asarray([0, 1], dtype=jnp.uint32),
        authority,
    )
    selected = planner.propose(state, model_state, authority, anchor)
    assert bool(jnp.all(selected.diagnostics.selected_actions == 1))
    assert bool(
        jnp.all(selected.diagnostics.epistemic_disagreements[:, 0] > 0.0)
    )

    masked_authority = _authority(
        planner,
        model_state,
        policy_bias=(0.0, 0.0),
        support=(10, 0),
    )
    masked_state = planner.init(
        jr.key(9, impl="threefry2x32"),
        model_state,
        masked_authority,
    )
    masked_anchor = planner.bind_real_anchor(
        OBSERVATION,
        jnp.asarray([0, 1], dtype=jnp.uint32),
        masked_authority,
    )
    masked = planner.propose(
        masked_state,
        model_state,
        masked_authority,
        masked_anchor,
    )
    assert bool(jnp.all(masked.diagnostics.selected_actions == 0))
    assert bool(jnp.all(masked.proposals.path_accepted))


def test_support_readiness_epistemic_and_residual_gates_are_independent() -> None:
    planner, model_state, _, _, _ = _system()
    unsupported_authority = _authority(planner, model_state, support=(0, 0))
    unsupported_state = planner.init(
        jr.key(2, impl="threefry2x32"),
        model_state,
        unsupported_authority,
    )
    unsupported_anchor = planner.bind_real_anchor(
        OBSERVATION,
        jnp.asarray([0, 1], dtype=jnp.uint32),
        unsupported_authority,
    )
    unsupported = planner.propose(
        unsupported_state,
        model_state,
        unsupported_authority,
        unsupported_anchor,
    )
    assert not bool(unsupported.diagnostics.support_valid[0, 0])
    assert not bool(jnp.any(unsupported.proposals.path_accepted))

    ready_config = _config(require_residual_proxy_ready=True)
    ready_planner, ready_model, ready_authority, ready_state, ready_anchor = _system(
        config=ready_config
    )
    cold = ready_planner.propose(
        ready_state,
        ready_model,
        ready_authority,
        ready_anchor,
    )
    assert not bool(cold.diagnostics.residual_proxy_ready[0, 0])
    assert not bool(jnp.any(cold.proposals.path_accepted))

    high = (0.0, 0.0, 10.0, 0.5)
    low = (0.0, 0.0, -10.0, 0.5)
    epistemic_planner, epistemic_model, epistemic_authority, epistemic_state, anchor = (
        _system(
            config=_config(max_epistemic_disagreement=0.01),
            outputs=((high, high), (low, low)),
        )
    )
    epistemic = epistemic_planner.propose(
        epistemic_state,
        epistemic_model,
        epistemic_authority,
        anchor,
    )
    assert not bool(epistemic.diagnostics.epistemic_valid[0, 0])

    residual_state = model_state.replace(
        residual_variances=jnp.full_like(model_state.residual_variances, 2.0)
    )
    assert bool(planner.ensemble.state_valid(residual_state))
    residual_authority = _authority(planner, residual_state)
    residual_planner = EnsembleShortRolloutPlanner(
        planner.ensemble,
        _config(max_residual_variance=1.0),
    )
    residual_authority = _authority(residual_planner, residual_state)
    residual_lane = residual_planner.init(
        jr.key(2, impl="threefry2x32"),
        residual_state,
        residual_authority,
    )
    residual_anchor = residual_planner.bind_real_anchor(
        OBSERVATION,
        jnp.asarray([0, 1], dtype=jnp.uint32),
        residual_authority,
    )
    residual = residual_planner.propose(
        residual_lane,
        residual_state,
        residual_authority,
        residual_anchor,
    )
    assert not bool(residual.diagnostics.residual_variance_valid[0, 0])


def test_stale_revisions_tag_aliases_duplicate_decisions_and_tamper_are_atomic_noops() -> None:
    planner, model_state, authority, state, anchor = _system()
    first = planner.propose(state, model_state, authority, anchor)
    duplicate = planner.propose(first.state, model_state, authority, anchor)
    assert not bool(duplicate.diagnostics.decision_identity_valid)
    assert not bool(duplicate.diagnostics.transaction_applied)
    _assert_tree_equal(duplicate.state, first.state)

    tampered = anchor.replace(observation=anchor.observation.at[0].set(2.0))
    blocked_tamper = planner.propose(state, model_state, authority, tampered)
    assert not bool(blocked_tamper.diagnostics.anchor_identity_valid)
    _assert_tree_equal(blocked_tamper.state, state)

    aliased = _authority(
        planner,
        model_state,
        revision=1,
        policy_bias=(-20.0, 20.0),
    )
    aliased_anchor = planner.bind_real_anchor(
        OBSERVATION,
        jnp.asarray([0, 2], dtype=jnp.uint32),
        aliased,
    )
    alias_result = planner.propose(first.state, model_state, aliased, aliased_anchor)
    assert not bool(alias_result.diagnostics.revisions_monotonic)
    _assert_tree_equal(alias_result.state, first.state)

    advanced = _authority(planner, model_state, revision=2)
    advanced_anchor = planner.bind_real_anchor(
        OBSERVATION,
        jnp.asarray([0, 2], dtype=jnp.uint32),
        advanced,
    )
    second = planner.propose(first.state, model_state, advanced, advanced_anchor)
    assert bool(second.diagnostics.transaction_applied)
    stale_anchor = planner.bind_real_anchor(
        OBSERVATION,
        jnp.asarray([0, 3], dtype=jnp.uint32),
        authority,
    )
    stale = planner.propose(second.state, model_state, authority, stale_anchor)
    assert not bool(stale.diagnostics.revisions_monotonic)
    assert not bool(stale.diagnostics.transaction_applied)
    _assert_tree_equal(stale.state, second.state)


def test_same_revision_alternate_model_content_is_an_atomic_alias_rejection() -> None:
    planner, model_state, authority, state, anchor = _system()
    first = planner.propose(state, model_state, authority, anchor)
    member = model_state.member_states[0]
    learner = member.learner_state
    changed_heads = learner.head_params.replace(
        biases=(
            learner.head_params.biases[0].at[0].add(0.25),
            *learner.head_params.biases[1:],
        )
    )
    changed_member = member.replace(
        learner_state=learner.replace(head_params=changed_heads)
    )
    aliased_model = cast(
        WorldModelEnsembleState,
        model_state.replace(
            member_states=(changed_member, *model_state.member_states[1:])
        ),
    )
    assert bool(planner.ensemble.state_valid(aliased_model))
    assert jnp.array_equal(
        aliased_model.event_count_words,
        model_state.event_count_words,
    )
    aliased_authority = _authority(planner, aliased_model)
    assert int(aliased_authority.model_integrity_tag) != int(
        authority.model_integrity_tag
    )
    aliased_anchor = planner.bind_real_anchor(
        OBSERVATION,
        jnp.asarray([0, 2], dtype=jnp.uint32),
        aliased_authority,
    )

    result = planner.propose(
        first.state,
        aliased_model,
        aliased_authority,
        aliased_anchor,
    )

    assert bool(result.diagnostics.authority_valid)
    assert not bool(result.diagnostics.revisions_monotonic)
    assert not bool(result.diagnostics.transaction_applied)
    _assert_tree_equal(result.state, first.state)


def test_invalid_model_corrupt_clock_and_capacity_exhaustion_preserve_rng() -> None:
    config = _config(
        rollout_budget=1,
        max_proposal_calls=1,
        max_rollout_attempts=1,
        max_imagined_steps=3,
    )
    planner, model_state, authority, state, anchor = _system(config=config)
    invalid_model = model_state.replace(
        residual_variances=model_state.residual_variances.at[0, 0].set(jnp.nan)
    )
    invalid = planner.propose(state, invalid_model, authority, anchor)
    assert not bool(invalid.diagnostics.model_state_valid)
    _assert_tree_equal(invalid.state, state)

    corrupt = state.replace(
        accepted_rollout_count_words=jnp.asarray([0, 1], dtype=jnp.uint32)
    )
    corrupt_result = planner.propose(corrupt, model_state, authority, anchor)
    assert not bool(corrupt_result.diagnostics.state_valid)
    _assert_tree_equal(corrupt_result.state, corrupt)

    first = planner.propose(state, model_state, authority, anchor)
    second_anchor = planner.bind_real_anchor(
        OBSERVATION,
        jnp.asarray([0, 2], dtype=jnp.uint32),
        authority,
    )
    exhausted = planner.propose(first.state, model_state, authority, second_anchor)
    assert not bool(exhausted.diagnostics.call_capacity_available)
    assert not bool(exhausted.diagnostics.transaction_applied)
    _assert_tree_equal(exhausted.state, first.state)


def test_rollout_rng_is_deterministic_and_model_rng_is_read_only() -> None:
    config = _config(policy_temperature=2.0)
    planner, model_state, _, _, _ = _system(config=config)
    authority = _authority(planner, model_state, policy_bias=(0.0, 0.0))
    state = planner.init(jr.key(17, impl="threefry2x32"), model_state, authority)
    anchor = planner.bind_real_anchor(
        OBSERVATION,
        jnp.asarray([0, 1], dtype=jnp.uint32),
        authority,
    )
    model_before = _materialize_keys(model_state)
    authority_before = _materialize_keys(authority)
    first = planner.propose(state, model_state, authority, anchor)
    repeated = planner.propose(state, model_state, authority, anchor)
    _assert_tree_equal(first.proposals, repeated.proposals)
    _assert_tree_equal(model_state, model_before)
    _assert_tree_equal(authority, authority_before)
    assert not bool(
        jnp.array_equal(
            jr.key_data(first.state.rollout_key),
            jr.key_data(state.rollout_key),
        )
    )


def test_checkpoint_round_trip_preserves_next_proposal_exactly(tmp_path: Path) -> None:
    planner, model_state, authority, state, anchor = _system()
    first = planner.propose(state, model_state, authority, anchor)
    path = tmp_path / "rollout-lane"
    save_ensemble_short_rollout_checkpoint(planner, first.state, path)
    restored_planner, restored_state = load_ensemble_short_rollout_checkpoint(path)
    assert restored_planner.to_config() == planner.to_config()
    _assert_tree_equal(restored_state, first.state)

    next_anchor = planner.bind_real_anchor(
        OBSERVATION,
        jnp.asarray([0, 2], dtype=jnp.uint32),
        authority,
    )
    expected = planner.propose(first.state, model_state, authority, next_anchor)
    restored_anchor = restored_planner.bind_real_anchor(
        OBSERVATION,
        jnp.asarray([0, 2], dtype=jnp.uint32),
        authority,
    )
    actual = restored_planner.propose(
        restored_state,
        model_state,
        authority,
        restored_anchor,
    )
    _assert_tree_equal(actual, expected)
