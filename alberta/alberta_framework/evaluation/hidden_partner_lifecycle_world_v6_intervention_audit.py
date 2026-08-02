# mypy: disable-error-code="attr-defined,call-arg,no-any-return,operator"
"""Per-step intervention audits for the development-only v6 study.

This module has no execution or evidence authority.  It replays source-bound
bounded components and independently recomputes their intervention gates,
routing, RNG, and control algebra, then returns two fixed-width boolean
vectors.  Component replay is a proposal/persistence audit, not an independent
implementation oracle for the component learner itself.  A false check is
always a structural failure.  Witnesses are deliberately separate: they
establish that a disabled path had a non-zero, behaviorally relevant
opportunity and therefore prevent vacuous matched-control conclusions.

The caller supplies the pre-step bridge state and the already-computed bridge
result.  No persistent audit state, replay, host conversion, or second bridge
step is introduced.  ``control`` and ``agent`` are static Python objects when
this function is used under :func:`jax.jit` or :func:`jax.lax.scan`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.core import Tracer
from jaxtyping import Bool

from alberta_framework.core.grounded_joint_world_model import GroundedJointWorldUpdateResult
from alberta_framework.core.integrated_hidden_partner import (
    ACTIVE_PAIR_SLOTS,
    BASE_FEATURE_DIM,
    CANDIDATE_PAIR_SLOTS,
    HIDDEN_STATE_DIM,
    N_ACTIONS,
    RAW_OBSERVATION_DIM,
    IntegratedGroundedPlannerEvaluation,
    IntegratedHiddenPartnerAgent,
)
from alberta_framework.core.interaction_features import (
    InteractionCurationPriorityOverride,
    InteractionFeatureState,
    InteractionFeatureUpdateResult,
)
from alberta_framework.core.representation_gradient_mixer import (
    RepresentationGradientMixResult,
    mix_representation_gradients,
)
from alberta_framework.core.state_builder import OnlineGatedStateBuilderState
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6 import (
    PRIMARY_CONDITION_ORDER,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_controls import (
    V6_DIAGNOSTIC_ORDER,
    HiddenPartnerLifecycleWorldV6Control,
    build_v6_full_agent_config,
    validate_v6_control,
)
from alberta_framework.evaluation.hidden_partner_world_online_bridge import (
    HiddenPartnerWorldMechanismTrace,
    HiddenPartnerWorldOnlineState,
    HiddenPartnerWorldOnlineStep,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    CUE_1_INDEX,
    CUE_2_INDEX,
    HiddenPartnerWorldFeedbackConfig,
)

V6_INTERVENTION_AUDIT_ORDER: tuple[str, ...] = (
    "behavior_credit_replay",
    "grounded_credit_replay",
    "gradient_mix_mode_bounded_replay",
    "gradient_chain_bounded_replay",
    "state_learning_gate_bounded_replay",
    "grounded_learning_gate_exact",
    "memory_mask_exact",
    "planner_reward_source_exact",
    "planning_application_exact",
    "partner_belief_exact",
    "lifecycle_commit_gate_exact",
    "identity_carry_mode_exact",
    "retention_floor_exact",
    "retirement_gate_exact",
    "random_curation_exact",
    "uniform_action_exact",
    "cue_sampling_exact",
    "row_bias_exact",
)

V6_INTERVENTION_WITNESS_ORDER: tuple[str, ...] = (
    "behavior_credit_nonzero",
    "grounded_credit_nonzero",
    "state_parameter_proposal_nonzero",
    "grounded_parameter_proposal_nonzero",
    "lifecycle_proposal_event",
    "applied_descriptor_change",
    "retention_floor_counterfactual_bind",
    "retirement_eligible",
    "random_selection_differs_from_utility_selection",
    "masked_hidden_state_downstream_learning_effect",
    "table_and_grounded_rewards_disagree",
    "planner_model_term_nonzero",
    "partner_prediction_nonuniform",
    "forced_action_differs_from_ordinary",
    "equal_cue_differs_from_base_counterfactual",
    "row_bias_proposal_nonzero",
)

if len(V6_INTERVENTION_AUDIT_ORDER) != 18:
    raise RuntimeError("v6 intervention audit width must remain exactly 18")
if len(V6_INTERVENTION_WITNESS_ORDER) != 16:
    raise RuntimeError("v6 intervention witness width must remain exactly 16")
if len(set(V6_INTERVENTION_AUDIT_ORDER)) != len(V6_INTERVENTION_AUDIT_ORDER):
    raise RuntimeError("v6 intervention audit names must be unique")
if len(set(V6_INTERVENTION_WITNESS_ORDER)) != len(V6_INTERVENTION_WITNESS_ORDER):
    raise RuntimeError("v6 intervention witness names must be unique")


# Every named requirement must receive positive support over a complete run.
# Empty tuples are intentional: those arms still require all 18 structural checks,
# but have no arm-specific non-vacuity obligation beyond the matched suite.
V6_CONTROL_REQUIRED_WITNESSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("full", ()),
    ("grounded_model_frozen", ("grounded_parameter_proposal_nonzero",)),
    (
        "world_credit_off",
        ("behavior_credit_nonzero", "grounded_credit_nonzero"),
    ),
    (
        "behavior_credit_off",
        ("behavior_credit_nonzero", "grounded_credit_nonzero"),
    ),
    (
        "all_representation_credit_off",
        ("behavior_credit_nonzero", "grounded_credit_nonzero"),
    ),
    ("state_frozen", ("state_parameter_proposal_nonzero",)),
    (
        "recurrent_memory_masked",
        ("masked_hidden_state_downstream_learning_effect",),
    ),
    ("table_planner", ("table_and_grounded_rewards_disagree",)),
    ("no_planning", ("planner_model_term_nonzero",)),
    ("uniform_partner", ("partner_prediction_nonuniform",)),
    ("lifecycle_frozen", ("lifecycle_proposal_event",)),
    ("no_identity_carry", ("applied_descriptor_change",)),
    ("no_retention_floor", ("retention_floor_counterfactual_bind",)),
    ("retirement_disabled", ("retirement_eligible",)),
    (
        "random_curation",
        ("random_selection_differs_from_utility_selection",),
    ),
    ("uniform_action", ("forced_action_differs_from_ordinary",)),
    (
        "equal_cue",
        ("equal_cue_differs_from_base_counterfactual",),
    ),
    ("row_bias", ("row_bias_proposal_nonzero",)),
)

if tuple(name for name, _ in V6_CONTROL_REQUIRED_WITNESSES) != (
    *PRIMARY_CONDITION_ORDER,
    *V6_DIAGNOSTIC_ORDER,
):
    raise RuntimeError("v6 intervention witness mapping differs from control order")
if any(
    witness not in V6_INTERVENTION_WITNESS_ORDER
    for _, witnesses in V6_CONTROL_REQUIRED_WITNESSES
    for witness in witnesses
):
    raise RuntimeError("v6 control mapping names an unknown intervention witness")


# The audit API receives the pre/post persistent states in addition to the
# compact transient trace.  At this contract version no predicate needs a new
# compact field.  Keeping this exported tuple explicit makes future loss of a
# required pre/post or trace field fail review rather than turn into a silent
# approximation.
V6_INTERVENTION_COMPACT_FIELD_BLOCKERS: tuple[tuple[str, str], ...] = ()

_INT32_MAX = 2**31 - 1
V6_FLOAT32_REPLAY_RTOL = 2.0**-20
V6_FLOAT32_REPLAY_ATOL = 2.0**-22
_FULL_RETENTION_DECAY = build_v6_full_agent_config().active_utility_retention_decay
if _FULL_RETENTION_DECAY is None:
    raise RuntimeError("v6 full control must retain an explicit utility floor")


@chex.dataclass(frozen=True)
class V6InterventionStepAudit:
    """One accepted transition's fixed-width checks and non-vacuity witnesses."""

    checks: Bool[Array, " 18"]
    witnesses: Bool[Array, " 16"]


def _exact(left: Array | object, right: Array | object) -> Array:
    """Return scalar bit equality for one same-shaped JAX leaf."""

    lhs = jnp.asarray(left)
    rhs = jnp.asarray(right)
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        return jnp.asarray(False, dtype=jnp.bool_)
    if jax.dtypes.issubdtype(lhs.dtype, jax.dtypes.prng_key):
        return jnp.array_equal(jr.key_data(lhs), jr.key_data(rhs))
    if jnp.issubdtype(lhs.dtype, jnp.floating) and lhs.dtype != jnp.float32:
        raise TypeError("v6 exact audit supports only float32 floating leaves")
    if lhs.dtype == jnp.float32:
        lhs = jax.lax.bitcast_convert_type(lhs, jnp.uint32)
        rhs = jax.lax.bitcast_convert_type(rhs, jnp.uint32)
    return jnp.array_equal(lhs, rhs)


def _replay_equal(left: Array | object, right: Array | object) -> Array:
    """Compare replayed float32 arithmetic tightly and all other leaves exactly.

    A separately staged component replay can differ from the bridge-fused XLA
    program by one float32 ULP.  This tolerance is limited to numerical replay
    leaves; static mode, masks, counters, identities, RNG, and persistence
    choices continue to use :func:`_exact`.
    """

    lhs = jnp.asarray(left)
    rhs = jnp.asarray(right)
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        return jnp.asarray(False, dtype=jnp.bool_)
    if lhs.dtype != jnp.float32:
        return _exact(lhs, rhs)
    return jnp.all(
        jnp.isclose(
            lhs,
            rhs,
            rtol=jnp.float32(V6_FLOAT32_REPLAY_RTOL),
            atol=jnp.float32(V6_FLOAT32_REPLAY_ATOL),
        )
    )


def _tree_exact(left: object, right: object) -> Array:
    """Return scalar bit equality for two equal-structure array PyTrees."""

    left_structure = jax.tree_util.tree_structure(left)
    right_structure = jax.tree_util.tree_structure(right)
    if left_structure != right_structure:
        return jnp.asarray(False, dtype=jnp.bool_)
    comparisons = tuple(
        _exact(lhs, rhs)
        for lhs, rhs in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
            strict=True,
        )
    )
    if not comparisons:
        return jnp.asarray(True, dtype=jnp.bool_)
    return jnp.all(jnp.stack(comparisons))


def _tree_replay_equal(left: object, right: object) -> Array:
    """Return tight replay equality for equal-structure array PyTrees."""

    left_structure = jax.tree_util.tree_structure(left)
    right_structure = jax.tree_util.tree_structure(right)
    if left_structure != right_structure:
        return jnp.asarray(False, dtype=jnp.bool_)
    comparisons = tuple(
        _replay_equal(lhs, rhs)
        for lhs, rhs in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
            strict=True,
        )
    )
    if not comparisons:
        return jnp.asarray(True, dtype=jnp.bool_)
    return jnp.all(jnp.stack(comparisons))


def _state_learning_persistence_exact(
    *,
    enabled: bool,
    pre_state: OnlineGatedStateBuilderState,
    post_state: OnlineGatedStateBuilderState,
    replay_state: OnlineGatedStateBuilderState,
) -> Array:
    """Bind the replay and make the disabled parameter gate bit-exact."""

    replay_matches = _tree_replay_equal(post_state, replay_state)
    if enabled:
        return replay_matches
    # OnlineGatedStateBuilderState is kept structural here so the helper stays
    # cheap to exercise with adversarial unit states.
    return (
        replay_matches
        & _exact(post_state.parameters, pre_state.parameters)
        & _exact(post_state.update_count, pre_state.update_count)
        & _exact(
            post_state.last_gradient_norm,
            pre_state.last_gradient_norm,
        )
    )


def _lifecycle_persistence_exact(
    *,
    enabled: bool,
    pre_state: InteractionFeatureState,
    post_state: InteractionFeatureState,
    replay_state: InteractionFeatureState,
) -> Array:
    """Bind ordinary learning and bit-exactly suppress lifecycle-owned mutation."""

    replay_matches = _tree_exact(post_state, replay_state)
    if enabled:
        return replay_matches
    for field in (
        "feature_left",
        "feature_right",
        "candidate_left",
        "candidate_right",
        "feature_parent_a",
        "feature_parent_b",
        "feature_generator",
        "candidate_parent_a",
        "candidate_parent_b",
        "candidate_generator",
    ):
        replay_matches &= _exact(getattr(post_state, field), getattr(pre_state, field))
    return replay_matches


def _row_update_isolation_exact(
    weight_change_mask: Array,
    bias_change_mask: Array,
    joint_index: Array,
) -> Array:
    """Require that no dynamically unexecuted joint-action row can change."""

    executed = jnp.arange(N_ACTIONS * N_ACTIONS, dtype=jnp.int32) == joint_index
    return ~jnp.any(jnp.asarray(weight_change_mask, dtype=jnp.bool_) & ~executed) & ~jnp.any(
        jnp.asarray(bias_change_mask, dtype=jnp.bool_) & ~executed
    )


def _nonzero(value: Array) -> Array:
    raw = jnp.asarray(value, dtype=jnp.float32)
    return jnp.all(jnp.isfinite(raw)) & jnp.any(raw != jnp.float32(0.0))


def _unmasked_chi_counterfactual(
    phi: Array,
    descriptors: Array,
    consumer_active_mask: Array,
) -> Array:
    """Build the same deployed vector with learned memory left unmasked."""

    base = jnp.asarray(phi, dtype=jnp.float32)
    pairs = jnp.asarray(descriptors, dtype=jnp.int32)
    consumer_mask = jnp.asarray(consumer_active_mask, dtype=jnp.bool_)
    live = _live_descriptors(pairs)
    safe_left = jnp.where(live, pairs[:, 0], 0)
    safe_right = jnp.where(live, pairs[:, 1], 0)
    products = (
        base[safe_left]
        * base[safe_right]
        * live.astype(jnp.float32)
        * consumer_mask.astype(jnp.float32)
    )
    return jnp.concatenate((base, products))


def _masked_memory_downstream_learning_effect(
    hidden_state: Array,
    pre_hidden_weights: Array,
    masked_post_hidden_weights: Array,
    unmasked_post_hidden_weights: Array,
) -> Array:
    """Require learned memory to change a finite downstream update proposal."""

    masked_delta = jnp.asarray(masked_post_hidden_weights, dtype=jnp.float32) - jnp.asarray(
        pre_hidden_weights,
        dtype=jnp.float32,
    )
    unmasked_delta = jnp.asarray(unmasked_post_hidden_weights, dtype=jnp.float32) - jnp.asarray(
        pre_hidden_weights,
        dtype=jnp.float32,
    )
    return _nonzero(hidden_state) & _nonzero(unmasked_delta - masked_delta)


def _live_descriptors(descriptors: Array) -> Array:
    pairs = jnp.asarray(descriptors, dtype=jnp.int32)
    return (
        (pairs[:, 0] >= 0)
        & (pairs[:, 1] >= 0)
        & (pairs[:, 0] < BASE_FEATURE_DIM)
        & (pairs[:, 1] < BASE_FEATURE_DIM)
        & (pairs[:, 0] < pairs[:, 1])
    )


def _expected_route_identity(
    old_descriptors: Array,
    new_descriptors: Array,
) -> tuple[Array, Array, Array, Array]:
    old = jnp.asarray(old_descriptors, dtype=jnp.int32)
    new = jnp.asarray(new_descriptors, dtype=jnp.int32)
    old_live = _live_descriptors(old)
    new_live = _live_descriptors(new)
    matches = jnp.all(new[:, None, :] == old[None, :, :], axis=-1)
    matches &= new_live[:, None] & old_live[None, :]
    survivor = new_live & jnp.any(matches, axis=1)
    source = jnp.where(
        survivor,
        jnp.argmax(matches, axis=1).astype(jnp.int32),
        jnp.asarray(-1, dtype=jnp.int32),
    )
    new_mask = new_live & ~survivor
    evicted = old_live & ~jnp.any(matches, axis=0)
    return source, survivor, new_mask, evicted


def _route_last_axis(
    values: Array,
    source_slots: Array,
    survivor_mask: Array,
    *,
    carry_survivors: bool,
    descriptors_changed: Array,
) -> Array:
    """Recompute the integrated router's stable prefix and dynamic tail."""

    raw = jnp.asarray(values, dtype=jnp.float32)
    tail = raw[..., BASE_FEATURE_DIM:]
    safe_sources = jnp.clip(
        jnp.asarray(source_slots, dtype=jnp.int32),
        jnp.int32(0),
        jnp.int32(ACTIVE_PAIR_SLOTS - 1),
    )
    gathered = jnp.take(tail, safe_sources, axis=-1)
    carried = jnp.where(
        jnp.asarray(survivor_mask, dtype=jnp.bool_),
        gathered,
        jnp.zeros_like(gathered),
    )
    if not carry_survivors:
        carried = jnp.where(descriptors_changed, jnp.zeros_like(carried), carried)
    return jnp.concatenate((raw[..., :BASE_FEATURE_DIM], carried), axis=-1)


def _replay_grounded_update(
    agent: IntegratedHiddenPartnerAgent,
    pre_state: HiddenPartnerWorldOnlineState,
    result: HiddenPartnerWorldOnlineStep,
) -> GroundedJointWorldUpdateResult:
    model = agent.grounded_world_model
    grounded = pre_state.agent.grounded_world
    if model is None or grounded is None:
        raise ValueError("v6 intervention audit requires the grounded model lane")
    saturated = grounded.update_count == jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    replay_input = grounded.replace(
        update_count=jnp.where(
            saturated,
            jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32),
            grounded.update_count,
        )
    )
    trace = result.trace
    replay = model.update(
        replay_input,
        trace.chi_pre,
        trace.focal_action,
        trace.partner_action,
        trace.next_observation,
        trace.reward,
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    return replay.replace(
        state=replay.state.replace(
            update_count=jnp.where(
                saturated,
                grounded.update_count,
                replay.state.update_count,
            )
        )
    )


def _grounded_projection_exact(
    mechanism: HiddenPartnerWorldMechanismTrace,
    replay: GroundedJointWorldUpdateResult,
    cached: IntegratedGroundedPlannerEvaluation,
    focal_action: Array,
    partner_action: Array,
) -> Array:
    joint_index = N_ACTIONS * focal_action + partner_action
    prediction_matches = (
        _exact(replay.prediction.joint_action_index, joint_index)
        & _exact(
            replay.prediction.raw_predictions,
            cached.grounded_raw_predictions[joint_index],
        )
        & _exact(
            replay.prediction.raw_predictions,
            replay.prediction.feature_contribution + replay.prediction.row_bias,
        )
    )
    checks = (
        mechanism.grounded_enabled,
        _exact(mechanism.grounded_prediction_valid, replay.prediction.valid),
        _exact(mechanism.grounded_target_valid, replay.diagnostics.target_valid),
        _exact(mechanism.grounded_gradient_valid, replay.gradient_valid),
        _exact(
            mechanism.grounded_prediction_matches_decision,
            prediction_matches,
        ),
        _exact(
            mechanism.grounded_row_update_isolated,
            replay.diagnostics.row_update_isolated,
        ),
        _exact(mechanism.grounded_update_applied, replay.diagnostics.applied),
        _exact(mechanism.grounded_executed_joint_index, joint_index),
        _exact(
            mechanism.grounded_feature_contribution,
            replay.prediction.feature_contribution,
        ),
        _exact(mechanism.grounded_row_bias, replay.prediction.row_bias),
        _exact(mechanism.grounded_raw_predictions, replay.prediction.raw_predictions),
        _exact(mechanism.grounded_targets, replay.targets),
        _exact(mechanism.grounded_errors, replay.errors),
        _exact(mechanism.grounded_fit_loss_by_head, replay.fit_loss_by_head),
        _exact(
            mechanism.grounded_representation_loss_by_head,
            replay.representation_loss_by_head,
        ),
        _exact(
            mechanism.grounded_representation_gradient,
            replay.representation_gradient,
        ),
        _exact(
            mechanism.grounded_representation_gradient_by_head,
            replay.representation_gradient_by_head,
        ),
        _exact(
            mechanism.grounded_representation_gradient_norm_by_head,
            replay.representation_gradient_norm_by_head,
        ),
        _exact(
            mechanism.grounded_proposed_weight_row_bit_change_mask,
            replay.proposed_weight_row_bit_change_mask,
        ),
        _exact(
            mechanism.grounded_proposed_bias_row_bit_change_mask,
            replay.proposed_bias_row_bit_change_mask,
        ),
        _exact(
            mechanism.grounded_executed_weight_row_delta_norm_by_head,
            replay.executed_weight_row_delta_norm_by_head,
        ),
        _exact(
            mechanism.grounded_executed_bias_row_delta_by_head,
            replay.executed_bias_row_delta_by_head,
        ),
        _exact(
            mechanism.grounded_credit_gradient_chi,
            replay.representation_gradient,
        ),
        _exact(mechanism.grounded_credit_valid, replay.gradient_valid),
    )
    return jnp.all(jnp.stack(tuple(jnp.asarray(check, dtype=jnp.bool_) for check in checks)))


def _mix_projection_exact(
    mechanism: HiddenPartnerWorldMechanismTrace,
    replay: RepresentationGradientMixResult,
) -> Array:
    return (
        _replay_equal(mechanism.mixed_credit_gradient_chi, replay.gradient)
        & _exact(mechanism.mixed_credit_valid, replay.valid)
        & _exact(mechanism.mixed_credit_applied, replay.applied)
        & _exact(mechanism.mixed_credit_conflict, replay.diagnostics.conflict)
    )


def _interaction_replay(
    agent: IntegratedHiddenPartnerAgent,
    pre_state: HiddenPartnerWorldOnlineState,
    result: HiddenPartnerWorldOnlineStep,
) -> tuple[InteractionFeatureUpdateResult, InteractionCurationPriorityOverride]:
    partner_target = jnp.asarray(2.0, dtype=jnp.float32) * result.trace.partner_action.astype(
        jnp.float32
    ) - jnp.asarray(1.0, dtype=jnp.float32)
    # Recompute the reviewed rank streams independently of the production
    # helper so an incorrect fold-in tag or permutation cannot self-validate.
    interaction_key = pre_state.agent.interaction.key
    active_key = jr.fold_in(interaction_key, jnp.uint32(0x43555241))
    candidate_key = jr.fold_in(interaction_key, jnp.uint32(0x43555243))
    priority = InteractionCurationPriorityOverride(
        enabled=jnp.asarray(agent.config.random_feature_curation, dtype=jnp.bool_),
        active_ranks=jr.permutation(active_key, ACTIVE_PAIR_SLOTS).astype(jnp.float32),
        candidate_ranks=jr.permutation(candidate_key, CANDIDATE_PAIR_SLOTS).astype(jnp.float32),
    )
    replay = agent.interaction_learner.update(
        pre_state.agent.interaction,
        agent._deployed_phi(pre_state.agent.phi),  # noqa: SLF001
        jnp.reshape(partner_target, (1,)),
        external_read_mask=pre_state.agent.consumer_active_mask,
        curation_priority_override=priority,
    )
    return replay, priority


def _interaction_projection_exact(
    mechanism: HiddenPartnerWorldMechanismTrace,
    replay: InteractionFeatureUpdateResult,
    *,
    lifecycle_enabled: bool,
) -> Array:
    proposal_descriptors = jnp.stack(
        (replay.state.feature_left, replay.state.feature_right), axis=1
    ).astype(jnp.int32)
    committed = replay.state if lifecycle_enabled else replay.pre_curation_state
    applied_descriptors = jnp.stack(
        (committed.feature_left, committed.feature_right), axis=1
    ).astype(jnp.int32)
    proposed = (
        (replay.replaced_slot >= 0) | (replay.refreshed_candidate >= 0) | (replay.retired_slot >= 0)
    )
    absent = jnp.asarray(-1, dtype=jnp.int32)
    applied_replaced = jnp.where(lifecycle_enabled, replay.replaced_slot, absent)
    applied_promoted = jnp.where(lifecycle_enabled, replay.promoted_candidate, absent)
    applied_refreshed = jnp.where(lifecycle_enabled, replay.refreshed_candidate, absent)
    applied_retired = jnp.where(lifecycle_enabled, replay.retired_slot, absent)
    applied_left = jnp.where(lifecycle_enabled, replay.retired_left, absent)
    applied_right = jnp.where(lifecycle_enabled, replay.retired_right, absent)
    checks = (
        _exact(mechanism.lifecycle_enabled, jnp.asarray(lifecycle_enabled)),
        _exact(mechanism.lifecycle_proposed, proposed),
        _exact(mechanism.lifecycle_applied, proposed & lifecycle_enabled),
        _exact(mechanism.lifecycle_proposal_descriptors, proposal_descriptors),
        _exact(mechanism.lifecycle_applied_descriptors, applied_descriptors),
        _exact(mechanism.lifecycle_proposal_replaced_slot, replay.replaced_slot),
        _exact(
            mechanism.lifecycle_proposal_promoted_candidate,
            replay.promoted_candidate,
        ),
        _exact(
            mechanism.lifecycle_proposal_refreshed_candidate,
            replay.refreshed_candidate,
        ),
        _exact(mechanism.lifecycle_proposal_retired_slot, replay.retired_slot),
        _exact(mechanism.lifecycle_proposal_retired_left, replay.retired_left),
        _exact(mechanism.lifecycle_proposal_retired_right, replay.retired_right),
        _exact(mechanism.lifecycle_applied_replaced_slot, applied_replaced),
        _exact(mechanism.lifecycle_applied_promoted_candidate, applied_promoted),
        _exact(mechanism.lifecycle_applied_refreshed_candidate, applied_refreshed),
        _exact(mechanism.lifecycle_applied_retired_slot, applied_retired),
        _exact(mechanism.lifecycle_applied_retired_left, applied_left),
        _exact(mechanism.lifecycle_applied_retired_right, applied_right),
        _exact(mechanism.lifecycle_active_evidence_refreshed, replay.evidence_refreshed),
        _exact(
            mechanism.lifecycle_retention_evidence_refreshed,
            replay.retention_evidence_refreshed,
        ),
        _exact(mechanism.lifecycle_durable_read_mask, replay.durable_read_mask),
        _exact(mechanism.lifecycle_relevance_probe_scores, replay.relevance_probe_scores),
        _exact(mechanism.lifecycle_relevance_probe_errors, replay.relevance_probe_errors),
        _exact(
            mechanism.lifecycle_candidate_promotion_signal,
            replay.candidate_promotion_signal,
        ),
        _exact(
            mechanism.lifecycle_candidate_promotion_raw_evidence,
            replay.candidate_promotion_raw_evidence,
        ),
        _exact(
            mechanism.lifecycle_candidate_promotion_confirmed,
            replay.candidate_promotion_confirmed,
        ),
    )
    return jnp.all(jnp.stack(tuple(jnp.asarray(check, dtype=jnp.bool_) for check in checks)))


def _retention_algebra(
    agent: IntegratedHiddenPartnerAgent,
    pre_state: HiddenPartnerWorldOnlineState,
    replay: InteractionFeatureUpdateResult,
) -> tuple[Array, Array]:
    cfg = agent.config
    old = pre_state.agent.interaction
    live = _live_descriptors(jnp.stack((old.feature_left, old.feature_right), axis=1))
    scores = replay.relevance_probe_scores
    grace = cfg.active_utility_retention_grace_steps
    if grace is None:
        refreshed = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_)
        streak = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32)
        idle = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32)
        ema = (
            jnp.float32(cfg.interaction_utility_decay) * old.utilities
            + jnp.float32(1.0 - cfg.interaction_utility_decay) * scores
        )
        expected = jnp.where(live, ema, 0.0)
        counterfactual_bind = jnp.asarray(False, dtype=jnp.bool_)
    else:
        raw = (
            live
            & jnp.isfinite(scores)
            & (scores >= jnp.float32(cfg.active_utility_evidence_threshold))
        )
        if cfg.evidence_gated_feature_memory:
            incremented = (
                jnp.minimum(
                    jnp.maximum(old.utility_evidence_streak, 0),
                    jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32),
                )
                + 1
            )
            streak = jnp.where(raw, incremented, 0)
            refreshed = raw & (streak >= jnp.int32(cfg.feature_evidence_confirmation_steps))
        else:
            streak = jnp.zeros_like(old.utility_evidence_streak)
            refreshed = raw
        incremented_idle = (
            jnp.minimum(
                jnp.maximum(old.evidence_idle_steps, 0),
                jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32),
            )
            + 1
        )
        idle = jnp.where(live, jnp.where(refreshed, 0, incremented_idle), 0)
        ema = (
            jnp.float32(cfg.interaction_utility_decay) * old.utilities
            + jnp.float32(1.0 - cfg.interaction_utility_decay) * scores
        )
        protected = live & (idle <= jnp.int32(grace))
        if cfg.active_utility_retention_decay is None:
            expected = ema
        else:
            retained = jnp.float32(cfg.active_utility_retention_decay) * old.utilities
            expected = jnp.where(protected, jnp.maximum(ema, retained), ema)
        expected = jnp.where(live, expected, 0.0)
        full_retained = jnp.float32(_FULL_RETENTION_DECAY) * old.utilities
        counterfactual_bind = jnp.any(protected & (full_retained > ema))
    exact = (
        _exact(replay.retention_evidence_refreshed, refreshed)
        & _exact(replay.pre_curation_state.utility_evidence_streak, streak)
        & _exact(replay.pre_curation_state.evidence_idle_steps, idle)
        & _exact(replay.pre_curation_state.utilities, expected)
    )
    return exact, counterfactual_bind


def _retirement_algebra(
    agent: IntegratedHiddenPartnerAgent,
    pre_state: HiddenPartnerWorldOnlineState,
    replay: InteractionFeatureUpdateResult,
    mechanism: HiddenPartnerWorldMechanismTrace,
) -> tuple[Array, Array]:
    cfg = agent.config
    pre = pre_state.agent.interaction
    ordinary = replay.pre_curation_state
    live = _live_descriptors(jnp.stack((pre.feature_left, pre.feature_right), axis=1))
    grace = cfg.active_utility_retention_grace_steps
    grace_value = -1 if grace is None else grace
    stale_scores = jnp.where(
        live
        & (ordinary.ages >= jnp.int32(cfg.min_feature_age))
        & (ordinary.evidence_idle_steps > jnp.int32(grace_value)),
        ordinary.evidence_idle_steps,
        jnp.asarray(-1, dtype=jnp.int32),
    )
    stale_slot = jnp.argmax(stale_scores).astype(jnp.int32)
    cadence = (ordinary.step_count % jnp.int32(max(cfg.replacement_interval, 1))) == 0
    eligible = (
        (cfg.replacement_interval > 0) & cadence & jnp.any(stale_scores >= 0) & ~jnp.any(~live)
    )
    should_retire = eligible & jnp.asarray(cfg.retire_stale_features)
    absent = jnp.asarray(-1, dtype=jnp.int32)
    expected_slot = jnp.where(should_retire, stale_slot, absent)
    expected_left = jnp.where(should_retire, pre.feature_left[stale_slot], absent)
    expected_right = jnp.where(should_retire, pre.feature_right[stale_slot], absent)
    exact = (
        _exact(replay.retired_slot, expected_slot)
        & _exact(replay.retired_left, expected_left)
        & _exact(replay.retired_right, expected_right)
        & _exact(mechanism.lifecycle_proposal_retired_slot, expected_slot)
        & _exact(mechanism.lifecycle_proposal_retired_left, expected_left)
        & _exact(mechanism.lifecycle_proposal_retired_right, expected_right)
    )
    return exact, eligible


def _random_curation_exact(
    agent: IntegratedHiddenPartnerAgent,
    pre_state: HiddenPartnerWorldOnlineState,
    replay: InteractionFeatureUpdateResult,
    priority: InteractionCurationPriorityOverride,
    mechanism: HiddenPartnerWorldMechanismTrace,
    partner_action: Array,
    *,
    compute_counterfactual: bool,
) -> tuple[Array, Array]:
    enabled = jnp.asarray(agent.config.random_feature_curation, dtype=jnp.bool_)
    exact = (
        _exact(priority.enabled, enabled)
        & _exact(mechanism.random_curation_enabled, enabled)
        & _exact(
            mechanism.random_curation_active_priorities,
            priority.active_ranks,
        )
        & _exact(
            mechanism.random_curation_candidate_priorities,
            priority.candidate_ranks,
        )
        & _exact(mechanism.random_curation_attempted, replay.curation_attempted)
        & _exact(
            mechanism.random_curation_applied,
            replay.curation_priority_override_applied,
        )
        & _exact(
            mechanism.random_curation_selected_active_worst_slot,
            replay.curation_selected_active_worst_slot,
        )
        & _exact(
            mechanism.random_curation_selected_promotion_candidate,
            replay.curation_selected_promotion_candidate,
        )
        & _exact(
            mechanism.random_curation_selected_refresh_candidate,
            replay.curation_selected_refresh_candidate,
        )
    )
    if not compute_counterfactual:
        return exact, jnp.asarray(False, dtype=jnp.bool_)
    utility_priority = InteractionCurationPriorityOverride(
        enabled=jnp.asarray(False, dtype=jnp.bool_),
        active_ranks=jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.float32),
        candidate_ranks=jnp.zeros((CANDIDATE_PAIR_SLOTS,), dtype=jnp.float32),
    )
    target = jnp.reshape(
        jnp.float32(2.0) * partner_action.astype(jnp.float32) - jnp.float32(1.0),
        (1,),
    )
    utility_replay = agent.interaction_learner.update(
        pre_state.agent.interaction,
        agent._deployed_phi(pre_state.agent.phi),  # noqa: SLF001
        target,
        external_read_mask=pre_state.agent.consumer_active_mask,
        curation_priority_override=utility_priority,
    )
    random_selected = jnp.stack(
        (
            replay.curation_selected_active_worst_slot,
            replay.curation_selected_promotion_candidate,
            replay.curation_selected_refresh_candidate,
        )
    )
    utility_selected = jnp.stack(
        (
            utility_replay.curation_selected_active_worst_slot,
            utility_replay.curation_selected_promotion_candidate,
            utility_replay.curation_selected_refresh_candidate,
        )
    )
    differs = (
        replay.curation_priority_override_applied
        & jnp.any(random_selected != utility_selected)
        & ~_tree_exact(replay.state, utility_replay.state)
    )
    return exact, differs


def required_v6_intervention_witness_names(
    control: HiddenPartnerLifecycleWorldV6Control,
) -> tuple[str, ...]:
    """Return the exact positive-support obligations for one canonical control."""

    validated = validate_v6_control(control)
    mapping = dict(V6_CONTROL_REQUIRED_WITNESSES)
    return mapping[validated.name]


def required_v6_intervention_witness_mask(
    control: HiddenPartnerLifecycleWorldV6Control,
) -> Bool[Array, " 16"]:
    """Return a static bool[16] mask in the frozen witness order."""

    required = frozenset(required_v6_intervention_witness_names(control))
    return jnp.asarray(
        tuple(name in required for name in V6_INTERVENTION_WITNESS_ORDER),
        dtype=jnp.bool_,
    )


def v6_required_witnesses_satisfied(
    control: HiddenPartnerLifecycleWorldV6Control,
    witness_counts: Array,
) -> Bool[Array, ""]:
    """Return a traced scalar requiring positive support for every named witness."""

    counts = jnp.asarray(witness_counts)
    if counts.shape != (len(V6_INTERVENTION_WITNESS_ORDER),):
        raise ValueError("v6 intervention witness counts must have shape (16,)")
    if counts.dtype != jnp.int32:
        raise TypeError("v6 intervention witness counts must have dtype int32")
    required = required_v6_intervention_witness_mask(control)
    return jnp.all(~required | (counts > 0)) & jnp.all(counts >= 0)


def missing_v6_required_witnesses(
    control: HiddenPartnerLifecycleWorldV6Control,
    witness_counts: Array | Sequence[int],
) -> tuple[str, ...]:
    """Host-only strict validator returning the names with zero support."""

    counts = np.asarray(jax.device_get(witness_counts))
    if counts.shape != (len(V6_INTERVENTION_WITNESS_ORDER),):
        raise ValueError("v6 intervention witness counts must have shape (16,)")
    if counts.dtype != np.dtype(np.int32):
        raise TypeError("v6 intervention witness counts must have dtype int32")
    if np.any(counts < 0):
        raise ValueError("v6 intervention witness counts must be non-negative")
    required = frozenset(required_v6_intervention_witness_names(control))
    return tuple(
        name
        for name, count in zip(V6_INTERVENTION_WITNESS_ORDER, counts, strict=True)
        if name in required and int(count) == 0
    )


def validate_v6_intervention_step_audit(
    audit: V6InterventionStepAudit,
) -> V6InterventionStepAudit:
    """Validate the exact fixed-width JAX result contract without materializing it."""

    if type(audit) is not V6InterventionStepAudit:
        raise TypeError("audit must be an exact V6InterventionStepAudit")
    for name, width in (("checks", 18), ("witnesses", 16)):
        raw_value = getattr(audit, name)
        if not isinstance(raw_value, (jax.Array, Tracer)):
            raise TypeError(f"audit.{name} must be a JAX array")
        value = jnp.asarray(raw_value)
        if value.shape != (width,):
            raise ValueError(f"audit.{name} must have shape ({width},)")
        if value.dtype != jnp.bool_:
            raise TypeError(f"audit.{name} must have dtype bool")
    return audit


def audit_v6_intervention_step(
    control: HiddenPartnerLifecycleWorldV6Control,
    agent: IntegratedHiddenPartnerAgent,
    pre_state: HiddenPartnerWorldOnlineState,
    result: HiddenPartnerWorldOnlineStep,
) -> V6InterventionStepAudit:
    """Replay and audit one v6 bridge transition without advancing it again."""

    validated = validate_v6_control(control)
    if type(agent) is not IntegratedHiddenPartnerAgent:
        raise TypeError("agent must be an exact IntegratedHiddenPartnerAgent")
    if validated.agent_config is None or validated.agent_config != agent.config:
        raise ValueError("control and audit agent must have the exact same config")
    if not isinstance(pre_state, HiddenPartnerWorldOnlineState):
        raise TypeError("pre_state must be a HiddenPartnerWorldOnlineState")
    if not isinstance(result, HiddenPartnerWorldOnlineStep):
        raise TypeError("result must be a HiddenPartnerWorldOnlineStep")

    cfg = agent.config
    trace = result.trace
    mechanism = trace.mechanism
    post_agent = result.state.agent
    accepted = trace.accepted & mechanism.valid

    behavior = agent.behavior_model.input_loss_gradient(
        pre_state.agent.behavior,
        trace.chi_pre,
        trace.partner_action,
    )
    behavior_valid = (
        jnp.all(jnp.isfinite(behavior.gradient))
        & jnp.all(jnp.isfinite(behavior.probabilities))
        & jnp.isfinite(behavior.loss)
    )
    behavior_update = agent.behavior_model.update(
        pre_state.agent.behavior,
        trace.chi_pre,
        trace.partner_action,
    )
    if cfg.evidence_gated_consumer_memory:
        committed_behavior_tail = jnp.where(
            mechanism.consumer_confirmed_write_pre[None, :],
            behavior_update.state.weights[:, BASE_FEATURE_DIM:],
            pre_state.agent.behavior.weights[:, BASE_FEATURE_DIM:],
        )
        committed_behavior_weights = jnp.concatenate(
            (
                behavior_update.state.weights[:, :BASE_FEATURE_DIM],
                committed_behavior_tail,
            ),
            axis=1,
        )
    else:
        committed_behavior_weights = behavior_update.state.weights
    expected_behavior_weights = _route_last_axis(
        committed_behavior_weights,
        mechanism.router_source_slots,
        mechanism.router_survivor_mask,
        carry_survivors=cfg.carry_survivors,
        descriptors_changed=mechanism.router_descriptors_changed,
    )
    expected_behavior_state = behavior_update.state.replace(weights=expected_behavior_weights)
    predicted_behavior_exact = _exact(
        behavior.probabilities,
        pre_state.agent.current_evaluation.predicted_partner_probabilities,
    )
    behavior_credit_exact = (
        _exact(mechanism.behavior_credit_gradient_chi, behavior.gradient)
        & _exact(mechanism.behavior_credit_valid, behavior_valid)
        & _exact(mechanism.behavior_prediction_matches_decision, predicted_behavior_exact)
        & _tree_replay_equal(post_agent.behavior, expected_behavior_state)
    )

    grounded_replay = _replay_grounded_update(agent, pre_state, result)
    cached_grounded = pre_state.agent.current_evaluation.grounded_world
    if cached_grounded is None:
        raise ValueError("v6 intervention audit requires cached grounded evaluation")
    grounded_credit_exact = _grounded_projection_exact(
        mechanism,
        grounded_replay,
        cached_grounded,
        trace.focal_action,
        trace.partner_action,
    ) & _exact(
        mechanism.grounded_learning_enabled,
        jnp.asarray(cfg.grounded_world_learning_enabled),
    )

    mixer_config = cfg.representation_gradient_mixer
    if mixer_config is None:
        raise ValueError("v6 intervention audit requires the representation mixer")
    mix_replay = mix_representation_gradients(
        mixer_config,
        behavior.gradient,
        grounded_replay.representation_gradient,
        behavior_valid=behavior_valid,
        grounded_world_valid=grounded_replay.gradient_valid,
    )
    mix_exact = _mix_projection_exact(mechanism, mix_replay)
    if mixer_config.mode == "discard":
        mix_exact &= (
            mix_replay.valid
            & ~mix_replay.applied
            & _exact(mix_replay.gradient, jnp.zeros_like(mix_replay.gradient))
        )
    elif mixer_config.mode == "behavior_only":
        mix_exact &= (
            mix_replay.diagnostics.behavior_active & ~mix_replay.diagnostics.grounded_world_active
        )
    elif mixer_config.mode == "world_only":
        mix_exact &= (
            ~mix_replay.diagnostics.behavior_active & mix_replay.diagnostics.grounded_world_active
        )
    else:
        mix_exact &= (
            mix_replay.diagnostics.behavior_active & mix_replay.diagnostics.grounded_world_active
        )

    expected_read_mask = (
        pre_state.agent.consumer_active_mask
        & pre_state.agent.interaction.active_output_memory_committed
        if cfg.independent_relevance_probe
        else pre_state.agent.consumer_active_mask
    )
    if cfg.memory_masked:
        unmasked_chi = _unmasked_chi_counterfactual(
            trace.phi_pre,
            mechanism.lifecycle_pre_descriptors,
            expected_read_mask,
        )
        unmasked_behavior_update = agent.behavior_model.update(
            pre_state.agent.behavior,
            unmasked_chi,
            trace.partner_action,
        )
        memory_downstream_effect = _masked_memory_downstream_learning_effect(
            trace.phi_pre[RAW_OBSERVATION_DIM:],
            pre_state.agent.behavior.weights[:, RAW_OBSERVATION_DIM:BASE_FEATURE_DIM],
            behavior_update.state.weights[:, RAW_OBSERVATION_DIM:BASE_FEATURE_DIM],
            unmasked_behavior_update.state.weights[:, RAW_OBSERVATION_DIM:BASE_FEATURE_DIM],
        )
    else:
        memory_downstream_effect = jnp.asarray(False, dtype=jnp.bool_)
    behavior_phi = agent.chain_chi_gradient_to_phi(
        trace.phi_pre,
        mechanism.lifecycle_pre_descriptors,
        behavior.gradient,
        expected_read_mask,
    )
    mixed_phi = agent.chain_chi_gradient_to_phi(
        trace.phi_pre,
        mechanism.lifecycle_pre_descriptors,
        mix_replay.gradient,
        expected_read_mask,
    )
    full_mix_replay = mix_representation_gradients(
        dataclasses.replace(mixer_config, mode="full"),
        behavior.gradient,
        grounded_replay.representation_gradient,
        behavior_valid=behavior_valid,
        grounded_world_valid=grounded_replay.gradient_valid,
    )
    full_mixed_phi = agent.chain_chi_gradient_to_phi(
        trace.phi_pre,
        mechanism.lifecycle_pre_descriptors,
        full_mix_replay.gradient,
        expected_read_mask,
    )
    credit_counterfactual_effect = (
        jnp.asarray(True, dtype=jnp.bool_)
        if mixer_config.mode == "full"
        else (full_mix_replay.valid & ~_replay_equal(mixed_phi, full_mixed_phi))
    )
    chain_exact = (
        _exact(mechanism.lifecycle_durable_read_mask, expected_read_mask)
        & _replay_equal(mechanism.behavior_credit_gradient_phi, behavior_phi)
        & _replay_equal(mechanism.mixed_credit_gradient_phi, mixed_phi)
    )

    learned_builder, builder_learning = agent.state_builder.learn(
        pre_state.agent.state_builder,
        mixed_phi,
    )
    deployed_builder = (
        learned_builder if cfg.state_learning_enabled else pre_state.agent.state_builder
    )
    expected_builder, expected_phi = agent.state_builder.update(
        deployed_builder,
        trace.next_observation,
        trace.focal_action,
        trace.reward,
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    state_learning_exact = (
        _exact(mechanism.state_learning_enabled, jnp.asarray(cfg.state_learning_enabled))
        & _exact(mechanism.state_learning_proposal_valid, builder_learning.proposal_valid)
        & _exact(mechanism.state_learning_component_applied, builder_learning.applied)
        & _exact(
            mechanism.state_learning_committed,
            builder_learning.applied & cfg.state_learning_enabled,
        )
        & _exact(mechanism.state_learning_valid, builder_learning.valid)
        & _exact(mechanism.state_learning_rejected, builder_learning.rejected)
        & _replay_equal(mechanism.state_learning_gradient_norm, builder_learning.gradient_norm)
        & _replay_equal(
            mechanism.state_learning_clipped_gradient_norm,
            builder_learning.clipped_gradient_norm,
        )
        & _replay_equal(
            mechanism.state_learning_parameter_update_norm,
            builder_learning.parameter_update_norm,
        )
        & _state_learning_persistence_exact(
            enabled=cfg.state_learning_enabled,
            pre_state=pre_state.agent.state_builder,
            post_state=post_agent.state_builder,
            replay_state=expected_builder,
        )
        & _replay_equal(post_agent.phi, expected_phi)
    )

    pre_grounded = pre_state.agent.grounded_world
    post_grounded = post_agent.grounded_world
    if pre_grounded is None or post_grounded is None:
        raise ValueError("v6 intervention audit requires persistent grounded state")
    proposed_grounded = (
        grounded_replay.state if cfg.grounded_world_learning_enabled else pre_grounded
    )
    if cfg.evidence_gated_consumer_memory:
        committed_tail = jnp.where(
            mechanism.consumer_confirmed_write_pre[None, None, :],
            proposed_grounded.weights[..., BASE_FEATURE_DIM:],
            pre_grounded.weights[..., BASE_FEATURE_DIM:],
        )
        committed_weights = jnp.concatenate(
            (proposed_grounded.weights[..., :BASE_FEATURE_DIM], committed_tail),
            axis=-1,
        )
    else:
        committed_weights = proposed_grounded.weights
    expected_grounded_weights = _route_last_axis(
        committed_weights,
        mechanism.router_source_slots,
        mechanism.router_survivor_mask,
        carry_survivors=cfg.carry_survivors,
        descriptors_changed=mechanism.router_descriptors_changed,
    )
    grounded_learning_exact = (
        _exact(post_grounded.weights, expected_grounded_weights)
        & _exact(post_grounded.bias, proposed_grounded.bias)
        & _exact(post_grounded.update_count, proposed_grounded.update_count)
        & mechanism.consumer_route_grounded_values_exact
    )

    expected_pre_chi = agent.build_chi(
        trace.phi_pre,
        mechanism.lifecycle_pre_descriptors,
        expected_read_mask,
    )
    post_read_mask = (
        post_agent.consumer_active_mask & post_agent.interaction.active_output_memory_committed
        if cfg.independent_relevance_probe
        else post_agent.consumer_active_mask
    )
    expected_post_chi = agent.build_chi(
        post_agent.phi,
        post_agent.router.descriptors,
        post_read_mask,
    )
    memory_exact = _exact(trace.chi_pre, expected_pre_chi) & _exact(
        post_agent.chi, expected_post_chi
    )
    if cfg.memory_masked:
        hidden_zeros = jnp.zeros((HIDDEN_STATE_DIM,), dtype=jnp.float32)
        memory_exact &= (
            _exact(
                trace.chi_pre[RAW_OBSERVATION_DIM:BASE_FEATURE_DIM],
                hidden_zeros,
            )
            & _exact(behavior_phi[RAW_OBSERVATION_DIM:], hidden_zeros)
            & _exact(mixed_phi[RAW_OBSERVATION_DIM:], hidden_zeros)
        )
    else:
        memory_exact &= _exact(trace.chi_pre[:BASE_FEATURE_DIM], trace.phi_pre)

    evaluation_replay = agent.evaluate_models(
        pre_state.agent.behavior,
        pre_state.agent.joint_world,
        pre_state.agent.control,
        trace.chi_pre,
        pre_grounded,
    )
    grounded_evaluation = evaluation_replay.grounded_world
    if grounded_evaluation is None:
        raise ValueError("v6 intervention audit requires both planner reward surfaces")
    cached_evaluation = pre_state.agent.current_evaluation
    cached_grounded = cached_evaluation.grounded_world
    if cached_grounded is None:
        raise ValueError("v6 intervention audit requires cached planner reward surfaces")
    cache_exact = jnp.all(
        agent._current_decision_cache_check_vector(  # noqa: SLF001
            pre_state.agent,
            evaluation_replay,
        )
    )
    expected_rewards = (
        cached_grounded.grounded_expected_rewards
        if cfg.grounded_world_planning_enabled
        else cached_grounded.table_expected_rewards
    )
    planner_source_exact = (
        cache_exact
        & _exact(cached_evaluation.expected_rewards, expected_rewards)
        & _exact(
            cached_grounded.planner_applied,
            jnp.asarray(cfg.grounded_world_planning_enabled and cfg.planning_enabled),
        )
    )
    centered = expected_rewards - jnp.mean(expected_rewards)
    model_term = jnp.float32(cfg.planner_lambda) * centered
    applied_model_term = model_term if cfg.planning_enabled else jnp.zeros_like(model_term)
    planning_exact = (
        cache_exact
        & _exact(cached_evaluation.centered_expected_rewards, centered)
        & _exact(cached_evaluation.model_term, model_term)
        & _exact(cached_evaluation.applied_model_term, applied_model_term)
        & _exact(
            cached_evaluation.planner_scores,
            cached_evaluation.q_values + applied_model_term,
        )
    )
    expected_partner = (
        jnp.full((N_ACTIONS,), 1.0 / N_ACTIONS, dtype=jnp.float32)
        if cfg.uniform_partner_belief
        else behavior.probabilities
    )
    partner_belief_exact = (
        cache_exact
        & _exact(cached_evaluation.predicted_partner_probabilities, behavior.probabilities)
        & _exact(cached_evaluation.partner_probabilities, expected_partner)
        & cached_evaluation.partner_probabilities_valid
    )

    interaction_replay, priority = _interaction_replay(agent, pre_state, result)
    expected_interaction = (
        interaction_replay.state
        if cfg.feature_lifecycle_enabled
        else interaction_replay.pre_curation_state
    )
    lifecycle_exact = _interaction_projection_exact(
        mechanism,
        interaction_replay,
        lifecycle_enabled=cfg.feature_lifecycle_enabled,
    ) & _lifecycle_persistence_exact(
        enabled=cfg.feature_lifecycle_enabled,
        pre_state=pre_state.agent.interaction,
        post_state=post_agent.interaction,
        replay_state=expected_interaction,
    )

    expected_source, expected_survivor, expected_new, expected_evicted = _expected_route_identity(
        mechanism.lifecycle_pre_descriptors,
        mechanism.lifecycle_applied_descriptors,
    )
    descriptors_changed = jnp.any(
        mechanism.lifecycle_pre_descriptors != mechanism.lifecycle_applied_descriptors
    )
    expected_carry_flag = jnp.asarray(cfg.carry_survivors) | ~descriptors_changed
    route_count_after = (
        jnp.minimum(
            jnp.maximum(mechanism.router_route_count_before, 0),
            jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32),
        )
        + 1
    )
    generation_after = jnp.where(
        descriptors_changed,
        jnp.minimum(
            jnp.maximum(mechanism.router_generation_count_before, 0),
            jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32),
        )
        + 1,
        mechanism.router_generation_count_before,
    )
    identity_exact = (
        mechanism.router_valid
        & mechanism.router_applied
        & _exact(mechanism.router_descriptors_changed, descriptors_changed)
        & _exact(mechanism.router_carry_survivors, expected_carry_flag)
        & _exact(mechanism.router_source_slots, expected_source)
        & _exact(mechanism.router_survivor_mask, expected_survivor)
        & _exact(mechanism.router_new_mask, expected_new)
        & _exact(mechanism.router_evicted_mask, expected_evicted)
        & _exact(mechanism.router_route_count_after, route_count_after)
        & _exact(mechanism.router_generation_count_after, generation_after)
        & _exact(post_agent.router.descriptors, mechanism.lifecycle_applied_descriptors)
        & _exact(post_agent.router.route_count, route_count_after)
        & _exact(post_agent.router.generation_count, generation_after)
        & mechanism.consumer_route_source_slots_exact
        & mechanism.consumer_route_identity_masks_exact
        & mechanism.consumer_route_stable_prefix_exact
        & mechanism.consumer_route_survivor_values_exact
        & mechanism.consumer_route_reset_values_exact
        & mechanism.consumer_route_no_carry_reset_exact
        & mechanism.consumer_route_behavior_values_exact
        & mechanism.consumer_route_q_values_exact
        & mechanism.consumer_route_trace_values_exact
        & mechanism.consumer_route_last_observation_exact
        & mechanism.consumer_route_grounded_values_exact
        & mechanism.consumer_route_values_exact
        & mechanism.consumer_lifecycle_destination_reset_exact
    )

    retention_exact, retention_bind = _retention_algebra(
        agent,
        pre_state,
        interaction_replay,
    )
    retirement_exact, retirement_eligible = _retirement_algebra(
        agent,
        pre_state,
        interaction_replay,
        mechanism,
    )
    random_exact, random_differs = _random_curation_exact(
        agent,
        pre_state,
        interaction_replay,
        priority,
        mechanism,
        trace.partner_action,
        compute_counterfactual=validated.name == "random_curation",
    )

    current_selection = pre_state.agent.current_selection
    next_selection = post_agent.current_selection
    current_ordinary = jnp.where(
        current_selection.explored,
        current_selection.random_action,
        current_selection.noisy_greedy_action,
    )
    next_ordinary = jnp.where(
        next_selection.explored,
        next_selection.random_action,
        next_selection.noisy_greedy_action,
    )
    forced = validated.focal_action_policy == "balanced_external"
    if forced:
        current_expected = jnp.bitwise_xor(
            jnp.int32(validated.initial_external_action),
            jnp.bitwise_and(pre_state.step_count, jnp.int32(1)),
        )
        next_expected = jnp.bitwise_xor(
            jnp.int32(validated.initial_external_action),
            jnp.bitwise_and(pre_state.step_count + 1, jnp.int32(1)),
        )
    else:
        current_expected = current_ordinary
        next_expected = next_ordinary
    action_exact = (
        _exact(trace.focal_action, current_expected)
        & _exact(trace.next_action, next_expected)
        & _exact(trace.focal_action_ordinary_policy_action, current_ordinary)
        & _exact(trace.next_action_ordinary_policy_action, next_ordinary)
        & _exact(trace.focal_action_externally_forced, jnp.asarray(forced))
        & _exact(trace.next_action_externally_forced, jnp.asarray(forced))
        & trace.policy_replay_valid
        & trace.next_policy_replay_valid
    )

    next_cue_key, cue_sample_key = jr.split(pre_state.world.cue_key)
    cue_probabilities = jnp.asarray(
        validated.world_config.cue_flip_probabilities,
        dtype=jnp.float32,
    )
    cue_flips = jr.bernoulli(cue_sample_key, p=cue_probabilities, shape=(2,))
    expected_cues = jnp.where(
        cue_flips,
        -result.state.world.world_sign,
        result.state.world.world_sign,
    ).astype(jnp.float32)
    cue_exact = (
        _exact(result.state.world.cue_key, next_cue_key)
        & _exact(result.state.world.current_cues, expected_cues)
        & _exact(
            trace.next_observation[jnp.asarray((CUE_1_INDEX, CUE_2_INDEX))],
            expected_cues,
        )
        & _exact(trace.oracle_next_world_cue_flipped, cue_flips)
    )
    base_probabilities = jnp.asarray(
        HiddenPartnerWorldFeedbackConfig().cue_flip_probabilities,
        dtype=jnp.float32,
    )
    base_flips = jr.bernoulli(cue_sample_key, p=base_probabilities, shape=(2,))
    cue_counterfactual_differs = jnp.asarray(validated.name == "equal_cue") & jnp.any(
        cue_flips != base_flips
    )

    grounded_model_config = cfg.grounded_world_model
    if grounded_model_config is None:
        raise ValueError("v6 intervention audit requires grounded model config")
    joint_index = N_ACTIONS * trace.focal_action + trace.partner_action
    row_isolation_exact = _row_update_isolation_exact(
        mechanism.grounded_proposed_weight_row_bit_change_mask,
        mechanism.grounded_proposed_bias_row_bit_change_mask,
        joint_index,
    ) & _exact(
        mechanism.grounded_raw_predictions,
        mechanism.grounded_feature_contribution + mechanism.grounded_row_bias,
    )
    if grounded_model_config.feature_path_mode == "row_bias_only":
        row_mode_exact = (
            _exact(
                mechanism.grounded_feature_contribution,
                jnp.zeros_like(mechanism.grounded_feature_contribution),
            )
            & _exact(
                mechanism.grounded_representation_gradient,
                jnp.zeros_like(mechanism.grounded_representation_gradient),
            )
            & _exact(
                mechanism.grounded_representation_gradient_by_head,
                jnp.zeros_like(mechanism.grounded_representation_gradient_by_head),
            )
            & ~jnp.any(mechanism.grounded_proposed_weight_row_bit_change_mask)
            & _exact(
                mechanism.grounded_executed_weight_row_delta_norm_by_head,
                jnp.zeros_like(mechanism.grounded_executed_weight_row_delta_norm_by_head),
            )
            & _exact(pre_grounded.weights, jnp.zeros_like(pre_grounded.weights))
            & _exact(post_grounded.weights, jnp.zeros_like(post_grounded.weights))
        )
    else:
        row_mode_exact = (
            grounded_replay.feature_path_enabled & grounded_replay.representation_credit_enabled
        )
    row_bias_exact = grounded_credit_exact & row_isolation_exact & row_mode_exact

    checks = jnp.stack(
        (
            behavior_credit_exact,
            grounded_credit_exact,
            mix_exact,
            chain_exact,
            state_learning_exact,
            grounded_learning_exact,
            memory_exact,
            planner_source_exact,
            planning_exact,
            partner_belief_exact,
            lifecycle_exact,
            identity_exact,
            retention_exact,
            retirement_exact,
            random_exact,
            action_exact,
            cue_exact,
            row_bias_exact,
        )
    ).astype(jnp.bool_)

    grounded_parameter_nonzero = _nonzero(
        grounded_replay.executed_weight_row_delta_norm_by_head
    ) | _nonzero(grounded_replay.executed_bias_row_delta_by_head)
    lifecycle_proposal = (
        (interaction_replay.replaced_slot >= 0)
        | (interaction_replay.refreshed_candidate >= 0)
        | (interaction_replay.retired_slot >= 0)
    )
    uniform = jnp.full((N_ACTIONS,), 1.0 / N_ACTIONS, dtype=jnp.float32)
    table_centered = cached_grounded.table_expected_rewards - jnp.mean(
        cached_grounded.table_expected_rewards
    )
    grounded_centered = cached_grounded.grounded_expected_rewards - jnp.mean(
        cached_grounded.grounded_expected_rewards
    )
    planner_source_effect = ~_exact(table_centered, grounded_centered)
    predicted_table_rewards = agent.joint_world_model.marginalize(
        pre_state.agent.joint_world,
        behavior.probabilities,
    ).expected_rewards
    uniform_table_rewards = agent.joint_world_model.marginalize(
        pre_state.agent.joint_world,
        uniform,
    ).expected_rewards
    predicted_grounded_rewards = cached_grounded.grounded_reward_cells @ behavior.probabilities
    uniform_grounded_rewards = cached_grounded.grounded_reward_cells @ uniform
    predicted_partner_rewards = (
        predicted_grounded_rewards
        if cfg.grounded_world_planning_enabled
        else predicted_table_rewards
    )
    uniform_partner_rewards = (
        uniform_grounded_rewards if cfg.grounded_world_planning_enabled else uniform_table_rewards
    )
    partner_belief_effect = ~_exact(
        predicted_partner_rewards - jnp.mean(predicted_partner_rewards),
        uniform_partner_rewards - jnp.mean(uniform_partner_rewards),
    )
    carry_counterfactual_differs = jnp.asarray(False, dtype=jnp.bool_)
    for consumer_values in (
        pre_state.agent.behavior.weights,
        pre_state.agent.control.q_weights,
        pre_state.agent.control.q_trace_weights,
        pre_state.agent.control.last_observation,
        committed_weights,
    ):
        carried = _route_last_axis(
            consumer_values,
            mechanism.router_source_slots,
            mechanism.router_survivor_mask,
            carry_survivors=True,
            descriptors_changed=mechanism.router_descriptors_changed,
        )
        reset = _route_last_axis(
            consumer_values,
            mechanism.router_source_slots,
            mechanism.router_survivor_mask,
            carry_survivors=False,
            descriptors_changed=mechanism.router_descriptors_changed,
        )
        carry_counterfactual_differs |= ~_exact(carried, reset)
    dormant_weight_gradient = (
        grounded_replay.errors[:, None]
        / jnp.float32(grounded_model_config.target_dim)
        * trace.chi_pre[None, :]
    )
    witnesses = jnp.stack(
        (
            _nonzero(behavior.gradient) & credit_counterfactual_effect,
            _nonzero(grounded_replay.representation_gradient) & credit_counterfactual_effect,
            builder_learning.applied & (builder_learning.parameter_update_norm > 0.0),
            grounded_parameter_nonzero,
            lifecycle_proposal,
            mechanism.router_descriptors_changed & carry_counterfactual_differs,
            retention_bind,
            retirement_eligible,
            random_differs,
            memory_downstream_effect,
            planner_source_effect,
            _nonzero(model_term),
            ~_exact(behavior.probabilities, uniform) & partner_belief_effect,
            jnp.asarray(forced)
            & ((trace.focal_action != current_ordinary) | (trace.next_action != next_ordinary)),
            cue_counterfactual_differs,
            jnp.asarray(validated.name == "row_bias") & _nonzero(dormant_weight_gradient),
        )
    ).astype(jnp.bool_)

    audit = V6InterventionStepAudit(
        checks=accepted & checks,
        witnesses=accepted & witnesses,
    )
    return validate_v6_intervention_step_audit(audit)


__all__ = [
    "V6_CONTROL_REQUIRED_WITNESSES",
    "V6_FLOAT32_REPLAY_ATOL",
    "V6_FLOAT32_REPLAY_RTOL",
    "V6_INTERVENTION_AUDIT_ORDER",
    "V6_INTERVENTION_COMPACT_FIELD_BLOCKERS",
    "V6_INTERVENTION_WITNESS_ORDER",
    "V6InterventionStepAudit",
    "audit_v6_intervention_step",
    "missing_v6_required_witnesses",
    "required_v6_intervention_witness_mask",
    "required_v6_intervention_witness_names",
    "v6_required_witnesses_satisfied",
    "validate_v6_intervention_step_audit",
]
