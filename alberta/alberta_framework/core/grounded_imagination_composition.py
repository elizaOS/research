# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Atomic planner-to-gauge-to-learner grounded-imagination composition.

This module is the smallest strict end-to-end L0 composition of
``EnsembleShortRolloutPlanner``, ``ImaginedRolloutSelectionGauge``, and
``AuthorizedImaginedRolloutActorCritic``.  One call derives the planner's
linear policy and value snapshot from the live learner state, asks the planner
for one fixed-shape batch, passes that exact in-scope value directly to the
gauge, forms an autodiff-free authorization proposal, and attempts one guarded
actor/critic commit.  There is deliberately no public rollout-batch input on
the composed call, so a caller cannot replace planner tensors between the
planner receipt and authorization.

The planner policy/value revision is the live actor/critic ``update_count +
1``.  This is only a nonzero identity for the exact parameter snapshot being
read: deriving it performs no learner update.  After an accepted transaction,
the learner update count advances into that revision and the next snapshot is
therefore assigned the following word-pair identity.

The complete planner, grounded-audit, and learner states live in one frozen,
content-sealed state.  A call adopts all three candidate states only when the
planner advances once, authorization advances once, the source proposal is
valid, and the commit applies after exactly one backward pass.  Every failure
returns the original state byte-for-byte, including the planner RNG and all
child and composition clocks.

All external facts remain attestations.  The model snapshot and action-support
counts are caller-owned; the real anchor is not environment-authenticated;
region assignments are not authenticated; and the safety/protected masks do
not grant this module safety authority.  Unkeyed content tags detect accidental
post-mint corruption but do not authenticate reality.  This is an L0,
development-only, not-assessed mechanism with no dispatch, output, efficacy,
or scientific-promotion authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Int, UInt

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.ensemble_short_rollouts import (
    EnsembleShortRolloutPlanner,
    EnsembleShortRolloutState,
    ImaginedRolloutBatch,
    RealStateRolloutAnchor,
    RolloutPolicyValueAuthority,
)
from alberta_framework.core.imagined_rollout_selection_gauge import (
    AuthorizedImaginedRolloutActorCritic,
    ImaginedRolloutActorCriticCommitTrace,
    ImaginedRolloutActorCriticState,
    ImaginedRolloutActorCriticUpdateProposal,
    ImaginedRolloutAuthorizationReceipt,
    ImaginedRolloutSelectionGauge,
    ImaginedRolloutSelectionGaugeState,
)
from alberta_framework.core.world_model_ensemble import WorldModelEnsembleState

GROUNDED_IMAGINATION_COMPOSITION_CONFIG_SCHEMA = (
    "alberta.grounded-imagination-composition.config.v1"
)
GROUNDED_IMAGINATION_COMPOSITION_CHECKPOINT_SCHEMA = (
    "alberta.grounded-imagination-composition.checkpoint.v1"
)
GROUNDED_IMAGINATION_COMPOSITION_MECHANISM_STATUS = (
    "l0-development-only-not-assessed"
)
GROUNDED_IMAGINATION_COMPOSITION_EVIDENCE_LEVEL = "L0"
GROUNDED_IMAGINATION_COMPOSITION_SCIENTIFIC_PROMOTION_ALLOWED = False
GROUNDED_IMAGINATION_REAL_ENVIRONMENT_AUTHENTICATED = False
GROUNDED_IMAGINATION_MODEL_SUPPORT_AUTHENTICATED = False
GROUNDED_IMAGINATION_REGION_ASSIGNMENTS_AUTHENTICATED = False
GROUNDED_IMAGINATION_SAFETY_PROTECTION_AUTHENTICATED = False

_TAG_OFFSET = 2_166_136_261
_TAG_PRIME = 16_777_619
_STATE_TAG_SALT = 0x47494353


def _array_contract(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: Any,
) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and tuple(cast(Any, value).shape) == shape
        and cast(Any, value).dtype == jnp.dtype(dtype)
    )


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    if not _array_contract(value, shape=shape, dtype=dtype):
        raise TypeError(
            f"{name} must have exact shape {shape} and dtype {jnp.dtype(dtype)}"
        )


def _words_nonzero(words: Array) -> Bool[Array, ""]:
    return jnp.any(words != jnp.asarray(0, dtype=jnp.uint32))


def _checked_words_add_small(words: Array, amount: int) -> tuple[Array, Array]:
    increment = jnp.asarray(amount, dtype=jnp.uint32)
    low = words[1] + increment
    carry = (low < words[1]).astype(jnp.uint32)
    high = words[0] + carry
    overflow = (carry != 0) & (high == jnp.asarray(0, dtype=jnp.uint32))
    candidate = jnp.stack((high, low)).astype(jnp.uint32)
    return jnp.where(overflow, words, candidate), ~overflow


def _checked_words_add(left: Array, right: Array) -> tuple[Array, Array]:
    low = left[1] + right[1]
    carry = (low < left[1]).astype(jnp.uint32)
    high_without_carry = left[0] + right[0]
    overflow_high = high_without_carry < left[0]
    high = high_without_carry + carry
    overflow_carry = (carry != 0) & (high == jnp.asarray(0, dtype=jnp.uint32))
    return jnp.stack((high, low)), ~(overflow_high | overflow_carry)


def _float_words(value: Array) -> Array:
    return jax.lax.bitcast_convert_type(value, jnp.uint32)


def _tree_content_words(tree: object) -> Array:
    parts: list[Array] = []
    for leaf in jax.tree.leaves(tree):
        if leaf is None:
            continue
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            words = jr.key_data(leaf)
        elif array.dtype == jnp.dtype(jnp.float32):
            words = _float_words(array)
        elif array.dtype in (
            jnp.dtype(jnp.int32),
            jnp.dtype(jnp.uint32),
            jnp.dtype(jnp.bool_),
        ):
            words = array.astype(jnp.uint32)
        else:
            raise TypeError(f"unsupported composition receipt dtype {array.dtype}")
        parts.append(jnp.ravel(words))
    if not parts:
        return jnp.zeros((0,), dtype=jnp.uint32)
    return jnp.concatenate(parts)


def _mix_words(words: Array, *, salt: int) -> UInt[Array, ""]:
    flat = jnp.ravel(words).astype(jnp.uint32)

    def body(index: int, tag: Array) -> Array:
        position = (jnp.asarray(index, dtype=jnp.uint32) + 1) * jnp.asarray(
            0x9E3779B9,
            dtype=jnp.uint32,
        )
        mixed = (tag ^ flat[index] ^ position) * jnp.asarray(
            _TAG_PRIME,
            dtype=jnp.uint32,
        )
        return (mixed << jnp.asarray(13, dtype=jnp.uint32)) | (
            mixed >> jnp.asarray(19, dtype=jnp.uint32)
        )

    tag = jax.lax.fori_loop(
        0,
        flat.shape[0],
        body,
        jnp.asarray(_TAG_OFFSET ^ salt, dtype=jnp.uint32),
    )
    return jnp.where(
        tag == jnp.asarray(0, dtype=jnp.uint32),
        jnp.asarray(salt, dtype=jnp.uint32),
        tag,
    )


def _tree_static_signature(tree: object) -> tuple[object, tuple[tuple[object, ...], ...]]:
    leaves, structure = jax.tree.flatten(tree)
    signatures: list[tuple[object, ...]] = []
    for leaf in leaves:
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            signatures.append((array.shape, "typed-prng", str(jr.key_impl(leaf))))
        else:
            signatures.append((array.shape, np.dtype(array.dtype).str))
    return structure, tuple(signatures)


def _logical_tree_size(tree: object) -> tuple[int, int]:
    scalars = 0
    nbytes = 0
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(leaf)
        scalars += int(array.size)
        nbytes += int(array.nbytes)
    return scalars, nbytes


def _config_digest(config: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_fingerprint(config: Mapping[str, object]) -> UInt[Array, " 8"]:
    digest = bytes.fromhex(_config_digest(config))
    return jnp.asarray(
        [
            int.from_bytes(digest[offset : offset + 4], "big")
            for offset in range(0, 32, 4)
        ],
        dtype=jnp.uint32,
    )


@chex.dataclass(frozen=True)
class GroundedImaginationCompositionState:
    """All persistent child states, lineage baselines, and exact bindings."""

    planner_state: EnsembleShortRolloutState
    gauge_state: ImaginedRolloutSelectionGaugeState
    actor_critic_state: ImaginedRolloutActorCriticState
    planner_config_fingerprint: UInt[Array, " 8"]
    gauge_config_fingerprint: UInt[Array, " 8"]
    actor_critic_config_fingerprint: UInt[Array, " 8"]
    composition_config_fingerprint: UInt[Array, " 8"]
    baseline_planner_call_words: UInt[Array, " 2"]
    baseline_authorization_words: UInt[Array, " 2"]
    baseline_actor_update_words: UInt[Array, " 2"]
    baseline_actor_dream_update_words: UInt[Array, " 2"]
    baseline_actor_real_update_words: UInt[Array, " 2"]
    transaction_count_words: UInt[Array, " 2"]
    state_integrity_tag: UInt[Array, ""]


@chex.dataclass(frozen=True)
class GroundedImaginationCompositionDiagnostics:
    """One-call stage, work-accounting, rollback, and attestation audit."""

    state_valid_before: Bool[Array, ""]
    state_valid_after: Bool[Array, ""]
    config_fingerprints_valid: Bool[Array, ""]
    live_actor_policy_value_bound: Bool[Array, ""]
    planner_transaction_applied: Bool[Array, ""]
    planner_batch_nonempty: Bool[Array, ""]
    planner_call_delta_exact: Bool[Array, ""]
    planner_output_forwarded_directly: Bool[Array, ""]
    exact_planner_batch_receipt_bound: Bool[Array, ""]
    caller_rollout_batch_input_available: Bool[Array, ""]
    authorization_transaction_applied: Bool[Array, ""]
    authorization_receipt_valid: Bool[Array, ""]
    authorization_granted: Bool[Array, ""]
    authorization_call_delta_exact: Bool[Array, ""]
    proposal_valid: Bool[Array, ""]
    proposal_autodiff_pass_count: Int[Array, ""]
    commit_preflight_valid: Bool[Array, ""]
    commit_backward_work_performed: Bool[Array, ""]
    commit_autodiff_pass_count: Int[Array, ""]
    actor_update_delta_exact: Bool[Array, ""]
    actor_dream_update_delta_exact: Bool[Array, ""]
    actor_backward_delta_exact: Bool[Array, ""]
    commit_applied: Bool[Array, ""]
    child_candidate_valid: Bool[Array, ""]
    real_environment_authenticated: Bool[Array, ""]
    model_support_authenticated: Bool[Array, ""]
    region_assignments_authenticated: Bool[Array, ""]
    safety_protection_authenticated: Bool[Array, ""]
    scientific_promotion_allowed: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    pre_transaction_count_words: UInt[Array, " 2"]
    post_transaction_count_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class GroundedImaginationCompositionResult:
    """Final state plus the exact in-call batch, receipt, proposal, and trace."""

    state: GroundedImaginationCompositionState
    imagined_batch: ImaginedRolloutBatch
    authorization_receipt: ImaginedRolloutAuthorizationReceipt
    update_proposal: ImaginedRolloutActorCriticUpdateProposal
    commit_trace: ImaginedRolloutActorCriticCommitTrace
    diagnostics: GroundedImaginationCompositionDiagnostics


@dataclasses.dataclass(frozen=True)
class GroundedImaginationCompositionResourceBudget:
    """Exact persistent size and source-level one-call work ceilings."""

    persistent_state_scalars: int
    persistent_state_bytes: int
    persistent_typed_prng_keys: int
    config_fingerprint_uint32_scalars: int
    max_planner_calls_per_call: int
    max_authorization_calls_per_call: int
    max_actor_critic_proposals_per_call: int
    proposal_autodiff_passes: int
    max_actor_critic_commits_per_call: int
    max_autodiff_passes_per_call: int
    accepted_call_autodiff_passes: int
    max_backward_transitions_per_call: int
    max_ensemble_prediction_calls_per_call: int
    max_member_predictions_per_call: int
    caller_rollout_batch_inputs: int
    model_state_owned: int
    dispatch_authority: int
    safety_authority: int
    output_authority: int
    scientific_promotion_allowed: bool

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


class GroundedImaginationComposition:
    """Strict transactional composition over one exact child configuration."""

    def __init__(
        self,
        planner: EnsembleShortRolloutPlanner,
        gauge: ImaginedRolloutSelectionGauge,
        actor_critic: AuthorizedImaginedRolloutActorCritic,
    ) -> None:
        if not isinstance(planner, EnsembleShortRolloutPlanner):
            raise TypeError("planner must be an EnsembleShortRolloutPlanner")
        if not isinstance(gauge, ImaginedRolloutSelectionGauge):
            raise TypeError("gauge must be an ImaginedRolloutSelectionGauge")
        if not isinstance(actor_critic, AuthorizedImaginedRolloutActorCritic):
            raise TypeError(
                "actor_critic must be an AuthorizedImaginedRolloutActorCritic"
            )
        planner_config = planner.to_config()
        gauge_config = gauge.to_config()
        actor_config = actor_critic.to_config()
        if gauge_config.get("planner") != planner_config:
            raise ValueError("gauge is not bound to the supplied exact planner config")
        if actor_config.get("gauge") != gauge_config:
            raise ValueError("actor/critic is not bound to the supplied exact gauge config")
        learner_budget = actor_critic.resource_budget
        if (
            learner_budget.proposal_autodiff_passes != 0
            or learner_budget.max_autodiff_passes_per_preflight_valid_commit != 1
            or learner_budget.rejected_preflight_autodiff_passes != 0
        ):
            raise ValueError(
                "actor/critic work contract must be zero-autodiff proposal and "
                "one guarded commit pass"
            )
        self._planner = planner
        self._gauge = gauge
        self._actor_critic = actor_critic
        self._planner_fingerprint = _config_fingerprint(planner_config)
        self._gauge_fingerprint = _config_fingerprint(gauge_config)
        self._actor_fingerprint = _config_fingerprint(actor_config)
        self._composition_fingerprint = _config_fingerprint(self.to_config())
        self._state_signature = _tree_static_signature(self._template_state())

    @property
    def planner(self) -> EnsembleShortRolloutPlanner:
        return self._planner

    @property
    def gauge(self) -> ImaginedRolloutSelectionGauge:
        return self._gauge

    @property
    def actor_critic(self) -> AuthorizedImaginedRolloutActorCritic:
        return self._actor_critic

    def to_config(self) -> dict[str, object]:
        return {
            "schema": GROUNDED_IMAGINATION_COMPOSITION_CONFIG_SCHEMA,
            "type": type(self).__name__,
            "mechanism_status": GROUNDED_IMAGINATION_COMPOSITION_MECHANISM_STATUS,
            "evidence_level": GROUNDED_IMAGINATION_COMPOSITION_EVIDENCE_LEVEL,
            "scientific_promotion_allowed": False,
            "control_benefit_assessed": False,
            "real_environment_authenticated": False,
            "model_support_authenticated": False,
            "region_assignments_authenticated": False,
            "safety_protection_authenticated": False,
            "caller_rollout_batch_input_available": False,
            "policy_value_snapshot_revision_rule": (
                "actor_critic_update_count_plus_one_no_update"
            ),
            "dispatch_authority": False,
            "safety_authority": False,
            "output_authority": False,
            "composition_order": [
                "planner_once",
                "authorize_exact_planner_batch_once",
                "autodiff_free_proposal_once",
                "guarded_commit_once",
            ],
            "planner": self._planner.to_config(),
            "gauge": self._gauge.to_config(),
            "actor_critic": self._actor_critic.to_config(),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> GroundedImaginationComposition:
        payload = dict(config)
        expected = {
            "schema",
            "type",
            "mechanism_status",
            "evidence_level",
            "scientific_promotion_allowed",
            "control_benefit_assessed",
            "real_environment_authenticated",
            "model_support_authenticated",
            "region_assignments_authenticated",
            "safety_protection_authenticated",
            "caller_rollout_batch_input_available",
            "policy_value_snapshot_revision_rule",
            "dispatch_authority",
            "safety_authority",
            "output_authority",
            "composition_order",
            "planner",
            "gauge",
            "actor_critic",
        }
        if set(payload) != expected:
            raise ValueError("grounded-imagination composition fields do not match v1")
        if payload.pop("schema") != GROUNDED_IMAGINATION_COMPOSITION_CONFIG_SCHEMA:
            raise ValueError("unsupported grounded-imagination composition schema")
        if payload.pop("type") != cls.__name__:
            raise ValueError("unexpected grounded-imagination composition type")
        if (
            payload.pop("mechanism_status")
            != GROUNDED_IMAGINATION_COMPOSITION_MECHANISM_STATUS
        ):
            raise ValueError("grounded-imagination composition must remain not assessed")
        if payload.pop("evidence_level") != "L0":
            raise ValueError("grounded-imagination composition must remain L0")
        for name in (
            "scientific_promotion_allowed",
            "control_benefit_assessed",
            "real_environment_authenticated",
            "model_support_authenticated",
            "region_assignments_authenticated",
            "safety_protection_authenticated",
            "caller_rollout_batch_input_available",
            "dispatch_authority",
            "safety_authority",
            "output_authority",
        ):
            if payload.pop(name) is not False:
                raise ValueError(f"grounded-imagination {name} must remain false")
        if (
            payload.pop("policy_value_snapshot_revision_rule")
            != "actor_critic_update_count_plus_one_no_update"
        ):
            raise ValueError("grounded-imagination snapshot revision rule changed")
        if payload.pop("composition_order") != [
            "planner_once",
            "authorize_exact_planner_batch_once",
            "autodiff_free_proposal_once",
            "guarded_commit_once",
        ]:
            raise ValueError("grounded-imagination composition order changed")
        planner_payload = payload.pop("planner")
        gauge_payload = payload.pop("gauge")
        actor_payload = payload.pop("actor_critic")
        if not all(
            isinstance(item, Mapping)
            for item in (planner_payload, gauge_payload, actor_payload)
        ):
            raise ValueError("grounded-imagination nested configs are missing")
        instance = cls(
            EnsembleShortRolloutPlanner.from_config(
                cast(Mapping[str, object], planner_payload)
            ),
            ImaginedRolloutSelectionGauge.from_config(
                cast(Mapping[str, object], gauge_payload)
            ),
            AuthorizedImaginedRolloutActorCritic.from_config(
                cast(Mapping[str, object], actor_payload)
            ),
        )
        if instance.to_config() != dict(config):
            raise ValueError("grounded-imagination config is not canonical")
        return instance

    def _template_state(self) -> GroundedImaginationCompositionState:
        planner_state = self._planner._empty_state(
            jr.key(0, impl="threefry2x32"),
            self._planner._template_authority(),
        )
        gauge_state = self._gauge._empty_state()
        actor_state = self._actor_critic._zero_state()
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        provisional = GroundedImaginationCompositionState(
            planner_state=planner_state,
            gauge_state=gauge_state,
            actor_critic_state=actor_state,
            planner_config_fingerprint=self._planner_fingerprint,
            gauge_config_fingerprint=self._gauge_fingerprint,
            actor_critic_config_fingerprint=self._actor_fingerprint,
            composition_config_fingerprint=self._composition_fingerprint,
            baseline_planner_call_words=zero_words,
            baseline_authorization_words=zero_words,
            baseline_actor_update_words=zero_words,
            baseline_actor_dream_update_words=zero_words,
            baseline_actor_real_update_words=zero_words,
            transaction_count_words=zero_words,
            state_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
        )
        return self._seal_state(provisional)

    def _state_tag(
        self,
        state: GroundedImaginationCompositionState,
    ) -> UInt[Array, ""]:
        payload = cast(
            GroundedImaginationCompositionState,
            cast(Any, state).replace(
                state_integrity_tag=jnp.asarray(0, dtype=jnp.uint32)
            ),
        )
        return _mix_words(
            _tree_content_words((self._composition_fingerprint, payload)),
            salt=_STATE_TAG_SALT,
        )

    def _seal_state(
        self,
        state: GroundedImaginationCompositionState,
    ) -> GroundedImaginationCompositionState:
        return cast(
            GroundedImaginationCompositionState,
            cast(Any, state).replace(state_integrity_tag=self._state_tag(state)),
        )

    def _state_static_valid(self, state: object) -> bool:
        return (
            isinstance(state, GroundedImaginationCompositionState)
            and _tree_static_signature(state) == self._state_signature
        )

    def _config_fingerprints_valid(
        self,
        state: GroundedImaginationCompositionState,
    ) -> Bool[Array, ""]:
        return (
            jnp.array_equal(
                state.planner_config_fingerprint,
                self._planner_fingerprint,
            )
            & jnp.array_equal(state.gauge_config_fingerprint, self._gauge_fingerprint)
            & jnp.array_equal(
                state.actor_critic_config_fingerprint,
                self._actor_fingerprint,
            )
            & jnp.array_equal(
                state.composition_config_fingerprint,
                self._composition_fingerprint,
            )
        )

    def _state_valid(
        self,
        state: GroundedImaginationCompositionState,
    ) -> Bool[Array, ""]:
        expected_planner, planner_clock_ok = _checked_words_add(
            state.baseline_planner_call_words,
            state.transaction_count_words,
        )
        expected_authorization, authorization_clock_ok = _checked_words_add(
            state.baseline_authorization_words,
            state.transaction_count_words,
        )
        expected_actor, actor_clock_ok = _checked_words_add(
            state.baseline_actor_update_words,
            state.transaction_count_words,
        )
        expected_dream, dream_clock_ok = _checked_words_add(
            state.baseline_actor_dream_update_words,
            state.transaction_count_words,
        )
        planner_static = self._planner._state_static_valid(state.planner_state)
        gauge_static = self._gauge._state_static_valid(state.gauge_state)
        actor_static = self._actor_critic._state_static_valid(state.actor_critic_state)
        if not (planner_static and gauge_static and actor_static):
            return jnp.asarray(False, dtype=jnp.bool_)
        child_clocks_valid = (
            planner_clock_ok
            & authorization_clock_ok
            & actor_clock_ok
            & dream_clock_ok
            & jnp.array_equal(
                state.planner_state.proposal_call_count_words,
                expected_planner,
            )
            & jnp.array_equal(
                state.gauge_state.authorization_count_words,
                expected_authorization,
            )
            & jnp.array_equal(
                state.actor_critic_state.update_count_words,
                expected_actor,
            )
            & jnp.array_equal(
                state.actor_critic_state.dream_update_count_words,
                expected_dream,
            )
            & jnp.array_equal(
                state.actor_critic_state.real_update_count_words,
                state.baseline_actor_real_update_words,
            )
        )
        generation_bound = (
            jnp.array_equal(
                state.planner_state.bound_source_revision_words,
                state.gauge_state.bound_source_revision_words,
            )
            & jnp.array_equal(
                state.planner_state.bound_model_revision_words,
                state.gauge_state.bound_model_revision_words,
            )
            & (
                state.planner_state.bound_source_integrity_tag
                == state.gauge_state.bound_source_integrity_tag
            )
            & (
                state.planner_state.bound_model_integrity_tag
                == state.gauge_state.bound_model_integrity_tag
            )
        )
        authorization_lineage = (~_words_nonzero(state.transaction_count_words)) | (
            jnp.array_equal(
                state.actor_critic_state.last_dream_authorization_words,
                state.gauge_state.authorization_count_words,
            )
        )
        return (
            self._config_fingerprints_valid(state)
            & self._planner._state_valid(state.planner_state)
            & self._gauge._state_valid(state.gauge_state)
            & self._actor_critic._state_valid(state.actor_critic_state)
            & child_clocks_valid
            & generation_bound
            & authorization_lineage
            & (state.state_integrity_tag == self._state_tag(state))
        )

    def state_valid(
        self,
        state: GroundedImaginationCompositionState,
    ) -> Bool[Array, ""]:
        if not self._state_static_valid(state):
            raise TypeError("state has the wrong grounded-imagination static contract")
        return self._state_valid(state)

    def _authority_from_live_actor(
        self,
        actor_state: ImaginedRolloutActorCriticState,
        model_state: WorldModelEnsembleState,
        action_support_counts: Array,
        source_revision_words: Array,
    ) -> tuple[RolloutPolicyValueAuthority, Array]:
        """Bind the current parameters to ``update_count + 1`` identity words.

        The addition mints a nonzero content-snapshot revision only.  It does
        not mutate the learner and does not count as a learner update.
        """

        revision_words, revision_valid = _checked_words_add_small(
            actor_state.update_count_words,
            1,
        )
        zero_tag = jnp.asarray(0, dtype=jnp.uint32)
        provisional = RolloutPolicyValueAuthority(
            policy_weights=actor_state.actor_parameters.weights,
            policy_bias=actor_state.actor_parameters.bias,
            value_weights=actor_state.critic_parameters.weights,
            value_bias=actor_state.critic_parameters.bias,
            action_support_counts=action_support_counts,
            source_revision_words=source_revision_words,
            model_revision_words=model_state.event_count_words,
            policy_revision_words=revision_words,
            value_revision_words=revision_words,
            source_integrity_tag=zero_tag,
            model_integrity_tag=zero_tag,
            policy_integrity_tag=zero_tag,
            value_integrity_tag=zero_tag,
            authority_integrity_tag=zero_tag,
        )
        source_tag = self._planner._source_tag(
            action_support_counts,
            source_revision_words,
        )
        model_tag = self._planner._model_tag(model_state)
        policy_tag = self._planner._policy_tag(
            provisional.policy_weights,
            provisional.policy_bias,
            revision_words,
        )
        value_tag = self._planner._value_tag(
            provisional.value_weights,
            provisional.value_bias,
            revision_words,
        )
        return (
            cast(
                RolloutPolicyValueAuthority,
                cast(Any, provisional).replace(
                    source_integrity_tag=source_tag,
                    model_integrity_tag=model_tag,
                    policy_integrity_tag=policy_tag,
                    value_integrity_tag=value_tag,
                    authority_integrity_tag=self._planner._authority_tag(
                        source_tag,
                        model_tag,
                        policy_tag,
                        value_tag,
                        provisional.model_revision_words,
                    ),
                ),
            ),
            revision_valid,
        )

    def _anchor(
        self,
        observation: Array,
        decision_id_words: Array,
        authority: RolloutPolicyValueAuthority,
    ) -> RealStateRolloutAnchor:
        provisional = RealStateRolloutAnchor(
            observation=observation,
            decision_id_words=decision_id_words,
            source_revision_words=authority.source_revision_words,
            model_revision_words=authority.model_revision_words,
            policy_revision_words=authority.policy_revision_words,
            value_revision_words=authority.value_revision_words,
            authority_integrity_tag=authority.authority_integrity_tag,
            model_integrity_tag=authority.model_integrity_tag,
            anchor_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
        )
        return cast(
            RealStateRolloutAnchor,
            cast(Any, provisional).replace(
                anchor_integrity_tag=self._planner._anchor_tag(provisional)
            ),
        )

    def init(
        self,
        *,
        planner_key: Array,
        actor_critic_key: Array,
        model_state: WorldModelEnsembleState,
        action_support_counts: Array,
        source_revision_words: Array,
        grounded_gauge_state: ImaginedRolloutSelectionGaugeState,
    ) -> GroundedImaginationCompositionState:
        """Initialize live actor/planner state and adopt one prior grounded audit.

        ``grounded_gauge_state`` must already contain whatever caller-grounded
        audit history will be used for authorization.  Its observations and
        realized outcomes remain caller attestations; adoption does not turn
        them into environment-authenticated facts.
        """

        if not self._planner._model_static_valid(model_state):
            raise TypeError("model_state has the wrong ensemble static contract")
        _require_array(
            action_support_counts,
            name="action_support_counts",
            shape=(self._planner.n_actions,),
            dtype=jnp.int32,
        )
        _require_array(
            source_revision_words,
            name="source_revision_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        if not self._gauge._state_static_valid(grounded_gauge_state):
            raise TypeError("grounded_gauge_state has the wrong static contract")
        if not bool(jax.device_get(self._gauge._state_valid(grounded_gauge_state))):
            raise ValueError("grounded_gauge_state is dynamically invalid")
        actor_state = self._actor_critic.init(actor_critic_key)
        authority, revision_valid = self._authority_from_live_actor(
            actor_state,
            model_state,
            action_support_counts,
            source_revision_words,
        )
        if not bool(jax.device_get(revision_valid)):
            raise ValueError("initial live actor revision overflowed")
        planner_state = self._planner.init(planner_key, model_state, authority)
        generation_matches = (
            jnp.array_equal(
                grounded_gauge_state.bound_source_revision_words,
                authority.source_revision_words,
            )
            & jnp.array_equal(
                grounded_gauge_state.bound_model_revision_words,
                authority.model_revision_words,
            )
            & (
                grounded_gauge_state.bound_source_integrity_tag
                == authority.source_integrity_tag
            )
            & (
                grounded_gauge_state.bound_model_integrity_tag
                == authority.model_integrity_tag
            )
        )
        if not bool(jax.device_get(generation_matches)):
            raise ValueError(
                "grounded gauge generation does not match model/support source"
            )
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        provisional = GroundedImaginationCompositionState(
            planner_state=planner_state,
            gauge_state=grounded_gauge_state,
            actor_critic_state=actor_state,
            planner_config_fingerprint=self._planner_fingerprint,
            gauge_config_fingerprint=self._gauge_fingerprint,
            actor_critic_config_fingerprint=self._actor_fingerprint,
            composition_config_fingerprint=self._composition_fingerprint,
            baseline_planner_call_words=planner_state.proposal_call_count_words,
            baseline_authorization_words=(
                grounded_gauge_state.authorization_count_words
            ),
            baseline_actor_update_words=actor_state.update_count_words,
            baseline_actor_dream_update_words=actor_state.dream_update_count_words,
            baseline_actor_real_update_words=actor_state.real_update_count_words,
            transaction_count_words=zero_words,
            state_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
        )
        state = self._seal_state(provisional)
        if not bool(jax.device_get(self._state_valid(state))):
            raise ValueError("failed to construct a valid grounded-imagination state")
        return state

    def _zero_diagnostics(
        self,
        state: GroundedImaginationCompositionState,
        *,
        state_valid: Array,
    ) -> GroundedImaginationCompositionDiagnostics:
        false = jnp.asarray(False, dtype=jnp.bool_)
        zero_i = jnp.asarray(0, dtype=jnp.int32)
        return GroundedImaginationCompositionDiagnostics(
            state_valid_before=state_valid,
            state_valid_after=state_valid,
            config_fingerprints_valid=self._config_fingerprints_valid(state),
            live_actor_policy_value_bound=false,
            planner_transaction_applied=false,
            planner_batch_nonempty=false,
            planner_call_delta_exact=false,
            planner_output_forwarded_directly=false,
            exact_planner_batch_receipt_bound=false,
            caller_rollout_batch_input_available=false,
            authorization_transaction_applied=false,
            authorization_receipt_valid=false,
            authorization_granted=false,
            authorization_call_delta_exact=false,
            proposal_valid=false,
            proposal_autodiff_pass_count=zero_i,
            commit_preflight_valid=false,
            commit_backward_work_performed=false,
            commit_autodiff_pass_count=zero_i,
            actor_update_delta_exact=false,
            actor_dream_update_delta_exact=false,
            actor_backward_delta_exact=false,
            commit_applied=false,
            child_candidate_valid=false,
            real_environment_authenticated=false,
            model_support_authenticated=false,
            region_assignments_authenticated=false,
            safety_protection_authenticated=false,
            scientific_promotion_allowed=false,
            transaction_applied=false,
            pre_transaction_count_words=state.transaction_count_words,
            post_transaction_count_words=state.transaction_count_words,
        )

    def step(
        self,
        state: GroundedImaginationCompositionState,
        *,
        model_state: WorldModelEnsembleState,
        action_support_counts: Array,
        source_revision_words: Array,
        real_observation: Array,
        decision_id_words: Array,
        region_ids: Array,
        safety_admitted: Array,
        protected: Array,
    ) -> GroundedImaginationCompositionResult:
        """Attempt one exact planner→authorize→proposal→commit transaction."""

        if not self._state_static_valid(state):
            raise TypeError("state has the wrong grounded-imagination static contract")
        if not self._planner._model_static_valid(model_state):
            raise TypeError("model_state has the wrong ensemble static contract")
        shape = (self._gauge.rollout_budget, self._gauge.rollout_horizon)
        for name, value, expected_shape, dtype in (
            (
                "action_support_counts",
                action_support_counts,
                (self._planner.n_actions,),
                jnp.int32,
            ),
            ("source_revision_words", source_revision_words, (2,), jnp.uint32),
            (
                "real_observation",
                real_observation,
                (self._planner.observation_dim,),
                jnp.float32,
            ),
            ("decision_id_words", decision_id_words, (2,), jnp.uint32),
            ("region_ids", region_ids, shape, jnp.int32),
            ("safety_admitted", safety_admitted, shape, jnp.bool_),
            ("protected", protected, shape, jnp.bool_),
        ):
            _require_array(
                value,
                name=name,
                shape=expected_shape,
                dtype=dtype,
            )
        return self._step_jit(
            state,
            model_state,
            action_support_counts,
            source_revision_words,
            real_observation,
            decision_id_words,
            region_ids,
            safety_admitted,
            protected,
        )

    @jax.jit(static_argnums=(0,))
    def _step_jit(
        self,
        state: GroundedImaginationCompositionState,
        model_state: WorldModelEnsembleState,
        action_support_counts: Array,
        source_revision_words: Array,
        real_observation: Array,
        decision_id_words: Array,
        region_ids: Array,
        safety_admitted: Array,
        protected: Array,
    ) -> GroundedImaginationCompositionResult:
        state_valid = self._state_valid(state)
        authority, revision_valid = self._authority_from_live_actor(
            state.actor_critic_state,
            model_state,
            action_support_counts,
            source_revision_words,
        )
        live_bound = revision_valid & self._planner._authority_valid(
            authority,
            model_state,
        )
        anchor = self._anchor(real_observation, decision_id_words, authority)
        preflight = state_valid & revision_valid

        def reject_preflight() -> GroundedImaginationCompositionResult:
            return GroundedImaginationCompositionResult(
                state=state,
                imagined_batch=self._gauge._zero_batch(),
                authorization_receipt=self._gauge._zero_receipt(),
                update_proposal=self._actor_critic._zero_proposal(
                    state.actor_critic_state
                ),
                commit_trace=self._actor_critic._zero_commit_trace(
                    state.actor_critic_state
                ),
                diagnostics=self._zero_diagnostics(
                    state,
                    state_valid=state_valid,
                ),
            )

        def attempt() -> GroundedImaginationCompositionResult:
            planner_result = self._planner.propose(
                state.planner_state,
                model_state,
                authority,
                anchor,
            )
            authorization = self._gauge.authorize(
                state.gauge_state,
                planner_result.proposals,
                region_ids=region_ids,
                safety_admitted=safety_admitted,
                protected=protected,
            )
            proposal = self._actor_critic.propose_dream_update(
                state.actor_critic_state,
                planner_result.proposals,
                authorization.receipt,
                authorization.state,
            )
            commit = self._actor_critic.commit_dream_update(
                state.actor_critic_state,
                proposal,
                planner_result.proposals,
                authorization.receipt,
                authorization.state,
            )
            expected_planner_words, planner_clock_ok = _checked_words_add_small(
                state.planner_state.proposal_call_count_words,
                1,
            )
            expected_authorization_words, authorization_clock_ok = (
                _checked_words_add_small(
                    state.gauge_state.authorization_count_words,
                    1,
                )
            )
            expected_actor_words, actor_clock_ok = _checked_words_add_small(
                state.actor_critic_state.update_count_words,
                1,
            )
            expected_dream_words, dream_clock_ok = _checked_words_add_small(
                state.actor_critic_state.dream_update_count_words,
                1,
            )
            actual_backward_count = proposal.eligible_transition_count.astype(
                jnp.uint32
            )
            expected_backward_words, backward_clock_ok = _checked_words_add(
                state.actor_critic_state.backward_transition_count_words,
                jnp.stack(
                    (
                        jnp.asarray(0, dtype=jnp.uint32),
                        actual_backward_count,
                    )
                ),
            )
            planner_delta_exact = planner_clock_ok & jnp.array_equal(
                planner_result.state.proposal_call_count_words,
                expected_planner_words,
            )
            authorization_delta_exact = authorization_clock_ok & jnp.array_equal(
                authorization.state.authorization_count_words,
                expected_authorization_words,
            )
            actor_delta_exact = actor_clock_ok & jnp.array_equal(
                commit.state.update_count_words,
                expected_actor_words,
            )
            dream_delta_exact = dream_clock_ok & jnp.array_equal(
                commit.state.dream_update_count_words,
                expected_dream_words,
            ) & jnp.array_equal(
                commit.state.real_update_count_words,
                state.actor_critic_state.real_update_count_words,
            )
            backward_delta_exact = (
                backward_clock_ok
                & jnp.array_equal(
                    commit.state.backward_transition_count_words,
                    expected_backward_words,
                )
                & (
                    commit.trace.backward_transition_count
                    == proposal.eligible_transition_count
                )
            )
            planner_nonempty = jnp.any(planner_result.proposals.transition_valid)
            exact_batch_bound = (
                authorization.receipt.proposal_content_tag
                == self._gauge.proposal_content_tag(planner_result.proposals)
            )
            accepted_stages = (
                live_bound
                & planner_result.diagnostics.transaction_applied
                & planner_nonempty
                & planner_delta_exact
                & authorization.diagnostics.transaction_applied
                & authorization.diagnostics.receipt_valid
                & authorization.receipt.authorized
                & exact_batch_bound
                & authorization_delta_exact
                & proposal.valid
                & commit.diagnostics.preflight_valid
                & commit.diagnostics.backward_work_performed
                & (commit.diagnostics.autodiff_pass_count == 1)
                & actor_delta_exact
                & dream_delta_exact
                & backward_delta_exact
                & commit.diagnostics.applied
            )
            proposed_transactions, transaction_clock_ok = _checked_words_add_small(
                state.transaction_count_words,
                1,
            )
            provisional = GroundedImaginationCompositionState(
                planner_state=planner_result.state,
                gauge_state=authorization.state,
                actor_critic_state=commit.state,
                planner_config_fingerprint=state.planner_config_fingerprint,
                gauge_config_fingerprint=state.gauge_config_fingerprint,
                actor_critic_config_fingerprint=(
                    state.actor_critic_config_fingerprint
                ),
                composition_config_fingerprint=(
                    state.composition_config_fingerprint
                ),
                baseline_planner_call_words=state.baseline_planner_call_words,
                baseline_authorization_words=state.baseline_authorization_words,
                baseline_actor_update_words=state.baseline_actor_update_words,
                baseline_actor_dream_update_words=(
                    state.baseline_actor_dream_update_words
                ),
                baseline_actor_real_update_words=(
                    state.baseline_actor_real_update_words
                ),
                transaction_count_words=proposed_transactions,
                state_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            )
            candidate = self._seal_state(provisional)
            child_candidate_valid = self._state_valid(candidate)
            applied = (
                accepted_stages
                & transaction_clock_ok
                & child_candidate_valid
            )
            next_state = cast(
                GroundedImaginationCompositionState,
                jax.lax.cond(applied, lambda: candidate, lambda: state),
            )
            return GroundedImaginationCompositionResult(
                state=next_state,
                imagined_batch=planner_result.proposals,
                authorization_receipt=authorization.receipt,
                update_proposal=proposal,
                commit_trace=commit.trace,
                diagnostics=GroundedImaginationCompositionDiagnostics(
                    state_valid_before=state_valid,
                    state_valid_after=self._state_valid(next_state),
                    config_fingerprints_valid=self._config_fingerprints_valid(state),
                    live_actor_policy_value_bound=live_bound,
                    planner_transaction_applied=(
                        planner_result.diagnostics.transaction_applied
                    ),
                    planner_batch_nonempty=planner_nonempty,
                    planner_call_delta_exact=planner_delta_exact,
                    planner_output_forwarded_directly=jnp.asarray(True),
                    exact_planner_batch_receipt_bound=exact_batch_bound,
                    caller_rollout_batch_input_available=jnp.asarray(False),
                    authorization_transaction_applied=(
                        authorization.diagnostics.transaction_applied
                    ),
                    authorization_receipt_valid=(
                        authorization.diagnostics.receipt_valid
                    ),
                    authorization_granted=authorization.receipt.authorized,
                    authorization_call_delta_exact=authorization_delta_exact,
                    proposal_valid=proposal.valid,
                    proposal_autodiff_pass_count=jnp.asarray(0, dtype=jnp.int32),
                    commit_preflight_valid=commit.diagnostics.preflight_valid,
                    commit_backward_work_performed=(
                        commit.diagnostics.backward_work_performed
                    ),
                    commit_autodiff_pass_count=(
                        commit.diagnostics.autodiff_pass_count
                    ),
                    actor_update_delta_exact=actor_delta_exact,
                    actor_dream_update_delta_exact=dream_delta_exact,
                    actor_backward_delta_exact=backward_delta_exact,
                    commit_applied=commit.diagnostics.applied,
                    child_candidate_valid=child_candidate_valid,
                    real_environment_authenticated=jnp.asarray(False),
                    model_support_authenticated=jnp.asarray(False),
                    region_assignments_authenticated=jnp.asarray(False),
                    safety_protection_authenticated=jnp.asarray(False),
                    scientific_promotion_allowed=jnp.asarray(False),
                    transaction_applied=applied,
                    pre_transaction_count_words=state.transaction_count_words,
                    post_transaction_count_words=next_state.transaction_count_words,
                ),
            )

        return cast(
            GroundedImaginationCompositionResult,
            jax.lax.cond(preflight, attempt, reject_preflight),
        )

    @property
    def resource_budget(self) -> GroundedImaginationCompositionResourceBudget:
        template = self._template_state()
        scalars, nbytes = _logical_tree_size(template)
        planner_budget = self._planner.resource_budget
        return GroundedImaginationCompositionResourceBudget(
            persistent_state_scalars=scalars,
            persistent_state_bytes=nbytes,
            persistent_typed_prng_keys=1,
            config_fingerprint_uint32_scalars=32,
            max_planner_calls_per_call=1,
            max_authorization_calls_per_call=1,
            max_actor_critic_proposals_per_call=1,
            proposal_autodiff_passes=0,
            max_actor_critic_commits_per_call=1,
            max_autodiff_passes_per_call=1,
            accepted_call_autodiff_passes=1,
            max_backward_transitions_per_call=(
                self._actor_critic.max_transition_budget
            ),
            max_ensemble_prediction_calls_per_call=(
                planner_budget.max_ensemble_prediction_calls_per_call
            ),
            max_member_predictions_per_call=(
                planner_budget.max_member_predictions_per_call
            ),
            caller_rollout_batch_inputs=0,
            model_state_owned=0,
            dispatch_authority=0,
            safety_authority=0,
            output_authority=0,
            scientific_promotion_allowed=False,
        )


def save_grounded_imagination_composition_checkpoint(
    composition: GroundedImaginationComposition,
    state: GroundedImaginationCompositionState,
    path: str | Path,
) -> None:
    """Persist the complete owned composition state and no external inputs."""

    if not bool(jax.device_get(composition.state_valid(state))):
        raise ValueError("refusing to save an invalid grounded-imagination state")
    config = composition.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": GROUNDED_IMAGINATION_COMPOSITION_CHECKPOINT_SCHEMA,
            "composition_config": config,
            "config_sha256": _config_digest(config),
            "resource_budget": composition.resource_budget.to_config(),
            "planner_gauge_actor_states_included": True,
            "model_or_support_inputs_included": False,
            "real_anchor_included": False,
            "rollout_batch_included": False,
            "dispatch_authority": False,
            "safety_authority": False,
            "output_authority": False,
            "scientific_promotion_allowed": False,
        },
    )


def load_grounded_imagination_composition_checkpoint(
    path: str | Path,
) -> tuple[GroundedImaginationComposition, GroundedImaginationCompositionState]:
    """Strictly restore the sole current complete-composition schema."""

    metadata = load_checkpoint_metadata(path)
    if metadata.get("schema") != GROUNDED_IMAGINATION_COMPOSITION_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not a grounded-imagination v1 checkpoint")
    config = metadata.get("composition_config")
    if not isinstance(config, Mapping):
        raise ValueError("grounded-imagination checkpoint lacks composition_config")
    config_dict = dict(config)
    if metadata.get("config_sha256") != _config_digest(config_dict):
        raise ValueError("grounded-imagination config digest does not match")
    composition = GroundedImaginationComposition.from_config(config_dict)
    if metadata.get("resource_budget") != composition.resource_budget.to_config():
        raise ValueError("grounded-imagination resource budget does not match")
    if metadata.get("planner_gauge_actor_states_included") is not True:
        raise ValueError("grounded-imagination child states must be included")
    for name in (
        "model_or_support_inputs_included",
        "real_anchor_included",
        "rollout_batch_included",
        "dispatch_authority",
        "safety_authority",
        "output_authority",
        "scientific_promotion_allowed",
    ):
        if metadata.get(name) is not False:
            raise ValueError(f"grounded-imagination checkpoint {name} must be false")
    restored, second_metadata = load_checkpoint(composition._template_state(), path)
    if second_metadata != metadata:
        raise ValueError("grounded-imagination metadata changed between reads")
    state = cast(GroundedImaginationCompositionState, restored)
    if not bool(jax.device_get(composition.state_valid(state))):
        raise ValueError("grounded-imagination checkpoint restored invalid state")
    if _logical_tree_size(state)[1] != composition.resource_budget.persistent_state_bytes:
        raise ValueError("grounded-imagination restored state size is invalid")
    return composition, state


__all__ = [
    "GROUNDED_IMAGINATION_COMPOSITION_CHECKPOINT_SCHEMA",
    "GROUNDED_IMAGINATION_COMPOSITION_CONFIG_SCHEMA",
    "GROUNDED_IMAGINATION_COMPOSITION_EVIDENCE_LEVEL",
    "GROUNDED_IMAGINATION_COMPOSITION_MECHANISM_STATUS",
    "GROUNDED_IMAGINATION_COMPOSITION_SCIENTIFIC_PROMOTION_ALLOWED",
    "GROUNDED_IMAGINATION_MODEL_SUPPORT_AUTHENTICATED",
    "GROUNDED_IMAGINATION_REAL_ENVIRONMENT_AUTHENTICATED",
    "GROUNDED_IMAGINATION_REGION_ASSIGNMENTS_AUTHENTICATED",
    "GROUNDED_IMAGINATION_SAFETY_PROTECTION_AUTHENTICATED",
    "GroundedImaginationComposition",
    "GroundedImaginationCompositionDiagnostics",
    "GroundedImaginationCompositionResourceBudget",
    "GroundedImaginationCompositionResult",
    "GroundedImaginationCompositionState",
    "load_grounded_imagination_composition_checkpoint",
    "save_grounded_imagination_composition_checkpoint",
]
