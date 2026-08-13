# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Versioned calibration-readiness sidecars for existing WP4 planners.

This module leaves :class:`RealStateOneStepDyna` and
:class:`EnsembleShortRolloutPlanner` unchanged.  It evaluates their existing
functional result, binds every selected model/action/anchor/decision identity
to the exact current :class:`WorldModelRegionCalibration` state and cell, and
then treats calibration readiness as one additional noncompensating
conjunction.  No legacy support, uncertainty, continuation, termination, or
finite-value gate is removed or relaxed.

Preparation is read-only.  Execution recomputes the exact legacy proposal and
readiness receipt from the same immutable inputs.  A current, byte-equivalent
receipt whose every required calibration gate passes may commit the untouched
legacy result.  Otherwise the readiness owner, planner state, planning RNG,
control state, and proposal batch all roll back.  Short-rollout eligibility is
prefix closed: a failed calibration gate at one required step makes every
later step on that path ineligible.

The region IDs and their revisions remain caller declarations.  Receipt tags
detect substitution and same-revision content aliasing; they do not
authenticate region assignment, environment provenance, safety, or planner
issuance.  Both readiness receipts explicitly carry no planning or safety
authority.  This is bounded L0 ``not_assessed`` mechanism infrastructure, not
evidence of probabilistic calibration, safety, planning benefit, or WP4
completion.
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
from jax import Array

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.ensemble_short_rollouts import (
    EnsembleShortRolloutDiagnostics,
    EnsembleShortRolloutPlanner,
    EnsembleShortRolloutState,
    ImaginedRolloutBatch,
    RealStateRolloutAnchor,
    RolloutPolicyValueAuthority,
)
from alberta_framework.core.multi_head_learner import MultiHeadMLPState
from alberta_framework.core.one_step_dyna import (
    OneStepDynaAuthority,
    OneStepDynaDiagnostics,
    OneStepDynaState,
    RealStateOneStepDyna,
)
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemblePrediction,
    WorldModelEnsembleState,
)
from alberta_framework.core.world_model_region_calibration import (
    WorldModelPredictBeforeOutcomeReceipt,
    WorldModelRegionCalibration,
    WorldModelRegionCalibrationState,
)

WORLD_MODEL_PLANNER_READINESS_CONFIG_SCHEMA = (
    "alberta.world-model-planner-readiness.config.v1"
)
WORLD_MODEL_PLANNER_READINESS_CHECKPOINT_SCHEMA = (
    "alberta.world-model-planner-readiness.checkpoint.v1"
)
WORLD_MODEL_PLANNER_READINESS_EVIDENCE_LEVEL = "L0"
WORLD_MODEL_PLANNER_READINESS_OUTCOME_STATUS = "not_assessed"
WORLD_MODEL_PLANNER_READINESS_SCIENTIFIC_PROMOTION_ALLOWED = False
WORLD_MODEL_PLANNER_CALIBRATION_GATE_NAMES = (
    "epistemic",
    "aleatoric",
    "next_state_error",
    "reward_error",
    "termination",
)

_UINT64_MAX = 2**64 - 1
_TAG_OFFSET = 2_166_136_261
_TAG_PRIME = 16_777_619
_DYNA_RECEIPT_SALT = 0x44594E41
_ROLLOUT_RECEIPT_SALT = 0x524F4C4C


def _strict_positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or not 1 <= value <= _UINT64_MAX - 1:
        raise ValueError(f"{name} must be an exact integer in [1, {_UINT64_MAX - 1}]")
    return value


@dataclasses.dataclass(frozen=True)
class WorldModelPlannerReadinessConfig:
    """Fail-stop execution limits for the two readiness sidecars."""

    max_dyna_executions: int = _UINT64_MAX - 1
    max_rollout_executions: int = _UINT64_MAX - 1

    def __post_init__(self) -> None:
        _strict_positive_int(self.max_dyna_executions, name="max_dyna_executions")
        _strict_positive_int(
            self.max_rollout_executions,
            name="max_rollout_executions",
        )

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema": WORLD_MODEL_PLANNER_READINESS_CONFIG_SCHEMA,
            "max_dyna_executions": self.max_dyna_executions,
            "max_rollout_executions": self.max_rollout_executions,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> WorldModelPlannerReadinessConfig:
        expected = {
            "type",
            "schema",
            "max_dyna_executions",
            "max_rollout_executions",
        }
        if type(config) is not dict or set(config) != expected:
            raise ValueError("planner-readiness config fields are not exact")
        if config.get("type") != cls.__name__:
            raise ValueError("planner-readiness config type is unsupported")
        if config.get("schema") != WORLD_MODEL_PLANNER_READINESS_CONFIG_SCHEMA:
            raise ValueError("planner-readiness config schema is unsupported")
        restored = cls(
            max_dyna_executions=cast(int, config["max_dyna_executions"]),
            max_rollout_executions=cast(int, config["max_rollout_executions"]),
        )
        if restored.to_config() != dict(config):
            raise ValueError("planner-readiness config is not canonical")
        return restored


@chex.dataclass(frozen=True)
class WorldModelPlannerReadinessState:
    """Small owner state; every child planner/calibrator state stays external."""

    lifecycle_id_words: Array
    bound_calibration_revision_words: Array
    bound_calibration_content_tag: Array
    dyna_execution_count_words: Array
    rollout_execution_count_words: Array
    last_dyna_receipt_integrity_tag: Array
    last_rollout_receipt_integrity_tag: Array


@chex.dataclass(frozen=True)
class DynaPlannerReadinessReceipt:
    """One read-only calibration binding for a functional Dyna proposal."""

    lifecycle_id_words: Array
    owner_execution_count_words: Array
    calibration_revision_words: Array
    calibration_state_content_tag: Array
    current_model_revision_words: Array
    current_representation_revision_words: Array
    selected_anchor_indices: Array
    selected_actions: Array
    selected_decision_id_words: Array
    selected_anchor_model_revision_words: Array
    region_ids: Array
    action_revision_words: Array
    region_revision_words: Array
    calibration_cell_revision_words: Array
    calibration_cell_content_tags: Array
    calibration_prediction_content_tags: Array
    calibration_receipt_integrity_tags: Array
    gate_available: Array
    gate_passed: Array
    aleatoric_noise_vetoed: Array
    legacy_guard_passed: Array
    calibration_gate_passed: Array
    combined_gate_passed: Array
    descriptive_all_required_eligible: Array
    receipt_integrity_tag: Array
    planning_authority: Array
    safety_authority: Array
    valid: Array


@chex.dataclass(frozen=True)
class ShortRolloutPlannerReadinessReceipt:
    """Fixed-shape per-step calibration binding with prefix-closed eligibility."""

    lifecycle_id_words: Array
    owner_execution_count_words: Array
    calibration_revision_words: Array
    calibration_state_content_tag: Array
    model_revision_words: Array
    representation_revision_words: Array
    decision_id_words: Array
    selected_actions: Array
    region_ids: Array
    action_revision_words: Array
    region_revision_words: Array
    calibration_cell_revision_words: Array
    calibration_cell_content_tags: Array
    calibration_prediction_content_tags: Array
    calibration_receipt_integrity_tags: Array
    gate_available: Array
    gate_passed: Array
    aleatoric_noise_vetoed: Array
    legacy_guard_passed: Array
    calibration_gate_passed: Array
    combined_gate_passed: Array
    prefix_eligible: Array
    path_eligible: Array
    descriptive_all_required_eligible: Array
    receipt_integrity_tag: Array
    planning_authority: Array
    safety_authority: Array
    valid: Array


@chex.dataclass(frozen=True)
class PlannerReadinessExecutionDiagnostics:
    """Shared exact receipt, legacy-gate, calibration, and rollback audit."""

    owner_state_valid: Array
    calibration_state_valid: Array
    receipt_static_contract_valid: Array
    receipt_content_valid: Array
    calibration_revision_monotonic: Array
    calibration_content_alias_valid: Array
    execution_capacity_available: Array
    legacy_transaction_evaluated: Array
    legacy_gates_preserved: Array
    calibration_is_additional_conjunction: Array
    prefix_closed: Array
    planning_authority: Array
    safety_authority: Array
    candidate_state_valid: Array
    applied: Array
    rejected: Array
    pre_execution_count_words: Array
    post_execution_count_words: Array


@chex.dataclass(frozen=True)
class CalibratedDynaExecutionResult:
    """Selected Dyna/control states and sidecar audit from one execution."""

    readiness_state: WorldModelPlannerReadinessState
    dyna_state: OneStepDynaState
    control_state: MultiHeadMLPState
    receipt: DynaPlannerReadinessReceipt
    legacy_diagnostics: OneStepDynaDiagnostics
    diagnostics: PlannerReadinessExecutionDiagnostics


@chex.dataclass(frozen=True)
class CalibratedShortRolloutExecutionResult:
    """Selected rollout state/proposals and sidecar audit from one execution."""

    readiness_state: WorldModelPlannerReadinessState
    rollout_state: EnsembleShortRolloutState
    proposals: ImaginedRolloutBatch
    receipt: ShortRolloutPlannerReadinessReceipt
    legacy_diagnostics: EnsembleShortRolloutDiagnostics
    diagnostics: PlannerReadinessExecutionDiagnostics


@dataclasses.dataclass(frozen=True)
class WorldModelPlannerReadinessResourceBudget:
    """Exact sidecar/receipt bytes and source-level child-call ceilings."""

    persistent_state_scalars: int
    persistent_state_bytes: int
    dyna_receipt_scalars: int
    dyna_receipt_bytes: int
    rollout_receipt_scalars: int
    rollout_receipt_bytes: int
    max_calibration_receipts_per_dyna_call: int
    max_calibration_receipts_per_rollout_call: int
    max_legacy_planner_evaluations_per_prepare: int
    max_legacy_planner_evaluations_per_execute: int
    max_dyna_executions: int
    max_rollout_executions: int
    model_state_owned: int
    control_state_owned: int
    calibration_state_owned: int
    planner_state_owned: int
    planning_authority: int
    safety_authority: int
    scientific_promotion_allowed: bool

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _array_contract(value: object, *, shape: tuple[int, ...], dtype: Any) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and cast(Any, value).shape == shape
        and cast(Any, value).dtype == jnp.dtype(dtype)
    )


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not _array_contract(value, shape=shape, dtype=dtype):
        raise TypeError(
            f"{name} must have exact shape {shape} and dtype {jnp.dtype(dtype)}"
        )
    return cast(Array, value)


def _words_nonzero(words: Array) -> Array:
    return jnp.any(words != jnp.asarray(0, dtype=jnp.uint32))


def _words_less_equal(left: Array, right: Array) -> Array:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] <= right[1]))


def _checked_words_increment(words: Array) -> tuple[Array, Array]:
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == 0).astype(jnp.uint32)
    high = words[0] + carry
    overflow = (carry != 0) & (high == 0)
    candidate = jnp.stack((high, low)).astype(jnp.uint32)
    return jnp.where(overflow, words, candidate), ~overflow


def _words_leq_limit(words: Array, limit: int) -> Array:
    limit_words = jnp.asarray(
        ((limit >> 32) & 0xFFFFFFFF, limit & 0xFFFFFFFF),
        dtype=jnp.uint32,
    )
    return _words_less_equal(words, limit_words)


def _tree_equal(left: object, right: object) -> Array:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(Any, left_tree) != right_tree or len(left_leaves) != len(right_leaves):
        return jnp.asarray(False, dtype=jnp.bool_)
    equal = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        equal = equal & jnp.array_equal(jnp.asarray(left_leaf), jnp.asarray(right_leaf))
    return equal


def _logical_tree_size(tree: object) -> tuple[int, int]:
    scalars = 0
    nbytes = 0
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        scalars += int(array.size)
        nbytes += int(array.nbytes)
    return scalars, nbytes


def _content_tag(tree: object, *, salt: int) -> Array:
    parts: list[Array] = []
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        if array.dtype == jnp.dtype(jnp.float32):
            words = jax.lax.bitcast_convert_type(array, jnp.uint32)
        elif array.dtype in (
            jnp.dtype(jnp.uint32),
            jnp.dtype(jnp.int32),
            jnp.dtype(jnp.bool_),
        ):
            words = array.astype(jnp.uint32)
        else:
            raise TypeError(f"unsupported readiness receipt dtype {array.dtype}")
        parts.append(jnp.ravel(words))
    words = jnp.concatenate(parts) if parts else jnp.zeros((0,), dtype=jnp.uint32)

    def body(index: int, tag: Array) -> Array:
        position = (jnp.asarray(index, dtype=jnp.uint32) + 1) * jnp.asarray(
            0x9E3779B9,
            dtype=jnp.uint32,
        )
        mixed = (tag ^ words[index] ^ position) * jnp.asarray(
            _TAG_PRIME,
            dtype=jnp.uint32,
        )
        return (mixed << jnp.asarray(13, dtype=jnp.uint32)) | (
            mixed >> jnp.asarray(19, dtype=jnp.uint32)
        )

    tag = jax.lax.fori_loop(
        0,
        words.shape[0],
        body,
        jnp.asarray(_TAG_OFFSET ^ salt, dtype=jnp.uint32),
    )
    return jnp.where(tag == 0, jnp.asarray(salt, dtype=jnp.uint32), tag)


def _config_digest(config: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class WorldModelPlannerReadiness:
    """Transactional readiness owner over unchanged Dyna and rollout planners."""

    def __init__(
        self,
        calibrator: WorldModelRegionCalibration,
        dyna: RealStateOneStepDyna,
        rollout: EnsembleShortRolloutPlanner,
        config: WorldModelPlannerReadinessConfig | None = None,
    ) -> None:
        if type(calibrator) is not WorldModelRegionCalibration:
            raise TypeError("calibrator must be an exact WorldModelRegionCalibration")
        if type(dyna) is not RealStateOneStepDyna:
            raise TypeError("dyna must be an exact RealStateOneStepDyna")
        if type(rollout) is not EnsembleShortRolloutPlanner:
            raise TypeError("rollout must be an exact EnsembleShortRolloutPlanner")
        resolved = WorldModelPlannerReadinessConfig() if config is None else config
        if type(resolved) is not WorldModelPlannerReadinessConfig:
            raise TypeError("config must be an exact WorldModelPlannerReadinessConfig")
        if dyna.ensemble.to_config() != rollout.ensemble.to_config():
            raise ValueError("Dyna and rollout must bind the same ensemble construction")
        cal_cfg = calibrator.config
        model_cfg = dyna.ensemble.config
        if cal_cfg.observation_dim != model_cfg.model.observation_dim:
            raise ValueError("calibrator observation_dim must match both planners")
        if cal_cfg.n_actions != model_cfg.model.n_actions:
            raise ValueError("calibrator n_actions must match both planners")
        if cal_cfg.ensemble_size != model_cfg.ensemble_size:
            raise ValueError("calibrator ensemble_size must match both planners")
        self._calibrator = calibrator
        self._dyna = dyna
        self._rollout = rollout
        self._config = resolved

    @property
    def calibrator(self) -> WorldModelRegionCalibration:
        return self._calibrator

    @property
    def dyna(self) -> RealStateOneStepDyna:
        return self._dyna

    @property
    def rollout(self) -> EnsembleShortRolloutPlanner:
        return self._rollout

    @property
    def config(self) -> WorldModelPlannerReadinessConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema": WORLD_MODEL_PLANNER_READINESS_CONFIG_SCHEMA,
            "evidence_level": WORLD_MODEL_PLANNER_READINESS_EVIDENCE_LEVEL,
            "outcome_status": WORLD_MODEL_PLANNER_READINESS_OUTCOME_STATUS,
            "scientific_promotion_allowed": False,
            "readiness": self._config.to_config(),
            "calibrator": self._calibrator.to_config(),
            "dyna": self._dyna.to_config(),
            "rollout": self._rollout.to_config(),
            "existing_planner_apis_modified": False,
            "planning_authority": False,
            "safety_authority": False,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> WorldModelPlannerReadiness:
        expected = {
            "type",
            "schema",
            "evidence_level",
            "outcome_status",
            "scientific_promotion_allowed",
            "readiness",
            "calibrator",
            "dyna",
            "rollout",
            "existing_planner_apis_modified",
            "planning_authority",
            "safety_authority",
        }
        if type(config) is not dict or set(config) != expected:
            raise ValueError("planner-readiness construction fields are not exact")
        if config.get("type") != cls.__name__:
            raise ValueError("planner-readiness construction type is unsupported")
        if config.get("schema") != WORLD_MODEL_PLANNER_READINESS_CONFIG_SCHEMA:
            raise ValueError("planner-readiness construction schema is unsupported")
        if config.get("evidence_level") != WORLD_MODEL_PLANNER_READINESS_EVIDENCE_LEVEL:
            raise ValueError("planner-readiness construction must remain L0")
        if config.get("outcome_status") != WORLD_MODEL_PLANNER_READINESS_OUTCOME_STATUS:
            raise ValueError("planner-readiness construction must remain not_assessed")
        for name in (
            "scientific_promotion_allowed",
            "existing_planner_apis_modified",
            "planning_authority",
            "safety_authority",
        ):
            if config.get(name) is not False:
                raise ValueError(f"planner-readiness {name} must be false")
        nested_names = ("readiness", "calibrator", "dyna", "rollout")
        if any(type(config[name]) is not dict for name in nested_names):
            raise ValueError("planner-readiness child constructions must be exact dicts")
        owner = cls(
            WorldModelRegionCalibration.from_config(
                cast(dict[str, object], config["calibrator"])
            ),
            RealStateOneStepDyna.from_config(cast(dict[str, object], config["dyna"])),
            EnsembleShortRolloutPlanner.from_config(
                cast(dict[str, object], config["rollout"])
            ),
            WorldModelPlannerReadinessConfig.from_config(
                cast(dict[str, object], config["readiness"])
            ),
        )
        if owner.to_config() != dict(config):
            raise ValueError("planner-readiness construction is not canonical")
        return owner

    def _empty_state(
        self,
        calibration_state: WorldModelRegionCalibrationState,
    ) -> WorldModelPlannerReadinessState:
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        zero_tag = jnp.asarray(0, dtype=jnp.uint32)
        return WorldModelPlannerReadinessState(
            lifecycle_id_words=calibration_state.lifecycle_id_words,
            bound_calibration_revision_words=calibration_state.accepted_count_words,
            bound_calibration_content_tag=self._calibrator.content_tag(calibration_state),
            dyna_execution_count_words=zero_words,
            rollout_execution_count_words=zero_words,
            last_dyna_receipt_integrity_tag=zero_tag,
            last_rollout_receipt_integrity_tag=zero_tag,
        )

    def _state_static_valid(self, state: object) -> bool:
        if type(state) is not WorldModelPlannerReadinessState:
            return False
        checks = (
            (state.lifecycle_id_words, (2,), jnp.uint32),
            (state.bound_calibration_revision_words, (2,), jnp.uint32),
            (state.bound_calibration_content_tag, (), jnp.uint32),
            (state.dyna_execution_count_words, (2,), jnp.uint32),
            (state.rollout_execution_count_words, (2,), jnp.uint32),
            (state.last_dyna_receipt_integrity_tag, (), jnp.uint32),
            (state.last_rollout_receipt_integrity_tag, (), jnp.uint32),
        )
        return all(
            _array_contract(value, shape=shape, dtype=dtype)
            for value, shape, dtype in checks
        )

    def _state_valid(self, state: WorldModelPlannerReadinessState) -> Array:
        dyna_any = _words_nonzero(state.dyna_execution_count_words)
        rollout_any = _words_nonzero(state.rollout_execution_count_words)
        return (
            _words_nonzero(state.lifecycle_id_words)
            & (state.bound_calibration_content_tag != 0)
            & _words_leq_limit(
                state.dyna_execution_count_words,
                self._config.max_dyna_executions,
            )
            & _words_leq_limit(
                state.rollout_execution_count_words,
                self._config.max_rollout_executions,
            )
            & (dyna_any == (state.last_dyna_receipt_integrity_tag != 0))
            & (rollout_any == (state.last_rollout_receipt_integrity_tag != 0))
        )

    def init(
        self,
        calibration_state: WorldModelRegionCalibrationState,
    ) -> WorldModelPlannerReadinessState:
        """Bind an empty sidecar to one exact current calibration state."""

        if not bool(jax.device_get(self._calibrator.state_valid(calibration_state))):
            raise ValueError("cannot initialize from an invalid calibration state")
        state = self._empty_state(calibration_state)
        if not bool(jax.device_get(self._state_valid(state))):
            raise RuntimeError("initial planner-readiness state is invalid")
        return state

    def state_valid(self, state: WorldModelPlannerReadinessState) -> Array:
        if not self._state_static_valid(state):
            raise TypeError("planner-readiness state has the wrong static contract")
        return self._state_valid(state)

    def _calibration_alias_valid(
        self,
        state: WorldModelPlannerReadinessState,
        calibration_state: WorldModelRegionCalibrationState,
    ) -> tuple[Array, Array]:
        monotonic = _words_less_equal(
            state.bound_calibration_revision_words,
            calibration_state.accepted_count_words,
        )
        current_tag = self._calibrator.content_tag(calibration_state)
        alias_valid = (~jnp.array_equal(
            state.bound_calibration_revision_words,
            calibration_state.accepted_count_words,
        )) | (state.bound_calibration_content_tag == current_tag)
        lifecycle = jnp.array_equal(
            state.lifecycle_id_words,
            calibration_state.lifecycle_id_words,
        )
        return monotonic & lifecycle, alias_valid

    def _prediction_receipt(
        self,
        calibration_state: WorldModelRegionCalibrationState,
        *,
        decision_id_words: Array,
        model_revision_words: Array,
        representation_revision_words: Array,
        action_revision_words: Array,
        region_revision_words: Array,
        action: Array,
        region: Array,
        prediction: WorldModelEnsemblePrediction,
    ) -> WorldModelPredictBeforeOutcomeReceipt:
        cfg = self._calibrator.config
        member_means = jnp.concatenate(
            (
                prediction.member_next_observations,
                prediction.member_rewards[:, None],
            ),
            axis=1,
        )
        member_variances = jnp.concatenate(
            (
                prediction.residual_variances[:, : cfg.observation_dim],
                prediction.residual_variances[:, cfg.observation_dim : cfg.observation_dim + 1],
            ),
            axis=1,
        )
        gamma = jnp.asarray(
            self._dyna.ensemble.config.model.gamma,
            dtype=jnp.float32,
        )
        termination_probabilities = jnp.clip(
            1.0 - prediction.member_discounts / gamma,
            0.0,
            1.0,
        )
        return self._calibrator.issue_prediction(
            calibration_state,
            lifecycle_id_words=calibration_state.lifecycle_id_words,
            decision_id_words=decision_id_words,
            model_revision_words=model_revision_words,
            representation_revision_words=representation_revision_words,
            action_revision_words=action_revision_words,
            region_revision_words=region_revision_words,
            action=action,
            region=region,
            member_mean_predictions=member_means,
            member_aleatoric_variances=member_variances,
            member_termination_probabilities=termination_probabilities,
        )

    @staticmethod
    def _gate_arrays(
        receipt: WorldModelPredictBeforeOutcomeReceipt,
    ) -> tuple[Array, Array, Array, Array]:
        gates = receipt.gates
        available = jnp.stack(
            (
                gates.epistemic.available,
                gates.aleatoric.available,
                gates.next_state_error.available,
                gates.reward_error.available,
                gates.termination.available,
            )
        )
        passed = jnp.stack(
            (
                gates.epistemic.passed,
                gates.aleatoric.passed,
                gates.next_state_error.passed,
                gates.reward_error.passed,
                gates.termination.passed,
            )
        )
        noise_vetoed = gates.aleatoric.noise_vetoed
        calibration_passed = jnp.all(available & passed) & ~noise_vetoed
        return available, passed, noise_vetoed, calibration_passed

    def _dyna_receipt_static_valid(self, receipt: object) -> bool:
        if type(receipt) is not DynaPlannerReadinessReceipt:
            return False
        budget = self._dyna.config.backup_budget
        checks = (
            (receipt.lifecycle_id_words, (2,), jnp.uint32),
            (receipt.owner_execution_count_words, (2,), jnp.uint32),
            (receipt.calibration_revision_words, (2,), jnp.uint32),
            (receipt.calibration_state_content_tag, (), jnp.uint32),
            (receipt.current_model_revision_words, (2,), jnp.uint32),
            (receipt.current_representation_revision_words, (2,), jnp.uint32),
            (receipt.selected_anchor_indices, (budget,), jnp.int32),
            (receipt.selected_actions, (budget,), jnp.int32),
            (receipt.selected_decision_id_words, (budget, 2), jnp.uint32),
            (
                receipt.selected_anchor_model_revision_words,
                (budget, 2),
                jnp.uint32,
            ),
            (receipt.region_ids, (budget,), jnp.int32),
            (receipt.action_revision_words, (budget, 2), jnp.uint32),
            (receipt.region_revision_words, (budget, 2), jnp.uint32),
            (
                receipt.calibration_cell_revision_words,
                (budget, 2),
                jnp.uint32,
            ),
            (receipt.calibration_cell_content_tags, (budget,), jnp.uint32),
            (
                receipt.calibration_prediction_content_tags,
                (budget,),
                jnp.uint32,
            ),
            (
                receipt.calibration_receipt_integrity_tags,
                (budget,),
                jnp.uint32,
            ),
            (receipt.gate_available, (budget, 5), jnp.bool_),
            (receipt.gate_passed, (budget, 5), jnp.bool_),
            (receipt.aleatoric_noise_vetoed, (budget,), jnp.bool_),
            (receipt.legacy_guard_passed, (budget,), jnp.bool_),
            (receipt.calibration_gate_passed, (budget,), jnp.bool_),
            (receipt.combined_gate_passed, (budget,), jnp.bool_),
            (receipt.descriptive_all_required_eligible, (), jnp.bool_),
            (receipt.receipt_integrity_tag, (), jnp.uint32),
            (receipt.planning_authority, (), jnp.bool_),
            (receipt.safety_authority, (), jnp.bool_),
            (receipt.valid, (), jnp.bool_),
        )
        return all(
            _array_contract(value, shape=shape, dtype=dtype)
            for value, shape, dtype in checks
        )

    def _short_rollout_receipt_static_valid(self, receipt: object) -> bool:
        if type(receipt) is not ShortRolloutPlannerReadinessReceipt:
            return False
        budget = self._rollout.config.rollout_budget
        horizon = self._rollout.config.rollout_horizon
        steps = (budget, horizon)
        revisions = (budget, horizon, 2)
        checks = (
            (receipt.lifecycle_id_words, (2,), jnp.uint32),
            (receipt.owner_execution_count_words, (2,), jnp.uint32),
            (receipt.calibration_revision_words, (2,), jnp.uint32),
            (receipt.calibration_state_content_tag, (), jnp.uint32),
            (receipt.model_revision_words, (2,), jnp.uint32),
            (receipt.representation_revision_words, (2,), jnp.uint32),
            (receipt.decision_id_words, (2,), jnp.uint32),
            (receipt.selected_actions, steps, jnp.int32),
            (receipt.region_ids, steps, jnp.int32),
            (receipt.action_revision_words, revisions, jnp.uint32),
            (receipt.region_revision_words, revisions, jnp.uint32),
            (receipt.calibration_cell_revision_words, revisions, jnp.uint32),
            (receipt.calibration_cell_content_tags, steps, jnp.uint32),
            (receipt.calibration_prediction_content_tags, steps, jnp.uint32),
            (receipt.calibration_receipt_integrity_tags, steps, jnp.uint32),
            (receipt.gate_available, (*steps, 5), jnp.bool_),
            (receipt.gate_passed, (*steps, 5), jnp.bool_),
            (receipt.aleatoric_noise_vetoed, steps, jnp.bool_),
            (receipt.legacy_guard_passed, steps, jnp.bool_),
            (receipt.calibration_gate_passed, steps, jnp.bool_),
            (receipt.combined_gate_passed, steps, jnp.bool_),
            (receipt.prefix_eligible, steps, jnp.bool_),
            (receipt.path_eligible, (budget,), jnp.bool_),
            (receipt.descriptive_all_required_eligible, (), jnp.bool_),
            (receipt.receipt_integrity_tag, (), jnp.uint32),
            (receipt.planning_authority, (), jnp.bool_),
            (receipt.safety_authority, (), jnp.bool_),
            (receipt.valid, (), jnp.bool_),
        )
        return all(
            _array_contract(value, shape=shape, dtype=dtype)
            for value, shape, dtype in checks
        )

    @staticmethod
    def _dyna_receipt_integrity_valid(receipt: DynaPlannerReadinessReceipt) -> Array:
        unsigned = receipt.replace(
            receipt_integrity_tag=jnp.asarray(0, dtype=jnp.uint32)
        )
        expected = _content_tag(unsigned, salt=_DYNA_RECEIPT_SALT)
        return (
            (receipt.receipt_integrity_tag != 0)
            & (receipt.receipt_integrity_tag == expected)
            & ~receipt.planning_authority
            & ~receipt.safety_authority
        )

    @staticmethod
    def _short_rollout_receipt_integrity_valid(
        receipt: ShortRolloutPlannerReadinessReceipt,
    ) -> Array:
        unsigned = receipt.replace(
            receipt_integrity_tag=jnp.asarray(0, dtype=jnp.uint32)
        )
        expected = _content_tag(unsigned, salt=_ROLLOUT_RECEIPT_SALT)
        return (
            (receipt.receipt_integrity_tag != 0)
            & (receipt.receipt_integrity_tag == expected)
            & ~receipt.planning_authority
            & ~receipt.safety_authority
        )

    @staticmethod
    def _seal_dyna_receipt(
        receipt: DynaPlannerReadinessReceipt,
    ) -> DynaPlannerReadinessReceipt:
        return receipt.replace(
            receipt_integrity_tag=_content_tag(receipt, salt=_DYNA_RECEIPT_SALT)
        )

    @staticmethod
    def _seal_short_rollout_receipt(
        receipt: ShortRolloutPlannerReadinessReceipt,
    ) -> ShortRolloutPlannerReadinessReceipt:
        return receipt.replace(
            receipt_integrity_tag=_content_tag(receipt, salt=_ROLLOUT_RECEIPT_SALT)
        )

    def _zero_dyna_receipt(
        self,
        state: WorldModelPlannerReadinessState,
        calibration_state: WorldModelRegionCalibrationState,
    ) -> DynaPlannerReadinessReceipt:
        budget = self._dyna.config.backup_budget
        words = jnp.zeros((budget, 2), dtype=jnp.uint32)
        tags = jnp.zeros((budget,), dtype=jnp.uint32)
        flags = jnp.zeros((budget,), dtype=jnp.bool_)
        receipt = DynaPlannerReadinessReceipt(
            lifecycle_id_words=state.lifecycle_id_words,
            owner_execution_count_words=state.dyna_execution_count_words,
            calibration_revision_words=calibration_state.accepted_count_words,
            calibration_state_content_tag=self._calibrator.content_tag(
                calibration_state
            ),
            current_model_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
            current_representation_revision_words=jnp.zeros(
                (2,), dtype=jnp.uint32
            ),
            selected_anchor_indices=jnp.full((budget,), -1, dtype=jnp.int32),
            selected_actions=jnp.full((budget,), -1, dtype=jnp.int32),
            selected_decision_id_words=words,
            selected_anchor_model_revision_words=words,
            region_ids=jnp.zeros((budget,), dtype=jnp.int32),
            action_revision_words=words,
            region_revision_words=words,
            calibration_cell_revision_words=words,
            calibration_cell_content_tags=tags,
            calibration_prediction_content_tags=tags,
            calibration_receipt_integrity_tags=tags,
            gate_available=jnp.zeros((budget, 5), dtype=jnp.bool_),
            gate_passed=jnp.zeros((budget, 5), dtype=jnp.bool_),
            aleatoric_noise_vetoed=flags,
            legacy_guard_passed=flags,
            calibration_gate_passed=flags,
            combined_gate_passed=flags,
            descriptive_all_required_eligible=jnp.asarray(False),
            receipt_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            planning_authority=jnp.asarray(False),
            safety_authority=jnp.asarray(False),
            valid=jnp.asarray(False),
        )
        return self._seal_dyna_receipt(receipt)

    def _zero_short_rollout_receipt(
        self,
        state: WorldModelPlannerReadinessState,
        calibration_state: WorldModelRegionCalibrationState,
    ) -> ShortRolloutPlannerReadinessReceipt:
        budget = self._rollout.config.rollout_budget
        horizon = self._rollout.config.rollout_horizon
        steps = (budget, horizon)
        revisions = (budget, horizon, 2)
        flags = jnp.zeros(steps, dtype=jnp.bool_)
        receipt = ShortRolloutPlannerReadinessReceipt(
            lifecycle_id_words=state.lifecycle_id_words,
            owner_execution_count_words=state.rollout_execution_count_words,
            calibration_revision_words=calibration_state.accepted_count_words,
            calibration_state_content_tag=self._calibrator.content_tag(
                calibration_state
            ),
            model_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
            representation_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
            decision_id_words=jnp.zeros((2,), dtype=jnp.uint32),
            selected_actions=jnp.zeros(steps, dtype=jnp.int32),
            region_ids=jnp.zeros(steps, dtype=jnp.int32),
            action_revision_words=jnp.zeros(revisions, dtype=jnp.uint32),
            region_revision_words=jnp.zeros(revisions, dtype=jnp.uint32),
            calibration_cell_revision_words=jnp.zeros(
                revisions, dtype=jnp.uint32
            ),
            calibration_cell_content_tags=jnp.zeros(steps, dtype=jnp.uint32),
            calibration_prediction_content_tags=jnp.zeros(
                steps, dtype=jnp.uint32
            ),
            calibration_receipt_integrity_tags=jnp.zeros(
                steps, dtype=jnp.uint32
            ),
            gate_available=jnp.zeros((*steps, 5), dtype=jnp.bool_),
            gate_passed=jnp.zeros((*steps, 5), dtype=jnp.bool_),
            aleatoric_noise_vetoed=flags,
            legacy_guard_passed=flags,
            calibration_gate_passed=flags,
            combined_gate_passed=flags,
            prefix_eligible=flags,
            path_eligible=jnp.zeros((budget,), dtype=jnp.bool_),
            descriptive_all_required_eligible=jnp.asarray(False),
            receipt_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            planning_authority=jnp.asarray(False),
            safety_authority=jnp.asarray(False),
            valid=jnp.asarray(False),
        )
        return self._seal_short_rollout_receipt(receipt)

    def _build_dyna_receipt(
        self,
        state: WorldModelPlannerReadinessState,
        calibration_state: WorldModelRegionCalibrationState,
        dyna_state: OneStepDynaState,
        model_state: WorldModelEnsembleState,
        authority: OneStepDynaAuthority,
        legacy: Any,
        *,
        region_ids: Array,
        action_revision_words: Array,
        region_revision_words: Array,
    ) -> DynaPlannerReadinessReceipt:
        budget = self._dyna.config.backup_budget
        cell_revisions: list[Array] = []
        cell_tags: list[Array] = []
        prediction_tags: list[Array] = []
        receipt_tags: list[Array] = []
        available_rows: list[Array] = []
        passed_rows: list[Array] = []
        vetoed: list[Array] = []
        calibration_passed: list[Array] = []
        child_valid: list[Array] = []
        for index in range(budget):
            anchor_index = jnp.clip(
                legacy.diagnostics.selected_anchor_indices[index],
                0,
                self._dyna.config.anchor_capacity - 1,
            )
            action = legacy.diagnostics.selected_actions[index]
            safe_action = jnp.clip(action, 0, self._dyna.n_primitive_actions - 1)
            prediction = self._dyna.ensemble.predict(
                model_state,
                dyna_state.anchor_observations[anchor_index],
                safe_action,
            )
            child = self._prediction_receipt(
                calibration_state,
                decision_id_words=legacy.diagnostics.selected_decision_id_words[
                    index
                ],
                model_revision_words=authority.model_revision_words,
                representation_revision_words=(
                    dyna_state.anchor_representation_revision_words[anchor_index]
                ),
                action_revision_words=action_revision_words[index],
                region_revision_words=region_revision_words[index],
                action=action,
                region=region_ids[index],
                prediction=prediction,
            )
            available, passed, noise_veto, cal_pass = self._gate_arrays(child)
            cell_revisions.append(child.cell_revision_words)
            cell_tags.append(child.calibration_content_tag)
            prediction_tags.append(child.prediction_content_tag)
            receipt_tags.append(child.receipt_integrity_tag)
            available_rows.append(available)
            passed_rows.append(passed)
            vetoed.append(noise_veto)
            calibration_passed.append(cal_pass)
            child_valid.append(child.valid)

        legacy_guard = legacy.diagnostics.guard_passed
        calibration_gate = jnp.stack(calibration_passed)
        combined = legacy_guard & calibration_gate
        all_required = jnp.all(~legacy_guard | calibration_gate)
        next_count, count_valid = _checked_words_increment(
            state.dyna_execution_count_words
        )
        monotonic, alias_valid = self._calibration_alias_valid(
            state,
            calibration_state,
        )
        valid = (
            self._state_valid(state)
            & self._calibrator.state_valid(calibration_state)
            & monotonic
            & alias_valid
            & legacy.diagnostics.transaction_applied
            & jnp.all(jnp.stack(child_valid))
            & count_valid
            & _words_leq_limit(next_count, self._config.max_dyna_executions)
        )
        receipt = DynaPlannerReadinessReceipt(
            lifecycle_id_words=state.lifecycle_id_words,
            owner_execution_count_words=state.dyna_execution_count_words,
            calibration_revision_words=calibration_state.accepted_count_words,
            calibration_state_content_tag=self._calibrator.content_tag(
                calibration_state
            ),
            current_model_revision_words=authority.model_revision_words,
            current_representation_revision_words=(
                authority.representation_revision_words
            ),
            selected_anchor_indices=legacy.diagnostics.selected_anchor_indices,
            selected_actions=legacy.diagnostics.selected_actions,
            selected_decision_id_words=(
                legacy.diagnostics.selected_decision_id_words
            ),
            selected_anchor_model_revision_words=(
                legacy.diagnostics.selected_anchor_model_revision_words
            ),
            region_ids=region_ids,
            action_revision_words=action_revision_words,
            region_revision_words=region_revision_words,
            calibration_cell_revision_words=jnp.stack(cell_revisions),
            calibration_cell_content_tags=jnp.stack(cell_tags),
            calibration_prediction_content_tags=jnp.stack(prediction_tags),
            calibration_receipt_integrity_tags=jnp.stack(receipt_tags),
            gate_available=jnp.stack(available_rows),
            gate_passed=jnp.stack(passed_rows),
            aleatoric_noise_vetoed=jnp.stack(vetoed),
            legacy_guard_passed=legacy_guard,
            calibration_gate_passed=calibration_gate,
            combined_gate_passed=combined,
            descriptive_all_required_eligible=all_required,
            receipt_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            planning_authority=jnp.asarray(False),
            safety_authority=jnp.asarray(False),
            valid=valid,
        )
        return self._seal_dyna_receipt(receipt)

    def prepare_dyna(
        self,
        state: WorldModelPlannerReadinessState,
        calibration_state: WorldModelRegionCalibrationState,
        dyna_state: OneStepDynaState,
        model_state: WorldModelEnsembleState,
        control_state: MultiHeadMLPState,
        authority: OneStepDynaAuthority,
        *,
        region_ids: Array,
        action_revision_words: Array,
        region_revision_words: Array,
    ) -> DynaPlannerReadinessReceipt:
        """Functionally evaluate unchanged Dyna and bind calibration cells."""

        if not self._state_static_valid(state):
            raise TypeError("planner-readiness state has the wrong static contract")
        budget = self._dyna.config.backup_budget
        regions = _require_array(
            region_ids,
            name="region_ids",
            shape=(budget,),
            dtype=jnp.int32,
        )
        action_revisions = _require_array(
            action_revision_words,
            name="action_revision_words",
            shape=(budget, 2),
            dtype=jnp.uint32,
        )
        region_revisions = _require_array(
            region_revision_words,
            name="region_revision_words",
            shape=(budget, 2),
            dtype=jnp.uint32,
        )
        legacy = self._dyna.plan(
            dyna_state,
            model_state,
            control_state,
            authority,
        )
        return self._build_dyna_receipt(
            state,
            calibration_state,
            dyna_state,
            model_state,
            authority,
            legacy,
            region_ids=regions,
            action_revision_words=action_revisions,
            region_revision_words=region_revisions,
        )

    def _candidate_dyna_state(
        self,
        state: WorldModelPlannerReadinessState,
        calibration_state: WorldModelRegionCalibrationState,
        next_count: Array,
        receipt_tag: Array,
    ) -> WorldModelPlannerReadinessState:
        return state.replace(
            bound_calibration_revision_words=(
                calibration_state.accepted_count_words
            ),
            bound_calibration_content_tag=self._calibrator.content_tag(
                calibration_state
            ),
            dyna_execution_count_words=next_count,
            last_dyna_receipt_integrity_tag=receipt_tag,
        )

    def _zero_execution_diagnostics(
        self,
        *,
        state: WorldModelPlannerReadinessState,
        calibration_state_valid: Array,
        legacy_transaction_evaluated: Array,
        lane_count_words: Array,
    ) -> PlannerReadinessExecutionDiagnostics:
        false = jnp.asarray(False, dtype=jnp.bool_)
        owner_valid = (
            self._state_valid(state)
            if self._state_static_valid(state)
            else false
        )
        return PlannerReadinessExecutionDiagnostics(
            owner_state_valid=owner_valid,
            calibration_state_valid=calibration_state_valid,
            receipt_static_contract_valid=false,
            receipt_content_valid=false,
            calibration_revision_monotonic=false,
            calibration_content_alias_valid=false,
            execution_capacity_available=false,
            legacy_transaction_evaluated=legacy_transaction_evaluated,
            legacy_gates_preserved=false,
            calibration_is_additional_conjunction=false,
            prefix_closed=false,
            planning_authority=false,
            safety_authority=false,
            candidate_state_valid=false,
            applied=false,
            rejected=jnp.asarray(True, dtype=jnp.bool_),
            pre_execution_count_words=lane_count_words,
            post_execution_count_words=lane_count_words,
        )

    def execute_dyna(
        self,
        state: WorldModelPlannerReadinessState,
        calibration_state: WorldModelRegionCalibrationState,
        dyna_state: OneStepDynaState,
        model_state: WorldModelEnsembleState,
        control_state: MultiHeadMLPState,
        authority: OneStepDynaAuthority,
        receipt: DynaPlannerReadinessReceipt,
    ) -> CalibratedDynaExecutionResult:
        """Commit the untouched Dyna result only under the added conjunction."""

        if not self._state_static_valid(state):
            raise TypeError("planner-readiness state has the wrong static contract")
        legacy = self._dyna.plan(
            dyna_state,
            model_state,
            control_state,
            authority,
        )
        calibration_valid = self._calibrator.state_valid(calibration_state)
        if not self._dyna_receipt_static_valid(receipt):
            zero_receipt = self._zero_dyna_receipt(state, calibration_state)
            diagnostics = self._zero_execution_diagnostics(
                state=state,
                calibration_state_valid=calibration_valid,
                legacy_transaction_evaluated=legacy.diagnostics.transaction_applied,
                lane_count_words=state.dyna_execution_count_words,
            )
            return CalibratedDynaExecutionResult(
                readiness_state=state,
                dyna_state=dyna_state,
                control_state=control_state,
                receipt=zero_receipt,
                legacy_diagnostics=legacy.diagnostics,
                diagnostics=diagnostics,
            )

        expected = self._build_dyna_receipt(
            state,
            calibration_state,
            dyna_state,
            model_state,
            authority,
            legacy,
            region_ids=receipt.region_ids,
            action_revision_words=receipt.action_revision_words,
            region_revision_words=receipt.region_revision_words,
        )
        owner_valid = self._state_valid(state)
        monotonic, alias_valid = self._calibration_alias_valid(
            state,
            calibration_state,
        )
        next_count, count_valid = _checked_words_increment(
            state.dyna_execution_count_words
        )
        capacity = count_valid & _words_leq_limit(
            next_count,
            self._config.max_dyna_executions,
        )
        content_valid = (
            receipt.valid
            & expected.valid
            & self._dyna_receipt_integrity_valid(receipt)
            & _tree_equal(receipt, expected)
        )
        legacy_preserved = jnp.array_equal(
            expected.legacy_guard_passed,
            legacy.diagnostics.guard_passed,
        )
        additional_conjunction = jnp.array_equal(
            expected.combined_gate_passed,
            expected.legacy_guard_passed
            & expected.calibration_gate_passed,
        )
        candidate = self._candidate_dyna_state(
            state,
            calibration_state,
            next_count,
            expected.receipt_integrity_tag,
        )
        candidate_valid = self._state_valid(candidate)
        applied = (
            owner_valid
            & calibration_valid
            & content_valid
            & monotonic
            & alias_valid
            & capacity
            & legacy.diagnostics.transaction_applied
            & legacy_preserved
            & additional_conjunction
            & expected.descriptive_all_required_eligible
            & candidate_valid
        )
        next_readiness = cast(
            WorldModelPlannerReadinessState,
            jax.lax.cond(applied, lambda _: candidate, lambda _: state, None),
        )
        next_dyna = cast(
            OneStepDynaState,
            jax.lax.cond(
                applied,
                lambda _: legacy.state,
                lambda _: dyna_state,
                None,
            ),
        )
        next_control = cast(
            MultiHeadMLPState,
            jax.lax.cond(
                applied,
                lambda _: legacy.control_state,
                lambda _: control_state,
                None,
            ),
        )
        diagnostics = PlannerReadinessExecutionDiagnostics(
            owner_state_valid=owner_valid,
            calibration_state_valid=calibration_valid,
            receipt_static_contract_valid=jnp.asarray(True),
            receipt_content_valid=content_valid,
            calibration_revision_monotonic=monotonic,
            calibration_content_alias_valid=alias_valid,
            execution_capacity_available=capacity,
            legacy_transaction_evaluated=legacy.diagnostics.transaction_applied,
            legacy_gates_preserved=legacy_preserved,
            calibration_is_additional_conjunction=additional_conjunction,
            prefix_closed=jnp.asarray(True),
            planning_authority=jnp.asarray(False),
            safety_authority=jnp.asarray(False),
            candidate_state_valid=candidate_valid,
            applied=applied,
            rejected=~applied,
            pre_execution_count_words=state.dyna_execution_count_words,
            post_execution_count_words=next_readiness.dyna_execution_count_words,
        )
        return CalibratedDynaExecutionResult(
            readiness_state=next_readiness,
            dyna_state=next_dyna,
            control_state=next_control,
            receipt=receipt,
            legacy_diagnostics=legacy.diagnostics,
            diagnostics=diagnostics,
        )

    def _build_short_rollout_receipt(
        self,
        state: WorldModelPlannerReadinessState,
        calibration_state: WorldModelRegionCalibrationState,
        model_state: WorldModelEnsembleState,
        authority: RolloutPolicyValueAuthority,
        anchor: RealStateRolloutAnchor,
        legacy: Any,
        *,
        region_ids: Array,
        action_revision_words: Array,
        region_revision_words: Array,
    ) -> ShortRolloutPlannerReadinessReceipt:
        budget = self._rollout.config.rollout_budget
        horizon = self._rollout.config.rollout_horizon
        cell_revision_rows: list[Array] = []
        cell_tag_rows: list[Array] = []
        prediction_tag_rows: list[Array] = []
        receipt_tag_rows: list[Array] = []
        available_path_rows: list[Array] = []
        passed_path_rows: list[Array] = []
        veto_path_rows: list[Array] = []
        calibration_path_rows: list[Array] = []
        child_valid_path_rows: list[Array] = []

        for path_index in range(budget):
            observation = anchor.observation
            cell_revisions: list[Array] = []
            cell_tags: list[Array] = []
            prediction_tags: list[Array] = []
            receipt_tags: list[Array] = []
            available_rows: list[Array] = []
            passed_rows: list[Array] = []
            vetoed: list[Array] = []
            calibration_passed: list[Array] = []
            child_valid: list[Array] = []
            for step_index in range(horizon):
                action = legacy.diagnostics.selected_actions[
                    path_index,
                    step_index,
                ]
                safe_action = jnp.clip(action, 0, self._rollout.n_actions - 1)
                prediction = self._rollout.ensemble.predict(
                    model_state,
                    observation,
                    safe_action,
                )
                child = self._prediction_receipt(
                    calibration_state,
                    decision_id_words=anchor.decision_id_words,
                    model_revision_words=authority.model_revision_words,
                    representation_revision_words=authority.source_revision_words,
                    action_revision_words=action_revision_words[
                        path_index,
                        step_index,
                    ],
                    region_revision_words=region_revision_words[
                        path_index,
                        step_index,
                    ],
                    action=action,
                    region=region_ids[path_index, step_index],
                    prediction=prediction,
                )
                available, passed, noise_veto, cal_pass = self._gate_arrays(child)
                cell_revisions.append(child.cell_revision_words)
                cell_tags.append(child.calibration_content_tag)
                prediction_tags.append(child.prediction_content_tag)
                receipt_tags.append(child.receipt_integrity_tag)
                available_rows.append(available)
                passed_rows.append(passed)
                vetoed.append(noise_veto)
                calibration_passed.append(cal_pass)
                child_valid.append(child.valid)
                observation = jnp.where(
                    legacy.diagnostics.guard_passed[path_index, step_index],
                    prediction.mean_next_observation,
                    observation,
                )

            cell_revision_rows.append(jnp.stack(cell_revisions))
            cell_tag_rows.append(jnp.stack(cell_tags))
            prediction_tag_rows.append(jnp.stack(prediction_tags))
            receipt_tag_rows.append(jnp.stack(receipt_tags))
            available_path_rows.append(jnp.stack(available_rows))
            passed_path_rows.append(jnp.stack(passed_rows))
            veto_path_rows.append(jnp.stack(vetoed))
            calibration_path_rows.append(jnp.stack(calibration_passed))
            child_valid_path_rows.append(jnp.stack(child_valid))

        legacy_guard = legacy.diagnostics.guard_passed
        calibration_gate = jnp.stack(calibration_path_rows)
        prefix_rows: list[Array] = []
        for path_index in range(budget):
            prefix = jnp.asarray(True, dtype=jnp.bool_)
            step_eligibility: list[Array] = []
            for step_index in range(horizon):
                current = prefix & calibration_gate[path_index, step_index]
                step_eligibility.append(current)
                prefix = prefix & (
                    ~legacy_guard[path_index, step_index]
                    | calibration_gate[path_index, step_index]
                )
            prefix_rows.append(jnp.stack(step_eligibility))
        prefix_eligible = jnp.stack(prefix_rows)
        combined = legacy_guard & prefix_eligible
        path_eligible = jnp.all(~legacy_guard | combined, axis=1)
        all_required = jnp.all(path_eligible)
        next_count, count_valid = _checked_words_increment(
            state.rollout_execution_count_words
        )
        monotonic, alias_valid = self._calibration_alias_valid(
            state,
            calibration_state,
        )
        valid = (
            self._state_valid(state)
            & self._calibrator.state_valid(calibration_state)
            & monotonic
            & alias_valid
            & legacy.diagnostics.transaction_applied
            & jnp.all(jnp.stack(child_valid_path_rows))
            & count_valid
            & _words_leq_limit(next_count, self._config.max_rollout_executions)
        )
        receipt = ShortRolloutPlannerReadinessReceipt(
            lifecycle_id_words=state.lifecycle_id_words,
            owner_execution_count_words=state.rollout_execution_count_words,
            calibration_revision_words=calibration_state.accepted_count_words,
            calibration_state_content_tag=self._calibrator.content_tag(
                calibration_state
            ),
            model_revision_words=authority.model_revision_words,
            representation_revision_words=authority.source_revision_words,
            decision_id_words=anchor.decision_id_words,
            selected_actions=legacy.diagnostics.selected_actions,
            region_ids=region_ids,
            action_revision_words=action_revision_words,
            region_revision_words=region_revision_words,
            calibration_cell_revision_words=jnp.stack(cell_revision_rows),
            calibration_cell_content_tags=jnp.stack(cell_tag_rows),
            calibration_prediction_content_tags=jnp.stack(prediction_tag_rows),
            calibration_receipt_integrity_tags=jnp.stack(receipt_tag_rows),
            gate_available=jnp.stack(available_path_rows),
            gate_passed=jnp.stack(passed_path_rows),
            aleatoric_noise_vetoed=jnp.stack(veto_path_rows),
            legacy_guard_passed=legacy_guard,
            calibration_gate_passed=calibration_gate,
            combined_gate_passed=combined,
            prefix_eligible=prefix_eligible,
            path_eligible=path_eligible,
            descriptive_all_required_eligible=all_required,
            receipt_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            planning_authority=jnp.asarray(False),
            safety_authority=jnp.asarray(False),
            valid=valid,
        )
        return self._seal_short_rollout_receipt(receipt)

    def prepare_short_rollout(
        self,
        state: WorldModelPlannerReadinessState,
        calibration_state: WorldModelRegionCalibrationState,
        rollout_state: EnsembleShortRolloutState,
        model_state: WorldModelEnsembleState,
        authority: RolloutPolicyValueAuthority,
        anchor: RealStateRolloutAnchor,
        *,
        region_ids: Array,
        action_revision_words: Array,
        region_revision_words: Array,
    ) -> ShortRolloutPlannerReadinessReceipt:
        """Functionally evaluate unchanged rollout and bind every step cell."""

        if not self._state_static_valid(state):
            raise TypeError("planner-readiness state has the wrong static contract")
        budget = self._rollout.config.rollout_budget
        horizon = self._rollout.config.rollout_horizon
        regions = _require_array(
            region_ids,
            name="region_ids",
            shape=(budget, horizon),
            dtype=jnp.int32,
        )
        action_revisions = _require_array(
            action_revision_words,
            name="action_revision_words",
            shape=(budget, horizon, 2),
            dtype=jnp.uint32,
        )
        region_revisions = _require_array(
            region_revision_words,
            name="region_revision_words",
            shape=(budget, horizon, 2),
            dtype=jnp.uint32,
        )
        legacy = self._rollout.propose(
            rollout_state,
            model_state,
            authority,
            anchor,
        )
        return self._build_short_rollout_receipt(
            state,
            calibration_state,
            model_state,
            authority,
            anchor,
            legacy,
            region_ids=regions,
            action_revision_words=action_revisions,
            region_revision_words=region_revisions,
        )

    def _candidate_rollout_state(
        self,
        state: WorldModelPlannerReadinessState,
        calibration_state: WorldModelRegionCalibrationState,
        next_count: Array,
        receipt_tag: Array,
    ) -> WorldModelPlannerReadinessState:
        return state.replace(
            bound_calibration_revision_words=(
                calibration_state.accepted_count_words
            ),
            bound_calibration_content_tag=self._calibrator.content_tag(
                calibration_state
            ),
            rollout_execution_count_words=next_count,
            last_rollout_receipt_integrity_tag=receipt_tag,
        )

    def execute_short_rollout(
        self,
        state: WorldModelPlannerReadinessState,
        calibration_state: WorldModelRegionCalibrationState,
        rollout_state: EnsembleShortRolloutState,
        model_state: WorldModelEnsembleState,
        authority: RolloutPolicyValueAuthority,
        anchor: RealStateRolloutAnchor,
        receipt: ShortRolloutPlannerReadinessReceipt,
    ) -> CalibratedShortRolloutExecutionResult:
        """Commit an untouched rollout only when every required prefix passes."""

        if not self._state_static_valid(state):
            raise TypeError("planner-readiness state has the wrong static contract")
        legacy = self._rollout.propose(
            rollout_state,
            model_state,
            authority,
            anchor,
        )
        calibration_valid = self._calibrator.state_valid(calibration_state)
        zero_proposals = cast(
            ImaginedRolloutBatch,
            jax.tree.map(jnp.zeros_like, legacy.proposals),
        )
        if not self._short_rollout_receipt_static_valid(receipt):
            zero_receipt = self._zero_short_rollout_receipt(
                state,
                calibration_state,
            )
            diagnostics = self._zero_execution_diagnostics(
                state=state,
                calibration_state_valid=calibration_valid,
                legacy_transaction_evaluated=legacy.diagnostics.transaction_applied,
                lane_count_words=state.rollout_execution_count_words,
            )
            return CalibratedShortRolloutExecutionResult(
                readiness_state=state,
                rollout_state=rollout_state,
                proposals=zero_proposals,
                receipt=zero_receipt,
                legacy_diagnostics=legacy.diagnostics,
                diagnostics=diagnostics,
            )

        expected = self._build_short_rollout_receipt(
            state,
            calibration_state,
            model_state,
            authority,
            anchor,
            legacy,
            region_ids=receipt.region_ids,
            action_revision_words=receipt.action_revision_words,
            region_revision_words=receipt.region_revision_words,
        )
        owner_valid = self._state_valid(state)
        monotonic, alias_valid = self._calibration_alias_valid(
            state,
            calibration_state,
        )
        next_count, count_valid = _checked_words_increment(
            state.rollout_execution_count_words
        )
        capacity = count_valid & _words_leq_limit(
            next_count,
            self._config.max_rollout_executions,
        )
        content_valid = (
            receipt.valid
            & expected.valid
            & self._short_rollout_receipt_integrity_valid(receipt)
            & _tree_equal(receipt, expected)
        )
        legacy_preserved = jnp.array_equal(
            expected.legacy_guard_passed,
            legacy.diagnostics.guard_passed,
        )
        additional_conjunction = jnp.array_equal(
            expected.combined_gate_passed,
            expected.legacy_guard_passed & expected.prefix_eligible,
        )
        candidate = self._candidate_rollout_state(
            state,
            calibration_state,
            next_count,
            expected.receipt_integrity_tag,
        )
        candidate_valid = self._state_valid(candidate)
        applied = (
            owner_valid
            & calibration_valid
            & content_valid
            & monotonic
            & alias_valid
            & capacity
            & legacy.diagnostics.transaction_applied
            & legacy_preserved
            & additional_conjunction
            & expected.descriptive_all_required_eligible
            & candidate_valid
        )
        next_readiness = cast(
            WorldModelPlannerReadinessState,
            jax.lax.cond(applied, lambda _: candidate, lambda _: state, None),
        )
        next_rollout = cast(
            EnsembleShortRolloutState,
            jax.lax.cond(
                applied,
                lambda _: legacy.state,
                lambda _: rollout_state,
                None,
            ),
        )
        next_proposals = cast(
            ImaginedRolloutBatch,
            jax.lax.cond(
                applied,
                lambda _: legacy.proposals,
                lambda _: zero_proposals,
                None,
            ),
        )
        diagnostics = PlannerReadinessExecutionDiagnostics(
            owner_state_valid=owner_valid,
            calibration_state_valid=calibration_valid,
            receipt_static_contract_valid=jnp.asarray(True),
            receipt_content_valid=content_valid,
            calibration_revision_monotonic=monotonic,
            calibration_content_alias_valid=alias_valid,
            execution_capacity_available=capacity,
            legacy_transaction_evaluated=legacy.diagnostics.transaction_applied,
            legacy_gates_preserved=legacy_preserved,
            calibration_is_additional_conjunction=additional_conjunction,
            prefix_closed=jnp.asarray(True),
            planning_authority=jnp.asarray(False),
            safety_authority=jnp.asarray(False),
            candidate_state_valid=candidate_valid,
            applied=applied,
            rejected=~applied,
            pre_execution_count_words=state.rollout_execution_count_words,
            post_execution_count_words=(
                next_readiness.rollout_execution_count_words
            ),
        )
        return CalibratedShortRolloutExecutionResult(
            readiness_state=next_readiness,
            rollout_state=next_rollout,
            proposals=next_proposals,
            receipt=receipt,
            legacy_diagnostics=legacy.diagnostics,
            diagnostics=diagnostics,
        )

    @property
    def resource_budget(self) -> WorldModelPlannerReadinessResourceBudget:
        """Return exact sidecar/receipt sizes and fixed child-call ceilings."""

        calibration_state = self._calibrator.init(
            jnp.asarray((0, 1), dtype=jnp.uint32)
        )
        state = self._empty_state(calibration_state)
        dyna_receipt = self._zero_dyna_receipt(state, calibration_state)
        rollout_receipt = self._zero_short_rollout_receipt(
            state,
            calibration_state,
        )
        state_scalars, state_bytes = _logical_tree_size(state)
        dyna_scalars, dyna_bytes = _logical_tree_size(dyna_receipt)
        rollout_scalars, rollout_bytes = _logical_tree_size(rollout_receipt)
        return WorldModelPlannerReadinessResourceBudget(
            persistent_state_scalars=state_scalars,
            persistent_state_bytes=state_bytes,
            dyna_receipt_scalars=dyna_scalars,
            dyna_receipt_bytes=dyna_bytes,
            rollout_receipt_scalars=rollout_scalars,
            rollout_receipt_bytes=rollout_bytes,
            max_calibration_receipts_per_dyna_call=(
                self._dyna.config.backup_budget
            ),
            max_calibration_receipts_per_rollout_call=(
                self._rollout.config.rollout_budget
                * self._rollout.config.rollout_horizon
            ),
            max_legacy_planner_evaluations_per_prepare=1,
            max_legacy_planner_evaluations_per_execute=1,
            max_dyna_executions=self._config.max_dyna_executions,
            max_rollout_executions=self._config.max_rollout_executions,
            model_state_owned=0,
            control_state_owned=0,
            calibration_state_owned=0,
            planner_state_owned=0,
            planning_authority=0,
            safety_authority=0,
            scientific_promotion_allowed=False,
        )


def measure_world_model_planner_readiness_state_nbytes(
    state: WorldModelPlannerReadinessState,
) -> int:
    """Measure every persistent JAX-array leaf in one sidecar state."""

    if type(state) is not WorldModelPlannerReadinessState:
        raise TypeError("state must be an exact WorldModelPlannerReadinessState")
    return _logical_tree_size(state)[1]


def save_world_model_planner_readiness_checkpoint(
    owner: WorldModelPlannerReadiness,
    state: WorldModelPlannerReadinessState,
    path: str | Path,
) -> None:
    """Persist one exact L0 readiness sidecar without any child state."""

    if type(owner) is not WorldModelPlannerReadiness:
        raise TypeError("owner must be an exact WorldModelPlannerReadiness")
    if not bool(jax.device_get(owner.state_valid(state))):
        raise ValueError("refusing to save an invalid planner-readiness state")
    config = owner.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": WORLD_MODEL_PLANNER_READINESS_CHECKPOINT_SCHEMA,
            "owner_config": config,
            "config_sha256": _config_digest(config),
            "resource_budget": owner.resource_budget.to_config(),
            "evidence_level": WORLD_MODEL_PLANNER_READINESS_EVIDENCE_LEVEL,
            "outcome_status": WORLD_MODEL_PLANNER_READINESS_OUTCOME_STATUS,
            "scientific_promotion_allowed": False,
            "readiness_state_included": True,
            "child_states_included": False,
            "model_state_included": False,
            "control_state_included": False,
            "calibration_state_included": False,
            "planner_state_included": False,
            "planning_authority": False,
            "safety_authority": False,
        },
    )


def load_world_model_planner_readiness_checkpoint(
    owner: WorldModelPlannerReadiness,
    path: str | Path,
) -> WorldModelPlannerReadinessState:
    """Strictly restore v1 into the caller's exact current construction."""

    if type(owner) is not WorldModelPlannerReadiness:
        raise TypeError("owner must be an exact WorldModelPlannerReadiness")
    metadata = load_checkpoint_metadata(path)
    expected_fields = {
        "schema",
        "owner_config",
        "config_sha256",
        "resource_budget",
        "evidence_level",
        "outcome_status",
        "scientific_promotion_allowed",
        "readiness_state_included",
        "child_states_included",
        "model_state_included",
        "control_state_included",
        "calibration_state_included",
        "planner_state_included",
        "planning_authority",
        "safety_authority",
    }
    if set(metadata) != expected_fields:
        raise ValueError("planner-readiness checkpoint metadata fields are not exact")
    if metadata.get("schema") != WORLD_MODEL_PLANNER_READINESS_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not a planner-readiness v1 checkpoint")
    config = metadata.get("owner_config")
    if type(config) is not dict:
        raise ValueError("planner-readiness checkpoint lacks exact owner_config")
    if metadata.get("config_sha256") != _config_digest(config):
        raise ValueError("planner-readiness checkpoint config digest does not match")
    if config != owner.to_config():
        raise ValueError("planner-readiness checkpoint owner construction does not match")
    if metadata.get("resource_budget") != owner.resource_budget.to_config():
        raise ValueError("planner-readiness checkpoint resource budget does not match")
    if metadata.get("evidence_level") != WORLD_MODEL_PLANNER_READINESS_EVIDENCE_LEVEL:
        raise ValueError("planner-readiness checkpoint must remain L0")
    if metadata.get("outcome_status") != WORLD_MODEL_PLANNER_READINESS_OUTCOME_STATUS:
        raise ValueError("planner-readiness checkpoint must remain not_assessed")
    if metadata.get("readiness_state_included") is not True:
        raise ValueError("planner-readiness checkpoint must include its sidecar state")
    for name in (
        "scientific_promotion_allowed",
        "child_states_included",
        "model_state_included",
        "control_state_included",
        "calibration_state_included",
        "planner_state_included",
        "planning_authority",
        "safety_authority",
    ):
        if metadata.get(name) is not False:
            raise ValueError(f"planner-readiness checkpoint {name} must be false")
    calibration_template = owner.calibrator.init(
        jnp.asarray((0, 1), dtype=jnp.uint32)
    )
    template = owner._empty_state(calibration_template)
    restored, second_metadata = load_checkpoint(template, path)
    if second_metadata != metadata:
        raise ValueError("planner-readiness checkpoint metadata changed between reads")
    state = cast(WorldModelPlannerReadinessState, restored)
    if not bool(jax.device_get(owner.state_valid(state))):
        raise ValueError("planner-readiness checkpoint restored an invalid state")
    if measure_world_model_planner_readiness_state_nbytes(state) != (
        owner.resource_budget.persistent_state_bytes
    ):
        raise ValueError("planner-readiness checkpoint restored a wrong-size state")
    return state


__all__ = [
    "WORLD_MODEL_PLANNER_CALIBRATION_GATE_NAMES",
    "WORLD_MODEL_PLANNER_READINESS_CHECKPOINT_SCHEMA",
    "WORLD_MODEL_PLANNER_READINESS_CONFIG_SCHEMA",
    "WORLD_MODEL_PLANNER_READINESS_EVIDENCE_LEVEL",
    "WORLD_MODEL_PLANNER_READINESS_OUTCOME_STATUS",
    "WORLD_MODEL_PLANNER_READINESS_SCIENTIFIC_PROMOTION_ALLOWED",
    "CalibratedDynaExecutionResult",
    "CalibratedShortRolloutExecutionResult",
    "DynaPlannerReadinessReceipt",
    "PlannerReadinessExecutionDiagnostics",
    "ShortRolloutPlannerReadinessReceipt",
    "WorldModelPlannerReadiness",
    "WorldModelPlannerReadinessConfig",
    "WorldModelPlannerReadinessResourceBudget",
    "WorldModelPlannerReadinessState",
    "load_world_model_planner_readiness_checkpoint",
    "measure_world_model_planner_readiness_state_nbytes",
    "save_world_model_planner_readiness_checkpoint",
]
