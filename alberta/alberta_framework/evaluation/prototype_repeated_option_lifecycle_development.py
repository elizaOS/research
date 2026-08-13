# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-untyped-call,type-var"
"""Nonwriting schedule-driven development harness for repeated Prototype cycles.

The harness exercises mechanism only.  A versioned, calibration-consumed
deterministic synthetic caller issues exact
but unauthenticated retirement and replacement receipts according to a fixed
two-cycle schedule.  It does not infer a retirement threshold, make a go/no-go
decision, authenticate an external principal, or confer autonomy on the
bridge.  Every control transition has positive discount.  The versioned schedule
uses an explicitly censored execution boundary to make option-use → control-
return → primitive-fallback traces bounded; this is not a natural option-
termination or efficacy claim.

The routed arm attempts two caller-authorized use → retire → replace cycles
through one persistent Prototype→OaK→STOMP owner.  The consumed v1 outcome
completes cycle 0 and then exhausts the scheduler-derived eight-attempt bound
while refreshing the cycle-1 candidate.  It raises a typed fail-closed outcome
that binds the attempt diagnostics, unchanged valid source, completed first
cycle, and an unassessed checkpoint suffix.  Consequently the routing-disabled
opportunity arm and parity replay are not reached, no report is produced, and
no winner or benefit can be inferred.  No method accepts an output path and
this module performs no filesystem or network writes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Any, ClassVar, cast

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from alberta_framework.core.authorized_option_retirement import (
    OptionRetirementAuthorityReceipt,
)
from alberta_framework.core.cumulant_option_installation import CumulantOptionLiveInputs
from alberta_framework.core.cumulant_option_scheduler import (
    CumulantOptionInstallationAuthorityReceipt,
    CumulantOptionSchedulerArmInputs,
    CumulantOptionSchedulerObservation,
    CumulantOptionSchedulerState,
)
from alberta_framework.core.prototype_agent import PrototypeTransition
from alberta_framework.core.prototype_option_authority_bridge import (
    _prototype_oak_state,
    _tree_exact_equal,
)
from alberta_framework.core.prototype_repeated_option_authority_bridge import (
    PrototypeRepeatedOptionAuthorityBridge,
    PrototypeRepeatedOptionAuthorityBridgeResourceBudget,
    PrototypeRepeatedOptionAuthorityBridgeState,
)

PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_CONFIG_SCHEMA = (
    "alberta.prototype-repeated-option-lifecycle-development.config.v1"
)
PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_REPORT_SCHEMA = (
    "alberta.prototype-repeated-option-lifecycle-development.report.v1"
)
PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_ASSESSMENT = "not_assessed"
PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_MECHANISM_STATUS = (
    "l0-schedule-driven-caller-authorized-mechanism-only"
)
PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_OUTCOME_STATUS = (
    "blocked-cycle-1-replacement-attempt-exhaustion"
)
PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_OUTPUT_WRITES = False
PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_CALLER_AUTHENTICATED = False
PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_AUTONOMOUS_SKILL_POLICY = False
PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_BENEFIT_CLAIM = False
PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_EFFICACY_CLAIM = False
PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_WINNER_SELECTION = False
PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_TUNED_THRESHOLD = False
PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_EVIDENCE_AUTHORITY = False
PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_PROMOTION_AUTHORITY = False
PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_SCIENTIFIC_PROMOTION_ALLOWED = False

_FROZEN_INITIAL_OBSERVATION = (-0.25, -0.15, -0.05, 0.05, 0.15, 0.25)
_FROZEN_CONTROL_UPDATE_CAPS = (1, 5)
_FROZEN_CYCLE_SEEDS = (1401, 1402)
_FROZEN_PHASE_ONE_SEEDS = (901, 903)
_FROZEN_PHASE_TWO_SEEDS = (902, 904)
_FROZEN_AUTHORITY_REVISIONS = (1, 2)
_FROZEN_DECLINE_FIRST_ATTEMPT = (True, False)
_FROZEN_OPEN_BLOCKERS = (
    "the consumed v1 schedule exhausts cycle-1 replacement attempts before completion",
    "external retirement/replacement/go-no-go authority is caller-owned",
    "development receipts are integrity-bound but caller authentication is absent",
    "censored execution boundaries do not establish natural option termination",
    "mechanism traces do not establish option benefit or autonomous skill policy",
    "only ordinary control-update opportunities are matched; total routed work is not",
    "the deterministic schedule was calibration-consumed during mechanism debugging",
)


def _tree_sha256_hex(value: object) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(value):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        host = jax.device_get(array)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(memoryview(host).tobytes())
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeRepeatedOptionLifecycleDevelopmentConfig:
    """Versioned calibration-consumed schedule; customization is rejected."""

    initial_observation: tuple[float, ...] = _FROZEN_INITIAL_OBSERVATION
    control_update_caps: tuple[int, int] = _FROZEN_CONTROL_UPDATE_CAPS
    cycle_key_seeds: tuple[int, int] = _FROZEN_CYCLE_SEEDS
    phase_one_key_seeds: tuple[int, int] = _FROZEN_PHASE_ONE_SEEDS
    phase_two_key_seeds: tuple[int, int] = _FROZEN_PHASE_TWO_SEEDS
    retirement_authority_revisions: tuple[int, int] = _FROZEN_AUTHORITY_REVISIONS
    decline_first_replacement_attempt: tuple[bool, bool] = _FROZEN_DECLINE_FIRST_ATTEMPT
    max_replacement_attempts: int = 8
    control_reward: float = -1.0
    control_discount: float = 0.9
    observation_delta: float = 0.05
    censored_execution_boundary: bool = True
    stop_on_first_post_use_primitive_fallback: bool = True
    checkpoint_after_completed_cycles: int = 1

    SCHEMA_VERSION: ClassVar[str] = PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "initial_observation": _FROZEN_INITIAL_OBSERVATION,
            "control_update_caps": _FROZEN_CONTROL_UPDATE_CAPS,
            "cycle_key_seeds": _FROZEN_CYCLE_SEEDS,
            "phase_one_key_seeds": _FROZEN_PHASE_ONE_SEEDS,
            "phase_two_key_seeds": _FROZEN_PHASE_TWO_SEEDS,
            "retirement_authority_revisions": _FROZEN_AUTHORITY_REVISIONS,
            "decline_first_replacement_attempt": _FROZEN_DECLINE_FIRST_ATTEMPT,
            "max_replacement_attempts": 8,
            "control_reward": -1.0,
            "control_discount": 0.9,
            "observation_delta": 0.05,
            "censored_execution_boundary": True,
            "stop_on_first_post_use_primitive_fallback": True,
            "checkpoint_after_completed_cycles": 1,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} is fixed for the v1 development schedule")

    def to_config(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "initial_observation": list(self.initial_observation),
            "control_update_caps": list(self.control_update_caps),
            "cycle_key_seeds": list(self.cycle_key_seeds),
            "phase_one_key_seeds": list(self.phase_one_key_seeds),
            "phase_two_key_seeds": list(self.phase_two_key_seeds),
            "retirement_authority_revisions": list(self.retirement_authority_revisions),
            "decline_first_replacement_attempt": list(self.decline_first_replacement_attempt),
            "max_replacement_attempts": self.max_replacement_attempts,
            "control_reward": self.control_reward,
            "control_discount": self.control_discount,
            "observation_delta": self.observation_delta,
            "censored_execution_boundary": self.censored_execution_boundary,
            "stop_on_first_post_use_primitive_fallback": (
                self.stop_on_first_post_use_primitive_fallback
            ),
            "checkpoint_after_completed_cycles": (self.checkpoint_after_completed_cycles),
            "comparator_match_scope": "ordinary_prototype_stomp_update_opportunities_only",
            "total_work_matched": False,
            "resource_comparability": "not_assessed",
            "calibration_consumed": True,
            "preregistered": False,
            "cycle_one_cap_derivation": "option_budget_plus_one",
            "replacement_attempt_cap_derivation": (
                "scheduler_config.max_install_attempts"
            ),
            "comparator_horizon": "endogenous_to_routed_realized_updates",
            "independently_fixed_comparator": False,
            "outcome_status": (
                PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_OUTCOME_STATUS
            ),
            "assessment": PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_ASSESSMENT,
            "output_writes": False,
            "caller_authenticated": False,
            "autonomous_skill_policy": False,
            "scientific_promotion_allowed": False,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_config(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeRepeatedOptionControlEvent:
    cycle_index: int
    global_control_index: int
    executing_option_before: int
    executing_option_after: int
    primitive_action_before: int
    primitive_action_after: int
    censored_execution_boundary: bool
    option_used: bool
    control_returned: bool
    primitive_fallback: bool
    stomp_update_evaluations: int


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeRepeatedOptionReplacementAttemptDiagnostic:
    """Host-observed facts for one consumed replacement attempt."""

    attempt_index: int
    scheduler_step_before: tuple[int, int]
    scheduler_step_after: tuple[int, int]
    scheduler_observation_count_before: int
    scheduler_observation_count_after: int
    proposal_due: bool
    proposal_ready: bool
    candidate_ready_for_authority: bool
    rejection_reason: str
    selected_candidate_indices: tuple[int, ...]
    selected_descriptors: tuple[tuple[int, int, int, int], ...]
    changed_slots: tuple[bool, ...]
    semantic_generation: int
    source_digest_words: tuple[int, int]
    installed_slot_mask_before: tuple[bool, ...]
    installed_slot_mask_after: tuple[bool, ...]
    cold_slot_mask_before: tuple[bool, ...]
    cold_slot_mask_after: tuple[bool, ...]
    installation_count_before: int
    installation_count_after: int
    scheduler_install_attempts_before: tuple[int, int]
    scheduler_install_attempts_after: tuple[int, int]
    ordinary_advance_applied: bool


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeRepeatedOptionCycleTrace:
    cycle_index: int
    cycle_key_words: tuple[int, int]
    control_events: tuple[PrototypeRepeatedOptionControlEvent, ...]
    option_use_count: int
    control_return_count: int
    primitive_fallback_count: int
    stale_retirement_replay_rejected: bool
    declined_replacement_observed: bool
    stale_replacement_replay_rejected: bool
    fresh_retry_required: bool
    replacement_attempts: int
    completed_cycles_before: int
    completed_cycles_after: int
    retirement_revision_words: tuple[int, int]
    replacement_revision_words: tuple[int, int]
    persistent_stomp_state_owners: int


class ReplacementAttemptExhaustedError(RuntimeError):
    """Typed fail-closed outcome for the consumed v1 replacement bound."""

    __slots__ = (
        "attempt_diagnostics",
        "attempts",
        "benefit_claim",
        "checkpoint_created",
        "checkpoint_suffix_assessment",
        "checkpoint_suffix_parity",
        "completed_cycle_traces",
        "completed_cycles_before",
        "cycle_index",
        "cycle_zero_completed",
        "efficacy_claim",
        "failed_cycle_control_events",
        "promotion_authority",
        "report_produced",
        "scheduler_attempt_cap",
        "source_sha256_after",
        "source_sha256_before",
        "source_state_unchanged",
        "source_state_valid_after",
        "source_state_valid_before",
        "winner_selected",
    )

    def __init__(
        self,
        *,
        cycle_index: int,
        attempts: int,
        scheduler_attempt_cap: int,
        completed_cycles_before: int,
        completed_cycle_traces: tuple[PrototypeRepeatedOptionCycleTrace, ...],
        failed_cycle_control_events: tuple[PrototypeRepeatedOptionControlEvent, ...],
        attempt_diagnostics: tuple[PrototypeRepeatedOptionReplacementAttemptDiagnostic, ...],
        source_state_valid_before: bool,
        source_state_valid_after: bool,
        source_sha256_before: str,
        source_sha256_after: str,
        checkpoint_created: bool,
    ) -> None:
        super().__init__(
            "replacement attempt bound exhausted: "
            f"cycle={cycle_index}, attempts={attempts}, scheduler_cap={scheduler_attempt_cap}"
        )
        self.cycle_index = cycle_index
        self.attempts = attempts
        self.scheduler_attempt_cap = scheduler_attempt_cap
        self.completed_cycles_before = completed_cycles_before
        self.completed_cycle_traces = completed_cycle_traces
        self.failed_cycle_control_events = failed_cycle_control_events
        self.attempt_diagnostics = attempt_diagnostics
        self.source_state_valid_before = source_state_valid_before
        self.source_state_valid_after = source_state_valid_after
        self.source_sha256_before = source_sha256_before
        self.source_sha256_after = source_sha256_after
        self.source_state_unchanged = source_sha256_before == source_sha256_after
        self.cycle_zero_completed = (
            completed_cycles_before == 1
            and len(completed_cycle_traces) == 1
            and completed_cycle_traces[0].cycle_index == 0
            and completed_cycle_traces[0].completed_cycles_after == 1
        )
        self.checkpoint_created = checkpoint_created
        self.checkpoint_suffix_assessment = "not_assessed"
        self.checkpoint_suffix_parity = None
        self.report_produced = False
        self.winner_selected = False
        self.benefit_claim = False
        self.efficacy_claim = False
        self.promotion_authority = False


class _ReplacementAttemptBoundExhaustedError(RuntimeError):
    """Internal cycle-local carrier enriched by ``run`` with source facts."""

    def __init__(
        self,
        *,
        cycle_index: int,
        attempts: int,
        completed_cycles_before: int,
        control_events: tuple[PrototypeRepeatedOptionControlEvent, ...],
        attempt_diagnostics: tuple[PrototypeRepeatedOptionReplacementAttemptDiagnostic, ...],
    ) -> None:
        super().__init__("replacement attempt bound exhausted")
        self.cycle_index = cycle_index
        self.attempts = attempts
        self.completed_cycles_before = completed_cycles_before
        self.control_events = control_events
        self.attempt_diagnostics = attempt_diagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeRepeatedOptionDevelopmentArm:
    lifecycle_routing_enabled: bool
    final_state: PrototypeRepeatedOptionAuthorityBridgeState
    final_state_sha256: str
    control_events: tuple[PrototypeRepeatedOptionControlEvent, ...]
    cycle_traces: tuple[PrototypeRepeatedOptionCycleTrace, ...]
    stomp_update_evaluations: int
    completed_cycles: int
    persistent_stomp_state_owners: int


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeRepeatedOptionLifecycleDevelopmentReport:
    schema_version: str
    config: dict[str, object]
    config_fingerprint: str
    routed: PrototypeRepeatedOptionDevelopmentArm
    lifecycle_routing_disabled: PrototypeRepeatedOptionDevelopmentArm
    checkpoint_suffix_parity: bool
    checkpoint_suffix_nonempty: bool
    same_control_work: bool
    ordinary_control_opportunities_matched: bool
    total_work_matched: bool
    resource_comparability: str
    calibration_consumed: bool
    preregistered: bool
    independently_fixed_comparator: bool
    resource_budget: PrototypeRepeatedOptionAuthorityBridgeResourceBudget
    stage_latency_ns: tuple[tuple[str, int], ...]
    latency_clock: str
    latency_synchronization: str
    latency_warmup_policy: str
    open_blockers: tuple[str, ...]
    mechanism_status: str
    assessment: str
    output_writes: bool
    caller_authenticated: bool
    autonomous_skill_policy: bool
    benefit_claim: bool
    efficacy_claim: bool
    winner_selected: bool
    threshold_tuned: bool
    evidence_authority: bool
    promotion_authority: bool
    scientific_promotion_allowed: bool


class PrototypeRepeatedOptionLifecycleDevelopmentHarness:
    """Host-only fixed-schedule runner over one versioned bridge."""

    def __init__(
        self,
        bridge: PrototypeRepeatedOptionAuthorityBridge,
        config: PrototypeRepeatedOptionLifecycleDevelopmentConfig | None = None,
    ) -> None:
        if type(bridge) is not PrototypeRepeatedOptionAuthorityBridge:
            raise TypeError("bridge must be an exact repeated Prototype bridge")
        self._bridge = bridge
        self._config = config or PrototypeRepeatedOptionLifecycleDevelopmentConfig()
        if self._bridge.lifecycle.config.max_cycles != 2:
            raise ValueError("development harness requires the fixed two-cycle cap")
        option_budget = self._bridge.lifecycle.replacement.scheduler.discovery.config.option_budget
        if self._config.control_update_caps[1] != option_budget + 1:
            raise ValueError("cycle-one control cap must equal option_budget + 1")
        scheduler_attempt_cap = (
            self._bridge.lifecycle.replacement.scheduler.config.max_install_attempts
        )
        if self._config.max_replacement_attempts != scheduler_attempt_cap:
            raise ValueError(
                "replacement attempt cap must equal scheduler max_install_attempts"
            )

    @property
    def config(self) -> PrototypeRepeatedOptionLifecycleDevelopmentConfig:
        return self._config

    def _attached(self, state: PrototypeRepeatedOptionAuthorityBridgeState) -> Any:
        repeated, valid = self._bridge._attach_source(state)
        if not bool(jax.device_get(valid & self._bridge.state_valid(state))):
            raise ValueError("development harness source is not exactly synchronized")
        return repeated

    @staticmethod
    def _words(value: Array) -> tuple[int, int]:
        host = jax.device_get(value)
        return int(host[0]), int(host[1])

    @staticmethod
    def _bool_tuple(value: Array) -> tuple[bool, ...]:
        host = jax.device_get(value)
        return tuple(bool(item) for item in host.tolist())

    @staticmethod
    def _int_tuple(value: Array) -> tuple[int, ...]:
        host = jax.device_get(value)
        return tuple(int(item) for item in host.tolist())

    @staticmethod
    def _descriptor_tuple(value: Array) -> tuple[tuple[int, int, int, int], ...]:
        host = jax.device_get(value)
        return tuple(
            (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
            for row in host.tolist()
        )

    @staticmethod
    def _replacement_rejection_reason(diagnostics: Any) -> str:
        checks = (
            ("source_state_invalid", diagnostics.source_state_valid),
            ("arm_binding_invalid", diagnostics.arm_binding_valid),
            (
                "ordinary_scheduler_transaction_invalid",
                diagnostics.ordinary_scheduler_transaction_valid,
            ),
            ("fallback_state_invalid", diagnostics.fallback_state_valid),
            ("proposal_not_due", diagnostics.proposal_due),
            ("proposal_not_ready", diagnostics.proposal_ready),
            ("proposal_binding_invalid", diagnostics.proposal_binding_valid),
            ("transition_not_fresh", diagnostics.fresh_transition),
            ("cold_slot_cardinality_invalid", diagnostics.exactly_one_cold_slot),
            ("target_not_cold", diagnostics.target_still_cold),
            (
                "semantic_change_mask_mismatch",
                diagnostics.exact_one_slot_semantic_change,
            ),
            (
                "live_slot_semantics_changed",
                diagnostics.live_slots_semantically_preserved,
            ),
            ("not_quiescent", diagnostics.quiescent),
            (
                "scheduler_attempt_capacity_unavailable",
                diagnostics.scheduler_attempt_capacity_available,
            ),
            ("installer_capacity_unavailable", diagnostics.installer_capacity_available),
            ("transaction_invalid", diagnostics.transaction_valid),
        )
        failed = [
            reason
            for reason, passed in checks
            if not bool(jax.device_get(passed))
        ]
        return "+".join(failed) if failed else "ready_for_authority"

    def _control_transition(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
        global_index: int,
    ) -> PrototypeTransition:
        prototype = state.bridge_state.prototype_state
        initial = jnp.asarray(self._config.initial_observation, dtype=jnp.float32)
        successor = initial + jnp.float32(self._config.observation_delta * float(global_index + 1))
        return PrototypeTransition(
            observation=prototype.current_raw_observation,
            action=prototype.current_action,
            decision_id=prototype.current_decision_id,
            reward=jnp.asarray(self._config.control_reward, dtype=jnp.float32),
            discount=jnp.asarray(self._config.control_discount, dtype=jnp.float32),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            truncated=jnp.asarray(
                self._config.censored_execution_boundary,
                dtype=jnp.bool_,
            ),
            next_observation=successor,
            next_decision_observation=successor,
        )

    def _run_control_window(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
        *,
        cycle_index: int,
        first_global_index: int,
        exact_updates: int | None = None,
    ) -> tuple[
        PrototypeRepeatedOptionAuthorityBridgeState,
        tuple[PrototypeRepeatedOptionControlEvent, ...],
    ]:
        current = state
        events: list[PrototypeRepeatedOptionControlEvent] = []
        update_bound = (
            self._config.control_update_caps[cycle_index]
            if exact_updates is None
            else exact_updates
        )
        for offset in range(update_bound):
            global_index = first_global_index + offset
            before_oak = _prototype_oak_state(current.bridge_state.prototype_state.oak_state)
            before_option = int(jax.device_get(before_oak.stomp_state.executing_option))
            before_action = int(jax.device_get(current.bridge_state.prototype_state.current_action))
            result = self._bridge.update_transition(
                current,
                self._control_transition(current, global_index),
            )
            if not bool(jax.device_get(result.transaction_applied)):
                raise RuntimeError("fixed ordinary control transition was refused")
            after_oak = _prototype_oak_state(result.state.bridge_state.prototype_state.oak_state)
            after_option = int(jax.device_get(after_oak.stomp_state.executing_option))
            after_action = int(
                jax.device_get(result.state.bridge_state.prototype_state.current_action)
            )
            option_used = before_option >= 0
            control_returned = option_used and after_option != before_option
            primitive_fallback = after_option < 0
            events.append(
                PrototypeRepeatedOptionControlEvent(
                    cycle_index=cycle_index,
                    global_control_index=global_index,
                    executing_option_before=before_option,
                    executing_option_after=after_option,
                    primitive_action_before=before_action,
                    primitive_action_after=after_action,
                    censored_execution_boundary=bool(
                        jax.device_get(result.bridge.prototype.oak_execution_boundary)
                    ),
                    option_used=option_used,
                    control_returned=control_returned,
                    primitive_fallback=primitive_fallback,
                    stomp_update_evaluations=int(
                        jax.device_get(result.bridge.stomp_update_evaluations)
                    ),
                )
            )
            current = result.state
            if (
                exact_updates is None
                and self._config.stop_on_first_post_use_primitive_fallback
                and option_used
                and control_returned
                and primitive_fallback
            ):
                break
        return current, tuple(events)

    @staticmethod
    def _snapshot(step: int) -> dict[str, Array]:
        value = float(step)
        action = float(step % 2)
        return {
            "raw": jnp.asarray([value, value * value + 0.25], dtype=jnp.float32),
            "event": jnp.asarray([1.0 + 2.0 * action + 0.1 * value], dtype=jnp.float32),
            "atom": jnp.asarray([0.5 + action + 0.2 * value], dtype=jnp.float32),
            "bottleneck": jnp.asarray([2.0 - action + 0.15 * value], dtype=jnp.float32),
            "probe": jnp.asarray([1.0 + 0.1 * value], dtype=jnp.float32),
            "incumbent": jnp.asarray([20.0 + value], dtype=jnp.float32),
            "hand": jnp.arange(4, dtype=jnp.float32) + value,
        }

    @staticmethod
    def _transition_id(step: int) -> Array:
        return jnp.asarray([0xD15C, step], dtype=jnp.uint32)

    def _replacement_inputs(
        self,
        state: CumulantOptionSchedulerState,
        step: int,
    ) -> tuple[
        CumulantOptionSchedulerArmInputs,
        CumulantOptionSchedulerObservation,
        CumulantOptionLiveInputs,
    ]:
        discovery = self._bridge.lifecycle.replacement.scheduler.discovery.config
        if (
            discovery.raw_feature_dim != 2
            or discovery.probe_feature_dim != 1
            or discovery.n_actions != 2
            or discovery.option_budget != 4
        ):
            raise ValueError("fixed replacement fixture requires the 2/1/2/4 test geometry")
        current = self._snapshot(step)
        successor = self._snapshot(step + 1)
        intervention = jnp.zeros((2,), dtype=jnp.bool_).at[step % 2].set(True)
        identity = jnp.asarray(discovery.hand_comparator_identity, dtype=jnp.uint32)
        transition_id = self._transition_id(step)
        generation = state.installation_state.installed_bundle.semantic_generation
        source_digest = state.installation_state.installed_bundle.source_digest
        arm = CumulantOptionSchedulerArmInputs(
            current_raw_features=current["raw"],
            current_raw_available=jnp.ones((2,), dtype=jnp.bool_),
            current_controllable_events=current["event"],
            current_controllable_events_available=jnp.ones((1,), dtype=jnp.bool_),
            current_transition_atoms=current["atom"],
            current_transition_atoms_available=jnp.full((1,), step > 0, dtype=jnp.bool_),
            current_bottleneck_values=current["bottleneck"],
            current_bottleneck_available=jnp.ones((1,), dtype=jnp.bool_),
            probe_features=current["probe"],
            current_incumbent_values=current["incumbent"],
            current_incumbent_available=jnp.ones((1,), dtype=jnp.bool_),
            current_hand_values=current["hand"],
            current_hand_available=jnp.ones((4,), dtype=jnp.bool_),
            hand_comparator_identity=identity,
            reward_base_predictions=jnp.zeros((1,), dtype=jnp.float32),
            model_base_predictions=jnp.zeros((1,), dtype=jnp.float32),
            action=jnp.asarray(step % 2, dtype=jnp.int32),
            behavior_propensity=jnp.asarray(0.5, dtype=jnp.float32),
            randomized=jnp.asarray(True, dtype=jnp.bool_),
            transition_id=transition_id,
            semantic_generation=generation,
            source_digest=source_digest,
        )
        observation = CumulantOptionSchedulerObservation(
            next_raw_features=successor["raw"],
            next_raw_available=jnp.ones((2,), dtype=jnp.bool_),
            next_controllable_events=successor["event"],
            next_controllable_events_available=jnp.ones((1,), dtype=jnp.bool_),
            next_transition_atoms=successor["atom"],
            next_transition_atoms_available=jnp.ones((1,), dtype=jnp.bool_),
            next_bottleneck_values=successor["bottleneck"],
            next_bottleneck_available=jnp.ones((1,), dtype=jnp.bool_),
            bottleneck_epistemic=jnp.asarray([0.5], dtype=jnp.float32),
            bottleneck_progress=jnp.asarray([0.25], dtype=jnp.float32),
            bottleneck_aleatoric=jnp.asarray([0.1], dtype=jnp.float32),
            bottleneck_evidence_available=jnp.ones((1,), dtype=jnp.bool_),
            randomized_action_evidence=intervention,
            next_incumbent_values=successor["incumbent"],
            next_incumbent_available=jnp.ones((1,), dtype=jnp.bool_),
            next_hand_values=successor["hand"],
            next_hand_available=jnp.ones((4,), dtype=jnp.bool_),
            hand_comparator_identity=identity,
            reward_targets=jnp.zeros((1,), dtype=jnp.float32),
            reward_targets_available=jnp.ones((1,), dtype=jnp.bool_),
            model_targets=jnp.zeros((1,), dtype=jnp.float32),
            model_targets_available=jnp.ones((1,), dtype=jnp.bool_),
            transition_id=transition_id,
            semantic_generation=generation,
            source_digest=source_digest,
        )
        live = CumulantOptionLiveInputs(
            raw_features=successor["raw"],
            raw_available=jnp.ones((2,), dtype=jnp.bool_),
            controllable_events=successor["event"],
            controllable_events_available=jnp.ones((1,), dtype=jnp.bool_),
            transition_atoms=successor["atom"],
            transition_atoms_available=jnp.ones((1,), dtype=jnp.bool_),
            bottleneck_values=successor["bottleneck"],
            bottleneck_available=jnp.ones((1,), dtype=jnp.bool_),
            semantic_generation=generation,
            source_digest=source_digest,
            canonical_digest=state.discovery_state.canonical_digest,
            transition_id=transition_id,
            state_observation_count=state.discovery_state.observation_count + 1,
        )
        return arm, observation, live

    @staticmethod
    def _installation_receipt(
        state: CumulantOptionSchedulerState,
        live: CumulantOptionLiveInputs,
        *,
        authorized: bool,
    ) -> CumulantOptionInstallationAuthorityReceipt:
        revision = int(jax.device_get(state.install_applied_words[1])) + 1
        return CumulantOptionInstallationAuthorityReceipt(
            go_no_go_authorized=jnp.asarray(authorized, dtype=jnp.bool_),
            safety_boundary_authorized=jnp.asarray(authorized, dtype=jnp.bool_),
            semantic_generation=live.semantic_generation,
            source_digest=live.source_digest,
            canonical_digest=state.discovery_state.canonical_digest,
            valid_from_step_words=jnp.asarray([0, 0], dtype=jnp.uint32),
            valid_through_step_words=jnp.asarray([0, 16], dtype=jnp.uint32),
            issuer_digest=state.expected_authority_issuer_digest,
            authority_revision_words=jnp.asarray([0, revision], dtype=jnp.uint32),
        )

    @staticmethod
    def _retirement_receipt(
        repeated: Any,
        handoff: Any,
        phase_one_key: Array,
        phase_two_key: Array,
        *,
        revision: int,
    ) -> OptionRetirementAuthorityReceipt:
        state = repeated.cycle_state
        projected = cast(
            Any,
            state,
        )
        installation = projected.scheduler_state.installation_state
        lifecycle = installation.lifecycle_state
        audit = lifecycle.audit_state
        return OptionRetirementAuthorityReceipt(
            retirement_authorized=jnp.asarray(True, dtype=jnp.bool_),
            go_no_go_authorized=jnp.asarray(True, dtype=jnp.bool_),
            safety_boundary_authorized=jnp.asarray(True, dtype=jnp.bool_),
            issuer_digest=projected.expected_retirement_authority_issuer_digest,
            controller_owner_digest=projected.controller_owner_digest,
            authority_revision_words=jnp.asarray([0, revision], dtype=jnp.uint32),
            valid_from_scheduler_step_words=jnp.asarray([0, 1], dtype=jnp.uint32),
            valid_through_scheduler_step_words=jnp.asarray([0, 100], dtype=jnp.uint32),
            scheduler_step_words=handoff.scheduler_step_words,
            descriptor_generation=projected.descriptor_generation,
            descriptor_digest=projected.descriptor_digest,
            discovery_source_digest=installation.installed_bundle.source_digest,
            discovery_canonical_digest=installation.installed_bundle.canonical_digest,
            consumer_source_digest=installation.consumer_source_digest,
            consumer_representation_digest=(installation.consumer_representation_digest),
            lifecycle_id=installation.lifecycle_id,
            installation_revision=installation.revision,
            lifecycle_revision=lifecycle.revision,
            audit_revision=audit.revision,
            controller_revision=projected.controller_revision,
            option_semantic_digests=installation.installed_semantic_digests,
            option_semantic_generations=audit.semantic_generations,
            retirement_slots=handoff.proposed_retirement_slots,
            retirement_mask=handoff.proposed_retirement_mask,
            phase_one_key_data=jr.key_data(phase_one_key),
            phase_two_key_data=jr.key_data(phase_two_key),
        )

    def _run_cycle(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
        *,
        cycle_index: int,
        first_global_index: int,
    ) -> tuple[
        PrototypeRepeatedOptionAuthorityBridgeState,
        PrototypeRepeatedOptionCycleTrace,
    ]:
        before = self._attached(state)
        completed_before = int(jax.device_get(before.completed_cycles))
        controlled, events = self._run_control_window(
            state,
            cycle_index=cycle_index,
            first_global_index=first_global_index,
        )
        if not events or not any(event.option_used for event in events):
            raise RuntimeError("fixed cycle did not exercise an installed option")
        if not any(event.control_returned for event in events):
            raise RuntimeError("fixed cycle did not return control from an option")
        if not any(event.primitive_fallback for event in events):
            raise RuntimeError("fixed cycle did not reach primitive fallback")
        if cycle_index == 0 and not (
            len(events) == 1
            and events[0].option_used
            and events[0].control_returned
            and events[0].primitive_fallback
        ):
            raise RuntimeError(
                "fixed first cycle must use, return, and fall back in its one transition"
            )
        repeated = self._attached(controlled)
        oak = _prototype_oak_state(controlled.bridge_state.prototype_state.oak_state)
        if int(jax.device_get(oak.stomp_state.executing_option)) >= 0:
            raise RuntimeError("fixed cycle is not quiescent for caller retirement")
        scheduler = self._bridge.lifecycle.replacement.scheduler
        scheduler_state = repeated.cycle_state.scheduler_state
        handoff = scheduler._retirement_handoff(
            scheduler_state.discovery_state,
            scheduler_state.installation_state,
            scheduler_state.step_words,
            available=jnp.asarray(True, dtype=jnp.bool_),
        )
        if not bool(jax.device_get(handoff.available & jnp.any(handoff.proposed_retirement_mask))):
            raise RuntimeError("fixed audit did not produce a retirement handoff")
        cycle_key = jr.key(self._config.cycle_key_seeds[cycle_index], impl="threefry2x32")
        phase_one = jr.key(self._config.phase_one_key_seeds[cycle_index], impl="threefry2x32")
        phase_two = jr.key(self._config.phase_two_key_seeds[cycle_index], impl="threefry2x32")
        child_receipt = self._retirement_receipt(
            repeated,
            handoff,
            phase_one,
            phase_two,
            revision=self._config.retirement_authority_revisions[cycle_index],
        )
        receipt = self._bridge.retirement_authority_receipt(
            controlled,
            child_receipt,
            cycle_key,
        )
        prepared = self._bridge.prepare_retirement(
            controlled,
            handoff,
            receipt,
            cycle_key,
            phase_one,
            phase_two,
        )
        retired = self._bridge.commit_retirement(controlled, prepared)
        if not bool(jax.device_get(retired.transaction_applied)):
            raise RuntimeError("fixed retirement transaction was refused")
        stale_retirement = self._bridge.commit_retirement(retired.state, prepared)
        stale_retirement_rejected = not bool(
            jax.device_get(stale_retirement.transaction_applied)
        ) and bool(jax.device_get(_tree_exact_equal(stale_retirement.state, retired.state)))

        current = retired.state
        declined_observed = False
        stale_replacement_rejected = False
        attempts = 0
        completed = False
        attempt_diagnostics: list[PrototypeRepeatedOptionReplacementAttemptDiagnostic] = []
        while attempts < self._config.max_replacement_attempts:
            repeated_current = self._attached(current)
            replacement_state = repeated_current.cycle_state.scheduler_state
            step = int(jax.device_get(replacement_state.step_words[1]))
            arm_inputs, observation, live = self._replacement_inputs(
                replacement_state,
                step,
            )
            arm = self._bridge.arm(current, arm_inputs)
            replacement_prepared = self._bridge.prepare_replacement(
                current,
                arm,
                observation,
                live,
            )
            child_prepared = replacement_prepared.lifecycle_prepared.replacement_prepared
            prepare_diagnostics = child_prepared.diagnostics
            proposal_bundle = child_prepared.scheduler_result.discovery.discovered
            ready = bool(
                jax.device_get(
                    prepare_diagnostics.candidate_ready_for_authority
                )
            )
            force_decline = (
                attempts == 0 and self._config.decline_first_replacement_attempt[cycle_index]
            )
            authorized = ready and not force_decline
            installation_authority = self._installation_receipt(
                replacement_state,
                live,
                authorized=authorized,
            )
            replacement_receipt = self._bridge.replacement_authority_receipt(
                current,
                replacement_prepared,
                installation_authority,
                cycle_key,
                replacement_authorized=authorized,
            )
            replacement = self._bridge.commit_replacement(
                current,
                replacement_prepared,
                replacement_receipt,
                cycle_key,
            )
            attempts += 1
            repeated_after_attempt = self._attached(replacement.state)
            replacement_state_after = repeated_after_attempt.cycle_state.scheduler_state
            attempt_diagnostics.append(
                PrototypeRepeatedOptionReplacementAttemptDiagnostic(
                    attempt_index=attempts,
                    scheduler_step_before=self._words(replacement_state.step_words),
                    scheduler_step_after=self._words(replacement_state_after.step_words),
                    scheduler_observation_count_before=int(
                        jax.device_get(replacement_state.discovery_state.observation_count)
                    ),
                    scheduler_observation_count_after=int(
                        jax.device_get(
                            replacement_state_after.discovery_state.observation_count
                        )
                    ),
                    proposal_due=bool(jax.device_get(prepare_diagnostics.proposal_due)),
                    proposal_ready=bool(jax.device_get(prepare_diagnostics.proposal_ready)),
                    candidate_ready_for_authority=ready,
                    rejection_reason=self._replacement_rejection_reason(
                        prepare_diagnostics
                    ),
                    selected_candidate_indices=self._int_tuple(
                        proposal_bundle.selected_candidate_indices
                    ),
                    selected_descriptors=self._descriptor_tuple(
                        proposal_bundle.selected_descriptors
                    ),
                    changed_slots=self._bool_tuple(child_prepared.changed_slots),
                    semantic_generation=int(jax.device_get(live.semantic_generation)),
                    source_digest_words=self._words(live.source_digest),
                    installed_slot_mask_before=self._bool_tuple(
                        repeated_current.cycle_state.installed_slot_mask
                    ),
                    installed_slot_mask_after=self._bool_tuple(
                        repeated_after_attempt.cycle_state.installed_slot_mask
                    ),
                    cold_slot_mask_before=self._bool_tuple(
                        ~repeated_current.cycle_state.installed_slot_mask
                    ),
                    cold_slot_mask_after=self._bool_tuple(
                        ~repeated_after_attempt.cycle_state.installed_slot_mask
                    ),
                    installation_count_before=int(
                        jax.device_get(
                            replacement_state.installation_state.installation_count
                        )
                    ),
                    installation_count_after=int(
                        jax.device_get(
                            replacement_state_after.installation_state.installation_count
                        )
                    ),
                    scheduler_install_attempts_before=self._words(
                        replacement_state.install_attempt_words
                    ),
                    scheduler_install_attempts_after=self._words(
                        replacement_state_after.install_attempt_words
                    ),
                    ordinary_advance_applied=bool(replacement.transaction_applied),
                )
            )
            if not replacement.transaction_applied:
                raise RuntimeError("fixed replacement ordinary advance was refused")
            if not replacement.cycle_completed:
                declined_observed = declined_observed or force_decline
                replay = self._bridge.commit_replacement(
                    replacement.state,
                    replacement_prepared,
                    replacement_receipt,
                    cycle_key,
                )
                stale_replacement_rejected = stale_replacement_rejected or (
                    not replay.transaction_applied
                    and bool(jax.device_get(_tree_exact_equal(replay.state, replacement.state)))
                )
                current = replacement.state
                continue
            current = replacement.state
            completed = True
            break
        if not completed:
            raise _ReplacementAttemptBoundExhaustedError(
                cycle_index=cycle_index,
                attempts=attempts,
                completed_cycles_before=completed_before,
                control_events=events,
                attempt_diagnostics=tuple(attempt_diagnostics),
            )
        after = self._attached(current)
        budget = self._bridge.resource_budget(current)
        trace = PrototypeRepeatedOptionCycleTrace(
            cycle_index=cycle_index,
            cycle_key_words=self._words(jr.key_data(cycle_key)),
            control_events=events,
            option_use_count=sum(event.option_used for event in events),
            control_return_count=sum(event.control_returned for event in events),
            primitive_fallback_count=sum(event.primitive_fallback for event in events),
            stale_retirement_replay_rejected=stale_retirement_rejected,
            declined_replacement_observed=declined_observed,
            stale_replacement_replay_rejected=stale_replacement_rejected,
            fresh_retry_required=declined_observed and stale_replacement_rejected,
            replacement_attempts=attempts,
            completed_cycles_before=completed_before,
            completed_cycles_after=int(jax.device_get(after.completed_cycles)),
            retirement_revision_words=self._words(after.last_retirement_authority_revision_words),
            replacement_revision_words=self._words(after.last_replacement_authority_revision_words),
            persistent_stomp_state_owners=budget.persistent_stomp_state_owners,
        )
        return current, trace

    def _start(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
    ) -> PrototypeRepeatedOptionAuthorityBridgeState:
        result = self._bridge.start(
            state,
            jnp.asarray(self._config.initial_observation, dtype=jnp.float32),
        )
        if not bool(jax.device_get(result.transaction_applied)):
            raise RuntimeError("fixed Prototype start was refused")
        return result.state

    def _run_disabled_arm(
        self,
        source: PrototypeRepeatedOptionAuthorityBridgeState,
        routed_update_counts: tuple[int, int],
    ) -> PrototypeRepeatedOptionDevelopmentArm:
        current = self._start(source)
        events: list[PrototypeRepeatedOptionControlEvent] = []
        global_index = 0
        for cycle_index in range(2):
            current, cycle_events = self._run_control_window(
                current,
                cycle_index=cycle_index,
                first_global_index=global_index,
                exact_updates=routed_update_counts[cycle_index],
            )
            events.extend(cycle_events)
            global_index += routed_update_counts[cycle_index]
        repeated = self._attached(current)
        budget = self._bridge.resource_budget(current)
        return PrototypeRepeatedOptionDevelopmentArm(
            lifecycle_routing_enabled=False,
            final_state=current,
            final_state_sha256=_tree_sha256_hex(current),
            control_events=tuple(events),
            cycle_traces=(),
            stomp_update_evaluations=sum(event.stomp_update_evaluations for event in events),
            completed_cycles=int(jax.device_get(repeated.completed_cycles)),
            persistent_stomp_state_owners=budget.persistent_stomp_state_owners,
        )

    def run(
        self,
        source: PrototypeRepeatedOptionAuthorityBridgeState,
    ) -> PrototypeRepeatedOptionLifecycleDevelopmentReport:
        """Run the fixed in-memory protocol; never write or promote anything."""

        source_state_valid_before = bool(jax.device_get(self._bridge.state_valid(source)))
        if not source_state_valid_before:
            raise ValueError("development source must satisfy the v2 bridge contract")
        source_sha256_before = _tree_sha256_hex(source)
        initial = self._attached(source)
        if int(jax.device_get(initial.completed_cycles)) != 0:
            raise ValueError("development source must begin before all repeated cycles")
        if bool(jax.device_get(source.bridge_state.prototype_state.started)):
            raise ValueError("development source must be an unstarted continuing owner")

        timings: list[tuple[str, int]] = []
        started_at = time.perf_counter_ns()
        routed_state = self._start(source)
        jax.block_until_ready(routed_state.binding_checksum)
        timings.append(("routed_start", time.perf_counter_ns() - started_at))

        traces: list[PrototypeRepeatedOptionCycleTrace] = []
        events: list[PrototypeRepeatedOptionControlEvent] = []
        global_index = 0
        checkpoint_payload: dict[str, object] | None = None
        checkpoint_state: PrototypeRepeatedOptionAuthorityBridgeState | None = None
        for cycle_index in range(2):
            stage_start = time.perf_counter_ns()
            try:
                routed_state, trace = self._run_cycle(
                    routed_state,
                    cycle_index=cycle_index,
                    first_global_index=global_index,
                )
            except _ReplacementAttemptBoundExhaustedError as exhausted:
                source_state_valid_after = bool(
                    jax.device_get(self._bridge.state_valid(source))
                )
                source_sha256_after = _tree_sha256_hex(source)
                raise ReplacementAttemptExhaustedError(
                    cycle_index=exhausted.cycle_index,
                    attempts=exhausted.attempts,
                    scheduler_attempt_cap=self._config.max_replacement_attempts,
                    completed_cycles_before=exhausted.completed_cycles_before,
                    completed_cycle_traces=tuple(traces),
                    failed_cycle_control_events=exhausted.control_events,
                    attempt_diagnostics=exhausted.attempt_diagnostics,
                    source_state_valid_before=source_state_valid_before,
                    source_state_valid_after=source_state_valid_after,
                    source_sha256_before=source_sha256_before,
                    source_sha256_after=source_sha256_after,
                    checkpoint_created=(
                        checkpoint_payload is not None and checkpoint_state is not None
                    ),
                ) from None
            jax.block_until_ready(routed_state.binding_checksum)
            timings.append((f"routed_cycle_{cycle_index}", time.perf_counter_ns() - stage_start))
            traces.append(trace)
            events.extend(trace.control_events)
            global_index += len(trace.control_events)
            if trace.completed_cycles_after == self._config.checkpoint_after_completed_cycles:
                checkpoint_state = routed_state
                checkpoint_payload = self._bridge.checkpoint_payload(routed_state)
        if checkpoint_payload is None or checkpoint_state is None:
            raise RuntimeError("fixed checkpoint cut was not reached")

        routed_budget = self._bridge.resource_budget(routed_state)
        routed = PrototypeRepeatedOptionDevelopmentArm(
            lifecycle_routing_enabled=True,
            final_state=routed_state,
            final_state_sha256=_tree_sha256_hex(routed_state),
            control_events=tuple(events),
            cycle_traces=tuple(traces),
            stomp_update_evaluations=sum(event.stomp_update_evaluations for event in events),
            completed_cycles=int(jax.device_get(self._attached(routed_state).completed_cycles)),
            persistent_stomp_state_owners=(routed_budget.persistent_stomp_state_owners),
        )

        suffix_start = time.perf_counter_ns()
        restored = self._bridge.restore_checkpoint(
            checkpoint_payload,
            expected_completed_cycles=1,
            expected_revision=checkpoint_state.revision,
        )
        suffix_first_index = len(traces[0].control_events)
        replayed_final, replayed_trace = self._run_cycle(
            restored,
            cycle_index=1,
            first_global_index=suffix_first_index,
        )
        jax.block_until_ready(replayed_final.binding_checksum)
        timings.append(("checkpoint_suffix", time.perf_counter_ns() - suffix_start))
        suffix_parity = (
            bool(jax.device_get(_tree_exact_equal(replayed_final, routed_state)))
            and replayed_trace == traces[1]
        )

        disabled_start = time.perf_counter_ns()
        routed_update_counts = (
            len(traces[0].control_events),
            len(traces[1].control_events),
        )
        disabled = self._run_disabled_arm(
            source,
            routed_update_counts,
        )
        jax.block_until_ready(disabled.final_state.binding_checksum)
        timings.append(("routing_disabled", time.perf_counter_ns() - disabled_start))
        same_work = (
            routed.stomp_update_evaluations
            == disabled.stomp_update_evaluations
            == sum(routed_update_counts)
        )
        if routed.completed_cycles != 2:
            raise RuntimeError("routed arm did not complete both fixed cycles")
        if disabled.completed_cycles != 0:
            raise RuntimeError("disabled arm unexpectedly routed lifecycle authority")
        if not suffix_parity or not same_work:
            raise RuntimeError("checkpoint suffix or same-work contract failed")
        if any(trace.persistent_stomp_state_owners != 1 for trace in traces):
            raise RuntimeError("a routed cycle did not preserve the sole STOMP owner")

        return PrototypeRepeatedOptionLifecycleDevelopmentReport(
            schema_version=(PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_REPORT_SCHEMA),
            config=self._config.to_config(),
            config_fingerprint=self._config.fingerprint,
            routed=routed,
            lifecycle_routing_disabled=disabled,
            checkpoint_suffix_parity=suffix_parity,
            checkpoint_suffix_nonempty=True,
            same_control_work=same_work,
            ordinary_control_opportunities_matched=same_work,
            total_work_matched=False,
            resource_comparability="not_assessed",
            calibration_consumed=True,
            preregistered=False,
            independently_fixed_comparator=False,
            resource_budget=routed_budget,
            stage_latency_ns=tuple(timings),
            latency_clock="time.perf_counter_ns",
            latency_synchronization="jax.block_until_ready(state.binding_checksum)",
            latency_warmup_policy=(
                "none; eager first-call compilation/dispatch is included per stage"
            ),
            open_blockers=_FROZEN_OPEN_BLOCKERS,
            mechanism_status=(PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_MECHANISM_STATUS),
            assessment=PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_ASSESSMENT,
            output_writes=False,
            caller_authenticated=False,
            autonomous_skill_policy=False,
            benefit_claim=False,
            efficacy_claim=False,
            winner_selected=False,
            threshold_tuned=False,
            evidence_authority=False,
            promotion_authority=False,
            scientific_promotion_allowed=False,
        )


__all__ = [
    "PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_ASSESSMENT",
    "PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_AUTONOMOUS_SKILL_POLICY",
    "PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_BENEFIT_CLAIM",
    "PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_CALLER_AUTHENTICATED",
    "PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_CONFIG_SCHEMA",
    "PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_EFFICACY_CLAIM",
    "PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_EVIDENCE_AUTHORITY",
    "PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_MECHANISM_STATUS",
    "PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_OUTCOME_STATUS",
    "PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_OUTPUT_WRITES",
    "PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_PROMOTION_AUTHORITY",
    "PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_REPORT_SCHEMA",
    "PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_SCIENTIFIC_PROMOTION_ALLOWED",
    "PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_TUNED_THRESHOLD",
    "PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_WINNER_SELECTION",
    "PrototypeRepeatedOptionControlEvent",
    "PrototypeRepeatedOptionCycleTrace",
    "PrototypeRepeatedOptionDevelopmentArm",
    "PrototypeRepeatedOptionLifecycleDevelopmentConfig",
    "PrototypeRepeatedOptionLifecycleDevelopmentHarness",
    "PrototypeRepeatedOptionLifecycleDevelopmentReport",
    "PrototypeRepeatedOptionReplacementAttemptDiagnostic",
    "ReplacementAttemptExhaustedError",
]
