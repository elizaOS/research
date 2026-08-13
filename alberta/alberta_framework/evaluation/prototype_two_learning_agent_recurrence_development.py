# mypy: disable-error-code="arg-type,attr-defined,call-arg,index,no-any-return"
"""Development-only A-B-A recurrence with two independent Prototype learners.

This is a deliberately separate rung from the singular scripted-partner
recurrence harness.  Both agents learn, both dispatch primitive actions through
``RecurringTwoAgentWorld.step_result``, and neither receives partner fusion,
intelligence amplification, communication, or evaluator-oracle inputs.

Every event is staged from immutable prestates.  Four joint environment
proposals measure actual, unilateral-base, and joint-base dispatches; two
no-memory Prototype previews are discarded; two memory-sidecar candidates are
then computed from the same agent prestates.  The live environment and both
live agents advance only after every proposal and candidate accepts.

Reports are canonical in-memory development records.  Ephemeral checkpoint
directories are used only for boundary shadow round trips and are removed
before the runner returns.  There is no artifact writer, threshold, evidence
seed, acceptance gate, or scientific-promotion path in this module.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.prototype_agent import (
    PROTOTYPE_FEATURE_WORLD_MODEL_CHECKPOINT_SCHEMA,
    PrototypeAgent,
    PrototypeAgentState,
    PrototypeFeatureWorldModelState,
    PrototypeTransition,
    load_prototype_checkpoint,
    measure_prototype_agent_state_resources,
    save_prototype_checkpoint,
)
from alberta_framework.core.world_model import ActionConditionedWorldModel
from alberta_framework.evaluation.prototype_feature_memory_recurrence_development import (
    RecurrenceReportValidation,
    _canonical_json,
    _critical_counts,
    _decision_id,
    _digest,
    _exact_json_equal,
    _feature_bundle,
    _horde_spec,
    _json_clone,
    _masked_observation,
    _memory_input,
    _phase_for_step,
    _primitive_to_continuous,
    _require_exact_float,
    _require_exact_int,
    _transition,
    _tree_bit_exact,
    _words,
    _words_value,
)
from alberta_framework.evaluation.prototype_feature_memory_recurrence_development import (
    _agent_config as _single_agent_config,
)
from alberta_framework.evaluation.prototype_feature_memory_recurrence_development import (
    _resource_payload as _single_agent_resource_payload,
)
from alberta_framework.evaluation.prototype_feature_memory_recurrence_development import (
    _validate_resources as _validate_single_agent_resources,
)
from alberta_framework.streams.recurring_multiagent import (
    RECURRING_TWO_AGENT_CHECKPOINT_SCHEMA,
    RecurringTwoAgentState,
    RecurringTwoAgentWorld,
    load_recurring_two_agent_checkpoint,
    save_recurring_two_agent_checkpoint,
)

PROTOTYPE_TWO_LEARNING_AGENT_RECURRENCE_PROTOCOL_SCHEMA: Final = (
    "alberta.prototype-two-learning-agent-recurrence-development.protocol.v1"
)
PROTOTYPE_TWO_LEARNING_AGENT_RECURRENCE_REPORT_SCHEMA: Final = (
    "alberta.prototype-two-learning-agent-recurrence-development.report.v1"
)
ACCEPTANCE_STATUS: Final = "not-assessed"
DEVELOPMENT_ONLY: Final = True
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
ACCEPTED_SCIENTIFIC_EVIDENCE: Final = False

INTERPRETATION: Final = (
    "Development-only visible-cue A-B-A trace with two independent learning agents; "
    "not scientific evidence and not an Alberta Plan completion certificate."
)
LIMITATIONS: Final = (
    "the meet/avoid cue is visible to both learners except in the declared "
    "same-shape counterexample",
    "the agents have no communication, partner fusion, intelligence amplification, "
    "or partner-action input",
    "each stable-base world model conditions only on its owner's primitive action, so "
    "joint dynamics under a changing learning partner are structurally marginal",
    "the environment checkpoint serializes its configured default partner policy even "
    "though this harness exclusively uses external joint actions",
    "the synthetic bounded world supplies exact zero uncertainty and safety cost to "
    "exercise memory readout; these gates are not learned or calibrated",
    "boundary checkpoints are ephemeral independent shadows, not an atomic composite "
    "crash-recovery mechanism",
    "report validation replays deterministic environment causality but does not "
    "re-execute Prototype learner, feature, memory, world-model, or Horde updates",
    "logical array bytes and declared work exclude compiler workspaces, allocator "
    "residency, FLOPs, and latency",
    "there are no thresholds, held-out or evidence seeds, confidence intervals, "
    "artifact writer, evidence registration, or promotion path",
)

CLAIM_ASSESSMENTS: Final = {
    "standard_forward_transfer": {
        "status": "not-assessed",
        "reason": "one online A-B-A life has no independent held-out task baseline",
    },
    "partner_learning_uplift": {
        "status": "not-assessed",
        "reason": "Prototype has no exact matched learning-freeze arm",
    },
    "joint_partner_world_prediction": {
        "status": "not-assessed",
        "reason": "the stable-base model receives only the owner's action",
    },
    "communication_or_intelligence_amplification": {
        "status": "not-assessed",
        "reason": "partner fusion, communication, and IA are disabled",
    },
    "scientific_evidence": {
        "status": "not-assessed",
        "reason": "development protocol with no evidence or promotion path",
    },
}

_PHASE_NAMES: Final = ("A1", "B", "A2")
_CHECKPOINT_LABELS: Final = ("initial", "A1", "B", "A2")
_N_AGENTS: Final = 2
_N_ACTIONS: Final = 2
_N_HORDE_DEMONS: Final = 2
_UINT32_MAX: Final = 2**32 - 1
_LIFECYCLE_TAG: Final = 0x32544C52  # ASCII-ish "2TLR"
_SHA256_HEX_LENGTH: Final = 64
_MEMORY_PRESTATE_QUERIES_PER_CANDIDATE: Final = 2
_JOINT_PROPOSAL_NAMES: Final = (
    "actual_actual",
    "base0_actual1",
    "actual0_base1",
    "base_base",
)


@dataclasses.dataclass(frozen=True, slots=True)
class TwoLearningAgentRecurrenceArm:
    """One symmetric two-learner development arm."""

    name: str
    memory_readout_enabled: bool
    feature_promotion_enabled: bool
    cue_visible: bool
    role: str

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


TWO_LEARNING_AGENT_RECURRENCE_ARMS: Final = (
    TwoLearningAgentRecurrenceArm(
        "joint_full",
        memory_readout_enabled=True,
        feature_promotion_enabled=True,
        cue_visible=True,
        role="symmetric two-learning-agent integrated candidate",
    ),
    TwoLearningAgentRecurrenceArm(
        "memory_readout_blocked",
        memory_readout_enabled=False,
        feature_promotion_enabled=True,
        cue_visible=True,
        role="symmetric matched memory behavioral-authority ablation",
    ),
    TwoLearningAgentRecurrenceArm(
        "feature_promotion_blocked",
        memory_readout_enabled=True,
        feature_promotion_enabled=False,
        cue_visible=True,
        role="symmetric matched feature-promotion ablation",
    ),
    TwoLearningAgentRecurrenceArm(
        "dual_blocked",
        memory_readout_enabled=False,
        feature_promotion_enabled=False,
        cue_visible=True,
        role="symmetric matched joint-ablation reference",
    ),
    TwoLearningAgentRecurrenceArm(
        "cue_masked_counterexample",
        memory_readout_enabled=True,
        feature_promotion_enabled=True,
        cue_visible=False,
        role="symmetric same-shape visible-cue-dependence counterexample",
    ),
)
_ARMS_BY_NAME: Final = {arm.name: arm for arm in TWO_LEARNING_AGENT_RECURRENCE_ARMS}
_CANONICAL_ARM_NAMES: Final = tuple(arm.name for arm in TWO_LEARNING_AGENT_RECURRENCE_ARMS)


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeTwoLearningAgentRecurrenceProtocol:
    """Static three-segment development protocol for two Prototype learners.

    The default retains the singular rung's declared geometry.  Unit and
    integration checks may use a one-step segment while preserving all five
    arms, all four joint proposals, both previews, both commits, stale replay,
    and all four checkpoint boundaries.
    """

    segment_length: int = 512
    nuisance_dim: int = 2
    nuisance_scale: float = 1.0
    active_pair_slots: int = 4
    memory_capacity: int = 64
    replacement_interval: int = 64
    metric_window: int = 64
    arm_names: tuple[str, ...] = _CANONICAL_ARM_NAMES
    schema_version: str = PROTOTYPE_TWO_LEARNING_AGENT_RECURRENCE_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PROTOTYPE_TWO_LEARNING_AGENT_RECURRENCE_PROTOCOL_SCHEMA:
            raise ValueError("protocol schema_version is unsupported")
        _require_exact_int(
            self.segment_length,
            name="segment_length",
            minimum=1,
            maximum=(2**31 - 1) // 3,
        )
        _require_exact_int(self.nuisance_dim, name="nuisance_dim", minimum=0, maximum=16)
        _require_exact_float(
            self.nuisance_scale,
            name="nuisance_scale",
            minimum=0.0,
            maximum=100.0,
        )
        _require_exact_int(
            self.active_pair_slots,
            name="active_pair_slots",
            minimum=1,
            maximum=self.candidate_pair_slots,
        )
        _require_exact_int(
            self.memory_capacity,
            name="memory_capacity",
            minimum=1,
            maximum=4096,
        )
        _require_exact_int(
            self.replacement_interval,
            name="replacement_interval",
            minimum=1,
            maximum=2**31 - 2,
        )
        _require_exact_int(
            self.metric_window,
            name="metric_window",
            minimum=1,
            maximum=self.segment_length,
        )
        if type(self.arm_names) is not tuple or not self.arm_names:
            raise ValueError("arm_names must be a nonempty exact tuple")
        if any(type(name) is not str or name not in _ARMS_BY_NAME for name in self.arm_names):
            raise ValueError("arm_names contains an unsupported arm")
        canonical_subset = tuple(name for name in _CANONICAL_ARM_NAMES if name in self.arm_names)
        if self.arm_names != canonical_subset or len(set(self.arm_names)) != len(self.arm_names):
            raise ValueError("arm_names must be a unique canonical-order subset")

    @property
    def total_steps(self) -> int:
        return 3 * self.segment_length

    @property
    def base_observation_dim(self) -> int:
        return 6 + self.nuisance_dim

    @property
    def candidate_pair_slots(self) -> int:
        base = self.base_observation_dim
        return base * (base - 1) // 2

    def to_config(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "type": type(self).__name__,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "seed_contract": "caller-supplied development seed; no held-out/evidence seeds",
            "schedule": list(_PHASE_NAMES),
            "segment_length": self.segment_length,
            "total_steps": self.total_steps,
            "agent_count": _N_AGENTS,
            "nuisance_dim": self.nuisance_dim,
            "nuisance_scale": self.nuisance_scale,
            "active_pair_slots": self.active_pair_slots,
            "candidate_pair_slots": self.candidate_pair_slots,
            "memory_capacity": self.memory_capacity,
            "replacement_interval": self.replacement_interval,
            "metric_window": self.metric_window,
            "arm_names": list(self.arm_names),
            "transaction_contract": {
                "joint_environment_proposals_per_event": 4,
                "discarded_no_memory_previews_per_event": 2,
                "committed_agent_candidates_per_event": 2,
                "carry_only_if_every_proposal_and_candidate_accepts": True,
                "simultaneous_immutable_prestates": True,
            },
            "checkpoint_contract": {
                "labels": list(_CHECKPOINT_LABELS),
                "ephemeral_shadow_only": True,
                "atomic_composite_recovery_claimed": False,
            },
            "world_model_contract": {
                "coordinates": "stable_base_only",
                "action_scope": "owner_primitive_action_only",
                "partner_action_observed": False,
                "generated_pair_tail_modeled": False,
                "gamma": 1.0,
                "buffer_capacity": 1,
                "update_calls_per_event": 4,
                "carried_updates_per_event": 2,
            },
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> PrototypeTwoLearningAgentRecurrenceProtocol:
        expected = set(cls().to_config())
        if set(payload) != expected:
            raise ValueError("protocol fields do not match the v1 schema")
        if payload.get("type") != cls.__name__:
            raise ValueError("protocol type is unsupported")
        raw_arms = payload.get("arm_names")
        if not isinstance(raw_arms, list) or any(type(name) is not str for name in raw_arms):
            raise ValueError("protocol arm_names must be a JSON list of strings")
        protocol = cls(
            segment_length=cast(int, payload["segment_length"]),
            nuisance_dim=cast(int, payload["nuisance_dim"]),
            nuisance_scale=cast(float, payload["nuisance_scale"]),
            active_pair_slots=cast(int, payload["active_pair_slots"]),
            memory_capacity=cast(int, payload["memory_capacity"]),
            replacement_interval=cast(int, payload["replacement_interval"]),
            metric_window=cast(int, payload["metric_window"]),
            arm_names=tuple(cast(list[str], raw_arms)),
            schema_version=cast(str, payload["schema_version"]),
        )
        if not _exact_json_equal(protocol.to_config(), dict(payload)):
            raise ValueError("protocol payload is not canonical")
        return protocol


def _execution_contract() -> dict[str, object]:
    return {
        "two_independent_learning_agents": True,
        "environment_step_api": "RecurringTwoAgentWorld.step_result",
        "external_joint_actions": True,
        "configured_partner_policy_used": False,
        "partner_policy_fusion_enabled": False,
        "intelligence_amplification_enabled": False,
        "communication_enabled": False,
        "joint_action_atomicity": "stage all candidates, then carry all-or-none",
        "world_model_behavioral_authority": False,
        "report_validation_scope": (
            "exact schema/arithmetic plus deterministic environment replay; "
            "learner trajectories are not re-executed"
        ),
    }


def _agent_config(
    protocol: PrototypeTwoLearningAgentRecurrenceProtocol,
    *,
    feature_promotion_enabled: bool,
) -> Any:
    """Reuse the singular lane's canonical all-in-one Prototype composition."""

    config = _single_agent_config(
        protocol,
        feature_promotion_enabled=feature_promotion_enabled,
    )
    if config.world_model is None or config.world_model.gamma != 1.0:
        raise RuntimeError("two-learner stable-base world model must use gamma=1.0")
    if config.partner_policy_fusion is not None or config.ia is not None:
        raise RuntimeError("two-learner recurrence excludes partner fusion and IA")
    return config


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return float(sum(values) / len(values))


def _phase_events(
    trace: Sequence[Mapping[str, object]],
    phase: str,
) -> list[Mapping[str, object]]:
    return [event for event in trace if event["phase"] == phase]


def _phase_window(
    trace: Sequence[Mapping[str, object]],
    phase: str,
    window: int,
    *,
    tail: bool,
) -> list[Mapping[str, object]]:
    events = _phase_events(trace, phase)
    return events[-window:] if tail else events[:window]


def _agent_payload(event: Mapping[str, object], agent_index: int) -> Mapping[str, object]:
    return cast(list[Mapping[str, object]], event["agents"])[agent_index]


def _world_model_state(state: PrototypeAgentState) -> Any:
    wrapper = state.world_model_state
    if type(wrapper) is not PrototypeFeatureWorldModelState:
        raise RuntimeError("two-learner recurrence requires the v17 feature/world wrapper")
    return wrapper.model_state


def _world_prediction_diagnostic(
    model: ActionConditionedWorldModel,
    state: PrototypeAgentState,
    *,
    next_observation: jax.Array,
    reward: jax.Array,
    discount: jax.Array,
) -> dict[str, object]:
    prediction = model.predict(
        _world_model_state(state),
        state.current_raw_observation,
        state.current_action,
    )
    predicted_next = [float(value) for value in np.asarray(prediction.next_observation)]
    target_next = [float(value) for value in np.asarray(next_observation)]
    next_errors = [left - right for left, right in zip(predicted_next, target_next, strict=True)]
    observation_mse = _mean([value * value for value in next_errors])
    physical_mse = _mean([value * value for value in next_errors[:4]])
    cue_mse = _mean([value * value for value in next_errors[4:6]])
    nuisance_errors = next_errors[6:]
    nuisance_mse = (
        _mean([value * value for value in nuisance_errors]) if nuisance_errors else 0.0
    )
    reward_prediction = float(prediction.reward)
    reward_target = float(reward)
    discount_prediction = float(prediction.discount)
    discount_target = float(discount)
    reward_squared_error = (reward_prediction - reward_target) ** 2
    discount_squared_error = (discount_prediction - discount_target) ** 2
    total = observation_mse + reward_squared_error + discount_squared_error
    return {
        "action_scope": "owner_primitive_action_only",
        "partner_action_observed": False,
        "next_observation_prediction": predicted_next,
        "next_observation_target": target_next,
        "reward_prediction": reward_prediction,
        "reward_target": reward_target,
        "discount_prediction": discount_prediction,
        "discount_target": discount_target,
        "errors": {
            "next_observation": next_errors,
            "observation_mse": observation_mse,
            "physical_mse": physical_mse,
            "cue_mse": cue_mse,
            "nuisance_mse": nuisance_mse,
            "reward_squared_error": reward_squared_error,
            "discount_squared_error": discount_squared_error,
            "total": total,
        },
    }


def _tree_copy(value: Any) -> Any:
    return jax.tree.map(
        lambda leaf: leaf if leaf is None else jnp.array(leaf),
        value,
        is_leaf=lambda leaf: leaf is None,
    )


def _tree_witness(value: object) -> dict[str, object]:
    """Hash one pytree's structure, leaf geometry, and exact leaf bytes."""

    leaves, tree = jax.tree_util.tree_flatten(value)
    geometry: list[dict[str, object]] = []
    leaf_records: list[dict[str, object]] = []
    for leaf in leaves:
        materialized = jax.device_get(leaf)
        dtype = getattr(materialized, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            materialized = jax.device_get(jr.key_data(materialized))
        array = np.ascontiguousarray(np.asarray(materialized))
        leaf_geometry = {
            "shape": [int(dimension) for dimension in array.shape],
            "dtype": str(array.dtype),
        }
        geometry.append(leaf_geometry)
        leaf_records.append(
            {
                **leaf_geometry,
                "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        )
    return {
        "leaf_count": len(leaves),
        "tree_structure_sha256": hashlib.sha256(str(tree).encode("utf-8")).hexdigest(),
        "leaf_geometry_sha256": _digest(geometry),
        "leaf_bytes_sha256": _digest(leaf_records),
    }


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 hex digest")
    return value


def _validate_tree_witness(value: object, *, name: str) -> Mapping[str, object]:
    expected = {
        "leaf_count",
        "tree_structure_sha256",
        "leaf_geometry_sha256",
        "leaf_bytes_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} fields are invalid")
    if type(value["leaf_count"]) is not int or value["leaf_count"] <= 0:
        raise ValueError(f"{name}.leaf_count must be a positive exact integer")
    for field in expected - {"leaf_count"}:
        _require_sha256(value[field], name=f"{name}.{field}")
    return value


_WORLD_ERROR_NAMES: Final = (
    "observation_mse",
    "physical_mse",
    "cue_mse",
    "nuisance_mse",
    "reward_squared_error",
    "discount_squared_error",
    "total",
)


def _summary(
    values: Sequence[float],
    *,
    entry: Sequence[float],
    tail: Sequence[float],
) -> dict[str, float]:
    return {
        "mean": _mean(values),
        "entry": _mean(entry),
        "tail": _mean(tail),
    }


def _metrics_from_trace(
    trace: Sequence[Mapping[str, object]],
    protocol: PrototypeTwoLearningAgentRecurrenceProtocol,
) -> dict[str, object]:
    per_agent: list[dict[str, object]] = []
    for agent_index in range(_N_AGENTS):
        phase_reward: dict[str, object] = {}
        phase_horde_mse: dict[str, object] = {}
        phase_world_error: dict[str, object] = {}
        phase_features: dict[str, object] = {}
        phase_memory: dict[str, object] = {}
        for phase in _PHASE_NAMES:
            events = _phase_events(trace, phase)
            entry_events = _phase_window(
                trace,
                phase,
                protocol.metric_window,
                tail=False,
            )
            tail_events = _phase_window(
                trace,
                phase,
                protocol.metric_window,
                tail=True,
            )
            payloads = [_agent_payload(event, agent_index) for event in events]
            entries = [_agent_payload(event, agent_index) for event in entry_events]
            tails = [_agent_payload(event, agent_index) for event in tail_events]
            phase_reward[phase] = _summary(
                [cast(float, item["reward"]) for item in payloads],
                entry=[cast(float, item["reward"]) for item in entries],
                tail=[cast(float, item["reward"]) for item in tails],
            )
            phase_horde_mse[phase] = {
                key: [
                    _mean(
                        [
                            cast(list[float], item["horde_squared_error"])[demon]
                            for item in values
                        ]
                    )
                    for demon in range(_N_HORDE_DEMONS)
                ]
                for key, values in (("mean", payloads), ("entry", entries), ("tail", tails))
            }
            phase_world_error[phase] = {
                error_name: _summary(
                    [
                        cast(
                            float,
                            cast(Mapping[str, object], item["world_model"])["errors"][
                                error_name
                            ],
                        )
                        for item in payloads
                    ],
                    entry=[
                        cast(
                            float,
                            cast(Mapping[str, object], item["world_model"])["errors"][
                                error_name
                            ],
                        )
                        for item in entries
                    ],
                    tail=[
                        cast(
                            float,
                            cast(Mapping[str, object], item["world_model"])["errors"][
                                error_name
                            ],
                        )
                        for item in tails
                    ],
                )
                for error_name in _WORLD_ERROR_NAMES
            }
            phase_features[phase] = {
                "a_critical_pair_fraction": _mean(
                    [float(cast(int, item["a_critical_pair_count"]) > 0) for item in payloads]
                ),
                "b_critical_pair_fraction": _mean(
                    [float(cast(int, item["b_critical_pair_count"]) > 0) for item in payloads]
                ),
                "curation_commits": sum(bool(item["curation_committed"]) for item in payloads),
                "memory_rebinds": sum(
                    bool(item["feature_memory_rebind_applied"]) for item in payloads
                ),
                "rows_reencoded": sum(
                    cast(int, item["memory_rows_reencoded"]) for item in payloads
                ),
            }
            phase_memory[phase] = {
                "query_before_write_events": sum(
                    bool(item["memory_query_before_write"]) for item in payloads
                ),
                "writes": sum(bool(item["memory_wrote"]) for item in payloads),
                "retrievals_available": sum(
                    bool(item["memory_retrieval_available"]) for item in payloads
                ),
                "action_changes": sum(bool(item["memory_action_changed"]) for item in payloads),
                "action_change_rate": _mean(
                    [float(bool(item["memory_action_changed"])) for item in payloads]
                ),
            }

        a1_reward = cast(Mapping[str, float], phase_reward["A1"])
        a2_reward = cast(Mapping[str, float], phase_reward["A2"])
        a1_horde = cast(Mapping[str, list[float]], phase_horde_mse["A1"])
        a2_horde = cast(Mapping[str, list[float]], phase_horde_mse["A2"])
        a1_world = cast(Mapping[str, Mapping[str, float]], phase_world_error["A1"])
        a2_world = cast(Mapping[str, Mapping[str, float]], phase_world_error["A2"])
        a1_features = cast(Mapping[str, object], phase_features["A1"])
        a2_features = cast(Mapping[str, object], phase_features["A2"])
        a1_memory = cast(Mapping[str, object], phase_memory["A1"])
        a2_memory = cast(Mapping[str, object], phase_memory["A2"])
        per_agent.append(
            {
                "agent_index": agent_index,
                "phase_reward": phase_reward,
                "phase_horde_mse": phase_horde_mse,
                "phase_world_prediction_error": phase_world_error,
                "phase_features": phase_features,
                "phase_memory": phase_memory,
                "recurrence": {
                    "a2_entry_minus_a1_tail_reward": (
                        a2_reward["entry"] - a1_reward["tail"]
                    ),
                    "a2_tail_minus_a1_tail_reward": (
                        a2_reward["tail"] - a1_reward["tail"]
                    ),
                    "a2_reward_reacquisition_gain": (
                        a2_reward["tail"] - a2_reward["entry"]
                    ),
                    "a2_entry_minus_a1_tail_horde_mse": [
                        a2_horde["entry"][demon] - a1_horde["tail"][demon]
                        for demon in range(_N_HORDE_DEMONS)
                    ],
                    "a2_horde_reacquisition_gain": [
                        a2_horde["entry"][demon] - a2_horde["tail"][demon]
                        for demon in range(_N_HORDE_DEMONS)
                    ],
                    "a2_entry_minus_a1_tail_world_total_error": (
                        a2_world["total"]["entry"] - a1_world["total"]["tail"]
                    ),
                    "a2_world_total_reacquisition_gain": (
                        a2_world["total"]["entry"] - a2_world["total"]["tail"]
                    ),
                    "a2_a_pair_fraction_minus_a1": (
                        cast(float, a2_features["a_critical_pair_fraction"])
                        - cast(float, a1_features["a_critical_pair_fraction"])
                    ),
                    "a2_action_change_rate_minus_a1": (
                        cast(float, a2_memory["action_change_rate"])
                        - cast(float, a1_memory["action_change_rate"])
                    ),
                },
            }
        )

    joint_effects: dict[str, object] = {}
    for phase in _PHASE_NAMES:
        events = _phase_events(trace, phase)
        entries = _phase_window(trace, phase, protocol.metric_window, tail=False)
        tails = _phase_window(trace, phase, protocol.metric_window, tail=True)
        joint_effects[phase] = {
            name: _summary(
                [
                    cast(
                        float,
                        cast(Mapping[str, object], event["joint_dispatch"])["effects"][name],
                    )
                    for event in events
                ],
                entry=[
                    cast(
                        float,
                        cast(Mapping[str, object], event["joint_dispatch"])["effects"][name],
                    )
                    for event in entries
                ],
                tail=[
                    cast(
                        float,
                        cast(Mapping[str, object], event["joint_dispatch"])["effects"][name],
                    )
                    for event in tails
                ],
            )
            for name in ("agent0_unilateral", "agent1_unilateral", "joint", "interaction")
        }
    return {
        "per_agent": per_agent,
        "phase_joint_reward_effects": joint_effects,
        "standard_forward_transfer_assessed": False,
        "partner_learning_uplift_assessed": False,
    }


def _checkpoint_identity(
    *,
    label: str,
    event_count: int,
    world: RecurringTwoAgentWorld,
    environment_state: RecurringTwoAgentState,
    agents: Sequence[PrototypeAgent],
    agent_states: Sequence[PrototypeAgentState],
    base_actions: Sequence[int],
) -> dict[str, object]:
    return {
        "label": label,
        "event_count": event_count,
        "environment_step_words": _words(environment_state.step_words),
        "agent_step_words": [_words(state.step_words) for state in agent_states],
        "agent_decision_ids": [_decision_id(state.current_decision_id) for state in agent_states],
        "agent_current_actions": [int(state.current_action) for state in agent_states],
        "harness_base_actions": list(base_actions),
        "environment_config_sha256": _digest(world.to_config()),
        "agent_config_sha256": [_digest(agent.to_config()) for agent in agents],
        "environment_state_witness": _tree_witness(environment_state),
        "agent_state_witnesses": [_tree_witness(state) for state in agent_states],
    }


def _checkpoint_shadow_round_trip(
    *,
    directory: Path,
    label: str,
    event_count: int,
    world: RecurringTwoAgentWorld,
    environment_state: RecurringTwoAgentState,
    agents: Sequence[PrototypeAgent],
    agent_states: Sequence[PrototypeAgentState],
    base_actions: Sequence[int],
) -> dict[str, object]:
    identity = _checkpoint_identity(
        label=label,
        event_count=event_count,
        world=world,
        environment_state=environment_state,
        agents=agents,
        agent_states=agent_states,
        base_actions=base_actions,
    )
    environment_path = directory / f"{label}-environment"
    save_recurring_two_agent_checkpoint(world, environment_state, environment_path)
    restored_world, restored_environment = load_recurring_two_agent_checkpoint(environment_path)
    restored_environment_config_sha256 = _digest(restored_world.to_config())
    restored_environment_witness = _tree_witness(restored_environment)
    environment_exact = (
        _exact_json_equal(restored_world.to_config(), world.to_config())
        and _exact_json_equal(
            restored_environment_witness,
            identity["environment_state_witness"],
        )
        and _tree_bit_exact(restored_environment, environment_state)
    )

    restored_agent_exact: list[bool] = []
    restored_agent_config_exact: list[bool] = []
    restored_agent_config_sha256: list[str] = []
    restored_agent_state_witnesses: list[dict[str, object]] = []
    for agent_index, (agent, state) in enumerate(zip(agents, agent_states, strict=True)):
        agent_path = directory / f"{label}-agent-{agent_index}"
        save_prototype_checkpoint(agent, state, agent_path)
        restored_agent, restored_state = load_prototype_checkpoint(agent_path)
        restored_config = restored_agent.to_config()
        restored_witness = _tree_witness(restored_state)
        restored_agent_config_sha256.append(_digest(restored_config))
        restored_agent_state_witnesses.append(restored_witness)
        restored_agent_config_exact.append(
            _exact_json_equal(restored_config, agent.to_config())
        )
        restored_agent_exact.append(
            _exact_json_equal(
                restored_witness,
                cast(list[object], identity["agent_state_witnesses"])[agent_index],
            )
            and _tree_bit_exact(restored_state, state)
        )
    audit = {
        **identity,
        "environment_checkpoint_schema": RECURRING_TWO_AGENT_CHECKPOINT_SCHEMA,
        "agent_checkpoint_schema": PROTOTYPE_FEATURE_WORLD_MODEL_CHECKPOINT_SCHEMA,
        "environment_round_trip_bit_exact": environment_exact,
        "agent_config_round_trip_exact": restored_agent_config_exact,
        "agent_state_round_trip_bit_exact": restored_agent_exact,
        "restored_environment_config_sha256": restored_environment_config_sha256,
        "restored_agent_config_sha256": restored_agent_config_sha256,
        "restored_environment_state_witness": restored_environment_witness,
        "restored_agent_state_witnesses": restored_agent_state_witnesses,
        "checkpoint_state_carried": False,
        "atomic_composite_recovery_claimed": False,
        "composite_identity_sha256": _digest(identity),
    }
    if not environment_exact or not all(restored_agent_config_exact) or not all(
        restored_agent_exact
    ):
        raise RuntimeError(f"{label} checkpoint shadow round trip was not bit-exact")
    return cast(dict[str, object], _json_clone(audit))


def _resources_payload(
    *,
    protocol: PrototypeTwoLearningAgentRecurrenceProtocol,
    world: RecurringTwoAgentWorld,
    agents: Sequence[PrototypeAgent],
    initial_states: Sequence[PrototypeAgentState],
    final_states: Sequence[PrototypeAgentState],
    phase_boundaries: Sequence[Sequence[int]],
    peaks: Sequence[int],
) -> dict[str, object]:
    per_agent: list[dict[str, object]] = []
    initial_agent_totals: list[int] = []
    final_agent_totals: list[int] = []
    for agent_index in range(_N_AGENTS):
        declaration = _single_agent_resource_payload(
            agents[agent_index],
            world,
            initial_states[agent_index],
            final_states[agent_index],
            list(phase_boundaries[agent_index]),
            peaks[agent_index],
        )
        per_agent.append({"agent_index": agent_index, "declaration": declaration})
        initial_agent_totals.append(
            cast(
                int,
                cast(Mapping[str, object], declaration["initial_state"])["total_nbytes"],
            )
        )
        final_agent_totals.append(
            cast(
                int,
                cast(Mapping[str, object], declaration["final_state"])["total_nbytes"],
            )
        )
    environment = world.resource_budget.to_dict()
    environment_bytes = cast(int, environment["state_nbytes"])
    boundary_totals = [
        environment_bytes + sum(phase_boundaries[agent][boundary] for agent in range(_N_AGENTS))
        for boundary in range(len(_CHECKPOINT_LABELS))
    ]
    initial_total = environment_bytes + sum(initial_agent_totals, 0)
    final_total = environment_bytes + sum(final_agent_totals, 0)
    return {
        "environment": environment,
        "per_agent": per_agent,
        "combined": {
            "initial_total_nbytes": initial_total,
            "final_total_nbytes": final_total,
            "phase_boundary_total_nbytes": boundary_totals,
            "peak_total_nbytes": environment_bytes + sum(peaks),
            "logical_fixed_allocation": True,
        },
    }


def _work_from_trace(
    trace: Sequence[Mapping[str, object]],
    resources: Mapping[str, object],
) -> dict[str, int]:
    steps = len(trace)
    per_agent = cast(list[Mapping[str, object]], resources["per_agent"])
    first_declaration = cast(Mapping[str, object], per_agent[0]["declaration"])
    feature = cast(Mapping[str, object], first_declaration["feature_lifecycle"])
    memory_queries = sum(
        cast(int, agent["memory_prestate_query_count"])
        for event in trace
        for agent in cast(list[Mapping[str, object]], event["agents"])
    )
    memory_writes = sum(
        bool(agent["memory_wrote"])
        for event in trace
        for agent in cast(list[Mapping[str, object]], event["agents"])
    )
    feature_observes = 4 * steps
    return {
        "requested_joint_transitions": steps,
        "requested_agent_transitions": 2 * steps,
        "environment_proposal_calls": 4 * steps,
        "counterfactual_environment_proposal_calls": 3 * steps,
        "committed_environment_transitions": steps,
        "joint_transaction_commits": steps,
        "discarded_preview_update_calls": 2 * steps,
        "committed_candidate_update_calls": 2 * steps,
        "regular_prototype_update_calls": 4 * steps,
        "stale_identity_probe_update_calls": 2,
        "total_prototype_update_calls": 4 * steps + 2,
        "world_model_update_calls": 4 * steps,
        "world_model_discarded_preview_updates": 2 * steps,
        "world_model_carried_updates": 2 * steps,
        "explicit_world_model_prediction_calls": 2 * steps,
        "explicit_horde_prediction_calls": 2 * steps,
        "explicit_horde_predictions_emitted": 2 * _N_HORDE_DEMONS * steps,
        "managed_horde_update_calls": 4 * steps,
        "regular_memory_sidecars_supplied": 2 * steps,
        "stale_probe_memory_sidecars_supplied": 0,
        "total_memory_sidecars_supplied": 2 * steps,
        "memory_deterministic_prestate_queries": memory_queries,
        "memory_writes": memory_writes,
        "prototype_feature_observe_calls": feature_observes,
        "configured_max_active_pair_products": (
            feature_observes * cast(int, feature["max_active_pair_products_per_observe"])
        ),
        "configured_max_candidate_pair_products": (
            feature_observes * cast(int, feature["max_candidate_pair_products_per_observe"])
        ),
        "checkpoint_shadow_boundaries": len(_CHECKPOINT_LABELS),
        "checkpoint_shadow_object_save_calls": 3 * len(_CHECKPOINT_LABELS),
        "checkpoint_shadow_object_load_calls": 3 * len(_CHECKPOINT_LABELS),
        "external_partner_policy_calls": 0,
    }


def _joint_action(first: int, second: int) -> jax.Array:
    return jnp.stack(
        (
            _primitive_to_continuous(first),
            _primitive_to_continuous(second),
        )
    ).astype(jnp.float32)


def _run_arm(
    protocol: PrototypeTwoLearningAgentRecurrenceProtocol,
    arm: TwoLearningAgentRecurrenceArm,
    *,
    seed: int,
    arm_index: int,
    world: RecurringTwoAgentWorld,
    agents: Sequence[PrototypeAgent],
    horde: HordeLearner,
) -> dict[str, object]:
    environment_state = world.init(jr.key(seed))
    initial_observations = world.observe(environment_state)
    lifecycle_ids = [
        jnp.asarray((_LIFECYCLE_TAG, 2 * arm_index + agent_index + 1), dtype=jnp.uint32)
        for agent_index in range(_N_AGENTS)
    ]
    agent_states = [
        agents[agent_index].start(
            agents[agent_index].init(
                jr.key(seed ^ (0x13579BDF + agent_index * 0x10203)),
                lifecycle_id=lifecycle_ids[agent_index],
            ),
            _masked_observation(
                initial_observations[agent_index],
                cue_visible=arm.cue_visible,
            ),
        )
        for agent_index in range(_N_AGENTS)
    ]
    initial_agent_states = list(agent_states)
    base_actions = [int(state.current_action) for state in agent_states]
    trace: list[dict[str, object]] = []
    phase_boundaries = [
        [measure_prototype_agent_state_resources(state).total_nbytes]
        for state in agent_states
    ]
    peaks = [values[0] for values in phase_boundaries]
    stale_decision_ids: list[jax.Array | None] = [None, None]
    stale_replay_audit: dict[str, object] | None = None
    world_models = [
        ActionConditionedWorldModel(cast(Any, agent.config.world_model))
        for agent in agents
    ]

    with tempfile.TemporaryDirectory(prefix="alberta-two-learner-shadow-") as temp_name:
        shadow_directory = Path(temp_name)
        checkpoint_audits = [
            _checkpoint_shadow_round_trip(
                directory=shadow_directory,
                label="initial",
                event_count=0,
                world=world,
                environment_state=environment_state,
                agents=agents,
                agent_states=agent_states,
                base_actions=base_actions,
            )
        ]

        for event_index in range(protocol.total_steps):
            environment_source = _tree_copy(environment_state)
            agent_sources = [_tree_copy(state) for state in agent_states]
            current_actions = [int(state.current_action) for state in agent_states]
            if any(action not in (0, 1) for action in current_actions + base_actions):
                raise RuntimeError("Prototype emitted an invalid primitive action")

            proposal_actions = {
                "actual_actual": _joint_action(current_actions[0], current_actions[1]),
                "base0_actual1": _joint_action(base_actions[0], current_actions[1]),
                "actual0_base1": _joint_action(current_actions[0], base_actions[1]),
                "base_base": _joint_action(base_actions[0], base_actions[1]),
            }
            proposal_primitives = {
                "actual_actual": list(current_actions),
                "base0_actual1": [base_actions[0], current_actions[1]],
                "actual0_base1": [current_actions[0], base_actions[1]],
                "base_base": list(base_actions),
            }
            environment_proposals = {
                name: world.step_result(environment_state, proposal_actions[name])
                for name in _JOINT_PROPOSAL_NAMES
            }
            if not all(bool(result.update_applied) for result in environment_proposals.values()):
                raise RuntimeError("a bounded joint environment proposal was rejected")
            expected_pre_words = _words(environment_state.step_words)
            proposal_word_pairs = {
                name: {
                    "pre": _words(result.pre_step_words),
                    "post": _words(result.post_step_words),
                }
                for name, result in environment_proposals.items()
            }
            if any(
                words["pre"] != expected_pre_words
                or _words_value(words["post"], name="proposal post words")
                != event_index + 1
                for words in proposal_word_pairs.values()
            ):
                raise RuntimeError("joint proposals did not share one exact event identity")

            actual_environment = environment_proposals["actual_actual"]
            raw_next_observations = actual_environment.transition.next_observation
            next_observations = [
                _masked_observation(
                    raw_next_observations[agent_index],
                    cue_visible=arm.cue_visible,
                )
                for agent_index in range(_N_AGENTS)
            ]
            transitions = [
                _transition(
                    agent_states[agent_index],
                    reward=actual_environment.transition.reward[agent_index],
                    discount=actual_environment.transition.discount,
                    terminated=actual_environment.transition.terminated,
                    next_observation=next_observations[agent_index],
                )
                for agent_index in range(_N_AGENTS)
            ]
            if stale_decision_ids[0] is None:
                stale_decision_ids = [transition.decision_id for transition in transitions]

            if event_index == 2 * protocol.segment_length:
                if any(value is None for value in stale_decision_ids):
                    raise RuntimeError("both A1 decision identities must be captured")
                environment_snapshot = _tree_copy(environment_state)
                probe_transitions = [
                    cast(
                        PrototypeTransition,
                        transitions[agent_index].replace(
                            decision_id=cast(jax.Array, stale_decision_ids[agent_index])
                        ),
                    )
                    for agent_index in range(_N_AGENTS)
                ]
                probe_assessments = [
                    agents[agent_index].assess_transition(
                        agent_states[agent_index], probe_transitions[agent_index]
                    )
                    for agent_index in range(_N_AGENTS)
                ]
                replay_results = [
                    agents[agent_index].update_transition(
                        agent_states[agent_index], probe_transitions[agent_index]
                    )
                    for agent_index in range(_N_AGENTS)
                ]
                replay_agents: list[dict[str, object]] = []
                for agent_index, (assessment, replay) in enumerate(
                    zip(probe_assessments, replay_results, strict=True)
                ):
                    nonidentity_prechecks_valid = all(
                        bool(value)
                        for value in (
                            assessment.outer_counter_valid,
                            assessment.current_counter_capacity_available,
                            assessment.started,
                            assessment.inputs_finite,
                            assessment.action_in_range,
                            assessment.observation_matches,
                            assessment.action_matches,
                            assessment.next_generation_available,
                            assessment.next_counter_capacity_available,
                            assessment.discount_valid,
                            assessment.boundary_semantics_valid,
                            assessment.state_consistent,
                        )
                    )
                    replay_agents.append(
                        {
                            "agent_index": agent_index,
                            "stale_decision_id": _decision_id(
                                cast(jax.Array, stale_decision_ids[agent_index])
                            ),
                            "current_decision_id": _decision_id(
                                agent_states[agent_index].current_decision_id
                            ),
                            "observation_matches": bool(
                                assessment.observation_matches
                            ),
                            "action_matches": bool(assessment.action_matches),
                            "decision_id_matches": bool(
                                assessment.decision_id_matches
                            ),
                            "update_decision_id_matches": bool(
                                replay.transition_diagnostics.decision_id_matches
                            ),
                            "update_reported_rejected": bool(
                                replay.transition_diagnostics.rejected
                            ),
                            "nonidentity_prechecks_valid": (
                                nonidentity_prechecks_valid
                            ),
                            "stale_decision_rejected": bool(
                                assessment.rejected
                                and not assessment.decision_id_matches
                                and replay.transition_diagnostics.rejected
                                and not replay.transition_diagnostics.decision_id_matches
                            ),
                            "state_bit_exact": _tree_bit_exact(
                                replay.state, agent_states[agent_index]
                            ),
                            "returned_action_unchanged": (
                                int(replay.action)
                                == int(agent_states[agent_index].current_action)
                            ),
                            "agent_clock_unchanged": (
                                _words(replay.state.step_words)
                                == _words(agent_states[agent_index].step_words)
                                and _words(replay.state.observation_event_words)
                                == _words(
                                    agent_states[agent_index].observation_event_words
                                )
                            ),
                        }
                    )
                stale_replay_audit = {
                    "aba_replay_attempted": True,
                    "probe_isolated_to_decision_id": True,
                    "environment_unchanged": _tree_bit_exact(
                        environment_state, environment_snapshot
                    ),
                    "agent_results": replay_agents,
                    "replay_update_calls": _N_AGENTS,
                    "memory_sidecars_supplied": 0,
                }
                if (
                    not bool(stale_replay_audit["environment_unchanged"])
                    or not all(
                        result["observation_matches"] is True
                        and result["action_matches"] is True
                        and result["decision_id_matches"] is False
                        and result["update_decision_id_matches"] is False
                        and result["update_reported_rejected"] is True
                        and result["nonidentity_prechecks_valid"] is True
                        and result["stale_decision_rejected"] is True
                        and result["state_bit_exact"] is True
                        and result["returned_action_unchanged"] is True
                        and result["agent_clock_unchanged"] is True
                        for result in replay_agents
                    )
                ):
                    raise RuntimeError("isolated stale decision-ID probe did not fail closed")

            horde_predictions: list[jax.Array] = []
            horde_cumulants: list[jax.Array] = []
            horde_td_errors: list[jax.Array] = []
            horde_squared_errors: list[jax.Array] = []
            world_diagnostics: list[dict[str, object]] = []
            for agent_index in range(_N_AGENTS):
                bundle = _feature_bundle(agent_states[agent_index])
                prediction = horde.predict(
                    bundle.horde_state,
                    agent_states[agent_index].current_representation,
                )
                cumulants = cast(jax.Array, transitions[agent_index].horde_cumulants)
                horde_predictions.append(prediction)
                horde_cumulants.append(cumulants)
                td_errors = cumulants - prediction
                horde_td_errors.append(td_errors)
                horde_squared_errors.append(jnp.square(td_errors))
                world_diagnostics.append(
                    _world_prediction_diagnostic(
                        world_models[agent_index],
                        agent_states[agent_index],
                        next_observation=next_observations[agent_index],
                        reward=actual_environment.transition.reward[agent_index],
                        discount=actual_environment.transition.discount,
                    )
                )

            previews = [
                agents[agent_index].update_transition(
                    agent_states[agent_index],
                    transitions[agent_index],
                )
                for agent_index in range(_N_AGENTS)
            ]
            if not all(bool(preview.transition_diagnostics.valid) for preview in previews):
                raise RuntimeError("a discarded no-memory preview was rejected")
            preview_actions = [int(preview.action) for preview in previews]
            memory_inputs = [
                _memory_input(
                    agent_states[agent_index],
                    previews[agent_index].state,
                    event_index=event_index,
                    reward=actual_environment.transition.reward[agent_index],
                    safe_action=(
                        None if arm.memory_readout_enabled else preview_actions[agent_index]
                    ),
                )
                for agent_index in range(_N_AGENTS)
            ]
            candidates = [
                agents[agent_index].update_transition(
                    agent_states[agent_index],
                    transitions[agent_index],
                    experiential_memory_input=memory_inputs[agent_index],
                )
                for agent_index in range(_N_AGENTS)
            ]
            if not all(candidate.transition_diagnostics.valid for candidate in candidates):
                raise RuntimeError("a committed Prototype candidate was rejected")
            if not _tree_bit_exact(environment_state, environment_source) or not all(
                _tree_bit_exact(agent_states[index], agent_sources[index])
                for index in range(_N_AGENTS)
            ):
                raise RuntimeError("a staged proposal mutated an immutable prestate")

            proposal_rewards = {
                name: float(result.transition.reward[0])
                for name, result in environment_proposals.items()
            }
            if any(
                float(result.transition.reward[1]) != reward
                for result, reward in zip(
                    environment_proposals.values(),
                    proposal_rewards.values(),
                    strict=True,
                )
            ):
                raise RuntimeError("cooperative reward unexpectedly differed by agent")
            actual_reward = proposal_rewards["actual_actual"]
            effects = {
                "agent0_unilateral": (
                    actual_reward - proposal_rewards["base0_actual1"]
                ),
                "agent1_unilateral": (
                    actual_reward - proposal_rewards["actual0_base1"]
                ),
                "joint": actual_reward - proposal_rewards["base_base"],
                "interaction": (
                    actual_reward
                    - proposal_rewards["base0_actual1"]
                    - proposal_rewards["actual0_base1"]
                    + proposal_rewards["base_base"]
                ),
            }

            agent_events: list[dict[str, object]] = []
            next_base_actions: list[int] = []
            for agent_index, candidate in enumerate(candidates):
                memory_diagnostics = candidate.experiential_memory_diagnostics
                feature_diagnostics = candidate.prototype_feature_lifecycle_diagnostics
                feature_memory_diagnostics = candidate.prototype_feature_memory_diagnostics
                if (
                    memory_diagnostics is None
                    or feature_diagnostics is None
                    or feature_memory_diagnostics is None
                    or candidate.world_model_error is None
                ):
                    raise RuntimeError("configured all-in-one component omitted diagnostics")
                for update_name, update_result in (
                    ("preview", previews[agent_index]),
                    ("candidate", candidate),
                ):
                    if update_result.horde_td_errors is None or not np.allclose(
                        np.asarray(update_result.horde_td_errors),
                        np.asarray(horde_td_errors[agent_index]),
                        rtol=1e-6,
                        atol=1e-7,
                    ):
                        raise RuntimeError(
                            f"managed Horde {update_name} TD errors disagree with "
                            "the explicit pre-update prediction"
                        )
                reported_world_error = float(candidate.world_model_error)
                computed_world_error = cast(
                    float,
                    cast(Mapping[str, object], world_diagnostics[agent_index]["errors"])[
                        "total"
                    ],
                )
                if not math.isfinite(reported_world_error) or not np.isclose(
                    reported_world_error,
                    computed_world_error,
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    raise RuntimeError("Prototype and explicit world-model errors disagree")
                world_diagnostics[agent_index]["prototype_reported_total"] = (
                    reported_world_error
                )
                if int(memory_diagnostics.counterfactual_base_action) != preview_actions[
                    agent_index
                ]:
                    raise RuntimeError("memory counterfactual action disagrees with preview")
                if (
                    not arm.memory_readout_enabled
                    and int(candidate.action) != preview_actions[agent_index]
                ):
                    raise RuntimeError("readout-blocked memory changed the preview action")
                a_count, b_count = _critical_counts(candidate.state)
                source_bundle = _feature_bundle(agent_states[agent_index])
                destination_bundle = _feature_bundle(candidate.state)
                agent_events.append(
                    {
                        "agent_index": agent_index,
                        "prototype_pre_step_words": _words(
                            agent_states[agent_index].step_words
                        ),
                        "prototype_post_step_words": _words(candidate.state.step_words),
                        "prototype_decision_id": _decision_id(
                            agent_states[agent_index].current_decision_id
                        ),
                        "action": current_actions[agent_index],
                        "counterfactual_base_action": base_actions[agent_index],
                        "next_preview_action": preview_actions[agent_index],
                        "next_committed_action": int(candidate.action),
                        "reward": actual_reward,
                        "horde_prediction": [
                            float(value) for value in np.asarray(horde_predictions[agent_index])
                        ],
                        "horde_cumulant": [
                            float(value) for value in np.asarray(horde_cumulants[agent_index])
                        ],
                        "prototype_reported_horde_td_error": [
                            float(value)
                            for value in np.asarray(candidate.horde_td_errors)
                        ],
                        "horde_squared_error": [
                            float(value)
                            for value in np.asarray(horde_squared_errors[agent_index])
                        ],
                        "world_model": world_diagnostics[agent_index],
                        "feature_generation_pre_words": _words(
                            source_bundle.consumer_binding.semantic_generation_words
                        ),
                        "feature_generation_post_words": _words(
                            destination_bundle.consumer_binding.semantic_generation_words
                        ),
                        "a_critical_pair_count": a_count,
                        "b_critical_pair_count": b_count,
                        "curation_committed": bool(
                            feature_diagnostics.lifecycle.curation_committed
                        ),
                        "feature_memory_rebind_applied": bool(
                            feature_memory_diagnostics.rebind.transaction_applied
                        ),
                        "memory_rows_reencoded": int(
                            feature_memory_diagnostics.rebind.valid_rows_reencoded
                        ),
                        "memory_query_before_write": bool(
                            memory_diagnostics.query_before_write
                        ),
                        "memory_prestate_query_count": int(
                            memory_diagnostics.deterministic_prestate_query_count
                        ),
                        "memory_wrote": bool(memory_diagnostics.wrote),
                        "memory_retrieval_available": bool(
                            memory_diagnostics.proposal.available
                        ),
                        "memory_action_changed": bool(
                            memory_diagnostics.dispatch_replacement.applied
                        ),
                        "source_state_preserved": True,
                        "preview_state_discarded": True,
                        "candidate_accepted": True,
                    }
                )
                next_base_actions.append(
                    int(memory_diagnostics.counterfactual_base_action)
                )

            trace.append(
                {
                    "event_index": event_index,
                    "phase": _phase_for_step(event_index, protocol.segment_length),
                    "phase_step": event_index % protocol.segment_length,
                    "environment_pre_words": expected_pre_words,
                    "environment_post_words": _words(
                        actual_environment.state.step_words
                    ),
                    "environment_source_state_preserved": True,
                    "joint_dispatch": {
                        "primitive_actions": proposal_primitives,
                        "continuous_actions": {
                            name: [int(value) for value in np.asarray(action)]
                            for name, action in proposal_actions.items()
                        },
                        "proposal_event_words": proposal_word_pairs,
                        "rewards": proposal_rewards,
                        "effects": effects,
                        "all_proposals_accepted": True,
                    },
                    "agents": agent_events,
                    "all_agent_candidates_accepted": True,
                    "joint_transaction_committed": True,
                }
            )
            # This is the only live-state carry point for the event.
            environment_state = actual_environment.state
            agent_states = [candidate.state for candidate in candidates]
            base_actions = next_base_actions
            for agent_index, state in enumerate(agent_states):
                measured = measure_prototype_agent_state_resources(state)
                peaks[agent_index] = max(peaks[agent_index], measured.total_nbytes)
            if (event_index + 1) % protocol.segment_length == 0:
                boundary_index = (event_index + 1) // protocol.segment_length
                for agent_index, state in enumerate(agent_states):
                    phase_boundaries[agent_index].append(
                        measure_prototype_agent_state_resources(state).total_nbytes
                    )
                checkpoint_audits.append(
                    _checkpoint_shadow_round_trip(
                        directory=shadow_directory,
                        label=_CHECKPOINT_LABELS[boundary_index],
                        event_count=event_index + 1,
                        world=world,
                        environment_state=environment_state,
                        agents=agents,
                        agent_states=agent_states,
                        base_actions=base_actions,
                    )
                )

    if stale_replay_audit is None:
        raise RuntimeError("A-B-A stale replay audit was not executed")
    resources = _resources_payload(
        protocol=protocol,
        world=world,
        agents=agents,
        initial_states=initial_agent_states,
        final_states=agent_states,
        phase_boundaries=phase_boundaries,
        peaks=peaks,
    )
    agent_configs = [
        cast(dict[str, object], _json_clone(agent.to_config())) for agent in agents
    ]
    run = {
        "arm": arm.name,
        "seed": seed,
        "lifecycle_ids": [
            [_LIFECYCLE_TAG, 2 * arm_index + agent_index + 1]
            for agent_index in range(_N_AGENTS)
        ],
        "agent_configs": agent_configs,
        "agent_config_sha256": [_digest(config) for config in agent_configs],
        "trace": trace,
        "trace_sha256": _digest(trace),
        "metrics": _metrics_from_trace(trace, protocol),
        "resources": resources,
        "work": _work_from_trace(trace, resources),
        "stale_replay_audit": stale_replay_audit,
        "checkpoint_shadow_audits": checkpoint_audits,
        "temporary_checkpoint_storage_retained": False,
    }
    return cast(dict[str, object], _json_clone(run))


def _comparison_contract(runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    initial_agent_geometries = [
        (
            witness["tree_structure_sha256"],
            witness["leaf_geometry_sha256"],
            witness["leaf_count"],
        )
        for run in runs
        for witness in cast(
            list[Mapping[str, object]],
            cast(list[Mapping[str, object]], run["checkpoint_shadow_audits"])[0][
                "agent_state_witnesses"
            ],
        )
    ]
    work_keys = (
        "requested_joint_transitions",
        "requested_agent_transitions",
        "environment_proposal_calls",
        "counterfactual_environment_proposal_calls",
        "committed_environment_transitions",
        "joint_transaction_commits",
        "discarded_preview_update_calls",
        "committed_candidate_update_calls",
        "regular_prototype_update_calls",
        "stale_identity_probe_update_calls",
        "world_model_update_calls",
        "world_model_carried_updates",
        "explicit_horde_prediction_calls",
        "managed_horde_update_calls",
        "regular_memory_sidecars_supplied",
        "stale_probe_memory_sidecars_supplied",
        "total_memory_sidecars_supplied",
        "prototype_feature_observe_calls",
        "checkpoint_shadow_object_save_calls",
        "checkpoint_shadow_object_load_calls",
    )
    work_rows = [
        tuple(cast(Mapping[str, object], run["work"])[key] for key in work_keys)
        for run in runs
    ]
    all_events = [
        event
        for run in runs
        for event in cast(list[Mapping[str, object]], run["trace"])
    ]
    return {
        "paired_development_seed": len({run["seed"] for run in runs}) == 1,
        "arm_order": [run["arm"] for run in runs],
        "two_symmetric_agent_configs_per_arm": all(
            cast(list[object], run["agent_configs"])[0]
            == cast(list[object], run["agent_configs"])[1]
            for run in runs
        ),
        "persistent_state_shape_matched": len(set(initial_agent_geometries)) == 1,
        "transaction_and_shadow_work_matched": len(set(work_rows)) == 1,
        "simultaneous_immutable_prestates": all(
            event["environment_source_state_preserved"] is True
            and all(
                agent["source_state_preserved"] is True
                for agent in cast(list[Mapping[str, object]], event["agents"])
            )
            for event in all_events
        ),
        "preview_state_carried": False,
        "checkpoint_shadow_state_carried": False,
        "realized_compute_or_allocator_parity_claimed": False,
    }


def run_prototype_two_learning_agent_recurrence_development(
    protocol: PrototypeTwoLearningAgentRecurrenceProtocol | None = None,
    *,
    seed: int = 0,
) -> dict[str, object]:
    """Run one paired, bounded, nonpromoting two-learner development life."""

    resolved = (
        PrototypeTwoLearningAgentRecurrenceProtocol() if protocol is None else protocol
    )
    if type(resolved) is not PrototypeTwoLearningAgentRecurrenceProtocol:
        raise TypeError(
            "protocol must be an exact PrototypeTwoLearningAgentRecurrenceProtocol"
        )
    _require_exact_int(seed, name="seed", minimum=0, maximum=_UINT32_MAX)
    world = RecurringTwoAgentWorld(
        context_length=resolved.segment_length,
        nuisance_dim=resolved.nuisance_dim,
        nuisance_scale=resolved.nuisance_scale,
    )
    horde = HordeLearner(_horde_spec(), hidden_sizes=(), step_size=0.05)
    agent_by_promotion = {
        enabled: PrototypeAgent(
            _agent_config(resolved, feature_promotion_enabled=enabled)
        )
        for enabled in (True, False)
    }
    runs = [
        _run_arm(
            resolved,
            _ARMS_BY_NAME[name],
            seed=seed,
            arm_index=_CANONICAL_ARM_NAMES.index(name),
            world=world,
            agents=(
                agent_by_promotion[_ARMS_BY_NAME[name].feature_promotion_enabled],
                agent_by_promotion[_ARMS_BY_NAME[name].feature_promotion_enabled],
            ),
            horde=horde,
        )
        for name in resolved.arm_names
    ]
    protocol_payload = resolved.to_config()
    environment_payload = world.to_config()
    report: dict[str, object] = {
        "schema_version": PROTOTYPE_TWO_LEARNING_AGENT_RECURRENCE_REPORT_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "acceptance_status": ACCEPTANCE_STATUS,
        "accepted_scientific_evidence": False,
        "interpretation": INTERPRETATION,
        "protocol": protocol_payload,
        "protocol_sha256": _digest(protocol_payload),
        "environment_config": environment_payload,
        "environment_config_sha256": _digest(environment_payload),
        "execution_contract": _execution_contract(),
        "arm_definitions": [
            _ARMS_BY_NAME[name].to_config() for name in resolved.arm_names
        ],
        "runs": runs,
        "comparison_contract": _comparison_contract(runs),
        "claim_assessments": _json_clone(CLAIM_ASSESSMENTS),
        "limitations": list(LIMITATIONS),
    }
    report["report_sha256"] = _digest(report)
    canonical = cast(dict[str, object], _json_clone(report))
    validation = validate_prototype_two_learning_agent_recurrence_report(canonical)
    if not validation.valid:
        raise RuntimeError(
            "internally built two-learner recurrence report failed validation: "
            + "; ".join(validation.errors)
        )
    return canonical


_AGENT_EVENT_FIELDS: Final = {
    "agent_index",
    "prototype_pre_step_words",
    "prototype_post_step_words",
    "prototype_decision_id",
    "action",
    "counterfactual_base_action",
    "next_preview_action",
    "next_committed_action",
    "reward",
    "horde_prediction",
    "horde_cumulant",
    "prototype_reported_horde_td_error",
    "horde_squared_error",
    "world_model",
    "feature_generation_pre_words",
    "feature_generation_post_words",
    "a_critical_pair_count",
    "b_critical_pair_count",
    "curation_committed",
    "feature_memory_rebind_applied",
    "memory_rows_reencoded",
    "memory_query_before_write",
    "memory_prestate_query_count",
    "memory_wrote",
    "memory_retrieval_available",
    "memory_action_changed",
    "source_state_preserved",
    "preview_state_discarded",
    "candidate_accepted",
}
_EVENT_FIELDS: Final = {
    "event_index",
    "phase",
    "phase_step",
    "environment_pre_words",
    "environment_post_words",
    "environment_source_state_preserved",
    "joint_dispatch",
    "agents",
    "all_agent_candidates_accepted",
    "joint_transaction_committed",
}


def _require_finite_float(value: object, *, name: str, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite exact float")
    result = value
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _require_float_vector(value: object, *, name: str, length: int) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(type(item) is not float or not math.isfinite(item) for item in value)
    ):
        raise ValueError(f"{name} must contain {length} finite exact floats")
    return cast(list[float], value)


def _validate_world_diagnostic(
    value: object,
    *,
    name: str,
    protocol: PrototypeTwoLearningAgentRecurrenceProtocol,
    reward: float,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    expected = {
        "action_scope",
        "partner_action_observed",
        "next_observation_prediction",
        "next_observation_target",
        "reward_prediction",
        "reward_target",
        "discount_prediction",
        "discount_target",
        "errors",
        "prototype_reported_total",
    }
    if set(value) != expected:
        raise ValueError(f"{name} fields are invalid")
    if value["action_scope"] != "owner_primitive_action_only":
        raise ValueError(f"{name} action scope changed")
    if value["partner_action_observed"] is not False:
        raise ValueError(f"{name} cannot claim partner-action input")
    predicted = _require_float_vector(
        value["next_observation_prediction"],
        name=f"{name}.next_observation_prediction",
        length=protocol.base_observation_dim,
    )
    target = _require_float_vector(
        value["next_observation_target"],
        name=f"{name}.next_observation_target",
        length=protocol.base_observation_dim,
    )
    reward_prediction = _require_finite_float(
        value["reward_prediction"], name=f"{name}.reward_prediction"
    )
    reward_target = _require_finite_float(
        value["reward_target"], name=f"{name}.reward_target"
    )
    discount_prediction = _require_finite_float(
        value["discount_prediction"], name=f"{name}.discount_prediction"
    )
    discount_target = _require_finite_float(
        value["discount_target"], name=f"{name}.discount_target"
    )
    if reward_target != reward or discount_target != 1.0:
        raise ValueError(f"{name} targets do not match the continuing transition")
    raw_errors = value["errors"]
    if not isinstance(raw_errors, Mapping) or set(raw_errors) != {
        "next_observation",
        *_WORLD_ERROR_NAMES,
    }:
        raise ValueError(f"{name}.errors fields are invalid")
    recorded_next_errors = _require_float_vector(
        raw_errors["next_observation"],
        name=f"{name}.errors.next_observation",
        length=protocol.base_observation_dim,
    )
    expected_next_errors = [
        left - right for left, right in zip(predicted, target, strict=True)
    ]
    if not np.allclose(recorded_next_errors, expected_next_errors, rtol=0.0, atol=0.0):
        raise ValueError(f"{name} next-observation errors do not reconstruct")
    squared = [error * error for error in expected_next_errors]
    expected_errors = {
        "observation_mse": _mean(squared),
        "physical_mse": _mean(squared[:4]),
        "cue_mse": _mean(squared[4:6]),
        "nuisance_mse": _mean(squared[6:]) if squared[6:] else 0.0,
        "reward_squared_error": (reward_prediction - reward_target) ** 2,
        "discount_squared_error": (discount_prediction - discount_target) ** 2,
    }
    expected_errors["total"] = (
        expected_errors["observation_mse"]
        + expected_errors["reward_squared_error"]
        + expected_errors["discount_squared_error"]
    )
    for error_name, expected_value in expected_errors.items():
        recorded = _require_finite_float(
            raw_errors[error_name],
            name=f"{name}.errors.{error_name}",
            nonnegative=True,
        )
        if recorded != expected_value:
            raise ValueError(f"{name}.{error_name} does not reconstruct")
    reported = _require_finite_float(
        value["prototype_reported_total"],
        name=f"{name}.prototype_reported_total",
        nonnegative=True,
    )
    if not np.isclose(reported, expected_errors["total"], rtol=1e-5, atol=1e-6):
        raise ValueError(f"{name} Prototype total does not match explicit prediction")


def _validate_agent_event(
    value: object,
    *,
    name: str,
    index: int,
    agent_index: int,
    lifecycle_id: list[int],
    protocol: PrototypeTwoLearningAgentRecurrenceProtocol,
    arm: TwoLearningAgentRecurrenceArm,
    expected_action: int,
    expected_base_action: int,
    expected_feature_generation: int,
    reward: float,
) -> int:
    if not isinstance(value, Mapping) or set(value) != _AGENT_EVENT_FIELDS:
        raise ValueError(f"{name} fields are invalid")
    if type(value["agent_index"]) is not int or value["agent_index"] != agent_index:
        raise ValueError(f"{name}.agent_index is invalid")
    for field, expected in (
        ("prototype_pre_step_words", index),
        ("prototype_post_step_words", index + 1),
    ):
        if _words_value(value[field], name=f"{name}.{field}") != expected:
            raise ValueError(f"{name}.{field} is not the exact event identity")
    decision = value["prototype_decision_id"]
    if not isinstance(decision, list) or len(decision) != 4:
        raise ValueError(f"{name}.prototype_decision_id is invalid")
    if not _exact_json_equal(decision[:2], lifecycle_id) or _words_value(
        decision[2:], name=f"{name}.decision_generation"
    ) != index:
        raise ValueError(f"{name}.prototype_decision_id is not current")
    for field in (
        "action",
        "counterfactual_base_action",
        "next_preview_action",
        "next_committed_action",
    ):
        if type(value[field]) is not int or value[field] not in (0, 1):
            raise ValueError(f"{name}.{field} is not a primitive action")
    if value["action"] != expected_action or value["counterfactual_base_action"] != (
        expected_base_action
    ):
        raise ValueError(f"{name} dispatch action does not match the joint proposal")
    recorded_reward = _require_finite_float(value["reward"], name=f"{name}.reward")
    if recorded_reward != reward:
        raise ValueError(f"{name}.reward does not match the shared reward")
    vectors = {
        field: _require_float_vector(
            value[field],
            name=f"{name}.{field}",
            length=_N_HORDE_DEMONS,
        )
        for field in (
            "horde_prediction",
            "horde_cumulant",
            "prototype_reported_horde_td_error",
            "horde_squared_error",
        )
    }
    expected_td_errors = [
        vectors["horde_cumulant"][demon] - vectors["horde_prediction"][demon]
        for demon in range(_N_HORDE_DEMONS)
    ]
    if not np.allclose(
        vectors["prototype_reported_horde_td_error"],
        expected_td_errors,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError(f"{name} managed Horde TD errors do not reconstruct")
    expected_squared = [
        error**2 for error in expected_td_errors
    ]
    if not np.allclose(
        vectors["horde_squared_error"],
        expected_squared,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError(f"{name} Horde error does not reconstruct")
    _validate_world_diagnostic(
        value["world_model"],
        name=f"{name}.world_model",
        protocol=protocol,
        reward=reward,
    )
    feature_generation_pre = _words_value(
        value["feature_generation_pre_words"],
        name=f"{name}.feature_generation_pre_words",
    )
    feature_generation_post = _words_value(
        value["feature_generation_post_words"],
        name=f"{name}.feature_generation_post_words",
    )
    if feature_generation_pre != expected_feature_generation or (
        feature_generation_post not in (feature_generation_pre, feature_generation_pre + 1)
    ):
        raise ValueError(f"{name} feature generation is not continuous and bounded")
    for field in (
        "a_critical_pair_count",
        "b_critical_pair_count",
        "memory_rows_reencoded",
        "memory_prestate_query_count",
    ):
        if type(value[field]) is not int or cast(int, value[field]) < 0:
            raise ValueError(f"{name}.{field} must be a non-negative exact integer")
    for field in (
        "curation_committed",
        "feature_memory_rebind_applied",
        "memory_query_before_write",
        "memory_wrote",
        "memory_retrieval_available",
        "memory_action_changed",
        "source_state_preserved",
        "preview_state_discarded",
        "candidate_accepted",
    ):
        if type(value[field]) is not bool:
            raise ValueError(f"{name}.{field} must be boolean")
    if (
        value["source_state_preserved"] is not True
        or value["preview_state_discarded"] is not True
        or value["candidate_accepted"] is not True
    ):
        raise ValueError(f"{name} did not satisfy the atomic staging contract")
    generation_changed = feature_generation_post != feature_generation_pre
    if value["curation_committed"] is not generation_changed or (
        value["feature_memory_rebind_applied"] is not generation_changed
    ):
        raise ValueError(f"{name} feature curation/rebind does not match generation")
    if value["a_critical_pair_count"] not in (0, 1) or value[
        "b_critical_pair_count"
    ] not in (0, 1):
        raise ValueError(f"{name} critical pair counts must be binary")
    if cast(int, value["memory_rows_reencoded"]) > protocol.memory_capacity or (
        not generation_changed and value["memory_rows_reencoded"] != 0
    ):
        raise ValueError(f"{name} memory re-encoding count is invalid")
    if (
        value["memory_query_before_write"] is not True
        or value["memory_prestate_query_count"]
        != _MEMORY_PRESTATE_QUERIES_PER_CANDIDATE
        or value["memory_wrote"] is not True
    ):
        raise ValueError(f"{name} memory query/write ordering is invalid")
    action_changed = value["next_committed_action"] != value["next_preview_action"]
    if value["memory_action_changed"] is not action_changed or (
        action_changed and value["memory_retrieval_available"] is not True
    ):
        raise ValueError(f"{name} memory action attribution is invalid")
    if not arm.memory_readout_enabled and (
        value["memory_action_changed"] is not False
        or value["next_committed_action"] != value["next_preview_action"]
    ):
        raise ValueError(f"{name} violates the readout-blocked arm")
    return feature_generation_post


def _validate_event(
    event: object,
    *,
    index: int,
    protocol: PrototypeTwoLearningAgentRecurrenceProtocol,
    arm: TwoLearningAgentRecurrenceArm,
    lifecycle_ids: list[list[int]],
    expected_feature_generations: list[int],
) -> list[int]:
    name = f"trace[{index}]"
    if not isinstance(event, Mapping) or set(event) != _EVENT_FIELDS:
        raise ValueError(f"{name} fields are invalid")
    if type(event["event_index"]) is not int or event["event_index"] != index:
        raise ValueError(f"{name}.event_index is invalid")
    if event["phase"] != _phase_for_step(index, protocol.segment_length):
        raise ValueError(f"{name}.phase is invalid")
    if type(event["phase_step"]) is not int or event["phase_step"] != (
        index % protocol.segment_length
    ):
        raise ValueError(f"{name}.phase_step is invalid")
    if _words_value(event["environment_pre_words"], name=f"{name}.environment_pre") != index:
        raise ValueError(f"{name}.environment_pre_words is invalid")
    if _words_value(event["environment_post_words"], name=f"{name}.environment_post") != (
        index + 1
    ):
        raise ValueError(f"{name}.environment_post_words is invalid")
    if event["environment_source_state_preserved"] is not True:
        raise ValueError(f"{name} mutated the environment prestate")
    joint = event["joint_dispatch"]
    if not isinstance(joint, Mapping) or set(joint) != {
        "primitive_actions",
        "continuous_actions",
        "proposal_event_words",
        "rewards",
        "effects",
        "all_proposals_accepted",
    }:
        raise ValueError(f"{name}.joint_dispatch fields are invalid")
    primitives = joint["primitive_actions"]
    continuous = joint["continuous_actions"]
    proposal_words = joint["proposal_event_words"]
    rewards = joint["rewards"]
    effects = joint["effects"]
    for field_name, raw in (
        ("primitive_actions", primitives),
        ("continuous_actions", continuous),
        ("proposal_event_words", proposal_words),
        ("rewards", rewards),
    ):
        if not isinstance(raw, Mapping) or set(raw) != set(_JOINT_PROPOSAL_NAMES):
            raise ValueError(f"{name}.{field_name} proposal names are invalid")
    primitive_rows: dict[str, list[int]] = {}
    for proposal_name in _JOINT_PROPOSAL_NAMES:
        row = cast(Mapping[str, object], primitives)[proposal_name]
        if (
            not isinstance(row, list)
            or len(row) != _N_AGENTS
            or any(type(action) is not int or action not in (0, 1) for action in row)
        ):
            raise ValueError(f"{name}.{proposal_name} primitive actions are invalid")
        primitive_rows[proposal_name] = cast(list[int], row)
        continuous_row = cast(Mapping[str, object], continuous)[proposal_name]
        expected_continuous = [-1 if action == 0 else 1 for action in row]
        if not _exact_json_equal(continuous_row, expected_continuous):
            raise ValueError(f"{name}.{proposal_name} continuous actions are invalid")
        words = cast(Mapping[str, object], proposal_words)[proposal_name]
        if not isinstance(words, Mapping) or set(words) != {"pre", "post"}:
            raise ValueError(f"{name}.{proposal_name} proposal words are invalid")
        if _words_value(words["pre"], name=f"{name}.{proposal_name}.pre") != index or (
            _words_value(words["post"], name=f"{name}.{proposal_name}.post")
            != index + 1
        ):
            raise ValueError(f"{name}.{proposal_name} did not share event identity")
    actual_actions = primitive_rows["actual_actual"]
    base_actions = primitive_rows["base_base"]
    if primitive_rows["base0_actual1"] != [base_actions[0], actual_actions[1]] or (
        primitive_rows["actual0_base1"] != [actual_actions[0], base_actions[1]]
    ):
        raise ValueError(f"{name} unilateral proposals do not reconstruct")
    reward_values = {
        proposal_name: _require_finite_float(
            cast(Mapping[str, object], rewards)[proposal_name],
            name=f"{name}.rewards.{proposal_name}",
        )
        for proposal_name in _JOINT_PROPOSAL_NAMES
    }
    expected_effects = {
        "agent0_unilateral": (
            reward_values["actual_actual"] - reward_values["base0_actual1"]
        ),
        "agent1_unilateral": (
            reward_values["actual_actual"] - reward_values["actual0_base1"]
        ),
        "joint": reward_values["actual_actual"] - reward_values["base_base"],
        "interaction": (
            reward_values["actual_actual"]
            - reward_values["base0_actual1"]
            - reward_values["actual0_base1"]
            + reward_values["base_base"]
        ),
    }
    if not isinstance(effects, Mapping) or set(effects) != set(expected_effects):
        raise ValueError(f"{name}.effects fields are invalid")
    for effect_name, expected_value in expected_effects.items():
        recorded = _require_finite_float(
            effects[effect_name], name=f"{name}.effects.{effect_name}"
        )
        if recorded != expected_value:
            raise ValueError(f"{name}.{effect_name} does not reconstruct")
    if joint["all_proposals_accepted"] is not True:
        raise ValueError(f"{name} contains a rejected environment proposal")
    agents = event["agents"]
    if not isinstance(agents, list) or len(agents) != _N_AGENTS:
        raise ValueError(f"{name}.agents must contain two entries")
    next_feature_generations = [
        _validate_agent_event(
            agent,
            name=f"{name}.agents[{agent_index}]",
            index=index,
            agent_index=agent_index,
            lifecycle_id=lifecycle_ids[agent_index],
            protocol=protocol,
            arm=arm,
            expected_action=actual_actions[agent_index],
            expected_base_action=base_actions[agent_index],
            expected_feature_generation=expected_feature_generations[agent_index],
            reward=reward_values["actual_actual"],
        )
        for agent_index, agent in enumerate(agents)
    ]
    if (
        event["all_agent_candidates_accepted"] is not True
        or event["joint_transaction_committed"] is not True
    ):
        raise ValueError(f"{name} did not commit both learners atomically")
    return next_feature_generations


def _validate_environment_causality(
    trace: Sequence[Mapping[str, object]],
    *,
    protocol: PrototypeTwoLearningAgentRecurrenceProtocol,
    arm: TwoLearningAgentRecurrenceArm,
    seed: int,
) -> None:
    """Replay the cheap deterministic world and bind trace targets to it."""

    world = RecurringTwoAgentWorld(
        context_length=protocol.segment_length,
        nuisance_dim=protocol.nuisance_dim,
        nuisance_scale=protocol.nuisance_scale,
    )
    state = world.init(jr.key(seed))
    for event_index, event in enumerate(trace):
        joint = cast(Mapping[str, object], event["joint_dispatch"])
        primitives = cast(Mapping[str, list[int]], joint["primitive_actions"])
        replayed = {
            proposal_name: world.step_result(
                state,
                _joint_action(
                    primitives[proposal_name][0],
                    primitives[proposal_name][1],
                ),
            )
            for proposal_name in _JOINT_PROPOSAL_NAMES
        }
        if not all(bool(result.update_applied) for result in replayed.values()):
            raise ValueError(f"trace[{event_index}] environment replay rejected")
        recorded_rewards = cast(Mapping[str, object], joint["rewards"])
        for proposal_name, result in replayed.items():
            replayed_reward = float(result.transition.reward[0])
            if recorded_rewards[proposal_name] != replayed_reward or float(
                result.transition.reward[1]
            ) != replayed_reward:
                raise ValueError(
                    f"trace[{event_index}].rewards.{proposal_name} does not replay"
                )
        actual = replayed["actual_actual"]
        next_observations = [
            _masked_observation(
                actual.transition.next_observation[agent_index],
                cue_visible=arm.cue_visible,
            )
            for agent_index in range(_N_AGENTS)
        ]
        for agent_index, agent in enumerate(
            cast(list[Mapping[str, object]], event["agents"])
        ):
            world_diagnostic = cast(Mapping[str, object], agent["world_model"])
            expected_target = [
                float(value) for value in np.asarray(next_observations[agent_index])
            ]
            if not _exact_json_equal(
                world_diagnostic["next_observation_target"], expected_target
            ):
                raise ValueError(
                    f"trace[{event_index}].agents[{agent_index}] world target "
                    "does not replay"
                )
            expected_cumulants = [
                float(actual.transition.reward[agent_index]),
                float(jnp.abs(next_observations[agent_index][1])),
            ]
            if not _exact_json_equal(agent["horde_cumulant"], expected_cumulants):
                raise ValueError(
                    f"trace[{event_index}].agents[{agent_index}] Horde cumulants "
                    "do not replay"
                )
        state = actual.state


def _validate_resources_payload(
    value: object,
    *,
    protocol: PrototypeTwoLearningAgentRecurrenceProtocol,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "environment",
        "per_agent",
        "combined",
    }:
        raise ValueError("resources fields are invalid")
    expected_environment = RecurringTwoAgentWorld(
        context_length=protocol.segment_length,
        nuisance_dim=protocol.nuisance_dim,
        nuisance_scale=protocol.nuisance_scale,
    ).resource_budget.to_dict()
    if not _exact_json_equal(value["environment"], expected_environment):
        raise ValueError("environment resources do not reconstruct")
    per_agent = value["per_agent"]
    if not isinstance(per_agent, list) or len(per_agent) != _N_AGENTS:
        raise ValueError("resources.per_agent must contain two entries")
    declarations: list[Mapping[str, object]] = []
    for agent_index, entry in enumerate(per_agent):
        if not isinstance(entry, Mapping) or set(entry) != {"agent_index", "declaration"}:
            raise ValueError(f"resources.per_agent[{agent_index}] fields are invalid")
        if (
            type(entry["agent_index"]) is not int
            or entry["agent_index"] != agent_index
            or not isinstance(entry["declaration"], Mapping)
        ):
            raise ValueError(f"resources.per_agent[{agent_index}] identity is invalid")
        declaration = cast(Mapping[str, object], entry["declaration"])
        _validate_single_agent_resources(declaration, protocol)
        declarations.append(declaration)
    combined = value["combined"]
    if not isinstance(combined, Mapping) or set(combined) != {
        "initial_total_nbytes",
        "final_total_nbytes",
        "phase_boundary_total_nbytes",
        "peak_total_nbytes",
        "logical_fixed_allocation",
    }:
        raise ValueError("resources.combined fields are invalid")
    environment_bytes = cast(int, expected_environment["state_nbytes"])
    initial_total = environment_bytes + sum(
        cast(int, cast(Mapping[str, object], declaration["initial_state"])["total_nbytes"])
        for declaration in declarations
    )
    final_total = environment_bytes + sum(
        cast(int, cast(Mapping[str, object], declaration["final_state"])["total_nbytes"])
        for declaration in declarations
    )
    boundary_total = [
        environment_bytes
        + sum(
            cast(list[int], declaration["phase_boundary_total_nbytes"])[boundary]
            for declaration in declarations
        )
        for boundary in range(len(_CHECKPOINT_LABELS))
    ]
    peak_total = environment_bytes + sum(
        cast(int, declaration["peak_total_nbytes"]) for declaration in declarations
    )
    if not _exact_json_equal(dict(combined), {
        "initial_total_nbytes": initial_total,
        "final_total_nbytes": final_total,
        "phase_boundary_total_nbytes": boundary_total,
        "peak_total_nbytes": peak_total,
        "logical_fixed_allocation": True,
    }):
        raise ValueError("combined resources do not reconstruct")


def _validate_stale_replay(
    value: object,
    *,
    protocol: PrototypeTwoLearningAgentRecurrenceProtocol,
    lifecycle_ids: list[list[int]],
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "aba_replay_attempted",
        "probe_isolated_to_decision_id",
        "environment_unchanged",
        "agent_results",
        "replay_update_calls",
        "memory_sidecars_supplied",
    }:
        raise ValueError("stale_replay_audit fields are invalid")
    if (
        value["aba_replay_attempted"] is not True
        or value["probe_isolated_to_decision_id"] is not True
        or value["environment_unchanged"] is not True
    ):
        raise ValueError("stale replay did not preserve the environment")
    if (
        type(value["replay_update_calls"]) is not int
        or value["replay_update_calls"] != _N_AGENTS
        or type(value["memory_sidecars_supplied"]) is not int
        or value["memory_sidecars_supplied"] != 0
    ):
        raise ValueError("stale replay work is invalid")
    agents = value["agent_results"]
    if not isinstance(agents, list) or len(agents) != _N_AGENTS:
        raise ValueError("stale replay must contain two agent results")
    for agent_index, result in enumerate(agents):
        if not isinstance(result, Mapping) or set(result) != {
            "agent_index",
            "stale_decision_id",
            "current_decision_id",
            "observation_matches",
            "action_matches",
            "decision_id_matches",
            "update_decision_id_matches",
            "update_reported_rejected",
            "nonidentity_prechecks_valid",
            "stale_decision_rejected",
            "state_bit_exact",
            "returned_action_unchanged",
            "agent_clock_unchanged",
        }:
            raise ValueError(f"stale replay agent {agent_index} fields are invalid")
        if type(result["agent_index"]) is not int or result["agent_index"] != agent_index:
            raise ValueError(f"stale replay agent {agent_index} identity is invalid")
        for field, expected_generation in (
            ("stale_decision_id", 0),
            ("current_decision_id", 2 * protocol.segment_length),
        ):
            decision = result[field]
            if (
                not isinstance(decision, list)
                or len(decision) != 4
                or not _exact_json_equal(decision[:2], lifecycle_ids[agent_index])
                or _words_value(decision[2:], name=f"stale replay agent {agent_index}.{field}")
                != expected_generation
            ):
                raise ValueError(f"stale replay agent {agent_index} {field} is invalid")
        expected_booleans = {
            "observation_matches": True,
            "action_matches": True,
            "decision_id_matches": False,
            "update_decision_id_matches": False,
            "update_reported_rejected": True,
            "nonidentity_prechecks_valid": True,
            "stale_decision_rejected": True,
            "state_bit_exact": True,
            "returned_action_unchanged": True,
            "agent_clock_unchanged": True,
        }
        if any(
            type(result[field]) is not bool or result[field] is not expected
            for field, expected in expected_booleans.items()
        ):
            raise ValueError(f"stale replay agent {agent_index} did not isolate identity")


def _validate_checkpoint_audits(
    value: object,
    *,
    protocol: PrototypeTwoLearningAgentRecurrenceProtocol,
    lifecycle_ids: list[list[int]],
    expected_environment_config_sha256: str,
    expected_agent_config_sha256: list[str],
) -> None:
    if not isinstance(value, list) or len(value) != len(_CHECKPOINT_LABELS):
        raise ValueError("checkpoint shadows must cover all four boundaries")
    expected_counts = (
        0,
        protocol.segment_length,
        2 * protocol.segment_length,
        3 * protocol.segment_length,
    )
    expected_fields = {
        "label",
        "event_count",
        "environment_step_words",
        "agent_step_words",
        "agent_decision_ids",
        "agent_current_actions",
        "harness_base_actions",
        "environment_config_sha256",
        "agent_config_sha256",
        "environment_state_witness",
        "agent_state_witnesses",
        "environment_checkpoint_schema",
        "agent_checkpoint_schema",
        "environment_round_trip_bit_exact",
        "agent_config_round_trip_exact",
        "agent_state_round_trip_bit_exact",
        "restored_environment_config_sha256",
        "restored_agent_config_sha256",
        "restored_environment_state_witness",
        "restored_agent_state_witnesses",
        "checkpoint_state_carried",
        "atomic_composite_recovery_claimed",
        "composite_identity_sha256",
    }
    for boundary, audit in enumerate(value):
        name = f"checkpoint_shadow_audits[{boundary}]"
        if not isinstance(audit, Mapping) or set(audit) != expected_fields:
            raise ValueError(f"{name} fields are invalid")
        label = _CHECKPOINT_LABELS[boundary]
        event_count = expected_counts[boundary]
        if (
            type(audit["label"]) is not str
            or audit["label"] != label
            or type(audit["event_count"]) is not int
            or audit["event_count"] != event_count
        ):
            raise ValueError(f"{name} boundary identity is invalid")
        if _words_value(audit["environment_step_words"], name=f"{name}.environment") != (
            event_count
        ):
            raise ValueError(f"{name} environment clock is invalid")
        agent_words = audit["agent_step_words"]
        decision_ids = audit["agent_decision_ids"]
        current_actions = audit["agent_current_actions"]
        base_actions = audit["harness_base_actions"]
        if (
            not isinstance(agent_words, list)
            or len(agent_words) != _N_AGENTS
            or not isinstance(decision_ids, list)
            or len(decision_ids) != _N_AGENTS
            or not isinstance(base_actions, list)
            or len(base_actions) != _N_AGENTS
            or not isinstance(current_actions, list)
            or len(current_actions) != _N_AGENTS
        ):
            raise ValueError(f"{name} agent identities are invalid")
        for agent_index in range(_N_AGENTS):
            if _words_value(agent_words[agent_index], name=f"{name}.agent_words") != (
                event_count
            ):
                raise ValueError(f"{name} agent clock is invalid")
            decision = decision_ids[agent_index]
            if (
                not isinstance(decision, list)
                or len(decision) != 4
                or not _exact_json_equal(decision[:2], lifecycle_ids[agent_index])
                or _words_value(decision[2:], name=f"{name}.decision") != event_count
            ):
                raise ValueError(f"{name} decision identity is invalid")
            for action_field, actions in (
                ("current", current_actions),
                ("base", base_actions),
            ):
                if type(actions[agent_index]) is not int or actions[agent_index] not in (
                    0,
                    1,
                ):
                    raise ValueError(f"{name} {action_field} action is invalid")
        source_environment_witness = _validate_tree_witness(
            audit["environment_state_witness"],
            name=f"{name}.environment_state_witness",
        )
        restored_environment_witness = _validate_tree_witness(
            audit["restored_environment_state_witness"],
            name=f"{name}.restored_environment_state_witness",
        )
        source_agent_witnesses = audit["agent_state_witnesses"]
        restored_agent_witnesses = audit["restored_agent_state_witnesses"]
        if (
            not isinstance(source_agent_witnesses, list)
            or len(source_agent_witnesses) != _N_AGENTS
            or not isinstance(restored_agent_witnesses, list)
            or len(restored_agent_witnesses) != _N_AGENTS
        ):
            raise ValueError(f"{name} state witness lists are invalid")
        for agent_index in range(_N_AGENTS):
            _validate_tree_witness(
                source_agent_witnesses[agent_index],
                name=f"{name}.agent_state_witnesses[{agent_index}]",
            )
            _validate_tree_witness(
                restored_agent_witnesses[agent_index],
                name=f"{name}.restored_agent_state_witnesses[{agent_index}]",
            )
        config_fields = (
            ("environment_config_sha256", expected_environment_config_sha256),
            (
                "restored_environment_config_sha256",
                expected_environment_config_sha256,
            ),
        )
        for config_field, expected_digest in config_fields:
            _require_sha256(audit[config_field], name=f"{name}.{config_field}")
            if audit[config_field] != expected_digest:
                raise ValueError(f"{name}.{config_field} is invalid")
        for config_field in (
            "agent_config_sha256",
            "restored_agent_config_sha256",
        ):
            digests = audit[config_field]
            if (
                not isinstance(digests, list)
                or len(digests) != _N_AGENTS
                or any(
                    _require_sha256(digest, name=f"{name}.{config_field}")
                    != expected_agent_config_sha256[agent_index]
                    for agent_index, digest in enumerate(digests)
                )
            ):
                raise ValueError(f"{name}.{config_field} is invalid")
        if (
            audit["environment_checkpoint_schema"]
            != RECURRING_TWO_AGENT_CHECKPOINT_SCHEMA
            or audit["agent_checkpoint_schema"]
            != PROTOTYPE_FEATURE_WORLD_MODEL_CHECKPOINT_SCHEMA
            or audit["environment_round_trip_bit_exact"] is not True
            or not _exact_json_equal(
                audit["agent_config_round_trip_exact"], [True, True]
            )
            or not _exact_json_equal(
                audit["agent_state_round_trip_bit_exact"], [True, True]
            )
            or not _exact_json_equal(
                source_environment_witness, restored_environment_witness
            )
            or not _exact_json_equal(
                source_agent_witnesses, restored_agent_witnesses
            )
            or audit["checkpoint_state_carried"] is not False
            or audit["atomic_composite_recovery_claimed"] is not False
        ):
            raise ValueError(f"{name} checkpoint contract is invalid")
        identity = {
            "label": label,
            "event_count": event_count,
            "environment_step_words": audit["environment_step_words"],
            "agent_step_words": agent_words,
            "agent_decision_ids": decision_ids,
            "agent_current_actions": current_actions,
            "harness_base_actions": base_actions,
            "environment_config_sha256": audit["environment_config_sha256"],
            "agent_config_sha256": audit["agent_config_sha256"],
            "environment_state_witness": source_environment_witness,
            "agent_state_witnesses": source_agent_witnesses,
        }
        _require_sha256(
            audit["composite_identity_sha256"],
            name=f"{name}.composite_identity_sha256",
        )
        if audit["composite_identity_sha256"] != _digest(identity):
            raise ValueError(f"{name} composite identity digest is invalid")


def _reconstruct_report(report: Mapping[str, object]) -> dict[str, object]:
    expected_fields = {
        "schema_version",
        "development_only",
        "scientific_promotion_allowed",
        "acceptance_status",
        "accepted_scientific_evidence",
        "interpretation",
        "protocol",
        "protocol_sha256",
        "environment_config",
        "environment_config_sha256",
        "execution_contract",
        "arm_definitions",
        "runs",
        "comparison_contract",
        "claim_assessments",
        "limitations",
        "report_sha256",
    }
    if set(report) != expected_fields:
        raise ValueError("report fields do not match the v1 schema")
    if (
        type(report["schema_version"]) is not str
        or report["schema_version"]
        != PROTOTYPE_TWO_LEARNING_AGENT_RECURRENCE_REPORT_SCHEMA
    ):
        raise ValueError("report schema_version is unsupported")
    if report["development_only"] is not True:
        raise ValueError("report must remain development-only")
    if report["scientific_promotion_allowed"] is not False:
        raise ValueError("report cannot allow scientific promotion")
    if type(report["acceptance_status"]) is not str or report[
        "acceptance_status"
    ] != ACCEPTANCE_STATUS:
        raise ValueError("report acceptance_status must remain not-assessed")
    if report["accepted_scientific_evidence"] is not False:
        raise ValueError("development report cannot claim accepted evidence")
    if type(report["interpretation"]) is not str or report["interpretation"] != INTERPRETATION:
        raise ValueError("report interpretation changed")
    if not _exact_json_equal(report["limitations"], list(LIMITATIONS)):
        raise ValueError("report limitations changed")
    if not _exact_json_equal(report["claim_assessments"], _json_clone(CLAIM_ASSESSMENTS)):
        raise ValueError("report claim assessments changed")
    if not _exact_json_equal(report["execution_contract"], _execution_contract()):
        raise ValueError("report execution contract changed")

    raw_protocol = report["protocol"]
    if not isinstance(raw_protocol, Mapping):
        raise ValueError("report protocol must be a mapping")
    protocol = PrototypeTwoLearningAgentRecurrenceProtocol.from_config(raw_protocol)
    _require_sha256(report["protocol_sha256"], name="report.protocol_sha256")
    if report["protocol_sha256"] != _digest(dict(raw_protocol)):
        raise ValueError("report protocol digest is invalid")
    expected_world = RecurringTwoAgentWorld(
        context_length=protocol.segment_length,
        nuisance_dim=protocol.nuisance_dim,
        nuisance_scale=protocol.nuisance_scale,
    )
    expected_environment = expected_world.to_config()
    if not _exact_json_equal(report["environment_config"], expected_environment):
        raise ValueError("environment config does not reconstruct")
    expected_environment_config_sha256 = _digest(expected_environment)
    _require_sha256(
        report["environment_config_sha256"],
        name="report.environment_config_sha256",
    )
    if report["environment_config_sha256"] != _digest(report["environment_config"]):
        raise ValueError("environment config digest is invalid")
    expected_arms = [_ARMS_BY_NAME[name].to_config() for name in protocol.arm_names]
    if not _exact_json_equal(report["arm_definitions"], expected_arms):
        raise ValueError("arm definitions are invalid")

    raw_runs = report["runs"]
    if not isinstance(raw_runs, list) or len(raw_runs) != len(protocol.arm_names):
        raise ValueError("report runs do not match protocol arms")
    run_fields = {
        "arm",
        "seed",
        "lifecycle_ids",
        "agent_configs",
        "agent_config_sha256",
        "trace",
        "trace_sha256",
        "metrics",
        "resources",
        "work",
        "stale_replay_audit",
        "checkpoint_shadow_audits",
        "temporary_checkpoint_storage_retained",
    }
    reconstructed_runs: list[dict[str, object]] = []
    for run_index, (arm_name, raw_run) in enumerate(
        zip(protocol.arm_names, raw_runs, strict=True)
    ):
        run_name = f"runs[{run_index}]"
        if not isinstance(raw_run, Mapping) or set(raw_run) != run_fields:
            raise ValueError(f"{run_name} fields are invalid")
        if type(raw_run["arm"]) is not str or raw_run["arm"] != arm_name:
            raise ValueError(f"{run_name} arm order is invalid")
        if type(raw_run["seed"]) is not int or not 0 <= raw_run["seed"] <= _UINT32_MAX:
            raise ValueError(f"{run_name} seed is invalid")
        arm = _ARMS_BY_NAME[arm_name]
        lifecycle_ids = raw_run["lifecycle_ids"]
        expected_lifecycle_ids = [
            [_LIFECYCLE_TAG, 2 * _CANONICAL_ARM_NAMES.index(arm_name) + agent_index + 1]
            for agent_index in range(_N_AGENTS)
        ]
        if not _exact_json_equal(lifecycle_ids, expected_lifecycle_ids):
            raise ValueError(f"{run_name} lifecycle identities are invalid")
        agent_configs = raw_run["agent_configs"]
        config_digests = raw_run["agent_config_sha256"]
        expected_agent_config = _agent_config(
            protocol,
            feature_promotion_enabled=arm.feature_promotion_enabled,
        ).to_config()
        expected_config_digest = _digest(expected_agent_config)
        if (
            not isinstance(agent_configs, list)
            or len(agent_configs) != _N_AGENTS
            or not _exact_json_equal(
                agent_configs, [expected_agent_config, expected_agent_config]
            )
            or not isinstance(config_digests, list)
            or len(config_digests) != _N_AGENTS
            or any(
                _require_sha256(digest, name=f"{run_name}.agent_config_sha256")
                != expected_config_digest
                for digest in config_digests
            )
            or any(
                digest != _digest(config)
                for digest, config in zip(config_digests, agent_configs, strict=True)
            )
        ):
            raise ValueError(f"{run_name} agent configs are invalid")
        world_config = cast(Mapping[str, object], expected_agent_config["world_model"])
        if world_config.get("gamma") != 1.0:
            raise ValueError(f"{run_name} world model gamma is not continuing")
        if "partner_policy_fusion" in expected_agent_config or "ia" in expected_agent_config:
            raise ValueError(f"{run_name} unexpectedly enables partner fusion or IA")

        trace = raw_run["trace"]
        if not isinstance(trace, list) or len(trace) != protocol.total_steps:
            raise ValueError(f"{run_name} trace length is invalid")
        feature_generations = [0, 0]
        for event_index, event in enumerate(trace):
            feature_generations = _validate_event(
                event,
                index=event_index,
                protocol=protocol,
                arm=arm,
                lifecycle_ids=expected_lifecycle_ids,
                expected_feature_generations=feature_generations,
            )
            joint = cast(Mapping[str, object], cast(Mapping[str, object], event)["joint_dispatch"])
            primitives = cast(Mapping[str, list[int]], joint["primitive_actions"])
            event_agents = cast(
                list[Mapping[str, object]],
                cast(Mapping[str, object], event)["agents"],
            )
            if event_index == 0:
                if primitives["actual_actual"] != primitives["base_base"]:
                    raise ValueError(f"{run_name} initial base actions must equal dispatch")
            else:
                previous_agents = cast(
                    list[Mapping[str, object]],
                    cast(Mapping[str, object], trace[event_index - 1])["agents"],
                )
                if primitives["actual_actual"] != [
                    cast(int, item["next_committed_action"]) for item in previous_agents
                ]:
                    raise ValueError(f"{run_name} committed action alignment is invalid")
                if primitives["base_base"] != [
                    cast(int, item["next_preview_action"]) for item in previous_agents
                ]:
                    raise ValueError(f"{run_name} counterfactual action alignment is invalid")
            if [cast(int, item["action"]) for item in event_agents] != primitives[
                "actual_actual"
            ]:
                raise ValueError(f"{run_name} agent dispatch alignment is invalid")
        _validate_environment_causality(
            cast(list[Mapping[str, object]], trace),
            protocol=protocol,
            arm=arm,
            seed=raw_run["seed"],
        )
        _require_sha256(raw_run["trace_sha256"], name=f"{run_name}.trace_sha256")
        if raw_run["trace_sha256"] != _digest(trace):
            raise ValueError(f"{run_name} trace digest is invalid")
        expected_metrics = _metrics_from_trace(
            cast(list[Mapping[str, object]], trace), protocol
        )
        if not _exact_json_equal(raw_run["metrics"], expected_metrics):
            raise ValueError(f"{run_name} metrics do not reconstruct")
        _validate_resources_payload(raw_run["resources"], protocol=protocol)
        expected_work = _work_from_trace(
            cast(list[Mapping[str, object]], trace),
            cast(Mapping[str, object], raw_run["resources"]),
        )
        if not _exact_json_equal(raw_run["work"], expected_work):
            raise ValueError(f"{run_name} work does not reconstruct")
        _validate_stale_replay(
            raw_run["stale_replay_audit"],
            protocol=protocol,
            lifecycle_ids=expected_lifecycle_ids,
        )
        _validate_checkpoint_audits(
            raw_run["checkpoint_shadow_audits"],
            protocol=protocol,
            lifecycle_ids=expected_lifecycle_ids,
            expected_environment_config_sha256=expected_environment_config_sha256,
            expected_agent_config_sha256=[expected_config_digest] * _N_AGENTS,
        )
        audits = cast(list[Mapping[str, object]], raw_run["checkpoint_shadow_audits"])
        initial_actions = cast(
            Mapping[str, list[int]],
            cast(
                Mapping[str, object],
                cast(Mapping[str, object], trace[0])["joint_dispatch"],
            )["primitive_actions"],
        )["actual_actual"]
        if not _exact_json_equal(audits[0]["harness_base_actions"], initial_actions) or not (
            _exact_json_equal(audits[0]["agent_current_actions"], initial_actions)
        ):
            raise ValueError(f"{run_name} initial checkpoint base actions are invalid")
        for boundary in range(1, len(_CHECKPOINT_LABELS)):
            preceding_event = boundary * protocol.segment_length - 1
            preceding_agents = cast(
                list[Mapping[str, object]],
                cast(Mapping[str, object], trace[preceding_event])["agents"],
            )
            expected_base_actions = [
                cast(int, item["next_preview_action"]) for item in preceding_agents
            ]
            expected_current_actions = [
                cast(int, item["next_committed_action"]) for item in preceding_agents
            ]
            if not _exact_json_equal(
                audits[boundary]["harness_base_actions"], expected_base_actions
            ) or not _exact_json_equal(
                audits[boundary]["agent_current_actions"], expected_current_actions
            ):
                raise ValueError(f"{run_name} checkpoint base-action identity is invalid")
        if raw_run["temporary_checkpoint_storage_retained"] is not False:
            raise ValueError(f"{run_name} retained temporary checkpoint storage")
        reconstructed_runs.append(cast(dict[str, object], _json_clone(raw_run)))

    seeds = {run["seed"] for run in reconstructed_runs}
    if len(seeds) != 1:
        raise ValueError("all arms must use one paired development seed")
    expected_comparison = _comparison_contract(reconstructed_runs)
    if not _exact_json_equal(report["comparison_contract"], expected_comparison):
        raise ValueError("comparison contract does not reconstruct")
    all_lifecycle_ids = [
        tuple(lifecycle)
        for run in reconstructed_runs
        for lifecycle in cast(list[list[int]], run["lifecycle_ids"])
    ]
    if len(all_lifecycle_ids) != len(set(all_lifecycle_ids)):
        raise ValueError("learner lifecycle identities must be unique across arms")
    without_digest = {key: value for key, value in report.items() if key != "report_sha256"}
    _require_sha256(report["report_sha256"], name="report.report_sha256")
    if report["report_sha256"] != _digest(without_digest):
        raise ValueError("report digest is invalid")
    return cast(dict[str, object], _json_clone(report))


def validate_prototype_two_learning_agent_recurrence_report(
    report: Mapping[str, object],
) -> RecurrenceReportValidation:
    """Validate exact structure, arithmetic, and deterministic world replay.

    This deliberately does not re-execute the expensive Prototype learners;
    feature, memory, world-model, and Horde state trajectories remain
    generator-attested development diagnostics rather than scientific proof.
    """

    try:
        _reconstruct_report(report)
    except (KeyError, TypeError, ValueError) as error:
        return RecurrenceReportValidation(False, (str(error),))
    return RecurrenceReportValidation(True, ())


def prototype_two_learning_agent_recurrence_report_json(
    report: Mapping[str, object],
) -> str:
    """Return canonical JSON only for an exact structurally valid report."""

    validation = validate_prototype_two_learning_agent_recurrence_report(report)
    if not validation.valid:
        raise ValueError(
            "invalid two-learner recurrence report: " + "; ".join(validation.errors)
        )
    return _canonical_json(report)


__all__ = [
    "ACCEPTANCE_STATUS",
    "ACCEPTED_SCIENTIFIC_EVIDENCE",
    "CLAIM_ASSESSMENTS",
    "DEVELOPMENT_ONLY",
    "INTERPRETATION",
    "LIMITATIONS",
    "PROTOTYPE_TWO_LEARNING_AGENT_RECURRENCE_PROTOCOL_SCHEMA",
    "PROTOTYPE_TWO_LEARNING_AGENT_RECURRENCE_REPORT_SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "TWO_LEARNING_AGENT_RECURRENCE_ARMS",
    "PrototypeTwoLearningAgentRecurrenceProtocol",
    "TwoLearningAgentRecurrenceArm",
    "prototype_two_learning_agent_recurrence_report_json",
    "run_prototype_two_learning_agent_recurrence_development",
    "validate_prototype_two_learning_agent_recurrence_report",
]
