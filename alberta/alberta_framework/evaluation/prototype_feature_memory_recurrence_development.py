# mypy: disable-error-code="arg-type,attr-defined,call-arg,index,no-any-return"
"""Bounded, nonpromoting Prototype feature-memory recurrence harness.

This development lane runs one :class:`PrototypeAgent` through a single
``meet -> avoid -> meet`` life in :class:`RecurringTwoAgentWorld`.  One exact
transaction simultaneously exercises linear OaK control, a managed linear
Horde, the pair-feature lifecycle, feature-bound experiential memory, and a
legacy action-conditioned world model over the stable base prefix.

The visible meet/avoid cue makes this the smallest useful integration rung,
not a hidden-task result.  ``cue_masked_counterexample`` removes those two
channels without changing shape or work.  The partner is scripted rather than
learning.  Reports are in-memory, strictly reconstructable, always
``not-assessed``, and have no artifact writer or evidence-registration path.

Every arm pays for one discarded no-memory preview update per real event.  The
preview identifies OaK's counterfactual next action.  Readout-blocked arms pass
a one-hot safety mask for exactly that action, so memory still performs its
query-before-write transaction but cannot replace the action.  The preview
state is never carried into the life.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, NamedTuple, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.experiential_memory import ExperientialMemoryConfig
from alberta_framework.core.experiential_memory_policy import (
    ExperientialMemoryAdvantageGateConfig,
    ExperientialMemoryAdvantageGateDiagnostics,
)
from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeExperientialMemoryInput,
    PrototypeFeatureOaKHordeState,
    PrototypeTransition,
    measure_prototype_agent_state_resources,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureLifecycleConfig,
)
from alberta_framework.core.state_builder import IdentityStateBuilderConfig
from alberta_framework.core.types import DemonType, GVFSpec, HordeSpec, create_horde_spec
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.streams.recurring_multiagent import (
    AVOID_CONTEXT_INDEX,
    MEET_CONTEXT_INDEX,
    RELATIVE_POSITION_INDEX,
    RecurringTwoAgentState,
    RecurringTwoAgentWorld,
)

PROTOTYPE_FEATURE_MEMORY_RECURRENCE_PROTOCOL_SCHEMA: Final = (
    "alberta.prototype-feature-memory-recurrence-development.protocol.v1"
)
PROTOTYPE_FEATURE_MEMORY_RECURRENCE_REPORT_SCHEMA: Final = (
    "alberta.prototype-feature-memory-recurrence-development.report.v1"
)
ACCEPTANCE_STATUS: Final = "not-assessed"
DEVELOPMENT_ONLY: Final = True
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
ACCEPTED_SCIENTIFIC_EVIDENCE: Final = False

INTERPRETATION: Final = (
    "Development-only visible-cue A-B-A integration trace; not scientific evidence "
    "and not an Alberta Plan completion certificate."
)
LIMITATIONS: Final = (
    "the meet/avoid task cue is learner-visible in every arm except the declared counterexample",
    "the partner is scripted and does not learn or co-adapt",
    "the world model consumes only the stable base prefix; generated pair tails are "
    "not modeled, and dreaming, IA, and partner fusion remain excluded",
    "the synthetic bounded world supplies exact zero uncertainty and safety cost to "
    "exercise memory readout; these gates are not learned or calibrated",
    "the experiential-memory similarity gate is fixed analytically from the visible-cue "
    "key separation; it is neither learned nor calibrated, and the consumed development "
    "sweep is not evidence",
    "the conservative outcome gate uses the exact defaults-off mechanism defaults without "
    "threshold tuning; its local immediate-reward evidence is associational, has no "
    "delayed-return credit or uncertainty interval, and cannot identify aliased contexts",
    "feature-promotion-blocked arms disable replacement scheduling while retaining the "
    "fixed candidate bank and candidate-product learning work; curation cadence therefore "
    "differs by design",
    "logical persistent-state bytes and work bounds exclude compiled trace buffers, "
    "compiler workspaces, allocator residency, and FLOPs",
    "the eager reference and compiled scan runners make no hardware-latency or "
    "realized-compute-parity claim",
    "cross-engine gate authority is compared exactly, but outer JIT fusion may introduce "
    "bounded Horde float drift; full-state float tolerance and semantic-digest equality "
    "are not claimed",
    "on a rejected event the compiled runner guarantees exact carry rollback and no later "
    "active events, not eager-equivalent short-circuit work or first-failure timing inside "
    "that event",
    "semantic state digests normalize only documented non-learning birth_timestamp and "
    "uptime_s wall-clock telemetry while hashing every causal state leaf",
    "there are no thresholds, held-out seeds, confidence intervals, artifact "
    "writer, or promotion path",
)

_PHASE_NAMES: Final = ("A1", "B", "A2")
_N_HORDE_DEMONS: Final = 2
_N_ACTIONS: Final = 2
_N_OPTIONS: Final = 1
_WORLD_MODEL_BUFFER_CAPACITY: Final = 1
_WORLD_MODEL_STEP_SIZE: Final = 0.02
_UINT32_MAX: Final = 2**32 - 1
_LIFECYCLE_TAG: Final = 0x50464D52  # ASCII "PFMR"
_SOURCE_ID: Final = 0x5046
_SEMANTIC_STATE_DIGEST_SCHEMA: Final = (
    "alberta.prototype-feature-memory-recurrence-development.semantic-state.v1"
)
_EXECUTION_CONTRACT_SCHEMA: Final = (
    "alberta.prototype-feature-memory-recurrence-development.execution.v1"
)
_EAGER_EXECUTION_ENGINE: Final = "python-eager-reference"
_COMPILED_EXECUTION_ENGINE: Final = "jax-jit-scan"
_EXECUTION_ENGINES: Final = (_EAGER_EXECUTION_ENGINE, _COMPILED_EXECUTION_ENGINE)
_CROSS_ENGINE_HORDE_FLOAT_MAX_ABS_TOLERANCE: Final = 1.0e-7
_NON_SEMANTIC_WALL_CLOCK_SUFFIXES: Final = (".birth_timestamp", ".uptime_s")
_SEMANTIC_STATE_NORMALIZATION: Final = (
    "documented non-learning birth_timestamp and uptime_s leaves are zero-normalized"
)


@dataclasses.dataclass(frozen=True, slots=True)
class RecurrenceArm:
    """One exact matched development arm."""

    name: str
    memory_readout_enabled: bool
    feature_promotion_enabled: bool
    conservative_outcome_gate_enabled: bool
    cue_visible: bool
    role: str

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


RECURRENCE_ARMS: Final = (
    RecurrenceArm(
        "full",
        memory_readout_enabled=True,
        feature_promotion_enabled=True,
        conservative_outcome_gate_enabled=False,
        cue_visible=True,
        role="integrated candidate",
    ),
    RecurrenceArm(
        "memory_readout_blocked",
        memory_readout_enabled=False,
        feature_promotion_enabled=True,
        conservative_outcome_gate_enabled=False,
        cue_visible=True,
        role="matched memory behavioral-authority ablation",
    ),
    RecurrenceArm(
        "feature_promotion_blocked",
        memory_readout_enabled=True,
        feature_promotion_enabled=False,
        conservative_outcome_gate_enabled=False,
        cue_visible=True,
        role="fixed-shape promotion ablation with replacement scheduling disabled",
    ),
    RecurrenceArm(
        "dual_blocked",
        memory_readout_enabled=False,
        feature_promotion_enabled=False,
        conservative_outcome_gate_enabled=False,
        cue_visible=True,
        role="fixed-shape joint ablation with replacement scheduling disabled",
    ),
    RecurrenceArm(
        "cue_masked_counterexample",
        memory_readout_enabled=True,
        feature_promotion_enabled=True,
        conservative_outcome_gate_enabled=False,
        cue_visible=False,
        role="same-shape visible-cue-dependence counterexample",
    ),
    RecurrenceArm(
        "conservative_outcome_gate",
        memory_readout_enabled=True,
        feature_promotion_enabled=True,
        conservative_outcome_gate_enabled=True,
        cue_visible=True,
        role="exact-default local immediate-reward authority gate",
    ),
    RecurrenceArm(
        "conservative_outcome_gate_cue_masked",
        memory_readout_enabled=True,
        feature_promotion_enabled=True,
        conservative_outcome_gate_enabled=True,
        cue_visible=False,
        role="same-shape gate context-aliasing counterexample",
    ),
)
_ARMS_BY_NAME: Final = {arm.name: arm for arm in RECURRENCE_ARMS}
_CANONICAL_ARM_NAMES: Final = tuple(arm.name for arm in RECURRENCE_ARMS)


def _conservative_outcome_gate_contract() -> dict[str, object]:
    """Return the exact untuned gate construction owned by the two new arms."""

    return {
        "configured_arms": [
            "conservative_outcome_gate",
            "conservative_outcome_gate_cue_masked",
        ],
        "config": ExperientialMemoryAdvantageGateConfig().to_config(),
        "threshold_tuning_performed": False,
        "evidence_semantics": "local similarity-weighted immediate observed reward",
        "causal_or_delayed_return_claimed": False,
    }


def _require_exact_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an exact integer in [{minimum}, {maximum}]")
    return value


def _require_exact_float(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    if type(value) is not float or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be a finite exact float in [{minimum}, {maximum}]")
    return value


def _exact_json_equal(left: object, right: object) -> bool:
    """Compare canonical JSON values without Python's bool/int aliasing."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_mapping = cast(dict[object, object], left)
        right_mapping = cast(dict[object, object], right)
        return set(left_mapping) == set(right_mapping) and all(
            _exact_json_equal(left_mapping[key], right_mapping[key])
            for key in left_mapping
        )
    if type(left) is list:
        left_values = cast(list[object], left)
        right_values = cast(list[object], right)
        return len(left_values) == len(right_values) and all(
            _exact_json_equal(left_value, right_value)
            for left_value, right_value in zip(left_values, right_values, strict=True)
        )
    return bool(left == right)


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeFeatureMemoryRecurrenceProtocol:
    """Static three-segment development protocol.

    The default is the declared ``3 x 512`` life.  Tests may use shorter
    positive segment lengths while preserving the identical transaction and
    schema contracts.
    """

    segment_length: int = 512
    nuisance_dim: int = 2
    nuisance_scale: float = 1.0
    active_pair_slots: int = 4
    memory_capacity: int = 64
    replacement_interval: int = 64
    metric_window: int = 64
    arm_names: tuple[str, ...] = _CANONICAL_ARM_NAMES
    schema_version: str = PROTOTYPE_FEATURE_MEMORY_RECURRENCE_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PROTOTYPE_FEATURE_MEMORY_RECURRENCE_PROTOCOL_SCHEMA:
            raise ValueError("protocol schema_version is unsupported")
        _require_exact_int(
            self.segment_length,
            name="segment_length",
            minimum=1,
            maximum=(2**31 - 1) // 3,
        )
        _require_exact_int(
            self.nuisance_dim,
            name="nuisance_dim",
            minimum=0,
            maximum=16,
        )
        _require_exact_float(
            self.nuisance_scale,
            name="nuisance_scale",
            minimum=0.0,
            maximum=100.0,
        )
        base_dim = 6 + self.nuisance_dim
        pair_space = base_dim * (base_dim - 1) // 2
        _require_exact_int(
            self.active_pair_slots,
            name="active_pair_slots",
            minimum=1,
            maximum=pair_space,
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
            "schedule": list(_PHASE_NAMES),
            "segment_length": self.segment_length,
            "total_steps": self.total_steps,
            "nuisance_dim": self.nuisance_dim,
            "nuisance_scale": self.nuisance_scale,
            "active_pair_slots": self.active_pair_slots,
            "candidate_pair_slots": self.candidate_pair_slots,
            "memory_capacity": self.memory_capacity,
            "replacement_interval": self.replacement_interval,
            "metric_window": self.metric_window,
            "arm_names": list(self.arm_names),
            "preview_contract": (
                "every arm executes and discards one no-memory preview update per event"
            ),
            "conservative_outcome_gate_contract": (
                _conservative_outcome_gate_contract()
            ),
            "world_model_contract": {
                "coordinates": "stable_base_only",
                "generated_pair_tail_modeled": False,
                "buffer_capacity": _WORLD_MODEL_BUFFER_CAPACITY,
                "real_update_calls_per_event": 2,
                "committed_updates_per_event": 1,
            },
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> PrototypeFeatureMemoryRecurrenceProtocol:
        expected = {
            "schema_version",
            "type",
            "development_only",
            "scientific_promotion_allowed",
            "schedule",
            "segment_length",
            "total_steps",
            "nuisance_dim",
            "nuisance_scale",
            "active_pair_slots",
            "candidate_pair_slots",
            "memory_capacity",
            "replacement_interval",
            "metric_window",
            "arm_names",
            "preview_contract",
            "conservative_outcome_gate_contract",
            "world_model_contract",
        }
        if set(payload) != expected:
            raise ValueError("protocol fields do not match the v1 schema")
        if payload["type"] != cls.__name__:
            raise ValueError("protocol type is unsupported")
        if payload["development_only"] is not True:
            raise ValueError("protocol must remain development-only")
        if payload["scientific_promotion_allowed"] is not False:
            raise ValueError("protocol cannot allow scientific promotion")
        if payload["schedule"] != list(_PHASE_NAMES):
            raise ValueError("protocol schedule must remain A1-B-A2")
        if payload["preview_contract"] != (
            "every arm executes and discards one no-memory preview update per event"
        ):
            raise ValueError("protocol preview contract changed")
        if not _exact_json_equal(
            payload["conservative_outcome_gate_contract"],
            _conservative_outcome_gate_contract(),
        ):
            raise ValueError("protocol conservative outcome-gate contract changed")
        if payload["world_model_contract"] != {
            "coordinates": "stable_base_only",
            "generated_pair_tail_modeled": False,
            "buffer_capacity": _WORLD_MODEL_BUFFER_CAPACITY,
            "real_update_calls_per_event": 2,
            "committed_updates_per_event": 1,
        }:
            raise ValueError("protocol world-model contract changed")
        raw_arms = payload["arm_names"]
        if not isinstance(raw_arms, list) or any(type(value) is not str for value in raw_arms):
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
        if payload["total_steps"] != protocol.total_steps:
            raise ValueError("protocol total_steps does not reconstruct")
        if payload["candidate_pair_slots"] != protocol.candidate_pair_slots:
            raise ValueError("protocol candidate_pair_slots does not reconstruct")
        if not _exact_json_equal(protocol.to_config(), dict(payload)):
            raise ValueError("protocol payload is not canonical")
        return protocol


@dataclasses.dataclass(frozen=True, slots=True)
class RecurrenceReportValidation:
    """Strict in-memory report validation result."""

    valid: bool
    errors: tuple[str, ...]


class _CompiledRecurrenceCarry(NamedTuple):
    """Fixed scan carry for one exact compiled arm."""

    environment_state: RecurringTwoAgentState
    agent_state: PrototypeAgentState
    counterfactual_base_action: jax.Array
    stale_transition: PrototypeTransition
    stale_memory_input: PrototypeExperientialMemoryInput
    stale_fixture_captured: jax.Array
    life_valid: jax.Array


class _CompiledTraceEvent(NamedTuple):
    """Device-side trace projection plus private fail-closed checks."""

    environment_pre_words: jax.Array
    environment_post_words: jax.Array
    prototype_pre_step_words: jax.Array
    prototype_post_step_words: jax.Array
    prototype_decision_id: jax.Array
    action: jax.Array
    counterfactual_base_action: jax.Array
    reward: jax.Array
    counterfactual_reward: jax.Array
    horde_prediction: jax.Array
    horde_cumulant: jax.Array
    horde_squared_error: jax.Array
    world_model_prediction_error: jax.Array
    feature_generation_pre_words: jax.Array
    feature_generation_post_words: jax.Array
    a_critical_pair_count: jax.Array
    b_critical_pair_count: jax.Array
    curation_committed: jax.Array
    feature_memory_rebind_applied: jax.Array
    memory_rows_reencoded: jax.Array
    memory_query_before_write: jax.Array
    memory_prestate_query_count: jax.Array
    memory_wrote: jax.Array
    memory_retrieval_available: jax.Array
    memory_action_changed: jax.Array
    memory_advantage_gate_configured: jax.Array
    memory_advantage_gate_evidence_valid: jax.Array
    memory_advantage_gate_actions_differ: jax.Array
    memory_advantage_gate_base_support_count: jax.Array
    memory_advantage_gate_proposed_support_count: jax.Array
    memory_advantage_gate_base_action_weight_mass: jax.Array
    memory_advantage_gate_proposed_action_weight_mass: jax.Array
    memory_advantage_gate_weight_mass_ready: jax.Array
    memory_advantage_gate_support_ready: jax.Array
    memory_advantage_gate_base_reward_mean: jax.Array
    memory_advantage_gate_proposed_reward_mean: jax.Array
    memory_advantage_gate_reward_advantage: jax.Array
    memory_advantage_gate_advantage_ready: jax.Array
    memory_advantage_gate_replacement_allowed: jax.Array
    memory_advantage_gate_dispatch_consistent: jax.Array
    actual_environment_applied: jax.Array
    counterfactual_environment_applied: jax.Array
    environment_identity_preserved: jax.Array
    action_valid: jax.Array
    preview_valid: jax.Array
    committed_update_valid: jax.Array
    world_model_error_finite: jax.Array
    preview_action_matches: jax.Array
    blocked_action_preserved: jax.Array
    source_state_consistent: jax.Array
    event_committed: jax.Array


class _CompiledReplayAudit(NamedTuple):
    """Device-side exact stale-transition replay result."""

    stale_decision_rejected: jax.Array
    state_bit_exact: jax.Array
    agent_clock_unchanged: jax.Array


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_clone(value: object) -> object:
    return json.loads(_canonical_json(value))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _execution_contract(engine: str) -> dict[str, object]:
    """Bind one report to its declared runner and the exact live module source."""

    if engine not in _EXECUTION_ENGINES:
        raise ValueError("execution engine is unsupported")
    module_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    engine_source_sha256 = hashlib.sha256(
        f"{engine}\0{module_source_sha256}".encode()
    ).hexdigest()
    return {
        "schema_version": _EXECUTION_CONTRACT_SCHEMA,
        "engine": engine,
        "module_source_sha256": module_source_sha256,
        "engine_source_sha256": engine_source_sha256,
        "accepted_event_semantics": (
            "same committed causal transaction contract; cross-engine float bit equality "
            "is not claimed"
        ),
        "cross_engine_gate_authority_exact_fields": [
            "actions",
            "rewards",
            "gate_diagnostics",
            "agent_config",
            "resources",
            "work",
        ],
        "focused_cross_engine_horde_float_max_abs_tolerance": (
            _CROSS_ENGINE_HORDE_FLOAT_MAX_ABS_TOLERANCE
        ),
        "cross_engine_full_state_float_tolerance_claimed": False,
        "cross_engine_semantic_state_digest_equality_claimed": False,
        "rejected_event_semantics": (
            "atomic carry rollback and no later active events; short-circuit work and "
            "first-failure timing are not claimed equivalent"
        ),
    }


def _pytree_semantic_sha256(value: object) -> str:
    """Hash a complete PyTree structure and every exact array byte."""

    path_leaves, tree = jax.tree_util.tree_flatten_with_path(value)
    leaves: list[dict[str, object]] = []
    for path, leaf in path_leaves:
        path_string = jax.tree_util.keystr(path)
        dtype = getattr(leaf, "dtype", None)
        prng_impl: str | None = None
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            prng_impl = str(jr.key_impl(leaf))
            leaf = jr.key_data(leaf)
        host = np.asarray(jax.device_get(leaf))
        if host.dtype.hasobject:
            raise TypeError("semantic state digest does not support object arrays")
        canonical_dtype = host.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(host.astype(canonical_dtype, copy=False))
        wall_clock_telemetry_normalized = path_string.endswith(
            _NON_SEMANTIC_WALL_CLOCK_SUFFIXES
        )
        if wall_clock_telemetry_normalized:
            canonical = np.zeros_like(canonical)
        leaves.append(
            {
                "path": path_string,
                "shape": [int(size) for size in canonical.shape],
                "dtype": canonical.dtype.str,
                "weak_type": bool(getattr(leaf, "weak_type", False)),
                "prng_impl": prng_impl,
                "wall_clock_telemetry_normalized": wall_clock_telemetry_normalized,
                "data_sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
            }
        )
    return _digest(
        {
            "root_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "tree": str(tree),
            "leaves": leaves,
        }
    )


def _semantic_state_snapshot(
    label: str,
    event_count: int,
    agent_state: PrototypeAgentState,
    environment_state: RecurringTwoAgentState,
    counterfactual_base_action: object,
) -> dict[str, object]:
    """Project one full causal carry boundary into exact component digests."""

    base_action = int(np.asarray(jax.device_get(counterfactual_base_action)))
    if base_action not in (0, 1):
        raise RuntimeError("semantic state snapshot has an invalid base action")
    components = {
        "agent_state_sha256": _pytree_semantic_sha256(agent_state),
        "environment_state_sha256": _pytree_semantic_sha256(environment_state),
        "counterfactual_base_action": base_action,
    }
    return {
        "label": label,
        "event_count": event_count,
        **components,
        "joint_state_sha256": _digest(components),
    }


def _semantic_state_audit(
    phase_boundaries: list[dict[str, object]],
) -> dict[str, object]:
    """Seal the four initial/phase-boundary snapshots for one arm."""

    if len(phase_boundaries) != 4:
        raise ValueError("semantic state audit requires four phase boundaries")
    return {
        "schema_version": _SEMANTIC_STATE_DIGEST_SCHEMA,
        "normalization": _SEMANTIC_STATE_NORMALIZATION,
        "phase_boundaries": phase_boundaries,
        "final_joint_state_sha256": phase_boundaries[-1]["joint_state_sha256"],
    }


def _words(value: Any) -> list[int]:
    array = np.asarray(jax.device_get(value), dtype=np.uint32)
    if array.shape != (2,):
        raise ValueError("exact clock must contain two uint32 words")
    return [int(array[0]), int(array[1])]


def _decision_id(value: Any) -> list[int]:
    array = np.asarray(jax.device_get(value), dtype=np.uint32)
    if array.shape != (4,):
        raise ValueError("Prototype decision identity must contain four uint32 words")
    return [int(item) for item in array]


def _words_value(value: object, *, name: str) -> int:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain two words")
    high, low = value
    for index, word in enumerate((high, low)):
        if type(word) is not int or not 0 <= word <= _UINT32_MAX:
            raise ValueError(f"{name}[{index}] must be uint32")
    return cast(int, high) * 2**32 + cast(int, low)


def _phase_for_step(step: int, segment_length: int) -> str:
    return _PHASE_NAMES[step // segment_length]


def _masked_observation(observation: Any, *, cue_visible: bool) -> jax.Array:
    result = jnp.asarray(observation, dtype=jnp.float32)
    if cue_visible:
        return result
    return result.at[
        jnp.asarray((MEET_CONTEXT_INDEX, AVOID_CONTEXT_INDEX), dtype=jnp.int32)
    ].set(0.0)


def _horde_spec() -> HordeSpec:
    return create_horde_spec(
        (
            GVFSpec(
                name="prequential_reward",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=0,
            ),
            GVFSpec(
                name="prequential_next_distance",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=1,
            ),
        )
    )


def _agent_config(
    protocol: PrototypeFeatureMemoryRecurrenceProtocol,
    *,
    feature_promotion_enabled: bool,
    conservative_outcome_gate_enabled: bool = False,
) -> PrototypeAgentConfig:
    base = protocol.base_observation_dim
    total = base + protocol.active_pair_slots
    horde_spec = _horde_spec()
    lifecycle = PrototypeFeatureLifecycleConfig(
        base_feature_dim=base,
        active_pair_slots=protocol.active_pair_slots,
        candidate_pair_slots=protocol.candidate_pair_slots,
        n_tasks=1 + _N_HORDE_DEMONS,
        n_options=_N_OPTIONS,
        n_primitive_actions=_N_ACTIONS,
        option_subtask_feature_indices=(RELATIVE_POSITION_INDEX,),
        step_size_output=0.02,
        utility_decay=0.99,
        replacement_interval=(
            protocol.replacement_interval if feature_promotion_enabled else 0
        ),
        min_feature_age=1,
        candidate_min_age=1,
        promotion_margin=1.0,
        scale_normalizer_decay=0.99,
        scale_normalizer_epsilon=1.0e-6,
        carry_survivors=True,
        max_observations=protocol.total_steps,
        managed_horde_demons=_N_HORDE_DEMONS,
    )
    oak = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(
                    feature_index=RELATIVE_POSITION_INDEX,
                    threshold=1.0e6,
                    max_option_steps=4,
                ),
            ),
            observation_dim=total,
            n_primitive_actions=_N_ACTIONS,
            base_hidden_sizes=(),
            base_step_size=0.02,
            option_step_size=0.02,
            epsilon_base=0.1,
            epsilon_option=0.1,
        )
    )
    capacity = protocol.memory_capacity
    memory = ExperientialMemoryConfig(
        capacity=capacity,
        observation_dim=total,
        key_dim=total,
        action_dim=_N_ACTIONS,
        outcome_dim=total + 1,
        top_k=min(4, capacity),
        min_neighbors=1,
        distance_scale=1.0,
        min_similarity=min(1.0, math.exp(-2.0 / float(total)) + 1.0e-6),
        min_effective_reliability=0.01,
        max_uncertainty=1.0,
        max_safety_cost=1.0,
        max_age=protocol.total_steps,
        staleness_scale=float(protocol.total_steps),
        utility_decay=0.99,
        eviction_utility_weight=1.0,
        eviction_recency_weight=1.0,
        recency_scale=float(capacity),
    )
    return PrototypeAgentConfig(
        oak=oak,
        state_builder=IdentityStateBuilderConfig(observation_dim=base),
        world_model=ActionConditionedWorldModelConfig(
            observation_dim=base,
            n_actions=_N_ACTIONS,
            gamma=1.0,
            hidden_sizes=(),
            step_size=_WORLD_MODEL_STEP_SIZE,
            sparsity=0.0,
            use_layer_norm=False,
            include_action_interactions=True,
        ),
        buffer_capacity=_WORLD_MODEL_BUFFER_CAPACITY,
        horde_spec=horde_spec,
        horde_hidden_sizes=(),
        horde_step_size=0.05,
        prototype_feature_lifecycle=lifecycle,
        experiential_memory=memory,
        experiential_memory_advantage_gate=(
            ExperientialMemoryAdvantageGateConfig()
            if conservative_outcome_gate_enabled
            else None
        ),
    )


def _feature_bundle(state: PrototypeAgentState) -> PrototypeFeatureOaKHordeState:
    if type(state.oak_state) is not PrototypeFeatureOaKHordeState:
        raise RuntimeError("recurrence harness requires the exact shared feature-Horde bundle")
    return state.oak_state


def _descriptors(state: PrototypeAgentState) -> np.ndarray:
    return np.asarray(
        jax.device_get(_feature_bundle(state).consumer_binding.descriptors),
        dtype=np.int32,
    )


def _critical_counts(state: PrototypeAgentState) -> tuple[int, int]:
    descriptors = {tuple(int(value) for value in row) for row in _descriptors(state)}
    return (
        int((RELATIVE_POSITION_INDEX, MEET_CONTEXT_INDEX) in descriptors),
        int((RELATIVE_POSITION_INDEX, AVOID_CONTEXT_INDEX) in descriptors),
    )


def _tree_bit_exact(left: object, right: object) -> bool:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if left_tree != right_tree or len(left_leaves) != len(right_leaves):  # type: ignore[operator]
        return False
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_dtype = getattr(left_leaf, "dtype", None)
        right_dtype = getattr(right_leaf, "dtype", None)
        if left_dtype is not None and jax.dtypes.issubdtype(left_dtype, jax.dtypes.prng_key):
            left_leaf = jr.key_data(left_leaf)
        if right_dtype is not None and jax.dtypes.issubdtype(right_dtype, jax.dtypes.prng_key):
            right_leaf = jr.key_data(right_leaf)
        if not np.array_equal(
            np.asarray(jax.device_get(left_leaf)),
            np.asarray(jax.device_get(right_leaf)),
        ):
            return False
    return True


def _primitive_to_continuous(action: int) -> jax.Array:
    if action not in (0, 1):
        raise ValueError("Prototype primitive action must be zero or one")
    return jnp.asarray(-1.0 if action == 0 else 1.0, dtype=jnp.float32)


def _memory_input(
    state: PrototypeAgentState,
    preview_state: PrototypeAgentState,
    *,
    event_index: int,
    reward: jax.Array,
    safe_action: int | None,
) -> PrototypeExperientialMemoryInput:
    binding = _feature_bundle(state).consumer_binding
    safety_mask = (
        jnp.ones((_N_ACTIONS,), dtype=jnp.bool_)
        if safe_action is None
        else jax.nn.one_hot(safe_action, _N_ACTIONS, dtype=jnp.bool_)
    )
    return PrototypeExperientialMemoryInput(
        available=jnp.asarray(True, dtype=jnp.bool_),
        current_prototype_decision_id=state.current_decision_id,
        next_prototype_decision_id=preview_state.current_decision_id,
        query_representation_version=binding.semantic_generation,
        entry_representation_version=binding.semantic_generation,
        query_uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
        query_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        entry_uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
        entry_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(True, dtype=jnp.bool_),
        reliability=jnp.asarray(1.0, dtype=jnp.float32),
        utility=jnp.maximum(jnp.asarray(reward, dtype=jnp.float32), 0.0),
        utility_available=jnp.asarray(True, dtype=jnp.bool_),
        provenance_id=jnp.asarray(event_index, dtype=jnp.int32),
        source_id=jnp.asarray(_SOURCE_ID, dtype=jnp.int32),
        next_action_safety_mask=safety_mask,
    )


def _transition(
    state: PrototypeAgentState,
    *,
    reward: jax.Array,
    discount: jax.Array,
    terminated: jax.Array,
    next_observation: jax.Array,
) -> PrototypeTransition:
    cumulants = jnp.asarray(
        (reward, jnp.abs(next_observation[RELATIVE_POSITION_INDEX])),
        dtype=jnp.float32,
    )
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(discount, dtype=jnp.float32),
        terminated=jnp.asarray(terminated, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=next_observation,
        next_decision_observation=next_observation,
        horde_cumulants=cumulants,
        horde_discounts=jnp.zeros((_N_HORDE_DEMONS,), dtype=jnp.float32),
    )


def _compiled_masked_observation(
    observation: jax.Array,
    cue_visible: jax.Array,
) -> jax.Array:
    """Mask the visible task cue without a Python/static arm branch."""

    visible = jnp.asarray(observation, dtype=jnp.float32)
    masked = visible.at[
        jnp.asarray((MEET_CONTEXT_INDEX, AVOID_CONTEXT_INDEX), dtype=jnp.int32)
    ].set(0.0)
    return jnp.where(jnp.asarray(cue_visible, dtype=jnp.bool_), visible, masked)


def _compiled_continuous_action(action: jax.Array) -> jax.Array:
    """Map the two primitive actions to the world's scalar action array."""

    primitive = jnp.asarray(action, dtype=jnp.int32)
    return jnp.where(primitive == 0, -1.0, 1.0).astype(jnp.float32)


def _compiled_memory_input(
    state: PrototypeAgentState,
    preview_state: PrototypeAgentState,
    *,
    preview_action: jax.Array,
    event_index: jax.Array,
    reward: jax.Array,
    memory_readout_enabled: jax.Array,
) -> PrototypeExperientialMemoryInput:
    """Build the fixed sidecar while selecting readout authority dynamically."""

    sidecar = _memory_input(
        state,
        preview_state,
        event_index=event_index,
        reward=reward,
        safe_action=None,
    )
    safety_mask = jnp.where(
        jnp.asarray(memory_readout_enabled, dtype=jnp.bool_),
        jnp.ones((_N_ACTIONS,), dtype=jnp.bool_),
        jax.nn.one_hot(preview_action, _N_ACTIONS, dtype=jnp.bool_),
    )
    return cast(
        PrototypeExperientialMemoryInput,
        sidecar.replace(next_action_safety_mask=safety_mask),
    )


def _compiled_tree_bit_exact(left: object, right: object) -> jax.Array:
    """Return exact PyTree equality without materialising device leaves."""

    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if left_tree != right_tree or len(left_leaves) != len(right_leaves):  # type: ignore[operator]
        return jnp.asarray(False, dtype=jnp.bool_)
    exact = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_dtype = getattr(left_leaf, "dtype", None)
        right_dtype = getattr(right_leaf, "dtype", None)
        if left_dtype is not None and jax.dtypes.issubdtype(
            left_dtype,
            jax.dtypes.prng_key,
        ):
            left_leaf = jr.key_data(left_leaf)
        if right_dtype is not None and jax.dtypes.issubdtype(
            right_dtype,
            jax.dtypes.prng_key,
        ):
            right_leaf = jr.key_data(right_leaf)
        exact = exact & jnp.array_equal(
            jnp.asarray(left_leaf),
            jnp.asarray(right_leaf),
        )
    return exact


@functools.partial(jax.jit, static_argnums=(0,))
def _compiled_start_agent(
    agent: PrototypeAgent,
    state: PrototypeAgentState,
    initial_observation: jax.Array,
) -> PrototypeAgentState:
    """Prime an already-initialised state behind one stable JIT boundary."""

    return agent.start(state, initial_observation)


def _compiled_initialize_arm(
    agent: PrototypeAgent,
    world: RecurringTwoAgentWorld,
    environment_key: jax.Array,
    agent_key: jax.Array,
    lifecycle_id: jax.Array,
    cue_visible: jax.Array,
) -> _CompiledRecurrenceCarry:
    """Initialise host-validating components, then compile the expensive start."""

    environment_state = world.init(environment_key)
    initial_observation = _compiled_masked_observation(
        world.observe(environment_state)[0],
        cue_visible,
    )
    agent_state = _compiled_start_agent(
        agent,
        agent.init(agent_key, lifecycle_id=lifecycle_id),
        initial_observation,
    )
    placeholder_transition = _transition(
        agent_state,
        reward=jnp.asarray(0.0, dtype=jnp.float32),
        discount=jnp.asarray(1.0, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=agent_state.current_raw_observation,
    )
    placeholder_memory_input = _memory_input(
        agent_state,
        agent_state,
        event_index=0,
        reward=jnp.asarray(0.0, dtype=jnp.float32),
        safe_action=None,
    )
    action_valid = (agent_state.current_action >= 0) & (
        agent_state.current_action < _N_ACTIONS
    )
    return _CompiledRecurrenceCarry(
        environment_state=environment_state,
        agent_state=agent_state,
        counterfactual_base_action=agent_state.current_action,
        stale_transition=placeholder_transition,
        stale_memory_input=placeholder_memory_input,
        stale_fixture_captured=jnp.asarray(False, dtype=jnp.bool_),
        life_valid=(
            agent_state.started & action_valid & agent.validate_state(agent_state)
        ),
    )


def _compiled_stopped_trace(
    carry: _CompiledRecurrenceCarry,
) -> _CompiledTraceEvent:
    """Return a fixed sentinel without performing work after a failed event."""

    false = jnp.asarray(False, dtype=jnp.bool_)
    zero_float = jnp.asarray(0.0, dtype=jnp.float32)
    zero_int = jnp.asarray(0, dtype=jnp.int32)
    zero_horde = jnp.zeros((_N_HORDE_DEMONS,), dtype=jnp.float32)
    environment_words = cast(jax.Array, carry.environment_state.step_words)
    feature_words = _feature_bundle(
        carry.agent_state
    ).consumer_binding.semantic_generation_words
    return _CompiledTraceEvent(
        environment_pre_words=environment_words,
        environment_post_words=environment_words,
        prototype_pre_step_words=carry.agent_state.step_words,
        prototype_post_step_words=carry.agent_state.step_words,
        prototype_decision_id=carry.agent_state.current_decision_id,
        action=carry.agent_state.current_action,
        counterfactual_base_action=carry.counterfactual_base_action,
        reward=zero_float,
        counterfactual_reward=zero_float,
        horde_prediction=zero_horde,
        horde_cumulant=zero_horde,
        horde_squared_error=zero_horde,
        world_model_prediction_error=zero_float,
        feature_generation_pre_words=feature_words,
        feature_generation_post_words=feature_words,
        a_critical_pair_count=zero_int,
        b_critical_pair_count=zero_int,
        curation_committed=false,
        feature_memory_rebind_applied=false,
        memory_rows_reencoded=zero_int,
        memory_query_before_write=false,
        memory_prestate_query_count=zero_int,
        memory_wrote=false,
        memory_retrieval_available=false,
        memory_action_changed=false,
        memory_advantage_gate_configured=false,
        memory_advantage_gate_evidence_valid=false,
        memory_advantage_gate_actions_differ=false,
        memory_advantage_gate_base_support_count=zero_int,
        memory_advantage_gate_proposed_support_count=zero_int,
        memory_advantage_gate_base_action_weight_mass=zero_float,
        memory_advantage_gate_proposed_action_weight_mass=zero_float,
        memory_advantage_gate_weight_mass_ready=false,
        memory_advantage_gate_support_ready=false,
        memory_advantage_gate_base_reward_mean=zero_float,
        memory_advantage_gate_proposed_reward_mean=zero_float,
        memory_advantage_gate_reward_advantage=zero_float,
        memory_advantage_gate_advantage_ready=false,
        memory_advantage_gate_replacement_allowed=false,
        memory_advantage_gate_dispatch_consistent=false,
        actual_environment_applied=false,
        counterfactual_environment_applied=false,
        environment_identity_preserved=false,
        action_valid=false,
        preview_valid=false,
        committed_update_valid=false,
        world_model_error_finite=false,
        preview_action_matches=false,
        blocked_action_preserved=false,
        source_state_consistent=false,
        event_committed=false,
    )


def _compiled_active_event(
    agent: PrototypeAgent,
    world: RecurringTwoAgentWorld,
    horde: HordeLearner,
    carry: _CompiledRecurrenceCarry,
    event_index: jax.Array,
    cue_visible: jax.Array,
    memory_readout_enabled: jax.Array,
) -> tuple[_CompiledRecurrenceCarry, _CompiledTraceEvent]:
    """Stage and atomically commit one exact recurrence event."""

    environment_state = carry.environment_state
    agent_state = carry.agent_state
    current_action = agent_state.current_action
    counterfactual_base_action = carry.counterfactual_base_action
    action_valid = (
        (current_action >= 0)
        & (current_action < _N_ACTIONS)
        & (counterfactual_base_action >= 0)
        & (counterfactual_base_action < _N_ACTIONS)
    )
    actual_environment = world.step_with_partner_result(
        environment_state,
        _compiled_continuous_action(current_action),
    )
    counterfactual_environment = world.step_with_partner_result(
        environment_state,
        _compiled_continuous_action(counterfactual_base_action),
    )
    environment_identity_preserved = jnp.array_equal(
        actual_environment.pre_step_words,
        counterfactual_environment.pre_step_words,
    ) & jnp.array_equal(
        actual_environment.post_step_words,
        counterfactual_environment.post_step_words,
    )
    next_observation = _compiled_masked_observation(
        actual_environment.transition.next_observation[0],
        cue_visible,
    )
    reward = actual_environment.transition.reward[0]
    transition = _transition(
        agent_state,
        reward=reward,
        discount=actual_environment.transition.discount,
        terminated=actual_environment.transition.terminated,
        next_observation=next_observation,
    )
    feature_bundle = _feature_bundle(agent_state)
    predictions = horde.predict(
        feature_bundle.horde_state,
        agent_state.current_representation,
    )
    cumulants = cast(jax.Array, transition.horde_cumulants)
    squared_error = jnp.square(cumulants - predictions)
    preview = agent.update_transition(agent_state, transition)
    memory_input = _compiled_memory_input(
        agent_state,
        preview.state,
        preview_action=preview.action,
        event_index=event_index,
        reward=reward,
        memory_readout_enabled=memory_readout_enabled,
    )
    result = agent.update_transition(
        agent_state,
        transition,
        experiential_memory_input=memory_input,
    )
    memory_diagnostics = result.experiential_memory_diagnostics
    feature_diagnostics = result.prototype_feature_lifecycle_diagnostics
    feature_memory_diagnostics = result.prototype_feature_memory_diagnostics
    if (
        memory_diagnostics is None
        or feature_diagnostics is None
        or feature_memory_diagnostics is None
    ):
        raise RuntimeError("configured recurrence component omitted diagnostics")
    if result.world_model_error is None:
        raise RuntimeError("configured stable-base world model omitted its error")
    world_model_error = jnp.asarray(result.world_model_error, dtype=jnp.float32)
    preview_action_matches = (
        memory_diagnostics.counterfactual_base_action == preview.action
    )
    blocked_action_preserved = jnp.asarray(
        memory_readout_enabled,
        dtype=jnp.bool_,
    ) | (result.action == preview.action)
    advantage_gate = memory_diagnostics.advantage_gate
    if advantage_gate is None:
        advantage_gate_configured = jnp.asarray(False, dtype=jnp.bool_)
        advantage_gate_evidence_valid = jnp.asarray(False, dtype=jnp.bool_)
        advantage_gate_actions_differ = jnp.asarray(False, dtype=jnp.bool_)
        advantage_gate_base_support_count = jnp.asarray(0, dtype=jnp.int32)
        advantage_gate_proposed_support_count = jnp.asarray(0, dtype=jnp.int32)
        advantage_gate_base_action_weight_mass = jnp.asarray(0.0, dtype=jnp.float32)
        advantage_gate_proposed_action_weight_mass = jnp.asarray(
            0.0, dtype=jnp.float32
        )
        advantage_gate_weight_mass_ready = jnp.asarray(False, dtype=jnp.bool_)
        advantage_gate_support_ready = jnp.asarray(False, dtype=jnp.bool_)
        advantage_gate_base_reward_mean = jnp.asarray(0.0, dtype=jnp.float32)
        advantage_gate_proposed_reward_mean = jnp.asarray(0.0, dtype=jnp.float32)
        advantage_gate_reward_advantage = jnp.asarray(0.0, dtype=jnp.float32)
        advantage_gate_advantage_ready = jnp.asarray(False, dtype=jnp.bool_)
        advantage_gate_replacement_allowed = jnp.asarray(False, dtype=jnp.bool_)
        advantage_gate_dispatch_consistent = jnp.asarray(True, dtype=jnp.bool_)
    else:
        advantage_gate_configured = jnp.asarray(True, dtype=jnp.bool_)
        advantage_gate_evidence_valid = advantage_gate.evidence_valid
        advantage_gate_actions_differ = advantage_gate.actions_differ
        advantage_gate_base_support_count = advantage_gate.base_support_count
        advantage_gate_proposed_support_count = advantage_gate.proposed_support_count
        advantage_gate_base_action_weight_mass = advantage_gate.base_action_weight_mass
        advantage_gate_proposed_action_weight_mass = (
            advantage_gate.proposed_action_weight_mass
        )
        advantage_gate_weight_mass_ready = advantage_gate.weight_mass_ready
        advantage_gate_support_ready = advantage_gate.support_ready
        advantage_gate_base_reward_mean = advantage_gate.base_reward_mean
        advantage_gate_proposed_reward_mean = advantage_gate.proposed_reward_mean
        advantage_gate_reward_advantage = advantage_gate.reward_advantage
        advantage_gate_advantage_ready = advantage_gate.advantage_ready
        advantage_gate_replacement_allowed = advantage_gate.replacement_allowed
        advantage_gate_dispatch_consistent = (
            memory_diagnostics.dispatch_replacement.applied
            == advantage_gate.replacement_allowed
        )
    source_state_consistent = result.transition_diagnostics.state_consistent
    event_committed = (
        action_valid
        & actual_environment.update_applied
        & counterfactual_environment.update_applied
        & environment_identity_preserved
        & preview.transition_diagnostics.valid
        & result.transition_diagnostics.valid
        & jnp.isfinite(world_model_error)
        & preview_action_matches
        & blocked_action_preserved
        & advantage_gate_dispatch_consistent
        & source_state_consistent
    )
    destination_descriptors = _feature_bundle(
        result.state
    ).consumer_binding.descriptors
    a_critical_pair_count = jnp.any(
        jnp.all(
            destination_descriptors
            == jnp.asarray(
                (RELATIVE_POSITION_INDEX, MEET_CONTEXT_INDEX),
                dtype=jnp.int32,
            ),
            axis=1,
        )
    ).astype(jnp.int32)
    b_critical_pair_count = jnp.any(
        jnp.all(
            destination_descriptors
            == jnp.asarray(
                (RELATIVE_POSITION_INDEX, AVOID_CONTEXT_INDEX),
                dtype=jnp.int32,
            ),
            axis=1,
        )
    ).astype(jnp.int32)
    trace = _CompiledTraceEvent(
        environment_pre_words=actual_environment.pre_step_words,
        environment_post_words=actual_environment.post_step_words,
        prototype_pre_step_words=agent_state.step_words,
        prototype_post_step_words=result.state.step_words,
        prototype_decision_id=agent_state.current_decision_id,
        action=current_action,
        counterfactual_base_action=counterfactual_base_action,
        reward=reward,
        counterfactual_reward=(
            counterfactual_environment.transition.reward[0]
        ),
        horde_prediction=predictions,
        horde_cumulant=cumulants,
        horde_squared_error=squared_error,
        world_model_prediction_error=world_model_error,
        feature_generation_pre_words=(
            feature_bundle.consumer_binding.semantic_generation_words
        ),
        feature_generation_post_words=(
            _feature_bundle(
                result.state
            ).consumer_binding.semantic_generation_words
        ),
        a_critical_pair_count=a_critical_pair_count,
        b_critical_pair_count=b_critical_pair_count,
        curation_committed=feature_diagnostics.lifecycle.curation_committed,
        feature_memory_rebind_applied=(
            feature_memory_diagnostics.rebind.transaction_applied
        ),
        memory_rows_reencoded=(
            feature_memory_diagnostics.rebind.valid_rows_reencoded
        ),
        memory_query_before_write=memory_diagnostics.query_before_write,
        memory_prestate_query_count=(
            memory_diagnostics.deterministic_prestate_query_count
        ),
        memory_wrote=memory_diagnostics.wrote,
        memory_retrieval_available=memory_diagnostics.proposal.available,
        memory_action_changed=(
            memory_diagnostics.dispatch_replacement.applied
        ),
        memory_advantage_gate_configured=advantage_gate_configured,
        memory_advantage_gate_evidence_valid=advantage_gate_evidence_valid,
        memory_advantage_gate_actions_differ=advantage_gate_actions_differ,
        memory_advantage_gate_base_support_count=(
            advantage_gate_base_support_count
        ),
        memory_advantage_gate_proposed_support_count=(
            advantage_gate_proposed_support_count
        ),
        memory_advantage_gate_base_action_weight_mass=(
            advantage_gate_base_action_weight_mass
        ),
        memory_advantage_gate_proposed_action_weight_mass=(
            advantage_gate_proposed_action_weight_mass
        ),
        memory_advantage_gate_weight_mass_ready=(
            advantage_gate_weight_mass_ready
        ),
        memory_advantage_gate_support_ready=advantage_gate_support_ready,
        memory_advantage_gate_base_reward_mean=advantage_gate_base_reward_mean,
        memory_advantage_gate_proposed_reward_mean=(
            advantage_gate_proposed_reward_mean
        ),
        memory_advantage_gate_reward_advantage=(
            advantage_gate_reward_advantage
        ),
        memory_advantage_gate_advantage_ready=advantage_gate_advantage_ready,
        memory_advantage_gate_replacement_allowed=(
            advantage_gate_replacement_allowed
        ),
        memory_advantage_gate_dispatch_consistent=(
            advantage_gate_dispatch_consistent
        ),
        actual_environment_applied=actual_environment.update_applied,
        counterfactual_environment_applied=(
            counterfactual_environment.update_applied
        ),
        environment_identity_preserved=environment_identity_preserved,
        action_valid=action_valid,
        preview_valid=preview.transition_diagnostics.valid,
        committed_update_valid=result.transition_diagnostics.valid,
        world_model_error_finite=jnp.isfinite(world_model_error),
        preview_action_matches=preview_action_matches,
        blocked_action_preserved=blocked_action_preserved,
        source_state_consistent=source_state_consistent,
        event_committed=event_committed,
    )
    capture_stale_fixture = (event_index == 0) & event_committed
    stale_transition, stale_memory_input = jax.lax.cond(
        capture_stale_fixture,
        lambda _: (transition, memory_input),
        lambda _: (carry.stale_transition, carry.stale_memory_input),
        operand=None,
    )
    accepted = _CompiledRecurrenceCarry(
        environment_state=actual_environment.state,
        agent_state=result.state,
        counterfactual_base_action=(
            memory_diagnostics.counterfactual_base_action
        ),
        stale_transition=stale_transition,
        stale_memory_input=stale_memory_input,
        stale_fixture_captured=(
            carry.stale_fixture_captured | capture_stale_fixture
        ),
        life_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    rejected = _CompiledRecurrenceCarry(
        environment_state=carry.environment_state,
        agent_state=carry.agent_state,
        counterfactual_base_action=carry.counterfactual_base_action,
        stale_transition=carry.stale_transition,
        stale_memory_input=carry.stale_memory_input,
        stale_fixture_captured=carry.stale_fixture_captured,
        life_valid=jnp.asarray(False, dtype=jnp.bool_),
    )
    next_carry = jax.lax.cond(
        event_committed,
        lambda _: accepted,
        lambda _: rejected,
        operand=None,
    )
    return next_carry, trace


def _compiled_event(
    agent: PrototypeAgent,
    world: RecurringTwoAgentWorld,
    horde: HordeLearner,
    carry: _CompiledRecurrenceCarry,
    event_index: jax.Array,
    cue_visible: jax.Array,
    memory_readout_enabled: jax.Array,
) -> tuple[_CompiledRecurrenceCarry, _CompiledTraceEvent]:
    """Execute one event or a fixed no-work sentinel after fail-stop."""

    return jax.lax.cond(
        carry.life_valid,
        lambda _: _compiled_active_event(
            agent,
            world,
            horde,
            carry,
            event_index,
            cue_visible,
            memory_readout_enabled,
        ),
        lambda _: (carry, _compiled_stopped_trace(carry)),
        operand=None,
    )


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def _compiled_scan_phase(
    agent: PrototypeAgent,
    world: RecurringTwoAgentWorld,
    horde: HordeLearner,
    carry: _CompiledRecurrenceCarry,
    event_indices: jax.Array,
    cue_visible: jax.Array,
    memory_readout_enabled: jax.Array,
) -> tuple[_CompiledRecurrenceCarry, _CompiledTraceEvent]:
    """Scan one equal-length phase under the exact event transaction."""

    return jax.lax.scan(
        lambda phase_carry, event_index: _compiled_event(
            agent,
            world,
            horde,
            phase_carry,
            event_index,
            cue_visible,
            memory_readout_enabled,
        ),
        carry,
        event_indices,
    )


@functools.partial(jax.jit, static_argnums=(0,))
def _compiled_stale_replay(
    agent: PrototypeAgent,
    state: PrototypeAgentState,
    transition: PrototypeTransition,
    memory_input: PrototypeExperientialMemoryInput,
) -> _CompiledReplayAudit:
    """Execute the one stale A1 replay without carrying its result."""

    replay = agent.update_transition(
        state,
        transition,
        experiential_memory_input=memory_input,
    )
    return _CompiledReplayAudit(
        stale_decision_rejected=replay.transition_diagnostics.rejected,
        state_bit_exact=_compiled_tree_bit_exact(replay.state, state),
        agent_clock_unchanged=(
            jnp.array_equal(replay.state.step_words, state.step_words)
            & jnp.array_equal(
                replay.state.observation_event_words,
                state.observation_event_words,
            )
        ),
    )


def _advantage_gate_trace_payload(
    diagnostics: ExperientialMemoryAdvantageGateDiagnostics | None,
    *,
    configured: bool,
    dispatch_applied: bool,
) -> dict[str, object]:
    """Return one exact host trace projection or a neutral unconfigured row."""

    if configured != (diagnostics is not None):
        raise RuntimeError("recurrence advantage-gate configuration omitted diagnostics")
    if diagnostics is None:
        return {
            "memory_advantage_gate_configured": False,
            "memory_advantage_gate_evidence_valid": False,
            "memory_advantage_gate_actions_differ": False,
            "memory_advantage_gate_base_support_count": 0,
            "memory_advantage_gate_proposed_support_count": 0,
            "memory_advantage_gate_base_action_weight_mass": 0.0,
            "memory_advantage_gate_proposed_action_weight_mass": 0.0,
            "memory_advantage_gate_weight_mass_ready": False,
            "memory_advantage_gate_support_ready": False,
            "memory_advantage_gate_base_reward_mean": 0.0,
            "memory_advantage_gate_proposed_reward_mean": 0.0,
            "memory_advantage_gate_reward_advantage": 0.0,
            "memory_advantage_gate_advantage_ready": False,
            "memory_advantage_gate_replacement_allowed": False,
            "memory_advantage_gate_dispatch_consistent": True,
        }
    replacement_allowed = bool(diagnostics.replacement_allowed)
    if dispatch_applied != replacement_allowed:
        raise RuntimeError("advantage-gate authority and memory dispatch disagreed")
    return {
        "memory_advantage_gate_configured": True,
        "memory_advantage_gate_evidence_valid": bool(diagnostics.evidence_valid),
        "memory_advantage_gate_actions_differ": bool(diagnostics.actions_differ),
        "memory_advantage_gate_base_support_count": int(
            diagnostics.base_support_count
        ),
        "memory_advantage_gate_proposed_support_count": int(
            diagnostics.proposed_support_count
        ),
        "memory_advantage_gate_base_action_weight_mass": float(
            diagnostics.base_action_weight_mass
        ),
        "memory_advantage_gate_proposed_action_weight_mass": float(
            diagnostics.proposed_action_weight_mass
        ),
        "memory_advantage_gate_weight_mass_ready": bool(
            diagnostics.weight_mass_ready
        ),
        "memory_advantage_gate_support_ready": bool(diagnostics.support_ready),
        "memory_advantage_gate_base_reward_mean": float(
            diagnostics.base_reward_mean
        ),
        "memory_advantage_gate_proposed_reward_mean": float(
            diagnostics.proposed_reward_mean
        ),
        "memory_advantage_gate_reward_advantage": float(
            diagnostics.reward_advantage
        ),
        "memory_advantage_gate_advantage_ready": bool(
            diagnostics.advantage_ready
        ),
        "memory_advantage_gate_replacement_allowed": replacement_allowed,
        "memory_advantage_gate_dispatch_consistent": True,
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return float(sum(values) / len(values))


def _phase_window(
    trace: Sequence[Mapping[str, object]],
    phase: str,
    window: int,
    *,
    tail: bool,
) -> list[Mapping[str, object]]:
    values = [event for event in trace if event["phase"] == phase]
    return values[-window:] if tail else values[:window]


def _advantage_gate_summary(
    trace: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    configured_events = [
        event for event in trace if bool(event["memory_advantage_gate_configured"])
    ]
    reason_counts = {
        "invalid_evidence": 0,
        "actions_not_different": 0,
        "insufficient_weight_mass": 0,
        "insufficient_support": 0,
        "insufficient_reward_advantage": 0,
        "unclassified": 0,
    }
    allowed = 0
    for event in configured_events:
        if bool(event["memory_advantage_gate_replacement_allowed"]):
            allowed += 1
        elif not bool(event["memory_advantage_gate_evidence_valid"]):
            reason_counts["invalid_evidence"] += 1
        elif not bool(event["memory_advantage_gate_actions_differ"]):
            reason_counts["actions_not_different"] += 1
        elif not bool(event["memory_advantage_gate_weight_mass_ready"]):
            reason_counts["insufficient_weight_mass"] += 1
        elif not bool(event["memory_advantage_gate_support_ready"]):
            reason_counts["insufficient_support"] += 1
        elif not bool(event["memory_advantage_gate_advantage_ready"]):
            reason_counts["insufficient_reward_advantage"] += 1
        else:
            reason_counts["unclassified"] += 1
    assessments = len(configured_events)
    abstained = assessments - allowed
    base_support = [
        float(cast(int, event["memory_advantage_gate_base_support_count"]))
        for event in configured_events
    ]
    proposed_support = [
        float(cast(int, event["memory_advantage_gate_proposed_support_count"]))
        for event in configured_events
    ]
    base_mass = [
        cast(float, event["memory_advantage_gate_base_action_weight_mass"])
        for event in configured_events
    ]
    proposed_mass = [
        cast(float, event["memory_advantage_gate_proposed_action_weight_mass"])
        for event in configured_events
    ]
    advantages = [
        cast(float, event["memory_advantage_gate_reward_advantage"])
        for event in configured_events
    ]
    return {
        "configured": bool(configured_events),
        "assessments_reported": assessments,
        "replacement_allowed_events": allowed,
        "abstained_events": abstained,
        "evidence_valid_events": sum(
            bool(event["memory_advantage_gate_evidence_valid"])
            for event in configured_events
        ),
        "actions_differ_events": sum(
            bool(event["memory_advantage_gate_actions_differ"])
            for event in configured_events
        ),
        "weight_mass_ready_events": sum(
            bool(event["memory_advantage_gate_weight_mass_ready"])
            for event in configured_events
        ),
        "support_ready_events": sum(
            bool(event["memory_advantage_gate_support_ready"])
            for event in configured_events
        ),
        "advantage_ready_events": sum(
            bool(event["memory_advantage_gate_advantage_ready"])
            for event in configured_events
        ),
        "abstention_reasons": reason_counts,
        "mean_base_support_count": _mean(base_support) if base_support else 0.0,
        "mean_proposed_support_count": (
            _mean(proposed_support) if proposed_support else 0.0
        ),
        "mean_base_action_weight_mass": _mean(base_mass) if base_mass else 0.0,
        "mean_proposed_action_weight_mass": (
            _mean(proposed_mass) if proposed_mass else 0.0
        ),
        "mean_reward_advantage": _mean(advantages) if advantages else 0.0,
        "minimum_reward_advantage": min(advantages) if advantages else 0.0,
        "maximum_reward_advantage": max(advantages) if advantages else 0.0,
    }


def _metrics_from_trace(
    trace: Sequence[Mapping[str, object]],
    protocol: PrototypeFeatureMemoryRecurrenceProtocol,
) -> dict[str, object]:
    phase_reward: dict[str, object] = {}
    phase_horde_mse: dict[str, object] = {}
    phase_world_model_error: dict[str, object] = {}
    for phase in _PHASE_NAMES:
        events = [event for event in trace if event["phase"] == phase]
        entry = _phase_window(trace, phase, protocol.metric_window, tail=False)
        tail = _phase_window(trace, phase, protocol.metric_window, tail=True)
        phase_reward[phase] = {
            "mean": _mean([cast(float, event["reward"]) for event in events]),
            "entry": _mean([cast(float, event["reward"]) for event in entry]),
            "tail": _mean([cast(float, event["reward"]) for event in tail]),
        }
        phase_horde_mse[phase] = {
            "mean": [
                _mean(
                    [cast(list[float], event["horde_squared_error"])[demon] for event in events]
                )
                for demon in range(_N_HORDE_DEMONS)
            ],
            "entry": [
                _mean(
                    [cast(list[float], event["horde_squared_error"])[demon] for event in entry]
                )
                for demon in range(_N_HORDE_DEMONS)
            ],
            "tail": [
                _mean(
                    [cast(list[float], event["horde_squared_error"])[demon] for event in tail]
                )
                for demon in range(_N_HORDE_DEMONS)
            ],
        }
        phase_world_model_error[phase] = {
            "mean": _mean(
                [
                    cast(float, event["world_model_prediction_error"])
                    for event in events
                ]
            ),
            "entry": _mean(
                [
                    cast(float, event["world_model_prediction_error"])
                    for event in entry
                ]
            ),
            "tail": _mean(
                [
                    cast(float, event["world_model_prediction_error"])
                    for event in tail
                ]
            ),
        }

    a1_reward = cast(dict[str, float], phase_reward["A1"])
    a2_reward = cast(dict[str, float], phase_reward["A2"])
    a1_mse = cast(dict[str, list[float]], phase_horde_mse["A1"])
    a2_mse = cast(dict[str, list[float]], phase_horde_mse["A2"])
    a1_world = cast(dict[str, float], phase_world_model_error["A1"])
    a2_world = cast(dict[str, float], phase_world_model_error["A2"])
    a1_tail = _phase_window(trace, "A1", protocol.metric_window, tail=True)
    b_tail = _phase_window(trace, "B", protocol.metric_window, tail=True)
    a2_entry = _phase_window(trace, "A2", protocol.metric_window, tail=False)
    a2_tail = _phase_window(trace, "A2", protocol.metric_window, tail=True)

    intervention_deltas = [cast(float, event["counterfactual_reward_delta"]) for event in trace]
    advantage_gate = _advantage_gate_summary(trace)
    advantage_gate["phase"] = {
        phase: _advantage_gate_summary(
            [event for event in trace if event["phase"] == phase]
        )
        for phase in _PHASE_NAMES
    }
    return {
        "phase_reward": phase_reward,
        "phase_horde_mse": phase_horde_mse,
        "phase_world_model_prediction_error": phase_world_model_error,
        "recurrence": {
            "a2_entry_minus_a1_tail_reward": a2_reward["entry"] - a1_reward["tail"],
            "a2_tail_minus_a1_tail_reward": a2_reward["tail"] - a1_reward["tail"],
            "a2_reward_reacquisition_gain": a2_reward["tail"] - a2_reward["entry"],
            "a2_entry_minus_a1_tail_horde_mse": [
                a2_mse["entry"][index] - a1_mse["tail"][index]
                for index in range(_N_HORDE_DEMONS)
            ],
            "a2_horde_reacquisition_gain": [
                a2_mse["entry"][index] - a2_mse["tail"][index]
                for index in range(_N_HORDE_DEMONS)
            ],
            "a2_entry_minus_a1_tail_world_model_error": (
                a2_world["entry"] - a1_world["tail"]
            ),
            "a2_world_model_reacquisition_gain": (
                a2_world["entry"] - a2_world["tail"]
            ),
        },
        "features": {
            "curation_commits": sum(bool(event["curation_committed"]) for event in trace),
            "memory_rebinds": sum(bool(event["feature_memory_rebind_applied"]) for event in trace),
            "rows_reencoded": sum(cast(int, event["memory_rows_reencoded"]) for event in trace),
            "a1_tail_a_pair_fraction": _mean(
                [float(cast(int, event["a_critical_pair_count"]) > 0) for event in a1_tail]
            ),
            "b_tail_b_pair_fraction": _mean(
                [float(cast(int, event["b_critical_pair_count"]) > 0) for event in b_tail]
            ),
            "a2_entry_a_pair_fraction": _mean(
                [float(cast(int, event["a_critical_pair_count"]) > 0) for event in a2_entry]
            ),
            "a2_tail_a_pair_fraction": _mean(
                [float(cast(int, event["a_critical_pair_count"]) > 0) for event in a2_tail]
            ),
        },
        "memory": {
            "query_before_write_events": sum(
                bool(event["memory_query_before_write"]) for event in trace
            ),
            "writes": sum(bool(event["memory_wrote"]) for event in trace),
            "retrievals_available": sum(
                bool(event["memory_retrieval_available"]) for event in trace
            ),
            "action_changes": sum(bool(event["memory_action_changed"]) for event in trace),
            "helpful_interventions": sum(value > 0.0 for value in intervention_deltas),
            "harmful_interventions": sum(value < 0.0 for value in intervention_deltas),
            "neutral_events": sum(value == 0.0 for value in intervention_deltas),
            "cumulative_counterfactual_reward_delta": float(sum(intervention_deltas)),
            "advantage_gate": advantage_gate,
        },
    }


def _work_from_trace(
    trace: Sequence[Mapping[str, object]],
    resources: Mapping[str, object],
) -> dict[str, int]:
    steps = len(trace)
    feature = cast(Mapping[str, object], resources["feature_lifecycle"])
    memory = cast(Mapping[str, object], resources["experiential_memory"])
    gate = cast(
        Mapping[str, object],
        resources["experiential_memory_advantage_gate"],
    )
    gate_configured = cast(bool, gate["configured"])
    gate_resources = cast(Mapping[str, object] | None, gate["resources"])
    gate_assessments = 2 * steps + 1 if gate_configured else 0
    if gate_configured != (gate_resources is not None):
        raise ValueError("advantage-gate resources do not match configuration")
    gate_action_values = (
        cast(int, gate_resources["neighbor_action_values_interpreted"])
        if gate_resources is not None
        else 0
    )
    gate_reward_values = (
        cast(int, gate_resources["neighbor_reward_values_interpreted"])
        if gate_resources is not None
        else 0
    )
    gate_weight_values = (
        cast(int, gate_resources["neighbor_weight_values_interpreted"])
        if gate_resources is not None
        else 0
    )
    gate_random_draws = (
        cast(int, gate_resources["random_draws_per_assessment"])
        if gate_resources is not None
        else 0
    )
    prototype_observes = 2 * steps
    committed_memory_queries = sum(
        cast(int, event["memory_prestate_query_count"]) for event in trace
    )
    return {
        "requested_transitions": steps,
        "committed_environment_transitions": steps,
        "counterfactual_environment_calls": steps,
        "prototype_update_calls": 2 * steps + 1,
        "discarded_preview_update_calls": steps,
        "committed_prototype_update_calls": steps,
        "identity_probe_update_calls": 1,
        "oak_update_calls": 2 * steps,
        "oak_discarded_preview_updates": steps,
        "oak_committed_updates": steps,
        "world_model_update_calls": 2 * steps,
        "world_model_discarded_preview_updates": steps,
        "world_model_committed_updates": steps,
        "horde_prediction_calls": steps,
        "horde_predictions_emitted": _N_HORDE_DEMONS * steps,
        "horde_update_calls": 2 * steps,
        "horde_discarded_preview_updates": steps,
        "horde_committed_updates": steps,
        "memory_sidecars_supplied": steps + 1,
        "memory_real_transition_sidecars_supplied": steps,
        "memory_stale_replay_sidecars_supplied": 1,
        "memory_deterministic_prestate_queries": committed_memory_queries,
        "memory_preview_prestate_query_calls": steps,
        "memory_stale_replay_prestate_query_calls": 1,
        "memory_total_prestate_query_calls": committed_memory_queries + steps + 1,
        "memory_writes": sum(bool(event["memory_wrote"]) for event in trace),
        "memory_advantage_gate_preview_assessments": steps if gate_configured else 0,
        "memory_advantage_gate_committed_assessments": (
            steps if gate_configured else 0
        ),
        "memory_advantage_gate_stale_replay_assessments": (
            1 if gate_configured else 0
        ),
        "memory_advantage_gate_total_assessments": gate_assessments,
        "memory_advantage_gate_reported_event_assessments": (
            steps if gate_configured else 0
        ),
        "memory_advantage_gate_neighbor_action_values_interpreted": (
            gate_assessments * gate_action_values
        ),
        "memory_advantage_gate_neighbor_reward_values_interpreted": (
            gate_assessments * gate_reward_values
        ),
        "memory_advantage_gate_neighbor_weight_values_interpreted": (
            gate_assessments * gate_weight_values
        ),
        "memory_advantage_gate_random_draws": (
            gate_assessments * gate_random_draws
        ),
        "prototype_feature_observe_calls": prototype_observes,
        "configured_max_active_pair_products": (
            prototype_observes * cast(int, feature["max_active_pair_products_per_observe"])
        ),
        "configured_max_candidate_pair_products": (
            prototype_observes * cast(int, feature["max_candidate_pair_products_per_observe"])
        ),
        "configured_memory_queries_per_committed_update": cast(
            int, memory["total_deterministic_prestate_queries"]
        ),
    }


def _resource_payload(
    agent: PrototypeAgent,
    world: RecurringTwoAgentWorld,
    initial_state: PrototypeAgentState,
    final_state: PrototypeAgentState,
    phase_boundary_total_nbytes: list[int],
    peak_total_nbytes: int,
) -> dict[str, object]:
    lifecycle = agent.prototype_feature_lifecycle
    feature_memory = agent.prototype_feature_memory_resource_budget
    memory = agent.experiential_memory_resource_declaration
    advantage_gate = agent.experiential_memory_advantage_gate
    world_model = agent.config.world_model
    if (
        lifecycle is None
        or feature_memory is None
        or memory is None
        or world_model is None
    ):
        raise RuntimeError(
            "recurrence harness resources require lifecycle, memory, and world model"
        )
    initial_resources = measure_prototype_agent_state_resources(initial_state)
    final_resources = measure_prototype_agent_state_resources(final_state)
    return {
        "initial_state": initial_resources.to_config(),
        "final_state": final_resources.to_config(),
        "phase_boundary_total_nbytes": phase_boundary_total_nbytes,
        "peak_total_nbytes": peak_total_nbytes,
        "environment": world.resource_budget.to_dict(),
        "feature_lifecycle": lifecycle.resource_budget().to_config(),
        "feature_memory": feature_memory.to_config(),
        "experiential_memory": memory.to_config(),
        "experiential_memory_advantage_gate": {
            "configured": advantage_gate is not None,
            "config": (
                advantage_gate.to_config() if advantage_gate is not None else None
            ),
            "resources": (
                advantage_gate.resource_declaration().to_config()
                if advantage_gate is not None
                else None
            ),
        },
        "stable_base_world_model": {
            "coordinates": "stable_base_only",
            "generated_pair_tail_modeled": False,
            "observation_dim": world_model.observation_dim,
            "buffer_capacity": agent.config.buffer_capacity,
            "world_model_bundle_nbytes": initial_resources.world_model_bundle_nbytes,
            "buffer_nbytes": initial_resources.buffer_nbytes,
        },
    }


def _run_arm(
    protocol: PrototypeFeatureMemoryRecurrenceProtocol,
    arm: RecurrenceArm,
    *,
    seed: int,
    arm_index: int,
    world: RecurringTwoAgentWorld,
    agent: PrototypeAgent,
    horde: HordeLearner,
) -> dict[str, object]:
    environment_state = world.init(jr.key(seed))
    initial_observation = _masked_observation(
        world.observe(environment_state)[0],
        cue_visible=arm.cue_visible,
    )
    lifecycle_id = jnp.asarray((_LIFECYCLE_TAG, arm_index + 1), dtype=jnp.uint32)
    agent_state = agent.start(
        agent.init(jr.key(seed ^ 0x13579BDF), lifecycle_id=lifecycle_id),
        initial_observation,
    )
    initial_agent_state = agent_state
    base_action_for_current = int(agent_state.current_action)
    semantic_phase_boundaries = [
        _semantic_state_snapshot(
            "initial",
            0,
            agent_state,
            environment_state,
            base_action_for_current,
        )
    ]
    trace: list[dict[str, object]] = []
    phase_boundary_total_nbytes = [
        measure_prototype_agent_state_resources(agent_state).total_nbytes
    ]
    peak_total_nbytes = phase_boundary_total_nbytes[0]
    stale_transition: PrototypeTransition | None = None
    stale_memory_input: PrototypeExperientialMemoryInput | None = None
    identity_audit: dict[str, object] | None = None

    for event_index in range(protocol.total_steps):
        if event_index == 2 * protocol.segment_length:
            if stale_transition is None or stale_memory_input is None:
                raise RuntimeError("A1 replay fixture was not captured")
            replay_source = agent_state
            replay = agent.update_transition(
                replay_source,
                stale_transition,
                experiential_memory_input=stale_memory_input,
            )
            identity_audit = {
                "aba_replay_attempted": True,
                "stale_decision_rejected": bool(replay.transition_diagnostics.rejected),
                "state_bit_exact": _tree_bit_exact(replay.state, replay_source),
                "agent_clock_unchanged": (
                    _words(replay.state.step_words) == _words(replay_source.step_words)
                    and _words(replay.state.observation_event_words)
                    == _words(replay_source.observation_event_words)
                ),
                "environment_unchanged": True,
                "replay_update_calls": 1,
            }

        pre_environment_words = _words(environment_state.step_words)
        pre_agent_words = _words(agent_state.step_words)
        decision = _decision_id(agent_state.current_decision_id)
        current_action = int(agent_state.current_action)
        if current_action not in (0, 1):
            raise RuntimeError("Prototype emitted an invalid primitive action")

        actual_environment = world.step_with_partner_result(
            environment_state,
            _primitive_to_continuous(current_action),
        )
        counterfactual_environment = world.step_with_partner_result(
            environment_state,
            _primitive_to_continuous(base_action_for_current),
        )
        if not bool(actual_environment.update_applied) or not bool(
            counterfactual_environment.update_applied
        ):
            raise RuntimeError("recurrence environment rejected a bounded valid event")
        if (
            _words(actual_environment.pre_step_words)
            != _words(counterfactual_environment.pre_step_words)
            or _words(actual_environment.post_step_words)
            != _words(counterfactual_environment.post_step_words)
        ):
            raise RuntimeError("counterfactual environment did not preserve event identity")

        raw_next_observation = actual_environment.transition.next_observation[0]
        next_observation = _masked_observation(
            raw_next_observation,
            cue_visible=arm.cue_visible,
        )
        reward = actual_environment.transition.reward[0]
        transition = _transition(
            agent_state,
            reward=reward,
            discount=actual_environment.transition.discount,
            terminated=actual_environment.transition.terminated,
            next_observation=next_observation,
        )

        bundle = _feature_bundle(agent_state)
        predictions = horde.predict(bundle.horde_state, agent_state.current_representation)
        cumulants = cast(jax.Array, transition.horde_cumulants)
        squared_error = jnp.square(cumulants - predictions)

        preview = agent.update_transition(agent_state, transition)
        if not bool(preview.transition_diagnostics.valid):
            raise RuntimeError("discarded no-memory preview update was rejected")
        preview_action = int(preview.action)
        sidecar = _memory_input(
            agent_state,
            preview.state,
            event_index=event_index,
            reward=reward,
            safe_action=(None if arm.memory_readout_enabled else preview_action),
        )
        result = agent.update_transition(
            agent_state,
            transition,
            experiential_memory_input=sidecar,
        )
        if not bool(result.transition_diagnostics.valid):
            raise RuntimeError("committed Prototype update was rejected")
        memory_diagnostics = result.experiential_memory_diagnostics
        feature_diagnostics = result.prototype_feature_lifecycle_diagnostics
        feature_memory_diagnostics = result.prototype_feature_memory_diagnostics
        if (
            memory_diagnostics is None
            or feature_diagnostics is None
            or feature_memory_diagnostics is None
        ):
            raise RuntimeError("configured recurrence component omitted diagnostics")
        if result.world_model_error is None:
            raise RuntimeError("configured stable-base world model omitted its error")
        world_model_prediction_error = float(result.world_model_error)
        if not math.isfinite(world_model_prediction_error):
            raise RuntimeError("stable-base world-model error was non-finite")
        if int(memory_diagnostics.counterfactual_base_action) != preview_action:
            raise RuntimeError("preview action did not match memory's counterfactual base action")
        if not arm.memory_readout_enabled and int(result.action) != preview_action:
            raise RuntimeError("readout-blocked memory changed the preview action")
        if not _tree_bit_exact(agent_state, agent_state):
            raise RuntimeError("source-state identity check failed")

        a_count, b_count = _critical_counts(result.state)
        actual_reward = float(reward)
        counterfactual_reward = float(counterfactual_environment.transition.reward[0])
        advantage_gate_payload = _advantage_gate_trace_payload(
            memory_diagnostics.advantage_gate,
            configured=arm.conservative_outcome_gate_enabled,
            dispatch_applied=bool(
                memory_diagnostics.dispatch_replacement.applied
            ),
        )
        trace.append(
            {
                "event_index": event_index,
                "phase": _phase_for_step(event_index, protocol.segment_length),
                "phase_step": event_index % protocol.segment_length,
                "environment_pre_words": pre_environment_words,
                "environment_post_words": _words(actual_environment.state.step_words),
                "prototype_pre_step_words": pre_agent_words,
                "prototype_post_step_words": _words(result.state.step_words),
                "prototype_decision_id": decision,
                "action": current_action,
                "counterfactual_base_action": base_action_for_current,
                "reward": actual_reward,
                "counterfactual_reward": counterfactual_reward,
                "counterfactual_reward_delta": actual_reward - counterfactual_reward,
                "horde_prediction": [float(value) for value in np.asarray(predictions)],
                "horde_cumulant": [float(value) for value in np.asarray(cumulants)],
                "horde_squared_error": [float(value) for value in np.asarray(squared_error)],
                "world_model_prediction_error": world_model_prediction_error,
                "feature_generation_pre_words": _words(
                    bundle.consumer_binding.semantic_generation_words
                ),
                "feature_generation_post_words": _words(
                    _feature_bundle(result.state).consumer_binding.semantic_generation_words
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
                "memory_query_before_write": bool(memory_diagnostics.query_before_write),
                "memory_prestate_query_count": int(
                    memory_diagnostics.deterministic_prestate_query_count
                ),
                "memory_wrote": bool(memory_diagnostics.wrote),
                "memory_retrieval_available": bool(memory_diagnostics.proposal.available),
                "memory_action_changed": bool(
                    memory_diagnostics.dispatch_replacement.applied
                ),
                **advantage_gate_payload,
                "preview_state_discarded": True,
                "transition_valid": True,
            }
        )
        if stale_transition is None:
            stale_transition = transition
            stale_memory_input = sidecar

        agent_state = result.state
        environment_state = actual_environment.state
        base_action_for_current = int(memory_diagnostics.counterfactual_base_action)
        measured = measure_prototype_agent_state_resources(agent_state)
        peak_total_nbytes = max(peak_total_nbytes, measured.total_nbytes)
        if (event_index + 1) % protocol.segment_length == 0:
            phase_boundary_total_nbytes.append(measured.total_nbytes)
            phase_index = (event_index + 1) // protocol.segment_length - 1
            semantic_phase_boundaries.append(
                _semantic_state_snapshot(
                    f"after_{_PHASE_NAMES[phase_index]}",
                    event_index + 1,
                    agent_state,
                    environment_state,
                    base_action_for_current,
                )
            )

    if identity_audit is None:
        raise RuntimeError("A-B-A replay audit was not executed")
    resources = _resource_payload(
        agent,
        world,
        initial_agent_state,
        agent_state,
        phase_boundary_total_nbytes,
        peak_total_nbytes,
    )
    config = cast(dict[str, object], _json_clone(agent.to_config()))
    run = {
        "arm": arm.name,
        "seed": seed,
        "lifecycle_id": [_LIFECYCLE_TAG, arm_index + 1],
        "agent_config": config,
        "agent_config_sha256": _digest(config),
        "trace": trace,
        "trace_sha256": _digest(trace),
        "metrics": _metrics_from_trace(trace, protocol),
        "semantic_state": _semantic_state_audit(semantic_phase_boundaries),
        "resources": resources,
        "work": _work_from_trace(trace, resources),
        "identity_audit": identity_audit,
    }
    return cast(dict[str, object], _json_clone(run))


def _require_compiled_phase_valid(
    carry: _CompiledRecurrenceCarry,
    trace: _CompiledTraceEvent,
    *,
    phase: str,
    phase_offset: int,
) -> None:
    """Synchronise one phase boundary and reject its first failed event."""

    if bool(jax.device_get(carry.life_valid)):
        return
    host_trace = cast(_CompiledTraceEvent, jax.device_get(trace))
    committed = np.asarray(host_trace.event_committed, dtype=np.bool_)
    failed = np.flatnonzero(~committed)
    local_index = int(failed[0]) if failed.size else 0
    gate_names = (
        "actual_environment_applied",
        "counterfactual_environment_applied",
        "environment_identity_preserved",
        "action_valid",
        "preview_valid",
        "committed_update_valid",
        "world_model_error_finite",
        "preview_action_matches",
        "blocked_action_preserved",
        "memory_advantage_gate_dispatch_consistent",
        "source_state_consistent",
    )
    failed_gates = [
        name
        for name in gate_names
        if not bool(np.asarray(getattr(host_trace, name))[local_index])
    ]
    detail = ", ".join(failed_gates) if failed_gates else "unknown gate"
    raise RuntimeError(
        f"compiled recurrence phase {phase} rejected event "
        f"{phase_offset + local_index}: {detail}"
    )


def _compiled_trace_to_events(
    trace: _CompiledTraceEvent,
    protocol: PrototypeFeatureMemoryRecurrenceProtocol,
) -> list[dict[str, object]]:
    """Bulk-materialise fixed device arrays into the unchanged v1 trace."""

    host = cast(_CompiledTraceEvent, jax.device_get(trace))
    committed = np.asarray(host.event_committed, dtype=np.bool_)
    if committed.shape != (protocol.total_steps,) or not bool(np.all(committed)):
        raise RuntimeError("compiled recurrence trace contains an uncommitted event")
    events: list[dict[str, object]] = []
    for event_index in range(protocol.total_steps):
        actual_reward = float(np.asarray(host.reward)[event_index])
        counterfactual_reward = float(
            np.asarray(host.counterfactual_reward)[event_index]
        )
        events.append(
            {
                "event_index": event_index,
                "phase": _phase_for_step(event_index, protocol.segment_length),
                "phase_step": event_index % protocol.segment_length,
                "environment_pre_words": _words(
                    np.asarray(host.environment_pre_words)[event_index]
                ),
                "environment_post_words": _words(
                    np.asarray(host.environment_post_words)[event_index]
                ),
                "prototype_pre_step_words": _words(
                    np.asarray(host.prototype_pre_step_words)[event_index]
                ),
                "prototype_post_step_words": _words(
                    np.asarray(host.prototype_post_step_words)[event_index]
                ),
                "prototype_decision_id": _decision_id(
                    np.asarray(host.prototype_decision_id)[event_index]
                ),
                "action": int(np.asarray(host.action)[event_index]),
                "counterfactual_base_action": int(
                    np.asarray(host.counterfactual_base_action)[event_index]
                ),
                "reward": actual_reward,
                "counterfactual_reward": counterfactual_reward,
                "counterfactual_reward_delta": (
                    actual_reward - counterfactual_reward
                ),
                "horde_prediction": [
                    float(value)
                    for value in np.asarray(host.horde_prediction)[event_index]
                ],
                "horde_cumulant": [
                    float(value)
                    for value in np.asarray(host.horde_cumulant)[event_index]
                ],
                "horde_squared_error": [
                    float(value)
                    for value in np.asarray(host.horde_squared_error)[event_index]
                ],
                "world_model_prediction_error": float(
                    np.asarray(host.world_model_prediction_error)[event_index]
                ),
                "feature_generation_pre_words": _words(
                    np.asarray(host.feature_generation_pre_words)[event_index]
                ),
                "feature_generation_post_words": _words(
                    np.asarray(host.feature_generation_post_words)[event_index]
                ),
                "a_critical_pair_count": int(
                    np.asarray(host.a_critical_pair_count)[event_index]
                ),
                "b_critical_pair_count": int(
                    np.asarray(host.b_critical_pair_count)[event_index]
                ),
                "curation_committed": bool(
                    np.asarray(host.curation_committed)[event_index]
                ),
                "feature_memory_rebind_applied": bool(
                    np.asarray(host.feature_memory_rebind_applied)[event_index]
                ),
                "memory_rows_reencoded": int(
                    np.asarray(host.memory_rows_reencoded)[event_index]
                ),
                "memory_query_before_write": bool(
                    np.asarray(host.memory_query_before_write)[event_index]
                ),
                "memory_prestate_query_count": int(
                    np.asarray(host.memory_prestate_query_count)[event_index]
                ),
                "memory_wrote": bool(
                    np.asarray(host.memory_wrote)[event_index]
                ),
                "memory_retrieval_available": bool(
                    np.asarray(host.memory_retrieval_available)[event_index]
                ),
                "memory_action_changed": bool(
                    np.asarray(host.memory_action_changed)[event_index]
                ),
                "memory_advantage_gate_configured": bool(
                    np.asarray(host.memory_advantage_gate_configured)[event_index]
                ),
                "memory_advantage_gate_evidence_valid": bool(
                    np.asarray(host.memory_advantage_gate_evidence_valid)[event_index]
                ),
                "memory_advantage_gate_actions_differ": bool(
                    np.asarray(host.memory_advantage_gate_actions_differ)[event_index]
                ),
                "memory_advantage_gate_base_support_count": int(
                    np.asarray(host.memory_advantage_gate_base_support_count)[event_index]
                ),
                "memory_advantage_gate_proposed_support_count": int(
                    np.asarray(host.memory_advantage_gate_proposed_support_count)[
                        event_index
                    ]
                ),
                "memory_advantage_gate_base_action_weight_mass": float(
                    np.asarray(host.memory_advantage_gate_base_action_weight_mass)[
                        event_index
                    ]
                ),
                "memory_advantage_gate_proposed_action_weight_mass": float(
                    np.asarray(host.memory_advantage_gate_proposed_action_weight_mass)[
                        event_index
                    ]
                ),
                "memory_advantage_gate_weight_mass_ready": bool(
                    np.asarray(host.memory_advantage_gate_weight_mass_ready)[event_index]
                ),
                "memory_advantage_gate_support_ready": bool(
                    np.asarray(host.memory_advantage_gate_support_ready)[event_index]
                ),
                "memory_advantage_gate_base_reward_mean": float(
                    np.asarray(host.memory_advantage_gate_base_reward_mean)[event_index]
                ),
                "memory_advantage_gate_proposed_reward_mean": float(
                    np.asarray(host.memory_advantage_gate_proposed_reward_mean)[event_index]
                ),
                "memory_advantage_gate_reward_advantage": float(
                    np.asarray(host.memory_advantage_gate_reward_advantage)[event_index]
                ),
                "memory_advantage_gate_advantage_ready": bool(
                    np.asarray(host.memory_advantage_gate_advantage_ready)[event_index]
                ),
                "memory_advantage_gate_replacement_allowed": bool(
                    np.asarray(host.memory_advantage_gate_replacement_allowed)[event_index]
                ),
                "memory_advantage_gate_dispatch_consistent": bool(
                    np.asarray(host.memory_advantage_gate_dispatch_consistent)[event_index]
                ),
                "preview_state_discarded": True,
                "transition_valid": True,
            }
        )
    return events


def _concatenate_compiled_traces(
    traces: Sequence[_CompiledTraceEvent],
) -> _CompiledTraceEvent:
    """Concatenate the three equal-length phase projections leafwise."""

    if len(traces) != len(_PHASE_NAMES):
        raise ValueError("compiled recurrence requires exactly three phase traces")
    return cast(
        _CompiledTraceEvent,
        jax.tree.map(
            lambda *values: jnp.concatenate(values, axis=0),
            *traces,
        ),
    )


def _run_arm_compiled(
    protocol: PrototypeFeatureMemoryRecurrenceProtocol,
    arm: RecurrenceArm,
    *,
    seed: int,
    arm_index: int,
    world: RecurringTwoAgentWorld,
    agent: PrototypeAgent,
    horde: HordeLearner,
) -> dict[str, object]:
    """Run one arm through three compiled scans and one exact stale replay."""

    lifecycle_id = jnp.asarray(
        (_LIFECYCLE_TAG, arm_index + 1),
        dtype=jnp.uint32,
    )
    cue_visible = jnp.asarray(arm.cue_visible, dtype=jnp.bool_)
    memory_readout_enabled = jnp.asarray(
        arm.memory_readout_enabled,
        dtype=jnp.bool_,
    )
    carry = _compiled_initialize_arm(
        agent,
        world,
        jr.key(seed),
        jr.key(seed ^ 0x13579BDF),
        lifecycle_id,
        cue_visible,
    )
    if not bool(jax.device_get(carry.life_valid)):
        raise RuntimeError("compiled recurrence failed to initialise an exact arm")
    initial_agent_state = carry.agent_state
    initial_resources = measure_prototype_agent_state_resources(
        initial_agent_state
    )
    phase_boundary_total_nbytes = [initial_resources.total_nbytes]
    semantic_phase_boundaries = [
        _semantic_state_snapshot(
            "initial",
            0,
            carry.agent_state,
            carry.environment_state,
            carry.counterfactual_base_action,
        )
    ]
    phase_traces: list[_CompiledTraceEvent] = []
    identity_audit: dict[str, object] | None = None
    for phase_index, phase in enumerate(_PHASE_NAMES):
        phase_offset = phase_index * protocol.segment_length
        event_indices = jnp.arange(
            phase_offset,
            phase_offset + protocol.segment_length,
            dtype=jnp.int32,
        )
        carry, phase_trace = _compiled_scan_phase(
            agent,
            world,
            horde,
            carry,
            event_indices,
            cue_visible,
            memory_readout_enabled,
        )
        _require_compiled_phase_valid(
            carry,
            phase_trace,
            phase=phase,
            phase_offset=phase_offset,
        )
        phase_traces.append(phase_trace)
        phase_boundary_total_nbytes.append(
            measure_prototype_agent_state_resources(
                carry.agent_state
            ).total_nbytes
        )
        semantic_phase_boundaries.append(
            _semantic_state_snapshot(
                f"after_{phase}",
                phase_offset + protocol.segment_length,
                carry.agent_state,
                carry.environment_state,
                carry.counterfactual_base_action,
            )
        )
        if phase_index == 0 and not bool(
            jax.device_get(carry.stale_fixture_captured)
        ):
            raise RuntimeError("compiled recurrence did not capture its A1 fixture")
        if phase_index == 1:
            replay = _compiled_stale_replay(
                agent,
                carry.agent_state,
                carry.stale_transition,
                carry.stale_memory_input,
            )
            replay_host = cast(_CompiledReplayAudit, jax.device_get(replay))
            identity_audit = {
                "aba_replay_attempted": True,
                "stale_decision_rejected": bool(
                    replay_host.stale_decision_rejected
                ),
                "state_bit_exact": bool(replay_host.state_bit_exact),
                "agent_clock_unchanged": bool(
                    replay_host.agent_clock_unchanged
                ),
                "environment_unchanged": True,
                "replay_update_calls": 1,
            }
            if any(
                identity_audit[name] is not True
                for name in (
                    "stale_decision_rejected",
                    "state_bit_exact",
                    "agent_clock_unchanged",
                    "environment_unchanged",
                )
            ):
                raise RuntimeError("compiled recurrence stale replay was not an exact no-op")

    if identity_audit is None:
        raise RuntimeError("compiled recurrence omitted its A-B-A replay audit")
    trace = _compiled_trace_to_events(
        _concatenate_compiled_traces(phase_traces),
        protocol,
    )
    resources = _resource_payload(
        agent,
        world,
        initial_agent_state,
        carry.agent_state,
        phase_boundary_total_nbytes,
        max(phase_boundary_total_nbytes),
    )
    config = cast(dict[str, object], _json_clone(agent.to_config()))
    run = {
        "arm": arm.name,
        "seed": seed,
        "lifecycle_id": [_LIFECYCLE_TAG, arm_index + 1],
        "agent_config": config,
        "agent_config_sha256": _digest(config),
        "trace": trace,
        "trace_sha256": _digest(trace),
        "metrics": _metrics_from_trace(trace, protocol),
        "semantic_state": _semantic_state_audit(semantic_phase_boundaries),
        "resources": resources,
        "work": _work_from_trace(trace, resources),
        "identity_audit": identity_audit,
    }
    return cast(dict[str, object], _json_clone(run))


def _comparison_contract(runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    initial_bytes = [
        cast(Mapping[str, object], cast(Mapping[str, object], run["resources"])["initial_state"])[
            "total_nbytes"
        ]
        for run in runs
    ]
    work_keys = (
        "requested_transitions",
        "committed_environment_transitions",
        "counterfactual_environment_calls",
        "prototype_update_calls",
        "discarded_preview_update_calls",
        "committed_prototype_update_calls",
        "identity_probe_update_calls",
        "oak_update_calls",
        "oak_discarded_preview_updates",
        "oak_committed_updates",
        "world_model_update_calls",
        "world_model_discarded_preview_updates",
        "world_model_committed_updates",
        "horde_prediction_calls",
        "horde_update_calls",
        "horde_discarded_preview_updates",
        "horde_committed_updates",
        "memory_sidecars_supplied",
        "memory_real_transition_sidecars_supplied",
        "memory_stale_replay_sidecars_supplied",
        "memory_total_prestate_query_calls",
        "prototype_feature_observe_calls",
    )
    work_rows = [
        tuple(cast(Mapping[str, object], run["work"])[key] for key in work_keys)
        for run in runs
    ]
    gate_runs = [
        run
        for run in runs
        if cast(
            bool,
            cast(
                Mapping[str, object],
                cast(Mapping[str, object], run["resources"])[
                    "experiential_memory_advantage_gate"
                ],
            )["configured"],
        )
    ]
    gate_work_rows = [
        _canonical_json(cast(Mapping[str, object], run["work"]))
        for run in gate_runs
    ]
    gate_initial_bytes = [
        cast(
            Mapping[str, object],
            cast(Mapping[str, object], run["resources"])["initial_state"],
        )["total_nbytes"]
        for run in gate_runs
    ]
    return {
        "paired_seed": len({run["seed"] for run in runs}) == 1,
        "arm_order": [run["arm"] for run in runs],
        "persistent_state_shape_matched": len(set(initial_bytes)) == 1,
        "preview_and_transaction_work_matched": len(set(work_rows)) == 1,
        "advantage_gate_additional_work_declared": all(
            "memory_advantage_gate_total_assessments"
            in cast(Mapping[str, object], run["work"])
            for run in runs
        ),
        "conservative_outcome_gate_pair_work_matched": (
            len(set(gate_work_rows)) <= 1
        ),
        "conservative_outcome_gate_pair_persistent_state_shape_matched": (
            len(set(gate_initial_bytes)) <= 1
        ),
        "preview_state_carried": False,
        "realized_compute_or_allocator_parity_claimed": False,
        "rejected_event_short_circuit_work_parity_claimed": False,
    }


def _build_recurrence_report(
    protocol: PrototypeFeatureMemoryRecurrenceProtocol,
    world: RecurringTwoAgentWorld,
    runs: list[dict[str, object]],
    *,
    execution_engine: str,
) -> dict[str, object]:
    """Build and strictly validate the one unchanged v1 report shape."""

    protocol_payload = protocol.to_config()
    environment_payload = world.to_config()
    report: dict[str, object] = {
        "schema_version": PROTOTYPE_FEATURE_MEMORY_RECURRENCE_REPORT_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "acceptance_status": ACCEPTANCE_STATUS,
        "accepted_scientific_evidence": False,
        "interpretation": INTERPRETATION,
        "protocol": protocol_payload,
        "protocol_sha256": _digest(protocol_payload),
        "environment_config": environment_payload,
        "environment_config_sha256": _digest(environment_payload),
        "execution": _execution_contract(execution_engine),
        "arm_definitions": [
            _ARMS_BY_NAME[name].to_config() for name in protocol.arm_names
        ],
        "runs": runs,
        "comparison_contract": _comparison_contract(runs),
        "limitations": list(LIMITATIONS),
    }
    report["report_sha256"] = _digest(report)
    canonical = cast(dict[str, object], _json_clone(report))
    validation = validate_prototype_feature_memory_recurrence_report(canonical)
    if not validation.valid:
        raise RuntimeError(
            "internally built recurrence report failed validation: "
            + "; ".join(validation.errors)
        )
    return canonical


def run_prototype_feature_memory_recurrence_development(
    protocol: PrototypeFeatureMemoryRecurrenceProtocol | None = None,
    *,
    seed: int = 0,
) -> dict[str, object]:
    """Run one bounded paired development life and return a strict report."""

    resolved = PrototypeFeatureMemoryRecurrenceProtocol() if protocol is None else protocol
    if type(resolved) is not PrototypeFeatureMemoryRecurrenceProtocol:
        raise TypeError("protocol must be an exact PrototypeFeatureMemoryRecurrenceProtocol")
    _require_exact_int(seed, name="seed", minimum=0, maximum=_UINT32_MAX)
    world = RecurringTwoAgentWorld(
        context_length=resolved.segment_length,
        nuisance_dim=resolved.nuisance_dim,
        nuisance_scale=resolved.nuisance_scale,
    )
    horde = HordeLearner(_horde_spec(), hidden_sizes=(), step_size=0.05)
    arm_agent_keys = tuple(
        dict.fromkeys(
            (
                _ARMS_BY_NAME[name].feature_promotion_enabled,
                _ARMS_BY_NAME[name].conservative_outcome_gate_enabled,
            )
            for name in resolved.arm_names
        )
    )
    agents = {
        key: PrototypeAgent(
            _agent_config(
                resolved,
                feature_promotion_enabled=key[0],
                conservative_outcome_gate_enabled=key[1],
            )
        )
        for key in arm_agent_keys
    }
    runs = [
        _run_arm(
            resolved,
            _ARMS_BY_NAME[name],
            seed=seed,
            arm_index=_CANONICAL_ARM_NAMES.index(name),
            world=world,
            agent=agents[
                (
                    _ARMS_BY_NAME[name].feature_promotion_enabled,
                    _ARMS_BY_NAME[name].conservative_outcome_gate_enabled,
                )
            ],
            horde=horde,
        )
        for name in resolved.arm_names
    ]
    return _build_recurrence_report(
        resolved,
        world,
        runs,
        execution_engine=_EAGER_EXECUTION_ENGINE,
    )


def run_compiled_prototype_feature_memory_recurrence_development(
    protocol: PrototypeFeatureMemoryRecurrenceProtocol | None = None,
    *,
    seed: int = 0,
) -> dict[str, object]:
    """Run the same v1 life through exact module-stable JIT/scan transactions.

    This execution path changes neither protocol nor report semantics. It has
    no artifact writer and cannot promote evidence. Compiler workspaces and
    transient device trace arrays are deliberately outside the report's
    persistent-state resource accounting, and no realized-compute or latency
    parity is claimed.
    """

    resolved = PrototypeFeatureMemoryRecurrenceProtocol() if protocol is None else protocol
    if type(resolved) is not PrototypeFeatureMemoryRecurrenceProtocol:
        raise TypeError("protocol must be an exact PrototypeFeatureMemoryRecurrenceProtocol")
    _require_exact_int(seed, name="seed", minimum=0, maximum=_UINT32_MAX)
    world = RecurringTwoAgentWorld(
        context_length=resolved.segment_length,
        nuisance_dim=resolved.nuisance_dim,
        nuisance_scale=resolved.nuisance_scale,
    )
    horde = HordeLearner(_horde_spec(), hidden_sizes=(), step_size=0.05)
    arm_agent_keys = tuple(
        dict.fromkeys(
            (
                _ARMS_BY_NAME[name].feature_promotion_enabled,
                _ARMS_BY_NAME[name].conservative_outcome_gate_enabled,
            )
            for name in resolved.arm_names
        )
    )
    agents = {
        key: PrototypeAgent(
            _agent_config(
                resolved,
                feature_promotion_enabled=key[0],
                conservative_outcome_gate_enabled=key[1],
            )
        )
        for key in arm_agent_keys
    }
    runs = [
        _run_arm_compiled(
            resolved,
            _ARMS_BY_NAME[name],
            seed=seed,
            arm_index=_CANONICAL_ARM_NAMES.index(name),
            world=world,
            agent=agents[
                (
                    _ARMS_BY_NAME[name].feature_promotion_enabled,
                    _ARMS_BY_NAME[name].conservative_outcome_gate_enabled,
                )
            ],
            horde=horde,
        )
        for name in resolved.arm_names
    ]
    return _build_recurrence_report(
        resolved,
        world,
        runs,
        execution_engine=_COMPILED_EXECUTION_ENGINE,
    )


_EVENT_FIELDS: Final = {
    "event_index",
    "phase",
    "phase_step",
    "environment_pre_words",
    "environment_post_words",
    "prototype_pre_step_words",
    "prototype_post_step_words",
    "prototype_decision_id",
    "action",
    "counterfactual_base_action",
    "reward",
    "counterfactual_reward",
    "counterfactual_reward_delta",
    "horde_prediction",
    "horde_cumulant",
    "horde_squared_error",
    "world_model_prediction_error",
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
    "memory_advantage_gate_configured",
    "memory_advantage_gate_evidence_valid",
    "memory_advantage_gate_actions_differ",
    "memory_advantage_gate_base_support_count",
    "memory_advantage_gate_proposed_support_count",
    "memory_advantage_gate_base_action_weight_mass",
    "memory_advantage_gate_proposed_action_weight_mass",
    "memory_advantage_gate_weight_mass_ready",
    "memory_advantage_gate_support_ready",
    "memory_advantage_gate_base_reward_mean",
    "memory_advantage_gate_proposed_reward_mean",
    "memory_advantage_gate_reward_advantage",
    "memory_advantage_gate_advantage_ready",
    "memory_advantage_gate_replacement_allowed",
    "memory_advantage_gate_dispatch_consistent",
    "preview_state_discarded",
    "transition_valid",
}


def _validate_event(
    event: Mapping[str, object],
    *,
    index: int,
    protocol: PrototypeFeatureMemoryRecurrenceProtocol,
    lifecycle_id: list[int],
    conservative_outcome_gate_enabled: bool,
) -> None:
    if set(event) != _EVENT_FIELDS:
        raise ValueError(f"trace[{index}] fields are invalid")
    if event["event_index"] != index:
        raise ValueError(f"trace[{index}] event_index is invalid")
    if event["phase"] != _phase_for_step(index, protocol.segment_length):
        raise ValueError(f"trace[{index}] phase is invalid")
    if event["phase_step"] != index % protocol.segment_length:
        raise ValueError(f"trace[{index}] phase_step is invalid")
    for name, expected in (
        ("environment_pre_words", index),
        ("environment_post_words", index + 1),
        ("prototype_pre_step_words", index),
        ("prototype_post_step_words", index + 1),
    ):
        if _words_value(event[name], name=f"trace[{index}].{name}") != expected:
            raise ValueError(f"trace[{index}] {name} is not the exact event identity")
    decision = event["prototype_decision_id"]
    if not isinstance(decision, list) or len(decision) != 4:
        raise ValueError(f"trace[{index}] decision id is invalid")
    if decision[:2] != lifecycle_id:
        raise ValueError(f"trace[{index}] decision lifecycle is invalid")
    if _words_value(decision[2:], name=f"trace[{index}].decision_generation") != index:
        raise ValueError(f"trace[{index}] decision generation is invalid")
    for name in ("action", "counterfactual_base_action"):
        if type(event[name]) is not int or event[name] not in (0, 1):
            raise ValueError(f"trace[{index}] {name} is invalid")
    for name in (
        "reward",
        "counterfactual_reward",
        "counterfactual_reward_delta",
        "world_model_prediction_error",
    ):
        if type(event[name]) is not float or not math.isfinite(cast(float, event[name])):
            raise ValueError(f"trace[{index}] {name} must be finite")
    if cast(float, event["world_model_prediction_error"]) < 0.0:
        raise ValueError(f"trace[{index}] world-model prediction error must be non-negative")
    expected_delta = cast(float, event["reward"]) - cast(float, event["counterfactual_reward"])
    if event["counterfactual_reward_delta"] != expected_delta:
        raise ValueError(f"trace[{index}] counterfactual delta does not reconstruct")
    vectors: dict[str, list[float]] = {}
    for name in ("horde_prediction", "horde_cumulant", "horde_squared_error"):
        value = event[name]
        if (
            not isinstance(value, list)
            or len(value) != _N_HORDE_DEMONS
            or any(type(item) is not float or not math.isfinite(item) for item in value)
        ):
            raise ValueError(f"trace[{index}] {name} is invalid")
        vectors[name] = cast(list[float], value)
    expected_squared = [
        (vectors["horde_cumulant"][demon] - vectors["horde_prediction"][demon]) ** 2
        for demon in range(_N_HORDE_DEMONS)
    ]
    if not np.allclose(vectors["horde_squared_error"], expected_squared, rtol=1e-6, atol=1e-7):
        raise ValueError(f"trace[{index}] Horde squared error does not reconstruct")
    for name in ("feature_generation_pre_words", "feature_generation_post_words"):
        _words_value(event[name], name=f"trace[{index}].{name}")
    for name in (
        "a_critical_pair_count",
        "b_critical_pair_count",
        "memory_rows_reencoded",
        "memory_prestate_query_count",
        "memory_advantage_gate_base_support_count",
        "memory_advantage_gate_proposed_support_count",
    ):
        if type(event[name]) is not int or cast(int, event[name]) < 0:
            raise ValueError(f"trace[{index}] {name} is invalid")
    for name in (
        "curation_committed",
        "feature_memory_rebind_applied",
        "memory_query_before_write",
        "memory_wrote",
        "memory_retrieval_available",
        "memory_action_changed",
        "memory_advantage_gate_configured",
        "memory_advantage_gate_evidence_valid",
        "memory_advantage_gate_actions_differ",
        "memory_advantage_gate_weight_mass_ready",
        "memory_advantage_gate_support_ready",
        "memory_advantage_gate_advantage_ready",
        "memory_advantage_gate_replacement_allowed",
        "memory_advantage_gate_dispatch_consistent",
        "preview_state_discarded",
        "transition_valid",
    ):
        if type(event[name]) is not bool:
            raise ValueError(f"trace[{index}] {name} must be boolean")
    if event["preview_state_discarded"] is not True or event["transition_valid"] is not True:
        raise ValueError(f"trace[{index}] did not commit the required transaction")
    if event["memory_advantage_gate_dispatch_consistent"] is not True:
        raise ValueError(f"trace[{index}] advantage-gate dispatch is inconsistent")
    if (
        event["memory_advantage_gate_configured"]
        is not conservative_outcome_gate_enabled
    ):
        raise ValueError(f"trace[{index}] advantage-gate configuration is invalid")
    gate_float_names = (
        "memory_advantage_gate_base_action_weight_mass",
        "memory_advantage_gate_proposed_action_weight_mass",
        "memory_advantage_gate_base_reward_mean",
        "memory_advantage_gate_proposed_reward_mean",
        "memory_advantage_gate_reward_advantage",
    )
    for name in gate_float_names:
        if type(event[name]) is not float or not math.isfinite(cast(float, event[name])):
            raise ValueError(f"trace[{index}] {name} must be finite")
    for name in (
        "memory_advantage_gate_base_action_weight_mass",
        "memory_advantage_gate_proposed_action_weight_mass",
    ):
        if cast(float, event[name]) < 0.0:
            raise ValueError(f"trace[{index}] {name} must be non-negative")

    if not conservative_outcome_gate_enabled:
        neutral_false = (
            "memory_advantage_gate_evidence_valid",
            "memory_advantage_gate_actions_differ",
            "memory_advantage_gate_weight_mass_ready",
            "memory_advantage_gate_support_ready",
            "memory_advantage_gate_advantage_ready",
            "memory_advantage_gate_replacement_allowed",
        )
        neutral_zero = (
            "memory_advantage_gate_base_support_count",
            "memory_advantage_gate_proposed_support_count",
            *gate_float_names,
        )
        if any(event[name] is not False for name in neutral_false) or any(
            event[name] != 0 for name in neutral_zero
        ):
            raise ValueError(f"trace[{index}] unconfigured advantage gate is not neutral")
        return

    evidence_valid = cast(bool, event["memory_advantage_gate_evidence_valid"])
    actions_differ = cast(bool, event["memory_advantage_gate_actions_differ"])
    base_support = cast(int, event["memory_advantage_gate_base_support_count"])
    proposed_support = cast(
        int, event["memory_advantage_gate_proposed_support_count"]
    )
    base_mass = cast(float, event["memory_advantage_gate_base_action_weight_mass"])
    proposed_mass = cast(
        float, event["memory_advantage_gate_proposed_action_weight_mass"]
    )
    support_floor = ExperientialMemoryAdvantageGateConfig().min_action_support
    mass_floor = float(
        np.float32(ExperientialMemoryAdvantageGateConfig().min_action_weight_mass)
    )
    advantage_floor = float(
        np.float32(ExperientialMemoryAdvantageGateConfig().min_reward_advantage)
    )
    expected_weight_mass_ready = (
        evidence_valid and base_mass >= mass_floor and proposed_mass >= mass_floor
    )
    if event["memory_advantage_gate_weight_mass_ready"] is not expected_weight_mass_ready:
        raise ValueError(f"trace[{index}] advantage-gate mass decision is invalid")
    expected_support_ready = (
        evidence_valid
        and base_support >= support_floor
        and proposed_support >= support_floor
        and expected_weight_mass_ready
    )
    if event["memory_advantage_gate_support_ready"] is not expected_support_ready:
        raise ValueError(f"trace[{index}] advantage-gate support decision is invalid")
    base_mean = cast(float, event["memory_advantage_gate_base_reward_mean"])
    proposed_mean = cast(float, event["memory_advantage_gate_proposed_reward_mean"])
    reward_advantage = cast(float, event["memory_advantage_gate_reward_advantage"])
    if not np.isclose(
        reward_advantage,
        float(np.float32(proposed_mean) - np.float32(base_mean)),
        rtol=1.0e-6,
        atol=1.0e-7,
    ):
        raise ValueError(f"trace[{index}] advantage-gate reward advantage is invalid")
    expected_advantage_ready = (
        expected_support_ready and reward_advantage > advantage_floor
    )
    if event["memory_advantage_gate_advantage_ready"] is not expected_advantage_ready:
        raise ValueError(f"trace[{index}] advantage-gate reward decision is invalid")
    expected_replacement = actions_differ and expected_advantage_ready
    if event["memory_advantage_gate_replacement_allowed"] is not expected_replacement:
        raise ValueError(f"trace[{index}] advantage-gate authority decision is invalid")
    if event["memory_action_changed"] is not expected_replacement:
        raise ValueError(f"trace[{index}] advantage-gate dispatch did not reconstruct")


def _validate_resources(
    resources: Mapping[str, object],
    protocol: PrototypeFeatureMemoryRecurrenceProtocol,
    *,
    conservative_outcome_gate_enabled: bool,
) -> None:
    expected = {
        "initial_state",
        "final_state",
        "phase_boundary_total_nbytes",
        "peak_total_nbytes",
        "environment",
        "feature_lifecycle",
        "feature_memory",
        "experiential_memory",
        "experiential_memory_advantage_gate",
        "stable_base_world_model",
    }
    if set(resources) != expected:
        raise ValueError("resource fields are invalid")
    initial = cast(Mapping[str, object], resources["initial_state"])
    final = cast(Mapping[str, object], resources["final_state"])
    if set(initial) != set(final) or initial != final:
        raise ValueError("initial and final Prototype state resources must match exactly")
    total = initial.get("total_nbytes")
    if type(total) is not int or total <= 0:
        raise ValueError("Prototype total_nbytes is invalid")
    boundaries = resources["phase_boundary_total_nbytes"]
    if boundaries != [total] * 4:
        raise ValueError("phase-boundary state bytes must remain exact and constant")
    if resources["peak_total_nbytes"] != total:
        raise ValueError("peak logical state bytes must equal the fixed allocation")
    feature = cast(Mapping[str, object], resources["feature_lifecycle"])
    if (
        feature.get("base_feature_slots") != protocol.base_observation_dim
        or feature.get("active_pair_slots") != protocol.active_pair_slots
        or feature.get("candidate_pair_slots") != protocol.candidate_pair_slots
        or feature.get("max_observations") != protocol.total_steps
    ):
        raise ValueError("feature-lifecycle resource declaration does not match protocol")
    feature_memory = cast(Mapping[str, object], resources["feature_memory"])
    memory = cast(Mapping[str, object], resources["experiential_memory"])
    if feature_memory.get("capacity_entries") != protocol.memory_capacity:
        raise ValueError("feature-memory capacity does not match protocol")
    if feature_memory.get("memory_state_nbytes") != memory.get("persistent_state_bytes"):
        raise ValueError("memory persistent byte declarations disagree")
    if initial.get("interaction_memory_bundle_nbytes") != feature_memory.get(
        "wrapper_state_nbytes"
    ):
        raise ValueError("whole-state memory ownership does not match adapter bytes")
    advantage_gate = resources["experiential_memory_advantage_gate"]
    if not isinstance(advantage_gate, Mapping) or set(advantage_gate) != {
        "configured",
        "config",
        "resources",
    }:
        raise ValueError("advantage-gate resource fields are invalid")
    if advantage_gate["configured"] is not conservative_outcome_gate_enabled:
        raise ValueError("advantage-gate resource configuration is invalid")
    expected_gate_config = (
        ExperientialMemoryAdvantageGateConfig().to_config()
        if conservative_outcome_gate_enabled
        else None
    )
    expected_top_k = min(4, protocol.memory_capacity)
    expected_gate_resources = (
        {
            "n_actions": _N_ACTIONS,
            "top_k": expected_top_k,
            "neighbor_action_values_interpreted": expected_top_k * _N_ACTIONS,
            "neighbor_reward_values_interpreted": expected_top_k,
            "neighbor_weight_values_interpreted": expected_top_k,
            "owned_persistent_state_bytes": 0,
            "random_draws_per_assessment": 0,
        }
        if conservative_outcome_gate_enabled
        else None
    )
    if not _exact_json_equal(advantage_gate["config"], expected_gate_config):
        raise ValueError("advantage-gate default configuration changed")
    if not _exact_json_equal(
        advantage_gate["resources"], expected_gate_resources
    ):
        raise ValueError("advantage-gate logical resources are invalid")
    world_model = cast(Mapping[str, object], resources["stable_base_world_model"])
    if set(world_model) != {
        "coordinates",
        "generated_pair_tail_modeled",
        "observation_dim",
        "buffer_capacity",
        "world_model_bundle_nbytes",
        "buffer_nbytes",
    }:
        raise ValueError("stable-base world-model resource fields are invalid")
    if (
        world_model["coordinates"] != "stable_base_only"
        or world_model["generated_pair_tail_modeled"] is not False
        or world_model["observation_dim"] != protocol.base_observation_dim
        or world_model["buffer_capacity"] != _WORLD_MODEL_BUFFER_CAPACITY
        or world_model["world_model_bundle_nbytes"]
        != initial.get("world_model_bundle_nbytes")
        or world_model["buffer_nbytes"] != initial.get("buffer_nbytes")
        or type(world_model["world_model_bundle_nbytes"]) is not int
        or world_model["world_model_bundle_nbytes"] <= 0
        or world_model["buffer_nbytes"]
        != 4 * _WORLD_MODEL_BUFFER_CAPACITY * protocol.base_observation_dim + 8
    ):
        raise ValueError("stable-base world-model resource declaration is invalid")
    environment = cast(Mapping[str, object], resources["environment"])
    if type(environment.get("state_nbytes")) is not int or cast(
        int, environment["state_nbytes"]
    ) <= 0:
        raise ValueError("environment resource declaration is invalid")


def _validate_identity_audit(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("identity_audit must be a mapping")
    expected = {
        "aba_replay_attempted",
        "stale_decision_rejected",
        "state_bit_exact",
        "agent_clock_unchanged",
        "environment_unchanged",
        "replay_update_calls",
    }
    if set(value) != expected:
        raise ValueError("identity_audit fields are invalid")
    for name in expected - {"replay_update_calls"}:
        if value[name] is not True:
            raise ValueError(f"identity_audit.{name} must be true")
    if value["replay_update_calls"] != 1:
        raise ValueError("identity replay work must equal one update")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_semantic_state_audit(
    value: object,
    protocol: PrototypeFeatureMemoryRecurrenceProtocol,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "normalization",
        "phase_boundaries",
        "final_joint_state_sha256",
    }:
        raise ValueError("semantic_state fields are invalid")
    if value["schema_version"] != _SEMANTIC_STATE_DIGEST_SCHEMA:
        raise ValueError("semantic_state schema is unsupported")
    if value["normalization"] != _SEMANTIC_STATE_NORMALIZATION:
        raise ValueError("semantic_state normalization contract is invalid")
    boundaries = value["phase_boundaries"]
    if not isinstance(boundaries, list) or len(boundaries) != 4:
        raise ValueError("semantic_state must contain four phase boundaries")
    expected_labels = ["initial", *(f"after_{phase}" for phase in _PHASE_NAMES)]
    expected_counts = [0, *(index * protocol.segment_length for index in range(1, 4))]
    for index, (boundary, label, event_count) in enumerate(
        zip(boundaries, expected_labels, expected_counts, strict=True)
    ):
        if not isinstance(boundary, Mapping) or set(boundary) != {
            "label",
            "event_count",
            "agent_state_sha256",
            "environment_state_sha256",
            "counterfactual_base_action",
            "joint_state_sha256",
        }:
            raise ValueError(f"semantic_state.phase_boundaries[{index}] fields are invalid")
        if boundary["label"] != label or boundary["event_count"] != event_count:
            raise ValueError(f"semantic_state.phase_boundaries[{index}] identity is invalid")
        if type(boundary["counterfactual_base_action"]) is not int or boundary[
            "counterfactual_base_action"
        ] not in (0, 1):
            raise ValueError(
                f"semantic_state.phase_boundaries[{index}] base action is invalid"
            )
        for name in (
            "agent_state_sha256",
            "environment_state_sha256",
            "joint_state_sha256",
        ):
            if not _is_sha256(boundary[name]):
                raise ValueError(
                    f"semantic_state.phase_boundaries[{index}].{name} is invalid"
                )
        components = {
            "agent_state_sha256": boundary["agent_state_sha256"],
            "environment_state_sha256": boundary["environment_state_sha256"],
            "counterfactual_base_action": boundary["counterfactual_base_action"],
        }
        if boundary["joint_state_sha256"] != _digest(components):
            raise ValueError(
                f"semantic_state.phase_boundaries[{index}] joint digest is invalid"
            )
    if value["final_joint_state_sha256"] != cast(
        Mapping[str, object], boundaries[-1]
    )["joint_state_sha256"]:
        raise ValueError("semantic_state final digest is invalid")


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
        "execution",
        "arm_definitions",
        "runs",
        "comparison_contract",
        "limitations",
        "report_sha256",
    }
    if set(report) != expected_fields:
        raise ValueError("report fields do not match the v1 schema")
    if report["schema_version"] != PROTOTYPE_FEATURE_MEMORY_RECURRENCE_REPORT_SCHEMA:
        raise ValueError("report schema_version is unsupported")
    if report["development_only"] is not True:
        raise ValueError("report must remain development-only")
    if report["scientific_promotion_allowed"] is not False:
        raise ValueError("report cannot allow scientific promotion")
    if report["acceptance_status"] != ACCEPTANCE_STATUS:
        raise ValueError("report acceptance_status must remain not-assessed")
    if report["accepted_scientific_evidence"] is not False:
        raise ValueError("development report cannot claim accepted evidence")
    if report["interpretation"] != INTERPRETATION:
        raise ValueError("report interpretation changed")
    if report["limitations"] != list(LIMITATIONS):
        raise ValueError("report limitations changed")
    raw_protocol = report["protocol"]
    if not isinstance(raw_protocol, Mapping):
        raise ValueError("report protocol must be a mapping")
    protocol = PrototypeFeatureMemoryRecurrenceProtocol.from_config(raw_protocol)
    protocol_payload = protocol.to_config()
    if report["protocol_sha256"] != _digest(protocol_payload):
        raise ValueError("report protocol digest is invalid")
    environment_config = report["environment_config"]
    if not isinstance(environment_config, Mapping):
        raise ValueError("environment_config must be a mapping")
    expected_environment = RecurringTwoAgentWorld(
        context_length=protocol.segment_length,
        nuisance_dim=protocol.nuisance_dim,
        nuisance_scale=protocol.nuisance_scale,
    ).to_config()
    if dict(environment_config) != expected_environment:
        raise ValueError("environment_config does not reconstruct from protocol")
    if report["environment_config_sha256"] != _digest(expected_environment):
        raise ValueError("environment config digest is invalid")
    execution = report["execution"]
    if not isinstance(execution, Mapping):
        raise ValueError("execution must be a mapping")
    engine = execution.get("engine")
    if type(engine) is not str or engine not in _EXECUTION_ENGINES:
        raise ValueError("execution engine is unsupported")
    expected_execution = _execution_contract(engine)
    if not _exact_json_equal(dict(execution), expected_execution):
        raise ValueError("execution contract does not match the live runner source")
    expected_arm_definitions = [
        _ARMS_BY_NAME[name].to_config() for name in protocol.arm_names
    ]
    if report["arm_definitions"] != expected_arm_definitions:
        raise ValueError("arm definitions are invalid")
    raw_runs = report["runs"]
    if not isinstance(raw_runs, list) or len(raw_runs) != len(protocol.arm_names):
        raise ValueError("report runs do not match protocol arms")
    reconstructed_runs: list[dict[str, object]] = []
    run_fields = {
        "arm",
        "seed",
        "lifecycle_id",
        "agent_config",
        "agent_config_sha256",
        "trace",
        "trace_sha256",
        "metrics",
        "semantic_state",
        "resources",
        "work",
        "identity_audit",
    }
    for run_index, (arm_name, raw_run) in enumerate(
        zip(protocol.arm_names, raw_runs, strict=True)
    ):
        if not isinstance(raw_run, Mapping) or set(raw_run) != run_fields:
            raise ValueError(f"runs[{run_index}] fields are invalid")
        if raw_run["arm"] != arm_name:
            raise ValueError(f"runs[{run_index}] arm order is invalid")
        if (
            type(raw_run["seed"]) is not int
            or not 0 <= raw_run["seed"] <= _UINT32_MAX
        ):
            raise ValueError(f"runs[{run_index}] seed is invalid")
        lifecycle_id = raw_run["lifecycle_id"]
        expected_lifecycle = [_LIFECYCLE_TAG, _CANONICAL_ARM_NAMES.index(arm_name) + 1]
        if lifecycle_id != expected_lifecycle:
            raise ValueError(f"runs[{run_index}] lifecycle_id is invalid")
        agent_config = raw_run["agent_config"]
        if not isinstance(agent_config, Mapping):
            raise ValueError(f"runs[{run_index}] agent_config must be a mapping")
        expected_agent_config = _agent_config(
            protocol,
            feature_promotion_enabled=_ARMS_BY_NAME[arm_name].feature_promotion_enabled,
            conservative_outcome_gate_enabled=(
                _ARMS_BY_NAME[arm_name].conservative_outcome_gate_enabled
            ),
        ).to_config()
        if dict(agent_config) != expected_agent_config:
            raise ValueError(f"runs[{run_index}] agent_config does not reconstruct")
        if raw_run["agent_config_sha256"] != _digest(expected_agent_config):
            raise ValueError(f"runs[{run_index}] agent config digest is invalid")
        trace = raw_run["trace"]
        if not isinstance(trace, list) or len(trace) != protocol.total_steps:
            raise ValueError(f"runs[{run_index}] trace length is invalid")
        for event_index, event in enumerate(trace):
            if not isinstance(event, Mapping):
                raise ValueError(f"runs[{run_index}].trace[{event_index}] is invalid")
            _validate_event(
                event,
                index=event_index,
                protocol=protocol,
                lifecycle_id=cast(list[int], lifecycle_id),
                conservative_outcome_gate_enabled=(
                    _ARMS_BY_NAME[arm_name].conservative_outcome_gate_enabled
                ),
            )
        if raw_run["trace_sha256"] != _digest(trace):
            raise ValueError(f"runs[{run_index}] trace digest is invalid")
        if not _ARMS_BY_NAME[arm_name].feature_promotion_enabled and any(
            bool(cast(Mapping[str, object], event)["curation_committed"])
            or cast(Mapping[str, object], event)["feature_generation_pre_words"]
            != [0, 0]
            or cast(Mapping[str, object], event)["feature_generation_post_words"]
            != [0, 0]
            for event in trace
        ):
            raise ValueError(f"runs[{run_index}] blocked feature arm changed generation")
        metrics = _metrics_from_trace(cast(list[Mapping[str, object]], trace), protocol)
        if raw_run["metrics"] != metrics:
            raise ValueError(f"runs[{run_index}] metrics do not reconstruct")
        _validate_semantic_state_audit(raw_run["semantic_state"], protocol)
        resources = raw_run["resources"]
        if not isinstance(resources, Mapping):
            raise ValueError(f"runs[{run_index}] resources must be a mapping")
        _validate_resources(
            resources,
            protocol,
            conservative_outcome_gate_enabled=(
                _ARMS_BY_NAME[arm_name].conservative_outcome_gate_enabled
            ),
        )
        work = _work_from_trace(cast(list[Mapping[str, object]], trace), resources)
        if raw_run["work"] != work:
            raise ValueError(f"runs[{run_index}] work does not reconstruct")
        _validate_identity_audit(raw_run["identity_audit"])
        reconstructed_runs.append(cast(dict[str, object], _json_clone(raw_run)))

    comparison = _comparison_contract(reconstructed_runs)
    if report["comparison_contract"] != comparison:
        raise ValueError("comparison contract does not reconstruct")
    if len({tuple(cast(list[int], run["lifecycle_id"])) for run in reconstructed_runs}) != len(
        reconstructed_runs
    ):
        raise ValueError("run lifecycle identities are not unique")
    without_digest = {
        key: value for key, value in report.items() if key != "report_sha256"
    }
    expected_report_digest = _digest(without_digest)
    if report["report_sha256"] != expected_report_digest:
        raise ValueError("report digest is invalid")
    return cast(dict[str, object], _json_clone(report))


def validate_prototype_feature_memory_recurrence_report(
    report: Mapping[str, object],
) -> RecurrenceReportValidation:
    """Strictly reconstruct one development report without running a life."""

    try:
        _reconstruct_report(report)
    except (KeyError, TypeError, ValueError) as error:
        return RecurrenceReportValidation(False, (str(error),))
    return RecurrenceReportValidation(True, ())


def prototype_feature_memory_recurrence_report_json(
    report: Mapping[str, object],
) -> str:
    """Return canonical JSON only for a strictly valid development report."""

    validation = validate_prototype_feature_memory_recurrence_report(report)
    if not validation.valid:
        raise ValueError("invalid recurrence report: " + "; ".join(validation.errors))
    return _canonical_json(report)


__all__ = [
    "ACCEPTANCE_STATUS",
    "ACCEPTED_SCIENTIFIC_EVIDENCE",
    "DEVELOPMENT_ONLY",
    "INTERPRETATION",
    "LIMITATIONS",
    "PROTOTYPE_FEATURE_MEMORY_RECURRENCE_PROTOCOL_SCHEMA",
    "PROTOTYPE_FEATURE_MEMORY_RECURRENCE_REPORT_SCHEMA",
    "RECURRENCE_ARMS",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "PrototypeFeatureMemoryRecurrenceProtocol",
    "RecurrenceArm",
    "RecurrenceReportValidation",
    "prototype_feature_memory_recurrence_report_json",
    "run_compiled_prototype_feature_memory_recurrence_development",
    "run_prototype_feature_memory_recurrence_development",
    "validate_prototype_feature_memory_recurrence_report",
]
