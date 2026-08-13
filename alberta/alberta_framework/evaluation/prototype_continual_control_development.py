"""Bounded WP1 report for PrototypeAgent and two ordinary baselines.

Each fixed development seed runs one uninterrupted
:class:`ContinualControlEvaluator` stream containing ``PrototypeAgent``, an
observation-agnostic running reward mean, and a fixed-action learner.  The
evaluator gives each condition an independent functional environment state;
only the pre-generated exogenous observation/reward-offset schedule is paired.
Learners receive numeric observations and owned transitions, never evaluator
regime identifiers or reset instructions.

This lane is deliberately narrow.  It records exact action/decision ownership,
the evaluator's reconstructing performance report, logical state bytes,
operation opportunities, deterministic logical latency, and the diagnostics
that the fixed linear Prototype configuration can measure directly.  Every
other WP1 diagnostic is marked either ``inapplicable`` or ``unavailable``;
missing measurements are never encoded as zero.  A fresh-per-regime reference
is not executed because it requires evaluator privilege and belongs to
``PrivilegedContinualControlReferenceSuite``.

Reports are in-memory L0 development records, always ``not_assessed``.  There
are no acceptance thresholds, winner selection, output writer, efficacy
claim, evidence claim, or promotion authority.  SHA-256 values provide
integrity and source/runtime binding, not authenticity.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.oak import OaKAgent, OaKConfig, OaKState
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    measure_prototype_agent_state_resources,
)
from alberta_framework.evaluation.continual_control_evaluator import (
    CONTROL_CHECKPOINT_SCHEMA,
    ContinualControlEvaluator,
    ContinualControlRunState,
    ContinuingControlBudget,
    ContinuingControlLearner,
    ContinuingControlProtocol,
    ControlDecision,
    ControlEnvironmentUpdate,
    ControlProbe,
    ControlTransition,
    FrozenActionControlBaseline,
    PrototypeAgentControlAdapter,
    RunningRewardBanditControlBaseline,
    validate_continual_control_report,
)

PROTOTYPE_CONTROL_DEVELOPMENT_CONFIG_SCHEMA = (
    "alberta.prototype-continual-control-development.config.v1"
)
PROTOTYPE_CONTROL_DEVELOPMENT_ENVIRONMENT_SCHEMA = (
    "alberta.prototype-continual-control-development.environment.v1"
)
PROTOTYPE_CONTROL_DEVELOPMENT_REPORT_SCHEMA = (
    "alberta.prototype-continual-control-development.report.v1"
)
PROTOTYPE_CONTROL_DEVELOPMENT_PROTOCOL_ID = (
    "alberta.prototype-continual-control-development.protocol.v1"
)

ASSESSMENT_STATUS = "not_assessed"
DEVELOPMENT_SEEDS: tuple[int, int] = (1701, 1702)
REGIME_SCHEDULE: tuple[str, ...] = ("A", "A", "B", "B", "A", "A")
CHECKPOINT_STEPS: tuple[int, ...] = (2, 4, 6)
REGIME_IDS: tuple[str, ...] = ("A", "B")
HORIZON = len(REGIME_SCHEDULE)
CHECKPOINT_SPLIT = 3
LOGICAL_CLOCK_INCREMENT_NS = 1_000
LATENCY_DEADLINE_MS = 0.0005
LATENCY_MEASUREMENT_METHOD = (
    "deterministic logical test clock: exactly 1000 ns per measured call; "
    "not wall-clock latency"
)
OUTPUT_WRITES = False
PROMOTION_AUTHORITY = False
SCIENTIFIC_PROMOTION_ALLOWED = False

_MAX_REPORT_BYTES = 32 * 1024 * 1024
_UINT32_MAX = 2**32 - 1
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATHS = (
    Path("alberta_framework/core/continual_backprop.py"),
    Path("alberta_framework/core/multi_head_learner.py"),
    Path("alberta_framework/core/normalizers.py"),
    Path("alberta_framework/core/oak.py"),
    Path("alberta_framework/core/optimizers.py"),
    Path("alberta_framework/core/options.py"),
    Path("alberta_framework/core/prototype_agent.py"),
    Path("alberta_framework/core/types.py"),
    Path("alberta_framework/evaluation/continual_control_evaluator.py"),
    Path("alberta_framework/evaluation/prototype_continual_control_development.py"),
    Path("pyproject.toml"),
)

_LIMITATIONS = (
    "L0 development mechanism with two consumed seeds; no promotion inference is permitted",
    "logical deterministic-call latency is measured; wall-clock latency is unavailable",
    "shared ceilings and identical opportunities do not establish equal realized compute",
    "logical JAX payload bytes do not measure allocator residency or transient buffers",
    "energy measurement is unavailable",
    "the held-out policy churn diagnostic is deterministic greedy primitive-action churn, not "
    "the complete option-aware behavior distribution",
    "gradient norm and sampled NTK rank are unavailable because the adapter does not expose "
    "their causal update-time values",
    "the fixed linear configuration has no hidden activation diagnostics or dynamic component "
    "creation/removal survival curves",
    "standalone and ensemble world models are disabled, so their calibration metrics do not apply",
    "fresh-per-regime evaluation requires the separate privileged reference suite and is not "
    "executed by this ordinary-observation lane",
    "the declared source manifest binds the executed Python surfaces but is not a compiled "
    "dependency or hardware attestation",
    "synthetic safety costs are evaluator measurements, not physical-safety evidence",
)

type DiagnosticStatus = Literal["available", "unavailable", "inapplicable"]
type DecisionId = tuple[int, int, int, int]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_clone(value: object) -> object:
    return json.loads(_canonical_json_bytes(value))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json_equal(actual: object, expected: object) -> bool:
    if expected is None:
        return actual is None
    if type(expected) in {bool, int, float, str}:
        return type(actual) is type(expected) and actual == expected
    if type(expected) is list:
        return (
            type(actual) is list
            and len(cast(list[object], actual)) == len(cast(list[object], expected))
            and all(
                _strict_json_equal(left, right)
                for left, right in zip(
                    cast(list[object], actual),
                    cast(list[object], expected),
                    strict=True,
                )
            )
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_json_equal(actual[key], expected[key]) for key in expected)
        )
    return False


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be an object with exact string keys")
    return cast(Mapping[str, object], value)


def _list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    return cast(list[object], value)


def _exact_keys(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields differ")


def _exact_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        raise ValueError(f"{name} must be an exact bounded integer")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _observation(value: object, *, name: str) -> tuple[float, float]:
    if type(value) not in {list, tuple} or len(cast(Sequence[object], value)) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    raw = cast(Sequence[object], value)
    return (
        _finite_float(raw[0], name=f"{name}[0]"),
        _finite_float(raw[1], name=f"{name}[1]"),
    )


def _decision_id(value: object, *, name: str) -> DecisionId:
    if type(value) not in {list, tuple} or len(cast(Sequence[object], value)) != 4:
        raise ValueError(f"{name} must contain four uint32 words")
    words = tuple(
        _exact_int(item, name=f"{name}[{index}]", maximum=_UINT32_MAX)
        for index, item in enumerate(cast(Sequence[object], value))
    )
    return cast(DecisionId, words)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prototype_continual_control_source_manifest(
    root: Path = _REPO_ROOT,
) -> dict[str, str]:
    """Hash the declared executed-source surface of this bounded lane."""

    return {
        path.as_posix(): _sha256_file(root / path)
        for path in _SOURCE_PATHS
    }


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def prototype_continual_control_runtime_identity() -> dict[str, object]:
    """Return observable non-secret runtime and JAX execution identity."""

    devices = tuple(jax.devices())
    return {
        "identity_scope": (
            "observable Python/JAX/device/config identity; exact source-bound causal replay "
            "is authoritative"
        ),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "jax_version": str(jax.__version__),
        "jaxlib_version": _package_version("jaxlib"),
        "numpy_version": str(np.__version__),
        "chex_version": _package_version("chex"),
        "default_backend": jax.default_backend(),
        "device_count": len(devices),
        "local_device_count": int(jax.local_device_count()),
        "device_platforms": [str(device.platform) for device in devices],
        "device_kinds": [str(device.device_kind) for device in devices],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "jax_disable_jit": bool(jax.config.jax_disable_jit),
    }


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeContinualControlDevelopmentConfig:
    """Frozen, finite development protocol; no field is an acceptance gate."""

    development_seeds: tuple[int, int] = DEVELOPMENT_SEEDS
    regime_schedule: tuple[str, ...] = REGIME_SCHEDULE
    checkpoint_steps: tuple[int, ...] = CHECKPOINT_STEPS
    horizon: int = HORIZON
    checkpoint_split: int = CHECKPOINT_SPLIT
    observation_dim: int = 2
    action_count: int = 2
    logical_clock_increment_ns: int = LOGICAL_CLOCK_INCREMENT_NS
    latency_deadline_ms: float = LATENCY_DEADLINE_MS
    schema_version: str = PROTOTYPE_CONTROL_DEVELOPMENT_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "development_seeds": DEVELOPMENT_SEEDS,
            "regime_schedule": REGIME_SCHEDULE,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "horizon": HORIZON,
            "checkpoint_split": CHECKPOINT_SPLIT,
            "observation_dim": 2,
            "action_count": 2,
            "logical_clock_increment_ns": LOGICAL_CLOCK_INCREMENT_NS,
            "latency_deadline_ms": LATENCY_DEADLINE_MS,
            "schema_version": PROTOTYPE_CONTROL_DEVELOPMENT_CONFIG_SCHEMA,
        }
        for name, expected_value in expected.items():
            value = getattr(self, name)
            if type(value) is not type(expected_value) or value != expected_value:
                raise ValueError(f"{name} is frozen at {expected_value!r}")

    def to_config(self) -> dict[str, object]:
        return cast(dict[str, object], _json_clone(dataclasses.asdict(self)))

    @classmethod
    def from_config(
        cls,
        payload: object,
    ) -> PrototypeContinualControlDevelopmentConfig:
        raw = _mapping(payload, name="config")
        expected = {field.name for field in dataclasses.fields(cls)}
        _exact_keys(raw, expected, name="config")
        converted = dict(raw)
        for name in ("development_seeds", "regime_schedule", "checkpoint_steps"):
            converted[name] = tuple(_list(converted[name], name=f"config.{name}"))
        return cls(**cast(dict[str, Any], converted))


@dataclasses.dataclass(frozen=True, slots=True)
class _TransitionRecord:
    step: int
    evaluator_regime_id: str
    observation: tuple[float, float]
    action: int
    decision_id: DecisionId
    armed: bool
    reward: float
    discount: float
    bootstrap_observation: tuple[float, float]
    reward_offset: float
    exogenous_distractor: float
    next_exogenous_distractor: float
    near_miss: bool
    safety_cost: float

    def to_config(self) -> dict[str, object]:
        return cast(dict[str, object], _json_clone(dataclasses.asdict(self)))


@dataclasses.dataclass(frozen=True, slots=True)
class _EnvironmentState:
    step: int
    observation: tuple[float, float]
    records: tuple[_TransitionRecord, ...]


class _PairedCueEnvironment:
    """Functional numeric-observation environment with evaluator-only regimes."""

    def __init__(self, *, seed: int) -> None:
        self._seed = _exact_int(seed, name="environment seed", maximum=_UINT32_MAX)
        key = jr.key(self._seed)
        distractor_key, reward_key = jr.split(key)
        distractors = jr.uniform(
            distractor_key,
            (HORIZON + 1,),
            minval=-0.25,
            maxval=0.25,
            dtype=jnp.float32,
        )
        reward_offsets = jr.uniform(
            reward_key,
            (HORIZON,),
            minval=-0.025,
            maxval=0.025,
            dtype=jnp.float32,
        )
        self._distractors = tuple(float(value) for value in np.asarray(distractors))
        self._reward_offsets = tuple(float(value) for value in np.asarray(reward_offsets))
        cues = tuple(0.0 if regime == "A" else 1.0 for regime in REGIME_SCHEDULE)
        self._observations = tuple(
            (cue, self._distractors[index])
            for index, cue in enumerate(cues)
        ) + ((cues[-1], self._distractors[-1]),)

    @property
    def n_actions(self) -> int:
        return 2

    def to_config(self) -> dict[str, object]:
        return {
            "type": "PairedCueEnvironment",
            "schema_version": PROTOTYPE_CONTROL_DEVELOPMENT_ENVIRONMENT_SCHEMA,
            "seed": self._seed,
            "regime_schedule_sha256": _digest(list(REGIME_SCHEDULE)),
            "observations": [list(value) for value in self._observations],
            "reward_offsets": list(self._reward_offsets),
            "action_score_table": {"A": [1.0, 0.0], "B": [0.0, 1.0]},
            "discount": 0.9,
            "boundary_mode": "uninterrupted_continuing_no_reset",
            "learner_visible_regime_metadata": False,
            "exogenous_randomness_paired_only": True,
        }

    def init(self) -> _EnvironmentState:
        return _EnvironmentState(0, self._observations[0], ())

    def observation(self, state: Any) -> tuple[float, ...]:
        resolved = cast(_EnvironmentState, state)
        if type(resolved) is not _EnvironmentState:
            raise TypeError("environment state type is invalid")
        return resolved.observation

    def step(
        self,
        state: Any,
        decision: ControlDecision,
        evaluator_regime_id: str,
    ) -> ControlEnvironmentUpdate:
        resolved = cast(_EnvironmentState, state)
        if type(resolved) is not _EnvironmentState:
            raise TypeError("environment state type is invalid")
        if not 0 <= resolved.step < HORIZON:
            raise ValueError("environment horizon is exhausted")
        expected_regime = REGIME_SCHEDULE[resolved.step]
        if evaluator_regime_id != expected_regime:
            raise ValueError("evaluator regime does not match frozen environment schedule")
        if decision.observation != resolved.observation or not decision.armed:
            raise ValueError("environment decision does not own the current observation")
        correct_action = 0 if expected_regime == "A" else 1
        offset = self._reward_offsets[resolved.step]
        reward = float((1.0 if decision.action == correct_action else 0.0) + offset)
        next_observation = self._observations[resolved.step + 1]
        near_miss = decision.action != correct_action
        safety_cost = 0.1 if near_miss else 0.0
        transition = ControlTransition(
            observation=resolved.observation,
            action=decision.action,
            decision_id=decision.decision_id,
            reward=reward,
            discount=0.9,
            terminated=False,
            truncated=False,
            bootstrap_observation=next_observation,
            reset_observation=None,
            safety_violation=False,
            intervention=False,
            near_miss=near_miss,
            safety_cost=safety_cost,
            near_miss_cost=safety_cost,
        )
        record = _TransitionRecord(
            step=resolved.step,
            evaluator_regime_id=expected_regime,
            observation=resolved.observation,
            action=decision.action,
            decision_id=decision.decision_id,
            armed=decision.armed,
            reward=reward,
            discount=0.9,
            bootstrap_observation=next_observation,
            reward_offset=offset,
            exogenous_distractor=resolved.observation[1],
            next_exogenous_distractor=next_observation[1],
            near_miss=near_miss,
            safety_cost=safety_cost,
        )
        return ControlEnvironmentUpdate(
            state=_EnvironmentState(
                resolved.step + 1,
                next_observation,
                (*resolved.records, record),
            ),
            transition=transition,
        )

    def state_to_config(self, state: Any) -> object:
        resolved = cast(_EnvironmentState, state)
        if type(resolved) is not _EnvironmentState:
            raise TypeError("environment state type is invalid")
        return {
            "step": resolved.step,
            "observation": list(resolved.observation),
            "records": [record.to_config() for record in resolved.records],
        }

    def state_from_config(self, payload: object) -> _EnvironmentState:
        raw = _mapping(payload, name="environment state")
        _exact_keys(raw, {"step", "observation", "records"}, name="environment state")
        step = _exact_int(raw["step"], name="environment state.step", maximum=HORIZON)
        records: list[_TransitionRecord] = []
        for index, value in enumerate(_list(raw["records"], name="environment state.records")):
            location = f"environment state.records[{index}]"
            raw_record = _mapping(value, name=location)
            expected = {field.name for field in dataclasses.fields(_TransitionRecord)}
            _exact_keys(raw_record, expected, name=location)
            if type(raw_record["evaluator_regime_id"]) is not str:
                raise ValueError(f"{location}.evaluator_regime_id must be an exact string")
            if (
                type(raw_record["armed"]) is not bool
                or type(raw_record["near_miss"]) is not bool
            ):
                raise ValueError(f"{location} boolean fields are invalid")
            records.append(
                _TransitionRecord(
                    step=_exact_int(raw_record["step"], name=f"{location}.step"),
                    evaluator_regime_id=raw_record["evaluator_regime_id"],
                    observation=_observation(
                        raw_record["observation"], name=f"{location}.observation"
                    ),
                    action=_exact_int(
                        raw_record["action"], name=f"{location}.action", maximum=1
                    ),
                    decision_id=_decision_id(
                        raw_record["decision_id"], name=f"{location}.decision_id"
                    ),
                    armed=raw_record["armed"],
                    reward=_finite_float(
                        raw_record["reward"], name=f"{location}.reward"
                    ),
                    discount=_finite_float(
                        raw_record["discount"], name=f"{location}.discount"
                    ),
                    bootstrap_observation=_observation(
                        raw_record["bootstrap_observation"],
                        name=f"{location}.bootstrap_observation",
                    ),
                    reward_offset=_finite_float(
                        raw_record["reward_offset"], name=f"{location}.reward_offset"
                    ),
                    exogenous_distractor=_finite_float(
                        raw_record["exogenous_distractor"],
                        name=f"{location}.exogenous_distractor",
                    ),
                    next_exogenous_distractor=_finite_float(
                        raw_record["next_exogenous_distractor"],
                        name=f"{location}.next_exogenous_distractor",
                    ),
                    near_miss=raw_record["near_miss"],
                    safety_cost=_finite_float(
                        raw_record["safety_cost"], name=f"{location}.safety_cost"
                    ),
                )
            )
        state = _EnvironmentState(
            step=step,
            observation=_observation(raw["observation"], name="environment state.observation"),
            records=tuple(records),
        )
        if len(records) != step or state.observation != self._observations[step]:
            raise ValueError("environment state prefix is inconsistent")
        expected_prefix = self.init()
        for index, parsed_record in enumerate(records):
            decision = ControlDecision(
                observation=parsed_record.observation,
                action=parsed_record.action,
                decision_id=parsed_record.decision_id,
                armed=parsed_record.armed,
            )
            update = self.step(expected_prefix, decision, REGIME_SCHEDULE[index])
            expected_record = cast(_EnvironmentState, update.state).records[-1]
            if expected_record != parsed_record:
                raise ValueError("environment record does not reconstruct")
            expected_prefix = cast(_EnvironmentState, update.state)
        if expected_prefix != state:
            raise ValueError("environment state is not canonical")
        return state


class _DeterministicClock:
    def __init__(self) -> None:
        self._now = 0

    def __call__(self) -> int:
        value = self._now
        self._now += LOGICAL_CLOCK_INCREMENT_NS
        return value


@dataclasses.dataclass(frozen=True)
class _SeedHarness:
    seed: int
    environment_seed: int
    agent: PrototypeAgent
    environment: _PairedCueEnvironment
    candidate: PrototypeAgentControlAdapter
    flat: RunningRewardBanditControlBaseline
    frozen: FrozenActionControlBaseline
    evaluator: ContinualControlEvaluator

    @property
    def learners(self) -> tuple[ContinuingControlLearner, ...]:
        return (self.candidate, self.flat, self.frozen)


def _protocol() -> ContinuingControlProtocol:
    return ContinuingControlProtocol(
        protocol_id=PROTOTYPE_CONTROL_DEVELOPMENT_PROTOCOL_ID,
        higher_is_better=True,
        regime_schedule=REGIME_SCHEDULE,
        evaluator_regime_ids=REGIME_IDS,
        checkpoint_steps=CHECKPOINT_STEPS,
        first_exposure_checkpoint={"A": 0, "B": 1},
        # This is the actually executed frozen-action-zero reference, not a
        # fabricated fresh-agent score.
        forward_transfer_reference={"A": 1.0, "B": 0.0},
        # Descriptive recovery/stability metric references, never acceptance gates.
        recovery_thresholds={"A": 0.5, "B": 0.5},
        stability_references={"A": 1.0, "B": 1.0},
        recovery_window=1,
        worst_window_size=2,
        operation_latency_deadline_ms=LATENCY_DEADLINE_MS,
    )


def _budget() -> ContinuingControlBudget:
    return ContinuingControlBudget(
        transition_limit=HORIZON,
        decision_call_limit=HORIZON,
        environment_call_limit=HORIZON,
        update_call_limit=HORIZON,
        probe_call_limit=len(CHECKPOINT_STEPS) * len(REGIME_IDS),
        backward_call_limit=HORIZON,
        persistent_state_bytes_limit=32 * 1024 * 1024,
        state_scalar_count_limit=4 * 1024 * 1024,
        trainable_parameter_count_limit=None,
        stored_decision_id_limit=HORIZON,
    )


def _probes() -> dict[str, tuple[ControlProbe, ...]]:
    return {
        "A": (ControlProbe((0.0, 0.0), (1.0, 0.0)),),
        "B": (ControlProbe((1.0, 0.0), (0.0, 1.0)),),
    }


def _prototype_agent() -> PrototypeAgent:
    stomp = STOMPConfig(
        subtask_specs=(
            SubtaskSpec(feature_index=0, threshold=1.0e6, max_option_steps=4),
        ),
        observation_dim=2,
        n_primitive_actions=2,
        base_step_size=0.05,
        base_avg_reward_step_size=0.01,
        base_trace_decay=0.0,
        base_hidden_sizes=(),
        option_step_size=0.05,
        option_avg_reward_step_size=0.01,
        option_trace_decay=0.0,
        option_gamma=0.9,
        option_planning_backups_per_step=0,
        epsilon_base=0.0,
        epsilon_option=0.0,
    )
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=OaKConfig(stomp=stomp),
            auto_curate_every=0,
        )
    )


def _environment_seed(seed: int) -> int:
    return int((seed ^ 0xA17E5EED) & _UINT32_MAX)


def _make_seed_harness(seed: int) -> _SeedHarness:
    if seed not in DEVELOPMENT_SEEDS:
        raise ValueError("seed must be one of the fixed consumed development_seeds")
    environment_seed = _environment_seed(seed)
    environment = _PairedCueEnvironment(seed=environment_seed)
    agent = _prototype_agent()
    candidate = PrototypeAgentControlAdapter(
        agent,
        seed=seed,
        lifecycle_id=(0x50524F54, seed),  # ASCII "PROT"
        name="prototype_agent",
    )
    flat = RunningRewardBanditControlBaseline(
        n_actions=2,
        name="flat_running_reward_mean",
        lifecycle_id=(0x464C4154, seed),  # ASCII "FLAT"
    )
    frozen = FrozenActionControlBaseline(
        n_actions=2,
        action=0,
        name="frozen_action_zero",
        lifecycle_id=(0x46524F5A, seed),  # ASCII "FROZ"
    )
    evaluator = ContinualControlEvaluator(
        run_id=f"alberta.prototype-control.seed-{seed}.v1",
        protocol=_protocol(),
        environment=environment,
        probes=_probes(),
        candidate=candidate,
        baselines=(flat, frozen),
        budget=_budget(),
        clock_ns=_DeterministicClock(),
        latency_measurement_method=LATENCY_MEASUREMENT_METHOD,
    )
    return _SeedHarness(
        seed=seed,
        environment_seed=environment_seed,
        agent=agent,
        environment=environment,
        candidate=candidate,
        flat=flat,
        frozen=frozen,
        evaluator=evaluator,
    )


def prototype_continual_control_evaluator_for_seed(
    seed: int,
) -> ContinualControlEvaluator:
    """Return the exact core evaluator used by one fixed development seed.

    The factory exists so callers can use the core evaluator's strict,
    configuration-bound checkpoint API without adding an output writer to this
    in-memory report module.
    """

    return _make_seed_harness(seed).evaluator


def _array_parameter_summary(arrays: Sequence[jax.Array]) -> tuple[int, float]:
    count = sum(int(value.size) for value in arrays)
    squared = math.fsum(
        float(np.sum(np.square(np.asarray(jax.device_get(value)), dtype=np.float64)))
        for value in arrays
    )
    return count, math.sqrt(squared)


def _candidate_parameter_summary(state: PrototypeAgentState) -> tuple[int, float]:
    if type(state.oak_state) is not OaKState:
        raise TypeError("fixed development candidate must contain an exact OaKState")
    stomp = state.oak_state.stomp_state
    learner = stomp.base_learner_state
    arrays = (
        *learner.trunk_params.weights,
        *learner.trunk_params.biases,
        *learner.head_params.weights,
        *learner.head_params.biases,
        stomp.option_policies.q_weights,
        stomp.option_models.next_state_weights,
    )
    return _array_parameter_summary(arrays)


def _candidate_action_values(
    agent: PrototypeAgent,
    state: PrototypeAgentState,
    observation: tuple[float, ...],
) -> list[float]:
    if type(state.oak_state) is not OaKState:
        raise TypeError("fixed development candidate must contain an exact OaKState")
    values = OaKAgent(agent.config.oak).base_q_values(
        state.oak_state,
        jnp.asarray(observation, dtype=jnp.float32),
    )[: agent.config.oak.n_primitive_actions]
    return [float(value) for value in np.asarray(jax.device_get(values))]


def _flat_values(learner: RunningRewardBanditControlBaseline, state: Any) -> list[float]:
    payload = _mapping(learner.state_to_config(state), name="flat state")
    sums = _list(payload["reward_sums"], name="flat reward_sums")
    counts = _list(payload["action_counts"], name="flat action_counts")
    return [
        _finite_float(total, name="flat reward sum") / _exact_int(count, name="flat count")
        if _exact_int(count, name="flat count") > 0
        else 0.0
        for total, count in zip(sums, counts, strict=True)
    ]


def _diagnostic_checkpoint(
    harness: _SeedHarness,
    state: ContinualControlRunState,
) -> dict[str, object]:
    records: dict[str, object] = {}
    for index, (learner, condition) in enumerate(
        zip(harness.learners, state.conditions, strict=True)
    ):
        before_payload = _json_clone(
            learner.state_to_config(condition.learner_state)
        )
        actions = {
            regime_id: learner.probe_action(condition.learner_state, probe[0].observation)
            for regime_id, probe in _probes().items()
        }
        usage = learner.resource_usage(condition.learner_state)
        values: dict[str, list[float]] | None
        parameter_count: int
        parameter_norm: float | None
        prototype_resources: dict[str, int] | None = None
        if index == 0:
            candidate_state = cast(PrototypeAgentState, condition.learner_state)
            parameter_count, resolved_norm = _candidate_parameter_summary(candidate_state)
            parameter_norm = resolved_norm
            values = {
                regime_id: _candidate_action_values(
                    harness.agent,
                    candidate_state,
                    probe[0].observation,
                )
                for regime_id, probe in _probes().items()
            }
            prototype_resources = measure_prototype_agent_state_resources(
                candidate_state
            ).to_config()
        elif index == 1:
            flat_values = _flat_values(harness.flat, condition.learner_state)
            parameter_count = len(flat_values)
            parameter_norm = math.sqrt(math.fsum(value * value for value in flat_values))
            values = {regime_id: list(flat_values) for regime_id in REGIME_IDS}
        else:
            parameter_count = 0
            parameter_norm = None
            values = None
        after_payload = _json_clone(
            learner.state_to_config(condition.learner_state)
        )
        if not _strict_json_equal(after_payload, before_payload):
            raise ValueError("development diagnostic mutated live learner state")
        records[learner.name] = {
            "greedy_primitive_action_by_regime": actions,
            "primitive_action_values_by_regime": values,
            "parameter_count": parameter_count,
            "parameter_l2_norm": parameter_norm,
            "logical_state_resources": dataclasses.asdict(usage),
            "prototype_top_level_resource_partition": prototype_resources,
            "component_inventory": {
                "feature_components": 0,
                "prediction_components": 0,
                "option_components": 1 if index == 0 else 0,
                "model_components": 1 if index == 0 else 0,
                "dynamic_creation_or_removal_enabled": False,
            },
            "live_state_nonmutation_verified": True,
        }
    return {"step": state.step, "conditions": records}


def _diagnostic_record(
    status: DiagnosticStatus,
    *,
    value: object,
    measurement_method: str | None,
    reason: str | None,
) -> dict[str, object]:
    if status == "available":
        if value is None or measurement_method is None or reason is not None:
            raise ValueError("available diagnostic fields are inconsistent")
        applicable, available = True, True
    elif status == "unavailable":
        if value is not None or reason is None:
            raise ValueError("unavailable diagnostic fields are inconsistent")
        applicable, available = True, False
    elif status == "inapplicable":
        if value is not None or reason is None:
            raise ValueError("inapplicable diagnostic fields are inconsistent")
        applicable, available = False, False
    else:
        raise ValueError("diagnostic status is invalid")
    return {
        "status": status,
        "applicable": applicable,
        "available": available,
        "value": _json_clone(value),
        "measurement_method": measurement_method,
        "reason": reason,
    }


def _policy_churn(
    checkpoints: Sequence[Mapping[str, object]],
    *,
    condition_name: str,
) -> dict[str, object]:
    events: list[dict[str, object]] = []
    values: list[float] = []
    for before, after in zip(checkpoints[:-1], checkpoints[1:], strict=True):
        before_condition = _mapping(
            _mapping(before["conditions"], name="diagnostic conditions")[condition_name],
            name="diagnostic condition",
        )
        after_condition = _mapping(
            _mapping(after["conditions"], name="diagnostic conditions")[condition_name],
            name="diagnostic condition",
        )
        before_actions = _mapping(
            before_condition["greedy_primitive_action_by_regime"],
            name="before actions",
        )
        after_actions = _mapping(
            after_condition["greedy_primitive_action_by_regime"],
            name="after actions",
        )
        per_regime = {
            regime_id: 0.0 if before_actions[regime_id] == after_actions[regime_id] else 2.0
            for regime_id in REGIME_IDS
        }
        mean = math.fsum(per_regime.values()) / len(per_regime)
        values.append(mean)
        events.append(
            {
                "from_step": before["step"],
                "to_step": after["step"],
                "per_regime_one_hot_l1": per_regime,
                "mean_one_hot_l1": mean,
            }
        )
    return {
        "mean_over_checkpoint_intervals": math.fsum(values) / len(values),
        "events": events,
    }


def _value_churn(
    checkpoints: Sequence[Mapping[str, object]],
    *,
    condition_name: str,
) -> dict[str, object]:
    events: list[dict[str, object]] = []
    values: list[float] = []
    for before, after in zip(checkpoints[:-1], checkpoints[1:], strict=True):
        before_condition = _mapping(
            _mapping(before["conditions"], name="diagnostic conditions")[condition_name],
            name="diagnostic condition",
        )
        after_condition = _mapping(
            _mapping(after["conditions"], name="diagnostic conditions")[condition_name],
            name="diagnostic condition",
        )
        before_values = _mapping(
            before_condition["primitive_action_values_by_regime"],
            name="before values",
        )
        after_values = _mapping(
            after_condition["primitive_action_values_by_regime"],
            name="after values",
        )
        per_regime: dict[str, float] = {}
        for regime_id in REGIME_IDS:
            left = _list(before_values[regime_id], name="before value vector")
            right = _list(after_values[regime_id], name="after value vector")
            per_regime[regime_id] = math.fsum(
                abs(
                    _finite_float(a, name="before value")
                    - _finite_float(b, name="after value")
                )
                for a, b in zip(left, right, strict=True)
            ) / len(left)
        mean = math.fsum(per_regime.values()) / len(per_regime)
        values.append(mean)
        events.append(
            {
                "from_step": before["step"],
                "to_step": after["step"],
                "per_regime_mean_absolute_primitive_q_change": per_regime,
                "mean_absolute_primitive_q_change": mean,
            }
        )
    return {
        "mean_over_checkpoint_intervals": math.fsum(values) / len(values),
        "events": events,
    }


def _condition_diagnostics(
    checkpoints: Sequence[Mapping[str, object]],
    *,
    condition_name: str,
) -> dict[str, object]:
    is_prototype = condition_name == "prototype_agent"
    is_flat = condition_name == "flat_running_reward_mean"
    has_parameters = is_prototype or is_flat
    parameter_trajectory = [
        {
            "step": checkpoint["step"],
            "parameter_count": _mapping(
                _mapping(checkpoint["conditions"], name="conditions")[condition_name],
                name="condition",
            )["parameter_count"],
            "l2_norm": _mapping(
                _mapping(checkpoint["conditions"], name="conditions")[condition_name],
                name="condition",
            )["parameter_l2_norm"],
        }
        for checkpoint in checkpoints
    ]
    no_hidden_reason = (
        "fixed candidate uses a linear base learner and no learned state-builder trunk"
        if is_prototype
        else "baseline has no hidden representation units"
    )
    plasticity = {
        name: _diagnostic_record(
            "inapplicable",
            value=None,
            measurement_method=None,
            reason=no_hidden_reason,
        )
        for name in (
            "dormant_units",
            "activation_entropy",
            "effective_rank",
            "stable_rank",
        )
    }
    plasticity["parameter_norm"] = (
        _diagnostic_record(
            "available",
            value={"trajectory": parameter_trajectory},
            measurement_method=(
                "float64 host reduction over the fixed schema's base-Q head weights/biases, "
                "intra-option Q weights, and option next-state weights"
                if is_prototype
                else "L2 norm of causal sample-mean action-value parameters"
            ),
            reason=None,
        )
        if has_parameters
        else _diagnostic_record(
            "inapplicable",
            value=None,
            measurement_method=None,
            reason="fixed-action learner has no trainable parameters",
        )
    )
    plasticity["gradient_norm"] = (
        _diagnostic_record(
            "unavailable",
            value=None,
            measurement_method=None,
            reason=(
                "causal update-time gradients are not exposed by the control adapter"
                if is_prototype
                else "sample-mean update exposes no gradient object"
            ),
        )
        if has_parameters
        else _diagnostic_record(
            "inapplicable",
            value=None,
            measurement_method=None,
            reason="fixed-action learner has no gradient update",
        )
    )
    plasticity["sampled_ntk_rank"] = (
        _diagnostic_record(
            "unavailable",
            value=None,
            measurement_method=None,
            reason="no frozen Jacobian/NTK probe set is configured for this bounded lane",
        )
        if is_prototype
        else _diagnostic_record(
            "inapplicable",
            value=None,
            measurement_method=None,
            reason="baseline does not expose a differentiable neural function",
        )
    )
    plasticity["policy_churn"] = _diagnostic_record(
        "available",
        value=_policy_churn(checkpoints, condition_name=condition_name),
        measurement_method=(
            "mean L1 distance between deterministic one-hot greedy primitive actions on the "
            "same evaluator-owned probes at adjacent checkpoints"
        ),
        reason=None,
    )
    plasticity["value_churn"] = (
        _diagnostic_record(
            "available",
            value=_value_churn(checkpoints, condition_name=condition_name),
            measurement_method=(
                "mean absolute primitive-action value change on fixed evaluator-owned probes"
            ),
            reason=None,
        )
        if has_parameters
        else _diagnostic_record(
            "inapplicable",
            value=None,
            measurement_method=None,
            reason="fixed-action learner has no value estimate",
        )
    )

    component_reasons = {
        "feature_survival_curve": (
            "Prototype feature lifecycle is disabled in the fixed base configuration"
            if is_prototype
            else "baseline has no feature-component lifecycle"
        ),
        "prediction_survival_curve": (
            "GVF Horde prediction lifecycle is disabled in the fixed base configuration"
            if is_prototype
            else "baseline has no prediction-component lifecycle"
        ),
        "option_survival_curve": (
            "the one configured option is static; no option creation/removal lifecycle runs"
            if is_prototype
            else "baseline has no option-component lifecycle"
        ),
        "model_survival_curve": (
            "the option outcome model inventory is static; no model creation/removal lifecycle runs"
            if is_prototype
            else "baseline has no model-component lifecycle"
        ),
    }
    component_retention = {
        name: _diagnostic_record(
            "inapplicable",
            value=None,
            measurement_method=None,
            reason=reason,
        )
        for name, reason in component_reasons.items()
    }
    inventories = [
        {
            "step": checkpoint["step"],
            **cast(
                dict[str, object],
                _mapping(
                    _mapping(
                        _mapping(checkpoint["conditions"], name="conditions")[condition_name],
                        name="condition",
                    )["component_inventory"],
                    name="component inventory",
                ),
            ),
        }
        for checkpoint in checkpoints
    ]
    return {
        "plasticity": plasticity,
        "component_retention": component_retention,
        "static_component_inventory_trace": inventories,
    }


def _opportunity_payload(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "step": record["step"],
            "evaluator_regime_id": record["evaluator_regime_id"],
            "observation": record["observation"],
            "bootstrap_observation": record["bootstrap_observation"],
            "reward_offset": record["reward_offset"],
            "exogenous_distractor": record["exogenous_distractor"],
            "next_exogenous_distractor": record["next_exogenous_distractor"],
        }
        for record in records
    ]


def _validate_generated_seed_run(run: Mapping[str, object]) -> None:
    expected_run_fields = {
        "seed",
        "seed_role",
        "candidate_seed",
        "environment_seed",
        "environment_seed_shared_across_conditions",
        "learner_states_independent",
        "environment_states_independent",
        "control_report",
        "ownership_traces",
        "opportunity_accounting",
        "diagnostics",
    }
    _exact_keys(run, expected_run_fields, name="seed run")
    seed = _exact_int(run["seed"], name="seed run.seed", maximum=_UINT32_MAX)
    if seed not in DEVELOPMENT_SEEDS:
        raise ValueError("seed run is not one of the fixed development seeds")
    if (
        run["seed_role"] != "consumed_development_seed_nonpromoting"
        or run["candidate_seed"] != seed
        or run["environment_seed"] != _environment_seed(seed)
        or run["environment_seed_shared_across_conditions"] is not True
        or run["learner_states_independent"] is not True
        or run["environment_states_independent"] is not True
    ):
        raise ValueError("seed run identity/independence declaration differs")
    control = _mapping(run["control_report"], name="control report")
    validation = validate_continual_control_report(control)
    if not validation.valid:
        raise ValueError(
            "core control report failed reconstruction: "
            + "; ".join(validation.errors)
        )
    conditions = _list(control["conditions"], name="control conditions")
    ownership = _list(run["ownership_traces"], name="ownership traces")
    if len(conditions) != 3 or len(ownership) != 3:
        raise ValueError("seed run must contain exactly three conditions")
    expected_names = (
        "prototype_agent",
        "flat_running_reward_mean",
        "frozen_action_zero",
    )
    expected_lifecycle_tags = (0x50524F54, 0x464C4154, 0x46524F5A)
    opportunities: list[list[dict[str, object]]] = []
    realized: dict[str, int] = {}
    for condition_index, (condition_raw, ownership_raw) in enumerate(
        zip(conditions, ownership, strict=True)
    ):
        condition = _mapping(condition_raw, name="control condition")
        trace = _mapping(ownership_raw, name="ownership trace")
        if (
            condition["name"] != expected_names[condition_index]
            or trace["name"] != condition["name"]
        ):
            raise ValueError("ownership trace condition order differs")
        _exact_keys(
            trace,
            {
                "name",
                "records",
                "consumed_decision_ids",
                "environment_final_step",
                "environment_final_observation",
                "regime_metadata_location",
            },
            name="ownership trace",
        )
        if trace["regime_metadata_location"] != "evaluator-owned raw trace only":
            raise ValueError("ownership regime-metadata boundary differs")
        records = [
            _mapping(record, name="ownership record")
            for record in _list(trace["records"], name="ownership records")
        ]
        if len(records) != HORIZON:
            raise ValueError("ownership record count differs from horizon")
        consumed = [
            _decision_id(value, name="consumed decision ID")
            for value in _list(trace["consumed_decision_ids"], name="consumed IDs")
        ]
        record_ids = [
            _decision_id(record["decision_id"], name="record decision ID")
            for record in records
        ]
        if consumed != record_ids or len(set(record_ids)) != len(record_ids):
            raise ValueError("ownership decision identity trace is invalid")
        if any(
            value[:2] != (expected_lifecycle_tags[condition_index], seed)
            for value in record_ids
        ):
            raise ValueError("ownership lifecycle identity differs")
        rewards = _list(
            _mapping(condition["trace"], name="condition trace")["rewards"],
            name="condition rewards",
        )
        if rewards != [record["reward"] for record in records]:
            raise ValueError("ownership rewards differ from core trace")
        counts = _mapping(
            _mapping(condition["operations"], name="condition operations")["counts"],
            name="condition operation counts",
        )
        if (
            counts["processed_transitions"] != HORIZON
            or counts["dropped_transitions"] != 0
            or counts["environment_calls"] != HORIZON
            or counts["update_calls"] != HORIZON
        ):
            raise ValueError("condition realized work differs from ownership trace")
        for step, record in enumerate(records):
            _exact_keys(
                record,
                {field.name for field in dataclasses.fields(_TransitionRecord)},
                name="ownership record",
            )
            if (
                record["step"] != step
                or record["evaluator_regime_id"] != REGIME_SCHEDULE[step]
                or type(record["armed"]) is not bool
                or record["armed"] is not True
            ):
                raise ValueError("ownership event ordering or arming differs")
            action = _exact_int(record["action"], name="record action", maximum=1)
            correct = 0 if REGIME_SCHEDULE[step] == "A" else 1
            observation = _observation(record["observation"], name="record observation")
            bootstrap = _observation(
                record["bootstrap_observation"],
                name="record bootstrap observation",
            )
            if (
                observation[0] != (0.0 if REGIME_SCHEDULE[step] == "A" else 1.0)
                or observation[1] != record["exogenous_distractor"]
                or bootstrap[1] != record["next_exogenous_distractor"]
                or (step > 0 and observation != _observation(
                    records[step - 1]["bootstrap_observation"],
                    name="previous bootstrap observation",
                ))
            ):
                raise ValueError("ownership observation/exogenous reconstruction differs")
            expected_reward = (1.0 if action == correct else 0.0) + _finite_float(
                record["reward_offset"], name="reward offset"
            )
            expected_near_miss = action != correct
            expected_safety_cost = 0.1 if expected_near_miss else 0.0
            if (
                record["reward"] != expected_reward
                or record["discount"] != 0.9
                or record["near_miss"] is not expected_near_miss
                or record["safety_cost"] != expected_safety_cost
            ):
                raise ValueError("ownership action/outcome reconstruction differs")
        if (
            trace["environment_final_step"] != HORIZON
            or trace["environment_final_observation"]
            != records[-1]["bootstrap_observation"]
        ):
            raise ValueError("ownership final environment state differs")
        opportunity = _opportunity_payload(records)
        opportunities.append(opportunity)
        realized[condition["name"]] = len(records)
    if any(value != opportunities[0] for value in opportunities[1:]):
        raise ValueError("conditions do not share identical exogenous opportunities")
    accounting = _mapping(run["opportunity_accounting"], name="opportunity accounting")
    expected_accounting = {
        "declared_transitions_per_condition": HORIZON,
        "declared_observation_opportunities_per_condition": HORIZON,
        "condition_count": 3,
        "realized_transition_counts": realized,
        "realized_observation_counts": realized,
        "identical_exogenous_opportunities": True,
        "exogenous_opportunity_sha256": _digest(opportunities[0]),
        "learner_environment_states_independent": True,
        "paired_randomness_scope": "pre-generated exogenous observations and reward offsets only",
        "matched_realized_compute_claimed": False,
    }
    if not _strict_json_equal(accounting, expected_accounting):
        raise ValueError("opportunity accounting does not reconstruct")


def _run_seed(seed: int) -> dict[str, object]:
    harness = _make_seed_harness(seed)
    state = harness.evaluator.init()
    checkpoints: list[dict[str, object]] = [_diagnostic_checkpoint(harness, state)]
    for _ in range(HORIZON):
        state = harness.evaluator.advance(state, steps=1)
        if state.step in CHECKPOINT_STEPS:
            checkpoints.append(_diagnostic_checkpoint(harness, state))
    control_report = harness.evaluator.build_report(state)
    validation = validate_continual_control_report(
        control_report,
        expected_evaluator_config=harness.evaluator.to_config(),
    )
    if not validation.valid:
        raise ValueError("generated core control report is invalid")

    ownership: list[dict[str, object]] = []
    conditions = cast(list[Mapping[str, object]], control_report["conditions"])
    for condition_report, condition_state in zip(
        conditions,
        state.conditions,
        strict=True,
    ):
        environment_payload = _mapping(
            harness.environment.state_to_config(condition_state.environment_state),
            name="environment payload",
        )
        ownership.append(
            {
                "name": condition_report["name"],
                "records": _json_clone(environment_payload["records"]),
                "consumed_decision_ids": [
                    list(value) for value in condition_state.used_decision_ids
                ],
                "environment_final_step": environment_payload["step"],
                "environment_final_observation": environment_payload["observation"],
                "regime_metadata_location": "evaluator-owned raw trace only",
            }
        )
    first_records = cast(list[Mapping[str, object]], ownership[0]["records"])
    realized = {
        cast(str, condition["name"]): len(cast(list[object], trace["records"]))
        for condition, trace in zip(conditions, ownership, strict=True)
    }
    opportunity = {
        "declared_transitions_per_condition": HORIZON,
        "declared_observation_opportunities_per_condition": HORIZON,
        "condition_count": 3,
        "realized_transition_counts": realized,
        "realized_observation_counts": dict(realized),
        "identical_exogenous_opportunities": True,
        "exogenous_opportunity_sha256": _digest(_opportunity_payload(first_records)),
        "learner_environment_states_independent": True,
        "paired_randomness_scope": "pre-generated exogenous observations and reward offsets only",
        "matched_realized_compute_claimed": False,
    }
    diagnostics = {
        "raw_checkpoint_trace": checkpoints,
        "conditions": {
            learner.name: _condition_diagnostics(
                checkpoints,
                condition_name=learner.name,
            )
            for learner in harness.learners
        },
        "diagnostic_checkpoint_steps": [0, *CHECKPOINT_STEPS],
        "probe_regime_order": list(REGIME_IDS),
        "probe_mutation_checked_by_core_evaluator": True,
    }
    run = {
        "seed": seed,
        "seed_role": "consumed_development_seed_nonpromoting",
        "candidate_seed": seed,
        "environment_seed": harness.environment_seed,
        "environment_seed_shared_across_conditions": True,
        "learner_states_independent": True,
        "environment_states_independent": True,
        "control_report": control_report,
        "ownership_traces": ownership,
        "opportunity_accounting": opportunity,
        "diagnostics": diagnostics,
    }
    canonical = cast(dict[str, object], _json_clone(run))
    _validate_generated_seed_run(canonical)
    return canonical


def _field_record(
    status: DiagnosticStatus,
    *,
    report_path: str | None,
    reason: str | None,
) -> dict[str, object]:
    return {
        "status": status,
        "applicable": status != "inapplicable",
        "available": status == "available",
        "report_path": report_path,
        "reason": reason,
    }


def _wp1_field_coverage() -> dict[str, object]:
    performance_names = (
        "prequential_return",
        "lifetime_return",
        "per_regime_online_mean_return",
        "adaptation_auc",
        "recovery",
        "per_regime_final_performance",
        "mean_final_performance",
        "forgetting",
        "backward_transfer",
        "forward_transfer",
        "stability",
        "worst_window",
    )
    operation_path = "runs[*].control_report.conditions[*].operations"
    resource_path = "runs[*].control_report.conditions[*].resources"
    resources = {
        name: _field_record(
            "available",
            report_path=operation_path,
            reason=None,
        )
        for name in (
            "processed_observations",
            "delayed_observations",
            "dropped_observations",
            "decision_calls",
            "environment_calls",
            "update_calls",
            "held_out_probe_calls",
            "logical_latency_method_and_quantiles",
        )
    }
    resources.update(
        {
            "internal_forward_calls": _field_record(
                "unavailable",
                report_path=None,
                reason=(
                    "adapter-level decision/probe calls are counted, but Prototype's internal "
                    "forward evaluations are not instrumented"
                ),
            ),
            "backward_calls": _field_record(
                "unavailable",
                report_path=operation_path,
                reason=(
                    "flat/frozen conditions report exact zero; Prototype's adapter reports "
                    "backward calls unavailable rather than fabricating zero"
                ),
            ),
            "persistent_state_bytes_high_water": _field_record(
                "available",
                report_path=resource_path,
                reason=None,
            ),
            "state_scalar_count_high_water": _field_record(
                "available",
                report_path=resource_path,
                reason=None,
            ),
            "host_allocator_high_water": _field_record(
                "unavailable",
                report_path=None,
                reason="no process allocator-residency sampler is connected",
            ),
            "accelerator_allocator_high_water": _field_record(
                "unavailable",
                report_path=None,
                reason="no accelerator allocator-residency sampler is connected",
            ),
        }
    )
    world_reason = "standalone/ensemble world-model lanes are disabled by the fixed config"
    return {
        "performance": {
            name: _field_record(
                "available",
                report_path=f"runs[*].control_report.conditions[*].metrics.{name}",
                reason=None,
            )
            for name in performance_names
        },
        "resources": resources,
        "safety": _field_record(
            "available",
            report_path="runs[*].control_report.conditions[*].safety",
            reason=None,
        ),
        "plasticity": _field_record(
            "available",
            report_path="runs[*].diagnostics.conditions[*].plasticity",
            reason=None,
        ),
        "component_retention": _field_record(
            "available",
            report_path="runs[*].diagnostics.conditions[*].component_retention",
            reason=None,
        ),
        "world_model": {
            name: _field_record(
                "inapplicable",
                report_path=None,
                reason=world_reason,
            )
            for name in (
                "observation_nll_or_mse",
                "reward_calibration",
                "termination_calibration",
                "ensemble_disagreement_calibration",
                "multi_step_rollout_error",
            )
        },
        "energy_proxy": _field_record(
            "unavailable",
            report_path=None,
            reason="no energy meter or calibrated energy proxy is connected",
        ),
        "wall_clock_latency": _field_record(
            "unavailable",
            report_path=None,
            reason="this exact-replay lane uses deterministic logical latency only",
        ),
    }


def _build_report(
    config: PrototypeContinualControlDevelopmentConfig,
) -> dict[str, object]:
    source_before = prototype_continual_control_source_manifest()
    runtime = prototype_continual_control_runtime_identity()
    runs = [_run_seed(seed) for seed in config.development_seeds]
    source_after = prototype_continual_control_source_manifest()
    if source_after != source_before:
        raise RuntimeError("declared source files changed during report construction")
    body: dict[str, object] = {
        "schema_version": PROTOTYPE_CONTROL_DEVELOPMENT_REPORT_SCHEMA,
        "type": "PrototypeContinualControlDevelopmentReport",
        "assessment_status": ASSESSMENT_STATUS,
        "interpretation": (
            "L0 in-memory descriptive report for one uninterrupted ordinary-observation "
            "Prototype/flat/frozen comparison per consumed development seed"
        ),
        "config": config.to_config(),
        "config_sha256": _digest(config.to_config()),
        "source_manifest": source_before,
        "source_manifest_sha256": _digest(source_before),
        "runtime_identity": runtime,
        "runtime_identity_sha256": _digest(runtime),
        "development_seeds": list(config.development_seeds),
        "seed_policy": {
            "fixed": True,
            "consumed": True,
            "promotion_eligible": False,
            "seed_count": len(config.development_seeds),
            "multi_seed_inference_performed": False,
            "confidence_intervals_available": False,
        },
        "stream_contract": {
            "uninterrupted_continuing_stream": True,
            "environment_resets": 0,
            "learner_visible_regime_or_task_labels": False,
            "decision_before_outcome_before_update": True,
            "exact_action_decision_ownership": True,
            "independent_learner_state": True,
            "independent_environment_state": True,
            "paired_randomness_scope": (
                "pre-generated exogenous observations and reward offsets only"
            ),
        },
        "conditions": {
            "candidate": {
                "name": "prototype_agent",
                "type": "PrototypeAgentControlAdapter",
                "base_prototype": True,
            },
            "ordinary_baselines": [
                {
                    "name": "flat_running_reward_mean",
                    "type": "RunningRewardBanditControlBaseline",
                    "no_continual_mechanism": True,
                    "semantics": "causal observation-agnostic sample-mean action values",
                },
                {
                    "name": "frozen_action_zero",
                    "type": "FrozenActionControlBaseline",
                    "no_learning": True,
                    "semantics": "actual fixed-action condition on the same continuing stream",
                },
            ],
        },
        "references": {
            "frozen": {
                "status": "available",
                "ordinary_lane_executed": True,
                "condition": "frozen_action_zero",
                "forward_transfer_reference": {"A": 1.0, "B": 0.0},
                "semantics": "held-out action score of the actually executed fixed action",
            },
            "fresh_per_regime": {
                "status": "unavailable",
                "ordinary_lane_executed": False,
                "value": None,
                "reason": (
                    "regime-routed fresh state requires evaluator privilege; exposing that "
                    "routing to an ordinary learner would violate this lane's label-free contract"
                ),
                "available_via": (
                    "alberta_framework.evaluation.continual_control_reference_suite."
                    "PrivilegedContinualControlReferenceSuite role "
                    "retained_fresh_once_per_regime_identity"
                ),
            },
        },
        "runs": runs,
        "wp1_field_coverage": _wp1_field_coverage(),
        "checkpoint_contract": {
            "available": True,
            "owner": "ContinualControlEvaluator.save_checkpoint/load_checkpoint",
            "schema_version": CONTROL_CHECKPOINT_SCHEMA,
            "exact_seed_factory": (
                "prototype_continual_control_evaluator_for_seed"
            ),
            "configuration_bound": True,
            "report_builder_writes_checkpoint": False,
            "checkpoint_split_for_integration_replay": CHECKPOINT_SPLIT,
        },
        "raw_traces_reconstructable": True,
        "source_runtime_bound": True,
        "exact_causal_replay_required": True,
        "latency_method": LATENCY_MEASUREMENT_METHOD,
        "thresholds": [],
        "winner": None,
        "efficacy_claimed": False,
        "sota_claimed": False,
        "safety_claimed": False,
        "evidence_claimed": False,
        "accepted_scientific_evidence": False,
        "promotion_authority": PROMOTION_AUTHORITY,
        "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        "output_written": OUTPUT_WRITES,
        "output_path": None,
        "limitations": list(_LIMITATIONS),
    }
    canonical_body = cast(dict[str, object], _json_clone(body))
    report = {**canonical_body, "report_sha256": _digest(canonical_body)}
    if len(_canonical_json_bytes(report)) > _MAX_REPORT_BYTES:
        raise ValueError("Prototype continual-control report exceeds byte cap")
    return report


def build_prototype_continual_control_development_report(
    config: PrototypeContinualControlDevelopmentConfig | None = None,
) -> dict[str, object]:
    """Run the fixed seeds and return one strict in-memory L0 report."""

    resolved = PrototypeContinualControlDevelopmentConfig() if config is None else config
    if type(resolved) is not PrototypeContinualControlDevelopmentConfig:
        raise TypeError("config must be an exact PrototypeContinualControlDevelopmentConfig")
    return _build_report(resolved)


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeContinualControlDevelopmentValidationReceipt:
    """Fail-closed receipt with no evidence or promotion authority."""

    valid: bool
    assessment_status: str
    source_runtime_bound: bool
    raw_ownership_valid: bool
    exact_replay: bool
    output_written: bool
    promotion_authority: bool


def validate_prototype_continual_control_development_report(
    report: object,
) -> PrototypeContinualControlDevelopmentValidationReceipt:
    """Validate only through exact source/runtime-bound causal replay."""

    raw = _mapping(report, name="report")
    if len(_canonical_json_bytes(raw)) > _MAX_REPORT_BYTES:
        raise ValueError("Prototype continual-control report exceeds byte cap")
    if (
        raw.get("schema_version") != PROTOTYPE_CONTROL_DEVELOPMENT_REPORT_SCHEMA
        or raw.get("type") != "PrototypeContinualControlDevelopmentReport"
    ):
        raise ValueError("report schema/type differs")
    body = {name: raw[name] for name in raw if name != "report_sha256"}
    if raw.get("report_sha256") != _digest(body):
        raise ValueError("report digest integrity check failed")
    config = PrototypeContinualControlDevelopmentConfig.from_config(raw.get("config"))
    if raw.get("config_sha256") != _digest(config.to_config()):
        raise ValueError("report config digest differs")
    if not _strict_json_equal(
        raw.get("source_manifest"),
        prototype_continual_control_source_manifest(),
    ):
        raise ValueError("report source manifest differs")
    if raw.get("source_manifest_sha256") != _digest(raw.get("source_manifest")):
        raise ValueError("report source-manifest digest differs")
    if not _strict_json_equal(
        raw.get("runtime_identity"),
        prototype_continual_control_runtime_identity(),
    ):
        raise ValueError("report runtime identity differs")
    if raw.get("runtime_identity_sha256") != _digest(raw.get("runtime_identity")):
        raise ValueError("report runtime-identity digest differs")
    required: dict[str, object] = {
        "assessment_status": ASSESSMENT_STATUS,
        "thresholds": [],
        "winner": None,
        "efficacy_claimed": False,
        "sota_claimed": False,
        "safety_claimed": False,
        "evidence_claimed": False,
        "accepted_scientific_evidence": False,
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
        "output_written": False,
        "output_path": None,
        "raw_traces_reconstructable": True,
        "source_runtime_bound": True,
        "exact_causal_replay_required": True,
    }
    for name, expected in required.items():
        if not _strict_json_equal(raw.get(name), expected):
            raise ValueError(f"report {name} differs")
    for run in _list(raw.get("runs"), name="report.runs"):
        _validate_generated_seed_run(_mapping(run, name="report run"))
    expected = _build_report(config)
    if not _strict_json_equal(dict(raw), expected):
        raise ValueError("report differs from exact causal replay")
    return PrototypeContinualControlDevelopmentValidationReceipt(
        valid=True,
        assessment_status=ASSESSMENT_STATUS,
        source_runtime_bound=True,
        raw_ownership_valid=True,
        exact_replay=True,
        output_written=False,
        promotion_authority=False,
    )


def prototype_continual_control_development_report_json(
    report: Mapping[str, object],
) -> str:
    """Serialize only a report that passes exact replay validation."""

    validate_prototype_continual_control_development_report(report)
    return json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _reject_nonstandard_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def load_prototype_continual_control_development_report_json(
    payload: str,
) -> dict[str, object]:
    """Strictly parse in-memory JSON; no file reader/writer is provided."""

    if type(payload) is not str:
        raise TypeError("payload must be an exact string")
    parsed = json.loads(
        payload,
        parse_constant=_reject_nonstandard_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    report = cast(dict[str, object], _mapping(parsed, name="report"))
    validate_prototype_continual_control_development_report(report)
    return report


__all__ = [
    "ASSESSMENT_STATUS",
    "CHECKPOINT_SPLIT",
    "CHECKPOINT_STEPS",
    "DEVELOPMENT_SEEDS",
    "HORIZON",
    "LATENCY_MEASUREMENT_METHOD",
    "OUTPUT_WRITES",
    "PROMOTION_AUTHORITY",
    "PROTOTYPE_CONTROL_DEVELOPMENT_CONFIG_SCHEMA",
    "PROTOTYPE_CONTROL_DEVELOPMENT_ENVIRONMENT_SCHEMA",
    "PROTOTYPE_CONTROL_DEVELOPMENT_PROTOCOL_ID",
    "PROTOTYPE_CONTROL_DEVELOPMENT_REPORT_SCHEMA",
    "PrototypeContinualControlDevelopmentConfig",
    "PrototypeContinualControlDevelopmentValidationReceipt",
    "REGIME_IDS",
    "REGIME_SCHEDULE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "build_prototype_continual_control_development_report",
    "load_prototype_continual_control_development_report_json",
    "prototype_continual_control_development_report_json",
    "prototype_continual_control_evaluator_for_seed",
    "prototype_continual_control_runtime_identity",
    "prototype_continual_control_source_manifest",
    "validate_prototype_continual_control_development_report",
]
