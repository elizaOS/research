# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Paired, reset-free development diagnostic over the embodied Prototype harness.

The two arms receive bit-identical exogenous preparation inputs while retaining
separately owned policy, plant, envelope, and grounded-shadow trajectories.  The
control differs in exactly five STOMP step-size configuration fields.  That is a
scoped optimizer control, not a claim that every mutable memory or shadow surface
is frozen.

This module is deliberately nonpromoting.  It writes no files, declares no
winner or threshold, and does not expose or assess delight, KondoGate forward
admission intent, or KondoSparseActor execution-level backward inclusion.  The
``GradientJoy`` names are retained elsewhere only for historical compatibility.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import platform
import struct
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final, Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.oak import OaKState
from alberta_framework.core.prototype_consolidated_semantic_memory import (
    PrototypeConsolidatedSemanticMemoryState,
)
from alberta_framework.core.prototype_embodied_development_harness import (
    PrototypeEmbodiedDevelopmentHarness,
    PrototypeEmbodiedDevelopmentHarnessPreparationInput,
    PrototypeEmbodiedDevelopmentHarnessSettlementResult,
    PrototypeEmbodiedDevelopmentHarnessState,
)
from alberta_framework.core.types import LMSState

REPO_ROOT = Path(__file__).resolve().parents[2]

PROTOTYPE_EMBODIED_PAIRED_CONFIG_SCHEMA = (
    "alberta.prototype-embodied-paired-development.config.v1"
)
PROTOTYPE_EMBODIED_PAIRED_REPORT_SCHEMA = (
    "alberta.prototype-embodied-paired-development.report.v1"
)
PROTOTYPE_EMBODIED_PAIRED_CHECKPOINT_SCHEMA = (
    "alberta.prototype-embodied-paired-development.checkpoint.v1"
)

ASSESSMENT_STATUS: Final = "not_assessed"
OUTPUT_WRITES: Final = False
PHYSICAL_DISPATCH_COUNT: Final = 0
EVIDENCE_AUTHORITY: Final = False
PROMOTION_AUTHORITY: Final = False
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
NO_ACTION_SENTINEL: Final = -1
UNAVAILABLE_REWARD_SENTINEL: Final = 0.0

AdaptiveArm = Literal["adaptive_stomp", "zero_stomp_step_size_control"]
ARM_ORDER: Final[tuple[AdaptiveArm, ...]] = (
    "adaptive_stomp",
    "zero_stomp_step_size_control",
)

PERMITTED_STOMP_CONFIG_DIFFERENCES: Final[tuple[str, ...]] = (
    "base_step_size",
    "base_avg_reward_step_size",
    "option_step_size",
    "option_avg_reward_step_size",
    "option_model_step_size",
)

_SOURCE_PATHS: Final = (
    Path("alberta_framework/benchmarks/prototype_embodied_paired_development.py"),
    Path("alberta_framework/benchmarks/prototype_embodied_paired_development_rig.py"),
    Path("alberta_framework/core/embodied_safety_envelope.py"),
    Path("alberta_framework/core/grounded_joint_world_model.py"),
    Path("alberta_framework/core/multi_head_learner.py"),
    Path("alberta_framework/core/oak.py"),
    Path("alberta_framework/core/options.py"),
    Path("alberta_framework/core/prototype_agent.py"),
    Path("alberta_framework/core/prototype_consolidated_memory.py"),
    Path("alberta_framework/core/prototype_consolidated_semantic_memory.py"),
    Path("alberta_framework/core/prototype_embodied_command_adapter.py"),
    Path("alberta_framework/core/prototype_embodied_development_harness.py"),
    Path("alberta_framework/core/types.py"),
)

_LIMITATIONS: Final = (
    "finite consumed development trajectory; every assessment status is not_assessed",
    "the deterministic development key is not an untouched or promotion-eligible seed",
    "paired inputs are exogenous only; each arm owns its causal policy and plant trajectory",
    "equal arm behavior is valid and is never treated as a failed diagnostic",
    "the zero-step-size control freezes only five declared STOMP update rates",
    "option-model decay statistics, semantic memory, and grounded shadow may still change",
    "the fixed primitive plant supports lifetime curves, not post-change adaptation claims",
    "delight and KondoGate forward admission intent are not exposed or assessed",
    "KondoSparseActor execution-level backward inclusion is not exposed or assessed",
    "GradientJoy names are historical compatibility names and are not assessed",
    "the hard envelope is not a physical-safety certificate or deployment authority",
    "SHA-256 fields are integrity/source bindings, not authentication",
    "the selected source manifest is not a complete transitive dependency lock",
    "the selected runtime fields are not a complete JAX/XLA environment or binary lock",
    "no result grants efficacy, safety, evidence, deployment, or promotion authority",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json_equal(actual: object, expected: object) -> bool:
    if expected is None:
        return actual is None
    if type(expected) is float:
        return type(actual) is float and struct.pack(">d", actual) == struct.pack(
            ">d", expected
        )
    if type(expected) in {bool, int, str}:
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


def _array_payload(value: object) -> dict[str, object]:
    array = jnp.asarray(value)
    typed_prng_key = jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key)
    logical_dtype = str(array.dtype)
    logical_shape = list(array.shape)
    prng_impl = str(jr.key_impl(array)) if typed_prng_key else None
    if typed_prng_key:
        array = jr.key_data(array)
    host = np.asarray(jax.device_get(array))
    return {
        "typed_prng_key": typed_prng_key,
        "prng_impl": prng_impl,
        "logical_dtype": logical_dtype,
        "logical_shape": logical_shape,
        "dtype": str(host.dtype),
        "shape": list(host.shape),
        "bytes_hex": host.tobytes(order="C").hex(),
    }


def _tree_payload(value: object) -> dict[str, object]:
    leaves, tree = jax.tree_util.tree_flatten(value)
    payload = {
        "treedef": str(tree),
        "leaves": [_array_payload(leaf) for leaf in leaves],
    }
    return {**payload, "tree_sha256": _canonical_sha256(payload)}


def _tree_sha256(value: object) -> str:
    return cast(str, _tree_payload(value)["tree_sha256"])


def _tree_bits_equal(left: object, right: object) -> bool:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if cast(Any, left_tree) != cast(Any, right_tree):
        return False
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        left_is_key = jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key)
        right_is_key = jax.dtypes.issubdtype(right_array.dtype, jax.dtypes.prng_key)
        if left_is_key != right_is_key:
            return False
        if left_is_key:
            if str(jr.key_impl(left_array)) != str(jr.key_impl(right_array)):
                return False
            left_array = jr.key_data(left_array)
            right_array = jr.key_data(right_array)
        left_host = np.asarray(jax.device_get(left_array))
        right_host = np.asarray(jax.device_get(right_array))
        if (
            left_host.shape != right_host.shape
            or left_host.dtype != right_host.dtype
            or left_host.tobytes(order="C") != right_host.tobytes(order="C")
        ):
            return False
    return True


def _words(value: int) -> Array:
    return jnp.asarray(((value >> 32) & 0xFFFFFFFF, value & 0xFFFFFFFF), dtype=jnp.uint32)


def _host_bool(value: object) -> bool:
    return bool(np.asarray(jax.device_get(value)))


def _host_int(value: object) -> int:
    return int(np.asarray(jax.device_get(value)))


def _host_float(value: object) -> float:
    result = float(np.asarray(jax.device_get(value)))
    if not math.isfinite(result):
        raise ValueError("development trace contains a nonfinite scalar")
    return result


def _host_list(value: object) -> list[object]:
    return cast(list[object], np.asarray(jax.device_get(value)).tolist())


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def prototype_embodied_paired_source_manifest() -> dict[str, object]:
    """Hash the live selected mechanism sources, including this module."""

    files: list[dict[str, object]] = []
    for relative in _SOURCE_PATHS:
        path = REPO_ROOT / relative
        payload = path.read_bytes()
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return {
        "schema": "alberta.prototype-embodied-paired-development.sources.v1",
        "selected_not_transitive": True,
        "files": files,
    }


def prototype_embodied_paired_runtime_identity() -> dict[str, object]:
    """Return the selected host/JAX runtime fields bound into replay."""

    devices = [
        {
            "platform": device.platform,
            "device_kind": device.device_kind,
            "id": int(device.id),
        }
        for device in jax.devices()
    ]
    return {
        "schema": "alberta.prototype-embodied-paired-development.runtime.v1",
        "selected_not_complete_environment_or_binary_lock": True,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": _package_version("jax"),
        "jaxlib": _package_version("jaxlib"),
        "numpy": _package_version("numpy"),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "devices": devices,
    }


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeEmbodiedPairedDevelopmentConfig:
    """Finite, consumed development schedule; no reset or task identifiers."""

    attempts: int = 4
    development_key: int = 0
    bridge_disconnect_attempts: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        if type(self.attempts) is not int or self.attempts != 4:
            raise ValueError("v1 attempts must be the exact int 4")
        if (
            type(self.development_key) is not int
            or not 0 <= self.development_key <= 0xFFFFFFFF
        ):
            raise ValueError("development_key must be an exact uint32 integer")
        if type(self.bridge_disconnect_attempts) is not tuple:
            raise TypeError("bridge_disconnect_attempts must be an exact tuple")
        if any(type(index) is not int for index in self.bridge_disconnect_attempts):
            raise TypeError("bridge disconnect indices must be exact ints")
        if self.bridge_disconnect_attempts != (1,):
            raise ValueError("v1 bridge_disconnect_attempts must be exactly (1,)")

    def to_config(self) -> dict[str, object]:
        return {
            "schema": PROTOTYPE_EMBODIED_PAIRED_CONFIG_SCHEMA,
            "attempts": self.attempts,
            "development_key": self.development_key,
            "bridge_disconnect_attempts": list(self.bridge_disconnect_attempts),
            "continuing": True,
            "environment_resets": 0,
            "task_identifiers": False,
            "paired_scope": "exogenous_preparation_inputs_only",
            "independent_causal_policy_and_plant_trajectories": True,
            "fixed_v1_protocol": True,
            "expected_committed_plant_transitions_per_arm": 3,
            "expected_no_action_attempts_per_arm": 1,
            "assessment_status": ASSESSMENT_STATUS,
            "thresholds": [],
            "winner_declared": False,
            "output_writes": OUTPUT_WRITES,
            "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeEmbodiedPairedRunState:
    """Integrity-sealed causal prefix for two separately owned harness states."""

    attempt_index: int
    adaptive_stomp: PrototypeEmbodiedDevelopmentHarnessState
    zero_stomp_step_size_control: PrototypeEmbodiedDevelopmentHarnessState
    chain_heads: tuple[str, str]
    records_json: tuple[str, ...]
    binding_sha256: str
    integrity_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeEmbodiedPairedValidationReceipt:
    """Strict replay receipt without assessment or promotion authority."""

    valid: bool
    assessment_status: str
    source_runtime_bound: bool
    exact_causal_replay: bool
    output_written: bool
    physical_dispatch_count: int
    evidence_authority: bool
    promotion_authority: bool


def _stomp_config(value: dict[str, object]) -> dict[str, object]:
    try:
        adapter = cast(dict[str, object], value["adapter"])
        semantic = cast(dict[str, object], adapter["semantic"])
        composition = cast(dict[str, object], semantic["composition"])
        prototype = cast(dict[str, object], composition["prototype"])
        oak = cast(dict[str, object], prototype["oak"])
        stomp = cast(dict[str, object], oak["stomp"])
    except (KeyError, TypeError) as exc:
        raise ValueError("harness config lacks the expected semantic Prototype/STOMP path") from exc
    if type(stomp) is not dict:
        raise ValueError("harness STOMP config must be an exact dict")
    return stomp


def _normalized_harness_config(value: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(value)
    stomp = _stomp_config(normalized)
    for name in PERMITTED_STOMP_CONFIG_DIFFERENCES:
        if name not in stomp:
            raise ValueError(f"harness STOMP config lacks permitted field {name}")
        stomp[name] = 0.0
    return normalized


def _prototype_state(state: PrototypeEmbodiedDevelopmentHarnessState) -> object:
    return state.adapter.semantic.composition.prototype


def _base_optimizer_step_sizes(
    semantic: PrototypeConsolidatedSemanticMemoryState,
) -> tuple[Array, ...]:
    prototype = semantic.composition.prototype
    oak = cast(OaKState, prototype.oak_state)
    learner = oak.stomp_state.base_learner_state
    states: list[object] = list(learner.trunk_optimizer_states)
    for pair in learner.head_optimizer_states:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("base learner optimizer pair layout differs")
        states.extend(pair)
    if not states or any(type(item) is not LMSState for item in states):
        raise ValueError("paired development benchmark requires exact LMS base optimizers")
    return tuple(cast(LMSState, item).step_size for item in states)


def _normalize_base_optimizer_step_sizes(
    semantic: PrototypeConsolidatedSemanticMemoryState,
) -> PrototypeConsolidatedSemanticMemoryState:
    prototype = semantic.composition.prototype
    oak = cast(OaKState, prototype.oak_state)
    learner = oak.stomp_state.base_learner_state

    def cleared(value: object) -> LMSState:
        if type(value) is not LMSState:
            raise ValueError("paired development benchmark requires exact LMS base optimizers")
        return value.replace(step_size=jnp.asarray(0.0, dtype=jnp.float32))

    trunk = tuple(cleared(item) for item in learner.trunk_optimizer_states)
    heads = tuple(
        tuple(cleared(item) for item in cast(tuple[object, ...], pair))
        for pair in learner.head_optimizer_states
    )
    learner = learner.replace(
        trunk_optimizer_states=trunk,
        head_optimizer_states=heads,
    )
    stomp = oak.stomp_state.replace(base_learner_state=learner)
    normalized_prototype = prototype.replace(oak_state=oak.replace(stomp_state=stomp))
    composition = semantic.composition.replace(prototype=normalized_prototype)
    return semantic.replace(composition=composition)


def _step_size_governed_parameter_tree(
    state: PrototypeEmbodiedDevelopmentHarnessState,
) -> object:
    prototype = state.adapter.semantic.composition.prototype
    oak = cast(OaKState, prototype.oak_state)
    stomp = oak.stomp_state
    learner = stomp.base_learner_state
    return (
        learner.trunk_params,
        learner.head_params,
        stomp.base_average_reward,
        stomp.option_policies.q_weights,
        stomp.option_policies.average_rewards,
        stomp.option_models.next_state_weights,
    )


def _normalized_trapezoid(values: list[float]) -> float | None:
    """Return trapezoidal area divided by its equally spaced index span."""

    if len(values) < 2:
        return None
    area = sum((left + right) * 0.5 for left, right in zip(values, values[1:]))
    return float(area / (len(values) - 1))


def _validate_fixed_v1_arm_record(
    arm: AdaptiveArm,
    record: dict[str, object],
    *,
    expected_connected: bool,
) -> None:
    """Reject any arm outcome outside the fixed v1 3-commit/1-no-action trace."""

    action_available = record.get("action_available") is True
    reward_available = record.get("reward_available") is True
    no_action = record.get("no_action") is True
    plant = record.get("plant")
    plant_committed = type(plant) is dict and plant.get("committed") is True
    if (
        action_available != expected_connected
        or reward_available != expected_connected
        or no_action == expected_connected
        or plant_committed != expected_connected
    ):
        raise ValueError(f"{arm} did not realize the fixed v1 connected/no-action outcome")
    if not expected_connected and (
        type(record.get("executed_action")) is not int
        or record.get("executed_action") != NO_ACTION_SENTINEL
        or not _strict_json_equal(
            record.get("reward"),
            UNAVAILABLE_REWARD_SENTINEL,
        )
    ):
        raise ValueError(f"{arm} did not preserve unavailable-value sentinels")


def _assessment_payload() -> dict[str, object]:
    """Return the fixed nonassessment/authority surface for every report."""

    return {
        "status": ASSESSMENT_STATUS,
        "winner": None,
        "verdict": None,
        "thresholds": [],
        "performance_claimed": False,
        "adaptation_efficacy_claimed": False,
        "safety_claimed": False,
        "delight_assessed": False,
        "kondo_gate_forward_admission_intent_assessed": False,
        "kondo_sparse_actor_backward_inclusion_assessed": False,
        "historical_gradientjoy_compatibility_names_assessed": False,
        "evidence_authority": EVIDENCE_AUTHORITY,
        "promotion_authority": PROMOTION_AUTHORITY,
        "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
    }


def _validate_clean_v1_initial_causal_surface(
    *,
    pending_available: tuple[bool, bool],
    commit_available: tuple[bool, bool],
    plant_counts: tuple[int, int],
    prototype_steps: tuple[list[object], list[object]],
    oak_steps: tuple[list[object], list[object]],
    adapter_receipt_clocks: tuple[list[object], list[object]],
    adapter_settled: tuple[bool, bool],
    maximum_plant_transitions: tuple[int, int],
    connected_opportunities: int,
) -> tuple[int, int]:
    """Validate the fixed v1 zero-clock start and return remaining capacities."""

    if any(pending_available):
        raise ValueError("paired initial states must not contain pending receipts")
    if any(commit_available):
        raise ValueError("paired initial states must not contain prior commit records")
    word_clocks = (*prototype_steps, *oak_steps, *adapter_receipt_clocks)
    if (
        any(count != 0 for count in plant_counts)
        or any(words != [0, 0] for words in word_clocks)
        or any(adapter_settled)
    ):
        raise ValueError("v1 paired initial causal clocks and ledgers must be clean")
    remaining = tuple(
        maximum - current
        for maximum, current in zip(
            maximum_plant_transitions,
            plant_counts,
            strict=True,
        )
    )
    if any(value < connected_opportunities for value in remaining):
        raise ValueError("remaining plant capacity cannot cover connected opportunities")
    return cast(tuple[int, int], remaining)


class PrototypeEmbodiedPairedDevelopmentBenchmark:
    """Strict host runner around two actual embodied development harnesses."""

    def __init__(
        self,
        config: PrototypeEmbodiedPairedDevelopmentConfig,
        *,
        adaptive_harness: PrototypeEmbodiedDevelopmentHarness,
        adaptive_initial_state: PrototypeEmbodiedDevelopmentHarnessState,
        zero_step_harness: PrototypeEmbodiedDevelopmentHarness,
        zero_step_initial_state: PrototypeEmbodiedDevelopmentHarnessState,
        common_preparation_template: PrototypeEmbodiedDevelopmentHarnessPreparationInput,
    ) -> None:
        if type(config) is not PrototypeEmbodiedPairedDevelopmentConfig:
            raise TypeError("config must be an exact paired development config")
        if type(adaptive_harness) is not PrototypeEmbodiedDevelopmentHarness:
            raise TypeError("adaptive_harness must be an exact development harness")
        if type(zero_step_harness) is not PrototypeEmbodiedDevelopmentHarness:
            raise TypeError("zero_step_harness must be an exact development harness")
        if type(adaptive_initial_state) is not PrototypeEmbodiedDevelopmentHarnessState:
            raise TypeError("adaptive_initial_state has the wrong exact type")
        if type(zero_step_initial_state) is not PrototypeEmbodiedDevelopmentHarnessState:
            raise TypeError("zero_step_initial_state has the wrong exact type")
        if (
            type(common_preparation_template)
            is not PrototypeEmbodiedDevelopmentHarnessPreparationInput
        ):
            raise TypeError("common_preparation_template has the wrong exact type")
        self.config = config
        self.adaptive_harness = adaptive_harness
        self.zero_step_harness = zero_step_harness
        self._adaptive_initial = adaptive_initial_state
        self._zero_initial = zero_step_initial_state
        self._template = common_preparation_template
        self._initial_match = self._validate_initial_pair()
        self._binding = self._binding_payload()
        self._binding_sha256 = _canonical_sha256(self._binding)
        self._initial_chain_heads = (
            self._initial_chain_head("adaptive_stomp", adaptive_initial_state),
            self._initial_chain_head(
                "zero_stomp_step_size_control", zero_step_initial_state
            ),
        )

    def _validate_initial_pair(self) -> dict[str, object]:
        if not _host_bool(self.adaptive_harness.state_valid(self._adaptive_initial)):
            raise ValueError("adaptive initial harness state is invalid")
        if not _host_bool(self.zero_step_harness.state_valid(self._zero_initial)):
            raise ValueError("zero-step initial harness state is invalid")
        adaptive_config = self.adaptive_harness.to_config()
        control_config = self.zero_step_harness.to_config()
        adaptive_stomp = _stomp_config(adaptive_config)
        control_stomp = _stomp_config(control_config)
        adaptive_values: dict[str, float] = {}
        control_values: dict[str, float] = {}
        for name in PERMITTED_STOMP_CONFIG_DIFFERENCES:
            adaptive = adaptive_stomp.get(name)
            control = control_stomp.get(name)
            if type(adaptive) not in {int, float} or type(control) not in {int, float}:
                raise ValueError(f"paired STOMP field {name} must be numeric")
            adaptive_value = float(cast(float, adaptive))
            control_value = float(cast(float, control))
            if not math.isfinite(adaptive_value) or adaptive_value <= 0.0:
                raise ValueError(f"adaptive STOMP field {name} must be finite and positive")
            if control_value != 0.0:
                raise ValueError(f"control STOMP field {name} must be exactly zero")
            adaptive_values[name] = adaptive_value
            control_values[name] = control_value
        if _canonical_json_bytes(
            _normalized_harness_config(adaptive_config)
        ) != _canonical_json_bytes(_normalized_harness_config(control_config)):
            raise ValueError(
                "paired harness configs differ outside the five permitted STOMP fields"
            )

        adaptive_semantic = self._adaptive_initial.adapter.semantic
        control_semantic = self._zero_initial.adapter.semantic
        adaptive_steps = _base_optimizer_step_sizes(adaptive_semantic)
        control_steps = _base_optimizer_step_sizes(control_semantic)
        if len(adaptive_steps) != len(control_steps):
            raise ValueError("paired base optimizer layouts differ")
        expected_adaptive = np.float32(adaptive_values["base_step_size"])
        if any(np.float32(_host_float(value)) != expected_adaptive for value in adaptive_steps):
            raise ValueError("adaptive LMS step-size leaves do not match base_step_size")
        if any(_host_float(value) != 0.0 for value in control_steps):
            raise ValueError("control LMS step-size leaves are not exactly zero")
        if not _tree_bits_equal(
            _normalize_base_optimizer_step_sizes(adaptive_semantic),
            _normalize_base_optimizer_step_sizes(control_semantic),
        ):
            raise ValueError(
                "initial semantic states differ outside base LMS step-size scalar leaves"
            )

        exact_pairs = {
            "plant": _tree_bits_equal(self._adaptive_initial.plant, self._zero_initial.plant),
            "grounded_shadow": _tree_bits_equal(
                self._adaptive_initial.shadow, self._zero_initial.shadow
            ),
            "hard_envelope": _tree_bits_equal(
                self._adaptive_initial.adapter.envelope,
                self._zero_initial.adapter.envelope,
            ),
            "adapter_causal_ledger": _tree_bits_equal(
                (
                    self._adaptive_initial.adapter.receipt_clock_words,
                    self._adaptive_initial.adapter.has_settled_prototype_decision,
                    self._adaptive_initial.adapter.last_settled_prototype_decision_id,
                    self._adaptive_initial.adapter.pending,
                ),
                (
                    self._zero_initial.adapter.receipt_clock_words,
                    self._zero_initial.adapter.has_settled_prototype_decision,
                    self._zero_initial.adapter.last_settled_prototype_decision_id,
                    self._zero_initial.adapter.pending,
                ),
            ),
            "harness_pending": _tree_bits_equal(
                self._adaptive_initial.pending,
                self._zero_initial.pending,
            ),
            "last_commit": _tree_bits_equal(
                self._adaptive_initial.last_commit,
                self._zero_initial.last_commit,
            ),
        }
        if not all(exact_pairs.values()):
            raise ValueError("paired initial causal state differs outside permitted LMS leaves")
        adaptive_budget = self.adaptive_harness.resource_budget(self._adaptive_initial)
        control_budget = self.zero_step_harness.resource_budget(self._zero_initial)
        matched_budget_fields = (
            "persistent_state_nbytes",
            "plant_observation_dim",
            "primitive_actions",
            "maximum_plant_transitions",
            "pending_receipts",
            "last_commit_records",
            "maximum_shadow_planner_calls_per_prepare",
            "maximum_shadow_autodiff_passes_per_prepare",
            "maximum_shadow_rng_draws_per_prepare",
        )
        if any(
            getattr(adaptive_budget, name) != getattr(control_budget, name)
            for name in matched_budget_fields
        ):
            raise ValueError("paired initial shape/capacity resource budgets differ")
        connected_opportunities = self.config.attempts - len(
            self.config.bridge_disconnect_attempts
        )
        initial_counts = {
            "adaptive_stomp": _host_int(self._adaptive_initial.plant.transition_count),
            "zero_stomp_step_size_control": _host_int(
                self._zero_initial.plant.transition_count
            ),
        }
        prototype_steps = {
            "adaptive_stomp": _host_list(
                self._adaptive_initial.adapter.semantic.composition.prototype.step_words
            ),
            "zero_stomp_step_size_control": _host_list(
                self._zero_initial.adapter.semantic.composition.prototype.step_words
            ),
        }
        oak_steps = {
            "adaptive_stomp": _host_list(
                cast(
                    OaKState,
                    self._adaptive_initial.adapter.semantic.composition.prototype.oak_state,
                ).step_words
            ),
            "zero_stomp_step_size_control": _host_list(
                cast(
                    OaKState,
                    self._zero_initial.adapter.semantic.composition.prototype.oak_state,
                ).step_words
            ),
        }
        adapter_receipt_clocks = {
            "adaptive_stomp": _host_list(
                self._adaptive_initial.adapter.receipt_clock_words
            ),
            "zero_stomp_step_size_control": _host_list(
                self._zero_initial.adapter.receipt_clock_words
            ),
        }
        adapter_settled = {
            "adaptive_stomp": _host_bool(
                self._adaptive_initial.adapter.has_settled_prototype_decision
            ),
            "zero_stomp_step_size_control": _host_bool(
                self._zero_initial.adapter.has_settled_prototype_decision
            ),
        }
        remaining_values = _validate_clean_v1_initial_causal_surface(
            pending_available=(
                _host_bool(self._adaptive_initial.pending.available),
                _host_bool(self._zero_initial.pending.available),
            ),
            commit_available=(
                _host_bool(self._adaptive_initial.last_commit.available),
                _host_bool(self._zero_initial.last_commit.available),
            ),
            plant_counts=tuple(initial_counts.values()),
            prototype_steps=cast(
                tuple[list[object], list[object]], tuple(prototype_steps.values())
            ),
            oak_steps=cast(tuple[list[object], list[object]], tuple(oak_steps.values())),
            adapter_receipt_clocks=cast(
                tuple[list[object], list[object]],
                tuple(adapter_receipt_clocks.values()),
            ),
            adapter_settled=tuple(adapter_settled.values()),
            maximum_plant_transitions=(
                adaptive_budget.maximum_plant_transitions,
                control_budget.maximum_plant_transitions,
            ),
            connected_opportunities=connected_opportunities,
        )
        remaining_capacities = dict(zip(ARM_ORDER, remaining_values, strict=True))

        return {
            "matched": True,
            "permitted_stomp_config_fields": list(PERMITTED_STOMP_CONFIG_DIFFERENCES),
            "permitted_config_difference_count": len(
                PERMITTED_STOMP_CONFIG_DIFFERENCES
            ),
            "adaptive_values": adaptive_values,
            "zero_step_control_values": control_values,
            "normalized_harness_configs_bit_equal": True,
            "materialized_base_lms_step_size_scalar_leaf_count": len(adaptive_steps),
            "semantic_states_bit_equal_after_base_lms_step_size_normalization": True,
            "all_other_initial_semantic_arrays_rng_caches_observations_traces_clocks_equal": (
                True
            ),
            "exact_initial_state_pairs": exact_pairs,
            "initial_shapes_and_capacities_equal": True,
            "clean_initial_pending_and_commit_records": True,
            "zero_initial_prototype_oak_adapter_and_plant_clocks": True,
            "remaining_plant_capacities": remaining_capacities,
            "outer_config_digests_and_integrity_checksums_expected_to_differ": True,
        }

    def _binding_payload(self) -> dict[str, object]:
        return {
            "schema": "alberta.prototype-embodied-paired-development.binding.v1",
            "benchmark_config": self.config.to_config(),
            "adaptive_harness_config": self.adaptive_harness.to_config(),
            "zero_step_harness_config": self.zero_step_harness.to_config(),
            "common_preparation_template_sha256": _tree_sha256(self._template),
            "adaptive_initial_state_sha256": _tree_sha256(self._adaptive_initial),
            "zero_step_initial_state_sha256": _tree_sha256(self._zero_initial),
            "initial_match_receipt": self._initial_match,
            "source_manifest": prototype_embodied_paired_source_manifest(),
            "runtime_identity": prototype_embodied_paired_runtime_identity(),
        }

    def _initial_chain_head(
        self,
        arm: AdaptiveArm,
        state: PrototypeEmbodiedDevelopmentHarnessState,
    ) -> str:
        return _canonical_sha256(
            {
                "arm": arm,
                "initial_state_sha256": _tree_sha256(state),
                "binding_sha256": _canonical_sha256(self._binding_payload()),
            }
        )

    def common_preparation(
        self,
        attempt_index: int,
    ) -> PrototypeEmbodiedDevelopmentHarnessPreparationInput:
        """Return one bit-identical exogenous input object for both arms."""

        if type(attempt_index) is not int or not 0 <= attempt_index < self.config.attempts:
            raise ValueError("attempt_index is outside the configured schedule")
        identity = attempt_index + 1
        tick_base = 10 * identity
        telemetry = self._template.envelope.telemetry.replace(
            bridge_connected=jnp.asarray(
                attempt_index not in self.config.bridge_disconnect_attempts,
                dtype=jnp.bool_,
            ),
            telemetry_id=_words(identity),
            sample_tick=_words(tick_base),
        )
        key = jr.fold_in(
            jr.key(self.config.development_key, impl="threefry2x32"),
            np.uint32(attempt_index),
        )
        untrusted_reward = jr.uniform(
            key,
            (),
            minval=jnp.float32(-1.0),
            maxval=jnp.float32(1.0),
            dtype=jnp.float32,
        )
        envelope = self._template.envelope.replace(
            telemetry=telemetry,
            envelope_decision_id=_words(identity),
            envelope_action_id=_words(identity),
            control_tick=_words(tick_base + 2),
            control_deadline_tick=_words(tick_base + 5),
            untrusted_reward=untrusted_reward,
        )
        return self._template.replace(envelope=envelope)

    def common_schedule_payload(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for index in range(self.config.attempts):
            preparation = self.common_preparation(index)
            result.append(
                {
                    "attempt_index": index,
                    "development_key": self.config.development_key,
                    "bridge_connected": _host_bool(
                        preparation.envelope.telemetry.bridge_connected
                    ),
                    "telemetry_id": _host_list(
                        preparation.envelope.telemetry.telemetry_id
                    ),
                    "sample_tick": _host_list(preparation.envelope.telemetry.sample_tick),
                    "control_tick": _host_list(preparation.envelope.control_tick),
                    "control_deadline_tick": _host_list(
                        preparation.envelope.control_deadline_tick
                    ),
                    "preparation_sha256": _tree_sha256(preparation),
                    "model_state_sha256": _tree_sha256(preparation.model_state),
                }
            )
        return result

    def _state_body(self, state: PrototypeEmbodiedPairedRunState) -> dict[str, object]:
        return {
            "attempt_index": state.attempt_index,
            "adaptive_state_sha256": _tree_sha256(state.adaptive_stomp),
            "zero_step_state_sha256": _tree_sha256(
                state.zero_stomp_step_size_control
            ),
            "chain_heads": list(state.chain_heads),
            "records_json": list(state.records_json),
            "binding_sha256": state.binding_sha256,
        }

    def _live_source_runtime_matches_binding(self) -> bool:
        return (
            _canonical_sha256(prototype_embodied_paired_source_manifest())
            == _canonical_sha256(self._binding["source_manifest"])
            and _canonical_sha256(prototype_embodied_paired_runtime_identity())
            == _canonical_sha256(self._binding["runtime_identity"])
        )

    def _seal_state(
        self,
        state: PrototypeEmbodiedPairedRunState,
    ) -> PrototypeEmbodiedPairedRunState:
        candidate = dataclasses.replace(state, integrity_sha256="")
        return dataclasses.replace(
            candidate,
            integrity_sha256=_canonical_sha256(self._state_body(candidate)),
        )

    def init(self) -> PrototypeEmbodiedPairedRunState:
        state = PrototypeEmbodiedPairedRunState(
            attempt_index=0,
            adaptive_stomp=self._adaptive_initial,
            zero_stomp_step_size_control=self._zero_initial,
            chain_heads=self._initial_chain_heads,
            records_json=(),
            binding_sha256=self._binding_sha256,
            integrity_sha256="",
        )
        return self._seal_state(state)

    def _arm_record(
        self,
        *,
        before: PrototypeEmbodiedDevelopmentHarnessState,
        result: PrototypeEmbodiedDevelopmentHarnessSettlementResult,
    ) -> dict[str, object]:
        after = result.state
        envelope = result.envelope
        transition = result.transition
        action_available = _host_bool(result.diagnostics.action_transaction_committed)
        reward_available = _host_bool(transition.committed)
        reward = _host_float(transition.reward) if reward_available else UNAVAILABLE_REWARD_SENTINEL
        action = _host_int(result.action) if action_available else NO_ACTION_SENTINEL
        prototype = after.adapter.semantic.composition.prototype
        oak = cast(OaKState, prototype.oak_state)
        return {
            "selected_action": _host_int(before.pending.selected_action),
            "action_available": action_available,
            "executed_action": action,
            "no_action_sentinel": NO_ACTION_SENTINEL,
            "reward_available": reward_available,
            "reward": reward,
            "unavailable_reward_sentinel": UNAVAILABLE_REWARD_SENTINEL,
            "fallback_used": _host_bool(envelope.fallback_used),
            "proposed_accepted": _host_bool(envelope.proposed_accepted),
            "no_action": not action_available,
            "envelope": {
                "transaction_applied": _host_bool(envelope.transaction_applied),
                "action_available": _host_bool(envelope.action_available),
                "fallback_certified": _host_bool(envelope.fallback_certified),
                "unavailable_reason": _host_int(envelope.unavailable_reason),
                "hard_violation": _host_bool(envelope.hard_violation),
            },
            "plant": {
                "proposal_requested": _host_bool(transition.requested),
                "proposal_applied": _host_bool(transition.proposal_applied),
                "committed": reward_available,
                "pre_transition_count": _host_int(before.plant.transition_count),
                "post_transition_count": _host_int(after.plant.transition_count),
                "pre_observation": _host_list(transition.pre_observation),
                "post_observation": _host_list(transition.post_observation),
                "terminated": _host_bool(transition.terminated),
                "truncated": _host_bool(transition.truncated),
            },
            "clocks": {
                "prototype_step_words": _host_list(prototype.step_words),
                "oak_step_words": _host_list(oak.step_words),
                "plant_transition_count": _host_int(after.plant.transition_count),
                "prototype_decision_id": _host_list(prototype.current_decision_id),
            },
            "semantic": {
                "transition_requested": _host_bool(
                    result.diagnostics.semantic_transition_requested
                ),
                "transition_committed": _host_bool(
                    result.diagnostics.semantic_transition_committed
                ),
                "successor_rearmed": _host_bool(
                    result.diagnostics.semantic_successor_rearmed
                ),
                "prototype_learning_updates_adopted": _host_int(
                    result.diagnostics.prototype_learning_updates_adopted
                ),
            },
            "shadow": {
                "adopted": _host_bool(result.diagnostics.shadow_adopted),
                "content_matches_receipt": _host_bool(
                    result.diagnostics.shadow_result_content_matches_receipt
                ),
                "state_valid": _host_bool(
                    result.diagnostics.shadow_result_state_valid
                ),
                "authenticated": _host_bool(
                    result.diagnostics.shadow_result_authenticated
                ),
                "backward_evaluations": _host_int(
                    result.diagnostics.shadow_backward_evaluations_per_prepare
                ),
            },
            "logical_work": {
                "adapter_prepare_calls": 1,
                "envelope_evaluations": 1,
                "adapter_settlements": 1,
                "plant_proposals_requested": int(_host_bool(transition.requested)),
                "semantic_transition_calls": int(
                    _host_bool(result.diagnostics.semantic_transition_requested)
                ),
                "prototype_learning_updates_adopted": _host_int(
                    result.diagnostics.prototype_learning_updates_adopted
                ),
                "shadow_steps": 1,
            },
            "transaction_committed": _host_bool(
                result.diagnostics.transaction_committed
            ),
            "step_size_governed_parameters_before_sha256": _tree_sha256(
                _step_size_governed_parameter_tree(before)
            ),
            "step_size_governed_parameters_after_sha256": _tree_sha256(
                _step_size_governed_parameter_tree(after)
            ),
            "state_before_sha256": _tree_sha256(before),
            "state_after_sha256": _tree_sha256(after),
        }

    def _advance_arm(
        self,
        harness: PrototypeEmbodiedDevelopmentHarness,
        state: PrototypeEmbodiedDevelopmentHarnessState,
        preparation: PrototypeEmbodiedDevelopmentHarnessPreparationInput,
    ) -> tuple[PrototypeEmbodiedDevelopmentHarnessState, dict[str, object]]:
        prepared = harness.prepare(state, preparation)
        if not _host_bool(prepared.diagnostics.prepared):
            raise ValueError("paired harness preparation failed closed")
        envelope = harness.evaluate_pending_envelope(prepared.state)
        settled = harness.settle(prepared.state, envelope, prepared.shadow)
        if not _host_bool(settled.diagnostics.transaction_committed):
            raise ValueError("paired harness settlement failed closed")
        if not _host_bool(harness.state_valid(settled.state)):
            raise ValueError("paired harness produced an invalid successor")
        return settled.state, self._arm_record(before=prepared.state, result=settled)

    def _advance_structurally_valid_once(
        self,
        state: PrototypeEmbodiedPairedRunState,
    ) -> PrototypeEmbodiedPairedRunState:
        if not self._state_structurally_valid(state):
            raise ValueError("cannot advance an invalid paired run state")
        if state.attempt_index >= self.config.attempts:
            raise ValueError("paired development schedule is already complete")
        preparation = self.common_preparation(state.attempt_index)
        adaptive, adaptive_record = self._advance_arm(
            self.adaptive_harness,
            state.adaptive_stomp,
            preparation,
        )
        control, control_record = self._advance_arm(
            self.zero_step_harness,
            state.zero_stomp_step_size_control,
            preparation,
        )
        record = {
            "attempt_index": state.attempt_index,
            "common_preparation_sha256": _tree_sha256(preparation),
            "arms": {
                "adaptive_stomp": adaptive_record,
                "zero_stomp_step_size_control": control_record,
            },
        }
        arm_records = cast(dict[str, dict[str, object]], record["arms"])
        expected_connected = (
            state.attempt_index not in self.config.bridge_disconnect_attempts
        )
        for arm in ARM_ORDER:
            _validate_fixed_v1_arm_record(
                arm,
                arm_records[arm],
                expected_connected=expected_connected,
            )
        chain_heads = tuple(
            _canonical_sha256(
                {
                    "previous": state.chain_heads[index],
                    "attempt_index": state.attempt_index,
                    "arm": arm,
                    "record": arm_records[arm],
                    "common_preparation_sha256": record[
                        "common_preparation_sha256"
                    ],
                }
            )
            for index, arm in enumerate(ARM_ORDER)
        )
        candidate = PrototypeEmbodiedPairedRunState(
            attempt_index=state.attempt_index + 1,
            adaptive_stomp=adaptive,
            zero_stomp_step_size_control=control,
            chain_heads=cast(tuple[str, str], chain_heads),
            records_json=(*state.records_json, _canonical_json_bytes(record).decode("utf-8")),
            binding_sha256=self._binding_sha256,
            integrity_sha256="",
        )
        return self._seal_state(candidate)

    def _state_structurally_valid(self, state: object) -> bool:
        """Check local shape, binding, child, record, and hash-chain integrity only."""

        if type(state) is not PrototypeEmbodiedPairedRunState:
            return False
        candidate = state
        if (
            not self._live_source_runtime_matches_binding()
            or
            type(candidate.attempt_index) is not int
            or not 0 <= candidate.attempt_index <= self.config.attempts
            or len(candidate.records_json) != candidate.attempt_index
            or candidate.binding_sha256 != self._binding_sha256
            or candidate.integrity_sha256
            != _canonical_sha256(
                self._state_body(dataclasses.replace(candidate, integrity_sha256=""))
            )
        ):
            return False
        if not _host_bool(self.adaptive_harness.state_valid(candidate.adaptive_stomp)):
            return False
        if not _host_bool(
            self.zero_step_harness.state_valid(candidate.zero_stomp_step_size_control)
        ):
            return False
        if _host_bool(candidate.adaptive_stomp.pending.available) or _host_bool(
            candidate.zero_stomp_step_size_control.pending.available
        ):
            return False
        heads = list(self._initial_chain_heads)
        for index, record_json in enumerate(candidate.records_json):
            try:
                record = json.loads(record_json)
            except (json.JSONDecodeError, TypeError):
                return False
            if _canonical_json_bytes(record).decode("utf-8") != record_json:
                return False
            if record.get("attempt_index") != index:
                return False
            arms = record.get("arms")
            if type(arms) is not dict or set(arms) != set(ARM_ORDER):
                return False
            for arm_index, arm in enumerate(ARM_ORDER):
                heads[arm_index] = _canonical_sha256(
                    {
                        "previous": heads[arm_index],
                        "attempt_index": index,
                        "arm": arm,
                        "record": arms[arm],
                        "common_preparation_sha256": record.get(
                            "common_preparation_sha256"
                        ),
                    }
                )
        return tuple(heads) == candidate.chain_heads

    def _reconstruct_prefix(self, attempt_index: int) -> PrototypeEmbodiedPairedRunState:
        """Recompute one fixed prefix without trusting or recursively validating a caller state."""

        if type(attempt_index) is not int or not 0 <= attempt_index <= self.config.attempts:
            raise ValueError("attempt_index is outside the fixed v1 schedule")
        current = self.init()
        while current.attempt_index < attempt_index:
            current = self._advance_structurally_valid_once(current)
        return current

    def validate_state(self, state: object) -> bool:
        """Accept only a structurally valid state exactly reconstructed from bound initials."""

        try:
            if not self._state_structurally_valid(state):
                return False
            candidate = cast(PrototypeEmbodiedPairedRunState, state)
            reconstructed = self._reconstruct_prefix(candidate.attempt_index)
        except Exception:
            return False
        return self._run_states_equal(candidate, reconstructed)

    def run(
        self,
        state: PrototypeEmbodiedPairedRunState | None = None,
        *,
        stop_after: int | None = None,
    ) -> PrototypeEmbodiedPairedRunState:
        if state is None:
            current = self.init()
            if not self._state_structurally_valid(current):
                raise ValueError("bound initial run state is invalid")
        else:
            current = state
            if not self.validate_state(current):
                raise ValueError("run source state is invalid")
        target = self.config.attempts if stop_after is None else stop_after
        if type(target) is not int or not current.attempt_index <= target <= self.config.attempts:
            raise ValueError("stop_after must be an exact reachable schedule prefix")
        while current.attempt_index < target:
            current = self._advance_structurally_valid_once(current)
        return current

    def _records(self, state: PrototypeEmbodiedPairedRunState) -> list[dict[str, object]]:
        return [cast(dict[str, object], json.loads(item)) for item in state.records_json]

    def _arm_summary(
        self,
        records: list[dict[str, object]],
        arm: AdaptiveArm,
        final_state: PrototypeEmbodiedDevelopmentHarnessState,
    ) -> dict[str, object]:
        arm_records = [
            cast(dict[str, object], cast(dict[str, object], record["arms"])[arm])
            for record in records
        ]
        rewards = [
            float(record["reward"])
            for record in arm_records
            if record["reward_available"] is True
        ]
        cumulative = [0.0]
        total = 0.0
        for record in arm_records:
            if record["reward_available"] is True:
                total += float(record["reward"])
            cumulative.append(total)
        actions = [
            int(cast(int, record["executed_action"]))
            for record in arm_records
            if record["action_available"] is True
        ]
        prototype = final_state.adapter.semantic.composition.prototype
        oak = cast(OaKState, prototype.oak_state)
        return {
            "attempts": len(arm_records),
            "committed_plant_transitions": len(rewards),
            "lifetime_reward_sum": float(sum(rewards)),
            "mean_reward_per_committed_plant_transition": (
                float(sum(rewards) / len(rewards)) if rewards else None
            ),
            "normalized_lifetime_reward_curve_auc_over_committed_transition_index": (
                _normalized_trapezoid(rewards)
            ),
            "normalized_lifetime_cumulative_reward_curve_auc_over_attempt_index": (
                _normalized_trapezoid(cumulative)
            ),
            "executed_actions": actions,
            "fallback_count": sum(bool(record["fallback_used"]) for record in arm_records),
            "no_action_count": sum(bool(record["no_action"]) for record in arm_records),
            "prototype_learning_updates_adopted": sum(
                int(
                    cast(
                        int,
                        cast(dict[str, object], record["semantic"])[
                            "prototype_learning_updates_adopted"
                        ],
                    )
                )
                for record in arm_records
            ),
            "semantic_successor_rearm_count": sum(
                bool(cast(dict[str, object], record["semantic"])["successor_rearmed"])
                for record in arm_records
            ),
            "shadow_adoption_count": sum(
                bool(cast(dict[str, object], record["shadow"])["adopted"])
                for record in arm_records
            ),
            "shadow_backward_evaluations": sum(
                int(
                    cast(
                        int,
                        cast(dict[str, object], record["shadow"])[
                            "backward_evaluations"
                        ],
                    )
                )
                for record in arm_records
            ),
            "envelope_decisions": sum(
                bool(cast(dict[str, object], record["envelope"])["transaction_applied"])
                for record in arm_records
            ),
            "final_prototype_step_words": _host_list(prototype.step_words),
            "final_oak_step_words": _host_list(oak.step_words),
            "final_plant_transition_count": _host_int(final_state.plant.transition_count),
        }

    def _resource_payload(
        self,
        state: PrototypeEmbodiedPairedRunState,
        records: list[dict[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "accounting_scope": "logical_calls_and_persistent_array_bytes",
            "compute_time_latency_and_peak_temporary_memory": "not_measured",
            "physical_dispatch_count": PHYSICAL_DISPATCH_COUNT,
            "output_writes": 0,
            "per_arm": {},
        }
        per_arm = cast(dict[str, object], result["per_arm"])
        states = {
            "adaptive_stomp": state.adaptive_stomp,
            "zero_stomp_step_size_control": state.zero_stomp_step_size_control,
        }
        harnesses = {
            "adaptive_stomp": self.adaptive_harness,
            "zero_stomp_step_size_control": self.zero_step_harness,
        }
        for arm in ARM_ORDER:
            budget = harnesses[arm].resource_budget(states[arm])
            arm_records = [
                cast(dict[str, object], cast(dict[str, object], record["arms"])[arm])
                for record in records
            ]
            logical: dict[str, int] = {}
            for name in (
                "adapter_prepare_calls",
                "envelope_evaluations",
                "adapter_settlements",
                "plant_proposals_requested",
                "semantic_transition_calls",
                "prototype_learning_updates_adopted",
                "shadow_steps",
            ):
                logical[name] = sum(
                    int(
                        cast(
                            int,
                            cast(dict[str, object], record["logical_work"])[name],
                        )
                    )
                    for record in arm_records
                )
            per_arm[arm] = {
                "final_harness_resource_budget": dataclasses.asdict(budget),
                "observed_logical_work": logical,
            }
        return result

    def report(self, state: PrototypeEmbodiedPairedRunState) -> dict[str, object]:
        """Build a complete source/runtime-bound, nonpromoting report."""

        if not self.validate_state(state) or state.attempt_index != self.config.attempts:
            raise ValueError("report requires one valid complete paired run")
        records = self._records(state)
        adaptive_summary = self._arm_summary(
            records, "adaptive_stomp", state.adaptive_stomp
        )
        control_summary = self._arm_summary(
            records,
            "zero_stomp_step_size_control",
            state.zero_stomp_step_size_control,
        )
        for arm, summary in (
            ("adaptive_stomp", adaptive_summary),
            ("zero_stomp_step_size_control", control_summary),
        ):
            if (
                summary["attempts"] != 4
                or summary["committed_plant_transitions"] != 3
                or summary["no_action_count"] != 1
            ):
                raise ValueError(f"{arm} does not satisfy the fixed v1 4/3/1 protocol")
        adaptive_parameter_changed = any(
            cast(dict[str, object], cast(dict[str, object], record["arms"])[
                "adaptive_stomp"
            ])["step_size_governed_parameters_before_sha256"]
            != cast(dict[str, object], cast(dict[str, object], record["arms"])[
                "adaptive_stomp"
            ])["step_size_governed_parameters_after_sha256"]
            for record in records
        )
        control_parameter_unchanged = all(
            cast(dict[str, object], cast(dict[str, object], record["arms"])[
                "zero_stomp_step_size_control"
            ])["step_size_governed_parameters_before_sha256"]
            == cast(dict[str, object], cast(dict[str, object], record["arms"])[
                "zero_stomp_step_size_control"
            ])["step_size_governed_parameters_after_sha256"]
            for record in records
        )
        trajectory_fields = (
            "selected_action",
            "action_available",
            "executed_action",
            "reward_available",
            "reward",
        )
        trajectories_diverged = False
        for record in records:
            arms = cast(dict[str, dict[str, object]], record["arms"])
            adaptive_record = arms["adaptive_stomp"]
            control_record = arms["zero_stomp_step_size_control"]
            field_difference = any(
                adaptive_record[name] != control_record[name]
                for name in trajectory_fields
            )
            adaptive_plant = cast(dict[str, object], adaptive_record["plant"])
            control_plant = cast(dict[str, object], control_record["plant"])
            trajectories_diverged = trajectories_diverged or field_difference or (
                adaptive_plant["post_observation"]
                != control_plant["post_observation"]
            )
        body: dict[str, object] = {
            "schema": PROTOTYPE_EMBODIED_PAIRED_REPORT_SCHEMA,
            "type": "PrototypeEmbodiedPairedDevelopmentReport",
            "assessment": _assessment_payload(),
            "protocol": self.config.to_config(),
            "arm_contract": {
                "adaptive_stomp": "five positive declared STOMP step sizes",
                "zero_stomp_step_size_control": (
                    "the same five STOMP step sizes are zero; no whole-agent freeze claim"
                ),
                "paired_inputs": "bit-identical exogenous preparation per attempt",
                "causal_ownership": "independent policy plant envelope and shadow state",
                "same_hard_envelope_and_grounded_shadow_opportunities": True,
                "equal_behavior_allowed": True,
            },
            "initial_match": self._initial_match,
            "common_schedule": self.common_schedule_payload(),
            "raw_records": records,
            "summaries": {
                "adaptive_stomp": adaptive_summary,
                "zero_stomp_step_size_control": control_summary,
                "descriptive_adaptive_minus_control": {
                    "lifetime_reward_sum": float(
                        cast(float, adaptive_summary["lifetime_reward_sum"])
                        - cast(float, control_summary["lifetime_reward_sum"])
                    ),
                    "verdict": None,
                },
            },
            "descriptive_metric_definition": {
                "normalized_auc": (
                    "trapezoidal area over equally spaced integer indices divided by "
                    "the index span"
                ),
                "reward_curve_x_axis": "committed plant transition index",
                "cumulative_reward_curve_x_axis": "attempt index including the initial zero",
                "inferential_or_adaptation_interpretation": False,
            },
            "parameter_witness": {
                "scope": "step_size_governed_real_owner_STOMP_parameter_surfaces",
                "adaptive_changed_at_least_once": adaptive_parameter_changed,
                "zero_step_control_unchanged_every_attempt": control_parameter_unchanged,
                "adaptive_change_required_for_validity": False,
                "excluded_mutable_surfaces": [
                    "option-model decay EMAs",
                    "semantic and consolidated memory",
                    "grounded shadow actor critic and model state",
                    "eligibility traces and causal caches",
                ],
            },
            "trajectory_relationship": {
                "independently_owned": True,
                "diverged_by_causal_policy_or_plant_trace": trajectories_diverged,
                "divergence_required_for_validity": False,
            },
            "resources": self._resource_payload(state, records),
            "bindings": {
                "binding_sha256": self._binding_sha256,
                "config_sha256": _canonical_sha256(self.config.to_config()),
                "source_manifest": self._binding["source_manifest"],
                "runtime_identity": self._binding["runtime_identity"],
                "source_scope": "selected mechanism files, not a transitive source lock",
                "runtime_scope": "selected host/JAX fields, not an XLA environment or binary lock",
                "initial_snapshot_sha256": _canonical_sha256(self._initial_match),
                "common_schedule_sha256": _canonical_sha256(
                    self.common_schedule_payload()
                ),
                "final_state_integrity_sha256": state.integrity_sha256,
                "chain_heads": list(state.chain_heads),
            },
            "replay_and_checkpoint": {
                "exact_replay_supported": True,
                "checkpoint_host_only": True,
                "checkpoint_restore_causally_reconstructs_prefix": True,
                "checkpoint_digest_authentication": False,
            },
            "outputs": {
                "writes": False,
                "path": None,
                "artifact_created": False,
            },
            "limitations": list(_LIMITATIONS),
        }
        return {**body, "report_sha256": _canonical_sha256(body)}

    def validate_report(self, report: object) -> PrototypeEmbodiedPairedValidationReceipt:
        """Fail closed against an exact deterministic full causal replay."""

        valid = False
        if isinstance(report, Mapping):
            raw = cast(Mapping[str, object], report)
            if set(raw) and "report_sha256" in raw:
                body = {key: raw[key] for key in raw if key != "report_sha256"}
                digest_valid = raw.get("report_sha256") == _canonical_sha256(body)
                if digest_valid:
                    expected = self.report(self.run())
                    valid = _strict_json_equal(dict(raw), expected)
        return PrototypeEmbodiedPairedValidationReceipt(
            valid=valid,
            assessment_status=ASSESSMENT_STATUS,
            source_runtime_bound=valid,
            exact_causal_replay=valid,
            output_written=False,
            physical_dispatch_count=PHYSICAL_DISPATCH_COUNT,
            evidence_authority=False,
            promotion_authority=False,
        )

    def checkpoint_payload(
        self,
        state: PrototypeEmbodiedPairedRunState,
    ) -> dict[str, object]:
        """Return a host-only checkpoint bound to deterministic prefix replay."""

        if not self.validate_state(state):
            raise ValueError("cannot checkpoint an invalid paired run state")
        state_sha256 = _canonical_sha256(self._state_body(state))
        body: dict[str, object] = {
            "schema": PROTOTYPE_EMBODIED_PAIRED_CHECKPOINT_SCHEMA,
            "type": "PrototypeEmbodiedPairedDevelopmentCheckpoint",
            "host_only": True,
            "attempt_index": state.attempt_index,
            "binding_sha256": self._binding_sha256,
            "config_sha256": _canonical_sha256(self.config.to_config()),
            "source_manifest_sha256": _canonical_sha256(
                self._binding["source_manifest"]
            ),
            "runtime_sha256": _canonical_sha256(
                self._binding["runtime_identity"]
            ),
            "common_schedule_prefix_sha256": _canonical_sha256(
                self.common_schedule_payload()[: state.attempt_index]
            ),
            "state_sha256": state_sha256,
            "assessment_status": ASSESSMENT_STATUS,
            "output_path": None,
            "physical_dispatch_count": PHYSICAL_DISPATCH_COUNT,
            "promotion_authority": False,
        }
        return {
            **body,
            "state": state,
            "checkpoint_sha256": _canonical_sha256(body),
        }

    def _run_states_equal(
        self,
        left: PrototypeEmbodiedPairedRunState,
        right: PrototypeEmbodiedPairedRunState,
    ) -> bool:
        return (
            left.attempt_index == right.attempt_index
            and _tree_bits_equal(left.adaptive_stomp, right.adaptive_stomp)
            and _tree_bits_equal(
                left.zero_stomp_step_size_control,
                right.zero_stomp_step_size_control,
            )
            and left.chain_heads == right.chain_heads
            and left.records_json == right.records_json
            and left.binding_sha256 == right.binding_sha256
            and left.integrity_sha256 == right.integrity_sha256
        )

    def restore_checkpoint(self, payload: object) -> PrototypeEmbodiedPairedRunState:
        """Restore only a prefix exactly reproduced from bound initial states."""

        if type(payload) is not dict:
            raise ValueError("paired development checkpoint must be an exact dict")
        raw = cast(dict[str, object], payload)
        expected_fields = {
            "schema",
            "type",
            "host_only",
            "attempt_index",
            "binding_sha256",
            "config_sha256",
            "source_manifest_sha256",
            "runtime_sha256",
            "common_schedule_prefix_sha256",
            "state_sha256",
            "assessment_status",
            "output_path",
            "physical_dispatch_count",
            "promotion_authority",
            "state",
            "checkpoint_sha256",
        }
        if set(raw) != expected_fields:
            raise ValueError("paired development checkpoint fields differ")
        if (
            raw["schema"] != PROTOTYPE_EMBODIED_PAIRED_CHECKPOINT_SCHEMA
            or raw["type"] != "PrototypeEmbodiedPairedDevelopmentCheckpoint"
            or raw["host_only"] is not True
            or raw["assessment_status"] != ASSESSMENT_STATUS
            or raw["output_path"] is not None
            or raw["physical_dispatch_count"] != 0
            or raw["promotion_authority"] is not False
        ):
            raise ValueError("paired development checkpoint authority/schema differs")
        attempt_index = raw["attempt_index"]
        if type(attempt_index) is not int or not 0 <= attempt_index <= self.config.attempts:
            raise ValueError("paired development checkpoint attempt index differs")
        state = raw["state"]
        if type(state) is not PrototypeEmbodiedPairedRunState:
            raise ValueError("paired development checkpoint state type differs")
        candidate = state
        body = {key: raw[key] for key in raw if key not in {"state", "checkpoint_sha256"}}
        expected_bindings = {
            "binding_sha256": self._binding_sha256,
            "config_sha256": _canonical_sha256(self.config.to_config()),
            "source_manifest_sha256": _canonical_sha256(
                self._binding["source_manifest"]
            ),
            "runtime_sha256": _canonical_sha256(
                self._binding["runtime_identity"]
            ),
            "common_schedule_prefix_sha256": _canonical_sha256(
                self.common_schedule_payload()[:attempt_index]
            ),
            "state_sha256": _canonical_sha256(self._state_body(candidate)),
        }
        if any(raw[name] != expected for name, expected in expected_bindings.items()):
            raise ValueError("paired development checkpoint binding differs")
        if raw["checkpoint_sha256"] != _canonical_sha256(body):
            raise ValueError("paired development checkpoint digest integrity failed")
        if not self.validate_state(candidate) or candidate.attempt_index != attempt_index:
            raise ValueError("paired development checkpoint state is invalid")
        reconstructed = self.run(stop_after=attempt_index)
        if not self._run_states_equal(candidate, reconstructed):
            raise ValueError("paired development checkpoint causal reconstruction differs")
        expected_payload = self.checkpoint_payload(reconstructed)
        for name in expected_fields - {"state"}:
            if not _strict_json_equal(raw[name], expected_payload[name]):
                raise ValueError("paired development checkpoint exact replay differs")
        return reconstructed


def build_prototype_embodied_paired_development_benchmark(
    *,
    development_key: int = 17,
) -> PrototypeEmbodiedPairedDevelopmentBenchmark:
    """Build the installed, deterministic, synthetic v1 development rig."""

    from alberta_framework.benchmarks.prototype_embodied_paired_development_rig import (
        build_prototype_embodied_paired_development_benchmark as build_rig,
    )

    return cast(
        PrototypeEmbodiedPairedDevelopmentBenchmark,
        build_rig(development_key=development_key),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the nonpromoting v1 diagnostic and emit only its in-memory JSON report."""

    parser = argparse.ArgumentParser(
        prog="alberta-prototype-embodied-paired-development",
        description=(
            "Run the fixed synthetic L0 paired embodied diagnostic; no artifact is written "
            "and no result has assessment, evidence, safety, or promotion authority."
        ),
    )
    parser.add_argument(
        "--development-key",
        type=int,
        default=17,
        help="Consumed uint32 development key (default: 17; never promotion-eligible).",
    )
    arguments = parser.parse_args(argv)
    benchmark = build_prototype_embodied_paired_development_benchmark(
        development_key=arguments.development_key
    )
    state = benchmark.run()
    report = benchmark.report(state)
    receipt = benchmark.validate_report(report)
    if not receipt.valid:
        raise RuntimeError("paired development report failed exact causal replay")
    sys.stdout.write(_canonical_json_bytes(report).decode("utf-8") + "\n")
    return 0


__all__ = [
    "ARM_ORDER",
    "ASSESSMENT_STATUS",
    "NO_ACTION_SENTINEL",
    "PERMITTED_STOMP_CONFIG_DIFFERENCES",
    "PROTOTYPE_EMBODIED_PAIRED_CHECKPOINT_SCHEMA",
    "PROTOTYPE_EMBODIED_PAIRED_CONFIG_SCHEMA",
    "PROTOTYPE_EMBODIED_PAIRED_REPORT_SCHEMA",
    "PrototypeEmbodiedPairedDevelopmentBenchmark",
    "PrototypeEmbodiedPairedDevelopmentConfig",
    "PrototypeEmbodiedPairedRunState",
    "PrototypeEmbodiedPairedValidationReceipt",
    "UNAVAILABLE_REWARD_SENTINEL",
    "build_prototype_embodied_paired_development_benchmark",
    "main",
    "prototype_embodied_paired_runtime_identity",
    "prototype_embodied_paired_source_manifest",
]


if __name__ == "__main__":  # pragma: no cover - exercised by the installed CLI
    raise SystemExit(main())
