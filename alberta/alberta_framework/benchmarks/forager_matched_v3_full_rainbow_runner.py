"""Non-authorizing Full Rainbow runner for the matched-v3 Forager task.

The exact production wrapper is present but requires an explicit unqualified
engineering flag.  The same state machine accepts dependency-injected kernels
and a tiny schedule for cheap boundary tests; such runs are never promoted to
runtime qualification or scientific evidence.
"""

from __future__ import annotations

import bisect
import copy
import hashlib
import hmac
import json
import math
import threading
import weakref
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, NoReturn, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jax import Array

from alberta_framework.benchmarks import forager_matched_v3_foragax_bridge as bridge
from alberta_framework.benchmarks import forager_matched_v3_full_rainbow as core

FULL_RAINBOW_RUNNER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.full_rainbow_runner.v1"
)
FULL_RAINBOW_ENGINEERING_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.full_rainbow_engineering_receipt.v1"
)
FULL_RAINBOW_RESULT_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.full_rainbow_result_receipt.v1"
)
FULL_RAINBOW_RUNNER_STATUS: Final = "implemented_unqualified"

BOUND_BRIDGE_DESCRIPTOR_SHA256: Final = (
    "1bf4f43bdf759a650e2f2662f8d5c86eb35d12eeb3a8399a3b5566b7bf8e45ab"
)
BOUND_BRIDGE_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_foragax_bridge.py"
)
BOUND_BRIDGE_IMPLEMENTATION_SHA256: Final = (
    "5aa304ee2ec185d038038fdd3e5cd093ecda85507ab7ee5e733ff1a47b21e362"
)
BOUND_CORE_CONFIG_SHA256: Final = (
    "835f02bdcf6844b7cd8c5e9fe33230a2a94f3a9c288c812cbfddf473c28b7e3f"
)
BOUND_CORE_DESCRIPTOR_SHA256: Final = (
    "5436200c47e1b003b0371c30606b52163b4c42427fa84e2fe2f4b2b2273ccae2"
)
BOUND_CORE_IMPLEMENTATION_SHA256: Final = (
    "7f75a0862ddc21160cea9c0a9faca221a0d757985fc90e5ef02b4673e3c14f5a"
)
BOUND_CORE_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_full_rainbow.py"
)

_MAX_CANONICAL_BYTES: Final = 2 * 1024 * 1024
_MAX_HORIZON: Final = 499_712
_MAX_REPLAY_CAPACITY: Final = 1_000_000
_MAX_BATCH_SIZE: Final = 32
_MAX_PAGE_SIZE: Final = 4_096
_PRIORITY_EPSILON: Final = 1e-10
_VALID_THREE_STEP_SCALED_RETURNS: Final = tuple(sorted({
    (first + 0.99 * second + 0.99**2 * third) / 30.0
    for first in (-1, 0, 1, 30)
    for second in (-1, 0, 1, 30)
    for third in (-1, 0, 1, 30)
}))


def _non_authorizing_claims() -> dict[str, bool]:
    return {
        "execution_ready": False,
        "execution_authorized": False,
        "runtime_qualified": False,
        "scientific_promotion_allowed": False,
        "performance_claim_allowed": False,
        "universal_sota_claim_allowed": False,
        "authority_granted": False,
    }


def _receipt_limitations() -> list[str]:
    return [
        "Receipt content and accounting do not independently prove execution.",
        "Receipt bytes grant no readiness, qualification, ingestion, or authority.",
        "Caller-supplied seed provenance is unverified; protected status is unknown.",
    ]


_AUTHORITY_FIELDS: Final = frozenset(
    {
        "authority_granted",
        "execution_authorized",
        "execution_ready",
        "ingestion_authorized",
        "performance_claim_allowed",
        "promotion_authorized",
        "runtime_qualified",
        "scientific_promotion_allowed",
        "universal_sota_claim_allowed",
    }
)


def _reject_authority_anywhere(value: object, *, label: str) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            mapping = cast(dict[str, object], item)
            for key, child in mapping.items():
                if key in _AUTHORITY_FIELDS and child is not False:
                    raise FullRainbowRunnerContractError(
                        f"{label} contains a non-false authority field {key}"
                    )
                pending.append(child)
        elif type(item) is list:
            pending.extend(cast(list[object], item))


def _source_sha256(module_file: object, expected_suffix: str) -> str:
    if type(module_file) is not str or not module_file.endswith(expected_suffix):
        raise RuntimeError(f"cannot resolve exact source path for {expected_suffix}")
    try:
        raw = Path(module_file).read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read exact source bytes for {expected_suffix}") from exc
    return hashlib.sha256(raw).hexdigest()

if bridge.FORAGAX_BRIDGE_DESCRIPTOR_SHA256 != BOUND_BRIDGE_DESCRIPTOR_SHA256:
    raise AssertionError("Full Rainbow runner bridge binding drifted")
if core.FULL_RAINBOW_CONFIG_SHA256 != BOUND_CORE_CONFIG_SHA256:
    raise AssertionError("Full Rainbow runner core configuration binding drifted")
if core.FULL_RAINBOW_DESCRIPTOR_SHA256 != BOUND_CORE_DESCRIPTOR_SHA256:
    raise AssertionError("Full Rainbow runner core descriptor binding drifted")
if not hmac.compare_digest(
    _source_sha256(bridge.__file__, BOUND_BRIDGE_IMPLEMENTATION_PATH),
    BOUND_BRIDGE_IMPLEMENTATION_SHA256,
):
    raise RuntimeError("Full Rainbow runner bridge implementation binding drifted")
if not hmac.compare_digest(
    _source_sha256(core.__file__, BOUND_CORE_IMPLEMENTATION_PATH),
    BOUND_CORE_IMPLEMENTATION_SHA256,
):
    raise RuntimeError("Full Rainbow runner core implementation binding drifted")


class FullRainbowRunnerContractError(ValueError):
    """A runner input, state transition, replay item, or receipt is invalid."""


class FullRainbowRunnerExecutionBlockedError(RuntimeError):
    """Execution was attempted without acknowledging unqualified engineering status."""


@dataclass(frozen=True, slots=True)
class FullRainbowRunnerSchedule:
    """Operational schedule; only one exact value is the production binding."""

    horizon: int
    replay_capacity: int
    batch_size: int
    minimum_replay_history: int
    update_period: int
    target_update_period: int
    page_size: int
    update_horizon: int = 3

    def __post_init__(self) -> None:
        positive = {
            "horizon": self.horizon,
            "replay_capacity": self.replay_capacity,
            "batch_size": self.batch_size,
            "update_period": self.update_period,
            "target_update_period": self.target_update_period,
            "page_size": self.page_size,
        }
        for name, value in positive.items():
            if type(value) is not int or value <= 0:
                raise FullRainbowRunnerContractError(
                    f"schedule {name} must be an exact positive integer"
                )
        if (
            type(self.minimum_replay_history) is not int
            or self.minimum_replay_history < 0
        ):
            raise FullRainbowRunnerContractError(
                "schedule minimum_replay_history must be an exact nonnegative integer"
            )
        if type(self.update_horizon) is not int or self.update_horizon != 3:
            raise FullRainbowRunnerContractError(
                "schedule update_horizon must remain exactly three"
            )
        if self.horizon > _MAX_HORIZON:
            raise FullRainbowRunnerContractError(
                "schedule horizon cannot exceed the exact production horizon"
            )
        if self.replay_capacity > _MAX_REPLAY_CAPACITY:
            raise FullRainbowRunnerContractError(
                "schedule replay_capacity cannot exceed the exact production capacity"
            )
        if self.batch_size > _MAX_BATCH_SIZE:
            raise FullRainbowRunnerContractError(
                "schedule batch_size cannot exceed the exact production batch"
            )
        if self.page_size > _MAX_PAGE_SIZE or self.page_size > self.replay_capacity:
            raise FullRainbowRunnerContractError(
                "schedule page_size exceeds the bounded replay layout"
            )
        if self.minimum_replay_history > _MAX_HORIZON:
            raise FullRainbowRunnerContractError(
                "schedule minimum_replay_history exceeds its bounded domain"
            )
        if self.update_period > _MAX_HORIZON or self.target_update_period > _MAX_HORIZON:
            raise FullRainbowRunnerContractError(
                "schedule periods exceed their bounded domain"
            )


def production_full_rainbow_schedule() -> FullRainbowRunnerSchedule:
    """Return the exact 499,712-interaction production schedule."""

    config = core.FullRainbowForagerConfig()
    return FullRainbowRunnerSchedule(
        horizon=config.horizon,
        update_horizon=config.update_horizon,
        replay_capacity=config.replay_capacity,
        batch_size=config.batch_size,
        minimum_replay_history=config.minimum_replay_history,
        update_period=config.update_period,
        target_update_period=config.target_update_period,
        page_size=4_096,
    )


def _first_due_transition(
    schedule: FullRainbowRunnerSchedule, period: int
) -> int | None:
    earliest = schedule.minimum_replay_history + schedule.update_horizon + 1
    first = ((earliest + period - 1) // period) * period
    return first if first <= schedule.horizon else None


def _count_periodic(first: int | None, stop: int, period: int) -> int:
    return 0 if first is None else ((stop - first) // period) + 1


def full_rainbow_schedule_accounting(
    schedule: FullRainbowRunnerSchedule,
) -> dict[str, int | None]:
    """Calculate every schedule boundary without allocating replay or running JAX."""

    if type(schedule) is not FullRainbowRunnerSchedule:
        raise FullRainbowRunnerContractError(
            "schedule must be an exact FullRainbowRunnerSchedule"
        )
    insertions = max(0, schedule.horizon - schedule.update_horizon)
    first_update = _first_due_transition(schedule, schedule.update_period)
    updates = _count_periodic(first_update, schedule.horizon, schedule.update_period)
    first_sync = _first_due_transition(schedule, schedule.target_update_period)
    syncs = _count_periodic(
        first_sync, schedule.horizon, schedule.target_update_period
    )
    return {
        "environment_interactions": schedule.horizon,
        "first_replay_insertion_transition": (
            schedule.update_horizon + 1 if insertions else None
        ),
        "replay_insertions": insertions,
        "maximum_replay_residency": min(insertions, schedule.replay_capacity),
        "replay_evictions": max(0, insertions - schedule.replay_capacity),
        "first_optimizer_update_transition": first_update,
        "optimizer_updates": updates,
        "first_target_sync_transition": first_sync,
        "target_syncs": syncs,
        "replay_samples": updates * schedule.batch_size,
        "priority_update_values": updates * schedule.batch_size,
    }


def _validate_observation(value: object) -> Array:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise FullRainbowRunnerContractError("observation must be an array") from exc
    if array.shape != (9, 9, 3) or array.dtype != np.dtype(np.float32):
        raise FullRainbowRunnerContractError(
            "observation must be exact float32 shape (9, 9, 3)"
        )
    if not bool(np.all(np.isfinite(array))):
        raise FullRainbowRunnerContractError("observation must be finite")
    if not bool(np.all((array == 0.0) | (array == 1.0))) or bool(
        np.any(np.sum(array, axis=-1) > 1.0)
    ):
        raise FullRainbowRunnerContractError(
            "observation must be binary zero-hot or one-hot per cell"
        )
    return jnp.asarray(value, dtype=jnp.float32)


def _validate_action(value: object) -> int:
    if type(value) is int:
        action = value
    else:
        array = np.asarray(value)
        if array.shape != () or array.dtype != np.dtype(np.int32):
            raise FullRainbowRunnerContractError(
                "action must be a Python int or exact int32 scalar"
            )
        action = int(array)
    if not 0 <= action < 4:
        raise FullRainbowRunnerContractError("action must lie in [0, 3]")
    return action


def _validate_raw_reward(value: object) -> int:
    if type(value) is not int or value not in (-1, 0, 1, 30):
        raise FullRainbowRunnerContractError(
            "raw reward must be an exact Python int in {-1, 0, 1, 30}"
        )
    return value


@dataclass(frozen=True, slots=True)
class _RawTransition:
    observation: Array
    action: int
    raw_reward: int
    transition: int


@dataclass(frozen=True, slots=True)
class FullRainbowAccumulatedTransition:
    """One source-faithful, nonterminal three-step replay element."""

    state: Array
    action: int
    next_state: Array
    scaled_n_step_reward: float
    bootstrap_discount: float
    source_transition: int
    available_after_transition: int


class FullRainbowThreeStepAccumulator:
    """Dopamine-compatible accumulator with deliberate one-transition latency."""

    def __init__(self, config: core.FullRainbowForagerConfig) -> None:
        if type(config) is not core.FullRainbowForagerConfig:
            raise FullRainbowRunnerContractError(
                "accumulator config must be an exact FullRainbowForagerConfig"
            )
        self._config = config
        self._trajectory: deque[_RawTransition] = deque(
            maxlen=config.update_horizon + config.stack_size
        )
        self._raw_transition_count = 0
        self._emitted_transition_count = 0

    @property
    def raw_transition_count(self) -> int:
        return self._raw_transition_count

    @property
    def emitted_transition_count(self) -> int:
        return self._emitted_transition_count

    def accumulate(
        self,
        *,
        observation: object,
        action: object,
        raw_reward: object,
        done: object,
        truncated: object,
    ) -> FullRainbowAccumulatedTransition | None:
        """Consume one raw transition and emit only after four are present."""

        if type(done) is not bool or type(truncated) is not bool or done or truncated:
            raise FullRainbowRunnerContractError(
                "accumulator accepts only continuing nonterminal transitions"
            )
        self._raw_transition_count += 1
        self._trajectory.append(
            _RawTransition(
                observation=_validate_observation(observation),
                action=_validate_action(action),
                raw_reward=_validate_raw_reward(raw_reward),
                transition=self._raw_transition_count,
            )
        )
        if len(self._trajectory) <= self._config.update_horizon:
            return None
        values = tuple(self._trajectory)
        first = values[0]
        next_observation = values[-1].observation
        accumulated = core.three_step_return(
            self._config,
            raw_rewards=tuple(
                value.raw_reward for value in values[: self._config.update_horizon]
            ),
            terminals=(False,) * self._config.update_horizon,
        )
        if accumulated.terminal or accumulated.bootstrap_discount == 0.0:
            raise FullRainbowRunnerContractError(
                "continuing accumulator unexpectedly produced a terminal return"
            )
        self._emitted_transition_count += 1
        return FullRainbowAccumulatedTransition(
            state=first.observation,
            action=first.action,
            next_state=next_observation,
            scaled_n_step_reward=accumulated.scaled_return,
            bootstrap_discount=accumulated.bootstrap_discount,
            source_transition=first.transition,
            available_after_transition=self._raw_transition_count,
        )


class _PrioritySumTree:
    def __init__(self, capacity: int) -> None:
        leaf_capacity = 1 << (capacity - 1).bit_length()
        self._leaf_capacity = leaf_capacity
        self._nodes = np.zeros((2 * leaf_capacity,), dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self._nodes[1])

    def get(self, index: int) -> float:
        return float(self._nodes[self._leaf_capacity + index])

    def set(self, index: int, value: float) -> None:
        node = self._leaf_capacity + index
        difference = value - self._nodes[node]
        if not math.isfinite(self.total + difference):
            raise FullRainbowRunnerContractError(
                "priority update would make the sum tree non-finite"
            )
        while node:
            self._nodes[node] += difference
            node //= 2

    def query(self, targets: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        nodes = np.ones(targets.shape, dtype=np.int64)
        remaining = targets.astype(np.float64, copy=True)
        while bool(np.any(nodes < self._leaf_capacity)):
            left = nodes * 2
            left_values = self._nodes[left]
            choose_right = remaining >= left_values
            remaining = np.where(choose_right, remaining - left_values, remaining)
            nodes = np.where(choose_right, left + 1, left)
        return (nodes - self._leaf_capacity).astype(np.int32)


@dataclass(slots=True)
class _ReplayPage:
    states: np.ndarray[Any, Any]
    actions: np.ndarray[Any, Any]
    next_states: np.ndarray[Any, Any]
    rewards: np.ndarray[Any, Any]
    discounts: np.ndarray[Any, Any]
    source_transitions: np.ndarray[Any, Any]
    available_transitions: np.ndarray[Any, Any]


@dataclass(frozen=True, slots=True)
class FullRainbowReplayDraw:
    """One with-replacement draw and the exact batch consumed by the core."""

    indices: Array
    sampling_probabilities: Array
    importance_weights: Array
    batch: core.FullRainbowReplayBatch


class CompactPrioritizedReplay:
    """Paged uint8 replay storage backed by an exact proportional sum tree."""

    duplicate_priority_resolution: Final = "first_occurrence_wins"

    def __init__(
        self,
        *,
        capacity: int,
        batch_size: int,
        observation_shape: tuple[int, int, int],
        page_size: int,
    ) -> None:
        for name, value in (
            ("capacity", capacity),
            ("batch_size", batch_size),
            ("page_size", page_size),
        ):
            if type(value) is not int or value <= 0:
                raise FullRainbowRunnerContractError(
                    f"replay {name} must be an exact positive integer"
                )
        if capacity > _MAX_REPLAY_CAPACITY:
            raise FullRainbowRunnerContractError(
                "replay capacity exceeds the exact bounded maximum"
            )
        if batch_size > _MAX_BATCH_SIZE:
            raise FullRainbowRunnerContractError(
                "replay batch_size exceeds the exact bounded maximum"
            )
        if page_size > _MAX_PAGE_SIZE or page_size > capacity:
            raise FullRainbowRunnerContractError(
                "replay page_size exceeds the bounded layout"
            )
        if type(observation_shape) is not tuple or observation_shape != (9, 9, 3):
            raise FullRainbowRunnerContractError(
                "replay observation_shape must be exact (9, 9, 3)"
            )
        self._capacity = capacity
        self._batch_size = batch_size
        self._observation_shape = observation_shape
        self._page_size = page_size
        self._pages: dict[int, _ReplayPage] = {}
        self._tree = _PrioritySumTree(capacity)
        self._size = 0
        self._insertions = 0
        self._next_index = 0
        self._max_recorded_priority = 1.0

    @property
    def size(self) -> int:
        return self._size

    @property
    def insertions(self) -> int:
        return self._insertions

    @property
    def evictions(self) -> int:
        return max(0, self._insertions - self._capacity)

    @property
    def next_index(self) -> int:
        return self._next_index

    @property
    def page_count(self) -> int:
        return len(self._pages)

    @property
    def max_recorded_priority(self) -> float:
        return self._max_recorded_priority

    def _page(self, slot: int, *, create: bool) -> tuple[_ReplayPage, int]:
        page_index, offset = divmod(slot, self._page_size)
        page = self._pages.get(page_index)
        if page is None:
            if not create:
                raise FullRainbowRunnerContractError("replay slot has no allocated page")
            length = min(self._page_size, self._capacity - page_index * self._page_size)
            page = _ReplayPage(
                states=np.zeros((length, *self._observation_shape), dtype=np.uint8),
                actions=np.zeros((length,), dtype=np.uint8),
                next_states=np.zeros(
                    (length, *self._observation_shape), dtype=np.uint8
                ),
                rewards=np.zeros((length,), dtype=np.float32),
                discounts=np.zeros((length,), dtype=np.float32),
                source_transitions=np.zeros((length,), dtype=np.int64),
                available_transitions=np.zeros((length,), dtype=np.int64),
            )
            self._pages[page_index] = page
        return page, offset

    def _validate_active_index(self, index: int) -> None:
        if not 0 <= index < self._size:
            raise FullRainbowRunnerContractError("replay index is not active")

    def add(self, value: FullRainbowAccumulatedTransition) -> int:
        """Overwrite the next ring slot using the monotonic recorded maximum."""

        if type(value) is not FullRainbowAccumulatedTransition:
            raise FullRainbowRunnerContractError(
                "replay value must be a FullRainbowAccumulatedTransition"
            )
        state = np.asarray(_validate_observation(value.state), dtype=np.uint8)
        next_state = np.asarray(
            _validate_observation(value.next_state), dtype=np.uint8
        )
        action = _validate_action(value.action)
        reward = float(value.scaled_n_step_reward)
        discount = float(value.bootstrap_discount)
        if not math.isfinite(reward) or not math.isfinite(discount):
            raise FullRainbowRunnerContractError("replay return values must be finite")
        insertion_point = bisect.bisect_left(
            _VALID_THREE_STEP_SCALED_RETURNS, reward
        )
        adjacent = _VALID_THREE_STEP_SCALED_RETURNS[
            max(0, insertion_point - 1) : insertion_point + 1
        ]
        if not any(
            math.isclose(reward, candidate, rel_tol=0.0, abs_tol=2e-6)
            for candidate in adjacent
        ):
            raise FullRainbowRunnerContractError(
                "replay scaled return is unreachable from exact task rewards"
            )
        if not math.isclose(discount, 0.99**3, rel_tol=0.0, abs_tol=1e-7):
            raise FullRainbowRunnerContractError(
                "replay bootstrap discount must be exact gamma cubed"
            )
        if (
            type(value.source_transition) is not int
            or value.source_transition <= 0
            or type(value.available_after_transition) is not int
            or value.available_after_transition
            != value.source_transition + 3
        ):
            raise FullRainbowRunnerContractError(
                "replay source/availability accounting is inconsistent"
            )
        slot = self._next_index
        page, offset = self._page(slot, create=True)
        page.states[offset] = state
        page.actions[offset] = action
        page.next_states[offset] = next_state
        page.rewards[offset] = np.float32(reward)
        page.discounts[offset] = np.float32(discount)
        page.source_transitions[offset] = value.source_transition
        page.available_transitions[offset] = value.available_after_transition
        self._tree.set(slot, self._max_recorded_priority)
        self._insertions += 1
        self._size = min(self._size + 1, self._capacity)
        self._next_index = (slot + 1) % self._capacity
        return slot

    def transition_at(self, index: int) -> FullRainbowAccumulatedTransition:
        if type(index) is not int:
            raise FullRainbowRunnerContractError("replay index must be an exact int")
        self._validate_active_index(index)
        page, offset = self._page(index, create=False)
        return FullRainbowAccumulatedTransition(
            state=jnp.asarray(page.states[offset], dtype=jnp.float32),
            action=int(page.actions[offset]),
            next_state=jnp.asarray(page.next_states[offset], dtype=jnp.float32),
            scaled_n_step_reward=float(page.rewards[offset]),
            bootstrap_discount=float(page.discounts[offset]),
            source_transition=int(page.source_transitions[offset]),
            available_after_transition=int(page.available_transitions[offset]),
        )

    def priority_at(self, index: int) -> float:
        if type(index) is not int:
            raise FullRainbowRunnerContractError("replay index must be an exact int")
        self._validate_active_index(index)
        return self._tree.get(index)

    def update_priorities(self, indices: object, priorities: object) -> None:
        """Apply the first value for duplicate slots, matching Dopamine SumTree.set."""

        index_values = np.asarray(indices)
        priority_values = np.asarray(priorities)
        if (
            index_values.ndim != 1
            or index_values.size == 0
            or not np.issubdtype(index_values.dtype, np.integer)
            or np.issubdtype(index_values.dtype, np.bool_)
            or priority_values.shape != index_values.shape
            or not np.issubdtype(priority_values.dtype, np.floating)
        ):
            raise FullRainbowRunnerContractError(
                "priority indices and values must be matching numeric vectors"
            )
        if (
            not bool(np.all(np.isfinite(priority_values)))
            or bool(np.any(priority_values < 0.0))
            or bool(np.any(priority_values > np.finfo(np.float64).max))
        ):
            raise FullRainbowRunnerContractError(
                "priority values must be finite and nonnegative"
            )
        float_priorities = priority_values.astype(np.float64, copy=False)
        integer_indices = index_values.astype(np.int64, copy=False)
        for index in integer_indices:
            self._validate_active_index(int(index))
        unique_indices, first_positions = np.unique(
            integer_indices, return_index=True
        )
        candidate_total = self._tree.total
        for index, position in zip(unique_indices, first_positions, strict=True):
            candidate_total += float(float_priorities[position]) - self._tree.get(
                int(index)
            )
        if not math.isfinite(candidate_total) or candidate_total < 0.0:
            raise FullRainbowRunnerContractError(
                "priority update would make the replay total invalid"
            )
        self._max_recorded_priority = max(
            self._max_recorded_priority,
            float(np.max(float_priorities)),
        )
        for index, position in zip(unique_indices, first_positions, strict=True):
            self._tree.set(int(index), float(float_priorities[position]))

    def sample(self, key: Array) -> FullRainbowReplayDraw:
        if self._size == 0:
            raise FullRainbowRunnerContractError("cannot sample empty replay")
        _validate_key(key, label="replay sampling key")
        total = self._tree.total
        if not math.isfinite(total) or total < 0.0:
            raise FullRainbowRunnerContractError("replay priority total is invalid")
        if total == 0.0:
            indices = np.asarray(
                jr.randint(key, (self._batch_size,), 0, self._size), dtype=np.int32
            )
            probabilities = np.full(
                (self._batch_size,), 1.0 / self._size, dtype=np.float32
            )
        else:
            targets = np.asarray(
                jr.uniform(
                    key,
                    (self._batch_size,),
                    minval=0.0,
                    maxval=total,
                    dtype=jnp.float32,
                ),
                dtype=np.float64,
            )
            indices = self._tree.query(targets)
            probabilities = np.asarray(
                [self._tree.get(int(index)) / total for index in indices],
                dtype=np.float32,
            )
        records = [self.transition_at(int(index)) for index in indices]
        states = jnp.stack([record.state for record in records]).astype(jnp.float32)
        actions = jnp.asarray([record.action for record in records], dtype=jnp.int32)
        next_states = jnp.stack([record.next_state for record in records]).astype(
            jnp.float32
        )
        rewards = jnp.asarray(
            [record.scaled_n_step_reward for record in records], dtype=jnp.float32
        )
        discounts = jnp.asarray(
            [record.bootstrap_discount for record in records], dtype=jnp.float32
        )
        selected_probabilities = jnp.asarray(probabilities, dtype=jnp.float32)
        weights = core.importance_sampling_weights(selected_probabilities)
        batch = core.FullRainbowReplayBatch(
            states=states,
            actions=actions,
            next_states=next_states,
            scaled_n_step_rewards=rewards,
            bootstrap_discounts=discounts,
            sampling_probabilities=selected_probabilities,
        )
        return FullRainbowReplayDraw(
            indices=jnp.asarray(indices, dtype=jnp.int32),
            sampling_probabilities=selected_probabilities,
            importance_weights=weights,
            batch=batch,
        )


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class FullRainbowRunnerDependencies:
    """Explicit environment/core boundary used by production and tiny tests."""

    dependency_identity: str
    environment_runtime: Any
    step_environment: Callable[..., Any]
    initialize_core: Callable[..., core.FullRainbowCoreState]
    action_q_values: Callable[..., Array]
    update_core: Callable[..., tuple[core.FullRainbowCoreState, core.FullRainbowTrainMetrics]]
    sync_target: Callable[[core.FullRainbowCoreState], core.FullRainbowCoreState]
    runtime_identity: Mapping[str, Any]
    compiled_action_kernel: bool
    compiled_update_kernel: bool

    def __post_init__(self) -> None:
        if type(self.dependency_identity) is not str or not self.dependency_identity:
            raise FullRainbowRunnerContractError(
                "dependency_identity must be a nonempty exact string"
            )
        if not callable(getattr(self.environment_runtime, "initialize", None)):
            raise FullRainbowRunnerContractError(
                "environment_runtime must expose callable initialize"
            )
        for name in (
            "step_environment",
            "initialize_core",
            "action_q_values",
            "update_core",
            "sync_target",
        ):
            if not callable(getattr(self, name)):
                raise FullRainbowRunnerContractError(
                    f"runner dependency {name} must be callable"
                )
        if type(self.runtime_identity) is not dict:
            raise FullRainbowRunnerContractError(
                "runtime_identity must be a plain mapping"
            )
        _assert_plain_unaliased_json(self.runtime_identity)
        identity = cast(Mapping[str, object], self.runtime_identity)
        if (
            type(identity.get("backend")) is not str
            or not identity["backend"]
            or identity.get("runtime_qualified") is not False
            or identity.get("foragax_runtime_parity_executed") is not False
        ):
            raise FullRainbowRunnerContractError(
                "runtime_identity must explicitly record an unqualified backend"
            )
        _reject_authority_anywhere(dict(identity), label="runtime_identity")
        object.__setattr__(
            self,
            "runtime_identity",
            MappingProxyType(copy.deepcopy(dict(identity))),
        )
        if (
            type(self.compiled_action_kernel) is not bool
            or type(self.compiled_update_kernel) is not bool
        ):
            raise FullRainbowRunnerContractError(
                "compiled kernel flags must be exact booleans"
            )


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class FullRainbowRunnerResult:
    """Compact immutable raw trace and its canonical unqualified receipt."""

    raw_reward_trace: bytes
    cumulative_raw_score: int
    interactions: int
    receipt_bytes: bytes
    production_runtime: bool


@dataclass(frozen=True, slots=True)
class _ProductionDependencyBinding:
    dependency_identity: str
    environment_runtime: object
    step_environment: object
    initialize_core: object
    action_q_values: object
    update_core: object
    sync_target: object
    runtime_identity: object
    runtime_identity_sha256: str
    compiled_action_kernel: bool
    compiled_update_kernel: bool


@dataclass(frozen=True, slots=True)
class _ProductionResultBinding:
    raw_reward_trace: bytes
    raw_reward_trace_sha256: str
    cumulative_raw_score: int
    interactions: int
    receipt_bytes: bytes
    receipt_sha256: str
    production_runtime: bool


_PRODUCTION_REGISTRY_LOCK: Final = threading.RLock()
_PRODUCTION_DEPENDENCY_REGISTRY: Final = weakref.WeakKeyDictionary[
    FullRainbowRunnerDependencies, _ProductionDependencyBinding
]()
_PRODUCTION_RESULT_REGISTRY: Final = weakref.WeakKeyDictionary[
    FullRainbowRunnerResult, _ProductionResultBinding
]()


def _validate_key(value: object, *, label: str) -> tuple[int, int]:
    try:
        dtype = cast(Any, value).dtype
        implementation = str(jr.key_impl(cast(Any, value)))
        words = np.asarray(jr.key_data(cast(Any, value)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise FullRainbowRunnerContractError(f"{label} is not a typed JAX key") from exc
    if (
        str(dtype) != "key<fry>"
        or implementation != "threefry2x32"
        or words.shape != (2,)
        or words.dtype != np.dtype(np.uint32)
    ):
        raise FullRainbowRunnerContractError(
            f"{label} must be exact Threefry2x32"
        )
    return int(words[0]), int(words[1])


def resolve_zero_epsilon_action(
    q_values: object, *, uniform_draw: object, random_action: object
) -> int:
    """Resolve Dopamine's ``uniform <= 0`` branch after its four-way split."""

    values = np.asarray(q_values)
    if values.shape != (4,) or values.dtype != np.dtype(np.float32):
        raise FullRainbowRunnerContractError(
            "action Q-values must be exact float32 shape (4,)"
        )
    if not bool(np.all(np.isfinite(values))):
        raise FullRainbowRunnerContractError("action Q-values must be finite")
    draw = np.asarray(uniform_draw)
    if draw.shape != () or not np.issubdtype(draw.dtype, np.floating):
        raise FullRainbowRunnerContractError("uniform draw must be a floating scalar")
    draw_value = float(draw)
    if not math.isfinite(draw_value) or not 0.0 <= draw_value < 1.0:
        raise FullRainbowRunnerContractError("uniform draw must lie in [0, 1)")
    exact_random_action = _validate_action(random_action)
    return exact_random_action if draw_value <= 0.0 else int(np.argmax(values))


def _tree_finite(value: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(value)
    if not leaves:
        return False
    return all(bool(np.all(np.isfinite(np.asarray(leaf)))) for leaf in leaves)


def _validate_initial_core_state(
    state: object,
    *,
    environment_seed: int,
    agent_seed: int,
) -> core.FullRainbowCoreState:
    if type(state) is not core.FullRainbowCoreState:
        raise FullRainbowRunnerContractError(
            "initialize_core must return an exact FullRainbowCoreState"
        )
    roots = core.full_rainbow_seed_roots(
        environment_seed=environment_seed,
        agent_seed=agent_seed,
    )
    expected_agent, _, _ = jr.split(roots.agent, 3)
    if _validate_key(state.environment_rng, label="core environment root") != (
        _validate_key(roots.environment, label="expected environment root")
    ):
        raise FullRainbowRunnerContractError(
            "core environment root does not match the raw environment seed"
        )
    if _validate_key(state.agent_rng, label="core agent continuation") != (
        _validate_key(expected_agent, label="expected agent continuation")
    ):
        raise FullRainbowRunnerContractError(
            "core initialization did not preserve the exact agent split order"
        )
    if type(state.optimizer_updates) is not int or state.optimizer_updates != 0:
        raise FullRainbowRunnerContractError(
            "core optimizer update accounting must start at zero"
        )
    if state.target_params is not state.online_params:
        raise FullRainbowRunnerContractError(
            "core target parameters must initially alias online parameters"
        )
    if not all(
        _tree_finite(value)
        for value in (state.online_params, state.target_params, state.optimizer_state)
    ):
        raise FullRainbowRunnerContractError(
            "initialized core parameter/optimizer trees must be finite"
        )
    return state


def _validate_environment_state(
    value: object,
    *,
    environment_seed: int,
    expected_step_count: int,
) -> Array:
    dynamic = cast(Any, value)
    for name in (
        "environment_seed",
        "observation",
        "reset_count",
        "step_count",
        "environment_key_use_count",
    ):
        if not hasattr(value, name):
            raise FullRainbowRunnerContractError(
                f"environment state lacks required field {name}"
            )
    if (
        type(dynamic.environment_seed) is not int
        or dynamic.environment_seed != environment_seed
        or type(dynamic.reset_count) is not int
        or dynamic.reset_count != 1
        or type(dynamic.step_count) is not int
        or dynamic.step_count != expected_step_count
        or type(dynamic.environment_key_use_count) is not int
        or dynamic.environment_key_use_count != expected_step_count + 1
    ):
        raise FullRainbowRunnerContractError(
            "environment state accounting disagrees with the runner"
        )
    return _validate_observation(dynamic.observation)


def _validate_transition(
    value: object,
    *,
    previous_state: object,
    environment_seed: int,
    expected_step_count: int,
    action: int,
) -> tuple[object, Array, int]:
    dynamic = cast(Any, value)
    for name in ("state", "action", "reward", "done", "truncated", "info_validated"):
        if not hasattr(value, name):
            raise FullRainbowRunnerContractError(
                f"environment transition lacks required field {name}"
            )
    if dynamic.state is previous_state:
        raise FullRainbowRunnerContractError(
            "environment state was reused instead of advanced"
        )
    observation = _validate_environment_state(
        dynamic.state,
        environment_seed=environment_seed,
        expected_step_count=expected_step_count,
    )
    if _validate_action(dynamic.action) != action:
        raise FullRainbowRunnerContractError("environment transition action drifted")
    reward = _validate_raw_reward(dynamic.reward)
    if (
        type(dynamic.done) is not bool
        or dynamic.done
        or type(dynamic.truncated) is not bool
        or dynamic.truncated
        or type(dynamic.info_validated) is not bool
        or not dynamic.info_validated
    ):
        raise FullRainbowRunnerContractError(
            "environment transition must remain continuing and info-validated"
        )
    return dynamic.state, observation, reward


def _validate_metrics(
    metrics: object, *, batch_size: int
) -> core.FullRainbowTrainMetrics:
    if type(metrics) is not core.FullRainbowTrainMetrics:
        raise FullRainbowRunnerContractError(
            "update_core must return exact FullRainbowTrainMetrics"
        )
    mean = np.asarray(metrics.mean_weighted_loss)
    losses = np.asarray(metrics.per_example_loss)
    weights = np.asarray(metrics.importance_weights)
    priorities = np.asarray(metrics.updated_priorities)
    actions = np.asarray(metrics.double_q_actions)
    if mean.shape != () or not bool(np.isfinite(mean)):
        raise FullRainbowRunnerContractError("mean update loss must be finite scalar")
    if any(array.shape != (batch_size,) for array in (losses, weights, priorities, actions)):
        raise FullRainbowRunnerContractError(
            "update metrics must exactly match replay batch size"
        )
    if not all(
        bool(np.all(np.isfinite(array))) for array in (losses, weights, priorities)
    ):
        raise FullRainbowRunnerContractError("update metrics must be finite")
    if bool(np.any(losses < 0.0)) or bool(np.any(priorities <= 0.0)):
        raise FullRainbowRunnerContractError(
            "losses must be nonnegative and priorities positive"
        )
    expected_priorities = np.sqrt(losses + _PRIORITY_EPSILON)
    expected_mean = np.mean(weights * losses, dtype=np.float32)
    if not bool(
        np.allclose(priorities, expected_priorities, rtol=1e-6, atol=1e-7)
    ) or not math.isclose(
        float(mean), float(expected_mean), rel_tol=1e-6, abs_tol=1e-7
    ):
        raise FullRainbowRunnerContractError(
            "update metrics disagree with exact weighted-loss priority semantics"
        )
    if (
        not np.issubdtype(actions.dtype, np.integer)
        or np.issubdtype(actions.dtype, np.bool_)
        or bool(np.any(actions < 0))
        or bool(np.any(actions >= 4))
    ):
        raise FullRainbowRunnerContractError(
            "Double-Q actions must be integer indices in [0, 3]"
        )
    return metrics


def _canonical_json(value: object) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise FullRainbowRunnerContractError(
            "runner value is not finite canonical JSON"
        ) from exc
    if len(raw) > _MAX_CANONICAL_BYTES:
        raise FullRainbowRunnerContractError("runner canonical payload is too large")
    return raw


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FullRainbowRunnerContractError(
            f"{label} must be an exact lowercase SHA-256"
        )
    return value


def _exact_plain_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_mapping = cast(dict[str, object], left)
        right_mapping = cast(dict[str, object], right)
        return set(left_mapping) == set(right_mapping) and all(
            _exact_plain_equal(left_mapping[key], right_mapping[key])
            for key in left_mapping
        )
    if type(left) is list:
        left_list = cast(list[object], left)
        right_list = cast(list[object], right)
        return len(left_list) == len(right_list) and all(
            _exact_plain_equal(left_item, right_item)
            for left_item, right_item in zip(left_list, right_list, strict=True)
        )
    return bool(left == right)


def _runner_descriptor() -> dict[str, Any]:
    schedule = production_full_rainbow_schedule()
    return {
        "schema_version": FULL_RAINBOW_RUNNER_DESCRIPTOR_SCHEMA_VERSION,
        "candidate_id": "adapted_full_rainbow",
        "status": FULL_RAINBOW_RUNNER_STATUS,
        "classification": "exact_runner_implemented_unqualified_non_authorizing",
        "implementation": {
            "module": (
                "alberta_framework.benchmarks.forager_matched_v3_full_rainbow_runner"
            ),
            "path": (
                "alberta_framework/benchmarks/forager_matched_v3_full_rainbow_runner.py"
            ),
            "source_self_hash_bound": False,
        },
        "bindings": {
            "bridge_descriptor_sha256": BOUND_BRIDGE_DESCRIPTOR_SHA256,
            "bridge_implementation_path": BOUND_BRIDGE_IMPLEMENTATION_PATH,
            "bridge_implementation_sha256": BOUND_BRIDGE_IMPLEMENTATION_SHA256,
            "core_configuration_sha256": BOUND_CORE_CONFIG_SHA256,
            "core_descriptor_sha256": BOUND_CORE_DESCRIPTOR_SHA256,
            "core_implementation_path": BOUND_CORE_IMPLEMENTATION_PATH,
            "core_implementation_sha256": BOUND_CORE_IMPLEMENTATION_SHA256,
        },
        "source": {
            "upstream_repository": "https://github.com/google/dopamine",
            "upstream_commit_git_sha1": "5873f5494ee0c2d7c016d0ab2ad530354fec59d0",
            "relationship": "runner_for_bound_modified_derivative_core",
            "source_review_complete": False,
            "source_closure_bound": False,
        },
        "schedule": asdict(schedule),
        "exact_accounting": full_rainbow_schedule_accounting(schedule),
        "accumulator": {
            "update_horizon": 3,
            "stack_size": 1,
            "first_emit_transition": 4,
            "nonterminal_stream_entries": "H-3",
            "one_transition_emission_latency": True,
            "terminal_flush": False,
        },
        "replay": {
            "storage": "lazy_paged_uint8_observations_bounded_ring",
            "configured_capacity": 1_000_000,
            "proportional_sampling": True,
            "sampling_with_replacement": True,
            "importance_weights": "max_normalized_inverse_square_root",
            "new_item_priority": "monotonic_max_recorded_priority_initial_1",
            "duplicate_priority_resolution": "first_occurrence_wins",
        },
        "rng": {
            "bridge_is_sole_environment_key_owner": True,
            "core_environment_rng_consumed": False,
            "core_environment_rng_preserved_unchanged": True,
            "agent_parameter_initialization": "split(agent_root,3):next,init,noise",
            "action_selection": (
                "split(agent_rng,4):next,uniform,noise,randint; uniform<=0 edge retained"
            ),
            "replay_sampling": "split(agent_rng,2):next,sample",
            "update": "split(agent_rng,3):current,target,next",
            "seed_source": "caller_supplied",
            "seed_provenance": "unverified",
            "protected_seed_status": "unknown_not_asserted",
        },
        "kernel_boundaries": {
            "action_network_jitted_in_production": True,
            "update_numeric_kernel_jitted_in_production": True,
            "host_validation_before_and_after_update": True,
            "environment_per_step_api_jitted": False,
        },
        "result": {
            "raw_reward_trace": "immutable_signed_int8_bytes",
            "raw_cumulative_score_separate_from_reward_scaling": True,
            "canonical_engineering_receipt_schema": (
                FULL_RAINBOW_ENGINEERING_RECEIPT_SCHEMA_VERSION
            ),
            "canonical_production_result_receipt_schema": (
                FULL_RAINBOW_RESULT_RECEIPT_SCHEMA_VERSION
            ),
            "injected_dependencies_can_emit_production_result": False,
            "production_dependency_capability": (
                "process_local_weak_identity_registry_all_fields_identity_value_bound"
            ),
            "production_result_capability": (
                "process_local_weak_identity_registry_all_fields_identity_value_digest_bound"
            ),
            "receipt_content_independently_proves_execution": False,
            "filesystem_writes": False,
        },
        "claims": _non_authorizing_claims(),
        "blockers": [
            "real_foragax_runtime_parity_unexecuted",
            "backend_and_full_horizon_resource_profile_unqualified",
            "runner_source_closure_and_reproducible_runtime_image_unbound",
            "result_ingestion_and_scientific_qualification_unimplemented",
        ],
        "limitations": [
            "Execution requires an explicit unqualified engineering flag.",
            "A completed result remains non-authorizing and runtime-unqualified.",
            (
                "No seed is embedded, generated, or accepted as authority; caller "
                "provenance and protected status remain unverified."
            ),
            "No checkpoint/resume or filesystem artifact writer is included.",
            "No real or full-horizon workload was run while implementing this layer.",
        ],
    }


_RUNNER_DESCRIPTOR_BYTES: Final = _canonical_json(_runner_descriptor())
FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256: Final = (
    "546009c19454a7839876df6e758b984db931db5eb234ac23833a232c387aa3bc"
)
if not hmac.compare_digest(
    hashlib.sha256(_RUNNER_DESCRIPTOR_BYTES).hexdigest(),
    FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256,
):
    raise RuntimeError("Full Rainbow runner descriptor identity drifted")


def full_rainbow_runner_descriptor() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(_RUNNER_DESCRIPTOR_BYTES.decode("ascii")))


def canonical_full_rainbow_runner_descriptor_bytes() -> bytes:
    return bytes(_RUNNER_DESCRIPTOR_BYTES)


def _assert_plain_unaliased_json(value: object) -> None:
    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if type(item) is dict:
            identity = id(item)
            if identity in seen:
                raise FullRainbowRunnerContractError("JSON contains aliased containers")
            seen.add(identity)
            mapping = cast(dict[object, object], item)
            if any(type(key) is not str for key in mapping):
                raise FullRainbowRunnerContractError("JSON contains a non-string key")
            pending.extend(mapping.values())
        elif type(item) is list:
            identity = id(item)
            if identity in seen:
                raise FullRainbowRunnerContractError("JSON contains aliased containers")
            seen.add(identity)
            pending.extend(cast(list[object], item))
        elif type(item) is float:
            if not math.isfinite(item):
                raise FullRainbowRunnerContractError("JSON contains a non-finite float")
        elif item is not None and type(item) not in {str, int, bool}:
            raise FullRainbowRunnerContractError("JSON contains a non-plain value")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FullRainbowRunnerContractError("JSON contains duplicate keys")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> NoReturn:
    raise FullRainbowRunnerContractError(f"JSON contains forbidden constant {token}")


def _strict_json_input(
    value: bytes | Mapping[str, Any], *, label: str
) -> tuple[dict[str, Any], bytes]:
    if type(value) is bytes:
        raw = value
    elif isinstance(value, Mapping):
        _assert_plain_unaliased_json(value)
        raw = _canonical_json(dict(value))
    else:
        raise FullRainbowRunnerContractError(
            f"{label} must be exact bytes or a plain mapping"
        )
    try:
        decoded = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except FullRainbowRunnerContractError:
        raise
    except (UnicodeError, ValueError) as exc:
        raise FullRainbowRunnerContractError(f"{label} is not strict JSON") from exc
    if type(decoded) is not dict or _canonical_json(decoded) != raw:
        raise FullRainbowRunnerContractError(f"{label} is not an exact canonical object")
    return cast(dict[str, Any], decoded), raw


def parse_full_rainbow_runner_descriptor(
    value: bytes | Mapping[str, Any],
) -> dict[str, Any]:
    decoded, raw = _strict_json_input(value, label="runner descriptor")
    if not hmac.compare_digest(raw, _RUNNER_DESCRIPTOR_BYTES):
        raise FullRainbowRunnerContractError(
            "runner descriptor does not match the frozen identity"
        )
    return copy.deepcopy(decoded)


def _schedule_from_payload(value: object) -> FullRainbowRunnerSchedule:
    if type(value) is not dict:
        raise FullRainbowRunnerContractError("receipt schedule must be an object")
    expected = {
        "horizon",
        "update_horizon",
        "replay_capacity",
        "batch_size",
        "minimum_replay_history",
        "update_period",
        "target_update_period",
        "page_size",
    }
    if set(value) != expected:
        raise FullRainbowRunnerContractError("receipt schedule schema drifted")
    return FullRainbowRunnerSchedule(**cast(dict[str, Any], value))


def canonical_full_rainbow_receipt_body_bytes(
    value: Mapping[str, Any],
) -> bytes:
    """Canonicalize a receipt body, removing its detached body hash if present."""

    if not isinstance(value, Mapping):
        raise FullRainbowRunnerContractError("receipt body must be a mapping")
    body = dict(value)
    body.pop("receipt_body_sha256", None)
    _assert_plain_unaliased_json(body)
    return _canonical_json(body)


def _receipt(
    *,
    environment_seed: int,
    agent_seed: int,
    schedule: FullRainbowRunnerSchedule,
    dependencies: FullRainbowRunnerDependencies,
    raw_trace: bytes,
    cumulative_score: int,
    accounting: dict[str, Any],
    rng_accounting: dict[str, int],
    production_runtime: bool,
) -> bytes:
    exact_schedule = schedule == production_full_rainbow_schedule()
    if production_runtime and not exact_schedule:
        raise FullRainbowRunnerContractError(
            "production result receipt requires the exact production schedule"
        )
    body: dict[str, Any] = {
        "schema_version": (
            FULL_RAINBOW_RESULT_RECEIPT_SCHEMA_VERSION
            if production_runtime
            else FULL_RAINBOW_ENGINEERING_RECEIPT_SCHEMA_VERSION
        ),
        "candidate_id": "adapted_full_rainbow",
        "status": (
            "completed_runtime_unqualified"
            if production_runtime
            else "completed_engineering_unqualified"
        ),
        "classification": (
            "production_runtime_unqualified_non_authorizing"
            if production_runtime
            else "synthetic_engineering_non_authorizing"
        ),
        "runner_descriptor_sha256": FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256,
        "bindings": {
            "bridge_descriptor_sha256": BOUND_BRIDGE_DESCRIPTOR_SHA256,
            "bridge_implementation_path": BOUND_BRIDGE_IMPLEMENTATION_PATH,
            "bridge_implementation_sha256": BOUND_BRIDGE_IMPLEMENTATION_SHA256,
            "core_configuration_sha256": BOUND_CORE_CONFIG_SHA256,
            "core_descriptor_sha256": BOUND_CORE_DESCRIPTOR_SHA256,
            "core_implementation_path": BOUND_CORE_IMPLEMENTATION_PATH,
            "core_implementation_sha256": BOUND_CORE_IMPLEMENTATION_SHA256,
        },
        "dependency_identity": dependencies.dependency_identity,
        "runtime_identity": dict(dependencies.runtime_identity),
        "kernel_boundaries": {
            "compiled_action_kernel": dependencies.compiled_action_kernel,
            "compiled_update_kernel": dependencies.compiled_update_kernel,
            "environment_per_step_api_jitted": False,
        },
        "schedule": asdict(schedule),
        "schedule_sha256": _canonical_sha256(asdict(schedule)),
        "seeds": {
            "environment_seed": environment_seed,
            "agent_seed": agent_seed,
            "domain": "uint31",
            "source": "caller_supplied",
            "seed_provenance_classification": "caller_supplied_unverified",
            "seed_provenance_verified": False,
            "protected_seed_status": "unknown_not_asserted",
        },
        "accounting": accounting,
        "rng_accounting": rng_accounting,
        "score": {
            "raw_reward_trace_encoding": "signed_int8_twos_complement",
            "raw_reward_trace_length": len(raw_trace),
            "raw_reward_trace_sha256": hashlib.sha256(raw_trace).hexdigest(),
            "raw_cumulative_score": cumulative_score,
            "reward_scaling_applied_to_score": False,
        },
        "completion": {
            "requested_horizon": schedule.horizon,
            "interactions_completed": len(raw_trace),
            "engineering_schedule_complete": len(raw_trace) == schedule.horizon,
            "schedule_is_exact_production": exact_schedule,
            "exact_matched_v3_horizon_complete": (
                production_runtime
                and exact_schedule
                and len(raw_trace) == core.FullRainbowForagerConfig().horizon
            ),
            "production_runtime_complete": production_runtime,
            "content_and_accounting_independently_prove_execution": False,
        },
        "claims": _non_authorizing_claims(),
        "limitations": _receipt_limitations(),
    }
    _reject_authority_anywhere(body, label="Full Rainbow receipt body")
    receipt = dict(body)
    receipt["receipt_body_sha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    return _canonical_json(receipt)


def _parse_full_rainbow_receipt(
    value: bytes | Mapping[str, Any], *, production_runtime: bool
) -> dict[str, Any]:
    label = "production result receipt" if production_runtime else "engineering receipt"
    receipt, _ = _strict_json_input(value, label=label)
    expected_keys = {
        "schema_version",
        "candidate_id",
        "status",
        "classification",
        "runner_descriptor_sha256",
        "bindings",
        "dependency_identity",
        "runtime_identity",
        "kernel_boundaries",
        "schedule",
        "schedule_sha256",
        "seeds",
        "accounting",
        "rng_accounting",
        "score",
        "completion",
        "claims",
        "limitations",
        "receipt_body_sha256",
    }
    if set(receipt) != expected_keys:
        raise FullRainbowRunnerContractError(f"{label} schema drifted")
    supplied_hash = _require_sha256(
        receipt["receipt_body_sha256"], label="receipt_body_sha256"
    )
    calculated_hash = hashlib.sha256(
        canonical_full_rainbow_receipt_body_bytes(receipt)
    ).hexdigest()
    if not hmac.compare_digest(supplied_hash, calculated_hash):
        raise FullRainbowRunnerContractError(f"{label} body hash drifted")
    expected_schema = (
        FULL_RAINBOW_RESULT_RECEIPT_SCHEMA_VERSION
        if production_runtime
        else FULL_RAINBOW_ENGINEERING_RECEIPT_SCHEMA_VERSION
    )
    expected_status = (
        "completed_runtime_unqualified"
        if production_runtime
        else "completed_engineering_unqualified"
    )
    expected_classification = (
        "production_runtime_unqualified_non_authorizing"
        if production_runtime
        else "synthetic_engineering_non_authorizing"
    )
    expected_bindings = {
        "bridge_descriptor_sha256": BOUND_BRIDGE_DESCRIPTOR_SHA256,
        "bridge_implementation_path": BOUND_BRIDGE_IMPLEMENTATION_PATH,
        "bridge_implementation_sha256": BOUND_BRIDGE_IMPLEMENTATION_SHA256,
        "core_configuration_sha256": BOUND_CORE_CONFIG_SHA256,
        "core_descriptor_sha256": BOUND_CORE_DESCRIPTOR_SHA256,
        "core_implementation_path": BOUND_CORE_IMPLEMENTATION_PATH,
        "core_implementation_sha256": BOUND_CORE_IMPLEMENTATION_SHA256,
    }
    if (
        receipt["schema_version"] != expected_schema
        or receipt["candidate_id"] != "adapted_full_rainbow"
        or receipt["status"] != expected_status
        or receipt["classification"] != expected_classification
        or receipt["runner_descriptor_sha256"]
        != FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256
        or not _exact_plain_equal(receipt["bindings"], expected_bindings)
        or not _exact_plain_equal(receipt["claims"], _non_authorizing_claims())
        or not _exact_plain_equal(receipt["limitations"], _receipt_limitations())
    ):
        raise FullRainbowRunnerContractError(
            f"{label} identity or authority fields drifted"
        )
    _reject_authority_anywhere(receipt, label=label)
    dependency_identity = receipt["dependency_identity"]
    if type(dependency_identity) is not str or not dependency_identity:
        raise FullRainbowRunnerContractError(f"{label} dependency identity is invalid")
    if production_runtime and dependency_identity != (
        "production_bridge_and_compiled_full_rainbow_v1"
    ):
        raise FullRainbowRunnerContractError(
            "production result receipt dependency identity drifted"
        )
    runtime_identity = receipt["runtime_identity"]
    if (
        type(runtime_identity) is not dict
        or type(runtime_identity.get("backend")) is not str
        or not runtime_identity["backend"]
        or runtime_identity.get("runtime_qualified") is not False
        or runtime_identity.get("foragax_runtime_parity_executed") is not False
    ):
        raise FullRainbowRunnerContractError(
            f"{label} runtime must remain explicitly unqualified"
        )
    kernel_boundaries = receipt["kernel_boundaries"]
    expected_kernel_keys = {
        "compiled_action_kernel",
        "compiled_update_kernel",
        "environment_per_step_api_jitted",
    }
    if (
        type(kernel_boundaries) is not dict
        or set(kernel_boundaries) != expected_kernel_keys
        or any(type(value) is not bool for value in kernel_boundaries.values())
        or kernel_boundaries["environment_per_step_api_jitted"] is not False
        or (
            production_runtime
            and (
                kernel_boundaries["compiled_action_kernel"] is not True
                or kernel_boundaries["compiled_update_kernel"] is not True
            )
        )
    ):
        raise FullRainbowRunnerContractError(f"{label} kernel boundary drifted")
    schedule = _schedule_from_payload(receipt["schedule"])
    schedule_hash = _require_sha256(receipt["schedule_sha256"], label="schedule_sha256")
    if not hmac.compare_digest(schedule_hash, _canonical_sha256(asdict(schedule))):
        raise FullRainbowRunnerContractError(f"{label} schedule hash drifted")
    exact_schedule = schedule == production_full_rainbow_schedule()
    if production_runtime and not exact_schedule:
        raise FullRainbowRunnerContractError(
            "production result receipt requires the exact production schedule"
        )
    expected_accounting = full_rainbow_schedule_accounting(schedule)
    expected_updates = [
        transition
        for transition in range(1, schedule.horizon + 1)
        if transition - schedule.update_horizon > schedule.minimum_replay_history
        and transition % schedule.update_period == 0
    ]
    expected_syncs = [
        transition
        for transition in range(1, schedule.horizon + 1)
        if transition - schedule.update_horizon > schedule.minimum_replay_history
        and transition % schedule.target_update_period == 0
    ]
    exact_accounting: dict[str, object] = {
        **expected_accounting,
        "update_transitions": expected_updates,
        "target_sync_transitions": expected_syncs,
    }
    if not _exact_plain_equal(receipt["accounting"], exact_accounting):
        raise FullRainbowRunnerContractError(f"{label} accounting drifted")
    seeds = receipt["seeds"]
    expected_seed_keys = {
        "environment_seed",
        "agent_seed",
        "domain",
        "source",
        "seed_provenance_classification",
        "seed_provenance_verified",
        "protected_seed_status",
    }
    if (
        type(seeds) is not dict
        or set(seeds) != expected_seed_keys
        or type(seeds["environment_seed"]) is not int
        or type(seeds["agent_seed"]) is not int
        or seeds["domain"] != "uint31"
        or seeds["source"] != "caller_supplied"
        or seeds["seed_provenance_classification"]
        != "caller_supplied_unverified"
        or seeds["seed_provenance_verified"] is not False
        or seeds["protected_seed_status"] != "unknown_not_asserted"
    ):
        raise FullRainbowRunnerContractError(f"{label} seed contract drifted")
    try:
        core.full_rainbow_seed_roots(
            environment_seed=seeds["environment_seed"],
            agent_seed=seeds["agent_seed"],
        )
    except core.FullRainbowContractError as exc:
        raise FullRainbowRunnerContractError(f"{label} seeds are invalid") from exc
    expected_completion = {
        "requested_horizon": schedule.horizon,
        "interactions_completed": schedule.horizon,
        "engineering_schedule_complete": True,
        "schedule_is_exact_production": exact_schedule,
        "exact_matched_v3_horizon_complete": production_runtime,
        "production_runtime_complete": production_runtime,
        "content_and_accounting_independently_prove_execution": False,
    }
    if not _exact_plain_equal(receipt["completion"], expected_completion):
        raise FullRainbowRunnerContractError(f"{label} completion drifted")
    score = receipt["score"]
    expected_score_keys = {
        "raw_reward_trace_encoding",
        "raw_reward_trace_length",
        "raw_reward_trace_sha256",
        "raw_cumulative_score",
        "reward_scaling_applied_to_score",
    }
    if (
        type(score) is not dict
        or set(score) != expected_score_keys
        or score["raw_reward_trace_encoding"] != "signed_int8_twos_complement"
        or type(score["raw_reward_trace_length"]) is not int
        or score["raw_reward_trace_length"] != schedule.horizon
        or type(score["raw_cumulative_score"]) is not int
        or not -schedule.horizon
        <= score["raw_cumulative_score"]
        <= 30 * schedule.horizon
        or score["reward_scaling_applied_to_score"] is not False
    ):
        raise FullRainbowRunnerContractError(f"{label} score contract drifted")
    _require_sha256(score["raw_reward_trace_sha256"], label="raw_reward_trace_sha256")
    updates = expected_accounting["optimizer_updates"]
    if type(updates) is not int:
        raise AssertionError("optimizer update accounting type drifted")
    expected_rng = {
        "agent_continuation_split_calls": 1 + schedule.horizon + 2 * updates,
        "agent_parameter_initialization_subkeys": 2,
        "agent_action_split_calls": schedule.horizon,
        "agent_action_subkeys_produced_per_call": 3,
        "agent_replay_sampling_split_calls": updates,
        "agent_update_split_calls": updates,
        "agent_update_subkeys_produced_per_call": 2,
        "core_environment_key_consumptions": 0,
        "bridge_environment_key_consumptions": schedule.horizon + 1,
    }
    if not _exact_plain_equal(receipt["rng_accounting"], expected_rng):
        raise FullRainbowRunnerContractError(f"{label} RNG accounting drifted")
    return copy.deepcopy(receipt)


def parse_full_rainbow_engineering_receipt(
    value: bytes | Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly parse a non-production engineering receipt."""

    return _parse_full_rainbow_receipt(value, production_runtime=False)


def parse_full_rainbow_result_receipt(
    value: bytes | Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly parse an unqualified exact-runtime production result receipt."""

    return _parse_full_rainbow_receipt(value, production_runtime=True)


def _validate_production_result_capability(
    result: FullRainbowRunnerResult,
) -> None:
    with _PRODUCTION_REGISTRY_LOCK:
        binding = _PRODUCTION_RESULT_REGISTRY.get(result)
    if (
        binding is None
        or result.raw_reward_trace is not binding.raw_reward_trace
        or not hmac.compare_digest(
            binding.raw_reward_trace_sha256,
            hashlib.sha256(result.raw_reward_trace).hexdigest(),
        )
        or result.cumulative_raw_score != binding.cumulative_raw_score
        or type(result.cumulative_raw_score) is not type(binding.cumulative_raw_score)
        or result.interactions != binding.interactions
        or type(result.interactions) is not type(binding.interactions)
        or result.receipt_bytes is not binding.receipt_bytes
        or not hmac.compare_digest(
            binding.receipt_sha256,
            hashlib.sha256(result.receipt_bytes).hexdigest(),
        )
        or result.production_runtime is not binding.production_runtime
        or binding.production_runtime is not True
    ):
        raise FullRainbowRunnerContractError(
            "production result lacks its exact process-local completion capability"
        )


def validate_full_rainbow_runner_result(
    result: object,
) -> FullRainbowRunnerResult:
    if type(result) is not FullRainbowRunnerResult:
        raise FullRainbowRunnerContractError(
            "result must be an exact FullRainbowRunnerResult"
        )
    if (
        type(result.raw_reward_trace) is not bytes
        or type(result.receipt_bytes) is not bytes
    ):
        raise FullRainbowRunnerContractError(
            "result trace and receipt must be exact bytes"
        )
    if (
        type(result.production_runtime) is not bool
        or type(result.interactions) is not int
        or result.interactions <= 0
        or type(result.cumulative_raw_score) is not int
    ):
        raise FullRainbowRunnerContractError("result scalar fields are invalid")
    with _PRODUCTION_REGISTRY_LOCK:
        registered_production_result = result in _PRODUCTION_RESULT_REGISTRY
    if result.production_runtime or registered_production_result:
        _validate_production_result_capability(result)
    receipt = (
        parse_full_rainbow_result_receipt(result.receipt_bytes)
        if result.production_runtime
        else parse_full_rainbow_engineering_receipt(result.receipt_bytes)
    )
    score = cast(dict[str, Any], receipt["score"])
    rewards = np.frombuffer(result.raw_reward_trace, dtype=np.int8)
    if rewards.size != result.interactions or result.interactions != score[
        "raw_reward_trace_length"
    ]:
        raise FullRainbowRunnerContractError("result trace length drifted")
    if not bool(np.all(np.isin(rewards, (-1, 0, 1, 30)))):
        raise FullRainbowRunnerContractError("result trace contains an invalid raw reward")
    if (
        int(np.sum(rewards, dtype=np.int64)) != result.cumulative_raw_score
        or result.cumulative_raw_score != score["raw_cumulative_score"]
        or hashlib.sha256(result.raw_reward_trace).hexdigest()
        != score["raw_reward_trace_sha256"]
    ):
        raise FullRainbowRunnerContractError("result trace identity or raw score drifted")
    return result


def canonical_full_rainbow_engineering_receipt_bytes(
    result: FullRainbowRunnerResult,
) -> bytes:
    validate_full_rainbow_runner_result(result)
    if result.production_runtime:
        raise FullRainbowRunnerContractError(
            "production result cannot be emitted as an engineering receipt"
        )
    return bytes(result.receipt_bytes)


def canonical_full_rainbow_result_receipt_bytes(
    result: FullRainbowRunnerResult,
) -> bytes:
    validate_full_rainbow_runner_result(result)
    if not result.production_runtime:
        raise FullRainbowRunnerContractError(
            "engineering result cannot be emitted as a production receipt"
        )
    return bytes(result.receipt_bytes)


def _require_execution_flag(value: object) -> None:
    if type(value) is not bool or not value:
        raise FullRainbowRunnerExecutionBlockedError(
            "Full Rainbow execution requires explicit unqualified_engineering=True; "
            "it grants no readiness, qualification, or authority"
        )


def _validate_production_dependencies(
    dependencies: FullRainbowRunnerDependencies,
) -> None:
    with _PRODUCTION_REGISTRY_LOCK:
        binding = _PRODUCTION_DEPENDENCY_REGISTRY.get(dependencies)
    if (
        binding is None
        or dependencies.dependency_identity != binding.dependency_identity
        or dependencies.dependency_identity
        != "production_bridge_and_compiled_full_rainbow_v1"
        or dependencies.environment_runtime is not binding.environment_runtime
        or type(dependencies.environment_runtime)
        is not bridge.MatchedV3ForagaxRuntime
        or dependencies.step_environment is not binding.step_environment
        or dependencies.step_environment is not bridge.step_matched_v3_foragax_bridge
        or dependencies.initialize_core is not binding.initialize_core
        or dependencies.action_q_values is not binding.action_q_values
        or dependencies.update_core is not binding.update_core
        or dependencies.sync_target is not binding.sync_target
        or dependencies.sync_target is not core.sync_full_rainbow_target
        or dependencies.runtime_identity is not binding.runtime_identity
        or dependencies.compiled_action_kernel is not binding.compiled_action_kernel
        or dependencies.compiled_action_kernel is not True
        or dependencies.compiled_update_kernel is not binding.compiled_update_kernel
        or dependencies.compiled_update_kernel is not True
    ):
        raise FullRainbowRunnerContractError(
            "production dependency capability binding drifted"
        )
    runtime_identity_sha256 = _canonical_sha256(dict(dependencies.runtime_identity))
    if not hmac.compare_digest(
        binding.runtime_identity_sha256, runtime_identity_sha256
    ):
        raise FullRainbowRunnerContractError(
            "production dependency capability binding drifted"
        )


def _run_full_rainbow(
    *,
    environment_seed: int,
    agent_seed: int,
    schedule: FullRainbowRunnerSchedule,
    dependencies: FullRainbowRunnerDependencies,
    production_runtime: bool,
    unqualified_engineering: bool = False,
) -> FullRainbowRunnerResult:
    """Run one complete schedule after its classification has been fixed."""

    _require_execution_flag(unqualified_engineering)
    if type(schedule) is not FullRainbowRunnerSchedule:
        raise FullRainbowRunnerContractError(
            "schedule must be an exact FullRainbowRunnerSchedule"
        )
    if type(dependencies) is not FullRainbowRunnerDependencies:
        raise FullRainbowRunnerContractError(
            "dependencies must be exact FullRainbowRunnerDependencies"
        )
    if type(production_runtime) is not bool:
        raise FullRainbowRunnerContractError(
            "production_runtime must be an exact internal boolean"
        )
    if production_runtime:
        _validate_production_dependencies(dependencies)
        if schedule != production_full_rainbow_schedule():
            raise FullRainbowRunnerContractError(
                "production runtime requires the exact production schedule"
            )
    try:
        roots = core.full_rainbow_seed_roots(
            environment_seed=environment_seed,
            agent_seed=agent_seed,
        )
    except core.FullRainbowContractError as exc:
        raise FullRainbowRunnerContractError("runner seeds must be exact uint31") from exc
    config = core.FullRainbowForagerConfig()
    environment_state = dependencies.environment_runtime.initialize(environment_seed)
    observation = _validate_environment_state(
        environment_state,
        environment_seed=environment_seed,
        expected_step_count=0,
    )
    core_state = _validate_initial_core_state(
        dependencies.initialize_core(config, environment_seed, agent_seed),
        environment_seed=environment_seed,
        agent_seed=agent_seed,
    )
    frozen_environment_rng = core_state.environment_rng
    environment_root_words = _validate_key(
        roots.environment, label="frozen core environment root"
    )
    replay = CompactPrioritizedReplay(
        capacity=schedule.replay_capacity,
        batch_size=schedule.batch_size,
        observation_shape=config.observation_shape,
        page_size=schedule.page_size,
    )
    accumulator = FullRainbowThreeStepAccumulator(config)
    raw_trace = bytearray()
    cumulative_score = 0
    update_transitions: list[int] = []
    sync_transitions: list[int] = []

    for transition_number in range(1, schedule.horizon + 1):
        next_agent, uniform_key, noise_key, randint_key = jr.split(
            core_state.agent_rng, 4
        )
        q_values = dependencies.action_q_values(
            config, core_state, observation, noise_key
        )
        uniform_draw = jr.uniform(uniform_key, (), dtype=jnp.float32)
        random_action = jr.randint(
            randint_key, (), 0, config.num_actions, dtype=jnp.int32
        )
        action = resolve_zero_epsilon_action(
            q_values,
            uniform_draw=uniform_draw,
            random_action=random_action,
        )
        core_state = replace(core_state, agent_rng=next_agent)
        if (
            _validate_key(core_state.environment_rng, label="core environment root")
            != environment_root_words
            or core_state.environment_rng is not frozen_environment_rng
        ):
            raise FullRainbowRunnerContractError(
                "action selection consumed or changed the core environment root"
            )

        raw_transition = dependencies.step_environment(environment_state, action)
        next_environment_state, next_observation, reward = _validate_transition(
            raw_transition,
            previous_state=environment_state,
            environment_seed=environment_seed,
            expected_step_count=transition_number,
            action=action,
        )
        raw_trace.append(reward & 0xFF)
        cumulative_score += reward
        replay_value = accumulator.accumulate(
            observation=observation,
            action=action,
            raw_reward=reward,
            done=False,
            truncated=False,
        )
        if replay_value is not None:
            if replay_value.available_after_transition != transition_number:
                raise FullRainbowRunnerContractError(
                    "accumulator availability accounting drifted"
                )
            replay.add(replay_value)

        if (
            replay.insertions > schedule.minimum_replay_history
            and transition_number % schedule.update_period == 0
        ):
            next_agent, sample_key = jr.split(core_state.agent_rng)
            core_state = replace(core_state, agent_rng=next_agent)
            draw = replay.sample(sample_key)
            expected_weights = core.importance_sampling_weights(
                draw.batch.sampling_probabilities
            )
            if not bool(
                np.array_equal(
                    np.asarray(draw.importance_weights), np.asarray(expected_weights)
                )
            ):
                raise FullRainbowRunnerContractError(
                    "replay importance weights drifted from the core contract"
                )
            before_update = core_state
            _, _, expected_next_agent = jr.split(before_update.agent_rng, 3)
            updated_state, metrics = dependencies.update_core(
                config, before_update, draw.batch
            )
            if type(updated_state) is not core.FullRainbowCoreState:
                raise FullRainbowRunnerContractError(
                    "update_core must return exact FullRainbowCoreState"
                )
            if (
                _validate_key(
                    updated_state.environment_rng, label="updated core environment root"
                )
                != environment_root_words
                or updated_state.environment_rng is not frozen_environment_rng
                or _validate_key(
                    updated_state.agent_rng, label="updated core agent continuation"
                )
                != _validate_key(
                    expected_next_agent, label="expected update agent continuation"
                )
                or type(updated_state.optimizer_updates) is not int
                or updated_state.optimizer_updates
                != before_update.optimizer_updates + 1
                or updated_state.target_params is not before_update.target_params
                or (
                    not production_runtime
                    and (
                        not _tree_finite(updated_state.online_params)
                        or not _tree_finite(updated_state.optimizer_state)
                    )
                )
            ):
                raise FullRainbowRunnerContractError(
                    "update_core violated RNG, target, or optimizer accounting"
                )
            metrics = _validate_metrics(metrics, batch_size=schedule.batch_size)
            if not bool(
                np.array_equal(
                    np.asarray(metrics.importance_weights),
                    np.asarray(draw.importance_weights),
                )
            ):
                raise FullRainbowRunnerContractError(
                    "update metrics importance weights drifted from replay"
                )
            replay.update_priorities(draw.indices, metrics.updated_priorities)
            core_state = updated_state
            update_transitions.append(transition_number)

        if (
            replay.insertions > schedule.minimum_replay_history
            and transition_number % schedule.target_update_period == 0
        ):
            before_sync = core_state
            synced = dependencies.sync_target(before_sync)
            if (
                type(synced) is not core.FullRainbowCoreState
                or _validate_key(
                    synced.environment_rng, label="synced core environment root"
                )
                != environment_root_words
                or synced.environment_rng is not frozen_environment_rng
                or _validate_key(synced.agent_rng, label="synced core agent key")
                != _validate_key(before_sync.agent_rng, label="pre-sync core agent key")
                or synced.agent_rng is not before_sync.agent_rng
                or synced.optimizer_updates != before_sync.optimizer_updates
                or synced.online_params is not before_sync.online_params
                or synced.target_params is not synced.online_params
                or synced.optimizer_state is not before_sync.optimizer_state
                or not _tree_finite(synced.online_params)
                or not _tree_finite(synced.target_params)
                or not _tree_finite(synced.optimizer_state)
            ):
                raise FullRainbowRunnerContractError(
                    "target sync violated parameter, RNG, or update accounting"
                )
            core_state = synced
            sync_transitions.append(transition_number)

        environment_state = next_environment_state
        observation = next_observation

    expected = full_rainbow_schedule_accounting(schedule)
    accounting: dict[str, Any] = {
        **expected,
        "update_transitions": update_transitions,
        "target_sync_transitions": sync_transitions,
    }
    if (
        accumulator.emitted_transition_count != expected["replay_insertions"]
        or replay.insertions != expected["replay_insertions"]
        or replay.size != expected["maximum_replay_residency"]
        or replay.evictions != expected["replay_evictions"]
        or core_state.optimizer_updates != expected["optimizer_updates"]
        or len(update_transitions) != expected["optimizer_updates"]
        or len(sync_transitions) != expected["target_syncs"]
    ):
        raise FullRainbowRunnerContractError(
            "completed driver accounting differs from the exact schedule"
        )
    updates = expected["optimizer_updates"]
    if type(updates) is not int:
        raise AssertionError("optimizer update accounting type drifted")
    rng_accounting = {
        "agent_continuation_split_calls": 1 + schedule.horizon + 2 * updates,
        "agent_parameter_initialization_subkeys": 2,
        "agent_action_split_calls": schedule.horizon,
        "agent_action_subkeys_produced_per_call": 3,
        "agent_replay_sampling_split_calls": updates,
        "agent_update_split_calls": updates,
        "agent_update_subkeys_produced_per_call": 2,
        "core_environment_key_consumptions": 0,
        "bridge_environment_key_consumptions": schedule.horizon + 1,
    }
    if production_runtime:
        _validate_production_dependencies(dependencies)
    trace_bytes = bytes(raw_trace)
    receipt_bytes = _receipt(
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        schedule=schedule,
        dependencies=dependencies,
        raw_trace=trace_bytes,
        cumulative_score=cumulative_score,
        accounting=accounting,
        rng_accounting=rng_accounting,
        production_runtime=production_runtime,
    )
    result = FullRainbowRunnerResult(
        raw_reward_trace=trace_bytes,
        cumulative_raw_score=cumulative_score,
        interactions=schedule.horizon,
        receipt_bytes=receipt_bytes,
        production_runtime=production_runtime,
    )
    if production_runtime:
        with _PRODUCTION_REGISTRY_LOCK:
            _PRODUCTION_RESULT_REGISTRY[result] = _ProductionResultBinding(
                raw_reward_trace=trace_bytes,
                raw_reward_trace_sha256=hashlib.sha256(trace_bytes).hexdigest(),
                cumulative_raw_score=cumulative_score,
                interactions=schedule.horizon,
                receipt_bytes=receipt_bytes,
                receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
                production_runtime=True,
            )
    try:
        return validate_full_rainbow_runner_result(result)
    except Exception:
        if production_runtime:
            with _PRODUCTION_REGISTRY_LOCK:
                _PRODUCTION_RESULT_REGISTRY.pop(result, None)
        raise


def run_full_rainbow_engineering(
    *,
    environment_seed: int,
    agent_seed: int,
    schedule: FullRainbowRunnerSchedule,
    dependencies: FullRainbowRunnerDependencies,
    unqualified_engineering: bool = False,
) -> FullRainbowRunnerResult:
    """Run injected dependencies, permanently classified as engineering-only."""

    return _run_full_rainbow(
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        schedule=schedule,
        dependencies=dependencies,
        production_runtime=False,
        unqualified_engineering=unqualified_engineering,
    )


def _production_dependencies() -> FullRainbowRunnerDependencies:
    """Build the real bridge and compiled action/update numerical boundaries."""

    config = core.FullRainbowForagerConfig()
    environment_runtime = bridge.open_matched_v3_foragax_runtime()

    @jax.jit
    def compiled_action(params: Any, observation: Array, noise_key: Array) -> Array:
        return core.apply_full_rainbow_network(
            config,
            params,
            observation,
            noise_key,
            eval_mode=False,
        ).q_values

    optimizer = optax.adam(
        config.learning_rate,
        b1=config.adam_beta1,
        b2=config.adam_beta2,
        eps=config.adam_epsilon,
    )

    @jax.jit
    def compiled_update(
        online_params: Any,
        target_params: Any,
        optimizer_state: Any,
        agent_rng: Array,
        states: Array,
        actions: Array,
        next_states: Array,
        rewards: Array,
        discounts: Array,
        probabilities: Array,
    ) -> tuple[Any, Any, Array, tuple[Array, Array, Array, Array, Array], Array]:
        current_key, target_key, next_agent_rng = jr.split(agent_rng, 3)
        target_output = core._batched_apply(
            config, target_params, next_states, target_key, eval_mode=False
        )
        online_next_output = core._batched_apply(
            config, online_params, next_states, target_key, eval_mode=False
        )
        double_q_actions = jnp.argmax(online_next_output.q_values, axis=1)
        chosen_target_probabilities = jnp.take_along_axis(
            target_output.probabilities,
            double_q_actions[:, None, None],
            axis=1,
        )[:, 0, :]
        support = core.frozen_support(config)
        target_atoms = rewards[:, None] + discounts[:, None] * support[None, :]
        targets = jax.vmap(core._project_c51, in_axes=(0, 0, None))(
            target_atoms, chosen_target_probabilities, support
        )
        targets = jax.lax.stop_gradient(targets)
        weights = 1.0 / jnp.sqrt(probabilities + _PRIORITY_EPSILON)
        weights = weights / jnp.max(weights)

        def loss_function(params: Any) -> tuple[Array, Array]:
            output = core._batched_apply(
                config, params, states, current_key, eval_mode=False
            )
            chosen_logits = jnp.take_along_axis(
                output.logits, actions[:, None, None], axis=1
            )[:, 0, :]
            losses = -jnp.sum(targets * jax.nn.log_softmax(chosen_logits), axis=1)
            return jnp.mean(weights * losses), losses

        (mean_loss, losses), gradients = jax.value_and_grad(
            loss_function, has_aux=True
        )(online_params)
        updates, next_optimizer_state = optimizer.update(
            gradients, optimizer_state, params=online_params
        )
        next_online_params = optax.apply_updates(online_params, updates)
        priorities = jnp.sqrt(losses + _PRIORITY_EPSILON)
        finite_leaves = [
            jnp.all(jnp.isfinite(leaf))
            for leaf in jax.tree_util.tree_leaves(
                (next_online_params, next_optimizer_state)
            )
        ]
        finite = jnp.all(jnp.stack(finite_leaves)) & jnp.all(
            jnp.isfinite(
                jnp.concatenate(
                    (
                        jnp.ravel(mean_loss),
                        jnp.ravel(losses),
                        jnp.ravel(weights),
                        jnp.ravel(priorities),
                    )
                )
            )
        )
        metrics = (mean_loss, losses, weights, priorities, double_q_actions)
        return next_online_params, next_optimizer_state, next_agent_rng, metrics, finite

    def initialize_core(
        exact_config: core.FullRainbowForagerConfig,
        environment_seed: int,
        agent_seed: int,
    ) -> core.FullRainbowCoreState:
        return core.initialize_full_rainbow_core(
            exact_config,
            environment_seed=environment_seed,
            agent_seed=agent_seed,
        )

    def action_q_values(
        exact_config: core.FullRainbowForagerConfig,
        state: core.FullRainbowCoreState,
        observation: Array,
        noise_key: Array,
    ) -> Array:
        if exact_config != config:
            raise FullRainbowRunnerContractError("production action config drifted")
        return cast(Array, compiled_action(state.online_params, observation, noise_key))

    def update_core(
        exact_config: core.FullRainbowForagerConfig,
        state: core.FullRainbowCoreState,
        batch: core.FullRainbowReplayBatch,
    ) -> tuple[core.FullRainbowCoreState, core.FullRainbowTrainMetrics]:
        if exact_config != config:
            raise FullRainbowRunnerContractError("production update config drifted")
        core.validate_replay_batch(config, batch)
        online_params, optimizer_state, agent_rng, values, finite = compiled_update(
            state.online_params,
            state.target_params,
            state.optimizer_state,
            state.agent_rng,
            batch.states,
            batch.actions,
            batch.next_states,
            batch.scaled_n_step_rewards,
            batch.bootstrap_discounts,
            batch.sampling_probabilities,
        )
        if not bool(np.asarray(finite)):
            raise FullRainbowRunnerContractError(
                "compiled update produced a non-finite parameter, optimizer, or metric"
            )
        mean, losses, weights, priorities, actions = values
        updated = core.FullRainbowCoreState(
            online_params=online_params,
            target_params=state.target_params,
            optimizer_state=optimizer_state,
            environment_rng=state.environment_rng,
            agent_rng=agent_rng,
            optimizer_updates=state.optimizer_updates + 1,
        )
        metrics = core.FullRainbowTrainMetrics(
            mean_weighted_loss=mean,
            per_example_loss=losses,
            importance_weights=weights,
            updated_priorities=priorities,
            double_q_actions=actions,
        )
        return updated, metrics

    runtime_identity = asdict(environment_runtime.runtime_identity)
    runtime_identity["foragax_runtime_parity_executed"] = False
    dependencies = FullRainbowRunnerDependencies(
        dependency_identity="production_bridge_and_compiled_full_rainbow_v1",
        environment_runtime=environment_runtime,
        step_environment=bridge.step_matched_v3_foragax_bridge,
        initialize_core=initialize_core,
        action_q_values=action_q_values,
        update_core=update_core,
        sync_target=core.sync_full_rainbow_target,
        runtime_identity=runtime_identity,
        compiled_action_kernel=True,
        compiled_update_kernel=True,
    )
    binding = _ProductionDependencyBinding(
        dependency_identity=dependencies.dependency_identity,
        environment_runtime=dependencies.environment_runtime,
        step_environment=dependencies.step_environment,
        initialize_core=dependencies.initialize_core,
        action_q_values=dependencies.action_q_values,
        update_core=dependencies.update_core,
        sync_target=dependencies.sync_target,
        runtime_identity=dependencies.runtime_identity,
        runtime_identity_sha256=_canonical_sha256(dict(dependencies.runtime_identity)),
        compiled_action_kernel=dependencies.compiled_action_kernel,
        compiled_update_kernel=dependencies.compiled_update_kernel,
    )
    with _PRODUCTION_REGISTRY_LOCK:
        _PRODUCTION_DEPENDENCY_REGISTRY[dependencies] = binding
    return dependencies


def run_matched_v3_full_rainbow(
    *,
    environment_seed: int,
    agent_seed: int,
    unqualified_engineering: bool = False,
) -> FullRainbowRunnerResult:
    """Run the exact production horizon only after explicit unqualified opt-in."""

    _require_execution_flag(unqualified_engineering)
    dependencies = _production_dependencies()
    return _run_full_rainbow(
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        schedule=production_full_rainbow_schedule(),
        dependencies=dependencies,
        production_runtime=True,
        unqualified_engineering=True,
    )


__all__ = [
    "BOUND_BRIDGE_DESCRIPTOR_SHA256",
    "BOUND_BRIDGE_IMPLEMENTATION_PATH",
    "BOUND_BRIDGE_IMPLEMENTATION_SHA256",
    "BOUND_CORE_CONFIG_SHA256",
    "BOUND_CORE_DESCRIPTOR_SHA256",
    "BOUND_CORE_IMPLEMENTATION_PATH",
    "BOUND_CORE_IMPLEMENTATION_SHA256",
    "CompactPrioritizedReplay",
    "FULL_RAINBOW_ENGINEERING_RECEIPT_SCHEMA_VERSION",
    "FULL_RAINBOW_RESULT_RECEIPT_SCHEMA_VERSION",
    "FULL_RAINBOW_RUNNER_DESCRIPTOR_SCHEMA_VERSION",
    "FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256",
    "FULL_RAINBOW_RUNNER_STATUS",
    "FullRainbowAccumulatedTransition",
    "FullRainbowReplayDraw",
    "FullRainbowRunnerContractError",
    "FullRainbowRunnerDependencies",
    "FullRainbowRunnerExecutionBlockedError",
    "FullRainbowRunnerResult",
    "FullRainbowRunnerSchedule",
    "FullRainbowThreeStepAccumulator",
    "canonical_full_rainbow_engineering_receipt_bytes",
    "canonical_full_rainbow_result_receipt_bytes",
    "canonical_full_rainbow_receipt_body_bytes",
    "canonical_full_rainbow_runner_descriptor_bytes",
    "full_rainbow_runner_descriptor",
    "full_rainbow_schedule_accounting",
    "parse_full_rainbow_engineering_receipt",
    "parse_full_rainbow_result_receipt",
    "parse_full_rainbow_runner_descriptor",
    "production_full_rainbow_schedule",
    "resolve_zero_epsilon_action",
    "run_full_rainbow_engineering",
    "run_matched_v3_full_rainbow",
    "validate_full_rainbow_runner_result",
]
