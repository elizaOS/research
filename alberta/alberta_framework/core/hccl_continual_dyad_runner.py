# mypy: disable-error-code="attr-defined,call-arg"
"""Explicit bounded host runner for the primitive-only HCCL continual dyad.

The runner is the production-owned execution boundary above
:mod:`hccl_continual_dyad_factory`.  A call to :meth:`HCCLContinualDyadRunner.run`
initializes a fresh dyad from caller key material and executes exactly one fixed
420-event mechanics-smoke, 8,998-event Core-L1, 71,984-event Core-L2, or
1,007,776-event Core-L3 life.  It never invokes a reset or regime-boundary
callback, including between the uninterrupted canonical-length longevity
cycles.

The four all-regime score columns are evaluator-only readouts.  They are
computed from the already committed PP world factors *after* atomic adoption;
neither those columns nor the evaluator regime identifier are passed to an
agent.  A rejection, malformed result, clock discontinuity, or non-finite
readout aborts the call without returning a partial trace.

This is an eager L0 development runner.  It has no CLI, writer, artifact,
threshold, benchmark, evidence, seed-reservation, or promotion authority.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from alberta_framework.core.hccl_continual_dyad_factory import (
    HCCLContinualDyadFactory,
    HCCLContinualDyadFactoryConfig,
    HCCLContinualDyadFactoryInitialization,
)
from alberta_framework.core.hccl_continual_dyad_transaction import (
    HCCLContinualDyadPreparationReceipt,
    HCCLContinualDyadPreparedTransaction,
    HCCLContinualDyadResult,
    HCCLContinualDyadState,
    HCCLContinualDyadTransaction,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
    HCCL_CAUSAL_CORE_L2_PROFILE,
    HCCL_CAUSAL_CORE_L3_PROFILE,
    HCCL_CAUSAL_CORE_REGIME_NAMES,
    HCCL_CAUSAL_CORE_SMOKE_PROFILE,
    HCCLCausalCoreFactors,
    hccl_causal_core_lifetime_for_profile,
    hccl_causal_core_schedule_for_profile,
)

if TYPE_CHECKING:
    from alberta_framework.evaluation.hccl_causal_core_endpoints import (
        HCCLCausalCoreCompleteTrace,
    )

HCCL_CONTINUAL_DYAD_RUNNER_CONFIG_SCHEMA: Final = (
    "alberta.hccl-continual-dyad-runner.config.v1"
)
HCCL_CONTINUAL_DYAD_LIFE_TRACE_SCHEMA: Final = (
    "alberta.hccl-continual-dyad-runner.life-trace.v1"
)
HCCL_CONTINUAL_DYAD_RUNNER_STATUS: Final = (
    "l0-development-bounded-primitive-only-life-runner"
)
HCCL_CONTINUAL_DYAD_RUNNER_EVIDENCE_LEVEL: Final = "L0"
HCCL_CONTINUAL_DYAD_RUNNER_LIMITATIONS: Final = (
    "host-eager-only",
    "primitive-only-v2-dyad-not-full-generated-feature-consumer-routing",
    "mechanics-smoke-is-not-canonical-endpoint-compatible",
    "caller-key-material-is-not-a-reserved-consumed-or-held-out-seed",
    "complete-in-memory-trace-only",
    "no-cli-writer-artifact-threshold-benchmark-evidence-or-promotion-authority",
)

_N_AGENTS: Final = 2
_N_ACTIONS: Final = 2
_N_REGIMES: Final = 4
_PP_SLOT: Final = 4
_SUPPORTED_PROFILES: Final = (
    HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
    HCCL_CAUSAL_CORE_SMOKE_PROFILE,
    HCCL_CAUSAL_CORE_L2_PROFILE,
    HCCL_CAUSAL_CORE_L3_PROFILE,
)

type Float32Array = NDArray[np.float32]
type Int32Array = NDArray[np.int32]
type UInt32Array = NDArray[np.uint32]
type BoolArray = NDArray[np.bool_]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _profile_steps(profile: str) -> int:
    if profile in _SUPPORTED_PROFILES:
        return hccl_causal_core_lifetime_for_profile(profile)
    raise ValueError("schedule_profile must select one fixed HCCL life")


def _profile_schedule(profile: str) -> tuple[tuple[str, int, int], ...]:
    if profile in _SUPPORTED_PROFILES:
        return hccl_causal_core_schedule_for_profile(profile)
    raise ValueError("schedule_profile must select one fixed HCCL life")


def _profile_for_steps(steps: int) -> str:
    if type(steps) is not int:
        raise ValueError("total_steps must select one exact bounded life")
    for profile in _SUPPORTED_PROFILES:
        if _profile_steps(profile) == steps:
            return profile
    raise ValueError("total_steps must select one exact bounded life")


def _expected_regime_ids(profile: str) -> Int32Array:
    steps = _profile_steps(profile)
    result = np.empty((steps,), dtype=np.int32)
    for regime_name, start, end in _profile_schedule(profile):
        result[start:end] = HCCL_CAUSAL_CORE_REGIME_NAMES.index(regime_name)
    return result


def _frozen_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> NDArray[Any]:
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be an exact numpy.ndarray")
    array = cast(NDArray[Any], value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def _host_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> NDArray[Any]:
    try:
        result = np.asarray(jax.device_get(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not a host-convertible array") from error
    if result.shape != shape or result.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    return result


def _host_true(value: object, *, name: str) -> bool:
    result = _host_array(
        value,
        name=name,
        shape=(),
        dtype=np.dtype(np.bool_),
    )
    if not bool(result):
        raise ValueError(f"{name} must be true")
    return True


def _host_all_true(value: object, *, name: str, shape: tuple[int, ...]) -> bool:
    result = _host_array(
        value,
        name=name,
        shape=shape,
        dtype=np.dtype(np.bool_),
    )
    if not bool(np.all(result)):
        raise ValueError(f"every {name} entry must be true")
    return True


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLContinualDyadRunnerConfig:
    """Select one exact factory-owned complete life; partial runs are unsupported."""

    factory_config: HCCLContinualDyadFactoryConfig = dataclasses.field(
        default_factory=HCCLContinualDyadFactoryConfig
    )

    def __post_init__(self) -> None:
        if type(self.factory_config) is not HCCLContinualDyadFactoryConfig:
            raise TypeError("factory_config must be exact HCCLContinualDyadFactoryConfig")

    @classmethod
    def mechanics_smoke(cls) -> HCCLContinualDyadRunnerConfig:
        """Select the exact 420-event mechanics-only life."""

        return cls(factory_config=HCCLContinualDyadFactoryConfig.mechanics_smoke())

    @classmethod
    def core_l2(cls) -> HCCLContinualDyadRunnerConfig:
        """Select the uninterrupted eight-cycle 71,984-event Core-L2 life."""

        return cls(factory_config=HCCLContinualDyadFactoryConfig.core_l2())

    @classmethod
    def core_l3(cls) -> HCCLContinualDyadRunnerConfig:
        """Select the uninterrupted 112-cycle 1,007,776-event Core-L3 life."""

        return cls(factory_config=HCCLContinualDyadFactoryConfig.core_l3())

    @property
    def schedule_profile(self) -> str:
        return self.factory_config.schedule_profile

    @property
    def total_steps(self) -> int:
        return self.factory_config.maximum_committed_transitions

    @property
    def canonical_endpoint_compatible(self) -> bool:
        return self.schedule_profile == HCCL_CAUSAL_CORE_CANONICAL_PROFILE

    @property
    def mechanics_smoke_only(self) -> bool:
        return self.schedule_profile == HCCL_CAUSAL_CORE_SMOKE_PROFILE

    @property
    def longevity_life(self) -> bool:
        return self.schedule_profile in (
            HCCL_CAUSAL_CORE_L2_PROFILE,
            HCCL_CAUSAL_CORE_L3_PROFILE,
        )

    def to_config(self) -> dict[str, object]:
        """Return the complete bounded-execution contract and explicit nonclaims."""

        return {
            "type": type(self).__name__,
            "schema": HCCL_CONTINUAL_DYAD_RUNNER_CONFIG_SCHEMA,
            "trace_schema": HCCL_CONTINUAL_DYAD_LIFE_TRACE_SCHEMA,
            "status": HCCL_CONTINUAL_DYAD_RUNNER_STATUS,
            "evidence_level": HCCL_CONTINUAL_DYAD_RUNNER_EVIDENCE_LEVEL,
            "factory_config": self.factory_config.to_config(),
            "schedule_profile": self.schedule_profile,
            "total_steps": self.total_steps,
            "complete_fixed_life_only": True,
            "partial_life_supported": False,
            "fresh_factory_initialization_per_run": True,
            "explicit_run_call_required": True,
            "bounded_development_life_execution_authorized": True,
            "factory_initialization_contract_itself_authorizes_execution": False,
            "reset_callback_count": 0,
            "boundary_callback_count": 0,
            "abort_on_first_rejection_or_malformed_event": True,
            "partial_trace_returned_on_failure": False,
            "evaluator_columns_computed_after_atomic_adoption": True,
            "evaluator_labels_exposed_to_learner": False,
            "counterfactual_score_columns_exposed_to_learner": False,
            "canonical_endpoint_compatible": self.canonical_endpoint_compatible,
            "mechanics_smoke_only": self.mechanics_smoke_only,
            "caller_key_material_used": True,
            "protocol_seed_reservation_or_consumption_authorized": False,
            "benchmark_execution_authorized": False,
            "output_writes_authorized": False,
            "artifact_authorized": False,
            "threshold_authorized": False,
            "evidence_authorized": False,
            "promotion_authorized": False,
            "scientific_promotion_allowed": False,
            "artifact_write_calls": 0,
            "artifact_bytes_written": 0,
            "limitations": list(HCCL_CONTINUAL_DYAD_RUNNER_LIMITATIONS),
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> HCCLContinualDyadRunnerConfig:
        """Reconstruct only an exact current runner manifest."""

        if type(payload) is not dict:
            raise TypeError("runner config must be an exact dict")
        nested = payload.get("factory_config")
        if type(nested) is not dict:
            raise ValueError("runner factory_config must be an exact dict")
        factory_config = HCCLContinualDyadFactoryConfig.from_config(nested)
        candidate = cls(factory_config=factory_config)
        if _canonical_json_bytes(payload) != _canonical_json_bytes(candidate.to_config()):
            raise ValueError("runner config is noncanonical or unsupported")
        return candidate


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLContinualDyadLifeTrace:
    """One complete in-memory fixed-profile primitive-only dyad life."""

    schedule_profile: str
    regime_ids: Int32Array
    transaction_committed: BoolArray
    pre_step_words: UInt32Array
    post_step_words: UInt32Array
    task_scores: Float32Array
    net_rewards: Float32Array
    all_regime_score_matrix: Float32Array
    reset_callback_count: int = 0
    boundary_callback_count: int = 0
    learner_received_evaluator_regime_ids: bool = False
    learner_received_counterfactual_scores: bool = False
    schema: str = HCCL_CONTINUAL_DYAD_LIFE_TRACE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.schedule_profile) is not str:
            raise TypeError("schedule_profile must be an exact string")
        steps = _profile_steps(self.schedule_profile)
        specifications = (
            ("regime_ids", (steps,), np.dtype(np.int32)),
            ("transaction_committed", (steps,), np.dtype(np.bool_)),
            ("pre_step_words", (steps, 2), np.dtype(np.uint32)),
            ("post_step_words", (steps, 2), np.dtype(np.uint32)),
            ("task_scores", (steps,), np.dtype(np.float32)),
            ("net_rewards", (steps, _N_AGENTS), np.dtype(np.float32)),
            (
                "all_regime_score_matrix",
                (steps, _N_REGIMES),
                np.dtype(np.float32),
            ),
        )
        for name, shape, dtype in specifications:
            object.__setattr__(
                self,
                name,
                _frozen_array(getattr(self, name), name=name, shape=shape, dtype=dtype),
            )
        self._validate()

    @property
    def total_steps(self) -> int:
        return _profile_steps(self.schedule_profile)

    @property
    def canonical_endpoint_compatible(self) -> bool:
        return self.schedule_profile == HCCL_CAUSAL_CORE_CANONICAL_PROFILE

    @property
    def longevity_life(self) -> bool:
        return self.schedule_profile in (
            HCCL_CAUSAL_CORE_L2_PROFILE,
            HCCL_CAUSAL_CORE_L3_PROFILE,
        )

    def _validate(self) -> None:
        steps = self.total_steps
        if type(self.schema) is not str or self.schema != HCCL_CONTINUAL_DYAD_LIFE_TRACE_SCHEMA:
            raise ValueError("life trace schema differs from the current schema")
        for name, shape, dtype in (
            ("regime_ids", (steps,), np.dtype(np.int32)),
            ("transaction_committed", (steps,), np.dtype(np.bool_)),
            ("pre_step_words", (steps, 2), np.dtype(np.uint32)),
            ("post_step_words", (steps, 2), np.dtype(np.uint32)),
            ("task_scores", (steps,), np.dtype(np.float32)),
            ("net_rewards", (steps, _N_AGENTS), np.dtype(np.float32)),
            (
                "all_regime_score_matrix",
                (steps, _N_REGIMES),
                np.dtype(np.float32),
            ),
        ):
            value = getattr(self, name)
            if type(value) is not np.ndarray:
                raise TypeError(f"{name} must remain an exact numpy.ndarray")
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f"{name} no longer has its exact shape and dtype")
            if value.flags.writeable or not value.flags.c_contiguous:
                raise ValueError(f"{name} must remain read-only and C-contiguous")
        for name in ("reset_callback_count", "boundary_callback_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) != 0:
                raise ValueError(f"{name} must be the exact integer zero")
        for name in (
            "learner_received_evaluator_regime_ids",
            "learner_received_counterfactual_scores",
        ):
            if type(getattr(self, name)) is not bool or getattr(self, name):
                raise ValueError(f"{name} must be exact False")
        if not np.array_equal(self.regime_ids, _expected_regime_ids(self.schedule_profile)):
            raise ValueError("regime_ids do not equal the selected fixed schedule")
        if not bool(np.all(self.transaction_committed)):
            raise ValueError("every transaction in a complete life must be committed")

        expected_pre = np.zeros((steps, 2), dtype=np.uint32)
        expected_post = np.zeros((steps, 2), dtype=np.uint32)
        expected_pre[:, 1] = np.arange(steps, dtype=np.uint32)
        expected_post[:, 1] = np.arange(1, steps + 1, dtype=np.uint32)
        if not np.array_equal(self.pre_step_words, expected_pre):
            raise ValueError("pre_step_words are not the exact monotone life clocks")
        if not np.array_equal(self.post_step_words, expected_post):
            raise ValueError("post_step_words are not the exact monotone committed clocks")
        for name in ("task_scores", "net_rewards", "all_regime_score_matrix"):
            if not bool(np.all(np.isfinite(getattr(self, name)))):
                raise ValueError(f"{name} must be entirely finite")
        selected = self.all_regime_score_matrix[np.arange(steps), self.regime_ids]
        if not np.array_equal(selected, self.task_scores):
            raise ValueError("task scores must equal their selected evaluator columns exactly")
        expected_net = np.broadcast_to(self.task_scores[:, None], (steps, _N_AGENTS))
        if not np.array_equal(self.net_rewards, expected_net):
            raise ValueError("causal-core net rewards must exactly equal task score per agent")

    def to_canonical_endpoint_trace(self) -> HCCLCausalCoreCompleteTrace:
        """Convert a canonical life to the strict endpoint evaluator input."""

        validate_hccl_continual_dyad_life_trace(self)
        if not self.canonical_endpoint_compatible:
            raise ValueError("the selected life is not a canonical endpoint trace")
        from alberta_framework.evaluation.hccl_causal_core_endpoints import (
            HCCLCausalCoreCompleteTrace,
        )

        return HCCLCausalCoreCompleteTrace(
            regime_ids=self.regime_ids,
            transaction_committed=self.transaction_committed,
            pre_step_words=self.pre_step_words,
            post_step_words=self.post_step_words,
            task_scores=self.task_scores,
            net_rewards=self.net_rewards,
            all_regime_score_matrix=self.all_regime_score_matrix,
            reset_callback_count=self.reset_callback_count,
            boundary_callback_count=self.boundary_callback_count,
            learner_received_evaluator_regime_ids=(
                self.learner_received_evaluator_regime_ids
            ),
            learner_received_counterfactual_scores=(
                self.learner_received_counterfactual_scores
            ),
        )


def validate_hccl_continual_dyad_life_trace(
    trace: HCCLContinualDyadLifeTrace,
) -> HCCLContinualDyadLifeTrace:
    """Revalidate an exact trace, including its post-construction freeze status."""

    if type(trace) is not HCCLContinualDyadLifeTrace:
        raise TypeError("trace must be exact HCCLContinualDyadLifeTrace")
    trace._validate()
    return trace


class HCCLContinualDyadLifeError(RuntimeError):
    """A fail-closed event failure; no partial trace accompanies this exception."""

    def __init__(self, step_index: int, stage: str, detail: str) -> None:
        self.step_index = step_index
        self.stage = stage
        super().__init__(f"HCCL life aborted at step {step_index} during {stage}: {detail}")


@dataclasses.dataclass(frozen=True, slots=True)
class _CommittedEvent:
    regime_id: int
    committed: bool
    pre_step_words: UInt32Array
    post_step_words: UInt32Array
    task_score: np.float32
    net_rewards: Float32Array
    all_regime_scores: Float32Array

    def __post_init__(self) -> None:
        if type(self.regime_id) is not int or not 0 <= self.regime_id < _N_REGIMES:
            raise ValueError("committed event regime_id must be an exact evaluator id")
        if type(self.committed) is not bool or not self.committed:
            raise ValueError("committed event must be exact True")
        if type(self.task_score) is not np.float32 or not np.isfinite(self.task_score):
            raise ValueError("committed event task_score must be finite float32")
        for name, shape, dtype in (
            ("pre_step_words", (2,), np.dtype(np.uint32)),
            ("post_step_words", (2,), np.dtype(np.uint32)),
            ("net_rewards", (_N_AGENTS,), np.dtype(np.float32)),
            ("all_regime_scores", (_N_REGIMES,), np.dtype(np.float32)),
        ):
            object.__setattr__(
                self,
                name,
                _frozen_array(getattr(self, name), name=name, shape=shape, dtype=dtype),
            )
        if not bool(np.all(np.isfinite(self.net_rewards))) or not bool(
            np.all(np.isfinite(self.all_regime_scores))
        ):
            raise ValueError("committed event score readouts must be finite")
        if self.all_regime_scores[self.regime_id] != self.task_score:
            raise ValueError("committed event selected evaluator score differs from task score")
        if not np.array_equal(
            self.net_rewards,
            np.full((_N_AGENTS,), self.task_score, dtype=np.float32),
        ):
            raise ValueError("committed event net rewards differ from causal-core task score")


class _LifeEventExecutor(Protocol):
    @property
    def final_state(self) -> object: ...

    def execute_event(self, step_index: int) -> _CommittedEvent: ...


class _ProductionLifeEventExecutor:
    """One fresh transaction/state pair; evaluator readout follows adoption."""

    def __init__(
        self,
        transaction: HCCLContinualDyadTransaction,
        state: HCCLContinualDyadState,
        *,
        total_steps: int,
    ) -> None:
        if type(transaction) is not HCCLContinualDyadTransaction:
            raise TypeError("transaction must be exact HCCLContinualDyadTransaction")
        if type(state) is not HCCLContinualDyadState:
            raise TypeError("state must be exact HCCLContinualDyadState")
        expected_profile = _profile_for_steps(total_steps)
        world_config = transaction.config.hccl.world_config
        if (
            world_config.maximum_committed_transitions != total_steps
            or world_config.schedule_profile != expected_profile
        ):
            raise ValueError("transaction world profile differs from the selected bounded life")
        if not bool(jax.device_get(transaction.state_valid(state))):
            raise ValueError("runner requires a valid freshly initialized dyad state")
        initial_words = np.asarray(
            jax.device_get(state.hccl_state.world_state.step_words), dtype=np.uint32
        )
        if not np.array_equal(initial_words, np.zeros((2,), dtype=np.uint32)):
            raise ValueError("runner initialization must begin at the exact zero clock")
        self._transaction = transaction
        self._state = state
        self._total_steps = total_steps
        self._next_hard_action_masks = jnp.ones(
            (_N_AGENTS, _N_ACTIONS), dtype=jnp.bool_
        )

    @property
    def final_state(self) -> object:
        return self._state

    @staticmethod
    def _abort(step_index: int, stage: str, error: Exception) -> HCCLContinualDyadLifeError:
        return HCCLContinualDyadLifeError(step_index, stage, str(error))

    def _validate_adoption(
        self,
        result: HCCLContinualDyadResult,
        prepared: HCCLContinualDyadPreparedTransaction,
        receipt: HCCLContinualDyadPreparationReceipt,
    ) -> None:
        if result.prepared is not prepared or result.receipt is not receipt:
            raise ValueError("adoption did not return the exact prepared transaction and receipt")
        for name in (
            "source_state_matches",
            "source_state_valid",
            "prepared_content_valid",
            "receipt_valid",
            "candidate_state_valid",
            "hccl_owner_committed",
            "planner_owner_committed",
            "update_applied",
        ):
            _host_true(getattr(result, name), name=f"result.{name}")
        for name in (
            "child_adoptions_valid",
            "action_stack_owners_committed",
            "context_owners_committed",
            "lineage_owners_committed",
        ):
            _host_all_true(getattr(result, name), name=f"result.{name}", shape=(2,))
        returned = _host_array(
            result.complete_source_returned,
            name="result.complete_source_returned",
            shape=(),
            dtype=np.dtype(np.bool_),
        )
        if bool(returned):
            raise ValueError("adoption returned the source instead of the committed candidate")
        _host_true(prepared.preparation_valid, name="prepared.preparation_valid")
        _host_true(prepared.hccl_result.update_applied, name="hccl_result.update_applied")
        _host_true(receipt.integrity_bound, name="receipt.integrity_bound")

    def execute_event(self, step_index: int) -> _CommittedEvent:
        if type(step_index) is not int or not 0 <= step_index < self._total_steps:
            raise HCCLContinualDyadLifeError(
                step_index, "bounds", "step index lies outside the selected fixed life"
            )
        expected_pre = np.asarray((0, step_index), dtype=np.uint32)
        source_words = np.asarray(
            jax.device_get(self._state.hccl_state.world_state.step_words),
            dtype=np.uint32,
        )
        if not np.array_equal(source_words, expected_pre):
            raise HCCLContinualDyadLifeError(
                step_index, "source-clock", "persistent source clock is discontinuous"
            )
        try:
            result = self._transaction.step(
                self._state,
                self._next_hard_action_masks,
            )
        except Exception as error:
            raise self._abort(step_index, "transaction-step", error) from error
        if type(result) is not HCCLContinualDyadResult:
            raise HCCLContinualDyadLifeError(
                step_index, "transaction-step", "malformed result type"
            )
        prepared = result.prepared
        receipt = result.receipt
        if type(prepared) is not HCCLContinualDyadPreparedTransaction:
            raise HCCLContinualDyadLifeError(
                step_index, "transaction-step", "malformed preparation type"
            )
        if type(receipt) is not HCCLContinualDyadPreparationReceipt:
            raise HCCLContinualDyadLifeError(
                step_index, "transaction-step", "malformed receipt type"
            )
        try:
            self._validate_adoption(result, prepared, receipt)
        except (AttributeError, TypeError, ValueError) as error:
            raise self._abort(step_index, "atomic-adoption-validation", error) from error

        # These evaluator-only reads deliberately occur after successful adoption.
        try:
            proposals = prepared.hccl_result.world_proposals
            regime_ids = _host_array(
                proposals.evaluator_regime_id,
                name="world_proposals.evaluator_regime_id",
                shape=(8,),
                dtype=np.dtype(np.int32),
            )
            task_scores = _host_array(
                proposals.signals.task_score,
                name="world_proposals.signals.task_score",
                shape=(8,),
                dtype=np.dtype(np.float32),
            )
            net_rewards = _host_array(
                proposals.signals.net_reward,
                name="world_proposals.signals.net_reward",
                shape=(8, _N_AGENTS),
                dtype=np.dtype(np.float32),
            )
            factors = HCCLCausalCoreFactors(
                gathering=proposals.factors.gathering[_PP_SLOT],
                velocity=proposals.factors.velocity[_PP_SLOT],
                convention_clean=proposals.factors.convention_clean[_PP_SLOT],
                convention_noisy=proposals.factors.convention_noisy[_PP_SLOT],
            )
            all_regime_scores = np.asarray(
                tuple(
                    np.float32(
                        jax.device_get(
                            self._transaction.hccl.world.task_score_for_regime(
                                regime_id, factors
                            )
                        )
                    )
                    for regime_id in range(_N_REGIMES)
                ),
                dtype=np.float32,
            )
            pre_words = np.asarray(
                jax.device_get(prepared.hccl_result.pre_transaction_words),
                dtype=np.uint32,
            )
            post_words = np.asarray(
                jax.device_get(prepared.hccl_result.post_transaction_words),
                dtype=np.uint32,
            )
            destination_words = np.asarray(
                jax.device_get(result.state.hccl_state.world_state.step_words),
                dtype=np.uint32,
            )
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise self._abort(step_index, "post-commit-evaluator-readout", error) from error

        expected_post = np.asarray((0, step_index + 1), dtype=np.uint32)
        if not (
            np.array_equal(pre_words, expected_pre)
            and np.array_equal(post_words, expected_post)
            and np.array_equal(destination_words, expected_post)
        ):
            raise HCCLContinualDyadLifeError(
                step_index,
                "post-commit-clock",
                "transaction clocks do not equal the exact committed life clocks",
            )
        regime_id = int(regime_ids[_PP_SLOT])
        task_score = np.float32(task_scores[_PP_SLOT])
        if not 0 <= regime_id < _N_REGIMES:
            raise HCCLContinualDyadLifeError(
                step_index,
                "post-commit-evaluator-readout",
                "committed PP evaluator regime id lies outside the fixed domain",
            )
        if all_regime_scores[regime_id] != task_score:
            raise HCCLContinualDyadLifeError(
                step_index,
                "post-commit-evaluator-readout",
                "selected evaluator score differs from committed PP task score",
            )
        committed_event = _CommittedEvent(
            regime_id=regime_id,
            committed=True,
            pre_step_words=pre_words,
            post_step_words=post_words,
            task_score=task_score,
            net_rewards=np.asarray(net_rewards[_PP_SLOT], dtype=np.float32),
            all_regime_scores=all_regime_scores,
        )
        self._state = result.state
        return committed_event


def _collect_bounded_life(
    config: HCCLContinualDyadRunnerConfig,
    executor: _LifeEventExecutor,
) -> HCCLContinualDyadLifeTrace:
    """CI-cheap orchestration seam; production supplies the exact executor above."""

    if type(config) is not HCCLContinualDyadRunnerConfig:
        raise TypeError("config must be exact HCCLContinualDyadRunnerConfig")
    steps = config.total_steps
    regime_ids = np.empty((steps,), dtype=np.int32)
    committed = np.empty((steps,), dtype=np.bool_)
    pre_words = np.empty((steps, 2), dtype=np.uint32)
    post_words = np.empty((steps, 2), dtype=np.uint32)
    task_scores = np.empty((steps,), dtype=np.float32)
    net_rewards = np.empty((steps, _N_AGENTS), dtype=np.float32)
    all_regime_scores = np.empty((steps, _N_REGIMES), dtype=np.float32)
    expected_regimes = _expected_regime_ids(config.schedule_profile)

    for step_index in range(steps):
        try:
            event = executor.execute_event(step_index)
        except HCCLContinualDyadLifeError:
            raise
        except Exception as error:
            raise HCCLContinualDyadLifeError(
                step_index, "event-executor", str(error)
            ) from error
        if type(event) is not _CommittedEvent:
            raise HCCLContinualDyadLifeError(
                step_index, "event-executor", "executor returned a malformed event type"
            )
        expected_pre = np.asarray((0, step_index), dtype=np.uint32)
        expected_post = np.asarray((0, step_index + 1), dtype=np.uint32)
        if not (
            event.committed
            and event.regime_id == int(expected_regimes[step_index])
            and np.array_equal(event.pre_step_words, expected_pre)
            and np.array_equal(event.post_step_words, expected_post)
        ):
            raise HCCLContinualDyadLifeError(
                step_index,
                "trace-collection",
                "event does not match the exact schedule, commit, or clock contract",
            )
        regime_ids[step_index] = event.regime_id
        committed[step_index] = event.committed
        pre_words[step_index] = event.pre_step_words
        post_words[step_index] = event.post_step_words
        task_scores[step_index] = event.task_score
        net_rewards[step_index] = event.net_rewards
        all_regime_scores[step_index] = event.all_regime_scores

    return HCCLContinualDyadLifeTrace(
        schedule_profile=config.schedule_profile,
        regime_ids=regime_ids,
        transaction_committed=committed,
        pre_step_words=pre_words,
        post_step_words=post_words,
        task_scores=task_scores,
        net_rewards=net_rewards,
        all_regime_score_matrix=all_regime_scores,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLContinualDyadLifeResult:
    """One complete trace and its exact final persistent dyad state."""

    config: HCCLContinualDyadRunnerConfig
    trace: HCCLContinualDyadLifeTrace
    state: HCCLContinualDyadState

    def __post_init__(self) -> None:
        if type(self.config) is not HCCLContinualDyadRunnerConfig:
            raise TypeError("result config must be exact HCCLContinualDyadRunnerConfig")
        if type(self.trace) is not HCCLContinualDyadLifeTrace:
            raise TypeError("result trace must be exact HCCLContinualDyadLifeTrace")
        if type(self.state) is not HCCLContinualDyadState:
            raise TypeError("result state must be exact HCCLContinualDyadState")
        validate_hccl_continual_dyad_life_trace(self.trace)
        if self.trace.schedule_profile != self.config.schedule_profile:
            raise ValueError("result trace profile differs from runner config")
        final_words = np.asarray(
            jax.device_get(self.state.hccl_state.world_state.step_words),
            dtype=np.uint32,
        )
        expected = np.asarray((0, self.config.total_steps), dtype=np.uint32)
        if not np.array_equal(final_words, expected):
            raise ValueError("result state does not have the exact completed-life clock")


class HCCLContinualDyadRunner:
    """Initialize through the factory and execute one explicit fixed-profile life."""

    def __init__(self, config: HCCLContinualDyadRunnerConfig | None = None) -> None:
        selected = HCCLContinualDyadRunnerConfig() if config is None else config
        if type(selected) is not HCCLContinualDyadRunnerConfig:
            raise TypeError("config must be exact HCCLContinualDyadRunnerConfig")
        self._config = selected
        self._factory = HCCLContinualDyadFactory(selected.factory_config)

    @property
    def config(self) -> HCCLContinualDyadRunnerConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    def run(self, key: Array) -> HCCLContinualDyadLifeResult:
        """Execute exactly one selected life; return nothing partial on failure."""

        try:
            initialized = self._factory.init(key)
        except Exception as error:
            raise HCCLContinualDyadLifeError(-1, "factory-initialization", str(error)) from error
        if type(initialized) is not HCCLContinualDyadFactoryInitialization:
            raise HCCLContinualDyadLifeError(
                -1, "factory-initialization", "factory returned a malformed initialization"
            )
        executor = _ProductionLifeEventExecutor(
            initialized.transaction,
            initialized.state,
            total_steps=self._config.total_steps,
        )
        trace = _collect_bounded_life(self._config, executor)
        final_state = executor.final_state
        if type(final_state) is not HCCLContinualDyadState:
            raise HCCLContinualDyadLifeError(
                self._config.total_steps,
                "final-state",
                "executor returned a malformed final state",
            )
        if not bool(jax.device_get(initialized.transaction.state_valid(final_state))):
            raise HCCLContinualDyadLifeError(
                self._config.total_steps,
                "final-state",
                "completed persistent dyad state is invalid",
            )
        return HCCLContinualDyadLifeResult(
            config=self._config,
            trace=trace,
            state=final_state,
        )


__all__ = (
    "HCCL_CONTINUAL_DYAD_LIFE_TRACE_SCHEMA",
    "HCCL_CONTINUAL_DYAD_RUNNER_CONFIG_SCHEMA",
    "HCCL_CONTINUAL_DYAD_RUNNER_EVIDENCE_LEVEL",
    "HCCL_CONTINUAL_DYAD_RUNNER_LIMITATIONS",
    "HCCL_CONTINUAL_DYAD_RUNNER_STATUS",
    "HCCLContinualDyadLifeError",
    "HCCLContinualDyadLifeResult",
    "HCCLContinualDyadLifeTrace",
    "HCCLContinualDyadRunner",
    "HCCLContinualDyadRunnerConfig",
    "validate_hccl_continual_dyad_life_trace",
)
