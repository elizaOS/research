# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Development-only H=2 lineage retention in the consumed hidden-rule life.

This evaluator composes the frozen :mod:`sequential_lineage_cache` sidecar
with the exact 4,000-transition, root-zero capacity-pressure dyad.  The base
environment, controllers, context banks, and evaluator birth ledgers remain
owned by :mod:`hidden_rule_capacity_pressure_development`; this module uses
that evaluator's event, ledger, scrub, clock, and resource helpers without
changing either source.

Every event snapshots both context reward-model banks and both sidecar rescue
vectors before the environment outcome is proposed.  The no-signal condition
dispatches exact zeros and the H=2 condition dispatches the already-live rescue
scores.  Both conditions still execute the same prioritized context calls,
sidecar proposals, controller scrub preparations, controller updates, and one
outer all-or-none commit.  Post-outcome evidence can therefore affect only a
future eviction decision.

The complete panel is an at-most-once, process-local, in-memory development
report.  It has no writer, artifact, threshold, default arm, winner, evidence,
or promotion surface.  The sole root is already consumed and cannot support a
scientific claim.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import json
import platform
import sys
import threading
from collections.abc import Callable, Mapping
from importlib.metadata import version as package_version
from pathlib import Path
from types import ModuleType
from typing import Final, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

import alberta_framework.core.average_reward as average_reward_module
import alberta_framework.core.context_inference as context_inference_module
import alberta_framework.core.pairwise_dominance_quarantine as pairwise_module
import alberta_framework.core.sequential_lineage_cache as sequential_lineage_module
import alberta_framework.streams.matrix_game as matrix_game_module
from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAState,
)
from alberta_framework.core.context_inference import (
    ContextInference,
    ContextInferencePrioritizedUpdateResult,
    ContextInferenceState,
)
from alberta_framework.core.sequential_lineage_cache import (
    SequentialLineageCache,
    SequentialLineageCacheConfig,
    SequentialLineageCacheEvent,
    SequentialLineageCacheProposal,
    SequentialLineageCacheResourceRecord,
    SequentialLineageCacheState,
    SequentialLineageCacheWorkRecord,
    measure_sequential_lineage_cache_state_nbytes,
)
from alberta_framework.evaluation import (
    hidden_rule_capacity_pressure_development as capacity_pressure,
)
from alberta_framework.streams.matrix_game import RecurringConventionGame

PROTOCOL_SCHEMA: Final = "alberta.hidden-rule-sequential-lineage-retention-development.protocol.v1"
REPORT_SCHEMA: Final = "alberta.hidden-rule-sequential-lineage-retention-development.report.v1"
DEVELOPMENT_ONLY: Final = True
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
EVIDENCE_AUTHORIZED: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
WRITER_AVAILABLE: Final = False
ARTIFACT_BYTES_WRITTEN: Final = 0
THRESHOLDS_USED: Final = False
WINNER_SELECTION_ALLOWED: Final = False
DEFAULT_CONDITION_AVAILABLE: Final = False
ARBITRARY_ROOT_EXECUTION_ALLOWED: Final = False
CALIBRATION_ROOT_CONSUMED: Final = True
ACCEPTANCE_STATUS: Final = "not-assessed"
PARAMETER_TRANSPLANT_ALLOWED: Final = False
EVALUATOR_MATCHED_OUTER_WORK_CLAIMED: Final = True
CORE_MATCHED_OUTER_WORK_CLAIMED: Final = False
CORE_HOST_TRANSITION_BINDING_CLAIMED: Final = (
    sequential_lineage_module.HOST_TRANSITION_BINDING_CLAIMED
)
CORE_STATE_CONTENT_INTEGRITY_CLAIMED: Final = (
    sequential_lineage_module.STATE_CONTENT_INTEGRITY_CLAIMED
)
CORE_EXTERNAL_STATE_PROVENANCE_CLAIMED: Final = (
    sequential_lineage_module.EXTERNAL_STATE_PROVENANCE_CLAIMED
)
FULL_PANEL_PROCESS_LOCAL_AT_MOST_ONCE: Final = True

NO_SIGNAL: Literal["no_signal"] = "no_signal"
H2_PREDICTIVE_RESCUE: Literal["h2_predictive_rescue"] = "h2_predictive_rescue"
SequentialLineageRetentionCondition = Literal[
    "no_signal",
    "h2_predictive_rescue",
]
CONDITIONS: Final[tuple[SequentialLineageRetentionCondition, ...]] = (
    NO_SIGNAL,
    H2_PREDICTIVE_RESCUE,
)

# Descriptive aliases make the condition namespace explicit to callers while
# deliberately providing no default or selected arm.
SEQUENTIAL_LINEAGE_NO_SIGNAL: Final = NO_SIGNAL
SEQUENTIAL_LINEAGE_H2_PREDICTIVE_RESCUE: Final = H2_PREDICTIVE_RESCUE
SEQUENTIAL_LINEAGE_RETENTION_CONDITIONS: Final = CONDITIONS
SEQUENTIAL_LINEAGE_RETENTION_NO_SIGNAL: Final = NO_SIGNAL
SEQUENTIAL_LINEAGE_RETENTION_H2_PREDICTIVE_RESCUE: Final = H2_PREDICTIVE_RESCUE

MAX_CONTEXTS: Final = capacity_pressure.MAX_CONTEXTS
N_ACTIONS: Final = capacity_pressure.N_ACTIONS
OBSERVATION_DIM: Final = capacity_pressure.CONTEXT_CONFIG.observation_dim
NUM_STEPS: Final = capacity_pressure.NUM_STEPS
N_AGENTS: Final = 2
CONSUMED_ROOT_INDEX: Final = capacity_pressure.CALIBRATION_ROOT_INDEX
AGENT_NAMESPACES: Final = (
    "hidden-rule-sequential-lineage-retention-agent-0",
    "hidden-rule-sequential-lineage-retention-agent-1",
)
EXPECTED_SEQUENTIAL_LINEAGE_CORE_SHA256: Final = (
    "4cdb68bc7125d0cf7d13709f515900c6262828b9961a33d615fc8c0c594e637a"
)
EXPECTED_BASE_SCAN_CARRY_NBYTES: Final = 962
EXPECTED_SIDECAR_NBYTES_PER_AGENT: Final = 563
EXPECTED_SIDECAR_PAIR_NBYTES: Final = 1_126
EXPECTED_COMPOSITE_SCAN_CARRY_NBYTES: Final = 2_088
SOURCE_MANIFEST_SCOPE: Final = "selected-direct-files-not-transitive-closure"
RUNTIME_IDENTITY_SCOPE: Final = (
    "selected Python, NumPy, JAX, backend, x64, and device fields; not an "
    "environment, accelerator-driver, XLA-flag, or compiler closure"
)
RESOURCE_ACCOUNTING_SCOPE: Final = (
    "exact persistent composite JAX-array bytes and fixed named logical calls; "
    "logical diagnostics exclude compiler workspaces, allocator residency, FLOPs, "
    "and latency"
)

SEQUENTIAL_LINEAGE_CONFIG: Final = SequentialLineageCacheConfig(
    max_contexts=MAX_CONTEXTS,
    n_actions=N_ACTIONS,
    observation_dim=OBSERVATION_DIM,
    initial_reward_estimate=capacity_pressure.CONTEXT_CONFIG.initial_reward_estimate,
)

_SELECTED_SOURCE_MODULES: Final[tuple[tuple[str, ModuleType], ...]] = (
    ("sequential_lineage_cache_core_sha256", sequential_lineage_module),
    ("capacity_pressure_evaluator_sha256", capacity_pressure),
    ("context_inference_core_sha256", context_inference_module),
    ("average_reward_core_sha256", average_reward_module),
    ("matrix_game_stream_sha256", matrix_game_module),
    ("pairwise_dominance_core_sha256", pairwise_module),
)


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenRuleSequentialLineageRetentionProtocol:
    """The exact non-customizable consumed-root protocol."""

    schema_version: str = PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PROTOCOL_SCHEMA:
            raise ValueError("sequential-lineage protocol schema is unsupported")

    def to_config(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "type": type(self).__name__,
            "development_only": DEVELOPMENT_ONLY,
            "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
            "evidence_authorized": EVIDENCE_AUTHORIZED,
            "output_writes_allowed": OUTPUT_WRITES_ALLOWED,
            "writer_available": WRITER_AVAILABLE,
            "artifact_bytes_written": ARTIFACT_BYTES_WRITTEN,
            "thresholds_used": THRESHOLDS_USED,
            "winner_selection_allowed": WINNER_SELECTION_ALLOWED,
            "default_condition_available": DEFAULT_CONDITION_AVAILABLE,
            "arbitrary_root_execution_allowed": ARBITRARY_ROOT_EXECUTION_ALLOWED,
            "root": {
                "namespace": capacity_pressure.CALIBRATION_ROOT.namespace,
                "index": CONSUMED_ROOT_INDEX,
                "key_seed": capacity_pressure.CALIBRATION_ROOT.key_seed,
                "consumed": CALIBRATION_ROOT_CONSUMED,
            },
            "conditions": list(CONDITIONS),
            "epsilon_grid": list(capacity_pressure.EPSILON_GRID),
            "schedule": {
                "phase_length": capacity_pressure.PHASE_LENGTH,
                "offsets": list(capacity_pressure.OFFSETS),
                "total_steps": NUM_STEPS,
            },
            "geometry": {
                "max_contexts": MAX_CONTEXTS,
                "n_actions": N_ACTIONS,
                "observation_dim": OBSERVATION_DIM,
                "initial_reward_estimate": SEQUENTIAL_LINEAGE_CONFIG.initial_reward_estimate,
                "comparison_bank_size": SEQUENTIAL_LINEAGE_CONFIG.comparison_bank_size,
                "confirmation_horizon": (
                    sequential_lineage_module.SEQUENTIAL_LINEAGE_CACHE_CONFIRMATION_HORIZON
                ),
                "archive_capacity_per_agent": (
                    sequential_lineage_module.SEQUENTIAL_LINEAGE_CACHE_ARCHIVE_CAPACITY
                ),
                "agent_namespaces": list(AGENT_NAMESPACES),
            },
            "causal_order": [
                "fix_joint_actions",
                "snapshot_pre_update_reward_weights_and_live_rescue_scores",
                "propose_environment_outcome",
                "propose_prioritized_context_updates",
                "derive_context_events_and_birth_ledgers_with_base_helpers",
                "propose_both_sequential_lineage_sidecars",
                "prepare_authenticated_controller_scrubs",
                "propose_controller_updates",
                "commit_complete_composite_all_or_none",
            ],
            "matching": {
                "same_base_and_sidecar_genesis": True,
                "same_calls_shapes_and_rng_advance": True,
                "sidecar_propose_called_for_every_agent_event_in_both_conditions": True,
                "evaluator_owns_matched_outer_work_claim": (EVALUATOR_MATCHED_OUTER_WORK_CLAIMED),
                "core_matched_outer_work_claimed": CORE_MATCHED_OUTER_WORK_CLAIMED,
            },
            "nonclaims": {
                "parameter_transplant": PARAMETER_TRANSPLANT_ALLOWED,
                "host_transition_binding_claimed_by_core": (CORE_HOST_TRANSITION_BINDING_CLAIMED),
                "external_sidecar_state_provenance_claimed": (
                    CORE_EXTERNAL_STATE_PROVENANCE_CLAIMED
                ),
                "transitive_source_closure": False,
                "environment_or_compiler_closure": False,
                "scientific_evidence": False,
                "Alberta_Plan_completion": False,
            },
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> HiddenRuleSequentialLineageRetentionProtocol:
        canonical = cls()
        if set(payload) != set(canonical.to_config()):
            raise ValueError("sequential-lineage protocol fields do not match")
        if dict(payload) != canonical.to_config():
            raise ValueError("sequential-lineage protocol is not the frozen declaration")
        return canonical


PROTOCOL: Final = HiddenRuleSequentialLineageRetentionProtocol()


class _ProcessAttemptLatch:
    """Execute one exact-string builder once, sealing success or BaseException."""

    def __init__(self, builder: Callable[[], str]) -> None:
        if not callable(builder):
            raise TypeError("process-attempt builder must be callable")
        self._builder = builder
        self._lock = threading.Lock()
        self._attempted = False
        self._value: str | None = None
        self._failure: BaseException | None = None

    def get(self) -> str:
        with self._lock:
            if self._attempted:
                if self._failure is not None:
                    raise RuntimeError(
                        "the process-local sequential-lineage panel is sealed after failure"
                    ) from self._failure
                if self._value is None:
                    raise RuntimeError("the process-local sequential-lineage latch is invalid")
                return self._value
            self._attempted = True
            try:
                value = self._builder()
                if type(value) is not str:
                    raise TypeError("process-attempt builder must return an exact string")
            except BaseException as error:
                self._failure = error
                raise
            self._value = value
            return value


@chex.dataclass(frozen=True)
class HiddenRuleSequentialLineageRetentionState:
    """One complete base dyad plus two independent fixed-capacity sidecars."""

    base: capacity_pressure.CapacityPressureState
    sidecar_0: SequentialLineageCacheState
    sidecar_1: SequentialLineageCacheState


@chex.dataclass(frozen=True)
class HiddenRuleSequentialLineageRetentionTrace:
    """Fixed-shape causal, binding, sidecar, scrub, clock, and rollback audit."""

    capacity: capacity_pressure.CapacityPressureStepTrace
    protection_enabled: Array
    source_scores_snapshotted_before_outcome: Array
    outcome_routed_to_current_protection: Array
    source_live_rescue_words: Array
    source_live_rescue_scores: Array
    source_rescue_scores_valid: Array
    dispatched_eviction_protection: Array
    dispatch_binding_valid: Array
    context_protection_input_bound: Array
    context_full_bank_eviction_requested: Array
    context_eviction_protection_used: Array
    context_eviction_target_adjusted: Array
    context_ordinary_lru_slots: Array
    context_protected_lru_slots: Array
    context_selected_eviction_slots: Array
    pre_update_reward_weights: Array
    event_source_reward_weights: Array
    pre_update_weight_binding_valid: Array
    event_source_step_words: Array
    event_post_step_words: Array
    event_source_birth_words: Array
    event_post_birth_words: Array
    event_source_in_use: Array
    event_post_in_use: Array
    event_observations: Array
    event_actions: Array
    event_rewards: Array
    event_allocated: Array
    event_evicted: Array
    event_target_slots: Array
    event_context_update_applied: Array
    event_binding_valid: Array
    proposal_source_state_valid: Array
    proposal_event_valid: Array
    proposal_predictive_inputs_finite: Array
    proposal_evidence_valid: Array
    proposal_candidate_state_valid: Array
    proposal_update_applied: Array
    proposal_fields_valid: Array
    proposal_full_bank_birth: Array
    proposal_cache_tested: Array
    proposal_quarantine_opened: Array
    proposal_quarantine_second_evidence: Array
    proposal_quarantine_confirmed: Array
    proposal_quarantine_rejected: Array
    proposal_target_identity_matched: Array
    proposal_target_survived: Array
    proposal_confirmation_commit_abstained: Array
    proposal_lineage_transferred: Array
    proposal_rescue_incremented: Array
    proposal_victim_staged: Array
    proposal_overlap_full_bank_birth: Array
    proposal_new_quarantine_suppressed: Array
    proposal_archive_locked_during_pending: Array
    proposal_archive_selected_source: Array
    proposal_archive_old_retained: Array
    proposal_archive_opening_victim_selected: Array
    proposal_archive_current_victim_selected: Array
    proposal_parameter_transplanted: Array
    proposal_predictions: Array
    proposal_losses: Array
    proposal_comparator_mask: Array
    proposal_never_worse: Array
    proposal_ever_strict: Array
    parameter_transplant_absent: Array
    source_sidecar_valid: Array
    candidate_sidecar_valid: Array
    committed_sidecar_valid: Array
    sidecar_config_tokens_bound: Array
    scrub_required: Array
    scrub_candidate_applied: Array
    scrub_preparation_valid: Array
    scrub_binding_valid: Array
    scrub_pre_bank_valid: Array
    scrub_post_bank_valid: Array
    scrub_pre_ledger_valid: Array
    scrub_post_ledger_valid: Array
    scrub_survivor_rows_untouched: Array
    scrub_rng_untouched_before_update: Array
    scrub_clock_untouched_before_update: Array
    controller_updates_proposed: Array
    source_clocks_aligned: Array
    candidate_clocks_aligned: Array
    committed_clocks_aligned: Array
    event_clocks_bound: Array
    candidate_state_finite: Array
    outer_candidate_valid: Array
    forced_outer_rejection: Array
    outer_update_applied: Array
    committed_candidate_exact: Array
    rollback_exact: Array
    all_or_none_commit_valid: Array


@chex.dataclass(frozen=True)
class HiddenRuleSequentialLineageRetentionStepResult:
    """Outer successor plus exact child proposals retained for local audit."""

    state: HiddenRuleSequentialLineageRetentionState
    trace: HiddenRuleSequentialLineageRetentionTrace
    events: tuple[SequentialLineageCacheEvent, SequentialLineageCacheEvent]
    proposals: tuple[SequentialLineageCacheProposal, SequentialLineageCacheProposal]
    context_results: tuple[
        ContextInferencePrioritizedUpdateResult,
        ContextInferencePrioritizedUpdateResult,
    ]
    prepared_controllers: tuple[DifferentialSARSAState, DifferentialSARSAState]


# Short aliases retain the file's hidden-rule qualification while offering the
# mechanism-first names used by the neighboring capacity-pressure evaluators.
SequentialLineageRetentionState = HiddenRuleSequentialLineageRetentionState
SequentialLineageRetentionTrace = HiddenRuleSequentialLineageRetentionTrace
SequentialLineageRetentionStepResult = HiddenRuleSequentialLineageRetentionStepResult


@dataclasses.dataclass(frozen=True, slots=True)
class SequentialLineageRetentionResourceBudget:
    """Exact persistent base, sidecar, pair, and composite byte formula."""

    base: capacity_pressure.CapacityPressureResourceBudget
    sidecar: SequentialLineageCacheResourceRecord
    measured_base_scan_carry_nbytes: int
    measured_sidecar_0_nbytes: int
    measured_sidecar_1_nbytes: int
    measured_sidecar_pair_nbytes: int
    measured_composite_scan_carry_nbytes: int
    exact_base_match: bool
    exact_sidecar_formula_match: bool
    exact_composite_match: bool
    fixed_shape: bool = True
    replay_capacity: int = 0

    @property
    def total_persistent_nbytes(self) -> int:
        return self.sidecar.total_scan_carry_nbytes

    def to_dict(self) -> dict[str, object]:
        return {
            "base": self.base.to_dict(),
            "sidecar": dataclasses.asdict(self.sidecar),
            "measured_base_scan_carry_nbytes": self.measured_base_scan_carry_nbytes,
            "measured_sidecar_0_nbytes": self.measured_sidecar_0_nbytes,
            "measured_sidecar_1_nbytes": self.measured_sidecar_1_nbytes,
            "measured_sidecar_pair_nbytes": self.measured_sidecar_pair_nbytes,
            "measured_composite_scan_carry_nbytes": (self.measured_composite_scan_carry_nbytes),
            "exact_base_match": self.exact_base_match,
            "exact_sidecar_formula_match": self.exact_sidecar_formula_match,
            "exact_composite_match": self.exact_composite_match,
            "total_persistent_nbytes": self.total_persistent_nbytes,
            "fixed_shape": self.fixed_shape,
            "replay_capacity": self.replay_capacity,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class SequentialLineageRetentionWorkBudget:
    """Evaluator-owned matched outer schedule plus the core standalone record."""

    total_steps: int
    n_agents: int
    environment_transition_proposals: int
    prioritized_context_update_proposals: int
    pre_outcome_reward_bank_snapshots: int
    pre_outcome_rescue_score_snapshots: int
    context_event_helper_calls: int
    birth_ledger_helper_calls: int
    sequential_lineage_proposals: int
    controller_scrub_preparations: int
    controller_update_proposals: int
    action_selection_calls_including_genesis: int
    outer_all_or_none_commit_decisions: int
    replay_updates: int
    reset_callbacks: int
    random_draws_added_by_sidecars: int
    same_calls_and_shapes_across_conditions: bool
    branch_independent_controller_rng_advance: bool
    evaluator_matched_outer_work_claimed: bool
    core_matched_outer_work_claimed: bool
    core: SequentialLineageCacheWorkRecord

    def to_dict(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["core"] = dataclasses.asdict(self.core)
        return payload


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _attach_report_hash(body: Mapping[str, object]) -> dict[str, object]:
    if "report_sha256" in body:
        raise ValueError("report body already contains report_sha256")
    materialized = dict(body)
    return {**materialized, "report_sha256": _json_sha256(materialized)}


def _report_hash_reconstructs(report: Mapping[str, object]) -> bool:
    expected = report.get("report_sha256")
    if type(expected) is not str:
        return False
    body = {name: value for name, value in report.items() if name != "report_sha256"}
    return expected == _json_sha256(body)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_source_path(module: ModuleType) -> Path:
    value = getattr(module, "__file__", None)
    if type(value) is not str:
        raise RuntimeError("a selected sequential-lineage source has no exact file path")
    return Path(value).resolve()


def _selected_source_hashes() -> dict[str, str]:
    files = {"evaluation_module_sha256": _sha256_file(Path(__file__).resolve())}
    for label, module in _SELECTED_SOURCE_MODULES:
        if label in files:
            raise RuntimeError("selected sequential-lineage source labels are not unique")
        files[label] = _sha256_file(_module_source_path(module))
    return files


_IMPORT_TIME_SELECTED_SOURCE_HASHES: Final = tuple(sorted(_selected_source_hashes().items()))


def _bound_source_manifest(*, stage: str) -> dict[str, str]:
    if type(stage) is not str or not stage:
        raise ValueError("source-binding stage must be a nonempty exact string")
    current = tuple(sorted(_selected_source_hashes().items()))
    if current != _IMPORT_TIME_SELECTED_SOURCE_HASHES:
        raise RuntimeError(
            "selected sequential-lineage source files differ from their import-time "
            f"bytes at {stage}"
        )
    files = dict(_IMPORT_TIME_SELECTED_SOURCE_HASHES)
    if files.get("sequential_lineage_cache_core_sha256") != EXPECTED_SEQUENTIAL_LINEAGE_CORE_SHA256:
        raise RuntimeError("the frozen sequential-lineage core hash differs")
    return {**files, "manifest_sha256": _json_sha256(files)}


def _runtime_identity() -> dict[str, object]:
    devices = jax.devices()
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "jax": jax.__version__,
        "jaxlib": package_version("jaxlib"),
        "numpy": np.__version__,
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_device_count": len(devices),
        "jax_device_kinds": [device.device_kind for device in devices],
    }


def _tree_exact_equal(left: object, right: object) -> Array:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    if len(left_leaves) != len(right_leaves):
        return jnp.asarray(False, dtype=jnp.bool_)
    exact = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return jnp.asarray(False, dtype=jnp.bool_)
        if jnp.issubdtype(left_array.dtype, jnp.floating):
            if left_array.dtype == jnp.dtype(jnp.float32):
                left_array = jax.lax.bitcast_convert_type(left_array, jnp.uint32)
                right_array = jax.lax.bitcast_convert_type(right_array, jnp.uint32)
            elif left_array.dtype == jnp.dtype(jnp.float16):
                left_array = jax.lax.bitcast_convert_type(left_array, jnp.uint16)
                right_array = jax.lax.bitcast_convert_type(right_array, jnp.uint16)
        exact = exact & jnp.all(left_array == right_array)
    return exact


def _tree_nbytes(tree: object) -> int:
    total = 0
    for leaf in jax.tree.leaves(tree):
        value = leaf
        dtype = getattr(value, "dtype", None)
        if dtype is None:
            continue
        if jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            value = jr.key_data(value)
        total += int(value.size) * int(value.dtype.itemsize)
    return total


def _tree_sha256(tree: object) -> str:
    digest = hashlib.sha256()
    leaves, structure = jax.tree.flatten(tree)
    digest.update(str(structure).encode("utf-8"))
    for leaf in leaves:
        value = leaf
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            value = jr.key_data(value)
        array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
        digest.update(
            _canonical_json({"shape": list(array.shape), "dtype": str(array.dtype)}).encode("ascii")
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sidecar_static_shapes_valid(state: SequentialLineageCacheState) -> bool:
    archive = state.archive
    pending = state.pending
    candidate = pending.candidate
    return (
        state.config_token.shape == (32,)
        and state.config_token.dtype == jnp.dtype(jnp.uint8)
        and state.content_token.shape == (32,)
        and state.content_token.dtype == jnp.dtype(jnp.uint8)
        and state.bound_birth_words.shape == (MAX_CONTEXTS, 2)
        and state.bound_birth_words.dtype == jnp.dtype(jnp.uint32)
        and state.live_lineage_words.shape == (MAX_CONTEXTS, 2)
        and state.live_lineage_words.dtype == jnp.dtype(jnp.uint32)
        and state.live_rescue_words.shape == (MAX_CONTEXTS, 2)
        and state.live_rescue_words.dtype == jnp.dtype(jnp.uint32)
        and archive.valid.shape == ()
        and archive.valid.dtype == jnp.dtype(jnp.bool_)
        and archive.source_birth_words.shape == (2,)
        and archive.lineage_words.shape == (2,)
        and archive.rescue_words.shape == (2,)
        and archive.reward_weights.shape == (N_ACTIONS, OBSERVATION_DIM)
        and archive.reward_weights.dtype == jnp.dtype(jnp.float32)
        and pending.valid.shape == ()
        and pending.valid.dtype == jnp.dtype(jnp.bool_)
        and candidate.reward_weights.shape == (N_ACTIONS, OBSERVATION_DIM)
        and pending.target_birth_words.shape == (2,)
        and pending.source_birth_words.shape == (MAX_CONTEXTS, 2)
        and pending.source_reward_weights.shape == (MAX_CONTEXTS, N_ACTIONS, OBSERVATION_DIM)
        and pending.source_reward_weights.dtype == jnp.dtype(jnp.float32)
        and pending.victim_lineage_words.shape == (2,)
        and pending.victim_rescue_words.shape == (2,)
        and pending.first_never_worse.shape == (MAX_CONTEXTS + 1,)
        and pending.first_ever_strict.shape == (MAX_CONTEXTS + 1,)
    )


def _live_rescue_scores(
    sidecar: SequentialLineageCacheState,
    context_state: ContextInferenceState,
) -> tuple[Array, Array]:
    """Project the fixed 4,000-step reachable rescue words exactly to float32."""

    words = sidecar.live_rescue_words
    low = words[:, 1]
    scores = low.astype(jnp.float32)
    unused_zero = jnp.all(
        jnp.where(
            context_state.in_use[:, None],
            jnp.asarray(True, dtype=jnp.bool_),
            words == jnp.asarray(0, dtype=jnp.uint32),
        )
    )
    valid = (
        jnp.all(words[:, 0] == jnp.asarray(0, dtype=jnp.uint32))
        & jnp.all(low <= jnp.asarray(NUM_STEPS, dtype=jnp.uint32))
        & jnp.all(scores.astype(jnp.uint32) == low)
        & jnp.all(jnp.isfinite(scores))
        & unused_zero
    )
    return jnp.where(context_state.in_use, scores, jnp.float32(0.0)), valid


def _stack_field(values: tuple[object, object], name: str) -> Array:
    return jnp.stack(tuple(jnp.asarray(getattr(value, name)) for value in values))


def _proposal_fields_valid(proposal: SequentialLineageCacheProposal) -> Array:
    bank = SEQUENTIAL_LINEAGE_CONFIG.comparison_bank_size
    shapes_valid = jnp.asarray(
        proposal.predictions.shape == (bank,)
        and proposal.losses.shape == (bank,)
        and proposal.comparator_mask.shape == (bank,)
        and proposal.never_worse.shape == (bank,)
        and proposal.ever_strict.shape == (bank,),
        dtype=jnp.bool_,
    )
    diagnostics_finite = jnp.all(jnp.isfinite(proposal.predictions)) & jnp.all(
        jnp.isfinite(proposal.losses)
    )
    relational_consistent = jnp.all((~proposal.ever_strict) | proposal.never_worse)
    transfer_consistent = proposal.lineage_transferred == proposal.rescue_incremented
    no_transplant = ~proposal.parameter_transplanted
    applied_contract = (~proposal.update_applied) | (
        proposal.source_state_valid
        & proposal.event_valid
        & proposal.predictive_inputs_finite
        & proposal.evidence_valid
        & proposal.candidate_state_valid
    )
    return (
        shapes_valid
        & diagnostics_finite
        & relational_consistent
        & transfer_consistent
        & no_transplant
        & applied_contract
    )


class HiddenRuleSequentialLineageRetentionEvaluator:
    """Exact two-sidecar wrapper for one epsilon and one explicit condition."""

    def __init__(
        self,
        epsilon: float,
        condition: SequentialLineageRetentionCondition,
    ) -> None:
        if epsilon not in capacity_pressure.EPSILON_GRID:
            raise ValueError("epsilon is not in the consumed-root grid")
        if condition not in CONDITIONS:
            raise ValueError("unknown sequential-lineage retention condition")
        self.epsilon = epsilon
        self.condition = condition
        self.agent = DifferentialSARSAAgent(capacity_pressure.control_config(epsilon))
        self.context = ContextInference(capacity_pressure.CONTEXT_CONFIG)
        self.game = RecurringConventionGame(capacity_pressure.GAME_CONFIG)
        self.sidecar = SequentialLineageCache(SEQUENTIAL_LINEAGE_CONFIG)

    def initialize(self) -> HiddenRuleSequentialLineageRetentionState:
        """Create identical base genesis and two independent empty sidecars."""

        return HiddenRuleSequentialLineageRetentionState(
            base=capacity_pressure.initialize_capacity_pressure_state(self.epsilon),
            sidecar_0=self.sidecar.init(),
            sidecar_1=self.sidecar.init(),
        )

    def _static_shapes_valid(self, state: HiddenRuleSequentialLineageRetentionState) -> bool:
        return (
            capacity_pressure._post_audit_static_shapes_valid(state.base)
            and _sidecar_static_shapes_valid(state.sidecar_0)
            and _sidecar_static_shapes_valid(state.sidecar_1)
        )

    def state_is_valid(self, state: HiddenRuleSequentialLineageRetentionState) -> Array:
        if not self._static_shapes_valid(state):
            return jnp.asarray(False, dtype=jnp.bool_)
        base = state.base
        return (
            capacity_pressure._clocks_aligned(base)
            & capacity_pressure._tree_finite(state)
            & capacity_pressure._birth_ledger_valid(self.context, base.context_0, base.ledger_0)
            & capacity_pressure._birth_ledger_valid(self.context, base.context_1, base.ledger_1)
            & self.sidecar.state_valid(
                state.sidecar_0,
                base.context_0.step_words,
                base.ledger_0.slot_birth_words,
                base.context_0.in_use,
            )
            & self.sidecar.state_valid(
                state.sidecar_1,
                base.context_1.step_words,
                base.ledger_1.slot_birth_words,
                base.context_1.in_use,
            )
        )

    def resource_budget(
        self,
        state: HiddenRuleSequentialLineageRetentionState | None = None,
    ) -> SequentialLineageRetentionResourceBudget:
        source = self.initialize() if state is None else state
        if not self._static_shapes_valid(source):
            raise ValueError("sequential-lineage state has invalid static shapes or dtypes")
        base_budget = capacity_pressure._resource_budget(source.base)
        sidecar_record = self.sidecar.resource_record(
            n_agents=N_AGENTS,
            base_scan_carry_nbytes=base_budget.total_scan_carry_nbytes,
        )
        base_measured = _tree_nbytes(source.base)
        sidecar_0 = measure_sequential_lineage_cache_state_nbytes(source.sidecar_0)
        sidecar_1 = measure_sequential_lineage_cache_state_nbytes(source.sidecar_1)
        pair = sidecar_0 + sidecar_1
        composite = _tree_nbytes(source)
        exact_base = (
            base_measured == base_budget.total_scan_carry_nbytes == EXPECTED_BASE_SCAN_CARRY_NBYTES
        )
        exact_sidecars = (
            sidecar_0
            == sidecar_1
            == sidecar_record.per_agent_state_nbytes
            == EXPECTED_SIDECAR_NBYTES_PER_AGENT
            and pair == sidecar_record.joint_state_nbytes == EXPECTED_SIDECAR_PAIR_NBYTES
        )
        exact_composite = (
            composite
            == base_measured + pair
            == sidecar_record.total_scan_carry_nbytes
            == EXPECTED_COMPOSITE_SCAN_CARRY_NBYTES
        )
        if not exact_base:
            raise RuntimeError("base persistent byte formula differs from the frozen dyad")
        if not exact_sidecars:
            raise RuntimeError("sequential-lineage sidecar byte formula differs")
        if not exact_composite:
            raise RuntimeError("base-plus-sidecar composite byte formula differs")
        return SequentialLineageRetentionResourceBudget(
            base=base_budget,
            sidecar=sidecar_record,
            measured_base_scan_carry_nbytes=base_measured,
            measured_sidecar_0_nbytes=sidecar_0,
            measured_sidecar_1_nbytes=sidecar_1,
            measured_sidecar_pair_nbytes=pair,
            measured_composite_scan_carry_nbytes=composite,
            exact_base_match=exact_base,
            exact_sidecar_formula_match=exact_sidecars,
            exact_composite_match=exact_composite,
        )

    def work_budget(self, total_steps: int = NUM_STEPS) -> SequentialLineageRetentionWorkBudget:
        if type(total_steps) is not int or total_steps < 0 or total_steps > NUM_STEPS:
            raise ValueError("total_steps must be an integer in the fixed life horizon")
        core = self.sidecar.work_record(total_steps=total_steps, n_agents=N_AGENTS)
        if core.matched_outer_work_claimed:
            raise RuntimeError("the standalone core unexpectedly claimed matched outer work")
        calls = N_AGENTS * total_steps
        return SequentialLineageRetentionWorkBudget(
            total_steps=total_steps,
            n_agents=N_AGENTS,
            environment_transition_proposals=total_steps,
            prioritized_context_update_proposals=calls,
            pre_outcome_reward_bank_snapshots=calls,
            pre_outcome_rescue_score_snapshots=calls,
            context_event_helper_calls=calls,
            birth_ledger_helper_calls=calls,
            sequential_lineage_proposals=calls,
            controller_scrub_preparations=calls,
            controller_update_proposals=calls,
            action_selection_calls_including_genesis=calls + N_AGENTS,
            outer_all_or_none_commit_decisions=total_steps,
            replay_updates=0,
            reset_callbacks=0,
            random_draws_added_by_sidecars=0,
            same_calls_and_shapes_across_conditions=True,
            branch_independent_controller_rng_advance=True,
            evaluator_matched_outer_work_claimed=True,
            core_matched_outer_work_claimed=core.matched_outer_work_claimed,
            core=core,
        )

    def _event(
        self,
        source_context: ContextInferenceState,
        result: ContextInferencePrioritizedUpdateResult,
        source_ledger: capacity_pressure.ContextBirthLedgerState,
        post_ledger: capacity_pressure.ContextBirthLedgerState,
        source_reward_weights: Array,
        observation: Array,
        action: Array,
        reward: Array,
        lifecycle: tuple[Array, Array, Array, Array],
    ) -> SequentialLineageCacheEvent:
        return SequentialLineageCacheEvent(
            source_step_words=source_context.step_words,
            post_step_words=result.post_step_words,
            source_birth_words=source_ledger.slot_birth_words,
            post_birth_words=post_ledger.slot_birth_words,
            source_in_use=source_context.in_use,
            post_in_use=result.state.in_use,
            source_reward_weights=source_reward_weights,
            observation=observation,
            action=jnp.asarray(action, dtype=jnp.int32),
            reward=jnp.asarray(reward, dtype=jnp.float32),
            allocated=jnp.asarray(lifecycle[1], dtype=jnp.bool_),
            evicted=jnp.asarray(lifecycle[2], dtype=jnp.bool_),
            target_slot=result.state.active_context.astype(jnp.int32),
            context_update_applied=result.update_applied,
        )

    def _event_binding_valid(
        self,
        event: SequentialLineageCacheEvent,
        source_context: ContextInferenceState,
        result: ContextInferencePrioritizedUpdateResult,
        source_ledger: capacity_pressure.ContextBirthLedgerState,
        post_ledger: capacity_pressure.ContextBirthLedgerState,
        source_reward_weights: Array,
        observation: Array,
        action: Array,
        reward: Array,
        lifecycle: tuple[Array, Array, Array, Array],
    ) -> Array:
        return (
            jnp.all(event.source_step_words == source_context.step_words)
            & jnp.all(event.post_step_words == result.post_step_words)
            & jnp.all(event.source_birth_words == source_ledger.slot_birth_words)
            & jnp.all(event.post_birth_words == post_ledger.slot_birth_words)
            & jnp.all(event.source_in_use == source_context.in_use)
            & jnp.all(event.post_in_use == result.state.in_use)
            & jnp.all(event.source_reward_weights == source_reward_weights)
            & jnp.all(event.observation == observation)
            & (event.action == jnp.asarray(action, dtype=jnp.int32))
            & (event.reward == jnp.asarray(reward, dtype=jnp.float32))
            & (event.allocated == lifecycle[1])
            & (event.evicted == lifecycle[2])
            & (event.target_slot == result.state.active_context)
            & (event.context_update_applied == result.update_applied)
        )

    def step(
        self,
        state: HiddenRuleSequentialLineageRetentionState,
        *,
        force_outer_rejection: Array = jnp.asarray(False, dtype=jnp.bool_),
    ) -> HiddenRuleSequentialLineageRetentionStepResult:
        """Propose one complete event and commit every child or no child."""

        if not self._static_shapes_valid(state):
            raise ValueError("sequential-lineage state has invalid static shapes or dtypes")
        force_reject = jnp.asarray(force_outer_rejection, dtype=jnp.bool_)
        if force_reject.shape != ():
            raise ValueError("force_outer_rejection must be scalar")

        base = state.base
        source_clocks_aligned = capacity_pressure._clocks_aligned(base)
        source_state_finite = capacity_pressure._tree_finite(state)
        source_ledger_valid = jnp.stack(
            (
                capacity_pressure._birth_ledger_valid(self.context, base.context_0, base.ledger_0),
                capacity_pressure._birth_ledger_valid(self.context, base.context_1, base.ledger_1),
            )
        ).astype(jnp.bool_)
        source_sidecar_valid = jnp.stack(
            (
                self.sidecar.state_valid(
                    state.sidecar_0,
                    base.context_0.step_words,
                    base.ledger_0.slot_birth_words,
                    base.context_0.in_use,
                ),
                self.sidecar.state_valid(
                    state.sidecar_1,
                    base.context_1.step_words,
                    base.ledger_1.slot_birth_words,
                    base.context_1.in_use,
                ),
            )
        ).astype(jnp.bool_)
        actions = jnp.stack((base.controller_0.last_action, base.controller_1.last_action)).astype(
            jnp.int32
        )
        pre_context_slots = jnp.stack(
            (base.context_0.active_context, base.context_1.active_context)
        ).astype(jnp.int32)
        pre_context_birth_words = jnp.stack(
            (
                base.ledger_0.slot_birth_words[base.context_0.active_context],
                base.ledger_1.slot_birth_words[base.context_1.active_context],
            )
        ).astype(jnp.uint32)

        # These snapshots are intentionally constructed before the environment
        # proposal.  Neither expression has a data dependency on this outcome.
        pre_reward_weights_0 = base.context_0.reward_weights
        pre_reward_weights_1 = base.context_1.reward_weights
        raw_score_0, score_valid_0 = _live_rescue_scores(state.sidecar_0, base.context_0)
        raw_score_1, score_valid_1 = _live_rescue_scores(state.sidecar_1, base.context_1)
        protection_enabled = self.condition == H2_PREDICTIVE_RESCUE
        dispatched_0 = jnp.where(
            protection_enabled,
            raw_score_0,
            jnp.zeros((MAX_CONTEXTS,), dtype=jnp.float32),
        )
        dispatched_1 = jnp.where(
            protection_enabled,
            raw_score_1,
            jnp.zeros((MAX_CONTEXTS,), dtype=jnp.float32),
        )

        environment_result = self.game.step_result(
            base.environment,
            actions[0],
            actions[1],
        )
        observation_0 = jax.nn.one_hot(actions[1], N_ACTIONS, dtype=jnp.float32)
        observation_1 = jax.nn.one_hot(actions[0], N_ACTIONS, dtype=jnp.float32)
        context_result_0 = self.context.update_result_with_eviction_protection(
            base.context_0,
            observation_0,
            actions[0],
            environment_result.reward,
            dispatched_0,
        )
        context_result_1 = self.context.update_result_with_eviction_protection(
            base.context_1,
            observation_1,
            actions[1],
            environment_result.reward,
            dispatched_1,
        )
        lifecycle_0 = capacity_pressure._context_event(
            self.context,
            base.context_0,
            context_result_0,
            observation_0,
            actions[0],
            environment_result.reward,
        )
        lifecycle_1 = capacity_pressure._context_event(
            self.context,
            base.context_1,
            context_result_1,
            observation_1,
            actions[1],
            environment_result.reward,
        )
        proposed_ledger_0 = capacity_pressure._propose_ledger(
            base.ledger_0, context_result_0, lifecycle_0[1]
        )
        proposed_ledger_1 = capacity_pressure._propose_ledger(
            base.ledger_1, context_result_1, lifecycle_1[1]
        )
        event_0 = self._event(
            base.context_0,
            context_result_0,
            base.ledger_0,
            proposed_ledger_0,
            pre_reward_weights_0,
            observation_0,
            actions[0],
            environment_result.reward,
            lifecycle_0,
        )
        event_1 = self._event(
            base.context_1,
            context_result_1,
            base.ledger_1,
            proposed_ledger_1,
            pre_reward_weights_1,
            observation_1,
            actions[1],
            environment_result.reward,
            lifecycle_1,
        )
        event_binding_0 = self._event_binding_valid(
            event_0,
            base.context_0,
            context_result_0,
            base.ledger_0,
            proposed_ledger_0,
            pre_reward_weights_0,
            observation_0,
            actions[0],
            environment_result.reward,
            lifecycle_0,
        )
        event_binding_1 = self._event_binding_valid(
            event_1,
            base.context_1,
            context_result_1,
            base.ledger_1,
            proposed_ledger_1,
            pre_reward_weights_1,
            observation_1,
            actions[1],
            environment_result.reward,
            lifecycle_1,
        )

        # Both conditions invoke both sidecars once on every event.  The core
        # returns a local proposal; only the later outer transaction can make
        # either proposal visible in the composite state.
        proposal_0 = self.sidecar.propose(state.sidecar_0, event_0)
        proposal_1 = self.sidecar.propose(state.sidecar_1, event_1)

        preparation_0 = capacity_pressure._prepare_controller_scrub(
            self.context,
            base.controller_0,
            base.context_0,
            context_result_0.state,
            base.ledger_0,
            proposed_ledger_0,
            lifecycle_0[1],
            context_result_0.post_step_words,
            -1,
        )
        preparation_1 = capacity_pressure._prepare_controller_scrub(
            self.context,
            base.controller_1,
            base.context_1,
            context_result_1.state,
            base.ledger_1,
            proposed_ledger_1,
            lifecycle_1[1],
            context_result_1.post_step_words,
            -1,
        )
        controller_result_0 = self.agent.update(
            preparation_0.state,
            environment_result.reward,
            context_result_0.context_onehot,
        )
        controller_result_1 = self.agent.update(
            preparation_1.state,
            environment_result.reward,
            context_result_1.context_onehot,
        )

        candidate_base = capacity_pressure.CapacityPressureState(
            environment=environment_result.state,
            controller_0=controller_result_0.state,
            controller_1=controller_result_1.state,
            context_0=context_result_0.state,
            context_1=context_result_1.state,
            ledger_0=proposed_ledger_0,
            ledger_1=proposed_ledger_1,
        )
        candidate = HiddenRuleSequentialLineageRetentionState(
            base=candidate_base,
            sidecar_0=proposal_0.state,
            sidecar_1=proposal_1.state,
        )
        candidate_clocks_aligned = capacity_pressure._clocks_aligned(candidate_base)
        candidate_state_finite = capacity_pressure._tree_finite(candidate)
        candidate_sidecar_valid = jnp.stack(
            (
                self.sidecar.state_valid(
                    proposal_0.state,
                    context_result_0.post_step_words,
                    proposed_ledger_0.slot_birth_words,
                    context_result_0.state.in_use,
                ),
                self.sidecar.state_valid(
                    proposal_1.state,
                    context_result_1.post_step_words,
                    proposed_ledger_1.slot_birth_words,
                    context_result_1.state.in_use,
                ),
            )
        ).astype(jnp.bool_)
        context_results = (context_result_0, context_result_1)
        proposals = (proposal_0, proposal_1)
        preparations = (preparation_0, preparation_1)
        context_updates = _stack_field(context_results, "update_applied").astype(jnp.bool_)
        controller_updates = jnp.stack(
            (controller_result_0.update_applied, controller_result_1.update_applied)
        ).astype(jnp.bool_)
        proposal_updates = _stack_field(proposals, "update_applied").astype(jnp.bool_)
        proposal_fields = jnp.stack(
            (_proposal_fields_valid(proposal_0), _proposal_fields_valid(proposal_1))
        ).astype(jnp.bool_)
        preparation_valid = _stack_field(preparations, "preparation_valid").astype(jnp.bool_)
        event_binding_valid = jnp.stack((event_binding_0, event_binding_1)).astype(jnp.bool_)
        score_valid = jnp.stack((score_valid_0, score_valid_1)).astype(jnp.bool_)
        source_children_valid = (
            environment_result.state_valid
            & context_result_0.source_state_valid
            & context_result_1.source_state_valid
            & controller_result_0.state_valid
            & controller_result_1.state_valid
            & jnp.all(source_ledger_valid)
            & jnp.all(source_sidecar_valid)
            & jnp.all(score_valid)
        )
        candidate_children_valid = (
            context_result_0.candidate_state_valid
            & context_result_1.candidate_state_valid
            & controller_result_0.candidate_state_finite
            & controller_result_1.candidate_state_finite
            & jnp.all(candidate_sidecar_valid)
            & jnp.all(proposal_fields)
        )
        outer_candidate_valid = (
            source_clocks_aligned
            & source_state_finite
            & source_children_valid
            & environment_result.update_applied
            & jnp.all(context_updates)
            & jnp.all(event_binding_valid)
            & jnp.all(proposal_updates)
            & jnp.all(preparation_valid)
            & jnp.all(controller_updates)
            & candidate_clocks_aligned
            & candidate_state_finite
            & candidate_children_valid
        )
        outer_update_applied = outer_candidate_valid & ~force_reject
        committed = jax.lax.cond(
            outer_update_applied,
            lambda _: candidate,
            lambda _: state,
            operand=None,
        )
        committed_base = committed.base
        committed_sidecar_valid = jnp.stack(
            (
                self.sidecar.state_valid(
                    committed.sidecar_0,
                    committed_base.context_0.step_words,
                    committed_base.ledger_0.slot_birth_words,
                    committed_base.context_0.in_use,
                ),
                self.sidecar.state_valid(
                    committed.sidecar_1,
                    committed_base.context_1.step_words,
                    committed_base.ledger_1.slot_birth_words,
                    committed_base.context_1.in_use,
                ),
            )
        ).astype(jnp.bool_)
        committed_candidate_exact = _tree_exact_equal(committed, candidate)
        rollback_exact = _tree_exact_equal(committed, state)
        all_or_none = jnp.where(
            outer_update_applied,
            committed_candidate_exact,
            rollback_exact,
        )

        switches = jnp.stack((lifecycle_0[0], lifecycle_1[0])) & outer_update_applied
        allocations = jnp.stack((lifecycle_0[1], lifecycle_1[1])) & outer_update_applied
        evictions = jnp.stack((lifecycle_0[2], lifecycle_1[2])) & outer_update_applied
        reuses = jnp.stack((lifecycle_0[3], lifecycle_1[3])) & outer_update_applied
        post_context_slots = jnp.stack(
            (
                committed_base.context_0.active_context,
                committed_base.context_1.active_context,
            )
        ).astype(jnp.int32)
        post_context_birth_words = jnp.stack(
            (
                committed_base.ledger_0.slot_birth_words[committed_base.context_0.active_context],
                committed_base.ledger_1.slot_birth_words[committed_base.context_1.active_context],
            )
        ).astype(jnp.uint32)
        capacity_trace = capacity_pressure.CapacityPressureStepTrace(
            reward=jnp.where(
                outer_update_applied,
                environment_result.reward,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            actions=actions,
            pre_context_slots=pre_context_slots,
            post_context_slots=post_context_slots,
            pre_context_birth_words=pre_context_birth_words,
            post_context_birth_words=post_context_birth_words,
            switches=switches,
            allocations=allocations,
            evictions=evictions,
            reuses=reuses,
            contexts_in_use=jnp.stack(
                (
                    self.context.num_contexts_in_use(committed_base.context_0),
                    self.context.num_contexts_in_use(committed_base.context_1),
                )
            ).astype(jnp.int32),
            environment_update_proposed=environment_result.update_applied,
            context_updates_proposed=context_updates,
            controller_updates_proposed=controller_updates,
            source_clocks_aligned=source_clocks_aligned,
            candidate_clocks_aligned=candidate_clocks_aligned,
            source_state_finite=source_state_finite,
            candidate_state_finite=candidate_state_finite,
            update_applied=outer_update_applied,
            pre_step_words=base.environment.step_words,
            post_step_words=committed_base.environment.step_words,
            controller_rng_key_words=jnp.stack(
                (
                    jr.key_data(committed_base.controller_0.rng_key),
                    jr.key_data(committed_base.controller_1.rng_key),
                )
            ).astype(jnp.uint32),
            controller_next_q_values=jnp.where(
                outer_update_applied,
                jnp.stack((controller_result_0.q_values, controller_result_1.q_values)),
                jnp.zeros((N_AGENTS, N_ACTIONS), dtype=jnp.float32),
            ),
            controller_next_actions=jnp.where(
                outer_update_applied,
                jnp.stack((controller_result_0.action, controller_result_1.action)),
                actions,
            ).astype(jnp.int32),
        )

        raw_scores = jnp.stack((raw_score_0, raw_score_1)).astype(jnp.float32)
        dispatched = jnp.stack((dispatched_0, dispatched_1)).astype(jnp.float32)
        expected_dispatched = jnp.where(
            protection_enabled,
            raw_scores,
            jnp.zeros_like(raw_scores),
        )
        event_weights = jnp.stack(
            (event_0.source_reward_weights, event_1.source_reward_weights)
        ).astype(jnp.float32)
        pre_weights = jnp.stack((pre_reward_weights_0, pre_reward_weights_1)).astype(jnp.float32)
        trace = HiddenRuleSequentialLineageRetentionTrace(
            capacity=capacity_trace,
            protection_enabled=jnp.asarray(protection_enabled, dtype=jnp.bool_),
            source_scores_snapshotted_before_outcome=jnp.asarray(True, dtype=jnp.bool_),
            outcome_routed_to_current_protection=jnp.asarray(False, dtype=jnp.bool_),
            source_live_rescue_words=jnp.stack(
                (state.sidecar_0.live_rescue_words, state.sidecar_1.live_rescue_words)
            ).astype(jnp.uint32),
            source_live_rescue_scores=raw_scores,
            source_rescue_scores_valid=score_valid,
            dispatched_eviction_protection=dispatched,
            dispatch_binding_valid=jnp.all(dispatched == expected_dispatched, axis=1),
            context_protection_input_bound=jnp.stack(
                (
                    jnp.all(context_result_0.eviction_protection == dispatched_0),
                    jnp.all(context_result_1.eviction_protection == dispatched_1),
                )
            ).astype(jnp.bool_),
            context_full_bank_eviction_requested=_stack_field(
                context_results, "full_bank_eviction_requested"
            ).astype(jnp.bool_),
            context_eviction_protection_used=_stack_field(
                context_results, "eviction_protection_used"
            ).astype(jnp.bool_),
            context_eviction_target_adjusted=_stack_field(
                context_results, "eviction_target_adjusted"
            ).astype(jnp.bool_),
            context_ordinary_lru_slots=_stack_field(context_results, "ordinary_lru_slot").astype(
                jnp.int32
            ),
            context_protected_lru_slots=_stack_field(context_results, "protected_lru_slot").astype(
                jnp.int32
            ),
            context_selected_eviction_slots=_stack_field(
                context_results, "selected_eviction_slot"
            ).astype(jnp.int32),
            pre_update_reward_weights=pre_weights,
            event_source_reward_weights=event_weights,
            pre_update_weight_binding_valid=jnp.all(event_weights == pre_weights, axis=(1, 2, 3)),
            event_source_step_words=jnp.stack(
                (event_0.source_step_words, event_1.source_step_words)
            ).astype(jnp.uint32),
            event_post_step_words=jnp.stack(
                (event_0.post_step_words, event_1.post_step_words)
            ).astype(jnp.uint32),
            event_source_birth_words=jnp.stack(
                (event_0.source_birth_words, event_1.source_birth_words)
            ).astype(jnp.uint32),
            event_post_birth_words=jnp.stack(
                (event_0.post_birth_words, event_1.post_birth_words)
            ).astype(jnp.uint32),
            event_source_in_use=jnp.stack((event_0.source_in_use, event_1.source_in_use)).astype(
                jnp.bool_
            ),
            event_post_in_use=jnp.stack((event_0.post_in_use, event_1.post_in_use)).astype(
                jnp.bool_
            ),
            event_observations=jnp.stack((event_0.observation, event_1.observation)).astype(
                jnp.float32
            ),
            event_actions=jnp.stack((event_0.action, event_1.action)).astype(jnp.int32),
            event_rewards=jnp.stack((event_0.reward, event_1.reward)).astype(jnp.float32),
            event_allocated=jnp.stack((event_0.allocated, event_1.allocated)).astype(jnp.bool_),
            event_evicted=jnp.stack((event_0.evicted, event_1.evicted)).astype(jnp.bool_),
            event_target_slots=jnp.stack((event_0.target_slot, event_1.target_slot)).astype(
                jnp.int32
            ),
            event_context_update_applied=jnp.stack(
                (event_0.context_update_applied, event_1.context_update_applied)
            ).astype(jnp.bool_),
            event_binding_valid=event_binding_valid,
            proposal_source_state_valid=_stack_field(proposals, "source_state_valid").astype(
                jnp.bool_
            ),
            proposal_event_valid=_stack_field(proposals, "event_valid").astype(jnp.bool_),
            proposal_predictive_inputs_finite=_stack_field(
                proposals, "predictive_inputs_finite"
            ).astype(jnp.bool_),
            proposal_evidence_valid=_stack_field(proposals, "evidence_valid").astype(jnp.bool_),
            proposal_candidate_state_valid=_stack_field(proposals, "candidate_state_valid").astype(
                jnp.bool_
            ),
            proposal_update_applied=proposal_updates,
            proposal_fields_valid=proposal_fields,
            proposal_full_bank_birth=_stack_field(proposals, "full_bank_birth").astype(jnp.bool_),
            proposal_cache_tested=_stack_field(proposals, "cache_tested").astype(jnp.bool_),
            proposal_quarantine_opened=_stack_field(proposals, "quarantine_opened").astype(
                jnp.bool_
            ),
            proposal_quarantine_second_evidence=_stack_field(
                proposals, "quarantine_second_evidence"
            ).astype(jnp.bool_),
            proposal_quarantine_confirmed=_stack_field(proposals, "quarantine_confirmed").astype(
                jnp.bool_
            ),
            proposal_quarantine_rejected=_stack_field(proposals, "quarantine_rejected").astype(
                jnp.bool_
            ),
            proposal_target_identity_matched=_stack_field(
                proposals, "target_identity_matched"
            ).astype(jnp.bool_),
            proposal_target_survived=_stack_field(proposals, "target_survived").astype(jnp.bool_),
            proposal_confirmation_commit_abstained=_stack_field(
                proposals, "confirmation_commit_abstained"
            ).astype(jnp.bool_),
            proposal_lineage_transferred=_stack_field(proposals, "lineage_transferred").astype(
                jnp.bool_
            ),
            proposal_rescue_incremented=_stack_field(proposals, "rescue_incremented").astype(
                jnp.bool_
            ),
            proposal_victim_staged=_stack_field(proposals, "victim_staged").astype(jnp.bool_),
            proposal_overlap_full_bank_birth=_stack_field(
                proposals, "overlap_full_bank_birth"
            ).astype(jnp.bool_),
            proposal_new_quarantine_suppressed=_stack_field(
                proposals, "new_quarantine_suppressed"
            ).astype(jnp.bool_),
            proposal_archive_locked_during_pending=_stack_field(
                proposals, "archive_locked_during_pending"
            ).astype(jnp.bool_),
            proposal_archive_selected_source=_stack_field(
                proposals, "archive_selected_source"
            ).astype(jnp.int32),
            proposal_archive_old_retained=_stack_field(proposals, "archive_old_retained").astype(
                jnp.bool_
            ),
            proposal_archive_opening_victim_selected=_stack_field(
                proposals, "archive_opening_victim_selected"
            ).astype(jnp.bool_),
            proposal_archive_current_victim_selected=_stack_field(
                proposals, "archive_current_victim_selected"
            ).astype(jnp.bool_),
            proposal_parameter_transplanted=_stack_field(
                proposals, "parameter_transplanted"
            ).astype(jnp.bool_),
            proposal_predictions=_stack_field(proposals, "predictions").astype(jnp.float32),
            proposal_losses=_stack_field(proposals, "losses").astype(jnp.float32),
            proposal_comparator_mask=_stack_field(proposals, "comparator_mask").astype(jnp.bool_),
            proposal_never_worse=_stack_field(proposals, "never_worse").astype(jnp.bool_),
            proposal_ever_strict=_stack_field(proposals, "ever_strict").astype(jnp.bool_),
            parameter_transplant_absent=(
                ~jnp.any(_stack_field(proposals, "parameter_transplanted"))
                & jnp.all(
                    candidate_base.context_0.reward_weights == context_result_0.state.reward_weights
                )
                & jnp.all(
                    candidate_base.context_1.reward_weights == context_result_1.state.reward_weights
                )
            ),
            source_sidecar_valid=source_sidecar_valid,
            candidate_sidecar_valid=candidate_sidecar_valid,
            committed_sidecar_valid=committed_sidecar_valid,
            sidecar_config_tokens_bound=jnp.stack(
                (
                    jnp.all(state.sidecar_0.config_token == state.sidecar_1.config_token),
                    jnp.all(proposal_0.state.config_token == proposal_1.state.config_token),
                )
            ).astype(jnp.bool_),
            scrub_required=_stack_field(preparations, "scrub_required").astype(jnp.bool_),
            scrub_candidate_applied=_stack_field(preparations, "scrub_candidate_applied").astype(
                jnp.bool_
            ),
            scrub_preparation_valid=preparation_valid,
            scrub_binding_valid=_stack_field(preparations, "binding_valid").astype(jnp.bool_),
            scrub_pre_bank_valid=_stack_field(preparations, "pre_bank_valid").astype(jnp.bool_),
            scrub_post_bank_valid=_stack_field(preparations, "post_bank_valid").astype(jnp.bool_),
            scrub_pre_ledger_valid=_stack_field(preparations, "pre_ledger_valid").astype(jnp.bool_),
            scrub_post_ledger_valid=_stack_field(preparations, "post_ledger_valid").astype(
                jnp.bool_
            ),
            scrub_survivor_rows_untouched=_stack_field(
                preparations, "survivor_rows_untouched"
            ).astype(jnp.bool_),
            scrub_rng_untouched_before_update=_stack_field(
                preparations, "rng_untouched_before_update"
            ).astype(jnp.bool_),
            scrub_clock_untouched_before_update=_stack_field(
                preparations, "clock_untouched_before_update"
            ).astype(jnp.bool_),
            controller_updates_proposed=controller_updates,
            source_clocks_aligned=source_clocks_aligned,
            candidate_clocks_aligned=candidate_clocks_aligned,
            committed_clocks_aligned=capacity_pressure._clocks_aligned(committed_base),
            event_clocks_bound=jnp.stack(
                (
                    jnp.all(event_0.source_step_words == base.context_0.step_words)
                    & jnp.all(event_0.post_step_words == context_result_0.post_step_words),
                    jnp.all(event_1.source_step_words == base.context_1.step_words)
                    & jnp.all(event_1.post_step_words == context_result_1.post_step_words),
                )
            ).astype(jnp.bool_),
            candidate_state_finite=candidate_state_finite,
            outer_candidate_valid=outer_candidate_valid,
            forced_outer_rejection=force_reject,
            outer_update_applied=outer_update_applied,
            committed_candidate_exact=committed_candidate_exact,
            rollback_exact=rollback_exact,
            all_or_none_commit_valid=all_or_none,
        )
        return HiddenRuleSequentialLineageRetentionStepResult(
            state=committed,
            trace=trace,
            events=(event_0, event_1),
            proposals=proposals,
            context_results=context_results,
            prepared_controllers=(preparation_0.state, preparation_1.state),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _scan_life(
        self,
        initial: HiddenRuleSequentialLineageRetentionState,
    ) -> tuple[
        HiddenRuleSequentialLineageRetentionState,
        HiddenRuleSequentialLineageRetentionTrace,
    ]:
        """Execute the fixed 4,000-event life; the public latch calls this once."""

        def scan_step(
            carry: HiddenRuleSequentialLineageRetentionState,
            _: None,
        ) -> tuple[
            HiddenRuleSequentialLineageRetentionState,
            HiddenRuleSequentialLineageRetentionTrace,
        ]:
            result = self.step(
                carry,
                force_outer_rejection=jnp.asarray(False, dtype=jnp.bool_),
            )
            return result.state, result.trace

        return jax.lax.scan(scan_step, initial, xs=None, length=NUM_STEPS)


def initialize_hidden_rule_sequential_lineage_retention_state(
    epsilon: float,
    condition: SequentialLineageRetentionCondition,
) -> HiddenRuleSequentialLineageRetentionState:
    """Initialize one explicit condition without advancing the consumed life."""

    return HiddenRuleSequentialLineageRetentionEvaluator(epsilon, condition).initialize()


def step_hidden_rule_sequential_lineage_retention(
    epsilon: float,
    condition: SequentialLineageRetentionCondition,
    state: HiddenRuleSequentialLineageRetentionState,
    *,
    force_outer_rejection: Array = jnp.asarray(False, dtype=jnp.bool_),
) -> HiddenRuleSequentialLineageRetentionStepResult:
    """Advance one local-test event; no alternative root is exposed."""

    return HiddenRuleSequentialLineageRetentionEvaluator(epsilon, condition).step(
        state,
        force_outer_rejection=force_outer_rejection,
    )


def initialize_sequential_lineage_retention_state(
    epsilon: float,
    condition: SequentialLineageRetentionCondition,
) -> HiddenRuleSequentialLineageRetentionState:
    """Mechanism-first alias for the explicit-condition initializer."""

    return initialize_hidden_rule_sequential_lineage_retention_state(epsilon, condition)


def step_sequential_lineage_retention(
    epsilon: float,
    condition: SequentialLineageRetentionCondition,
    state: HiddenRuleSequentialLineageRetentionState,
    *,
    force_outer_rejection: Array = jnp.asarray(False, dtype=jnp.bool_),
) -> HiddenRuleSequentialLineageRetentionStepResult:
    """Mechanism-first alias for one exact outer transaction."""

    return step_hidden_rule_sequential_lineage_retention(
        epsilon,
        condition,
        state,
        force_outer_rejection=force_outer_rejection,
    )


def sequential_lineage_retention_resource_budget(
    epsilon: float,
    condition: SequentialLineageRetentionCondition,
) -> SequentialLineageRetentionResourceBudget:
    """Return the exact base-plus-pair persistent resource formula."""

    return HiddenRuleSequentialLineageRetentionEvaluator(epsilon, condition).resource_budget()


def sequential_lineage_retention_work_budget(
    total_steps: int = NUM_STEPS,
) -> SequentialLineageRetentionWorkBudget:
    """Return the condition-independent evaluator-owned matched work record."""

    return HiddenRuleSequentialLineageRetentionEvaluator(
        capacity_pressure.EPSILON_GRID[0], NO_SIGNAL
    ).work_budget(total_steps)


def validate_static_contract() -> tuple[str, ...]:
    """Fail closed on geometry, causal surfaces, hashes, resources, and nonclaims."""

    errors: list[str] = []
    if CONDITIONS != (NO_SIGNAL, H2_PREDICTIVE_RESCUE):
        errors.append("the two explicit conditions drifted")
    if (MAX_CONTEXTS, N_ACTIONS, OBSERVATION_DIM) != (3, 4, 4):
        errors.append("the fixed K=3, A=4, D=4 geometry drifted")
    if NUM_STEPS != 4_000 or CONSUMED_ROOT_INDEX != 0:
        errors.append("the consumed 4,000-step root-zero life drifted")
    if SEQUENTIAL_LINEAGE_CONFIG.initial_reward_estimate != (
        capacity_pressure.CONTEXT_CONFIG.initial_reward_estimate
    ):
        errors.append("the sidecar fresh prior differs from CONTEXT_CONFIG")
    if SEQUENTIAL_LINEAGE_CONFIG != SequentialLineageCacheConfig(
        max_contexts=3,
        n_actions=4,
        observation_dim=4,
        initial_reward_estimate=capacity_pressure.CONTEXT_CONFIG.initial_reward_estimate,
    ):
        errors.append("the sequential-lineage configuration drifted")
    if (
        not DEVELOPMENT_ONLY
        or SCIENTIFIC_PROMOTION_ALLOWED
        or EVIDENCE_AUTHORIZED
        or OUTPUT_WRITES_ALLOWED
        or WRITER_AVAILABLE
        or ARTIFACT_BYTES_WRITTEN != 0
        or THRESHOLDS_USED
        or WINNER_SELECTION_ALLOWED
        or DEFAULT_CONDITION_AVAILABLE
        or ARBITRARY_ROOT_EXECUTION_ALLOWED
        or not CALIBRATION_ROOT_CONSUMED
        or PARAMETER_TRANSPLANT_ALLOWED
        or not EVALUATOR_MATCHED_OUTER_WORK_CLAIMED
        or CORE_MATCHED_OUTER_WORK_CLAIMED
        or CORE_HOST_TRANSITION_BINDING_CLAIMED
        or not CORE_STATE_CONTENT_INTEGRITY_CLAIMED
        or CORE_EXTERNAL_STATE_PROVENANCE_CLAIMED
        or not FULL_PANEL_PROCESS_LOCAL_AT_MOST_ONCE
    ):
        errors.append("the development-only nonpromotion surface drifted")
    source = inspect.getsource(HiddenRuleSequentialLineageRetentionEvaluator.step)
    for forbidden_call in (".rule_of(", ".phase_index_of(", ".observe("):
        if forbidden_call in source:
            errors.append(f"the learner transaction contains forbidden call {forbidden_call}")
    if "source_reward_weights=source_reward_weights" not in inspect.getsource(
        HiddenRuleSequentialLineageRetentionEvaluator._event
    ):
        errors.append("the event no longer binds the pre-update reward weights")
    import_hashes = dict(_IMPORT_TIME_SELECTED_SOURCE_HASHES)
    if (
        import_hashes.get("sequential_lineage_cache_core_sha256")
        != EXPECTED_SEQUENTIAL_LINEAGE_CORE_SHA256
    ):
        errors.append("the frozen sequential-lineage core hash differs")
    mechanism = SequentialLineageCache(SEQUENTIAL_LINEAGE_CONFIG)
    resources = mechanism.resource_record(
        n_agents=N_AGENTS,
        base_scan_carry_nbytes=EXPECTED_BASE_SCAN_CARRY_NBYTES,
    )
    work = mechanism.work_record(total_steps=NUM_STEPS, n_agents=N_AGENTS)
    if (
        resources.per_agent_state_nbytes != EXPECTED_SIDECAR_NBYTES_PER_AGENT
        or resources.joint_state_nbytes != EXPECTED_SIDECAR_PAIR_NBYTES
        or resources.total_scan_carry_nbytes != EXPECTED_COMPOSITE_SCAN_CARRY_NBYTES
    ):
        errors.append("the exact base-plus-sidecar resource formula drifted")
    if resources.parameter_transplant_allowed:
        errors.append("the core unexpectedly allows parameter transplant")
    if work.matched_outer_work_claimed:
        errors.append("the core unexpectedly claims matched outer work")
    if work.replay_updates or work.random_draws or work.reset_callbacks:
        errors.append("the sidecar added replay, randomness, or reset work")
    if PROTOCOL.to_config()["conditions"] != list(CONDITIONS):
        errors.append("the protocol condition order drifted")
    return tuple(errors)


def _run_condition(
    epsilon: float,
    condition: SequentialLineageRetentionCondition,
) -> dict[str, object]:
    evaluator = HiddenRuleSequentialLineageRetentionEvaluator(epsilon, condition)
    initial = evaluator.initialize()
    initial_sha256 = _tree_sha256(initial)
    initial_base_sha256 = _tree_sha256(initial.base)
    initial_sidecar_pair_sha256 = _tree_sha256((initial.sidecar_0, initial.sidecar_1))
    resources = evaluator.resource_budget(initial)
    work = evaluator.work_budget(NUM_STEPS)
    final, trace = evaluator._scan_life(initial)
    if not bool(jnp.all(trace.outer_update_applied)):
        raise RuntimeError("the full sequential-lineage life contained an outer rollback")
    if not bool(jnp.all(trace.all_or_none_commit_valid)):
        raise RuntimeError("the full sequential-lineage life violated atomic commit")
    if not bool(jnp.all(trace.event_binding_valid)):
        raise RuntimeError("the full sequential-lineage life violated event binding")
    if not bool(jnp.all(trace.pre_update_weight_binding_valid)):
        raise RuntimeError("the full sequential-lineage life violated pre-weight binding")
    if not bool(jnp.all(trace.proposal_fields_valid)):
        raise RuntimeError("the full sequential-lineage life violated proposal fields")
    if not bool(jnp.all(trace.candidate_sidecar_valid)):
        raise RuntimeError("the full sequential-lineage life produced an invalid sidecar")
    if bool(jnp.any(trace.proposal_parameter_transplanted)):
        raise RuntimeError("the full sequential-lineage life reported a parameter transplant")
    return {
        "epsilon": epsilon,
        "condition": condition,
        "initial_state_sha256": initial_sha256,
        "initial_base_sha256": initial_base_sha256,
        "initial_sidecar_pair_sha256": initial_sidecar_pair_sha256,
        "final_state_sha256": _tree_sha256(final),
        "trace_sha256": _tree_sha256(trace),
        "controller_rng_trace_sha256": _tree_sha256(trace.capacity.controller_rng_key_words),
        "resource_budget": resources.to_dict(),
        "work_budget": work.to_dict(),
        "metrics": {
            "mean_common_reward": float(jnp.mean(trace.capacity.reward)),
            "allocation_count": int(jnp.sum(trace.capacity.allocations)),
            "eviction_count": int(jnp.sum(trace.capacity.evictions)),
            "full_bank_birth_count": int(jnp.sum(trace.proposal_full_bank_birth)),
            "cache_test_count": int(jnp.sum(trace.proposal_cache_tested)),
            "quarantine_open_count": int(jnp.sum(trace.proposal_quarantine_opened)),
            "quarantine_confirmation_count": int(jnp.sum(trace.proposal_quarantine_confirmed)),
            "lineage_transfer_count": int(jnp.sum(trace.proposal_lineage_transferred)),
            "rescue_increment_count": int(jnp.sum(trace.proposal_rescue_incremented)),
            "eviction_target_adjustment_count": int(
                jnp.sum(trace.context_eviction_target_adjusted)
            ),
            "outer_commit_count": int(jnp.sum(trace.outer_update_applied)),
            "parameter_transplant_count": int(jnp.sum(trace.proposal_parameter_transplanted)),
        },
        "final_step_words": [int(value) for value in np.asarray(final.base.environment.step_words)],
    }


LIMITATIONS: Final = (
    "one already-consumed development root with no statistical inference",
    "a synthetic four-rule convention game is not broad transfer or physical embodiment",
    "H=2 relational evidence is bounded confirmation, not semantic ground truth",
    "the sidecar core does not authenticate the host transition supplied by this evaluator",
    "agent namespaces are evaluator-owned and not authenticated by the sidecar core",
    "the source manifest binds selected direct files, not a transitive source closure",
    "runtime identity binds selected fields, not an environment or compiler closure",
    "resource and work records exclude compiler workspaces, FLOPs, and latency",
    "no threshold, default, winner, writer, artifact, evidence, or promotion path",
)


def _build_report() -> dict[str, object]:
    errors = validate_static_contract()
    if errors:
        raise RuntimeError(f"sequential-lineage static contract failed: {errors}")
    source_manifest = _bound_source_manifest(stage="pre-run")
    runtime_identity = _runtime_identity()
    runs = [
        _run_condition(epsilon, condition)
        for epsilon in capacity_pressure.EPSILON_GRID
        for condition in CONDITIONS
    ]
    if _bound_source_manifest(stage="post-run") != source_manifest:
        raise RuntimeError("selected sequential-lineage sources changed during the panel")
    if _runtime_identity() != runtime_identity:
        raise RuntimeError("selected sequential-lineage runtime identity changed during the panel")
    for epsilon in capacity_pressure.EPSILON_GRID:
        pair = [run for run in runs if run["epsilon"] == epsilon]
        if len(pair) != len(CONDITIONS):
            raise RuntimeError("the sequential-lineage condition pair is incomplete")
        if len({cast(str, run["initial_state_sha256"]) for run in pair}) != 1:
            raise RuntimeError("paired conditions do not share exact composite genesis")
        if len({cast(str, run["initial_base_sha256"]) for run in pair}) != 1:
            raise RuntimeError("paired conditions do not share exact base genesis")
        if len({cast(str, run["initial_sidecar_pair_sha256"]) for run in pair}) != 1:
            raise RuntimeError("paired conditions do not share exact sidecar genesis")
        if len({_canonical_json(run["resource_budget"]) for run in pair}) != 1:
            raise RuntimeError("paired conditions do not have identical resources")
        if len({_canonical_json(run["work_budget"]) for run in pair}) != 1:
            raise RuntimeError("paired conditions do not have identical work")
        if len({cast(str, run["controller_rng_trace_sha256"]) for run in pair}) != 1:
            raise RuntimeError("paired conditions do not have identical RNG advance")
    body: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "acceptance_status": ACCEPTANCE_STATUS,
        "development_only": DEVELOPMENT_ONLY,
        "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        "evidence_authorized": EVIDENCE_AUTHORIZED,
        "accepted_scientific_evidence": False,
        "output_writes_allowed": OUTPUT_WRITES_ALLOWED,
        "writer_available": WRITER_AVAILABLE,
        "artifact_bytes_written": ARTIFACT_BYTES_WRITTEN,
        "thresholds_evaluated": THRESHOLDS_USED,
        "winner_selected": False,
        "default_condition_available": DEFAULT_CONDITION_AVAILABLE,
        "protocol": PROTOCOL.to_config(),
        "protocol_sha256": _json_sha256(PROTOCOL.to_config()),
        "source_manifest": source_manifest,
        "source_manifest_scope": SOURCE_MANIFEST_SCOPE,
        "source_hash_binding": {
            "captured_at_module_import": True,
            "pre_run_disk_equality_required": True,
            "post_run_disk_equality_required": True,
            "sequential_lineage_core_sha256_pinned": True,
        },
        "transitive_source_closure_claimed": False,
        "runtime_identity": runtime_identity,
        "runtime_identity_scope": RUNTIME_IDENTITY_SCOPE,
        "runtime_identity_bound_by_validation": True,
        "runtime_environment_or_compiler_closure_claimed": False,
        "resource_accounting_scope": RESOURCE_ACCOUNTING_SCOPE,
        "condition_order": list(CONDITIONS),
        "runs": runs,
        "matched_comparison": {
            "same_consumed_root": True,
            "bit_identical_composite_genesis_within_epsilon": True,
            "persistent_resources_matched": True,
            "logical_work_matched": True,
            "controller_rng_advance_matched": True,
            "same_prioritized_context_and_sidecar_calls": True,
            "evaluator_matched_outer_work_claimed": True,
            "core_matched_outer_work_claimed": False,
            "varying_field": "dispatched source rescue score versus exact zeros",
        },
        "nonclaims": {
            "parameter_transplant": False,
            "core_host_transition_binding": False,
            "core_external_state_provenance": False,
            "scientific_evidence": "not-assessed",
            "Alberta_Plan_completion": "not-assessed",
        },
        "limitations": list(LIMITATIONS),
    }
    report = _attach_report_hash(body)
    if not _report_hash_reconstructs(report):
        raise RuntimeError("sequential-lineage report hash does not reconstruct")
    return report


_FULL_PANEL_ATTEMPT = _ProcessAttemptLatch(lambda: _canonical_json(_build_report()))


def run_consumed_sequential_lineage_retention_panel() -> str:
    """Return the sole compact in-memory panel report, sealing any first failure."""

    return _FULL_PANEL_ATTEMPT.get()


def run_sequential_lineage_retention_paired_intervention() -> str:
    """Paired-intervention alias backed by the same process-local latch."""

    return run_consumed_sequential_lineage_retention_panel()


__all__ = [
    "ACCEPTANCE_STATUS",
    "AGENT_NAMESPACES",
    "ARBITRARY_ROOT_EXECUTION_ALLOWED",
    "ARTIFACT_BYTES_WRITTEN",
    "CALIBRATION_ROOT_CONSUMED",
    "CONDITIONS",
    "CONSUMED_ROOT_INDEX",
    "CORE_EXTERNAL_STATE_PROVENANCE_CLAIMED",
    "CORE_HOST_TRANSITION_BINDING_CLAIMED",
    "CORE_MATCHED_OUTER_WORK_CLAIMED",
    "CORE_STATE_CONTENT_INTEGRITY_CLAIMED",
    "DEFAULT_CONDITION_AVAILABLE",
    "DEVELOPMENT_ONLY",
    "EVALUATOR_MATCHED_OUTER_WORK_CLAIMED",
    "EVIDENCE_AUTHORIZED",
    "EXPECTED_BASE_SCAN_CARRY_NBYTES",
    "EXPECTED_COMPOSITE_SCAN_CARRY_NBYTES",
    "EXPECTED_SEQUENTIAL_LINEAGE_CORE_SHA256",
    "EXPECTED_SIDECAR_NBYTES_PER_AGENT",
    "EXPECTED_SIDECAR_PAIR_NBYTES",
    "FULL_PANEL_PROCESS_LOCAL_AT_MOST_ONCE",
    "H2_PREDICTIVE_RESCUE",
    "HiddenRuleSequentialLineageRetentionEvaluator",
    "HiddenRuleSequentialLineageRetentionProtocol",
    "HiddenRuleSequentialLineageRetentionState",
    "HiddenRuleSequentialLineageRetentionStepResult",
    "HiddenRuleSequentialLineageRetentionTrace",
    "NO_SIGNAL",
    "OUTPUT_WRITES_ALLOWED",
    "PARAMETER_TRANSPLANT_ALLOWED",
    "PROTOCOL",
    "PROTOCOL_SCHEMA",
    "REPORT_SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SEQUENTIAL_LINEAGE_CONFIG",
    "SEQUENTIAL_LINEAGE_H2_PREDICTIVE_RESCUE",
    "SEQUENTIAL_LINEAGE_NO_SIGNAL",
    "SEQUENTIAL_LINEAGE_RETENTION_CONDITIONS",
    "SEQUENTIAL_LINEAGE_RETENTION_H2_PREDICTIVE_RESCUE",
    "SEQUENTIAL_LINEAGE_RETENTION_NO_SIGNAL",
    "SequentialLineageRetentionState",
    "SequentialLineageRetentionStepResult",
    "SequentialLineageRetentionTrace",
    "SequentialLineageRetentionCondition",
    "SequentialLineageRetentionResourceBudget",
    "SequentialLineageRetentionWorkBudget",
    "THRESHOLDS_USED",
    "WINNER_SELECTION_ALLOWED",
    "WRITER_AVAILABLE",
    "initialize_hidden_rule_sequential_lineage_retention_state",
    "initialize_sequential_lineage_retention_state",
    "run_consumed_sequential_lineage_retention_panel",
    "run_sequential_lineage_retention_paired_intervention",
    "sequential_lineage_retention_resource_budget",
    "sequential_lineage_retention_work_budget",
    "step_hidden_rule_sequential_lineage_retention",
    "step_sequential_lineage_retention",
    "validate_static_contract",
]
