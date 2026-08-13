"""Silent-task contextual control life for autonomous compositional features.

This development-only lane places :class:`CompositionalFeatureLearner` on the
smallest grounded control problem that still requires multi-step feature
discovery.  The learner receives only six raw Rademacher values, its selected
action's scalar reward, and its own persistent state.  Phase identity,
boundaries, and the evaluator's product expressions are never learner inputs.

The primary arms use the production compositional update.  Opt-in novelty
admission gives zero-direct-utility intermediates a bounded route into the
active DAG, while ancestor utility backup lets a useful descendant protect its
parents.  The readout-blocked arm still computes and trains the same full
feature heads; only its behavior-time action values mask the composed tail.

Reports are strict in-memory development records.  They are always
``not-assessed``, have no thresholds or artifact writer, and cannot authorize
evidence or scientific promotion.  Algebraic product presence is tracked, but
the compiled scan does not carry the host-only authenticated v4 birth ledger.
Consequently disappearance/reappearance is reported only as bank-level
structural reacquisition, never as retained or fresh birth identity.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, NamedTuple, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

import alberta_framework.core.compositional_features as compositional_core
from alberta_framework.core.compositional_features import (
    GENERATION_DOVETAIL_PRODUCT_COVERAGE,
    GENERATION_ROBUST_RECURSIVE,
    OP_PRODUCT,
    OP_RAW,
    CompositionalFeatureLearner,
    CompositionalFeatureState,
    CompositionalRankingDiagnostics,
)
from alberta_framework.evaluation.compositional_discovery_development import (
    DEFAULT_DEVELOPMENT_SEEDS,
)
from alberta_framework.evaluation.generated_birth_identity_scrub_epoch import (
    generated_birth_identity_scrub_epoch_core_state_sha256,
)
from alberta_framework.evaluation.generated_class_lifecycle_scrub import (
    persistent_compositional_state_nbytes,
)

PROTOCOL_SCHEMA: Final = "alberta.compositional-control-life-development.protocol.v1"
REPORT_SCHEMA: Final = "alberta.compositional-control-life-development.report.v1"
ARM_EXECUTION_RECEIPT_SCHEMA: Final = (
    "alberta.compositional-control-life-development.arm-execution-receipt.v1"
)
ARM_ANALYSIS_RECEIPT_SCHEMA: Final = (
    "alberta.compositional-control-life-development.arm-analysis-receipt.v1"
)
STATUS: Final = "DEVELOPMENT_SILENT_TASK_CONTROL_NOT_ASSESSED"
ACCEPTANCE_STATUS: Final = "not-assessed"
DEVELOPMENT_ONLY: Final = True
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
EVIDENCE_AUTHORIZED: Final = False

RAW_DIM: Final = 6
ACTIVE_SLOTS: Final = 11
CANDIDATE_SLOTS: Final = 8
ACTION_HEADS: Final = 2
ALLOCATED_MAX_DEPTH: Final = 3
GENERATOR_CONTEXTS: Final = 1
GENERATOR_POLICIES: Final = 4
CURATION_INTERVAL: Final = 32
_VALIDATED_RECEIPT_TOKEN: Final = object()

PHASE_ORDER: Final = ("A", "B", "A", "D", "A", "C", "A", "B", "C", "A")
DEFAULT_PHASE_LENGTHS: Final = (769, 797, 829, 857, 883, 911, 941, 971, 1009, 1031)

SIGNATURE_NAMES: Final = ("A", "B", "C", "D", "shared_p45", "obsolete_p12")
SIGNATURE_ROLES: Final = (
    "recurring_root",
    "recurring_root",
    "recurring_root",
    "one_exposure_obsolete_root",
    "shared_recurring_intermediate",
    "one_exposure_obsolete_intermediate",
)
SIGNATURE_RAW_INDICES: Final = (
    (1, 4, 5),
    (2, 4, 5),
    (3, 4, 5),
    (1, 2, 3),
    (4, 5),
    (1, 2),
)
AUDITED_ADMISSION_SIGNATURE_NAMES: Final = (
    "A",
    "B",
    "C",
    "shared_p45",
    "obsolete_p12",
)
RAW_PAIR_INDICES: Final = tuple(
    (left, right)
    for left in range(RAW_DIM)
    for right in range(left + 1, RAW_DIM)
)
RAW_PAIR_NAMES: Final = tuple(
    f"p{left}{right}" for left, right in RAW_PAIR_INDICES
)

CONSUMED_DEVELOPMENT_SEEDS: Final = DEFAULT_DEVELOPMENT_SEEDS
DEFAULT_CONSUMED_SEED: Final = CONSUMED_DEVELOPMENT_SEEDS[0]

OBSERVATION_DOMAIN: Final = 0x4F425356  # OBSV
EXPLORATION_DOMAIN: Final = 0x4558504C  # EXPL
RANDOM_ACTION_DOMAIN: Final = 0x52414354  # RACT
LEARNER_DOMAIN: Final = 0x4C524E52  # LRNR

CURATION_COUNT_NAMES: Final = (
    "curation_due",
    "proposal",
    "root_change",
    "promotion",
    "cascade_refill",
    "ordinary_candidate_refresh",
    "post_promotion_candidate_refresh",
    "candidate_refresh",
    "candidate_rebound",
    "candidate_overdepth_regeneration",
    "logical_event",
)

INTERPRETATION: Final = (
    "Development-only silent-task contextual-bandit life testing whether bounded "
    "generated features acquire control authority, retain recurring structure, and "
    "retire a one-exposure structure. It is not an Alberta Plan completion result."
)
LIMITATIONS: Final = (
    "finite synthetic product grammar and iid contextual-bandit observations",
    "immediate selected-action reward prediction, not TD control or world modeling",
    "bank-level algebraic recurrence is not authenticated birth-identity reacquisition",
    "no thresholds, held-out seeds, confidence intervals, artifact writer, or promotion path",
    "persistent-state and structural-work accounting excludes compiler workspaces and FLOPs",
    "the depth-one arm matches arrays and update opportunities, not compiled instruction traces",
)


def _exact_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an exact integer in [{minimum}, {maximum}]")
    return value


def _exact_float(value: object, *, name: str, minimum: float, maximum: float) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite exact float")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionalControlLifeArm:
    """One fixed-shape development arm."""

    name: str
    generation_strategy: str
    candidate_novelty_admission_bonus: float
    ancestor_utility_backup_decay: float
    retention_slow_utility_decay: float
    effective_max_depth: int
    topology_headroom_reserve: bool
    topology_left_pack_destinations: bool
    composed_readout_enabled: bool
    role: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise TypeError("arm name must be a nonempty exact string")
        if self.generation_strategy not in {
            GENERATION_ROBUST_RECURSIVE,
            GENERATION_DOVETAIL_PRODUCT_COVERAGE,
        }:
            raise ValueError("arm generation strategy is not in the reviewed set")
        _exact_float(
            self.candidate_novelty_admission_bonus,
            name="candidate_novelty_admission_bonus",
            minimum=0.0,
            maximum=100.0,
        )
        _exact_float(
            self.ancestor_utility_backup_decay,
            name="ancestor_utility_backup_decay",
            minimum=0.0,
            maximum=1.0,
        )
        _exact_float(
            self.retention_slow_utility_decay,
            name="retention_slow_utility_decay",
            minimum=0.0,
            maximum=1.0 - 1e-12,
        )
        _exact_int(
            self.effective_max_depth,
            name="effective_max_depth",
            minimum=1,
            maximum=ALLOCATED_MAX_DEPTH,
        )
        if type(self.topology_headroom_reserve) is not bool:
            raise TypeError("topology_headroom_reserve must be an exact boolean")
        if type(self.topology_left_pack_destinations) is not bool:
            raise TypeError(
                "topology_left_pack_destinations must be an exact boolean"
            )
        if type(self.composed_readout_enabled) is not bool:
            raise TypeError("composed_readout_enabled must be an exact boolean")
        if type(self.role) is not str or not self.role:
            raise TypeError("arm role must be a nonempty exact string")

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


CONTROL_LIFE_ARMS: Final = (
    CompositionalControlLifeArm(
        name="myopic_full",
        generation_strategy=GENERATION_ROBUST_RECURSIVE,
        candidate_novelty_admission_bonus=0.0,
        ancestor_utility_backup_decay=0.0,
        retention_slow_utility_decay=0.999,
        effective_max_depth=3,
        topology_headroom_reserve=False,
        topology_left_pack_destinations=False,
        composed_readout_enabled=True,
        role="current myopic production lifecycle reference",
    ),
    CompositionalControlLifeArm(
        name="explore_ancestor",
        generation_strategy=GENERATION_ROBUST_RECURSIVE,
        candidate_novelty_admission_bonus=1.0,
        ancestor_utility_backup_decay=0.95,
        retention_slow_utility_decay=0.999,
        effective_max_depth=3,
        topology_headroom_reserve=False,
        topology_left_pack_destinations=False,
        composed_readout_enabled=True,
        role="novelty admission plus descendant-to-ancestor retention candidate",
    ),
    CompositionalControlLifeArm(
        name="dovetail_coverage_ancestor",
        generation_strategy=GENERATION_DOVETAIL_PRODUCT_COVERAGE,
        candidate_novelty_admission_bonus=1.0,
        ancestor_utility_backup_decay=0.95,
        retention_slow_utility_decay=0.999,
        effective_max_depth=3,
        topology_headroom_reserve=False,
        topology_left_pack_destinations=False,
        composed_readout_enabled=True,
        role="task-agnostic product coverage plus ancestor retention candidate",
    ),
    CompositionalControlLifeArm(
        name="dovetail_coverage_ancestor_headroom",
        generation_strategy=GENERATION_DOVETAIL_PRODUCT_COVERAGE,
        candidate_novelty_admission_bonus=1.0,
        ancestor_utility_backup_decay=0.95,
        retention_slow_utility_decay=0.999,
        effective_max_depth=3,
        topology_headroom_reserve=True,
        topology_left_pack_destinations=False,
        composed_readout_enabled=True,
        role="matched product coverage with remaining-depth topology headroom",
    ),
    CompositionalControlLifeArm(
        name="dovetail_coverage_ancestor_headroom_leftpack",
        generation_strategy=GENERATION_DOVETAIL_PRODUCT_COVERAGE,
        candidate_novelty_admission_bonus=1.0,
        ancestor_utility_backup_decay=0.95,
        retention_slow_utility_decay=0.999,
        effective_max_depth=3,
        topology_headroom_reserve=True,
        topology_left_pack_destinations=True,
        composed_readout_enabled=True,
        role=(
            "matched headroom arm with lowest-index margin-eligible destination "
            "placement"
        ),
    ),
    CompositionalControlLifeArm(
        name="explore_ancestor_readout_blocked",
        generation_strategy=GENERATION_ROBUST_RECURSIVE,
        candidate_novelty_admission_bonus=1.0,
        ancestor_utility_backup_decay=0.95,
        retention_slow_utility_decay=0.999,
        effective_max_depth=3,
        topology_headroom_reserve=False,
        topology_left_pack_destinations=False,
        composed_readout_enabled=False,
        role="matched composed-feature behavioral-authority ablation",
    ),
    CompositionalControlLifeArm(
        name="explore_ancestor_no_slow",
        generation_strategy=GENERATION_ROBUST_RECURSIVE,
        candidate_novelty_admission_bonus=1.0,
        ancestor_utility_backup_decay=0.95,
        retention_slow_utility_decay=0.0,
        effective_max_depth=3,
        topology_headroom_reserve=False,
        topology_left_pack_destinations=False,
        composed_readout_enabled=True,
        role="fast-utility-only retention ablation",
    ),
    CompositionalControlLifeArm(
        name="depth1_ceiling",
        generation_strategy=GENERATION_ROBUST_RECURSIVE,
        candidate_novelty_admission_bonus=1.0,
        ancestor_utility_backup_decay=0.95,
        retention_slow_utility_decay=0.999,
        effective_max_depth=1,
        topology_headroom_reserve=False,
        topology_left_pack_destinations=False,
        composed_readout_enabled=True,
        role="same-shape degree-one representational ceiling",
    ),
)
_ARMS_BY_NAME: Final = {arm.name: arm for arm in CONTROL_LIFE_ARMS}
_CANONICAL_ARM_NAMES: Final = tuple(arm.name for arm in CONTROL_LIFE_ARMS)


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionalControlLifeProtocol:
    """Static finite-life protocol; custom lengths exist only for cheap tests."""

    phase_lengths: tuple[int, ...] = DEFAULT_PHASE_LENGTHS
    epsilon: float = 0.1
    entry_window: int = 64
    tail_window: int = 64

    def __post_init__(self) -> None:
        if type(self.phase_lengths) is not tuple or len(self.phase_lengths) != len(
            PHASE_ORDER
        ):
            raise ValueError("phase_lengths must be an exact ten-element tuple")
        checked = tuple(
            _exact_int(value, name="phase length", minimum=1, maximum=2**31 - 1)
            for value in self.phase_lengths
        )
        if len(set(checked)) != len(checked):
            raise ValueError("phase lengths must be nonperiodic and unique")
        _exact_float(self.epsilon, name="epsilon", minimum=0.0, maximum=1.0)
        _exact_int(
            self.entry_window,
            name="entry_window",
            minimum=1,
            maximum=min(checked),
        )
        _exact_int(
            self.tail_window,
            name="tail_window",
            minimum=1,
            maximum=min(checked),
        )
        if self.total_steps > 2**31 - 1:
            raise ValueError("protocol exceeds the signed telemetry horizon")

    @property
    def total_steps(self) -> int:
        return sum(self.phase_lengths)

    @property
    def canonical_schedule(self) -> bool:
        return self.phase_lengths == DEFAULT_PHASE_LENGTHS

    def to_config(self) -> dict[str, object]:
        return {
            "schema": PROTOCOL_SCHEMA,
            "status": STATUS,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "phase_order": list(PHASE_ORDER),
            "phase_lengths": list(self.phase_lengths),
            "total_steps": self.total_steps,
            "canonical_schedule": self.canonical_schedule,
            "epsilon": self.epsilon,
            "entry_window": self.entry_window,
            "tail_window": self.tail_window,
            "raw_dim": RAW_DIM,
            "active_slots": ACTIVE_SLOTS,
            "candidate_slots": CANDIDATE_SLOTS,
            "action_heads": ACTION_HEADS,
            "allocated_max_depth": ALLOCATED_MAX_DEPTH,
            "curation_interval": CURATION_INTERVAL,
            "learner_observation_fields": ["raw_rademacher_values"],
            "learner_feedback_fields": ["selected_action_reward"],
            "evaluator_only_fields": [
                "phase_name",
                "phase_boundary",
                "target_expression",
                "counterfactual_action_reward",
            ],
            "resets_allowed": False,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> CompositionalControlLifeProtocol:
        expected = set(cls().to_config())
        if set(payload) != expected:
            raise ValueError("protocol fields do not match the v1 schema")
        if payload.get("schema") != PROTOCOL_SCHEMA or payload.get("status") != STATUS:
            raise ValueError("protocol schema or status is invalid")
        raw_lengths = payload.get("phase_lengths")
        if not isinstance(raw_lengths, list):
            raise TypeError("phase_lengths must be a JSON list")
        protocol = cls(
            phase_lengths=tuple(cast(list[int], raw_lengths)),
            epsilon=cast(float, payload["epsilon"]),
            entry_window=cast(int, payload["entry_window"]),
            tail_window=cast(int, payload["tail_window"]),
        )
        if protocol.to_config() != dict(payload):
            raise ValueError("protocol payload does not reconstruct exactly")
        return protocol


@dataclasses.dataclass(frozen=True, slots=True)
class BoundCompositionalControlLifeSource:
    """Root-agnostic exogenous arrays derived from four caller-bound keys."""

    key_manifest: Mapping[str, tuple[int, int]]
    observations: Array
    phase_indices: Array
    exploration_mask: Array
    random_actions: Array
    learner_key: Array
    curation_due_mask: Array
    stream_sha256: str
    cadence_bound_stream_sha256: str
    scientific_promotion_allowed: bool = False
    evidence_authorized: bool = False
    output_writes_allowed: bool = False

    def __post_init__(self) -> None:
        if (
            self.scientific_promotion_allowed
            or self.evidence_authorized
            or self.output_writes_allowed
        ):
            raise ValueError("a bound source cannot acquire scientific or output authority")


def build_default_protocol() -> CompositionalControlLifeProtocol:
    """Return the inert canonical 8,998-step development declaration."""

    return CompositionalControlLifeProtocol()


def build_short_test_protocol() -> CompositionalControlLifeProtocol:
    """Return a noncanonical curation-crossing protocol for focused tests."""

    return CompositionalControlLifeProtocol(
        phase_lengths=(33, 35, 37, 39, 41, 43, 45, 47, 49, 51),
        entry_window=8,
        tail_window=8,
    )


def _build_learner(arm: CompositionalControlLifeArm) -> CompositionalFeatureLearner:
    depth_bonus = 0.05 if arm.effective_max_depth > 1 else 0.0
    return CompositionalFeatureLearner(
        n_features=ACTIVE_SLOTS,
        n_tasks=ACTION_HEADS,
        candidate_count=CANDIDATE_SLOTS,
        step_size_output=0.01,
        step_size_theta=0.001,
        utility_decay=0.995,
        replacement_interval=CURATION_INTERVAL,
        min_feature_age=16,
        candidate_min_age=16,
        promotion_margin=1.0,
        promotion_blend=1.0,
        max_depth=arm.effective_max_depth,
        topology_headroom_reserve=arm.topology_headroom_reserve,
        topology_left_pack_destinations=arm.topology_left_pack_destinations,
        use_obgd=True,
        generation_strategy=arm.generation_strategy,
        parent_novelty_weight=0.1,
        parent_depth_prior=0.1 if arm.effective_max_depth > 1 else 0.0,
        retention_depth_bonus=depth_bonus,
        candidate_scoring_mode="energy_novelty",
        candidate_score_trace_decay=0.95,
        candidate_novelty_weight=0.25,
        candidate_novelty_admission_bonus=(
            arm.candidate_novelty_admission_bonus
        ),
        retention_slow_utility_decay=arm.retention_slow_utility_decay,
        ancestor_utility_backup_decay=arm.ancestor_utility_backup_decay,
        operation_prior=(
            None
            if arm.generation_strategy == GENERATION_DOVETAIL_PRODUCT_COVERAGE
            else (0.0, 0.25, 0.25, 0.25, 0.25)
        ),
        generator_resource_contexts=GENERATOR_CONTEXTS,
    )


def learner_config_for_arm(name: str) -> dict[str, Any]:
    """Return the exact production learner config for one declared arm."""

    if type(name) is not str or name not in _ARMS_BY_NAME:
        raise ValueError("unknown compositional control-life arm")
    return _build_learner(_ARMS_BY_NAME[name]).to_config()


def compositional_control_state_nbytes_formula(
    *,
    active_slots: int,
    candidate_slots: int,
    action_heads: int,
    generator_contexts: int = GENERATOR_CONTEXTS,
    generator_policies: int = GENERATOR_POLICIES,
) -> int:
    """Exact persistent-array bytes for the current compositional state schema."""

    for name, value in (
        ("active_slots", active_slots),
        ("action_heads", action_heads),
        ("generator_contexts", generator_contexts),
        ("generator_policies", generator_policies),
    ):
        _exact_int(value, name=name, minimum=1, maximum=2**31 - 1)
    _exact_int(
        candidate_slots,
        name="candidate_slots",
        minimum=0,
        maximum=2**31 - 1,
    )
    return (
        (56 + 12 * action_heads) * active_slots
        + 4 * active_slots * candidate_slots
        + (68 + 12 * action_heads) * candidate_slots
        + 12 * action_heads
        + 12 * generator_contexts * generator_policies
        + 32
    )


def _source_manifest() -> dict[str, str]:
    module_path = Path(__file__).resolve()
    core_path = Path(compositional_core.__file__).resolve()
    return {
        "evaluation_module_sha256": _sha256_file(module_path),
        "compositional_core_sha256": _sha256_file(core_path),
    }


def _key_words(key: Array) -> list[int]:
    if key.shape != () or str(jr.key_impl(key)) != "threefry2x32":
        raise ValueError("development keys must be scalar typed Threefry keys")
    words = np.asarray(jr.key_data(key), dtype=np.uint32)
    if words.shape != (2,):
        raise ValueError("typed key must contain exactly two uint32 words")
    return [int(words[0]), int(words[1])]


def _array_tree_sha256(tree: object) -> str:
    digest = hashlib.sha256()
    for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        if isinstance(leaf, Array) and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            leaf.dtype,
            jax.dtypes.prng_key,
        ):
            array = np.asarray(jr.key_data(leaf), dtype=np.uint32)
            dtype = "typed-prng-threefry2x32"
        else:
            array = np.asarray(leaf)
            dtype = array.dtype.str
        if array.dtype.hasobject:
            raise TypeError("object arrays cannot enter a trace hash")
        contiguous = np.ascontiguousarray(array)
        metadata = _canonical_json_bytes(
            {
                "index": index,
                "dtype": dtype,
                "shape": list(contiguous.shape),
                "nbytes": int(contiguous.nbytes),
            }
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _exponent_matrix(raw_indices: Sequence[Sequence[int]]) -> Array:
    signatures = np.zeros((len(raw_indices), RAW_DIM), dtype=np.int32)
    for row, indices in enumerate(raw_indices):
        for raw_index in indices:
            signatures[row, raw_index] += 1
    return jnp.asarray(signatures, dtype=jnp.int32)


_SIGNATURE_MATRIX: Final = _exponent_matrix(SIGNATURE_RAW_INDICES)
_RAW_PAIR_MATRIX: Final = _exponent_matrix(RAW_PAIR_INDICES)


def _product_signature_slot_matches(
    state: CompositionalFeatureState,
) -> tuple[Array, Array, Array, Array]:
    """Bind exact RAW/PRODUCT monomials to active and candidate slot ids."""

    n_features = state.ops.shape[0]
    signatures = jnp.zeros((n_features, RAW_DIM), dtype=jnp.int32)
    valid = jnp.zeros((n_features,), dtype=jnp.bool_)

    def active_slot(
        slot: int,
        carry: tuple[Array, Array],
    ) -> tuple[Array, Array]:
        values, validity = carry
        op = state.ops[slot]
        pa = state.parent_a[slot]
        pb = state.parent_b[slot]
        safe_pa = jnp.clip(pa, 0, n_features - 1)
        safe_pb = jnp.clip(pb, 0, n_features - 1)
        raw_valid = (op == OP_RAW) & (pa >= 0) & (pa < RAW_DIM)
        product_valid = (
            (op == OP_PRODUCT)
            & (pa >= 0)
            & (pa < slot)
            & (pb >= 0)
            & (pb < slot)
            & validity[safe_pa]
            & validity[safe_pb]
        )
        raw_signature = jax.nn.one_hot(
            jnp.clip(pa, 0, RAW_DIM - 1), RAW_DIM, dtype=jnp.int32
        )
        product_signature = values[safe_pa] + values[safe_pb]
        signature = jnp.where(raw_valid, raw_signature, 0)
        signature = jnp.where(product_valid, product_signature, signature)
        is_valid = raw_valid | product_valid
        return values.at[slot].set(signature), validity.at[slot].set(is_valid)

    signatures, valid = jax.lax.fori_loop(
        0,
        n_features,
        active_slot,
        (signatures, valid),
    )
    active_matches = valid[:, None] & jnp.all(
        signatures[:, None, :] == _SIGNATURE_MATRIX[None, :, :], axis=-1
    )

    safe_pa = jnp.clip(state.candidate_parent_a, 0, n_features - 1)
    safe_pb = jnp.clip(state.candidate_parent_b, 0, n_features - 1)
    candidate_product_valid = (
        (state.candidate_ops == OP_PRODUCT)
        & (state.candidate_parent_a >= 0)
        & (state.candidate_parent_a < n_features)
        & (state.candidate_parent_b >= 0)
        & (state.candidate_parent_b < n_features)
        & valid[safe_pa]
        & valid[safe_pb]
    )
    candidate_signatures = signatures[safe_pa] + signatures[safe_pb]
    candidate_matches = candidate_product_valid[:, None] & jnp.all(
        candidate_signatures[:, None, :] == _SIGNATURE_MATRIX[None, :, :],
        axis=-1,
    )
    active_pair_matches = valid[:, None] & jnp.all(
        signatures[:, None, :] == _RAW_PAIR_MATRIX[None, :, :], axis=-1
    )
    candidate_pair_matches = candidate_product_valid[:, None] & jnp.all(
        candidate_signatures[:, None, :] == _RAW_PAIR_MATRIX[None, :, :],
        axis=-1,
    )
    return active_matches, candidate_matches, active_pair_matches, candidate_pair_matches


def _product_signature_counts(
    state: CompositionalFeatureState,
) -> tuple[Array, Array, Array, Array]:
    """Count exact RAW/PRODUCT monomials in active and candidate banks."""

    active, candidate, active_pairs, candidate_pairs = (
        _product_signature_slot_matches(state)
    )
    return tuple(
        jnp.sum(matches.astype(jnp.int32), axis=0)
        for matches in (active, candidate, active_pairs, candidate_pairs)
    )  # type: ignore[return-value]


def product_signature_counts(state: CompositionalFeatureState) -> dict[str, object]:
    """Return host-readable algebraic counts without claiming birth identity."""

    active, candidate, raw_pair_active, raw_pair_candidate = (
        _product_signature_counts(state)
    )
    return {
        "active": {
            name: int(value)
            for name, value in zip(SIGNATURE_NAMES, np.asarray(active), strict=True)
        },
        "candidate": {
            name: int(value)
            for name, value in zip(
                SIGNATURE_NAMES, np.asarray(candidate), strict=True
            )
        },
        "raw_pair_active": {
            name: int(value)
            for name, value in zip(
                RAW_PAIR_NAMES, np.asarray(raw_pair_active), strict=True
            )
        },
        "raw_pair_candidate": {
            name: int(value)
            for name, value in zip(
                RAW_PAIR_NAMES, np.asarray(raw_pair_candidate), strict=True
            )
        },
    }


def _phase_target(observation: Array, phase_index: Array) -> Array:
    values = jnp.asarray(
        (
            observation[1] * observation[4] * observation[5],
            observation[2] * observation[4] * observation[5],
            observation[1] * observation[4] * observation[5],
            observation[1] * observation[2] * observation[3],
            observation[1] * observation[4] * observation[5],
            observation[3] * observation[4] * observation[5],
            observation[1] * observation[4] * observation[5],
            observation[2] * observation[4] * observation[5],
            observation[3] * observation[4] * observation[5],
            observation[1] * observation[4] * observation[5],
        ),
        dtype=jnp.float32,
    )
    return values[phase_index]


class _ScanEvents(NamedTuple):
    executed_reward: Array
    greedy_reward: Array
    executed_regret: Array
    greedy_regret: Array
    action: Array
    greedy_action: Array
    explored: Array
    target_value: Array
    full_q: Array
    raw_q: Array
    behavior_q: Array
    core_prediction_matches_full_q: Array
    curation_counts: Array
    lifetime_counter_valid: Array
    lifetime_capacity_available: Array
    ranking_contract_valid: Array
    raw_active_utilities: Array
    slow_active_utilities: Array
    direct_active_scores: Array
    backed_active_scores: Array
    raw_candidate_utilities: Array
    slow_candidate_utilities: Array
    direct_candidate_scores: Array
    candidate_novelty_scores: Array
    augmented_candidate_scores: Array
    candidate_mature: Array
    curation_trace: compositional_core.CompositionalCurationTrace
    pre_active_signature_slots: Array
    pre_candidate_signature_slots: Array
    post_active_signature_slots: Array
    post_candidate_signature_slots: Array
    active_signature_counts: Array
    candidate_signature_counts: Array
    active_raw_pair_counts: Array
    candidate_raw_pair_counts: Array
    eligible_recursive_parent_exists: Array


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionalControlLifeArmExecution:
    """Authority-free result of one bounded compositional control-life arm.

    This is a reusable mechanism boundary, not a protocol runner.  It has no
    root-selection, output-writing, evidence, or scientific-promotion authority.
    """

    initial_state: CompositionalFeatureState
    final_state: CompositionalFeatureState
    events: _ScanEvents
    initial_ranking_diagnostics: CompositionalRankingDiagnostics
    initial_active_signature_counts: Array
    initial_candidate_signature_counts: Array
    initial_active_raw_pair_counts: Array
    initial_candidate_raw_pair_counts: Array
    initial_state_sha256: str
    final_state_sha256: str
    trace_sha256: str
    expected_persistent_state_nbytes: int
    initial_persistent_state_nbytes: int
    final_persistent_state_nbytes: int
    scientific_promotion_allowed: bool = False
    evidence_authorized: bool = False
    output_writes_allowed: bool = False

    def __post_init__(self) -> None:
        if (
            self.scientific_promotion_allowed
            or self.evidence_authorized
            or self.output_writes_allowed
        ):
            raise ValueError("an arm execution cannot acquire scientific or output authority")


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionalControlLifeArmExecutionReceipt:
    """Immutable integrity projection produced only by the public validator.

    This receipt binds bytes supplied to one validation call.  It is not a
    replay proof and does not by itself authenticate a learner or source.
    """

    total_steps: int
    initial_state_sha256: str
    final_state_sha256: str
    trace_sha256: str
    expected_persistent_state_nbytes: int
    initial_persistent_state_nbytes: int
    final_persistent_state_nbytes: int
    final_step_count: int
    final_step_words_uint32: tuple[int, int]
    final_replacement_phase: int
    initial_state_finite: bool
    final_state_finite: bool
    all_lifetime_counters_valid: bool
    all_lifetime_capacity_available: bool
    all_ranking_contracts_valid: bool
    all_core_predictions_match_full_q: bool
    initial_target_signature_counts_zero: bool
    scientific_promotion_allowed: bool = False
    evidence_authorized: bool = False
    output_writes_allowed: bool = False
    _validation_token: object = dataclasses.field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._validation_token is not _VALIDATED_RECEIPT_TOKEN:
            raise TypeError("execution receipts must be produced by the validator")
        _exact_int(
            self.total_steps,
            name="execution receipt total_steps",
            minimum=1,
            maximum=2**31 - 1,
        )
        for hash_name, hash_value in (
            ("initial_state_sha256", self.initial_state_sha256),
            ("final_state_sha256", self.final_state_sha256),
            ("trace_sha256", self.trace_sha256),
        ):
            if not _is_sha256(hash_value):
                raise ValueError(
                    f"execution receipt {hash_name} is not a canonical SHA-256"
                )
        for byte_name, byte_value in (
            (
                "expected_persistent_state_nbytes",
                self.expected_persistent_state_nbytes,
            ),
            (
                "initial_persistent_state_nbytes",
                self.initial_persistent_state_nbytes,
            ),
            (
                "final_persistent_state_nbytes",
                self.final_persistent_state_nbytes,
            ),
        ):
            _exact_int(
                byte_value,
                name=byte_name,
                minimum=1,
                maximum=2**63 - 1,
            )
        if not (
            self.expected_persistent_state_nbytes
            == self.initial_persistent_state_nbytes
            == self.final_persistent_state_nbytes
        ):
            raise ValueError("execution receipt persistent-state byte counts disagree")
        _exact_int(
            self.final_step_count,
            name="execution receipt final_step_count",
            minimum=0,
            maximum=2**31 - 1,
        )
        if self.final_step_count != self.total_steps:
            raise ValueError("execution receipt final step count does not close")
        if (
            type(self.final_step_words_uint32) is not tuple
            or len(self.final_step_words_uint32) != 2
            or any(
                type(value) is not int or not 0 <= value <= 2**32 - 1
                for value in self.final_step_words_uint32
            )
            or self.final_step_words_uint32 != (0, self.total_steps)
        ):
            raise ValueError("execution receipt exact lifetime words do not close")
        _exact_int(
            self.final_replacement_phase,
            name="execution receipt final_replacement_phase",
            minimum=0,
            maximum=CURATION_INTERVAL - 1,
        )
        if self.final_replacement_phase != self.total_steps % CURATION_INTERVAL:
            raise ValueError("execution receipt replacement phase does not close")
        closures = (
            ("initial_state_finite", self.initial_state_finite),
            ("final_state_finite", self.final_state_finite),
            ("all_lifetime_counters_valid", self.all_lifetime_counters_valid),
            (
                "all_lifetime_capacity_available",
                self.all_lifetime_capacity_available,
            ),
            ("all_ranking_contracts_valid", self.all_ranking_contracts_valid),
            (
                "all_core_predictions_match_full_q",
                self.all_core_predictions_match_full_q,
            ),
            (
                "initial_target_signature_counts_zero",
                self.initial_target_signature_counts_zero,
            ),
        )
        if any(type(value) is not bool or not value for _name, value in closures):
            failed = ", ".join(
                name for name, value in closures if type(value) is not bool or not value
            )
            raise ValueError(f"execution receipt semantic closure failed: {failed}")
        authority = (
            self.scientific_promotion_allowed,
            self.evidence_authorized,
            self.output_writes_allowed,
        )
        if any(type(value) is not bool for value in authority) or any(authority):
            raise ValueError("execution receipt cannot acquire scientific or output authority")

    def to_config(self) -> dict[str, object]:
        """Return a fresh, strict JSON-native receipt payload."""

        return {
            "schema": ARM_EXECUTION_RECEIPT_SCHEMA,
            "total_steps": self.total_steps,
            "initial_state_sha256": self.initial_state_sha256,
            "final_state_sha256": self.final_state_sha256,
            "trace_sha256": self.trace_sha256,
            "expected_persistent_state_nbytes": self.expected_persistent_state_nbytes,
            "initial_persistent_state_nbytes": self.initial_persistent_state_nbytes,
            "final_persistent_state_nbytes": self.final_persistent_state_nbytes,
            "final_step_count": self.final_step_count,
            "final_step_words_uint32": list(self.final_step_words_uint32),
            "final_replacement_phase": self.final_replacement_phase,
            "initial_state_finite": self.initial_state_finite,
            "final_state_finite": self.final_state_finite,
            "all_lifetime_counters_valid": self.all_lifetime_counters_valid,
            "all_lifetime_capacity_available": self.all_lifetime_capacity_available,
            "all_ranking_contracts_valid": self.all_ranking_contracts_valid,
            "all_core_predictions_match_full_q": (
                self.all_core_predictions_match_full_q
            ),
            "initial_target_signature_counts_zero": (
                self.initial_target_signature_counts_zero
            ),
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
            "evidence_authorized": self.evidence_authorized,
            "output_writes_allowed": self.output_writes_allowed,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionalControlLifeArmAnalysisReceipt:
    """Immutable curation analysis with a fresh JSON projection on access."""

    curation_geometry_arm_name: str
    execution_receipt: CompositionalControlLifeArmExecutionReceipt
    curation_totals: tuple[int, ...]
    _active_structural_trajectories_json: str
    _candidate_structural_trajectories_json: str
    _curation_decision_audit_json: str
    curation_decision_audit_array_elements: int
    curation_decision_audit_ephemeral_bytes: int
    scientific_promotion_allowed: bool = False
    evidence_authorized: bool = False
    output_writes_allowed: bool = False
    _validation_token: object = dataclasses.field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            type(self.curation_geometry_arm_name) is not str
            or self.curation_geometry_arm_name not in _ARMS_BY_NAME
        ):
            raise ValueError("analysis receipt curation geometry arm is not declared")
        if self._validation_token is not _VALIDATED_RECEIPT_TOKEN:
            raise TypeError("analysis receipts must be produced by the analyzer")
        if (
            type(self.execution_receipt)
            is not CompositionalControlLifeArmExecutionReceipt
        ):
            raise TypeError("analysis receipt execution receipt has an invalid type")
        if (
            type(self.curation_totals) is not tuple
            or len(self.curation_totals) != len(CURATION_COUNT_NAMES)
            or any(type(value) is not int or value < 0 for value in self.curation_totals)
        ):
            raise ValueError("analysis receipt curation totals are invalid")
        if (
            self.curation_totals[0]
            != self.execution_receipt.total_steps // CURATION_INTERVAL
        ):
            raise ValueError("analysis receipt curation-due total does not close")

        decoded: list[object] = []
        for name, payload in (
            (
                "active structural trajectories",
                self._active_structural_trajectories_json,
            ),
            (
                "candidate structural trajectories",
                self._candidate_structural_trajectories_json,
            ),
            ("curation decision audit", self._curation_decision_audit_json),
        ):
            if type(payload) is not str:
                raise TypeError(f"analysis receipt {name} must be canonical JSON text")
            try:
                value = json.loads(payload)
            except (TypeError, ValueError) as error:
                raise ValueError(f"analysis receipt {name} is not valid JSON") from error
            if _canonical_json_bytes(value).decode("ascii") != payload:
                raise ValueError(f"analysis receipt {name} is not canonical JSON")
            decoded.append(value)
        active, candidate, audit = decoded
        if (
            not isinstance(active, Mapping)
            or set(active) != set(SIGNATURE_NAMES)
            or not isinstance(candidate, Mapping)
            or set(candidate) != set(SIGNATURE_NAMES)
        ):
            raise ValueError("analysis receipt structural trajectory namespace is invalid")
        if not isinstance(audit, Mapping):
            raise TypeError("analysis receipt curation audit must decode to a mapping")
        _exact_int(
            self.curation_decision_audit_array_elements,
            name="curation_decision_audit_array_elements",
            minimum=1,
            maximum=2**63 - 1,
        )
        _exact_int(
            self.curation_decision_audit_ephemeral_bytes,
            name="curation_decision_audit_ephemeral_bytes",
            minimum=1,
            maximum=2**63 - 1,
        )
        if (
            audit.get("due_curation_event_count") != self.curation_totals[0]
            or audit.get("ephemeral_array_elements")
            != self.curation_decision_audit_array_elements
            or audit.get("ephemeral_array_bytes")
            != self.curation_decision_audit_ephemeral_bytes
        ):
            raise ValueError("analysis receipt curation audit accounting does not close")
        authority = (
            self.scientific_promotion_allowed,
            self.evidence_authorized,
            self.output_writes_allowed,
        )
        if any(type(value) is not bool for value in authority) or any(authority):
            raise ValueError("analysis receipt cannot acquire scientific or output authority")

    @property
    def active_structural_trajectories(self) -> dict[str, object]:
        """Return a fresh copy of the validated active-bank trajectories."""

        return cast(
            dict[str, object],
            json.loads(self._active_structural_trajectories_json),
        )

    @property
    def candidate_structural_trajectories(self) -> dict[str, object]:
        """Return a fresh copy of the validated candidate-bank trajectories."""

        return cast(
            dict[str, object],
            json.loads(self._candidate_structural_trajectories_json),
        )

    @property
    def curation_decision_audit(self) -> dict[str, object]:
        """Return a fresh copy of the fully validated curation audit."""

        return cast(dict[str, object], json.loads(self._curation_decision_audit_json))

    def to_config(self) -> dict[str, object]:
        """Return a fresh, strict JSON-native analysis payload."""

        return {
            "schema": ARM_ANALYSIS_RECEIPT_SCHEMA,
            "curation_geometry_arm_name": self.curation_geometry_arm_name,
            "execution_receipt": self.execution_receipt.to_config(),
            "curation_totals": {
                name: value
                for name, value in zip(
                    CURATION_COUNT_NAMES,
                    self.curation_totals,
                    strict=True,
                )
            },
            "active_structural_trajectories": self.active_structural_trajectories,
            "candidate_structural_trajectories": (
                self.candidate_structural_trajectories
            ),
            "curation_decision_audit": self.curation_decision_audit,
            "curation_decision_audit_array_elements": (
                self.curation_decision_audit_array_elements
            ),
            "curation_decision_audit_ephemeral_bytes": (
                self.curation_decision_audit_ephemeral_bytes
            ),
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
            "evidence_authorized": self.evidence_authorized,
            "output_writes_allowed": self.output_writes_allowed,
        }


@functools.partial(jax.jit, static_argnums=(0, 2))
def _run_compiled_scan(
    learner: CompositionalFeatureLearner,
    initial_state: CompositionalFeatureState,
    composed_readout_enabled: bool,
    observations: Array,
    phase_indices: Array,
    exploration_mask: Array,
    random_actions: Array,
) -> tuple[CompositionalFeatureState, _ScanEvents]:
    """Run one arm as one compiled uninterrupted scan."""

    def step(
        state: CompositionalFeatureState,
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[CompositionalFeatureState, _ScanEvents]:
        observation, phase_index, explore, random_action = inputs
        features = learner.constructed_features(state, observation)
        full_q = state.output_weights @ features + state.output_bias
        raw_features = jnp.where(
            jnp.arange(ACTIVE_SLOTS, dtype=jnp.int32) < RAW_DIM,
            features,
            0.0,
        )
        raw_q = state.output_weights @ raw_features + state.output_bias
        behavior_q = full_q if composed_readout_enabled else raw_q
        greedy_action = jnp.argmax(behavior_q).astype(jnp.int32)
        action = jnp.where(explore, random_action, greedy_action).astype(jnp.int32)
        target_value = _phase_target(observation, phase_index)
        action_sign = 2.0 * action.astype(jnp.float32) - 1.0
        greedy_sign = 2.0 * greedy_action.astype(jnp.float32) - 1.0
        executed_reward = action_sign * target_value
        greedy_reward = greedy_sign * target_value
        targets = jnp.where(
            jnp.arange(ACTION_HEADS, dtype=jnp.int32) == action,
            executed_reward,
            jnp.nan,
        )
        (
            pre_active_signature_slots,
            pre_candidate_signature_slots,
            _pre_active_pair_slots,
            _pre_candidate_pair_slots,
        ) = _product_signature_slot_matches(state)
        result = learner.update(state, observation, targets)
        post_state = result.state
        ranking = learner.ranking_diagnostics(post_state, RAW_DIM)
        (
            post_active_signature_slots,
            post_candidate_signature_slots,
            post_active_pair_slots,
            post_candidate_pair_slots,
        ) = _product_signature_slot_matches(post_state)
        active_counts = jnp.sum(
            post_active_signature_slots.astype(jnp.int32), axis=0
        )
        candidate_counts = jnp.sum(
            post_candidate_signature_slots.astype(jnp.int32), axis=0
        )
        active_pair_counts = jnp.sum(
            post_active_pair_slots.astype(jnp.int32), axis=0
        )
        candidate_pair_counts = jnp.sum(
            post_candidate_pair_slots.astype(jnp.int32), axis=0
        )
        trace = result.curation_trace
        curation_counts = jnp.asarray(
            (
                trace.should_try_replace.astype(jnp.int32),
                trace.proposal_count,
                trace.root_change_count,
                trace.promotion_count,
                trace.cascade_refill_count,
                trace.ordinary_candidate_refresh_count,
                trace.post_promotion_candidate_refresh_count,
                trace.candidate_refresh_count,
                trace.candidate_rebound_count,
                trace.candidate_overdepth_regeneration_count,
                trace.logical_event_count,
            ),
            dtype=jnp.int32,
        )
        events = _ScanEvents(
            executed_reward=executed_reward,
            greedy_reward=greedy_reward,
            executed_regret=1.0 - executed_reward,
            greedy_regret=1.0 - greedy_reward,
            action=action,
            greedy_action=greedy_action,
            explored=explore,
            target_value=target_value,
            full_q=full_q,
            raw_q=raw_q,
            behavior_q=behavior_q,
            core_prediction_matches_full_q=jnp.array_equal(result.predictions, full_q),
            curation_counts=curation_counts,
            lifetime_counter_valid=trace.lifetime_counter_valid,
            lifetime_capacity_available=trace.lifetime_capacity_available,
            ranking_contract_valid=ranking.contract_valid,
            raw_active_utilities=post_state.utilities,
            slow_active_utilities=post_state.retention_slow_utilities,
            direct_active_scores=ranking.direct_active_scores,
            backed_active_scores=ranking.backed_active_scores,
            raw_candidate_utilities=post_state.candidate_utilities,
            slow_candidate_utilities=post_state.candidate_retention_slow_utilities,
            direct_candidate_scores=ranking.direct_candidate_scores,
            candidate_novelty_scores=ranking.candidate_novelty_scores,
            augmented_candidate_scores=ranking.augmented_candidate_scores,
            candidate_mature=ranking.candidate_mature,
            curation_trace=trace,
            pre_active_signature_slots=pre_active_signature_slots,
            pre_candidate_signature_slots=pre_candidate_signature_slots,
            post_active_signature_slots=post_active_signature_slots,
            post_candidate_signature_slots=post_candidate_signature_slots,
            active_signature_counts=active_counts,
            candidate_signature_counts=candidate_counts,
            active_raw_pair_counts=active_pair_counts,
            candidate_raw_pair_counts=candidate_pair_counts,
            eligible_recursive_parent_exists=jnp.any(
                (post_state.depth >= 1)
                & (post_state.depth + 1 <= learner._max_depth)
            ),
        )
        return post_state, events

    return jax.lax.scan(
        step,
        initial_state,
        (observations, phase_indices, exploration_mask, random_actions),
    )


def _state_is_finite(state: CompositionalFeatureState) -> bool:
    for leaf in jax.tree_util.tree_leaves(state):
        if isinstance(leaf, Array) and jnp.issubdtype(leaf.dtype, jnp.inexact):
            if not bool(jnp.all(jnp.isfinite(leaf))):
                return False
        elif type(leaf) is float and not math.isfinite(leaf):
            return False
    return True


def _ranking_record(
    *,
    raw_active: object,
    slow_active: object,
    direct_active: object,
    backed_active: object,
    raw_candidate: object,
    slow_candidate: object,
    direct_candidate: object,
    novelty_candidate: object,
    augmented_candidate: object,
    candidate_mature: object,
    contract_valid: object,
) -> dict[str, object]:
    def floats(value: object) -> list[float]:
        return [float(item) for item in np.asarray(value, dtype=np.float32)]

    return {
        "contract_valid": bool(np.asarray(contract_valid)),
        "raw_active_utilities": floats(raw_active),
        "slow_active_utilities": floats(slow_active),
        "direct_active_scores": floats(direct_active),
        "backed_active_scores": floats(backed_active),
        "raw_candidate_utilities": floats(raw_candidate),
        "slow_candidate_utilities": floats(slow_candidate),
        "direct_candidate_scores": floats(direct_candidate),
        "candidate_novelty_scores": floats(novelty_candidate),
        "augmented_candidate_scores": floats(augmented_candidate),
        "candidate_mature": [bool(item) for item in np.asarray(candidate_mature)],
    }


def _initial_ranking_record(
    state: CompositionalFeatureState,
    diagnostics: CompositionalRankingDiagnostics,
) -> dict[str, object]:
    return _ranking_record(
        raw_active=state.utilities,
        slow_active=state.retention_slow_utilities,
        direct_active=diagnostics.direct_active_scores,
        backed_active=diagnostics.backed_active_scores,
        raw_candidate=state.candidate_utilities,
        slow_candidate=state.candidate_retention_slow_utilities,
        direct_candidate=diagnostics.direct_candidate_scores,
        novelty_candidate=diagnostics.candidate_novelty_scores,
        augmented_candidate=diagnostics.augmented_candidate_scores,
        candidate_mature=diagnostics.candidate_mature,
        contract_valid=diagnostics.contract_valid,
    )


def _event_ranking_record(events: _ScanEvents, index: int) -> dict[str, object]:
    return _ranking_record(
        raw_active=events.raw_active_utilities[index],
        slow_active=events.slow_active_utilities[index],
        direct_active=events.direct_active_scores[index],
        backed_active=events.backed_active_scores[index],
        raw_candidate=events.raw_candidate_utilities[index],
        slow_candidate=events.slow_candidate_utilities[index],
        direct_candidate=events.direct_candidate_scores[index],
        novelty_candidate=events.candidate_novelty_scores[index],
        augmented_candidate=events.augmented_candidate_scores[index],
        candidate_mature=events.candidate_mature[index],
        contract_valid=events.ranking_contract_valid[index],
    )


def _window_metrics(events: _ScanEvents, start: int, stop: int) -> dict[str, float]:
    def mean(value: object) -> float:
        return float(np.mean(np.asarray(value)[start:stop], dtype=np.float64))

    return {
        "executed_reward": mean(events.executed_reward),
        "executed_regret": mean(events.executed_regret),
        "greedy_reward": mean(events.greedy_reward),
        "greedy_regret": mean(events.greedy_regret),
        "positive_action_fraction": mean(events.action),
        "greedy_positive_action_fraction": mean(events.greedy_action),
        "exploration_fraction": mean(events.explored),
    }


def _signature_count_record(counts: object) -> dict[str, int]:
    return {
        name: int(value)
        for name, value in zip(SIGNATURE_NAMES, np.asarray(counts), strict=True)
    }


def _signature_fraction_record(counts: object) -> dict[str, float]:
    array = np.asarray(counts)
    return {
        name: float(np.mean(array[:, index] > 0, dtype=np.float64))
        for index, name in enumerate(SIGNATURE_NAMES)
    }


def _raw_pair_coverage_record(
    initial_active_counts: object,
    initial_candidate_counts: object,
    active_counts: object,
    candidate_counts: object,
) -> dict[str, object]:
    """Summarize all fifteen distinct raw-pair signatures cumulatively."""

    initial_active = np.asarray(initial_active_counts) > 0
    initial_candidate = np.asarray(initial_candidate_counts) > 0
    ever_active = initial_active | np.any(np.asarray(active_counts) > 0, axis=0)
    ever_candidate = initial_candidate | np.any(
        np.asarray(candidate_counts) > 0, axis=0
    )
    ever_either = ever_active | ever_candidate

    def booleans(values: np.ndarray) -> dict[str, bool]:
        return {
            name: bool(value)
            for name, value in zip(RAW_PAIR_NAMES, values, strict=True)
        }

    def bitset(values: np.ndarray) -> int:
        return sum(int(value) << index for index, value in enumerate(values))

    return {
        "pair_order": list(RAW_PAIR_NAMES),
        "pair_count": len(RAW_PAIR_NAMES),
        "initial_active": booleans(initial_active),
        "initial_candidate": booleans(initial_candidate),
        "ever_active": booleans(ever_active),
        "ever_candidate": booleans(ever_candidate),
        "ever_either": booleans(ever_either),
        "new_after_genesis_active": booleans(ever_active & ~initial_active),
        "new_after_genesis_candidate": booleans(
            ever_candidate & ~initial_candidate
        ),
        "initial_active_bitset": bitset(initial_active),
        "initial_candidate_bitset": bitset(initial_candidate),
        "ever_active_bitset": bitset(ever_active),
        "ever_candidate_bitset": bitset(ever_candidate),
        "ever_either_bitset": bitset(ever_either),
        "ever_active_count": int(np.count_nonzero(ever_active)),
        "ever_candidate_count": int(np.count_nonzero(ever_candidate)),
        "ever_either_count": int(np.count_nonzero(ever_either)),
        "missing_either": [
            name
            for name, present in zip(RAW_PAIR_NAMES, ever_either, strict=True)
            if not present
        ],
    }


def _raw_pair_reachability_record(
    arm: CompositionalControlLifeArm,
    initial_recursive_parent_exists: bool,
    events: _ScanEvents,
    cascade_refill_count: int,
) -> dict[str, object]:
    post_recursive = np.asarray(events.eligible_recursive_parent_exists)
    full_trace_recursive = initial_recursive_parent_exists and bool(
        np.all(post_recursive)
    )
    dovetail_enabled = (
        arm.generation_strategy == GENERATION_DOVETAIL_PRODUCT_COVERAGE
    )
    conditional_theorem_applies = full_trace_recursive and not dovetail_enabled
    return {
        "conditional_theorem": (
            "under legacy robust_recursive with no forced operation, whenever at least one "
            "eligible depth>=1 active parent exists, _generate_one selects parent_a "
            "from depth>=1 and parent_b from depth==0; an ordinary fresh candidate "
            "therefore cannot be a raw-by-raw pair"
        ),
        "genesis_active_pair_scaffold": ["p01", "p02", "p03", "p04", "p05"],
        "eligible_recursive_parent_exists_at_genesis": (
            initial_recursive_parent_exists
        ),
        "eligible_recursive_parent_exists_after_every_step": bool(
            np.all(post_recursive)
        ),
        "conditional_theorem_applies_for_entire_observed_life": (
            conditional_theorem_applies
        ),
        "ordinary_fresh_raw_pair_support_for_entire_observed_life": (
            dovetail_enabled or not full_trace_recursive
        ),
        "dovetail_product_coverage_enabled": dovetail_enabled,
        "cascade_refill_is_raw_pair_support_loophole": True,
        "observed_cascade_refill_count": cascade_refill_count,
        "cascade_loophole_exercised": cascade_refill_count > 0,
        "depth1_ceiling_has_ordinary_raw_pair_support": (
            arm.effective_max_depth == 1
        ),
        "scope": (
            "structural proposal support only; nonlinear functional equivalence and "
            "authenticated birth identity are not claimed"
        ),
    }


def _structural_trajectory(
    initial_count: int,
    counts: object,
) -> dict[str, object]:
    values = np.concatenate(
        (
            np.asarray([initial_count > 0], dtype=np.bool_),
            np.asarray(counts) > 0,
        )
    )
    acquisitions = int(values[0]) + int(np.count_nonzero(values[1:] & ~values[:-1]))
    losses = int(np.count_nonzero(~values[1:] & values[:-1]))
    present_indices = np.flatnonzero(values)
    return {
        "initially_present": bool(values[0]),
        "ever_present": bool(present_indices.size),
        "present_at_end": bool(values[-1]),
        "first_present_post_step": (
            None if present_indices.size == 0 else int(present_indices[0])
        ),
        "last_present_post_step": (
            None if present_indices.size == 0 else int(present_indices[-1])
        ),
        "presence_fraction": float(np.mean(values, dtype=np.float64)),
        "acquisition_episode_count": acquisitions,
        "loss_episode_count": losses,
        "structural_reacquisition_count": max(0, acquisitions - 1),
        "identity_reacquisition_claimed": False,
    }


def _active_target_coexistence_record(
    counts: object,
    *,
    start_post_step: int,
) -> dict[str, object]:
    """Summarize exact simultaneous active-bank presence of A, B, and C."""

    values = np.asarray(counts, dtype=np.int64)
    target_indices = tuple(SIGNATURE_NAMES.index(name) for name in ("A", "B", "C"))
    if values.ndim != 2 or values.shape[1] != len(SIGNATURE_NAMES):
        raise RuntimeError("target coexistence telemetry has an invalid shape")
    present = values[:, target_indices] > 0
    active_count = np.sum(present, axis=1, dtype=np.int64)
    histogram = np.bincount(active_count, minlength=4)
    all_three = active_count == 3
    all_three_indices = np.flatnonzero(all_three)
    return {
        "target_order": ["A", "B", "C"],
        "steps": int(values.shape[0]),
        "steps_by_active_target_count": [int(value) for value in histogram],
        "maximum_active_target_count": int(np.max(active_count, initial=0)),
        "all_three_present_steps": int(np.count_nonzero(all_three)),
        "all_three_presence_fraction": float(
            np.mean(all_three, dtype=np.float64) if all_three.size else 0.0
        ),
        "first_all_three_post_step": (
            None
            if all_three_indices.size == 0
            else start_post_step + int(all_three_indices[0]) + 1
        ),
        "last_all_three_post_step": (
            None
            if all_three_indices.size == 0
            else start_post_step + int(all_three_indices[-1]) + 1
        ),
        "active_targets_at_end": (
            []
            if present.shape[0] == 0
            else [
                name
                for name, flag in zip(
                    ("A", "B", "C"), present[-1], strict=True
                )
                if flag
            ]
        ),
    }


def _phase_records(
    protocol: CompositionalControlLifeProtocol,
    initial_ranking: dict[str, object],
    initial_active_counts: Array,
    initial_candidate_counts: Array,
    events: _ScanEvents,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    start = 0
    for index, (phase_name, length) in enumerate(
        zip(PHASE_ORDER, protocol.phase_lengths, strict=True)
    ):
        stop = start + length
        entry_stop = start + protocol.entry_window
        tail_start = stop - protocol.tail_window
        entry_ranking = (
            initial_ranking if start == 0 else _event_ranking_record(events, start - 1)
        )
        entry_active = (
            initial_active_counts
            if start == 0
            else events.active_signature_counts[start - 1]
        )
        entry_candidate = (
            initial_candidate_counts
            if start == 0
            else events.candidate_signature_counts[start - 1]
        )
        records.append(
            {
                "phase_index": index,
                "phase_name": phase_name,
                "start_post_step": start,
                "end_post_step": stop,
                "steps": length,
                "entry_window": _window_metrics(events, start, entry_stop),
                "tail_window": _window_metrics(events, tail_start, stop),
                "whole_phase": _window_metrics(events, start, stop),
                "active_signature_presence_fraction": _signature_fraction_record(
                    events.active_signature_counts[start:stop]
                ),
                "candidate_signature_presence_fraction": _signature_fraction_record(
                    events.candidate_signature_counts[start:stop]
                ),
                "active_target_coexistence": _active_target_coexistence_record(
                    events.active_signature_counts[start:stop],
                    start_post_step=start,
                ),
                "entry_active_signature_counts": _signature_count_record(entry_active),
                "exit_active_signature_counts": _signature_count_record(
                    events.active_signature_counts[stop - 1]
                ),
                "entry_candidate_signature_counts": _signature_count_record(
                    entry_candidate
                ),
                "exit_candidate_signature_counts": _signature_count_record(
                    events.candidate_signature_counts[stop - 1]
                ),
                "entry_ranking": entry_ranking,
                "exit_ranking": _event_ranking_record(events, stop - 1),
            }
        )
        start = stop
    return records


def _f32_bits(value: object) -> int | list[Any]:
    """Return an exact JSON-safe float32 bit representation."""

    array = np.array(value, dtype=np.float32, copy=True, order="C")
    bits = array.view(np.uint32).reshape(array.shape)
    if bits.shape == ():
        return int(bits)
    return cast(list[Any], bits.tolist())


def _f32_from_bits(value: object) -> np.ndarray:
    """Decode an exact JSON-safe float32 bit payload without numeric coercion."""

    bits = np.asarray(value, dtype=np.uint32)
    return bits.view(np.float32).reshape(bits.shape)


def _int_values(value: object) -> list[Any]:
    return cast(list[Any], np.asarray(value, dtype=np.int64).tolist())


def _mask_bitset(value: object) -> int:
    values = np.asarray(value, dtype=np.bool_).reshape(-1)
    return sum(int(flag) << index for index, flag in enumerate(values))


def _signature_slot_record(matches: object) -> dict[str, object]:
    values = np.asarray(matches, dtype=np.bool_)
    if values.ndim != 2 or values.shape[1] != len(SIGNATURE_NAMES):
        raise RuntimeError("signature-slot telemetry has an invalid shape")
    return {
        "slot_count": int(values.shape[0]),
        "match_bitsets": {
            name: _mask_bitset(values[:, index])
            for index, name in enumerate(SIGNATURE_NAMES)
        },
        "matching_slots": {
            name: [int(slot) for slot in np.flatnonzero(values[:, index])]
            for index, name in enumerate(SIGNATURE_NAMES)
        },
    }


def _descriptor_record(
    *,
    ops: object,
    parent_a: object,
    parent_b: object,
    theta: object,
    depth: object,
    generator_policy: object,
) -> dict[str, object]:
    return {
        "ops": _int_values(ops),
        "parent_a": _int_values(parent_a),
        "parent_b": _int_values(parent_b),
        "theta_f32_bits": _f32_bits(theta),
        "depth": _int_values(depth),
        "generator_policy": _int_values(generator_policy),
    }


def _candidate_slot_reason(
    *,
    candidate_slot: int,
    mature: np.ndarray,
    active_eligible: np.ndarray,
    topology: np.ndarray,
    depth: np.ndarray,
    headroom: np.ndarray,
    compatible: np.ndarray,
    selected_candidate: int,
    margin_passed: bool,
    selected_topology_ok: bool,
    selected_depth_ok: bool,
    should_promote: bool,
    promotion_applied: bool,
) -> str:
    if not mature[candidate_slot]:
        return "candidate_immature"
    if not np.any(compatible[candidate_slot]):
        if not np.any(active_eligible):
            return "no_eligible_active_destination"
        if not np.any(active_eligible & topology[candidate_slot]):
            return "topology_blocked"
        if not np.any(
            active_eligible & topology[candidate_slot] & depth[candidate_slot]
        ):
            return "depth_blocked"
        if not np.any(
            active_eligible
            & topology[candidate_slot]
            & depth[candidate_slot]
            & headroom[candidate_slot]
        ):
            return "headroom_blocked"
        return "destination_mask_inconsistent"
    if selected_candidate != candidate_slot:
        return "candidate_selection_competition"
    if not margin_passed:
        return "promotion_margin_failed"
    if not selected_topology_ok:
        return "selected_topology_recheck_failed"
    if not selected_depth_ok:
        return "selected_depth_recheck_failed"
    if not should_promote:
        return "promotion_gate_not_requested"
    if not promotion_applied:
        return "promotion_commit_rollback"
    return "admitted"


def _target_admission_outcome(
    *,
    signature_index: int,
    pre_active_matches: np.ndarray,
    pre_candidate_matches: np.ndarray,
    post_active_matches: np.ndarray,
    trace: compositional_core.CompositionalCurationTrace,
    event_index: int,
) -> dict[str, object]:
    candidate_slots = np.flatnonzero(pre_candidate_matches[:, signature_index])
    pre_active = bool(np.any(pre_active_matches[:, signature_index]))
    post_active = bool(np.any(post_active_matches[:, signature_index]))
    mature = np.asarray(trace.decision_candidate_mature[event_index], dtype=np.bool_)
    active_eligible = np.asarray(
        trace.decision_active_eligible[event_index], dtype=np.bool_
    )
    topology = np.asarray(
        trace.decision_candidate_topology_compatible[event_index], dtype=np.bool_
    )
    depth = np.asarray(
        trace.decision_candidate_depth_compatible[event_index], dtype=np.bool_
    )
    headroom = np.asarray(
        trace.decision_candidate_headroom_compatible[event_index], dtype=np.bool_
    )
    compatible = np.asarray(
        trace.decision_candidate_destination_compatible[event_index], dtype=np.bool_
    )
    selected_candidate = int(trace.decision_selected_candidate[event_index])
    promotion_applied = bool(trace.promotion_applied[event_index])
    slot_outcomes = {
        str(int(slot)): _candidate_slot_reason(
            candidate_slot=int(slot),
            mature=mature,
            active_eligible=active_eligible,
            topology=topology,
            depth=depth,
            headroom=headroom,
            compatible=compatible,
            selected_candidate=selected_candidate,
            margin_passed=bool(trace.decision_margin_passed[event_index]),
            selected_topology_ok=bool(
                trace.decision_selected_topology_ok[event_index]
            ),
            selected_depth_ok=bool(trace.decision_selected_depth_ok[event_index]),
            should_promote=bool(trace.decision_should_promote[event_index]),
            promotion_applied=promotion_applied,
        )
        for slot in candidate_slots
    }
    if pre_active:
        outcome = "already_active"
    elif candidate_slots.size == 0:
        outcome = "candidate_absent"
    elif promotion_applied and selected_candidate in candidate_slots and post_active:
        outcome = "admitted"
    else:
        reasons = tuple(dict.fromkeys(slot_outcomes.values()))
        outcome = reasons[0] if len(reasons) == 1 else "multiple_candidate_constraints"
    return {
        "pre_active": pre_active,
        "post_active": post_active,
        "candidate_slots": [int(slot) for slot in candidate_slots],
        "selected_candidate_is_signature": selected_candidate in candidate_slots,
        "candidate_slot_outcomes": slot_outcomes,
        "outcome": outcome,
    }


def _active_p45_loss_record(
    trace: compositional_core.CompositionalCurationTrace,
    event_index: int,
    lost_slots: np.ndarray,
) -> dict[str, object]:
    root = np.asarray(trace.root_change_mask[event_index], dtype=np.bool_)
    cascade = np.asarray(trace.cascade_refill_mask[event_index], dtype=np.bool_)
    causes: dict[str, list[str]] = {}
    for raw_slot in lost_slots:
        slot = int(raw_slot)
        slot_causes: list[str] = []
        if root[slot]:
            slot_causes.append("promotion_root_replacement")
        if cascade[slot]:
            slot_causes.append("cascade_dependency_refill")
        if not slot_causes:
            slot_causes.append("unmarked_signature_dependency_change")
        causes[str(slot)] = slot_causes
    return {
        "bank_loss": True,
        "lost_slots": [int(slot) for slot in lost_slots],
        "slot_causes": causes,
        "all_lost_slots_accounted": all(
            "unmarked_signature_dependency_change" not in value
            for value in causes.values()
        ),
    }


def _active_signature_transition_record(
    trace: compositional_core.CompositionalCurationTrace,
    event_index: int,
    pre_slots: np.ndarray,
    post_slots: np.ndarray,
) -> dict[str, object]:
    """Attribute every active signature slot transition to exact curation masks."""

    root = np.asarray(trace.root_change_mask[event_index], dtype=np.bool_)
    cascade = np.asarray(trace.cascade_refill_mask[event_index], dtype=np.bool_)

    def slot_causes(slots: np.ndarray) -> dict[str, list[str]]:
        causes: dict[str, list[str]] = {}
        for raw_slot in slots:
            slot = int(raw_slot)
            labels: list[str] = []
            if root[slot]:
                labels.append("promotion_root_replacement")
            if cascade[slot]:
                labels.append("cascade_dependency_refill")
            if not labels:
                labels.append("unmarked_signature_dependency_change")
            causes[str(slot)] = labels
        return causes

    acquired_slots = np.flatnonzero(post_slots & ~pre_slots)
    lost_slots = np.flatnonzero(pre_slots & ~post_slots)
    acquired_causes = slot_causes(acquired_slots)
    lost_causes = slot_causes(lost_slots)
    all_changed_slots_accounted = all(
        "unmarked_signature_dependency_change" not in labels
        for labels in (*acquired_causes.values(), *lost_causes.values())
    )
    return {
        "pre_present": bool(np.any(pre_slots)),
        "post_present": bool(np.any(post_slots)),
        "bank_acquisition": bool(not np.any(pre_slots) and np.any(post_slots)),
        "bank_loss": bool(np.any(pre_slots) and not np.any(post_slots)),
        "acquired_slots": [int(slot) for slot in acquired_slots],
        "lost_slots": [int(slot) for slot in lost_slots],
        "acquired_slot_causes": acquired_causes,
        "lost_slot_causes": lost_causes,
        "all_changed_slots_accounted": all_changed_slots_accounted,
    }


def _candidate_p45_loss_record(
    trace: compositional_core.CompositionalCurationTrace,
    event_index: int,
    lost_slots: np.ndarray,
) -> dict[str, object]:
    ordinary = np.asarray(
        trace.ordinary_candidate_refresh_mask[event_index], dtype=np.bool_
    )
    post_promotion = np.asarray(
        trace.post_promotion_candidate_refresh_mask[event_index], dtype=np.bool_
    )
    rebound = np.asarray(trace.candidate_rebound_mask[event_index], dtype=np.bool_)
    overdepth = np.asarray(
        trace.candidate_overdepth_regeneration_mask[event_index], dtype=np.bool_
    )
    causes: dict[str, list[str]] = {}
    for raw_slot in lost_slots:
        slot = int(raw_slot)
        slot_causes: list[str] = []
        if ordinary[slot]:
            slot_causes.append("ordinary_candidate_refresh")
        if post_promotion[slot]:
            slot_causes.append("post_promotion_candidate_refresh")
        if rebound[slot]:
            slot_causes.append("active_dependency_rebound")
        if overdepth[slot]:
            slot_causes.append("overdepth_regeneration")
        if not slot_causes:
            slot_causes.append("unmarked_candidate_signature_change")
        causes[str(slot)] = slot_causes
    return {
        "bank_loss": True,
        "lost_slots": [int(slot) for slot in lost_slots],
        "slot_causes": causes,
        "all_lost_slots_accounted": all(
            "unmarked_candidate_signature_change" not in value
            for value in causes.values()
        ),
    }


def _audit_tree_size(tree: object) -> tuple[int, int]:
    elements = 0
    nbytes = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, Array) and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            leaf.dtype,
            jax.dtypes.prng_key,
        ):
            array = np.asarray(jr.key_data(leaf), dtype=np.uint32)
        else:
            array = np.asarray(leaf)
        elements += int(array.size)
        nbytes += int(array.nbytes)
    return elements, nbytes


def _curation_decision_audit(
    protocol: CompositionalControlLifeProtocol,
    events: _ScanEvents,
) -> tuple[dict[str, object], int, int]:
    """Build an exact, in-memory causal audit from ephemeral scan output."""

    trace = events.curation_trace
    pre_active_all = np.asarray(events.pre_active_signature_slots, dtype=np.bool_)
    pre_candidate_all = np.asarray(events.pre_candidate_signature_slots, dtype=np.bool_)
    post_active_all = np.asarray(events.post_active_signature_slots, dtype=np.bool_)
    post_candidate_all = np.asarray(events.post_candidate_signature_slots, dtype=np.bool_)
    due_indices = np.flatnonzero(
        np.arange(1, protocol.total_steps + 1, dtype=np.int64) % CURATION_INTERVAL == 0
    )
    p45_index = SIGNATURE_NAMES.index("shared_p45")
    records: list[dict[str, object]] = []
    active_p45_loss_count = 0
    candidate_p45_loss_count = 0
    all_p45_losses_accounted = True

    for raw_index in due_indices:
        index = int(raw_index)
        pre_active = pre_active_all[index]
        pre_candidate = pre_candidate_all[index]
        post_active = post_active_all[index]
        post_candidate = post_candidate_all[index]
        active_bank_loss = bool(
            np.any(pre_active[:, p45_index])
            and not np.any(post_active[:, p45_index])
        )
        candidate_bank_loss = bool(
            np.any(pre_candidate[:, p45_index])
            and not np.any(post_candidate[:, p45_index])
        )
        active_loss: dict[str, object] | None = None
        candidate_loss: dict[str, object] | None = None
        if active_bank_loss:
            active_p45_loss_count += 1
            active_loss = _active_p45_loss_record(
                trace,
                index,
                np.flatnonzero(
                    pre_active[:, p45_index] & ~post_active[:, p45_index]
                ),
            )
            all_p45_losses_accounted &= cast(
                bool, active_loss["all_lost_slots_accounted"]
            )
        if candidate_bank_loss:
            candidate_p45_loss_count += 1
            candidate_loss = _candidate_p45_loss_record(
                trace,
                index,
                np.flatnonzero(
                    pre_candidate[:, p45_index] & ~post_candidate[:, p45_index]
                ),
            )
            all_p45_losses_accounted &= cast(
                bool, candidate_loss["all_lost_slots_accounted"]
            )

        target_outcomes = {
            name: _target_admission_outcome(
                signature_index=SIGNATURE_NAMES.index(name),
                pre_active_matches=pre_active,
                pre_candidate_matches=pre_candidate,
                post_active_matches=post_active,
                trace=trace,
                event_index=index,
            )
            for name in AUDITED_ADMISSION_SIGNATURE_NAMES
        }
        active_signature_transitions = {
            name: _active_signature_transition_record(
                trace,
                index,
                pre_active[:, SIGNATURE_NAMES.index(name)],
                post_active[:, SIGNATURE_NAMES.index(name)],
            )
            for name in AUDITED_ADMISSION_SIGNATURE_NAMES
        }
        topology = np.asarray(
            trace.decision_candidate_topology_compatible[index], dtype=np.bool_
        )
        depth = np.asarray(
            trace.decision_candidate_depth_compatible[index], dtype=np.bool_
        )
        headroom = np.asarray(
            trace.decision_candidate_headroom_compatible[index], dtype=np.bool_
        )
        compatible = np.asarray(
            trace.decision_candidate_destination_compatible[index], dtype=np.bool_
        )
        record = {
            "pre_step": int(trace.pre_step[index]),
            "post_step": int(trace.post_step[index]),
            "decision_due": True,
            "should_try_replace": bool(trace.should_try_replace[index]),
            "decision_update_available": bool(
                trace.decision_update_available[index]
            ),
            "decision_commit_available": bool(
                trace.decision_commit_available[index]
            ),
            "pre_active_descriptors": _descriptor_record(
                ops=trace.decision_active_ops[index],
                parent_a=trace.decision_active_parent_a[index],
                parent_b=trace.decision_active_parent_b[index],
                theta=trace.decision_active_theta[index],
                depth=trace.decision_active_depth[index],
                generator_policy=trace.decision_active_generator_policy[index],
            ),
            "pre_candidate_descriptors": _descriptor_record(
                ops=trace.decision_candidate_ops[index],
                parent_a=trace.decision_candidate_parent_a[index],
                parent_b=trace.decision_candidate_parent_b[index],
                theta=trace.decision_candidate_theta[index],
                depth=trace.decision_candidate_depth[index],
                generator_policy=trace.decision_candidate_generator_policy[index],
            ),
            "post_active_descriptors": _descriptor_record(
                ops=trace.cascade_final_ops[index],
                parent_a=trace.cascade_final_parent_a[index],
                parent_b=trace.cascade_final_parent_b[index],
                theta=trace.cascade_final_theta[index],
                depth=trace.cascade_final_depth[index],
                generator_policy=trace.cascade_final_generator_policy[index],
            ),
            "post_candidate_descriptors": _descriptor_record(
                ops=trace.candidate_final_ops[index],
                parent_a=trace.candidate_final_parent_a[index],
                parent_b=trace.candidate_final_parent_b[index],
                theta=trace.candidate_final_theta[index],
                depth=trace.candidate_final_depth[index],
                generator_policy=trace.candidate_final_generator_policy[index],
            ),
            "pre_active_signatures": _signature_slot_record(pre_active),
            "pre_candidate_signatures": _signature_slot_record(pre_candidate),
            "post_active_signatures": _signature_slot_record(post_active),
            "post_candidate_signatures": _signature_slot_record(post_candidate),
            "active_ranking": {
                "ages": _int_values(trace.decision_active_ages[index]),
                "fast_utilities_f32_bits": _f32_bits(
                    trace.decision_active_fast_utilities[index]
                ),
                "slow_utilities_f32_bits": _f32_bits(
                    trace.decision_active_slow_utilities[index]
                ),
                "direct_scores_f32_bits": _f32_bits(
                    trace.decision_active_direct_scores[index]
                ),
                "backed_scores_f32_bits": _f32_bits(
                    trace.decision_active_backed_scores[index]
                ),
                "selection_scores_f32_bits": _f32_bits(
                    trace.decision_active_selection_scores[index]
                ),
            },
            "candidate_ranking": {
                "ages": _int_values(trace.decision_candidate_ages[index]),
                "fast_utilities_f32_bits": _f32_bits(
                    trace.decision_candidate_fast_utilities[index]
                ),
                "slow_utilities_f32_bits": _f32_bits(
                    trace.decision_candidate_slow_utilities[index]
                ),
                "direct_scores_f32_bits": _f32_bits(
                    trace.decision_candidate_direct_scores[index]
                ),
                "novelty_scores_f32_bits": _f32_bits(
                    trace.decision_candidate_novelty_scores[index]
                ),
                "augmented_scores_f32_bits": _f32_bits(
                    trace.decision_candidate_augmented_scores[index]
                ),
                "ranking_scores_f32_bits": _f32_bits(
                    trace.decision_candidate_ranking_scores[index]
                ),
                "refresh_utilities_f32_bits": _f32_bits(
                    trace.decision_candidate_refresh_utilities[index]
                ),
                "recomputed_depth": _int_values(
                    trace.decision_candidate_recomputed_depth[index]
                ),
            },
            "destination_masks": {
                "active_eligible_bitset": _mask_bitset(
                    trace.decision_active_eligible[index]
                ),
                "candidate_mature_bitset": _mask_bitset(
                    trace.decision_candidate_mature[index]
                ),
                "topology_compatible_active_bitsets": [
                    _mask_bitset(row) for row in topology
                ],
                "depth_compatible_active_bitsets": [
                    _mask_bitset(row) for row in depth
                ],
                "headroom_compatible_active_bitsets": [
                    _mask_bitset(row) for row in headroom
                ],
                "compatible_active_bitsets": [
                    _mask_bitset(row) for row in compatible
                ],
                "margin_eligible_active_bitsets": [
                    _mask_bitset(row)
                    for row in np.asarray(
                        trace.decision_candidate_margin_eligible[index],
                        dtype=np.bool_,
                    )
                ],
                "candidate_has_destination_bitset": _mask_bitset(
                    trace.decision_candidate_has_destination[index]
                ),
            },
            "selection": {
                "worst_active": int(trace.decision_worst_active[index]),
                "has_active_slot": bool(trace.decision_has_active_slot[index]),
                "selected_candidate": int(
                    trace.decision_selected_candidate[index]
                ),
                "has_candidate": bool(trace.decision_has_candidate[index]),
                "selected_destination": int(
                    trace.decision_selected_destination[index]
                ),
                "selected_refresh_candidate": int(
                    trace.decision_selected_refresh_candidate[index]
                ),
                "has_refresh_candidate": bool(
                    trace.decision_has_refresh_candidate[index]
                ),
                "left_pack_destinations_enabled": bool(
                    trace.decision_left_pack_destinations_enabled[index]
                ),
                "left_pack_destination_available": bool(
                    trace.decision_left_pack_destination_available[index]
                ),
                "effective_promotion_margin_f32_bits": _f32_bits(
                    trace.decision_effective_promotion_margin[index]
                ),
                "selected_candidate_score_f32_bits": _f32_bits(
                    trace.decision_selected_candidate_score[index]
                ),
                "selected_destination_backed_score_f32_bits": _f32_bits(
                    trace.decision_selected_destination_backed_score[index]
                ),
                "margin_rhs_f32_bits": _f32_bits(
                    trace.decision_margin_rhs[index]
                ),
                "margin_passed": bool(trace.decision_margin_passed[index]),
                "selected_topology_ok": bool(
                    trace.decision_selected_topology_ok[index]
                ),
                "selected_depth_ok": bool(
                    trace.decision_selected_depth_ok[index]
                ),
                "selected_headroom_ok": bool(
                    trace.decision_selected_headroom_ok[index]
                ),
                "selected_can_promote": bool(
                    trace.decision_selected_can_promote[index]
                ),
                "should_promote": bool(trace.decision_should_promote[index]),
                "should_refresh": bool(trace.decision_should_refresh[index]),
                "promotion_applied": bool(trace.promotion_applied[index]),
            },
            "structural_changes": {
                "root_change_bitset": _mask_bitset(trace.root_change_mask[index]),
                "cascade_refill_bitset": _mask_bitset(
                    trace.cascade_refill_mask[index]
                ),
                "active_change_bitset": _mask_bitset(
                    trace.active_change_mask[index]
                ),
                "ordinary_candidate_refresh_bitset": _mask_bitset(
                    trace.ordinary_candidate_refresh_mask[index]
                ),
                "post_promotion_candidate_refresh_bitset": _mask_bitset(
                    trace.post_promotion_candidate_refresh_mask[index]
                ),
                "candidate_rebound_bitset": _mask_bitset(
                    trace.candidate_rebound_mask[index]
                ),
                "candidate_overdepth_regeneration_bitset": _mask_bitset(
                    trace.candidate_overdepth_regeneration_mask[index]
                ),
                "post_root_pre_cascade": {
                    "slot": int(trace.post_root_pre_cascade_slot[index]),
                    "op": int(trace.post_root_pre_cascade_op[index]),
                    "parent_a": int(trace.post_root_pre_cascade_parent_a[index]),
                    "parent_b": int(trace.post_root_pre_cascade_parent_b[index]),
                    "theta_f32_bits": _f32_bits(
                        trace.post_root_pre_cascade_theta[index]
                    ),
                    "depth": int(trace.post_root_pre_cascade_depth[index]),
                    "generator_policy": int(
                        trace.post_root_pre_cascade_generator_policy[index]
                    ),
                },
            },
            "target_admission_outcomes": target_outcomes,
            "active_signature_transitions": active_signature_transitions,
            "shared_p45_active_bank_loss": active_loss,
            "shared_p45_candidate_bank_loss": candidate_loss,
        }
        records.append(record)

    outcome_counts: dict[str, dict[str, int]] = {}
    for name in AUDITED_ADMISSION_SIGNATURE_NAMES:
        counts: dict[str, int] = {}
        for record in records:
            outcomes = cast(
                Mapping[str, Mapping[str, object]],
                record["target_admission_outcomes"],
            )
            outcome = cast(str, outcomes[name]["outcome"])
            counts[outcome] = counts.get(outcome, 0) + 1
        outcome_counts[name] = counts

    transition_causes: dict[str, object] = {}
    cause_labels = (
        "promotion_root_replacement",
        "cascade_dependency_refill",
        "unmarked_signature_dependency_change",
    )
    for name in AUDITED_ADMISSION_SIGNATURE_NAMES:
        acquisitions: list[dict[str, object]] = []
        losses: list[dict[str, object]] = []
        acquisition_counts = {label: 0 for label in cause_labels}
        loss_counts = {label: 0 for label in cause_labels}
        all_changed_slots_accounted = True
        for record in records:
            transition = cast(
                Mapping[str, object],
                cast(Mapping[str, object], record["active_signature_transitions"])[
                    name
                ],
            )
            all_changed_slots_accounted &= cast(
                bool, transition["all_changed_slots_accounted"]
            )
            if transition["bank_acquisition"] is True:
                event = {
                    "post_step": record["post_step"],
                    "acquired_slots": transition["acquired_slots"],
                    "slot_causes": transition["acquired_slot_causes"],
                }
                acquisitions.append(event)
                for labels in cast(
                    Mapping[str, list[str]], transition["acquired_slot_causes"]
                ).values():
                    for label in labels:
                        acquisition_counts[label] += 1
            if transition["bank_loss"] is True:
                event = {
                    "post_step": record["post_step"],
                    "lost_slots": transition["lost_slots"],
                    "slot_causes": transition["lost_slot_causes"],
                }
                losses.append(event)
                for labels in cast(
                    Mapping[str, list[str]], transition["lost_slot_causes"]
                ).values():
                    for label in labels:
                        loss_counts[label] += 1
        transition_causes[name] = {
            "acquisition_episode_count": len(acquisitions),
            "loss_episode_count": len(losses),
            "acquisition_events": acquisitions,
            "loss_events": losses,
            "acquisition_slot_cause_counts": acquisition_counts,
            "loss_slot_cause_counts": loss_counts,
            "all_changed_slots_accounted": all_changed_slots_accounted,
        }

    detailed_telemetry = (
        trace,
        events.pre_active_signature_slots,
        events.pre_candidate_signature_slots,
        events.post_active_signature_slots,
        events.post_candidate_signature_slots,
    )
    telemetry_elements, telemetry_nbytes = _audit_tree_size(detailed_telemetry)
    records_bytes = len(_canonical_json_bytes(records))
    audit = {
        "scope": (
            "ephemeral exact decision algebra; no persistent state, RNG, policy, "
            "seed, artifact, threshold, or evidence mutation"
        ),
        "float_encoding": "IEEE-754 binary32 payload as unsigned integer bits",
        "due_curation_event_count": len(records),
        "due_curation_records": records,
        "records_sha256": _json_sha256(records),
        "records_canonical_json_bytes": records_bytes,
        "target_outcome_counts": outcome_counts,
        "all_target_due_events_accounted": all(
            sum(outcome_counts[name].values()) == len(records)
            for name in AUDITED_ADMISSION_SIGNATURE_NAMES
        ),
        "active_signature_transition_causes": transition_causes,
        "shared_p45_active_bank_loss_count": active_p45_loss_count,
        "shared_p45_candidate_bank_loss_count": candidate_p45_loss_count,
        "all_shared_p45_bank_losses_accounted": all_p45_losses_accounted,
        "ephemeral_array_elements": telemetry_elements,
        "ephemeral_array_bytes": telemetry_nbytes,
    }
    return audit, telemetry_elements, telemetry_nbytes


def execute_compositional_control_life_arm(
    protocol: CompositionalControlLifeProtocol,
    learner: CompositionalFeatureLearner,
    learner_key: Array,
    observations: Array,
    phase_indices: Array,
    exploration_mask: Array,
    random_actions: Array,
    *,
    composed_readout_enabled: bool,
) -> CompositionalControlLifeArmExecution:
    """Execute one strict arm without selecting a root or granting result authority."""

    if type(protocol) is not CompositionalControlLifeProtocol:
        raise TypeError("protocol must be an exact CompositionalControlLifeProtocol")
    if type(learner) is not CompositionalFeatureLearner:
        raise TypeError("learner must be an exact CompositionalFeatureLearner")
    if type(composed_readout_enabled) is not bool:
        raise TypeError("composed_readout_enabled must be an exact bool")
    if learner_key.shape != () or str(jr.key_impl(learner_key)) != "threefry2x32":
        raise ValueError("learner_key must be a scalar typed Threefry key")

    expected_arrays = (
        ("observations", observations, (protocol.total_steps, RAW_DIM), np.float32),
        ("phase_indices", phase_indices, (protocol.total_steps,), np.int32),
        ("exploration_mask", exploration_mask, (protocol.total_steps,), np.bool_),
        ("random_actions", random_actions, (protocol.total_steps,), np.int32),
    )
    for name, value, shape, dtype in expected_arrays:
        if not isinstance(value, Array):
            raise TypeError(f"{name} must be a JAX array")
        if value.shape != shape or np.dtype(value.dtype) != np.dtype(dtype):
            raise ValueError(f"{name} has an invalid shape or dtype")

    state = cast(
        CompositionalFeatureState,
        learner.init(RAW_DIM, learner_key).replace(  # type: ignore[attr-defined]
            birth_timestamp=0.0,
            uptime_s=0.0,
        ),
    )
    initial_ranking_diagnostics = learner.ranking_diagnostics(state, RAW_DIM)
    if not bool(initial_ranking_diagnostics.contract_valid):
        raise RuntimeError("initial ranking contract is invalid")
    (
        initial_active_counts,
        initial_candidate_counts,
        initial_active_pair_counts,
        initial_candidate_pair_counts,
    ) = _product_signature_counts(state)
    if bool(jnp.any(initial_active_counts)) or bool(jnp.any(initial_candidate_counts)):
        raise RuntimeError("a target or useful intermediate is prewired at genesis")

    expected_nbytes = compositional_control_state_nbytes_formula(
        active_slots=ACTIVE_SLOTS,
        candidate_slots=CANDIDATE_SLOTS,
        action_heads=ACTION_HEADS,
    )
    initial_nbytes = persistent_compositional_state_nbytes(state)
    if initial_nbytes != expected_nbytes:
        raise RuntimeError("initial compositional state violates the byte formula")
    initial_state_sha256 = generated_birth_identity_scrub_epoch_core_state_sha256(state)
    final_state, device_events = _run_compiled_scan(
        learner,
        state,
        composed_readout_enabled,
        observations,
        phase_indices,
        exploration_mask,
        random_actions,
    )
    final_state_sha256 = generated_birth_identity_scrub_epoch_core_state_sha256(
        final_state
    )
    events = cast(_ScanEvents, jax.device_get(device_events))
    final_nbytes = persistent_compositional_state_nbytes(final_state)
    if final_nbytes != expected_nbytes:
        raise RuntimeError("final compositional state violates the byte formula")
    if tuple(int(value) for value in np.asarray(final_state.step_words)) != (
        0,
        protocol.total_steps,
    ):
        raise RuntimeError("final exact lifetime clock does not match requested steps")

    semantic_integrity = (
        ("initial state", _state_is_finite(state)),
        ("final state", _state_is_finite(final_state)),
        (
            "lifetime counters",
            bool(np.all(np.asarray(events.lifetime_counter_valid))),
        ),
        (
            "lifetime capacity",
            bool(np.all(np.asarray(events.lifetime_capacity_available))),
        ),
        (
            "ranking contracts",
            bool(np.all(np.asarray(events.ranking_contract_valid))),
        ),
        (
            "core prediction parity",
            bool(np.all(np.asarray(events.core_prediction_matches_full_q))),
        ),
    )
    failed_integrity = tuple(name for name, valid in semantic_integrity if not valid)
    if failed_integrity:
        raise RuntimeError(
            "arm semantic integrity failed: " + ", ".join(failed_integrity)
        )

    return CompositionalControlLifeArmExecution(
        initial_state=state,
        final_state=final_state,
        events=events,
        initial_ranking_diagnostics=initial_ranking_diagnostics,
        initial_active_signature_counts=initial_active_counts,
        initial_candidate_signature_counts=initial_candidate_counts,
        initial_active_raw_pair_counts=initial_active_pair_counts,
        initial_candidate_raw_pair_counts=initial_candidate_pair_counts,
        initial_state_sha256=initial_state_sha256,
        final_state_sha256=final_state_sha256,
        trace_sha256=_array_tree_sha256(events),
        expected_persistent_state_nbytes=expected_nbytes,
        initial_persistent_state_nbytes=initial_nbytes,
        final_persistent_state_nbytes=final_nbytes,
    )


def _validate_execution_event_tree_geometry(
    events: _ScanEvents,
    *,
    total_steps: int,
) -> None:
    leaves = jax.tree_util.tree_leaves(events)
    if not leaves:
        raise ValueError("arm execution event tree cannot be empty")
    for index, leaf in enumerate(leaves):
        shape = getattr(leaf, "shape", None)
        if type(shape) is not tuple or not shape or shape[0] != total_steps:
            raise ValueError(
                f"arm execution event leaf {index} does not span the exact life"
            )
        if isinstance(leaf, Array) and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            leaf.dtype,
            jax.dtypes.prng_key,
        ):
            array = np.asarray(jr.key_data(leaf), dtype=np.uint32)
        else:
            array = np.asarray(leaf)
        if array.dtype.hasobject:
            raise TypeError("arm execution event tree cannot contain object arrays")


def _validate_execution_event_schema(
    events: _ScanEvents,
    *,
    total_steps: int,
) -> None:
    scalar_specs = (
        ("executed_reward", np.float32),
        ("greedy_reward", np.float32),
        ("executed_regret", np.float32),
        ("greedy_regret", np.float32),
        ("action", np.int32),
        ("greedy_action", np.int32),
        ("explored", np.bool_),
        ("target_value", np.float32),
        ("core_prediction_matches_full_q", np.bool_),
        ("lifetime_counter_valid", np.bool_),
        ("lifetime_capacity_available", np.bool_),
        ("ranking_contract_valid", np.bool_),
        ("eligible_recursive_parent_exists", np.bool_),
    )
    shaped_specs = (
        ("full_q", (ACTION_HEADS,), np.float32),
        ("raw_q", (ACTION_HEADS,), np.float32),
        ("behavior_q", (ACTION_HEADS,), np.float32),
        ("curation_counts", (len(CURATION_COUNT_NAMES),), np.int32),
        ("raw_active_utilities", (ACTIVE_SLOTS,), np.float32),
        ("slow_active_utilities", (ACTIVE_SLOTS,), np.float32),
        ("direct_active_scores", (ACTIVE_SLOTS,), np.float32),
        ("backed_active_scores", (ACTIVE_SLOTS,), np.float32),
        ("raw_candidate_utilities", (CANDIDATE_SLOTS,), np.float32),
        ("slow_candidate_utilities", (CANDIDATE_SLOTS,), np.float32),
        ("direct_candidate_scores", (CANDIDATE_SLOTS,), np.float32),
        ("candidate_novelty_scores", (CANDIDATE_SLOTS,), np.float32),
        ("augmented_candidate_scores", (CANDIDATE_SLOTS,), np.float32),
        ("candidate_mature", (CANDIDATE_SLOTS,), np.bool_),
        (
            "pre_active_signature_slots",
            (ACTIVE_SLOTS, len(SIGNATURE_NAMES)),
            np.bool_,
        ),
        (
            "pre_candidate_signature_slots",
            (CANDIDATE_SLOTS, len(SIGNATURE_NAMES)),
            np.bool_,
        ),
        (
            "post_active_signature_slots",
            (ACTIVE_SLOTS, len(SIGNATURE_NAMES)),
            np.bool_,
        ),
        (
            "post_candidate_signature_slots",
            (CANDIDATE_SLOTS, len(SIGNATURE_NAMES)),
            np.bool_,
        ),
        ("active_signature_counts", (len(SIGNATURE_NAMES),), np.int32),
        ("candidate_signature_counts", (len(SIGNATURE_NAMES),), np.int32),
        ("active_raw_pair_counts", (len(RAW_PAIR_NAMES),), np.int32),
        ("candidate_raw_pair_counts", (len(RAW_PAIR_NAMES),), np.int32),
    )
    for name, dtype in scalar_specs:
        array = np.asarray(getattr(events, name))
        if array.shape != (total_steps,) or array.dtype != np.dtype(dtype):
            raise TypeError(f"arm execution event {name} has an invalid shape or dtype")
        if np.issubdtype(array.dtype, np.inexact) and not np.all(np.isfinite(array)):
            raise ValueError(f"arm execution event {name} must be finite")
    for name, trailing_shape, dtype in shaped_specs:
        array = np.asarray(getattr(events, name))
        if array.shape != (total_steps, *trailing_shape) or array.dtype != np.dtype(
            dtype
        ):
            raise TypeError(f"arm execution event {name} has an invalid shape or dtype")
        if np.issubdtype(array.dtype, np.inexact) and not np.all(np.isfinite(array)):
            raise ValueError(f"arm execution event {name} must be finite")
    for name in ("action", "greedy_action"):
        actions = np.asarray(getattr(events, name))
        if np.any(actions < 0) or np.any(actions >= ACTION_HEADS):
            raise ValueError(f"arm execution event {name} is outside the action space")
    if np.any(np.asarray(events.curation_counts) < 0):
        raise ValueError("arm execution curation counts cannot be negative")
    _validate_curation_trace_schema(events.curation_trace, total_steps=total_steps)


def _validate_curation_trace_schema(
    trace: compositional_core.CompositionalCurationTrace,
    *,
    total_steps: int,
) -> None:
    if type(trace) is not compositional_core.CompositionalCurationTrace:
        raise TypeError("arm execution curation trace has an invalid type")
    fields = dataclasses.fields(cast(Any, trace))
    annotations = compositional_core.CompositionalCurationTrace.__annotations__
    if tuple(field.name for field in fields) != tuple(annotations):
        raise RuntimeError("curation trace fields and annotations disagree")
    dimensions = {
        "": (),
        "2": (2,),
        "n_features": (ACTIVE_SLOTS,),
        "n_features 2": (ACTIVE_SLOTS, 2),
        "n_candidates": (CANDIDATE_SLOTS,),
        "n_candidates 2": (CANDIDATE_SLOTS, 2),
        "n_candidates n_features": (CANDIDATE_SLOTS, ACTIVE_SLOTS),
    }
    dtypes = {
        "Bool": np.dtype(np.bool_),
        "Float": np.dtype(np.float32),
        "Int": np.dtype(np.int32),
        "UInt": np.dtype(np.uint32),
    }
    key_fields = {
        "decision_key",
        "curation_key",
        "proposal_key",
        "cascade_key",
        "candidate_overdepth_regeneration_key",
    }
    for field in fields:
        value = getattr(trace, field.name)
        annotation = annotations[field.name]
        dimension_string = getattr(annotation, "dim_str", None)
        if field.name in key_fields:
            if (
                dimension_string is not None
                or not isinstance(value, Array)
                or value.shape != (total_steps,)
                or str(jr.key_impl(value)) != "threefry2x32"
            ):
                raise TypeError(f"curation trace key {field.name} is invalid")
            key_data = np.asarray(jr.key_data(value))
            if key_data.shape != (total_steps, 2) or key_data.dtype != np.dtype(
                np.uint32
            ):
                raise TypeError(f"curation trace key data {field.name} is invalid")
            continue
        if dimension_string not in dimensions:
            raise RuntimeError(
                f"curation trace field {field.name} has an unknown dimension schema"
            )
        dtype_name = getattr(getattr(annotation, "dtype", None), "__name__", None)
        if type(dtype_name) is not str:
            raise RuntimeError(
                f"curation trace field {field.name} has no exact dtype name"
            )
        expected_dtype = dtypes.get(dtype_name)
        if expected_dtype is None:
            raise RuntimeError(
                f"curation trace field {field.name} has an unknown dtype schema"
            )
        array = np.asarray(value)
        expected_shape = (total_steps, *dimensions[dimension_string])
        if array.shape != expected_shape or array.dtype != expected_dtype:
            raise TypeError(
                f"curation trace field {field.name} has an invalid shape or dtype"
            )
        if expected_dtype == np.dtype(np.float32):
            if field.name == "decision_active_selection_scores":
                eligible = np.asarray(trace.decision_active_eligible)
                if (
                    np.any(np.isnan(array))
                    or np.any(np.isneginf(array))
                    or not np.array_equal(np.isposinf(array), ~eligible)
                    or not np.all(np.isfinite(array[eligible]))
                ):
                    raise ValueError(
                        "curation trace active selection sentinels are invalid"
                    )
            elif not np.all(np.isfinite(array)):
                raise ValueError(f"curation trace field {field.name} must be finite")


def _validate_execution_curation_semantics(
    protocol: CompositionalControlLifeProtocol,
    events: _ScanEvents,
    *,
    pinned_curation_due_mask: object,
) -> tuple[int, ...]:
    # Local import avoids a module-load cycle: the neutral engine imports this
    # control module, while validation occurs only after both modules exist.
    from alberta_framework.evaluation import (  # noqa: PLC0415
        _compositional_future_utility_calibration_engine as future_engine,
    )

    due = np.asarray(pinned_curation_due_mask)
    expected_due = (
        (np.arange(protocol.total_steps, dtype=np.int64) + 1)
        % CURATION_INTERVAL
        == 0
    )
    if (
        due.shape != (protocol.total_steps,)
        or due.dtype != np.dtype(np.bool_)
        or not np.array_equal(due, expected_due)
    ):
        raise ValueError("pinned curation-due mask does not match the exact schedule")
    geometry = future_engine.FutureUtilityEndpointGeometry(
        phase_order=PHASE_ORDER,
        phase_lengths=protocol.phase_lengths,
        target_names=("A", "B", "C"),
        curation_interval=CURATION_INTERVAL,
    )
    future_engine.validate_future_utility_trace_shapes(geometry, events)
    future_engine.validate_future_utility_eventwise_curation_semantics(
        geometry,
        events,
    )
    cadence_audit = future_engine.future_utility_cadence_audit_from_events(
        geometry,
        events,
        pinned_due_mask=due,
    )
    totals_array = np.sum(
        np.asarray(events.curation_counts),
        axis=0,
        dtype=np.int64,
    )
    totals = tuple(int(value) for value in totals_array)
    total_record = {
        name: value
        for name, value in zip(CURATION_COUNT_NAMES, totals, strict=True)
    }
    future_engine.validate_future_utility_curation_count_closure(
        cadence_audit,
        total_record,
    )
    return totals


def _validate_initial_ranking_diagnostics(
    diagnostics: CompositionalRankingDiagnostics,
) -> None:
    if type(diagnostics) is not CompositionalRankingDiagnostics:
        raise TypeError("initial ranking diagnostics have an invalid type")
    specs = (
        ("contract_valid", diagnostics.contract_valid, (), np.bool_),
        (
            "direct_active_scores",
            diagnostics.direct_active_scores,
            (ACTIVE_SLOTS,),
            np.float32,
        ),
        (
            "backed_active_scores",
            diagnostics.backed_active_scores,
            (ACTIVE_SLOTS,),
            np.float32,
        ),
        (
            "direct_candidate_scores",
            diagnostics.direct_candidate_scores,
            (CANDIDATE_SLOTS,),
            np.float32,
        ),
        (
            "candidate_novelty_scores",
            diagnostics.candidate_novelty_scores,
            (CANDIDATE_SLOTS,),
            np.float32,
        ),
        (
            "augmented_candidate_scores",
            diagnostics.augmented_candidate_scores,
            (CANDIDATE_SLOTS,),
            np.float32,
        ),
        (
            "candidate_mature",
            diagnostics.candidate_mature,
            (CANDIDATE_SLOTS,),
            np.bool_,
        ),
    )
    for name, value, shape, dtype in specs:
        array = np.asarray(value)
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise TypeError(f"initial ranking {name} has an invalid shape or dtype")
        if np.issubdtype(array.dtype, np.inexact) and not np.all(np.isfinite(array)):
            raise ValueError(f"initial ranking {name} must be finite")
    if bool(np.asarray(diagnostics.contract_valid)) is not True:
        raise ValueError("initial ranking contract is invalid")


def _validated_initial_count_array(
    value: object,
    *,
    expected: Array,
    name: str,
) -> np.ndarray[Any, Any]:
    if not isinstance(value, Array):
        raise TypeError(f"{name} must remain a JAX array")
    array = np.asarray(value)
    expected_array = np.asarray(expected)
    if (
        array.shape != expected_array.shape
        or array.dtype != np.dtype(np.int32)
        or not np.array_equal(array, expected_array)
    ):
        raise ValueError(f"{name} does not match the recomputed genesis structure")
    return array


def validate_compositional_control_life_arm_execution(
    protocol: CompositionalControlLifeProtocol,
    execution: CompositionalControlLifeArmExecution,
    *,
    pinned_curation_due_mask: object,
) -> CompositionalControlLifeArmExecutionReceipt:
    """Recompute hashes, schemas, cadence, bytes, clocks, and core integrity.

    The returned receipt is authority-free and contains no learner selection,
    root issuance, execution, artifact, threshold, or promotion capability. It
    does not replay the update or authenticate the learner/source that produced
    the supplied arrays; callers must bind those separately.
    """

    if type(protocol) is not CompositionalControlLifeProtocol:
        raise TypeError("protocol must be an exact CompositionalControlLifeProtocol")
    if type(execution) is not CompositionalControlLifeArmExecution:
        raise TypeError("execution must be an exact CompositionalControlLifeArmExecution")
    authority = (
        execution.scientific_promotion_allowed,
        execution.evidence_authorized,
        execution.output_writes_allowed,
    )
    if any(type(value) is not bool for value in authority) or any(authority):
        raise ValueError("arm execution has invalid authority flags")
    if type(execution.initial_state) is not CompositionalFeatureState or type(
        execution.final_state
    ) is not CompositionalFeatureState:
        raise TypeError("arm execution states have an invalid type")
    if type(execution.events) is not _ScanEvents:
        raise TypeError("arm execution events have an invalid type")
    _validate_execution_event_tree_geometry(
        execution.events,
        total_steps=protocol.total_steps,
    )
    _validate_execution_event_schema(
        execution.events,
        total_steps=protocol.total_steps,
    )
    _validate_execution_curation_semantics(
        protocol,
        execution.events,
        pinned_curation_due_mask=pinned_curation_due_mask,
    )
    _validate_initial_ranking_diagnostics(execution.initial_ranking_diagnostics)

    (
        expected_active,
        expected_candidate,
        expected_active_pairs,
        expected_candidate_pairs,
    ) = _product_signature_counts(execution.initial_state)
    initial_active = _validated_initial_count_array(
        execution.initial_active_signature_counts,
        expected=expected_active,
        name="initial active signature counts",
    )
    initial_candidate = _validated_initial_count_array(
        execution.initial_candidate_signature_counts,
        expected=expected_candidate,
        name="initial candidate signature counts",
    )
    _validated_initial_count_array(
        execution.initial_active_raw_pair_counts,
        expected=expected_active_pairs,
        name="initial active raw-pair counts",
    )
    _validated_initial_count_array(
        execution.initial_candidate_raw_pair_counts,
        expected=expected_candidate_pairs,
        name="initial candidate raw-pair counts",
    )
    initial_target_counts_zero = bool(
        not np.any(initial_active) and not np.any(initial_candidate)
    )
    if not initial_target_counts_zero:
        raise ValueError("arm execution has a prewired target at genesis")

    initial_state_sha256 = generated_birth_identity_scrub_epoch_core_state_sha256(
        execution.initial_state
    )
    final_state_sha256 = generated_birth_identity_scrub_epoch_core_state_sha256(
        execution.final_state
    )
    trace_sha256 = _array_tree_sha256(execution.events)
    for name, supplied, recomputed in (
        ("initial state", execution.initial_state_sha256, initial_state_sha256),
        ("final state", execution.final_state_sha256, final_state_sha256),
        ("event trace", execution.trace_sha256, trace_sha256),
    ):
        if not _is_sha256(supplied) or supplied != recomputed:
            raise ValueError(f"arm execution {name} SHA-256 does not close")

    expected_nbytes = compositional_control_state_nbytes_formula(
        active_slots=ACTIVE_SLOTS,
        candidate_slots=CANDIDATE_SLOTS,
        action_heads=ACTION_HEADS,
    )
    initial_nbytes = persistent_compositional_state_nbytes(execution.initial_state)
    final_nbytes = persistent_compositional_state_nbytes(execution.final_state)
    byte_values = (
        execution.expected_persistent_state_nbytes,
        execution.initial_persistent_state_nbytes,
        execution.final_persistent_state_nbytes,
    )
    if any(type(value) is not int for value in byte_values) or byte_values != (
        expected_nbytes,
        initial_nbytes,
        final_nbytes,
    ):
        raise ValueError("arm execution persistent-state byte accounting does not close")

    initial_step_count = np.asarray(execution.initial_state.step_count)
    initial_step_words = np.asarray(execution.initial_state.step_words)
    initial_replacement_phase = np.asarray(execution.initial_state.replacement_phase)
    if (
        initial_step_count.shape != ()
        or initial_step_count.dtype != np.dtype(np.int32)
        or int(initial_step_count) != 0
        or initial_step_words.shape != (2,)
        or initial_step_words.dtype != np.dtype(np.uint32)
        or tuple(int(value) for value in initial_step_words) != (0, 0)
        or initial_replacement_phase.shape != ()
        or initial_replacement_phase.dtype != np.dtype(np.int32)
        or int(initial_replacement_phase) != 0
    ):
        raise ValueError("arm execution genesis lifetime clocks are invalid")
    final_step_count_array = np.asarray(execution.final_state.step_count)
    final_step_words_array = np.asarray(execution.final_state.step_words)
    final_replacement_phase_array = np.asarray(execution.final_state.replacement_phase)
    if (
        final_step_count_array.shape != ()
        or final_step_count_array.dtype != np.dtype(np.int32)
        or final_step_words_array.shape != (2,)
        or final_step_words_array.dtype != np.dtype(np.uint32)
        or final_replacement_phase_array.shape != ()
        or final_replacement_phase_array.dtype != np.dtype(np.int32)
    ):
        raise TypeError("arm execution final lifetime clock shape or dtype is invalid")
    final_step_count = int(final_step_count_array)
    final_step_words = tuple(int(value) for value in final_step_words_array)
    final_replacement_phase = int(final_replacement_phase_array)
    if (
        final_step_count != protocol.total_steps
        or final_step_words != (0, protocol.total_steps)
        or final_replacement_phase != protocol.total_steps % CURATION_INTERVAL
    ):
        raise ValueError("arm execution final lifetime clocks do not close")

    closures = {
        "initial_state_finite": _state_is_finite(execution.initial_state),
        "final_state_finite": _state_is_finite(execution.final_state),
        "all_lifetime_counters_valid": bool(
            np.all(np.asarray(execution.events.lifetime_counter_valid))
        ),
        "all_lifetime_capacity_available": bool(
            np.all(np.asarray(execution.events.lifetime_capacity_available))
        ),
        "all_ranking_contracts_valid": bool(
            np.all(np.asarray(execution.events.ranking_contract_valid))
        ),
        "all_core_predictions_match_full_q": bool(
            np.all(np.asarray(execution.events.core_prediction_matches_full_q))
        ),
    }
    failed = tuple(name for name, valid in closures.items() if not valid)
    if failed:
        raise ValueError("arm execution semantic closure failed: " + ", ".join(failed))

    return CompositionalControlLifeArmExecutionReceipt(
        total_steps=protocol.total_steps,
        initial_state_sha256=initial_state_sha256,
        final_state_sha256=final_state_sha256,
        trace_sha256=trace_sha256,
        expected_persistent_state_nbytes=expected_nbytes,
        initial_persistent_state_nbytes=initial_nbytes,
        final_persistent_state_nbytes=final_nbytes,
        final_step_count=final_step_count,
        final_step_words_uint32=final_step_words,
        final_replacement_phase=final_replacement_phase,
        initial_state_finite=closures["initial_state_finite"],
        final_state_finite=closures["final_state_finite"],
        all_lifetime_counters_valid=closures["all_lifetime_counters_valid"],
        all_lifetime_capacity_available=closures[
            "all_lifetime_capacity_available"
        ],
        all_ranking_contracts_valid=closures["all_ranking_contracts_valid"],
        all_core_predictions_match_full_q=closures[
            "all_core_predictions_match_full_q"
        ],
        initial_target_signature_counts_zero=initial_target_counts_zero,
        _validation_token=_VALIDATED_RECEIPT_TOKEN,
    )


def analyze_compositional_control_life_arm_execution(
    protocol: CompositionalControlLifeProtocol,
    execution: CompositionalControlLifeArmExecution,
    *,
    curation_geometry_arm_name: str,
    pinned_curation_due_mask: object,
) -> CompositionalControlLifeArmAnalysisReceipt:
    """Validate curation geometry and build exact structural trajectories.

    ``curation_geometry_arm_name`` selects only the topology/headroom/placement
    rules needed to interpret the decision audit. It is deliberately not a
    learner, source, key, or stream identity claim.
    """

    if (
        type(curation_geometry_arm_name) is not str
        or curation_geometry_arm_name not in _ARMS_BY_NAME
    ):
        raise ValueError(
            "curation_geometry_arm_name must identify one declared control-life arm"
        )
    execution_receipt = validate_compositional_control_life_arm_execution(
        protocol,
        execution,
        pinned_curation_due_mask=pinned_curation_due_mask,
    )
    events = execution.events
    count_specs = (
        (
            "curation counts",
            events.curation_counts,
            (protocol.total_steps, len(CURATION_COUNT_NAMES)),
        ),
        (
            "active signature counts",
            events.active_signature_counts,
            (protocol.total_steps, len(SIGNATURE_NAMES)),
        ),
        (
            "candidate signature counts",
            events.candidate_signature_counts,
            (protocol.total_steps, len(SIGNATURE_NAMES)),
        ),
        (
            "active raw-pair counts",
            events.active_raw_pair_counts,
            (protocol.total_steps, len(RAW_PAIR_NAMES)),
        ),
        (
            "candidate raw-pair counts",
            events.candidate_raw_pair_counts,
            (protocol.total_steps, len(RAW_PAIR_NAMES)),
        ),
    )
    for name, value, count_shape in count_specs:
        array = np.asarray(value)
        if array.shape != count_shape or array.dtype != np.dtype(np.int32):
            raise TypeError(f"arm analysis {name} must use exact int32 telemetry")
        if np.any(array < 0):
            raise ValueError(f"arm analysis {name} cannot contain negative counts")
    slot_specs = (
        (
            "pre active signature slots",
            events.pre_active_signature_slots,
            (protocol.total_steps, ACTIVE_SLOTS, len(SIGNATURE_NAMES)),
        ),
        (
            "post active signature slots",
            events.post_active_signature_slots,
            (protocol.total_steps, ACTIVE_SLOTS, len(SIGNATURE_NAMES)),
        ),
        (
            "pre candidate signature slots",
            events.pre_candidate_signature_slots,
            (protocol.total_steps, CANDIDATE_SLOTS, len(SIGNATURE_NAMES)),
        ),
        (
            "post candidate signature slots",
            events.post_candidate_signature_slots,
            (protocol.total_steps, CANDIDATE_SLOTS, len(SIGNATURE_NAMES)),
        ),
    )
    for name, value, slot_shape in slot_specs:
        array = np.asarray(value)
        if array.shape != slot_shape or array.dtype != np.dtype(np.bool_):
            raise TypeError(f"arm analysis {name} must use exact boolean telemetry")
        if np.any(np.sum(array, axis=2, dtype=np.int32) > 1):
            raise ValueError(f"arm analysis {name} matches one slot to multiple signatures")
    pre_active_slots = np.asarray(events.pre_active_signature_slots)
    post_active_slots = np.asarray(events.post_active_signature_slots)
    pre_candidate_slots = np.asarray(events.pre_candidate_signature_slots)
    post_candidate_slots = np.asarray(events.post_candidate_signature_slots)
    (
        initial_active_slots,
        initial_candidate_slots,
        _initial_active_pair_slots,
        _initial_candidate_pair_slots,
    ) = _product_signature_slot_matches(execution.initial_state)
    if not np.array_equal(pre_active_slots[0], np.asarray(initial_active_slots)) or not (
        np.array_equal(pre_candidate_slots[0], np.asarray(initial_candidate_slots))
    ):
        raise ValueError("arm analysis first pre-state does not match genesis")
    if not np.array_equal(pre_active_slots[1:], post_active_slots[:-1]) or not (
        np.array_equal(pre_candidate_slots[1:], post_candidate_slots[:-1])
    ):
        raise ValueError("arm analysis pre/post structural trajectory is discontinuous")
    due = np.asarray(pinned_curation_due_mask)
    active_slot_changes = np.any(pre_active_slots != post_active_slots, axis=2)
    candidate_slot_changes = np.any(
        pre_candidate_slots != post_candidate_slots,
        axis=2,
    )
    if np.any(active_slot_changes[~due]) or np.any(candidate_slot_changes[~due]):
        raise ValueError("arm analysis contains an off-cadence structural transition")
    trace = events.curation_trace
    active_change_mask = np.asarray(trace.active_change_mask, dtype=np.bool_)
    candidate_change_mask = (
        np.asarray(trace.candidate_refresh_mask, dtype=np.bool_)
        | np.asarray(trace.candidate_rebound_mask, dtype=np.bool_)
        | np.asarray(
            trace.candidate_overdepth_regeneration_mask,
            dtype=np.bool_,
        )
    )
    if np.any(active_slot_changes & ~active_change_mask) or np.any(
        candidate_slot_changes & ~candidate_change_mask
    ):
        raise ValueError("arm analysis has an unmarked structural slot transition")
    if not np.array_equal(
        np.sum(post_active_slots, axis=1, dtype=np.int32),
        np.asarray(events.active_signature_counts),
    ) or not np.array_equal(
        np.sum(
            post_candidate_slots,
            axis=1,
            dtype=np.int32,
        ),
        np.asarray(events.candidate_signature_counts),
    ):
        raise ValueError("arm analysis signature counts do not match slot telemetry")
    for name, counts, initial_counts, slot_limit in (
        (
            "active raw-pair counts",
            np.asarray(events.active_raw_pair_counts),
            np.asarray(execution.initial_active_raw_pair_counts),
            ACTIVE_SLOTS,
        ),
        (
            "candidate raw-pair counts",
            np.asarray(events.candidate_raw_pair_counts),
            np.asarray(execution.initial_candidate_raw_pair_counts),
            CANDIDATE_SLOTS,
        ),
    ):
        if np.any(counts > slot_limit):
            raise ValueError(f"arm analysis {name} exceed bank capacity")
        previous = np.concatenate((initial_counts[None, :], counts[:-1]), axis=0)
        if np.any(np.any(counts != previous, axis=1) & ~due):
            raise ValueError(f"arm analysis {name} change off cadence")

    curation_totals = _validate_execution_curation_semantics(
        protocol,
        events,
        pinned_curation_due_mask=pinned_curation_due_mask,
    )
    active_trajectories: dict[str, object] = {}
    candidate_trajectories: dict[str, object] = {}
    initial_active = np.asarray(execution.initial_active_signature_counts)
    initial_candidate = np.asarray(execution.initial_candidate_signature_counts)
    for index, name in enumerate(SIGNATURE_NAMES):
        active_trajectories[name] = _structural_trajectory(
            int(initial_active[index]),
            events.active_signature_counts[:, index],
        )
        candidate_trajectories[name] = _structural_trajectory(
            int(initial_candidate[index]),
            events.candidate_signature_counts[:, index],
        )

    curation_audit, audit_elements, audit_nbytes = _curation_decision_audit(
        protocol,
        events,
    )
    _validate_curation_decision_audit(
        curation_audit,
        arm_name=curation_geometry_arm_name,
        protocol=protocol,
        active_trajectories=active_trajectories,
        candidate_trajectories=candidate_trajectories,
    )
    if curation_audit["shared_p45_active_bank_loss_count"] != cast(
        Mapping[str, object],
        active_trajectories["shared_p45"],
    )["loss_episode_count"]:
        raise ValueError("active shared-p45 loss audit does not close")
    if curation_audit["shared_p45_candidate_bank_loss_count"] != cast(
        Mapping[str, object],
        candidate_trajectories["shared_p45"],
    )["loss_episode_count"]:
        raise ValueError("candidate shared-p45 loss audit does not close")
    transition_causes = cast(
        Mapping[str, Mapping[str, object]],
        curation_audit["active_signature_transition_causes"],
    )
    for name in AUDITED_ADMISSION_SIGNATURE_NAMES:
        trajectory = cast(Mapping[str, object], active_trajectories[name])
        initial_acquisition = int(bool(trajectory["initially_present"]))
        if (
            transition_causes[name]["acquisition_episode_count"]
            != cast(int, trajectory["acquisition_episode_count"])
            - initial_acquisition
            or transition_causes[name]["loss_episode_count"]
            != trajectory["loss_episode_count"]
            or transition_causes[name]["all_changed_slots_accounted"] is not True
        ):
            raise ValueError(f"active {name} transition-cause audit does not close")

    return CompositionalControlLifeArmAnalysisReceipt(
        curation_geometry_arm_name=curation_geometry_arm_name,
        execution_receipt=execution_receipt,
        curation_totals=curation_totals,
        _active_structural_trajectories_json=_canonical_json_bytes(
            active_trajectories
        ).decode("ascii"),
        _candidate_structural_trajectories_json=_canonical_json_bytes(
            candidate_trajectories
        ).decode("ascii"),
        _curation_decision_audit_json=_canonical_json_bytes(curation_audit).decode(
            "ascii"
        ),
        curation_decision_audit_array_elements=audit_elements,
        curation_decision_audit_ephemeral_bytes=audit_nbytes,
        _validation_token=_VALIDATED_RECEIPT_TOKEN,
    )


def _run_arm(
    protocol: CompositionalControlLifeProtocol,
    arm: CompositionalControlLifeArm,
    learner_key: Array,
    observations: Array,
    phase_indices: Array,
    exploration_mask: Array,
    random_actions: Array,
) -> dict[str, object]:
    learner = _build_learner(arm)
    execution = execute_compositional_control_life_arm(
        protocol,
        learner,
        learner_key,
        observations,
        phase_indices,
        exploration_mask,
        random_actions,
        composed_readout_enabled=arm.composed_readout_enabled,
    )
    state = execution.initial_state
    final_state = execution.final_state
    events = execution.events
    initial_ranking_diagnostics = execution.initial_ranking_diagnostics
    initial_active_counts = execution.initial_active_signature_counts
    initial_candidate_counts = execution.initial_candidate_signature_counts
    initial_active_pair_counts = execution.initial_active_raw_pair_counts
    initial_candidate_pair_counts = execution.initial_candidate_raw_pair_counts
    expected_nbytes = execution.expected_persistent_state_nbytes
    initial_nbytes = execution.initial_persistent_state_nbytes
    final_nbytes = execution.final_persistent_state_nbytes
    initial_state_sha256 = execution.initial_state_sha256
    final_state_sha256 = execution.final_state_sha256
    final_state_finite = True

    initial_ranking = _initial_ranking_record(state, initial_ranking_diagnostics)
    curation_totals = np.sum(
        np.asarray(events.curation_counts, dtype=np.int64), axis=0
    )
    curation_total_record = {
        name: int(value)
        for name, value in zip(CURATION_COUNT_NAMES, curation_totals, strict=True)
    }
    active_trajectories: dict[str, object] = {}
    candidate_trajectories: dict[str, object] = {}
    for index, name in enumerate(SIGNATURE_NAMES):
        active_trajectories[name] = _structural_trajectory(
            int(initial_active_counts[index]),
            events.active_signature_counts[:, index],
        )
        candidate_trajectories[name] = _structural_trajectory(
            int(initial_candidate_counts[index]),
            events.candidate_signature_counts[:, index],
        )
    curation_decision_audit, audit_elements, audit_nbytes = (
        _curation_decision_audit(protocol, events)
    )
    if curation_decision_audit["shared_p45_active_bank_loss_count"] != cast(
        Mapping[str, object], active_trajectories["shared_p45"]
    )["loss_episode_count"]:
        raise RuntimeError("active shared-p45 loss audit does not close")
    if curation_decision_audit["shared_p45_candidate_bank_loss_count"] != cast(
        Mapping[str, object], candidate_trajectories["shared_p45"]
    )["loss_episode_count"]:
        raise RuntimeError("candidate shared-p45 loss audit does not close")
    transition_causes = cast(
        Mapping[str, Mapping[str, object]],
        curation_decision_audit["active_signature_transition_causes"],
    )
    for name in AUDITED_ADMISSION_SIGNATURE_NAMES:
        trajectory = cast(Mapping[str, object], active_trajectories[name])
        initial_acquisition = int(bool(trajectory["initially_present"]))
        if (
            transition_causes[name]["acquisition_episode_count"]
            != cast(int, trajectory["acquisition_episode_count"])
            - initial_acquisition
            or transition_causes[name]["loss_episode_count"]
            != trajectory["loss_episode_count"]
            or transition_causes[name]["all_changed_slots_accounted"] is not True
        ):
            raise RuntimeError(f"active {name} transition-cause audit does not close")

    return {
        "arm": arm.name,
        "role": arm.role,
        "arm_definition": arm.to_config(),
        "learner_config": learner.to_config(),
        "learner_config_sha256": _json_sha256(learner.to_config()),
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
        "trace_sha256": execution.trace_sha256,
        "initial_persistent_state_nbytes": initial_nbytes,
        "final_persistent_state_nbytes": final_nbytes,
        "expected_persistent_state_nbytes": expected_nbytes,
        "expected_persistent_state_nbytes_formula": (
            "(56+12H)N + 4NC + (68+12H)C + 12H + 12GK + 32"
        ),
        "final_step_count_telemetry": int(final_state.step_count),
        "final_step_words_uint32": [
            int(value) for value in np.asarray(final_state.step_words)
        ],
        "final_replacement_phase": int(final_state.replacement_phase),
        "initial_state_finite": _state_is_finite(state),
        "final_state_finite": final_state_finite,
        "all_lifetime_counters_valid": bool(
            np.all(np.asarray(events.lifetime_counter_valid))
        ),
        "all_lifetime_capacity_available": bool(
            np.all(np.asarray(events.lifetime_capacity_available))
        ),
        "all_ranking_contracts_valid": bool(
            np.all(np.asarray(events.ranking_contract_valid))
        ),
        "all_core_predictions_match_full_q": bool(
            np.all(np.asarray(events.core_prediction_matches_full_q))
        ),
        "curation_totals": curation_total_record,
        "lifetime_metrics": _window_metrics(events, 0, protocol.total_steps),
        "phase_metrics": _phase_records(
            protocol,
            initial_ranking,
            initial_active_counts,
            initial_candidate_counts,
            events,
        ),
        "signature_manifest": [
            {
                "name": name,
                "role": role,
                "raw_indices": list(indices),
                "exponents": [int(value) for value in np.asarray(_SIGNATURE_MATRIX[row])],
            }
            for row, (name, role, indices) in enumerate(
                zip(
                    SIGNATURE_NAMES,
                    SIGNATURE_ROLES,
                    SIGNATURE_RAW_INDICES,
                    strict=True,
                )
            )
        ],
        "active_structural_trajectories": active_trajectories,
        "candidate_structural_trajectories": candidate_trajectories,
        "active_target_coexistence": _active_target_coexistence_record(
            events.active_signature_counts,
            start_post_step=0,
        ),
        "curation_decision_audit": curation_decision_audit,
        "raw_pair_coverage": _raw_pair_coverage_record(
            initial_active_pair_counts,
            initial_candidate_pair_counts,
            events.active_raw_pair_counts,
            events.candidate_raw_pair_counts,
        ),
        "raw_pair_reachability": _raw_pair_reachability_record(
            arm,
            bool(
                jnp.any(
                    (state.depth >= 1)
                    & (state.depth + 1 <= arm.effective_max_depth)
                )
            ),
            events,
            curation_total_record["cascade_refill"],
        ),
        "initial_ranking": initial_ranking,
        "final_ranking": _event_ranking_record(events, protocol.total_steps - 1),
        "work": {
            "learner_updates": protocol.total_steps,
            "active_feature_evaluations": protocol.total_steps * ACTIVE_SLOTS,
            "candidate_feature_evaluations": protocol.total_steps * CANDIDATE_SLOTS,
            "action_head_evaluations": protocol.total_steps * ACTION_HEADS,
            "behavior_q_dot_products": protocol.total_steps * 2,
            "ranking_diagnostic_calls": protocol.total_steps + 1,
            "candidate_active_correlation_cells_per_step": (
                ACTIVE_SLOTS * CANDIDATE_SLOTS
            ),
            "curation_decision_audit_events": cast(
                int, curation_decision_audit["due_curation_event_count"]
            ),
            "curation_decision_audit_array_elements": audit_elements,
            "curation_decision_audit_ephemeral_bytes": audit_nbytes,
            "curation_decision_audit_report_json_bytes": cast(
                int, curation_decision_audit["records_canonical_json_bytes"]
            ),
            "persistent_search_archive_entries": 0,
        },
    }


def _resolve_arm_names(arm_names: Sequence[str] | None) -> tuple[str, ...]:
    if arm_names is None:
        return _CANONICAL_ARM_NAMES
    if type(arm_names) not in {tuple, list}:
        raise TypeError("arm_names must be a list/tuple canonical subset")
    resolved = tuple(arm_names)
    if not resolved or any(type(name) is not str for name in resolved):
        raise ValueError("arm_names must contain exact strings")
    canonical_subset = tuple(name for name in _CANONICAL_ARM_NAMES if name in resolved)
    if resolved != canonical_subset or len(set(resolved)) != len(resolved):
        raise ValueError("arm_names must be a unique canonical-order subset")
    return resolved


def build_bound_compositional_control_life_source(
    protocol: CompositionalControlLifeProtocol,
    *,
    observation_key: Array,
    exploration_key: Array,
    random_action_key: Array,
    learner_key: Array,
) -> BoundCompositionalControlLifeSource:
    """Build one root-agnostic source without selecting a protocol identity."""

    if type(protocol) is not CompositionalControlLifeProtocol:
        raise TypeError("protocol must be an exact CompositionalControlLifeProtocol")
    named_keys = (
        ("observations", observation_key),
        ("exploration", exploration_key),
        ("random_actions", random_action_key),
        ("learner_genesis", learner_key),
    )
    words: dict[str, tuple[int, int]] = {}
    for name, key in named_keys:
        key_words = _key_words(key)
        words[name] = (key_words[0], key_words[1])
    if len(set(words.values())) != len(words):
        raise ValueError("bound source keys must be pairwise distinct")

    observations = jnp.where(
        jr.bernoulli(
            observation_key,
            0.5,
            (protocol.total_steps, RAW_DIM),
        ),
        1.0,
        -1.0,
    ).astype(jnp.float32)
    exploration_mask = jr.bernoulli(
        exploration_key,
        protocol.epsilon,
        (protocol.total_steps,),
    )
    random_actions = jr.bernoulli(
        random_action_key,
        0.5,
        (protocol.total_steps,),
    ).astype(jnp.int32)
    phase_indices = jnp.asarray(
        np.repeat(
            np.arange(len(PHASE_ORDER), dtype=np.int32),
            np.asarray(protocol.phase_lengths, dtype=np.int32),
        ),
        dtype=jnp.int32,
    )
    curation_due_mask = (
        (jnp.arange(protocol.total_steps, dtype=jnp.int32) + 1)
        % CURATION_INTERVAL
        == 0
    )
    stream_arrays = (
        observations,
        phase_indices,
        exploration_mask,
        random_actions,
    )
    return BoundCompositionalControlLifeSource(
        key_manifest=MappingProxyType(words),
        observations=observations,
        phase_indices=phase_indices,
        exploration_mask=exploration_mask,
        random_actions=random_actions,
        learner_key=learner_key,
        curation_due_mask=curation_due_mask,
        stream_sha256=_array_tree_sha256(stream_arrays),
        cadence_bound_stream_sha256=_array_tree_sha256(
            (*stream_arrays, curation_due_mask)
        ),
    )


def _stream_arrays(
    protocol: CompositionalControlLifeProtocol,
    seed: int,
) -> tuple[dict[str, list[int]], Array, Array, Array, Array, str]:
    root = jr.key(seed, impl="threefry2x32")
    observation_key = jr.fold_in(root, jnp.uint32(OBSERVATION_DOMAIN))
    exploration_key = jr.fold_in(root, jnp.uint32(EXPLORATION_DOMAIN))
    random_action_key = jr.fold_in(root, jnp.uint32(RANDOM_ACTION_DOMAIN))
    learner_key = jr.fold_in(root, jnp.uint32(LEARNER_DOMAIN))
    source = build_bound_compositional_control_life_source(
        protocol,
        observation_key=observation_key,
        exploration_key=exploration_key,
        random_action_key=random_action_key,
        learner_key=learner_key,
    )
    manifest = {
        "root": _key_words(root),
        **{name: list(words) for name, words in source.key_manifest.items()},
    }
    return (
        manifest,
        source.observations,
        source.phase_indices,
        source.exploration_mask,
        source.random_actions,
        source.stream_sha256,
    )


_TOP_LEVEL_FIELDS: Final = {
    "schema",
    "status",
    "development_only",
    "acceptance_status",
    "scientific_promotion_allowed",
    "evidence_authorized",
    "artifact_bytes_written",
    "interpretation",
    "limitations",
    "protocol",
    "protocol_sha256",
    "source_manifest",
    "seed",
    "seed_role",
    "key_manifest",
    "stream_sha256",
    "arm_order",
    "arm_definitions",
    "runs",
    "identity_tracking",
    "work_resource_contract",
}


def _validate_raw_pair_coverage(coverage: object, *, arm_name: str) -> None:
    if not isinstance(coverage, Mapping):
        raise ValueError(f"{arm_name} raw-pair coverage is not a mapping")
    mapping_fields = (
        "initial_active",
        "initial_candidate",
        "ever_active",
        "ever_candidate",
        "ever_either",
        "new_after_genesis_active",
        "new_after_genesis_candidate",
    )
    decoded: dict[str, tuple[bool, ...]] = {}
    for field in mapping_fields:
        raw_values = coverage.get(field)
        if not isinstance(raw_values, Mapping) or tuple(raw_values) != RAW_PAIR_NAMES:
            raise ValueError(f"{arm_name} {field} pair order is invalid")
        values = tuple(raw_values[name] for name in RAW_PAIR_NAMES)
        if any(type(value) is not bool for value in values):
            raise ValueError(f"{arm_name} {field} values must be booleans")
        decoded[field] = cast(tuple[bool, ...], values)

    def bitset(values: tuple[bool, ...]) -> int:
        return sum(int(value) << index for index, value in enumerate(values))

    for field in (
        "initial_active",
        "initial_candidate",
        "ever_active",
        "ever_candidate",
        "ever_either",
    ):
        if coverage.get(f"{field}_bitset") != bitset(decoded[field]):
            raise ValueError(f"{arm_name} {field} bitset is invalid")
    ever_either = tuple(
        active or candidate
        for active, candidate in zip(
            decoded["ever_active"], decoded["ever_candidate"], strict=True
        )
    )
    if decoded["ever_either"] != ever_either:
        raise ValueError(f"{arm_name} ever-either raw-pair union is invalid")
    for bank in ("active", "candidate"):
        expected_new = tuple(
            ever and not initial
            for ever, initial in zip(
                decoded[f"ever_{bank}"],
                decoded[f"initial_{bank}"],
                strict=True,
            )
        )
        if decoded[f"new_after_genesis_{bank}"] != expected_new:
            raise ValueError(f"{arm_name} new {bank} raw-pair map is invalid")
        if coverage.get(f"ever_{bank}_count") != sum(decoded[f"ever_{bank}"]):
            raise ValueError(f"{arm_name} {bank} raw-pair count is invalid")
    if coverage.get("ever_either_count") != sum(ever_either):
        raise ValueError(f"{arm_name} either-bank raw-pair count is invalid")
    if coverage.get("pair_order") != list(RAW_PAIR_NAMES) or coverage.get(
        "pair_count"
    ) != len(RAW_PAIR_NAMES):
        raise ValueError(f"{arm_name} raw-pair namespace is invalid")
    expected_missing = [
        name
        for name, present in zip(RAW_PAIR_NAMES, ever_either, strict=True)
        if not present
    ]
    if coverage.get("missing_either") != expected_missing:
        raise ValueError(f"{arm_name} missing raw-pair list is invalid")


def _validate_active_target_coexistence_record(
    record: object,
    *,
    arm_name: str,
    expected_steps: int,
    first_post_step: int,
    last_post_step: int,
    expected_end: list[str],
) -> None:
    expected_fields = {
        "target_order",
        "steps",
        "steps_by_active_target_count",
        "maximum_active_target_count",
        "all_three_present_steps",
        "all_three_presence_fraction",
        "first_all_three_post_step",
        "last_all_three_post_step",
        "active_targets_at_end",
    }
    if not isinstance(record, Mapping) or set(record) != expected_fields:
        raise ValueError(f"{arm_name} active target coexistence fields are invalid")
    histogram = record.get("steps_by_active_target_count")
    if (
        record.get("target_order") != ["A", "B", "C"]
        or record.get("steps") != expected_steps
        or not isinstance(histogram, list)
        or len(histogram) != 4
        or any(type(value) is not int or value < 0 for value in histogram)
        or sum(histogram) != expected_steps
        or record.get("active_targets_at_end") != expected_end
    ):
        raise ValueError(f"{arm_name} active target coexistence payload is invalid")
    expected_maximum = max(
        (index for index, count in enumerate(histogram) if count),
        default=0,
    )
    all_three_steps = histogram[3]
    if (
        record.get("maximum_active_target_count") != expected_maximum
        or record.get("all_three_present_steps") != all_three_steps
        or record.get("all_three_presence_fraction")
        != all_three_steps / expected_steps
    ):
        raise ValueError(f"{arm_name} active target coexistence algebra is invalid")
    first = record.get("first_all_three_post_step")
    last = record.get("last_all_three_post_step")
    if all_three_steps == 0:
        if first is not None or last is not None:
            raise ValueError(
                f"{arm_name} active target coexistence empty timing is invalid"
            )
    elif (
        type(first) is not int
        or type(last) is not int
        or not first_post_step <= first <= last <= last_post_step
    ):
        raise ValueError(f"{arm_name} active target coexistence timing is invalid")


def _validate_curation_decision_audit(
    audit: object,
    *,
    arm_name: str,
    protocol: CompositionalControlLifeProtocol,
    active_trajectories: Mapping[str, object],
    candidate_trajectories: Mapping[str, object],
) -> None:
    if not isinstance(audit, Mapping) or set(audit) != {
        "scope",
        "float_encoding",
        "due_curation_event_count",
        "due_curation_records",
        "records_sha256",
        "records_canonical_json_bytes",
        "target_outcome_counts",
        "all_target_due_events_accounted",
        "active_signature_transition_causes",
        "shared_p45_active_bank_loss_count",
        "shared_p45_candidate_bank_loss_count",
        "all_shared_p45_bank_losses_accounted",
        "ephemeral_array_elements",
        "ephemeral_array_bytes",
    }:
        raise ValueError(f"{arm_name} curation decision audit fields are invalid")
    records = audit.get("due_curation_records")
    expected_count = protocol.total_steps // CURATION_INTERVAL
    if (
        not isinstance(records, list)
        or audit.get("due_curation_event_count") != expected_count
        or len(records) != expected_count
        or audit.get("records_sha256") != _json_sha256(records)
        or audit.get("records_canonical_json_bytes")
        != len(_canonical_json_bytes(records))
    ):
        raise ValueError(f"{arm_name} curation decision audit digest is invalid")
    expected_steps = list(
        range(CURATION_INTERVAL, protocol.total_steps + 1, CURATION_INTERVAL)
    )
    outcome_counts: dict[str, dict[str, int]] = {
        name: {} for name in AUDITED_ADMISSION_SIGNATURE_NAMES
    }
    cause_labels = (
        "promotion_root_replacement",
        "cascade_dependency_refill",
        "unmarked_signature_dependency_change",
    )
    transition_accumulators: dict[str, dict[str, object]] = {
        name: {
            "acquisition_events": [],
            "loss_events": [],
            "acquisition_slot_cause_counts": {
                label: 0 for label in cause_labels
            },
            "loss_slot_cause_counts": {label: 0 for label in cause_labels},
            "all_changed_slots_accounted": True,
        }
        for name in AUDITED_ADMISSION_SIGNATURE_NAMES
    }
    active_losses = 0
    candidate_losses = 0
    all_losses_accounted = True
    for expected_step, record in zip(expected_steps, records, strict=True):
        if not isinstance(record, Mapping):
            raise ValueError(f"{arm_name} curation decision audit record is invalid")
        if (
            record.get("post_step") != expected_step
            or record.get("pre_step") != expected_step - 1
            or record.get("decision_due") is not True
        ):
            raise ValueError(f"{arm_name} curation decision audit clock is invalid")
        for descriptor_name, slot_count in (
            ("pre_active_descriptors", ACTIVE_SLOTS),
            ("post_active_descriptors", ACTIVE_SLOTS),
            ("pre_candidate_descriptors", CANDIDATE_SLOTS),
            ("post_candidate_descriptors", CANDIDATE_SLOTS),
        ):
            descriptor = record.get(descriptor_name)
            if (
                not isinstance(descriptor, Mapping)
                or not isinstance(descriptor.get("ops"), list)
                or len(cast(list[object], descriptor["ops"])) != slot_count
            ):
                raise ValueError(
                    f"{arm_name} curation decision audit descriptors are invalid"
                )
        masks = record.get("destination_masks")
        if (
            not isinstance(masks, Mapping)
            or not isinstance(masks.get("compatible_active_bitsets"), list)
            or len(cast(list[object], masks["compatible_active_bitsets"]))
            != CANDIDATE_SLOTS
        ):
            raise ValueError(
                f"{arm_name} curation decision audit destination masks are invalid"
            )
        active_eligible = masks.get("active_eligible_bitset")
        mature = masks.get("candidate_mature_bitset")
        topology_masks = masks.get("topology_compatible_active_bitsets")
        depth_masks = masks.get("depth_compatible_active_bitsets")
        headroom_masks = masks.get("headroom_compatible_active_bitsets")
        compatible_masks = masks.get("compatible_active_bitsets")
        margin_eligible_masks = masks.get("margin_eligible_active_bitsets")
        candidate_ranking = record.get("candidate_ranking")
        recomputed_depth = (
            candidate_ranking.get("recomputed_depth")
            if isinstance(candidate_ranking, Mapping)
            else None
        )
        if (
            type(active_eligible) is not int
            or type(mature) is not int
            or not isinstance(topology_masks, list)
            or not isinstance(depth_masks, list)
            or not isinstance(headroom_masks, list)
            or not isinstance(compatible_masks, list)
            or not isinstance(margin_eligible_masks, list)
            or not isinstance(recomputed_depth, list)
            or any(
                len(values) != CANDIDATE_SLOTS
                for values in (
                    topology_masks,
                    depth_masks,
                    headroom_masks,
                    compatible_masks,
                    margin_eligible_masks,
                    recomputed_depth,
                )
            )
        ):
            raise ValueError(
                f"{arm_name} curation decision audit mask algebra is invalid"
            )
        for candidate_slot in range(CANDIDATE_SLOTS):
            expected_headroom = sum(
                1 << destination
                for destination in range(ACTIVE_SLOTS)
                if (
                    not _ARMS_BY_NAME[arm_name].topology_headroom_reserve
                    or destination
                    + (
                        _ARMS_BY_NAME[arm_name].effective_max_depth
                        - recomputed_depth[candidate_slot]
                    )
                    < ACTIVE_SLOTS
                )
            )
            if headroom_masks[candidate_slot] != expected_headroom:
                raise ValueError(
                    f"{arm_name} curation decision audit headroom mask is invalid"
                )
            expected_compatible = (
                active_eligible
                & topology_masks[candidate_slot]
                & depth_masks[candidate_slot]
                & headroom_masks[candidate_slot]
            )
            if not ((mature >> candidate_slot) & 1):
                expected_compatible = 0
            if compatible_masks[candidate_slot] != expected_compatible:
                raise ValueError(
                    f"{arm_name} curation decision audit final mask relation is invalid"
                )
        expected_has_destination = sum(
            int(bool(compatible_masks[candidate_slot])) << candidate_slot
            for candidate_slot in range(CANDIDATE_SLOTS)
        )
        if masks.get("candidate_has_destination_bitset") != expected_has_destination:
            raise ValueError(
                f"{arm_name} curation decision audit destination summary is invalid"
            )
        selection = record.get("selection")
        if not isinstance(selection, Mapping) or any(
            field not in selection
            for field in (
                "selected_candidate",
                "selected_destination",
                "effective_promotion_margin_f32_bits",
                "margin_rhs_f32_bits",
                "margin_passed",
                "selected_headroom_ok",
                "left_pack_destinations_enabled",
                "left_pack_destination_available",
            )
        ):
            raise ValueError(
                f"{arm_name} curation decision audit selection is invalid"
            )
        if type(selection["margin_passed"]) is not bool:
            raise ValueError(
                f"{arm_name} curation decision audit margin flag is invalid"
            )
        active_ranking = record.get("active_ranking")
        if not isinstance(active_ranking, Mapping) or not isinstance(
            candidate_ranking, Mapping
        ):
            raise ValueError(
                f"{arm_name} curation decision audit ranking payload is invalid"
            )
        active_backed_scores = _f32_from_bits(
            active_ranking.get("backed_scores_f32_bits")
        )
        candidate_scores = _f32_from_bits(
            candidate_ranking.get("augmented_scores_f32_bits")
        )
        promotion_margin = _f32_from_bits(
            selection.get("effective_promotion_margin_f32_bits")
        )
        if (
            active_backed_scores.shape != (ACTIVE_SLOTS,)
            or candidate_scores.shape != (CANDIDATE_SLOTS,)
            or promotion_margin.shape != ()
        ):
            raise ValueError(
                f"{arm_name} curation decision audit margin payload is invalid"
            )
        margin_rhs = np.multiply(
            promotion_margin,
            active_backed_scores,
            dtype=np.float32,
        )
        for candidate_slot in range(CANDIDATE_SLOTS):
            expected_margin_mask = sum(
                int(
                    bool((compatible_masks[candidate_slot] >> destination) & 1)
                    and bool(candidate_scores[candidate_slot] > margin_rhs[destination])
                )
                << destination
                for destination in range(ACTIVE_SLOTS)
            )
            if margin_eligible_masks[candidate_slot] != expected_margin_mask:
                raise ValueError(
                    f"{arm_name} curation decision audit margin mask is invalid"
                )
        leftpack_enabled = _ARMS_BY_NAME[
            arm_name
        ].topology_left_pack_destinations
        if selection["left_pack_destinations_enabled"] is not leftpack_enabled:
            raise ValueError(
                f"{arm_name} curation decision audit left-pack flag is invalid"
            )
        if leftpack_enabled:
            selected_candidate = selection["selected_candidate"]
            if type(selected_candidate) is not int:
                raise ValueError(
                    f"{arm_name} curation decision audit selected candidate is invalid"
                )
            selected_margin_mask = (
                0
                if selected_candidate < 0
                else margin_eligible_masks[selected_candidate]
            )
            available = bool(selected_margin_mask)
            expected_destination = (
                -1
                if not available
                else (selected_margin_mask & -selected_margin_mask).bit_length() - 1
            )
            if (
                selection["left_pack_destination_available"] is not available
                or selection["selected_destination"] != expected_destination
            ):
                raise ValueError(
                    f"{arm_name} curation decision audit left-pack selection is invalid"
                )
        elif selection["left_pack_destination_available"] is not False:
            raise ValueError(
                f"{arm_name} disabled left-pack availability must be false"
            )
        outcomes = record.get("target_admission_outcomes")
        if not isinstance(outcomes, Mapping) or tuple(outcomes) != (
            AUDITED_ADMISSION_SIGNATURE_NAMES
        ):
            raise ValueError(
                f"{arm_name} curation decision audit target outcomes are invalid"
            )
        for name in AUDITED_ADMISSION_SIGNATURE_NAMES:
            outcome_record = outcomes[name]
            if not isinstance(outcome_record, Mapping) or type(
                outcome_record.get("outcome")
            ) is not str:
                raise ValueError(
                    f"{arm_name} curation decision audit target outcome is invalid"
                )
            outcome = cast(str, outcome_record["outcome"])
            outcome_counts[name][outcome] = outcome_counts[name].get(outcome, 0) + 1
        transitions = record.get("active_signature_transitions")
        pre_signatures = record.get("pre_active_signatures")
        post_signatures = record.get("post_active_signatures")
        structural_changes = record.get("structural_changes")
        if (
            not isinstance(transitions, Mapping)
            or tuple(transitions) != AUDITED_ADMISSION_SIGNATURE_NAMES
            or not isinstance(pre_signatures, Mapping)
            or not isinstance(post_signatures, Mapping)
            or not isinstance(structural_changes, Mapping)
        ):
            raise ValueError(
                f"{arm_name} curation decision active transition payload is invalid"
            )
        pre_match_bitsets = pre_signatures.get("match_bitsets")
        post_match_bitsets = post_signatures.get("match_bitsets")
        root_bits = structural_changes.get("root_change_bitset")
        cascade_bits = structural_changes.get("cascade_refill_bitset")
        if (
            not isinstance(pre_match_bitsets, Mapping)
            or not isinstance(post_match_bitsets, Mapping)
            or type(root_bits) is not int
            or type(cascade_bits) is not int
        ):
            raise ValueError(
                f"{arm_name} curation decision transition masks are invalid"
            )
        active_mask = (1 << ACTIVE_SLOTS) - 1
        for name in AUDITED_ADMISSION_SIGNATURE_NAMES:
            transition = transitions[name]
            pre_bits = pre_match_bitsets.get(name)
            post_bits = post_match_bitsets.get(name)
            if (
                not isinstance(transition, Mapping)
                or type(pre_bits) is not int
                or type(post_bits) is not int
            ):
                raise ValueError(
                    f"{arm_name} curation decision signature transition is invalid"
                )
            acquired_bits = post_bits & ~pre_bits & active_mask
            lost_bits = pre_bits & ~post_bits & active_mask
            acquired_slots = [
                slot
                for slot in range(ACTIVE_SLOTS)
                if (acquired_bits >> slot) & 1
            ]
            lost_slots = [
                slot for slot in range(ACTIVE_SLOTS) if (lost_bits >> slot) & 1
            ]

            def expected_causes(slots: list[int]) -> dict[str, list[str]]:
                values: dict[str, list[str]] = {}
                for slot in slots:
                    labels: list[str] = []
                    if (root_bits >> slot) & 1:
                        labels.append("promotion_root_replacement")
                    if (cascade_bits >> slot) & 1:
                        labels.append("cascade_dependency_refill")
                    if not labels:
                        labels.append("unmarked_signature_dependency_change")
                    values[str(slot)] = labels
                return values

            acquired_causes = expected_causes(acquired_slots)
            lost_causes = expected_causes(lost_slots)
            accounted = all(
                "unmarked_signature_dependency_change" not in labels
                for labels in (*acquired_causes.values(), *lost_causes.values())
            )
            bank_acquisition = pre_bits == 0 and post_bits != 0
            bank_loss = pre_bits != 0 and post_bits == 0
            if transition != {
                "pre_present": pre_bits != 0,
                "post_present": post_bits != 0,
                "bank_acquisition": bank_acquisition,
                "bank_loss": bank_loss,
                "acquired_slots": acquired_slots,
                "lost_slots": lost_slots,
                "acquired_slot_causes": acquired_causes,
                "lost_slot_causes": lost_causes,
                "all_changed_slots_accounted": accounted,
            }:
                raise ValueError(
                    f"{arm_name} curation decision transition attribution is invalid"
                )
            accumulator = transition_accumulators[name]
            accumulator["all_changed_slots_accounted"] = bool(
                accumulator["all_changed_slots_accounted"]
            ) and bool(accounted)
            if bank_acquisition:
                cast(list[object], accumulator["acquisition_events"]).append(
                    {
                        "post_step": expected_step,
                        "acquired_slots": acquired_slots,
                        "slot_causes": acquired_causes,
                    }
                )
                counts = cast(
                    dict[str, int], accumulator["acquisition_slot_cause_counts"]
                )
                for labels in acquired_causes.values():
                    for label in labels:
                        counts[label] += 1
            if bank_loss:
                cast(list[object], accumulator["loss_events"]).append(
                    {
                        "post_step": expected_step,
                        "lost_slots": lost_slots,
                        "slot_causes": lost_causes,
                    }
                )
                counts = cast(
                    dict[str, int], accumulator["loss_slot_cause_counts"]
                )
                for labels in lost_causes.values():
                    for label in labels:
                        counts[label] += 1
        for field, bank in (
            ("shared_p45_active_bank_loss", "active"),
            ("shared_p45_candidate_bank_loss", "candidate"),
        ):
            loss = record.get(field)
            if loss is None:
                continue
            if not isinstance(loss, Mapping) or loss.get("bank_loss") is not True:
                raise ValueError(
                    f"{arm_name} curation decision audit p45 loss is invalid"
                )
            accounted = loss.get("all_lost_slots_accounted") is True
            all_losses_accounted &= accounted
            if bank == "active":
                active_losses += 1
            else:
                candidate_losses += 1
    if (
        audit.get("target_outcome_counts") != outcome_counts
        or audit.get("all_target_due_events_accounted") is not True
        or any(sum(counts.values()) != expected_count for counts in outcome_counts.values())
    ):
        raise ValueError(f"{arm_name} curation decision audit coverage is invalid")
    expected_transition_causes: dict[str, object] = {}
    for name in AUDITED_ADMISSION_SIGNATURE_NAMES:
        accumulator = transition_accumulators[name]
        acquisitions = cast(list[object], accumulator["acquisition_events"])
        losses = cast(list[object], accumulator["loss_events"])
        expected_transition_causes[name] = {
            "acquisition_episode_count": len(acquisitions),
            "loss_episode_count": len(losses),
            "acquisition_events": acquisitions,
            "loss_events": losses,
            "acquisition_slot_cause_counts": accumulator[
                "acquisition_slot_cause_counts"
            ],
            "loss_slot_cause_counts": accumulator["loss_slot_cause_counts"],
            "all_changed_slots_accounted": accumulator[
                "all_changed_slots_accounted"
            ],
        }
        trajectory = cast(Mapping[str, object], active_trajectories[name])
        initial_acquisition = int(bool(trajectory["initially_present"]))
        if (
            len(acquisitions)
            != cast(int, trajectory["acquisition_episode_count"])
            - initial_acquisition
            or len(losses) != trajectory["loss_episode_count"]
            or accumulator["all_changed_slots_accounted"] is not True
        ):
            raise ValueError(
                f"{arm_name} curation decision transition closure is invalid"
            )
    if audit.get("active_signature_transition_causes") != expected_transition_causes:
        raise ValueError(
            f"{arm_name} curation decision transition summary is invalid"
        )
    expected_active_losses = cast(
        Mapping[str, object], active_trajectories["shared_p45"]
    )["loss_episode_count"]
    expected_candidate_losses = cast(
        Mapping[str, object], candidate_trajectories["shared_p45"]
    )["loss_episode_count"]
    if (
        audit.get("shared_p45_active_bank_loss_count") != active_losses
        or audit.get("shared_p45_candidate_bank_loss_count") != candidate_losses
        or active_losses != expected_active_losses
        or candidate_losses != expected_candidate_losses
        or audit.get("all_shared_p45_bank_losses_accounted")
        is not all_losses_accounted
        or not all_losses_accounted
    ):
        raise ValueError(f"{arm_name} curation decision audit p45 accounting is invalid")
    for field in ("ephemeral_array_elements", "ephemeral_array_bytes"):
        if type(audit.get(field)) is not int or cast(int, audit[field]) <= 0:
            raise ValueError(
                f"{arm_name} curation decision audit telemetry size is invalid"
            )


def validate_compositional_control_life_report(
    report: Mapping[str, object],
    protocol: CompositionalControlLifeProtocol,
) -> None:
    """Strictly validate one in-memory descriptive report."""

    if set(report) != _TOP_LEVEL_FIELDS:
        raise ValueError("report fields do not match the v1 schema")
    if (
        report["schema"] != REPORT_SCHEMA
        or report["status"] != STATUS
        or report["development_only"] is not True
        or report["acceptance_status"] != ACCEPTANCE_STATUS
        or report["scientific_promotion_allowed"] is not False
        or report["evidence_authorized"] is not False
        or report["artifact_bytes_written"] != 0
    ):
        raise ValueError("report authority/status fields are invalid")
    if report["protocol"] != protocol.to_config():
        raise ValueError("report protocol does not match the supplied protocol")
    if report["protocol_sha256"] != _json_sha256(protocol.to_config()):
        raise ValueError("report protocol hash is invalid")
    if report["source_manifest"] != _source_manifest():
        raise ValueError("report source manifest is stale")
    if type(report["seed"]) is not int or report["seed"] not in CONSUMED_DEVELOPMENT_SEEDS:
        raise ValueError("report seed is not an already-consumed development seed")
    if report["seed_role"] != "consumed_development_nonpromoting":
        raise ValueError("report seed role is invalid")
    if not _is_sha256(report["stream_sha256"]):
        raise ValueError("stream hash is invalid")
    raw_order = report["arm_order"]
    if not isinstance(raw_order, list):
        raise TypeError("arm_order must be a JSON list")
    arm_order = _resolve_arm_names(cast(list[str], raw_order))
    expected_definitions = [_ARMS_BY_NAME[name].to_config() for name in arm_order]
    if report["arm_definitions"] != expected_definitions:
        raise ValueError("arm definitions are invalid")
    raw_runs = report["runs"]
    if not isinstance(raw_runs, list) or len(raw_runs) != len(arm_order):
        raise ValueError("report runs do not match arm order")
    expected_nbytes = compositional_control_state_nbytes_formula(
        active_slots=ACTIVE_SLOTS,
        candidate_slots=CANDIDATE_SLOTS,
        action_heads=ACTION_HEADS,
    )
    for arm_name, raw_run in zip(arm_order, raw_runs, strict=True):
        if not isinstance(raw_run, Mapping) or raw_run.get("arm") != arm_name:
            raise ValueError("run arm order is invalid")
        if raw_run.get("arm_definition") != _ARMS_BY_NAME[arm_name].to_config():
            raise ValueError(f"{arm_name} definition is invalid")
        expected_config = _build_learner(_ARMS_BY_NAME[arm_name]).to_config()
        if raw_run.get("learner_config") != expected_config:
            raise ValueError(f"{arm_name} learner config is invalid")
        if raw_run.get("learner_config_sha256") != _json_sha256(expected_config):
            raise ValueError(f"{arm_name} learner config hash is invalid")
        for field in ("initial_state_sha256", "final_state_sha256", "trace_sha256"):
            if not _is_sha256(raw_run.get(field)):
                raise ValueError(f"{arm_name} {field} is invalid")
        if any(
            raw_run.get(field) != expected_nbytes
            for field in (
                "initial_persistent_state_nbytes",
                "final_persistent_state_nbytes",
                "expected_persistent_state_nbytes",
            )
        ):
            raise ValueError(f"{arm_name} state byte accounting is invalid")
        if raw_run.get("final_step_words_uint32") != [0, protocol.total_steps]:
            raise ValueError(f"{arm_name} exact lifetime clock is invalid")
        for field in (
            "initial_state_finite",
            "final_state_finite",
            "all_lifetime_counters_valid",
            "all_lifetime_capacity_available",
            "all_ranking_contracts_valid",
            "all_core_predictions_match_full_q",
        ):
            if raw_run.get(field) is not True:
                raise ValueError(f"{arm_name} {field} must be true")
        phases = raw_run.get("phase_metrics")
        if not isinstance(phases, list) or len(phases) != len(PHASE_ORDER):
            raise ValueError(f"{arm_name} phase metrics are invalid")
        trajectories = raw_run.get("active_structural_trajectories")
        if not isinstance(trajectories, Mapping) or set(trajectories) != set(
            SIGNATURE_NAMES
        ):
            raise ValueError(f"{arm_name} active structural trajectories are invalid")
        candidate_trajectories = raw_run.get("candidate_structural_trajectories")
        if not isinstance(candidate_trajectories, Mapping) or set(
            candidate_trajectories
        ) != set(SIGNATURE_NAMES):
            raise ValueError(f"{arm_name} candidate trajectories are invalid")
        phase_coexistence: list[Mapping[str, object]] = []
        for phase_index, phase in enumerate(phases):
            if not isinstance(phase, Mapping):
                raise ValueError(f"{arm_name} phase payload is invalid")
            exit_counts = phase.get("exit_active_signature_counts")
            if not isinstance(exit_counts, Mapping):
                raise ValueError(f"{arm_name} phase exit counts are invalid")
            expected_phase_end = [
                name
                for name in ("A", "B", "C")
                if cast(int, exit_counts.get(name, 0)) > 0
            ]
            phase_start = sum(protocol.phase_lengths[:phase_index])
            phase_steps = protocol.phase_lengths[phase_index]
            raw_coexistence = phase.get("active_target_coexistence")
            _validate_active_target_coexistence_record(
                raw_coexistence,
                arm_name=arm_name,
                expected_steps=phase_steps,
                first_post_step=phase_start + 1,
                last_post_step=phase_start + phase_steps,
                expected_end=expected_phase_end,
            )
            phase_coexistence.append(cast(Mapping[str, object], raw_coexistence))
        expected_lifetime_end = [
            name
            for name in ("A", "B", "C")
            if cast(Mapping[str, object], trajectories[name])["present_at_end"]
            is True
        ]
        lifetime_coexistence = raw_run.get("active_target_coexistence")
        _validate_active_target_coexistence_record(
            lifetime_coexistence,
            arm_name=arm_name,
            expected_steps=protocol.total_steps,
            first_post_step=1,
            last_post_step=protocol.total_steps,
            expected_end=expected_lifetime_end,
        )
        lifetime_coexistence = cast(Mapping[str, object], lifetime_coexistence)
        aggregate_histogram = [
            sum(
                cast(list[int], phase["steps_by_active_target_count"])[count]
                for phase in phase_coexistence
            )
            for count in range(4)
        ]
        phase_firsts = [
            cast(int, phase["first_all_three_post_step"])
            for phase in phase_coexistence
            if phase["first_all_three_post_step"] is not None
        ]
        phase_lasts = [
            cast(int, phase["last_all_three_post_step"])
            for phase in phase_coexistence
            if phase["last_all_three_post_step"] is not None
        ]
        if (
            lifetime_coexistence["steps_by_active_target_count"]
            != aggregate_histogram
            or lifetime_coexistence["first_all_three_post_step"]
            != (None if not phase_firsts else min(phase_firsts))
            or lifetime_coexistence["last_all_three_post_step"]
            != (None if not phase_lasts else max(phase_lasts))
        ):
            raise ValueError(
                f"{arm_name} active target phase/lifetime coexistence is invalid"
            )
        _validate_curation_decision_audit(
            raw_run.get("curation_decision_audit"),
            arm_name=arm_name,
            protocol=protocol,
            active_trajectories=trajectories,
            candidate_trajectories=candidate_trajectories,
        )
        audit = cast(Mapping[str, object], raw_run["curation_decision_audit"])
        work = raw_run.get("work")
        if not isinstance(work, Mapping) or (
            work.get("curation_decision_audit_events")
            != audit["due_curation_event_count"]
            or work.get("curation_decision_audit_array_elements")
            != audit["ephemeral_array_elements"]
            or work.get("curation_decision_audit_ephemeral_bytes")
            != audit["ephemeral_array_bytes"]
            or work.get("curation_decision_audit_report_json_bytes")
            != audit["records_canonical_json_bytes"]
        ):
            raise ValueError(f"{arm_name} curation decision audit work is invalid")
        _validate_raw_pair_coverage(
            raw_run.get("raw_pair_coverage"), arm_name=arm_name
        )
        reachability = raw_run.get("raw_pair_reachability")
        if not isinstance(reachability, Mapping):
            raise ValueError(f"{arm_name} raw-pair reachability is invalid")
        cascade_count = cast(Mapping[str, object], raw_run.get("curation_totals")).get(
            "cascade_refill"
        )
        if (
            reachability.get("observed_cascade_refill_count") != cascade_count
            or reachability.get("cascade_loophole_exercised")
            is not (cast(int, cascade_count) > 0)
            or reachability.get("cascade_refill_is_raw_pair_support_loophole")
            is not True
            or reachability.get("depth1_ceiling_has_ordinary_raw_pair_support")
            is not (_ARMS_BY_NAME[arm_name].effective_max_depth == 1)
        ):
            raise ValueError(f"{arm_name} raw-pair reachability contract is invalid")
        applies = reachability.get(
            "conditional_theorem_applies_for_entire_observed_life"
        )
        support = reachability.get(
            "ordinary_fresh_raw_pair_support_for_entire_observed_life"
        )
        if type(applies) is not bool or type(support) is not bool or support is applies:
            raise ValueError(f"{arm_name} ordinary raw-pair support claim is invalid")
    identity = report["identity_tracking"]
    if identity != {
        "v4_birth_ledger_integrated": False,
        "retained_identity_assessed": False,
        "fresh_birth_identity_reacquisition_assessed": False,
        "reported_reacquisition_kind": "bank_level_algebraic_structural_only",
        "reason": (
            "the compiled scan does not execute authenticated host-ledger transitions; "
            "structural disappearance/reappearance cannot establish identity continuity"
        ),
    }:
        raise ValueError("identity limitation disclosure is invalid")
    if report["work_resource_contract"] != _work_resource_contract(
        protocol, arm_order
    ):
        raise ValueError("work/resource contract is invalid")


def _work_resource_contract(
    protocol: CompositionalControlLifeProtocol,
    arm_order: tuple[str, ...],
) -> dict[str, object]:
    expected_nbytes = compositional_control_state_nbytes_formula(
        active_slots=ACTIVE_SLOTS,
        candidate_slots=CANDIDATE_SLOTS,
        action_heads=ACTION_HEADS,
    )
    return {
        "selected_arm_count": len(arm_order),
        "steps_per_arm": protocol.total_steps,
        "learner_updates_per_event": 1,
        "full_and_raw_q_dot_products_per_event": 2,
        "ranking_diagnostic_calls_per_event": 1,
        "active_feature_evaluations_per_event": ACTIVE_SLOTS,
        "candidate_feature_evaluations_per_event": CANDIDATE_SLOTS,
        "action_heads_per_event": ACTION_HEADS,
        "candidate_active_correlation_cells_per_event": (
            ACTIVE_SLOTS * CANDIDATE_SLOTS
        ),
        "persistent_state_nbytes_per_arm": expected_nbytes,
        "persistent_state_formula": (
            "(56+12H)N + 4NC + (68+12H)C + 12H + 12GK + 32"
        ),
        "persistent_agent_state_complexity": "O(N*C + H*(N+C) + G*K)",
        "ordinary_step_structural_complexity": "O(N*C + H*(N+C))",
        "curation_structural_complexity": "O(N^2 + N*C)",
        "host_lineage_audit_complexity": "O((N+C)*2^depth)",
        "stream_and_action_keys_paired": True,
        "learner_genesis_key_paired": True,
        "persistent_shapes_matched": True,
        "update_opportunities_matched": True,
        "ranking_work_matched": True,
        "behavior_q_work_matched": True,
        "compiled_flop_equivalence_claimed": False,
        "depth1_static_generation_instruction_equivalence_claimed": False,
        "behavioral_experience_matching_claimed": False,
        "persistent_exhaustive_candidate_archive": False,
    }


def run_compositional_control_life_development(
    protocol: CompositionalControlLifeProtocol | None = None,
    *,
    seed: int = DEFAULT_CONSUMED_SEED,
    arm_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Run one paired consumed-seed life and return a strict in-memory report."""

    selected_protocol = build_default_protocol() if protocol is None else protocol
    if type(selected_protocol) is not CompositionalControlLifeProtocol:
        raise TypeError("protocol must be an exact CompositionalControlLifeProtocol")
    if type(seed) is not int or seed not in CONSUMED_DEVELOPMENT_SEEDS:
        raise ValueError("seed must be an already-consumed development root")
    selected_arms = _resolve_arm_names(arm_names)
    (
        key_manifest,
        observations,
        phase_indices,
        exploration_mask,
        random_actions,
        stream_sha256,
    ) = _stream_arrays(selected_protocol, seed)
    learner_key = jr.wrap_key_data(
        jnp.asarray(key_manifest["learner_genesis"], dtype=jnp.uint32),
        impl="threefry2x32",
    )
    runs = [
        _run_arm(
            selected_protocol,
            _ARMS_BY_NAME[name],
            learner_key,
            observations,
            phase_indices,
            exploration_mask,
            random_actions,
        )
        for name in selected_arms
    ]
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": STATUS,
        "development_only": DEVELOPMENT_ONLY,
        "acceptance_status": ACCEPTANCE_STATUS,
        "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        "evidence_authorized": EVIDENCE_AUTHORIZED,
        "artifact_bytes_written": 0,
        "interpretation": INTERPRETATION,
        "limitations": list(LIMITATIONS),
        "protocol": selected_protocol.to_config(),
        "protocol_sha256": _json_sha256(selected_protocol.to_config()),
        "source_manifest": _source_manifest(),
        "seed": seed,
        "seed_role": "consumed_development_nonpromoting",
        "key_manifest": key_manifest,
        "stream_sha256": stream_sha256,
        "arm_order": list(selected_arms),
        "arm_definitions": [_ARMS_BY_NAME[name].to_config() for name in selected_arms],
        "runs": runs,
        "identity_tracking": {
            "v4_birth_ledger_integrated": False,
            "retained_identity_assessed": False,
            "fresh_birth_identity_reacquisition_assessed": False,
            "reported_reacquisition_kind": "bank_level_algebraic_structural_only",
            "reason": (
                "the compiled scan does not execute authenticated host-ledger transitions; "
                "structural disappearance/reappearance cannot establish identity continuity"
            ),
        },
        "work_resource_contract": _work_resource_contract(
            selected_protocol, selected_arms
        ),
    }
    validate_compositional_control_life_report(report, selected_protocol)
    return report


__all__ = [
    "ACCEPTANCE_STATUS",
    "ACTIVE_SLOTS",
    "ALLOCATED_MAX_DEPTH",
    "ARM_ANALYSIS_RECEIPT_SCHEMA",
    "ARM_EXECUTION_RECEIPT_SCHEMA",
    "CANDIDATE_SLOTS",
    "CONSUMED_DEVELOPMENT_SEEDS",
    "CONTROL_LIFE_ARMS",
    "DEFAULT_CONSUMED_SEED",
    "DEFAULT_PHASE_LENGTHS",
    "DEVELOPMENT_ONLY",
    "EVIDENCE_AUTHORIZED",
    "PHASE_ORDER",
    "PROTOCOL_SCHEMA",
    "RAW_DIM",
    "RAW_PAIR_NAMES",
    "REPORT_SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SIGNATURE_NAMES",
    "STATUS",
    "BoundCompositionalControlLifeSource",
    "CompositionalControlLifeArmAnalysisReceipt",
    "CompositionalControlLifeArmExecution",
    "CompositionalControlLifeArmExecutionReceipt",
    "CompositionalControlLifeArm",
    "CompositionalControlLifeProtocol",
    "analyze_compositional_control_life_arm_execution",
    "build_default_protocol",
    "build_bound_compositional_control_life_source",
    "build_short_test_protocol",
    "compositional_control_state_nbytes_formula",
    "execute_compositional_control_life_arm",
    "learner_config_for_arm",
    "product_signature_counts",
    "run_compositional_control_life_development",
    "validate_compositional_control_life_arm_execution",
    "validate_compositional_control_life_report",
]
