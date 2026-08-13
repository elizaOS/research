"""Paired, nonpromoting development benchmark for compositional discovery.

This module replaces a lucky single-seed polynomial unit test with an explicit
multi-seed *development* lane.  Three online learners see the same stationary
stream ``y = x0 * x1 * x2 + noise`` for each root seed:

* ``raw_only`` learns a readout over a frozen raw-feature bank;
* ``prewired_depth2_oracle`` learns a readout over a frozen bank containing
  ``(x0 * x1) * x2``; and
* ``stochastic_discovery`` uses the former 20-active/20-candidate discovery
  configuration and its production curation path.

The fixed seeds are development seeds.  Outcomes are descriptive only: window
MSEs, their ratio, scheduled curation opportunities, applied active-root
replacements, promotion counts, and structural product-chain trajectories.
There is deliberately no pass threshold,
validator, evidence schema, held-out seed, or promotion path here.  Running or
rerunning this campaign cannot support an Alberta Plan completion claim.

Product-chain detection propagates exact raw-variable exponent signatures
through ``OP_PRODUCT`` nodes.  This makes ``(x0*x1)*x2`` and ``x0*(x1*x2)``
equivalent structurally.  It does *not* prove numeric equality to the target:
the production learner clips every intermediate feature value, so a matching
signature can depart from an unclipped polynomial on sufficiently large
inputs.  A disappearance followed by reappearance is reported as bank-level
structural recurrence, not as identity-level reacquisition.

Importing this module and calling :func:`build_development_plan` are inert.
Only :func:`run_development_campaign` executes learner updates, and it returns
in-memory dataclasses without writing an artifact.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Sequence
from typing import Any, Final

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from numpy.typing import NDArray

from alberta_framework.core.compositional_features import (
    OP_PRODUCT,
    OP_RAW,
    CompositionalFeatureLearner,
    CompositionalFeatureState,
)

logger = logging.getLogger(__name__)

COMPOSITIONAL_DISCOVERY_DEVELOPMENT_STATUS: Final = (
    "DEVELOPMENT_DESCRIPTIVE_NONPROMOTING"
)
THREEFRY_IMPLEMENTATION: Final = "threefry2x32"

# Stable big-endian ASCII fold-in domains.  Stream keys are shared across arms;
# learner keys are arm-specific.  These names and values are part of the
# development pairing contract, not a scientific seed-registration scheme.
STREAM_DOMAIN: Final = 0x5354524D  # STRM
OBSERVATION_DOMAIN: Final = 0x4F425356  # OBSV
NOISE_DOMAIN: Final = 0x4E4F4953  # NOIS
LEARNER_DOMAIN: Final = 0x4C524E52  # LRNR

RAW_ONLY: Final = "raw_only"
PREWIRED_DEPTH2_ORACLE: Final = "prewired_depth2_oracle"
STOCHASTIC_DISCOVERY: Final = "stochastic_discovery"

ARM_DOMAINS: Final[dict[str, int]] = {
    RAW_ONLY: 0x52415730,  # RAW0
    PREWIRED_DEPTH2_ORACLE: 0x4F52434C,  # ORCL
    STOCHASTIC_DISCOVERY: 0x44495343,  # DISC
}

# Fixed before this lane is run.  These are consumed development roots and
# must never be relabelled as untouched held-out/evidence seeds.
DEFAULT_DEVELOPMENT_SEEDS: Final[tuple[int, ...]] = (
    0x13A5C7E9,
    0x2476B8D1,
    0x35C9E2A7,
    0x46D1A3F5,
    0x57E4B629,
    0x68F2C4B7,
    0x79A6D835,
    0x8BC3E147,
)

PRODUCT_SIGNATURE_CLIPPING_CAVEAT: Final = (
    "Exact exponent signatures describe the stored RAW/PRODUCT graph only. "
    "Because production feature values are clipped after every intermediate "
    "operation, a matching graph need not equal the unclipped polynomial on "
    "large-magnitude observations."
)
STRUCTURAL_RECURRENCE_DEFINITION: Final = (
    "A recurrence is a false-to-true bank-level target-signature transition "
    "after its first acquisition; it is not a feature-identity or birth-ledger claim."
)


@dataclasses.dataclass(frozen=True, slots=True)
class DevelopmentArm:
    """One paired descriptive arm and its fixed learner budget."""

    name: str
    active_slots: int
    candidate_slots: int
    curation_interval: int
    frozen_structure: bool
    prewired_target: bool


DEVELOPMENT_ARMS: Final[tuple[DevelopmentArm, ...]] = (
    DevelopmentArm(
        name=RAW_ONLY,
        active_slots=4,
        candidate_slots=0,
        curation_interval=0,
        frozen_structure=True,
        prewired_target=False,
    ),
    DevelopmentArm(
        name=PREWIRED_DEPTH2_ORACLE,
        active_slots=6,
        candidate_slots=0,
        curation_interval=0,
        frozen_structure=True,
        prewired_target=True,
    ),
    DevelopmentArm(
        name=STOCHASTIC_DISCOVERY,
        active_slots=20,
        candidate_slots=20,
        curation_interval=20,
        frozen_structure=False,
        prewired_target=False,
    ),
)


@dataclasses.dataclass(frozen=True, slots=True)
class DevelopmentSeed:
    """One fixed, explicitly consumed development root."""

    index: int
    root_seed_uint32: int
    role: str = "development_consumed_nonpromoting"


@dataclasses.dataclass(frozen=True, slots=True)
class DevelopmentResourceAccounting:
    """Exact logical work requested by a plan, not a FLOP or peak-RAM claim."""

    paired_stream_count: int
    arm_count: int
    trial_count: int
    updates_per_trial: int
    total_learner_updates: int
    total_active_slot_update_exposures: int
    total_candidate_slot_update_exposures: int
    logical_descriptor_snapshots_per_trial: int
    total_initial_host_descriptor_int32_values: int
    total_scan_return_descriptor_int32_values: int
    total_scan_return_descriptor_int32_bytes: int
    total_logical_descriptor_int32_values: int
    total_metric_scalars_recorded: int
    total_event_index_scalars_recorded: int
    paired_stream_float32_values_per_seed: int
    paired_stream_float32_bytes_per_seed: int
    artifact_bytes_written_by_runner: int
    compiled_executable_or_peak_ram_bound_claimed: bool
    accounting_scope: str


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionalDiscoveryDevelopmentPlan:
    """Inert plan for the paired development campaign."""

    status: str
    development_only: bool
    scientific_evidence_authorized: bool
    promotion_authorized: bool
    key_implementation: str
    stream_domain: int
    observation_domain: int
    noise_domain: int
    learner_domain: int
    feature_dim: int
    target_raw_indices: tuple[int, int, int]
    num_steps: int
    window: int
    noise_std: float
    seeds: tuple[DevelopmentSeed, ...]
    arms: tuple[DevelopmentArm, ...]
    resource_accounting: DevelopmentResourceAccounting
    clipping_caveat: str
    recurrence_definition: str


@dataclasses.dataclass(frozen=True, slots=True)
class DevelopmentTrialKeys:
    """Auditable Threefry key words for one seed/arm pairing."""

    root_seed_uint32: int
    arm_name: str
    implementation: str
    root_key_words_uint32: tuple[int, int]
    observation_key_words_uint32: tuple[int, int]
    noise_key_words_uint32: tuple[int, int]
    learner_key_words_uint32: tuple[int, int]


@dataclasses.dataclass(frozen=True, slots=True)
class ProductChainSnapshot:
    """Slots structurally matching the target monomial in both banks."""

    active_slots: tuple[int, ...]
    candidate_slots: tuple[int, ...]

    @property
    def active_present(self) -> bool:
        return bool(self.active_slots)

    @property
    def candidate_present(self) -> bool:
        return bool(self.candidate_slots)


@dataclasses.dataclass(frozen=True, slots=True)
class ProductChainTrajectory:
    """Descriptive presence history for one bank."""

    present_at_final_step: bool
    ever_present: bool
    first_present_step: int | None
    presence_snapshot_count: int
    acquisition_episode_count: int
    recurrence_count: int


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionalDiscoveryTrialOutcome:
    """One arm/seed outcome with no acceptance interpretation."""

    status: str
    development_only: bool
    root_seed_uint32: int
    arm_name: str
    num_steps: int
    window: int
    initial_window_mse: float
    final_window_mse: float
    final_to_initial_mse_ratio: float
    scheduled_curation_opportunity_count: int
    active_root_replacement_count: int
    promotion_count: int
    active_product_chain: ProductChainTrajectory
    candidate_product_chain: ProductChainTrajectory


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionalDiscoveryArmSummary:
    """Across-seed descriptive aggregates for one arm."""

    arm_name: str
    trial_count: int
    mean_initial_window_mse: float
    mean_final_window_mse: float
    mean_final_to_initial_mse_ratio: float
    median_final_to_initial_mse_ratio: float
    total_scheduled_curation_opportunities: int
    total_active_root_replacements: int
    total_promotions: int
    active_ever_present_trials: int
    active_final_present_trials: int
    active_first_present_step_median: float | None
    active_total_recurrences: int
    candidate_ever_present_trials: int
    candidate_final_present_trials: int
    candidate_first_present_step_median: float | None
    candidate_total_recurrences: int


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionalDiscoveryCampaignResult:
    """In-memory result returned only by the explicit development runner."""

    plan: CompositionalDiscoveryDevelopmentPlan
    trials: tuple[CompositionalDiscoveryTrialOutcome, ...]
    arm_summaries: tuple[CompositionalDiscoveryArmSummary, ...]
    artifacts_written: int
    scientific_evidence_authorized: bool
    promotion_authorized: bool


@dataclasses.dataclass(frozen=True, slots=True)
class _DerivedKeys:
    observation: Array
    noise: Array
    learner: Array


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("development seeds must be exact integers")
    if not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError("development seeds must fit uint32")
    return seed


def _build_resource_accounting(
    *,
    seed_count: int,
    arms: tuple[DevelopmentArm, ...],
    num_steps: int,
    feature_dim: int,
) -> DevelopmentResourceAccounting:
    arm_count = len(arms)
    trial_count = seed_count * arm_count
    total_active_slots = sum(arm.active_slots for arm in arms)
    total_candidate_slots = sum(arm.candidate_slots for arm in arms)
    paired_values = num_steps * (feature_dim + 1)
    descriptor_values = (
        seed_count * num_steps * 3 * (total_active_slots + total_candidate_slots)
    )
    initial_descriptor_values = seed_count * 3 * (
        total_active_slots + total_candidate_slots
    )
    return DevelopmentResourceAccounting(
        paired_stream_count=seed_count,
        arm_count=arm_count,
        trial_count=trial_count,
        updates_per_trial=num_steps,
        total_learner_updates=trial_count * num_steps,
        total_active_slot_update_exposures=seed_count * num_steps * total_active_slots,
        total_candidate_slot_update_exposures=(
            seed_count * num_steps * total_candidate_slots
        ),
        logical_descriptor_snapshots_per_trial=num_steps + 1,
        total_initial_host_descriptor_int32_values=initial_descriptor_values,
        total_scan_return_descriptor_int32_values=descriptor_values,
        total_scan_return_descriptor_int32_bytes=(
            descriptor_values * np.dtype(np.int32).itemsize
        ),
        total_logical_descriptor_int32_values=(
            initial_descriptor_values + descriptor_values
        ),
        total_metric_scalars_recorded=trial_count * num_steps,
        total_event_index_scalars_recorded=trial_count * num_steps * 2,
        paired_stream_float32_values_per_seed=paired_values,
        paired_stream_float32_bytes_per_seed=paired_values * np.dtype(np.float32).itemsize,
        artifact_bytes_written_by_runner=0,
        compiled_executable_or_peak_ram_bound_claimed=False,
        accounting_scope=(
            "Logical learner updates, bank-slot exposures, one initial host-state plus "
            "post-update descriptor snapshots, exact scan-return descriptor/metric/event "
            "scalars, and one paired float32 stream. Initial descriptors are read from "
            "learner state and are not counted as scan-return bytes. JAX executable, "
            "allocator, state, and scan-intermediate peak memory are not estimated."
        ),
    )


def build_development_plan(
    *,
    root_seeds: Sequence[int] = DEFAULT_DEVELOPMENT_SEEDS,
    num_steps: int = 5_000,
    window: int = 500,
    feature_dim: int = 4,
    noise_std: float = 0.05,
) -> CompositionalDiscoveryDevelopmentPlan:
    """Build an inert, nonpromoting paired-campaign plan.

    Custom seeds are allowed for cheap development smoke tests, but they do
    not acquire held-out or evidence status by being passed here.
    """

    if isinstance(num_steps, bool) or not isinstance(num_steps, int):
        raise TypeError("num_steps must be an exact integer")
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be an exact integer")
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
        raise TypeError("feature_dim must be an exact integer")
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if not 1 <= window or 2 * window > num_steps:
        raise ValueError("window must be positive and initial/final windows must be disjoint")
    if feature_dim < 3:
        raise ValueError("feature_dim must expose raw inputs x0, x1, and x2")
    if not math.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("noise_std must be finite and nonnegative")

    seed_values = tuple(_validate_seed(seed) for seed in root_seeds)
    if not seed_values:
        raise ValueError("at least one development seed is required")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("development root seeds must be unique")
    seeds = tuple(
        DevelopmentSeed(index=index, root_seed_uint32=seed)
        for index, seed in enumerate(seed_values)
    )

    arms = tuple(
        dataclasses.replace(
            arm,
            active_slots=(
                feature_dim
                if arm.name == RAW_ONLY
                else feature_dim + 2
                if arm.name == PREWIRED_DEPTH2_ORACLE
                else arm.active_slots
            ),
        )
        for arm in DEVELOPMENT_ARMS
    )
    if feature_dim + 2 > next(
        arm.active_slots for arm in arms if arm.name == STOCHASTIC_DISCOVERY
    ):
        raise ValueError(
            "feature_dim plus two composed slots must fit the discovery arm's "
            "20-slot budget"
        )

    return CompositionalDiscoveryDevelopmentPlan(
        status=COMPOSITIONAL_DISCOVERY_DEVELOPMENT_STATUS,
        development_only=True,
        scientific_evidence_authorized=False,
        promotion_authorized=False,
        key_implementation=THREEFRY_IMPLEMENTATION,
        stream_domain=STREAM_DOMAIN,
        observation_domain=OBSERVATION_DOMAIN,
        noise_domain=NOISE_DOMAIN,
        learner_domain=LEARNER_DOMAIN,
        feature_dim=feature_dim,
        target_raw_indices=(0, 1, 2),
        num_steps=num_steps,
        window=window,
        noise_std=float(noise_std),
        seeds=seeds,
        arms=arms,
        resource_accounting=_build_resource_accounting(
            seed_count=len(seeds),
            arms=arms,
            num_steps=num_steps,
            feature_dim=feature_dim,
        ),
        clipping_caveat=PRODUCT_SIGNATURE_CLIPPING_CAVEAT,
        recurrence_definition=STRUCTURAL_RECURRENCE_DEFINITION,
    )


def validate_development_plan(
    plan: CompositionalDiscoveryDevelopmentPlan,
) -> CompositionalDiscoveryDevelopmentPlan:
    """Reconstruct and compare every field of a development plan."""

    if type(plan) is not CompositionalDiscoveryDevelopmentPlan:
        raise TypeError("plan must be an exact CompositionalDiscoveryDevelopmentPlan")
    canonical = build_development_plan(
        root_seeds=tuple(seed.root_seed_uint32 for seed in plan.seeds),
        num_steps=plan.num_steps,
        window=plan.window,
        feature_dim=plan.feature_dim,
        noise_std=plan.noise_std,
    )
    if plan != canonical:
        raise ValueError("development plan does not match canonical reconstruction")
    return plan


def _derive_keys(root_seed_uint32: int, arm_name: str) -> _DerivedKeys:
    seed = _validate_seed(root_seed_uint32)
    try:
        arm_domain = ARM_DOMAINS[arm_name]
    except KeyError as error:
        raise ValueError(f"unknown development arm: {arm_name!r}") from error
    root = jr.key(seed, impl=THREEFRY_IMPLEMENTATION)
    stream = jr.fold_in(root, np.uint32(STREAM_DOMAIN))
    learner_root = jr.fold_in(root, np.uint32(LEARNER_DOMAIN))
    return _DerivedKeys(
        observation=jr.fold_in(stream, np.uint32(OBSERVATION_DOMAIN)),
        noise=jr.fold_in(stream, np.uint32(NOISE_DOMAIN)),
        learner=jr.fold_in(learner_root, np.uint32(arm_domain)),
    )


def _key_words(key: Array) -> tuple[int, int]:
    words = np.asarray(jr.key_data(key), dtype=np.uint32).reshape(-1)
    if words.shape != (2,):
        raise RuntimeError("Threefry development key must contain exactly two uint32 words")
    return int(words[0]), int(words[1])


def derive_trial_key_manifest(
    root_seed_uint32: int,
    arm_name: str,
) -> DevelopmentTrialKeys:
    """Return stable key words without executing a learner or stream."""

    seed = _validate_seed(root_seed_uint32)
    root = jr.key(seed, impl=THREEFRY_IMPLEMENTATION)
    keys = _derive_keys(seed, arm_name)
    return DevelopmentTrialKeys(
        root_seed_uint32=seed,
        arm_name=arm_name,
        implementation=THREEFRY_IMPLEMENTATION,
        root_key_words_uint32=_key_words(root),
        observation_key_words_uint32=_key_words(keys.observation),
        noise_key_words_uint32=_key_words(keys.noise),
        learner_key_words_uint32=_key_words(keys.learner),
    )


def _int_vector(value: object, *, name: str) -> NDArray[np.int64]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be rank one")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype")
    return np.asarray(array, dtype=np.int64)


def _active_exponent_signatures(
    *,
    ops: object,
    parent_a: object,
    parent_b: object,
    feature_dim: int,
) -> tuple[tuple[int, ...] | None, ...]:
    op_values = _int_vector(ops, name="ops")
    parent_a_values = _int_vector(parent_a, name="parent_a")
    parent_b_values = _int_vector(parent_b, name="parent_b")
    if not (
        op_values.shape == parent_a_values.shape == parent_b_values.shape
    ):
        raise ValueError("active descriptor arrays must have identical shapes")

    signatures: list[tuple[int, ...] | None] = []
    for slot, op in enumerate(op_values):
        first = int(parent_a_values[slot])
        second = int(parent_b_values[slot])
        signature: tuple[int, ...] | None = None
        if int(op) == OP_RAW and 0 <= first < feature_dim:
            exponents = [0] * feature_dim
            exponents[first] = 1
            signature = tuple(exponents)
        elif (
            int(op) == OP_PRODUCT
            and 0 <= first < slot
            and 0 <= second < slot
            and signatures[first] is not None
            and signatures[second] is not None
        ):
            left = signatures[first]
            right = signatures[second]
            assert left is not None and right is not None
            signature = tuple(a + b for a, b in zip(left, right, strict=True))
        signatures.append(signature)
    return tuple(signatures)


def detect_product_chain(
    *,
    active_ops: object,
    active_parent_a: object,
    active_parent_b: object,
    candidate_ops: object,
    candidate_parent_a: object,
    candidate_parent_b: object,
    feature_dim: int,
    target_raw_indices: tuple[int, int, int] = (0, 1, 2),
) -> ProductChainSnapshot:
    """Detect exact target-monomial signatures in active and candidate banks.

    Only RAW and PRODUCT structure is interpreted.  SUM, TANH, and GATED
    nodes receive no polynomial signature.  Candidate parents are resolved
    against the active bank, matching production candidate semantics.
    """

    if feature_dim < 1:
        raise ValueError("feature_dim must be positive")
    if len(target_raw_indices) != 3:
        raise ValueError("the development target must name exactly three raw indices")
    target = [0] * feature_dim
    for raw_index in target_raw_indices:
        if not 0 <= raw_index < feature_dim:
            raise ValueError("target raw indices must fall inside feature_dim")
        target[raw_index] += 1
    target_signature = tuple(target)

    active_signatures = _active_exponent_signatures(
        ops=active_ops,
        parent_a=active_parent_a,
        parent_b=active_parent_b,
        feature_dim=feature_dim,
    )
    candidate_op_values = _int_vector(candidate_ops, name="candidate_ops")
    candidate_a_values = _int_vector(candidate_parent_a, name="candidate_parent_a")
    candidate_b_values = _int_vector(candidate_parent_b, name="candidate_parent_b")
    if not (
        candidate_op_values.shape
        == candidate_a_values.shape
        == candidate_b_values.shape
    ):
        raise ValueError("candidate descriptor arrays must have identical shapes")

    candidate_slots: list[int] = []
    for slot, op in enumerate(candidate_op_values):
        first = int(candidate_a_values[slot])
        second = int(candidate_b_values[slot])
        signature: tuple[int, ...] | None = None
        if int(op) == OP_RAW and 0 <= first < feature_dim:
            exponents = [0] * feature_dim
            exponents[first] = 1
            signature = tuple(exponents)
        elif (
            int(op) == OP_PRODUCT
            and 0 <= first < len(active_signatures)
            and 0 <= second < len(active_signatures)
            and active_signatures[first] is not None
            and active_signatures[second] is not None
        ):
            left = active_signatures[first]
            right = active_signatures[second]
            assert left is not None and right is not None
            signature = tuple(a + b for a, b in zip(left, right, strict=True))
        if signature == target_signature:
            candidate_slots.append(slot)

    active_slots = tuple(
        slot
        for slot, signature in enumerate(active_signatures)
        if signature == target_signature
    )
    return ProductChainSnapshot(
        active_slots=active_slots,
        candidate_slots=tuple(candidate_slots),
    )


def summarize_presence_history(presence: Sequence[bool]) -> ProductChainTrajectory:
    """Summarize snapshots indexed from initial state (step zero)."""

    values = tuple(bool(value) for value in presence)
    if not values:
        raise ValueError("presence history must contain at least the initial snapshot")
    first = next((index for index, value in enumerate(values) if value), None)
    acquisitions = int(values[0]) + sum(
        int(current and not previous)
        for previous, current in zip(values[:-1], values[1:], strict=True)
    )
    return ProductChainTrajectory(
        present_at_final_step=values[-1],
        ever_present=first is not None,
        first_present_step=first,
        presence_snapshot_count=sum(values),
        acquisition_episode_count=acquisitions,
        recurrence_count=max(0, acquisitions - 1),
    )


def _median_optional(values: Sequence[int | None]) -> float | None:
    present = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if present.size == 0:
        return None
    return float(np.median(present))


def summarize_arm_outcomes(
    arm_name: str,
    trials: Sequence[CompositionalDiscoveryTrialOutcome],
) -> CompositionalDiscoveryArmSummary:
    """Aggregate one arm descriptively; no field encodes acceptance."""

    selected = tuple(trial for trial in trials if trial.arm_name == arm_name)
    if not selected:
        raise ValueError(f"no outcomes supplied for arm {arm_name!r}")
    ratios = np.asarray(
        [trial.final_to_initial_mse_ratio for trial in selected], dtype=np.float64
    )
    if not bool(np.all(np.isfinite(ratios))):
        raise ValueError("all real-campaign MSE ratios must be finite")
    return CompositionalDiscoveryArmSummary(
        arm_name=arm_name,
        trial_count=len(selected),
        mean_initial_window_mse=float(
            np.mean([trial.initial_window_mse for trial in selected])
        ),
        mean_final_window_mse=float(
            np.mean([trial.final_window_mse for trial in selected])
        ),
        mean_final_to_initial_mse_ratio=float(np.mean(ratios)),
        median_final_to_initial_mse_ratio=float(np.median(ratios)),
        total_scheduled_curation_opportunities=sum(
            trial.scheduled_curation_opportunity_count for trial in selected
        ),
        total_active_root_replacements=sum(
            trial.active_root_replacement_count for trial in selected
        ),
        total_promotions=sum(trial.promotion_count for trial in selected),
        active_ever_present_trials=sum(
            trial.active_product_chain.ever_present for trial in selected
        ),
        active_final_present_trials=sum(
            trial.active_product_chain.present_at_final_step for trial in selected
        ),
        active_first_present_step_median=_median_optional(
            [trial.active_product_chain.first_present_step for trial in selected]
        ),
        active_total_recurrences=sum(
            trial.active_product_chain.recurrence_count for trial in selected
        ),
        candidate_ever_present_trials=sum(
            trial.candidate_product_chain.ever_present for trial in selected
        ),
        candidate_final_present_trials=sum(
            trial.candidate_product_chain.present_at_final_step for trial in selected
        ),
        candidate_first_present_step_median=_median_optional(
            [trial.candidate_product_chain.first_present_step for trial in selected]
        ),
        candidate_total_recurrences=sum(
            trial.candidate_product_chain.recurrence_count for trial in selected
        ),
    )


def _build_learner_and_state(
    arm: DevelopmentArm,
    *,
    feature_dim: int,
    key: Array,
) -> tuple[CompositionalFeatureLearner, CompositionalFeatureState]:
    common: dict[str, Any] = {
        "n_tasks": 1,
        "step_size_output": 0.05,
        "step_size_theta": 0.005,
        "utility_decay": 0.99,
        "max_depth": 3,
        "use_obgd": True,
        "obgd_kappa": 2.0,
    }
    if arm.name == STOCHASTIC_DISCOVERY:
        learner = CompositionalFeatureLearner(
            n_features=20,
            candidate_count=20,
            replacement_interval=20,
            min_feature_age=40,
            candidate_min_age=20,
            promotion_margin=1.05,
            promotion_blend=0.5,
            **common,
        )
        return learner, learner.init(feature_dim=feature_dim, key=key)

    learner = CompositionalFeatureLearner(
        n_features=arm.active_slots,
        candidate_count=0,
        replacement_interval=0,
        min_feature_age=100_000,
        **common,
    )
    state = learner.init(feature_dim=feature_dim, key=key)
    if arm.name == PREWIRED_DEPTH2_ORACLE:
        pair_slot = feature_dim
        target_slot = feature_dim + 1
        state = state.replace(  # type: ignore[attr-defined]
            ops=jnp.asarray(
                [*[OP_RAW] * feature_dim, OP_PRODUCT, OP_PRODUCT], dtype=jnp.int32
            ),
            parent_a=jnp.asarray(
                [*range(feature_dim), 0, pair_slot], dtype=jnp.int32
            ),
            parent_b=jnp.asarray(
                [*[-1] * feature_dim, 1, 2], dtype=jnp.int32
            ),
            depth=jnp.asarray([*[0] * feature_dim, 1, 2], dtype=jnp.int32),
        )
        if target_slot != arm.active_slots - 1:
            raise RuntimeError("oracle target slot must be the final active slot")
    elif arm.name != RAW_ONLY:
        raise ValueError(f"unknown development arm: {arm.name!r}")
    return learner, state


def _stationary_stream(
    plan: CompositionalDiscoveryDevelopmentPlan,
    root_seed_uint32: int,
) -> tuple[Array, Array]:
    # Stream keys are arm-independent by construction.  Use RAW_ONLY merely
    # to obtain the shared observation/noise branches.
    keys = _derive_keys(root_seed_uint32, RAW_ONLY)
    observations = jr.normal(
        keys.observation,
        (plan.num_steps, plan.feature_dim),
        dtype=jnp.float32,
    )
    noise = plan.noise_std * jr.normal(
        keys.noise,
        (plan.num_steps,),
        dtype=jnp.float32,
    )
    signal = observations[:, 0] * observations[:, 1] * observations[:, 2]
    return observations, (signal + noise)[:, None]


def _snapshot_from_descriptor_rows(
    *,
    active_ops: object,
    active_parent_a: object,
    active_parent_b: object,
    candidate_ops: object,
    candidate_parent_a: object,
    candidate_parent_b: object,
    plan: CompositionalDiscoveryDevelopmentPlan,
) -> ProductChainSnapshot:
    return detect_product_chain(
        active_ops=active_ops,
        active_parent_a=active_parent_a,
        active_parent_b=active_parent_b,
        candidate_ops=candidate_ops,
        candidate_parent_a=candidate_parent_a,
        candidate_parent_b=candidate_parent_b,
        feature_dim=plan.feature_dim,
        target_raw_indices=plan.target_raw_indices,
    )


def _run_trial(
    plan: CompositionalDiscoveryDevelopmentPlan,
    seed: DevelopmentSeed,
    arm: DevelopmentArm,
    observations: Array,
    targets: Array,
) -> CompositionalDiscoveryTrialOutcome:
    keys = _derive_keys(seed.root_seed_uint32, arm.name)
    learner, initial_state = _build_learner_and_state(
        arm,
        feature_dim=plan.feature_dim,
        key=keys.learner,
    )
    initial_snapshot = _snapshot_from_descriptor_rows(
        active_ops=initial_state.ops,
        active_parent_a=initial_state.parent_a,
        active_parent_b=initial_state.parent_b,
        candidate_ops=initial_state.candidate_ops,
        candidate_parent_a=initial_state.candidate_parent_a,
        candidate_parent_b=initial_state.candidate_parent_b,
        plan=plan,
    )

    def step_fn(
        state: CompositionalFeatureState,
        sample: tuple[Array, Array],
    ) -> tuple[CompositionalFeatureState, tuple[Array, ...]]:
        observation, target = sample
        result = learner.update(state, observation, target)
        next_state = result.state
        # Consume only established update-result/state fields.  The lifecycle
        # trace is intentionally outside this descriptive lane's contract.
        trace_row = (
            result.metrics[0],
            result.replaced_slot,
            result.promoted_candidate,
            next_state.ops,
            next_state.parent_a,
            next_state.parent_b,
            next_state.candidate_ops,
            next_state.candidate_parent_a,
            next_state.candidate_parent_b,
        )
        return next_state, trace_row

    _, trace = jax.lax.scan(step_fn, initial_state, (observations, targets))
    (
        mse_values,
        replaced_slots,
        promoted_candidates,
        active_ops,
        active_parent_a,
        active_parent_b,
        candidate_ops,
        candidate_parent_a,
        candidate_parent_b,
    ) = (np.asarray(value) for value in trace)

    active_presence = [initial_snapshot.active_present]
    candidate_presence = [initial_snapshot.candidate_present]
    for step in range(plan.num_steps):
        snapshot = _snapshot_from_descriptor_rows(
            active_ops=active_ops[step],
            active_parent_a=active_parent_a[step],
            active_parent_b=active_parent_b[step],
            candidate_ops=candidate_ops[step],
            candidate_parent_a=candidate_parent_a[step],
            candidate_parent_b=candidate_parent_b[step],
            plan=plan,
        )
        active_presence.append(snapshot.active_present)
        candidate_presence.append(snapshot.candidate_present)

    initial_mse = float(np.mean(mse_values[: plan.window], dtype=np.float64))
    final_mse = float(np.mean(mse_values[-plan.window :], dtype=np.float64))
    if not math.isfinite(initial_mse) or initial_mse <= 0.0:
        raise RuntimeError("initial-window MSE must be finite and positive")
    if not math.isfinite(final_mse) or final_mse < 0.0:
        raise RuntimeError("final-window MSE must be finite and nonnegative")
    ratio = final_mse / initial_mse
    return CompositionalDiscoveryTrialOutcome(
        status=COMPOSITIONAL_DISCOVERY_DEVELOPMENT_STATUS,
        development_only=True,
        root_seed_uint32=seed.root_seed_uint32,
        arm_name=arm.name,
        num_steps=plan.num_steps,
        window=plan.window,
        initial_window_mse=initial_mse,
        final_window_mse=final_mse,
        final_to_initial_mse_ratio=ratio,
        scheduled_curation_opportunity_count=(
            plan.num_steps // arm.curation_interval if arm.curation_interval > 0 else 0
        ),
        active_root_replacement_count=int(np.count_nonzero(replaced_slots >= 0)),
        promotion_count=int(np.count_nonzero(promoted_candidates >= 0)),
        active_product_chain=summarize_presence_history(active_presence),
        candidate_product_chain=summarize_presence_history(candidate_presence),
    )


def run_development_campaign(
    plan: CompositionalDiscoveryDevelopmentPlan | None = None,
) -> CompositionalDiscoveryCampaignResult:
    """Explicitly execute the nonpromoting campaign without artifact writes."""

    selected_plan = validate_development_plan(
        build_development_plan() if plan is None else plan
    )

    outcomes: list[CompositionalDiscoveryTrialOutcome] = []
    logger.info(
        "starting nonpromoting compositional discovery development campaign: "
        "%d seeds, %d arms, %d updates",
        len(selected_plan.seeds),
        len(selected_plan.arms),
        selected_plan.resource_accounting.total_learner_updates,
    )
    for seed in selected_plan.seeds:
        observations, targets = _stationary_stream(selected_plan, seed.root_seed_uint32)
        for arm in selected_plan.arms:
            logger.info(
                "running development seed=%d arm=%s",
                seed.root_seed_uint32,
                arm.name,
            )
            outcomes.append(
                _run_trial(selected_plan, seed, arm, observations, targets)
            )

    trial_tuple = tuple(outcomes)
    summaries = tuple(
        summarize_arm_outcomes(arm.name, trial_tuple) for arm in selected_plan.arms
    )
    logger.info("finished nonpromoting compositional discovery development campaign")
    return CompositionalDiscoveryCampaignResult(
        plan=selected_plan,
        trials=trial_tuple,
        arm_summaries=summaries,
        artifacts_written=0,
        scientific_evidence_authorized=False,
        promotion_authorized=False,
    )


__all__ = [
    "ARM_DOMAINS",
    "COMPOSITIONAL_DISCOVERY_DEVELOPMENT_STATUS",
    "DEFAULT_DEVELOPMENT_SEEDS",
    "DEVELOPMENT_ARMS",
    "LEARNER_DOMAIN",
    "NOISE_DOMAIN",
    "OBSERVATION_DOMAIN",
    "PREWIRED_DEPTH2_ORACLE",
    "PRODUCT_SIGNATURE_CLIPPING_CAVEAT",
    "RAW_ONLY",
    "STOCHASTIC_DISCOVERY",
    "STREAM_DOMAIN",
    "STRUCTURAL_RECURRENCE_DEFINITION",
    "THREEFRY_IMPLEMENTATION",
    "CompositionalDiscoveryArmSummary",
    "CompositionalDiscoveryCampaignResult",
    "CompositionalDiscoveryDevelopmentPlan",
    "CompositionalDiscoveryTrialOutcome",
    "DevelopmentArm",
    "DevelopmentResourceAccounting",
    "DevelopmentSeed",
    "DevelopmentTrialKeys",
    "ProductChainSnapshot",
    "ProductChainTrajectory",
    "build_development_plan",
    "derive_trial_key_manifest",
    "detect_product_chain",
    "run_development_campaign",
    "summarize_arm_outcomes",
    "summarize_presence_history",
    "validate_development_plan",
]
