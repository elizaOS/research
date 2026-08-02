# mypy: disable-error-code="call-arg"
"""Continuing ternary hidden-regime signaling world for L1 development.

A helper privately observes one uniform ternary cue and sends one ternary
message.  A beneficiary observes only the delivered message and chooses one
ternary action.  An evaluator-only hidden regime selects a target permutation;
both learners receive only their local input, local action, and common binary
reward.  Regime, target, segment index, and schedule never enter ordinary
observations.

The finite evaluator schedule either repeats or holds its last regime forever,
so the world never terminates or resets.  A bounded schedule cursor continues
after the diagnostic global int32 step counter saturates.  Cue and channel
randomness use named independent streams.

This is an L1 development substrate (``RESEARCH_STATUS.md`` evidence levels:
L1 = a component learns in a controlled toy problem) with no acceptance
threshold, artifact, seed namespace, or performance claim.  Its three-symbol
channel carries only log2(3) bits per step, and a learner with three durable
modules cannot retain an unbounded sequence of incompatible permutations.

The intended learners are the fixed-capacity slot-signaling agents
(:mod:`alberta_framework.core.slot_signaling_agent`), which operate in
fixed-length slot-residency windows called *leases*; the structural
manifests below are phased against that lease length.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal, cast

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, PRNGKeyArray

N_HIDDEN_REGIME_SYMBOLS = 3

DIRECT_TERNARY_CHANNEL: Literal["direct"] = "direct"
CONSTANT_ZERO_TERNARY_CHANNEL: Literal["constant_0"] = "constant_0"
CONSTANT_ONE_TERNARY_CHANNEL: Literal["constant_1"] = "constant_1"
CONSTANT_TWO_TERNARY_CHANNEL: Literal["constant_2"] = "constant_2"
SHUFFLED_TERNARY_CHANNEL: Literal["shuffled"] = "shuffled"

type HiddenRegimeChannel = Literal[
    "direct",
    "constant_0",
    "constant_1",
    "constant_2",
    "shuffled",
]

HIDDEN_REGIME_CHANNELS: tuple[HiddenRegimeChannel, ...] = (
    DIRECT_TERNARY_CHANNEL,
    CONSTANT_ZERO_TERNARY_CHANNEL,
    CONSTANT_ONE_TERNARY_CHANNEL,
    CONSTANT_TWO_TERNARY_CHANNEL,
    SHUFFLED_TERNARY_CHANNEL,
)

# Stable named tags are part of the world contract.  New random consumers must
# receive a new tag rather than positionally splitting either existing stream.
_CUE_RNG_TAG = 0x48524355  # ASCII "HRCU"
_CHANNEL_RNG_TAG = 0x48524348  # ASCII "HRCH"

# Default development schedule.  The regime order is designed around the
# learners' three durable slots: A and B recur throughout (retention under
# interleaving); C-old is used early and then permanently superseded by the
# incompatible C-new (stale-convention replacement); D-transient appears
# exactly once, for a single 16-step segment, as a distraction that should
# not evict a durable convention.  Manifest validation freezes exactly this
# structure (bank-fill prefix, two C-new before D and one after, 16-step D).
DEFAULT_REGIME_PERMUTATIONS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),  # A
    (1, 2, 0),  # B
    (1, 0, 2),  # C-old
    (2, 1, 0),  # C-new
    (0, 2, 1),  # D-transient
)
DEFAULT_SEGMENT_REGIMES: tuple[int, ...] = (
    0,
    1,
    0,
    2,
    1,
    0,
    2,
    0,
    1,
    3,
    0,
    1,
    3,
    4,
    0,
    1,
    3,
)
DEFAULT_SEGMENT_LENGTHS: tuple[int, ...] = (
    64,
    72,
    56,
    64,
    72,
    64,
    56,
    72,
    64,
    64,
    56,
    72,
    64,
    16,
    72,
    56,
    64,
)

# A "lease" is the fixed slot-residency window of the slot-signaling learners
# (:class:`~alberta_framework.core.slot_signaling_agent.SlotSignalingConfig`
# ``lease_length``); the hidden-regime development calibration pins it to 16
# steps (see ``evaluation/hidden_regime_signaling_evidence.py``).  Manifest
# change-point residues are measured modulo this window so that segment
# boundaries do not systematically align with learner lease boundaries.
#
# The structural-generalization candidates preserve the calibrated development
# exposure exactly: every ordinary regime segment is expanded by sixteen while
# the deliberately transient D occurrence remains sixteen steps (exactly one
# lease).  Perturbations below are applied after this expansion; scaling a
# shorter jittered schedule would erase all of its lease-phase variation.
# The perturbations are load-bearing: every default segment length is a
# multiple of eight, so an unperturbed schedule only ever places a change
# point at lease phase 0 or 8.  Manifest validation therefore requires the
# perturbed boundaries to cover at least eight distinct nonzero phases.
STRUCTURAL_GENERALIZATION_LEASE_STEPS = 16
STRUCTURAL_GENERALIZATION_BASELINE_SEGMENT_LENGTHS: tuple[int, ...] = tuple(
    length if regime == 4 else length * STRUCTURAL_GENERALIZATION_LEASE_STEPS
    for length, regime in zip(
        DEFAULT_SEGMENT_LENGTHS,
        DEFAULT_SEGMENT_REGIMES,
        strict=True,
    )
)

_STRUCTURAL_BASELINE_LENGTHS_BY_REGIME: tuple[tuple[int, ...], ...] = tuple(
    tuple(
        length
        for length, actual_regime in zip(
            STRUCTURAL_GENERALIZATION_BASELINE_SEGMENT_LENGTHS,
            DEFAULT_SEGMENT_REGIMES,
            strict=True,
        )
        if actual_regime == regime
    )
    for regime in range(len(DEFAULT_REGIME_PERMUTATIONS))
)
# Frozen as literals rather than recomputed: these are the per-regime
# occurrence counts and step exposures of the default schedule under the x16
# expansion (total = (1048 - 16) * 16 + 16 = 16,528).  Every candidate
# manifest must reproduce them exactly, so any drift in the derivation above
# fails closed instead of silently redefining the protected exposure budget.
_STRUCTURAL_EXPECTED_OCCURRENCES = (6, 5, 2, 3, 1)
_STRUCTURAL_EXPECTED_EXPOSURES = (6144, 5376, 1920, 3072, 16)
_STRUCTURAL_TOTAL_STEPS = 16_528
_IDENTITY_SYMBOL_RELABELING = tuple(range(N_HIDDEN_REGIME_SYMBOLS))

PROTECTED_CANDIDATE_PARTITION: Literal["protected_candidate"] = "protected_candidate"
CALIBRATION_ONLY_PARTITION: Literal["calibration_only"] = "calibration_only"
type HiddenRegimeManifestPartition = Literal["protected_candidate", "calibration_only"]

PROTECTED_CANDIDATE_PERFORMANCE_STATUS = "performance_unexecuted_protected_candidate"
CALIBRATION_ONLY_PERFORMANCE_STATUS = "available_for_nonpromoting_calibration"
PROTECTED_CANDIDATE_LEARNER_OUTCOMES_EXECUTED = False


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenRegimeScheduleManifest:
    """Immutable evaluator-only structural-generalization world definition.

    Each manifest changes schedule order, integer boundary phase, and visible
    cue/action labels while holding regime occurrence counts and total exposure
    fixed.  It contains no RNG seed or acceptance threshold and is not itself a
    preregistration or scientific result.
    """

    name: str
    use_partition: HiddenRegimeManifestPartition
    segment_lengths: tuple[int, ...]
    segment_regimes: tuple[int, ...]
    length_perturbations_by_regime: tuple[tuple[int, ...], ...]
    cue_symbol_relabeling: tuple[int, int, int]
    action_symbol_relabeling: tuple[int, int, int]

    def __post_init__(self) -> None:
        if type(self.name) is not str:
            raise TypeError("manifest name must be a string")
        if self.use_partition == PROTECTED_CANDIDATE_PARTITION:
            name_family = "structural"
        elif self.use_partition == CALIBRATION_ONLY_PARTITION:
            name_family = "calibration"
        else:
            raise ValueError("manifest use_partition must be explicit and recognized")
        name_pattern = rf"hidden-regime-{name_family}-[a-z0-9]+-v[1-9][0-9]*"
        if re.fullmatch(name_pattern, self.name) is None:
            raise ValueError("manifest name must match its explicit use partition")
        if not isinstance(self.segment_lengths, tuple) or len(self.segment_lengths) != len(
            DEFAULT_SEGMENT_LENGTHS
        ):
            raise ValueError("manifest segment_lengths must contain exactly 17 entries")
        if any(type(length) is not int or length < 1 for length in self.segment_lengths):
            raise ValueError("manifest segment_lengths must contain positive integers")
        if not isinstance(self.segment_regimes, tuple) or len(self.segment_regimes) != len(
            DEFAULT_SEGMENT_REGIMES
        ):
            raise ValueError("manifest segment_regimes must contain exactly 17 entries")
        if any(
            type(regime) is not int or regime < 0 or regime >= len(DEFAULT_REGIME_PERMUTATIONS)
            for regime in self.segment_regimes
        ):
            raise ValueError("manifest segment_regimes contains an invalid regime")
        if not isinstance(self.length_perturbations_by_regime, tuple) or len(
            self.length_perturbations_by_regime
        ) != len(DEFAULT_REGIME_PERMUTATIONS):
            raise ValueError("length perturbations must contain one tuple per regime")
        for regime, perturbations in enumerate(self.length_perturbations_by_regime):
            if not isinstance(perturbations, tuple) or len(perturbations) != len(
                _STRUCTURAL_BASELINE_LENGTHS_BY_REGIME[regime]
            ):
                raise ValueError("length perturbations must match baseline occurrence counts")
            if any(type(delta) is not int for delta in perturbations):
                raise ValueError("length perturbations must contain strict integers")
            if sum(perturbations) != 0:
                raise ValueError("length perturbations must sum to zero within every regime")

        self._validate_symbol_relabeling("cue_symbol_relabeling", self.cue_symbol_relabeling)
        self._validate_symbol_relabeling(
            "action_symbol_relabeling",
            self.action_symbol_relabeling,
        )
        if (
            self.cue_symbol_relabeling == _IDENTITY_SYMBOL_RELABELING
            and self.action_symbol_relabeling == _IDENTITY_SYMBOL_RELABELING
        ):
            raise ValueError("schedule manifests require a nonidentity symbol relabeling")

        occurrences = tuple(
            self.segment_regimes.count(regime) for regime in range(len(DEFAULT_REGIME_PERMUTATIONS))
        )
        if occurrences != _STRUCTURAL_EXPECTED_OCCURRENCES:
            raise ValueError("manifest must preserve exact per-regime occurrence counts")
        if sum(self.segment_lengths) != _STRUCTURAL_TOTAL_STEPS:
            raise ValueError("manifest must preserve the 16,528-step baseline total")
        exposures = tuple(
            sum(
                length
                for length, actual_regime in zip(
                    self.segment_lengths,
                    self.segment_regimes,
                    strict=True,
                )
                if actual_regime == regime
            )
            for regime in range(len(DEFAULT_REGIME_PERMUTATIONS))
        )
        if exposures != _STRUCTURAL_EXPECTED_EXPOSURES:
            raise ValueError("manifest must preserve exact per-regime exposure")
        if self.segment_lengths != self._reconstructed_segment_lengths():
            raise ValueError("segment lengths must equal baseline plus declared perturbations")

        transient_index = self.segment_regimes.index(4)
        if self.segment_lengths[transient_index] != STRUCTURAL_GENERALIZATION_LEASE_STEPS:
            raise ValueError("the sole transient D occurrence must remain exactly 16 steps")
        c_new_before_transient = self.segment_regimes[:transient_index].count(3)
        c_new_after_transient = self.segment_regimes[transient_index + 1 :].count(3)
        if c_new_before_transient != 2 or c_new_after_transient != 1:
            raise ValueError(
                "manifest must place exactly two C-new occurrences before D and exactly one after D"
            )
        first_c_new = self.segment_regimes.index(3)
        bank_fill_prefix = self.segment_regimes[:first_c_new]
        if any(bank_fill_prefix.count(regime) < 2 for regime in (0, 1, 2)):
            raise ValueError("manifest must provide a two-occurrence A/B/C-old bank-fill prefix")
        if not {0, 1, 3}.issubset(self.segment_regimes[transient_index + 1 :]):
            raise ValueError("manifest must contain post-transient A/B/C-new recurrence")
        residues = self.change_point_residues
        if any(residue == 0 for residue in residues) or len(set(residues)) < 8:
            raise ValueError(
                "all change points must cover at least eight distinct nonzero lease phases"
            )

    @staticmethod
    def _validate_symbol_relabeling(name: str, relabeling: object) -> None:
        if (
            not isinstance(relabeling, tuple)
            or len(relabeling) != N_HIDDEN_REGIME_SYMBOLS
            or any(type(symbol) is not int for symbol in relabeling)
            or tuple(sorted(relabeling)) != _IDENTITY_SYMBOL_RELABELING
        ):
            raise ValueError(f"{name} must contain exactly 0, 1, 2")

    def _reconstructed_segment_lengths(self) -> tuple[int, ...]:
        cursors = [0] * len(DEFAULT_REGIME_PERMUTATIONS)
        lengths: list[int] = []
        for regime in self.segment_regimes:
            ordinal = cursors[regime]
            lengths.append(
                _STRUCTURAL_BASELINE_LENGTHS_BY_REGIME[regime][ordinal]
                + self.length_perturbations_by_regime[regime][ordinal]
            )
            cursors[regime] += 1
        return tuple(lengths)

    @property
    def change_point_residues(self) -> tuple[int, ...]:
        """Return all internal boundary phases modulo the sixteen-step lease."""

        elapsed = 0
        residues: list[int] = []
        for length in self.segment_lengths[:-1]:
            elapsed += length
            residues.append(elapsed % STRUCTURAL_GENERALIZATION_LEASE_STEPS)
        return tuple(residues)

    @property
    def regime_permutations(self) -> tuple[tuple[int, int, int], ...]:
        """Return canonical targets conjugated into visible cue/action labels."""

        inverse_cue = [0] * N_HIDDEN_REGIME_SYMBOLS
        for canonical_cue, visible_cue in enumerate(self.cue_symbol_relabeling):
            inverse_cue[visible_cue] = canonical_cue
        permutations = tuple(
            tuple(
                self.action_symbol_relabeling[canonical_permutation[inverse_cue[visible_cue]]]
                for visible_cue in range(N_HIDDEN_REGIME_SYMBOLS)
            )
            for canonical_permutation in DEFAULT_REGIME_PERMUTATIONS
        )
        return cast(tuple[tuple[int, int, int], ...], permutations)

    def to_world_config(self, *, repeat_schedule: bool = False) -> HiddenRegimeWorldConfig:
        """Build a world config without retaining or exposing the manifest name."""

        if type(repeat_schedule) is not bool:
            raise ValueError("repeat_schedule must be boolean")
        return HiddenRegimeWorldConfig(
            segment_lengths=self.segment_lengths,
            segment_regimes=self.segment_regimes,
            regime_permutations=self.regime_permutations,
            repeat_schedule=repeat_schedule,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic manifest content for a future preregistration."""

        protected_candidate = self.use_partition == PROTECTED_CANDIDATE_PARTITION
        return {
            "schema": (
                "hidden-regime-structural-manifest-v1"
                if protected_candidate
                else "hidden-regime-calibration-manifest-v1"
            ),
            "name": self.name,
            "use_partition": self.use_partition,
            "segment_lengths": list(self.segment_lengths),
            "segment_regimes": list(self.segment_regimes),
            "length_perturbations_by_regime": [
                list(row) for row in self.length_perturbations_by_regime
            ],
            "cue_symbol_relabeling": list(self.cue_symbol_relabeling),
            "action_symbol_relabeling": list(self.action_symbol_relabeling),
            "regime_permutations": [list(row) for row in self.regime_permutations],
            "change_point_residues": list(self.change_point_residues),
            "total_schedule_steps": sum(self.segment_lengths),
            "candidate_only": protected_candidate,
            "protected_candidate": protected_candidate,
            "calibration_use_allowed": not protected_candidate,
            "scientific_promotion_allowed": False,
            "learner_outcome_status": (
                PROTECTED_CANDIDATE_PERFORMANCE_STATUS
                if protected_candidate
                else CALIBRATION_ONLY_PERFORMANCE_STATUS
            ),
            "learner_outcomes_executed": False if protected_candidate else None,
            "protected_seed_namespace": None,
        }


@dataclasses.dataclass(frozen=True)
class HiddenRegimeWorldConfig:
    """Evaluator-owned schedule and target permutations.

    Segment identities and boundaries are never emitted by :meth:`observe`.
    ``repeat_schedule=False`` makes the final regime persist indefinitely;
    ``True`` repeats the complete evaluator schedule without resetting state.
    """

    segment_lengths: tuple[int, ...] = DEFAULT_SEGMENT_LENGTHS
    segment_regimes: tuple[int, ...] = DEFAULT_SEGMENT_REGIMES
    regime_permutations: tuple[tuple[int, int, int], ...] = DEFAULT_REGIME_PERMUTATIONS
    repeat_schedule: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.segment_lengths, tuple) or not self.segment_lengths:
            raise ValueError("segment_lengths must be a nonempty tuple")
        if any(type(length) is not int or length < 1 for length in self.segment_lengths):
            raise ValueError("segment_lengths must contain positive integers")
        if sum(self.segment_lengths) > np.iinfo(np.int32).max:
            raise ValueError("total schedule length must fit int32")
        if not isinstance(self.segment_regimes, tuple) or len(self.segment_regimes) != len(
            self.segment_lengths
        ):
            raise ValueError("segment_regimes must match segment_lengths")
        if not isinstance(self.regime_permutations, tuple) or not self.regime_permutations:
            raise ValueError("regime_permutations must be a nonempty tuple")
        for permutation in self.regime_permutations:
            if (
                not isinstance(permutation, tuple)
                or len(permutation) != N_HIDDEN_REGIME_SYMBOLS
                or any(type(symbol) is not int for symbol in permutation)
                or tuple(sorted(permutation)) != tuple(range(N_HIDDEN_REGIME_SYMBOLS))
            ):
                raise ValueError("every regime permutation must contain exactly 0, 1, 2")
        if any(
            type(regime) is not int or regime < 0 or regime >= len(self.regime_permutations)
            for regime in self.segment_regimes
        ):
            raise ValueError("segment_regimes contains an invalid regime index")
        if type(self.repeat_schedule) is not bool:
            raise ValueError("repeat_schedule must be boolean")

    @property
    def total_schedule_steps(self) -> int:
        """Return the finite evaluator schedule length before repeat/hold."""

        return sum(self.segment_lengths)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible development-only world definition."""

        return {
            "segment_lengths": list(self.segment_lengths),
            "segment_regimes": list(self.segment_regimes),
            "regime_permutations": [list(row) for row in self.regime_permutations],
            "repeat_schedule": self.repeat_schedule,
            "total_schedule_steps": self.total_schedule_steps,
            "development_only": True,
            "scientific_promotion_allowed": False,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenRegimeManifestUseLedgerEntry:
    """Immutable source-level authorization and performance-use status."""

    manifest_name: str
    use_partition: HiddenRegimeManifestPartition
    learner_outcome_status: str
    learner_outcomes_executed: bool | None
    calibration_use_allowed: bool
    protected_evaluation_candidate: bool
    scientific_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        if type(self.manifest_name) is not str:
            raise TypeError("manifest ledger name must be a string")
        expected: tuple[str, bool | None, bool, bool]
        if self.use_partition == PROTECTED_CANDIDATE_PARTITION:
            expected = (
                PROTECTED_CANDIDATE_PERFORMANCE_STATUS,
                False,
                False,
                True,
            )
            name_family = "structural"
        elif self.use_partition == CALIBRATION_ONLY_PARTITION:
            expected = (
                CALIBRATION_ONLY_PERFORMANCE_STATUS,
                None,
                True,
                False,
            )
            name_family = "calibration"
        else:
            raise ValueError("manifest ledger use_partition is unsupported")
        actual = (
            self.learner_outcome_status,
            self.learner_outcomes_executed,
            self.calibration_use_allowed,
            self.protected_evaluation_candidate,
        )
        if actual != expected:
            raise ValueError("manifest ledger status conflicts with its use partition")
        pattern = rf"hidden-regime-{name_family}-[a-z0-9]+-v[1-9][0-9]*"
        if re.fullmatch(pattern, self.manifest_name) is None:
            raise ValueError("manifest ledger name conflicts with its use partition")
        if self.scientific_promotion_allowed is not False:
            raise ValueError("source-level manifest entries cannot authorize promotion")

    def to_dict(self) -> dict[str, object]:
        """Return strict role metadata without a seed or learner outcome."""

        return dataclasses.asdict(self)


HIDDEN_REGIME_STRUCTURAL_A_V1 = HiddenRegimeScheduleManifest(
    name="hidden-regime-structural-a-v1",
    use_partition=PROTECTED_CANDIDATE_PARTITION,
    segment_regimes=(0, 1, 2, 0, 1, 2, 0, 1, 3, 0, 3, 1, 0, 4, 1, 3, 0),
    segment_lengths=(
        1026,
        1155,
        1034,
        905,
        1141,
        886,
        1032,
        1031,
        1028,
        1149,
        1033,
        1158,
        890,
        16,
        891,
        1011,
        1142,
    ),
    length_perturbations_by_regime=(
        (2, 9, 8, -3, -6, -10),
        (3, -11, 7, 6, -5),
        (10, -10),
        (4, 9, -13),
        (0,),
    ),
    cue_symbol_relabeling=(1, 2, 0),
    action_symbol_relabeling=(2, 0, 1),
)

HIDDEN_REGIME_STRUCTURAL_B_V1 = HiddenRegimeScheduleManifest(
    name="hidden-regime-structural-b-v1",
    use_partition=PROTECTED_CANDIDATE_PARTITION,
    segment_regimes=(1, 0, 2, 1, 0, 2, 1, 0, 3, 1, 3, 0, 4, 3, 0, 1, 0),
    segment_lengths=(
        1153,
        1032,
        1029,
        1145,
        888,
        891,
        1033,
        1026,
        1020,
        1163,
        1023,
        1155,
        16,
        1029,
        896,
        882,
        1147,
    ),
    length_perturbations_by_regime=(
        (8, -8, 2, 3, 0, -5),
        (1, -7, 9, 11, -14),
        (5, -5),
        (-4, -1, 5),
        (0,),
    ),
    cue_symbol_relabeling=(2, 0, 1),
    action_symbol_relabeling=(1, 0, 2),
)

HIDDEN_REGIME_STRUCTURAL_C_V1 = HiddenRegimeScheduleManifest(
    name="hidden-regime-structural-c-v1",
    use_partition=PROTECTED_CANDIDATE_PARTITION,
    segment_regimes=(0, 2, 1, 0, 2, 1, 0, 3, 1, 0, 3, 1, 4, 0, 1, 3, 0),
    segment_lengths=(
        1028,
        1029,
        1149,
        889,
        891,
        1144,
        1025,
        1034,
        1031,
        1156,
        1022,
        1144,
        16,
        899,
        908,
        1016,
        1147,
    ),
    length_perturbations_by_regime=(
        (4, -7, 1, 4, 3, -5),
        (-3, -8, 7, -8, 12),
        (5, -5),
        (10, -2, -8),
        (0,),
    ),
    cue_symbol_relabeling=(0, 2, 1),
    action_symbol_relabeling=(2, 1, 0),
)

HIDDEN_REGIME_CALIBRATION_A_V1 = HiddenRegimeScheduleManifest(
    name="hidden-regime-calibration-a-v1",
    use_partition=CALIBRATION_ONLY_PARTITION,
    segment_regimes=(2, 0, 1, 2, 1, 0, 0, 1, 3, 0, 1, 3, 0, 4, 3, 1, 0),
    segment_lengths=(
        1021,
        1019,
        1166,
        899,
        1139,
        881,
        1018,
        1021,
        1028,
        1161,
        1146,
        1035,
        902,
        16,
        1009,
        904,
        1163,
    ),
    length_perturbations_by_regime=(
        (-5, -15, -6, 9, 6, 11),
        (14, -13, -3, -6, 8),
        (-3, 3),
        (4, 11, -15),
        (0,),
    ),
    cue_symbol_relabeling=(0, 1, 2),
    action_symbol_relabeling=(1, 2, 0),
)

HIDDEN_REGIME_CALIBRATION_B_V1 = HiddenRegimeScheduleManifest(
    name="hidden-regime-calibration-b-v1",
    use_partition=CALIBRATION_ONLY_PARTITION,
    segment_regimes=(1, 2, 0, 0, 1, 2, 1, 0, 3, 1, 0, 3, 4, 0, 1, 3, 0),
    segment_lengths=(
        1163,
        1030,
        1033,
        898,
        1142,
        890,
        1026,
        1018,
        1036,
        1153,
        1137,
        1021,
        16,
        908,
        892,
        1015,
        1150,
    ),
    length_perturbations_by_regime=(
        (9, 2, -6, -15, 12, -2),
        (11, -10, 2, 1, -4),
        (6, -6),
        (12, -3, -9),
        (0,),
    ),
    cue_symbol_relabeling=(1, 0, 2),
    action_symbol_relabeling=(0, 1, 2),
)

HIDDEN_REGIME_CALIBRATION_C_V1 = HiddenRegimeScheduleManifest(
    name="hidden-regime-calibration-c-v1",
    use_partition=CALIBRATION_ONLY_PARTITION,
    segment_regimes=(2, 1, 0, 1, 2, 0, 1, 0, 3, 0, 3, 1, 0, 4, 0, 3, 1),
    segment_lengths=(
        1023,
        1155,
        1009,
        1167,
        897,
        898,
        1027,
        1030,
        1030,
        1151,
        1027,
        1145,
        908,
        16,
        1148,
        1015,
        882,
    ),
    length_perturbations_by_regime=(
        (-15, 2, 6, -1, 12, -4),
        (3, 15, 3, -7, -14),
        (-1, 1),
        (6, 3, -9),
        (0,),
    ),
    cue_symbol_relabeling=(2, 1, 0),
    action_symbol_relabeling=(2, 0, 1),
)

HIDDEN_REGIME_STRUCTURAL_MANIFESTS: Mapping[str, HiddenRegimeScheduleManifest] = MappingProxyType(
    {
        manifest.name: manifest
        for manifest in (
            HIDDEN_REGIME_STRUCTURAL_A_V1,
            HIDDEN_REGIME_STRUCTURAL_B_V1,
            HIDDEN_REGIME_STRUCTURAL_C_V1,
        )
    }
)

HIDDEN_REGIME_CALIBRATION_MANIFESTS: Mapping[str, HiddenRegimeScheduleManifest] = MappingProxyType(
    {
        manifest.name: manifest
        for manifest in (
            HIDDEN_REGIME_CALIBRATION_A_V1,
            HIDDEN_REGIME_CALIBRATION_B_V1,
            HIDDEN_REGIME_CALIBRATION_C_V1,
        )
    }
)

HIDDEN_REGIME_MANIFEST_USE_LEDGER: Mapping[str, HiddenRegimeManifestUseLedgerEntry] = (
    MappingProxyType(
        {
            **{
                name: HiddenRegimeManifestUseLedgerEntry(
                    manifest_name=name,
                    use_partition=PROTECTED_CANDIDATE_PARTITION,
                    learner_outcome_status=PROTECTED_CANDIDATE_PERFORMANCE_STATUS,
                    learner_outcomes_executed=(PROTECTED_CANDIDATE_LEARNER_OUTCOMES_EXECUTED),
                    calibration_use_allowed=False,
                    protected_evaluation_candidate=True,
                )
                for name in HIDDEN_REGIME_STRUCTURAL_MANIFESTS
            },
            **{
                name: HiddenRegimeManifestUseLedgerEntry(
                    manifest_name=name,
                    use_partition=CALIBRATION_ONLY_PARTITION,
                    learner_outcome_status=CALIBRATION_ONLY_PERFORMANCE_STATUS,
                    learner_outcomes_executed=None,
                    calibration_use_allowed=True,
                    protected_evaluation_candidate=False,
                )
                for name in HIDDEN_REGIME_CALIBRATION_MANIFESTS
            },
        }
    )
)


def hidden_regime_structural_manifest(name: str) -> HiddenRegimeScheduleManifest:
    """Resolve one exact named candidate, refusing aliases and unknown names."""

    if type(name) is not str:
        raise TypeError("manifest name must be a string")
    try:
        return HIDDEN_REGIME_STRUCTURAL_MANIFESTS[name]
    except KeyError as error:
        raise ValueError(f"unknown hidden-regime structural manifest: {name!r}") from error


def hidden_regime_calibration_manifest(name: str) -> HiddenRegimeScheduleManifest:
    """Resolve one nonpromoting calibration manifest, refusing protected names."""

    if type(name) is not str:
        raise TypeError("calibration manifest name must be a string")
    try:
        return HIDDEN_REGIME_CALIBRATION_MANIFESTS[name]
    except KeyError as error:
        raise ValueError(f"unknown hidden-regime calibration manifest: {name!r}") from error


def hidden_regime_manifest_use(name: str) -> HiddenRegimeManifestUseLedgerEntry:
    """Resolve immutable use authorization without resolving a world definition."""

    if type(name) is not str:
        raise TypeError("manifest ledger name must be a string")
    try:
        return HIDDEN_REGIME_MANIFEST_USE_LEDGER[name]
    except KeyError as error:
        raise ValueError(f"unknown hidden-regime manifest ledger entry: {name!r}") from error


def hidden_regime_world_config_for_manifest(
    name: str,
    *,
    repeat_schedule: bool = False,
) -> HiddenRegimeWorldConfig:
    """Build one named structural world without placing its name in learner state."""

    return hidden_regime_structural_manifest(name).to_world_config(repeat_schedule=repeat_schedule)


def hidden_regime_calibration_world_config(
    name: str,
    *,
    repeat_schedule: bool = False,
) -> HiddenRegimeWorldConfig:
    """Build a calibration-only world while refusing every protected name."""

    return hidden_regime_calibration_manifest(name).to_world_config(repeat_schedule=repeat_schedule)


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenRegimeRepeatingPhaseDriftMetadata:
    """Disclosure for a derived repeating schedule that is not its source manifest.

    Construction is outcome-free and grants no execution authority.  In
    particular, deriving metadata from a protected candidate does not change
    that candidate's use-ledger status or permit a learner run.
    """

    source_manifest_name: str
    source_use_partition: HiddenRegimeManifestPartition
    final_regime_id: int
    final_regime_extension_steps: int
    base_total_schedule_steps: int
    repeated_total_schedule_steps: int
    cycle_phase_drift: int
    phase_period_cycles: int
    cycle_start_residues: tuple[int, ...]
    base_exposure_by_regime: tuple[int, ...]
    repeated_exposure_by_regime: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.source_manifest_name) is not str:
            raise TypeError("phase-drift source manifest name must be a string")
        if self.source_use_partition == PROTECTED_CANDIDATE_PARTITION:
            name_family = "structural"
        elif self.source_use_partition == CALIBRATION_ONLY_PARTITION:
            name_family = "calibration"
        else:
            raise ValueError("phase-drift source partition is unsupported")
        pattern = rf"hidden-regime-{name_family}-[a-z0-9]+-v[1-9][0-9]*"
        if re.fullmatch(pattern, self.source_manifest_name) is None:
            raise ValueError("phase-drift source name conflicts with its partition")
        strict_integers = (
            self.final_regime_id,
            self.final_regime_extension_steps,
            self.base_total_schedule_steps,
            self.repeated_total_schedule_steps,
            self.cycle_phase_drift,
            self.phase_period_cycles,
        )
        if any(type(value) is not int for value in strict_integers):
            raise TypeError("phase-drift metadata counters must be strict integers")
        extension = self.final_regime_extension_steps
        if not 1 <= extension < STRUCTURAL_GENERALIZATION_LEASE_STEPS:
            raise ValueError("phase-drift extension must be between 1 and 15 steps")
        if not 0 <= self.final_regime_id < len(DEFAULT_REGIME_PERMUTATIONS):
            raise ValueError("phase-drift final regime is invalid")
        if self.repeated_total_schedule_steps != self.base_total_schedule_steps + extension:
            raise ValueError("phase-drift total does not include the declared extension")
        expected_drift = self.repeated_total_schedule_steps % STRUCTURAL_GENERALIZATION_LEASE_STEPS
        if self.cycle_phase_drift != expected_drift or expected_drift == 0:
            raise ValueError("phase-drift cycle total must shift the lease phase")
        expected_period = STRUCTURAL_GENERALIZATION_LEASE_STEPS // math.gcd(
            STRUCTURAL_GENERALIZATION_LEASE_STEPS,
            expected_drift,
        )
        expected_residues = tuple(
            (cycle * expected_drift) % STRUCTURAL_GENERALIZATION_LEASE_STEPS
            for cycle in range(expected_period)
        )
        if self.phase_period_cycles != expected_period:
            raise ValueError("phase-drift period is inconsistent")
        if self.cycle_start_residues != expected_residues:
            raise ValueError("phase-drift cycle-start residues are inconsistent")
        if (
            len(self.base_exposure_by_regime) != len(DEFAULT_REGIME_PERMUTATIONS)
            or len(self.repeated_exposure_by_regime) != len(DEFAULT_REGIME_PERMUTATIONS)
            or any(
                type(value) is not int
                for value in (*self.base_exposure_by_regime, *self.repeated_exposure_by_regime)
            )
        ):
            raise ValueError("phase-drift exposure vectors must contain strict regime totals")
        expected_exposure = tuple(
            base + (extension if regime == self.final_regime_id else 0)
            for regime, base in enumerate(self.base_exposure_by_regime)
        )
        if self.repeated_exposure_by_regime != expected_exposure:
            raise ValueError("phase-drift exposure does not disclose the final-regime extension")

    def to_dict(self) -> dict[str, object]:
        """Return explicit provenance for the changed repeating schedule."""

        return {
            "schema": "hidden-regime-repeating-phase-drift-metadata-v1",
            "source_manifest_name": self.source_manifest_name,
            "source_use_partition": self.source_use_partition,
            "final_regime_id": self.final_regime_id,
            "final_regime_extension_steps": self.final_regime_extension_steps,
            "base_total_schedule_steps": self.base_total_schedule_steps,
            "repeated_total_schedule_steps": self.repeated_total_schedule_steps,
            "cycle_phase_drift": self.cycle_phase_drift,
            "phase_period_cycles": self.phase_period_cycles,
            "cycle_start_residues": list(self.cycle_start_residues),
            "base_exposure_by_regime": list(self.base_exposure_by_regime),
            "repeated_exposure_by_regime": list(self.repeated_exposure_by_regime),
            "source_manifest_unchanged": True,
            "finite_manifest_equivalent": False,
            "construction_is_outcome_free": True,
            "builder_authorizes_learner_execution": False,
            "source_calibration_use_allowed": (
                self.source_use_partition == CALIBRATION_ONLY_PARTITION
            ),
            "protected_source_performance_must_remain_unexecuted": (
                self.source_use_partition == PROTECTED_CANDIDATE_PARTITION
            ),
            "scientific_promotion_allowed": False,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenRegimeRepeatingPhaseDriftWorld:
    """A changed repeating world kept inseparable from its disclosure metadata."""

    world: HiddenRegimeWorldConfig
    metadata: HiddenRegimeRepeatingPhaseDriftMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.world, HiddenRegimeWorldConfig):
            raise TypeError("phase-drift world must contain a HiddenRegimeWorldConfig")
        if not isinstance(self.metadata, HiddenRegimeRepeatingPhaseDriftMetadata):
            raise TypeError("phase-drift world must contain disclosure metadata")
        if not self.world.repeat_schedule:
            raise ValueError("phase-drift world must repeat")
        if self.world.total_schedule_steps != self.metadata.repeated_total_schedule_steps:
            raise ValueError("phase-drift world and metadata totals disagree")

    def to_dict(self) -> dict[str, object]:
        """Serialize the derived world only inside its explicit disclosure wrapper."""

        return {
            "schema": "hidden-regime-repeating-phase-drift-world-v1",
            "derived_schedule_not_finite_manifest": True,
            "world": self.world.to_dict(),
            "metadata": self.metadata.to_dict(),
        }


def build_hidden_regime_repeating_phase_drift_world(
    manifest: HiddenRegimeScheduleManifest,
    *,
    final_regime_extension_steps: int,
) -> HiddenRegimeRepeatingPhaseDriftWorld:
    """Derive an outcome-free changed world without authorizing its execution."""

    if type(manifest) is not HiddenRegimeScheduleManifest:
        raise TypeError("phase-drift source must be an explicit schedule manifest")
    if type(final_regime_extension_steps) is not int:
        raise TypeError("final_regime_extension_steps must be a strict integer")
    if not 1 <= final_regime_extension_steps < STRUCTURAL_GENERALIZATION_LEASE_STEPS:
        raise ValueError("final_regime_extension_steps must be between 1 and 15")
    final_regime = manifest.segment_regimes[-1]
    changed_lengths = (
        *manifest.segment_lengths[:-1],
        manifest.segment_lengths[-1] + final_regime_extension_steps,
    )
    base_exposure = tuple(
        sum(
            length
            for length, actual_regime in zip(
                manifest.segment_lengths,
                manifest.segment_regimes,
                strict=True,
            )
            if actual_regime == regime
        )
        for regime in range(len(DEFAULT_REGIME_PERMUTATIONS))
    )
    repeated_exposure = tuple(
        exposure + (final_regime_extension_steps if regime == final_regime else 0)
        for regime, exposure in enumerate(base_exposure)
    )
    changed_total = sum(changed_lengths)
    cycle_drift = changed_total % STRUCTURAL_GENERALIZATION_LEASE_STEPS
    phase_period = STRUCTURAL_GENERALIZATION_LEASE_STEPS // math.gcd(
        STRUCTURAL_GENERALIZATION_LEASE_STEPS,
        cycle_drift,
    )
    metadata = HiddenRegimeRepeatingPhaseDriftMetadata(
        source_manifest_name=manifest.name,
        source_use_partition=manifest.use_partition,
        final_regime_id=final_regime,
        final_regime_extension_steps=final_regime_extension_steps,
        base_total_schedule_steps=sum(manifest.segment_lengths),
        repeated_total_schedule_steps=changed_total,
        cycle_phase_drift=cycle_drift,
        phase_period_cycles=phase_period,
        cycle_start_residues=tuple(
            (cycle * cycle_drift) % STRUCTURAL_GENERALIZATION_LEASE_STEPS
            for cycle in range(phase_period)
        ),
        base_exposure_by_regime=base_exposure,
        repeated_exposure_by_regime=repeated_exposure,
    )
    world = HiddenRegimeWorldConfig(
        segment_lengths=changed_lengths,
        segment_regimes=manifest.segment_regimes,
        regime_permutations=manifest.regime_permutations,
        repeat_schedule=True,
    )
    return HiddenRegimeRepeatingPhaseDriftWorld(world=world, metadata=metadata)


@chex.dataclass(frozen=True)
class HiddenRegimeWorldKeys:
    """Named independent random streams owned only by the world."""

    cue: PRNGKeyArray
    channel: PRNGKeyArray


def hidden_regime_world_keys(root_key: Array) -> HiddenRegimeWorldKeys:
    """Derive stable cue/channel streams without positional coupling."""

    return HiddenRegimeWorldKeys(
        cue=jr.fold_in(root_key, _CUE_RNG_TAG),
        channel=jr.fold_in(root_key, _CHANNEL_RNG_TAG),
    )


@chex.dataclass(frozen=True)
class HiddenRegimeObservation:
    """The helper's sole ordinary pre-message observation."""

    helper_cue: Int[Array, ""]


@chex.dataclass(frozen=True)
class HiddenRegimeWorldState:
    """Fixed-shape continuing state with a bounded evaluator schedule cursor."""

    cue_key: PRNGKeyArray
    channel_key: PRNGKeyArray
    cue: Int[Array, ""]
    step_count: Int[Array, ""]
    schedule_position: Int[Array, ""]


@chex.dataclass(frozen=True)
class HiddenRegimeOracle:
    """Evaluator-only facts forbidden from either learner policy."""

    step_count: Int[Array, ""]
    segment_index: Int[Array, ""]
    segment_step: Int[Array, ""]
    regime_id: Int[Array, ""]
    target: Int[Array, ""]


@chex.dataclass(frozen=True)
class HiddenRegimeTransition:
    """One continuing ternary signaling transition."""

    observation: HiddenRegimeObservation
    helper_message: Int[Array, ""]
    delivered_message: Int[Array, ""]
    beneficiary_action: Int[Array, ""]
    reward: Float[Array, ""]
    next_observation: HiddenRegimeObservation
    terminated: Bool[Array, ""]
    discount: Float[Array, ""]
    oracle: HiddenRegimeOracle


class HiddenRegimeSignalingWorld:
    """Pure-JAX continuing world with aliased hidden regimes."""

    def __init__(self, config: HiddenRegimeWorldConfig | None = None) -> None:
        self._config = config or HiddenRegimeWorldConfig()
        self._segment_ends = jnp.asarray(
            np.cumsum(self._config.segment_lengths),
            dtype=jnp.int32,
        )
        self._segment_regimes = jnp.asarray(
            self._config.segment_regimes,
            dtype=jnp.int32,
        )
        self._permutations = jnp.asarray(
            self._config.regime_permutations,
            dtype=jnp.int32,
        )

    @property
    def config(self) -> HiddenRegimeWorldConfig:
        """Return the evaluator-only static world configuration."""

        return self._config

    def schedule_position_of(self, source: HiddenRegimeWorldState | Array) -> Array:
        """Return the evaluator cursor from state or a safe nonrepeating counter.

        A saturated diagnostic counter cannot identify the live cursor of a
        repeating world.  Such worlds therefore require the authoritative
        :class:`HiddenRegimeWorldState`; raw counters remain supported for the
        backward-compatible nonrepeating schedule.
        """

        if isinstance(source, HiddenRegimeWorldState):
            return jnp.asarray(source.schedule_position, dtype=jnp.int32)
        if self._config.repeat_schedule:
            raise ValueError(
                "repeating schedule lookup requires authoritative HiddenRegimeWorldState"
            )
        step = jnp.asarray(source, dtype=jnp.int32)
        return jnp.minimum(
            step,
            jnp.asarray(self._config.total_schedule_steps - 1, dtype=jnp.int32),
        )

    def segment_index_of(self, source: HiddenRegimeWorldState | Array) -> Array:
        """Return evaluator-only segment index for state or a safe counter."""

        position = self.schedule_position_of(source)
        return self._segment_index_at(position)

    def _segment_index_at(self, schedule_position: Array) -> Array:
        """Return the segment index for an already bounded world cursor."""

        position = jnp.asarray(schedule_position, dtype=jnp.int32)
        return jnp.sum(position >= self._segment_ends).astype(jnp.int32)

    def segment_step_of(self, source: HiddenRegimeWorldState | Array) -> Array:
        """Return evaluator-only segment offset for state or a safe counter."""

        position = self.schedule_position_of(source)
        return self._segment_step_at(position)

    def _segment_step_at(self, schedule_position: Array) -> Array:
        """Return the within-segment offset for an already bounded cursor."""

        position = jnp.asarray(schedule_position, dtype=jnp.int32)
        segment = self._segment_index_at(position)
        previous_end = jnp.where(
            segment == 0,
            jnp.asarray(0, dtype=jnp.int32),
            self._segment_ends[jnp.maximum(segment - 1, 0)],
        )
        return (position - previous_end).astype(jnp.int32)

    def regime_of(self, source: HiddenRegimeWorldState | Array) -> Array:
        """Return evaluator-only hidden regime for state or a safe counter."""

        return self._regime_at(self.schedule_position_of(source))

    def _regime_at(self, schedule_position: Array) -> Array:
        """Return the hidden regime for an already bounded world cursor."""

        return self._segment_regimes[self._segment_index_at(schedule_position)]

    def target_of(self, source: HiddenRegimeWorldState | Array, cue: Array) -> Array:
        """Return evaluator-only target action for state/counter and one cue."""

        return self._target_at(self.schedule_position_of(source), cue)

    def _target_at(self, schedule_position: Array, cue: Array) -> Array:
        """Return the target for an already bounded world cursor."""

        return self._permutations[
            self._regime_at(schedule_position),
            jnp.asarray(cue, dtype=jnp.int32),
        ]

    def _next_schedule_position(self, schedule_position: Array) -> Array:
        """Advance the bounded cursor without integer overflow."""

        position = jnp.asarray(schedule_position, dtype=jnp.int32)
        final_position = jnp.asarray(
            self._config.total_schedule_steps - 1,
            dtype=jnp.int32,
        )
        if self._config.repeat_schedule:
            return jnp.where(
                position == final_position,
                jnp.asarray(0, dtype=jnp.int32),
                position + jnp.asarray(1, dtype=jnp.int32),
            )
        return jnp.where(
            position < final_position,
            position + jnp.asarray(1, dtype=jnp.int32),
            position,
        )

    @staticmethod
    def _next_step_count(step_count: Array) -> Array:
        """Saturate the diagnostic global counter at the int32 maximum."""

        step = jnp.asarray(step_count, dtype=jnp.int32)
        maximum = jnp.asarray(np.iinfo(np.int32).max, dtype=jnp.int32)
        return jnp.where(
            step < maximum,
            step + jnp.asarray(1, dtype=jnp.int32),
            step,
        )

    def init(self, keys: HiddenRegimeWorldKeys) -> HiddenRegimeWorldState:
        """Sample the first cue and initialize a continuing global step."""

        cue_draw_key, next_cue_key = jr.split(keys.cue)
        cue = jr.randint(
            cue_draw_key,
            (),
            0,
            N_HIDDEN_REGIME_SYMBOLS,
            dtype=jnp.int32,
        )
        return HiddenRegimeWorldState(
            cue_key=next_cue_key,
            channel_key=keys.channel,
            cue=cue,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            schedule_position=jnp.asarray(0, dtype=jnp.int32),
        )

    def observe(self, state: HiddenRegimeWorldState) -> HiddenRegimeObservation:
        """Return the ordinary observation, containing only the private cue."""

        return HiddenRegimeObservation(helper_cue=state.cue)

    def deliver(
        self,
        state: HiddenRegimeWorldState,
        helper_message: Array,
        channel: HiddenRegimeChannel,
    ) -> Array:
        """Resolve a ternary channel without advancing any world stream."""

        if channel == DIRECT_TERNARY_CHANNEL:
            return jnp.asarray(helper_message, dtype=jnp.int32)
        if channel == CONSTANT_ZERO_TERNARY_CHANNEL:
            return jnp.asarray(0, dtype=jnp.int32)
        if channel == CONSTANT_ONE_TERNARY_CHANNEL:
            return jnp.asarray(1, dtype=jnp.int32)
        if channel == CONSTANT_TWO_TERNARY_CHANNEL:
            return jnp.asarray(2, dtype=jnp.int32)
        if channel == SHUFFLED_TERNARY_CHANNEL:
            draw_key, _ = jr.split(state.channel_key)
            return jr.randint(
                draw_key,
                (),
                0,
                N_HIDDEN_REGIME_SYMBOLS,
                dtype=jnp.int32,
            )
        raise ValueError(f"unknown hidden-regime channel: {channel!r}")

    def step(
        self,
        state: HiddenRegimeWorldState,
        helper_message: Array,
        beneficiary_action: Array,
        channel: HiddenRegimeChannel = DIRECT_TERNARY_CHANNEL,
    ) -> tuple[HiddenRegimeTransition, HiddenRegimeWorldState]:
        """Apply a message/action pair and advance without termination."""

        delivered = self.deliver(state, helper_message, channel)
        return self.step_with_delivery(
            state,
            helper_message,
            delivered,
            beneficiary_action,
        )

    def step_with_delivery(
        self,
        state: HiddenRegimeWorldState,
        helper_message: Array,
        delivered_message: Array,
        beneficiary_action: Array,
    ) -> tuple[HiddenRegimeTransition, HiddenRegimeWorldState]:
        """Advance using an already resolved evaluator-declared channel output."""

        observation = self.observe(state)
        delivered = jnp.asarray(delivered_message, dtype=jnp.int32)
        action = jnp.asarray(beneficiary_action, dtype=jnp.int32)
        segment = self._segment_index_at(state.schedule_position)
        regime = self._regime_at(state.schedule_position)
        target = self._target_at(state.schedule_position, observation.helper_cue)
        reward = (action == target).astype(jnp.float32)

        cue_draw_key, next_cue_key = jr.split(state.cue_key)
        next_cue = jr.randint(
            cue_draw_key,
            (),
            0,
            N_HIDDEN_REGIME_SYMBOLS,
            dtype=jnp.int32,
        )
        _, next_channel_key = jr.split(state.channel_key)
        next_state = HiddenRegimeWorldState(
            cue_key=next_cue_key,
            channel_key=next_channel_key,
            cue=next_cue,
            step_count=self._next_step_count(state.step_count),
            schedule_position=self._next_schedule_position(state.schedule_position),
        )
        transition = HiddenRegimeTransition(
            observation=observation,
            helper_message=jnp.asarray(helper_message, dtype=jnp.int32),
            delivered_message=delivered,
            beneficiary_action=action,
            reward=reward,
            next_observation=self.observe(next_state),
            terminated=jnp.asarray(False),
            discount=jnp.asarray(1.0, dtype=jnp.float32),
            oracle=HiddenRegimeOracle(
                step_count=state.step_count,
                segment_index=segment,
                segment_step=self._segment_step_at(state.schedule_position),
                regime_id=regime,
                target=target,
            ),
        )
        return transition, next_state
