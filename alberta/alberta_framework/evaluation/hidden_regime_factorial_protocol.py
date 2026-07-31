"""Frozen, non-executing design for hidden-regime factorial calibration.

This module is a protocol specification, not an evaluator.  It contains no
world construction, learner update, command-line entry point, or artifact
writer.  The thirty seed pairs are permanently consumed for development by
their publication here, even before any outcome is observed.  Nothing produced
under this namespace can support evidence promotion.

The primary retention attribution is generation lineage, fixed without looking
at recurrence performance.  At every recurrence the evaluator must select the
latest prior synchronized commit for the same evaluator-only regime whose exact
slot and generation remain present in both roles.  A post-hoc best dormant slot
can therefore never become a primary endpoint.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

from alberta_framework.streams.hidden_regime_signaling import (
    CALIBRATION_ONLY_PARTITION,
    HIDDEN_REGIME_CALIBRATION_MANIFESTS,
    HIDDEN_REGIME_MANIFEST_USE_LEDGER,
    PROTECTED_CANDIDATE_LEARNER_OUTCOMES_EXECUTED,
    PROTECTED_CANDIDATE_PARTITION,
)

DESIGN_SCHEMA = "alberta.hidden-regime-factorial.calibration-design.v1"
DESIGN_ENVELOPE_SCHEMA = "alberta.hidden-regime-factorial.calibration-envelope.v1"
SEED_DERIVATION_SCHEMA = "alberta.hidden-regime-factorial.seed-pairs.sha256-be32.v1"
THRESHOLD_FREEZE_RECEIPT_SCHEMA = (
    "alberta.hidden-regime-factorial.threshold-freeze-receipt.v1"
)
CALIBRATION_READINESS_RECEIPT_SCHEMA = (
    "alberta.hidden-regime-factorial.calibration-readiness-receipt.v1"
)
BOUND_DEVELOPMENT_SUMMARY_SCHEMA = "alberta.hidden-regime-signaling.development.v5"
BOUND_PRIMITIVE_TRACE_SCHEMA = "alberta.hidden-regime-signaling.primitive-trace.v3"
PROTOCOL_STATUS = "calibration_design_frozen_outcomes_unexecuted"

CONSUMED_CALIBRATION_NAMESPACE = (
    "hidden-regime-factorial-calibration-v1-consumed-nonpromoting"
)
N_SEED_PAIRS = 30
N_CONDITIONS = 8
N_MATCHED_CASES = N_SEED_PAIRS * N_CONDITIONS
EXPECTED_ROLE_STATE_SCALARS = 69
EXPECTED_ROLE_STATE_BYTES = 276
EXPECTED_DYAD_STATE_SCALARS = 138
EXPECTED_DYAD_STATE_BYTES = 552

CALIBRATION_MANIFEST_ORDER: tuple[str, ...] = (
    "hidden-regime-calibration-a-v1",
    "hidden-regime-calibration-b-v1",
    "hidden-regime-calibration-c-v1",
)
CALIBRATION_MANIFEST_PAYLOAD_SHA256: tuple[str, ...] = (
    "57a4bd7a0bb9edf2a3ff962869eb3d6b78d54296c033676af6c6f62483882209",
    "84ce1a104d8859fdeb9cbe0d9bae7e62df75d731621ef0b25e95426d8a3e6864",
    "8cf72a724d922f685216ca54678f486f992fbd10763245d0c06be84e8361febe",
)

CANONICAL_CONDITION_ORDER: tuple[str, ...] = (
    "selective_full",
    "writable_evidence",
    "selective_lru",
    "writable_lru",
    "helper_frozen",
    "beneficiary_frozen",
    "constant_channel_0",
    "shuffled_channel",
)

FACTORIAL_CELL_ORDER: tuple[str, ...] = ("SE", "WE", "SL", "WL")
CONTROL_CONDITION_ORDER: tuple[str, ...] = (
    "helper_frozen",
    "beneficiary_frozen",
    "constant_channel_0",
    "shuffled_channel",
)

_UINT32_MAX = (1 << 32) - 1
_SHA256_HEX_LENGTH = 64
_PROHIBITED_NAMESPACE_TOKENS = (
    "heldout",
    "protected",
    "reserved",
    "candidate",
    "structural",
)


type MetricRole = Literal["primary", "secondary", "diagnostic"]
type MetricOrientation = Literal["higher", "lower"]
type GateMode = Literal["level_and_contrast", "contrast_only", "diagnostic_only"]
type EstimandRole = Literal["primary", "replication", "secondary", "diagnostic"]


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_json_value(value: object, *, location: str = "$") -> None:
    """Reject non-canonical JSON types, floats, and non-string mapping keys."""

    if value is None or type(value) in (str, int, bool):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, location=f"{location}[{index}]")
        return
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise TypeError(f"{location} contains a non-string mapping key")
            _validate_json_value(item, location=f"{location}.{key}")
        return
    raise TypeError(f"{location} contains unsupported JSON type {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Encode the protocol's integer/string-only canonical JSON representation."""

    _validate_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: object) -> str:
    """Return the lowercase SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _seed_preimage(index: int, lane: Literal["world", "learner"]) -> bytes:
    return (
        f"{SEED_DERIVATION_SCHEMA}|{CONSUMED_CALIBRATION_NAMESPACE}|{index}|{lane}"
    ).encode("ascii")


def derive_seed_pair_for_audit(index: int) -> tuple[int, int]:
    """Reconstruct one pair for integrity audit; literals remain authoritative."""

    if type(index) is not int or not 0 <= index < N_SEED_PAIRS:
        raise ValueError("seed-pair index must be a strict integer in [0, 30)")
    world_seed = int.from_bytes(hashlib.sha256(_seed_preimage(index, "world")).digest()[:4], "big")
    learner_seed = int.from_bytes(
        hashlib.sha256(_seed_preimage(index, "learner")).digest()[:4], "big"
    )
    return world_seed, learner_seed


@dataclass(frozen=True, slots=True)
class FrozenSeedPair:
    """One snapshotted uint32 world/learner pair in the consumed namespace."""

    index: int
    world_seed: int
    learner_seed: int

    def __post_init__(self) -> None:
        if type(self.index) is not int or not 0 <= self.index < N_SEED_PAIRS:
            raise ValueError("seed index must be a strict integer in [0, 30)")
        for name, value in (
            ("world_seed", self.world_seed),
            ("learner_seed", self.learner_seed),
        ):
            if type(value) is not int or not 0 <= value <= _UINT32_MAX:
                raise ValueError(f"{name} must be a strict uint32 integer")

    def to_payload(self) -> dict[str, int]:
        return {
            "index": self.index,
            "world_seed": self.world_seed,
            "learner_seed": self.learner_seed,
        }


FROZEN_SEED_PAIRS: tuple[FrozenSeedPair, ...] = (
    FrozenSeedPair(0, 1468689570, 1546104370),
    FrozenSeedPair(1, 590055347, 804438077),
    FrozenSeedPair(2, 3767879322, 1643509633),
    FrozenSeedPair(3, 80497473, 4189926585),
    FrozenSeedPair(4, 1945129402, 1883887917),
    FrozenSeedPair(5, 3873172020, 4083017348),
    FrozenSeedPair(6, 2583037211, 1182736045),
    FrozenSeedPair(7, 1877171996, 3683415448),
    FrozenSeedPair(8, 2204510677, 3993401959),
    FrozenSeedPair(9, 1545678539, 2068255135),
    FrozenSeedPair(10, 3836258998, 867857100),
    FrozenSeedPair(11, 777079313, 733824999),
    FrozenSeedPair(12, 722647547, 1529927928),
    FrozenSeedPair(13, 3826183176, 1166058410),
    FrozenSeedPair(14, 240076701, 485464268),
    FrozenSeedPair(15, 671721364, 2568884807),
    FrozenSeedPair(16, 2386694814, 1824260680),
    FrozenSeedPair(17, 3723998196, 4004534928),
    FrozenSeedPair(18, 3207110778, 584646942),
    FrozenSeedPair(19, 1625637479, 1661070870),
    FrozenSeedPair(20, 658805409, 1843539644),
    FrozenSeedPair(21, 2093172873, 1960429933),
    FrozenSeedPair(22, 3069546086, 3769537048),
    FrozenSeedPair(23, 3656452392, 2262967555),
    FrozenSeedPair(24, 879596909, 1087485336),
    FrozenSeedPair(25, 1509934539, 3292002002),
    FrozenSeedPair(26, 1116346787, 2120556918),
    FrozenSeedPair(27, 4284997650, 2893975960),
    FrozenSeedPair(28, 32033416, 1154546367),
    FrozenSeedPair(29, 3585556973, 2948504861),
)

SEED_SNAPSHOT_SHA256 = "1733afb917902d180c1c784563e7b557162eb36c6904dc6bc79b4b721ce008f3"


def seed_snapshot_payload() -> dict[str, object]:
    """Return the exact payload covered by :data:`SEED_SNAPSHOT_SHA256`."""

    return {
        "schema": SEED_DERIVATION_SCHEMA,
        "namespace": CONSUMED_CALIBRATION_NAMESPACE,
        "pairs": [pair.to_payload() for pair in FROZEN_SEED_PAIRS],
    }


@dataclass(frozen=True, slots=True)
class CalibrationManifestBinding:
    """Content binding for one calibration-only schedule manifest."""

    name: str
    use_partition: str
    manifest_payload_sha256: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in CALIBRATION_MANIFEST_ORDER:
            raise ValueError("manifest binding must name a frozen calibration manifest")
        if self.use_partition != CALIBRATION_ONLY_PARTITION:
            raise ValueError("manifest binding must use the calibration-only partition")
        if not _is_sha256(self.manifest_payload_sha256):
            raise ValueError("manifest binding digest must be lowercase SHA-256")

    def to_payload(self) -> dict[str, object]:
        return dataclasses.asdict(self)


_EXPECTED_RECURRENCE_BINDINGS: Mapping[
    str,
    tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, int, int, int, int],
        tuple[tuple[int, int, int], ...],
    ],
] = MappingProxyType({
    "hidden-regime-calibration-a-v1": (
        (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
        (3, 4, 5, 7, 9, 10, 11, 12, 14, 15, 16),
        (4, 4, 1, 2, 0),
        (
            (3, 2, 1),
            (4, 1, 1),
            (5, 0, 1),
            (7, 1, 2),
            (9, 0, 2),
            (10, 1, 3),
            (11, 3, 1),
            (12, 0, 3),
            (14, 3, 2),
            (15, 1, 4),
            (16, 0, 4),
        ),
    ),
    "hidden-regime-calibration-b-v1": (
        (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
        (4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16),
        (4, 4, 1, 2, 0),
        (
            (4, 1, 1),
            (5, 2, 1),
            (6, 1, 2),
            (7, 0, 1),
            (9, 1, 3),
            (10, 0, 2),
            (11, 3, 1),
            (13, 0, 3),
            (14, 1, 4),
            (15, 3, 2),
            (16, 0, 4),
        ),
    ),
    "hidden-regime-calibration-c-v1": (
        tuple(range(17)),
        (3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16),
        (5, 4, 1, 2, 0),
        (
            (3, 1, 1),
            (4, 2, 1),
            (5, 0, 1),
            (6, 1, 2),
            (7, 0, 2),
            (9, 0, 3),
            (10, 3, 1),
            (11, 1, 3),
            (12, 0, 4),
            (14, 0, 5),
            (15, 3, 2),
            (16, 1, 4),
        ),
    ),
})


@dataclass(frozen=True, slots=True)
class RecurrenceEligibilityBinding:
    """Manifest-specific recall entries after adjacent regimes are coalesced."""

    manifest_name: str
    coalesced_episode_start_segment_indices: tuple[int, ...]
    eligible_recurrence_start_segment_indices: tuple[int, ...]
    eligible_recurrence_counts_by_regime: tuple[int, int, int, int, int]
    eligible_recurrence_identities: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if type(self.manifest_name) is not str or self.manifest_name not in (
            CALIBRATION_MANIFEST_ORDER
        ):
            raise ValueError("recurrence binding must name a frozen calibration manifest")
        expected = _EXPECTED_RECURRENCE_BINDINGS[self.manifest_name]
        if (
            self.coalesced_episode_start_segment_indices,
            self.eligible_recurrence_start_segment_indices,
            self.eligible_recurrence_counts_by_regime,
            self.eligible_recurrence_identities,
        ) != expected:
            raise ValueError("recurrence binding differs from frozen coalesced episodes")
        for name, values in (
            (
                "coalesced_episode_start_segment_indices",
                self.coalesced_episode_start_segment_indices,
            ),
            (
                "eligible_recurrence_start_segment_indices",
                self.eligible_recurrence_start_segment_indices,
            ),
            ("eligible_recurrence_counts_by_regime", self.eligible_recurrence_counts_by_regime),
        ):
            if not isinstance(values, tuple) or any(type(value) is not int for value in values):
                raise ValueError(f"{name} must be a tuple of strict integers")
        if sum(self.eligible_recurrence_counts_by_regime) != len(
            self.eligible_recurrence_start_segment_indices
        ):
            raise ValueError("recurrence counts must equal the eligible entry count")
        if not isinstance(self.eligible_recurrence_identities, tuple) or any(
            not isinstance(identity, tuple)
            or len(identity) != 3
            or any(type(value) is not int for value in identity)
            for identity in self.eligible_recurrence_identities
        ):
            raise ValueError("recurrence identities must be strict integer triples")
        if tuple(identity[0] for identity in self.eligible_recurrence_identities) != (
            self.eligible_recurrence_start_segment_indices
        ):
            raise ValueError("recurrence identities must match eligible segment starts")

    def to_payload(self) -> dict[str, object]:
        return {
            "manifest_name": self.manifest_name,
            "eligibility_rule": (
                "episode entry for a previously observed exact regime after at least one complete "
                "intervening transition under a different regime; adjacent equal-regime segments "
                "are one uninterrupted episode and their boundary is never a recurrence"
            ),
            "required_runtime_helper": "hidden_regime_lineage_recurrence_segments(world)",
            "required_trace_fields": [
                "segment_index",
                "regime_id",
                "occurrence_index",
                "raw_segment_occurrence_index",
            ],
            "repeat_schedule": False,
            "expected_total_steps": 16_528,
            "execution_horizon": "exactly_one_finite_schedule",
            "coalesced_episode_start_segment_indices": list(
                self.coalesced_episode_start_segment_indices
            ),
            "eligible_recurrence_start_segment_indices": list(
                self.eligible_recurrence_start_segment_indices
            ),
            "eligible_recurrence_count": len(
                self.eligible_recurrence_start_segment_indices
            ),
            "eligible_recurrence_counts_by_regime": list(
                self.eligible_recurrence_counts_by_regime
            ),
            "eligible_recurrence_identities": [
                list(identity) for identity in self.eligible_recurrence_identities
            ],
        }


@dataclass(frozen=True, slots=True)
class PriorCommitLineageAuditRecord:
    """Minimal evaluator-only record used to prove full lineage enumeration."""

    lineage_index: int
    commit_step: int
    regime_id: int
    slot: int
    generation: int
    synchronized_commit: bool
    acquisition_qualified: bool
    survives_exact_generation_at_entry: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("lineage_index", self.lineage_index),
            ("commit_step", self.commit_step),
            ("regime_id", self.regime_id),
            ("slot", self.slot),
            ("generation", self.generation),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative strict integer")
        for name, value in (
            ("synchronized_commit", self.synchronized_commit),
            ("acquisition_qualified", self.acquisition_qualified),
            ("survives_exact_generation_at_entry", self.survives_exact_generation_at_entry),
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a strict boolean")
        if self.synchronized_commit is not True:
            raise ValueError("the prior-commit ledger contains only synchronized dyad commits")


@dataclass(frozen=True, slots=True)
class LineageSelectionAudit:
    """Derived recurrence selection with the complete prior collection retained."""

    all_prior_lineage_indices: tuple[int, ...]
    qualified_prior_lineage_indices: tuple[int, ...]
    latest_prior_qualified_lineage_index: int | None
    surviving_qualified_lineage_indices: tuple[int, ...]
    selected_latest_surviving_qualified_lineage_index: int | None


def audit_prior_commit_lineage_serialization(
    complete_commit_ledger: tuple[PriorCommitLineageAuditRecord, ...],
    *,
    recurrence_regime_id: int,
    recurrence_entry_step: int,
    serialized_prior_lineage_indices: tuple[int, ...],
) -> LineageSelectionAudit:
    """Reject omission of any earlier synchronized same-regime commit lineage."""

    if not isinstance(complete_commit_ledger, tuple):
        raise TypeError("complete_commit_ledger must be a tuple")
    if any(type(record) is not PriorCommitLineageAuditRecord for record in complete_commit_ledger):
        raise TypeError("complete_commit_ledger contains an invalid record")
    if type(recurrence_regime_id) is not int or recurrence_regime_id < 0:
        raise ValueError("recurrence_regime_id must be a nonnegative strict integer")
    if type(recurrence_entry_step) is not int or recurrence_entry_step < 1:
        raise ValueError("recurrence_entry_step must be a positive strict integer")
    if not isinstance(serialized_prior_lineage_indices, tuple) or any(
        type(lineage_index) is not int or lineage_index < 0
        for lineage_index in serialized_prior_lineage_indices
    ):
        raise ValueError(
            "serialized_prior_lineage_indices must be a tuple of nonnegative integers"
        )
    lineage_indices = tuple(record.lineage_index for record in complete_commit_ledger)
    if len(set(lineage_indices)) != len(lineage_indices):
        raise ValueError("complete commit lineage identifiers must be unique")
    ordered_ledger = tuple(
        sorted(
            complete_commit_ledger,
            key=lambda record: (
                record.commit_step,
                record.slot,
                record.generation,
                record.lineage_index,
            ),
        )
    )
    if complete_commit_ledger != ordered_ledger:
        raise ValueError("complete commit ledger must be in canonical lineage order")
    prior = tuple(
        record
        for record in complete_commit_ledger
        if record.regime_id == recurrence_regime_id
        and record.commit_step < recurrence_entry_step
    )
    expected_indices = tuple(record.lineage_index for record in prior)
    if serialized_prior_lineage_indices != expected_indices:
        raise ValueError(
            "serialized recurrence lineages must include every earlier synchronized "
            "same-regime commit, including unqualified and evicted lineages"
        )
    qualified = tuple(record for record in prior if record.acquisition_qualified)
    surviving = tuple(
        record for record in qualified if record.survives_exact_generation_at_entry
    )
    return LineageSelectionAudit(
        all_prior_lineage_indices=expected_indices,
        qualified_prior_lineage_indices=tuple(
            record.lineage_index for record in qualified
        ),
        latest_prior_qualified_lineage_index=(
            None if not qualified else qualified[-1].lineage_index
        ),
        surviving_qualified_lineage_indices=tuple(
            record.lineage_index for record in surviving
        ),
        selected_latest_surviving_qualified_lineage_index=(
            None if not surviving else surviving[-1].lineage_index
        ),
    )


@dataclass(frozen=True, slots=True)
class CalibrationAssignment:
    """Round-robin assignment of one matched seed pair to one manifest."""

    seed_index: int
    manifest_name: str

    def __post_init__(self) -> None:
        if type(self.seed_index) is not int or not 0 <= self.seed_index < N_SEED_PAIRS:
            raise ValueError("assignment seed_index must be a strict integer in [0, 30)")
        if type(self.manifest_name) is not str or self.manifest_name not in (
            CALIBRATION_MANIFEST_ORDER
        ):
            raise ValueError("assignment must use a frozen calibration-only manifest")
        expected = CALIBRATION_MANIFEST_ORDER[self.seed_index % len(CALIBRATION_MANIFEST_ORDER)]
        if self.manifest_name != expected:
            raise ValueError("assignment must follow frozen A/B/C round-robin order")

    def to_payload(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class MatchedCalibrationCase:
    """One condition within a seed-pair/world-matched eight-condition block."""

    case_index: int
    seed_index: int
    manifest_name: str
    world_seed: int
    learner_seed: int
    condition: str

    def __post_init__(self) -> None:
        if type(self.case_index) is not int or not 0 <= self.case_index < N_MATCHED_CASES:
            raise ValueError("case_index must be a strict integer in [0, 240)")
        if type(self.seed_index) is not int or not 0 <= self.seed_index < N_SEED_PAIRS:
            raise ValueError("case seed_index must be a strict integer in [0, 30)")
        if type(self.manifest_name) is not str or self.manifest_name not in (
            CALIBRATION_MANIFEST_ORDER
        ):
            raise ValueError("case must use a frozen calibration-only manifest")
        if type(self.condition) is not str or self.condition not in CANONICAL_CONDITION_ORDER:
            raise ValueError("case condition must be in the canonical order")
        expected_case = self.seed_index * N_CONDITIONS + CANONICAL_CONDITION_ORDER.index(
            self.condition
        )
        if self.case_index != expected_case:
            raise ValueError("case index must encode seed-major canonical condition order")
        pair = FROZEN_SEED_PAIRS[self.seed_index]
        if self.world_seed != pair.world_seed or self.learner_seed != pair.learner_seed:
            raise ValueError("case seeds must equal the frozen matched seed pair")
        expected_manifest = CALIBRATION_MANIFEST_ORDER[
            self.seed_index % len(CALIBRATION_MANIFEST_ORDER)
        ]
        if self.manifest_name != expected_manifest:
            raise ValueError("case manifest must equal its frozen round-robin assignment")

    def to_payload(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class FactorialCell:
    """One cell of durable-write policy by replacement-target policy."""

    code: str
    condition: str
    durable_write_policy: str
    replacement_target_policy: str

    def __post_init__(self) -> None:
        expected = {
            "SE": ("selective_full", "selective", "evidence"),
            "WE": ("writable_evidence", "writable", "evidence"),
            "SL": ("selective_lru", "selective", "lru"),
            "WL": ("writable_lru", "writable", "lru"),
        }
        if type(self.code) is not str or self.code not in expected:
            raise ValueError("unknown factorial cell code")
        if (
            self.condition,
            self.durable_write_policy,
            self.replacement_target_policy,
        ) != expected[self.code]:
            raise ValueError("factorial cell semantics do not match its code")

    def to_payload(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class ConditionRuntimeBinding:
    """Exact evaluator literals for one of the eight matched interventions."""

    condition: str
    durable_write_policy: str
    replacement_target_policy: str
    helper_learning_enabled: bool
    beneficiary_learning_enabled: bool
    delivery_mode: str
    constant_delivery_symbol: int | None
    shuffle_low_inclusive: int | None
    shuffle_high_exclusive: int | None
    shuffle_dtype: str | None
    shuffle_key_rule: str | None
    channel_key_advance_rule: str

    def __post_init__(self) -> None:
        if type(self.condition) is not str or self.condition not in CANONICAL_CONDITION_ORDER:
            raise ValueError("runtime binding condition is not canonical")
        expected = _EXPECTED_RUNTIME_BINDINGS[self.condition]
        actual = (
            self.durable_write_policy,
            self.replacement_target_policy,
            self.helper_learning_enabled,
            self.beneficiary_learning_enabled,
            self.delivery_mode,
            self.constant_delivery_symbol,
            self.shuffle_low_inclusive,
            self.shuffle_high_exclusive,
            self.shuffle_dtype,
            self.shuffle_key_rule,
            self.channel_key_advance_rule,
        )
        if actual != expected:
            raise ValueError("runtime binding differs from frozen evaluator literals")
        if type(self.helper_learning_enabled) is not bool or type(
            self.beneficiary_learning_enabled
        ) is not bool:
            raise ValueError("runtime learning permissions must be strict booleans")
        if self.constant_delivery_symbol is not None and type(
            self.constant_delivery_symbol
        ) is not int:
            raise ValueError("constant delivery symbol must be a strict integer or null")
        for name, value in (
            ("shuffle_low_inclusive", self.shuffle_low_inclusive),
            ("shuffle_high_exclusive", self.shuffle_high_exclusive),
        ):
            if value is not None and type(value) is not int:
                raise ValueError(f"{name} must be a strict integer or null")

    def to_payload(self) -> dict[str, object]:
        return {
            "condition": self.condition,
            "durable_write_policy": self.durable_write_policy,
            "replacement_target_policy": self.replacement_target_policy,
            "helper_learning_enabled": self.helper_learning_enabled,
            "beneficiary_learning_enabled": self.beneficiary_learning_enabled,
            "delivery_mode": self.delivery_mode,
            "constant_delivery_symbol": self.constant_delivery_symbol,
            "shuffle_low_inclusive": self.shuffle_low_inclusive,
            "shuffle_high_exclusive": self.shuffle_high_exclusive,
            "shuffle_dtype": self.shuffle_dtype,
            "shuffle_key_rule": self.shuffle_key_rule,
            "channel_key_advance_rule": self.channel_key_advance_rule,
            "evaluator_field_binding": {
                "helper_learning_enabled": "HiddenRegimeConditionSpec.helper_write",
                "beneficiary_learning_enabled": "HiddenRegimeConditionSpec.beneficiary_write",
                "delivery_mode": "HiddenRegimeConditionSpec.channel",
                "durable_write_policy": "HiddenRegimeConditionSpec.durable_write_policy",
                "replacement_target_policy": (
                    "HiddenRegimeConditionSpec.replacement_target_policy"
                ),
            },
        }


_DIRECT_SHUFFLE_RULE: None = None
_SHUFFLED_KEY_RULE = (
    "jax.random.split(world.channel_key)[0] then "
    "jax.random.randint(key,shape=(),minval=0,maxval=3,dtype=int32)"
)
_CHANNEL_KEY_ADVANCE_RULE = "world channel_key advances exactly once on every transition"

_EXPECTED_RUNTIME_BINDINGS: Mapping[
    str,
    tuple[
        str,
        str,
        bool,
        bool,
        str,
        int | None,
        int | None,
        int | None,
        str | None,
        str | None,
        str,
    ],
] = MappingProxyType({
    "selective_full": (
        "selective",
        "evidence",
        True,
        True,
        "direct",
        None,
        None,
        None,
        None,
        _DIRECT_SHUFFLE_RULE,
        _CHANNEL_KEY_ADVANCE_RULE,
    ),
    "writable_evidence": (
        "writable",
        "evidence",
        True,
        True,
        "direct",
        None,
        None,
        None,
        None,
        _DIRECT_SHUFFLE_RULE,
        _CHANNEL_KEY_ADVANCE_RULE,
    ),
    "selective_lru": (
        "selective",
        "lru",
        True,
        True,
        "direct",
        None,
        None,
        None,
        None,
        _DIRECT_SHUFFLE_RULE,
        _CHANNEL_KEY_ADVANCE_RULE,
    ),
    "writable_lru": (
        "writable",
        "lru",
        True,
        True,
        "direct",
        None,
        None,
        None,
        None,
        _DIRECT_SHUFFLE_RULE,
        _CHANNEL_KEY_ADVANCE_RULE,
    ),
    "helper_frozen": (
        "selective",
        "evidence",
        False,
        True,
        "direct",
        None,
        None,
        None,
        None,
        _DIRECT_SHUFFLE_RULE,
        _CHANNEL_KEY_ADVANCE_RULE,
    ),
    "beneficiary_frozen": (
        "selective",
        "evidence",
        True,
        False,
        "direct",
        None,
        None,
        None,
        None,
        _DIRECT_SHUFFLE_RULE,
        _CHANNEL_KEY_ADVANCE_RULE,
    ),
    "constant_channel_0": (
        "selective",
        "evidence",
        True,
        True,
        "constant_0",
        0,
        None,
        None,
        None,
        _DIRECT_SHUFFLE_RULE,
        _CHANNEL_KEY_ADVANCE_RULE,
    ),
    "shuffled_channel": (
        "selective",
        "evidence",
        True,
        True,
        "shuffled",
        None,
        0,
        3,
        "int32",
        _SHUFFLED_KEY_RULE,
        _CHANNEL_KEY_ADVANCE_RULE,
    ),
})


def _runtime_bindings() -> tuple[ConditionRuntimeBinding, ...]:
    return tuple(
        ConditionRuntimeBinding(condition, *_EXPECTED_RUNTIME_BINDINGS[condition])
        for condition in CANONICAL_CONDITION_ORDER
    )


@dataclass(frozen=True, slots=True)
class BaseEvaluatorConfigBinding:
    """Exact common learner/evaluator configuration shared by all 240 cases."""

    learning_rate_decimal: str
    epsilon_decimal: str
    relevance_rate_decimal: str
    lease_length: int
    confirmation_steps: int
    durable_retrieval_threshold_decimal: str
    candidate_confirmation_threshold_decimal: str
    candidate_confirmation_leases: int
    scratch_training_leases_before_retest: int
    writable_lru_ablation: bool
    requested_durable_write_policy: None
    requested_replacement_target_policy: None
    effective_base_durable_write_policy: str
    effective_base_replacement_target_policy: str
    metric_window: int
    signal_symbols: int
    slot_inputs: int
    slot_actions: int
    durable_slots: int
    total_slots_per_role: int
    scratch_slot: int
    repeat_schedule: bool
    execute_exactly_one_finite_schedule: bool
    expected_total_steps_per_manifest: int

    def __post_init__(self) -> None:
        for name in (
            "learning_rate_decimal",
            "epsilon_decimal",
            "relevance_rate_decimal",
            "durable_retrieval_threshold_decimal",
            "candidate_confirmation_threshold_decimal",
            "effective_base_durable_write_policy",
            "effective_base_replacement_target_policy",
        ):
            if type(getattr(self, name)) is not str:
                raise ValueError(f"{name} must be a strict string")
        for name in (
            "lease_length",
            "confirmation_steps",
            "candidate_confirmation_leases",
            "scratch_training_leases_before_retest",
            "metric_window",
            "signal_symbols",
            "slot_inputs",
            "slot_actions",
            "durable_slots",
            "total_slots_per_role",
            "scratch_slot",
            "expected_total_steps_per_manifest",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be a strict integer")
        for name in (
            "writable_lru_ablation",
            "repeat_schedule",
            "execute_exactly_one_finite_schedule",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a strict boolean")
        if self.requested_durable_write_policy is not None or (
            self.requested_replacement_target_policy is not None
        ):
            raise ValueError("base requested policies must be null before condition replacement")
        actual = (
            self.learning_rate_decimal,
            self.epsilon_decimal,
            self.relevance_rate_decimal,
            self.lease_length,
            self.confirmation_steps,
            self.durable_retrieval_threshold_decimal,
            self.candidate_confirmation_threshold_decimal,
            self.candidate_confirmation_leases,
            self.scratch_training_leases_before_retest,
            self.writable_lru_ablation,
            self.requested_durable_write_policy,
            self.requested_replacement_target_policy,
            self.effective_base_durable_write_policy,
            self.effective_base_replacement_target_policy,
            self.metric_window,
            self.signal_symbols,
            self.slot_inputs,
            self.slot_actions,
            self.durable_slots,
            self.total_slots_per_role,
            self.scratch_slot,
            self.repeat_schedule,
            self.execute_exactly_one_finite_schedule,
            self.expected_total_steps_per_manifest,
        )
        expected = (
            "0.25",
            "0.1",
            "0.1",
            16,
            8,
            "0.5",
            "0.75",
            3,
            16,
            False,
            None,
            None,
            "selective",
            "evidence",
            128,
            3,
            3,
            3,
            3,
            4,
            0,
            False,
            True,
            16_528,
        )
        if actual != expected:
            raise ValueError("base evaluator configuration differs from frozen literals")

    def to_payload(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["condition_axis_application"] = (
            "for each case replace only durable_write_policy and replacement_target_policy with "
            "the condition runtime binding; helper/beneficiary learning and delivery use their "
            "separate binding fields; every other learner/evaluator field remains exact"
        )
        payload["retention_window_semantics"] = (
            "exactly lease_length prequential transitions from each coalesced recurrence episode "
            "entry, without alignment to learner lease boundaries"
        )
        return payload

def _base_config_binding() -> BaseEvaluatorConfigBinding:
    return BaseEvaluatorConfigBinding(
        learning_rate_decimal="0.25",
        epsilon_decimal="0.1",
        relevance_rate_decimal="0.1",
        lease_length=16,
        confirmation_steps=8,
        durable_retrieval_threshold_decimal="0.5",
        candidate_confirmation_threshold_decimal="0.75",
        candidate_confirmation_leases=3,
        scratch_training_leases_before_retest=16,
        writable_lru_ablation=False,
        requested_durable_write_policy=None,
        requested_replacement_target_policy=None,
        effective_base_durable_write_policy="selective",
        effective_base_replacement_target_policy="evidence",
        metric_window=128,
        signal_symbols=3,
        slot_inputs=3,
        slot_actions=3,
        durable_slots=3,
        total_slots_per_role=4,
        scratch_slot=0,
        repeat_schedule=False,
        execute_exactly_one_finite_schedule=True,
        expected_total_steps_per_manifest=16_528,
    )


@dataclass(frozen=True, slots=True)
class MetricContract:
    """Pre-outcome definition and orientation for one retention endpoint."""

    metric_id: str
    role: MetricRole
    orientation: MetricOrientation
    gate_mode: GateMode
    source_fields: tuple[str, ...]
    aggregation: str
    eligibility: str
    missingness: str
    null_value_decimal: str | None

    def __post_init__(self) -> None:
        if type(self.metric_id) is not str or not self.metric_id:
            raise ValueError("metric_id must be a nonempty string")
        if self.role not in ("primary", "secondary", "diagnostic"):
            raise ValueError("unknown metric role")
        if self.orientation not in ("higher", "lower"):
            raise ValueError("unknown metric orientation")
        if self.gate_mode not in (
            "level_and_contrast",
            "contrast_only",
            "diagnostic_only",
        ):
            raise ValueError("unknown metric gate mode")
        if not isinstance(self.source_fields, tuple) or not self.source_fields:
            raise ValueError("source_fields must be a nonempty tuple")
        if any(type(field) is not str or not field for field in self.source_fields):
            raise ValueError("source_fields must contain nonempty strings")
        for name, value in (
            ("aggregation", self.aggregation),
            ("eligibility", self.eligibility),
            ("missingness", self.missingness),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if self.null_value_decimal is not None and (
            type(self.null_value_decimal) is not str or not self.null_value_decimal
        ):
            raise ValueError("null_value_decimal must be a nonempty decimal string or null")
        if self.role == "diagnostic" and self.gate_mode != "diagnostic_only":
            raise ValueError("diagnostic metrics cannot become gates")

    def to_payload(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["source_fields"] = list(self.source_fields)
        return payload


@dataclass(frozen=True, slots=True)
class PairedPopulationSupportMetric:
    """Coverage of the exact recurrence intersection supporting a paired estimand."""

    metric_id: str
    estimand_id: str
    conditions: tuple[str, ...]
    source_fields: tuple[str, ...]
    numerator: str
    denominator: str
    per_seed_aggregation: str
    mandatory: bool
    orientation: Literal["higher"]
    null_value_decimal: Literal["0"]

    def __post_init__(self) -> None:
        if type(self.metric_id) is not str or not self.metric_id:
            raise ValueError("paired support metric_id must be a nonempty string")
        if type(self.estimand_id) is not str or not self.estimand_id:
            raise ValueError("paired support estimand_id must be a nonempty string")
        if not isinstance(self.conditions, tuple) or len(self.conditions) < 2:
            raise ValueError("paired support conditions require at least two entries")
        if any(
            type(condition) is not str or condition not in CANONICAL_CONDITION_ORDER
            for condition in self.conditions
        ) or len(set(self.conditions)) != len(self.conditions):
            raise ValueError("paired support conditions must be unique and canonical")
        if not isinstance(self.source_fields, tuple) or not self.source_fields:
            raise ValueError("paired support source_fields must be a nonempty tuple")
        if any(type(field) is not str or not field for field in self.source_fields):
            raise ValueError("paired support source_fields must contain nonempty strings")
        for name, value in (
            ("numerator", self.numerator),
            ("denominator", self.denominator),
            ("per_seed_aggregation", self.per_seed_aggregation),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if self.mandatory is not True or type(self.mandatory) is not bool:
            raise ValueError("paired support metrics are mandatory strict booleans")
        if self.orientation != "higher" or self.null_value_decimal != "0":
            raise ValueError("paired support metrics require higher-is-better exact null 0")

    def to_payload(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["conditions"] = list(self.conditions)
        payload["source_fields"] = list(self.source_fields)
        return payload


@dataclass(frozen=True, slots=True)
class EstimandContract:
    """Exact oriented within-pair contrast evaluated separately per metric."""

    estimand_id: str
    role: EstimandRole
    formula: str
    condition_terms: tuple[tuple[str, int], ...]
    population_rule: str
    metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.estimand_id) is not str or not self.estimand_id:
            raise ValueError("estimand_id must be a nonempty string")
        if self.role not in ("primary", "replication", "secondary", "diagnostic"):
            raise ValueError("unknown estimand role")
        if type(self.formula) is not str or not self.formula:
            raise ValueError("estimand formula must be a nonempty string")
        if not isinstance(self.condition_terms, tuple) or len(self.condition_terms) < 2:
            raise ValueError("estimand condition terms must contain at least two entries")
        if any(
            not isinstance(term, tuple)
            or len(term) != 2
            or type(term[0]) is not str
            or term[0] not in CANONICAL_CONDITION_ORDER
            or type(term[1]) is not int
            or term[1] == 0
            for term in self.condition_terms
        ):
            raise ValueError("estimand condition terms are invalid")
        conditions = tuple(term[0] for term in self.condition_terms)
        if len(set(conditions)) != len(conditions) or sum(
            term[1] for term in self.condition_terms
        ) != 0:
            raise ValueError("estimand terms require unique conditions and zero-sum coefficients")
        if type(self.population_rule) is not str or not self.population_rule:
            raise ValueError("estimand population rule must be a nonempty string")
        if not isinstance(self.metrics, tuple) or not self.metrics:
            raise ValueError("estimand metrics must be a nonempty tuple")
        if any(type(metric) is not str or not metric for metric in self.metrics):
            raise ValueError("estimand metrics must contain nonempty strings")

    def to_payload(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["condition_terms"] = [
            {"condition": condition, "coefficient": coefficient}
            for condition, coefficient in self.condition_terms
        ]
        payload["metrics"] = list(self.metrics)
        return payload


@dataclass(frozen=True, slots=True)
class AuditRequirement:
    """A mandatory non-statistical execution or trace invariant."""

    requirement_id: str
    scope: str
    predicate: str
    failure_disposition: str

    def __post_init__(self) -> None:
        for name, value in (
            ("requirement_id", self.requirement_id),
            ("scope", self.scope),
            ("predicate", self.predicate),
            ("failure_disposition", self.failure_disposition),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty string")

    def to_payload(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class StatisticalSummaryPlan:
    """Frozen descriptive and confidence-bound calculations."""

    paired_unit: str
    paired_recurrence_alignment: str
    orientation_transform: str
    pooled_expected_n: int
    manifest_expected_n: int
    confidence_basis_points: int
    standard_deviation: str
    standard_error: str
    one_sided_lower_bound: str
    win_definition: str
    tie_definition: str
    loss_definition: str
    missingness_policy: str
    pooled_summary: tuple[str, ...]
    manifest_stratified_summary: tuple[str, ...]
    worst_manifest_summary: tuple[str, ...]
    multiplicity_scope: str

    def __post_init__(self) -> None:
        if self.pooled_expected_n != 30 or self.manifest_expected_n != 10:
            raise ValueError("statistical plan requires n=30 pooled and n=10 per manifest")
        if self.confidence_basis_points != 9500:
            raise ValueError("statistical plan requires a frozen one-sided 95% bound")
        for name in (
            "paired_unit",
            "paired_recurrence_alignment",
            "orientation_transform",
            "standard_deviation",
            "standard_error",
            "one_sided_lower_bound",
            "win_definition",
            "tie_definition",
            "loss_definition",
            "missingness_policy",
            "multiplicity_scope",
        ):
            string_value = cast(str, getattr(self, name))
            if type(string_value) is not str or not string_value:
                raise ValueError(f"{name} must be a nonempty string")
        for name in (
            "pooled_summary",
            "manifest_stratified_summary",
            "worst_manifest_summary",
        ):
            tuple_value = cast(tuple[str, ...], getattr(self, name))
            if not isinstance(tuple_value, tuple) or not tuple_value:
                raise ValueError(f"{name} must be a nonempty tuple")
            if any(type(item) is not str or not item for item in tuple_value):
                raise ValueError(f"{name} must contain nonempty strings")

    def to_payload(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        for name in (
            "pooled_summary",
            "manifest_stratified_summary",
            "worst_manifest_summary",
        ):
            payload[name] = list(cast(tuple[str, ...], getattr(self, name)))
        return payload


@dataclass(frozen=True, slots=True)
class ThresholdFreezeRule:
    """Deterministic future rule; this design intentionally has no thresholds."""

    status: str
    frozen_thresholds: tuple[object, ...]
    threshold_freeze_receipt: None
    minimum_margin_ratio_numerator: int
    minimum_margin_ratio_denominator: int
    rounding_quantum_decimal: str
    higher_is_better_rule: str
    lower_is_better_rule: str
    wins_rule: str
    mandatory_missingness_rule: str
    conservative_bound_rule: str
    receipt_schema: str
    receipt_required_fields: tuple[str, ...]
    receipt_immutability: str
    post_protected_adjustment: str

    def __post_init__(self) -> None:
        if self.status != "thresholds_unset_pending_consumed_calibration_outcomes":
            raise ValueError("threshold rule status is not frozen")
        if self.frozen_thresholds != () or self.threshold_freeze_receipt is not None:
            raise ValueError("thresholds and freeze receipt must remain explicitly unset")
        if (
            self.minimum_margin_ratio_numerator,
            self.minimum_margin_ratio_denominator,
        ) != (2, 1):
            raise ValueError("threshold rule must require at least a twofold margin")
        if self.rounding_quantum_decimal != "0.0001":
            raise ValueError("threshold rounding quantum must remain 0.0001")
        if self.receipt_schema != THRESHOLD_FREEZE_RECEIPT_SCHEMA:
            raise ValueError("threshold receipt schema is not frozen")
        if not isinstance(self.receipt_required_fields, tuple) or not (
            self.receipt_required_fields
        ):
            raise ValueError("threshold receipt fields must be a nonempty tuple")
        if len(set(self.receipt_required_fields)) != len(self.receipt_required_fields):
            raise ValueError("threshold receipt fields must be unique")
        for name in (
            "higher_is_better_rule",
            "lower_is_better_rule",
            "wins_rule",
            "mandatory_missingness_rule",
            "conservative_bound_rule",
            "receipt_immutability",
            "post_protected_adjustment",
        ):
            value = cast(str, getattr(self, name))
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty string")

    def to_payload(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["frozen_thresholds"] = []
        payload["receipt_required_fields"] = list(self.receipt_required_fields)
        return payload


@dataclass(frozen=True, slots=True)
class GateFamilySpec:
    """Exact mandatory/descriptive scope fixed before any calibration outcome."""

    gate_family_id: str
    mandatory: bool
    conditions: tuple[str, ...]
    metric_ids: tuple[str, ...]
    estimand_ids: tuple[str, ...]
    required_components: tuple[str, ...]
    null_rule: str
    qualification_and_missingness_rule: str
    failure_disposition: str

    def __post_init__(self) -> None:
        if type(self.gate_family_id) is not str or not self.gate_family_id:
            raise ValueError("gate family identifier must be a nonempty string")
        if type(self.mandatory) is not bool:
            raise ValueError("gate family mandatory flag must be a strict boolean")
        if not isinstance(self.conditions, tuple) or any(
            type(condition) is not str or condition not in CANONICAL_CONDITION_ORDER
            for condition in self.conditions
        ):
            raise ValueError("gate family conditions must be canonical")
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("gate family conditions must be unique")
        for name, values in (
            ("metric_ids", self.metric_ids),
            ("estimand_ids", self.estimand_ids),
            ("required_components", self.required_components),
        ):
            if not isinstance(values, tuple) or any(
                type(value) is not str or not value for value in values
            ):
                raise ValueError(f"{name} must be a tuple of nonempty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
        if not self.required_components:
            raise ValueError("gate family required components cannot be empty")
        for name, value in (
            ("null_rule", self.null_rule),
            ("qualification_and_missingness_rule", self.qualification_and_missingness_rule),
            ("failure_disposition", self.failure_disposition),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty string")

    def to_payload(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        for name in ("conditions", "metric_ids", "estimand_ids", "required_components"):
            payload[name] = list(cast(tuple[str, ...], getattr(self, name)))
        return payload


_COMMIT_GENERATION_LINEAGE_FIELDS: tuple[str, ...] = (
    "lineage_index",
    "commit_step",
    "commit_segment_index",
    "commit_segment_step",
    "regime_id",
    "regime_label",
    "slot",
    "generation",
    "target_mapping",
    "committed_composed_greedy_mapping",
    "committed_composed_greedy_accuracy",
    "committed_composed_greedy_tie_free",
    "acquisition_qualified",
    "helper_table_uint32_bits",
    "beneficiary_table_uint32_bits",
)

_RECURRENCE_LINEAGE_PROBE_FIELDS: tuple[str, ...] = (
    "lineage_index",
    "commit_step",
    "commit_segment_index",
    "slot",
    "generation",
    "acquisition_qualified",
    "helper_entry_slot_status",
    "helper_entry_slot_generation",
    "helper_slot_generation_present",
    "beneficiary_entry_slot_status",
    "beneficiary_entry_slot_generation",
    "beneficiary_slot_generation_present",
    "synchronized_generation_survives",
    "helper_active_at_entry",
    "beneficiary_active_at_entry",
    "entry_activity_status",
    "helper_entry_table_uint32_bits",
    "beneficiary_entry_table_uint32_bits",
    "entry_composed_greedy_mapping",
    "entry_composed_greedy_accuracy",
    "entry_minus_commit_accuracy",
    "helper_bit_exact_preserved",
    "beneficiary_bit_exact_preserved",
    "joint_bit_exact_preserved",
    "zero_helper_accuracy",
    "zero_beneficiary_accuracy",
    "role_swapped_accuracy",
)

PRIMARY_METRIC_IDS: tuple[str, ...] = (
    "qualified_first_entry_window_error_rate",
    "latest_prior_qualified_lineage_survival_rate",
    "selected_lineage_joint_bit_exact_preservation_rate",
    "selected_lineage_entry_composed_accuracy",
    "selected_lineage_commit_to_entry_accuracy_change",
    "selected_lineage_exact_generation_relock_rate",
    "selected_lineage_retrieval_before_scratch_rate",
    "recurrence_minus_latest_qualified_acquisition_error_rate",
)

PRIMARY_LEVEL_METRIC_IDS: tuple[str, ...] = (
    "qualified_first_entry_window_error_rate",
    "latest_prior_qualified_lineage_survival_rate",
    "selected_lineage_joint_bit_exact_preservation_rate",
    "selected_lineage_entry_composed_accuracy",
    "selected_lineage_exact_generation_relock_rate",
    "selected_lineage_retrieval_before_scratch_rate",
)

SECONDARY_METRIC_IDS: tuple[str, ...] = (
    "acquisition_qualified_recurrence_coverage_rate",
    "all_recurrence_first_entry_window_error_rate",
    "any_qualified_lineage_survival_rate",
    "mean_prequential_reward",
    "all_surviving_qualified_lineage_entry_composed_accuracy",
    "all_dormant_probe_composed_accuracy",
    "all_dormant_probe_composed_minus_zero_helper_accuracy",
    "all_dormant_probe_composed_minus_zero_beneficiary_accuracy",
    "all_dormant_probe_composed_minus_role_swapped_accuracy",
)

DIAGNOSTIC_METRIC_IDS: tuple[str, ...] = (
    "best_dormant_composed_accuracy",
    "best_dormant_zero_helper_accuracy",
    "best_dormant_zero_beneficiary_accuracy",
    "best_dormant_role_swapped_accuracy",
    "selected_lineage_dormant_at_entry_rate",
    "selected_lineage_active_at_entry_rate",
)


def _metric_contracts() -> tuple[MetricContract, ...]:
    all_recurrence_eligibility = (
        "every evaluator-known episode entry after an earlier episode of that exact manifest "
        "regime and at least one complete intervening transition under a different regime; "
        "adjacent equal-regime segments are coalesced and are never recurrence entries"
    )
    qualified_recurrence_eligibility = (
        all_recurrence_eligibility
        + "; restricted to entries having at least one earlier synchronized commit for the same "
        "exact manifest regime with exact committed_composed_greedy_accuracy=1.0 and "
        "committed_composed_greedy_tie_free=true; "
        "qualification is fixed before recurrence outcomes"
    )
    return (
        MetricContract(
            metric_id="qualified_first_entry_window_error_rate",
            role="primary",
            orientation="lower",
            gate_mode="level_and_contrast",
            source_fields=(
                "summary.recurrence_retention[*].lineage_retention_applicable",
                "summary.recurrence_retention[*].first_world_window_errors",
                "summary.recurrence_retention[*].first_world_window_length",
                "summary.recurrence_retention[*].first_world_window_complete",
            ),
            aggregation=(
                "sum errors divided by sum window lengths across complete exact lease-length "
                "windows, then one value per matched run"
            ),
            eligibility=qualified_recurrence_eligibility,
            missingness=(
                "a recurrence without an earlier qualified lineage is acquisition-unqualified and "
                "reported as missing_from_retention_n; an incomplete window is separately missing"
            ),
            null_value_decimal="0.6666666666666667",
        ),
        MetricContract(
            metric_id="latest_prior_qualified_lineage_survival_rate",
            role="primary",
            orientation="higher",
            gate_mode="level_and_contrast",
            source_fields=(
                "summary.retention.latest_qualified_version_survival_count",
                "summary.retention.latest_qualified_version_survival_denominator",
                "summary.retention.latest_qualified_version_survival_missing_count",
                "summary.retention.latest_qualified_version_survival_fraction",
                "summary.recurrence_retention[*].lineage_retention_applicable",
                "summary.recurrence_retention[*].latest_prior_qualified_lineage_index",
                "summary.recurrence_retention[*].latest_prior_qualified_commit_step",
                "summary.recurrence_retention[*].latest_prior_qualified_survived",
            ),
            aggregation=(
                "fraction of acquisition-qualified recurrences where the latest prior qualified "
                "same-regime generation remains present in both roles at entry"
            ),
            eligibility=qualified_recurrence_eligibility,
            missingness=(
                "no earlier qualified lineage is missing_from_retention_n, not a forgetting "
                "failure; an evicted latest qualified lineage contributes zero survival"
            ),
            null_value_decimal="0",
        ),
        MetricContract(
            metric_id="selected_lineage_joint_bit_exact_preservation_rate",
            role="primary",
            orientation="higher",
            gate_mode="level_and_contrast",
            source_fields=(
                "summary.retention.selected_joint_bit_exact_preservation_count",
                "summary.retention.selected_bit_exact_preservation_all_qualified_denominator",
                "summary.retention.selected_joint_bit_exact_preservation_fraction_all_qualified",
                "summary.recurrence_retention[*].lineage_retention_applicable",
                "summary.recurrence_retention[*].selected_lineage_available",
                "summary.recurrence_retention[*].selected_lineage_index",
                "summary.recurrence_retention[*].selected_lineage_joint_bit_exact_preserved",
            ),
            aggregation=(
                "fraction of acquisition-qualified recurrences where the selected latest surviving "
                "qualified generation exists and every helper and beneficiary value bit equals "
                "its synchronized commit snapshot"
            ),
            eligibility=qualified_recurrence_eligibility,
            missingness=(
                "within the qualified denominator, no surviving lineage or any bit mismatch "
                "contributes zero; acquisition-unqualified recurrences are reported separately"
            ),
            null_value_decimal="0",
        ),
        MetricContract(
            metric_id="selected_lineage_entry_composed_accuracy",
            role="primary",
            orientation="higher",
            gate_mode="level_and_contrast",
            source_fields=(
                "summary.retention.selected_entry_metric_denominator",
                "summary.retention.selected_entry_composed_greedy_accuracy_mean",
                "summary.recurrence_retention[*].selected_lineage_available",
                "summary.recurrence_retention[*].selected_lineage_index",
                "summary.recurrence_retention[*].selected_lineage_generation",
                "summary.recurrence_retention[*].selected_lineage_entry_composed_greedy_accuracy",
            ),
            aggregation=(
                "arithmetic mean across recurrence entries having a selected latest surviving "
                "qualified lineage"
            ),
            eligibility=qualified_recurrence_eligibility,
            missingness=(
                "no selected lineage is recorded separately as survival failure and is missing "
                "for conditional accuracy; both counts must be reported"
            ),
            null_value_decimal="0.3333333333333333",
        ),
        MetricContract(
            metric_id="selected_lineage_commit_to_entry_accuracy_change",
            role="primary",
            orientation="higher",
            gate_mode="contrast_only",
            source_fields=(
                "summary.retention.selected_entry_metric_denominator",
                "summary.retention.selected_entry_minus_commit_accuracy_mean",
                "summary.commit_generation_lineages[*].lineage_index",
                "summary.commit_generation_lineages[*].committed_composed_greedy_accuracy",
                "summary.recurrence_retention[*].selected_lineage_available",
                "summary.recurrence_retention[*].selected_lineage_index",
                "summary.recurrence_retention[*].selected_lineage_entry_minus_commit_accuracy",
            ),
            aggregation=(
                "arithmetic mean of selected_lineage_entry_composed_greedy_accuracy minus "
                "committed_composed_greedy_accuracy for the exact selected generation"
            ),
            eligibility=qualified_recurrence_eligibility,
            missingness="missing whenever no lineage survives; never choose another lineage",
            null_value_decimal=None,
        ),
        MetricContract(
            metric_id="selected_lineage_exact_generation_relock_rate",
            role="primary",
            orientation="higher",
            gate_mode="level_and_contrast",
            source_fields=(
                "summary.retention.selected_exact_generation_relock_count",
                "summary.retention.selected_exact_generation_relock_all_qualified_denominator",
                "summary.retention.selected_exact_generation_relock_fraction_all_qualified",
                "summary.recurrence_retention[*].selected_lineage_available",
                "summary.recurrence_retention[*].selected_exact_generation_relock_observed",
                "summary.recurrence_retention[*].selected_first_exact_generation_relock_step",
                "summary.recurrence_retention[*].selected_first_exact_generation_relock_segment_step",
                "summary.recurrence_retention[*].selected_exact_generation_relock_phase",
                "summary.recurrence_retention[*].selected_observed_learner_boundaries_until_relock",
            ),
            aggregation="fraction of acquisition-qualified recurrence entries with an exact relock",
            eligibility=qualified_recurrence_eligibility,
            missingness=(
                "within the qualified denominator, no surviving lineage or no relock contributes "
                "zero; acquisition-unqualified recurrences are reported separately"
            ),
            null_value_decimal="0",
        ),
        MetricContract(
            metric_id="selected_lineage_retrieval_before_scratch_rate",
            role="primary",
            orientation="higher",
            gate_mode="level_and_contrast",
            source_fields=(
                "summary.retention.selected_durable_retrieval_before_scratch_count",
                "summary.retention.selected_durable_retrieval_before_scratch_all_qualified_denominator",
                "summary.retention.selected_durable_retrieval_before_scratch_fraction_all_qualified",
                "summary.recurrence_retention[*].selected_lineage_available",
                "summary.recurrence_retention[*].selected_first_scratch_entry_step",
                "summary.recurrence_retention[*].selected_first_scratch_entry_segment_step",
                "summary.recurrence_retention[*].selected_first_scratch_entry_phase",
                "summary.recurrence_retention[*].selected_scratch_entered_before_relock",
                "summary.recurrence_retention[*].selected_scratch_entered_before_relock_or_segment_end",
                "summary.recurrence_retention[*].selected_durable_retrieval_before_scratch",
            ),
            aggregation=(
                "fraction of acquisition-qualified recurrence entries retrieving the selected "
                "exact generation before either role enters scratch"
            ),
            eligibility=qualified_recurrence_eligibility,
            missingness=(
                "within the qualified denominator, no surviving lineage or no retrieval "
                "contributes zero; acquisition-unqualified recurrences are reported separately"
            ),
            null_value_decimal="0",
        ),
        MetricContract(
            metric_id="recurrence_minus_latest_qualified_acquisition_error_rate",
            role="primary",
            orientation="lower",
            gate_mode="contrast_only",
            source_fields=(
                "summary.retention.latest_qualified_acquisition_comparison_available_count",
                "summary.retention.latest_qualified_acquisition_comparison_denominator",
                "summary.retention.latest_qualified_acquisition_comparison_missing_count",
                "summary.retention.latest_qualified_acquisition_comparison_not_applicable_count",
                "summary.retention.recurrence_minus_latest_qualified_acquisition_error_rate_delta_mean",
                "summary.commit_generation_lineages[*].commit_step",
                "summary.commit_generation_lineages[*].commit_segment_index",
                "summary.commit_generation_lineages[*].acquisition_qualified",
                "summary.recurrence_retention[*].lineage_retention_applicable",
                "summary.recurrence_retention[*].prior_same_regime_lineages[*].lineage_index",
                "summary.recurrence_retention[*].prior_same_regime_lineages[*].commit_step",
                "summary.recurrence_retention[*].prior_same_regime_lineages[*].commit_segment_index",
                "summary.recurrence_retention[*].prior_same_regime_lineages[*].acquisition_qualified",
                "summary.recurrence_retention[*].latest_prior_qualified_lineage_index",
                "summary.recurrence_retention[*].latest_prior_qualified_commit_step",
                "summary.recurrence_retention[*].latest_qualified_acquisition_segment_index",
                "summary.recurrence_retention[*].latest_qualified_acquisition_episode_length",
                "summary.recurrence_retention[*].latest_qualified_acquisition_world_window_complete",
                "summary.recurrence_retention[*].latest_qualified_acquisition_world_window_errors",
                "summary.recurrence_retention[*].latest_qualified_acquisition_world_window_error_rate",
                "summary.recurrence_retention[*].latest_qualified_acquisition_comparison_available",
                "summary.recurrence_retention[*].first_world_window_complete",
                "summary.recurrence_retention[*].first_world_window_error_rate",
                "summary.recurrence_retention[*].recurrence_minus_latest_qualified_acquisition_error_rate_delta",
            ),
            aggregation=(
                "arithmetic mean of recurrence entry error rate minus the entry error rate of "
                "the coalesced episode containing the exact latest prior acquisition-qualified "
                "commit, over complete paired lease-length windows"
            ),
            eligibility=qualified_recurrence_eligibility,
            missingness=(
                "no prior qualified commit is not_applicable; a qualified recurrence with either "
                "incomplete paired window is missing; both counts and the qualified denominator "
                "are mandatory"
            ),
            null_value_decimal=None,
        ),
        MetricContract(
            metric_id="acquisition_qualified_recurrence_coverage_rate",
            role="secondary",
            orientation="higher",
            gate_mode="level_and_contrast",
            source_fields=(
                "summary.retention.lineage_retention_applicable_count",
                "summary.retention.acquisition_coverage_failure_count",
                "summary.retention.qualification_coverage_denominator",
                "summary.retention.qualification_coverage_fraction",
                "summary.recurrence_retention[*].lineage_retention_applicable",
                "summary.recurrence_retention[*].acquisition_coverage_failure",
                "summary.recurrence_retention[*].prior_same_regime_lineages[*].acquisition_qualified",
            ),
            aggregation=(
                "fraction of all evaluator-known recurrence entries with at least one exact "
                "acquisition-qualified prior lineage"
            ),
            eligibility=all_recurrence_eligibility,
            missingness="only an absent or invalid recurrence record is missing",
            null_value_decimal="0",
        ),
        MetricContract(
            metric_id="all_recurrence_first_entry_window_error_rate",
            role="secondary",
            orientation="lower",
            gate_mode="contrast_only",
            source_fields=(
                "summary.retention.complete_first_world_window_count",
                "summary.retention.missing_first_world_window_count",
                "summary.retention.first_world_window_error_rate_mean",
                "summary.recurrence_retention[*].first_world_window_errors",
                "summary.recurrence_retention[*].first_world_window_length",
                "summary.recurrence_retention[*].first_world_window_complete",
            ),
            aggregation=(
                "sum errors divided by sum window lengths across all complete recurrence entry "
                "windows, including acquisition-unqualified recurrences"
            ),
            eligibility=all_recurrence_eligibility,
            missingness="an incomplete required first-entry window is missing, never zero",
            null_value_decimal=None,
        ),
        MetricContract(
            metric_id="any_qualified_lineage_survival_rate",
            role="secondary",
            orientation="higher",
            gate_mode="contrast_only",
            source_fields=(
                "summary.retention.any_qualified_knowledge_survival_count",
                "summary.retention.any_qualified_knowledge_survival_denominator",
                "summary.retention.any_qualified_knowledge_survival_missing_count",
                "summary.retention.any_qualified_knowledge_survival_fraction",
                "summary.recurrence_retention[*].lineage_retention_applicable",
                "summary.recurrence_retention[*].any_prior_qualified_survived",
            ),
            aggregation=(
                "fraction of acquisition-qualified recurrences where at least one qualified prior "
                "same-regime exact generation remains present in both roles"
            ),
            eligibility=qualified_recurrence_eligibility,
            missingness=(
                "acquisition-unqualified recurrences are missing_from_retention_n; no surviving "
                "qualified lineage contributes zero"
            ),
            null_value_decimal="0",
        ),
        MetricContract(
            metric_id="mean_prequential_reward",
            role="secondary",
            orientation="higher",
            gate_mode="level_and_contrast",
            source_fields=("summary.mean_prequential_reward",),
            aggregation="one uninterrupted-life arithmetic mean per matched run",
            eligibility="all 240 matched cases",
            missingness="any absent or nonfinite reward invalidates the case",
            null_value_decimal="0.3333333333333333",
        ),
        MetricContract(
            metric_id="all_surviving_qualified_lineage_entry_composed_accuracy",
            role="secondary",
            orientation="higher",
            gate_mode="contrast_only",
            source_fields=(
                "summary.recurrence_retention[*].prior_same_regime_lineages[*].acquisition_qualified",
                "summary.recurrence_retention[*].prior_same_regime_lineages[*].synchronized_generation_survives",
                "summary.recurrence_retention[*].prior_same_regime_lineages[*].entry_composed_greedy_accuracy",
            ),
            aggregation=(
                "arithmetic mean over every surviving qualified lineage, while all evicted and "
                "surviving qualified lineages remain serialized; no best selection"
            ),
            eligibility=qualified_recurrence_eligibility,
            missingness="zero surviving qualified lineages is missing and separately counted",
            null_value_decimal=None,
        ),
        MetricContract(
            metric_id="all_dormant_probe_composed_accuracy",
            role="secondary",
            orientation="higher",
            gate_mode="contrast_only",
            source_fields=(
                "summary.recurrence_retention[*].eligible_dormant_generations[*].composed_greedy_accuracy",
            ),
            aggregation="arithmetic mean over every synchronized dormant-generation probe",
            eligibility=all_recurrence_eligibility,
            missingness="zero dormant probes is missing and separately counted",
            null_value_decimal=None,
        ),
        MetricContract(
            metric_id="all_dormant_probe_composed_minus_zero_helper_accuracy",
            role="secondary",
            orientation="higher",
            gate_mode="contrast_only",
            source_fields=(
                "summary.recurrence_retention[*].eligible_dormant_generations[*].composed_greedy_accuracy",
                "summary.recurrence_retention[*].eligible_dormant_generations[*].zero_helper_accuracy",
            ),
            aggregation="mean paired probe-level composed minus zero-helper accuracy",
            eligibility=all_recurrence_eligibility,
            missingness="zero dormant probes is missing and separately counted",
            null_value_decimal="0",
        ),
        MetricContract(
            metric_id="all_dormant_probe_composed_minus_zero_beneficiary_accuracy",
            role="secondary",
            orientation="higher",
            gate_mode="contrast_only",
            source_fields=(
                "summary.recurrence_retention[*].eligible_dormant_generations[*].composed_greedy_accuracy",
                "summary.recurrence_retention[*].eligible_dormant_generations[*].zero_beneficiary_accuracy",
            ),
            aggregation="mean paired probe-level composed minus zero-beneficiary accuracy",
            eligibility=all_recurrence_eligibility,
            missingness="zero dormant probes is missing and separately counted",
            null_value_decimal="0",
        ),
        MetricContract(
            metric_id="all_dormant_probe_composed_minus_role_swapped_accuracy",
            role="secondary",
            orientation="higher",
            gate_mode="contrast_only",
            source_fields=(
                "summary.recurrence_retention[*].eligible_dormant_generations[*].composed_greedy_accuracy",
                "summary.recurrence_retention[*].eligible_dormant_generations[*].role_swapped_accuracy",
            ),
            aggregation="mean paired probe-level composed minus role-swapped accuracy",
            eligibility=all_recurrence_eligibility,
            missingness="zero dormant probes is missing and separately counted",
            null_value_decimal="0",
        ),
        MetricContract(
            metric_id="best_dormant_composed_accuracy",
            role="diagnostic",
            orientation="higher",
            gate_mode="diagnostic_only",
            source_fields=(
                "summary.recurrence_retention[*].best_dormant_composed_greedy_accuracy",
            ),
            aggregation="mean of post-hoc best dormant probe values",
            eligibility=all_recurrence_eligibility,
            missingness="reported explicitly",
            null_value_decimal=None,
        ),
        MetricContract(
            metric_id="best_dormant_zero_helper_accuracy",
            role="diagnostic",
            orientation="higher",
            gate_mode="diagnostic_only",
            source_fields=(
                "summary.recurrence_retention[*].best_dormant_zero_helper_accuracy",
            ),
            aggregation="mean zero-helper value paired to each post-hoc best probe",
            eligibility=all_recurrence_eligibility,
            missingness="reported explicitly",
            null_value_decimal=None,
        ),
        MetricContract(
            metric_id="best_dormant_zero_beneficiary_accuracy",
            role="diagnostic",
            orientation="higher",
            gate_mode="diagnostic_only",
            source_fields=(
                "summary.recurrence_retention[*].best_dormant_zero_beneficiary_accuracy",
            ),
            aggregation="mean zero-beneficiary value paired to each post-hoc best probe",
            eligibility=all_recurrence_eligibility,
            missingness="reported explicitly",
            null_value_decimal=None,
        ),
        MetricContract(
            metric_id="best_dormant_role_swapped_accuracy",
            role="diagnostic",
            orientation="higher",
            gate_mode="diagnostic_only",
            source_fields=(
                "summary.recurrence_retention[*].best_dormant_role_swapped_accuracy",
            ),
            aggregation="mean role-swapped value paired to each post-hoc best probe",
            eligibility=all_recurrence_eligibility,
            missingness="reported explicitly",
            null_value_decimal=None,
        ),
        MetricContract(
            metric_id="selected_lineage_dormant_at_entry_rate",
            role="diagnostic",
            orientation="higher",
            gate_mode="diagnostic_only",
            source_fields=(
                "summary.retention.selected_entry_metric_denominator",
                "summary.retention.selected_entry_dormant_count",
                "summary.recurrence_retention[*].selected_lineage_entry_activity_status",
            ),
            aggregation="fraction of selected lineages dormant in both roles immediately pre-entry",
            eligibility=qualified_recurrence_eligibility,
            missingness="no selected lineage is missing",
            null_value_decimal=None,
        ),
        MetricContract(
            metric_id="selected_lineage_active_at_entry_rate",
            role="diagnostic",
            orientation="higher",
            gate_mode="diagnostic_only",
            source_fields=(
                "summary.retention.selected_entry_metric_denominator",
                "summary.retention.selected_entry_active_count",
                "summary.recurrence_retention[*].selected_lineage_entry_activity_status",
            ),
            aggregation="fraction of selected lineages active in both roles immediately pre-entry",
            eligibility=qualified_recurrence_eligibility,
            missingness="no selected lineage is missing",
            null_value_decimal=None,
        ),
    )


def _factorial_estimands() -> tuple[EstimandContract, ...]:
    metrics = PRIMARY_METRIC_IDS + SECONDARY_METRIC_IDS
    formula_prefix = "z_m(x)=x for higher-is-better and z_m(x)=-x for lower-is-better; "
    population_rule = (
        "for qualification-conditioned unconditional metrics use the exact intersection of "
        "recurrence identities acquisition-qualified in every condition term and encode no "
        "survivor/relock/retrieval/bit-preservation as observed zero; for "
        "selected_lineage_entry_composed_accuracy and "
        "selected_lineage_commit_to_entry_accuracy_change use the narrower exact intersection of "
        "recurrence identities with a selected surviving qualified lineage in every condition "
        "term; for all-surviving-lineage or dormant-probe conditional diagnostics use the exact "
        "intersection with at least one corresponding observation in every term; acquisition "
        "coverage, all-recurrence, and whole-life metrics retain their full denominator; report "
        "all excluded identifiers and paired intersection coverage separately"
    )
    return (
        EstimandContract(
            estimand_id="immutability_evidence_primary",
            role="primary",
            formula=formula_prefix + "paired z_m(SE)-z_m(WE)",
            condition_terms=(("selective_full", 1), ("writable_evidence", -1)),
            population_rule=population_rule,
            metrics=metrics,
        ),
        EstimandContract(
            estimand_id="immutability_lru_replication",
            role="replication",
            formula=formula_prefix + "paired z_m(SL)-z_m(WL)",
            condition_terms=(("selective_lru", 1), ("writable_lru", -1)),
            population_rule=population_rule,
            metrics=metrics,
        ),
        EstimandContract(
            estimand_id="replacement_target_selective_secondary",
            role="secondary",
            formula=formula_prefix + "paired z_m(SE)-z_m(SL)",
            condition_terms=(("selective_full", 1), ("selective_lru", -1)),
            population_rule=population_rule,
            metrics=metrics,
        ),
        EstimandContract(
            estimand_id="replacement_target_writable_secondary",
            role="secondary",
            formula=formula_prefix + "paired z_m(WE)-z_m(WL)",
            condition_terms=(("writable_evidence", 1), ("writable_lru", -1)),
            population_rule=population_rule,
            metrics=metrics,
        ),
        EstimandContract(
            estimand_id="write_by_replacement_interaction_secondary",
            role="secondary",
            formula=(
                formula_prefix + "paired [z_m(SE)-z_m(WE)]-[z_m(SL)-z_m(WL)]"
            ),
            condition_terms=(
                ("selective_full", 1),
                ("writable_evidence", -1),
                ("selective_lru", -1),
                ("writable_lru", 1),
            ),
            population_rule=population_rule,
            metrics=metrics,
        ),
    )


def _control_estimands() -> tuple[EstimandContract, ...]:
    control_metrics = (
        "mean_prequential_reward",
        "all_recurrence_first_entry_window_error_rate",
        "acquisition_qualified_recurrence_coverage_rate",
    )
    return tuple(
        EstimandContract(
            estimand_id=f"selective_evidence_vs_{condition}",
            role="secondary",
            formula=(
                "z_m(x)=x for higher-is-better and z_m(x)=-x for lower-is-better; "
                f"paired z_m(SE)-z_m({condition})"
            ),
            condition_terms=(("selective_full", 1), (condition, -1)),
            population_rule=(
                "whole-life and all-recurrence metrics use all records from the matched seed pair"
            ),
            metrics=control_metrics,
        )
        for condition in CONTROL_CONDITION_ORDER
    )


def _paired_population_support_metrics() -> tuple[PairedPopulationSupportMetric, ...]:
    denominator = (
        "all frozen coalesced eligible recurrence identities for the assigned manifest: "
        "11 for calibration A, 11 for calibration B, and 12 for calibration C"
    )
    return tuple(
        metric
        for prefix, estimand_id, conditions in (
            (
                "se_we",
                "immutability_evidence_primary",
                ("selective_full", "writable_evidence"),
            ),
            (
                "sl_wl",
                "immutability_lru_replication",
                ("selective_lru", "writable_lru"),
            ),
        )
        for metric in (
            PairedPopulationSupportMetric(
                metric_id=f"{prefix}_paired_qualification_intersection_coverage_rate",
                estimand_id=estimand_id,
                conditions=conditions,
                source_fields=(
                    "summary.recurrence_retention[*].segment_index",
                    "summary.recurrence_retention[*].regime_id",
                    "summary.recurrence_retention[*].occurrence_index",
                    "summary.recurrence_retention[*].raw_segment_occurrence_index",
                    "summary.recurrence_retention[*].lineage_retention_applicable",
                ),
                numerator=(
                    "count of exact recurrence identities acquisition-qualified in every named "
                    "condition"
                ),
                denominator=denominator,
                per_seed_aggregation="numerator divided by assigned-manifest denominator",
                mandatory=True,
                orientation="higher",
                null_value_decimal="0",
            ),
            PairedPopulationSupportMetric(
                metric_id=f"{prefix}_paired_selected_survival_intersection_coverage_rate",
                estimand_id=estimand_id,
                conditions=conditions,
                source_fields=(
                    "summary.recurrence_retention[*].segment_index",
                    "summary.recurrence_retention[*].regime_id",
                    "summary.recurrence_retention[*].occurrence_index",
                    "summary.recurrence_retention[*].raw_segment_occurrence_index",
                    "summary.recurrence_retention[*].lineage_retention_applicable",
                    "summary.recurrence_retention[*].selected_lineage_available",
                ),
                numerator=(
                    "count of exact recurrence identities with a selected surviving qualified "
                    "lineage in every named condition"
                ),
                denominator=denominator,
                per_seed_aggregation="numerator divided by assigned-manifest denominator",
                mandatory=True,
                orientation="higher",
                null_value_decimal="0",
            ),
        )
    )


def _gate_families() -> tuple[GateFamilySpec, ...]:
    invalid = (
        "mandatory failure is a valid calibration rejection; no retry, seed replacement, "
        "threshold retuning, endpoint substitution, or promotion"
    )
    paired_components = (
        "pooled_n_equals_30",
        "pooled_oriented_mean_and_sample_sd",
        "pooled_one_sided_95_percent_paired_t_lower_bound",
        "three_manifest_strata_each_n_equals_10",
        "worst_manifest_mean_and_one_sided_95_percent_paired_t_lower_bound",
        "pooled_and_each_manifest_wins_ties_losses",
        "structural_missing_case_n_equals_zero",
        "frozen_twofold_margin_threshold_and_win_gate",
    )
    level_components = (
        "pooled_n_equals_30",
        "pooled_oriented_mean_and_one_sided_95_percent_bound",
        "three_manifest_strata_each_n_equals_10",
        "worst_manifest_oriented_bound",
        "wins_ties_losses_relative_to_metric_null",
        "structural_missing_case_n_equals_zero",
        "frozen_twofold_margin_level_and_win_gate",
    )
    qualified_rule = (
        "qualification-conditioned paired metrics use their exact predeclared recurrence "
        "intersection, never separate condition subsets; serialize included/excluded identifiers "
        "and report eligible, acquisition_unqualified, conditional_unobserved, and "
        "structural_missing counts separately; only structural_missing cases violate the "
        "zero-missing invariant, while coverage and survival failures enter mandatory endpoints"
    )
    return (
        GateFamilySpec(
            gate_family_id="candidate_primary_level_gates",
            mandatory=True,
            conditions=("selective_full",),
            metric_ids=PRIMARY_LEVEL_METRIC_IDS,
            estimand_ids=(),
            required_components=level_components,
            null_rule=(
                "use each MetricContract.null_value_decimal and orientation; each condition/metric "
                "has a separate frozen level threshold"
            ),
            qualification_and_missingness_rule=qualified_rule,
            failure_disposition=invalid,
        ),
        GateFamilySpec(
            gate_family_id="acquisition_coverage_level_gates",
            mandatory=True,
            conditions=("selective_full",),
            metric_ids=("acquisition_qualified_recurrence_coverage_rate",),
            estimand_ids=(),
            required_components=level_components,
            null_rule="higher-is-better level null is exactly 0",
            qualification_and_missingness_rule=(
                "denominator is every non-adjacent evaluator-known recurrence; zero qualified "
                "lineage is an observed coverage failure, never statistical missingness"
            ),
            failure_disposition=invalid,
        ),
        GateFamilySpec(
            gate_family_id="primary_paired_population_support_gates",
            mandatory=True,
            conditions=("selective_full", "writable_evidence"),
            metric_ids=(
                "se_we_paired_qualification_intersection_coverage_rate",
                "se_we_paired_selected_survival_intersection_coverage_rate",
            ),
            estimand_ids=("immutability_evidence_primary",),
            required_components=level_components,
            null_rule=(
                "each higher-is-better paired-population support level has exact null 0 and a "
                "separate frozen twofold-margin threshold"
            ),
            qualification_and_missingness_rule=(
                "denominator is every frozen eligible recurrence identity for the assigned "
                "manifest; no common qualified/surviving lineage contributes observed zero"
            ),
            failure_disposition=invalid,
        ),
        GateFamilySpec(
            gate_family_id="primary_immutability_contrast_gates",
            mandatory=True,
            conditions=("selective_full", "writable_evidence"),
            metric_ids=PRIMARY_METRIC_IDS,
            estimand_ids=("immutability_evidence_primary",),
            required_components=paired_components,
            null_rule=(
                "every oriented paired contrast, including contrast-only metrics, has exact null 0"
            ),
            qualification_and_missingness_rule=qualified_rule,
            failure_disposition=invalid,
        ),
        GateFamilySpec(
            gate_family_id="replication_paired_population_support_gates",
            mandatory=True,
            conditions=("selective_lru", "writable_lru"),
            metric_ids=(
                "sl_wl_paired_qualification_intersection_coverage_rate",
                "sl_wl_paired_selected_survival_intersection_coverage_rate",
            ),
            estimand_ids=("immutability_lru_replication",),
            required_components=level_components,
            null_rule=(
                "each higher-is-better paired-population support level has exact null 0 and a "
                "separate frozen twofold-margin threshold"
            ),
            qualification_and_missingness_rule=(
                "denominator is every frozen eligible recurrence identity for the assigned "
                "manifest; no common qualified/surviving lineage contributes observed zero"
            ),
            failure_disposition=invalid,
        ),
        GateFamilySpec(
            gate_family_id="lru_immutability_replication_gates",
            mandatory=True,
            conditions=("selective_lru", "writable_lru"),
            metric_ids=PRIMARY_METRIC_IDS,
            estimand_ids=("immutability_lru_replication",),
            required_components=paired_components,
            null_rule=(
                "every oriented paired contrast, including contrast-only metrics, has exact null "
                "0; this mandatory mechanism replication tests write-policy invariance under LRU "
                "but does not require absolute LRU success"
            ),
            qualification_and_missingness_rule=qualified_rule,
            failure_disposition=invalid,
        ),
        GateFamilySpec(
            gate_family_id="selective_lru_absolute_levels_descriptive",
            mandatory=False,
            conditions=("selective_lru",),
            metric_ids=PRIMARY_LEVEL_METRIC_IDS
            + ("acquisition_qualified_recurrence_coverage_rate",),
            estimand_ids=(),
            required_components=level_components[:-1],
            null_rule=(
                "descriptive metric-specific nulls only; LRU is a replacement-policy ablation and "
                "its absolute weakness cannot reject the selective-evidence candidate"
            ),
            qualification_and_missingness_rule=qualified_rule,
            failure_disposition="descriptive only; cannot accept or reject calibration",
        ),
        GateFamilySpec(
            gate_family_id="causal_control_contrast_gates",
            mandatory=True,
            conditions=("selective_full",) + CONTROL_CONDITION_ORDER,
            metric_ids=(
                "mean_prequential_reward",
                "all_recurrence_first_entry_window_error_rate",
            ),
            estimand_ids=tuple(
                f"selective_evidence_vs_{condition}" for condition in CONTROL_CONDITION_ORDER
            ),
            required_components=paired_components,
            null_rule="every oriented paired causal-control contrast has exact null 0",
            qualification_and_missingness_rule=(
                "whole-life and coalesced all-recurrence records are paired by seed and manifest; "
                "structural missingness is forbidden"
            ),
            failure_disposition=invalid,
        ),
        GateFamilySpec(
            gate_family_id="mandatory_trace_and_lifecycle_audits",
            mandatory=True,
            conditions=CANONICAL_CONDITION_ORDER,
            metric_ids=(),
            estimand_ids=(),
            required_components=tuple(
                audit.requirement_id for audit in _audit_requirements()
            ),
            null_rule="not statistical: every named audit predicate must be exactly true",
            qualification_and_missingness_rule="no audit record may be absent or indeterminate",
            failure_disposition=invalid,
        ),
        GateFamilySpec(
            gate_family_id="replacement_and_interaction_descriptive",
            mandatory=False,
            conditions=(
                "selective_full",
                "writable_evidence",
                "selective_lru",
                "writable_lru",
            ),
            metric_ids=PRIMARY_METRIC_IDS + SECONDARY_METRIC_IDS,
            estimand_ids=(
                "replacement_target_selective_secondary",
                "replacement_target_writable_secondary",
                "write_by_replacement_interaction_secondary",
            ),
            required_components=paired_components[:-1],
            null_rule="descriptive oriented contrast reference is 0; no gate may be frozen",
            qualification_and_missingness_rule=qualified_rule,
            failure_disposition="descriptive only; cannot accept or reject calibration",
        ),
        GateFamilySpec(
            gate_family_id="probe_and_posthoc_best_diagnostics",
            mandatory=False,
            conditions=(
                "selective_full",
                "writable_evidence",
                "selective_lru",
                "writable_lru",
            ),
            metric_ids=(
                "all_dormant_probe_composed_accuracy",
                "all_dormant_probe_composed_minus_zero_helper_accuracy",
                "all_dormant_probe_composed_minus_zero_beneficiary_accuracy",
                "all_dormant_probe_composed_minus_role_swapped_accuracy",
            )
            + DIAGNOSTIC_METRIC_IDS,
            estimand_ids=(),
            required_components=(
                "counts",
                "means",
                "manifest_strata",
                "missingness",
            ),
            null_rule="descriptive only; post-hoc best values have no gate or null test",
            qualification_and_missingness_rule=(
                "serialize every probe and report missingness; no post-hoc best can replace lineage"
            ),
            failure_disposition="descriptive only; cannot accept or reject calibration",
        ),
    )


def _audit_requirements() -> tuple[AuditRequirement, ...]:
    invalid = "invalidate the calibration case; no imputation, retry, or seed replacement"
    return (
        AuditRequirement(
            "lineage_serialization",
            "every synchronized durable commit and every evaluator-known recurrence",
            (
                "serialize every earlier synchronized same-regime commit lineage, including "
                "unqualified and evicted lineages, in ascending "
                "(commit_step, slot, generation, lineage_index) order; independently reconstruct "
                "the expected collection from the complete commit ledger; acquisition "
                "qualification is a per-lineage filter requiring "
                "synchronized commit and "
                "exact committed_composed_greedy_accuracy=1.0 and "
                "committed_composed_greedy_tie_free=true; separately mark "
                "the latest prior qualified lineage with exact "
                "summary.recurrence_retention[*].latest_prior_qualified_lineage_index and "
                "select the latest surviving qualified lineage by commit_step before reading "
                "recurrence rewards or accuracies; "
                "join recurrence probes to the global commit ledger by exact lineage_index; "
                "all CommitGenerationLineage fields are mandatory: "
                + ",".join(_COMMIT_GENERATION_LINEAGE_FIELDS)
                + "; all RecurrenceLineageProbe fields are mandatory: "
                + ",".join(_RECURRENCE_LINEAGE_PROBE_FIELDS)
            ),
            invalid,
        ),
        AuditRequirement(
            "both_roles_learning",
            "SE, WE, SL, and WL in every seed pair",
            (
                "helper and beneficiary each have positive value-write and effective-learning "
                "counts and summary.both_roles_learned is true"
            ),
            invalid,
        ),
        AuditRequirement(
            "atomic_c_old_to_c_new_replacement",
            "SE in every seed pair; SL/WE/WL replacement outcomes are descriptive",
            (
                "exactly one synchronized target slot replaces C-old with C-new; helper and "
                "beneficiary retire the same old slot/generation and commit the same incremented "
                "generation on one lease boundary, with no half-updated trace state"
            ),
            invalid,
        ),
        AuditRequirement(
            "d_short_non_displacement",
            "SE in every seed pair; SL/WE/WL displacement outcomes are descriptive",
            (
                "the short D exposure causes no durable commit, retirement, generation change, "
                "or durable-bank displacement in either role"
            ),
            invalid,
        ),
        AuditRequirement(
            "constant_resource",
            "all 240 matched cases at every trace step",
            (
                "each role remains exactly 69 scalars/276 bytes and the dyad exactly 138 "
                "scalars/552 bytes; all eight conditions use the same state shape and budget"
            ),
            invalid,
        ),
        AuditRequirement(
            "complete_role_lifecycle_oracle",
            "every helper and beneficiary transition",
            (
                "independent host oracle reconstructs full pre/post role state, decision, writes, "
                "lease/search/retest/commit/replacement events and sequence continuity without "
                "calling learner update"
            ),
            invalid,
        ),
        AuditRequirement(
            "complete_world_oracle",
            "every world transition and channel delivery",
            (
                "independent host oracle reconstructs cue, schedule position, regime target, "
                "reward, direct/constant/shuffled delivery, RNG-key advance and state continuity "
                "plus actual terminated and discount fields without calling world step or delivery"
            ),
            invalid,
        ),
        AuditRequirement(
            "source_bound_trace_contract",
            "every case",
            (
                "trace binds manifest digest, seed pair, condition policies, full initial/final "
                "world and role state, role-private inputs, actions, common reward, diagnostics, "
                "and both oracle validations"
            ),
            invalid,
        ),
        AuditRequirement(
            "decentralized_role_equivalence",
            "every case",
            (
                "separate local helper/beneficiary updates using only local state/input/action, "
                "common reward and frozen lifecycle permission reproduce the joint trajectory"
            ),
            invalid,
        ),
        AuditRequirement(
            "checkpoint_resume_equivalence",
            "cuts inside leases, search, retest, commit, replacement, and regime changes",
            "resumed trace and final world/dyad state are bit-exact to uninterrupted execution",
            invalid,
        ),
        AuditRequirement(
            "frozen_role_causal_controls",
            "helper_frozen and beneficiary_frozen",
            (
                "named frozen role has zero learning writes/commits/replacements while the other "
                "role retains its normal resource allocation; no parameter or memory is removed"
            ),
            invalid,
        ),
        AuditRequirement(
            "channel_causal_controls",
            "constant_channel_0 and shuffled_channel",
            (
                "constant delivery is always zero; shuffled delivery is independently oracle-"
                "reconstructed; both retain the same dyad state budget and learner permissions"
            ),
            invalid,
        ),
    )


def _statistical_plan() -> StatisticalSummaryPlan:
    return StatisticalSummaryPlan(
        paired_unit=(
            "one literal world/learner seed pair assigned to one manifest; all eight conditions "
            "share that world definition and seed pair"
        ),
        paired_recurrence_alignment=(
            "never compare conditional means over different recurrence subsets: use the exact "
            "per-metric intersection specified by EstimandContract.population_rule, serialize "
            "included/excluded recurrence identities, and separately gate paired qualification "
            "and selected-survival intersection coverage"
        ),
        orientation_transform=(
            "z_m(x)=x for higher-is-better metrics and z_m(x)=-x for lower-is-better metrics"
        ),
        pooled_expected_n=30,
        manifest_expected_n=10,
        confidence_basis_points=9500,
        standard_deviation="sample standard deviation with Bessel correction (denominator n-1)",
        standard_error="sample_standard_deviation/sqrt(n)",
        one_sided_lower_bound=(
            "mean(oriented paired deltas)-t_quantile(0.95,df=n-1)*standard_error"
        ),
        win_definition="oriented paired delta strictly greater than zero",
        tie_definition="oriented paired delta exactly equal to zero before display rounding",
        loss_definition="oriented paired delta strictly less than zero",
        missingness_policy=(
            "report eligible_n, observed_n, missing_n and missing identifiers; no imputation, "
            "seed replacement, condition replacement, or pairwise metric substitution; every "
            "mandatory gate requires missing_n=0"
        ),
        pooled_summary=(
            "observed_n",
            "missing_n",
            "mean_oriented_delta",
            "sample_standard_deviation",
            "standard_error",
            "one_sided_95_percent_lower_confidence_bound",
            "wins",
            "ties",
            "losses",
        ),
        manifest_stratified_summary=(
            "manifest_name",
            "observed_n",
            "missing_n",
            "mean_oriented_delta",
            "one_sided_95_percent_lower_confidence_bound",
            "wins",
            "ties",
            "losses",
        ),
        worst_manifest_summary=(
            "minimum_manifest_mean_oriented_delta",
            "minimum_manifest_one_sided_95_percent_lower_confidence_bound",
            "minimum_manifest_wins",
            "maximum_manifest_missing_n",
        ),
        multiplicity_scope=(
            "all predeclared gates are conjunctive; bounds are per-endpoint descriptive 95% "
            "bounds and no familywise probability claim is made"
        ),
    )


def _threshold_rule() -> ThresholdFreezeRule:
    return ThresholdFreezeRule(
        status="thresholds_unset_pending_consumed_calibration_outcomes",
        frozen_thresholds=(),
        threshold_freeze_receipt=None,
        minimum_margin_ratio_numerator=2,
        minimum_margin_ratio_denominator=1,
        rounding_quantum_decimal="0.0001",
        higher_is_better_rule=(
            "let B be the conservative favorable bound and N the predeclared null; require B>N; "
            "set q=floor_to_0.0001((B-N)/2), require q>=0.0001, and freeze T=N+q, "
            "so B-N>=2*(T-N); a null-equivalent rounded gate cannot freeze"
        ),
        lower_is_better_rule=(
            "let B be the conservative favorable upper bound and N the predeclared adverse null; "
            "require B<N; set q=floor_to_0.0001((N-B)/2), require q>=0.0001, and freeze "
            "T=N-q (rounding toward N), so N-B>=2*(N-T); a null-equivalent gate cannot freeze"
        ),
        wins_rule=(
            "for each even-n directional paired-win gate use W0=n/2 (pooled n=30 gives 15; "
            "manifest n=10 gives 5); with W>W0 set q=floor((W-W0)/2), require integer q>=1, "
            "and freeze T=W0+q; ties are not wins and a null-equivalent gate cannot freeze"
        ),
        mandatory_missingness_rule="freeze missing_n threshold at exactly zero",
        conservative_bound_rule=(
            "B is the less favorable of the pooled one-sided 95% bound and the worst of the "
            "three manifest-stratified one-sided 95% bounds; a nonfinite or missing bound cannot "
            "freeze a threshold"
        ),
        receipt_schema=THRESHOLD_FREEZE_RECEIPT_SCHEMA,
        receipt_required_fields=(
            "receipt_schema",
            "protocol_payload_sha256",
            "seed_snapshot_sha256",
            "readiness_receipt_sha256",
            "gate_matrix_sha256",
            "calibration_outcomes_payload_sha256",
            "source_closure_sha256",
            "environment_identity_sha256",
            "frozen_thresholds",
            "mandatory_gate_results",
            "descriptive_only_results",
            "rounding_worked_examples",
            "all_calibration_seeds_consumed",
            "protected_namespace_derived",
            "protected_outcomes_observed",
            "scientific_promotion_allowed",
            "amendments_allowed",
            "receipt_payload_sha256",
        ),
        receipt_immutability=(
            "canonical JSON receipt is content-addressed and created exclusively at a new path; "
            "it is never overwritten, amended, or regenerated in place"
        ),
        post_protected_adjustment=(
            "forbidden: after any protected seed is derived or outcome is observed, thresholds, "
            "rounding, estimands, exclusions, missingness, manifests, conditions, or gates cannot "
            "change; a failed gate remains a valid rejection"
        ),
    )


@dataclass(frozen=True, slots=True)
class HiddenRegimeFactorialCalibrationDesign:
    """Fully frozen, outcome-free calibration design."""

    manifest_bindings: tuple[CalibrationManifestBinding, ...]
    recurrence_bindings: tuple[RecurrenceEligibilityBinding, ...]
    seed_pairs: tuple[FrozenSeedPair, ...]
    assignments: tuple[CalibrationAssignment, ...]
    cases: tuple[MatchedCalibrationCase, ...]
    factorial_cells: tuple[FactorialCell, ...]
    condition_runtime_bindings: tuple[ConditionRuntimeBinding, ...]
    base_config_binding: BaseEvaluatorConfigBinding
    metrics: tuple[MetricContract, ...]
    paired_population_support_metrics: tuple[PairedPopulationSupportMetric, ...]
    factorial_estimands: tuple[EstimandContract, ...]
    control_estimands: tuple[EstimandContract, ...]
    audits: tuple[AuditRequirement, ...]
    gate_families: tuple[GateFamilySpec, ...]
    statistical_plan: StatisticalSummaryPlan
    threshold_rule: ThresholdFreezeRule

    def __post_init__(self) -> None:
        if tuple(binding.name for binding in self.manifest_bindings) != (
            CALIBRATION_MANIFEST_ORDER
        ):
            raise ValueError("manifest bindings must follow frozen A/B/C order")
        if tuple(binding.manifest_payload_sha256 for binding in self.manifest_bindings) != (
            CALIBRATION_MANIFEST_PAYLOAD_SHA256
        ):
            raise ValueError("manifest binding digests do not match frozen values")
        if self.recurrence_bindings != _recurrence_bindings():
            raise ValueError("recurrence eligibility bindings differ from frozen coalescing")
        if self.seed_pairs != FROZEN_SEED_PAIRS:
            raise ValueError("seed pairs must equal the authoritative literal snapshot")
        expected_assignments = tuple(
            CalibrationAssignment(
                seed_index=index,
                manifest_name=CALIBRATION_MANIFEST_ORDER[index % 3],
            )
            for index in range(N_SEED_PAIRS)
        )
        if self.assignments != expected_assignments:
            raise ValueError("assignments must equal the frozen round-robin schedule")
        if len(self.cases) != N_MATCHED_CASES:
            raise ValueError("design must contain exactly 240 matched cases")
        expected_cases = _matched_cases(expected_assignments)
        if self.cases != expected_cases:
            raise ValueError("cases must equal the seed-major matched Cartesian schedule")
        if tuple(cell.code for cell in self.factorial_cells) != FACTORIAL_CELL_ORDER:
            raise ValueError("factorial cells must follow SE, WE, SL, WL order")
        if self.condition_runtime_bindings != _runtime_bindings():
            raise ValueError("all eight condition runtime bindings must equal frozen literals")
        for cell, runtime in zip(
            self.factorial_cells,
            self.condition_runtime_bindings[:4],
            strict=True,
        ):
            if (
                cell.condition,
                cell.durable_write_policy,
                cell.replacement_target_policy,
            ) != (
                runtime.condition,
                runtime.durable_write_policy,
                runtime.replacement_target_policy,
            ):
                raise ValueError("factorial cells and runtime bindings disagree")
        if self.base_config_binding != _base_config_binding():
            raise ValueError("base evaluator configuration differs from frozen binding")
        metric_ids = tuple(metric.metric_id for metric in self.metrics)
        expected_metric_ids = PRIMARY_METRIC_IDS + SECONDARY_METRIC_IDS + DIAGNOSTIC_METRIC_IDS
        if metric_ids != expected_metric_ids or len(set(metric_ids)) != len(metric_ids):
            raise ValueError("metric contracts must equal the frozen unique order")
        if self.paired_population_support_metrics != _paired_population_support_metrics():
            raise ValueError("paired-population support metrics differ from frozen scope")
        referenced_metrics = {
            metric
            for estimand in self.factorial_estimands + self.control_estimands
            for metric in estimand.metrics
        }
        if not referenced_metrics.issubset(set(metric_ids)):
            raise ValueError("an estimand references an unknown metric")
        if tuple(item.estimand_id for item in self.factorial_estimands) != (
            "immutability_evidence_primary",
            "immutability_lru_replication",
            "replacement_target_selective_secondary",
            "replacement_target_writable_secondary",
            "write_by_replacement_interaction_secondary",
        ):
            raise ValueError("factorial estimands must equal the frozen order")
        expected_terms = {
            "immutability_evidence_primary": (
                ("selective_full", 1),
                ("writable_evidence", -1),
            ),
            "immutability_lru_replication": (("selective_lru", 1), ("writable_lru", -1)),
            "replacement_target_selective_secondary": (
                ("selective_full", 1),
                ("selective_lru", -1),
            ),
            "replacement_target_writable_secondary": (
                ("writable_evidence", 1),
                ("writable_lru", -1),
            ),
            "write_by_replacement_interaction_secondary": (
                ("selective_full", 1),
                ("writable_evidence", -1),
                ("selective_lru", -1),
                ("writable_lru", 1),
            ),
        }
        if {
            item.estimand_id: item.condition_terms for item in self.factorial_estimands
        } != expected_terms:
            raise ValueError("factorial estimand condition terms differ from frozen coefficients")
        if len(self.control_estimands) != len(CONTROL_CONDITION_ORDER):
            raise ValueError("one control estimand is required per causal control")
        if tuple(
            item.condition_terms for item in self.control_estimands
        ) != tuple(
            (("selective_full", 1), (condition, -1))
            for condition in CONTROL_CONDITION_ORDER
        ):
            raise ValueError("control estimand terms differ from frozen matched contrasts")
        if len({audit.requirement_id for audit in self.audits}) != len(self.audits):
            raise ValueError("audit requirement identifiers must be unique")
        if self.gate_families != _gate_families():
            raise ValueError("mandatory/descriptive gate matrix differs from frozen scope")
        metric_id_set = set(metric_ids) | {
            item.metric_id for item in self.paired_population_support_metrics
        }
        estimand_id_set = {
            item.estimand_id for item in self.factorial_estimands + self.control_estimands
        }
        estimand_conditions = {
            item.estimand_id: tuple(term[0] for term in item.condition_terms)
            for item in self.factorial_estimands + self.control_estimands
        }
        if any(
            item.estimand_id not in estimand_id_set
            or item.conditions != estimand_conditions[item.estimand_id]
            for item in self.paired_population_support_metrics
        ):
            raise ValueError("paired support metric conditions disagree with their estimand")
        if any(
            not set(family.metric_ids).issubset(metric_id_set)
            or not set(family.estimand_ids).issubset(estimand_id_set)
            for family in self.gate_families
        ):
            raise ValueError("gate matrix references an unknown metric or estimand")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> HiddenRegimeFactorialCalibrationDesign:
        """Strictly accept only the byte-canonical frozen payload."""

        if type(payload) is not dict:
            raise TypeError("design payload must be a plain dict")
        candidate_bytes = canonical_json_bytes(payload)
        expected = build_hidden_regime_factorial_calibration_design()
        expected_bytes = canonical_json_bytes(expected.to_payload())
        if candidate_bytes != expected_bytes:
            raise ValueError("design payload differs from the frozen canonical design")
        return expected

    def to_payload(self) -> dict[str, object]:
        """Return the complete JSON-compatible protocol payload."""

        gate_payload = [family.to_payload() for family in self.gate_families]
        return {
            "schema_version": DESIGN_SCHEMA,
            "protocol_status": PROTOCOL_STATUS,
            "status_time_scope": "facts_observed_when_this_design_was_frozen",
            "development_only": True,
            "outcomes_observed_at_design_freeze": False,
            "design_module_execution_api_available": False,
            "design_module_artifact_writer_available": False,
            "outcome_artifact_written_at_design_freeze": False,
            "scientific_promotion_allowed": False,
            "runtime_schema_bindings": {
                "development_summary_schema": BOUND_DEVELOPMENT_SUMMARY_SCHEMA,
                "primitive_trace_schema": BOUND_PRIMITIVE_TRACE_SCHEMA,
            },
            "namespace": {
                "value": CONSUMED_CALIBRATION_NAMESPACE,
                "derivation_schema": SEED_DERIVATION_SCHEMA,
                "disposition": (
                    "permanently_consumed_nonpromoting_at_design_freeze_even_before_execution"
                ),
                "outcomes_observed_at_design_freeze": False,
                "replay_only_after_first_execution": True,
                "promotion_eligible": False,
                "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
                "protected_seed_namespace": None,
            },
            "execution_policy": {
                "permitted_partition": CALIBRATION_ONLY_PARTITION,
                "authorization_rule": (
                    "calibration execution is permitted if and only if a separate "
                    "content-addressed readiness receipt validates every required certification"
                ),
                "readiness_receipt_schema": CALIBRATION_READINESS_RECEIPT_SCHEMA,
                "readiness_receipt_required_certifications": [
                    "protocol_payload_and_seed_snapshot_digests",
                    "calibration_manifest_payload_and_source_closure_digests",
                    "coalesced_nonadjacent_recurrence_binding_validator",
                    "generation_lineage_serialization_selection_and_validator",
                    "full_role_and_world_oracle_trace_audit_including_actual_terminated_and_discount",
                    "checkpoint_resume_and_decentralized_role_bit_exact_equivalence",
                    "runtime_schema_environment_identity_and_dependency_lock_digests",
                    "protected_candidate_outcome_ledger_false_before_calibration_execution",
                ],
                "protected_candidate_execution_permitted": False,
                "promotion_permitted": False,
            },
            "protected_candidate_guard": {
                "learner_outcomes_observed_at_design_freeze": False,
                "outcome_ledger_all_false_at_design_freeze": True,
                "manifest_names_serialized": False,
            },
            "manifest_bindings": [item.to_payload() for item in self.manifest_bindings],
            "recurrence_eligibility_bindings": [
                item.to_payload() for item in self.recurrence_bindings
            ],
            "seed_pairs": [pair.to_payload() for pair in self.seed_pairs],
            "assignment_rule": "manifest_order[seed_index modulo 3]",
            "assignments": [assignment.to_payload() for assignment in self.assignments],
            "condition_order": list(CANONICAL_CONDITION_ORDER),
            "condition_runtime_bindings": [
                item.to_payload() for item in self.condition_runtime_bindings
            ],
            "base_config_binding": self.base_config_binding.to_payload(),
            "matched_case_count": N_MATCHED_CASES,
            "cases": [case.to_payload() for case in self.cases],
            "factorial_cells": [cell.to_payload() for cell in self.factorial_cells],
            "factorial_estimands": [item.to_payload() for item in self.factorial_estimands],
            "causal_control_estimands": [item.to_payload() for item in self.control_estimands],
            "metric_contracts": [metric.to_payload() for metric in self.metrics],
            "paired_population_support_metrics": [
                metric.to_payload() for metric in self.paired_population_support_metrics
            ],
            "gate_mode_semantics": {
                "matrix_authority": "gate_families is exhaustive; no post-hoc scope is allowed",
                "level_and_contrast": (
                    "separate level and paired-contrast gates exist only where a mandatory gate "
                    "family explicitly lists the metric, conditions, and estimand"
                ),
                "contrast_only": (
                    "no level gate; each explicitly listed mandatory paired contrast uses null 0"
                ),
                "diagnostic_only": "never mandatory and never thresholded",
                "unlisted_metric_estimand_pair": "descriptive_only",
            },
            "gate_scope_rationale": {
                "main_candidate": "SE selective writes plus evidence replacement targeting",
                "mandatory_absolute_levels": "SE only",
                "lru_axis": (
                    "SL is a replacement-policy ablation: SL-vs-WL is a mandatory directional "
                    "write-policy replication, while SL absolute levels and evidence-like "
                    "replacement success are descriptive and cannot reject SE"
                ),
                "latest_qualified_acquisition_delta": (
                    "contrast_only: acquisition-qualified identifies an exact correct commit, "
                    "but its containing episode may be initial acquisition, relearning, or an "
                    "already-mastered refresh; therefore a strictly negative within-condition "
                    "recurrence-minus-episode delta is not an absolute continual-learning "
                    "requirement and has no level null, while its oriented matched-condition "
                    "contrasts remain predeclared"
                ),
                "replacement_contrasts_and_interaction": "descriptive_secondary",
            },
            "gate_families": gate_payload,
            "gate_matrix_sha256": canonical_sha256(gate_payload),
            "lineage_selection_contract": {
                "observer": "evaluator_only_not_learner_visible",
                "global_commit_collection_path": "summary.commit_generation_lineages",
                "global_commit_record_fields": list(_COMMIT_GENERATION_LINEAGE_FIELDS),
                "global_commit_count_paths": [
                    "summary.synchronized_commit_lineage_count",
                    "summary.acquisition_qualified_commit_lineage_count",
                    "summary.acquisition_unqualified_commit_lineage_count",
                ],
                "recurrence_lineage_collection_path": (
                    "summary.recurrence_retention[*].prior_same_regime_lineages"
                ),
                "recurrence_lineage_probe_fields": list(
                    _RECURRENCE_LINEAGE_PROBE_FIELDS
                ),
                "join_key": "lineage_index exact inner identity join to global commit collection",
                "acquisition_qualification": (
                    "earlier synchronized same-manifest-regime commit with exact "
                    "committed_composed_greedy_accuracy=1.0 and "
                    "committed_composed_greedy_tie_free=true"
                ),
                "all_prior_generation_lineages_serialized": (
                    "all_including_unqualified_and_evicted"
                ),
                "prior_generation_lineage_order": [
                    "commit_step",
                    "slot",
                    "generation",
                    "lineage_index",
                ],
                "omission_detection": (
                    "independent validator derives every earlier synchronized same-regime commit "
                    "from the complete commit ledger and requires exact ordered identifier equality"
                ),
                "qualified_prior_lineages": (
                    "derived filter prior_same_regime_lineages[*].acquisition_qualified=true; "
                    "never a replacement serialized list"
                ),
                "latest_prior_qualified_lineage_index_path": (
                    "summary.recurrence_retention[*].latest_prior_qualified_lineage_index"
                ),
                "latest_prior_qualified_commit_step_path": (
                    "summary.recurrence_retention[*].latest_prior_qualified_commit_step"
                ),
                "latest_prior_qualified_lineage": (
                    "the exact lineage_index of the last acquisition-qualified prior probe under "
                    "prior_generation_lineage_order; its commit_step must equal the serialized "
                    "latest_prior_qualified_commit_step and it must join exactly to both the "
                    "recurrence probe collection and global commit collection"
                ),
                "latest_qualified_acquisition_episode_binding": (
                    "coalesced episode bounds containing the commit_segment_index of the exact "
                    "latest_prior_qualified_lineage_index; never first exposure or another "
                    "qualified lineage"
                ),
                "selected_probe_lineage": (
                    "maximum commit_step among qualified prior lineages whose exact slot and "
                    "generation remain present in both roles immediately pre-entry"
                ),
                "retention_denominator": (
                    "only recurrences with at least one qualified prior lineage; all other "
                    "recurrences are acquisition-unqualified and reported separately"
                ),
                "acquisition_coverage_denominator": (
                    "all manifest-bound non-adjacent evaluator-known recurrence episode entries"
                ),
                "non_substitution_rule": (
                    "acquisition-qualified coverage, latest-prior-qualified survival, "
                    "any-qualified survival, joint bit-exact preservation, behavioral accuracy, "
                    "and first-window adaptation are reported separately; no endpoint can "
                    "substitute for a failed lineage gate"
                ),
                "selection_uses_recurrence_performance": False,
                "posthoc_best_dormant_primary_allowed": False,
            },
            "audit_requirements": [audit.to_payload() for audit in self.audits],
            "resource_contract": {
                "per_role_scalars": EXPECTED_ROLE_STATE_SCALARS,
                "per_role_bytes": EXPECTED_ROLE_STATE_BYTES,
                "dyad_scalars": EXPECTED_DYAD_STATE_SCALARS,
                "dyad_bytes": EXPECTED_DYAD_STATE_BYTES,
                "constant_at_every_step": True,
                "matched_across_all_conditions": True,
            },
            "statistical_plan": self.statistical_plan.to_payload(),
            "threshold_freeze_rule": self.threshold_rule.to_payload(),
            "frozen_thresholds": [],
            "threshold_freeze_receipt": None,
            "claim_scope": (
                "nonpromoting calibration of a finite hidden-regime signaling factorial; no "
                "Alberta Plan completion, unbounded retention, or general continual-learning claim"
            ),
        }


def _matched_cases(
    assignments: tuple[CalibrationAssignment, ...],
) -> tuple[MatchedCalibrationCase, ...]:
    cases: list[MatchedCalibrationCase] = []
    for assignment in assignments:
        pair = FROZEN_SEED_PAIRS[assignment.seed_index]
        for condition_index, condition in enumerate(CANONICAL_CONDITION_ORDER):
            cases.append(
                MatchedCalibrationCase(
                    case_index=assignment.seed_index * N_CONDITIONS + condition_index,
                    seed_index=assignment.seed_index,
                    manifest_name=assignment.manifest_name,
                    world_seed=pair.world_seed,
                    learner_seed=pair.learner_seed,
                    condition=condition,
                )
            )
    return tuple(cases)


def _assert_external_partition_guards() -> None:
    if PROTECTED_CANDIDATE_LEARNER_OUTCOMES_EXECUTED is not False:
        raise RuntimeError("protected-candidate learner-outcome constant is no longer false")
    protected_entries = tuple(
        entry
        for entry in HIDDEN_REGIME_MANIFEST_USE_LEDGER.values()
        if entry.use_partition == PROTECTED_CANDIDATE_PARTITION
    )
    if not protected_entries or any(
        entry.learner_outcomes_executed is not False for entry in protected_entries
    ):
        raise RuntimeError("protected-candidate outcome ledger is not uniformly false")
    if set(HIDDEN_REGIME_CALIBRATION_MANIFESTS) != set(CALIBRATION_MANIFEST_ORDER):
        raise RuntimeError("calibration manifest registry differs from frozen design")
    for name in CALIBRATION_MANIFEST_ORDER:
        ledger = HIDDEN_REGIME_MANIFEST_USE_LEDGER[name]
        if (
            ledger.use_partition != CALIBRATION_ONLY_PARTITION
            or ledger.calibration_use_allowed is not True
            or ledger.protected_evaluation_candidate is not False
            or ledger.scientific_promotion_allowed is not False
        ):
            raise RuntimeError("calibration manifest ledger is not calibration-only")


def _manifest_bindings() -> tuple[CalibrationManifestBinding, ...]:
    bindings: list[CalibrationManifestBinding] = []
    for name, expected_digest in zip(
        CALIBRATION_MANIFEST_ORDER,
        CALIBRATION_MANIFEST_PAYLOAD_SHA256,
        strict=True,
    ):
        manifest = HIDDEN_REGIME_CALIBRATION_MANIFESTS[name]
        if manifest.use_partition != CALIBRATION_ONLY_PARTITION:
            raise RuntimeError("frozen manifest moved out of calibration-only partition")
        actual_digest = canonical_sha256(manifest.to_dict())
        if actual_digest != expected_digest:
            raise RuntimeError(f"calibration manifest content changed: {name}")
        bindings.append(
            CalibrationManifestBinding(
                name=name,
                use_partition=CALIBRATION_ONLY_PARTITION,
                manifest_payload_sha256=expected_digest,
            )
        )
    return tuple(bindings)


def _recurrence_bindings() -> tuple[RecurrenceEligibilityBinding, ...]:
    bindings: list[RecurrenceEligibilityBinding] = []
    for name in CALIBRATION_MANIFEST_ORDER:
        manifest = HIDDEN_REGIME_CALIBRATION_MANIFESTS[name]
        if sum(manifest.segment_lengths) != 16_528:
            raise RuntimeError("calibration manifest total is not the frozen 16,528 steps")
        episode_starts = tuple(
            index
            for index, regime in enumerate(manifest.segment_regimes)
            if index == 0 or manifest.segment_regimes[index - 1] != regime
        )
        observed_episode_counts: Counter[int] = Counter()
        recurrence_starts: list[int] = []
        recurrence_identities: list[tuple[int, int, int]] = []
        recurrence_counts: Counter[int] = Counter()
        for index in episode_starts:
            regime = manifest.segment_regimes[index]
            if observed_episode_counts[regime] > 0:
                recurrence_starts.append(index)
                recurrence_identities.append(
                    (index, regime, observed_episode_counts[regime])
                )
                recurrence_counts[regime] += 1
            observed_episode_counts[regime] += 1
        binding = RecurrenceEligibilityBinding(
            manifest_name=name,
            coalesced_episode_start_segment_indices=episode_starts,
            eligible_recurrence_start_segment_indices=tuple(recurrence_starts),
            eligible_recurrence_counts_by_regime=cast(
                tuple[int, int, int, int, int],
                tuple(recurrence_counts[regime] for regime in range(5)),
            ),
            eligible_recurrence_identities=tuple(recurrence_identities),
        )
        bindings.append(binding)
    return tuple(bindings)


def _validate_seed_snapshot() -> None:
    if any(token in CONSUMED_CALIBRATION_NAMESPACE for token in _PROHIBITED_NAMESPACE_TOKENS):
        raise RuntimeError("consumed calibration namespace contains a protected-use token")
    if tuple(pair.index for pair in FROZEN_SEED_PAIRS) != tuple(range(N_SEED_PAIRS)):
        raise RuntimeError("literal seed-pair indices are not exactly 0 through 29")
    flat_seeds = tuple(
        seed
        for pair in FROZEN_SEED_PAIRS
        for seed in (pair.world_seed, pair.learner_seed)
    )
    if len(set(flat_seeds)) != 2 * N_SEED_PAIRS:
        raise RuntimeError("all sixty world/learner uint32 seeds must be globally unique")
    for pair in FROZEN_SEED_PAIRS:
        if (pair.world_seed, pair.learner_seed) != derive_seed_pair_for_audit(pair.index):
            raise RuntimeError("literal seed pair differs from disclosed derivation")
    if canonical_sha256(seed_snapshot_payload()) != SEED_SNAPSHOT_SHA256:
        raise RuntimeError("literal seed snapshot digest differs from frozen digest")


def build_hidden_regime_factorial_calibration_design(
) -> HiddenRegimeFactorialCalibrationDesign:
    """Construct the frozen design without constructing a world or running a learner."""

    _assert_external_partition_guards()
    _validate_seed_snapshot()
    assignments = tuple(
        CalibrationAssignment(
            seed_index=index,
            manifest_name=CALIBRATION_MANIFEST_ORDER[index % 3],
        )
        for index in range(N_SEED_PAIRS)
    )
    design = HiddenRegimeFactorialCalibrationDesign(
        manifest_bindings=_manifest_bindings(),
        recurrence_bindings=_recurrence_bindings(),
        seed_pairs=FROZEN_SEED_PAIRS,
        assignments=assignments,
        cases=_matched_cases(assignments),
        factorial_cells=(
            FactorialCell("SE", "selective_full", "selective", "evidence"),
            FactorialCell("WE", "writable_evidence", "writable", "evidence"),
            FactorialCell("SL", "selective_lru", "selective", "lru"),
            FactorialCell("WL", "writable_lru", "writable", "lru"),
        ),
        condition_runtime_bindings=_runtime_bindings(),
        base_config_binding=_base_config_binding(),
        metrics=_metric_contracts(),
        paired_population_support_metrics=_paired_population_support_metrics(),
        factorial_estimands=_factorial_estimands(),
        control_estimands=_control_estimands(),
        audits=_audit_requirements(),
        gate_families=_gate_families(),
        statistical_plan=_statistical_plan(),
        threshold_rule=_threshold_rule(),
    )
    counts = Counter(assignment.manifest_name for assignment in design.assignments)
    if counts != Counter({name: 10 for name in CALIBRATION_MANIFEST_ORDER}):
        raise RuntimeError("calibration manifests are not assigned ten seed pairs each")
    return design


def calibration_design_payload() -> dict[str, object]:
    """Return a fresh canonicalizable payload for the frozen design."""

    return build_hidden_regime_factorial_calibration_design().to_payload()


def calibration_design_payload_sha256() -> str:
    """Return the frozen design payload digest."""

    return canonical_sha256(calibration_design_payload())


# Filled from ``calibration_design_payload_sha256`` after the source-level
# protocol was frozen.  Validation compares this literal rather than trusting a
# digest carried by an input payload.
CALIBRATION_DESIGN_PAYLOAD_SHA256 = (
    "735ceb533717e8b71c0159372b44041b2fd533ec14b62e78234de2c3552dd47d"
)


def calibration_design_envelope() -> dict[str, object]:
    """Return an in-memory digest envelope; this function performs no writes."""

    payload = calibration_design_payload()
    digest = canonical_sha256(payload)
    if digest != CALIBRATION_DESIGN_PAYLOAD_SHA256:
        raise RuntimeError("calibration design source differs from its frozen payload digest")
    return {
        "envelope_schema": DESIGN_ENVELOPE_SCHEMA,
        "payload": payload,
        "payload_sha256": digest,
    }


def validate_calibration_design_payload(
    payload: Mapping[str, object],
) -> HiddenRegimeFactorialCalibrationDesign:
    """Strictly validate an in-memory design payload against the frozen source."""

    return HiddenRegimeFactorialCalibrationDesign.from_payload(payload)


def validate_calibration_design_envelope(
    envelope: Mapping[str, object],
) -> HiddenRegimeFactorialCalibrationDesign:
    """Validate exact keys, digest, and canonical frozen payload in an envelope."""

    if type(envelope) is not dict:
        raise TypeError("design envelope must be a plain dict")
    if set(envelope) != {"envelope_schema", "payload", "payload_sha256"}:
        raise ValueError("design envelope keys are not exact")
    if envelope.get("envelope_schema") != DESIGN_ENVELOPE_SCHEMA:
        raise ValueError("design envelope schema is unsupported")
    digest = envelope.get("payload_sha256")
    if not _is_sha256(digest):
        raise ValueError("design envelope digest must be lowercase SHA-256")
    payload = envelope.get("payload")
    if type(payload) is not dict:
        raise TypeError("design envelope payload must be a plain dict")
    if canonical_sha256(payload) != digest:
        raise ValueError("design envelope digest does not match payload")
    if digest != CALIBRATION_DESIGN_PAYLOAD_SHA256:
        raise ValueError("design envelope digest differs from frozen source digest")
    return validate_calibration_design_payload(cast(dict[str, object], payload))


__all__ = [
    "BOUND_DEVELOPMENT_SUMMARY_SCHEMA",
    "BOUND_PRIMITIVE_TRACE_SCHEMA",
    "CALIBRATION_DESIGN_PAYLOAD_SHA256",
    "CALIBRATION_MANIFEST_ORDER",
    "CALIBRATION_READINESS_RECEIPT_SCHEMA",
    "CANONICAL_CONDITION_ORDER",
    "CONSUMED_CALIBRATION_NAMESPACE",
    "DESIGN_ENVELOPE_SCHEMA",
    "DESIGN_SCHEMA",
    "FACTORIAL_CELL_ORDER",
    "FROZEN_SEED_PAIRS",
    "N_MATCHED_CASES",
    "PRIMARY_LEVEL_METRIC_IDS",
    "PRIMARY_METRIC_IDS",
    "PROTOCOL_STATUS",
    "SEED_DERIVATION_SCHEMA",
    "SEED_SNAPSHOT_SHA256",
    "AuditRequirement",
    "BaseEvaluatorConfigBinding",
    "CalibrationAssignment",
    "CalibrationManifestBinding",
    "ConditionRuntimeBinding",
    "EstimandContract",
    "FactorialCell",
    "FrozenSeedPair",
    "GateFamilySpec",
    "HiddenRegimeFactorialCalibrationDesign",
    "MatchedCalibrationCase",
    "MetricContract",
    "PairedPopulationSupportMetric",
    "PriorCommitLineageAuditRecord",
    "RecurrenceEligibilityBinding",
    "StatisticalSummaryPlan",
    "ThresholdFreezeRule",
    "audit_prior_commit_lineage_serialization",
    "build_hidden_regime_factorial_calibration_design",
    "calibration_design_envelope",
    "calibration_design_payload",
    "calibration_design_payload_sha256",
    "canonical_json_bytes",
    "canonical_sha256",
    "derive_seed_pair_for_audit",
    "seed_snapshot_payload",
    "validate_calibration_design_envelope",
    "validate_calibration_design_payload",
]
