"""Development-only contract for a D-mapping-never-seen twin.

The twin changes only the evaluator-owned scalar targets in the first D phase.
It cyclically permutes the block's already-preregistered true D target values so
that the exact float32 value multiset is preserved but no observation in that
designated D block remains paired with its true D value.  The second D phase is
unchanged and is therefore the twin's first phase designated to expose true D.
Unrelated target functions can coincidentally equal D numerically; those raw
incidental matches are counted rather than misreported as D-phase experience.

The block permutation is intentionally evaluator-noncausal: another timestep's
preregistered target can be used at the current timestep.  The full synthetic
schedule and its two named Threefry keys are fixed before any learner exists and
are never included in learner input.  This module constructs no learner, runs no
campaign, writes no artifact, defines no threshold, and has no evidence or
promotion authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from numpy.typing import NDArray

from alberta_framework.core.compositional_features import FEATURE_VALUE_CLIP
from alberta_framework.evaluation.generated_class_recurrence import (
    DEVELOPMENT_EXPRESSION_NAMESPACE,
    GeneratedClassRecurrenceV0Protocol,
    GeneratedExpression,
    build_generated_class_recurrence_v0_protocol,
    derive_expression_manifest,
    expression_digest,
)

D_MAPPING_TWIN_SCHEMA = "alberta.generated-class-d-mapping-twin.development.v0"
D_MAPPING_TWIN_STATUS = "DEVELOPMENT_CONTRACT_NO_RUNNER_OR_EVIDENCE_AUTHORITY"

D_MAPPING_TWIN_OBSERVATION_NAMESPACE = (
    "alberta/generated-class-recurrence/v0/evaluator/d-twin-observations"
)
D_MAPPING_TWIN_DERANGEMENT_NAMESPACE = (
    "alberta/generated-class-recurrence/v0/evaluator/d-twin-derangement"
)

REFERENCE_TRUE_MAPPING = "reference_true_mapping"
SHAM_TRUE_MAPPING = "sham_true_mapping"
D_MAPPING_NEVER_SEEN_TWIN = "d_mapping_never_seen_twin"

_OBSERVATION_KEY_DATA = (0x91F4A62B, 0x3C07D8E5)
_DERANGEMENT_KEY_DATA = (0xD27B10C4, 0x6AE5938F)
_PRNG_IMPL = "threefry2x32"
_GRID_DENOMINATOR = 64

# Filled from the exact named-key derivation below.  These constants turn a
# dependency/algorithm drift into a construction error instead of silently
# changing the supposedly fixed development stream.
_EXPECTED_TOTAL_STEPS = 4_559
_EXPECTED_FIRST_D_LENGTH = 421
_EXPECTED_SECOND_D_LENGTH = 389
_EXPECTED_OBSERVATION_SHA256 = (
    "ab94bf71dca223c504f2c14f80fcbfcf57e590be6cd8448fb395462a0cb9dba7"
)
_EXPECTED_SOURCE_INDEX_SHA256 = (
    "0bb8b638c39dbd0c689830d269c555e2d62b61419b39b55d034a50a277b7fa51"
)


class DMappingTwinConstructionError(RuntimeError):
    """Raised when the fixed twin contract cannot be constructed exactly."""


@dataclasses.dataclass(frozen=True, slots=True)
class DMappingTwinRngContract:
    """Named evaluator RNG ownership and exact draw accounting."""

    prng_impl: str
    observation_namespace: str
    observation_key_data: tuple[int, int]
    derangement_namespace: str
    derangement_key_data: tuple[int, int]
    observation_uint32_words_per_draw: int
    derangement_candidate_shifts_per_draw: int
    logical_named_rng_streams: int
    logical_rng_draw_sites: int
    builder_construction_draw_invocations: int
    builder_validation_replay_draw_invocations: int
    builder_total_draw_invocations: int
    per_consumer_validation_replay_draw_invocations: int
    learner_rng_draw_invocations_owned: int
    rng_accounting_scope: str


@dataclasses.dataclass(frozen=True, slots=True)
class DMappingTwinResourceContract:
    """Exact persistent-array shapes, dtypes, and byte count."""

    observation_shape: tuple[int, int]
    target_shape: tuple[int]
    derangement_index_shape: tuple[int]
    observation_dtype: str
    target_dtype: str
    derangement_index_dtype: str
    persistent_array_nbytes: int
    persistent_array_nbytes_formula: str
    host_metadata_included: bool


@dataclasses.dataclass(frozen=True, slots=True)
class DMappingTwinWork:
    """Exact counts for four named construction/replay work dimensions."""

    observation_uint32_words_generated: int
    true_target_rows_evaluated: int
    cyclic_shift_candidates_audited: int
    cyclic_value_comparisons: int


@dataclasses.dataclass(frozen=True, slots=True)
class DMappingTwinOperationContract:
    """Logical construction and literal replay counts in an explicit scope."""

    logical_construction_work: DMappingTwinWork
    builder_validation_replay_work: DMappingTwinWork
    builder_total_work: DMappingTwinWork
    per_consumer_validation_replay_work: DMappingTwinWork
    raw_diagnostics_post_validation_target_rows_evaluated: int
    persistent_target_rows_materialized: int
    persistent_derangement_indices_materialized: int
    learner_update_steps: int
    runner_steps: int
    accounting_scope: str
    flop_or_hlo_equivalence_claimed: bool
    latency_measurement: str
    wall_clock_threshold: float | None


@dataclasses.dataclass(frozen=True, slots=True)
class DMappingTwinContract:
    """Hash-bound, non-executable D-mapping twin declaration."""

    schema: str
    status: str
    development_only: bool
    execution_authorized: bool
    runner_authorized: bool
    campaign_authorized: bool
    evidence_authorized: bool
    artifact_writes_authorized: bool
    scientific_promotion_allowed: bool
    recurrence_schema: str
    expression_manifest_sha256: str
    phase_length_manifest_sha256: str
    phase_order: tuple[str, ...]
    phase_lengths: tuple[int, ...]
    total_steps: int
    input_dim: int
    first_d_phase_index: int
    first_d_start: int
    first_d_stop: int
    first_d_length: int
    second_d_phase_index: int
    second_d_start: int
    second_d_stop: int
    second_d_length: int
    d_expression_whole_tree_digest: str
    observation_sampling: str
    preregistered_evaluator_only_noncausal_permutation: bool
    designated_first_d_true_mapping_exposures: int
    second_d_is_first_true_d_phase_experience: bool
    learner_input_fields: tuple[str, ...]
    learner_target_fields: tuple[str, ...]
    evaluator_only_fields: tuple[str, ...]
    paired_agent_initial_key_required: bool
    paired_agent_rng_call_count_required: bool
    paired_agent_rng_audit_implemented: bool
    target_conditioned_state_divergence_expected: bool
    rng_contract: DMappingTwinRngContract
    resource_contract: DMappingTwinResourceContract
    operation_contract: DMappingTwinOperationContract
    contract_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class DMappingTwinArm:
    """One resource-matched arm declaration; none is runnable here."""

    name: str
    role: str
    target_mode: str
    pairing_manifest_sha256: str
    resource_contract: DMappingTwinResourceContract
    operation_contract: DMappingTwinOperationContract
    derangement_computed: bool
    derangement_applied: bool
    execution_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class DMappingTwinDerangementAudit:
    """Exact structural facts for the selected cyclic permutation."""

    block_length: int
    cyclic_shift: int
    selected_candidate_rank: int
    candidate_count_evaluated: int
    index_fixed_point_count: int
    exact_equal_value_pair_count: int
    exact_changed_value_pair_count: int
    source_index_sha256: str
    true_value_multiset_sha256: str
    twin_value_multiset_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class DMappingTwinDataset:
    """Fixed evaluator arrays; learner views expose only observations/targets."""

    contract: DMappingTwinContract
    observations: Array
    reference_targets: Array
    sham_targets: Array
    twin_targets: Array
    first_d_source_indices: Array
    derangement_audit: DMappingTwinDerangementAudit
    observation_sha256: str
    reference_target_sha256: str
    sham_target_sha256: str
    twin_target_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class DMappingTwinRawDiagnostics:
    """Raw exact counts and hashes only; there is no threshold or verdict."""

    first_d_changed_value_rows: int
    first_d_equal_value_rows: int
    non_first_d_target_bit_mismatches: int
    second_d_target_bit_mismatches: int
    reference_sham_target_bit_mismatches: int
    source_index_fixed_points: int
    incidental_non_d_pre_second_d_true_value_matches: int
    observation_sha256: str
    reference_target_sha256: str
    sham_target_sha256: str
    twin_target_sha256: str
    first_d_true_multiset_sha256: str
    first_d_twin_multiset_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _float32_bits(values: Array) -> NDArray[np.uint32]:
    array = np.asarray(values)
    if array.dtype != np.float32:
        raise TypeError("target/observation array must have dtype float32")
    return cast(NDArray[np.uint32], array.view(np.uint32))


def _float32_array_sha256(values: Array) -> str:
    bits = _float32_bits(values)
    stable_bytes = bits.astype(">u4", copy=False).tobytes(order="C")
    return hashlib.sha256(stable_bytes).hexdigest()


def _uint32_multiset_sha256(values: NDArray[np.uint32]) -> str:
    sorted_bits = np.sort(values.reshape(-1)).astype(">u4", copy=False)
    return hashlib.sha256(sorted_bits.tobytes(order="C")).hexdigest()


def _source_index_sha256(indices: NDArray[np.int32]) -> str:
    return _sha256_json(tuple(int(index) for index in indices))


def _typed_threefry_key(key_data: tuple[int, int]) -> Array:
    data = jnp.asarray(key_data, dtype=jnp.uint32)
    if data.shape != (2,):
        raise DMappingTwinConstructionError("Threefry key data must have shape (2,)")
    return cast(Array, jr.wrap_key_data(data, impl=_PRNG_IMPL))


def _validate_typed_threefry_key(key: object) -> Array:
    if not isinstance(key, Array):
        raise TypeError("derangement key must be a typed JAX key")
    try:
        implementation = str(jr.key_impl(key))
        key_data = jr.key_data(key)
    except (TypeError, ValueError) as exc:
        raise TypeError("derangement key must be a typed JAX key") from exc
    if implementation != _PRNG_IMPL:
        raise ValueError("derangement key must use threefry2x32")
    if key.shape != () or key_data.shape != (2,) or key_data.dtype != jnp.uint32:
        raise ValueError("derangement key must contain exactly two uint32 words")
    return key


def _canonical_recurrence_protocol(
    recurrence_protocol: GeneratedClassRecurrenceV0Protocol | None,
) -> GeneratedClassRecurrenceV0Protocol:
    canonical = build_generated_class_recurrence_v0_protocol()
    if recurrence_protocol is None:
        return canonical
    if type(recurrence_protocol) is not GeneratedClassRecurrenceV0Protocol:
        raise TypeError("recurrence_protocol has the wrong concrete type")
    if recurrence_protocol != canonical:
        raise ValueError("D-mapping twin requires the canonical recurrence v0 protocol")
    return recurrence_protocol


def _phase_bounds(
    phase_lengths: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    starts: list[int] = []
    stops: list[int] = []
    cursor = 0
    for length in phase_lengths:
        starts.append(cursor)
        cursor += length
        stops.append(cursor)
    return tuple(starts), tuple(stops)


def _contract_sha256(contract: DMappingTwinContract) -> str:
    payload = dataclasses.asdict(contract)
    payload.pop("contract_sha256")
    return _sha256_json(payload)


def build_d_mapping_never_seen_contract(
    recurrence_protocol: GeneratedClassRecurrenceV0Protocol | None = None,
) -> DMappingTwinContract:
    """Bind the twin to the canonical recurrence protocol without running it."""

    recurrence = _canonical_recurrence_protocol(recurrence_protocol)
    if recurrence.n_tasks != 1 or recurrence.context_id != 0:
        raise DMappingTwinConstructionError("D twin requires the one-head recurrence v0")
    d_indices = tuple(
        index for index, phase_name in enumerate(recurrence.phase_order) if phase_name == "D"
    )
    if d_indices != (3, 7):
        raise DMappingTwinConstructionError("D twin requires D phases at indices 3 and 7")
    starts, stops = _phase_bounds(recurrence.phase_lengths)
    first_index, second_index = d_indices
    first_length = recurrence.phase_lengths[first_index]
    second_length = recurrence.phase_lengths[second_index]
    total_steps = sum(recurrence.phase_lengths)
    if (
        total_steps != _EXPECTED_TOTAL_STEPS
        or first_length != _EXPECTED_FIRST_D_LENGTH
        or second_length != _EXPECTED_SECOND_D_LENGTH
    ):
        raise DMappingTwinConstructionError("stable recurrence schedule shape changed")
    manifest = derive_expression_manifest(DEVELOPMENT_EXPRESSION_NAMESPACE)
    d_target = next(target for target in manifest.targets if target.name == "D")

    rng_contract = DMappingTwinRngContract(
        prng_impl=_PRNG_IMPL,
        observation_namespace=D_MAPPING_TWIN_OBSERVATION_NAMESPACE,
        observation_key_data=_OBSERVATION_KEY_DATA,
        derangement_namespace=D_MAPPING_TWIN_DERANGEMENT_NAMESPACE,
        derangement_key_data=_DERANGEMENT_KEY_DATA,
        observation_uint32_words_per_draw=total_steps * recurrence.input_dim,
        derangement_candidate_shifts_per_draw=first_length - 1,
        logical_named_rng_streams=2,
        logical_rng_draw_sites=2,
        builder_construction_draw_invocations=2,
        builder_validation_replay_draw_invocations=2,
        builder_total_draw_invocations=4,
        per_consumer_validation_replay_draw_invocations=2,
        learner_rng_draw_invocations_owned=0,
        rng_accounting_scope="twin_owned_keys_only_bound_recurrence_rng_is_upstream",
    )
    persistent_array_nbytes = 4 * (
        total_steps * recurrence.input_dim + 3 * total_steps + first_length
    )
    resources = DMappingTwinResourceContract(
        observation_shape=(total_steps, recurrence.input_dim),
        target_shape=(total_steps,),
        derangement_index_shape=(first_length,),
        observation_dtype="float32",
        target_dtype="float32",
        derangement_index_dtype="int32",
        persistent_array_nbytes=persistent_array_nbytes,
        persistent_array_nbytes_formula="4 * (T*K + 3*T + D1)",
        host_metadata_included=False,
    )
    logical_work = DMappingTwinWork(
        observation_uint32_words_generated=total_steps * recurrence.input_dim,
        true_target_rows_evaluated=total_steps,
        cyclic_shift_candidates_audited=first_length - 1,
        cyclic_value_comparisons=first_length * (first_length - 1),
    )
    builder_total_work = DMappingTwinWork(
        observation_uint32_words_generated=(
            2 * logical_work.observation_uint32_words_generated
        ),
        true_target_rows_evaluated=2 * logical_work.true_target_rows_evaluated,
        cyclic_shift_candidates_audited=(
            2 * logical_work.cyclic_shift_candidates_audited
        ),
        cyclic_value_comparisons=2 * logical_work.cyclic_value_comparisons,
    )
    operations = DMappingTwinOperationContract(
        logical_construction_work=logical_work,
        builder_validation_replay_work=logical_work,
        builder_total_work=builder_total_work,
        per_consumer_validation_replay_work=logical_work,
        raw_diagnostics_post_validation_target_rows_evaluated=starts[second_index],
        persistent_target_rows_materialized=3 * total_steps,
        persistent_derangement_indices_materialized=first_length,
        learner_update_steps=0,
        runner_steps=0,
        accounting_scope=(
            "named_rng_target_and_derangement_dimensions_not_flop_or_hlo_total"
        ),
        flop_or_hlo_equivalence_claimed=False,
        latency_measurement="structural_accounting_only_no_wall_clock_acceptance",
        wall_clock_threshold=None,
    )
    provisional = DMappingTwinContract(
        schema=D_MAPPING_TWIN_SCHEMA,
        status=D_MAPPING_TWIN_STATUS,
        development_only=True,
        execution_authorized=False,
        runner_authorized=False,
        campaign_authorized=False,
        evidence_authorized=False,
        artifact_writes_authorized=False,
        scientific_promotion_allowed=False,
        recurrence_schema=recurrence.schema,
        expression_manifest_sha256=recurrence.expression_manifest_sha256,
        phase_length_manifest_sha256=recurrence.phase_length_manifest_sha256,
        phase_order=recurrence.phase_order,
        phase_lengths=recurrence.phase_lengths,
        total_steps=total_steps,
        input_dim=recurrence.input_dim,
        first_d_phase_index=first_index,
        first_d_start=starts[first_index],
        first_d_stop=stops[first_index],
        first_d_length=first_length,
        second_d_phase_index=second_index,
        second_d_start=starts[second_index],
        second_d_stop=stops[second_index],
        second_d_length=second_length,
        d_expression_whole_tree_digest=expression_digest(d_target.expression),
        observation_sampling="iid_fixed_grid_from_named_threefry_uint32_bits",
        preregistered_evaluator_only_noncausal_permutation=True,
        designated_first_d_true_mapping_exposures=0,
        second_d_is_first_true_d_phase_experience=True,
        learner_input_fields=("raw_features",),
        learner_target_fields=("scalar_target",),
        evaluator_only_fields=(
            "phase_label",
            "phase_boundary",
            "arm_name",
            "twin_flag",
            "target_mapping_mode",
        ),
        paired_agent_initial_key_required=True,
        paired_agent_rng_call_count_required=True,
        paired_agent_rng_audit_implemented=False,
        target_conditioned_state_divergence_expected=True,
        rng_contract=rng_contract,
        resource_contract=resources,
        operation_contract=operations,
        contract_sha256="",
    )
    return dataclasses.replace(provisional, contract_sha256=_contract_sha256(provisional))


def _validate_contract(contract: object) -> DMappingTwinContract:
    if type(contract) is not DMappingTwinContract:
        raise TypeError("contract must be an exact DMappingTwinContract")
    canonical = build_d_mapping_never_seen_contract()
    if contract != canonical or contract.contract_sha256 != _contract_sha256(contract):
        raise ValueError("D-mapping twin contract is not canonical")
    return contract


def _fixed_observations(contract: DMappingTwinContract) -> Array:
    key = _typed_threefry_key(contract.rng_contract.observation_key_data)
    raw_bits = jr.bits(
        key,
        contract.resource_contract.observation_shape,
        dtype=jnp.uint32,
    )
    codes = (raw_bits >> jnp.uint32(24)).astype(jnp.int32) - jnp.int32(128)
    observations = codes.astype(jnp.float32) / jnp.float32(_GRID_DENOMINATOR)
    if observations.shape != contract.resource_contract.observation_shape:
        raise DMappingTwinConstructionError("fixed observation shape mismatch")
    if observations.dtype != jnp.float32 or not bool(jnp.all(jnp.isfinite(observations))):
        raise DMappingTwinConstructionError("fixed observations must be finite float32")
    return cast(Array, observations)


def _evaluate_expression_batch(
    expression: GeneratedExpression,
    observations: Array,
) -> Array:
    if observations.ndim != 2 or observations.dtype != jnp.float32:
        raise TypeError("batch observations must be a rank-two float32 JAX array")

    def evaluate(node: GeneratedExpression) -> Array:
        if node.op == "raw":
            if node.raw_index is None or node.raw_index >= observations.shape[1]:
                raise DMappingTwinConstructionError("target raw index is out of range")
            value = observations[:, node.raw_index]
        else:
            if node.left is None or node.right is None:
                raise DMappingTwinConstructionError("target expression is missing a child")
            left = evaluate(node.left)
            right = evaluate(node.right)
            if node.op == "sum":
                value = left + right
            elif node.op == "product":
                value = left * right
            elif node.op == "tanh":
                value = jnp.tanh(
                    jnp.float32(node.theta0) * left + jnp.float32(node.theta1) * right
                )
            elif node.op == "gate":
                value = left * jax.nn.sigmoid(right)
            else:
                raise DMappingTwinConstructionError("target op is outside the grammar")
        return jnp.clip(value, -jnp.float32(FEATURE_VALUE_CLIP), jnp.float32(FEATURE_VALUE_CLIP))

    result = evaluate(expression)
    if result.shape != (observations.shape[0],) or result.dtype != jnp.float32:
        raise DMappingTwinConstructionError("target batch has the wrong shape or dtype")
    return result


def _reference_targets(contract: DMappingTwinContract, observations: Array) -> Array:
    manifest = derive_expression_manifest(DEVELOPMENT_EXPRESSION_NAMESPACE)
    expressions = {target.name: target.expression for target in manifest.targets}
    starts, stops = _phase_bounds(contract.phase_lengths)
    targets = jnp.zeros((contract.total_steps,), dtype=jnp.float32)
    for phase_name, start, stop in zip(contract.phase_order, starts, stops, strict=True):
        phase_targets = _evaluate_expression_batch(
            expressions[phase_name],
            observations[start:stop],
        )
        targets = targets.at[start:stop].set(phase_targets)
    if targets.shape != contract.resource_contract.target_shape or not bool(
        jnp.all(jnp.isfinite(targets))
    ):
        raise DMappingTwinConstructionError("reference target stream is invalid")
    return targets


def choose_cyclic_value_derangement(
    values: Array,
    key: Array,
) -> tuple[Array, Array, DMappingTwinDerangementAudit]:
    """Choose a named-key-ordered nonzero cyclic shift with no correct value rows.

    Every nonzero shift is audited.  The first full value derangement in the
    Threefry-permuted candidate order is selected.  If repeated float32 values
    make every cyclic shift retain at least one true row, construction fails.
    """

    if not isinstance(values, Array):
        raise TypeError("values must be a JAX array")
    if values.ndim != 1 or values.dtype != jnp.float32:
        raise TypeError("values must be a rank-one float32 JAX array")
    if values.shape[0] < 2:
        raise ValueError("cyclic derangement requires at least two values")
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("cyclic derangement values must be finite")
    checked_key = _validate_typed_threefry_key(key)
    block_length = values.shape[0]
    candidate_order = np.asarray(
        jr.permutation(
            checked_key,
            jnp.arange(1, block_length, dtype=jnp.int32),
        ),
        dtype=np.int32,
    )
    true_bits = _float32_bits(values).reshape(-1)
    row_indices = np.arange(block_length, dtype=np.int32)
    selected_shift: int | None = None
    selected_rank: int | None = None
    selected_indices: NDArray[np.int32] | None = None
    for rank, shift_scalar in enumerate(candidate_order):
        shift = int(shift_scalar)
        source_indices = cast(
            NDArray[np.int32],
            (row_indices + shift) % block_length,
        )
        changed_count = int(np.count_nonzero(true_bits[source_indices] != true_bits))
        if changed_count == block_length and selected_shift is None:
            selected_shift = shift
            selected_rank = rank
            selected_indices = source_indices
    if selected_shift is None or selected_rank is None or selected_indices is None:
        raise DMappingTwinConstructionError(
            "no full value derangement exists among the nonzero cyclic shifts"
        )
    fixed_points = int(np.count_nonzero(selected_indices == row_indices))
    if fixed_points != 0:
        raise DMappingTwinConstructionError("selected cyclic indices are not a derangement")
    deranged = values[jnp.asarray(selected_indices, dtype=jnp.int32)]
    twin_bits = _float32_bits(deranged).reshape(-1)
    equal_count = int(np.count_nonzero(twin_bits == true_bits))
    changed_count = block_length - equal_count
    true_multiset_sha256 = _uint32_multiset_sha256(true_bits)
    twin_multiset_sha256 = _uint32_multiset_sha256(twin_bits)
    if equal_count != 0 or true_multiset_sha256 != twin_multiset_sha256:
        raise DMappingTwinConstructionError("selected shift is not a full multiset derangement")
    audit = DMappingTwinDerangementAudit(
        block_length=block_length,
        cyclic_shift=selected_shift,
        selected_candidate_rank=selected_rank,
        candidate_count_evaluated=block_length - 1,
        index_fixed_point_count=fixed_points,
        exact_equal_value_pair_count=equal_count,
        exact_changed_value_pair_count=changed_count,
        source_index_sha256=_source_index_sha256(selected_indices),
        true_value_multiset_sha256=true_multiset_sha256,
        twin_value_multiset_sha256=twin_multiset_sha256,
    )
    return deranged, jnp.asarray(selected_indices, dtype=jnp.int32), audit


def _validate_dataset(dataset: object) -> DMappingTwinDataset:
    if type(dataset) is not DMappingTwinDataset:
        raise TypeError("dataset must be an exact DMappingTwinDataset")
    contract = _validate_contract(dataset.contract)
    resources = contract.resource_contract
    arrays_and_contracts = (
        (dataset.observations, resources.observation_shape, jnp.float32),
        (dataset.reference_targets, resources.target_shape, jnp.float32),
        (dataset.sham_targets, resources.target_shape, jnp.float32),
        (dataset.twin_targets, resources.target_shape, jnp.float32),
        (
            dataset.first_d_source_indices,
            resources.derangement_index_shape,
            jnp.int32,
        ),
    )
    for array, shape, dtype in arrays_and_contracts:
        if not isinstance(array, Array) or array.shape != shape or array.dtype != dtype:
            raise DMappingTwinConstructionError("dataset array shape/dtype mismatch")

    # Reconstruct every load-bearing value from the canonical contract and its
    # named keys. Dataset-owned hashes and audit fields are never authorities
    # for the arrays they accompany.
    expected_observations = _fixed_observations(contract)
    expected_observation_sha256 = _float32_array_sha256(expected_observations)
    if expected_observation_sha256 != _EXPECTED_OBSERVATION_SHA256:
        raise DMappingTwinConstructionError("canonical fixed observation digest changed")
    if not np.array_equal(
        _float32_bits(dataset.observations),
        _float32_bits(expected_observations),
    ):
        raise DMappingTwinConstructionError("observations differ from the named-key stream")

    expected_reference_targets = _reference_targets(contract, expected_observations)
    if not np.array_equal(
        _float32_bits(dataset.reference_targets),
        _float32_bits(expected_reference_targets),
    ):
        raise DMappingTwinConstructionError("reference targets differ from the manifest")
    expected_first_d = expected_reference_targets[
        contract.first_d_start : contract.first_d_stop
    ]
    derangement_key = _typed_threefry_key(contract.rng_contract.derangement_key_data)
    expected_deranged_first_d, expected_source_indices, expected_audit = (
        choose_cyclic_value_derangement(expected_first_d, derangement_key)
    )
    if expected_audit.source_index_sha256 != _EXPECTED_SOURCE_INDEX_SHA256:
        raise DMappingTwinConstructionError("canonical derangement index digest changed")
    if not np.array_equal(
        np.asarray(dataset.first_d_source_indices, dtype=np.int32),
        np.asarray(expected_source_indices, dtype=np.int32),
    ):
        raise DMappingTwinConstructionError("source indices differ from the named-key choice")
    if type(dataset.derangement_audit) is not DMappingTwinDerangementAudit:
        raise DMappingTwinConstructionError("derangement audit has the wrong concrete type")
    if dataset.derangement_audit != expected_audit:
        raise DMappingTwinConstructionError("derangement audit differs from reconstruction")

    expected_sham_targets = expected_reference_targets
    expected_twin_targets = expected_reference_targets.at[
        contract.first_d_start : contract.first_d_stop
    ].set(expected_deranged_first_d)
    if not np.array_equal(
        _float32_bits(dataset.sham_targets),
        _float32_bits(expected_sham_targets),
    ):
        raise DMappingTwinConstructionError("sham targets differ from canonical reference")
    if not np.array_equal(
        _float32_bits(dataset.twin_targets),
        _float32_bits(expected_twin_targets),
    ):
        raise DMappingTwinConstructionError("twin targets differ from named-key derangement")

    expected_hashes = (
        expected_observation_sha256,
        _float32_array_sha256(expected_reference_targets),
        _float32_array_sha256(expected_sham_targets),
        _float32_array_sha256(expected_twin_targets),
    )
    actual_hashes = (
        dataset.observation_sha256,
        dataset.reference_target_sha256,
        dataset.sham_target_sha256,
        dataset.twin_target_sha256,
    )
    if actual_hashes != expected_hashes:
        raise DMappingTwinConstructionError("dataset hashes differ from reconstruction")
    persistent_nbytes = sum(
        int(array.nbytes)
        for array in (
            dataset.observations,
            dataset.reference_targets,
            dataset.sham_targets,
            dataset.twin_targets,
            dataset.first_d_source_indices,
        )
    )
    if persistent_nbytes != resources.persistent_array_nbytes:
        raise DMappingTwinConstructionError("persistent resource byte count mismatch")
    return dataset


def build_d_mapping_twin_dataset(
    contract: DMappingTwinContract,
) -> DMappingTwinDataset:
    """Construct fixed evaluator arrays only; no learner or runner is invoked."""

    checked = _validate_contract(contract)
    observations = _fixed_observations(checked)
    observation_sha256 = _float32_array_sha256(observations)
    if _EXPECTED_OBSERVATION_SHA256 and observation_sha256 != _EXPECTED_OBSERVATION_SHA256:
        raise DMappingTwinConstructionError("fixed observation digest changed")
    reference_targets = _reference_targets(checked, observations)
    true_first_d = reference_targets[checked.first_d_start : checked.first_d_stop]
    derangement_key = _typed_threefry_key(checked.rng_contract.derangement_key_data)
    deranged_first_d, source_indices, audit = choose_cyclic_value_derangement(
        true_first_d,
        derangement_key,
    )
    if _EXPECTED_SOURCE_INDEX_SHA256 and audit.source_index_sha256 != (
        _EXPECTED_SOURCE_INDEX_SHA256
    ):
        raise DMappingTwinConstructionError("fixed derangement index digest changed")
    sham_targets = jnp.array(reference_targets, dtype=jnp.float32, copy=True)
    twin_targets = reference_targets.at[checked.first_d_start : checked.first_d_stop].set(
        deranged_first_d
    )
    dataset = DMappingTwinDataset(
        contract=checked,
        observations=observations,
        reference_targets=reference_targets,
        sham_targets=sham_targets,
        twin_targets=twin_targets,
        first_d_source_indices=source_indices,
        derangement_audit=audit,
        observation_sha256=observation_sha256,
        reference_target_sha256=_float32_array_sha256(reference_targets),
        sham_target_sha256=_float32_array_sha256(sham_targets),
        twin_target_sha256=_float32_array_sha256(twin_targets),
    )
    return _validate_dataset(dataset)


def build_d_mapping_twin_arms(
    contract: DMappingTwinContract,
) -> tuple[DMappingTwinArm, ...]:
    """Declare reference, sham, and twin arms with identical allocated work."""

    checked = _validate_contract(contract)

    def arm(
        name: str,
        role: str,
        target_mode: str,
        *,
        derangement_applied: bool,
    ) -> DMappingTwinArm:
        return DMappingTwinArm(
            name=name,
            role=role,
            target_mode=target_mode,
            pairing_manifest_sha256=checked.contract_sha256,
            resource_contract=checked.resource_contract,
            operation_contract=checked.operation_contract,
            derangement_computed=True,
            derangement_applied=derangement_applied,
            execution_authorized=False,
            evidence_authorized=False,
            scientific_promotion_allowed=False,
        )

    return (
        arm(
            REFERENCE_TRUE_MAPPING,
            "reference",
            "true_targets_after_shared_derangement_bookkeeping",
            derangement_applied=False,
        ),
        arm(
            SHAM_TRUE_MAPPING,
            "matched_sham",
            "true_targets_after_shared_derangement_bookkeeping",
            derangement_applied=False,
        ),
        arm(
            D_MAPPING_NEVER_SEEN_TWIN,
            "mapping_never_seen_twin",
            "first_d_deranged_then_true_targets",
            derangement_applied=True,
        ),
    )


def learner_view(
    dataset: DMappingTwinDataset,
    arm_name: str,
) -> tuple[Array, Array]:
    """Return only raw observations and scalar targets for one declared arm."""

    checked = _validate_dataset(dataset)
    if type(arm_name) is not str:
        raise TypeError("arm_name must be an exact string")
    if arm_name == REFERENCE_TRUE_MAPPING:
        targets = checked.reference_targets
    elif arm_name == SHAM_TRUE_MAPPING:
        targets = checked.sham_targets
    elif arm_name == D_MAPPING_NEVER_SEEN_TWIN:
        targets = checked.twin_targets
    else:
        raise ValueError("unknown D-mapping twin arm")
    return checked.observations, targets


def measure_d_mapping_twin_array_nbytes(dataset: DMappingTwinDataset) -> int:
    """Measure only the five persistent JAX arrays named by the resource contract."""

    checked = _validate_dataset(dataset)
    arrays = (
        checked.observations,
        checked.reference_targets,
        checked.sham_targets,
        checked.twin_targets,
        checked.first_d_source_indices,
    )
    return sum(int(array.nbytes) for array in arrays)


def _raw_diagnostics_unchecked(dataset: DMappingTwinDataset) -> DMappingTwinRawDiagnostics:
    contract = dataset.contract
    reference_bits = _float32_bits(dataset.reference_targets).reshape(-1)
    sham_bits = _float32_bits(dataset.sham_targets).reshape(-1)
    twin_bits = _float32_bits(dataset.twin_targets).reshape(-1)
    first_slice = slice(contract.first_d_start, contract.first_d_stop)
    second_slice = slice(contract.second_d_start, contract.second_d_stop)
    non_first_mask = np.ones(contract.total_steps, dtype=np.bool_)
    non_first_mask[first_slice] = False
    pre_second_non_d_mask = np.ones(contract.second_d_start, dtype=np.bool_)
    pre_second_non_d_mask[contract.first_d_start : contract.first_d_stop] = False
    source_indices = np.asarray(dataset.first_d_source_indices, dtype=np.int32)
    row_indices = np.arange(contract.first_d_length, dtype=np.int32)
    first_true = reference_bits[first_slice]
    first_twin = twin_bits[first_slice]
    manifest = derive_expression_manifest(DEVELOPMENT_EXPRESSION_NAMESPACE)
    d_expression = next(
        target.expression for target in manifest.targets if target.name == "D"
    )
    true_d_before_second = _evaluate_expression_batch(
        d_expression,
        dataset.observations[: contract.second_d_start],
    )
    true_d_before_second_bits = _float32_bits(true_d_before_second).reshape(-1)
    return DMappingTwinRawDiagnostics(
        first_d_changed_value_rows=int(np.count_nonzero(first_true != first_twin)),
        first_d_equal_value_rows=int(np.count_nonzero(first_true == first_twin)),
        non_first_d_target_bit_mismatches=int(
            np.count_nonzero(reference_bits[non_first_mask] != twin_bits[non_first_mask])
        ),
        second_d_target_bit_mismatches=int(
            np.count_nonzero(reference_bits[second_slice] != twin_bits[second_slice])
        ),
        reference_sham_target_bit_mismatches=int(
            np.count_nonzero(reference_bits != sham_bits)
        ),
        source_index_fixed_points=int(np.count_nonzero(source_indices == row_indices)),
        incidental_non_d_pre_second_d_true_value_matches=int(
            np.count_nonzero(
                twin_bits[: contract.second_d_start][pre_second_non_d_mask]
                == true_d_before_second_bits[pre_second_non_d_mask]
            )
        ),
        observation_sha256=dataset.observation_sha256,
        reference_target_sha256=dataset.reference_target_sha256,
        sham_target_sha256=dataset.sham_target_sha256,
        twin_target_sha256=dataset.twin_target_sha256,
        first_d_true_multiset_sha256=_uint32_multiset_sha256(first_true),
        first_d_twin_multiset_sha256=_uint32_multiset_sha256(first_twin),
    )


def raw_d_mapping_twin_diagnostics(
    dataset: DMappingTwinDataset,
) -> DMappingTwinRawDiagnostics:
    """Return raw exact mismatch counts and hashes without a gate or verdict."""

    checked = _validate_dataset(dataset)
    return _raw_diagnostics_unchecked(checked)


def require_d_mapping_twin_runner_authorized(contract: DMappingTwinContract) -> None:
    """Fail closed: this module has no learner runner or campaign authority."""

    _validate_contract(contract)
    raise DMappingTwinConstructionError(
        "D-mapping twin runner is unauthorized: paired agent RNG audit, runner, "
        "artifact schema, thresholds, evidence authority, and promotion path are absent"
    )


__all__ = [
    "D_MAPPING_NEVER_SEEN_TWIN",
    "D_MAPPING_TWIN_DERANGEMENT_NAMESPACE",
    "D_MAPPING_TWIN_OBSERVATION_NAMESPACE",
    "REFERENCE_TRUE_MAPPING",
    "SHAM_TRUE_MAPPING",
    "DMappingTwinConstructionError",
    "DMappingTwinContract",
    "DMappingTwinDataset",
    "build_d_mapping_never_seen_contract",
    "build_d_mapping_twin_arms",
    "build_d_mapping_twin_dataset",
    "choose_cyclic_value_derangement",
    "learner_view",
    "measure_d_mapping_twin_array_nbytes",
    "raw_d_mapping_twin_diagnostics",
    "require_d_mapping_twin_runner_authorized",
]
