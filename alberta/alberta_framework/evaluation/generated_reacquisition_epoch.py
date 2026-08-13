"""Development-only post-scrub freeze and fresh RNG-epoch contract.

This module prepares one evaluator-owned contract after an exact expanded-tree
scrub has succeeded.  It binds every pre- and post-scrub state bit, independently
replays the structural scrub validator, declares a fixed half-open interval in
which structural generation writes must be suppressed, and derives the key for
the following reacquisition epoch from a preregistered evaluator root.

The key derivation is deliberately separate from the scrub target.  Its only
inputs are the fixed root, a named domain-separation namespace, one seed from a
fixed development-life manifest, and the fixed monotone epoch counter.  The life
seed must be shared across matched arms and is distinct across manifest lives.
Phase labels, target identities, observations, targets, predictions, outcomes,
and arm order are not inputs.  Evaluator metadata is not part of the learner
observation contract.

``CompositionalFeatureState`` exposes one learner key rather than a dedicated
generation-only substream.  Replacing that key therefore intervenes on both the
generator-policy decision and structural-generation randomness.  This module
only prepares the typed Threefry key; it neither mutates the learner state nor
executes the fixed freeze window.  A future runner must form the identical
curation proposal and mask every conditional curation component on the complete
reviewed local-state leaf surface, while retaining ordinary non-curation learning
writes, then apply the key at the exact endpoint.  That write-surface review is
bound to the complete learner module bytes and inspect-source hashes for
initialization, operation sampling, stage guidance, update, proposal generation,
and both cascade surfaces.  The learner now exposes a public fixed-shape curation
trace, but this prerequisite neither consumes that trace nor binds it into this
contract; trace availability is not lifecycle authentication.  The intervention
does not establish stochastic independence, target acquisition, adequate
acquisition probability, or behavioral-memory erasure.

No learner update, generated life, campaign, artifact, threshold, evidence, or
promotion path is authorized here.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import struct
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.compositional_features import (
    CompositionalCurationTrace,
    CompositionalFeatureLearner,
    CompositionalFeatureState,
    CompositionalFeatureUpdateResult,
)
from alberta_framework.evaluation.generated_class_lifecycle_scrub import (
    ACTIVE_MASKED_LEAF_PATHS,
    CANDIDATE_MASKED_LEAF_PATHS,
    COMPOSITIONAL_STATE_LEAF_PATHS,
    CROSS_MASKED_LEAF_PATHS,
    GeneratedClassScrubConfig,
    compositional_state_leaf_paths,
    persistent_compositional_state_nbytes,
)
from alberta_framework.evaluation.generated_class_recurrence import (
    GeneratedClassRecurrenceV0Protocol,
    GeneratedExpression,
    build_generated_class_recurrence_v0_protocol,
)
from alberta_framework.evaluation.generated_expression_lineage import (
    ExpandedExpressionLineageConfig,
    ExpandedExpressionLineagePlan,
    validate_post_scrub_expanded_expression_absence,
)

GENERATED_REACQUISITION_EPOCH_SCHEMA = (
    "alberta.generated-reacquisition-epoch.development.v1"
)
GENERATED_REACQUISITION_EPOCH_STATUS = (
    "DEVELOPMENT_PLAN_NO_FREEZE_RUNNER_OR_EXECUTION_AUTHORITY"
)
GENERATED_REACQUISITION_KEY_NAMESPACE = (
    "alberta/generated-class-recurrence/v0/evaluator/reacquisition-generation-epoch"
)
GENERATED_REACQUISITION_LIFE_SEED_MANIFEST_SCHEMA = (
    "alberta.generated-reacquisition-life-seeds.development.v0"
)
GENERATED_REACQUISITION_LIFE_SEED_NAMESPACE = (
    "alberta/generated-class-recurrence/v0/evaluator/reacquisition-development-lives"
)

_PRNG_IMPL = "threefry2x32"
_EVALUATOR_ROOT_KEY_DATA = (0xC63A0F51, 0x74BD982E)
_NAMESPACE_SHA256 = "01e87c71433e9ab9575503bda23f447ed59750f197f95f156b136e815a892d9f"
_NAMESPACE_FOLD_WORDS = (0x01E87C71, 0x433E9AB9)
_BASELINE_EPOCH_COUNTER = 0
_REACQUISITION_EPOCH_COUNTER = 1
_GENERATION_WRITE_FREEZE_UPDATES = 32
_DEVELOPMENT_PAIRED_LIFE_SEEDS = (101, 211, 307, 401, 503, 601, 701, 809)
_DEVELOPMENT_PAIRED_LIFE_SEED_MANIFEST_SHA256 = (
    "84d1989c959837994bc3fdb2454f92c871de41201573416bd2bacee0ce568610"
)
_EXPECTED_EPOCH_KEY_DATA_BY_LIFE_SEED = {
    101: ((0x8402811F, 0x8CDF6549), (0x070E65CE, 0x787A2732)),
    211: ((0x607AFAA3, 0xB30E5E2E), (0x092D9271, 0x0678A1BB)),
    307: ((0xC29722F4, 0x64164673), (0x71A63583, 0xC2571CC1)),
    401: ((0xE3A513F3, 0xFB2C3E46), (0x4B7676CE, 0xE1DEE6E8)),
    503: ((0xDF1BED48, 0xFA3A6ABB), (0x88168C45, 0x1DF7FC45)),
    601: ((0x0C072125, 0xBFD4F712), (0x2C6307CA, 0x7C74336C)),
    701: ((0x244FC55E, 0xAEF2A37A), (0x11D1553D, 0x32251370)),
    809: ((0x36B069F0, 0x9F3EDE64), (0x8DBD17AE, 0xD4AB123D)),
}
_EXPECTED_LEARNER_UPDATE_SOURCE_SHA256 = (
    "9c86a5c984c9d7cbcb454543f87589c938d8d7873797c81aa057a71f0903b8eb"
)
_EXPECTED_COMPOSITIONAL_FEATURES_MODULE_BYTE_SHA256 = (
    "767f054bb3413b2408e664a17bcb8690a9f83018f638d6acfcfde2e9debf5b5a"
)
_EXPECTED_LEARNER_INIT_SOURCE_SHA256 = (
    "1ae246e6736ad932e798d21d5f7238477414ff908636289cdbfcd73f42f3ce94"
)
_EXPECTED_LEARNER_OP_LOGITS_SOURCE_SHA256 = (
    "c6d1a059bd29d074596809716cc775df2396fb11ec1a5386dbc78f7dc148fd82"
)
_EXPECTED_LEARNER_GENERATE_ONE_SOURCE_SHA256 = (
    "d6aafc8401b45ed915323997f7560d866e36fd42aac375d3db3ab660d625c960"
)
_EXPECTED_LEARNER_CURATION_STAGE_GUIDANCE_SOURCE_SHA256 = (
    "27c997e2478a7eef95ffd5adb3e592b085a313f3ce2fa1d78ab3330bccd0f725"
)
_EXPECTED_LEARNER_CASCADE_REPLACE_SOURCE_SHA256 = (
    "2994d446d0ff9a667ec67a304b92cf7ff003292fc278d62e4f66ae849fe836a2"
)
_EXPECTED_LEARNER_CASCADE_REPLACE_WITH_MASK_SOURCE_SHA256 = (
    "8a26857076565a2c8a4f064b079ac3a412b6194c4a2c4095925dec462951fa98"
)

_KEY_DERIVATION_INPUT_FIELDS = (
    "preregistered_evaluator_root_key",
    "domain_separation_namespace",
    "paired_life_seed_from_development_manifest",
    "reacquisition_epoch_counter",
)
_KEY_DERIVATION_FORBIDDEN_INPUT_FIELDS = (
    "phase_label",
    "phase_boundary",
    "target_name",
    "target_expression",
    "target_digest",
    "observations",
    "targets",
    "predictions",
    "losses",
    "outcomes",
    "arm_name",
    "arm_order",
)
_CONDITIONAL_CURATION_WRITE_LEAF_PATHS = tuple(
    sorted(
        ACTIVE_MASKED_LEAF_PATHS
        | CANDIDATE_MASKED_LEAF_PATHS
        | CROSS_MASKED_LEAF_PATHS
        | {
            "generator_resource_state.action_counts",
            "generator_resource_state.log_weights",
            "generator_resource_state.reward_ema",
            "generator_resource_state.step_count",
            "replacement_accumulator",
        }
    )
)
if not set(_CONDITIONAL_CURATION_WRITE_LEAF_PATHS) <= COMPOSITIONAL_STATE_LEAF_PATHS:
    raise RuntimeError("conditional curation-write audit names an unknown state leaf")
_EVALUATOR_ONLY_FIELDS = (
    "structural_scrub_target",
    "pre_scrub_state_hash",
    "post_scrub_state_hash",
    "structural_scrub_validation",
    "generation_write_freeze_bounds",
    "reviewed_compositional_features_module_byte_sha256",
    "reviewed_learner_init_source_sha256",
    "reviewed_learner_op_logits_source_sha256",
    "reviewed_learner_update_source_sha256",
    "reviewed_learner_generate_one_source_sha256",
    "reviewed_learner_curation_stage_guidance_source_sha256",
    "reviewed_learner_cascade_replace_source_sha256",
    "reviewed_learner_cascade_replace_with_mask_source_sha256",
    "public_curation_trace_available",
    "public_curation_trace_consumed",
    "public_curation_trace_bound",
    "conditional_curation_write_leaf_paths",
    "paired_life_seed_manifest",
    "paired_life_seed",
    "generation_key_namespace",
    "generation_epoch_counter",
    "fresh_learner_state_key",
)


class GeneratedReacquisitionEpochConstructionError(RuntimeError):
    """Raised when a fresh epoch cannot be prepared without overclaiming."""


class GeneratedReacquisitionEpochExecutionUnauthorizedError(RuntimeError):
    """Raised whenever this non-runner contract is treated as executable."""


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedReacquisitionEpochConfig:
    """Canonical v1 evaluator configuration; every field is fail-closed."""

    generation_write_freeze_updates: int = _GENERATION_WRITE_FREEZE_UPDATES
    previous_epoch_counter: int = _BASELINE_EPOCH_COUNTER
    reacquisition_epoch_counter: int = _REACQUISITION_EPOCH_COUNTER
    paired_life_seed: int = _DEVELOPMENT_PAIRED_LIFE_SEEDS[0]
    key_namespace: str = GENERATED_REACQUISITION_KEY_NAMESPACE
    evaluator_root_key_data_uint32: tuple[int, int] = _EVALUATOR_ROOT_KEY_DATA
    key_derivation_input_fields: tuple[str, ...] = _KEY_DERIVATION_INPUT_FIELDS
    key_derivation_forbidden_input_fields: tuple[str, ...] = (
        _KEY_DERIVATION_FORBIDDEN_INPUT_FIELDS
    )
    conditional_curation_write_leaf_paths: tuple[str, ...] = (
        _CONDITIONAL_CURATION_WRITE_LEAF_PATHS
    )
    schema: str = GENERATED_REACQUISITION_EPOCH_SCHEMA
    status: str = GENERATED_REACQUISITION_EPOCH_STATUS
    development_only: bool = True
    evaluator_intervention: bool = True
    future_target_blind_key_derivation: bool = True
    evaluator_metadata_in_learner_observations: bool = False
    public_curation_trace_available: bool = True
    public_curation_trace_consumed: bool = False
    public_curation_trace_bound: bool = False
    freeze_runner_implemented: bool = False
    fresh_key_application_implemented: bool = False
    generation_only_substream_claimed: bool = False
    stochastic_independence_claimed: bool = False
    acquisition_claimed: bool = False
    adequate_acquisition_probability_claimed: bool = False
    behavioral_memory_erasure_claimed: bool = False
    execution_authorized: bool = False
    runner_authorized: bool = False
    campaign_authorized: bool = False
    artifact_writes_authorized: bool = False
    threshold_authorized: bool = False
    evidence_authorized: bool = False
    scientific_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            "generation_write_freeze_updates",
            "previous_epoch_counter",
            "reacquisition_epoch_counter",
            "paired_life_seed",
        )
        for name in integer_fields:
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact Python integer")
        if self.generation_write_freeze_updates != _GENERATION_WRITE_FREEZE_UPDATES:
            raise ValueError("generation-write freeze length is not the fixed v0 value")
        if self.previous_epoch_counter != _BASELINE_EPOCH_COUNTER:
            raise ValueError("previous epoch counter is not the preregistered v0 baseline")
        if self.reacquisition_epoch_counter != self.previous_epoch_counter + 1:
            raise ValueError("reacquisition epoch counter must be the next monotone epoch")
        if self.reacquisition_epoch_counter != _REACQUISITION_EPOCH_COUNTER:
            raise ValueError("reacquisition epoch counter is not the preregistered v0 value")
        if self.paired_life_seed not in _DEVELOPMENT_PAIRED_LIFE_SEEDS:
            raise ValueError("paired_life_seed is outside the fixed development manifest")

        if type(self.key_namespace) is not str:
            raise TypeError("key_namespace must be an exact Python string")
        if self.key_namespace != GENERATED_REACQUISITION_KEY_NAMESPACE:
            raise ValueError("reacquisition key namespace is not canonical")
        if type(self.evaluator_root_key_data_uint32) is not tuple:
            raise TypeError("evaluator root key data must be an exact tuple")
        if self.evaluator_root_key_data_uint32 != _EVALUATOR_ROOT_KEY_DATA:
            raise ValueError("evaluator root key is not the preregistered v0 root")
        for index, word in enumerate(self.evaluator_root_key_data_uint32):
            if type(word) is not int or not 0 <= word <= np.iinfo(np.uint32).max:
                raise ValueError(f"evaluator root key word {index} is not uint32")

        tuple_fields = (
            "key_derivation_input_fields",
            "key_derivation_forbidden_input_fields",
            "conditional_curation_write_leaf_paths",
        )
        for name in tuple_fields:
            value = getattr(self, name)
            if type(value) is not tuple or not all(type(item) is str for item in value):
                raise TypeError(f"{name} must be an exact tuple of exact strings")
        if self.key_derivation_input_fields != _KEY_DERIVATION_INPUT_FIELDS:
            raise ValueError("key derivation inputs are not canonical")
        if self.key_derivation_forbidden_input_fields != (
            _KEY_DERIVATION_FORBIDDEN_INPUT_FIELDS
        ):
            raise ValueError("key derivation forbidden-input declaration is not canonical")
        if set(self.key_derivation_input_fields) & set(
            self.key_derivation_forbidden_input_fields
        ):
            raise ValueError("key derivation includes a forbidden target/outcome field")
        if self.conditional_curation_write_leaf_paths != (
            _CONDITIONAL_CURATION_WRITE_LEAF_PATHS
        ):
            raise ValueError("conditional curation-write leaf set is not canonical")
        if len(set(self.conditional_curation_write_leaf_paths)) != len(
            self.conditional_curation_write_leaf_paths
        ):
            raise ValueError("conditional curation-write leaf paths must be unique")

        if type(self.schema) is not str or type(self.status) is not str:
            raise TypeError("schema and status must be exact Python strings")
        if self.schema != GENERATED_REACQUISITION_EPOCH_SCHEMA:
            raise ValueError("reacquisition epoch schema is not canonical")
        if self.status != GENERATED_REACQUISITION_EPOCH_STATUS:
            raise ValueError("reacquisition epoch status is not canonical")

        boolean_fields = (
            "development_only",
            "evaluator_intervention",
            "future_target_blind_key_derivation",
            "evaluator_metadata_in_learner_observations",
            "public_curation_trace_available",
            "public_curation_trace_consumed",
            "public_curation_trace_bound",
            "freeze_runner_implemented",
            "fresh_key_application_implemented",
            "generation_only_substream_claimed",
            "stochastic_independence_claimed",
            "acquisition_claimed",
            "adequate_acquisition_probability_claimed",
            "behavioral_memory_erasure_claimed",
            "execution_authorized",
            "runner_authorized",
            "campaign_authorized",
            "artifact_writes_authorized",
            "threshold_authorized",
            "evidence_authorized",
            "scientific_promotion_allowed",
        )
        for name in boolean_fields:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact Python boolean")
        if not (
            self.development_only
            and self.evaluator_intervention
            and self.future_target_blind_key_derivation
            and self.public_curation_trace_available
        ):
            raise ValueError(
                "the contract must remain target-blind and expose the public trace"
            )
        forbidden_claims = (
            self.evaluator_metadata_in_learner_observations,
            self.public_curation_trace_consumed,
            self.public_curation_trace_bound,
            self.freeze_runner_implemented,
            self.fresh_key_application_implemented,
            self.generation_only_substream_claimed,
            self.stochastic_independence_claimed,
            self.acquisition_claimed,
            self.adequate_acquisition_probability_claimed,
            self.behavioral_memory_erasure_claimed,
            self.execution_authorized,
            self.runner_authorized,
            self.campaign_authorized,
            self.artifact_writes_authorized,
            self.threshold_authorized,
            self.evidence_authorized,
            self.scientific_promotion_allowed,
        )
        if any(forbidden_claims):
            raise ValueError("development reacquisition config cannot grant claims or authority")


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedReacquisitionEpochContract:
    """Hash-bound declaration for one post-scrub freeze and fresh epoch."""

    schema: str
    status: str
    development_only: bool
    recurrence_schema: str
    expression_manifest_sha256: str
    phase_length_manifest_sha256: str
    curation_interval: int
    structural_lineage_schema: str
    structural_lineage_plan_sha256: str
    structural_lineage_pre_state_bit_sha256: str
    structural_scrub_validation_sha256: str
    structural_scrub_target_sha256: str
    structural_scrub_valid: bool
    pre_scrub_state_bit_sha256: str
    post_scrub_state_bit_sha256: str
    pre_scrub_learner_key_data_uint32: tuple[int, int]
    post_scrub_learner_key_data_uint32: tuple[int, int]
    scrub_preserved_learner_key_exactly: bool
    key_prng_impl: str
    key_namespace: str
    key_namespace_sha256: str
    key_namespace_fold_words_uint32: tuple[int, int]
    evaluator_root_key_data_uint32: tuple[int, int]
    paired_life_seed_manifest_schema: str
    paired_life_seed_namespace: str
    paired_life_seed_manifest_sha256: str
    paired_life_seeds_uint32: tuple[int, ...]
    paired_life_seed: int
    paired_life_seed_matched_across_arms_required: bool
    all_manifest_lives_required_for_campaign: bool
    distinct_epoch_keys_across_manifest_lives_observed: bool
    previous_epoch_counter: int
    reacquisition_epoch_counter: int
    previous_epoch_key_data_uint32: tuple[int, int]
    fresh_learner_key_data_uint32: tuple[int, int]
    key_derivation_algorithm: str
    key_derivation_input_fields: tuple[str, ...]
    key_derivation_forbidden_input_fields: tuple[str, ...]
    future_target_blind_key_derivation: bool
    fresh_key_collision_with_bound_or_manifest_prior_count: int
    manifest_epoch_key_collision_count: int
    fresh_key_matches_bound_pre_or_post_state_key: bool
    learner_key_scope: str
    generation_only_substream_claimed: bool
    freeze_window_coordinate: str
    generation_write_freeze_start_state_step_count: int
    generation_write_freeze_end_state_step_count: int
    generation_write_freeze_updates: int
    scheduled_curation_decision_slots_in_freeze: int
    reviewed_compositional_features_module_byte_sha256: str
    reviewed_learner_init_source_sha256: str
    reviewed_learner_op_logits_source_sha256: str
    reviewed_learner_update_source_sha256: str
    reviewed_learner_generate_one_source_sha256: str
    reviewed_learner_curation_stage_guidance_source_sha256: str
    reviewed_learner_cascade_replace_source_sha256: str
    reviewed_learner_cascade_replace_with_mask_source_sha256: str
    public_curation_trace_available: bool
    public_curation_trace_consumed: bool
    public_curation_trace_bound: bool
    identical_curation_proposal_required: bool
    conditional_curation_write_leaf_paths: tuple[str, ...]
    conditional_curation_write_components_permitted_during_freeze: int
    ordinary_noncuration_learning_writes_permitted: bool
    freeze_runner_implemented: bool
    fresh_key_apply_state_step_count: int
    fresh_key_application_implemented: bool
    learner_observation_fields: tuple[str, ...]
    evaluator_only_fields: tuple[str, ...]
    evaluator_metadata_in_learner_observations: bool
    evaluator_intervention: bool
    generator_policy_state_reset: bool
    global_rng_use_ledger_implemented: bool
    stochastic_independence_claimed: bool
    acquisition_claimed: bool
    adequate_acquisition_probability_claimed: bool
    behavioral_memory_erasure_claimed: bool
    pre_scrub_persistent_array_nbytes: int
    post_scrub_persistent_array_nbytes: int
    evaluator_fresh_key_nbytes: int
    learner_updates_executed: int
    generated_life_steps_executed: int
    learner_rng_draws_executed: int
    artifact_bytes_written: int
    wall_clock_threshold: float | None
    execution_authorized: bool
    runner_authorized: bool
    campaign_authorized: bool
    artifact_writes_authorized: bool
    threshold_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool
    contract_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedReacquisitionEpochPlan:
    """Canonical host metadata plus the unapplied typed learner key."""

    contract: GeneratedReacquisitionEpochContract
    fresh_learner_key: Array


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedReacquisitionEpochValidation:
    """Raw validation of an untrusted plan against current bound inputs."""

    supplied_contract_self_hash_valid: bool
    contract_matches_canonical: bool
    fresh_key_is_typed_threefry: bool
    fresh_key_matches_supplied_contract: bool
    fresh_key_matches_canonical: bool
    monotone_epoch_counter_valid: bool
    target_or_outcome_derivation_fields_absent: bool
    pre_scrub_state_hash_matches: bool
    post_scrub_state_hash_matches: bool
    structural_scrub_valid: bool
    no_detectable_key_reuse: bool
    evaluator_metadata_excluded_from_learner_observations: bool
    authority_closed: bool
    valid: bool


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


def _canonical_config() -> GeneratedReacquisitionEpochConfig:
    return GeneratedReacquisitionEpochConfig()


def _validate_config(
    config: object,
) -> GeneratedReacquisitionEpochConfig:
    if type(config) is not GeneratedReacquisitionEpochConfig:
        raise TypeError("config must be an exact GeneratedReacquisitionEpochConfig")
    canonical_for_life = dataclasses.replace(
        _canonical_config(),
        paired_life_seed=config.paired_life_seed,
    )
    if config != canonical_for_life:
        raise ValueError("reacquisition epoch config is not canonical")
    return config


def _life_seed_manifest_sha256() -> str:
    payload = {
        "schema": GENERATED_REACQUISITION_LIFE_SEED_MANIFEST_SCHEMA,
        "namespace": GENERATED_REACQUISITION_LIFE_SEED_NAMESPACE,
        "paired_life_seeds_uint32": list(_DEVELOPMENT_PAIRED_LIFE_SEEDS),
        "development_only": True,
        "matched_exactly_across_arms": True,
        "all_manifest_lives_required_for_campaign": True,
    }
    return _sha256_json(payload)


def _typed_threefry_key(words: tuple[int, int]) -> Array:
    data = jnp.asarray(words, dtype=jnp.uint32)
    if data.shape != (2,):
        raise GeneratedReacquisitionEpochConstructionError(
            "typed Threefry key data must have exactly two words"
        )
    key = cast(Array, jr.wrap_key_data(data, impl=_PRNG_IMPL))
    _validate_typed_threefry_key(key, name="derived key")
    return key


def _validate_typed_threefry_key(key: object, *, name: str) -> Array:
    if not isinstance(key, Array):
        raise TypeError(f"{name} must be a typed JAX key")
    if not jax.dtypes.issubdtype(  # type: ignore[attr-defined]
        key.dtype,
        jax.dtypes.prng_key,
    ):
        raise TypeError(f"{name} must use a typed JAX PRNG dtype")
    try:
        implementation = str(jr.key_impl(key))
        data = jr.key_data(key)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a typed JAX key") from exc
    if implementation != _PRNG_IMPL:
        raise ValueError(f"{name} must use {_PRNG_IMPL}")
    if key.shape != () or data.shape != (2,) or data.dtype != jnp.uint32:
        raise ValueError(f"{name} must be a scalar key containing two uint32 words")
    return key


def _key_data_tuple(key: object, *, name: str) -> tuple[int, int]:
    checked = _validate_typed_threefry_key(key, name=name)
    words = tuple(int(value) for value in np.asarray(jr.key_data(checked)).reshape(-1))
    if len(words) != 2:
        raise GeneratedReacquisitionEpochConstructionError(
            f"{name} did not expose exactly two key-data words"
        )
    return words


def derive_generated_reacquisition_epoch_key(
    epoch_counter: int,
    *,
    config: GeneratedReacquisitionEpochConfig,
) -> Array:
    """Derive one exact typed key without accepting any target/outcome input.

    Only the fixed root, namespace, paired development-life seed, and either of
    the two fixed v0 counters are available.  ``fold_in`` is deterministic key
    derivation, not a learner RNG draw and not a claim of independent outcomes.
    """

    checked = _validate_config(config)
    if type(epoch_counter) is not int:
        raise TypeError("epoch_counter must be an exact Python integer")
    if epoch_counter not in {
        checked.previous_epoch_counter,
        checked.reacquisition_epoch_counter,
    }:
        raise ValueError("epoch_counter is outside the preregistered v0 epoch sequence")
    key = _typed_threefry_key(checked.evaluator_root_key_data_uint32)
    for word in _NAMESPACE_FOLD_WORDS:
        key = jr.fold_in(key, jnp.asarray(word, dtype=jnp.uint32))
    key = jr.fold_in(
        key,
        jnp.asarray(checked.paired_life_seed, dtype=jnp.uint32),
    )
    key = jr.fold_in(key, jnp.asarray(epoch_counter, dtype=jnp.uint32))
    checked_key = _validate_typed_threefry_key(key, name="derived epoch key")
    expected = _EXPECTED_EPOCH_KEY_DATA_BY_LIFE_SEED[checked.paired_life_seed][
        epoch_counter
    ]
    if _key_data_tuple(checked_key, name="derived epoch key") != expected:
        raise GeneratedReacquisitionEpochConstructionError(
            "derived epoch key drifted from the pinned development-life manifest"
        )
    return checked_key


def _path_text(path: tuple[Any, ...]) -> str:
    names: list[str] = []
    for key in path:
        name = getattr(key, "name", None)
        if not isinstance(name, str):
            raise TypeError(f"unsupported compositional-state path key: {key!r}")
        names.append(name)
    return ".".join(names)


def _normalized_array_bytes(array: np.ndarray[Any, Any]) -> bytes:
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot be state-hash leaves")
    if array.dtype.byteorder == "|":
        normalized = np.ascontiguousarray(array)
    else:
        normalized = np.ascontiguousarray(array.astype(array.dtype.newbyteorder(">"), copy=False))
    return normalized.tobytes(order="C")


def _state_bit_sha256(state: CompositionalFeatureState) -> str:
    if type(state) is not CompositionalFeatureState:
        raise TypeError("state must be an exact CompositionalFeatureState")
    compositional_state_leaf_paths(state)
    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        path_name = _path_text(path)
        if isinstance(leaf, Array) and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            leaf.dtype,
            jax.dtypes.prng_key,
        ):
            checked = _validate_typed_threefry_key(leaf, name=f"state.{path_name}")
            array = np.asarray(jr.key_data(checked), dtype=np.uint32)
            kind = "typed-prng-threefry2x32"
            dtype = ">u4"
            raw = _normalized_array_bytes(array.astype(">u4", copy=False))
        elif isinstance(leaf, Array):
            array = np.asarray(leaf)
            kind = "jax-array"
            dtype = array.dtype.newbyteorder(">").str
            raw = _normalized_array_bytes(array)
        elif type(leaf) is float:
            array = np.asarray(leaf, dtype=">f8")
            kind = "python-float"
            dtype = ">f8"
            raw = struct.pack(">d", leaf)
        else:
            raise TypeError(f"unsupported state-hash leaf at {path_name}: {type(leaf)!r}")
        metadata = _canonical_json_bytes(
            {
                "path": path_name,
                "kind": kind,
                "dtype": dtype,
                "shape": list(array.shape),
                "nbytes": len(raw),
            }
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _state_step_count(state: CompositionalFeatureState, *, name: str) -> int:
    array = np.asarray(state.step_count)
    if array.shape != () or array.dtype != np.int32:
        raise TypeError(f"{name}.step_count must be a scalar int32 array")
    value = int(array)
    if value < 0:
        raise ValueError(f"{name}.step_count must be non-negative")
    return value


def _compositional_features_module_byte_sha256() -> str:
    source_path = inspect.getsourcefile(CompositionalFeatureLearner)
    if source_path is None:
        raise GeneratedReacquisitionEpochConstructionError(
            "compositional-feature learner source path is unavailable"
        )
    return hashlib.sha256(Path(source_path).read_bytes()).hexdigest()


def _canonical_protocol() -> GeneratedClassRecurrenceV0Protocol:
    if not dataclasses.is_dataclass(CompositionalCurationTrace) or (
        "curation_trace"
        not in getattr(CompositionalFeatureUpdateResult, "__dataclass_fields__", {})
    ):
        raise GeneratedReacquisitionEpochConstructionError(
            "public compositional curation trace is unavailable"
        )
    module_byte_sha256 = _compositional_features_module_byte_sha256()
    if (
        module_byte_sha256
        != _EXPECTED_COMPOSITIONAL_FEATURES_MODULE_BYTE_SHA256
    ):
        raise GeneratedReacquisitionEpochConstructionError(
            "reviewed compositional_features module bytes changed; "
            "curation-write audit is stale"
        )
    reviewed_methods = (
        (
            "__init__",
            CompositionalFeatureLearner.__init__,
            _EXPECTED_LEARNER_INIT_SOURCE_SHA256,
        ),
        (
            "_op_logits",
            CompositionalFeatureLearner._op_logits,
            _EXPECTED_LEARNER_OP_LOGITS_SOURCE_SHA256,
        ),
        (
            "update",
            CompositionalFeatureLearner.update,
            _EXPECTED_LEARNER_UPDATE_SOURCE_SHA256,
        ),
        (
            "_generate_one",
            CompositionalFeatureLearner._generate_one,
            _EXPECTED_LEARNER_GENERATE_ONE_SOURCE_SHA256,
        ),
        (
            "_curation_stage_guidance",
            CompositionalFeatureLearner._curation_stage_guidance,
            _EXPECTED_LEARNER_CURATION_STAGE_GUIDANCE_SOURCE_SHA256,
        ),
        (
            "_cascade_replace",
            CompositionalFeatureLearner._cascade_replace,
            _EXPECTED_LEARNER_CASCADE_REPLACE_SOURCE_SHA256,
        ),
        (
            "_cascade_replace_with_mask",
            CompositionalFeatureLearner._cascade_replace_with_mask,
            _EXPECTED_LEARNER_CASCADE_REPLACE_WITH_MASK_SOURCE_SHA256,
        ),
    )
    for method_name, method, expected_sha256 in reviewed_methods:
        source_sha256 = hashlib.sha256(
            inspect.getsource(method).encode("utf-8")
        ).hexdigest()
        if source_sha256 != expected_sha256:
            raise GeneratedReacquisitionEpochConstructionError(
                f"reviewed learner {method_name} source changed; "
                "curation-write audit is stale"
            )
    protocol = build_generated_class_recurrence_v0_protocol()
    if protocol.execution_authorized or protocol.evidence_authorized:
        raise GeneratedReacquisitionEpochConstructionError(
            "reacquisition development plan cannot inspect an authorized protocol"
        )
    if protocol.scientific_promotion_allowed:
        raise GeneratedReacquisitionEpochConstructionError(
            "reacquisition development plan cannot allow scientific promotion"
        )
    if protocol.curation_opportunity_audit.curation_interval != (
        _GENERATION_WRITE_FREEZE_UPDATES
    ):
        raise GeneratedReacquisitionEpochConstructionError(
            "recurrence curation cadence no longer matches the fixed freeze window"
        )
    if protocol.learner_observation_fields != ("raw_features",):
        raise GeneratedReacquisitionEpochConstructionError(
            "learner observation contract changed; target-blind audit is stale"
        )
    return protocol


def _contract_payload(contract: GeneratedReacquisitionEpochContract) -> dict[str, object]:
    payload = cast(dict[str, object], dataclasses.asdict(contract))
    payload.pop("contract_sha256")
    return payload


def _contract_self_hash_valid(contract: GeneratedReacquisitionEpochContract) -> bool:
    return contract.contract_sha256 == _sha256_json(_contract_payload(contract))


def build_generated_reacquisition_epoch_plan(
    pre_scrub_state: CompositionalFeatureState,
    post_scrub_state: CompositionalFeatureState,
    target: GeneratedExpression,
    lineage_plan: ExpandedExpressionLineagePlan,
    *,
    lineage_config: ExpandedExpressionLineageConfig,
    scrub_config: GeneratedClassScrubConfig,
    config: GeneratedReacquisitionEpochConfig,
) -> GeneratedReacquisitionEpochPlan:
    """Prepare, but do not apply or execute, the canonical fresh-epoch plan."""

    checked = _validate_config(config)
    protocol = _canonical_protocol()
    life_seed_manifest_sha256 = _life_seed_manifest_sha256()
    if life_seed_manifest_sha256 != _DEVELOPMENT_PAIRED_LIFE_SEED_MANIFEST_SHA256:
        raise GeneratedReacquisitionEpochConstructionError(
            "development paired-life seed manifest digest changed"
        )
    structural = validate_post_scrub_expanded_expression_absence(
        pre_scrub_state,
        post_scrub_state,
        target,
        lineage_plan,
        config=lineage_config,
        scrub_config=scrub_config,
    )
    if not structural.valid:
        raise GeneratedReacquisitionEpochConstructionError(
            "expanded structural scrub validation did not accept the bound transaction"
        )

    pre_hash = _state_bit_sha256(pre_scrub_state)
    post_hash = _state_bit_sha256(post_scrub_state)
    if pre_hash == post_hash:
        raise GeneratedReacquisitionEpochConstructionError(
            "successful nonempty structural scrub did not change the bound state"
        )

    pre_key_data = _key_data_tuple(pre_scrub_state.key, name="pre-scrub learner key")
    post_key_data = _key_data_tuple(post_scrub_state.key, name="post-scrub learner key")
    scrub_preserved_key = pre_key_data == post_key_data
    if not scrub_preserved_key:
        raise GeneratedReacquisitionEpochConstructionError(
            "structural scrub unexpectedly changed the learner key"
        )

    namespace_digest = hashlib.sha256(checked.key_namespace.encode("ascii")).hexdigest()
    if namespace_digest != _NAMESPACE_SHA256:
        raise GeneratedReacquisitionEpochConstructionError(
            "reacquisition namespace digest no longer matches its preregistration"
        )
    digest_bytes = bytes.fromhex(namespace_digest)
    namespace_words = tuple(
        int.from_bytes(digest_bytes[index : index + 4], "big")
        for index in (0, 4)
    )
    if namespace_words != _NAMESPACE_FOLD_WORDS:
        raise GeneratedReacquisitionEpochConstructionError(
            "reacquisition namespace fold words no longer match their preregistration"
        )

    previous_key = derive_generated_reacquisition_epoch_key(
        checked.previous_epoch_counter,
        config=checked,
    )
    fresh_key = derive_generated_reacquisition_epoch_key(
        checked.reacquisition_epoch_counter,
        config=checked,
    )
    previous_key_data = _key_data_tuple(previous_key, name="previous epoch key")
    fresh_key_data = _key_data_tuple(fresh_key, name="fresh reacquisition epoch key")
    manifest_epoch_keys: list[tuple[int, int]] = []
    for paired_life_seed in _DEVELOPMENT_PAIRED_LIFE_SEEDS:
        life_config = dataclasses.replace(checked, paired_life_seed=paired_life_seed)
        for epoch_counter in (_BASELINE_EPOCH_COUNTER, _REACQUISITION_EPOCH_COUNTER):
            manifest_epoch_keys.append(
                _key_data_tuple(
                    derive_generated_reacquisition_epoch_key(
                        epoch_counter,
                        config=life_config,
                    ),
                    name="manifest epoch key",
                )
            )
    manifest_collision_count = len(manifest_epoch_keys) - len(set(manifest_epoch_keys))
    if manifest_collision_count != 0:
        raise GeneratedReacquisitionEpochConstructionError(
            "development life/epoch key manifest contains a collision"
        )
    other_manifest_epoch_keys = set(manifest_epoch_keys)
    other_manifest_epoch_keys.remove(fresh_key_data)
    detectable_prior_keys = {
        checked.evaluator_root_key_data_uint32,
        pre_key_data,
        post_key_data,
        *other_manifest_epoch_keys,
    }
    if fresh_key_data in detectable_prior_keys:
        raise GeneratedReacquisitionEpochConstructionError(
            "fresh reacquisition key reuses a bound or prior namespace key"
        )

    pre_step = _state_step_count(pre_scrub_state, name="pre_scrub_state")
    post_step = _state_step_count(post_scrub_state, name="post_scrub_state")
    if pre_step != post_step:
        raise GeneratedReacquisitionEpochConstructionError(
            "structural scrub unexpectedly changed the learner step counter"
        )
    freeze_stop = post_step + checked.generation_write_freeze_updates
    if freeze_stop > np.iinfo(np.int32).max:
        raise GeneratedReacquisitionEpochConstructionError(
            "generation-write freeze endpoint exceeds learner int32 step capacity"
        )
    scheduled_curation_slots = (
        freeze_stop // protocol.curation_opportunity_audit.curation_interval
        - post_step // protocol.curation_opportunity_audit.curation_interval
    )
    if scheduled_curation_slots != 1:
        raise GeneratedReacquisitionEpochConstructionError(
            "fixed freeze window must contain exactly one scheduled curation decision"
        )

    pre_nbytes = persistent_compositional_state_nbytes(pre_scrub_state)
    post_nbytes = persistent_compositional_state_nbytes(post_scrub_state)
    if pre_nbytes != post_nbytes:
        raise GeneratedReacquisitionEpochConstructionError(
            "structural scrub changed the persistent learner resource budget"
        )
    if pre_nbytes != structural.pre_state_persistent_array_nbytes:
        raise GeneratedReacquisitionEpochConstructionError(
            "persistent resource accounting disagrees with structural validation"
        )
    if post_nbytes != structural.post_state_persistent_array_nbytes:
        raise GeneratedReacquisitionEpochConstructionError(
            "post-scrub resource accounting disagrees with structural validation"
        )

    contract_without_hash = GeneratedReacquisitionEpochContract(
        schema=checked.schema,
        status=checked.status,
        development_only=True,
        recurrence_schema=protocol.schema,
        expression_manifest_sha256=protocol.expression_manifest_sha256,
        phase_length_manifest_sha256=protocol.phase_length_manifest_sha256,
        curation_interval=protocol.curation_opportunity_audit.curation_interval,
        structural_lineage_schema=structural.schema,
        structural_lineage_plan_sha256=structural.canonical_plan_sha256,
        structural_lineage_pre_state_bit_sha256=(
            lineage_plan.audit.pre_state_bit_sha256
        ),
        structural_scrub_validation_sha256=structural.validation_sha256,
        structural_scrub_target_sha256=structural.target_expression_sha256,
        structural_scrub_valid=True,
        pre_scrub_state_bit_sha256=pre_hash,
        post_scrub_state_bit_sha256=post_hash,
        pre_scrub_learner_key_data_uint32=pre_key_data,
        post_scrub_learner_key_data_uint32=post_key_data,
        scrub_preserved_learner_key_exactly=True,
        key_prng_impl=_PRNG_IMPL,
        key_namespace=checked.key_namespace,
        key_namespace_sha256=namespace_digest,
        key_namespace_fold_words_uint32=namespace_words,
        evaluator_root_key_data_uint32=checked.evaluator_root_key_data_uint32,
        paired_life_seed_manifest_schema=(
            GENERATED_REACQUISITION_LIFE_SEED_MANIFEST_SCHEMA
        ),
        paired_life_seed_namespace=GENERATED_REACQUISITION_LIFE_SEED_NAMESPACE,
        paired_life_seed_manifest_sha256=life_seed_manifest_sha256,
        paired_life_seeds_uint32=_DEVELOPMENT_PAIRED_LIFE_SEEDS,
        paired_life_seed=checked.paired_life_seed,
        paired_life_seed_matched_across_arms_required=True,
        all_manifest_lives_required_for_campaign=True,
        distinct_epoch_keys_across_manifest_lives_observed=True,
        previous_epoch_counter=checked.previous_epoch_counter,
        reacquisition_epoch_counter=checked.reacquisition_epoch_counter,
        previous_epoch_key_data_uint32=previous_key_data,
        fresh_learner_key_data_uint32=fresh_key_data,
        key_derivation_algorithm=(
            "threefry2x32 evaluator root; fold_in(namespace_sha256_u32_be[0]); "
            "fold_in(namespace_sha256_u32_be[1]); fold_in(paired_life_seed_u32); "
            "fold_in(epoch_counter_u32)"
        ),
        key_derivation_input_fields=checked.key_derivation_input_fields,
        key_derivation_forbidden_input_fields=(
            checked.key_derivation_forbidden_input_fields
        ),
        future_target_blind_key_derivation=True,
        fresh_key_collision_with_bound_or_manifest_prior_count=0,
        manifest_epoch_key_collision_count=manifest_collision_count,
        fresh_key_matches_bound_pre_or_post_state_key=False,
        learner_key_scope=(
            "single CompositionalFeatureState.key split into generator-policy and "
            "curation-root channels; proposal, cascade, and candidate overdepth "
            "regeneration domains derive from the curation root"
        ),
        generation_only_substream_claimed=False,
        freeze_window_coordinate=(
            "learner transitions after the start state through and including "
            "the end state step_count"
        ),
        generation_write_freeze_start_state_step_count=post_step,
        generation_write_freeze_end_state_step_count=freeze_stop,
        generation_write_freeze_updates=checked.generation_write_freeze_updates,
        scheduled_curation_decision_slots_in_freeze=scheduled_curation_slots,
        reviewed_compositional_features_module_byte_sha256=(
            _EXPECTED_COMPOSITIONAL_FEATURES_MODULE_BYTE_SHA256
        ),
        reviewed_learner_init_source_sha256=(
            _EXPECTED_LEARNER_INIT_SOURCE_SHA256
        ),
        reviewed_learner_op_logits_source_sha256=(
            _EXPECTED_LEARNER_OP_LOGITS_SOURCE_SHA256
        ),
        reviewed_learner_update_source_sha256=(
            _EXPECTED_LEARNER_UPDATE_SOURCE_SHA256
        ),
        reviewed_learner_generate_one_source_sha256=(
            _EXPECTED_LEARNER_GENERATE_ONE_SOURCE_SHA256
        ),
        reviewed_learner_curation_stage_guidance_source_sha256=(
            _EXPECTED_LEARNER_CURATION_STAGE_GUIDANCE_SOURCE_SHA256
        ),
        reviewed_learner_cascade_replace_source_sha256=(
            _EXPECTED_LEARNER_CASCADE_REPLACE_SOURCE_SHA256
        ),
        reviewed_learner_cascade_replace_with_mask_source_sha256=(
            _EXPECTED_LEARNER_CASCADE_REPLACE_WITH_MASK_SOURCE_SHA256
        ),
        public_curation_trace_available=True,
        public_curation_trace_consumed=False,
        public_curation_trace_bound=False,
        identical_curation_proposal_required=True,
        conditional_curation_write_leaf_paths=(
            checked.conditional_curation_write_leaf_paths
        ),
        conditional_curation_write_components_permitted_during_freeze=0,
        ordinary_noncuration_learning_writes_permitted=True,
        freeze_runner_implemented=False,
        fresh_key_apply_state_step_count=freeze_stop,
        fresh_key_application_implemented=False,
        learner_observation_fields=protocol.learner_observation_fields,
        evaluator_only_fields=_EVALUATOR_ONLY_FIELDS,
        evaluator_metadata_in_learner_observations=False,
        evaluator_intervention=True,
        generator_policy_state_reset=False,
        global_rng_use_ledger_implemented=False,
        stochastic_independence_claimed=False,
        acquisition_claimed=False,
        adequate_acquisition_probability_claimed=False,
        behavioral_memory_erasure_claimed=False,
        pre_scrub_persistent_array_nbytes=pre_nbytes,
        post_scrub_persistent_array_nbytes=post_nbytes,
        evaluator_fresh_key_nbytes=2 * np.dtype(np.uint32).itemsize,
        learner_updates_executed=0,
        generated_life_steps_executed=0,
        learner_rng_draws_executed=0,
        artifact_bytes_written=0,
        wall_clock_threshold=None,
        execution_authorized=False,
        runner_authorized=False,
        campaign_authorized=False,
        artifact_writes_authorized=False,
        threshold_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
        contract_sha256="",
    )
    contract = dataclasses.replace(
        contract_without_hash,
        contract_sha256=_sha256_json(_contract_payload(contract_without_hash)),
    )
    return GeneratedReacquisitionEpochPlan(
        contract=contract,
        fresh_learner_key=fresh_key,
    )


def validate_generated_reacquisition_epoch_plan(
    plan: GeneratedReacquisitionEpochPlan,
    pre_scrub_state: CompositionalFeatureState,
    post_scrub_state: CompositionalFeatureState,
    target: GeneratedExpression,
    lineage_plan: ExpandedExpressionLineagePlan,
    *,
    lineage_config: ExpandedExpressionLineageConfig,
    scrub_config: GeneratedClassScrubConfig,
    config: GeneratedReacquisitionEpochConfig,
) -> GeneratedReacquisitionEpochValidation:
    """Independently rebuild and validate an untrusted unapplied epoch plan."""

    if type(plan) is not GeneratedReacquisitionEpochPlan:
        raise TypeError("plan must be an exact GeneratedReacquisitionEpochPlan")
    if type(plan.contract) is not GeneratedReacquisitionEpochContract:
        raise TypeError("plan.contract must be an exact GeneratedReacquisitionEpochContract")
    supplied_key = _validate_typed_threefry_key(
        plan.fresh_learner_key,
        name="plan fresh learner key",
    )
    canonical = build_generated_reacquisition_epoch_plan(
        pre_scrub_state,
        post_scrub_state,
        target,
        lineage_plan,
        lineage_config=lineage_config,
        scrub_config=scrub_config,
        config=config,
    )
    supplied_key_data = _key_data_tuple(supplied_key, name="plan fresh learner key")
    canonical_key_data = _key_data_tuple(
        canonical.fresh_learner_key,
        name="canonical fresh learner key",
    )
    contract = plan.contract
    contract_matches = contract == canonical.contract
    monotone = bool(
        type(contract.previous_epoch_counter) is int
        and type(contract.reacquisition_epoch_counter) is int
        and contract.reacquisition_epoch_counter == contract.previous_epoch_counter + 1
        and contract.previous_epoch_counter == _BASELINE_EPOCH_COUNTER
        and contract.reacquisition_epoch_counter == _REACQUISITION_EPOCH_COUNTER
    )
    target_fields_absent = bool(
        contract.key_derivation_input_fields == _KEY_DERIVATION_INPUT_FIELDS
        and contract.key_derivation_forbidden_input_fields
        == _KEY_DERIVATION_FORBIDDEN_INPUT_FIELDS
        and not (
            set(contract.key_derivation_input_fields)
            & set(contract.key_derivation_forbidden_input_fields)
        )
    )
    pre_hash_matches = (
        contract.pre_scrub_state_bit_sha256 == _state_bit_sha256(pre_scrub_state)
    )
    post_hash_matches = (
        contract.post_scrub_state_bit_sha256 == _state_bit_sha256(post_scrub_state)
    )
    canonical_manifest_keys = {
        key_data
        for life_keys in _EXPECTED_EPOCH_KEY_DATA_BY_LIFE_SEED.values()
        for key_data in life_keys
    }
    canonical_manifest_keys.remove(canonical_key_data)
    detectable_prior_keys = {
        contract.evaluator_root_key_data_uint32,
        contract.pre_scrub_learner_key_data_uint32,
        contract.post_scrub_learner_key_data_uint32,
        *canonical_manifest_keys,
    }
    no_reuse = bool(
        supplied_key_data not in detectable_prior_keys
        and contract.fresh_learner_key_data_uint32 not in detectable_prior_keys
        and contract.fresh_key_collision_with_bound_or_manifest_prior_count == 0
        and contract.manifest_epoch_key_collision_count == 0
        and not contract.fresh_key_matches_bound_pre_or_post_state_key
    )
    metadata_excluded = bool(
        not contract.evaluator_metadata_in_learner_observations
        and contract.learner_observation_fields == ("raw_features",)
        and not (set(contract.learner_observation_fields) & set(contract.evaluator_only_fields))
    )
    authority_closed = bool(
        contract.public_curation_trace_available
        and not any(
            (
                contract.public_curation_trace_consumed,
                contract.public_curation_trace_bound,
                contract.execution_authorized,
                contract.runner_authorized,
                contract.campaign_authorized,
                contract.artifact_writes_authorized,
                contract.threshold_authorized,
                contract.evidence_authorized,
                contract.scientific_promotion_allowed,
                contract.freeze_runner_implemented,
                contract.fresh_key_application_implemented,
                contract.generation_only_substream_claimed,
                contract.stochastic_independence_claimed,
                contract.acquisition_claimed,
                contract.adequate_acquisition_probability_claimed,
                contract.behavioral_memory_erasure_claimed,
            )
        )
    )
    supplied_self_hash = _contract_self_hash_valid(contract)
    supplied_key_matches_contract = (
        supplied_key_data == contract.fresh_learner_key_data_uint32
    )
    supplied_key_matches_canonical = supplied_key_data == canonical_key_data
    valid = bool(
        supplied_self_hash
        and contract_matches
        and supplied_key_matches_contract
        and supplied_key_matches_canonical
        and monotone
        and target_fields_absent
        and pre_hash_matches
        and post_hash_matches
        and contract.structural_scrub_valid
        and no_reuse
        and metadata_excluded
        and authority_closed
    )
    return GeneratedReacquisitionEpochValidation(
        supplied_contract_self_hash_valid=supplied_self_hash,
        contract_matches_canonical=contract_matches,
        fresh_key_is_typed_threefry=True,
        fresh_key_matches_supplied_contract=supplied_key_matches_contract,
        fresh_key_matches_canonical=supplied_key_matches_canonical,
        monotone_epoch_counter_valid=monotone,
        target_or_outcome_derivation_fields_absent=target_fields_absent,
        pre_scrub_state_hash_matches=pre_hash_matches,
        post_scrub_state_hash_matches=post_hash_matches,
        structural_scrub_valid=contract.structural_scrub_valid,
        no_detectable_key_reuse=no_reuse,
        evaluator_metadata_excluded_from_learner_observations=metadata_excluded,
        authority_closed=authority_closed,
        valid=valid,
    )


def require_generated_reacquisition_epoch_executable(
    plan: GeneratedReacquisitionEpochPlan,
) -> None:
    """Always fail: this module prepares metadata and owns no runner."""

    if type(plan) is not GeneratedReacquisitionEpochPlan:
        raise TypeError("plan must be an exact GeneratedReacquisitionEpochPlan")
    raise GeneratedReacquisitionEpochExecutionUnauthorizedError(
        "generated reacquisition epoch is not executable: freeze-write masking, "
        "public-trace consumption and binding, per-update validation, and endpoint key "
        "application require a future runner"
    )


__all__ = [
    "GENERATED_REACQUISITION_EPOCH_SCHEMA",
    "GENERATED_REACQUISITION_EPOCH_STATUS",
    "GENERATED_REACQUISITION_KEY_NAMESPACE",
    "GENERATED_REACQUISITION_LIFE_SEED_MANIFEST_SCHEMA",
    "GENERATED_REACQUISITION_LIFE_SEED_NAMESPACE",
    "GeneratedReacquisitionEpochConfig",
    "GeneratedReacquisitionEpochConstructionError",
    "GeneratedReacquisitionEpochContract",
    "GeneratedReacquisitionEpochExecutionUnauthorizedError",
    "GeneratedReacquisitionEpochPlan",
    "GeneratedReacquisitionEpochValidation",
    "build_generated_reacquisition_epoch_plan",
    "derive_generated_reacquisition_epoch_key",
    "require_generated_reacquisition_epoch_executable",
    "validate_generated_reacquisition_epoch_plan",
]
