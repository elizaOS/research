# mypy: disable-error-code="attr-defined,call-arg,type-var"
"""Strict L0 prospective exploration selection with a post-generation shield.

The historical filename predates the repository's narrow delight vocabulary.
Nothing in this module computes DG delight or establishes that a policy-gradient
contribution entered an executed actor backward pass.  Its canonical public
quantity is an expected-improvement--surprisal exploration score.

The Alberta WP5.6 extension scores a fixed batch of exploratory candidates as

``expected_improvement * min(-log(host_policy_probability), surprisal_cap)``.

The host probability is the probability of the candidate action under the
pre-decision host policy.  The controller also exposes random, epsilon-greedy,
ensemble-disagreement, information-gain, and learning-progress comparators.
Every mode consumes the same statically configured candidate batch and the
same logical random-draw schedule.

The boundary is deliberately narrow.  Pandora equivalence is exact only for
the revealed-value search model.  For a noisy independent-arm bandit, the
caller-supplied expected improvement is a value-of-perfect-information proxy
that upper-bounds one-step knowledge gradient; it is not exact sequential
value of information.  This module estimates neither quantity itself.

Candidate generation is completed before the caller-owned hard safety mask is
consulted.  A permitted candidate becomes a proposed executable action; a
rejected candidate falls back to the host action only when that action is also
declared safe.  The controller never dispatches an action and has no policy,
deployment, physical-safety, evidence, or promotion authority.

All score inputs are explicitly bound to one source event, owner identities,
and nondecreasing producer revisions.  A caller must attest that the values
were available before the decision.  That attestation is auditable ownership
metadata, not proof that a producer was causal or calibrated.  Dynamic invalid
input, stale ownership, and exhausted exact uint64 clocks are atomic no-ops.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.checkpoints import load_checkpoint as _load_checkpoint
from alberta_framework.core.checkpoints import (
    load_checkpoint_metadata as _load_checkpoint_metadata,
)
from alberta_framework.core.checkpoints import save_checkpoint as _save_checkpoint

DELIGHTFUL_EXPLORATION_CONFIG_SCHEMA = "alberta.delightful-exploration.config.v1"
DELIGHTFUL_EXPLORATION_STATE_SCHEMA = "alberta.delightful-exploration.state.v1"
DELIGHTFUL_EXPLORATION_CHECKPOINT_SCHEMA = "alberta.delightful-exploration.checkpoint.v1"
DELIGHTFUL_EXPLORATION_RESOURCE_SCHEMA = "alberta.delightful-exploration.resource.v1"
PROSPECTIVE_EXPLORATION_CONFIG_SCHEMA = "alberta.prospective-exploration.config.v2"
PROSPECTIVE_EXPLORATION_STATE_SCHEMA = "alberta.prospective-exploration.state.v2"
PROSPECTIVE_EXPLORATION_CHECKPOINT_SCHEMA = (
    "alberta.prospective-exploration.checkpoint.v2"
)
PROSPECTIVE_EXPLORATION_RESOURCE_SCHEMA = "alberta.prospective-exploration.resource.v2"
DELIGHTFUL_EXPLORATION_EVIDENCE_LEVEL = "L0"
DELIGHTFUL_EXPLORATION_OUTCOME_STATUS = "not_assessed"
DELIGHTFUL_EXPLORATION_LIFETIME_SEMANTICS = "exact-uint64-fail-stop"
DELIGHTFUL_EXPLORATION_REVEALED_VALUE_EQUIVALENCE = (
    "exact-only-in-the-revealed-value-search-model"
)
DELIGHTFUL_EXPLORATION_NOISY_BANDIT_SEMANTICS = (
    "expected-improvement-is-a-value-of-perfect-information-proxy-that-upper-bounds-"
    "one-step-knowledge-gradient-not-exact-sequential-value-of-information"
)
DELIGHTFUL_EXPLORATION_ACTION_DISPATCH_AUTHORITY = False
DELIGHTFUL_EXPLORATION_POLICY_OVERRIDE_AUTHORITY = False
DELIGHTFUL_EXPLORATION_PHYSICAL_SAFETY_CLAIM = False
DELIGHTFUL_EXPLORATION_SCIENTIFIC_PROMOTION_ALLOWED = False
DELIGHTFUL_EXPLORATION_OUTPUT_WRITE_AUTHORITY = False
PROSPECTIVE_EXPLORATION_SCORE_SEMANTICS = (
    "expected-improvement-times-capped-host-relative-surprisal"
)
PROSPECTIVE_EXPLORATION_GRADIENT_DELIGHT_SEMANTICS = False
PROSPECTIVE_EXPLORATION_EXECUTES_ACTOR_BACKWARD = False

ExplorationMode = Literal[
    "expected_improvement_surprisal",
    "random",
    "epsilon_greedy",
    "ensemble_disagreement",
    "information_gain",
    "learning_progress",
]

DELIGHTFUL_EXPLORATION_MODES: tuple[ExplorationMode, ...] = (
    "expected_improvement_surprisal",
    "random",
    "epsilon_greedy",
    "ensemble_disagreement",
    "information_gain",
    "learning_progress",
)

_DIGEST_WORDS = 8
_UINT32_MAX = 2**32 - 1
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)


def _exact_manifest(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError(f"{label} must be an exact dict")
    fields = dict(payload)
    supplied = set(fields)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(f"{label} field manifest is not exact; missing={missing}, extra={extra}")
    return fields


def _exact_positive_int(value: object, *, label: str, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{label} must be in [1, {maximum}]")
    return value


def _exact_float32(
    value: object,
    *,
    label: str,
    minimum: float,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if type(value) is not float:
        raise TypeError(f"{label} must be an exact Python float")
    scalar = float(value)
    represented = float(np.float32(scalar))
    below = scalar <= minimum if strict_minimum else scalar < minimum
    if (
        not math.isfinite(scalar)
        or not math.isfinite(represented)
        or below
        or (maximum is not None and scalar > maximum)
    ):
        comparator = ">" if strict_minimum else ">="
        upper = "" if maximum is None else f" and <= {maximum}"
        raise ValueError(f"{label} must be finite, {comparator} {minimum}{upper}, and float32")
    if scalar != 0.0 and abs(represented) < _FLOAT32_TINY:
        raise ValueError(f"{label} must not become subnormal in float32")
    return scalar


def _digest_tuple(value: object, *, label: str) -> tuple[int, ...]:
    if type(value) is not tuple or len(value) != _DIGEST_WORDS:
        raise TypeError(f"{label} must be an exact {_DIGEST_WORDS}-word tuple")
    result: list[int] = []
    for index, word in enumerate(value):
        if type(word) is not int or not 0 <= word <= _UINT32_MAX:
            raise ValueError(f"{label}[{index}] must be uint32-compatible")
        result.append(word)
    if not any(result):
        raise ValueError(f"{label} must be nonzero")
    return tuple(result)


def _mode(value: object) -> ExplorationMode:
    if type(value) is not str:
        raise TypeError("mode must be an exact string")
    if value not in DELIGHTFUL_EXPLORATION_MODES:
        raise ValueError(f"mode must be one of {DELIGHTFUL_EXPLORATION_MODES}")
    return value


def _require_array(
    value: Any,
    *,
    label: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    if getattr(value, "shape", None) != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if getattr(value, "dtype", None) != dtype:
        raise TypeError(f"{label} must have dtype {dtype}")
    return jnp.asarray(value)


def _require_threefry_key(value: Any, *, label: str) -> None:
    try:
        data = jr.key_data(value)
        implementation = str(jr.key_impl(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be one typed Threefry JAX key") from exc
    if (
        getattr(value, "shape", None) != ()
        or data.shape != (2,)
        or data.dtype != jnp.dtype(jnp.uint32)
        or implementation != "threefry2x32"
    ):
        raise TypeError(f"{label} must be one typed Threefry JAX key")


def _words_greater(left: Array, right: Array) -> Bool[Array, ""]:
    return (left[0] > right[0]) | ((left[0] == right[0]) & (left[1] > right[1]))


def _words_greater_equal(left: Array, right: Array) -> Bool[Array, ""]:
    return jnp.all(left == right) | _words_greater(left, right)


def _increment_words(words: Array) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    carry = words[1] == maximum
    capacity = ~(carry & (words[0] == maximum))
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    high = words[0] + carry.astype(jnp.uint32)
    return jnp.stack((high, low), dtype=jnp.uint32), capacity


def _array_nbytes(value: Array) -> int:
    return int(value.size) * int(value.dtype.itemsize)


@dataclasses.dataclass(frozen=True, slots=True)
class DelightfulExplorationConfig:
    """Static candidate budget, score bounds, comparator, and owner bindings."""

    n_actions: int
    candidate_budget: int
    mode: ExplorationMode
    epsilon: float
    host_surprisal_cap: float
    max_expected_improvement: float
    max_ensemble_disagreement: float
    max_information_gain: float
    max_learning_progress: float
    source_owner_digest: tuple[int, ...]
    host_policy_owner_digest: tuple[int, ...]
    candidate_owner_digest: tuple[int, ...]
    score_owner_digest: tuple[int, ...]
    safety_owner_digest: tuple[int, ...]

    def __post_init__(self) -> None:
        _exact_positive_int(self.n_actions, label="n_actions", maximum=2**31 - 1)
        if self.n_actions < 2:
            raise ValueError("n_actions must be at least 2 for exploration")
        _exact_positive_int(
            self.candidate_budget,
            label="candidate_budget",
            maximum=self.n_actions,
        )
        _mode(self.mode)
        _exact_float32(self.epsilon, label="epsilon", minimum=0.0, maximum=1.0)
        _exact_float32(
            self.host_surprisal_cap,
            label="host_surprisal_cap",
            minimum=0.0,
            strict_minimum=True,
        )
        for name in (
            "max_expected_improvement",
            "max_ensemble_disagreement",
            "max_information_gain",
            "max_learning_progress",
        ):
            _exact_float32(
                getattr(self, name),
                label=name,
                minimum=0.0,
                strict_minimum=True,
            )
        if self.max_expected_improvement * self.host_surprisal_cap > _FLOAT32_MAX:
            raise ValueError("expected-improvement and surprisal caps must have a finite product")
        owners = (
            _digest_tuple(self.source_owner_digest, label="source_owner_digest"),
            _digest_tuple(self.host_policy_owner_digest, label="host_policy_owner_digest"),
            _digest_tuple(self.candidate_owner_digest, label="candidate_owner_digest"),
            _digest_tuple(self.score_owner_digest, label="score_owner_digest"),
            _digest_tuple(self.safety_owner_digest, label="safety_owner_digest"),
        )
        if len(set(owners)) != len(owners):
            raise ValueError("source, policy, candidate, score, and safety owners must be distinct")

    def to_config(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        for name in (
            "source_owner_digest",
            "host_policy_owner_digest",
            "candidate_owner_digest",
            "score_owner_digest",
            "safety_owner_digest",
        ):
            payload[name] = list(payload[name])
        return {
            "type": "ProspectiveExploration",
            "schema": PROSPECTIVE_EXPLORATION_CONFIG_SCHEMA,
            "evidence_level": DELIGHTFUL_EXPLORATION_EVIDENCE_LEVEL,
            "outcome_status": DELIGHTFUL_EXPLORATION_OUTCOME_STATUS,
            "revealed_value_equivalence": DELIGHTFUL_EXPLORATION_REVEALED_VALUE_EQUIVALENCE,
            "noisy_bandit_semantics": DELIGHTFUL_EXPLORATION_NOISY_BANDIT_SEMANTICS,
            "candidate_budget_contract": "same-fixed-budget-for-all-comparator-modes",
            "candidate_generation_before_safety_shield": True,
            "causal_attestation_is_proof": False,
            "action_dispatch_authority": DELIGHTFUL_EXPLORATION_ACTION_DISPATCH_AUTHORITY,
            "policy_override_authority": DELIGHTFUL_EXPLORATION_POLICY_OVERRIDE_AUTHORITY,
            "physical_safety_claim": DELIGHTFUL_EXPLORATION_PHYSICAL_SAFETY_CLAIM,
            "scientific_promotion_allowed": DELIGHTFUL_EXPLORATION_SCIENTIFIC_PROMOTION_ALLOWED,
            "output_write_authority": DELIGHTFUL_EXPLORATION_OUTPUT_WRITE_AUTHORITY,
            **payload,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> DelightfulExplorationConfig:
        fixed = {
            "type",
            "schema",
            "evidence_level",
            "outcome_status",
            "revealed_value_equivalence",
            "noisy_bandit_semantics",
            "candidate_budget_contract",
            "candidate_generation_before_safety_shield",
            "causal_attestation_is_proof",
            "action_dispatch_authority",
            "policy_override_authority",
            "physical_safety_claim",
            "scientific_promotion_allowed",
            "output_write_authority",
        }
        expected = {field.name for field in dataclasses.fields(cls)} | fixed
        fields = _exact_manifest(payload, expected, label="prospective exploration config")
        supplied_type = fields["type"]
        supplied_schema = fields["schema"]
        canonical_identity = (
            supplied_type == "ProspectiveExploration"
            and supplied_schema == PROSPECTIVE_EXPLORATION_CONFIG_SCHEMA
        )
        legacy_identity = (
            supplied_type == "DelightfulExploration"
            and supplied_schema == DELIGHTFUL_EXPLORATION_CONFIG_SCHEMA
        )
        if not (canonical_identity or legacy_identity):
            raise ValueError("type and schema are incompatible with prospective exploration")
        required_fixed: dict[str, object] = {
            "evidence_level": DELIGHTFUL_EXPLORATION_EVIDENCE_LEVEL,
            "outcome_status": DELIGHTFUL_EXPLORATION_OUTCOME_STATUS,
            "revealed_value_equivalence": DELIGHTFUL_EXPLORATION_REVEALED_VALUE_EQUIVALENCE,
            "noisy_bandit_semantics": DELIGHTFUL_EXPLORATION_NOISY_BANDIT_SEMANTICS,
            "candidate_budget_contract": "same-fixed-budget-for-all-comparator-modes",
            "candidate_generation_before_safety_shield": True,
            "causal_attestation_is_proof": False,
            "action_dispatch_authority": False,
            "policy_override_authority": False,
            "physical_safety_claim": False,
            "scientific_promotion_allowed": False,
            "output_write_authority": False,
        }
        fields.pop("type")
        fields.pop("schema")
        for name, expected_value in required_fixed.items():
            if fields.pop(name) != expected_value:
                raise ValueError(f"{name} is incompatible with the strict mechanism boundary")
        for name in (
            "source_owner_digest",
            "host_policy_owner_digest",
            "candidate_owner_digest",
            "score_owner_digest",
            "safety_owner_digest",
        ):
            value = fields[name]
            if type(value) is not list:
                raise TypeError(f"{name} must be a JSON list")
            fields[name] = tuple(value)
        if legacy_identity and fields["mode"] == "prospective_delight":
            fields["mode"] = "expected_improvement_surprisal"
        return cls(**fields)


@chex.dataclass(frozen=True)
class DelightfulExplorationState:
    """Fixed persistent state: typed RNG, exact clock, and last causal receipts."""

    rng_key: Array
    decision_words: UInt[Array, " 2"]
    last_source_event_words: UInt[Array, " 2"]
    last_host_policy_revision_words: UInt[Array, " 2"]
    last_candidate_revision_words: UInt[Array, " 2"]
    last_score_revision_words: UInt[Array, " 2"]
    last_safety_revision_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class ExplorationCandidateBatch:
    """One fixed-budget, pre-decision candidate/source/ownership receipt."""

    candidate_actions: Int[Array, " candidate"]
    candidate_identity_words: UInt[Array, "candidate 2"]
    candidate_valid: Bool[Array, " candidate"]
    host_policy: Float[Array, " action"]
    host_action: Int[Array, ""]
    expected_improvement: Float[Array, " candidate"]
    ensemble_disagreement: Float[Array, " candidate"]
    information_gain: Float[Array, " candidate"]
    learning_progress: Float[Array, " candidate"]
    candidate_safety_allowed: Bool[Array, " candidate"]
    host_action_safety_allowed: Bool[Array, ""]
    source_event_words: UInt[Array, " 2"]
    candidate_source_event_words: UInt[Array, " 2"]
    score_source_event_words: UInt[Array, " 2"]
    host_policy_source_event_words: UInt[Array, " 2"]
    safety_source_event_words: UInt[Array, " 2"]
    host_policy_revision_words: UInt[Array, " 2"]
    candidate_revision_words: UInt[Array, " 2"]
    score_revision_words: UInt[Array, " 2"]
    safety_revision_words: UInt[Array, " 2"]
    source_owner_digest: UInt[Array, " digest"]
    host_policy_owner_digest: UInt[Array, " digest"]
    candidate_owner_digest: UInt[Array, " digest"]
    score_owner_digest: UInt[Array, " digest"]
    safety_owner_digest: UInt[Array, " digest"]
    causal_pre_decision_attested: Bool[Array, ""]


@chex.dataclass(frozen=True)
class DelightfulExplorationResult:
    """Generated candidate, post-generation shield result, and audit diagnostics."""

    state: DelightfulExplorationState
    host_relative_surprisal: Float[Array, " candidate"]
    expected_improvement_surprisal_score: Float[Array, " candidate"]
    comparator_priority: Float[Array, " candidate"]
    selected_index: Int[Array, ""]
    selected_candidate_action: Int[Array, ""]
    selected_candidate_identity_words: UInt[Array, " 2"]
    selected_expected_improvement: Float[Array, ""]
    selected_host_relative_surprisal: Float[Array, ""]
    selected_expected_improvement_surprisal_score: Float[Array, ""]
    proposed_executable_action: Int[Array, ""]
    pre_decision_words: UInt[Array, " 2"]
    post_decision_words: UInt[Array, " 2"]
    state_valid: Bool[Array, ""]
    owner_binding_valid: Bool[Array, ""]
    causal_binding_valid: Bool[Array, ""]
    host_policy_valid: Bool[Array, ""]
    candidate_batch_valid: Bool[Array, ""]
    score_valid: Bool[Array, ""]
    source_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    decision_applied: Bool[Array, ""]
    candidate_generated: Bool[Array, ""]
    candidate_passed_hard_shield: Bool[Array, ""]
    candidate_override_proposed: Bool[Array, ""]
    host_fallback_used: Bool[Array, ""]
    proposed_executable_action_available: Bool[Array, ""]

    @property
    def prospective_delight(self) -> Float[Array, " candidate"]:
        """Deprecated v1 alias; this score is not policy-gradient delight."""

        return self.expected_improvement_surprisal_score

    @property
    def selected_prospective_delight(self) -> Float[Array, ""]:
        """Deprecated v1 alias; no actor backward executes in this selector."""

        return self.selected_expected_improvement_surprisal_score


@dataclasses.dataclass(frozen=True, slots=True)
class DelightfulExplorationResourceBudget:
    """Exact persistent state bytes and fixed logical per-decision resources."""

    schema: str
    persistent_bytes_scope: str
    temporary_bytes_scope: str
    rng_nbytes: int
    clock_and_receipt_nbytes: int
    total_state_nbytes: int
    fixed_candidate_budget: int
    logical_uniform_draws_per_decision: int
    candidate_metric_scalars_per_decision: int


@chex.dataclass(frozen=True)
class DelightfulExplorationScanResult:
    """Final state and selected diagnostics from a JAX scan."""

    state: DelightfulExplorationState
    selected_indices: Int[Array, " steps"]
    selected_candidate_actions: Int[Array, " steps"]
    proposed_executable_actions: Int[Array, " steps"]
    selected_expected_improvement_surprisal_score: Float[Array, " steps"]
    decision_applied: Bool[Array, " steps"]
    candidate_passed_hard_shield: Bool[Array, " steps"]
    proposed_executable_action_available: Bool[Array, " steps"]

    @property
    def selected_prospective_delight(self) -> Float[Array, " steps"]:
        """Deprecated v1 alias for checkpoint/read compatibility only."""

        return self.selected_expected_improvement_surprisal_score


def measure_delightful_exploration_state_nbytes(state: DelightfulExplorationState) -> int:
    """Return bytes occupied by every persistent array leaf."""

    return sum(_array_nbytes(leaf) for leaf in jax.tree.leaves(state))


class DelightfulExploration:
    """Fixed-budget prospective selector with a caller-owned hard shield."""

    def __init__(self, config: DelightfulExplorationConfig) -> None:
        if type(config) is not DelightfulExplorationConfig:
            raise TypeError("config must be an exact ProspectiveExplorationConfig")
        self._config = config

    @property
    def config(self) -> DelightfulExplorationConfig:
        return self._config

    def to_config(self) -> dict[str, Any]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> DelightfulExploration:
        return cls(DelightfulExplorationConfig.from_config(payload))

    def init(self, key: Array) -> DelightfulExplorationState:
        """Initialize a zero-receipt state from one typed Threefry key."""

        _require_threefry_key(key, label="key")
        zero = jnp.zeros((2,), dtype=jnp.uint32)
        return DelightfulExplorationState(
            rng_key=key,
            decision_words=zero,
            last_source_event_words=zero,
            last_host_policy_revision_words=zero,
            last_candidate_revision_words=zero,
            last_score_revision_words=zero,
            last_safety_revision_words=zero,
        )

    def _require_state_contract(self, state: DelightfulExplorationState) -> None:
        if type(state) is not DelightfulExplorationState:
            raise TypeError("state must be an exact ProspectiveExplorationState")
        _require_threefry_key(state.rng_key, label="state.rng_key")
        for name in (
            "decision_words",
            "last_source_event_words",
            "last_host_policy_revision_words",
            "last_candidate_revision_words",
            "last_score_revision_words",
            "last_safety_revision_words",
        ):
            _require_array(
                getattr(state, name),
                label=name,
                shape=(2,),
                dtype=jnp.dtype(jnp.uint32),
            )

    def _require_batch_contract(self, batch: ExplorationCandidateBatch) -> None:
        if type(batch) is not ExplorationCandidateBatch:
            raise TypeError("batch must be an exact ExplorationCandidateBatch")
        budget = self._config.candidate_budget
        actions = self._config.n_actions
        contracts = (
            (batch.candidate_actions, (budget,), jnp.int32, "candidate_actions"),
            (
                batch.candidate_identity_words,
                (budget, 2),
                jnp.uint32,
                "candidate_identity_words",
            ),
            (batch.candidate_valid, (budget,), jnp.bool_, "candidate_valid"),
            (batch.host_policy, (actions,), jnp.float32, "host_policy"),
            (batch.host_action, (), jnp.int32, "host_action"),
            (
                batch.expected_improvement,
                (budget,),
                jnp.float32,
                "expected_improvement",
            ),
            (
                batch.ensemble_disagreement,
                (budget,),
                jnp.float32,
                "ensemble_disagreement",
            ),
            (batch.information_gain, (budget,), jnp.float32, "information_gain"),
            (batch.learning_progress, (budget,), jnp.float32, "learning_progress"),
            (
                batch.candidate_safety_allowed,
                (budget,),
                jnp.bool_,
                "candidate_safety_allowed",
            ),
            (
                batch.host_action_safety_allowed,
                (),
                jnp.bool_,
                "host_action_safety_allowed",
            ),
            (
                batch.causal_pre_decision_attested,
                (),
                jnp.bool_,
                "causal_pre_decision_attested",
            ),
        )
        for value, shape, dtype, label in contracts:
            _require_array(value, label=label, shape=shape, dtype=jnp.dtype(dtype))
        for name in (
            "source_event_words",
            "candidate_source_event_words",
            "score_source_event_words",
            "host_policy_source_event_words",
            "safety_source_event_words",
            "host_policy_revision_words",
            "candidate_revision_words",
            "score_revision_words",
            "safety_revision_words",
        ):
            _require_array(
                getattr(batch, name),
                label=name,
                shape=(2,),
                dtype=jnp.dtype(jnp.uint32),
            )
        for name in (
            "source_owner_digest",
            "host_policy_owner_digest",
            "candidate_owner_digest",
            "score_owner_digest",
            "safety_owner_digest",
        ):
            _require_array(
                getattr(batch, name),
                label=name,
                shape=(_DIGEST_WORDS,),
                dtype=jnp.dtype(jnp.uint32),
            )

    def state_valid(self, state: DelightfulExplorationState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return jnp.asarray(True, dtype=jnp.bool_)

    def select(
        self,
        state: DelightfulExplorationState,
        batch: ExplorationCandidateBatch,
    ) -> DelightfulExplorationResult:
        """Generate one candidate, then apply the hard shield to its proposal."""

        self._require_state_contract(state)
        self._require_batch_contract(batch)
        config = self._config
        budget = config.candidate_budget

        state_valid = self.state_valid(state)
        expected_owners = (
            config.source_owner_digest,
            config.host_policy_owner_digest,
            config.candidate_owner_digest,
            config.score_owner_digest,
            config.safety_owner_digest,
        )
        observed_owners = (
            batch.source_owner_digest,
            batch.host_policy_owner_digest,
            batch.candidate_owner_digest,
            batch.score_owner_digest,
            batch.safety_owner_digest,
        )
        owner_binding_valid = jnp.asarray(True, dtype=jnp.bool_)
        for observed, expected in zip(observed_owners, expected_owners, strict=True):
            owner_binding_valid = owner_binding_valid & jnp.all(
                observed == jnp.asarray(expected, dtype=jnp.uint32)
            )

        same_source = (
            jnp.all(batch.candidate_source_event_words == batch.source_event_words)
            & jnp.all(batch.score_source_event_words == batch.source_event_words)
            & jnp.all(batch.host_policy_source_event_words == batch.source_event_words)
            & jnp.all(batch.safety_source_event_words == batch.source_event_words)
        )
        causal_binding_valid = (
            batch.causal_pre_decision_attested
            & same_source
            & _words_greater(batch.source_event_words, state.last_source_event_words)
            & _words_greater_equal(
                batch.host_policy_revision_words,
                state.last_host_policy_revision_words,
            )
            & _words_greater_equal(
                batch.candidate_revision_words,
                state.last_candidate_revision_words,
            )
            & _words_greater_equal(
                batch.score_revision_words,
                state.last_score_revision_words,
            )
            & _words_greater_equal(
                batch.safety_revision_words,
                state.last_safety_revision_words,
            )
        )

        policy_sum = jnp.sum(batch.host_policy, dtype=jnp.float32)
        policy_tolerance = jnp.asarray(8.0 * np.finfo(np.float32).eps, dtype=jnp.float32)
        host_policy_valid = (
            jnp.all(jnp.isfinite(batch.host_policy))
            & jnp.all(batch.host_policy >= 0.0)
            & jnp.all(batch.host_policy <= 1.0)
            & (jnp.abs(policy_sum - 1.0) <= policy_tolerance)
            & (batch.host_action >= 0)
            & (batch.host_action < config.n_actions)
        )
        safe_host_action = jnp.clip(batch.host_action, 0, config.n_actions - 1)
        host_policy_valid = host_policy_valid & (batch.host_policy[safe_host_action] > 0.0)

        valid = batch.candidate_valid
        action_is_valid = (batch.candidate_actions >= 0) & (
            batch.candidate_actions < config.n_actions
        )
        padded_action_is_canonical = jnp.where(
            valid,
            action_is_valid,
            batch.candidate_actions == -1,
        )
        identity_nonzero = jnp.any(batch.candidate_identity_words != 0, axis=1)
        padded_identity_is_canonical = jnp.where(
            valid,
            identity_nonzero,
            ~identity_nonzero,
        )
        pair_valid = valid[:, None] & valid[None, :]
        off_diagonal = ~jnp.eye(budget, dtype=jnp.bool_)
        duplicate_action = (
            pair_valid
            & off_diagonal
            & (batch.candidate_actions[:, None] == batch.candidate_actions[None, :])
        )
        duplicate_identity = (
            pair_valid
            & off_diagonal
            & jnp.all(
                batch.candidate_identity_words[:, None, :]
                == batch.candidate_identity_words[None, :, :],
                axis=2,
            )
        )
        candidate_batch_valid = (
            jnp.any(valid)
            & jnp.all(padded_action_is_canonical)
            & jnp.all(padded_identity_is_canonical)
            & ~jnp.any(duplicate_action)
            & ~jnp.any(duplicate_identity)
            & jnp.all(valid | ~batch.candidate_safety_allowed)
        )

        metric_contracts = (
            (batch.expected_improvement, config.max_expected_improvement),
            (batch.ensemble_disagreement, config.max_ensemble_disagreement),
            (batch.information_gain, config.max_information_gain),
            (batch.learning_progress, config.max_learning_progress),
        )
        score_valid = jnp.asarray(True, dtype=jnp.bool_)
        for values, maximum in metric_contracts:
            score_valid = score_valid & jnp.all(jnp.isfinite(values))
            score_valid = score_valid & jnp.all(
                jnp.where(valid, (values >= 0.0) & (values <= maximum), values == 0.0)
            )

        safe_candidate_actions = jnp.clip(batch.candidate_actions, 0, config.n_actions - 1)
        candidate_host_probability = batch.host_policy[safe_candidate_actions]
        cap = jnp.asarray(config.host_surprisal_cap, dtype=jnp.float32)
        uncapped_surprisal = jnp.where(
            candidate_host_probability > 0.0,
            -jnp.log(
                jnp.maximum(
                    candidate_host_probability,
                    jnp.asarray(_FLOAT32_TINY, dtype=jnp.float32),
                )
            ),
            cap,
        )
        host_relative_surprisal = jnp.where(
            valid,
            jnp.minimum(uncapped_surprisal, cap),
            jnp.float32(0.0),
        )
        expected_improvement_surprisal_score = jnp.where(
            valid,
            batch.expected_improvement * host_relative_surprisal,
            jnp.float32(0.0),
        )
        score_valid = score_valid & jnp.all(
            jnp.isfinite(expected_improvement_surprisal_score)
        )

        next_key, priority_key, epsilon_key = jr.split(state.rng_key, 3)
        lower = jnp.asarray(np.finfo(np.float32).tiny, dtype=jnp.float32)
        uniforms = jr.uniform(
            priority_key,
            (budget,),
            dtype=jnp.float32,
            minval=lower,
            maxval=jnp.float32(1.0),
        )
        random_priority = -jnp.log(-jnp.log(uniforms))
        epsilon_draw = jr.uniform(epsilon_key, (), dtype=jnp.float32)
        if config.mode == "expected_improvement_surprisal":
            comparator_priority = expected_improvement_surprisal_score
        elif config.mode == "random":
            comparator_priority = random_priority
        elif config.mode == "epsilon_greedy":
            comparator_priority = jnp.where(
                epsilon_draw < jnp.asarray(config.epsilon, dtype=jnp.float32),
                random_priority,
                batch.expected_improvement,
            )
        elif config.mode == "ensemble_disagreement":
            comparator_priority = batch.ensemble_disagreement
        elif config.mode == "information_gain":
            comparator_priority = batch.information_gain
        else:
            comparator_priority = batch.learning_progress

        source_valid = (
            state_valid
            & owner_binding_valid
            & causal_binding_valid
            & host_policy_valid
            & candidate_batch_valid
            & score_valid
        )
        proposed_decision_words, lifetime_capacity = _increment_words(state.decision_words)
        decision_applied = source_valid & lifetime_capacity

        masked_priority = jnp.where(valid, comparator_priority, -jnp.inf)
        generated_index = jnp.argmax(masked_priority).astype(jnp.int32)
        safe_index = jnp.clip(generated_index, 0, budget - 1)
        selected_action = batch.candidate_actions[safe_index]
        selected_identity = batch.candidate_identity_words[safe_index]
        selected_shield = batch.candidate_safety_allowed[safe_index]

        candidate_passed_hard_shield = decision_applied & selected_shield
        host_fallback_used = (
            decision_applied & ~selected_shield & batch.host_action_safety_allowed
        )
        proposal_available = candidate_passed_hard_shield | host_fallback_used
        proposed_action = jnp.where(
            candidate_passed_hard_shield,
            selected_action,
            jnp.where(host_fallback_used, batch.host_action, jnp.int32(-1)),
        )
        candidate_override_proposed = (
            candidate_passed_hard_shield & (selected_action != batch.host_action)
        )

        candidate_state = DelightfulExplorationState(
            rng_key=next_key,
            decision_words=proposed_decision_words,
            last_source_event_words=batch.source_event_words,
            last_host_policy_revision_words=batch.host_policy_revision_words,
            last_candidate_revision_words=batch.candidate_revision_words,
            last_score_revision_words=batch.score_revision_words,
            last_safety_revision_words=batch.safety_revision_words,
        )
        next_state = jax.lax.cond(
            decision_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )
        diagnostic_scores = jnp.where(decision_applied & valid, comparator_priority, 0.0)
        return DelightfulExplorationResult(
            state=next_state,
            host_relative_surprisal=jnp.where(
                decision_applied,
                host_relative_surprisal,
                jnp.zeros_like(host_relative_surprisal),
            ),
            expected_improvement_surprisal_score=jnp.where(
                decision_applied,
                expected_improvement_surprisal_score,
                jnp.zeros_like(expected_improvement_surprisal_score),
            ),
            comparator_priority=diagnostic_scores,
            selected_index=jnp.where(decision_applied, generated_index, jnp.int32(-1)),
            selected_candidate_action=jnp.where(
                decision_applied,
                selected_action,
                jnp.int32(-1),
            ),
            selected_candidate_identity_words=jnp.where(
                decision_applied,
                selected_identity,
                jnp.zeros((2,), dtype=jnp.uint32),
            ),
            selected_expected_improvement=jnp.where(
                decision_applied,
                batch.expected_improvement[safe_index],
                jnp.float32(0.0),
            ),
            selected_host_relative_surprisal=jnp.where(
                decision_applied,
                host_relative_surprisal[safe_index],
                jnp.float32(0.0),
            ),
            selected_expected_improvement_surprisal_score=jnp.where(
                decision_applied,
                expected_improvement_surprisal_score[safe_index],
                jnp.float32(0.0),
            ),
            proposed_executable_action=proposed_action,
            pre_decision_words=state.decision_words,
            post_decision_words=next_state.decision_words,
            state_valid=state_valid,
            owner_binding_valid=owner_binding_valid,
            causal_binding_valid=causal_binding_valid,
            host_policy_valid=host_policy_valid,
            candidate_batch_valid=candidate_batch_valid,
            score_valid=score_valid,
            source_valid=source_valid,
            lifetime_capacity_available=lifetime_capacity,
            decision_applied=decision_applied,
            candidate_generated=decision_applied,
            candidate_passed_hard_shield=candidate_passed_hard_shield,
            candidate_override_proposed=candidate_override_proposed,
            host_fallback_used=host_fallback_used,
            proposed_executable_action_available=proposal_available,
        )

    def resource_budget(
        self,
        state: DelightfulExplorationState,
    ) -> DelightfulExplorationResourceBudget:
        """Partition persistent bytes and disclose fixed logical scratch scope."""

        self._require_state_contract(state)
        rng_nbytes = _array_nbytes(state.rng_key)
        receipts = (
            state.decision_words,
            state.last_source_event_words,
            state.last_host_policy_revision_words,
            state.last_candidate_revision_words,
            state.last_score_revision_words,
            state.last_safety_revision_words,
        )
        receipt_nbytes = sum(_array_nbytes(value) for value in receipts)
        total = rng_nbytes + receipt_nbytes
        if total != measure_delightful_exploration_state_nbytes(state):
            raise AssertionError("prospective exploration resource partition is incomplete")
        return DelightfulExplorationResourceBudget(
            schema=PROSPECTIVE_EXPLORATION_RESOURCE_SCHEMA,
            persistent_bytes_scope="all-persistent-state-array-leaves",
            temporary_bytes_scope=(
                "source-level-fixed-candidate-score-and-rng-arrays; excludes-input-batch,"
                "host-object-overhead,compiler-and-xla-workspaces; not-a-measured-device-peak"
            ),
            rng_nbytes=rng_nbytes,
            clock_and_receipt_nbytes=receipt_nbytes,
            total_state_nbytes=total,
            fixed_candidate_budget=self._config.candidate_budget,
            logical_uniform_draws_per_decision=self._config.candidate_budget + 1,
            candidate_metric_scalars_per_decision=4 * self._config.candidate_budget,
        )

    def save_checkpoint(self, state: DelightfulExplorationState, path: str | Path) -> None:
        """Save a valid state with an exact construction and resource receipt."""

        self._require_state_contract(state)
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot checkpoint an invalid prospective exploration state")
        _save_checkpoint(
            state,
            path,
            metadata={
                "schema": PROSPECTIVE_EXPLORATION_CHECKPOINT_SCHEMA,
                "state_schema": PROSPECTIVE_EXPLORATION_STATE_SCHEMA,
                "construction": self.to_config(),
                "state_nbytes": measure_delightful_exploration_state_nbytes(state),
                "resource_budget": dataclasses.asdict(self.resource_budget(state)),
            },
        )

    def checkpoint_metadata(self, path: str | Path) -> dict[str, Any]:
        """Load and validate the exact checkpoint metadata manifest."""

        metadata = _load_checkpoint_metadata(path)
        expected = {
            "schema",
            "state_schema",
            "construction",
            "state_nbytes",
            "resource_budget",
        }
        fields = _exact_manifest(metadata, expected, label="prospective exploration checkpoint")
        if fields["schema"] != PROSPECTIVE_EXPLORATION_CHECKPOINT_SCHEMA:
            raise ValueError("prospective exploration checkpoint schema is unsupported")
        if fields["state_schema"] != PROSPECTIVE_EXPLORATION_STATE_SCHEMA:
            raise ValueError("prospective exploration state schema is unsupported")
        if fields["construction"] != self.to_config():
            raise ValueError("prospective exploration checkpoint construction is incompatible")
        _exact_positive_int(
            fields["state_nbytes"],
            label="checkpoint state_nbytes",
            maximum=2**63 - 1,
        )
        template_budget = dataclasses.asdict(
            self.resource_budget(self.init(jr.key(0, impl="threefry2x32")))
        )
        if fields["resource_budget"] != template_budget:
            raise ValueError("prospective exploration checkpoint resource budget differs")
        return fields

    def load_checkpoint(
        self,
        state_template: DelightfulExplorationState,
        path: str | Path,
    ) -> DelightfulExplorationState:
        """Restore only an exact, construction-compatible, valid state."""

        self._require_state_contract(state_template)
        metadata = self.checkpoint_metadata(path)
        loaded_raw, restored_metadata = _load_checkpoint(state_template, path)
        loaded = cast(DelightfulExplorationState, loaded_raw)
        if restored_metadata != metadata:
            raise ValueError("checkpoint metadata changed between validation and restore")
        self._require_state_contract(loaded)
        if not bool(jax.device_get(self.state_valid(loaded))):
            raise ValueError("restored prospective exploration state is invalid")
        if measure_delightful_exploration_state_nbytes(loaded) != metadata["state_nbytes"]:
            raise ValueError("restored prospective exploration state size is invalid")
        return loaded


def run_delightful_exploration_from_batches(
    controller: DelightfulExploration,
    state: DelightfulExplorationState,
    batches: ExplorationCandidateBatch,
) -> DelightfulExplorationScanResult:
    """Run exact selection transactions through ``jax.lax.scan``."""

    if type(controller) is not DelightfulExploration:
        raise TypeError("controller must be an exact ProspectiveExploration")
    controller._require_state_contract(state)
    if type(batches) is not ExplorationCandidateBatch:
        raise TypeError("batches must be an exact ExplorationCandidateBatch")
    if getattr(batches.candidate_actions, "ndim", None) != 2:
        raise ValueError("batched candidate_actions must have rank 2")
    steps = batches.candidate_actions.shape[0]
    if steps < 1:
        raise ValueError("batches must contain at least one decision")
    for leaf in jax.tree.leaves(batches):
        if getattr(leaf, "ndim", 0) < 1 or leaf.shape[0] != steps:
            raise ValueError("every batched receipt leaf must share the leading step dimension")

    def body(
        carry: DelightfulExplorationState,
        batch: ExplorationCandidateBatch,
    ) -> tuple[DelightfulExplorationState, tuple[Array, ...]]:
        result = controller.select(carry, batch)
        output = (
            result.selected_index,
            result.selected_candidate_action,
            result.proposed_executable_action,
            result.selected_expected_improvement_surprisal_score,
            result.decision_applied,
            result.candidate_passed_hard_shield,
            result.proposed_executable_action_available,
        )
        return result.state, output

    final_state, outputs = jax.lax.scan(body, state, batches)
    return DelightfulExplorationScanResult(
        state=final_state,
        selected_indices=outputs[0],
        selected_candidate_actions=outputs[1],
        proposed_executable_actions=outputs[2],
        selected_expected_improvement_surprisal_score=outputs[3],
        decision_applied=outputs[4],
        candidate_passed_hard_shield=outputs[5],
        proposed_executable_action_available=outputs[6],
    )


# Canonical non-gradient API.  The implementation's historical class and
# function names remain import-compatible for v1 callers, but new code should
# use these names so “delight” stays reserved for an actor-gradient sample.
ProspectiveExploration = DelightfulExploration
ProspectiveExplorationConfig = DelightfulExplorationConfig
ProspectiveExplorationResourceBudget = DelightfulExplorationResourceBudget
ProspectiveExplorationResult = DelightfulExplorationResult
ProspectiveExplorationScanResult = DelightfulExplorationScanResult
ProspectiveExplorationState = DelightfulExplorationState
for _canonical_type, _canonical_name in (
    (ProspectiveExploration, "ProspectiveExploration"),
    (ProspectiveExplorationConfig, "ProspectiveExplorationConfig"),
    (ProspectiveExplorationResourceBudget, "ProspectiveExplorationResourceBudget"),
    (ProspectiveExplorationResult, "ProspectiveExplorationResult"),
    (ProspectiveExplorationScanResult, "ProspectiveExplorationScanResult"),
    (ProspectiveExplorationState, "ProspectiveExplorationState"),
):
    _canonical_type.__name__ = _canonical_name
    _canonical_type.__qualname__ = _canonical_name
PROSPECTIVE_EXPLORATION_MODES = DELIGHTFUL_EXPLORATION_MODES
PROSPECTIVE_EXPLORATION_EVIDENCE_LEVEL = DELIGHTFUL_EXPLORATION_EVIDENCE_LEVEL
PROSPECTIVE_EXPLORATION_OUTCOME_STATUS = DELIGHTFUL_EXPLORATION_OUTCOME_STATUS
PROSPECTIVE_EXPLORATION_LIFETIME_SEMANTICS = DELIGHTFUL_EXPLORATION_LIFETIME_SEMANTICS
PROSPECTIVE_EXPLORATION_REVEALED_VALUE_EQUIVALENCE = (
    DELIGHTFUL_EXPLORATION_REVEALED_VALUE_EQUIVALENCE
)
PROSPECTIVE_EXPLORATION_NOISY_BANDIT_SEMANTICS = (
    DELIGHTFUL_EXPLORATION_NOISY_BANDIT_SEMANTICS
)
PROSPECTIVE_EXPLORATION_ACTION_DISPATCH_AUTHORITY = (
    DELIGHTFUL_EXPLORATION_ACTION_DISPATCH_AUTHORITY
)
PROSPECTIVE_EXPLORATION_POLICY_OVERRIDE_AUTHORITY = (
    DELIGHTFUL_EXPLORATION_POLICY_OVERRIDE_AUTHORITY
)
PROSPECTIVE_EXPLORATION_PHYSICAL_SAFETY_CLAIM = (
    DELIGHTFUL_EXPLORATION_PHYSICAL_SAFETY_CLAIM
)
PROSPECTIVE_EXPLORATION_SCIENTIFIC_PROMOTION_ALLOWED = (
    DELIGHTFUL_EXPLORATION_SCIENTIFIC_PROMOTION_ALLOWED
)
PROSPECTIVE_EXPLORATION_OUTPUT_WRITE_AUTHORITY = (
    DELIGHTFUL_EXPLORATION_OUTPUT_WRITE_AUTHORITY
)
measure_prospective_exploration_state_nbytes = measure_delightful_exploration_state_nbytes
run_prospective_exploration_from_batches = run_delightful_exploration_from_batches


__all__ = [
    "PROSPECTIVE_EXPLORATION_ACTION_DISPATCH_AUTHORITY",
    "PROSPECTIVE_EXPLORATION_CHECKPOINT_SCHEMA",
    "PROSPECTIVE_EXPLORATION_CONFIG_SCHEMA",
    "PROSPECTIVE_EXPLORATION_EVIDENCE_LEVEL",
    "PROSPECTIVE_EXPLORATION_EXECUTES_ACTOR_BACKWARD",
    "PROSPECTIVE_EXPLORATION_GRADIENT_DELIGHT_SEMANTICS",
    "PROSPECTIVE_EXPLORATION_LIFETIME_SEMANTICS",
    "PROSPECTIVE_EXPLORATION_MODES",
    "PROSPECTIVE_EXPLORATION_NOISY_BANDIT_SEMANTICS",
    "PROSPECTIVE_EXPLORATION_OUTCOME_STATUS",
    "PROSPECTIVE_EXPLORATION_OUTPUT_WRITE_AUTHORITY",
    "PROSPECTIVE_EXPLORATION_PHYSICAL_SAFETY_CLAIM",
    "PROSPECTIVE_EXPLORATION_POLICY_OVERRIDE_AUTHORITY",
    "PROSPECTIVE_EXPLORATION_RESOURCE_SCHEMA",
    "PROSPECTIVE_EXPLORATION_REVEALED_VALUE_EQUIVALENCE",
    "PROSPECTIVE_EXPLORATION_SCIENTIFIC_PROMOTION_ALLOWED",
    "PROSPECTIVE_EXPLORATION_SCORE_SEMANTICS",
    "PROSPECTIVE_EXPLORATION_STATE_SCHEMA",
    "ProspectiveExploration",
    "ProspectiveExplorationConfig",
    "ProspectiveExplorationResourceBudget",
    "ProspectiveExplorationResult",
    "ProspectiveExplorationScanResult",
    "ProspectiveExplorationState",
    "measure_prospective_exploration_state_nbytes",
    "run_prospective_exploration_from_batches",
    "DELIGHTFUL_EXPLORATION_ACTION_DISPATCH_AUTHORITY",
    "DELIGHTFUL_EXPLORATION_CHECKPOINT_SCHEMA",
    "DELIGHTFUL_EXPLORATION_CONFIG_SCHEMA",
    "DELIGHTFUL_EXPLORATION_EVIDENCE_LEVEL",
    "DELIGHTFUL_EXPLORATION_LIFETIME_SEMANTICS",
    "DELIGHTFUL_EXPLORATION_MODES",
    "DELIGHTFUL_EXPLORATION_NOISY_BANDIT_SEMANTICS",
    "DELIGHTFUL_EXPLORATION_OUTCOME_STATUS",
    "DELIGHTFUL_EXPLORATION_OUTPUT_WRITE_AUTHORITY",
    "DELIGHTFUL_EXPLORATION_PHYSICAL_SAFETY_CLAIM",
    "DELIGHTFUL_EXPLORATION_POLICY_OVERRIDE_AUTHORITY",
    "DELIGHTFUL_EXPLORATION_RESOURCE_SCHEMA",
    "DELIGHTFUL_EXPLORATION_REVEALED_VALUE_EQUIVALENCE",
    "DELIGHTFUL_EXPLORATION_SCIENTIFIC_PROMOTION_ALLOWED",
    "DELIGHTFUL_EXPLORATION_STATE_SCHEMA",
    "DelightfulExploration",
    "DelightfulExplorationConfig",
    "DelightfulExplorationResourceBudget",
    "DelightfulExplorationResult",
    "DelightfulExplorationScanResult",
    "DelightfulExplorationState",
    "ExplorationCandidateBatch",
    "ExplorationMode",
    "measure_delightful_exploration_state_nbytes",
    "run_delightful_exploration_from_batches",
]
