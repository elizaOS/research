"""Focused contracts for authenticated curation-trace to ledger binding."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core import compositional_features as cf
from alberta_framework.evaluation import generated_birth_identity_ledger as ledger_module
from alberta_framework.evaluation import generated_birth_identity_trace_binding as binding_module
from alberta_framework.evaluation.generated_birth_identity_ledger import (
    GENERATED_BIRTH_IDENTITY_LEDGER_V4_SCHEMA,
    GeneratedBirthIdentityLedgerV4Config,
    derive_generated_birth_identity_v4,
    generated_birth_identity_ledger_v4_state_sha256,
)
from alberta_framework.evaluation.generated_birth_identity_trace_binding import (
    GeneratedBirthIdentityTraceBindingError,
    attach_generated_birth_identity_ledger_at_core_genesis,
    authenticate_generated_birth_identity_trace_by_source_replay,
    bind_generated_birth_identity_trace_structurally,
)

pytestmark = pytest.mark.unit


@dataclasses.dataclass(frozen=True)
class _Case:
    learner: cf.CompositionalFeatureLearner
    state: cf.CompositionalFeatureState
    config: GeneratedBirthIdentityLedgerV4Config
    ledger: ledger_module.GeneratedBirthIdentityLedgerV4State
    observation: jax.Array
    targets: jax.Array
    result: cf.CompositionalFeatureUpdateResult


def _case(
    *,
    learner: cf.CompositionalFeatureLearner,
    state: cf.CompositionalFeatureState,
    raw_features: int,
    namespace: str,
    observation: tuple[float, ...],
    targets: tuple[float, ...],
    seed: int,
) -> _Case:
    config = GeneratedBirthIdentityLedgerV4Config(
        namespace=namespace,
        active_slots=learner._n_features,  # noqa: SLF001
        candidate_slots=learner._candidate_count,  # noqa: SLF001
        raw_feature_slots=raw_features,
        max_depth=learner._max_depth,  # noqa: SLF001
        learn_generator_resources=learner._learn_generator_resources,  # noqa: SLF001
    )
    sidecar = attach_generated_birth_identity_ledger_at_core_genesis(
        config,
        learner_pre_state=state,
        paired_development_life_seed=seed,
    )
    observation_array = jnp.asarray(observation, dtype=jnp.float32)
    target_array = jnp.asarray(targets, dtype=jnp.float32)
    result = learner.update(state, observation_array, target_array)
    result.state.step_count.block_until_ready()
    return _Case(
        learner=learner,
        state=state,
        config=config,
        ledger=sidecar,
        observation=observation_array,
        targets=target_array,
        result=result,
    )


@pytest.fixture(scope="module")
def no_event_case() -> _Case:
    learner = cf.CompositionalFeatureLearner(
        n_features=3,
        n_tasks=1,
        candidate_count=1,
        replacement_interval=32,
        min_feature_age=100,
        candidate_min_age=16,
        use_obgd=False,
    )
    return _case(
        learner=learner,
        state=learner.init(feature_dim=2, key=jr.key(4100)),
        raw_features=2,
        namespace="trace-binding-no-event-development",
        observation=(0.25, -0.5),
        targets=(0.75,),
        seed=4100,
    )


@pytest.fixture(scope="module")
def ordinary_case() -> _Case:
    learner = cf.CompositionalFeatureLearner(
        n_features=4,
        n_tasks=1,
        candidate_count=1,
        step_size_output=0.0,
        step_size_theta=0.0,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0e6,
        use_obgd=False,
    )
    state = learner.init(feature_dim=2, key=jr.key(4101)).replace(  # type: ignore[attr-defined]
        ages=jnp.full((4,), 10, dtype=jnp.int32),
        utilities=jnp.asarray((10.0, 10.0, 1.0, 1.0), dtype=jnp.float32),
        candidate_ages=jnp.asarray((10,), dtype=jnp.int32),
        candidate_utilities=jnp.asarray((0.0,), dtype=jnp.float32),
    )
    return _case(
        learner=learner,
        state=state,
        raw_features=2,
        namespace="trace-binding-ordinary-refresh-development",
        observation=(0.2, -0.1),
        targets=(0.0,),
        seed=4101,
    )


@pytest.fixture(scope="module")
def promotion_case() -> _Case:
    learner = cf.CompositionalFeatureLearner(
        n_features=5,
        n_tasks=1,
        candidate_count=2,
        step_size_output=0.0,
        step_size_theta=0.0,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=0.0,
        max_depth=4,
        use_obgd=False,
    )
    state = learner.init(feature_dim=3, key=jr.key(4102)).replace(  # type: ignore[attr-defined]
        ops=jnp.asarray(
            (cf.OP_RAW, cf.OP_RAW, cf.OP_RAW, cf.OP_PRODUCT, cf.OP_PRODUCT),
            dtype=jnp.int32,
        ),
        parent_a=jnp.asarray((0, 1, 2, 0, 3), dtype=jnp.int32),
        parent_b=jnp.asarray((-1, -1, -1, 1, 2), dtype=jnp.int32),
        theta=jnp.zeros((5, 2), dtype=jnp.float32),
        depth=jnp.asarray((0, 0, 0, 1, 2), dtype=jnp.int32),
        output_weights=jnp.zeros((1, 5), dtype=jnp.float32),
        utilities=jnp.asarray((10.0, 10.0, 10.0, 0.0, 10.0), dtype=jnp.float32),
        ages=jnp.full((5,), 10, dtype=jnp.int32),
        feature_generator_policy=jnp.zeros((5,), dtype=jnp.int32),
        candidate_ops=jnp.asarray((cf.OP_SUM, cf.OP_PRODUCT), dtype=jnp.int32),
        candidate_parent_a=jnp.asarray((0, 4), dtype=jnp.int32),
        candidate_parent_b=jnp.asarray((1, 2), dtype=jnp.int32),
        candidate_theta=jnp.zeros((2, 2), dtype=jnp.float32),
        candidate_depth=jnp.asarray((1, 3), dtype=jnp.int32),
        candidate_output_weights=jnp.asarray(((6.0, 7.0),), dtype=jnp.float32),
        candidate_utilities=jnp.asarray((100.0, 1.0), dtype=jnp.float32),
        candidate_ages=jnp.asarray((10, 10), dtype=jnp.int32),
        candidate_generator_policy=jnp.zeros((2,), dtype=jnp.int32),
    )
    return _case(
        learner=learner,
        state=state,
        raw_features=3,
        namespace="trace-binding-promotion-cascade-development",
        observation=(1.0, 1.0, 1.0),
        targets=(0.0,),
        seed=4102,
    )


@pytest.fixture(scope="module")
def direct_case() -> _Case:
    learner = cf.CompositionalFeatureLearner(
        n_features=3,
        n_tasks=1,
        candidate_count=0,
        step_size_output=0.0,
        step_size_theta=0.0,
        replacement_interval=1,
        min_feature_age=0,
        max_depth=2,
        use_obgd=False,
    )
    state = learner.init(feature_dim=2, key=jr.key(4103)).replace(  # type: ignore[attr-defined]
        ages=jnp.full((3,), 10, dtype=jnp.int32),
        utilities=jnp.asarray((10.0, 10.0, 0.0), dtype=jnp.float32),
    )
    return _case(
        learner=learner,
        state=state,
        raw_features=2,
        namespace="trace-binding-direct-raw-parent-development",
        observation=(0.3, -0.6),
        targets=(0.0,),
        seed=4103,
    )


def _authenticate(case: _Case) -> binding_module.GeneratedBirthIdentityTraceBinding:
    return authenticate_generated_birth_identity_trace_by_source_replay(
        case.learner,
        case.config,
        case.ledger,
        learner_pre_state=case.state,
        learner_post_state=case.result.state,
        supplied_update_result=case.result,
        observation=case.observation,
        targets=case.targets,
    )


def test_pins_and_manifests_are_exact_current_source_contracts() -> None:
    assert (
        binding_module.PINNED_COMPOSITIONAL_FEATURES_MODULE_SHA256
        == hashlib.sha256(Path(cf.__file__).read_bytes()).hexdigest()
    )
    assert (
        binding_module.PINNED_GENERATED_BIRTH_IDENTITY_LEDGER_MODULE_SHA256
        == hashlib.sha256(Path(ledger_module.__file__).read_bytes()).hexdigest()
    )
    assert binding_module.PINNED_COMPOSITIONAL_FEATURE_STATE_FIELD_MANIFEST == tuple(
        field.name for field in dataclasses.fields(cast(Any, cf.CompositionalFeatureState))
    )
    assert binding_module.PINNED_COMPOSITIONAL_CURATION_TRACE_FIELD_MANIFEST == tuple(
        field.name for field in dataclasses.fields(cast(Any, cf.CompositionalCurationTrace))
    )
    assert binding_module.PINNED_COMPOSITIONAL_UPDATE_RESULT_FIELD_MANIFEST == tuple(
        field.name for field in dataclasses.fields(cast(Any, cf.CompositionalFeatureUpdateResult))
    )


def test_no_event_is_authenticated_and_advances_only_canonical_words(
    no_event_case: _Case,
) -> None:
    binding = _authenticate(no_event_case)

    assert binding.source_replay_authenticated
    assert binding.complete_result_bit_compared
    assert binding.ledger_schema == GENERATED_BIRTH_IDENTITY_LEDGER_V4_SCHEMA
    assert not bool(no_event_case.result.curation_trace.has_event)
    np.testing.assert_array_equal(binding.event.pre_step_words, (0, 0))
    np.testing.assert_array_equal(binding.event.post_step_words, (0, 1))
    assert binding.transaction.audit.applied_identity_event_count == 0
    assert not binding.execution_authorized
    assert not binding.runner_authorized
    assert not binding.artifact_writes_authorized
    assert not binding.evidence_authorized
    assert not binding.scientific_promotion_allowed


def test_ordinary_refresh_is_derived_without_caller_event_masks(
    ordinary_case: _Case,
) -> None:
    binding = _authenticate(ordinary_case)
    event = binding.event.structural_event

    np.testing.assert_array_equal(event.ordinary_candidate_refresh_mask, (True,))
    assert event.ordinary_candidate_refresh_slot == 0
    assert event.promotion_active_slot == -1
    assert binding.transaction.audit.applied_identity_event_count == 1
    assert np.any(binding.transaction.assignments.ordinary_candidate_birth_identity[0])


def test_promotion_cascade_and_fresh_proposal_are_all_bound(
    promotion_case: _Case,
) -> None:
    binding = _authenticate(promotion_case)
    trace = promotion_case.result.curation_trace
    event = binding.event.structural_event

    assert event.promotion_active_slot == 3
    assert event.promotion_candidate_slot == 0
    np.testing.assert_array_equal(
        event.cascade_refill_mask,
        (False, False, False, False, True),
    )
    np.testing.assert_array_equal(
        event.post_promotion_candidate_refresh_mask,
        (True, False),
    )
    assert int(trace.proposal_destination_slot) == 0
    assert binding.transaction.audit.promotion_transfer_count == 1
    assert binding.transaction.audit.applied_identity_event_count >= 2
    assert np.any(binding.transaction.assignments.post_promotion_candidate_birth_identity[0])


def test_candidate_free_direct_replacement_binds_raw_parents(
    direct_case: _Case,
) -> None:
    binding = _authenticate(direct_case)
    event = binding.event.structural_event

    assert event.direct_active_replacement_slot == 2
    assert int(event.active_parent_a[2]) < direct_case.config.raw_feature_slots
    assert int(event.active_parent_b[2]) < direct_case.config.raw_feature_slots
    assert np.any(binding.transaction.assignments.direct_active_birth_identity[2])
    assert binding.transaction.audit.applied_identity_event_count == 1


class _OverlapRegenerationLearner(cf.CompositionalFeatureLearner):
    """Force a transient promotion refresh to require later ODRG repair."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.expected_proposal_key_data = jnp.zeros((2,), dtype=jnp.uint32)
        super().__init__(*args, **kwargs)

    def prepare_update(self, state: cf.CompositionalFeatureState) -> None:
        _post, _decision, curation = jr.split(state.key, 3)
        proposal, _cascade = cf.compositional_curation_keys(curation)
        self.expected_proposal_key_data = jr.key_data(proposal)

    def _generate_one(
        self,
        key: jax.Array,
        existing_depth: jax.Array,
        existing_utilities: jax.Array | None = None,
        existing_ages: jax.Array | None = None,
        feature_values: jax.Array | None = None,
        feature_credit: jax.Array | None = None,
        forced_op: jax.Array | None = None,
        parent_mode: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        del (
            existing_utilities,
            existing_ages,
            feature_values,
            feature_credit,
            forced_op,
            parent_mode,
        )
        primary = jnp.all(jr.key_data(key) == self.expected_proposal_key_data)
        parent_a = jnp.where(primary, 4, 0).astype(jnp.int32)
        parent_b = jnp.asarray(0, dtype=jnp.int32)
        depth = jnp.maximum(existing_depth[parent_a], existing_depth[parent_b]) + 1
        return (
            jnp.asarray(cf.OP_PRODUCT, dtype=jnp.int32),
            parent_a,
            parent_b,
            jnp.asarray((0.5, -0.5), dtype=jnp.float32),
            depth.astype(jnp.int32),
        )

    def _cascade_replace_with_mask(self, *args: Any, **kwargs: Any) -> Any:
        (
            ops,
            parent_a,
            parent_b,
            theta,
            depth,
            utilities,
            ages,
            output_weights,
            cascade_mask,
        ) = super()._cascade_replace_with_mask(*args, **kwargs)
        force_slot = cascade_mask[4]
        forced_depth = (jnp.maximum(depth[3], depth[0]) + 1).astype(jnp.int32)
        ops = ops.at[4].set(jnp.where(force_slot, cf.OP_PRODUCT, ops[4]))
        parent_a = parent_a.at[4].set(jnp.where(force_slot, 3, parent_a[4]))
        parent_b = parent_b.at[4].set(jnp.where(force_slot, 0, parent_b[4]))
        theta = theta.at[4].set(jnp.where(force_slot, jnp.zeros((2,)), theta[4]))
        depth = depth.at[4].set(jnp.where(force_slot, forced_depth, depth[4]))
        return (
            ops,
            parent_a,
            parent_b,
            theta,
            depth,
            utilities,
            ages,
            output_weights,
            cascade_mask,
        )


def test_structural_overlap_binds_transient_refresh_then_odrg_as_unauthenticated() -> None:
    learner = _OverlapRegenerationLearner(
        n_features=5,
        n_tasks=1,
        candidate_count=2,
        step_size_output=0.0,
        step_size_theta=0.0,
        utility_decay=0.999999,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=0.0,
        max_depth=3,
        candidate_imprint_scale=1.0,
        use_obgd=False,
    )
    state = learner.init(feature_dim=2, key=jr.key(4104)).replace(  # type: ignore[attr-defined]
        ops=jnp.asarray(
            (cf.OP_RAW, cf.OP_RAW, cf.OP_PRODUCT, cf.OP_PRODUCT, cf.OP_PRODUCT),
            dtype=jnp.int32,
        ),
        parent_a=jnp.asarray((0, 1, 0, 0, 3), dtype=jnp.int32),
        parent_b=jnp.asarray((-1, -1, 1, 1, 1), dtype=jnp.int32),
        theta=jnp.zeros((5, 2), dtype=jnp.float32),
        depth=jnp.asarray((0, 0, 1, 1, 2), dtype=jnp.int32),
        output_weights=jnp.asarray(((1.0, 2.0, 3.0, 4.0, 5.0),), dtype=jnp.float32),
        utilities=jnp.asarray((10.0, 10.0, 10.0, 0.0, 10.0), dtype=jnp.float32),
        ages=jnp.full((5,), 10, dtype=jnp.int32),
        candidate_ops=jnp.asarray((cf.OP_SUM, cf.OP_PRODUCT), dtype=jnp.int32),
        candidate_parent_a=jnp.asarray((2, 0), dtype=jnp.int32),
        candidate_parent_b=jnp.asarray((0, 1), dtype=jnp.int32),
        candidate_theta=jnp.asarray(((0.1, 0.2), (0.3, 0.4)), dtype=jnp.float32),
        candidate_depth=jnp.asarray((2, 1), dtype=jnp.int32),
        candidate_output_weights=jnp.asarray(((6.0, 7.0),), dtype=jnp.float32),
        candidate_utilities=jnp.asarray((100.0, 1.0), dtype=jnp.float32),
        candidate_ages=jnp.asarray((10, 10), dtype=jnp.int32),
        candidate_generator_policy=jnp.zeros((2,), dtype=jnp.int32),
    )
    learner.prepare_update(state)
    case = _case(
        learner=learner,
        state=state,
        raw_features=2,
        namespace="trace-binding-overlap-structural-development",
        observation=(1.25, 0.75),
        targets=(1.0,),
        seed=4104,
    )
    binding = bind_generated_birth_identity_trace_structurally(
        case.config,
        case.ledger,
        learner_pre_state=case.state,
        learner_post_state=case.result.state,
        update_result=case.result,
    )

    assert not binding.source_replay_authenticated
    np.testing.assert_array_equal(
        binding.event.structural_event.post_promotion_candidate_refresh_mask,
        (True, False),
    )
    np.testing.assert_array_equal(
        binding.event.candidate_overdepth_regeneration_mask,
        (True, False),
    )
    assert np.any(binding.transaction.assignments.post_promotion_candidate_birth_identity[0])
    assert np.any(binding.transaction.assignments.candidate_overdepth_regeneration_identity[0])


def test_genesis_attach_rejects_nonzero_core_lifetime(no_event_case: _Case) -> None:
    with pytest.raises(GeneratedBirthIdentityTraceBindingError, match="only valid at core step 0"):
        attach_generated_birth_identity_ledger_at_core_genesis(
            no_event_case.config,
            learner_pre_state=no_event_case.result.state,
            paired_development_life_seed=999,
        )


def _immutable_words(high: int, low: int) -> np.ndarray:
    value = np.asarray((high, low), dtype=np.uint32)
    return np.frombuffer(value.tobytes(order="C"), dtype=np.uint32)


def _ledger_with_words(
    state: ledger_module.GeneratedBirthIdentityLedgerV4State,
    high: int,
    low: int,
) -> ledger_module.GeneratedBirthIdentityLedgerV4State:
    changed = dataclasses.replace(
        state,
        step_words=_immutable_words(high, low),
        integrity_sha256="0" * 64,
    )
    return dataclasses.replace(
        changed,
        integrity_sha256=generated_birth_identity_ledger_v4_state_sha256(changed),
    )


def test_v4_identity_domain_and_rollover_use_both_exact_words(
    no_event_case: _Case,
) -> None:
    low_identity = derive_generated_birth_identity_v4(
        namespace=no_event_case.config.namespace,
        paired_development_life_seed=4100,
        learner_step_words=np.asarray((0, 0), dtype=np.uint32),
        event_channel=ledger_module.ORDINARY_CANDIDATE_REFRESH_CHANNEL,
        slot=0,
        ordinal=0,
    )
    high_identity = derive_generated_birth_identity_v4(
        namespace=no_event_case.config.namespace,
        paired_development_life_seed=4100,
        learner_step_words=np.asarray((1, 0), dtype=np.uint32),
        event_channel=ledger_module.ORDINARY_CANDIDATE_REFRESH_CHANNEL,
        slot=0,
        ordinal=0,
    )
    assert low_identity != high_identity

    int32_max = 2**31 - 1
    pre = no_event_case.state.replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(int32_max, dtype=jnp.int32),
        step_words=jnp.asarray((0, 2**32 - 1), dtype=jnp.uint32),
        replacement_phase=jnp.asarray(0, dtype=jnp.int32),
    )
    ledger_pre = _ledger_with_words(no_event_case.ledger, 0, 2**32 - 1)
    result = no_event_case.learner.update(
        pre,
        no_event_case.observation,
        no_event_case.targets,
    )
    result.state.step_count.block_until_ready()
    binding = bind_generated_birth_identity_trace_structurally(
        no_event_case.config,
        ledger_pre,
        learner_pre_state=pre,
        learner_post_state=result.state,
        update_result=result,
    )

    np.testing.assert_array_equal(binding.event.pre_step_words, (0, 2**32 - 1))
    np.testing.assert_array_equal(binding.event.post_step_words, (1, 0))
    assert int(result.state.step_count) == int32_max


def test_exhausted_lifetime_words_fail_closed_before_ledger_binding(
    no_event_case: _Case,
) -> None:
    int32_max = 2**31 - 1
    terminal = no_event_case.state.replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(int32_max, dtype=jnp.int32),
        step_words=jnp.asarray((2**32 - 1, 2**32 - 1), dtype=jnp.uint32),
        replacement_phase=jnp.asarray(0, dtype=jnp.int32),
    )
    ledger_terminal = _ledger_with_words(
        no_event_case.ledger,
        2**32 - 1,
        2**32 - 1,
    )
    result = no_event_case.learner.update(
        terminal,
        no_event_case.observation,
        no_event_case.targets,
    )
    result.state.step_count.block_until_ready()
    assert bool(result.curation_trace.lifetime_counter_valid)
    assert not bool(result.curation_trace.lifetime_capacity_available)
    np.testing.assert_array_equal(
        np.asarray(jr.key_data(result.state.key)),
        np.asarray(jr.key_data(terminal.key)),
    )
    np.testing.assert_array_equal(result.state.step_words, terminal.step_words)
    np.testing.assert_array_equal(result.state.ops, terminal.ops)
    np.testing.assert_array_equal(result.state.output_weights, terminal.output_weights)
    assert int(result.state.replacement_phase) == int(terminal.replacement_phase)

    with pytest.raises(GeneratedBirthIdentityTraceBindingError, match="capacity is exhausted"):
        bind_generated_birth_identity_trace_structurally(
            no_event_case.config,
            ledger_terminal,
            learner_pre_state=terminal,
            learner_post_state=result.state,
            update_result=result,
        )


def test_forged_trace_state_result_subclass_and_hash_fail_closed(
    no_event_case: _Case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = no_event_case.result.curation_trace
    forged_trace = dataclasses.replace(
        cast(Any, trace),
        logical_event_count=trace.logical_event_count + jnp.asarray(1, dtype=jnp.int32),
    )
    forged_trace_result = dataclasses.replace(
        cast(Any, no_event_case.result),
        curation_trace=forged_trace,
    )
    with pytest.raises(GeneratedBirthIdentityTraceBindingError, match="logical-event count"):
        bind_generated_birth_identity_trace_structurally(
            no_event_case.config,
            no_event_case.ledger,
            learner_pre_state=no_event_case.state,
            learner_post_state=no_event_case.result.state,
            update_result=forged_trace_result,
        )

    forged_post = no_event_case.result.state.replace(  # type: ignore[attr-defined]
        output_bias=no_event_case.result.state.output_bias + jnp.asarray((1.0,), dtype=jnp.float32)
    )
    with pytest.raises(GeneratedBirthIdentityTraceBindingError, match="bits mismatch"):
        bind_generated_birth_identity_trace_structurally(
            no_event_case.config,
            no_event_case.ledger,
            learner_pre_state=no_event_case.state,
            learner_post_state=forged_post,
            update_result=no_event_case.result,
        )

    forged_result = dataclasses.replace(
        cast(Any, no_event_case.result),
        metrics=no_event_case.result.metrics.at[0].add(jnp.float32(1.0)),
    )
    with pytest.raises(GeneratedBirthIdentityTraceBindingError, match="complete_update_result"):
        authenticate_generated_birth_identity_trace_by_source_replay(
            no_event_case.learner,
            no_event_case.config,
            no_event_case.ledger,
            learner_pre_state=no_event_case.state,
            learner_post_state=no_event_case.result.state,
            supplied_update_result=forged_result,
            observation=no_event_case.observation,
            targets=no_event_case.targets,
        )

    subclass = type("ForgedLearnerSubclass", (cf.CompositionalFeatureLearner,), {})
    forged_learner = subclass(
        n_features=3,
        n_tasks=1,
        candidate_count=1,
        replacement_interval=32,
        min_feature_age=100,
        candidate_min_age=16,
        use_obgd=False,
    )
    with pytest.raises(GeneratedBirthIdentityTraceBindingError, match="exact"):
        authenticate_generated_birth_identity_trace_by_source_replay(
            forged_learner,
            no_event_case.config,
            no_event_case.ledger,
            learner_pre_state=no_event_case.state,
            learner_post_state=no_event_case.result.state,
            supplied_update_result=no_event_case.result,
            observation=no_event_case.observation,
            targets=no_event_case.targets,
        )

    monkeypatch.setattr(
        binding_module,
        "PINNED_COMPOSITIONAL_FEATURES_MODULE_SHA256",
        "0" * 64,
    )
    with pytest.raises(GeneratedBirthIdentityTraceBindingError, match="module bytes"):
        authenticate_generated_birth_identity_trace_by_source_replay(
            no_event_case.learner,
            no_event_case.config,
            no_event_case.ledger,
            learner_pre_state=no_event_case.state,
            learner_post_state=no_event_case.result.state,
            supplied_update_result=no_event_case.result,
            observation=no_event_case.observation,
            targets=no_event_case.targets,
        )
