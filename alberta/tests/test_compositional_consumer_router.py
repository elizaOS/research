# mypy: disable-error-code="attr-defined,call-arg,no-any-return,no-untyped-def"
"""Transactional compositional-bank routing for exact linear consumers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_consumer_router import (
    CompositionalConsumerRouter,
    CompositionalConsumerState,
)
from alberta_framework.core.compositional_feature_adapter import (
    CompositionalFeatureAdapter,
    CompositionalFeatureAdapterPreparedUpdate,
    CompositionalFeatureAdapterState,
)
from alberta_framework.core.compositional_features import (
    OP_PRODUCT,
    OP_RAW,
    CompositionalFeatureLearner,
)
from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.oak import OaKAgent, OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec

pytestmark = pytest.mark.unit


def _adapter(*, structural: bool) -> CompositionalFeatureAdapter:
    return CompositionalFeatureAdapter(
        CompositionalFeatureLearner(
            n_features=4,
            n_tasks=1,
            candidate_count=0,
            step_size_output=0.0,
            step_size_theta=0.0,
            utility_decay=0.999,
            replacement_interval=1 if structural else 0,
            min_feature_age=0,
            use_obgd=False,
            train_candidate_theta=False,
        ),
        base_feature_dim=2,
    )


def _source(adapter: CompositionalFeatureAdapter) -> CompositionalFeatureAdapterState:
    state = adapter.init(jr.key(3101))
    learner_state = state.learner_state.replace(
        ops=jnp.asarray((OP_RAW, OP_RAW, OP_PRODUCT, OP_PRODUCT), dtype=jnp.int32),
        parent_a=jnp.asarray((0, 1, 0, 0), dtype=jnp.int32),
        parent_b=jnp.asarray((-1, -1, 1, 1), dtype=jnp.int32),
        theta=jnp.zeros((4, 2), dtype=jnp.float32),
        depth=jnp.asarray((0, 0, 1, 1), dtype=jnp.int32),
        utilities=jnp.asarray((10.0, 10.0, 0.0, 10.0), dtype=jnp.float32),
        ages=jnp.full((4,), 10, dtype=jnp.int32),
    )
    return adapter.rebind_pristine_state(learner_state)


def _consumer_templates(width: int, *, horde: bool = True):
    oak_config = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
            observation_dim=width,
            n_primitive_actions=2,
            base_hidden_sizes=(),
        )
    )
    oak = OaKAgent(oak_config).init(jr.key(3102))
    horde_state = None
    if horde:
        spec = create_horde_spec(
            (
                GVFSpec(
                    name="first",
                    demon_type=DemonType.PREDICTION,
                    gamma=0.0,
                    lamda=0.0,
                    cumulant_index=0,
                ),
                GVFSpec(
                    name="second",
                    demon_type=DemonType.PREDICTION,
                    gamma=0.9,
                    lamda=0.8,
                    cumulant_index=1,
                ),
            )
        )
        horde_state = HordeLearner(spec, hidden_sizes=(), step_size=0.1).init(
            width, jr.key(3103)
        )
        # Make the declared demon order observably distinct at pristine bind.
        horde_state = horde_state.replace(
            head_params=horde_state.head_params.replace(
                weights=tuple(
                    weight + jnp.float32(index + 1)
                    for index, weight in enumerate(horde_state.head_params.weights)
                )
            )
        )
    return oak_config, oak, horde_state


def _filled(state: CompositionalConsumerState) -> CompositionalConsumerState:
    oak = state.oak_state
    stomp = oak.stomp_state
    base = stomp.base_learner_state
    base_weights = tuple(
        jnp.arange(weight.size, dtype=jnp.float32).reshape(weight.shape)
        + jnp.float32(10 * (index + 1))
        for index, weight in enumerate(base.head_params.weights)
    )
    base_traces = tuple(
        (
            jnp.arange(pair[0].size, dtype=jnp.float32).reshape(pair[0].shape)
            + jnp.float32(100 * (index + 1)),
            pair[1],
        )
        for index, pair in enumerate(base.head_traces)
    )
    base = base.replace(
        head_params=base.head_params.replace(weights=base_weights),
        head_traces=base_traces,
    )
    width = int(stomp.base_last_obs.shape[0])
    option_shape = stomp.option_policies.q_weights.shape
    policy_values = jnp.arange(np.prod(option_shape), dtype=jnp.float32).reshape(
        option_shape
    ) + 200.0
    model_shape = stomp.option_models.next_state_weights.shape
    model_values = jnp.arange(np.prod(model_shape), dtype=jnp.float32).reshape(
        model_shape
    ) + 300.0
    next_stomp = stomp.replace(
        base_learner_state=base,
        base_last_obs=jnp.arange(width, dtype=jnp.float32) + 400.0,
        option_start_obs=jnp.arange(width, dtype=jnp.float32) + 500.0,
        option_policies=stomp.option_policies.replace(
            q_weights=policy_values,
            traces=policy_values + 50.0,
        ),
        option_models=stomp.option_models.replace(
            next_state_weights=model_values,
        ),
    )
    next_oak = oak.replace(stomp_state=next_stomp)
    next_horde = state.horde_state
    if next_horde is not None:
        weights = tuple(
            jnp.arange(weight.size, dtype=jnp.float32).reshape(weight.shape)
            + jnp.float32(600 + 100 * index)
            for index, weight in enumerate(next_horde.head_params.weights)
        )
        traces = tuple(
            (
                jnp.arange(pair[0].size, dtype=jnp.float32).reshape(pair[0].shape)
                + jnp.float32(800 + 100 * index),
                pair[1],
            )
            for index, pair in enumerate(next_horde.head_traces)
        )
        next_horde = next_horde.replace(
            head_params=next_horde.head_params.replace(weights=weights),
            head_traces=traces,
        )
    return state.replace(oak_state=next_oak, horde_state=next_horde)


def _tree_equal(left: object, right: object) -> bool:
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        left_dtype = getattr(left_leaf, "dtype", None)
        if left_dtype is not None and jax.dtypes.issubdtype(
            left_dtype, jax.dtypes.prng_key
        ):
            left_leaf = jr.key_data(left_leaf)
            right_leaf = jr.key_data(right_leaf)
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return False
        if left_array.dtype == jnp.dtype(jnp.float32):
            left_array = jax.lax.bitcast_convert_type(left_array, jnp.uint32)
            right_array = jax.lax.bitcast_convert_type(right_array, jnp.uint32)
        if not bool(jnp.all(left_array == right_array)):
            return False
    return True


def _post_update_fixture(
    router: CompositionalConsumerRouter,
    state: CompositionalConsumerState,
    proposal: CompositionalFeatureAdapterPreparedUpdate,
) -> CompositionalConsumerState:
    """Attest consumers that have taken one old-bank transition."""

    words = proposal.candidate_state.learner_state.step_words
    telemetry = proposal.candidate_state.learner_state.step_count
    source_cache = router.adapter.representation(
        proposal.source_state,
        proposal.observation,
    )
    oak = state.oak_state
    stomp = oak.stomp_state
    base = stomp.base_learner_state.replace(
        step_count=telemetry,
        step_words=words,
    )
    stomp = stomp.replace(
        base_learner_state=base,
        base_last_obs=source_cache,
        step_count=telemetry,
        step_words=words,
    )
    oak = oak.replace(
        stomp_state=stomp,
        step_count=telemetry,
        step_words=words,
    )
    horde = state.horde_state
    if horde is not None:
        horde = horde.replace(step_count=telemetry, step_words=words)
    return state.replace(oak_state=oak, horde_state=horde)


def _prepared_transaction(*, structural: bool = True):
    adapter = _adapter(structural=structural)
    source = _source(adapter)
    oak_config, oak, horde = _consumer_templates(adapter.n_features)
    router = CompositionalConsumerRouter(adapter, oak_config, oak, horde)
    stable_consumers = _filled(router.bind_pristine(source, oak, horde))
    proposal = adapter.prepare_update(
        source,
        jnp.asarray((0.5, 0.25), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    )
    post_update_consumers = _post_update_fixture(
        router,
        stable_consumers,
        proposal,
    )
    prepared = router.prepare_route(
        stable_consumers,
        post_update_consumers,
        proposal,
    )
    return (
        adapter,
        router,
        source,
        stable_consumers,
        post_update_consumers,
        proposal,
        prepared,
    )


def test_structural_route_uses_births_preserves_survivors_and_scrubs_all_changed_columns() -> None:
    (
        adapter,
        router,
        source,
        _,
        consumers,
        proposal,
        prepared,
    ) = _prepared_transaction()
    assert bool(proposal.diagnostics.active_bank_changed)
    assert bool(prepared.receipt.consumers_ready)
    changed = np.asarray(prepared.receipt.changed_birth_mask)
    assert not np.any(changed[: adapter.base_feature_dim])
    assert np.any(changed[adapter.base_feature_dim :])
    assert bool(prepared.diagnostics.full_binding_transition_valid)
    assert bool(prepared.diagnostics.post_update_clock_parity_valid)
    assert bool(prepared.diagnostics.safe_route_boundary)
    assert bool(prepared.diagnostics.source_cache_matches)
    assert bool(prepared.diagnostics.candidate_cache_recomputed)
    assert bool(prepared.diagnostics.survivor_columns_bit_exact)
    assert bool(prepared.diagnostics.changed_columns_scrubbed)
    assert bool(prepared.diagnostics.optimizer_state_authenticated)

    candidate = prepared.candidate_state
    np.testing.assert_array_equal(
        candidate.oak_state.stomp_state.base_last_obs,
        router.adapter.representation(
            proposal.candidate_state,
            proposal.observation,
        ),
    )
    old_base = consumers.oak_state.stomp_state.base_learner_state
    new_base = candidate.oak_state.stomp_state.base_learner_state
    for old, new in zip(
        old_base.head_params.weights, new_base.head_params.weights, strict=True
    ):
        np.testing.assert_array_equal(new[..., ~changed], old[..., ~changed])
        np.testing.assert_array_equal(new[..., changed], 0.0)
    for old, new in zip(old_base.head_traces, new_base.head_traces, strict=True):
        np.testing.assert_array_equal(new[0][..., ~changed], old[0][..., ~changed])
        np.testing.assert_array_equal(new[0][..., changed], 0.0)

    old_model = consumers.oak_state.stomp_state.option_models.next_state_weights
    new_model = candidate.oak_state.stomp_state.option_models.next_state_weights
    survivor_cells = (~changed)[:, None] & (~changed)[None, :]
    np.testing.assert_array_equal(
        np.asarray(new_model)[:, survivor_cells],
        np.asarray(old_model)[:, survivor_cells],
    )
    assert not np.any(np.asarray(new_model)[:, ~survivor_cells])

    assert consumers.horde_state is not None
    assert candidate.horde_state is not None
    for old, new in zip(
        consumers.horde_state.head_params.weights,
        candidate.horde_state.head_params.weights,
        strict=True,
    ):
        np.testing.assert_array_equal(new[..., ~changed], old[..., ~changed])
        np.testing.assert_array_equal(new[..., changed], 0.0)

    # Slots 2 and 3 began as descriptor-identical products. Only a changed
    # birth scrubs a column; descriptor equality never moves or merges it.
    assert source.binding.parent_a[2] == source.binding.parent_a[3]
    assert source.binding.parent_b[2] == source.binding.parent_b[3]
    for slot in np.flatnonzero(~changed):
        for old, new in zip(
            old_base.head_params.weights, new_base.head_params.weights, strict=True
        ):
            np.testing.assert_array_equal(new[..., slot], old[..., slot])


def test_combined_commit_adopts_adapter_and_consumers_atomically() -> None:
    _, router, source, stable, _, proposal, prepared = _prepared_transaction()

    committed = router.commit_prepared_route(source, stable, prepared)

    assert bool(committed.diagnostics.route_integrity)
    assert bool(committed.diagnostics.consumer_source_matches)
    assert bool(committed.diagnostics.consumers_ready)
    assert bool(committed.diagnostics.adapter_applied)
    assert bool(committed.diagnostics.applied)
    assert _tree_equal(committed.adapter_state, proposal.candidate_state)
    assert _tree_equal(committed.consumer_state, prepared.candidate_state)
    assert _tree_equal(
        committed.consumer_state.binding,
        committed.adapter_state.binding,
    )


def test_stale_tampered_and_misbound_transactions_are_atomic_noops() -> None:
    _, router, source, stable, consumers, _, prepared = _prepared_transaction()
    first = router.commit_prepared_route(source, stable, prepared)

    stale = router.commit_prepared_route(
        first.adapter_state,
        first.consumer_state,
        prepared,
    )
    assert not bool(stale.diagnostics.consumer_source_matches)
    assert not bool(stale.diagnostics.applied)
    assert _tree_equal(stale.adapter_state, first.adapter_state)
    assert _tree_equal(stale.consumer_state, first.consumer_state)

    base = prepared.candidate_state.oak_state.stomp_state.base_learner_state
    tampered_base = base.replace(
        head_params=base.head_params.replace(
            weights=(
                base.head_params.weights[0].at[0, 0].add(jnp.float32(1.0)),
                *base.head_params.weights[1:],
            )
        )
    )
    tampered = prepared.replace(
        candidate_state=prepared.candidate_state.replace(
            oak_state=prepared.candidate_state.oak_state.replace(
                stomp_state=prepared.candidate_state.oak_state.stomp_state.replace(
                    base_learner_state=tampered_base
                )
            )
        )
    )
    refused = router.commit_prepared_route(source, stable, tampered)
    assert not bool(refused.diagnostics.route_integrity)
    assert not bool(refused.diagnostics.applied)
    assert _tree_equal(refused.adapter_state, source)
    assert _tree_equal(refused.consumer_state, stable)
    assert router.state_valid(refused.consumer_state, refused.adapter_state)

    misbound = consumers.replace(
        binding=consumers.binding.replace(
            parent_a=consumers.binding.parent_a.at[2].set(jnp.int32(1))
        )
    )
    rejected = router.prepare_route(stable, misbound, prepared.adapter_proposal)
    assert not bool(rejected.diagnostics.post_update_binding_matches)
    assert not bool(rejected.receipt.consumers_ready)
    assert _tree_equal(rejected.candidate_state, stable)


def test_nonfinite_consumer_or_adapter_proposal_fails_closed() -> None:
    _, router, source, stable, consumers, _, prepared = _prepared_transaction()
    base = consumers.oak_state.stomp_state.base_learner_state
    bad_base = base.replace(
        head_params=base.head_params.replace(
            weights=(
                base.head_params.weights[0].at[0, 3].set(jnp.nan),
                *base.head_params.weights[1:],
            )
        )
    )
    nonfinite = consumers.replace(
        oak_state=consumers.oak_state.replace(
            stomp_state=consumers.oak_state.stomp_state.replace(
                base_learner_state=bad_base
            )
        )
    )
    rejected = router.prepare_route(stable, nonfinite, prepared.adapter_proposal)
    assert not bool(rejected.diagnostics.post_update_consumers_valid)
    assert not bool(rejected.receipt.consumers_ready)
    assert _tree_equal(rejected.candidate_state, stable)

    tampered_adapter = prepared.replace(
        adapter_proposal=prepared.adapter_proposal.replace(
            predictions=prepared.adapter_proposal.predictions.at[0].set(jnp.nan)
        )
    )
    commit = router.commit_prepared_route(source, stable, tampered_adapter)
    assert not bool(commit.diagnostics.route_integrity)
    assert not bool(commit.diagnostics.applied)
    assert _tree_equal(commit.adapter_state, source)
    assert _tree_equal(commit.consumer_state, stable)
    assert router.state_valid(commit.consumer_state, commit.adapter_state)


def test_no_birth_change_is_ready_and_bit_exact_for_every_consumer() -> None:
    _, router, source, stable, consumers, proposal, prepared = _prepared_transaction(
        structural=False
    )
    assert not bool(proposal.diagnostics.active_bank_changed)
    assert not bool(jnp.any(prepared.receipt.changed_birth_mask))
    assert bool(prepared.receipt.consumers_ready)
    assert _tree_equal(prepared.candidate_state.oak_state, consumers.oak_state)
    assert _tree_equal(prepared.candidate_state.horde_state, consumers.horde_state)

    committed = router.commit_prepared_route(source, stable, prepared)
    assert bool(committed.diagnostics.applied)
    assert _tree_equal(committed.consumer_state.oak_state, consumers.oak_state)
    assert _tree_equal(committed.consumer_state.horde_state, consumers.horde_state)


def test_exact_persistent_transient_bytes_and_fixed_work_close() -> None:
    _, router, _, stable, consumers, _, prepared = _prepared_transaction()
    budget = router.resource_budget(stable)

    assert budget.total_persistent_state_nbytes == router.measure_state_nbytes(
        stable
    )
    assert budget.binding_persistent_nbytes == 32 * 4 + 44
    assert budget.horde_enabled is True
    assert budget.managed_feature_scalars > 0
    assert budget.fixed_consumer_route_feature_scalar_evaluations == (
        3 * budget.managed_feature_scalars
    )
    assert budget.fixed_cache_representation_calls_per_route == 2
    assert budget.fixed_cache_representation_feature_slot_evaluations_per_route == 8
    assert budget.route_recomputations_per_commit == 1
    assert budget.fixed_prepare_commit_cache_representation_calls == 4
    assert (
        budget.fixed_prepare_commit_cache_representation_feature_slot_evaluations
        == 16
    )
    assert (
        budget.fixed_prepare_commit_consumer_route_feature_scalar_evaluations
        == 6 * budget.managed_feature_scalars
    )
    assert int(prepared.receipt.managed_feature_scalars) == (
        budget.managed_feature_scalars
    )
    assert int(prepared.receipt.consumer_route_feature_scalar_evaluations) == (
        3 * budget.managed_feature_scalars
    )
    assert int(prepared.receipt.cache_representation_calls) == 2
    assert int(prepared.receipt.cache_representation_feature_slot_evaluations) == 8
    eager_nbytes = router.measure_prepared_route_nbytes(prepared)
    compiled = jax.jit(router.prepare_route)(
        stable,
        consumers,
        prepared.adapter_proposal,
    )
    assert router.measure_prepared_route_nbytes(compiled) == eager_nbytes
    assert budget.internal_pristine_template_nbytes > 0


def test_eager_jit_and_scan_transactions_are_bit_exact() -> None:
    _, router, source, stable, _, _, prepared = _prepared_transaction(
        structural=False
    )
    eager = router.commit_prepared_route(source, stable, prepared)
    compiled = jax.jit(router.commit_prepared_route)(source, stable, prepared)
    assert _tree_equal(eager, compiled)

    observation = jnp.asarray((0.5, 0.25), dtype=jnp.float32)
    targets = jnp.asarray((0.0,), dtype=jnp.float32)

    def step(carry, _):
        adapter_state, consumer_state = carry
        adapter_proposal = router.adapter.prepare_update(
            adapter_state, observation, targets
        )
        post_update_consumers = _post_update_fixture(
            router,
            consumer_state,
            adapter_proposal,
        )
        route = router.prepare_route(
            consumer_state,
            post_update_consumers,
            adapter_proposal,
        )
        result = router.commit_prepared_route(
            adapter_state, consumer_state, route
        )
        return (result.adapter_state, result.consumer_state), result.diagnostics.applied

    (final_adapter, final_consumers), applied = jax.jit(
        lambda carry: jax.lax.scan(step, carry, jnp.arange(3))
    )((source, stable))
    np.testing.assert_array_equal(applied, (True, True, True))
    np.testing.assert_array_equal(final_adapter.learner_state.step_words, (0, 3))
    assert router.state_valid(final_consumers, final_adapter)


def test_route_defers_on_unsafe_boundary_cache_mismatch_or_clock_drift() -> None:
    _, router, source, stable, consumers, _, prepared = _prepared_transaction()

    unsafe_stomp = consumers.oak_state.stomp_state.replace(executing_option=jnp.int32(0))
    unsafe = consumers.replace(
        oak_state=consumers.oak_state.replace(stomp_state=unsafe_stomp)
    )
    deferred = router.prepare_route(stable, unsafe, prepared.adapter_proposal)
    assert not bool(deferred.diagnostics.safe_route_boundary)
    assert not bool(deferred.receipt.consumers_ready)
    assert _tree_equal(deferred.candidate_state, stable)

    mismatched_stomp = consumers.oak_state.stomp_state.replace(
        base_last_obs=consumers.oak_state.stomp_state.base_last_obs.at[2].add(1.0)
    )
    mismatched = consumers.replace(
        oak_state=consumers.oak_state.replace(stomp_state=mismatched_stomp)
    )
    rejected_cache = router.prepare_route(
        stable,
        mismatched,
        prepared.adapter_proposal,
    )
    assert not bool(rejected_cache.diagnostics.source_cache_matches)
    assert not bool(rejected_cache.receipt.consumers_ready)

    source_words = source.learner_state.step_words
    source_step = source.learner_state.step_count
    lag_stomp = consumers.oak_state.stomp_state
    lag_base = lag_stomp.base_learner_state.replace(
        step_count=source_step,
        step_words=source_words,
    )
    lag_stomp = lag_stomp.replace(
        base_learner_state=lag_base,
        step_count=source_step,
        step_words=source_words,
    )
    lag_oak = consumers.oak_state.replace(
        stomp_state=lag_stomp,
        step_count=source_step,
        step_words=source_words,
    )
    assert consumers.horde_state is not None
    lag_horde = consumers.horde_state.replace(
        step_count=source_step,
        step_words=source_words,
    )
    lagged = consumers.replace(oak_state=lag_oak, horde_state=lag_horde)
    rejected_clock = router.prepare_route(
        stable,
        lagged,
        prepared.adapter_proposal,
    )
    assert not bool(rejected_clock.diagnostics.post_update_clock_parity_valid)
    assert not bool(rejected_clock.receipt.consumers_ready)


def test_no_birth_change_commits_while_an_option_is_executing() -> None:
    _, router, source, stable, consumers, _, prepared = _prepared_transaction(
        structural=False
    )
    stomp = consumers.oak_state.stomp_state.replace(
        executing_option=jnp.int32(0),
        base_last_action=jnp.int32(router.oak_config.n_primitive_actions),
    )
    executing = consumers.replace(
        oak_state=consumers.oak_state.replace(stomp_state=stomp)
    )
    route = router.prepare_route(stable, executing, prepared.adapter_proposal)

    assert not bool(jnp.any(route.receipt.changed_birth_mask))
    assert bool(route.diagnostics.safe_route_boundary)
    assert bool(route.receipt.consumers_ready)
    committed = router.commit_prepared_route(source, stable, route)
    assert bool(committed.diagnostics.applied)
    assert router.state_valid(committed.consumer_state, committed.adapter_state)


def test_denied_unsafe_curation_commits_learning_then_cures_at_safe_cadence() -> None:
    adapter = _adapter(structural=True)
    source = _source(adapter)
    oak_config, oak, horde = _consumer_templates(adapter.n_features)
    router = CompositionalConsumerRouter(adapter, oak_config, oak, horde)
    stable = _filled(router.bind_pristine(source, oak, horde))
    observation = jnp.asarray((0.5, 0.25), dtype=jnp.float32)
    targets = jnp.asarray((0.0,), dtype=jnp.float32)

    denied_proposal = adapter.prepare_update(
        source,
        observation,
        targets,
        curation_allowed=jnp.asarray(False, dtype=jnp.bool_),
    )
    assert not bool(denied_proposal.diagnostics.active_bank_changed)
    unsafe = _post_update_fixture(router, stable, denied_proposal)
    unsafe_stomp = unsafe.oak_state.stomp_state.replace(
        executing_option=jnp.int32(0),
        base_last_action=jnp.int32(oak_config.n_primitive_actions),
    )
    unsafe = unsafe.replace(
        oak_state=unsafe.oak_state.replace(stomp_state=unsafe_stomp)
    )
    denied_route = router.prepare_route(stable, unsafe, denied_proposal)
    first = router.commit_prepared_route(source, stable, denied_route)

    assert bool(denied_route.receipt.consumers_ready)
    assert bool(first.diagnostics.applied)
    np.testing.assert_array_equal(first.adapter_state.learner_state.step_words, (0, 1))
    assert router.state_valid(first.consumer_state, first.adapter_state)

    admitted_proposal = adapter.prepare_update(
        first.adapter_state,
        observation,
        targets,
        curation_allowed=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert bool(admitted_proposal.diagnostics.active_bank_changed)
    safe = _post_update_fixture(router, first.consumer_state, admitted_proposal)
    safe_stomp = safe.oak_state.stomp_state.replace(
        executing_option=jnp.int32(-1),
        base_last_action=jnp.int32(0),
    )
    safe = safe.replace(oak_state=safe.oak_state.replace(stomp_state=safe_stomp))
    admitted_route = router.prepare_route(
        first.consumer_state,
        safe,
        admitted_proposal,
    )
    second = router.commit_prepared_route(
        first.adapter_state,
        first.consumer_state,
        admitted_route,
    )

    assert bool(admitted_route.diagnostics.safe_route_boundary)
    assert bool(admitted_route.receipt.consumers_ready)
    assert bool(second.diagnostics.applied)
    np.testing.assert_array_equal(second.adapter_state.learner_state.step_words, (0, 2))
    assert router.state_valid(second.consumer_state, second.adapter_state)


def test_oak_only_configuration_keeps_the_optional_horde_boundary_absent() -> None:
    adapter = _adapter(structural=False)
    source = _source(adapter)
    oak_config, oak, horde = _consumer_templates(adapter.n_features, horde=False)
    assert horde is None
    router = CompositionalConsumerRouter(adapter, oak_config, oak, None)
    stable = _filled(router.bind_pristine(source, oak, None))
    proposal = adapter.prepare_update(
        source,
        jnp.asarray((0.5, 0.25), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    )
    post_update = _post_update_fixture(router, stable, proposal)
    route = router.prepare_route(stable, post_update, proposal)
    committed = router.commit_prepared_route(source, stable, route)

    assert bool(committed.diagnostics.applied)
    assert committed.consumer_state.horde_state is None
    assert router.state_valid(committed.consumer_state, committed.adapter_state)
    assert router.resource_budget(stable).horde_enabled is False


def test_oak_subtasks_must_bind_only_the_stable_base_prefix() -> None:
    adapter = _adapter(structural=False)
    tail_config = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=adapter.base_feature_dim),),
            observation_dim=adapter.n_features,
            n_primitive_actions=2,
            base_hidden_sizes=(),
        )
    )
    tail_oak = OaKAgent(tail_config).init(jr.key(3189))
    with pytest.raises(ValueError, match="stable base prefix"):
        CompositionalConsumerRouter(adapter, tail_config, tail_oak, None)


def test_static_contract_rejects_nonlinear_oak_and_horde_presence_mismatch() -> None:
    adapter = _adapter(structural=False)
    oak = OaKAgent(
        OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(SubtaskSpec(feature_index=0),),
                observation_dim=adapter.n_features,
                n_primitive_actions=2,
                base_hidden_sizes=(3,),
            )
        )
    ).init(jr.key(3190))
    with pytest.raises(ValueError, match="linear OaK"):
        CompositionalConsumerRouter(
            adapter,
            OaKConfig(
                stomp=STOMPConfig(
                    subtask_specs=(SubtaskSpec(feature_index=0),),
                    observation_dim=adapter.n_features,
                    n_primitive_actions=2,
                    base_hidden_sizes=(3,),
                )
            ),
            oak,
            None,
        )

    oak_config, linear_oak, horde = _consumer_templates(adapter.n_features)
    assert horde is not None
    router = CompositionalConsumerRouter(adapter, oak_config, linear_oak, horde)
    source = _source(adapter)
    with pytest.raises(ValueError, match="Horde"):
        router.bind_pristine(source, linear_oak, None)

    # A same-shaped but reordered demon collection is not accepted as the
    # configured ordered Horde template.
    reordered = horde.replace(
        head_params=horde.head_params.replace(
            weights=tuple(reversed(horde.head_params.weights)),
            biases=tuple(reversed(horde.head_params.biases)),
        ),
        head_optimizer_states=tuple(reversed(horde.head_optimizer_states)),
        head_traces=tuple(reversed(horde.head_traces)),
    )
    with pytest.raises(ValueError, match="pristine Horde"):
        router.bind_pristine(source, linear_oak, reordered)
