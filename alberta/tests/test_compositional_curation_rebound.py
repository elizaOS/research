"""Applied-cascade and candidate-rebound contracts for compositional curation."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core import compositional_features as cf


class _GenerationProbeLearner(cf.CompositionalFeatureLearner):
    generation_events: list[int] = []

    def _generate_one(self, key: jax.Array, *args: Any, **kwargs: Any) -> Any:
        jax.debug.callback(
            lambda word: self.generation_events.append(int(word)),
            jr.key_data(key)[0],
            ordered=True,
        )
        return super()._generate_one(key, *args, **kwargs)


class _ControlledRefreshLearner(cf.CompositionalFeatureLearner):
    def __init__(
        self,
        *args: Any,
        refresh_parent: int,
        refresh_op: int = cf.OP_PRODUCT,
        **kwargs: Any,
    ) -> None:
        self.refresh_parent = refresh_parent
        self.refresh_op = refresh_op
        super().__init__(*args, **kwargs)

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
            key,
            existing_utilities,
            existing_ages,
            feature_values,
            feature_credit,
            forced_op,
            parent_mode,
        )
        parent_a = jnp.asarray(self.refresh_parent, dtype=jnp.int32)
        parent_b = jnp.asarray(0, dtype=jnp.int32)
        depth = jnp.maximum(existing_depth[parent_a], existing_depth[parent_b]) + 1
        return (
            jnp.asarray(self.refresh_op, dtype=jnp.int32),
            parent_a,
            parent_b,
            jnp.asarray((0.5, -0.5), dtype=jnp.float32),
            depth.astype(jnp.int32),
        )

    def _initial_candidate_output_weights(
        self,
        op: jax.Array,
        parent_a: jax.Array,
        parent_b: jax.Array,
        theta: jax.Array,
        active_values: jax.Array,
        observation: jax.Array,
        errors: jax.Array,
        active_count: jax.Array,
        imprint_scale: jax.Array | None = None,
    ) -> jax.Array:
        del (
            op,
            parent_a,
            parent_b,
            theta,
            active_values,
            observation,
            active_count,
            imprint_scale,
        )
        return jnp.full_like(errors, 23.0)


class _RegenerationKeyProbeLearner(_ControlledRefreshLearner):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.generated_key_words: list[tuple[int, ...]] = []
        super().__init__(*args, **kwargs)

    def _generate_one(self, key: jax.Array, *args: Any, **kwargs: Any) -> Any:
        jax.debug.callback(
            lambda words: self.generated_key_words.append(
                tuple(int(word) for word in words)
            ),
            jr.key_data(key),
            ordered=True,
        )
        return super()._generate_one(key, *args, **kwargs)


class _OverlapRegenerationLearner(cf.CompositionalFeatureLearner):
    """Force a promoted proposal to overlap a later depth-increasing cascade."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.expected_proposal_key_data = jnp.zeros((2,), dtype=jnp.uint32)
        self.generation_events: list[tuple[Any, Any, Any]] = []
        super().__init__(*args, **kwargs)

    def prepare_update(self, state: cf.CompositionalFeatureState) -> None:
        _post_key, _decision_key, curation_key = jr.split(state.key, 3)
        proposal_key, _cascade_key = cf.compositional_curation_keys(curation_key)
        self.expected_proposal_key_data = jr.key_data(proposal_key)
        self.generation_events.clear()

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
        del existing_utilities, existing_ages, forced_op, parent_mode
        is_primary_proposal = jnp.all(
            jr.key_data(key) == self.expected_proposal_key_data
        )
        parent_a = jnp.where(is_primary_proposal, 4, 0).astype(jnp.int32)
        parent_b = jnp.asarray(0, dtype=jnp.int32)
        depth = jnp.maximum(existing_depth[parent_a], existing_depth[parent_b]) + 1
        if feature_values is not None and feature_credit is not None:
            jax.debug.callback(
                lambda words, values, credit: self.generation_events.append(
                    (
                        tuple(int(word) for word in words),
                        np.array(values, dtype=np.float32, copy=True),
                        np.array(credit, dtype=np.float32, copy=True),
                    )
                ),
                jr.key_data(key),
                feature_values,
                feature_credit,
                ordered=True,
            )
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


def test_nonpromotion_without_refresh_executes_no_candidate_generation() -> None:
    learner = _GenerationProbeLearner(
        n_features=3,
        n_tasks=1,
        candidate_count=1,
        replacement_interval=32,
        min_feature_age=100,
        candidate_min_age=16,
        use_obgd=False,
    )
    learner.generation_events.clear()
    state = learner.init(feature_dim=2, key=jr.key(901))

    result = learner.update(
        state,
        jnp.asarray((0.25, -0.5), dtype=jnp.float32),
        jnp.asarray((0.75,), dtype=jnp.float32),
    )
    result.state.step_count.block_until_ready()

    assert int(result.replaced_slot) == -1
    assert int(result.promoted_candidate) == -1
    assert learner.generation_events == []


def _cascade_state(
    learner: cf.CompositionalFeatureLearner,
) -> cf.CompositionalFeatureState:
    state = learner.init(feature_dim=3, key=jr.key(902))
    return state.replace(  # type: ignore[attr-defined]
        ops=jnp.asarray(
            (cf.OP_RAW, cf.OP_RAW, cf.OP_RAW, cf.OP_PRODUCT, cf.OP_PRODUCT),
            dtype=jnp.int32,
        ),
        parent_a=jnp.asarray((0, 1, 2, 0, 3), dtype=jnp.int32),
        parent_b=jnp.asarray((-1, -1, -1, 1, 2), dtype=jnp.int32),
        depth=jnp.asarray((0, 0, 0, 1, 2), dtype=jnp.int32),
        utilities=jnp.asarray((10.0, 10.0, 10.0, 0.0, 5.0), dtype=jnp.float32),
        ages=jnp.full((5,), 10, dtype=jnp.int32),
    )


def test_cascade_reports_exact_mask_and_uses_slot_fold_in_keys() -> None:
    learner = cf.CompositionalFeatureLearner(
        n_features=5,
        n_tasks=1,
        candidate_count=0,
        max_depth=4,
        generation_strategy=cf.GENERATION_UNIFORM,
        operation_prior=(0.0, 1.0, 0.0, 0.0, 0.0),
    )
    state = _cascade_state(learner)
    root_key = jr.key(903)
    initial_mask = jnp.asarray((False, False, False, True, False), dtype=jnp.bool_)

    values = learner._cascade_replace_with_mask(  # noqa: SLF001
        state.ops,
        state.parent_a,
        state.parent_b,
        state.theta,
        state.depth,
        state.utilities,
        state.ages,
        state.output_weights,
        initial_mask,
        jnp.asarray((0.2, -0.4, 0.6), dtype=jnp.float32),
        root_key,
    )
    *replacement_values, cascaded_mask = values
    theta = replacement_values[3]

    np.testing.assert_array_equal(
        np.asarray(cascaded_mask),
        np.asarray((False, False, False, True, True)),
    )
    for slot in (3, 4):
        slot_key = jr.fold_in(root_key, jnp.asarray(slot, dtype=jnp.uint32))
        theta_key = jr.split(slot_key, 4)[3]
        expected_theta = 0.5 * jr.normal(theta_key, (2,), dtype=jnp.float32)
        np.testing.assert_array_equal(
            np.asarray(theta[slot]),
            np.asarray(expected_theta),
        )

    compatibility_result = learner._cascade_replace(  # noqa: SLF001
        state.ops,
        state.parent_a,
        state.parent_b,
        state.theta,
        state.depth,
        state.utilities,
        state.ages,
        state.output_weights,
        initial_mask,
        jnp.asarray((0.2, -0.4, 0.6), dtype=jnp.float32),
        root_key,
    )
    assert len(compatibility_result) == 8


def _promotion_and_rebound_state(
    learner: cf.CompositionalFeatureLearner,
    *,
    learned_generator_policy: bool,
) -> cf.CompositionalFeatureState:
    state = learner.init(feature_dim=3, key=jr.key(904))
    manager_state = state.generator_resource_state
    if learned_generator_policy:
        manager_state = manager_state.replace(  # type: ignore[attr-defined]
            log_weights=jnp.asarray(
                ((-1.0e9, -1.0e9, 0.0, -1.0e9),),
                dtype=jnp.float32,
            )
        )
    return state.replace(  # type: ignore[attr-defined]
        ops=jnp.asarray(
            (cf.OP_RAW, cf.OP_RAW, cf.OP_RAW, cf.OP_PRODUCT, cf.OP_PRODUCT),
            dtype=jnp.int32,
        ),
        parent_a=jnp.asarray((0, 1, 2, 0, 3), dtype=jnp.int32),
        parent_b=jnp.asarray((-1, -1, -1, 1, 2), dtype=jnp.int32),
        depth=jnp.asarray((0, 0, 0, 1, 2), dtype=jnp.int32),
        utilities=jnp.asarray((10.0, 10.0, 10.0, 0.0, 10.0), dtype=jnp.float32),
        ages=jnp.full((5,), 10, dtype=jnp.int32),
        feature_generator_policy=jnp.asarray((0, 0, 0, 1, 2), dtype=jnp.int32),
        candidate_ops=jnp.asarray((cf.OP_SUM, cf.OP_PRODUCT), dtype=jnp.int32),
        candidate_parent_a=jnp.asarray((0, 4), dtype=jnp.int32),
        candidate_parent_b=jnp.asarray((1, 2), dtype=jnp.int32),
        candidate_theta=jnp.asarray(((0.1, 0.2), (0.3, 0.4)), dtype=jnp.float32),
        candidate_depth=jnp.asarray((1, 3), dtype=jnp.int32),
        candidate_output_weights=jnp.asarray(((6.0, 7.0),), dtype=jnp.float32),
        candidate_utilities=jnp.asarray((100.0, 1.0), dtype=jnp.float32),
        candidate_utility_contribution_trace=jnp.asarray(
            ((8.0, 1.0),), dtype=jnp.float32
        ),
        candidate_utility_feature_trace=jnp.asarray((9.0, 1.0), dtype=jnp.float32),
        candidate_utility_feature_energy_trace=jnp.asarray(
            (10.0, 1.0), dtype=jnp.float32
        ),
        candidate_utility_signal_second_moment=jnp.asarray(
            (11.0, 1.0), dtype=jnp.float32
        ),
        candidate_score_residual_trace=jnp.asarray(
            ((12.0, 1.0),), dtype=jnp.float32
        ),
        candidate_score_energy_trace=jnp.asarray((13.0, 1.0), dtype=jnp.float32),
        candidate_retention_slow_utilities=jnp.asarray(
            (14.0, 1.0), dtype=jnp.float32
        ),
        candidate_active_correlation_trace=jnp.ones((2, 5), dtype=jnp.float32),
        candidate_ages=jnp.asarray((10, 10), dtype=jnp.int32),
        candidate_selector_log_weights=jnp.ones((2,), dtype=jnp.float32),
        candidate_selector_cumulative_loss=jnp.ones((2,), dtype=jnp.float32),
        candidate_selector_action_counts=jnp.ones((2,), dtype=jnp.float32),
        candidate_generator_policy=jnp.asarray((3, 2), dtype=jnp.int32),
        generator_resource_state=manager_state,
    )


@pytest.mark.parametrize(
    ("learned_generator_policy", "expected_cascade_policy"),
    ((False, cf.FIXED_GENERATOR_POLICY_PLACEHOLDER), (True, 2)),
)
def test_cascade_provenance_and_candidate_rebound_reset_are_exact(
    learned_generator_policy: bool,
    expected_cascade_policy: int,
) -> None:
    learner = _ControlledRefreshLearner(
        n_features=5,
        n_tasks=1,
        candidate_count=2,
        refresh_parent=3,
        replacement_interval=2 if learned_generator_policy else 1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=0.0,
        max_depth=4,
        learn_generator_resources=learned_generator_policy,
        generator_resource_learning_rate=0.0,
        generator_resource_exploration=0.0,
        use_obgd=False,
    )
    state = _promotion_and_rebound_state(
        learner,
        learned_generator_policy=learned_generator_policy,
    )
    if learned_generator_policy:
        state = state.replace(  # type: ignore[attr-defined]
            replacement_accumulator=jnp.asarray(0.5, dtype=jnp.float32)
        )
    rebound_descriptor = (
        state.candidate_ops[1],
        state.candidate_parent_a[1],
        state.candidate_parent_b[1],
        state.candidate_theta[1],
        state.candidate_depth[1],
        state.candidate_generator_policy[1],
    )

    result = learner.update(
        state,
        jnp.asarray((1.0, 1.0, 1.0), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    )
    post = result.state

    assert int(result.replaced_slot) == 3
    assert int(result.promoted_candidate) == 0
    assert int(post.feature_generator_policy[3]) == 3
    assert int(post.feature_generator_policy[4]) == expected_cascade_policy
    assert int(post.candidate_parent_a[0]) == 3
    assert float(post.candidate_output_weights[0, 0]) == 23.0

    for before, after in zip(
        rebound_descriptor,
        (
            post.candidate_ops[1],
            post.candidate_parent_a[1],
            post.candidate_parent_b[1],
            post.candidate_theta[1],
            post.candidate_depth[1],
            post.candidate_generator_policy[1],
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))

    assert int(post.candidate_ages[1]) == 0
    for value in (
        post.candidate_output_weights[:, 1],
        post.candidate_utilities[1],
        post.candidate_utility_contribution_trace[:, 1],
        post.candidate_utility_feature_trace[1],
        post.candidate_utility_feature_energy_trace[1],
        post.candidate_utility_signal_second_moment[1],
        post.candidate_score_residual_trace[:, 1],
        post.candidate_score_energy_trace[1],
        post.candidate_retention_slow_utilities[1],
        post.candidate_active_correlation_trace[1],
        post.candidate_selector_log_weights[1],
        post.candidate_selector_cumulative_loss[1],
        post.candidate_selector_action_counts[1],
    ):
        array = np.asarray(value)
        assert array.dtype == np.float32
        np.testing.assert_array_equal(
            array.view(np.uint32),
            np.zeros(array.shape, dtype=np.uint32),
        )

    # Promotion evidence transfers before the refreshed candidate is reset.
    assert float(post.feature_score_residual_trace[0, 3]) == 12.0
    assert float(post.feature_score_energy_trace[3]) == 13.0
    assert float(post.retention_slow_utilities[3]) == 14.0
    np.testing.assert_array_equal(
        np.asarray(post.candidate_score_residual_trace[:, 0]),
        np.zeros((1,), dtype=np.float32),
    )


def test_fresh_promoted_candidate_referencing_later_cascade_loses_imprint() -> None:
    learner = _ControlledRefreshLearner(
        n_features=5,
        n_tasks=1,
        candidate_count=2,
        refresh_parent=4,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=0.0,
        max_depth=4,
        use_obgd=False,
    )
    state = _promotion_and_rebound_state(
        learner,
        learned_generator_policy=False,
    )

    post = learner.update(
        state,
        jnp.asarray((1.0, 1.0, 1.0), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    ).state

    assert int(post.candidate_parent_a[0]) == 4
    imprint_bits = np.asarray(post.candidate_output_weights[:, 0]).view(np.uint32)
    np.testing.assert_array_equal(
        imprint_bits,
        np.zeros(imprint_bits.shape, dtype=np.uint32),
    )


def test_rebound_recomputes_depth_and_resets_only_preexisting_trainable_theta() -> None:
    learner = _ControlledRefreshLearner(
        n_features=5,
        n_tasks=1,
        candidate_count=2,
        refresh_parent=3,
        refresh_op=cf.OP_TANH,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=0.0,
        max_depth=4,
        train_candidate_theta=True,
        use_obgd=False,
    )
    state = _promotion_and_rebound_state(
        learner,
        learned_generator_policy=False,
    ).replace(  # type: ignore[attr-defined]
        candidate_ops=jnp.asarray((cf.OP_SUM, cf.OP_TANH), dtype=jnp.int32),
        candidate_depth=jnp.asarray((1, 99), dtype=jnp.int32),
    )

    post = learner.update(
        state,
        jnp.asarray((1.0, 1.0, 1.0), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    ).state

    # This proposal was sampled after the promoted root changed, so its fresh
    # TANH parameters remain valid and its root-only imprint is preserved.
    np.testing.assert_array_equal(
        np.asarray(post.candidate_theta[0]),
        np.asarray((0.5, -0.5), dtype=np.float32),
    )
    assert float(post.candidate_output_weights[0, 0]) == 23.0

    # Candidate 1 pre-dates its cascaded parent refill.  Its structural depth
    # is recomputed from final parents and learned TANH parameters cold-reset.
    expected_depth = max(
        int(post.depth[int(post.candidate_parent_a[1])]),
        int(post.depth[int(post.candidate_parent_b[1])]),
    ) + 1
    assert int(post.candidate_depth[1]) == expected_depth
    theta_bits = np.asarray(post.candidate_theta[1]).view(np.uint32)
    np.testing.assert_array_equal(
        theta_bits,
        np.zeros(theta_bits.shape, dtype=np.uint32),
    )


def test_over_depth_candidate_rebound_is_regenerated_within_budget() -> None:
    learner = _RegenerationKeyProbeLearner(
        n_features=5,
        n_tasks=1,
        candidate_count=2,
        refresh_parent=0,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=0.0,
        max_depth=2,
        use_obgd=False,
    )
    state = _promotion_and_rebound_state(
        learner,
        learned_generator_policy=False,
    ).replace(  # type: ignore[attr-defined]
        parent_a=jnp.asarray((0, 1, 2, 0, 1), dtype=jnp.int32),
        parent_b=jnp.asarray((-1, -1, -1, 1, 2), dtype=jnp.int32),
        depth=jnp.asarray((0, 0, 0, 1, 1), dtype=jnp.int32),
        utilities=jnp.asarray((10.0, 10.0, 10.0, 10.0, 0.0), dtype=jnp.float32),
        candidate_parent_a=jnp.asarray((3, 4), dtype=jnp.int32),
        candidate_parent_b=jnp.asarray((0, 1), dtype=jnp.int32),
        candidate_depth=jnp.asarray((2, 2), dtype=jnp.int32),
    )

    result = learner.update(
        state,
        jnp.asarray((1.0, 1.0, 1.0), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    )
    post = result.state
    post.step_count.block_until_ready()

    assert int(result.replaced_slot) == 4
    assert int(result.promoted_candidate) == 0
    np.testing.assert_array_equal(
        np.asarray(result.curation_trace.candidate_rebound_mask),
        np.asarray((False, False)),
    )
    np.testing.assert_array_equal(
        np.asarray(result.curation_trace.candidate_overdepth_regeneration_mask),
        np.asarray((False, True)),
    )
    assert int(result.curation_trace.candidate_rebound_count) == 0
    assert int(result.curation_trace.candidate_overdepth_regeneration_count) == 1
    assert int(result.curation_trace.logical_event_count) == 3
    assert int(post.depth[4]) == learner._max_depth  # noqa: SLF001

    # Candidate 1 was valid at depth 2 before its parent slot changed.  The
    # promoted root deepens that final parent to depth 2, so preserving the
    # descriptor would strand the rebound candidate at depth 3.  It must be
    # regenerated against the final active bank instead.
    assert int(post.candidate_ops[1]) == cf.OP_PRODUCT
    assert int(post.candidate_parent_a[1]) == 0
    assert int(post.candidate_parent_b[1]) == 0
    assert int(post.candidate_depth[1]) == 1
    assert float(post.candidate_output_weights[0, 1]) == 23.0
    assert (
        int(post.candidate_generator_policy[1])
        == cf.FIXED_GENERATOR_POLICY_PLACEHOLDER
    )
    assert bool(jnp.all(post.candidate_depth <= learner._max_depth))  # noqa: SLF001

    proposal_words = tuple(
        int(word) for word in jr.key_data(result.curation_trace.proposal_key)
    )
    regeneration_slot_key = jr.fold_in(
        result.curation_trace.candidate_overdepth_regeneration_key,
        jnp.uint32(1),
    )
    regeneration_words = tuple(int(word) for word in jr.key_data(regeneration_slot_key))
    assert learner.generated_key_words == [proposal_words, regeneration_words]


def test_post_promotion_refresh_then_overdepth_regeneration_is_explicit() -> None:
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
    state = learner.init(feature_dim=2, key=jr.key(905)).replace(  # type: ignore[attr-defined]
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
        candidate_generator_policy=jnp.asarray((3, 2), dtype=jnp.int32),
    )
    learner.prepare_update(state)
    observation = jnp.asarray((1.25, 0.75), dtype=jnp.float32)
    target = jnp.asarray((1.0,), dtype=jnp.float32)

    result = learner.update(state, observation, target)
    post = result.state
    post.step_count.block_until_ready()
    trace = result.curation_trace

    assert int(result.replaced_slot) == 3
    assert int(result.promoted_candidate) == 0
    np.testing.assert_array_equal(
        np.asarray(trace.post_promotion_candidate_refresh_mask),
        np.asarray((True, False)),
    )
    np.testing.assert_array_equal(
        np.asarray(trace.cascade_refill_mask),
        np.asarray((False, False, False, False, True)),
    )
    np.testing.assert_array_equal(
        np.asarray(trace.candidate_rebound_mask),
        np.asarray((False, False)),
    )
    np.testing.assert_array_equal(
        np.asarray(trace.candidate_overdepth_regeneration_mask),
        np.asarray((True, False)),
    )
    assert int(trace.candidate_refresh_count) == 1
    assert int(trace.candidate_rebound_count) == 0
    assert int(trace.candidate_overdepth_regeneration_count) == 1
    assert int(trace.logical_event_count) == 4

    # The primary proposal remains the transient post-promotion birth in the
    # trace; final candidate arrays record the later ODRG repair birth.
    assert int(trace.proposal_destination_slot) == 0
    assert int(trace.proposal_parent_a) == 4
    assert int(trace.proposal_depth) == 3
    assert int(trace.candidate_final_parent_a[0]) == 0
    assert int(trace.candidate_final_parent_b[0]) == 0
    assert int(trace.candidate_final_depth[0]) == 1
    assert bool(jnp.all(post.candidate_depth <= learner._max_depth))  # noqa: SLF001

    proposal_words = tuple(int(word) for word in jr.key_data(trace.proposal_key))
    cascade_words = tuple(int(word) for word in jr.key_data(trace.cascade_key))
    repair_root_words = tuple(
        int(word) for word in jr.key_data(trace.candidate_overdepth_regeneration_key)
    )
    repair_slot_key = jr.fold_in(
        trace.candidate_overdepth_regeneration_key,
        jnp.uint32(0),
    )
    repair_slot_words = tuple(int(word) for word in jr.key_data(repair_slot_key))
    assert len({proposal_words, cascade_words, repair_root_words}) == 3
    assert [event[0] for event in learner.generation_events] == [
        proposal_words,
        repair_slot_words,
    ]

    final_values = cf._compute_feature_values(  # noqa: SLF001
        post.ops,
        post.parent_a,
        post.parent_b,
        post.theta,
        observation,
    )
    final_credit = result.errors @ post.output_weights
    np.testing.assert_allclose(learner.generation_events[-1][1], final_values)
    np.testing.assert_allclose(learner.generation_events[-1][2], final_credit)
    assert not np.allclose(learner.generation_events[0][1], final_values)
    repaired_value = final_values[0] * final_values[0]
    expected_imprint = cf._imprint_candidate_output_weights(  # noqa: SLF001
        result.errors,
        repaired_value,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    np.testing.assert_allclose(post.candidate_output_weights[:, 0], expected_imprint)
    assert float(jnp.abs(post.candidate_output_weights[0, 0])) > 0.0
