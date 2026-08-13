"""Stage-exact residual guidance contracts for compositional curation."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core import compositional_features as cf


@pytest.fixture(autouse=True)
def _release_jax_compilation_cache() -> Iterator[None]:
    yield
    jax.clear_caches()  # type: ignore[no-untyped-call]


class _StageGuidanceProbeLearner(cf.CompositionalFeatureLearner):
    """Record runtime-only guidance consumers without changing their values."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.guidance_events: list[tuple[Any, Any]] = []
        self.generation_events: list[tuple[Any, Any]] = []
        self.cascade_events: list[tuple[Any, Any]] = []
        self.parent_events: list[tuple[Any, Any]] = []

    @staticmethod
    def _append_pair(target: list[tuple[Any, Any]], values: Any, credit: Any) -> None:
        target.append(
            (
                np.array(values, dtype=np.float32, copy=True),
                np.array(credit, dtype=np.float32, copy=True),
            )
        )

    def _curation_stage_guidance(
        self,
        ops: jax.Array,
        parent_a: jax.Array,
        parent_b: jax.Array,
        theta: jax.Array,
        output_weights: jax.Array,
        observation: jax.Array,
        errors: jax.Array,
        active_count: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        values, credit = super()._curation_stage_guidance(
            ops,
            parent_a,
            parent_b,
            theta,
            output_weights,
            observation,
            errors,
            active_count,
        )
        jax.debug.callback(
            lambda actual_values, actual_credit: self._append_pair(
                self.guidance_events,
                actual_values,
                actual_credit,
            ),
            values,
            credit,
            ordered=True,
        )
        return values, credit

    def _generate_one(self, key: jax.Array, *args: Any, **kwargs: Any) -> Any:
        values = kwargs.get("feature_values")
        credit = kwargs.get("feature_credit")
        if values is not None and credit is not None:
            jax.debug.callback(
                lambda actual_values, actual_credit: self._append_pair(
                    self.generation_events,
                    actual_values,
                    actual_credit,
                ),
                values,
                credit,
                ordered=True,
            )
        return super()._generate_one(key, *args, **kwargs)

    def _cascade_replace_with_mask(self, *args: Any, **kwargs: Any) -> Any:
        values = kwargs.get("feature_values")
        credit = kwargs.get("feature_credit")
        if values is not None and credit is not None:
            jax.debug.callback(
                lambda actual_values, actual_credit: self._append_pair(
                    self.cascade_events,
                    actual_values,
                    actual_credit,
                ),
                values,
                credit,
                ordered=True,
            )
        return super()._cascade_replace_with_mask(*args, **kwargs)

    def _parent_logits(
        self,
        eligible: jax.Array,
        utilities: jax.Array,
        feature_values: jax.Array | None = None,
        feature_credit: jax.Array | None = None,
        depth: jax.Array | None = None,
        ages: jax.Array | None = None,
        parent_mode: jax.Array | None = None,
    ) -> jax.Array:
        if feature_values is not None and feature_credit is not None:
            jax.debug.callback(
                lambda actual_values, actual_credit: self._append_pair(
                    self.parent_events,
                    actual_values,
                    actual_credit,
                ),
                feature_values,
                feature_credit,
                ordered=True,
            )
        return super()._parent_logits(
            eligible,
            utilities,
            feature_values=feature_values,
            feature_credit=feature_credit,
            depth=depth,
            ages=ages,
            parent_mode=parent_mode,
        )


def _post_gradient_active_stage(
    state: cf.CompositionalFeatureState,
    observation: jax.Array,
    targets: jax.Array,
    *,
    step_size_output: float,
    step_size_theta: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    pre_values = cf._compute_feature_values(  # noqa: SLF001
        state.ops,
        state.parent_a,
        state.parent_b,
        state.theta,
        observation,
    )
    errors = targets - (state.output_weights @ pre_values + state.output_bias)
    active_count = jnp.asarray(float(targets.shape[0]), dtype=jnp.float32)
    pre_credit = (errors @ state.output_weights) / active_count
    d_theta0, d_theta1 = cf._theta_local_grads(  # noqa: SLF001
        state.ops,
        state.parent_a,
        state.parent_b,
        state.theta,
        pre_values,
    )
    stage_theta = state.theta + step_size_theta * jnp.stack(
        (pre_credit * d_theta0, pre_credit * d_theta1),
        axis=-1,
    )
    stage_weights = state.output_weights + (
        step_size_output * errors[:, None] * pre_values[None, :] / active_count
    )
    stage_values = cf._compute_feature_values(  # noqa: SLF001
        state.ops,
        state.parent_a,
        state.parent_b,
        stage_theta,
        observation,
    )
    stage_credit = (errors @ stage_weights) / active_count
    return stage_values, stage_credit, errors, stage_theta, stage_weights


def _assert_pair(
    actual: tuple[Any, Any],
    expected_values: jax.Array,
    expected_credit: jax.Array,
) -> None:
    np.testing.assert_allclose(actual[0], np.asarray(expected_values), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual[1], np.asarray(expected_credit), rtol=1e-6, atol=1e-6)


def _promotion_state(
    learner: cf.CompositionalFeatureLearner,
) -> cf.CompositionalFeatureState:
    state = learner.init(feature_dim=3, key=jr.key(2101))
    return cast(
        cf.CompositionalFeatureState,
        state.replace(  # type: ignore[attr-defined]
            ops=jnp.asarray(
                (cf.OP_RAW, cf.OP_RAW, cf.OP_RAW, cf.OP_TANH, cf.OP_PRODUCT),
                dtype=jnp.int32,
            ),
            parent_a=jnp.asarray((0, 1, 2, 0, 3), dtype=jnp.int32),
            parent_b=jnp.asarray((-1, -1, -1, 1, 2), dtype=jnp.int32),
            theta=jnp.asarray(
                ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.4, -0.2), (0.3, 0.1)),
                dtype=jnp.float32,
            ),
            depth=jnp.asarray((0, 0, 0, 1, 2), dtype=jnp.int32),
            output_weights=jnp.asarray(
                ((0.2, -0.3, 0.5, 0.7, -0.4),), dtype=jnp.float32
            ),
            utilities=jnp.asarray(
                (10.0, 10.0, 10.0, 0.0, 10.0), dtype=jnp.float32
            ),
            ages=jnp.full((5,), 10, dtype=jnp.int32),
            candidate_ops=jnp.asarray((cf.OP_SUM, cf.OP_PRODUCT), dtype=jnp.int32),
            candidate_parent_a=jnp.asarray((0, 4), dtype=jnp.int32),
            candidate_parent_b=jnp.asarray((1, 2), dtype=jnp.int32),
            candidate_theta=jnp.asarray(
                ((0.25, -0.1), (0.3, 0.4)), dtype=jnp.float32
            ),
            candidate_depth=jnp.asarray((1, 3), dtype=jnp.int32),
            candidate_output_weights=jnp.asarray(
                ((0.6, -0.2),), dtype=jnp.float32
            ),
            candidate_utilities=jnp.asarray((100.0, 1.0), dtype=jnp.float32),
            candidate_ages=jnp.asarray((10, 10), dtype=jnp.int32),
        ),
    )


def _direct_state(
    learner: cf.CompositionalFeatureLearner,
) -> cf.CompositionalFeatureState:
    state = learner.init(feature_dim=3, key=jr.key(2102))
    return cast(
        cf.CompositionalFeatureState,
        state.replace(  # type: ignore[attr-defined]
            ops=jnp.asarray(
                (cf.OP_RAW, cf.OP_RAW, cf.OP_RAW, cf.OP_TANH, cf.OP_PRODUCT),
                dtype=jnp.int32,
            ),
            parent_a=jnp.asarray((0, 1, 2, 0, 3), dtype=jnp.int32),
            parent_b=jnp.asarray((-1, -1, -1, 1, 2), dtype=jnp.int32),
            theta=jnp.asarray(
                ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.4, -0.2), (0.3, 0.1)),
                dtype=jnp.float32,
            ),
            depth=jnp.asarray((0, 0, 0, 1, 2), dtype=jnp.int32),
            output_weights=jnp.asarray(
                ((0.2, -0.3, 0.5, 0.7, -0.4),), dtype=jnp.float32
            ),
            utilities=jnp.asarray(
                (10.0, 10.0, 10.0, 0.0, 10.0), dtype=jnp.float32
            ),
            ages=jnp.full((5,), 10, dtype=jnp.int32),
        ),
    )


def test_ordinary_refresh_uses_post_gradient_pre_root_guidance() -> None:
    step_size_output = 0.2
    step_size_theta = 0.3
    learner = _StageGuidanceProbeLearner(
        n_features=4,
        n_tasks=1,
        candidate_count=1,
        step_size_output=step_size_output,
        step_size_theta=step_size_theta,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0e6,
        generation_strategy=cf.GENERATION_RESIDUAL_IMPRINT,
        use_obgd=False,
    )
    state = learner.init(feature_dim=2, key=jr.key(2100)).replace(  # type: ignore[attr-defined]
        ops=jnp.asarray((cf.OP_RAW, cf.OP_RAW, cf.OP_TANH, cf.OP_PRODUCT)),
        parent_a=jnp.asarray((0, 1, 0, 2), dtype=jnp.int32),
        parent_b=jnp.asarray((-1, -1, 1, 1), dtype=jnp.int32),
        theta=jnp.asarray(((0.0, 0.0), (0.0, 0.0), (0.4, -0.3), (0.2, 0.1))),
        depth=jnp.asarray((0, 0, 1, 2), dtype=jnp.int32),
        output_weights=jnp.asarray(((0.2, -0.3, 0.7, -0.4),), dtype=jnp.float32),
        utilities=jnp.asarray((10.0, 10.0, 1.0, 1.0), dtype=jnp.float32),
        ages=jnp.full((4,), 10, dtype=jnp.int32),
        candidate_ages=jnp.asarray((10,), dtype=jnp.int32),
        candidate_utilities=jnp.asarray((0.0,), dtype=jnp.float32),
    )
    observation = jnp.asarray((0.8, -0.35), dtype=jnp.float32)
    targets = jnp.asarray((0.65,), dtype=jnp.float32)
    expected_values, expected_credit, *_ = _post_gradient_active_stage(
        state,
        observation,
        targets,
        step_size_output=step_size_output,
        step_size_theta=step_size_theta,
    )

    result = learner.update(state, observation, targets)
    result.state.step_count.block_until_ready()

    assert bool(result.curation_trace.ordinary_candidate_refresh_mask[0])
    assert len(learner.guidance_events) == 1
    assert len(learner.generation_events) == 1
    assert learner.cascade_events == []
    _assert_pair(learner.guidance_events[0], expected_values, expected_credit)
    _assert_pair(learner.generation_events[0], expected_values, expected_credit)


def test_promotion_refresh_and_cascade_share_post_promotion_guidance() -> None:
    step_size_output = 0.2
    step_size_theta = 0.3
    learner = _StageGuidanceProbeLearner(
        n_features=5,
        n_tasks=1,
        candidate_count=2,
        step_size_output=step_size_output,
        step_size_theta=step_size_theta,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=0.0,
        max_depth=4,
        generation_strategy=cf.GENERATION_RESIDUAL_IMPRINT,
        use_obgd=False,
    )
    state = _promotion_state(learner)
    observation = jnp.asarray((0.8, -0.35, 0.45), dtype=jnp.float32)
    targets = jnp.asarray((0.65,), dtype=jnp.float32)
    _, _, errors, active_theta, active_weights = _post_gradient_active_stage(
        state,
        observation,
        targets,
        step_size_output=step_size_output,
        step_size_theta=step_size_theta,
    )
    pre_values = cf._compute_feature_values(  # noqa: SLF001
        state.ops,
        state.parent_a,
        state.parent_b,
        state.theta,
        observation,
    )
    candidate_values = learner._candidate_features(  # noqa: SLF001
        state,
        pre_values,
        observation,
    )
    candidate_weights = state.candidate_output_weights + (
        step_size_output * errors[:, None] * candidate_values[None, :]
    )
    destination = 3
    source = 0
    stage_ops = state.ops.at[destination].set(state.candidate_ops[source])
    stage_parent_a = state.parent_a.at[destination].set(state.candidate_parent_a[source])
    stage_parent_b = state.parent_b.at[destination].set(state.candidate_parent_b[source])
    stage_theta = active_theta.at[destination].set(state.candidate_theta[source])
    promoted_weights = learner._promoted_output_weights(  # noqa: SLF001
        active_weights[:, destination],
        candidate_weights[:, source],
    )
    stage_weights = active_weights.at[:, destination].set(promoted_weights)
    expected_values = cf._compute_feature_values(  # noqa: SLF001
        stage_ops,
        stage_parent_a,
        stage_parent_b,
        stage_theta,
        observation,
    )
    expected_credit = errors @ stage_weights

    result = learner.update(state, observation, targets)
    result.state.step_count.block_until_ready()

    assert bool(result.curation_trace.promotion_applied)
    assert bool(result.curation_trace.cascade_refill_mask[4])
    assert len(learner.guidance_events) == 1
    assert len(learner.generation_events) == 1
    assert len(learner.cascade_events) == 1
    _assert_pair(learner.guidance_events[0], expected_values, expected_credit)
    _assert_pair(learner.generation_events[0], expected_values, expected_credit)
    _assert_pair(learner.cascade_events[0], expected_values, expected_credit)


def test_direct_root_and_cascade_use_pre_and_post_root_guidance() -> None:
    step_size_output = 0.2
    step_size_theta = 0.3
    learner = _StageGuidanceProbeLearner(
        n_features=5,
        n_tasks=1,
        candidate_count=0,
        step_size_output=step_size_output,
        step_size_theta=step_size_theta,
        replacement_interval=1,
        min_feature_age=0,
        max_depth=4,
        generation_strategy=cf.GENERATION_RESIDUAL_IMPRINT,
        use_obgd=False,
    )
    state = _direct_state(learner)
    observation = jnp.asarray((0.8, -0.35, 0.45), dtype=jnp.float32)
    targets = jnp.asarray((0.65,), dtype=jnp.float32)
    pre_values, pre_credit, errors, stage_theta, stage_weights = (
        _post_gradient_active_stage(
            state,
            observation,
            targets,
            step_size_output=step_size_output,
            step_size_theta=step_size_theta,
        )
    )

    result = learner.update(state, observation, targets)
    result.state.step_count.block_until_ready()

    root = int(result.replaced_slot)
    assert root == 3
    post_root_ops = state.ops.at[root].set(result.state.ops[root])
    post_root_parent_a = state.parent_a.at[root].set(result.state.parent_a[root])
    post_root_parent_b = state.parent_b.at[root].set(result.state.parent_b[root])
    post_root_theta = stage_theta.at[root].set(result.state.theta[root])
    post_root_weights = stage_weights.at[:, root].set(0.0)
    post_root_values = cf._compute_feature_values(  # noqa: SLF001
        post_root_ops,
        post_root_parent_a,
        post_root_parent_b,
        post_root_theta,
        observation,
    )
    post_root_credit = errors @ post_root_weights

    assert bool(result.curation_trace.cascade_refill_mask[4])
    assert len(learner.guidance_events) == 2
    assert len(learner.cascade_events) == 1
    _assert_pair(learner.guidance_events[0], pre_values, pre_credit)
    _assert_pair(learner.guidance_events[1], post_root_values, post_root_credit)
    _assert_pair(learner.cascade_events[0], post_root_values, post_root_credit)
    _assert_pair(learner.parent_events[0], pre_values, pre_credit)
    for parent_event in learner.parent_events[1:]:
        _assert_pair(parent_event, post_root_values, post_root_credit)


def test_no_event_executes_no_curation_guidance_or_generation() -> None:
    learner = _StageGuidanceProbeLearner(
        n_features=3,
        n_tasks=1,
        candidate_count=1,
        replacement_interval=32,
        min_feature_age=100,
        candidate_min_age=16,
        generation_strategy=cf.GENERATION_RESIDUAL_IMPRINT,
        use_obgd=False,
    )
    state = learner.init(feature_dim=2, key=jr.key(2103))

    result = learner.update(
        state,
        jnp.asarray((0.25, -0.5), dtype=jnp.float32),
        jnp.asarray((0.75,), dtype=jnp.float32),
    )
    result.state.step_count.block_until_ready()

    assert not bool(result.curation_trace.has_event)
    assert learner.guidance_events == []
    assert learner.generation_events == []
    assert learner.cascade_events == []
    assert learner.parent_events == []
