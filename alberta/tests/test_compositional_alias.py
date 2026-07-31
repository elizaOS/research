"""Regression tests: raw-input refill fallback must not alias raw indices.

When ``_cascade_replace`` refills a slot that has no surviving eligible
parent, it falls back to an ``OP_RAW`` passthrough.  The invariant (also
enforced by ``_compute_feature_values``) is that an ``OP_RAW`` slot's
``parent_a`` is a valid raw-input index in ``[0, observation_dim)``.
Historically the fallback clamped against ``n_features`` instead of the
raw-input dim, so refilled slots at index ``>= observation_dim`` stored
out-of-range raw indices that silently clamped onto the last raw input
at evaluation time, aliasing a wrong raw feature.
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.compositional_features import (
    OP_RAW,
    CompositionalFeatureLearner,
    _compute_feature_values,
)


class TestRawRefillFallbackAliasing:
    """The OP_RAW refill fallback must produce in-range raw indices."""

    def test_fallback_raw_indices_stay_within_observation_dim(self) -> None:
        """Refilled OP_RAW slots must reference a real raw input."""
        feature_dim = 3
        n_features = 6
        learner = CompositionalFeatureLearner(n_features=n_features, n_tasks=1)
        state = learner.init(feature_dim=feature_dim, key=jr.key(0))

        observation = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        # Replace every slot: each refill then sees no surviving earlier
        # parent, which forces the OP_RAW fallback for every slot —
        # including slots at index >= feature_dim.
        replaced_mask = jnp.ones(n_features, dtype=bool)
        ops_f, pa_f, pb_f, theta_f, depth_f, *_ = learner._cascade_replace(
            state.ops,
            state.parent_a,
            state.parent_b,
            state.theta,
            state.depth,
            state.utilities,
            state.ages,
            state.output_weights,
            replaced_mask,
            observation,
            jr.key(1),
        )

        ops_np = np.asarray(ops_f)
        pa_np = np.asarray(pa_f)
        pb_np = np.asarray(pb_f)
        depth_np = np.asarray(depth_f)
        assert (ops_np == OP_RAW).all()
        assert (pb_np == -1).all()
        assert (depth_np == 0).all()
        # Core invariant: OP_RAW parent_a is a raw-input index.  Under the
        # aliasing bug, slots 3..5 stored raw indices 3..5 (>= feature_dim).
        assert (pa_np >= 0).all()
        assert (pa_np < feature_dim).all(), (
            f"OP_RAW fallback produced out-of-range raw indices {pa_np} "
            f"for observation dim {feature_dim}"
        )

        # Evaluation must agree with plain (unclamped) raw indexing; under
        # the bug the stored indices are unindexable in the observation.
        values = np.asarray(
            _compute_feature_values(ops_f, pa_f, pb_f, theta_f, observation)
        )
        obs_np = np.asarray(observation)
        for slot in range(n_features):
            assert values[slot] == obs_np[pa_np[slot]]

    def test_fallback_prefix_slots_keep_identity_raw_indices(self) -> None:
        """Slots below the raw dim still pass through their own raw input."""
        feature_dim = 4
        n_features = 7
        learner = CompositionalFeatureLearner(n_features=n_features, n_tasks=2)
        state = learner.init(feature_dim=feature_dim, key=jr.key(2))

        observation = jnp.array([0.5, -1.0, 2.0, 0.25], dtype=jnp.float32)
        replaced_mask = jnp.ones(n_features, dtype=bool)
        _, pa_f, *_ = learner._cascade_replace(
            state.ops,
            state.parent_a,
            state.parent_b,
            state.theta,
            state.depth,
            state.utilities,
            state.ages,
            state.output_weights,
            replaced_mask,
            observation,
            jr.key(3),
        )

        pa_np = np.asarray(pa_f)
        # The raw prefix keeps the identity mapping, and overflow slots are
        # clamped to the last valid raw input rather than aliasing past it.
        np.testing.assert_array_equal(pa_np[:feature_dim], np.arange(feature_dim))
        assert (pa_np[feature_dim:] < feature_dim).all()
