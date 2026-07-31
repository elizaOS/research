# mypy: disable-error-code="call-arg"
"""Recycled prototype slots must start with a fresh readout row.

When the basis is full and a novel observation evicts a slot, the value-map
row learned for the evicted center must not be inherited by the new
prototype: stale weights for a different basis function would inject
arbitrary bias into predictions at the new center.
"""

import chex
import jax.numpy as jnp

from alberta_framework.core.prototype_basis import (
    PrototypeBasisBlock,
    PrototypeBasisConfig,
    run_prototype_basis_arrays,
)


def test_recycled_slot_readout_row_is_reset() -> None:
    block = PrototypeBasisBlock(
        PrototypeBasisConfig(
            input_dim=1,
            output_dim=2,
            n_prototypes=2,
            step_size=0.1,
            novelty_threshold=0.05,
            bandwidth=0.05,
        )
    )
    params, state = block.init()
    obs_a = jnp.asarray([0.0], dtype=jnp.float32)
    obs_b = jnp.asarray([1.0], dtype=jnp.float32)
    target_a = jnp.asarray([1.0, -0.5], dtype=jnp.float32)
    target_b = jnp.asarray([-1.0, 0.5], dtype=jnp.float32)

    # Fill both slots and train non-trivial readout rows for each.
    for _ in range(5):
        result = block.update(params, state, obs_a, target_a)
        params, state = result.params, result.state
        result = block.update(params, state, obs_b, target_b)
        params, state = result.params, result.state
    # Extra visits to A make B's slot the least-used replacement candidate.
    for _ in range(3):
        result = block.update(params, state, obs_a, target_a)
        params, state = result.params, result.state
    assert int(jnp.sum(state.counts > 0.0)) == 2

    # A far-away observation with no empty slots forces an eviction+recycle.
    obs_c = jnp.asarray([5.0], dtype=jnp.float32)
    target_c = jnp.asarray([0.0, 0.0], dtype=jnp.float32)
    _, _, slot, novel = block.update_centers_with_slot(state, obs_c)
    assert bool(novel)
    assert float(state.counts[slot]) > 0.0  # true recycle, not an empty slot
    assert int(slot) == 1
    stale_row = params.values[int(slot)]
    assert float(jnp.max(jnp.abs(stale_row))) > 1e-2

    result = block.update(params, state, obs_c, target_c)

    # The recycled slot starts fresh with the neutral zero readout row.
    chex.assert_trees_all_close(
        result.params.values[int(slot)],
        jnp.zeros(2, dtype=jnp.float32),
    )
    # Non-recycled rows still follow the plain LMS update.
    expected = params.values - 0.1 * result.activations[:, None] * result.error[None, :]
    chex.assert_trees_all_close(result.params.values[0], expected[0])
    # Recycled center state is reset as before.
    chex.assert_trees_all_close(result.state.centers[int(slot)], obs_c)
    assert float(result.state.counts[int(slot)]) == 1.0


def test_no_error_spike_after_recycle() -> None:
    block = PrototypeBasisBlock(
        PrototypeBasisConfig(
            input_dim=1,
            output_dim=1,
            n_prototypes=1,
            step_size=0.1,
            novelty_threshold=0.05,
            bandwidth=0.05,
        )
    )
    params, state = block.init()
    obs_a = jnp.asarray([0.0], dtype=jnp.float32)
    target_a = jnp.asarray([2.0], dtype=jnp.float32)
    for _ in range(60):
        result = block.update(params, state, obs_a, target_a)
        params, state = result.params, result.state

    # A distant observation recycles the single slot on this update.
    obs_b = jnp.asarray([3.0], dtype=jnp.float32)
    target_b = jnp.asarray([0.0], dtype=jnp.float32)
    result = block.update(params, state, obs_b, target_b)
    transition_error = float(jnp.abs(result.error)[0])
    params, state = result.params, result.state
    chex.assert_trees_all_close(state.centers[0], obs_b)
    assert float(state.counts[0]) == 1.0

    # Fresh slot: the recycled prototype contributes nothing, so the
    # prediction at the new center is the bias alone.
    chex.assert_trees_all_close(
        block.predict(params, state, obs_b),
        params.bias,
        atol=1e-6,
    )

    errors = []
    for _ in range(15):
        result = block.update(params, state, obs_b, target_b)
        params, state = result.params, result.state
        errors.append(float(jnp.abs(result.error)[0]))
    # No spike: post-recycle errors stay below the transition error and decay.
    assert max(errors) < transition_error
    assert errors[-1] < errors[0]
    assert errors[-1] < 0.2


def test_run_prototype_basis_arrays_recycles_cleanly_under_scan() -> None:
    block = PrototypeBasisBlock(
        PrototypeBasisConfig(
            input_dim=1,
            output_dim=1,
            n_prototypes=1,
            step_size=0.1,
            novelty_threshold=0.05,
            bandwidth=0.05,
        )
    )
    observations = jnp.concatenate(
        [
            jnp.zeros((20, 1), dtype=jnp.float32),
            jnp.full((20, 1), 3.0, dtype=jnp.float32),
        ]
    )
    targets = jnp.concatenate(
        [
            jnp.full((20, 1), 2.0, dtype=jnp.float32),
            jnp.zeros((20, 1), dtype=jnp.float32),
        ]
    )

    result = run_prototype_basis_arrays(block, observations, targets)

    chex.assert_tree_all_finite(result)
    # The regime switch allocates (recycles) the single slot exactly once more.
    assert float(result.metrics[20, 4]) == 1.0
    assert float(result.metrics[-1, 0]) < 0.25  # final mse settles near zero
