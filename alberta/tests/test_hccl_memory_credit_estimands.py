# mypy: disable-error-code="attr-defined,call-arg"
"""Pure contracts for comparing two-agent HCCL memory-credit estimands."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from alberta_framework.core.hccl_causal_attribution import HCCLTypedSignals
from alberta_framework.core.hccl_memory_credit_estimands import (
    HCCLMemoryCreditAgentEffects,
    HCCLMemoryCreditEstimandPanel,
    derive_hccl_memory_credit_estimands,
)
from alberta_framework.streams.hccl_causal_core import HCCLCausalCoreTypedSignals

pytestmark = pytest.mark.unit


def _signals(
    task_score: float,
    net_reward: tuple[float, float],
    safety_cost: tuple[float, float],
    message_charge: tuple[float, float],
) -> HCCLCausalCoreTypedSignals:
    return HCCLCausalCoreTypedSignals(
        task_score=jnp.asarray(task_score, dtype=jnp.float32),
        net_reward=jnp.asarray(net_reward, dtype=jnp.float32),
        message_charge=jnp.asarray(message_charge, dtype=jnp.float32),
        safety_cost=jnp.asarray(safety_cost, dtype=jnp.float32),
    )


def _zeros() -> HCCLCausalCoreTypedSignals:
    return _signals(0.0, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))


def _assert_effects(
    actual: HCCLMemoryCreditAgentEffects,
    *,
    task_score: Array,
    net_reward: Array,
    safety_cost: Array,
    message_charge: Array,
) -> None:
    chex.assert_trees_all_equal(actual.shared_task_score, task_score)
    chex.assert_trees_all_equal(actual.net_reward, net_reward)
    chex.assert_trees_all_equal(actual.safety_cost, safety_cost)
    chex.assert_trees_all_equal(actual.message_charge, message_charge)


def _assert_all_algebra_diagnostics(panel: HCCLMemoryCreditEstimandPanel) -> None:
    algebra = panel.algebra
    for leaf in jax.tree.leaves(algebra):
        assert np.asarray(leaf).dtype == np.dtype(np.bool_)
        assert bool(jnp.all(leaf))


def test_pure_synergy_distinguishes_all_three_credit_rules() -> None:
    zero = _zeros()
    mm = _signals(
        8.0,
        (4.0, 10.0),
        (2.0, 6.0),
        (12.0, 14.0),
    )

    panel = derive_hccl_memory_credit_estimands(
        mm=mm,
        b0m1=zero,
        m0b1=zero,
        bb=zero,
    )

    _assert_effects(
        panel.baseline_context_direct_effect,
        task_score=jnp.zeros((2,), dtype=jnp.float32),
        net_reward=jnp.zeros((2, 2), dtype=jnp.float32),
        safety_cost=jnp.zeros((2, 2), dtype=jnp.float32),
        message_charge=jnp.zeros((2, 2), dtype=jnp.float32),
    )
    expected_factual = HCCLMemoryCreditAgentEffects(
        shared_task_score=jnp.asarray((8.0, 8.0), dtype=jnp.float32),
        net_reward=jnp.asarray(((4.0, 10.0), (4.0, 10.0)), dtype=jnp.float32),
        safety_cost=jnp.asarray(((2.0, 6.0), (2.0, 6.0)), dtype=jnp.float32),
        message_charge=jnp.asarray(((12.0, 14.0), (12.0, 14.0)), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(panel.factual_context_leave_one_out, expected_factual)
    chex.assert_trees_all_equal(panel.componentwise_interaction.task_score, mm.task_score)
    chex.assert_trees_all_equal(panel.componentwise_interaction.net_reward, mm.net_reward)
    chex.assert_trees_all_equal(panel.componentwise_interaction.safety_cost, mm.safety_cost)
    chex.assert_trees_all_equal(
        panel.componentwise_interaction.message_charge,
        mm.message_charge,
    )
    _assert_effects(
        panel.shapley_allocation,
        task_score=jnp.asarray((4.0, 4.0), dtype=jnp.float32),
        net_reward=jnp.asarray(((2.0, 5.0), (2.0, 5.0)), dtype=jnp.float32),
        safety_cost=jnp.asarray(((1.0, 3.0), (1.0, 3.0)), dtype=jnp.float32),
        message_charge=jnp.asarray(((6.0, 7.0), (6.0, 7.0)), dtype=jnp.float32),
    )
    _assert_all_algebra_diagnostics(panel)


@pytest.mark.parametrize("changing_agent", (0, 1))
def test_one_sided_change_makes_baseline_factual_and_shapley_identical(
    changing_agent: int,
) -> None:
    bb = _signals(1.0, (2.0, 3.0), (4.0, 5.0), (6.0, 7.0))
    changed = _signals(9.0, (12.0, 23.0), (34.0, 45.0), (56.0, 67.0))
    if changing_agent == 0:
        mm, b0m1, m0b1 = changed, bb, changed
    else:
        mm, b0m1, m0b1 = changed, changed, bb

    panel = derive_hccl_memory_credit_estimands(
        mm=mm,
        b0m1=b0m1,
        m0b1=m0b1,
        bb=bb,
    )

    chex.assert_trees_all_equal(
        panel.baseline_context_direct_effect,
        panel.factual_context_leave_one_out,
    )
    chex.assert_trees_all_equal(
        panel.baseline_context_direct_effect,
        panel.shapley_allocation,
    )
    for leaf in jax.tree.leaves(panel.componentwise_interaction):
        chex.assert_trees_all_equal(leaf, jnp.zeros_like(leaf))
    _assert_all_algebra_diagnostics(panel)


def test_additive_asymmetric_channels_preserve_credit_and_signal_agent_axes() -> None:
    bb = _zeros()
    m0b1 = _signals(2.0, (3.0, 30.0), (4.0, 40.0), (5.0, 50.0))
    b0m1 = _signals(7.0, (11.0, 70.0), (13.0, 80.0), (17.0, 90.0))
    mm = _signals(9.0, (14.0, 100.0), (17.0, 120.0), (22.0, 140.0))

    panel = derive_hccl_memory_credit_estimands(
        mm=mm,
        b0m1=b0m1,
        m0b1=m0b1,
        bb=bb,
    )

    expected = HCCLMemoryCreditAgentEffects(
        shared_task_score=jnp.asarray((2.0, 7.0), dtype=jnp.float32),
        net_reward=jnp.asarray(((3.0, 30.0), (11.0, 70.0)), dtype=jnp.float32),
        safety_cost=jnp.asarray(((4.0, 40.0), (13.0, 80.0)), dtype=jnp.float32),
        message_charge=jnp.asarray(((5.0, 50.0), (17.0, 90.0)), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(panel.baseline_context_direct_effect, expected)
    chex.assert_trees_all_equal(panel.factual_context_leave_one_out, expected)
    chex.assert_trees_all_equal(panel.shapley_allocation, expected)
    chex.assert_trees_all_equal(
        jnp.diag(panel.baseline_context_direct_effect.net_reward),
        jnp.asarray((3.0, 70.0), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        jnp.sum(panel.shapley_allocation.net_reward, axis=0),
        panel.memory_total.net_reward,
    )
    chex.assert_trees_all_equal(
        jnp.sum(panel.shapley_allocation.shared_task_score),
        panel.memory_total.task_score,
    )
    _assert_all_algebra_diagnostics(panel)


@pytest.mark.parametrize(
    "field,value",
    (
        ("task_score", jnp.asarray(jnp.nan, dtype=jnp.float32)),
        ("net_reward", jnp.asarray((jnp.inf, 0.0), dtype=jnp.float32)),
        ("safety_cost", jnp.asarray((0.0, -jnp.inf), dtype=jnp.float32)),
        ("message_charge", jnp.asarray((jnp.nan, 0.0), dtype=jnp.float32)),
    ),
)
def test_nonfinite_input_is_rejected(field: str, value: Array) -> None:
    signal = _signals(0.0, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)).replace(
        **{field: value}
    )

    with pytest.raises(ValueError, match=rf"mm\.{field}.*finite"):
        derive_hccl_memory_credit_estimands(
            mm=signal,
            b0m1=_zeros(),
            m0b1=_zeros(),
            bb=_zeros(),
        )


@pytest.mark.parametrize(
    "mutate,exception,match",
    (
        (
            lambda value: value.replace(
                task_score=jnp.zeros((1,), dtype=jnp.float32)
            ),
            ValueError,
            r"mm\.task_score.*shape",
        ),
        (
            lambda value: value.replace(
                net_reward=jnp.zeros((3,), dtype=jnp.float32)
            ),
            ValueError,
            r"mm\.net_reward.*shape",
        ),
        (
            lambda value: value.replace(
                safety_cost=jnp.zeros((2,), dtype=jnp.int32)
            ),
            TypeError,
            r"mm\.safety_cost.*dtype",
        ),
    ),
)
def test_wrong_shape_or_dtype_is_rejected(
    mutate: Callable[[HCCLCausalCoreTypedSignals], HCCLCausalCoreTypedSignals],
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        derive_hccl_memory_credit_estimands(
            mm=mutate(_zeros()),
            b0m1=_zeros(),
            m0b1=_zeros(),
            bb=_zeros(),
        )


def test_nonexact_signal_type_is_rejected() -> None:
    wrong = HCCLTypedSignals(
        task_score=jnp.asarray(0.0, dtype=jnp.float32),
        net_reward=jnp.zeros((2,), dtype=jnp.float32),
        safety_cost=jnp.zeros((2,), dtype=jnp.float32),
        message_charge=jnp.zeros((2,), dtype=jnp.float32),
    )

    with pytest.raises(TypeError, match="mm.*exact HCCLCausalCoreTypedSignals"):
        derive_hccl_memory_credit_estimands(
            mm=wrong,  # type: ignore[arg-type]
            b0m1=_zeros(),
            m0b1=_zeros(),
            bb=_zeros(),
        )


def test_result_is_frozen_input_is_unchanged_and_jit_rejection_is_explicit() -> None:
    mm = _signals(8.0, (4.0, 10.0), (2.0, 6.0), (12.0, 14.0))
    before = jax.tree.map(np.asarray, mm)
    panel = derive_hccl_memory_credit_estimands(
        mm=mm,
        b0m1=_zeros(),
        m0b1=_zeros(),
        bb=_zeros(),
    )
    chex.assert_trees_all_equal(jax.tree.map(np.asarray, mm), before)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(panel, "baseline_context_direct_effect", panel.shapley_allocation)

    def traced(task_score: Array) -> Array:
        traced_mm = mm.replace(task_score=task_score)
        return derive_hccl_memory_credit_estimands(
            mm=traced_mm,
            b0m1=_zeros(),
            m0b1=_zeros(),
            bb=_zeros(),
        ).memory_total.task_score

    with pytest.raises(TypeError, match="host/eager.*finite"):
        jax.jit(traced)(mm.task_score)
