# mypy: disable-error-code="call-arg"
"""Pure two-agent memory-credit estimands for the four HCCL memory vertices.

The input order is named rather than positional: ``MM``, ``B0M1``, ``M0B1``,
and ``BB``.  This module owns no state and chooses no controller feedback.  It
only exposes three defensible two-player credit views over every typed signal
channel:

* baseline-context direct effects hold the other agent at ``B``;
* factual-context leave-one-out effects hold the other agent at executed ``M``;
* Shapley allocations average those two marginal orders and split interaction.

Per-agent signal fields have shape ``[credit_agent, signal_agent]``.  A live
controller's own-component readout is therefore the diagonal: ``[0, 0]`` for
agent 0 and ``[1, 1]`` for agent 1.  The task score is one shared scalar, so its
credit result has shape ``[credit_agent]`` rather than being silently broadcast
onto a signal-agent axis.  Summing either Shapley representation over its
credit-agent axis recovers the corresponding ``MM - BB`` signal.

Exact finite-value validation is deliberately host/eager-only.  Shape and dtype
contracts are static, but a traced finite check cannot raise a normal Python
exception.  Callers that need a compiled kernel should stage already-validated
arrays in their own checked boundary; this comparison helper itself fails
closed when traced.
"""

from __future__ import annotations

from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float

from alberta_framework.core.hccl_causal_attribution import HCCLSignalContrast
from alberta_framework.streams.hccl_causal_core import HCCLCausalCoreTypedSignals

_N_AGENTS = 2
_FLOAT32 = jnp.dtype(jnp.float32)


@chex.dataclass(frozen=True)
class HCCLMemoryCreditAgentEffects:
    """Effects allocated by acting agent, preserving every signal component.

    ``shared_task_score[a]`` is credit agent ``a``'s effect/allocation for the
    one shared task scalar.  Each matrix field ``x[a, s]`` is credit agent
    ``a``'s effect/allocation for signal-agent component ``s``.
    """

    shared_task_score: Float[Array, " 2"]
    net_reward: Float[Array, "2 2"]
    safety_cost: Float[Array, "2 2"]
    message_charge: Float[Array, "2 2"]


@chex.dataclass(frozen=True)
class HCCLMemoryCreditAlgebraDiagnostics:
    """Bit-exact, component-local identities for the returned estimands."""

    factual_equals_baseline_plus_interaction_task_score: Bool[Array, " 2"]
    factual_equals_baseline_plus_interaction_net_reward: Bool[Array, "2 2"]
    factual_equals_baseline_plus_interaction_safety_cost: Bool[Array, "2 2"]
    factual_equals_baseline_plus_interaction_message_charge: Bool[Array, "2 2"]
    shapley_sums_to_memory_total_task_score: Bool[Array, ""]
    shapley_sums_to_memory_total_net_reward: Bool[Array, " 2"]
    shapley_sums_to_memory_total_safety_cost: Bool[Array, " 2"]
    shapley_sums_to_memory_total_message_charge: Bool[Array, " 2"]
    all_identities_hold: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HCCLMemoryCreditEstimandPanel:
    """Named estimands and their exact two-player algebra diagnostics."""

    baseline_context_direct_effect: HCCLMemoryCreditAgentEffects
    factual_context_leave_one_out: HCCLMemoryCreditAgentEffects
    componentwise_interaction: HCCLSignalContrast
    shapley_allocation: HCCLMemoryCreditAgentEffects
    memory_total: HCCLSignalContrast
    algebra: HCCLMemoryCreditAlgebraDiagnostics


def _require_array(
    value: Any,
    *,
    shape: tuple[int, ...],
    label: str,
) -> Array:
    if getattr(value, "shape", None) != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if getattr(value, "dtype", None) != _FLOAT32:
        raise TypeError(f"{label} must have dtype float32")
    return jnp.asarray(value)


def _require_signal_contract(
    value: HCCLCausalCoreTypedSignals,
    *,
    label: str,
) -> None:
    if type(value) is not HCCLCausalCoreTypedSignals:
        raise TypeError(f"{label} must be exact HCCLCausalCoreTypedSignals")
    _require_array(value.task_score, shape=(), label=f"{label}.task_score")
    _require_array(value.net_reward, shape=(_N_AGENTS,), label=f"{label}.net_reward")
    _require_array(value.safety_cost, shape=(_N_AGENTS,), label=f"{label}.safety_cost")
    _require_array(
        value.message_charge,
        shape=(_N_AGENTS,),
        label=f"{label}.message_charge",
    )


def _contains_tracer(value: object) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _require_finite_signal(
    value: HCCLCausalCoreTypedSignals,
    *,
    label: str,
) -> None:
    for field in ("task_score", "net_reward", "safety_cost", "message_charge"):
        array = cast(Array, getattr(value, field))
        if not bool(jnp.all(jnp.isfinite(array))):
            raise ValueError(f"{label}.{field} must be finite")


def _subtract(
    left: HCCLCausalCoreTypedSignals | HCCLSignalContrast,
    right: HCCLCausalCoreTypedSignals | HCCLSignalContrast,
) -> HCCLSignalContrast:
    return HCCLSignalContrast(
        task_score=jnp.subtract(left.task_score, right.task_score),
        net_reward=jnp.subtract(left.net_reward, right.net_reward),
        safety_cost=jnp.subtract(left.safety_cost, right.safety_cost),
        message_charge=jnp.subtract(left.message_charge, right.message_charge),
    )


def _add(
    left: HCCLCausalCoreTypedSignals | HCCLSignalContrast,
    right: HCCLCausalCoreTypedSignals | HCCLSignalContrast,
) -> HCCLSignalContrast:
    return HCCLSignalContrast(
        task_score=jnp.add(left.task_score, right.task_score),
        net_reward=jnp.add(left.net_reward, right.net_reward),
        safety_cost=jnp.add(left.safety_cost, right.safety_cost),
        message_charge=jnp.add(left.message_charge, right.message_charge),
    )


def _agent_effects(
    agent_0: HCCLSignalContrast,
    agent_1: HCCLSignalContrast,
) -> HCCLMemoryCreditAgentEffects:
    return HCCLMemoryCreditAgentEffects(
        shared_task_score=jnp.stack((agent_0.task_score, agent_1.task_score)),
        net_reward=jnp.stack((agent_0.net_reward, agent_1.net_reward)),
        safety_cost=jnp.stack((agent_0.safety_cost, agent_1.safety_cost)),
        message_charge=jnp.stack((agent_0.message_charge, agent_1.message_charge)),
    )


def _repeat_interaction(
    interaction: HCCLSignalContrast,
) -> HCCLMemoryCreditAgentEffects:
    return _agent_effects(interaction, interaction)


def _add_effects(
    left: HCCLMemoryCreditAgentEffects,
    right: HCCLMemoryCreditAgentEffects,
) -> HCCLMemoryCreditAgentEffects:
    return HCCLMemoryCreditAgentEffects(
        shared_task_score=left.shared_task_score + right.shared_task_score,
        net_reward=left.net_reward + right.net_reward,
        safety_cost=left.safety_cost + right.safety_cost,
        message_charge=left.message_charge + right.message_charge,
    )


def _average_effects(
    left: HCCLMemoryCreditAgentEffects,
    right: HCCLMemoryCreditAgentEffects,
) -> HCCLMemoryCreditAgentEffects:
    half = jnp.asarray(0.5, dtype=jnp.float32)
    return HCCLMemoryCreditAgentEffects(
        shared_task_score=half * (left.shared_task_score + right.shared_task_score),
        net_reward=half * (left.net_reward + right.net_reward),
        safety_cost=half * (left.safety_cost + right.safety_cost),
        message_charge=half * (left.message_charge + right.message_charge),
    )


def _float_bits_equal(left: Array, right: Array) -> Bool[Array, ...]:
    return (
        jax.lax.bitcast_convert_type(left, jnp.uint32)
        == jax.lax.bitcast_convert_type(right, jnp.uint32)
    )


def _algebra_diagnostics(
    *,
    baseline: HCCLMemoryCreditAgentEffects,
    factual: HCCLMemoryCreditAgentEffects,
    interaction: HCCLSignalContrast,
    shapley: HCCLMemoryCreditAgentEffects,
    memory_total: HCCLSignalContrast,
) -> HCCLMemoryCreditAlgebraDiagnostics:
    factual_rhs = _add_effects(baseline, _repeat_interaction(interaction))
    factual_task = _float_bits_equal(
        factual.shared_task_score,
        factual_rhs.shared_task_score,
    )
    factual_net = _float_bits_equal(factual.net_reward, factual_rhs.net_reward)
    factual_safety = _float_bits_equal(factual.safety_cost, factual_rhs.safety_cost)
    factual_message = _float_bits_equal(
        factual.message_charge,
        factual_rhs.message_charge,
    )
    shapley_task = _float_bits_equal(
        jnp.sum(shapley.shared_task_score, axis=0),
        memory_total.task_score,
    )
    shapley_net = _float_bits_equal(
        jnp.sum(shapley.net_reward, axis=0),
        memory_total.net_reward,
    )
    shapley_safety = _float_bits_equal(
        jnp.sum(shapley.safety_cost, axis=0),
        memory_total.safety_cost,
    )
    shapley_message = _float_bits_equal(
        jnp.sum(shapley.message_charge, axis=0),
        memory_total.message_charge,
    )
    all_hold = (
        jnp.all(factual_task)
        & jnp.all(factual_net)
        & jnp.all(factual_safety)
        & jnp.all(factual_message)
        & jnp.all(shapley_task)
        & jnp.all(shapley_net)
        & jnp.all(shapley_safety)
        & jnp.all(shapley_message)
    )
    return HCCLMemoryCreditAlgebraDiagnostics(
        factual_equals_baseline_plus_interaction_task_score=factual_task,
        factual_equals_baseline_plus_interaction_net_reward=factual_net,
        factual_equals_baseline_plus_interaction_safety_cost=factual_safety,
        factual_equals_baseline_plus_interaction_message_charge=factual_message,
        shapley_sums_to_memory_total_task_score=shapley_task,
        shapley_sums_to_memory_total_net_reward=shapley_net,
        shapley_sums_to_memory_total_safety_cost=shapley_safety,
        shapley_sums_to_memory_total_message_charge=shapley_message,
        all_identities_hold=all_hold,
    )


def _require_finite_output(panel: HCCLMemoryCreditEstimandPanel) -> None:
    for leaf in jax.tree.leaves(
        (
            panel.baseline_context_direct_effect,
            panel.factual_context_leave_one_out,
            panel.componentwise_interaction,
            panel.shapley_allocation,
            panel.memory_total,
        )
    ):
        if not bool(jnp.all(jnp.isfinite(leaf))):
            raise ValueError("derived HCCL memory-credit estimands must be finite")


def derive_hccl_memory_credit_estimands(
    *,
    mm: HCCLCausalCoreTypedSignals,
    b0m1: HCCLCausalCoreTypedSignals,
    m0b1: HCCLCausalCoreTypedSignals,
    bb: HCCLCausalCoreTypedSignals,
) -> HCCLMemoryCreditEstimandPanel:
    """Derive baseline, factual, interaction, and Shapley memory credit.

    Rows in each per-agent effect matrix are the agent whose memory action is
    credited.  Columns retain the original typed signal's agent component.
    ``shared_task_score`` instead has one entry per credited agent because the
    source task score is a single shared scalar.
    """

    signals = (("mm", mm), ("b0m1", b0m1), ("m0b1", m0b1), ("bb", bb))
    for label, value in signals:
        _require_signal_contract(value, label=label)
    if _contains_tracer(tuple(value for _, value in signals)):
        raise TypeError(
            "HCCL memory-credit estimands are host/eager only because strict finite "
            "input validation cannot run while traced"
        )
    for label, value in signals:
        _require_finite_signal(value, label=label)

    baseline = _agent_effects(_subtract(m0b1, bb), _subtract(b0m1, bb))
    factual = _agent_effects(_subtract(mm, b0m1), _subtract(mm, m0b1))
    interaction = _add(_subtract(_subtract(mm, b0m1), m0b1), bb)
    memory_total = _subtract(mm, bb)
    shapley = _average_effects(baseline, factual)
    algebra = _algebra_diagnostics(
        baseline=baseline,
        factual=factual,
        interaction=interaction,
        shapley=shapley,
        memory_total=memory_total,
    )
    panel = HCCLMemoryCreditEstimandPanel(
        baseline_context_direct_effect=baseline,
        factual_context_leave_one_out=factual,
        componentwise_interaction=interaction,
        shapley_allocation=shapley,
        memory_total=memory_total,
        algebra=algebra,
    )
    _require_finite_output(panel)
    return panel
