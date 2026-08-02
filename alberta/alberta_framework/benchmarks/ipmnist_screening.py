"""Reduced-horizon mechanism-combination screening for the UPGD IPMNIST lane.

Screens optimizer/mechanism combinations that might exceed the reproduced
UPGD-W SOTA on the ICLR-2024 online Input-permuted MNIST protocol
(:mod:`alberta_framework.benchmarks.upgd_ipmnist`). The screening proxy is
the *same* protocol at a reduced horizon (default 60 tasks x 5,000 steps
instead of 200 tasks): because :func:`~alberta_framework.benchmarks.
upgd_ipmnist.build_schedule` folds the task index into per-seed keys, a
60-task run is an exact prefix of the corresponding 200-task run for the
same seed, so the proxy can be validated bit-for-bit against the completed
full-horizon shards in ``outputs/upgd_ipmnist/partials/``.

Screened combinations (all paired against a ``upgd_w_control`` arm run on
identical seeds):

- ``upgd_idbd`` / ``upgd_autostep``: UPGD's utility gate combined with
  per-weight step-size adaptation (IDBD, Meyer error-free variant; Autostep,
  Mahmood et al. 2012). The meta signal is the *gated* loss gradient — the
  update direction UPGD actually applies (perturbation noise excluded).
- ``upgd_cbp`` / ``adamw_cbp``: dormant-unit recycling in the style of
  Continual Backprop (Dohare et al.), adapted to the protocol MLP: per-unit
  utility EMA of ``|activation * dL/d_activation|``, accumulator-driven
  replacement of the lowest-utility mature unit, incoming weights redrawn
  from the protocol's PyTorch-default uniform init (upstream CBP uses a
  sparse init), outgoing weights zeroed, optimizer state reset per unit.
- ``upgd_w_idbd_swift``: the IDBD arm plus SwiftTD's two supervised-mode
  stabilizers (Javed, Sharifnassab & Sutton, RLC 2024; equation forms from
  :mod:`alberta_framework.core.swift_td`): a network-global overshoot bound
  capping ``sum_i alpha_i * z_i^2`` at ``eta``, and persistent step-size
  decay ``log_alpha_i += ln(eps) * z_i^2`` plus meta-trace reset whenever
  the bound triggers.
- ``upgd_w_fade_head``: FADE-style meta-learned per-parameter weight decay
  (Ramesh, Lewandowski & Schmidhuber, arXiv 2604.27063) on the output layer
  only -- ``lambda_i = exp(gamma_i)`` adapted through a forward-mode
  sensitivity trace of the head weights w.r.t. their log decay rates;
  UPGD-W unchanged elsewhere.
- ``upgd_l2init``: decoupled decay pulls toward the *initial* weights
  (L2-Init, Kumar et al.) instead of toward zero.
- ``upgd_ema_norm``: EMA input normalization (equation-parity with
  :class:`~alberta_framework.core.normalizers.EMANormalizer`) in front of
  the protocol MLP.
- ``upgd_w_wclip_*``: UPGD-W followed by per-layer weight clipping to
  ``[-kappa * s_l, +kappa * s_l]`` where ``s_l = 1/sqrt(fan_in)`` is the
  protocol's uniform-init bound (Elsayed, Lan, Lyle & Mahmood, RLC 2024),
  with ``kappa`` in {1, 2} crossed with weight decay in {0.01, 0}.
- ``upgd_w_localgate``: the lean UPGD-W step with the sigmoid utility gate
  normalized by the *per-tensor* utility max instead of the network-global
  max (the paper's local/global distinction).
- ``upgd_w_*`` hyperparameter-neighborhood star around the published
  UPGD-W configuration (sigma, utility decay, weight decay).
- ``guarded_cbp_adam``: AdamW+CBP (the screening leader) plus UPGD-style
  utility *protection only* — Adam's applied per-weight delta is scaled by
  ``1 - guard_scale * gate`` with the gate from the UPGD ``-w*g`` utility
  EMA; no perturbation (CBP supplies regeneration). ``guard_scale=0``
  reduces bit-exactly to ``adamw_cbp`` (pinned).
- ``adamw_cbp_noreset``: ``adamw_cbp`` WITHOUT the per-unit Adam
  moment/count reset at CBP replacement (the leader resets by default) —
  dissects whether optimizer-state freshness at recycle is load-bearing.
  ``cbp_replacement_rate=0`` reduces to ``adamw_control`` (pinned).
- ``upgd_w_sigma0``: lean UPGD-W with ``sigma=0`` — pure utility-gated
  SGD + decoupled decay, no perturbation; the noise draw (~85-90% of the
  UPGD step cost) is skipped entirely. Bit-exact against the control
  factory run at ``noise_std=0`` (pinned).
- ``upgd_alpha_utility``: UPGD-W whose protection signal is per-weight
  step-size relevance — an IDBD ``log_alpha``/trace pair maintained as a
  *passive statistic* on the raw gradient (never applied as a step size);
  the gate is a scale-free squashing of each weight's log-step-size drift
  from init. ``meta_step_size=0`` reduces to the closed-form half-gated
  step (pinned).
- ``adamw_cbp_{r3e5,r3e4,m50,m200}``: axis-aligned mini-star around the
  untuned ``adamw_cbp`` leader (replacement rate 3e-5/3e-4, maturity
  50/200).
- ``adamw_cbp_ema_norm``: the exact ``adamw_cbp`` update behind the exact
  ``upgd_ema_norm`` EMA input normalizer (same decay/eps/state threading) —
  composition of the screening's two orthogonal wins (input conditioning +
  capacity regeneration). ``norm_enabled=0`` skips the normalizer entirely
  and reduces bit-exactly to ``adamw_cbp`` (pinned).
- ``sgd_ema_norm``: the gate ablation closing the ``upgd_ema_norm`` /
  ``upgd_ema_norm_sigma0`` dissection — plain SGD with decoupled weight decay
  (``w <- w * (1 - lr*wd) - lr * grad``) behind the exact ``upgd_ema_norm``
  EMA input normalizer (same decay/eps/state threading); no utility, no gate,
  no noise. Pinned against a hand-computed trajectory, and the normalizer
  states are pinned bitwise against ``upgd_ema_norm``'s on a shared stream.
- ``sigma0_*``: single-axis frontier extensions on the confirmed
  ``upgd_ema_norm_sigma0`` champion (normalize + utility-gated SGD + decay,
  no noise), all built by one factory whose defaults reduce bit-exactly to
  that champion (pinned): normalizer decay {0.99, 0.9999} and epsilon
  {1e-6, 1e-4} stars (``ema_normalize`` already centers with the EMA mean,
  so the statistics themselves are the unexplored axis), stateless
  per-example RMS normalization of both hidden ReLU layers
  (``sigma0_hidden_norm``; no learnable parameters), utility-gate
  temperature ``sigmoid(beta * scaled_utility)`` with beta {0.5, 2}, and
  the per-tensor (local) gate normalization retested under conditioning
  (``sigma0_localgate``; measured -0.0008 on raw inputs).

Everything here is a development screening diagnostic — never promotable
scientific evidence. Benchmark executions happen through the CLI
(``run`` / ``merge`` / ``validate-proxy``), never inside pytest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.benchmarks.upgd_ipmnist import (
    _PLASTICITY_LOSS_FLOOR,
    ADAMW_PROTOCOL_HYPERPARAMETERS,
    UPGD_W_PROTOCOL_HYPERPARAMETERS,
    IPMNISTConfig,
    LeanUPGDState,
    LearnerInitFn,
    LearnerStepFn,
    _make_adamw_learner,
    _make_upgd_w_learner,
    _sorted_param_shapes,
    _split_flat_noise,
    build_schedule,
    cross_entropy_loss,
    default_openml_data_home,
    init_mlp_params,
    lean_upgd_w_update,
    load_mnist_train,
    mlp_logits,
)
from alberta_framework.evaluation.recurring_ipmnist_retention import (
    RecurringIPMNISTPhase,
    RecurringIPMNISTProtocol,
    RecurringIPMNISTRetentionReport,
    RecurringIPMNISTTrace,
    SentinelProbeBinding,
    SentinelProbeSnapshot,
    build_recurring_ipmnist_retention_report,
)

logger = logging.getLogger(__name__)

SHARD_SCHEMA = "alberta.ipmnist_screening.shard.v1"
SUMMARY_SCHEMA = "alberta.ipmnist_screening.summary.v1"
VALIDATION_SCHEMA = "alberta.ipmnist_screening.proxy_validation.v1"

#: Default reduced-horizon proxy: 60 tasks x 5,000 steps. At this horizon the
#: completed 10-seed full runs separate UPGD-W from AdamW by ~+0.022 average
#: online accuracy with every seed ordered correctly.
PROXY_N_TASKS = 60

#: Paired proxy improvement over the UPGD-W control above which a config is
#: flagged as a full-protocol confirmation candidate.
CONFIRMATION_THRESHOLD = 0.005

NONPROMOTING_POLICY: dict[str, object] = {
    "evidence_class": "development_screening_diagnostic",
    "development_only": True,
    "scientific_promotion_allowed": False,
}

_IDBD_LOG_ALPHA_MIN = -10.0
_IDBD_LOG_ALPHA_MAX = 0.0  # alpha <= 1 keeps per-weight decay factors positive
_AUTOSTEP_ALPHA_MIN = 1e-8
_AUTOSTEP_ALPHA_MAX = 1.0

# Metrics returned by every screening step: (accuracy, loss, plasticity).
StepMetrics = tuple[Array, Array, Array]
ScreeningStepFn = Callable[
    [dict[str, Array], Any, Array, Array, Array],
    tuple[dict[str, Array], Any, StepMetrics],
]
#: Pure noise-consuming update ``(params, state, grads, noise, hp)`` used by
#: the pool-noise confirmation path (only lean-UPGD-family arms provide one).
NoiseUpdateFn = Callable[
    [dict[str, Array], Any, dict[str, Array], dict[str, Array], Mapping[str, float]],
    tuple[dict[str, Array], Any],
]
FrozenProbeInputFn = Callable[[Any, Array, Mapping[str, float]], Array]


# =============================================================================
# Shared pieces
# =============================================================================


def _sorted_flat_noise(
    key: Array, params: dict[str, Array], noise_std: float
) -> dict[str, Array]:
    """Draw one flat N(0, sigma^2) vector and slice it per parameter.

    Identical construction (sorted names, one flat draw) to the lean UPGD-W
    learner in :mod:`alberta_framework.benchmarks.upgd_ipmnist`, so a combo
    that degenerates to plain UPGD-W consumes the same noise stream.
    """
    names = sorted(params)
    shapes = [params[name].shape for name in names]
    counts = [int(np.prod(shape)) for shape in shapes]
    flat = jr.normal(key, (sum(counts),), jnp.float32) * noise_std
    chunks = jnp.split(flat, np.cumsum(counts)[:-1])
    return {
        name: chunk.reshape(shape)
        for name, chunk, shape in zip(names, chunks, shapes, strict=True)
    }


def _upgd_utility_and_gate(
    params: dict[str, Array],
    grads: dict[str, Array],
    utility: dict[str, Array],
    count: Array,
    utility_decay: float,
) -> tuple[dict[str, Array], dict[str, Array]]:
    """UPGD utility EMA update + global-max sigmoid gate (lean-step equations)."""
    beta = utility_decay
    new_utility = {
        name: beta * utility[name] + (1.0 - beta) * (-grads[name] * params[name])
        for name in params
    }
    global_max = jnp.max(jnp.stack([jnp.max(new_utility[name]) for name in sorted(params)]))
    bias_correction = 1.0 - jnp.power(
        jnp.asarray(beta, dtype=jnp.float32), count.astype(jnp.float32)
    )
    gate = {
        name: jax.nn.sigmoid((new_utility[name] / bias_correction) / global_max)
        for name in params
    }
    return new_utility, gate


def _forward_with_activations(
    params: dict[str, Array], x: Array
) -> tuple[Array, Array, Array, Array, Array]:
    """Forward pass returning ``(logits, z1, a1, z2, a2)`` for CBP bookkeeping."""
    z1 = x @ params["w1"] + params["b1"]
    a1 = jax.nn.relu(z1)
    z2 = a1 @ params["w2"] + params["b2"]
    a2 = jax.nn.relu(z2)
    logits = a2 @ params["w3"] + params["b3"]
    return logits, z1, a1, z2, a2


def _activation_loss_grads(
    params: dict[str, Array], logits: Array, y: Array, z2: Array
) -> tuple[Array, Array]:
    """Analytic ``(dL/da1, dL/da2)`` for softmax cross-entropy on one example."""
    dlogits = jax.nn.softmax(logits) - jax.nn.one_hot(y, logits.shape[0], dtype=jnp.float32)
    da2 = params["w3"] @ dlogits
    da1 = params["w2"] @ (da2 * (z2 > 0).astype(jnp.float32))
    return da1, da2


def _step_metrics(
    params_after: dict[str, Array], x: Array, y: Array, loss: Array, logits: Array
) -> StepMetrics:
    """Protocol metrics: pre-update accuracy, loss, post-update plasticity."""
    accuracy = (jnp.argmax(logits) == y).astype(jnp.float32)
    loss_after, _ = cross_entropy_loss(params_after, x, y)
    plasticity = jnp.clip(
        1.0 - loss_after / jnp.maximum(loss, _PLASTICITY_LOSS_FLOOR), 0.0, 1.0
    )
    return accuracy, loss, plasticity


def _wrap_grad_learner(
    init_fn: LearnerInitFn, step_fn: LearnerStepFn
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Adapt an ``upgd_ipmnist`` (grads, key)-driven learner to the screening API.

    Mirrors the ``run_ipmnist`` inner-step ordering exactly so control arms
    reproduce the full-horizon lane bit-for-bit.
    """

    def full_step(
        params: dict[str, Array], state: Any, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], Any, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        new_params, new_state = step_fn(params, state, grads, key)
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


# =============================================================================
# (a) UPGD-W + per-weight step-sizes (IDBD / Autostep)
# =============================================================================


@chex.dataclass(frozen=True)
class UPGDIDBDState:
    """UPGD utility EMA/clock plus IDBD per-weight log step-sizes and traces."""

    utility: dict[str, Array]
    step: Array
    log_alpha: dict[str, Array]
    trace: dict[str, Array]


def upgd_idbd_update(
    params: dict[str, Array],
    state: UPGDIDBDState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], UPGDIDBDState]:
    """UPGD-W step with IDBD-style per-weight step-sizes.

    The meta signal is the gated loss gradient ``z = grad * (1 - gate)`` —
    the direction UPGD actually descends (perturbation noise excluded from
    meta-learning). Following the Meyer error-free variant implemented by
    :class:`~alberta_framework.core.optimizers.IDBD`:
    ``log_alpha += meta * z * h`` (old trace), then
    ``h = h * max(0, 1 - alpha * z^2) + alpha * z`` with the new alpha.
    ``log_alpha`` is clipped to ``[-10, 0]`` (alpha <= 1) so the per-weight
    decoupled decay ``1 - alpha * wd`` stays positive.

    With ``meta_step_size = 0`` and ``initial_step_size`` equal to the
    published UPGD-W step size this reduces exactly to the lean UPGD-W step
    (pinned by a unit test).
    """
    wd = hp["weight_decay"]
    meta = hp["meta_step_size"]
    count = state.step + jnp.array(1, dtype=jnp.int32)
    utility, gate = _upgd_utility_and_gate(
        params, grads, state.utility, count, hp["utility_decay"]
    )
    new_params: dict[str, Array] = {}
    new_log_alpha: dict[str, Array] = {}
    new_trace: dict[str, Array] = {}
    for name in params:
        keep = 1.0 - gate[name]
        z = grads[name] * keep
        log_alpha = jnp.clip(
            state.log_alpha[name] + meta * z * state.trace[name],
            _IDBD_LOG_ALPHA_MIN,
            _IDBD_LOG_ALPHA_MAX,
        )
        alpha = jnp.exp(log_alpha)
        new_params[name] = params[name] * (1.0 - alpha * wd) - alpha * (
            (grads[name] + noise[name]) * keep
        )
        trace_decay = jnp.maximum(0.0, 1.0 - alpha * z * z)
        new_log_alpha[name] = log_alpha
        new_trace[name] = state.trace[name] * trace_decay + alpha * z
    return new_params, UPGDIDBDState(  # type: ignore[call-arg]
        utility=utility, step=count, log_alpha=new_log_alpha, trace=new_trace
    )


def upgd_idbd_swift_update(
    params: dict[str, Array],
    state: UPGDIDBDState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], UPGDIDBDState]:
    """:func:`upgd_idbd_update` plus SwiftTD's two supervised-mode stabilizers.

    Same state, same meta signal (``z = grad * (1 - gate)``, the direction
    UPGD actually descends), same IDBD meta/trace equations. In the role of
    SwiftTD's feature ``phi_i`` (:mod:`alberta_framework.core.swift_td`) this
    supervised per-weight arm uses that same ``z_i``:

    - **Overshoot bound**: the network-global correction ratio
      ``tau = sum_i alpha_i * z_i^2`` is capped at ``swift_eta``. When
      ``tau > swift_eta`` every per-weight step this update applies is scaled
      by ``swift_eta / tau`` (weight decay and trace extension included,
      exactly as SwiftTD's ``bound_scale`` scales its whole ``z_delta``).
    - **Persistent step-size decay on trigger**: when the bound fires the
      stored log step-sizes decay by ``ln(swift_eps) * z_i^2`` (proportional
      to each weight's contribution, then re-clipped to the IDBD bounds) and
      the meta-learning traces reset to zero, mirroring the reference decay
      block that zeroes SwiftTD's ``h`` traces.

    With ``swift_eta = inf`` and ``swift_eps = 1`` this reduces exactly to
    :func:`upgd_idbd_update` (pinned by a unit test).
    """
    wd = hp["weight_decay"]
    meta = hp["meta_step_size"]
    eta = hp["swift_eta"]
    log_eps = math.log(hp["swift_eps"])
    count = state.step + jnp.array(1, dtype=jnp.int32)
    utility, gate = _upgd_utility_and_gate(
        params, grads, state.utility, count, hp["utility_decay"]
    )
    z_all: dict[str, Array] = {}
    log_alpha_all: dict[str, Array] = {}
    for name in params:
        z_all[name] = grads[name] * (1.0 - gate[name])
        log_alpha_all[name] = jnp.clip(
            state.log_alpha[name] + meta * z_all[name] * state.trace[name],
            _IDBD_LOG_ALPHA_MIN,
            _IDBD_LOG_ALPHA_MAX,
        )
    alpha_all = {name: jnp.exp(log_alpha_all[name]) for name in params}
    tau = jnp.sum(
        jnp.stack(
            [jnp.sum(alpha_all[name] * z_all[name] * z_all[name]) for name in sorted(params)]
        )
    )
    triggered = tau > eta
    bound_scale = jnp.where(triggered, eta / tau, 1.0)
    new_params: dict[str, Array] = {}
    new_log_alpha: dict[str, Array] = {}
    new_trace: dict[str, Array] = {}
    for name in params:
        keep = 1.0 - gate[name]
        z = z_all[name]
        alpha_eff = bound_scale * alpha_all[name]
        new_params[name] = params[name] * (1.0 - alpha_eff * wd) - alpha_eff * (
            (grads[name] + noise[name]) * keep
        )
        trace = state.trace[name] * jnp.maximum(0.0, 1.0 - alpha_eff * z * z) + alpha_eff * z
        new_trace[name] = jnp.where(triggered, 0.0, trace)
        new_log_alpha[name] = jnp.where(
            triggered,
            jnp.clip(
                log_alpha_all[name] + log_eps * z * z,
                _IDBD_LOG_ALPHA_MIN,
                _IDBD_LOG_ALPHA_MAX,
            ),
            log_alpha_all[name],
        )
    return new_params, UPGDIDBDState(  # type: ignore[call-arg]
        utility=utility, step=count, log_alpha=new_log_alpha, trace=new_trace
    )


#: Pure IDBD-family update ``(params, state, grads, noise, hp)``.
_IDBDUpdateFn = Callable[
    [dict[str, Array], UPGDIDBDState, dict[str, Array], dict[str, Array], Mapping[str, float]],
    tuple[dict[str, Array], UPGDIDBDState],
]


def _make_idbd_family_learner(
    hp: Mapping[str, float], update: _IDBDUpdateFn
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]

    def init_fn(params: dict[str, Array]) -> UPGDIDBDState:
        log_init = math.log(hp["initial_step_size"])
        return UPGDIDBDState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            log_alpha={
                name: jnp.full_like(value, log_init) for name, value in params.items()
            },
            trace={name: jnp.zeros_like(value) for name, value in params.items()},
        )

    def full_step(
        params: dict[str, Array], state: UPGDIDBDState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], UPGDIDBDState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        noise = _sorted_flat_noise(key, params, noise_std)
        new_params, new_state = update(params, state, grads, noise, hp)
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


def _make_upgd_idbd_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    return _make_idbd_family_learner(hp, upgd_idbd_update)


def _make_upgd_idbd_swift_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    return _make_idbd_family_learner(hp, upgd_idbd_swift_update)


@chex.dataclass(frozen=True)
class UPGDAutostepState:
    """UPGD utility EMA/clock plus Autostep step-sizes, traces, normalizers."""

    utility: dict[str, Array]
    step: Array
    alpha: dict[str, Array]
    trace: dict[str, Array]
    normalizer: dict[str, Array]


def upgd_autostep_update(
    params: dict[str, Array],
    state: UPGDAutostepState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], UPGDAutostepState]:
    """UPGD-W step with Autostep per-weight step-size adaptation.

    Mahmood et al. 2012 Table 1 with the error-free meta gradient
    ``z * h`` (``z`` = gated loss gradient), the self-regulated normalizer
    ``v``, and a *network-global* effective-step normalizer
    ``M = max(sum(alpha * z^2), 1)`` across all parameters.
    """
    wd = hp["weight_decay"]
    mu = hp["meta_step_size"]
    tau = hp["tau"]
    count = state.step + jnp.array(1, dtype=jnp.int32)
    utility, gate = _upgd_utility_and_gate(
        params, grads, state.utility, count, hp["utility_decay"]
    )
    z_all = {name: grads[name] * (1.0 - gate[name]) for name in params}
    raw_alpha: dict[str, Array] = {}
    new_normalizer: dict[str, Array] = {}
    for name in params:
        z = z_all[name]
        meta_gradient = z * state.trace[name]
        abs_meta = jnp.abs(meta_gradient)
        v_update = state.normalizer[name] + (1.0 / tau) * state.alpha[name] * z * z * (
            abs_meta - state.normalizer[name]
        )
        v_new = jnp.maximum(abs_meta, v_update)
        safe_v = jnp.maximum(v_new, 1e-38)
        raw_alpha[name] = jnp.where(
            v_new > 0.0,
            state.alpha[name] * jnp.exp(mu * meta_gradient / safe_v),
            state.alpha[name],
        )
        new_normalizer[name] = v_new
    effective = jnp.sum(
        jnp.stack(
            [jnp.sum(raw_alpha[name] * z_all[name] * z_all[name]) for name in sorted(params)]
        )
    )
    m_factor = jnp.maximum(effective, 1.0)
    new_params: dict[str, Array] = {}
    new_alpha: dict[str, Array] = {}
    new_trace: dict[str, Array] = {}
    for name in params:
        keep = 1.0 - gate[name]
        z = z_all[name]
        alpha = jnp.clip(raw_alpha[name] / m_factor, _AUTOSTEP_ALPHA_MIN, _AUTOSTEP_ALPHA_MAX)
        new_params[name] = params[name] * (1.0 - alpha * wd) - alpha * (
            (grads[name] + noise[name]) * keep
        )
        new_alpha[name] = alpha
        new_trace[name] = state.trace[name] * (1.0 - alpha * z * z) + alpha * z
    return new_params, UPGDAutostepState(  # type: ignore[call-arg]
        utility=utility,
        step=count,
        alpha=new_alpha,
        trace=new_trace,
        normalizer=new_normalizer,
    )


def _make_upgd_autostep_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]

    def init_fn(params: dict[str, Array]) -> UPGDAutostepState:
        return UPGDAutostepState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            alpha={
                name: jnp.full_like(value, hp["initial_step_size"])
                for name, value in params.items()
            },
            trace={name: jnp.zeros_like(value) for name, value in params.items()},
            normalizer={name: jnp.zeros_like(value) for name, value in params.items()},
        )

    def full_step(
        params: dict[str, Array], state: UPGDAutostepState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], UPGDAutostepState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        noise = _sorted_flat_noise(key, params, noise_std)
        new_params, new_state = upgd_autostep_update(params, state, grads, noise, hp)
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


# =============================================================================
# (i) UPGD-W + FADE meta-learned per-parameter weight decay on the head
# =============================================================================

#: ``gamma <= 0`` keeps ``lambda = exp(gamma) <= 1`` so the head decay factor
#: ``1 - lambda`` stays in ``[0, 1]`` (no sign-flipping overshoot).
_FADE_GAMMA_MAX = 0.0
_FADE_HEAD_PARAMS = ("w3", "b3")


@chex.dataclass(frozen=True)
class UPGDFadeHeadState:
    """UPGD utility EMA/clock plus FADE log decay rates and sensitivity traces.

    ``gamma``/``fade_trace`` carry entries for the head parameters
    (``w3``/``b3``) only; hidden layers keep the protocol's fixed decay.
    """

    utility: dict[str, Array]
    step: Array
    gamma: dict[str, Array]
    fade_trace: dict[str, Array]


def upgd_w_fade_head_update(
    params: dict[str, Array],
    state: UPGDFadeHeadState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    head_input: Array,
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], UPGDFadeHeadState]:
    """Lean UPGD-W step with FADE meta-learned weight decay on the output layer.

    FADE-style per-parameter weight decay (Ramesh, Lewandowski & Schmidhuber,
    arXiv:2604.27063; equations re-derived, full text not locally cached) on
    ``w3``/``b3`` only. Hidden layers (``w1/b1/w2/b2``) take the unchanged
    lean UPGD-W step with the fixed decoupled decay
    ``1 - step_size * weight_decay``. The head replaces that fixed decay with
    a per-parameter factor ``1 - lambda_i``, ``lambda_i = exp(gamma_i)``,
    meta-learned online:

    - Meta update (old trace first, IDBD convention):
      ``gamma_i += theta_lambda * delta_t * x_i * g_i``, then capped at
      ``gamma_i <= 0`` so ``lambda_i <= 1``. For the softmax cross-entropy
      head the error-times-input product is exactly
      ``delta_t * x_i = -dL/dw_i`` elementwise (``delta_t`` = one-hot target
      minus softmax at the output; ``x_i`` = head input activation ``a2`` for
      ``w3`` and the constant 1 for ``b3``, SwiftTD's bias-feature
      convention), so it is implemented as ``gamma += theta * (-grad) * g``.
    - Sensitivity trace (forward-mode ``g_i ~ d w_i / d gamma_i`` through the
      head update, diagonal/IDBD approximation), with the *new* ``lambda_i``
      and the *pre-update* weight:
      ``g_i <- g_i * max(0, 1 - lambda_i - fade_alpha * x_i^2)
      - lambda_i * w_i``. Both subtractions inside the ``max`` shrink the
      trace, so the contraction factor lies in ``[0, 1]`` (``lambda_i <= 1``)
      and ``|g_i|`` stays bounded by a geometric sum of ``lambda_i * |w_i|``
      -- the stable orientation of the trace recursion.

    Sign-convention reading (chosen so lambda shrinks when decay hurts):
    ``g_i`` accumulates ``-lambda_i * w_i``, i.e. it opposes the sign of a
    decayed weight, while ``delta_t * x_i = -grad_i`` points where descent
    wants the weight to move. When decay hurts (descent wants the weight to
    grow away from zero, ``-grad_i`` aligned with ``w_i``) the product
    ``(-grad_i) * g_i`` is negative and ``gamma_i`` falls (lambda shrinks);
    when decay helps (stale weight the new task's gradient pushes toward
    zero) the product is positive and ``gamma_i`` rises.

    ``fade_alpha`` is FADE's base step-size inside the trace contraction only
    (published 0.005); the applied gradient step keeps the protocol
    ``step_size`` -- UPGD-W's gate, noise, and descent are unchanged on every
    layer. With ``fade_theta_lambda = 0`` and ``fade_gamma0 = -inf``
    (``lambda = 0``) the head reduces exactly to the control update with zero
    head weight decay (pinned by a unit test).
    """
    step_size = hp["step_size"]
    theta = hp["fade_theta_lambda"]
    fade_alpha = hp["fade_alpha"]
    hidden_decay = 1.0 - step_size * hp["weight_decay"]
    count = state.step + jnp.array(1, dtype=jnp.int32)
    utility, gate = _upgd_utility_and_gate(
        params, grads, state.utility, count, hp["utility_decay"]
    )
    head_sq = {
        "w3": (head_input * head_input)[:, None],
        "b3": jnp.ones_like(params["b3"]),
    }
    new_params: dict[str, Array] = {}
    new_gamma: dict[str, Array] = {}
    new_trace: dict[str, Array] = {}
    for name in params:
        descent = step_size * ((grads[name] + noise[name]) * (1.0 - gate[name]))
        if name in _FADE_HEAD_PARAMS:
            gamma = jnp.minimum(
                state.gamma[name] + theta * (-grads[name]) * state.fade_trace[name],
                _FADE_GAMMA_MAX,
            )
            lam = jnp.exp(gamma)
            new_params[name] = params[name] * (1.0 - lam) - descent
            contraction = jnp.maximum(0.0, 1.0 - lam - fade_alpha * head_sq[name])
            new_gamma[name] = gamma
            new_trace[name] = state.fade_trace[name] * contraction - lam * params[name]
        else:
            new_params[name] = params[name] * hidden_decay - descent
    return new_params, UPGDFadeHeadState(  # type: ignore[call-arg]
        utility=utility, step=count, gamma=new_gamma, fade_trace=new_trace
    )


def _make_upgd_w_fade_head_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]
    gamma0 = hp["fade_gamma0"]

    def init_fn(params: dict[str, Array]) -> UPGDFadeHeadState:
        return UPGDFadeHeadState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            gamma={name: jnp.full_like(params[name], gamma0) for name in _FADE_HEAD_PARAMS},
            fade_trace={name: jnp.zeros_like(params[name]) for name in _FADE_HEAD_PARAMS},
        )

    def full_step(
        params: dict[str, Array], state: UPGDFadeHeadState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], UPGDFadeHeadState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        _, _, _, _, a2 = _forward_with_activations(params, x)
        noise = _sorted_flat_noise(key, params, noise_std)
        new_params, new_state = upgd_w_fade_head_update(params, state, grads, noise, a2, hp)
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


# =============================================================================
# (c) UPGD-W + L2-Init (decay toward initial weights)
# =============================================================================


@chex.dataclass(frozen=True)
class UPGDL2InitState:
    """UPGD utility EMA/clock plus a frozen copy of the initial parameters."""

    utility: dict[str, Array]
    step: Array
    init_params: dict[str, Array]


def upgd_l2init_update(
    params: dict[str, Array],
    state: UPGDL2InitState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], UPGDL2InitState]:
    """Lean UPGD-W step whose decoupled decay pulls toward the initial weights."""
    step_size = hp["step_size"]
    wd = hp["weight_decay"]
    count = state.step + jnp.array(1, dtype=jnp.int32)
    utility, gate = _upgd_utility_and_gate(
        params, grads, state.utility, count, hp["utility_decay"]
    )
    new_params = {
        name: params[name]
        - step_size * wd * (params[name] - state.init_params[name])
        - step_size * ((grads[name] + noise[name]) * (1.0 - gate[name]))
        for name in params
    }
    return new_params, UPGDL2InitState(  # type: ignore[call-arg]
        utility=utility, step=count, init_params=state.init_params
    )


def _make_upgd_l2init_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]

    def init_fn(params: dict[str, Array]) -> UPGDL2InitState:
        return UPGDL2InitState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            init_params={name: value for name, value in params.items()},
        )

    def full_step(
        params: dict[str, Array], state: UPGDL2InitState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], UPGDL2InitState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        noise = _sorted_flat_noise(key, params, noise_std)
        new_params, new_state = upgd_l2init_update(params, state, grads, noise, hp)
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


# =============================================================================
# (e) UPGD-W + EMA input normalization
# =============================================================================


@chex.dataclass(frozen=True)
class EMANormState:
    """Inline EMA normalizer state (mean, var, sample count)."""

    mean: Array
    var: Array
    count: Array


def ema_normalize(
    state: EMANormState, observation: Array, decay: float, epsilon: float
) -> tuple[Array, EMANormState]:
    """Equation-parity restatement of ``EMANormalizer.normalize`` (scan-friendly)."""
    new_count = state.count + 1.0
    effective_decay = jnp.minimum(decay, 1.0 - 1.0 / (new_count + 1.0))
    delta = observation - state.mean
    new_mean = state.mean + (1.0 - effective_decay) * delta
    delta2 = observation - new_mean
    new_var = jnp.maximum(
        effective_decay * state.var + (1.0 - effective_decay) * delta * delta2, epsilon
    )
    normalized = (observation - new_mean) / (jnp.sqrt(new_var) + epsilon)
    return normalized, EMANormState(  # type: ignore[call-arg]
        mean=new_mean, var=new_var, count=new_count
    )


@chex.dataclass(frozen=True)
class UPGDNormState:
    """Lean UPGD state plus the EMA input-normalizer state."""

    utility: dict[str, Array]
    step: Array
    norm: EMANormState


def _make_upgd_ema_norm_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]
    decay = hp["norm_decay"]
    epsilon = hp["norm_epsilon"]
    lean_hp = {
        name: hp[name] for name in ("step_size", "utility_decay", "noise_std", "weight_decay")
    }

    def init_fn(params: dict[str, Array]) -> UPGDNormState:
        input_dim = params["w1"].shape[0]
        return UPGDNormState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            norm=EMANormState(  # type: ignore[call-arg]
                mean=jnp.zeros(input_dim, dtype=jnp.float32),
                var=jnp.ones(input_dim, dtype=jnp.float32),
                count=jnp.array(0.0, dtype=jnp.float32),
            ),
        )

    def full_step(
        params: dict[str, Array], state: UPGDNormState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], UPGDNormState, StepMetrics]:
        x_norm, new_norm = ema_normalize(state.norm, x, decay, epsilon)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        noise = _sorted_flat_noise(key, params, noise_std)
        lean_state = LeanUPGDState(  # type: ignore[call-arg]
            utility=state.utility, step=state.step
        )
        new_params, new_lean = lean_upgd_w_update(params, lean_state, grads, noise, lean_hp)
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        return new_params, UPGDNormState(  # type: ignore[call-arg]
            utility=new_lean.utility, step=new_lean.step, norm=new_norm
        ), metrics

    return init_fn, full_step


@chex.dataclass(frozen=True)
class SGDNormState:
    """Just the EMA input-normalizer state (the gate-ablation arm is stateless
    beyond the normalizer: no utility EMA, no step clock)."""

    norm: EMANormState


def _make_sgd_ema_norm_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Plain SGD + decoupled weight decay behind the exact ``upgd_ema_norm``
    EMA input normalizer (same decay/eps/state threading).

    The final dissection of the normalized-UPGD result: ``upgd_ema_norm_sigma0``
    showed the perturbation is not load-bearing under input conditioning, so
    the method there is normalize + utility-GATED SGD + decay. This arm drops
    the gate too — ``w <- w * (1 - lr*wd) - lr * grad`` — no utility, no gate,
    no noise (the RNG key is deliberately unused). Pinned by a hand-computed
    trajectory test; the normalizer path is pinned bitwise against
    ``upgd_ema_norm``'s on a shared stream.
    """
    step_size = hp["step_size"]
    decay_factor = 1.0 - step_size * hp["weight_decay"]
    norm_decay = hp["norm_decay"]
    epsilon = hp["norm_epsilon"]

    def init_fn(params: dict[str, Array]) -> SGDNormState:
        input_dim = params["w1"].shape[0]
        return SGDNormState(  # type: ignore[call-arg]
            norm=EMANormState(  # type: ignore[call-arg]
                mean=jnp.zeros(input_dim, dtype=jnp.float32),
                var=jnp.ones(input_dim, dtype=jnp.float32),
                count=jnp.array(0.0, dtype=jnp.float32),
            ),
        )

    def full_step(
        params: dict[str, Array], state: SGDNormState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], SGDNormState, StepMetrics]:
        del key  # no perturbation: the per-step noise key is unused
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, epsilon)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_norm, y
        )
        new_params = {
            name: params[name] * decay_factor - step_size * grads[name]
            for name in params
        }
        metrics = _step_metrics(new_params, x_norm, y, loss, logits)
        return new_params, SGDNormState(norm=new_norm), metrics  # type: ignore[call-arg]

    return init_fn, full_step


# =============================================================================
# (n) sigma0_* frontier extensions on the normalized sigma0 champion
# =============================================================================


def _hidden_rms_normalize(activation: Array, epsilon: float) -> Array:
    """Stateless per-example RMS normalization of one hidden activation vector.

    ``a / sqrt(mean(a^2) + eps)`` — layer-norm-style conditioning with no
    learnable parameters and no running statistics (the stream-x recipe).
    The epsilon keeps an all-zero ReLU vector (fully dormant layer) exactly
    zero instead of NaN.
    """
    return activation / jnp.sqrt(jnp.mean(activation * activation) + epsilon)


#: Loss callable ``(params, x, y) -> (loss, logits)`` used by the extension
#: factory (protocol MLP or its hidden-RMS-normalized variant).
_ExtLossFn = Callable[[dict[str, Array], Array, Array], tuple[Array, Array]]


def _make_upgd_ema_norm_ext_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Frontier-extension factory on the ``upgd_ema_norm_sigma0`` champion.

    One factory, three orthogonal switches over the normalize + utility-gated
    SGD + decoupled-decay step (each inert at its default):

    - ``hidden_rms`` (default 0): RMS-normalize both hidden ReLU activation
      vectors per example (:func:`_hidden_rms_normalize`,
      ``hidden_rms_epsilon``) inside the forward pass — gradients, utilities,
      and metrics all see the normalized network.
    - ``gate_beta`` (default 1): utility-gate temperature — the sigmoid
      argument (bias-corrected utility over its max) is scaled by beta.
    - ``local_gate`` (default 0): normalize the gate by the per-tensor
      utility max (zero-guarded exactly as :func:`upgd_w_localgate_update`)
      instead of the network-global max.

    With every switch at its default the trajectory is bit-exact against
    ``upgd_ema_norm_sigma0`` (pinned by a unit test): the perturbation term
    is the same explicit zeros the champion's ``noise_std=0`` draw produces,
    without paying for the 282,160-element normal draw, and the RNG key is
    left untouched.  ``noise_std > 0`` keeps the champion's exact noise
    stream (``_sorted_flat_noise``) for completeness.
    """
    noise_std = hp["noise_std"]
    step_size = hp["step_size"]
    utility_decay = hp["utility_decay"]
    param_decay = 1.0 - step_size * hp["weight_decay"]
    norm_decay = hp["norm_decay"]
    norm_epsilon = hp["norm_epsilon"]
    gate_beta = hp.get("gate_beta", 1.0)
    local_gate = hp.get("local_gate", 0.0) != 0.0
    hidden_rms = hp.get("hidden_rms", 0.0) != 0.0
    rms_epsilon = hp.get("hidden_rms_epsilon", 1e-8)

    def _hidden_rms_loss(
        params: dict[str, Array], x: Array, y: Array
    ) -> tuple[Array, Array]:
        z1 = x @ params["w1"] + params["b1"]
        h1 = _hidden_rms_normalize(jax.nn.relu(z1), rms_epsilon)
        z2 = h1 @ params["w2"] + params["b2"]
        h2 = _hidden_rms_normalize(jax.nn.relu(z2), rms_epsilon)
        logits = h2 @ params["w3"] + params["b3"]
        return -jax.nn.log_softmax(logits)[y], logits

    loss_fn: _ExtLossFn = _hidden_rms_loss if hidden_rms else cross_entropy_loss

    def init_fn(params: dict[str, Array]) -> UPGDNormState:
        input_dim = params["w1"].shape[0]
        return UPGDNormState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            norm=EMANormState(  # type: ignore[call-arg]
                mean=jnp.zeros(input_dim, dtype=jnp.float32),
                var=jnp.ones(input_dim, dtype=jnp.float32),
                count=jnp.array(0.0, dtype=jnp.float32),
            ),
        )

    def full_step(
        params: dict[str, Array], state: UPGDNormState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], UPGDNormState, StepMetrics]:
        x_norm, new_norm = ema_normalize(state.norm, x, norm_decay, norm_epsilon)
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x_norm, y
        )
        if noise_std == 0.0:
            # Exactly the zeros the sigma=0 draw produces, minus the draw.
            noise = {name: jnp.zeros_like(value) for name, value in params.items()}
        else:
            noise = _sorted_flat_noise(key, params, noise_std)
        count = state.step + jnp.array(1, dtype=jnp.int32)
        utility = {
            name: utility_decay * state.utility[name]
            + (1.0 - utility_decay) * (-grads[name] * params[name])
            for name in params
        }
        bias_correction = 1.0 - jnp.power(
            jnp.asarray(utility_decay, dtype=jnp.float32), count.astype(jnp.float32)
        )
        global_max = jnp.max(
            jnp.stack([jnp.max(utility[name]) for name in sorted(params)])
        )
        new_params: dict[str, Array] = {}
        for name in params:
            if local_gate:
                local_max = jnp.max(utility[name])
                divisor = jnp.where(local_max == 0.0, 1.0, local_max)
            else:
                divisor = global_max
            scaled = (utility[name] / bias_correction) / divisor
            if gate_beta != 1.0:
                scaled = gate_beta * scaled
            gate = jax.nn.sigmoid(scaled)
            new_params[name] = params[name] * param_decay - step_size * (
                (grads[name] + noise[name]) * (1.0 - gate)
            )
        accuracy = (jnp.argmax(logits) == y).astype(jnp.float32)
        loss_after, _ = loss_fn(new_params, x_norm, y)
        plasticity = jnp.clip(
            1.0 - loss_after / jnp.maximum(loss, _PLASTICITY_LOSS_FLOOR), 0.0, 1.0
        )
        return new_params, UPGDNormState(  # type: ignore[call-arg]
            utility=utility, step=count, norm=new_norm
        ), (accuracy, loss, plasticity)

    return init_fn, full_step


# =============================================================================
# (f) UPGD-W + per-layer weight clipping (Elsayed, Lan, Lyle & Mahmood, 2024)
# =============================================================================


def _wclip_bound(params: dict[str, Array], name: str, kappa: float) -> float:
    """Clipping bound ``kappa * s_l`` for parameter ``name``.

    ``s_l = 1/sqrt(fan_in)`` is the protocol's PyTorch-default uniform init
    bound (:func:`~alberta_framework.benchmarks.upgd_ipmnist.init_mlp_params`
    draws both ``w{l}`` and ``b{l}`` from ``U(-s_l, s_l)``); the paper clips
    weights and biases of layer ``l`` to ``[-kappa * s_l, +kappa * s_l]``.
    """
    fan_in = params[f"w{name[1:]}"].shape[0]
    return kappa / math.sqrt(fan_in)


def upgd_w_wclip_update(
    params: dict[str, Array],
    state: LeanUPGDState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], LeanUPGDState]:
    """Lean UPGD-W step followed by per-layer weight clipping.

    Algorithm 1 of Elsayed et al. (RLC 2024): after the optimizer update,
    every weight and bias of layer ``l`` is clipped to
    ``[-kappa * s_l, +kappa * s_l]`` with ``s_l`` the uniform-init bound.
    With ``clip_kappa = inf`` the clip is a no-op and this reduces exactly
    to the lean UPGD-W step (pinned by a unit test).
    """
    kappa = hp["clip_kappa"]
    new_params, new_state = lean_upgd_w_update(params, state, grads, noise, dict(hp))
    clipped = {
        name: jnp.clip(
            new_params[name],
            -_wclip_bound(params, name, kappa),
            _wclip_bound(params, name, kappa),
        )
        for name in new_params
    }
    return clipped, new_state


def _make_upgd_w_wclip_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]

    def init_fn(params: dict[str, Array]) -> LeanUPGDState:
        return LeanUPGDState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )

    def full_step(
        params: dict[str, Array], state: LeanUPGDState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], LeanUPGDState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        noise = _sorted_flat_noise(key, params, noise_std)
        new_params, new_state = upgd_w_wclip_update(params, state, grads, noise, hp)
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


# =============================================================================
# (h) UPGD-W with per-tensor (local) utility-gate normalization
# =============================================================================


def upgd_w_localgate_update(
    params: dict[str, Array],
    state: LeanUPGDState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], LeanUPGDState]:
    """Lean UPGD-W step with the sigmoid gate normalized per parameter tensor.

    Identical to :func:`~alberta_framework.benchmarks.upgd_ipmnist.
    lean_upgd_w_update` except the utility scaling before the sigmoid divides
    by ``max(utility[name])`` of the *same tensor* rather than the
    network-global maximum. With a single parameter tensor the two coincide
    exactly (pinned by a unit test).

    Unlike the network-global maximum, a per-tensor utility max can be
    *exactly zero* (a tensor whose gradients are all zero, e.g. fully dead
    units); the global equation would then produce ``0/0``. In that case the
    divisor is replaced by 1, which yields the same ``sigmoid(0) = 0.5``
    gates the global rule assigns to zero utilities.
    """
    beta = hp["utility_decay"]
    step_size = hp["step_size"]
    decay = 1.0 - step_size * hp["weight_decay"]
    count = state.step + jnp.array(1, dtype=jnp.int32)
    utility = {
        name: beta * state.utility[name] + (1.0 - beta) * (-grads[name] * params[name])
        for name in params
    }
    bias_correction = 1.0 - jnp.power(
        jnp.asarray(beta, dtype=jnp.float32), count.astype(jnp.float32)
    )
    new_params = {}
    for name in params:
        local_max = jnp.max(utility[name])
        safe_max = jnp.where(local_max == 0.0, 1.0, local_max)
        gate = jax.nn.sigmoid((utility[name] / bias_correction) / safe_max)
        new_params[name] = params[name] * decay - step_size * (
            (grads[name] + noise[name]) * (1.0 - gate)
        )
    return new_params, LeanUPGDState(utility=utility, step=count)  # type: ignore[call-arg]


def _make_upgd_w_localgate_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]

    def init_fn(params: dict[str, Array]) -> LeanUPGDState:
        return LeanUPGDState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )

    def full_step(
        params: dict[str, Array], state: LeanUPGDState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], LeanUPGDState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        noise = _sorted_flat_noise(key, params, noise_std)
        new_params, new_state = upgd_w_localgate_update(params, state, grads, noise, hp)
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


# =============================================================================
# (b)/(g) CBP-style dormant-unit recycling on UPGD-W and AdamW
# =============================================================================


@chex.dataclass(frozen=True)
class CBPState:
    """Per-unit recycling state for the two hidden layers of the protocol MLP."""

    util1: Array
    util2: Array
    age1: Array
    age2: Array
    accumulator: Array  # shape (2,)


def _init_cbp_state(config_hidden1: int, config_hidden2: int) -> CBPState:
    return CBPState(  # type: ignore[call-arg]
        util1=jnp.zeros(config_hidden1, dtype=jnp.float32),
        util2=jnp.zeros(config_hidden2, dtype=jnp.float32),
        age1=jnp.zeros(config_hidden1, dtype=jnp.int32),
        age2=jnp.zeros(config_hidden2, dtype=jnp.int32),
        accumulator=jnp.zeros(2, dtype=jnp.float32),
    )


@dataclass(frozen=True)
class _CBPLayerRefs:
    """Static wiring of one hidden layer inside the params dict."""

    in_weight: str
    in_bias: str
    out_weight: str


_CBP_LAYERS = (
    _CBPLayerRefs(in_weight="w1", in_bias="b1", out_weight="w2"),
    _CBPLayerRefs(in_weight="w2", in_bias="b2", out_weight="w3"),
)


def cbp_maybe_replace_layer(
    params: dict[str, Array],
    opt_arrays: dict[str, Array] | None,
    utility: Array,
    age: Array,
    accumulator: Array,
    layer: _CBPLayerRefs,
    key: Array,
    replacement_rate: float,
    maturity_threshold: int,
) -> tuple[dict[str, Array], dict[str, Array] | None, Array, Array, Array]:
    """Accumulate the replacement budget and recycle at most one unit.

    ``opt_arrays`` maps parameter names to *stacked* per-element optimizer
    state arrays of shape ``(k, *param_shape)``; the recycled unit's slices
    are reset to zero. Incoming weights are redrawn from the protocol's
    PyTorch-default uniform init; the incoming bias, outgoing weights,
    utility, and age reset to zero.
    """
    n_units = utility.shape[0]
    new_accumulator = accumulator + replacement_rate * n_units
    mature = age >= jnp.asarray(maturity_threshold, dtype=age.dtype)
    fire = jnp.logical_and(new_accumulator >= 1.0, jnp.any(mature))
    masked = jnp.where(mature, utility, jnp.inf)
    idx = jnp.argmin(masked).astype(jnp.int32)

    w_in = params[layer.in_weight]
    fan_in = w_in.shape[0]
    bound = 1.0 / math.sqrt(fan_in)
    fresh_col = jr.uniform(key, (fan_in,), jnp.float32, -bound, bound)
    new_params = dict(params)
    new_params[layer.in_weight] = w_in.at[:, idx].set(
        jnp.where(fire, fresh_col, w_in[:, idx])
    )
    b_in = params[layer.in_bias]
    new_params[layer.in_bias] = b_in.at[idx].set(jnp.where(fire, 0.0, b_in[idx]))
    w_out = params[layer.out_weight]
    new_params[layer.out_weight] = w_out.at[idx, :].set(
        jnp.where(fire, jnp.zeros(w_out.shape[1], dtype=w_out.dtype), w_out[idx, :])
    )

    new_opt_arrays = opt_arrays
    if opt_arrays is not None:
        new_opt_arrays = dict(opt_arrays)
        stack_in = opt_arrays[layer.in_weight]
        new_opt_arrays[layer.in_weight] = stack_in.at[:, :, idx].set(
            jnp.where(fire, jnp.zeros_like(stack_in[:, :, idx]), stack_in[:, :, idx])
        )
        stack_bias = opt_arrays[layer.in_bias]
        new_opt_arrays[layer.in_bias] = stack_bias.at[:, idx].set(
            jnp.where(fire, jnp.zeros_like(stack_bias[:, idx]), stack_bias[:, idx])
        )
        stack_out = opt_arrays[layer.out_weight]
        new_opt_arrays[layer.out_weight] = stack_out.at[:, idx, :].set(
            jnp.where(fire, jnp.zeros_like(stack_out[:, idx, :]), stack_out[:, idx, :])
        )

    new_utility = utility.at[idx].set(jnp.where(fire, 0.0, utility[idx]))
    new_age = age.at[idx].set(jnp.where(fire, jnp.int32(0), age[idx]))
    new_accumulator = jnp.where(fire, new_accumulator - 1.0, new_accumulator)
    return new_params, new_opt_arrays, new_utility, new_age, new_accumulator


def _cbp_update(
    params: dict[str, Array],
    opt_arrays: dict[str, Array] | None,
    cbp: CBPState,
    a1: Array,
    da1: Array,
    a2: Array,
    da2: Array,
    key: Array,
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], dict[str, Array] | None, CBPState]:
    """Utility EMA + age update, then at most one replacement per layer."""
    decay = hp["cbp_decay_rate"]
    util1 = decay * cbp.util1 + (1.0 - decay) * jnp.abs(a1 * da1)
    util2 = decay * cbp.util2 + (1.0 - decay) * jnp.abs(a2 * da2)
    age1 = cbp.age1 + 1
    age2 = cbp.age2 + 1
    key1, key2 = jr.split(key)
    maturity = int(hp["cbp_maturity_threshold"])
    rate = hp["cbp_replacement_rate"]
    params, opt_arrays, util1, age1, acc0 = cbp_maybe_replace_layer(
        params, opt_arrays, util1, age1, cbp.accumulator[0], _CBP_LAYERS[0], key1, rate, maturity
    )
    params, opt_arrays, util2, age2, acc1 = cbp_maybe_replace_layer(
        params, opt_arrays, util2, age2, cbp.accumulator[1], _CBP_LAYERS[1], key2, rate, maturity
    )
    new_cbp = CBPState(  # type: ignore[call-arg]
        util1=util1,
        util2=util2,
        age1=age1,
        age2=age2,
        accumulator=jnp.stack([acc0, acc1]),
    )
    return params, opt_arrays, new_cbp


@chex.dataclass(frozen=True)
class UPGDCBPState:
    """Lean UPGD state plus CBP recycling state."""

    utility: dict[str, Array]
    step: Array
    cbp: CBPState


def _make_upgd_cbp_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]
    lean_hp = {
        name: hp[name] for name in ("step_size", "utility_decay", "noise_std", "weight_decay")
    }

    def init_fn(params: dict[str, Array]) -> UPGDCBPState:
        return UPGDCBPState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            cbp=_init_cbp_state(params["w1"].shape[1], params["w2"].shape[1]),
        )

    def full_step(
        params: dict[str, Array], state: UPGDCBPState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], UPGDCBPState, StepMetrics]:
        key_noise, key_cbp = jr.split(key)
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        _, _, a1, z2, a2 = _forward_with_activations(params, x)
        da1, da2 = _activation_loss_grads(params, logits, y, z2)
        noise = _sorted_flat_noise(key_noise, params, noise_std)
        lean_state = LeanUPGDState(  # type: ignore[call-arg]
            utility=state.utility, step=state.step
        )
        new_params, new_lean = lean_upgd_w_update(params, lean_state, grads, noise, lean_hp)
        opt_arrays = {name: new_lean.utility[name][None, ...] for name in new_params}
        new_params, opt_arrays, new_cbp = _cbp_update(
            new_params, opt_arrays, state.cbp, a1, da1, a2, da2, key_cbp, hp
        )
        assert opt_arrays is not None
        new_utility = {name: opt_arrays[name][0] for name in new_params}
        metrics = _step_metrics(new_params, x, y, loss, logits)
        return new_params, UPGDCBPState(  # type: ignore[call-arg]
            utility=new_utility, step=new_lean.step, cbp=new_cbp
        ), metrics

    return init_fn, full_step


@chex.dataclass(frozen=True)
class AdamCBPState:
    """Per-element Adam moments/counts plus CBP recycling state."""

    m: dict[str, Array]
    v: dict[str, Array]
    count: dict[str, Array]
    cbp: CBPState


def adam_elem_step(
    param: Array,
    m: Array,
    v: Array,
    count: Array,
    grad: Array,
    hp: Mapping[str, float],
) -> tuple[Array, Array, Array, Array]:
    """Adam *delta* with per-element bias-correction counts (not applied).

    Returns ``(step, new_m, new_v, new_count)`` so gated variants can scale
    the applied delta without touching the moment statistics
    (:func:`guarded_adam_update`); :func:`adam_elem_update` applies it as
    ``param - step``.
    """
    new_count = count + 1.0
    new_m = hp["beta1"] * m + (1.0 - hp["beta1"]) * grad
    new_v = hp["beta2"] * v + (1.0 - hp["beta2"]) * grad * grad
    m_hat = new_m / (1.0 - jnp.power(jnp.float32(hp["beta1"]), new_count))
    v_hat = new_v / (1.0 - jnp.power(jnp.float32(hp["beta2"]), new_count))
    step = hp["step_size"] * m_hat / (jnp.sqrt(v_hat) + hp["eps"])
    if hp["weight_decay"] != 0.0:
        step = step + hp["step_size"] * hp["weight_decay"] * param
    return step, new_m, new_v, new_count


def adam_elem_update(
    param: Array,
    m: Array,
    v: Array,
    count: Array,
    grad: Array,
    hp: Mapping[str, float],
) -> tuple[Array, Array, Array, Array]:
    """Adam step with per-element bias-correction counts.

    Matches ``baseline_optimizers.Adam.update_from_gradient`` exactly when
    every element shares the same count (pinned by a unit test); per-element
    counts let CBP restart bias correction for recycled units only.
    """
    step, new_m, new_v, new_count = adam_elem_step(param, m, v, count, grad, hp)
    return param - step, new_m, new_v, new_count


def _make_adamw_cbp_learner(
    hp: Mapping[str, float],
    *,
    reset_recycled_optimizer: bool = True,
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """AdamW+CBP; ``reset_recycled_optimizer=False`` keeps stale per-unit
    Adam moments/counts across CBP replacements (the ``adamw_cbp_noreset``
    dissection arm)."""

    def init_fn(params: dict[str, Array]) -> AdamCBPState:
        zeros = {name: jnp.zeros_like(value) for name, value in params.items()}
        return AdamCBPState(  # type: ignore[call-arg]
            m=dict(zeros),
            v=dict(zeros),
            count={name: jnp.zeros_like(value) for name, value in params.items()},
            cbp=_init_cbp_state(params["w1"].shape[1], params["w2"].shape[1]),
        )

    def full_step(
        params: dict[str, Array], state: AdamCBPState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], AdamCBPState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        _, _, a1, z2, a2 = _forward_with_activations(params, x)
        da1, da2 = _activation_loss_grads(params, logits, y, z2)
        new_params: dict[str, Array] = {}
        new_m: dict[str, Array] = {}
        new_v: dict[str, Array] = {}
        new_count: dict[str, Array] = {}
        for name, value in params.items():
            new_params[name], new_m[name], new_v[name], new_count[name] = adam_elem_update(
                value, state.m[name], state.v[name], state.count[name], grads[name], hp
            )
        if reset_recycled_optimizer:
            opt_arrays: dict[str, Array] | None = {
                name: jnp.stack([new_m[name], new_v[name], new_count[name]])
                for name in new_params
            }
            new_params, opt_arrays, new_cbp = _cbp_update(
                new_params, opt_arrays, state.cbp, a1, da1, a2, da2, key, hp
            )
            assert opt_arrays is not None
            new_m = {name: opt_arrays[name][0] for name in new_params}
            new_v = {name: opt_arrays[name][1] for name in new_params}
            new_count = {name: opt_arrays[name][2] for name in new_params}
        else:
            new_params, _, new_cbp = _cbp_update(
                new_params, None, state.cbp, a1, da1, a2, da2, key, hp
            )
        metrics = _step_metrics(new_params, x, y, loss, logits)
        return new_params, AdamCBPState(  # type: ignore[call-arg]
            m=new_m, v=new_v, count=new_count, cbp=new_cbp
        ), metrics

    return init_fn, full_step


def _make_adamw_cbp_noreset_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    return _make_adamw_cbp_learner(hp, reset_recycled_optimizer=False)


# =============================================================================
# (m) Composition: AdamW+CBP behind EMA input normalization
# =============================================================================


@chex.dataclass(frozen=True)
class AdamCBPNormState:
    """AdamW+CBP state plus the EMA input-normalizer state."""

    m: dict[str, Array]
    v: dict[str, Array]
    count: dict[str, Array]
    cbp: CBPState
    norm: EMANormState


def _make_adamw_cbp_ema_norm_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """The exact ``adamw_cbp`` update behind ``upgd_ema_norm``'s normalizer.

    The EMA input-normalization step (:func:`ema_normalize` equations,
    ``norm_decay``/``norm_epsilon``, per-step state threading) is identical
    to ``upgd_ema_norm``'s (pinned by unit tests); everything downstream —
    gradients, activations for CBP utility, the per-element AdamW step, and
    the recycling with optimizer-state reset — is the ``adamw_cbp`` step run
    on the normalized input. With ``norm_enabled = 0`` the normalizer is
    skipped entirely (state untouched) and the arm reduces bit-exactly to
    ``adamw_cbp`` (pinned by a unit test).
    """
    decay = hp["norm_decay"]
    epsilon = hp["norm_epsilon"]
    normalize = hp.get("norm_enabled", 1.0) != 0.0

    def init_fn(params: dict[str, Array]) -> AdamCBPNormState:
        zeros = {name: jnp.zeros_like(value) for name, value in params.items()}
        input_dim = params["w1"].shape[0]
        return AdamCBPNormState(  # type: ignore[call-arg]
            m=dict(zeros),
            v=dict(zeros),
            count={name: jnp.zeros_like(value) for name, value in params.items()},
            cbp=_init_cbp_state(params["w1"].shape[1], params["w2"].shape[1]),
            norm=EMANormState(  # type: ignore[call-arg]
                mean=jnp.zeros(input_dim, dtype=jnp.float32),
                var=jnp.ones(input_dim, dtype=jnp.float32),
                count=jnp.array(0.0, dtype=jnp.float32),
            ),
        )

    def full_step(
        params: dict[str, Array], state: AdamCBPNormState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], AdamCBPNormState, StepMetrics]:
        if normalize:
            x_in, new_norm = ema_normalize(state.norm, x, decay, epsilon)
        else:
            x_in, new_norm = x, state.norm
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x_in, y
        )
        _, _, a1, z2, a2 = _forward_with_activations(params, x_in)
        da1, da2 = _activation_loss_grads(params, logits, y, z2)
        new_params: dict[str, Array] = {}
        new_m: dict[str, Array] = {}
        new_v: dict[str, Array] = {}
        new_count: dict[str, Array] = {}
        for name, value in params.items():
            new_params[name], new_m[name], new_v[name], new_count[name] = adam_elem_update(
                value, state.m[name], state.v[name], state.count[name], grads[name], hp
            )
        opt_arrays: dict[str, Array] | None = {
            name: jnp.stack([new_m[name], new_v[name], new_count[name]])
            for name in new_params
        }
        new_params, opt_arrays, new_cbp = _cbp_update(
            new_params, opt_arrays, state.cbp, a1, da1, a2, da2, key, hp
        )
        assert opt_arrays is not None
        metrics = _step_metrics(new_params, x_in, y, loss, logits)
        return new_params, AdamCBPNormState(  # type: ignore[call-arg]
            m={name: opt_arrays[name][0] for name in new_params},
            v={name: opt_arrays[name][1] for name in new_params},
            count={name: opt_arrays[name][2] for name in new_params},
            cbp=new_cbp,
            norm=new_norm,
        ), metrics

    return init_fn, full_step


# =============================================================================
# (j) Guarded AdamW+CBP: utility protection on Adam's delta, CBP regeneration
# =============================================================================


@chex.dataclass(frozen=True)
class GuardedAdamCBPState:
    """Per-element Adam moments/counts, UPGD utility EMA + clock, CBP state."""

    m: dict[str, Array]
    v: dict[str, Array]
    count: dict[str, Array]
    utility: dict[str, Array]
    step: Array
    cbp: CBPState


def guarded_adam_update(
    params: dict[str, Array],
    m: dict[str, Array],
    v: dict[str, Array],
    count: dict[str, Array],
    grads: dict[str, Array],
    gate: dict[str, Array],
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array], dict[str, Array]]:
    """Adam step whose *applied* delta is scaled by ``1 - guard_scale * gate``.

    Protection only: the moment statistics see the raw gradients (exactly as
    UPGD's gate scales the applied update, not the utility bookkeeping), and
    there is no perturbation term. With ``guard_scale = 0`` the gating is
    skipped entirely and every parameter takes the plain
    :func:`adam_elem_step` delta, so the ``guarded_cbp_adam`` arm reduces
    bit-exactly to ``adamw_cbp`` (pinned by a unit test).
    """
    guard = hp["guard_scale"]
    new_params: dict[str, Array] = {}
    new_m: dict[str, Array] = {}
    new_v: dict[str, Array] = {}
    new_count: dict[str, Array] = {}
    for name in params:
        step, new_m[name], new_v[name], new_count[name] = adam_elem_step(
            params[name], m[name], v[name], count[name], grads[name], hp
        )
        if guard == 0.0:
            new_params[name] = params[name] - step
        else:
            new_params[name] = params[name] - step * (1.0 - guard * gate[name])
    return new_params, new_m, new_v, new_count


def _make_guarded_cbp_adam_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    def init_fn(params: dict[str, Array]) -> GuardedAdamCBPState:
        zeros = {name: jnp.zeros_like(value) for name, value in params.items()}
        return GuardedAdamCBPState(  # type: ignore[call-arg]
            m=dict(zeros),
            v=dict(zeros),
            count={name: jnp.zeros_like(value) for name, value in params.items()},
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            cbp=_init_cbp_state(params["w1"].shape[1], params["w2"].shape[1]),
        )

    def full_step(
        params: dict[str, Array],
        state: GuardedAdamCBPState,
        x: Array,
        y: Array,
        key: Array,
    ) -> tuple[dict[str, Array], GuardedAdamCBPState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        _, _, a1, z2, a2 = _forward_with_activations(params, x)
        da1, da2 = _activation_loss_grads(params, logits, y, z2)
        clock = state.step + jnp.array(1, dtype=jnp.int32)
        utility, gate = _upgd_utility_and_gate(
            params, grads, state.utility, clock, hp["utility_decay"]
        )
        new_params, new_m, new_v, new_count = guarded_adam_update(
            params, state.m, state.v, state.count, grads, gate, hp
        )
        # Recycled units also reset their guard utility (row 3): fresh units
        # restart at the neutral sigmoid(0) = 0.5 gate.
        opt_arrays: dict[str, Array] | None = {
            name: jnp.stack(
                [new_m[name], new_v[name], new_count[name], utility[name]]
            )
            for name in new_params
        }
        new_params, opt_arrays, new_cbp = _cbp_update(
            new_params, opt_arrays, state.cbp, a1, da1, a2, da2, key, hp
        )
        assert opt_arrays is not None
        metrics = _step_metrics(new_params, x, y, loss, logits)
        return new_params, GuardedAdamCBPState(  # type: ignore[call-arg]
            m={name: opt_arrays[name][0] for name in new_params},
            v={name: opt_arrays[name][1] for name in new_params},
            count={name: opt_arrays[name][2] for name in new_params},
            utility={name: opt_arrays[name][3] for name in new_params},
            step=clock,
            cbp=new_cbp,
        ), metrics

    return init_fn, full_step


# =============================================================================
# (k) Perturbation dissection: lean UPGD-W with sigma = 0
# =============================================================================


def _make_upgd_w_sigma0_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Zero-noise lean UPGD-W: utility gate + decoupled decay, no perturbation.

    Skips the per-step 282,160-element normal draw entirely (~85-90% of the
    UPGD-W step cost) instead of drawing and scaling by zero; the per-step
    RNG chain (``key, step_key = split(key)``) is untouched, so the
    trajectory is bit-exact against the control factory run with
    ``noise_std = 0`` (pinned by a unit test).
    """
    if hp["noise_std"] != 0.0:
        raise ValueError(
            f"upgd_w_sigma0 requires noise_std=0, got {hp['noise_std']!r}"
        )

    def init_fn(params: dict[str, Array]) -> LeanUPGDState:
        return LeanUPGDState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )

    def full_step(
        params: dict[str, Array], state: LeanUPGDState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], LeanUPGDState, StepMetrics]:
        del key  # no perturbation: the step consumes no randomness
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        zeros = {name: jnp.zeros_like(value) for name, value in params.items()}
        new_params, new_state = lean_upgd_w_update(params, state, grads, zeros, dict(hp))
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


# =============================================================================
# (l) UPGD-W gated by passive IDBD step-size relevance instead of -w*g utility
# =============================================================================


@chex.dataclass(frozen=True)
class UPGDAlphaGateState:
    """Clock plus the passive IDBD statistics that drive the protection gate."""

    step: Array
    log_alpha: dict[str, Array]
    trace: dict[str, Array]


def upgd_alpha_utility_update(
    params: dict[str, Array],
    state: UPGDAlphaGateState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    hp: Mapping[str, float],
) -> tuple[dict[str, Array], UPGDAlphaGateState]:
    """UPGD-W step whose protection signal is per-weight step-size relevance.

    An IDBD ``log_alpha``/trace pair (Meyer error-free equations, exactly as
    :func:`upgd_idbd_update`) is maintained as a *passive statistic* on the
    raw loss gradient — it is never applied as a step size; the applied step
    keeps the protocol's fixed ``step_size``, decoupled decay, and
    perturbation. Protection instead reads each weight's log-step-size drift
    from its initial value ``la0 = ln(initial_step_size)``:

    - ``log_alpha_i += meta * g_i * h_i`` (old trace), clipped to
      ``[-10, 0]``; ``h_i = h_i * max(0, 1 - alpha_i * g_i^2) + alpha_i * g_i``.
    - ``s_i = log_alpha_i - la0``; ``gate_i = sigmoid(s_i / max_j |s_j|)``
      (network-global normalizer, mirroring UPGD's global-max gate; when all
      drifts are zero the gate is exactly 0.5). The normalization is
      scale-free — only the *ordering and relative size* of drifts matters,
      the rank-like reading of "alpha as relevance".
    - ``w_i' = w_i * (1 - lr*wd) - lr * (g_i + xi_i) * (1 - gate_i)``.

    Weights whose gradients correlate over time (consistent learners) grow
    ``log_alpha`` and are protected; weights whose gradients decorrelate
    (e.g. the input layer right after a permutation switch) *shed* protection
    because sign-alternating meta-gradients drive ``log_alpha`` down. With
    ``meta_step_size = 0`` every drift stays zero and the update reduces
    bit-exactly to the closed-form half-gated step (pinned by a unit test).
    """
    step_size = hp["step_size"]
    decay = 1.0 - step_size * hp["weight_decay"]
    meta = hp["meta_step_size"]
    la0 = math.log(hp["initial_step_size"])
    count = state.step + jnp.array(1, dtype=jnp.int32)
    new_log_alpha: dict[str, Array] = {}
    new_trace: dict[str, Array] = {}
    for name in params:
        g = grads[name]
        la = jnp.clip(
            state.log_alpha[name] + meta * g * state.trace[name],
            _IDBD_LOG_ALPHA_MIN,
            _IDBD_LOG_ALPHA_MAX,
        )
        alpha = jnp.exp(la)
        new_log_alpha[name] = la
        new_trace[name] = state.trace[name] * jnp.maximum(0.0, 1.0 - alpha * g * g) + alpha * g
    drift = {name: new_log_alpha[name] - la0 for name in params}
    drift_max = jnp.max(
        jnp.stack([jnp.max(jnp.abs(drift[name])) for name in sorted(params)])
    )
    safe_max = jnp.where(drift_max > 0.0, drift_max, 1.0)
    new_params: dict[str, Array] = {}
    for name in params:
        gate = jax.nn.sigmoid(
            jnp.where(drift_max > 0.0, drift[name] / safe_max, 0.0)
        )
        new_params[name] = params[name] * decay - step_size * (
            (grads[name] + noise[name]) * (1.0 - gate)
        )
    return new_params, UPGDAlphaGateState(  # type: ignore[call-arg]
        step=count, log_alpha=new_log_alpha, trace=new_trace
    )


def _make_upgd_alpha_utility_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    noise_std = hp["noise_std"]
    la0 = math.log(hp["initial_step_size"])

    def init_fn(params: dict[str, Array]) -> UPGDAlphaGateState:
        return UPGDAlphaGateState(  # type: ignore[call-arg]
            step=jnp.array(0, dtype=jnp.int32),
            log_alpha={
                name: jnp.full_like(value, la0) for name, value in params.items()
            },
            trace={name: jnp.zeros_like(value) for name, value in params.items()},
        )

    def full_step(
        params: dict[str, Array], state: UPGDAlphaGateState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], UPGDAlphaGateState, StepMetrics]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        noise = _sorted_flat_noise(key, params, noise_std)
        new_params, new_state = upgd_alpha_utility_update(params, state, grads, noise, hp)
        return new_params, new_state, _step_metrics(new_params, x, y, loss, logits)

    return init_fn, full_step


# =============================================================================
# Config registry
# =============================================================================


def _raw_frozen_probe_input(
    state: Any, observation: Array, hyperparameters: Mapping[str, float]
) -> Array:
    """Return the fixed protocol input for learners without preprocessing."""
    del state, hyperparameters
    return observation


def _ema_frozen_probe_input(
    state: Any, observation: Array, hyperparameters: Mapping[str, float]
) -> Array:
    """Apply an EMA learner's current statistics without updating them.

    Online normalized arms update their EMA before predicting each training
    example.  A sentinel probe must be non-learning, so it uses the frozen
    checkpoint statistics.  The normalizer state is part of the checkpoint
    hash and the fixed pixel-permuted sentinel input is separately bound by
    :func:`ipmnist_sentinel_set_sha256`.
    """
    if hyperparameters.get("norm_enabled", 1.0) == 0.0:
        return observation
    norm = getattr(state, "norm", None)
    if not isinstance(norm, EMANormState):
        raise TypeError("an EMA frozen probe requires an EMANormState-backed learner")
    epsilon = hyperparameters.get("norm_epsilon")
    if epsilon is None or not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("an EMA frozen probe requires finite positive norm_epsilon")
    return (observation - norm.mean) / (jnp.sqrt(norm.var) + epsilon)


def _hidden_rms_frozen_probe_input(
    state: Any, observation: Array, hyperparameters: Mapping[str, float]
) -> Array:
    """Refuse sentinel probes for arms whose forward pass is not the plain MLP.

    ``sigma0_hidden_norm`` RMS-normalizes the hidden activations inside the
    forward pass; the probe harness computes logits with ``mlp_logits``, so
    any input-side transform would silently probe the wrong model.  Failing
    closed here is the honest option until the probe harness can accept a
    per-arm forward function.
    """
    del state, observation, hyperparameters
    raise NotImplementedError(
        "sentinel probes are unsupported for hidden-RMS-normalized arms: the "
        "deployed forward pass is not the plain protocol MLP"
    )


@dataclass(frozen=True)
class ScreeningSpec:
    """One screening arm: a named learner configuration.

    Attributes:
        name: Registry key and shard identity.
        base_learner: ``"upgd_w"`` or ``"adamw"`` (cost/reporting bucket).
        mechanism: Short mechanism tag for reporting.
        hyperparameters: Full resolved hyperparameters (JSON-serializable).
        factory: Builds ``(init_fn, step_fn)`` from the hyperparameters.
        description: One-line description for the summary.
        noise_update: Pure noise-consuming update for the pool-noise
            confirmation path (``None`` = pool mode unsupported for this arm).
        frozen_probe_input: Applies the learner's current input preprocessing
            without updating its state.  Raw-input learners use the identity
            transform; adaptive normalizers must opt in explicitly.
    """

    name: str
    base_learner: str
    mechanism: str
    hyperparameters: dict[str, float]
    factory: Callable[[Mapping[str, float]], tuple[LearnerInitFn, ScreeningStepFn]]
    description: str = ""
    noise_update: NoiseUpdateFn | None = None
    frozen_probe_input: FrozenProbeInputFn = _raw_frozen_probe_input


def _upgd_hp(**overrides: float) -> dict[str, float]:
    merged = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
    merged.update(overrides)
    return merged


def _sigma0_ext_hp(**overrides: float) -> dict[str, float]:
    """``upgd_ema_norm_sigma0``'s hyperparameters plus inert extension defaults."""
    merged = _upgd_hp(
        norm_decay=0.999,
        norm_epsilon=1e-8,
        noise_std=0.0,
        gate_beta=1.0,
        local_gate=0.0,
        hidden_rms=0.0,
    )
    merged.update(overrides)
    return merged


def _control_factory(
    make: Callable[[dict[str, float]], tuple[LearnerInitFn, LearnerStepFn]],
) -> Callable[[Mapping[str, float]], tuple[LearnerInitFn, ScreeningStepFn]]:
    def factory(hp: Mapping[str, float]) -> tuple[LearnerInitFn, ScreeningStepFn]:
        return _wrap_grad_learner(*make(dict(hp)))

    return factory


_CBP_DEFAULTS = {
    "cbp_decay_rate": 0.99,
    "cbp_replacement_rate": 1e-4,
    "cbp_maturity_threshold": 100.0,
}


def _build_registry() -> dict[str, ScreeningSpec]:
    specs = [
        ScreeningSpec(
            name="upgd_w_control",
            base_learner="upgd_w",
            mechanism="control",
            hyperparameters=_upgd_hp(),
            factory=_control_factory(_make_upgd_w_learner),
            description="Published UPGD-W (paired control arm; exact full-lane prefix).",
            noise_update=lean_upgd_w_update,
        ),
        ScreeningSpec(
            name="adamw_control",
            base_learner="adamw",
            mechanism="control",
            hyperparameters=dict(ADAMW_PROTOCOL_HYPERPARAMETERS),
            factory=_control_factory(_make_adamw_learner),
            description="Published AdamW baseline (proxy-ordering validation arm).",
        ),
        ScreeningSpec(
            name="upgd_idbd",
            base_learner="upgd_w",
            mechanism="per_weight_step_sizes",
            hyperparameters=_upgd_hp(meta_step_size=1e-3, initial_step_size=0.01),
            factory=_make_upgd_idbd_learner,
            description="UPGD-W with IDBD per-weight step-sizes on the gated gradient.",
        ),
        ScreeningSpec(
            name="upgd_idbd_meta1e2",
            base_learner="upgd_w",
            mechanism="per_weight_step_sizes",
            hyperparameters=_upgd_hp(meta_step_size=1e-2, initial_step_size=0.01),
            factory=_make_upgd_idbd_learner,
            description="UPGD-W + IDBD, faster meta step-size.",
        ),
        ScreeningSpec(
            name="upgd_autostep",
            base_learner="upgd_w",
            mechanism="per_weight_step_sizes",
            hyperparameters=_upgd_hp(
                meta_step_size=1e-2, initial_step_size=0.01, tau=1e4
            ),
            factory=_make_upgd_autostep_learner,
            description="UPGD-W with Autostep per-weight step-sizes on the gated gradient.",
        ),
        ScreeningSpec(
            name="upgd_w_idbd_swift",
            base_learner="upgd_w",
            mechanism="per_weight_step_sizes",
            hyperparameters=_upgd_hp(
                meta_step_size=1e-3,
                initial_step_size=0.01,
                swift_eta=0.1,
                swift_eps=0.99,
            ),
            factory=_make_upgd_idbd_swift_learner,
            description=(
                "UPGD-W + IDBD with SwiftTD's overshoot bound (eta) and "
                "persistent step-size decay on trigger (eps)."
            ),
        ),
        ScreeningSpec(
            name="upgd_w_fade_head",
            base_learner="upgd_w",
            mechanism="meta_learned_weight_decay",
            hyperparameters=_upgd_hp(
                fade_alpha=0.005, fade_gamma0=-6.9, fade_theta_lambda=0.1
            ),
            factory=_make_upgd_w_fade_head_learner,
            description=(
                "UPGD-W with FADE meta-learned per-parameter weight decay on "
                "the output layer (w3/b3); hidden layers unchanged."
            ),
        ),
        ScreeningSpec(
            name="upgd_l2init",
            base_learner="upgd_w",
            mechanism="l2_init",
            hyperparameters=_upgd_hp(),
            factory=_make_upgd_l2init_learner,
            description="UPGD-W whose weight decay pulls toward the initial weights.",
        ),
        ScreeningSpec(
            name="upgd_ema_norm",
            base_learner="upgd_w",
            mechanism="input_normalization",
            hyperparameters=_upgd_hp(norm_decay=0.999, norm_epsilon=1e-8),
            factory=_make_upgd_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description="UPGD-W behind an EMA input normalizer on the 784 pixels.",
        ),
        ScreeningSpec(
            name="upgd_cbp",
            base_learner="upgd_w",
            mechanism="dormant_unit_recycling",
            hyperparameters=_upgd_hp(**_CBP_DEFAULTS),
            factory=_make_upgd_cbp_learner,
            description="UPGD-W with CBP-style dormant-unit recycling.",
        ),
        # --- Wave 5: star around the confirmed upgd_ema_norm result (0.85357
        # at 200 tasks).  Its UPGD-W hyperparameters were tuned for RAW pixel
        # inputs; under EMA-normalized inputs the effective gradient scale,
        # the noise-to-gradient ratio, and the decay pressure all change, so
        # the published values are unlikely to remain optimal.  One axis per
        # arm, same factory.
        ScreeningSpec(
            name="upgd_ema_norm_wd0005",
            base_learner="upgd_w",
            mechanism="input_normalization",
            hyperparameters=_upgd_hp(
                norm_decay=0.999, norm_epsilon=1e-8, weight_decay=0.005
            ),
            factory=_make_upgd_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "upgd_ema_norm with the independently confirmed better weight "
                "decay 0.005 (composition of the two confirmed wins)."
            ),
        ),
        ScreeningSpec(
            name="upgd_ema_norm_lr003",
            base_learner="upgd_w",
            mechanism="input_normalization",
            hyperparameters=_upgd_hp(
                norm_decay=0.999, norm_epsilon=1e-8, step_size=0.03
            ),
            factory=_make_upgd_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description="upgd_ema_norm at 3x step size (normalized inputs change scale).",
        ),
        ScreeningSpec(
            name="upgd_ema_norm_lr0003",
            base_learner="upgd_w",
            mechanism="input_normalization",
            hyperparameters=_upgd_hp(
                norm_decay=0.999, norm_epsilon=1e-8, step_size=0.003
            ),
            factory=_make_upgd_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description="upgd_ema_norm at 1/3 step size.",
        ),
        ScreeningSpec(
            name="upgd_ema_norm_sigma0",
            base_learner="upgd_w",
            mechanism="input_normalization",
            hyperparameters=_upgd_hp(
                norm_decay=0.999, norm_epsilon=1e-8, noise_std=0.0
            ),
            factory=_make_upgd_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "upgd_ema_norm without the perturbation: is the noise "
                "load-bearing once inputs are conditioned?"
            ),
        ),
        # --- Wave 6: the final dissection of the normalized arm.  With
        # upgd_ema_norm_sigma0 tying upgd_ema_norm, the method reduces to
        # normalize + utility-gated SGD + decay; this arm drops the gate too.
        ScreeningSpec(
            name="sgd_ema_norm",
            base_learner="upgd_w",
            mechanism="input_normalization",
            hyperparameters={
                "step_size": 0.01,
                "weight_decay": 0.01,
                "norm_decay": 0.999,
                "norm_epsilon": 1e-8,
            },
            factory=_make_sgd_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "Gate ablation of upgd_ema_norm_sigma0: plain SGD + decoupled "
                "decay behind the exact EMA input normalizer — no utility, no "
                "gate, no noise."
            ),
        ),
        ScreeningSpec(
            name="adamw_cbp",
            base_learner="adamw",
            mechanism="dormant_unit_recycling",
            hyperparameters={**ADAMW_PROTOCOL_HYPERPARAMETERS, **_CBP_DEFAULTS},
            factory=_make_adamw_cbp_learner,
            description="AdamW with CBP-style recycling (Nature-combination reference arm).",
        ),
        # --- Wave 7: single-axis frontier extensions on the confirmed
        # upgd_ema_norm_sigma0 champion (0.85051 at 200 tasks).  The
        # decomposition attributes +0.061 to input conditioning and +0.011 to
        # the utility gate; these arms push the normalizer statistics
        # (ema_normalize already centers with the EMA mean, so decay/epsilon
        # are the unexplored axes), extend conditioning to the hidden layers,
        # and refine the gate under conditioning.  One axis per arm; the
        # shared factory's defaults reduce bit-exactly to the champion.
        ScreeningSpec(
            name="sigma0_hidden_norm",
            base_learner="upgd_w",
            mechanism="hidden_normalization",
            hyperparameters=_sigma0_ext_hp(hidden_rms=1.0, hidden_rms_epsilon=1e-8),
            factory=_make_upgd_ema_norm_ext_learner,
            frozen_probe_input=_hidden_rms_frozen_probe_input,
            description=(
                "upgd_ema_norm_sigma0 plus stateless per-example RMS "
                "normalization of both hidden ReLU layers (no learnable "
                "parameters — conditioning extended past the input)."
            ),
        ),
        ScreeningSpec(
            name="sigma0_localgate",
            base_learner="upgd_w",
            mechanism="local_gate_normalization",
            hyperparameters=_sigma0_ext_hp(local_gate=1.0),
            factory=_make_upgd_ema_norm_ext_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "upgd_ema_norm_sigma0 with the per-tensor gate normalization "
                "(-0.0008 on raw inputs; retested where conditioning rescales "
                "the utilities)."
            ),
        ),
        ScreeningSpec(
            name="guarded_cbp_adam",
            base_learner="adamw",
            mechanism="utility_guarded_recycling",
            hyperparameters={
                **ADAMW_PROTOCOL_HYPERPARAMETERS,
                **_CBP_DEFAULTS,
                "utility_decay": 0.9999,
                "guard_scale": 1.0,
            },
            factory=_make_guarded_cbp_adam_learner,
            description=(
                "AdamW+CBP with UPGD-style utility protection scaling Adam's "
                "applied delta by 1 - gate; no perturbation (CBP regenerates)."
            ),
        ),
        ScreeningSpec(
            name="adamw_cbp_noreset",
            base_learner="adamw",
            mechanism="dormant_unit_recycling",
            hyperparameters={**ADAMW_PROTOCOL_HYPERPARAMETERS, **_CBP_DEFAULTS},
            factory=_make_adamw_cbp_noreset_learner,
            description=(
                "adamw_cbp WITHOUT the per-unit Adam moment/count reset at "
                "replacement (moment-freshness dissection; the leader resets)."
            ),
        ),
        ScreeningSpec(
            name="adamw_cbp_ema_norm",
            base_learner="adamw",
            mechanism="input_normalization_recycling",
            hyperparameters={
                **ADAMW_PROTOCOL_HYPERPARAMETERS,
                **_CBP_DEFAULTS,
                "norm_decay": 0.999,
                "norm_epsilon": 1e-8,
                "norm_enabled": 1.0,
            },
            factory=_make_adamw_cbp_ema_norm_learner,
            frozen_probe_input=_ema_frozen_probe_input,
            description=(
                "adamw_cbp behind the exact upgd_ema_norm EMA input "
                "normalizer (composition of the two orthogonal wins)."
            ),
        ),
        ScreeningSpec(
            name="upgd_w_sigma0",
            base_learner="upgd_w",
            mechanism="perturbation_dissection",
            hyperparameters=_upgd_hp(noise_std=0.0),
            factory=_make_upgd_w_sigma0_learner,
            description=(
                "Lean UPGD-W with sigma=0: pure utility-gated SGD + decoupled "
                "decay, no perturbation (noise draw skipped entirely)."
            ),
        ),
        ScreeningSpec(
            name="upgd_alpha_utility",
            base_learner="upgd_w",
            mechanism="alpha_protection_signal",
            hyperparameters=_upgd_hp(meta_step_size=1e-2, initial_step_size=0.01),
            factory=_make_upgd_alpha_utility_learner,
            description=(
                "UPGD-W whose protection gate reads passive IDBD per-weight "
                "step-size drift instead of the -w*g utility EMA."
            ),
        ),
    ]
    for cbp_overrides, tag in (
        ({"cbp_replacement_rate": 3e-5}, "r3e5"),
        ({"cbp_replacement_rate": 3e-4}, "r3e4"),
        ({"cbp_maturity_threshold": 50.0}, "m50"),
        ({"cbp_maturity_threshold": 200.0}, "m200"),
    ):
        specs.append(
            ScreeningSpec(
                name=f"adamw_cbp_{tag}",
                base_learner="adamw",
                mechanism="dormant_unit_recycling",
                hyperparameters={
                    **ADAMW_PROTOCOL_HYPERPARAMETERS,
                    **_CBP_DEFAULTS,
                    **cbp_overrides,
                },
                factory=_make_adamw_cbp_learner,
                description=(
                    "adamw_cbp leader mini-star: "
                    + ", ".join(f"{k}={v}" for k, v in cbp_overrides.items())
                    + "."
                ),
            )
        )
    for kappa, wd, tag in (
        (1.0, 0.01, "k1"),
        (2.0, 0.01, "k2"),
        (1.0, 0.0, "k1_wd0"),
        (2.0, 0.0, "k2_wd0"),
    ):
        specs.append(
            ScreeningSpec(
                name=f"upgd_w_wclip_{tag}",
                base_learner="upgd_w",
                mechanism="weight_clipping",
                hyperparameters=_upgd_hp(clip_kappa=kappa, weight_decay=wd),
                factory=_make_upgd_w_wclip_learner,
                description=(
                    f"UPGD-W + per-layer weight clipping to kappa={kappa} times the "
                    f"init bound (weight_decay={wd})."
                ),
                noise_update=upgd_w_wclip_update,
            )
        )
    specs.append(
        ScreeningSpec(
            name="upgd_w_localgate",
            base_learner="upgd_w",
            mechanism="local_gate_normalization",
            hyperparameters=_upgd_hp(),
            factory=_make_upgd_w_localgate_learner,
            description="UPGD-W with the utility gate normalized per parameter tensor.",
            noise_update=upgd_w_localgate_update,
        )
    )
    for value, tag in ((0.05, "sigma005"), (0.2, "sigma02")):
        specs.append(
            ScreeningSpec(
                name=f"upgd_w_{tag}",
                base_learner="upgd_w",
                mechanism="hyperparameter_neighborhood",
                hyperparameters=_upgd_hp(noise_std=value),
                factory=_control_factory(_make_upgd_w_learner),
                description=f"UPGD-W with sigma={value}.",
                noise_update=lean_upgd_w_update,
            )
        )
    for value, tag in ((0.999, "udecay0999"), (0.99999, "udecay099999")):
        specs.append(
            ScreeningSpec(
                name=f"upgd_w_{tag}",
                base_learner="upgd_w",
                mechanism="hyperparameter_neighborhood",
                hyperparameters=_upgd_hp(utility_decay=value),
                factory=_control_factory(_make_upgd_w_learner),
                description=f"UPGD-W with utility_decay={value}.",
                noise_update=lean_upgd_w_update,
            )
        )
    for value, tag in ((0.005, "wd0005"), (0.02, "wd002")):
        specs.append(
            ScreeningSpec(
                name=f"upgd_w_{tag}",
                base_learner="upgd_w",
                mechanism="hyperparameter_neighborhood",
                hyperparameters=_upgd_hp(weight_decay=value),
                factory=_control_factory(_make_upgd_w_learner),
                description=f"UPGD-W with weight_decay={value}.",
                noise_update=lean_upgd_w_update,
            )
        )
    for value, tag in ((0.99, "ndecay099"), (0.9999, "ndecay09999")):
        specs.append(
            ScreeningSpec(
                name=f"sigma0_{tag}",
                base_learner="upgd_w",
                mechanism="input_normalization",
                hyperparameters=_sigma0_ext_hp(norm_decay=value),
                factory=_make_upgd_ema_norm_ext_learner,
                frozen_probe_input=_ema_frozen_probe_input,
                description=(
                    f"upgd_ema_norm_sigma0 with normalizer decay {value} "
                    "(champion 0.999)."
                ),
            )
        )
    for value, tag in ((1e-6, "eps1e6"), (1e-4, "eps1e4")):
        specs.append(
            ScreeningSpec(
                name=f"sigma0_{tag}",
                base_learner="upgd_w",
                mechanism="input_normalization",
                hyperparameters=_sigma0_ext_hp(norm_epsilon=value),
                factory=_make_upgd_ema_norm_ext_learner,
                frozen_probe_input=_ema_frozen_probe_input,
                description=(
                    f"upgd_ema_norm_sigma0 with normalizer epsilon {value} "
                    "(champion 1e-8; floors the variance and pads the divisor)."
                ),
            )
        )
    for value, tag in ((0.5, "gate_beta05"), (2.0, "gate_beta2")):
        specs.append(
            ScreeningSpec(
                name=f"sigma0_{tag}",
                base_learner="upgd_w",
                mechanism="gate_temperature",
                hyperparameters=_sigma0_ext_hp(gate_beta=value),
                factory=_make_upgd_ema_norm_ext_learner,
                frozen_probe_input=_ema_frozen_probe_input,
                description=(
                    f"upgd_ema_norm_sigma0 with utility-gate temperature beta={value} "
                    "(sigmoid of beta times the scaled utility)."
                ),
            )
        )
    return {spec.name: spec for spec in specs}


SCREENING_REGISTRY: Mapping[str, ScreeningSpec] = MappingProxyType(_build_registry())


def screening_spec(name: str) -> ScreeningSpec:
    """Look up a screening configuration by name."""
    if name not in SCREENING_REGISTRY:
        raise ValueError(
            f"unknown screening config {name!r}; expected one of "
            f"{sorted(SCREENING_REGISTRY)}"
        )
    return SCREENING_REGISTRY[name]


# =============================================================================
# Development-only recurring A/B/A retention adapter
# =============================================================================


RECURRING_IPMNIST_ADAPTER_SCHEMA = "alberta.ipmnist-screening.recurring-adapter.v1"
_MAX_UINT32 = 2**32 - 1


def _canonical_hash_array(array: object) -> np.ndarray:
    """Return a contiguous, little-endian, non-object array for hashing."""
    resolved = np.asarray(jax.device_get(array))
    if resolved.dtype.hasobject:
        raise TypeError("object arrays cannot enter a canonical SHA-256 binding")
    canonical_dtype = resolved.dtype.newbyteorder("<")
    return np.ascontiguousarray(resolved.astype(canonical_dtype, copy=False))


def _array_bundle_sha256(domain: str, arrays: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii") + b"\0")
    for name in sorted(arrays):
        encoded_name = name.encode("ascii")
        array = _canonical_hash_array(arrays[name])
        header = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "little"))
        digest.update(encoded_name)
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        payload = array.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _validated_permutation(permutation: object, *, input_dim: int) -> np.ndarray:
    raw = np.asarray(jax.device_get(permutation))
    if raw.dtype.kind not in {"i", "u"} or raw.ndim != 1:
        raise ValueError("each permutation must be a one-dimensional integer array")
    if raw.shape != (input_dim,):
        raise ValueError(f"each permutation must have shape ({input_dim},)")
    resolved = np.asarray(raw, dtype=np.int64)
    if not np.array_equal(np.sort(resolved), np.arange(input_dim, dtype=np.int64)):
        raise ValueError("each permutation must contain every input index exactly once")
    return np.asarray(resolved, dtype=np.int32)


def _validated_sentinel_indices(
    sentinel_indices: Sequence[int] | np.ndarray | Array, *, n_examples: int
) -> np.ndarray:
    raw = np.asarray(jax.device_get(sentinel_indices))
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
        raise TypeError("sentinel_indices must be a one-dimensional integer sequence")
    if raw.size == 0:
        raise ValueError("sentinel_indices must be non-empty")
    if np.any(raw < 0) or np.any(raw >= n_examples):
        raise ValueError("sentinel_indices must be in range for the supplied data")
    indices = np.asarray(raw, dtype=np.int64)
    if len(set(int(index) for index in indices)) != len(indices):
        raise ValueError("sentinel_indices must be unique and ordered explicitly")
    return indices


def _validated_recurring_phase_lengths(
    phase_lengths: Sequence[int],
) -> tuple[int, int, int]:
    lengths = tuple(phase_lengths)
    if len(lengths) != 3 or any(
        not isinstance(length, int) or isinstance(length, bool) or length <= 0
        for length in lengths
    ):
        raise ValueError("phase_lengths must contain exactly three positive integers")
    resolved = (int(lengths[0]), int(lengths[1]), int(lengths[2]))
    if resolved[0] != resolved[2]:
        raise ValueError("the two A phase lengths must be equal")
    return resolved


def _validated_recurring_seed(seed: int) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= _MAX_UINT32:
        raise ValueError("seed must be a canonical uint32 integer")
    return seed


def build_recurring_ipmnist_online_indices(
    *,
    seed: int,
    n_examples: int,
    phase_lengths: Sequence[int],
    sentinel_indices: Sequence[int] | np.ndarray | Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the evaluator-owned held-out A/B/A online example schedule.

    The first and recurring A exposures use the exact same ordered example
    indices (common random numbers); B uses an independent seed fold.  Every
    sentinel row is removed before any phase permutation is drawn.  This
    helper is intentionally public so those two properties can be checked
    without executing a learner.
    """
    resolved_seed = _validated_recurring_seed(seed)
    if (
        not isinstance(n_examples, int)
        or isinstance(n_examples, bool)
        or n_examples <= 0
    ):
        raise ValueError("n_examples must be a positive integer")
    lengths = _validated_recurring_phase_lengths(phase_lengths)
    indices = _validated_sentinel_indices(sentinel_indices, n_examples=n_examples)
    eligible_mask = np.ones(n_examples, dtype=np.bool_)
    eligible_mask[indices] = False
    eligible = np.flatnonzero(eligible_mask).astype(np.int32)
    if any(length > len(eligible) for length in lengths):
        raise ValueError(
            "each phase length must fit a without-replacement draw after holding out sentinels"
        )

    root = jr.key(jnp.uint32(resolved_seed))
    _, key_schedule, _ = jr.split(root, 3)
    _, key_sample = jr.split(key_schedule)

    def phase_order(fold_index: int, length: int) -> np.ndarray:
        offsets = np.asarray(
            jr.permutation(jr.fold_in(key_sample, fold_index), len(eligible))[:length]
        )
        return np.asarray(eligible[offsets], dtype=np.int32)

    a_order = phase_order(0, lengths[0])
    b_order = phase_order(1, lengths[1])
    return a_order, b_order, a_order.copy()


def _validated_ipmnist_data(
    data_x: np.ndarray | Array,
    data_y: np.ndarray | Array,
    *,
    input_dim: int | None,
    n_classes: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    raw_x = np.asarray(jax.device_get(data_x))
    raw_y = np.asarray(jax.device_get(data_y))
    if raw_x.ndim != 2:
        raise ValueError("data_x must be a two-dimensional example matrix")
    if input_dim is not None and raw_x.shape[1] != input_dim:
        raise ValueError(f"data_x must have shape (n_train, {input_dim})")
    if raw_y.shape != (raw_x.shape[0],):
        raise ValueError("data_y must be (n_train,) aligned with data_x")
    if raw_y.dtype.kind not in {"i", "u"}:
        raise ValueError("data_y must contain integer class labels")
    if np.any(raw_y < 0) or np.any(raw_y > np.iinfo(np.int32).max):
        raise ValueError("data_y class labels must fit non-negative int32")
    resolved_x = np.asarray(raw_x, dtype=np.float32)
    resolved_y = np.asarray(raw_y, dtype=np.int32)
    if not np.all(np.isfinite(resolved_x)):
        raise ValueError("data_x must contain only finite values")
    if n_classes is not None and np.any(resolved_y >= n_classes):
        raise ValueError(f"data_y class labels must be smaller than {n_classes}")
    return resolved_x, resolved_y


def ipmnist_permutation_sha256(permutation: np.ndarray | Array) -> str:
    """Bind one complete integer pixel permutation deterministically."""
    raw = np.asarray(jax.device_get(permutation))
    if raw.ndim != 1:
        raise ValueError("permutation must be one-dimensional")
    resolved = _validated_permutation(permutation, input_dim=int(raw.shape[0]))
    return _array_bundle_sha256(
        "alberta.ipmnist-screening.pixel-permutation.v1",
        {"permutation": resolved},
    )


def ipmnist_sentinel_set_sha256(
    data_x: np.ndarray | Array,
    data_y: np.ndarray | Array,
    permutation: np.ndarray | Array,
    sentinel_indices: Sequence[int] | np.ndarray | Array,
) -> str:
    """Bind ordered sentinel identities, labels, and pixel-permuted inputs.

    The digest covers the exact float32 source rows as well as the transformed
    rows.  Adaptive learner preprocessing (for example EMA normalization) is
    derived from the bound transformed rows and the separately hashed frozen
    learner state at each checkpoint.
    """
    resolved_x, resolved_y = _validated_ipmnist_data(
        data_x,
        data_y,
        input_dim=None,
        n_classes=None,
    )
    resolved_permutation = _validated_permutation(
        permutation, input_dim=int(resolved_x.shape[1])
    )
    indices = _validated_sentinel_indices(
        sentinel_indices, n_examples=int(resolved_x.shape[0])
    )
    raw_examples = resolved_x[indices]
    return _array_bundle_sha256(
        "alberta.ipmnist-screening.ordered-sentinel-set.v1",
        {
            "example_indices": indices,
            "labels": resolved_y[indices],
            "permutation": resolved_permutation,
            "pixel_permuted_inputs": raw_examples[:, resolved_permutation],
            "raw_examples": raw_examples,
        },
    )


def _declared_learner_state_sha256(
    params: dict[str, Array], state: Any, learner_key: Array
) -> str:
    """Hash parameters, optimizer/mechanism state, and the next learner RNG key."""
    bundle = {
        "learner_key": jr.key_data(learner_key),
        "optimizer_and_mechanism_state": state,
        "params": params,
    }
    path_leaves, tree = jax.tree_util.tree_flatten_with_path(bundle)
    digest = hashlib.sha256()
    digest.update(b"alberta.ipmnist-screening.full-learner-state.v1\0")
    tree_bytes = str(tree).encode("utf-8")
    digest.update(len(tree_bytes).to_bytes(8, "little"))
    digest.update(tree_bytes)
    for path, leaf in path_leaves:
        path_bytes = repr(path).encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, "little"))
        digest.update(path_bytes)
        array = _canonical_hash_array(leaf)
        header = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        payload = array.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _recurring_protocol_id(
    *,
    spec: ScreeningSpec,
    seed: int,
    config: IPMNISTConfig,
    phase_lengths: tuple[int, int, int],
    permutation_sha256: tuple[str, str, str],
    sentinel_indices_sha256: str,
    online_indices_sha256: tuple[str, str, str],
    relearning_window: int,
) -> str:
    manifest = {
        "schema": RECURRING_IPMNIST_ADAPTER_SCHEMA,
        "development_only": True,
        "config_name": spec.name,
        "base_learner": spec.base_learner,
        "hyperparameters": dict(spec.hyperparameters),
        "seed": seed,
        "config": config.to_config(),
        "phase_lengths": list(phase_lengths),
        "permutation_sha256": list(permutation_sha256),
        "sentinel_indices_sha256": sentinel_indices_sha256,
        "online_indices_sha256": list(online_indices_sha256),
        "relearning_window": relearning_window,
    }
    encoded = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"ipmnist-screening-aba-{hashlib.sha256(encoded).hexdigest()}.v1"


def run_recurring_ipmnist_retention_development(
    data_x: np.ndarray | Array,
    data_y: np.ndarray | Array,
    spec: ScreeningSpec,
    *,
    seed: int,
    config: IPMNISTConfig,
    phase_lengths: Sequence[int],
    permutations: Sequence[np.ndarray | Array],
    sentinel_indices: Sequence[int] | np.ndarray | Array,
    relearning_window: int,
) -> RecurringIPMNISTRetentionReport:
    """Run an explicit, threshold-free A/B/A retention diagnostic.

    This adapter reuses the screening arm's initialization, online update,
    input preprocessing, and same-example plasticity equations.  The learner
    receives only ``(x, y, rng_key)`` in a continuous key chain: phase,
    permutation, exposure, and sentinel identities remain evaluator-only.

    ``spec`` must be the exact object returned by :func:`screening_spec` for
    its name.  Cloned or custom specs are rejected even when their visible
    fields match: otherwise a substituted factory or stateful probe callback
    could silently change the semantics committed by the protocol identity.

    Every argument governing the recurrence is mandatory.  Sentinel rows are
    held out from the online examples, and every phase samples the remaining
    rows without replacement using the caller's seed.  There is deliberately
    no default protocol, threshold, artifact writer, or evidence path.

    The evaluator report schema stores the SHA-256 commitment to the adapter
    manifest in ``protocol_id``, not the manifest preimage.  Callers must keep
    these explicit arguments with the in-memory development report if they
    need standalone reconstruction; this function does not create artifacts.
    """
    if not isinstance(spec, ScreeningSpec):
        raise TypeError("spec must be a ScreeningSpec")
    if SCREENING_REGISTRY.get(spec.name) is not spec:
        raise ValueError(
            "spec must be the exact registered object returned by screening_spec(name)"
        )
    if not isinstance(config, IPMNISTConfig):
        raise TypeError("config must be an IPMNISTConfig")
    resolved_seed = _validated_recurring_seed(seed)
    if (
        not isinstance(relearning_window, int)
        or isinstance(relearning_window, bool)
        or relearning_window <= 0
    ):
        raise ValueError("relearning_window must be a positive integer")

    typed_lengths = _validated_recurring_phase_lengths(phase_lengths)
    if config.n_tasks != 3 or typed_lengths[0] != config.task_length:
        raise ValueError(
            "config must describe three phases and bind task_length to both A exposures"
        )
    if relearning_window > typed_lengths[0]:
        raise ValueError("relearning_window cannot exceed an A phase length")

    resolved_x, resolved_y = _validated_ipmnist_data(
        data_x,
        data_y,
        input_dim=config.input_dim,
        n_classes=config.n_classes,
    )
    raw_permutations = tuple(permutations)
    if len(raw_permutations) != 3:
        raise ValueError("permutations must contain the exact A/B/A phase tuple")
    resolved_permutations = tuple(
        _validated_permutation(permutation, input_dim=config.input_dim)
        for permutation in raw_permutations
    )
    if not np.array_equal(resolved_permutations[0], resolved_permutations[2]):
        raise ValueError("the first and third phase permutations must be exactly identical")
    if np.array_equal(resolved_permutations[0], resolved_permutations[1]):
        raise ValueError("the B permutation must be distinct from A")

    indices = _validated_sentinel_indices(
        sentinel_indices, n_examples=int(resolved_x.shape[0])
    )
    online_indices = build_recurring_ipmnist_online_indices(
        seed=resolved_seed,
        n_examples=int(resolved_x.shape[0]),
        phase_lengths=typed_lengths,
        sentinel_indices=indices,
    )

    root = jr.key(jnp.uint32(resolved_seed))
    key_init, _, key_noise = jr.split(root, 3)

    permutation_hashes = (
        ipmnist_permutation_sha256(resolved_permutations[0]),
        ipmnist_permutation_sha256(resolved_permutations[1]),
        ipmnist_permutation_sha256(resolved_permutations[2]),
    )
    sentinel_hashes = tuple(
        ipmnist_sentinel_set_sha256(resolved_x, resolved_y, permutation, indices)
        for permutation in resolved_permutations[:2]
    )
    online_hashes = tuple(
        _array_bundle_sha256(
            "alberta.ipmnist-screening.online-example-order.v1",
            {"example_indices": phase_indices},
        )
        for phase_indices in online_indices
    )
    typed_online_hashes = (online_hashes[0], online_hashes[1], online_hashes[2])
    sentinel_indices_hash = _array_bundle_sha256(
        "alberta.ipmnist-screening.sentinel-index-order.v1",
        {"sentinel_indices": indices},
    )
    protocol_id = _recurring_protocol_id(
        spec=spec,
        seed=resolved_seed,
        config=config,
        phase_lengths=typed_lengths,
        permutation_sha256=permutation_hashes,
        sentinel_indices_sha256=sentinel_indices_hash,
        online_indices_sha256=typed_online_hashes,
        relearning_window=relearning_window,
    )
    permutation_ids = (
        f"ipmnist-permutation-{permutation_hashes[0]}.v1",
        f"ipmnist-permutation-{permutation_hashes[1]}.v1",
    )
    sentinel_ids = (
        f"ipmnist-sentinel-{sentinel_hashes[0]}.v1",
        f"ipmnist-sentinel-{sentinel_hashes[1]}.v1",
    )
    phase_permutation_ids = (
        permutation_ids[0],
        permutation_ids[1],
        permutation_ids[0],
    )
    starts = (0, typed_lengths[0], typed_lengths[0] + typed_lengths[1])
    protocol = RecurringIPMNISTProtocol(
        protocol_id=protocol_id,
        phases=tuple(
            RecurringIPMNISTPhase(
                phase_index=index,
                start_step=starts[index],
                length=typed_lengths[index],
                permutation_id=phase_permutation_ids[index],
                exposure_index=0 if index < 2 else 1,
            )
            for index in range(3)
        ),
        sentinel_bindings=tuple(
            SentinelProbeBinding(
                permutation_id=permutation_ids[index],
                permutation_sha256=permutation_hashes[index],
                sentinel_set_id=sentinel_ids[index],
                sentinel_set_sha256=sentinel_hashes[index],
                sentinel_case_count=len(indices),
            )
            for index in range(2)
        ),
        relearning_window=relearning_window,
    )

    data_x_array = jnp.asarray(resolved_x, dtype=jnp.float32)
    data_y_array = jnp.asarray(resolved_y, dtype=jnp.int32)
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    params = init_mlp_params(key_init, config)
    state = init_fn(params)

    def run_phase(
        phase_params: dict[str, Array],
        phase_state: Any,
        learner_key: Array,
        permutation: Array,
        examples: Array,
    ) -> tuple[dict[str, Array], Any, Array, Array, Array]:
        def one_step(
            carry: tuple[dict[str, Array], Any, Array], example: Array
        ) -> tuple[tuple[dict[str, Array], Any, Array], tuple[Array, Array]]:
            step_params, step_state, next_key = carry
            x = data_x_array[example][permutation]
            y = data_y_array[example]
            next_key, step_key = jr.split(next_key)
            new_params, new_state, metrics = step_fn(
                step_params, step_state, x, y, step_key
            )
            accuracy, _, plasticity = metrics
            return (new_params, new_state, next_key), (accuracy, plasticity)

        (new_params, new_state, new_key), (accuracies, plasticities) = jax.lax.scan(
            one_step,
            (phase_params, phase_state, learner_key),
            examples,
        )
        return new_params, new_state, new_key, accuracies, plasticities

    run_phase_jit = jax.jit(run_phase)
    accuracy_trace: list[float] = []
    plasticity_trace: list[float] = []
    snapshots: list[SentinelProbeSnapshot] = []
    permutation_by_id = {
        permutation_ids[0]: resolved_permutations[0],
        permutation_ids[1]: resolved_permutations[1],
    }
    sentinel_labels = resolved_y[indices]

    for phase_index in range(3):
        params, state, key_noise, accuracies, plasticities = run_phase_jit(
            params,
            state,
            key_noise,
            jnp.asarray(resolved_permutations[phase_index], dtype=jnp.int32),
            jnp.asarray(online_indices[phase_index], dtype=jnp.int32),
        )
        accuracy_trace.extend(
            float(value) for value in np.asarray(jax.device_get(accuracies)).reshape(-1)
        )
        plasticity_trace.extend(
            float(value) for value in np.asarray(jax.device_get(plasticities)).reshape(-1)
        )

        requirements = tuple(
            requirement
            for requirement in protocol.required_probe_snapshots
            if requirement.phase_index == phase_index
        )
        for requirement in requirements:
            state_hash_before = _declared_learner_state_sha256(params, state, key_noise)
            permutation = permutation_by_id[requirement.permutation_id]
            sentinel_inputs = jnp.asarray(
                resolved_x[indices][:, permutation], dtype=jnp.float32
            )
            model_inputs = spec.frozen_probe_input(
                state, sentinel_inputs, spec.hyperparameters
            )
            if model_inputs.shape != sentinel_inputs.shape:
                raise ValueError("frozen_probe_input must preserve sentinel input shape")
            logits = np.asarray(jax.device_get(mlp_logits(params, model_inputs)))
            if not np.all(np.isfinite(logits)):
                raise ValueError("a frozen sentinel probe produced non-finite logits")
            correctness = tuple(
                bool(value)
                for value in np.asarray(np.argmax(logits, axis=-1) == sentinel_labels)
            )
            state_hash_after = _declared_learner_state_sha256(params, state, key_noise)
            snapshots.append(
                SentinelProbeSnapshot.from_requirement(
                    requirement,
                    learner_state_sha256_before=state_hash_before,
                    learner_state_sha256_after=state_hash_after,
                    correctness=correctness,
                )
            )

    trace = RecurringIPMNISTTrace(
        pre_update_online_accuracy=tuple(accuracy_trace),
        post_update_one_step_plasticity=tuple(plasticity_trace),
    )
    return build_recurring_ipmnist_retention_report(
        protocol=protocol,
        trace=trace,
        sentinel_snapshots=tuple(snapshots),
    )


# =============================================================================
# Runner (single seed, one process per seed)
# =============================================================================


@dataclass(frozen=True)
class ScreeningRunResult:
    """Host-side per-task results of one (config, seed) screening run."""

    config_name: str
    base_learner: str
    hyperparameters: dict[str, float]
    seed: int
    config: IPMNISTConfig
    per_task_accuracy: np.ndarray
    per_task_loss: np.ndarray
    per_task_plasticity: np.ndarray
    wall_clock_seconds: float
    noise_mode: str = "step"


def run_screening_config(
    data_x: np.ndarray | Array,
    data_y: np.ndarray | Array,
    spec: ScreeningSpec,
    seed: int,
    config: IPMNISTConfig,
    progress_every: int | None = None,
    noise_mode: str = "step",
    noise_pool_steps: int = 64,
) -> ScreeningRunResult:
    """Run one screening configuration for one seed.

    Seed derivation, schedules, init, and the per-step RNG chain mirror
    :func:`~alberta_framework.benchmarks.upgd_ipmnist.run_ipmnist` exactly,
    so control arms reproduce the full-horizon lane and every arm shares the
    control's task/example schedule for paired comparison.

    ``noise_mode="pool"`` mirrors ``run_ipmnist(noise_mode="pool")`` --
    including its per-task pool-key split and per-step offset draw, so the
    control arm reproduces the full lane's pool trajectories bit-for-bit
    (pinned by a unit test) -- but consumes ``spec.noise_update`` instead of
    the fixed UPGD-W equations. Pool shards are a screening-only
    approximation: they record ``noise_mode`` and never merge with exact
    shards nor pass proxy validation.
    """
    if noise_mode not in ("step", "pool"):
        raise ValueError(f"noise_mode must be 'step' or 'pool', got {noise_mode!r}")
    if noise_mode == "pool" and spec.noise_update is None:
        raise ValueError(
            f"noise_mode='pool' is unsupported for {spec.name!r}: the arm "
            "declares no noise-consuming update"
        )
    if noise_mode == "pool" and noise_pool_steps < 2:
        raise ValueError(f"noise_pool_steps must be >= 2, got {noise_pool_steps}")
    data_x = jnp.asarray(data_x, dtype=jnp.float32)
    data_y = jnp.asarray(data_y, dtype=jnp.int32)
    if data_x.ndim != 2 or data_x.shape[1] != config.input_dim:
        raise ValueError(
            f"data_x must have shape (n_train, {config.input_dim}), got {data_x.shape}"
        )
    if data_y.shape != (data_x.shape[0],):
        raise ValueError("data_y must be (n_train,) aligned with data_x")
    n_train = int(data_x.shape[0])

    init_fn, step_fn = spec.factory(spec.hyperparameters)

    root = jr.key(jnp.uint32(seed))
    key_init, key_schedule, key_noise = jr.split(root, 3)
    params = init_mlp_params(key_init, config)
    schedule = build_schedule(key_schedule, config, n_train)
    state = init_fn(params)

    def run_task(
        params: dict[str, Array],
        state: Any,
        key: Array,
        permutation: Array,
        examples: Array,
    ) -> tuple[dict[str, Array], Any, Array, Array, Array, Array]:
        def one_step(
            carry: tuple[dict[str, Array], Any, Array], example: Array
        ) -> tuple[tuple[dict[str, Array], Any, Array], StepMetrics]:
            step_params, step_state, key = carry
            x = data_x[example][permutation]
            y = data_y[example]
            key, step_key = jr.split(key)
            new_params, new_state, metrics = step_fn(step_params, step_state, x, y, step_key)
            return (new_params, new_state, key), metrics

        (params, state, key), (accuracies, losses, plasticities) = jax.lax.scan(
            one_step, (params, state, key), examples
        )
        return params, state, key, accuracies, losses, plasticities

    shapes = _sorted_param_shapes(config)
    n_flat = int(sum(np.prod(shape) for shape in shapes.values()))
    pool_len = int(noise_pool_steps) * n_flat
    pool_noise_std = float(spec.hyperparameters.get("noise_std", 0.0))
    noise_update = spec.noise_update
    hp = spec.hyperparameters

    def run_task_pool(
        params: dict[str, Array],
        state: Any,
        key: Array,
        permutation: Array,
        examples: Array,
    ) -> tuple[dict[str, Array], Any, Array, Array, Array, Array]:
        key, pool_key = jr.split(key)
        pool = jr.normal(pool_key, (pool_len,), jnp.float32) * pool_noise_std

        def one_step(
            carry: tuple[dict[str, Array], Any, Array], example: Array
        ) -> tuple[tuple[dict[str, Array], Any, Array], StepMetrics]:
            step_params, step_state, key = carry
            x = data_x[example][permutation]
            y = data_y[example]
            (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
                step_params, x, y
            )
            key, step_key = jr.split(key)
            offset = jr.randint(step_key, (), 0, pool_len - n_flat + 1)
            flat_noise = jax.lax.dynamic_slice(pool, (offset,), (n_flat,))
            noise = _split_flat_noise(flat_noise, shapes)
            assert noise_update is not None
            new_params, new_state = noise_update(step_params, step_state, grads, noise, hp)
            return (new_params, new_state, key), _step_metrics(
                new_params, x, y, loss, logits
            )

        (params, state, key), (accuracies, losses, plasticities) = jax.lax.scan(
            one_step, (params, state, key), examples
        )
        return params, state, key, accuracies, losses, plasticities

    run_task_jit = jax.jit(run_task_pool if noise_mode == "pool" else run_task)

    task_accuracy: list[float] = []
    task_loss: list[float] = []
    task_plasticity: list[float] = []
    started = time.monotonic()
    for task in range(config.n_tasks):
        params, state, key_noise, accuracies, losses, plasticities = run_task_jit(
            params,
            state,
            key_noise,
            schedule.permutations[task],
            schedule.example_indices[task],
        )
        task_accuracy.append(float(jnp.mean(accuracies)))
        task_loss.append(float(jnp.mean(losses)))
        task_plasticity.append(float(jnp.mean(plasticities)))
        if progress_every is not None and (task + 1) % progress_every == 0:
            elapsed = time.monotonic() - started
            logger.info(
                "%s seed=%d task %d/%d online_acc=%.4f elapsed=%.1fs",
                spec.name,
                seed,
                task + 1,
                config.n_tasks,
                task_accuracy[-1],
                elapsed,
            )
    return ScreeningRunResult(
        config_name=spec.name,
        base_learner=spec.base_learner,
        hyperparameters=dict(spec.hyperparameters),
        seed=int(seed),
        config=config,
        per_task_accuracy=np.asarray(task_accuracy, dtype=np.float64),
        per_task_loss=np.asarray(task_loss, dtype=np.float64),
        per_task_plasticity=np.asarray(task_plasticity, dtype=np.float64),
        wall_clock_seconds=time.monotonic() - started,
        noise_mode=noise_mode,
    )


# =============================================================================
# Shards, merge, summary
# =============================================================================


def shard_payload(result: ScreeningRunResult) -> dict[str, Any]:
    """Serialize one (config, seed) screening run to a mergeable shard."""
    return {
        "schema": SHARD_SCHEMA,
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "config_name": result.config_name,
        "base_learner": result.base_learner,
        "hyperparameters": result.hyperparameters,
        "seed": result.seed,
        "noise_mode": result.noise_mode,
        "config": result.config.to_config(),
        "per_task_accuracy": [round(float(v), 8) for v in result.per_task_accuracy],
        "per_task_loss": [round(float(v), 8) for v in result.per_task_loss],
        "per_task_plasticity": [round(float(v), 8) for v in result.per_task_plasticity],
        "wall_clock_seconds": round(result.wall_clock_seconds, 2),
        "created_unix": time.time(),
        "environment": {
            "jax": jax.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def load_shard(path: Path) -> dict[str, Any]:
    """Load and structurally validate one screening shard."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SHARD_SCHEMA:
        raise ValueError(f"{path}: not an {SHARD_SCHEMA} shard")
    config = IPMNISTConfig(**payload["config"])
    for fieldname in ("per_task_accuracy", "per_task_loss", "per_task_plasticity"):
        values = np.asarray(payload[fieldname], dtype=np.float64)
        if values.shape != (config.n_tasks,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: {fieldname} must be finite with shape ({config.n_tasks},)")
    if type(payload.get("seed")) is not int or payload["seed"] < 0:
        raise ValueError(f"{path}: seed must be a non-negative integer")
    if payload.get("config_name") not in SCREENING_REGISTRY:
        raise ValueError(f"{path}: unknown config_name {payload.get('config_name')!r}")
    return payload


def _late_window_slope(per_task_accuracy: np.ndarray, window: int) -> float:
    """OLS slope (accuracy per task) over the final ``window`` tasks."""
    tail = per_task_accuracy[-window:]
    x = np.arange(tail.shape[0], dtype=np.float64)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x * (tail - tail.mean())) / denom)


def merge_shards(
    paths: Sequence[Path],
    control_name: str = "upgd_w_control",
    slope_window: int = 15,
) -> dict[str, Any]:
    """Merge shards into a ranked screening summary with paired comparisons."""
    shards = [load_shard(Path(p)) for p in paths]
    if not shards:
        raise ValueError("no shards given")
    configs = {tuple(sorted(s["config"].items())) for s in shards}
    if len(configs) != 1:
        raise ValueError("shards span multiple protocol configs; merge them separately")
    noise_modes = {s.get("noise_mode", "step") for s in shards}
    if len(noise_modes) != 1:
        raise ValueError(
            "shards span multiple noise modes (pool results are a screening-only "
            "approximation); merge them separately"
        )
    noise_mode = noise_modes.pop()
    by_config: dict[str, dict[int, dict[str, Any]]] = {}
    for shard in shards:
        per_seed = by_config.setdefault(shard["config_name"], {})
        if shard["seed"] in per_seed:
            raise ValueError(
                f"duplicate shard for config={shard['config_name']} seed={shard['seed']}"
            )
        per_seed[shard["seed"]] = shard

    control = by_config.get(control_name, {})
    entries: list[dict[str, Any]] = []
    for name, per_seed in sorted(by_config.items()):
        seeds = sorted(per_seed)
        acc = np.stack(
            [np.asarray(per_seed[s]["per_task_accuracy"], dtype=np.float64) for s in seeds]
        )
        per_seed_avg = acc.mean(axis=1)
        slopes = np.asarray([_late_window_slope(acc[i], slope_window) for i in range(len(seeds))])
        entry: dict[str, Any] = {
            "config_name": name,
            "base_learner": per_seed[seeds[0]]["base_learner"],
            "hyperparameters": per_seed[seeds[0]]["hyperparameters"],
            "seeds": seeds,
            "n_seeds": len(seeds),
            "average_online_accuracy_mean": float(per_seed_avg.mean()),
            "average_online_accuracy_stderr": (
                float(per_seed_avg.std(ddof=1) / math.sqrt(len(seeds)))
                if len(seeds) > 1
                else 0.0
            ),
            "per_seed_average_online_accuracy": [round(float(v), 6) for v in per_seed_avg],
            "late_window_slope_mean": float(slopes.mean()),
            "per_seed_late_window_slope": [round(float(v), 8) for v in slopes],
            "average_plasticity_mean": float(
                np.mean(
                    [
                        np.mean(per_seed[s]["per_task_plasticity"])
                        for s in seeds
                    ]
                )
            ),
            "wall_clock_seconds_total": round(
                float(sum(per_seed[s]["wall_clock_seconds"] for s in seeds)), 2
            ),
        }
        common = [s for s in seeds if s in control]
        if name != control_name and common:
            control_avg = np.asarray(
                [
                    np.mean(np.asarray(control[s]["per_task_accuracy"], dtype=np.float64))
                    for s in common
                ]
            )
            ours_avg = np.asarray(
                [
                    np.mean(np.asarray(per_seed[s]["per_task_accuracy"], dtype=np.float64))
                    for s in common
                ]
            )
            diff = ours_avg - control_avg
            entry["paired_vs_control"] = {
                "control": control_name,
                "seeds": common,
                "per_seed_diff": [round(float(v), 6) for v in diff],
                "mean_diff": float(diff.mean()),
                "stderr_diff": (
                    float(diff.std(ddof=1) / math.sqrt(len(common)))
                    if len(common) > 1
                    else 0.0
                ),
                "all_seeds_improve": bool(np.all(diff > 0.0)),
                "beats_control": bool(diff.mean() > 0.0),
                "confirmation_candidate": bool(diff.mean() > CONFIRMATION_THRESHOLD),
            }
        entries.append(entry)

    entries.sort(key=lambda e: e["average_online_accuracy_mean"], reverse=True)
    return {
        "schema": SUMMARY_SCHEMA,
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "created_unix": time.time(),
        "protocol_config": dict(shards[0]["config"]),
        "noise_mode": noise_mode,
        "control_name": control_name,
        "confirmation_threshold": CONFIRMATION_THRESHOLD,
        "slope_window": slope_window,
        "n_shards": len(shards),
        "results": entries,
    }


def validate_proxy(
    shard_paths: Sequence[Path],
    partials_dir: Path,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Validate control shards against the completed full-horizon partials.

    Checks, per control shard, that the proxy per-task accuracy equals the
    first ``n_tasks`` entries of the corresponding 200-task shard (exact
    prefix property), and that the proxy horizon preserves the known
    UPGD-W > AdamW ordering both in the proxy runs and in the full-run
    prefixes at the same task index.
    """
    partials_dir = Path(partials_dir)
    checks: list[dict[str, Any]] = []
    proxy_avg: dict[str, list[float]] = {"upgd_w": [], "adamw": []}
    full_avg: dict[str, list[float]] = {"upgd_w": [], "adamw": []}
    n_tasks_seen: set[int] = set()
    for path in shard_paths:
        shard = load_shard(Path(path))
        if shard.get("noise_mode", "step") != "step":
            raise ValueError(
                f"{path}: proxy validation requires noise_mode='step' shards "
                f"(got {shard.get('noise_mode')!r})"
            )
        learner = {"upgd_w_control": "upgd_w", "adamw_control": "adamw"}.get(
            shard["config_name"]
        )
        if learner is None:
            raise ValueError(f"{path}: proxy validation accepts only control shards")
        seed = shard["seed"]
        n_tasks = int(shard["config"]["n_tasks"])
        n_tasks_seen.add(n_tasks)
        partial_path = partials_dir / f"{learner}_seed{seed}.json"
        reference = json.loads(partial_path.read_text(encoding="utf-8"))
        full_curve = np.asarray(reference["per_task_accuracy"][0], dtype=np.float64)
        proxy_curve = np.asarray(shard["per_task_accuracy"], dtype=np.float64)
        max_abs_diff = float(np.max(np.abs(proxy_curve - full_curve[:n_tasks])))
        proxy_avg[learner].append(float(proxy_curve.mean()))
        full_avg[learner].append(float(full_curve[:n_tasks].mean()))
        checks.append(
            {
                "config_name": shard["config_name"],
                "seed": seed,
                "reference_partial": partial_path.as_posix(),
                "max_abs_per_task_diff": max_abs_diff,
                "prefix_match": bool(max_abs_diff <= atol),
            }
        )
    ordering_proxy = (
        bool(np.mean(proxy_avg["upgd_w"]) > np.mean(proxy_avg["adamw"]))
        if proxy_avg["upgd_w"] and proxy_avg["adamw"]
        else None
    )
    ordering_full_prefix = (
        bool(np.mean(full_avg["upgd_w"]) > np.mean(full_avg["adamw"]))
        if full_avg["upgd_w"] and full_avg["adamw"]
        else None
    )
    return {
        "schema": VALIDATION_SCHEMA,
        "created_unix": time.time(),
        "atol": atol,
        "n_tasks": sorted(n_tasks_seen),
        "checks": checks,
        "all_prefixes_match": bool(all(c["prefix_match"] for c in checks)),
        "proxy_mean_average_online_accuracy": {
            k: (float(np.mean(v)) if v else None) for k, v in proxy_avg.items()
        },
        "full_prefix_mean_average_online_accuracy": {
            k: (float(np.mean(v)) if v else None) for k, v in full_avg.items()
        },
        "proxy_preserves_upgd_over_adamw": ordering_proxy,
        "full_prefix_preserves_upgd_over_adamw": ordering_full_prefix,
        "proxy_validated": bool(
            all(c["prefix_match"] for c in checks)
            and ordering_proxy is True
            and ordering_full_prefix is True
        ),
    }


# =============================================================================
# CLI
# =============================================================================


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    """Screening CLI: ``run`` one (config, seed); ``merge``; ``validate-proxy``."""
    parser = argparse.ArgumentParser(description="IPMNIST mechanism-combination screening")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run one (config, seed) shard")
    run_p.add_argument("--config-name", required=True, choices=sorted(SCREENING_REGISTRY))
    run_p.add_argument("--seed", type=int, required=True)
    run_p.add_argument("--n-tasks", type=int, default=PROXY_N_TASKS)
    run_p.add_argument("--task-length", type=int, default=5000)
    run_p.add_argument("--data-home", type=Path, default=None)
    run_p.add_argument("--out", type=Path, required=True)
    run_p.add_argument("--progress-every", type=int, default=10)
    run_p.add_argument(
        "--noise-mode", choices=("step", "pool"), default="step",
        help="'pool' = screening-only pool-noise approximation "
             "(lean-UPGD-family arms only; never mergeable with exact shards)",
    )
    run_p.add_argument("--noise-pool-steps", type=int, default=64)

    merge_p = sub.add_parser("merge", help="merge shards into a ranked summary")
    merge_p.add_argument("--shards", type=Path, nargs="+", required=True)
    merge_p.add_argument("--control-name", default="upgd_w_control")
    merge_p.add_argument("--slope-window", type=int, default=15)
    merge_p.add_argument("--output", type=Path, required=True)

    val_p = sub.add_parser("validate-proxy", help="validate control shards vs full partials")
    val_p.add_argument("--shards", type=Path, nargs="+", required=True)
    val_p.add_argument("--partials-dir", type=Path,
                       default=Path("outputs/upgd_ipmnist/partials"))
    val_p.add_argument("--atol", type=float, default=1e-6)
    val_p.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )

    if args.command == "run":
        spec = screening_spec(args.config_name)
        config = IPMNISTConfig(n_tasks=args.n_tasks, task_length=args.task_length)
        data_home = args.data_home if args.data_home is not None else default_openml_data_home()
        logger.info("loading MNIST from data_home=%s", data_home)
        data_x, data_y = load_mnist_train(data_home)
        logger.info(
            "running %s seed=%d for %d tasks x %d steps (noise_mode=%s)",
            spec.name, args.seed, config.n_tasks, config.task_length, args.noise_mode,
        )
        result = run_screening_config(
            data_x, data_y, spec, args.seed, config,
            progress_every=args.progress_every,
            noise_mode=args.noise_mode,
            noise_pool_steps=args.noise_pool_steps,
        )
        _atomic_write_json(args.out, shard_payload(result))
        logger.info(
            "%s seed=%d done: avg online acc %.4f (wall %.1fs) -> %s",
            spec.name,
            args.seed,
            float(result.per_task_accuracy.mean()),
            result.wall_clock_seconds,
            args.out,
        )
    elif args.command == "merge":
        summary = merge_shards(
            args.shards, control_name=args.control_name, slope_window=args.slope_window
        )
        _atomic_write_json(args.output, summary)
        logger.info("merged %d shards -> %s", summary["n_shards"], args.output)
    elif args.command == "validate-proxy":
        report = validate_proxy(args.shards, args.partials_dir, atol=args.atol)
        _atomic_write_json(args.output, report)
        logger.info(
            "proxy_validated=%s (prefix_match=%s ordering=%s) -> %s",
            report["proxy_validated"],
            report["all_prefixes_match"],
            report["proxy_preserves_upgd_over_adamw"],
            args.output,
        )


if __name__ == "__main__":
    main()
