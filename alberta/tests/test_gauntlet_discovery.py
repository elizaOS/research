"""Exhaustive pairwise feature-selection diagnostic on the Alberta Gauntlet.

The gauntlet diagnostic suite (tests/test_gauntlet_certification.py) shows on
its development seeds that recurrence performance is a *representation*
property:
over the oracle :class:`ContextGatedFeatures` products ``c_k * x_i`` an
Autostep learner scores savings_c ~9.8 / savings_d ~15.1, while the same
learner on raw observations scores ~1.2-1.6 and the fresh-reinit twin ~1.0.
This file asks a narrower Step 2 question: can a feature-selection learner,
given the raw observation ``[x, c]`` rather than precomputed gated features,
identify the context-interaction products that enable task memory?

Learner choice: :class:`FixedBudgetInteractionLearner`
(core/interaction_features.py) searches pairwise products ``obs[i] * obs[j]``
of the raw observation — the oracle's gated features are literally points in
its exhaustive search space (pairs ``(i, 12)`` / ``(i, 13)`` pair an x
channel with a supplied task-identity context channel).
``FixedBudgetFeatureLearner`` builds dense random linear
features (no interpretable parents) and ``CompositionalFeatureLearner`` is a
heavier DAG learner; the interaction learner is the minimal tool whose
hypothesis space contains the answer.  Configuration: 24 active slots, the
full 91-pair candidate archive (``candidate_strategy="all_pairs"``, trained on
its own deployed residual, promoted on signed causal utility),
``utility_retention_decay=0.9999``, and the opt-in scale-robust head.  The head
uses diagonal NLMS normalized by ``1 + phi.T D^-1 phi``; active utility is the
bounded loss increase from deleting a contribution, and candidate utility is
the bounded loss reduction from adding one.

This is not general or open-ended feature finding.  All pair descriptors and
candidate output weights remain resident in the counted 91-entry archive,
only degree-two products are considered, the context cue is supplied, and the
development seeds below were inspected without a held-out confidence
interval.  The result is a narrow diagnostic of online exhaustive pairwise
selection and active-bank retention, not completion of Alberta Plan Step 2.

Historical design-C development comparison (integer stream seeds 8--15,
``jr.key(123)`` learner init, default ``GauntletConfig``; medians).  These
figures predate v2's unique-per-task representation counts, end-segment-7
snapshot, D error guards, and scaled early/cumulative guards; they are not v2
acceptance values:

===========================  =============  ==============  =============
quantity                     robust+retain  matched legacy  robust no-ret
===========================  =============  ==============  =============
savings_c / savings_d        13.39 / 13.71  29.47 / 13.39   8.90 / 105.00
savings_c_final              5.38           1.19            7.67
scaled tail MSE (last 500)   4.34           208750.83       5.19
final-C tail MSE             0.040          8.257           0.051
nonlinear MSE                0.039          0.686           0.044
final relevant ctx slots     16 (min 15)    0               9 (min 8)
non-finite steps             0              0               0
===========================  =============  ==============  =============

The matched legacy arm has exactly the same descriptors, archive, lifecycle,
stream, and initialization, using its native OBGD mechanism.  The current
learner stores 24 idle counters and 24 evidence-streak counters (192 bytes),
plus 24 output-memory commitment flags (24 bytes), so the legacy arm costs
3916 persistent array bytes; robust normalization adds 116 float scalars
(464 bytes) for 4380 bytes total.  Removing retention sometimes improves
short-term errors and savings ratios, so retention is not claimed to be
performance-load-bearing.  Its narrower measured effect in the original
slot-count audit is representation persistence: the retained arm finishes
with a paired median gain of 7 relevant context slots and a 0.012 lower
final-tail MSE.

The finite nonlinear bank is also useful outside the context tasks: the
fresh segment-7 probe below requires it to identify the generating products
``(x_0 x_1, x_2 x_3, x_4 x_5, x_6 x_7)`` and beat a raw linear learner.

Scale-robust development audit (seeds 8--15 only; not held-out evidence):

* Design A combined a per-coordinate current-value normalizer, a fixed
  ``1 / 25`` active-head step, and the signed loss proxy now unit-tested
  below.  It stayed finite and kept a final context-pair median/min of
  6.5/4, but learned disastrously slowly: median first-C/recurrent-C MSE was
  5858.71/91.48 and scaled-segment MSE was 739.50.  Its apparently excellent
  final-C savings ratio was 953.12x (minimum 79.04x), solely because the
  acquisition denominator was catastrophic.  This rejected design is kept
  here because it demonstrates why a savings ratio without absolute
  acquisition and recurrence errors is not evidence of memory.

* Design B replaced the fixed divisor with diagonally preconditioned joint
  NLMS, normalized by ``1 + phi.T D^-1 phi`` including the bias.  It remained
  finite, but the legacy candidate rule still accumulated deployed residual
  gradients instead of fitting each candidate's proposed residual.  Context
  pairs fell to median/min 0/0 after both ordinary recurrence segments and
  recovered only to 4.5/4 at the end; first/recurrent-C MSE was 9.28/8.21,
  final-C early/tail MSE was 34.61/279.39, scaled tail MSE was 6487.40, and
  final savings was 0.51x.  A seed-8 C-only rank probe found irrelevant
  candidate weights with magnitude 2--5 while the maximum relevant
  context-candidate utility was only 0.0175.  That diagnosis motivated the
  robust-only candidate residual ``e_candidate = e_deployed - w_c * phi_c``.

* Design C kept the same joint NLMS and signed utilities but trained each
  dormant head as a proper one-dimensional residual model.  On seeds 8--15,
  it was finite with no NaNs; final relevant-context slots had median/min
  16/15; C/D/final-C savings were 13.39x/13.71x/5.38x; recurrent-C
  early MSE was 0.672; scaled-segment tail MSE was 4.34 (maximum 36.56);
  final-C early/tail MSE was 1.476/0.040; and nonlinear MSE was 0.0392.
  Unlike Design A, its acquisition error stayed bounded (first-C early/tail
  MSE 8.49/0.589), so the savings denominator is interpretable.

These designs were inspected during mechanism development.  The direct-key
v2 calibration rerun on seeds 8--15 subsequently froze the revised
representation, D-error, and scale-transient contract; its exact available
summary is stored in ``scale_robust_feature_calibration_v2.json``.  An older
primary-only confirmation already exposed seeds 30--45, so v2 excludes the
30--59 namespace and records a fresh SHA-256-derived 30-seed schedule.  That
fresh schedule remains unrun, and no promoted v2 result is reported by this
development test.

The representation gate deduplicates canonical descriptors before counting
them.  The historical context figures above remain explicitly slot-count
diagnostics; the later v2 calibration record contains the tightened
unique-product summaries used to freeze the thresholds.

Runtime depends on JAX compilation; scan+vmap runs are shared through
module-scoped fixtures.
"""

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array

from alberta_framework.core.interaction_features import FixedBudgetInteractionLearner
from alberta_framework.core.learners import LinearLearner
from alberta_framework.core.optimizers import Autostep
from alberta_framework.streams.gauntlet import (
    GauntletConfig,
    GauntletStream,
    gauntlet_scorecard,
    run_gauntlet_batched,
)

DEVELOPMENT_SEEDS = tuple(range(8, 16))
N_SEEDS = len(DEVELOPMENT_SEEDS)
LEARNER_INIT_KEY = 123
RETENTION_DECAY = 0.9999
PROBE_STEPS = 3000


def make_discovery_learner(
    retention: float | None = RETENTION_DECAY,
) -> FixedBudgetInteractionLearner:
    """The frozen scale-robust discovery configuration (see module docstring).

    24 active product slots over the 14-dim raw observation, the full 91-pair
    candidate archive fit to deployed residual error, signed loss utilities,
    diagonally normalized output updates, and (by default) the
    recurrent-context utility retention floor.  OBGD is disabled because the
    robust head has its own exact prediction-space contraction bound.
    """
    return FixedBudgetInteractionLearner(
        n_features=24,
        n_tasks=1,
        candidate_count=91,
        candidate_strategy="all_pairs",
        utility_retention_decay=retention,
        refresh_candidates=False,
        refresh_promoted_candidate=False,
        scale_robust=True,
        use_obgd=False,
    )


def bank_context_pair_counts(
    feature_left: Array, feature_right: Array, config: GauntletConfig
) -> tuple[Array, Array, Array]:
    """Count unique active products pairing context with an x channel.

    Returns ``(n_ctx, n_ctx_c_relevant, n_ctx_d_relevant)``.  The latter two
    counts are separated because a combined total could hide complete loss of
    either supplied task cue.  Repeated active descriptors count once.
    """
    d = config.input_dim
    left = jnp.minimum(feature_left, feature_right)
    right = jnp.maximum(feature_left, feature_right)
    is_ctx = (left >= d) ^ (right >= d)
    x_index = left
    ctx_c_rel = (right == d) & (x_index < config.relevant_dim)
    ctx_d_rel = (right == d + 1) & (x_index < config.relevant_dim)
    pair_code = left * config.observation_dim + right
    slot_count = pair_code.shape[-1]
    slot = jnp.arange(slot_count)
    earlier_slot = slot[None, :] < slot[:, None]
    repeated = jnp.any(
        (pair_code[..., :, None] == pair_code[..., None, :]) & earlier_slot,
        axis=-1,
    )
    first_occurrence = ~repeated
    return (
        jnp.sum(is_ctx & first_occurrence, axis=-1),
        jnp.sum(ctx_c_rel & first_occurrence, axis=-1),
        jnp.sum(ctx_d_rel & first_occurrence, axis=-1),
    )


class TestScaleRobustMechanism:
    """Focused mechanism checks independent of the long gauntlet program."""

    def test_context_pair_trace_counts_unique_descriptors(self):
        config = GauntletConfig()
        left = jnp.array([0, 0, 1, 8], dtype=jnp.int32)
        right = jnp.array([12, 12, 13, 12], dtype=jnp.int32)

        context, relevant_c, relevant_d = bank_context_pair_counts(left, right, config)

        assert int(context) == 3
        assert int(relevant_c) == 1
        assert int(relevant_d) == 1

    def test_joint_nlms_has_exact_prediction_space_contraction_bound(self):
        learner = FixedBudgetInteractionLearner(
            n_features=3,
            n_tasks=1,
            candidate_count=0,
            replacement_interval=0,
            step_size_output=0.3,
            scale_robust=True,
        )
        state = learner.init(feature_dim=4, key=jr.key(90)).replace(
            feature_left=jnp.array([0, 0, 2], dtype=jnp.int32),
            feature_right=jnp.array([1, 3, 3], dtype=jnp.int32),
            output_weights=jnp.array([[0.2, -0.1, 0.05]], dtype=jnp.float32),
            output_biases=jnp.array([0.4], dtype=jnp.float32),
        )
        observation = jnp.array([1e-5, 2.0, 10.0, -3.0], dtype=jnp.float32)
        target = jnp.array([7.0], dtype=jnp.float32)
        features = learner.constructed_features(state, observation)

        result = learner.update(state, observation, target)

        parameter_prediction_change = (
            (result.state.output_weights - state.output_weights) @ features
            + result.state.output_biases
            - state.output_biases
        )
        expected = 0.3 * result.errors
        assert jnp.allclose(
            parameter_prediction_change,
            expected,
            rtol=1e-5,
            atol=1e-6,
        )
        assert float(jnp.abs(parameter_prediction_change[0])) <= (
            0.3 * float(jnp.abs(result.errors[0])) + 1e-6
        )

    def test_update_and_loss_utilities_are_scale_equivariant(self):
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            candidate_count=1,
            replacement_interval=0,
            utility_decay=0.0,
            step_size_output=0.1,
            scale_robust=True,
        )
        state = learner.init(feature_dim=2, key=jr.key(91)).replace(
            feature_left=jnp.array([0], dtype=jnp.int32),
            feature_right=jnp.array([1], dtype=jnp.int32),
            candidate_left=jnp.array([0], dtype=jnp.int32),
            candidate_right=jnp.array([1], dtype=jnp.int32),
            output_weights=jnp.array([[1.0]], dtype=jnp.float32),
            candidate_output_weights=jnp.array([[0.5]], dtype=jnp.float32),
        )

        unit = learner.update(
            state,
            jnp.array([2.0, 1.0], dtype=jnp.float32),
            jnp.array([4.0], dtype=jnp.float32),
        )
        scaled = learner.update(
            state,
            jnp.array([20.0, 1.0], dtype=jnp.float32),
            jnp.array([40.0], dtype=jnp.float32),
        )

        # Both target and constructed feature scale by 10.  The raw weight
        # update and the normalized intervention utilities are unchanged;
        # the bias and prediction-space quantities scale with the target.
        assert jnp.allclose(
            unit.state.output_weights,
            scaled.state.output_weights,
            rtol=1e-6,
            atol=1e-7,
        )
        assert jnp.allclose(
            unit.state.candidate_output_weights,
            scaled.state.candidate_output_weights,
            rtol=1e-6,
            atol=1e-7,
        )
        assert jnp.allclose(
            unit.state.utilities,
            scaled.state.utilities,
            rtol=1e-6,
            atol=1e-7,
        )
        assert jnp.allclose(
            unit.state.candidate_utilities,
            scaled.state.candidate_utilities,
            rtol=1e-6,
            atol=1e-7,
        )
        assert jnp.allclose(
            scaled.state.feature_second_moments,
            100.0 * unit.state.feature_second_moments,
        )
        assert jnp.allclose(
            scaled.state.target_second_moments,
            100.0 * unit.state.target_second_moments,
        )
        assert jnp.allclose(
            scaled.state.output_biases,
            10.0 * unit.state.output_biases,
        )

    def test_signed_loss_proxy_rejects_large_harmful_contributions(self):
        learner = FixedBudgetInteractionLearner(
            n_features=2,
            n_tasks=1,
            candidate_count=2,
            replacement_interval=0,
            utility_decay=0.0,
            step_size_output=0.1,
            scale_robust=True,
        )
        state = learner.init(feature_dim=4, key=jr.key(92)).replace(
            feature_left=jnp.array([0, 2], dtype=jnp.int32),
            feature_right=jnp.array([1, 3], dtype=jnp.int32),
            candidate_left=jnp.array([0, 2], dtype=jnp.int32),
            candidate_right=jnp.array([1, 3], dtype=jnp.int32),
            # Contributions are +1 (helpful) and -3 (harmful).  The latter is
            # larger in magnitude, so abs(w*feature) would rank it backwards.
            output_weights=jnp.array([[1.0, -3.0]], dtype=jnp.float32),
            # Dormant candidate +2 reduces residual 4 -> 2; -2 worsens it.
            candidate_output_weights=jnp.array([[2.0, -2.0]], dtype=jnp.float32),
        )

        result = learner.update(
            state,
            jnp.ones(4, dtype=jnp.float32),
            jnp.array([2.0], dtype=jnp.float32),
        )

        assert float(result.state.utilities[0]) > 0.0
        assert float(result.state.utilities[1]) == 0.0
        assert float(result.state.candidate_utilities[0]) > 0.0
        assert float(result.state.candidate_utilities[1]) == 0.0
        assert float(jnp.max(result.state.utilities)) < 1.0
        assert float(jnp.max(result.state.candidate_utilities)) < 1.0

    def test_candidate_is_bounded_residual_regressor_not_gradient_accumulator(self):
        common = {
            "n_features": 1,
            "n_tasks": 1,
            "candidate_count": 2,
            "replacement_interval": 0,
            "step_size_output": 0.1,
            "use_obgd": False,
        }
        robust = FixedBudgetInteractionLearner(**common, scale_robust=True)
        legacy = FixedBudgetInteractionLearner(**common)

        def initialize(learner: FixedBudgetInteractionLearner):
            return learner.init(feature_dim=6, key=jr.key(95)).replace(
                # The deployed product is identically zero.  Candidate 0 is
                # z*1 and exactly predicts residual 2z; candidate 1 follows an
                # orthogonal four-step sign pattern.
                feature_left=jnp.array([0], dtype=jnp.int32),
                feature_right=jnp.array([1], dtype=jnp.int32),
                candidate_left=jnp.array([2, 4], dtype=jnp.int32),
                candidate_right=jnp.array([3, 5], dtype=jnp.int32),
            )

        def run(learner: FixedBudgetInteractionLearner, state):
            def step(carry, index):
                z = jnp.where(index % 2 == 0, 1.0, -1.0)
                orthogonal = jnp.where((index // 2) % 2 == 0, 1.0, -1.0)
                observation = jnp.array(
                    [0.0, 1.0, z, 1.0, orthogonal, 1.0],
                    dtype=jnp.float32,
                )
                result = learner.update(
                    carry,
                    observation,
                    jnp.array([2.0 * z], dtype=jnp.float32),
                )
                return result.state, None

            final, _ = jax.lax.scan(step, state, jnp.arange(800))
            return final

        robust_final = run(robust, initialize(robust))
        legacy_final = run(legacy, initialize(legacy))

        assert abs(float(robust_final.candidate_output_weights[0, 0]) - 2.0) < 0.15
        assert abs(float(robust_final.candidate_output_weights[0, 1])) < 0.2
        # The default path is intentionally unchanged: it accumulates the
        # deployed gradient rather than fitting a candidate residual model.
        assert float(legacy_final.candidate_output_weights[0, 0]) > 20.0

    def test_missing_target_freezes_robust_candidate_head_row(self):
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=2,
            candidate_count=1,
            replacement_interval=0,
            step_size_output=0.1,
            scale_robust=True,
        )
        state = learner.init(feature_dim=2, key=jr.key(96)).replace(
            feature_left=jnp.array([0], dtype=jnp.int32),
            feature_right=jnp.array([1], dtype=jnp.int32),
            candidate_left=jnp.array([0], dtype=jnp.int32),
            candidate_right=jnp.array([1], dtype=jnp.int32),
            candidate_output_weights=jnp.array([[0.5], [7.0]], dtype=jnp.float32),
        )

        result = learner.update(
            state,
            jnp.array([2.0, 1.0], dtype=jnp.float32),
            jnp.array([4.0, jnp.nan], dtype=jnp.float32),
        )

        assert float(result.state.candidate_output_weights[0, 0]) != 0.5
        assert float(result.state.candidate_output_weights[1, 0]) == 7.0
        assert float(result.state.target_second_moments[1]) == 0.0

    def test_normalizer_memory_is_fixed_and_archive_is_counted(self):
        kwargs = {
            "n_features": 24,
            "n_tasks": 1,
            "candidate_count": 91,
            "candidate_strategy": "all_pairs",
        }
        legacy = FixedBudgetInteractionLearner(**kwargs)
        robust = FixedBudgetInteractionLearner(**kwargs, scale_robust=True)
        legacy_state = legacy.init(feature_dim=14, key=jr.key(93))
        robust_state = robust.init(feature_dim=14, key=jr.key(93))

        legacy_memory = legacy.memory_accounting(legacy_state)
        robust_memory = robust.memory_accounting(robust_state)
        assert robust_memory["active_pair_slots"] == 24
        assert robust_memory["candidate_archive_slots"] == 91
        assert robust_memory["candidate_archive_is_counted"] is True
        assert robust_memory["candidate_output_weight_scalars"] == 91
        assert legacy_state.evidence_idle_steps.shape == (24,)
        assert legacy_state.evidence_idle_steps.dtype == jnp.int32
        assert robust_state.evidence_idle_steps.shape == (24,)
        assert robust_state.evidence_idle_steps.dtype == jnp.int32
        assert legacy_memory["evidence_idle_step_scalars"] == 24
        assert legacy_memory["evidence_idle_step_bytes"] == 96
        assert robust_memory["evidence_idle_step_scalars"] == 24
        assert robust_memory["evidence_idle_step_bytes"] == 96
        assert legacy_state.utility_evidence_streak.shape == (24,)
        assert legacy_state.utility_evidence_streak.dtype == jnp.int32
        assert robust_state.utility_evidence_streak.shape == (24,)
        assert robust_state.utility_evidence_streak.dtype == jnp.int32
        assert legacy_memory["utility_evidence_streak_scalars"] == 24
        assert legacy_memory["utility_evidence_streak_bytes"] == 96
        assert robust_memory["utility_evidence_streak_scalars"] == 24
        assert robust_memory["utility_evidence_streak_bytes"] == 96
        assert legacy_state.active_output_memory_committed.shape == (24,)
        assert legacy_state.active_output_memory_committed.dtype == jnp.bool_
        assert robust_state.active_output_memory_committed.shape == (24,)
        assert robust_state.active_output_memory_committed.dtype == jnp.bool_
        assert legacy_memory["active_output_memory_committed_scalars"] == 24
        assert legacy_memory["active_output_memory_committed_bytes"] == 24
        assert robust_memory["active_output_memory_committed_scalars"] == 24
        assert robust_memory["active_output_memory_committed_bytes"] == 24
        assert legacy_state.relevance_probe_weights.shape == (1, 24)
        assert robust_state.relevance_probe_weights.shape == (1, 24)
        assert legacy_state.relevance_probe_biases.shape == (1,)
        assert robust_state.relevance_probe_biases.shape == (1,)
        assert legacy_memory["relevance_probe_weight_bytes"] == 4 * 24
        assert robust_memory["relevance_probe_weight_bytes"] == 4 * 24
        assert legacy_memory["relevance_probe_bias_bytes"] == 4
        assert robust_memory["relevance_probe_bias_bytes"] == 4
        assert legacy_memory["relevance_probe_bytes"] == 4 * (24 + 1)
        assert robust_memory["relevance_probe_bytes"] == 4 * (24 + 1)
        assert legacy_state.candidate_promotion_evidence_streak.shape == (91,)
        assert robust_state.candidate_promotion_evidence_streak.shape == (91,)
        assert legacy_state.candidate_promotion_evidence_streak.dtype == jnp.int32
        assert robust_state.candidate_promotion_evidence_streak.dtype == jnp.int32
        assert legacy_memory["candidate_promotion_evidence_streak_scalars"] == 91
        assert robust_memory["candidate_promotion_evidence_streak_scalars"] == 91
        assert legacy_memory["candidate_promotion_evidence_streak_bytes"] == 4 * 91
        assert robust_memory["candidate_promotion_evidence_streak_bytes"] == 4 * 91
        assert legacy_state.candidate_reacquisition_required.shape == (91,)
        assert robust_state.candidate_reacquisition_required.shape == (91,)
        assert legacy_state.candidate_reacquisition_required.dtype == jnp.bool_
        assert robust_state.candidate_reacquisition_required.dtype == jnp.bool_
        assert legacy_memory["candidate_reacquisition_required_scalars"] == 91
        assert robust_memory["candidate_reacquisition_required_scalars"] == 91
        assert legacy_memory["candidate_reacquisition_required_bytes"] == 91
        assert robust_memory["candidate_reacquisition_required_bytes"] == 91
        added_lifecycle_bytes = (
            legacy_memory["utility_evidence_streak_bytes"]
            + legacy_memory["active_output_memory_committed_bytes"]
        )
        assert added_lifecycle_bytes == 120
        assert legacy_memory["normalizer_scalars"] == 0
        assert robust_memory["normalizer_scalars"] == 24 + 91 + 1
        assert robust_memory["normalizer_bytes"] == 4 * (24 + 91 + 1)
        added_probe_and_candidate_confirmation_bytes = (
            legacy_memory["relevance_probe_bytes"]
            + legacy_memory["candidate_promotion_evidence_streak_bytes"]
            + legacy_memory["candidate_reacquisition_required_bytes"]
        )
        assert added_probe_and_candidate_confirmation_bytes == 100 + 364 + 91 == 555
        assert legacy_memory["persistent_array_bytes"] == 3_916 + 555 == 4_471
        assert robust_memory["persistent_array_bytes"] == 4_471 + 464 == 4_935
        assert (
            robust_memory["persistent_array_bytes"]
            == legacy_memory["persistent_array_bytes"] + robust_memory["normalizer_bytes"]
        )

        updated = robust.update(
            robust_state,
            jnp.ones(14, dtype=jnp.float32),
            jnp.array([1.0], dtype=jnp.float32),
        ).state
        assert (
            robust.memory_accounting(updated)["persistent_array_bytes"]
            == robust_memory["persistent_array_bytes"]
        )

    def test_config_roundtrip_and_incompatible_raw_future_proxy(self):
        learner = FixedBudgetInteractionLearner(
            n_features=3,
            n_tasks=1,
            scale_robust=True,
            scale_normalizer_decay=0.95,
            scale_normalizer_epsilon=1e-5,
        )
        restored = FixedBudgetInteractionLearner.from_config(learner.to_config())
        assert restored.to_config() == learner.to_config()

        with pytest.raises(ValueError, match="future_utility_mix must be 0"):
            FixedBudgetInteractionLearner(
                n_features=3,
                n_tasks=1,
                scale_robust=True,
                future_utility_mix=0.5,
            )


def run_discovery_traced(
    learner: FixedBudgetInteractionLearner,
    stream: GauntletStream,
    num_steps: int,
    keys: Array,
    init_key: Array,
) -> tuple[Any, Array, Array, Array, Array]:
    """Vmapped gauntlet run that also traces the bank's ctx-pair counts.

    Returns ``(final_states, sq_errors, n_ctx, n_ctx_c, n_ctx_d)`` with the
    trailing four of shape ``(n_seeds, num_steps)``.
    """
    config = stream.config

    def one_seed(key: Array) -> tuple[Any, tuple[Array, Array, Array, Array]]:
        l_state = learner.init(stream.feature_dim, init_key)
        s_state = stream.init(key)

        def step_fn(
            carry: tuple[Any, Any], idx: Array
        ) -> tuple[tuple[Any, Any], tuple[Array, Array, Array, Array]]:
            ls, ss = carry
            timestep, new_ss = stream.step(ss, idx)
            result = learner.update(ls, timestep.observation, timestep.target)
            n_ctx, n_ctx_c, n_ctx_d = bank_context_pair_counts(
                result.state.feature_left, result.state.feature_right, config
            )
            sq = jnp.squeeze(result.errors) ** 2
            return (result.state, new_ss), (sq, n_ctx, n_ctx_c, n_ctx_d)

        (final, _), outputs = jax.lax.scan(step_fn, (l_state, s_state), jnp.arange(num_steps))
        return final, outputs

    finals, (sq, n_ctx, n_ctx_c, n_ctx_d) = jax.vmap(one_seed)(keys)
    return finals, sq, n_ctx, n_ctx_c, n_ctx_d


def run_probe_segment7(stream: GauntletStream, keys: Array, discovery: bool) -> tuple[Any, Array]:
    """Drop a fresh learner into segment 7 (pure nonlinear product target).

    Initializes the stream state with ``step_count`` at the segment-7 start
    (the same jump-in technique the certification suite uses for its oracle
    noise-floor check), so the learner sees only the product-target regime —
    isolating pairwise feature selection from the segment-6 scale shock.
    """
    start = 7 * stream.config.segment_length

    def one_seed(key: Array) -> tuple[Any, Array]:
        s_state = stream.init(key)
        s_state = s_state.replace(step_count=jnp.array(start, dtype=jnp.int32))
        if discovery:
            learner = make_discovery_learner()
            l_state = learner.init(stream.feature_dim, jr.key(LEARNER_INIT_KEY))
        else:
            learner = LinearLearner(optimizer=Autostep())
            l_state = learner.init(stream.feature_dim)

        def step_fn(carry: tuple[Any, Any], idx: Array) -> tuple[tuple[Any, Any], Array]:
            ls, ss = carry
            timestep, new_ss = stream.step(ss, idx)
            result = learner.update(ls, timestep.observation, timestep.target)
            err = result.errors if discovery else result.error
            return (result.state, new_ss), jnp.squeeze(err) ** 2

        (final, _), sq = jax.lax.scan(step_fn, (l_state, s_state), jnp.arange(PROBE_STEPS))
        return final, sq

    return jax.vmap(one_seed)(keys)


@pytest.fixture(scope="module")
def config() -> GauntletConfig:
    return GauntletConfig()


@pytest.fixture(scope="module")
def stream(config: GauntletConfig) -> GauntletStream:
    return GauntletStream(config)


@pytest.fixture(scope="module")
def keys():
    """The exact direct-key development schedule declared above."""
    return jnp.stack([jr.key(seed) for seed in DEVELOPMENT_SEEDS])


@pytest.fixture(scope="module")
def raw_sq(stream: GauntletStream, config: GauntletConfig, keys):
    """Raw-observation Autostep linear learner (the no-discovery baseline)."""
    return run_gauntlet_batched(LinearLearner(optimizer=Autostep()), stream, config.num_steps, keys)


@pytest.fixture(scope="module")
def discovery_run(stream: GauntletStream, config: GauntletConfig, keys):
    """Traced full-program run of the calibrated discovery learner."""
    return run_discovery_traced(
        make_discovery_learner(),
        stream,
        config.num_steps,
        keys,
        jr.key(LEARNER_INIT_KEY),
    )


@pytest.fixture(scope="module")
def noret_run(stream: GauntletStream, config: GauntletConfig, keys):
    """Ablation: same robust learner without the utility retention floor."""
    return run_discovery_traced(
        make_discovery_learner(retention=None),
        stream,
        config.num_steps,
        keys,
        jr.key(LEARNER_INIT_KEY),
    )


@pytest.fixture(scope="module")
def probe7(stream: GauntletStream, keys):
    """Segment-7 jump-in probes: (discovery finals, discovery sq, raw sq)."""
    disc_finals, disc_sq = run_probe_segment7(stream, keys, discovery=True)
    _, raw_probe_sq = run_probe_segment7(stream, keys, discovery=False)
    return disc_finals, disc_sq, raw_probe_sq


class TestDiscoveryMemory:
    def test_savings_beat_raw_observation_learner(
        self, discovery_run, raw_sq, config: GauntletConfig
    ):
        """(a) The discovery learner's re-acquisition savings on BOTH recurring
        tasks far exceed the raw-observation linear learner's.

        The raw learner is a diagnostic lower bound, not a capacity-matched
        comparator; the evidence protocol uses matched legacy OBGD instead.
        Absolute savings floors prevent the relative check from passing merely
        because the raw baseline is poor.
        """
        _, disc_sq, _, _, _ = discovery_run
        disc_score = gauntlet_scorecard(disc_sq, config)
        raw_score = gauntlet_scorecard(raw_sq, config)

        disc_c = float(jnp.median(disc_score["savings_c"]))
        disc_d = float(jnp.median(disc_score["savings_d"]))
        raw_c = float(jnp.median(raw_score["savings_c"]))
        raw_d = float(jnp.median(raw_score["savings_d"]))

        assert disc_c > 1.5 * raw_c
        assert disc_d > 1.5 * raw_d
        assert disc_c > 8.0
        assert disc_d > 3.0

    def test_reentry_error_matches_oracle_class(self, discovery_run, config: GauntletConfig):
        """The discovered representation re-enters task C near its old
        solution.  Frozen-design development median early recurrence MSE was
        0.672 versus 8.49 on first exposure.  The frozen absolute safeguard
        requires recurrence at or below 2.0 while confirming acquisition was
        nontrivial."""
        _, disc_sq, _, _, _ = discovery_run
        score = gauntlet_scorecard(disc_sq, config)
        assert float(jnp.median(score["early_mse_c_recur"])) <= 2.0
        assert float(jnp.median(score["early_mse_c_first"])) > 3.0

    def test_retention_snapshot_is_before_final_c_reacquisition(
        self,
        discovery_run,
        noret_run,
        config: GauntletConfig,
    ):
        """Expose separate C/D retention differences at end segment 7.

        The frozen evidence artifact requires all-seed non-inferiority and a
        strictly positive paired-mean interval for both differences at this
        pre-reacquisition snapshot.  This development test exposes the raw
        vectors without claiming held-out evidence.
        """
        _, _, _, retained_c, retained_d = discovery_run
        _, _, _, no_retention_c, no_retention_d = noret_run
        before_final_c = 8 * config.segment_length - 1
        paired_c = retained_c[:, before_final_c] - no_retention_c[:, before_final_c]
        paired_d = retained_d[:, before_final_c] - no_retention_d[:, before_final_c]
        assert paired_c.shape == (N_SEEDS,)
        assert paired_d.shape == (N_SEEDS,)
        assert bool(jnp.all(jnp.isfinite(paired_c)))
        assert bool(jnp.all(jnp.isfinite(paired_d)))


class TestDiscoveredRepresentation:
    def test_bank_contains_context_interaction_features(
        self, discovery_run, config: GauntletConfig
    ):
        """(b) At the end of the recurrence phase (segment 5) every seed's
        active bank pairs context channels with x channels.

        The historical slot-count audit held at least 12 relevant slots after
        segment 4 and 13 after segment 5.  The tightened floor below requires
        seven unique products per seed, a substantial fraction of the
        supplied oracle family, and must be revalidated on direct keys 8--15.
        """
        _, _, n_ctx, n_ctx_c, n_ctx_d = discovery_run
        length = config.segment_length
        end_seg4 = 5 * length - 1
        end_seg5 = 6 * length - 1

        for counts in (n_ctx, n_ctx_c, n_ctx_d):
            assert counts.shape == (N_SEEDS, config.num_steps)
            assert int(jnp.min(counts)) >= 0
        assert int(jnp.max(n_ctx_c[:, end_seg4])) <= config.relevant_dim
        assert int(jnp.max(n_ctx_d[:, end_seg5])) <= config.relevant_dim


class TestPairwiseFeatureFinding:
    def test_nonlinear_probe_builds_product_features(self, probe7, config: GauntletConfig):
        """(d) Dropped fresh into the pure nonlinear product segment, the
        discovery learner constructs the generating pairs and beats the raw
        linear learner by an order of magnitude.

        The test requires low absolute error, a 3.3x advantage over the raw
        linear learner, at least two generating pairs per seed, and all four
        at the median.  This remains a closed finite hypothesis-space probe.
        """
        disc_finals, disc_sq, raw_probe_sq = probe7
        tail = PROBE_STEPS // 2
        disc_tail = jnp.mean(disc_sq[:, tail:], axis=-1)
        raw_tail = jnp.mean(raw_probe_sq[:, tail:], axis=-1)

        assert float(jnp.median(disc_tail)) < 0.3
        assert float(jnp.median(disc_tail)) < 0.3 * float(jnp.median(raw_tail))

        fl = disc_finals.feature_left
        fr = disc_finals.feature_right
        generating = (fl % 2 == 0) & (fr == fl + 1) & (fl < config.relevant_dim)
        nl_counts = jnp.sum(generating, axis=-1)
        assert int(jnp.min(nl_counts)) >= 2
        assert float(jnp.median(nl_counts)) >= 4.0

    def test_nonlinear_segment_beats_raw_in_full_program(
        self, discovery_run, raw_sq, config: GauntletConfig
    ):
        """(d cont.) On the full program the discovery learner's segment-7
        error also beats the raw linear learner.  The fresh probe above is the
        cleaner feature-selection diagnostic because it removes earlier-phase
        interference."""
        _, disc_sq, _, _, _ = discovery_run
        disc_nl = float(jnp.median(gauntlet_scorecard(disc_sq, config)["nonlinear_mse"]))
        raw_nl = float(jnp.median(gauntlet_scorecard(raw_sq, config)["nonlinear_mse"]))
        assert disc_nl < 0.6 * raw_nl


class TestScaleRobustClosure:
    def test_scale_shock_retains_discovered_bank(self, discovery_run, config: GauntletConfig):
        """The robust learner survives scale shock and reacquires task C.

        Ratio gates are paired with absolute acquisition, scale-tail, and
        final-recurrence errors so catastrophic acquisition cannot manufacture
        apparent savings (the rejected Design A did exactly that).  ``max``
        assertions below are explicitly per-seed; the others are aggregate
        medians over the calibration batch.
        """
        _, disc_sq, n_ctx, n_ctx_c, n_ctx_d = discovery_run
        score = gauntlet_scorecard(disc_sq, config)
        length = config.segment_length
        first_early = jnp.mean(
            disc_sq[:, 2 * length : 2 * length + 200],
            axis=1,
        )
        first_tail = jnp.mean(
            disc_sq[:, 3 * length - 500 : 3 * length],
            axis=1,
        )
        recur_early = jnp.mean(
            disc_sq[:, 4 * length : 4 * length + 200],
            axis=1,
        )
        recur_tail = jnp.mean(
            disc_sq[:, 5 * length - 500 : 5 * length],
            axis=1,
        )
        scaled_tail = jnp.mean(
            disc_sq[:, 7 * length - 500 : 7 * length],
            axis=1,
        )
        final_early = jnp.mean(
            disc_sq[:, 8 * length : 8 * length + 200],
            axis=1,
        )
        final_tail = jnp.mean(
            disc_sq[:, 9 * length - 500 : 9 * length],
            axis=1,
        )

        # Per-seed stability and representation safeguards.
        assert int(jnp.sum(score["nan_steps"])) == 0
        for counts in (n_ctx, n_ctx_c, n_ctx_d):
            assert int(jnp.min(counts)) >= 0
        assert int(jnp.max(n_ctx_c)) <= config.relevant_dim
        assert int(jnp.max(n_ctx_d)) <= config.relevant_dim
        assert float(jnp.max(scaled_tail)) <= 50.0
        assert float(jnp.max(final_tail)) <= 0.1

        # Aggregate median acquisition, recurrence, and scale safeguards.
        assert float(jnp.median(first_early)) <= 15.0
        assert float(jnp.median(first_tail)) <= 2.0
        assert float(jnp.median(recur_early)) <= 2.0
        assert float(jnp.median(recur_tail)) <= 2.0
        assert float(jnp.median(scaled_tail)) <= 10.0
        assert float(jnp.median(final_early)) <= 3.0
        assert float(jnp.median(final_tail)) <= 0.1
        assert float(jnp.median(score["nonlinear_mse"])) <= 0.1
        assert float(jnp.median(score["savings_c"])) >= 8.0
        assert float(jnp.median(score["savings_d"])) >= 5.0
        assert float(jnp.median(score["savings_c_final"])) >= 5.0
