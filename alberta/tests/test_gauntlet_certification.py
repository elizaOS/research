"""Component diagnostics for the Alberta Gauntlet (streams/gauntlet.py).

The gauntlet is a compact supervised continual-learning stream whose scorecard
turns several early-plan properties into scalars. These tests provide
calibrated mechanism evidence, not an integrated Alberta Plan certification:
they use eight development seeds, medians without confidence intervals, and a
hand-designed oracle context-gated representation. Measured margins are:

- P1 tracking:      meta-learned step-sizes (IDBD) beat the best fixed
                    step-size from a sweep on the drift segment
                    (measured 0.044 vs 0.155 MSE).
- P2 relevance:     IDBD's learned per-weight step-sizes rank truly relevant
                    input dims above irrelevant ones.
- P3 plasticity:    abrupt task switches are recovered from quickly, and the
                    second switch is not slower than the first.
- P4 memory:        over context-gated features, task recurrence shows large
                    re-acquisition savings (measured ~10-26x) while the
                    fresh-reinit twin scores ~1.0 and the raw-observation
                    learner shows near-zero savings — retention comes from
                    the supplied representation in this diagnostic.
- P5 retention:     savings on the final recurrence survive an intervening
                    nonlinear task (measured ~36x).
- P6 stability:     Autostep survives the 10x input-scale segment with zero
                    non-finite steps; IDBD's divergence there is *detected*
                    by the gauntlet (documented pathology motivating
                    SwiftTD-class optimizers).

Total runtime target: well under two minutes on CPU (all runs are
scan+vmap-compiled and shared across tests via module-scoped fixtures).
"""

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array
from jaxtyping import Float

from alberta_framework.core.learners import LinearLearner, run_learning_loop
from alberta_framework.core.optimizers import IDBD, LMS, Autostep
from alberta_framework.core.swift_td import SwiftTD, SwiftTDState
from alberta_framework.core.types import StepSizeTrackingConfig
from alberta_framework.streams.gauntlet import (
    NUM_SEGMENTS,
    SEGMENT_NAMES,
    ContextGatedFeatures,
    GauntletConfig,
    GauntletStream,
    best_fixed_alpha_errors,
    gauntlet_scorecard,
    run_gauntlet,
    run_gauntlet_batched,
    savings_ratio,
    segment_mse,
)

N_SEEDS = 8


@pytest.fixture(scope="module")
def config() -> GauntletConfig:
    return GauntletConfig()


@pytest.fixture(scope="module")
def stream(config: GauntletConfig) -> GauntletStream:
    return GauntletStream(config)


@pytest.fixture(scope="module")
def gated(stream: GauntletStream) -> ContextGatedFeatures:
    return ContextGatedFeatures(stream)


@pytest.fixture(scope="module")
def keys():
    return jr.split(jr.key(7), N_SEEDS)


@pytest.fixture(scope="module")
def idbd_raw_sq(stream: GauntletStream, config: GauntletConfig, keys):
    return run_gauntlet_batched(
        LinearLearner(optimizer=IDBD()), stream, config.num_steps, keys
    )


@pytest.fixture(scope="module")
def autostep_raw_sq(stream: GauntletStream, config: GauntletConfig, keys):
    return run_gauntlet_batched(
        LinearLearner(optimizer=Autostep()), stream, config.num_steps, keys
    )


@pytest.fixture(scope="module")
def autostep_gated_sq(gated: ContextGatedFeatures, config: GauntletConfig, keys):
    return run_gauntlet_batched(
        LinearLearner(optimizer=Autostep()), gated, config.num_steps, keys
    )


@pytest.fixture(scope="module")
def reinit_gated_sq(gated: ContextGatedFeatures, config: GauntletConfig, keys):
    return run_gauntlet_batched(
        LinearLearner(optimizer=Autostep()),
        gated,
        config.num_steps,
        keys,
        reinit_each_segment=True,
        segment_length=config.segment_length,
    )


class TestGauntletStreamMechanics:
    def test_program_shape_and_names(self, config: GauntletConfig):
        assert len(SEGMENT_NAMES) == NUM_SEGMENTS
        assert config.num_steps == NUM_SEGMENTS * config.segment_length
        assert config.observation_dim == config.relevant_dim + config.irrelevant_dim + 2

    def test_segment_schedule(self, stream: GauntletStream, config: GauntletConfig):
        length = config.segment_length
        assert int(stream.segment_of(jnp.array(0))) == 0
        assert int(stream.segment_of(jnp.array(length - 1))) == 0
        assert int(stream.segment_of(jnp.array(length))) == 1
        assert int(stream.segment_of(jnp.array(7 * length + 5))) == 7
        # Beyond the program the schedule clips to the final segment.
        assert int(stream.segment_of(jnp.array(100 * length))) == NUM_SEGMENTS - 1

    def test_step_shapes_and_dtypes(self, stream: GauntletStream, config: GauntletConfig):
        state = stream.init(jr.key(0))
        timestep, new_state = stream.step(state, jnp.array(0))
        assert timestep.observation.shape == (config.observation_dim,)
        assert timestep.target.shape == (1,)
        assert timestep.observation.dtype == jnp.float32
        assert int(new_state.step_count) == 1

    def test_linear_tasks_zero_on_irrelevant_dims(self, stream: GauntletStream):
        state = stream.init(jr.key(1))
        cfg = stream.config
        for w in (state.w_a, state.w_c, state.w_d):
            assert jnp.all(w[cfg.relevant_dim :] == 0.0)
            assert jnp.any(w[: cfg.relevant_dim] != 0.0)

    def test_recurrence_segments_reuse_exact_task_weights(self, stream: GauntletStream):
        """Segments 2, 4, 6, and 8 all use the identical task C weights."""
        state = stream.init(jr.key(2))
        for seg in (2, 4, 6, 8):
            w = stream.true_linear_weights(state, jnp.array(seg))
            assert jnp.array_equal(w, state.w_c)
        for seg in (3, 5):
            w = stream.true_linear_weights(state, jnp.array(seg))
            assert jnp.array_equal(w, state.w_d)

    def test_context_channels_follow_task_identity(
        self, stream: GauntletStream, config: GauntletConfig
    ):
        """Mean context per segment matches the (noisy) context table."""
        state = stream.init(jr.key(3))
        num_steps = config.num_steps
        import jax

        def step_fn(carry, idx):
            ts, new_state = stream.step(carry, idx)
            return new_state, ts.observation[config.input_dim :]

        _, ctx = jax.lax.scan(step_fn, state, jnp.arange(num_steps))
        length = config.segment_length
        expected = {2: (1.0, 0.0), 3: (0.0, 1.0), 4: (1.0, 0.0), 7: (0.0, 0.0)}
        for seg, (c0, c1) in expected.items():
            seg_ctx = ctx[seg * length : (seg + 1) * length]
            assert jnp.allclose(jnp.mean(seg_ctx[:, 0]), c0, atol=0.02)
            assert jnp.allclose(jnp.mean(seg_ctx[:, 1]), c1, atol=0.02)

    def test_oracle_noise_floor(self, stream: GauntletStream, config: GauntletConfig):
        """Predicting with the true task weights achieves the noise floor."""
        import jax

        state = stream.init(jr.key(4))
        # Jump to segment 2 (stationary task C).
        state = state.replace(
            step_count=jnp.array(2 * config.segment_length, dtype=jnp.int32)
        )

        def step_fn(carry, idx):
            ts, new_state = stream.step(carry, idx)
            pred = jnp.dot(new_state.w_c, ts.observation[: config.input_dim])
            return new_state, (jnp.squeeze(ts.target) - pred) ** 2

        _, sq = jax.lax.scan(step_fn, state, jnp.arange(2000))
        assert jnp.mean(sq) == pytest.approx(config.noise_floor, rel=0.15)

    def test_gated_wrapper_dims_and_exclusivity(
        self, stream: GauntletStream, gated: ContextGatedFeatures, config: GauntletConfig
    ):
        d = config.input_dim
        assert gated.feature_dim == 3 * d + 2
        state = gated.init(jr.key(5))
        # Segment 2 (task C): base-gated block ~0, c0 block ~x, c1 block ~0.
        state = state.replace(
            step_count=jnp.array(2 * config.segment_length, dtype=jnp.int32)
        )
        ts, _ = gated.step(state, jnp.array(0))
        base_block = ts.observation[:d]
        c0_block = ts.observation[d + 2 : 2 * d + 2]
        c1_block = ts.observation[2 * d + 2 :]
        assert float(jnp.mean(jnp.abs(base_block))) < 0.2  # ~context-noise leakage
        assert float(jnp.mean(jnp.abs(c0_block))) > 0.5
        assert float(jnp.mean(jnp.abs(c1_block))) < 0.2


class TestCertificationLadder:
    def test_p1_tracking_meta_beats_best_fixed_alpha(
        self, stream: GauntletStream, config: GauntletConfig, keys, idbd_raw_sq
    ):
        """P1: IDBD beats the best fixed step-size on the drift segment.

        Calibration (10 seeds): IDBD 0.044 vs best-LMS 0.155 tail MSE.
        """
        lms_sq, best_alpha = best_fixed_alpha_errors(
            lambda a: LinearLearner(optimizer=LMS(step_size=a)),
            stream,
            config.num_steps,
            keys,
            step_sizes=(0.003, 0.01, 0.03),
        )
        idbd_track = segment_mse(idbd_raw_sq, 1, config.segment_length)
        lms_track = segment_mse(lms_sq, 1, config.segment_length)
        assert float(jnp.median(idbd_track)) < float(jnp.median(lms_track))
        # Paired per-seed wins: IDBD must win on at least 6 of 8 seeds.
        assert int(jnp.sum(idbd_track < lms_track)) >= N_SEEDS - 2

    def test_p2_step_size_relevance(self, config: GauntletConfig):
        """P2: after the drift segment, IDBD's per-weight step-sizes rank
        every relevant input dim above every irrelevant one on average."""
        stream = GauntletStream(config)
        learner = LinearLearner(optimizer=IDBD())
        # Run segments 0-1 only (stationary + drift; irrelevant dims carry
        # zero weight throughout).
        result = run_learning_loop(
            learner,
            stream,
            num_steps=2 * config.segment_length,
            key=jr.key(11),
            step_size_tracking=StepSizeTrackingConfig(interval=config.segment_length),
        )
        history = result[2]
        final_alphas = history.step_sizes[-1]
        relevant = final_alphas[: config.relevant_dim]
        irrelevant = final_alphas[config.relevant_dim : config.input_dim]
        assert float(jnp.min(relevant)) > float(jnp.max(irrelevant))

    def test_p3_plasticity_rapid_recovery(
        self, autostep_raw_sq, config: GauntletConfig
    ):
        """P3: both abrupt switches are recovered from within half a segment,
        and the second event is not slower than 2x the first (no rapid
        plasticity decay).  Calibration: ~620 steps each."""
        score = gauntlet_scorecard(autostep_raw_sq, config)
        rec_c = float(jnp.median(score["recovery_steps_c"]))
        rec_d = float(jnp.median(score["recovery_steps_d"]))
        assert rec_c < config.segment_length / 2
        assert rec_d < config.segment_length / 2
        assert rec_d < 2.0 * rec_c

    def test_p4_memory_savings_from_gated_representation(
        self, autostep_gated_sq, reinit_gated_sq, autostep_raw_sq, config: GauntletConfig
    ):
        """P4: the context-gated learner shows large re-acquisition savings on
        both recurring tasks; the fresh-reinit twin (perfect plasticity, zero
        memory) scores ~1; the raw-observation learner shows near-zero
        savings.  Retention is a property of the representation.

        Calibration (10 seeds): gated savings_c 9.8 / savings_d 15.1;
        reinit 1.00 / 1.01; raw 1.3 / 1.2.
        """
        gated_score = gauntlet_scorecard(autostep_gated_sq, config)
        reinit_score = gauntlet_scorecard(reinit_gated_sq, config)
        raw_score = gauntlet_scorecard(autostep_raw_sq, config)

        assert float(jnp.median(gated_score["savings_c"])) > 3.0
        assert float(jnp.median(gated_score["savings_d"])) > 3.0
        # The no-memory control sits at ~1 by construction.
        assert 0.6 < float(jnp.median(reinit_score["savings_c"])) < 1.6
        # Raw observations cannot retain contradictory tasks.
        assert float(jnp.median(raw_score["savings_c"])) < 2.0
        # Direct retention evidence: gated re-entry error is far below the
        # no-memory twin's re-entry error (calibration: 0.11 vs 1.72).
        assert float(jnp.median(gated_score["early_mse_c_recur"])) < 0.25 * float(
            jnp.median(reinit_score["early_mse_c_recur"])
        )

    def test_p5_retention_survives_nonlinear_interference(
        self, autostep_gated_sq, config: GauntletConfig
    ):
        """P5: savings on the final task C recurrence, after the intervening
        nonlinear segment, remain large (calibration: ~36x)."""
        score = gauntlet_scorecard(autostep_gated_sq, config)
        assert float(jnp.median(score["savings_c_final"])) > 3.0

    def test_p6_stability_autostep_no_nans(
        self, autostep_raw_sq, autostep_gated_sq, config: GauntletConfig
    ):
        """P6: Autostep survives the whole program, including the 10x-scale
        segment, with zero non-finite steps — and its scaled-segment error
        stays small on raw observations (calibration: 0.014)."""
        raw_score = gauntlet_scorecard(autostep_raw_sq, config)
        gated_score = gauntlet_scorecard(autostep_gated_sq, config)
        assert int(jnp.sum(raw_score["nan_steps"])) == 0
        assert int(jnp.sum(gated_score["nan_steps"])) == 0
        assert float(jnp.median(raw_score["scaled_mse"])) < 0.1

    def test_p6_gauntlet_detects_idbd_scale_divergence(
        self, idbd_raw_sq, config: GauntletConfig
    ):
        """The gauntlet *detects* IDBD's known divergence under a 10x input
        scale shift (segments 6+): a documented pathology of the 1992
        algorithm that motivates SwiftTD-class successors.  Most seeds are
        NaN-free until the scale segment, then diverge there."""
        length = config.segment_length
        finite_before = jnp.all(
            jnp.isfinite(idbd_raw_sq[..., : 6 * length]), axis=-1
        )
        nan_after = jnp.any(~jnp.isfinite(idbd_raw_sq[..., 6 * length :]), axis=-1)
        # The scale shift is what kills IDBD: among seeds that were healthy
        # through segment 5, the majority go non-finite at segment 6+.
        n_healthy = int(jnp.sum(finite_before))
        assert n_healthy >= N_SEEDS - 2
        assert int(jnp.sum(finite_before & nan_after)) >= n_healthy // 2

    def test_savings_ratio_of_memoryless_run_is_one(self, reinit_gated_sq, config):
        """The savings metric itself is calibrated: a fresh-reinit run scores
        ~1.0 on every recurrence pair (its early-window errors are i.i.d.)."""
        for first, revisit in ((2, 4), (3, 5), (2, 8)):
            ratio = float(
                jnp.median(
                    savings_ratio(reinit_gated_sq, first, revisit, config.segment_length)
                )
            )
            assert 0.6 < ratio < 1.6


class TestHarness:
    def test_run_gauntlet_single_seed_shapes(
        self, stream: GauntletStream, config: GauntletConfig
    ):
        final_state, sq = run_gauntlet(
            LinearLearner(optimizer=Autostep()), stream, 500, jr.key(0)
        )
        assert sq.shape == (500,)
        assert bool(jnp.all(jnp.isfinite(sq)))
        assert int(final_state.step_count) == 500

    def test_reinit_requires_segment_length(self, stream: GauntletStream):
        with pytest.raises(ValueError, match="segment_length"):
            run_gauntlet(
                LinearLearner(optimizer=Autostep()),
                stream,
                100,
                jr.key(0),
                reinit_each_segment=True,
            )


# =============================================================================
# SwiftTD diagnostic
# =============================================================================


@chex.dataclass(frozen=True)
class _SwiftTDLearnerState:
    """Weights + bias + optimizer state for the supervised SwiftTD adapter."""

    weights: Float[Array, " feature_dim"]
    bias: Float[Array, ""]
    optimizer_state: SwiftTDState


@chex.dataclass(frozen=True)
class _SwiftTDResult:
    """Update result mirroring the ``.state`` / ``.error`` learner protocol."""

    state: _SwiftTDLearnerState
    error: Float[Array, ""]


class _SwiftTDSupervisedLearner:
    """Adapter running SwiftTD under :func:`run_gauntlet`'s learner protocol.

    SwiftTD is a TD optimizer (``update(state, td_error, obs, next_obs,
    gamma)``); in the supervised limit (``gamma = 0``, fresh traces every
    step) it reduces to per-feature bounded LMS with IDBD-style
    meta-learning.  This wrapper owns the weights/bias and exposes the
    ``init(feature_dim)`` / ``update(state, observation, target)`` surface
    that :class:`LinearLearner` provides.
    """

    def __init__(self, optimizer: SwiftTD):
        self._optimizer = optimizer

    def init(self, feature_dim: int) -> _SwiftTDLearnerState:
        return _SwiftTDLearnerState(
            weights=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias=jnp.array(0.0, dtype=jnp.float32),
            optimizer_state=self._optimizer.init(feature_dim),
        )

    def update(
        self, state: _SwiftTDLearnerState, observation: Array, target: Array
    ) -> _SwiftTDResult:
        prediction = jnp.dot(state.weights, observation) + state.bias
        error = jnp.squeeze(target) - prediction
        upd = self._optimizer.update(
            state.optimizer_state, error, observation, observation, jnp.array(0.0)
        )
        new_state = _SwiftTDLearnerState(
            weights=state.weights + upd.weight_delta,
            bias=state.bias + upd.bias_delta,
            optimizer_state=upd.new_state,
        )
        return _SwiftTDResult(state=new_state, error=error)


@pytest.fixture(scope="module")
def swift_raw_sq(stream: GauntletStream, config: GauntletConfig, keys):
    """SwiftTD over the raw gauntlet program (shared across the class).

    Hyperparameters (calibrated on the gauntlet, 8 seeds x 3 seed batches):
    ``eta = 0.2`` keeps the overshoot bound tight; ``step_size_decay =
    0.999`` is gentler than the paper's 0.99 because the 10x-scale segment
    makes ``phi_i^2 ~ 100``, so each bound trigger multiplies step-sizes by
    ``decay**(phi_i^2)`` — 0.99 would crush them to ``eta_min`` within a
    few steps while 0.999 decays just enough to restore the bound.
    """
    learner = _SwiftTDSupervisedLearner(
        SwiftTD(
            initial_step_size=1e-2,
            meta_step_size=3e-3,
            eta=0.2,
            step_size_decay=0.999,
        )
    )
    return run_gauntlet_batched(learner, stream, config.num_steps, keys)


class TestSwiftTDCertification:
    """SwiftTD (Javed, Sharifnassab & Sutton, RLC 2024) on the gauntlet.

    All numbers below were calibrated on 8 seeds (``jr.key(7)``, the module
    fixture) and cross-checked on two more 8-seed batches (``jr.key(123)``,
    ``jr.key(2026)``):

    - nan_steps: 0 on every seed of every batch — including the 10x-scale
      segment where IDBD diverges (see
      ``test_p6_gauntlet_detects_idbd_scale_divergence``).
    - drift tail MSE: median 0.048-0.052 vs IDBD 0.042-0.044 (paired ratio
      1.14-1.18, all 8 seeds within 2.0x) and best fixed-alpha LMS 0.155.
    - scaled-segment tail MSE: median 0.0106-0.0109 vs Autostep 0.0121
      (noise floor 0.01).  A rare seed spikes during re-adaptation (max 1.2
      seen once in 24 seeds), so accuracy assertions are median-based;
      only finiteness is asserted per-step.
    """

    def test_p6_swift_survives_entire_program(self, swift_raw_sq, config: GauntletConfig):
        """P6 stability: zero non-finite steps over the whole nine-segment
        program, including the 10x input-scale segment that kills IDBD.
        Calibration: 0 non-finite steps on all 24 seeds tried."""
        score = gauntlet_scorecard(swift_raw_sq, config)
        assert int(jnp.sum(score["nan_steps"])) == 0
        assert bool(jnp.all(jnp.isfinite(swift_raw_sq)))

    def test_tracking_within_factor_of_idbd_and_beats_best_lms(
        self, swift_raw_sq, idbd_raw_sq, config: GauntletConfig
    ):
        """P1 tracking: SwiftTD's drift-segment tail MSE is within 2x of
        IDBD's and beats the best fixed-alpha LMS.

        Calibration: SwiftTD median 0.048-0.052; IDBD median 0.042-0.044
        (measured ratio 1.14-1.18, threshold 2.0); best fixed-alpha LMS
        0.155 (see ``test_p1_tracking_meta_beats_best_fixed_alpha``) —
        the 0.10 absolute bound sits ~1.9x above SwiftTD's measurement and
        ~1.6x below the LMS level.
        """
        swift_track = segment_mse(swift_raw_sq, 1, config.segment_length)
        idbd_track = segment_mse(idbd_raw_sq, 1, config.segment_length)
        assert float(jnp.median(swift_track)) < 2.0 * float(jnp.median(idbd_track))
        # Paired per-seed: nearly all seeds stay within 2.5x of IDBD
        # (calibration: 8/8 within 2.0x on all three seed batches).
        assert int(jnp.sum(swift_track < 2.5 * idbd_track)) >= N_SEEDS - 2
        # Better than the best fixed step-size from the sweep (0.155).
        assert float(jnp.median(swift_track)) < 0.10

    def test_scaled_segment_competitive_with_autostep(
        self, swift_raw_sq, autostep_raw_sq, config: GauntletConfig
    ):
        """P6 accuracy under scale shift: SwiftTD's 10x-segment tail MSE is
        competitive with Autostep's (the repo's stability reference).

        Calibration: SwiftTD median 0.0106-0.0109; Autostep median 0.0121;
        noise floor 0.01.  Thresholds: 3x Autostep (measured ratio ~0.9)
        and 0.05 absolute (~4.5x above measurement).
        """
        swift_scaled = segment_mse(swift_raw_sq, 6, config.segment_length)
        autostep_scaled = segment_mse(autostep_raw_sq, 6, config.segment_length)
        assert float(jnp.median(swift_scaled)) < 3.0 * float(jnp.median(autostep_scaled))
        assert float(jnp.median(swift_scaled)) < 0.05
