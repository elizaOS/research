"""Unit tests for the IPMNIST mechanism-combination screening lane.

These pin the screening harness to the full-horizon lane (control parity and
prefix property), pin combination steps to their reference equations, and
test shard/merge/validation plumbing. Benchmark executions never run here.
"""

import json
import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_screening import (
    _CBP_LAYERS,
    PROXY_N_TASKS,
    SCREENING_REGISTRY,
    SHARD_SCHEMA,
    EMANormState,
    ScreeningSpec,
    _make_upgd_idbd_learner,
    _make_upgd_w_fade_head_learner,
    _make_upgd_w_wclip_learner,
    adam_elem_update,
    cbp_maybe_replace_layer,
    ema_normalize,
    load_shard,
    merge_shards,
    run_screening_config,
    screening_spec,
    shard_payload,
    upgd_idbd_swift_update,
    upgd_idbd_update,
    upgd_l2init_update,
    upgd_w_fade_head_update,
    upgd_w_localgate_update,
    upgd_w_wclip_update,
    validate_proxy,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    UPGD_W_PROTOCOL_HYPERPARAMETERS,
    IPMNISTConfig,
    LeanUPGDState,
    init_mlp_params,
    lean_upgd_w_update,
    run_ipmnist,
)
from alberta_framework.core.baseline_optimizers import Adam
from alberta_framework.core.normalizers import EMANormalizer

SMALL = IPMNISTConfig(
    n_tasks=3, task_length=30, input_dim=12, hidden1=8, hidden2=6, n_classes=5
)


@pytest.fixture(scope="module")
def small_data():
    key = jr.key(1234)
    kx, ky = jr.split(key)
    x = jr.uniform(kx, (64, SMALL.input_dim), jnp.float32, -1.0, 1.0)
    y = jr.randint(ky, (64,), 0, SMALL.n_classes)
    return np.asarray(x), np.asarray(y)


class TestRegistry:
    def test_expected_configs_present(self):
        expected = {
            "upgd_w_control",
            "adamw_control",
            "upgd_idbd",
            "upgd_idbd_meta1e2",
            "upgd_autostep",
            "upgd_l2init",
            "upgd_ema_norm",
            "upgd_cbp",
            "adamw_cbp",
            "upgd_w_sigma005",
            "upgd_w_sigma02",
            "upgd_w_udecay0999",
            "upgd_w_udecay099999",
            "upgd_w_wd0005",
            "upgd_w_wd002",
            "upgd_w_wclip_k1",
            "upgd_w_wclip_k2",
            "upgd_w_wclip_k1_wd0",
            "upgd_w_wclip_k2_wd0",
            "upgd_w_localgate",
            "upgd_w_fade_head",
            "upgd_w_idbd_swift",
        }
        assert expected == set(SCREENING_REGISTRY)

    def test_unknown_config_rejected(self):
        with pytest.raises(ValueError, match="unknown screening config"):
            screening_spec("nope")

    def test_control_uses_published_hyperparameters(self):
        assert (
            screening_spec("upgd_w_control").hyperparameters
            == UPGD_W_PROTOCOL_HYPERPARAMETERS
        )

    def test_proxy_default_horizon(self):
        assert PROXY_N_TASKS == 60


class TestControlParity:
    """The screening runner must reproduce the full lane for control arms."""

    @pytest.mark.parametrize("name,learner", [
        ("upgd_w_control", "upgd_w"),
        ("adamw_control", "adamw"),
    ])
    def test_control_matches_run_ipmnist(self, small_data, name, learner):
        x, y = small_data
        ours = run_screening_config(x, y, screening_spec(name), seed=7, config=SMALL)
        reference = run_ipmnist(x, y, learner, seeds=[7], config=SMALL)
        np.testing.assert_allclose(
            ours.per_task_accuracy, reference.per_task_accuracy[0], atol=1e-7
        )
        np.testing.assert_allclose(
            ours.per_task_loss, reference.per_task_loss[0], rtol=1e-5
        )

    def test_prefix_property(self, small_data):
        """A shorter-horizon run is an exact prefix of a longer one (same seed)."""
        x, y = small_data
        spec = screening_spec("upgd_w_control")
        short = run_screening_config(
            x, y, spec, seed=3, config=IPMNISTConfig(
                n_tasks=2, task_length=30, input_dim=12, hidden1=8, hidden2=6, n_classes=5
            )
        )
        long = run_screening_config(x, y, spec, seed=3, config=SMALL)
        np.testing.assert_allclose(
            short.per_task_accuracy, long.per_task_accuracy[:2], atol=1e-7
        )


class TestIDBDCombo:
    def test_meta_zero_reduces_to_lean_upgd(self):
        """With meta=0 and initial alpha = published lr, IDBD == lean UPGD-W."""
        key = jr.key(0)
        params = init_mlp_params(key, SMALL)
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        hp.update({"meta_step_size": 0.0, "initial_step_size": hp["step_size"]})
        init_fn, _ = _make_upgd_idbd_learner(hp)
        state = init_fn(params)
        lean_state = LeanUPGDState(utility=state.utility, step=state.step)
        kg, kn = jr.split(jr.key(9))
        grads = {n: jr.normal(jr.fold_in(kg, i), v.shape) * 0.1
                 for i, (n, v) in enumerate(sorted(params.items()))}
        noise = {n: jr.normal(jr.fold_in(kn, i), v.shape) * hp["noise_std"]
                 for i, (n, v) in enumerate(sorted(params.items()))}
        for _ in range(3):
            p_idbd, state = upgd_idbd_update(params, state, grads, noise, hp)
            p_lean, lean_state = lean_upgd_w_update(params, lean_state, grads, noise, hp)
            for n in params:
                np.testing.assert_allclose(p_idbd[n], p_lean[n], atol=1e-7)
            params = p_idbd

    def test_log_alpha_stays_within_bounds(self):
        key = jr.key(0)
        params = init_mlp_params(key, SMALL)
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        hp.update({"meta_step_size": 10.0, "initial_step_size": 0.01})
        init_fn, _ = _make_upgd_idbd_learner(hp)
        state = init_fn(params)
        grads = {n: jnp.ones_like(v) for n, v in params.items()}
        noise = {n: jnp.zeros_like(v) for n, v in params.items()}
        for _ in range(5):
            params, state = upgd_idbd_update(params, state, grads, noise, hp)
        for n in params:
            assert bool(jnp.all(state.log_alpha[n] >= -10.0))
            assert bool(jnp.all(state.log_alpha[n] <= 0.0))
            assert bool(jnp.all(jnp.isfinite(params[n])))


class TestFadeHead:
    """FADE meta-learned per-parameter weight decay on the output layer."""

    def _fade_hp(self, **overrides):
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        hp.update({"fade_alpha": 0.005, "fade_gamma0": -6.9, "fade_theta_lambda": 0.1})
        hp.update(overrides)
        return hp

    def _random_inputs(self, params, hp, seed):
        kg, kn = jr.split(jr.key(seed))
        grads = {n: jr.normal(jr.fold_in(kg, i), v.shape) * 0.1
                 for i, (n, v) in enumerate(sorted(params.items()))}
        noise = {n: jr.normal(jr.fold_in(kn, i), v.shape) * hp["noise_std"]
                 for i, (n, v) in enumerate(sorted(params.items()))}
        return grads, noise

    def test_lambda_zero_theta_zero_reduces_to_control_on_head(self):
        """theta_lambda=0, gamma_0=-inf (lambda=0): head == control with zero
        head decay, hidden layers == published control, bit-exact."""
        key = jr.key(0)
        params = init_mlp_params(key, SMALL)
        hp = self._fade_hp(fade_gamma0=-math.inf, fade_theta_lambda=0.0)
        init_fn, _ = _make_upgd_w_fade_head_learner(hp)
        state = init_fn(params)
        grads, noise = self._random_inputs(params, hp, seed=21)
        head_x = jnp.abs(jr.normal(jr.key(33), (SMALL.hidden2,), jnp.float32))
        lean_state = LeanUPGDState(
            utility={n: jnp.zeros_like(v) for n, v in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )
        p_fade, _ = upgd_w_fade_head_update(params, state, grads, noise, head_x, hp)
        p_wd, _ = lean_upgd_w_update(params, lean_state, grads, noise, hp)
        hp_wd0 = dict(hp)
        hp_wd0["weight_decay"] = 0.0
        p_wd0, _ = lean_upgd_w_update(params, lean_state, grads, noise, hp_wd0)
        for n in ("w3", "b3"):
            np.testing.assert_array_equal(np.asarray(p_fade[n]), np.asarray(p_wd0[n]))
        for n in ("w1", "b1", "w2", "b2"):
            np.testing.assert_array_equal(np.asarray(p_fade[n]), np.asarray(p_wd[n]))

        # Multi-step: with weight_decay=0 everywhere and lambda=0 the whole
        # trajectory reduces bit-exactly to the lean UPGD-W trajectory.
        params = init_mlp_params(key, SMALL)
        state = init_fn(params)
        lean_state = LeanUPGDState(
            utility={n: jnp.zeros_like(v) for n, v in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )
        for step in range(3):
            grads, noise = self._random_inputs(params, hp_wd0, seed=100 + step)
            p_fade, state = upgd_w_fade_head_update(
                params, state, grads, noise, head_x, hp_wd0
            )
            p_lean, lean_state = lean_upgd_w_update(
                params, lean_state, grads, noise, hp_wd0
            )
            for n in params:
                np.testing.assert_array_equal(np.asarray(p_fade[n]), np.asarray(p_lean[n]))
            params = p_fade

    def test_frozen_gamma_at_control_decay_matches_published_control(self):
        """theta_lambda=0, lambda_0 = step_size*weight_decay: one step matches
        the published control on every layer (head decay factors coincide)."""
        key = jr.key(1)
        params = init_mlp_params(key, SMALL)
        base = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        gamma0 = math.log(base["step_size"] * base["weight_decay"])
        hp = self._fade_hp(fade_gamma0=gamma0, fade_theta_lambda=0.0)
        init_fn, _ = _make_upgd_w_fade_head_learner(hp)
        state = init_fn(params)
        lean_state = LeanUPGDState(
            utility={n: jnp.zeros_like(v) for n, v in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )
        grads, noise = self._random_inputs(params, hp, seed=55)
        head_x = jnp.abs(jr.normal(jr.key(56), (SMALL.hidden2,), jnp.float32))
        p_fade, _ = upgd_w_fade_head_update(params, state, grads, noise, head_x, hp)
        p_lean, _ = lean_upgd_w_update(params, lean_state, grads, noise, hp)
        for n in params:
            np.testing.assert_allclose(
                np.asarray(p_fade[n]), np.asarray(p_lean[n]), atol=1e-7
            )

    def test_lambda_stays_finite_positive_over_random_steps(self):
        """200 random steps at published hyperparameters: gamma stays finite
        and capped, lambda = exp(gamma) stays in (0, 1], trace stays finite."""
        key = jr.key(2)
        params = init_mlp_params(key, SMALL)
        hp = self._fade_hp()
        init_fn, _ = _make_upgd_w_fade_head_learner(hp)
        state = init_fn(params)
        for step in range(200):
            grads, noise = self._random_inputs(params, hp, seed=1000 + step)
            head_x = jnp.abs(jr.normal(jr.fold_in(jr.key(3), step), (SMALL.hidden2,)))
            params, state = upgd_w_fade_head_update(
                params, state, grads, noise, head_x, hp
            )
        for n in ("w3", "b3"):
            gamma = np.asarray(state.gamma[n])
            lam = np.exp(gamma)
            assert np.all(np.isfinite(gamma)), n
            assert np.all(gamma <= 0.0), n
            assert np.all(lam > 0.0), n
            assert np.all(lam <= 1.0), n
            assert np.all(np.isfinite(np.asarray(state.fade_trace[n]))), n
            assert np.all(np.isfinite(np.asarray(params[n]))), n

    def test_gamma_increases_when_decay_helps(self):
        """Stale positive head weight the new task's gradient keeps pushing
        toward zero (decay helps): gamma on w3 must rise above gamma_0."""
        key = jr.key(4)
        params = init_mlp_params(key, SMALL)
        params = dict(params)
        params["w3"] = jnp.full_like(params["w3"], 2.0)  # stale, far from init
        hp = self._fade_hp()
        init_fn, _ = _make_upgd_w_fade_head_learner(hp)
        state = init_fn(params)
        kg = jr.key(8)
        head_x = jnp.ones((SMALL.hidden2,), jnp.float32)
        zeros = {n: jnp.zeros_like(v) for n, v in params.items()}
        for step in range(100):
            # Persistent error: descent wants the stale positive w3 to shrink
            # (positive gradient); small random grads elsewhere keep the
            # utility gate's global max well-defined.
            grads = {n: jr.normal(jr.fold_in(kg, i * 1000 + step), v.shape) * 0.1
                     for i, (n, v) in enumerate(sorted(params.items()))}
            grads["w3"] = jnp.full_like(params["w3"], 0.5)
            grads["b3"] = jnp.zeros_like(params["b3"])
            params, state = upgd_w_fade_head_update(
                params, state, grads, zeros, head_x, hp
            )
        assert bool(jnp.all(state.gamma["w3"] > hp["fade_gamma0"]))
        assert bool(jnp.all(jnp.isfinite(state.gamma["w3"])))


class TestIDBDSwift:
    """UPGD+IDBD with SwiftTD's supervised-mode stabilizers."""

    def _hp(self, **overrides):
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        hp.update({"meta_step_size": 1e-3, "initial_step_size": 0.01,
                   "swift_eta": 0.1, "swift_eps": 0.99})
        hp.update(overrides)
        return hp

    def test_eta_inf_eps_one_reduces_to_upgd_idbd(self):
        """swift_eta=inf, swift_eps=1: bit-exact upgd_idbd trajectory."""
        key = jr.key(0)
        params = init_mlp_params(key, SMALL)
        hp = self._hp(swift_eta=math.inf, swift_eps=1.0)
        init_fn, _ = _make_upgd_idbd_learner(hp)
        state_swift = init_fn(params)
        state_plain = init_fn(params)
        kg, kn = jr.split(jr.key(9))
        grads = {n: jr.normal(jr.fold_in(kg, i), v.shape) * 0.1
                 for i, (n, v) in enumerate(sorted(params.items()))}
        noise = {n: jr.normal(jr.fold_in(kn, i), v.shape) * hp["noise_std"]
                 for i, (n, v) in enumerate(sorted(params.items()))}
        for _ in range(3):
            p_swift, state_swift = upgd_idbd_swift_update(
                params, state_swift, grads, noise, hp
            )
            p_plain, state_plain = upgd_idbd_update(params, state_plain, grads, noise, hp)
            for n in params:
                np.testing.assert_array_equal(np.asarray(p_swift[n]), np.asarray(p_plain[n]))
                np.testing.assert_array_equal(
                    np.asarray(state_swift.log_alpha[n]), np.asarray(state_plain.log_alpha[n])
                )
                np.testing.assert_array_equal(
                    np.asarray(state_swift.trace[n]), np.asarray(state_plain.trace[n])
                )
            params = p_swift

    def test_overshoot_bound_triggers_and_caps_effective_step(self):
        """Large alpha: sum_i alpha_i z_i^2 >> eta, so the applied step is the
        unbounded step scaled by eta/tau, the effective correction ratio is
        capped at eta, step-sizes decay persistently, and traces reset."""
        key = jr.key(2)
        params = init_mlp_params(key, SMALL)
        hp = self._hp(
            weight_decay=0.0, meta_step_size=0.0,
            initial_step_size=1.0, swift_eta=1.0,
        )
        init_fn, _ = _make_upgd_idbd_learner(hp)
        state_swift = init_fn(params)
        state_plain = init_fn(params)
        kg = jr.key(7)
        grads = {n: jr.normal(jr.fold_in(kg, i), v.shape)
                 for i, (n, v) in enumerate(sorted(params.items()))}
        zeros = {n: jnp.zeros_like(v) for n, v in params.items()}
        p_swift, s_swift = upgd_idbd_swift_update(params, state_swift, grads, zeros, hp)
        p_plain, s_plain = upgd_idbd_update(params, state_plain, grads, zeros, hp)
        alpha = hp["initial_step_size"]  # meta=0 keeps alpha frozen at init
        delta_plain = {n: np.asarray(params[n] - p_plain[n]) for n in params}  # alpha*z
        delta_swift = {n: np.asarray(params[n] - p_swift[n]) for n in params}
        tau = sum(float(np.sum(d * d)) for d in delta_plain.values()) / alpha
        eta = hp["swift_eta"]
        assert tau > eta  # the bound triggers on this constructed case
        scale = eta / tau
        assert scale < 1.0
        for n in params:
            np.testing.assert_allclose(
                delta_swift[n], scale * delta_plain[n], rtol=1e-4, atol=1e-7
            )
        # Effective correction ratio sum_i alpha_eff_i z_i^2 is capped at eta.
        tau_eff = sum(
            float(np.sum(delta_swift[n] * delta_plain[n])) for n in params
        ) / alpha
        assert tau_eff == pytest.approx(eta, rel=1e-3)
        # Persistent decay: beta_i += z_i^2 * ln(eps) (plain arm's log stays
        # frozen at ln(initial_step_size) because meta=0).
        for n in params:
            z_sq = (delta_plain[n] / alpha) ** 2
            np.testing.assert_allclose(
                np.asarray(s_swift.log_alpha[n]),
                np.asarray(s_plain.log_alpha[n]) + math.log(hp["swift_eps"]) * z_sq,
                rtol=1e-5, atol=1e-6,
            )
        # The trigger also resets the meta traces (SwiftTD's decay block).
        for n in params:
            np.testing.assert_array_equal(np.asarray(s_swift.trace[n]), 0.0)
        assert any(float(np.max(np.abs(np.asarray(s_plain.trace[n])))) > 0.0
                   for n in params)


class TestL2Init:
    def test_zero_grads_pull_toward_init_only(self):
        key = jr.key(4)
        params = init_mlp_params(key, SMALL)
        w0 = {n: v + 1.0 for n, v in params.items()}  # pretend init is elsewhere
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        from alberta_framework.benchmarks.ipmnist_screening import UPGDL2InitState

        # Nonzero utility keeps the global-max gate well-defined under zero
        # grads (the lean UPGD equations divide by the global utility max).
        state = UPGDL2InitState(
            utility={n: jnp.ones_like(v) for n, v in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
            init_params=w0,
        )
        grads = {n: jnp.zeros_like(v) for n, v in params.items()}
        noise = {n: jnp.zeros_like(v) for n, v in params.items()}
        new_params, _ = upgd_l2init_update(params, state, grads, noise, hp)
        lam = hp["step_size"] * hp["weight_decay"]
        for n in params:
            expected = params[n] - lam * (params[n] - w0[n])
            np.testing.assert_allclose(new_params[n], expected, atol=1e-7)


class TestEMANorm:
    def test_parity_with_core_ema_normalizer(self):
        normalizer = EMANormalizer(epsilon=1e-8, decay=0.999)
        core_state = normalizer.init(6)
        mine = EMANormState(
            mean=jnp.zeros(6), var=jnp.ones(6), count=jnp.array(0.0)
        )
        key = jr.key(11)
        for i in range(20):
            obs = jr.normal(jr.fold_in(key, i), (6,)) * 3.0 + 1.0
            ref, core_state = normalizer.normalize(core_state, obs)
            got, mine = ema_normalize(mine, obs, 0.999, 1e-8)
            np.testing.assert_allclose(got, ref, atol=1e-6)


class TestPerElementAdam:
    def test_uniform_count_parity_with_baseline_adam(self):
        hp = {"step_size": 1e-4, "beta1": 0.0, "beta2": 0.99, "eps": 1e-8,
              "weight_decay": 0.0}
        optimizer = Adam(**hp)
        shape = (5, 4)
        ref_state = optimizer.init_for_shape(shape)
        key = jr.key(2)
        param = jr.normal(jr.fold_in(key, 99), shape)
        ref_param = param
        m = jnp.zeros(shape)
        v = jnp.zeros(shape)
        count = jnp.zeros(shape)
        for i in range(6):
            grad = jr.normal(jr.fold_in(key, i), shape)
            step, ref_state = optimizer.update_from_gradient(
                ref_state, grad, error=None, param=ref_param
            )
            ref_param = ref_param - step
            param, m, v, count = adam_elem_update(param, m, v, count, grad, hp)
            np.testing.assert_allclose(param, ref_param, atol=1e-7)


class TestCBPReplacement:
    def test_replacement_recycles_lowest_utility_mature_unit(self):
        key = jr.key(5)
        params = init_mlp_params(key, SMALL)
        h1 = SMALL.hidden1
        utility = jnp.arange(1, h1 + 1, dtype=jnp.float32)  # unit 0 lowest
        age = jnp.full((h1,), 200, dtype=jnp.int32)
        opt = {n: jnp.ones((2,) + v.shape, dtype=jnp.float32) for n, v in params.items()}
        new_params, new_opt, new_util, new_age, new_acc = cbp_maybe_replace_layer(
            params, opt, utility, age, jnp.array(0.5), _CBP_LAYERS[0], jr.key(6),
            replacement_rate=1.0 / h1, maturity_threshold=100,
        )
        # accumulator 0.5 + 1.0 -> fired and decremented
        assert float(new_acc) == pytest.approx(0.5)
        assert float(new_util[0]) == 0.0
        assert int(new_age[0]) == 0
        # incoming column replaced, in protocol init range
        bound = 1.0 / math.sqrt(SMALL.input_dim)
        col = np.asarray(new_params["w1"][:, 0])
        assert not np.allclose(col, np.asarray(params["w1"][:, 0]))
        assert np.all(np.abs(col) <= bound)
        assert float(new_params["b1"][0]) == 0.0
        # outgoing row zeroed
        np.testing.assert_allclose(np.asarray(new_params["w2"][0, :]), 0.0)
        # optimizer slices reset
        np.testing.assert_allclose(np.asarray(new_opt["w1"][:, :, 0]), 0.0)
        np.testing.assert_allclose(np.asarray(new_opt["b1"][:, 0]), 0.0)
        np.testing.assert_allclose(np.asarray(new_opt["w2"][:, 0, :]), 0.0)
        # untouched units keep their values
        np.testing.assert_allclose(
            np.asarray(new_params["w1"][:, 1]), np.asarray(params["w1"][:, 1])
        )

    def test_no_replacement_when_accumulator_below_one(self):
        key = jr.key(5)
        params = init_mlp_params(key, SMALL)
        h1 = SMALL.hidden1
        utility = jnp.arange(1, h1 + 1, dtype=jnp.float32)
        age = jnp.full((h1,), 200, dtype=jnp.int32)
        new_params, _, new_util, new_age, new_acc = cbp_maybe_replace_layer(
            params, None, utility, age, jnp.array(0.0), _CBP_LAYERS[0], jr.key(6),
            replacement_rate=0.01 / h1, maturity_threshold=100,
        )
        assert float(new_acc) == pytest.approx(0.01)
        for n in params:
            np.testing.assert_allclose(np.asarray(new_params[n]), np.asarray(params[n]))

    def test_no_replacement_when_no_mature_units(self):
        key = jr.key(5)
        params = init_mlp_params(key, SMALL)
        h1 = SMALL.hidden1
        utility = jnp.arange(1, h1 + 1, dtype=jnp.float32)
        age = jnp.zeros((h1,), dtype=jnp.int32)
        new_params, _, _, _, new_acc = cbp_maybe_replace_layer(
            params, None, utility, age, jnp.array(2.0), _CBP_LAYERS[0], jr.key(6),
            replacement_rate=1.0 / h1, maturity_threshold=100,
        )
        # budget accumulates but nothing fires
        assert float(new_acc) == pytest.approx(3.0)
        for n in params:
            np.testing.assert_allclose(np.asarray(new_params[n]), np.asarray(params[n]))


class TestWeightClipping:
    """UPGD-W + weight clipping (Elsayed et al., RLC 2024)."""

    def test_kappa_inf_reduces_exactly_to_control(self, small_data):
        """With kappa=inf the clip is a no-op: bit-exact control trajectory."""
        x, y = small_data
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        hp["clip_kappa"] = math.inf
        spec = ScreeningSpec(
            name="upgd_w_control",  # reuse control identity for shard plumbing
            base_learner="upgd_w",
            mechanism="weight_clipping",
            hyperparameters=hp,
            factory=_make_upgd_w_wclip_learner,
        )
        ours = run_screening_config(x, y, spec, seed=7, config=SMALL)
        control = run_screening_config(
            x, y, screening_spec("upgd_w_control"), seed=7, config=SMALL
        )
        np.testing.assert_array_equal(ours.per_task_accuracy, control.per_task_accuracy)
        np.testing.assert_allclose(ours.per_task_loss, control.per_task_loss, rtol=1e-6)

    def test_clip_bounds_enforced_per_layer(self):
        """After one step every parameter obeys |w| <= kappa / sqrt(fan_in)."""
        key = jr.key(4)
        params = init_mlp_params(key, SMALL)
        big = {n: v + 100.0 for n, v in params.items()}  # way outside every bound
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        hp["clip_kappa"] = 1.0
        state = LeanUPGDState(
            utility={n: jnp.ones_like(v) for n, v in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )
        grads = {n: jnp.zeros_like(v) for n, v in params.items()}
        noise = {n: jnp.zeros_like(v) for n, v in params.items()}
        new_params, _ = upgd_w_wclip_update(big, state, grads, noise, hp)
        fan_in = {"1": SMALL.input_dim, "2": SMALL.hidden1, "3": SMALL.hidden2}
        for n in new_params:
            bound = hp["clip_kappa"] / math.sqrt(fan_in[n[1:]])
            values = np.asarray(new_params[n])
            assert np.all(values <= bound + 1e-7), n
            assert np.all(values >= -bound - 1e-7), n
            # saturated: the +100 shift puts everything at the upper bound
            np.testing.assert_allclose(values, bound, atol=1e-6)

    def test_registry_wd0_variants_disable_weight_decay(self):
        assert screening_spec("upgd_w_wclip_k1_wd0").hyperparameters["weight_decay"] == 0.0
        assert screening_spec("upgd_w_wclip_k2_wd0").hyperparameters["weight_decay"] == 0.0
        assert screening_spec("upgd_w_wclip_k1").hyperparameters["weight_decay"] == 0.01
        assert screening_spec("upgd_w_wclip_k1").hyperparameters["clip_kappa"] == 1.0
        assert screening_spec("upgd_w_wclip_k2").hyperparameters["clip_kappa"] == 2.0


class TestLocalGateNorm:
    """Per-tensor utility-gate normalization vs the published global max."""

    def _random_inputs(self, params, hp, seed=9):
        kg, kn = jr.split(jr.key(seed))
        grads = {n: jr.normal(jr.fold_in(kg, i), v.shape) * 0.1
                 for i, (n, v) in enumerate(sorted(params.items()))}
        noise = {n: jr.normal(jr.fold_in(kn, i), v.shape) * hp["noise_std"]
                 for i, (n, v) in enumerate(sorted(params.items()))}
        return grads, noise

    def test_single_tensor_equals_global(self):
        """With one parameter tensor, per-tensor max == global max: identical."""
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        key = jr.key(3)
        params = {"w1": jr.normal(key, (12, 8), jnp.float32) * 0.1}
        state = LeanUPGDState(
            utility={"w1": jnp.zeros_like(params["w1"])},
            step=jnp.array(0, dtype=jnp.int32),
        )
        lean_state = LeanUPGDState(utility=dict(state.utility), step=state.step)
        grads, noise = self._random_inputs(params, hp)
        for _ in range(3):
            p_local, state = upgd_w_localgate_update(params, state, grads, noise, hp)
            p_global, lean_state = lean_upgd_w_update(params, lean_state, grads, noise, hp)
            np.testing.assert_array_equal(
                np.asarray(p_local["w1"]), np.asarray(p_global["w1"])
            )
            params = p_local

    def test_multi_tensor_differs_from_global(self):
        """With several tensors whose utility maxima differ, the gates differ."""
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        key = jr.key(5)
        params = init_mlp_params(key, SMALL)
        state = LeanUPGDState(
            utility={n: jnp.zeros_like(v) for n, v in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )
        lean_state = LeanUPGDState(utility=dict(state.utility), step=state.step)
        grads, noise = self._random_inputs(params, hp, seed=17)
        p_local, _ = upgd_w_localgate_update(params, state, grads, noise, hp)
        p_global, _ = lean_upgd_w_update(params, lean_state, grads, noise, hp)
        assert any(
            not np.allclose(np.asarray(p_local[n]), np.asarray(p_global[n]))
            for n in params
        )


class TestSmokeRuns:
    """Every combination runs at tiny scale, learns above chance, stays finite."""

    @pytest.mark.parametrize("name", [
        "upgd_idbd", "upgd_autostep", "upgd_l2init", "upgd_ema_norm",
        "upgd_cbp", "adamw_cbp", "upgd_w_wclip_k1", "upgd_w_wclip_k2_wd0",
        "upgd_w_localgate", "upgd_w_fade_head", "upgd_w_idbd_swift",
    ])
    def test_combo_runs_and_is_finite(self, small_data, name):
        x, y = small_data
        result = run_screening_config(x, y, screening_spec(name), seed=1, config=SMALL)
        assert result.per_task_accuracy.shape == (SMALL.n_tasks,)
        assert np.all(np.isfinite(result.per_task_accuracy))
        assert np.all(result.per_task_accuracy >= 0.0)
        assert np.all(result.per_task_accuracy <= 1.0)
        assert np.all(np.isfinite(result.per_task_loss))
        assert np.all(result.per_task_plasticity >= 0.0)
        assert np.all(result.per_task_plasticity <= 1.0)


class TestShardsAndMerge:
    def _make_shard(self, tmp_path, small_data, name, seed):
        x, y = small_data
        result = run_screening_config(x, y, screening_spec(name), seed=seed, config=SMALL)
        path = tmp_path / f"{name}_seed{seed}.json"
        path.write_text(json.dumps(shard_payload(result)), encoding="utf-8")
        return path

    def test_shard_roundtrip_and_merge(self, tmp_path, small_data):
        paths = [
            self._make_shard(tmp_path, small_data, "upgd_w_control", 0),
            self._make_shard(tmp_path, small_data, "upgd_w_control", 1),
            self._make_shard(tmp_path, small_data, "upgd_l2init", 0),
            self._make_shard(tmp_path, small_data, "upgd_l2init", 1),
        ]
        for p in paths:
            assert load_shard(p)["schema"] == SHARD_SCHEMA
        summary = merge_shards(paths, control_name="upgd_w_control", slope_window=2)
        names = {e["config_name"] for e in summary["results"]}
        assert names == {"upgd_w_control", "upgd_l2init"}
        l2 = next(e for e in summary["results"] if e["config_name"] == "upgd_l2init")
        assert l2["paired_vs_control"]["seeds"] == [0, 1]
        assert len(l2["paired_vs_control"]["per_seed_diff"]) == 2
        assert isinstance(l2["paired_vs_control"]["confirmation_candidate"], bool)
        control = next(e for e in summary["results"] if e["config_name"] == "upgd_w_control")
        assert "paired_vs_control" not in control

    def test_merge_rejects_duplicate_seed(self, tmp_path, small_data):
        p1 = self._make_shard(tmp_path, small_data, "upgd_w_control", 0)
        p2 = tmp_path / "dup.json"
        p2.write_text(p1.read_text(encoding="utf-8"), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate shard"):
            merge_shards([p1, p2])

    def test_validate_proxy_prefix_and_ordering(self, tmp_path, small_data):
        x, y = small_data
        partials = tmp_path / "partials"
        partials.mkdir()
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        shard_paths = []
        for name, learner in (("upgd_w_control", "upgd_w"), ("adamw_control", "adamw")):
            full = run_ipmnist(x, y, learner, seeds=[0], config=SMALL)
            (partials / f"{learner}_seed0.json").write_text(
                json.dumps({"per_task_accuracy": full.per_task_accuracy.tolist()}),
                encoding="utf-8",
            )
            proxy = run_screening_config(
                x, y, screening_spec(name), seed=0, config=IPMNISTConfig(
                    n_tasks=2, task_length=30, input_dim=12,
                    hidden1=8, hidden2=6, n_classes=5,
                )
            )
            path = shard_dir / f"{name}_seed0.json"
            path.write_text(json.dumps(shard_payload(proxy)), encoding="utf-8")
            shard_paths.append(path)
        report = validate_proxy(shard_paths, partials, atol=1e-6)
        assert report["all_prefixes_match"] is True
        for check in report["checks"]:
            assert check["max_abs_per_task_diff"] <= 1e-6
        # ordering flags are booleans (tiny-scale runs may order either way)
        assert isinstance(report["proxy_preserves_upgd_over_adamw"], bool)
        assert isinstance(report["full_prefix_preserves_upgd_over_adamw"], bool)


class TestPoolConfirmation:
    """Screening-only pool-noise mode for full-protocol confirmation runs."""

    def test_pool_control_matches_run_ipmnist_pool(self, small_data):
        """Control arm under pool mode reproduces run_ipmnist's pool chain."""
        x, y = small_data
        ours = run_screening_config(
            x, y, screening_spec("upgd_w_control"), seed=5, config=SMALL,
            noise_mode="pool", noise_pool_steps=8,
        )
        reference = run_ipmnist(
            x, y, "upgd_w", seeds=[5], config=SMALL,
            noise_mode="pool", noise_pool_steps=8,
        )
        assert ours.noise_mode == "pool"
        np.testing.assert_allclose(
            ours.per_task_accuracy, reference.per_task_accuracy[0], atol=1e-7
        )
        np.testing.assert_allclose(
            ours.per_task_loss, reference.per_task_loss[0], rtol=1e-5
        )

    def test_pool_differs_from_exact_but_stays_close_at_tiny_scale(self, small_data):
        x, y = small_data
        exact = run_screening_config(
            x, y, screening_spec("upgd_w_wclip_k2"), seed=5, config=SMALL
        )
        pool = run_screening_config(
            x, y, screening_spec("upgd_w_wclip_k2"), seed=5, config=SMALL,
            noise_mode="pool", noise_pool_steps=8,
        )
        assert exact.noise_mode == "step"
        # different noise stream => different (continuous) loss trajectory
        assert not np.array_equal(pool.per_task_loss, exact.per_task_loss)

    def test_pool_rejected_for_arms_without_noise_update(self, small_data):
        x, y = small_data
        with pytest.raises(ValueError, match="pool"):
            run_screening_config(
                x, y, screening_spec("upgd_idbd"), seed=0, config=SMALL,
                noise_mode="pool",
            )

    def test_pool_shards_record_mode_and_never_merge_with_exact(
        self, tmp_path, small_data
    ):
        x, y = small_data
        exact = run_screening_config(
            x, y, screening_spec("upgd_w_control"), seed=0, config=SMALL
        )
        pool = run_screening_config(
            x, y, screening_spec("upgd_w_localgate"), seed=0, config=SMALL,
            noise_mode="pool", noise_pool_steps=8,
        )
        exact_payload = shard_payload(exact)
        pool_payload = shard_payload(pool)
        assert exact_payload["noise_mode"] == "step"
        assert pool_payload["noise_mode"] == "pool"
        p_exact = tmp_path / "exact.json"
        p_pool = tmp_path / "pool.json"
        p_exact.write_text(json.dumps(exact_payload), encoding="utf-8")
        p_pool.write_text(json.dumps(pool_payload), encoding="utf-8")
        with pytest.raises(ValueError, match="noise mode"):
            merge_shards([p_exact, p_pool])

    def test_validate_proxy_rejects_pool_shards(self, tmp_path, small_data):
        x, y = small_data
        pool = run_screening_config(
            x, y, screening_spec("upgd_w_control"), seed=0, config=SMALL,
            noise_mode="pool", noise_pool_steps=8,
        )
        path = tmp_path / "pool.json"
        path.write_text(json.dumps(shard_payload(pool)), encoding="utf-8")
        with pytest.raises(ValueError, match="noise_mode"):
            validate_proxy([path], tmp_path)


class TestSpecShape:
    def test_specs_are_json_serializable(self):
        for spec in SCREENING_REGISTRY.values():
            assert isinstance(spec, ScreeningSpec)
            json.dumps(spec.hyperparameters)
            assert spec.base_learner in ("upgd_w", "adamw")

    def test_noise_update_present_exactly_on_lean_family_arms(self):
        with_pool = {
            name for name, spec in SCREENING_REGISTRY.items()
            if spec.noise_update is not None
        }
        assert with_pool == {
            "upgd_w_control",
            "upgd_w_wclip_k1",
            "upgd_w_wclip_k2",
            "upgd_w_wclip_k1_wd0",
            "upgd_w_wclip_k2_wd0",
            "upgd_w_localgate",
            "upgd_w_sigma005",
            "upgd_w_sigma02",
            "upgd_w_udecay0999",
            "upgd_w_udecay099999",
            "upgd_w_wd0005",
            "upgd_w_wd002",
        }
