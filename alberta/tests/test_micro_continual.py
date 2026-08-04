"""Unit tests for the micro continual-learning suite.

Canonical first-principles suite: pins the M1-M4 Gaussian stream generators
to their construction (coordinate permutation / label permutation / scale
shift / recurrence over a heterogeneous-spectrum class mixture), pins the
analytic Bayes reference to the closed-form two-class formula and to
transform invariance, pins the method-ladder arms to the campaign's
registered equations and hyperparameters, and tests
shard/merge/transfer-validation plumbing. Benchmark executions (the actual
proxy-validation ladder) never run here.

Provisional digits-based suite (rule-discovery track compatibility): search
tasks (M1/M2/M3) and holdout tasks (M4/M1p) must stay disjoint; streams are
pure functions of (task config, seed). Never scientific evidence.
"""

import dataclasses
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_screening import (
    SCREENING_REGISTRY,
    screening_spec,
)
from alberta_framework.benchmarks.micro_continual import (
    FAMILIES,
    HOLDOUT_TASKS,
    LADDER_ARMS,
    MICRO_ARM_REGISTRY,
    MICRO_SHARD_SCHEMA,
    MICRO_SUITE,
    MICRO_SUITE_VERSION,
    MICRO_SUMMARY_SCHEMA,
    MICRO_VALIDATION_SCHEMA,
    NONPROMOTING_POLICY,
    SEARCH_TASKS,
    BayesReference,
    MicroStream,
    MicroStreamConfig,
    MicroTaskConfig,
    assemble_observed,
    bayes_predict,
    bayes_reference,
    build_micro_stream,
    class_geometry,
    dim_scale_spectrum,
    generate_stream,
    load_digits_features,
    load_micro_shard,
    main,
    merge_micro_shards,
    micro_arm_spec,
    micro_shard_path,
    micro_shard_payload,
    run_micro_arm,
    transfer_validation,
    transfer_validation_from_shards,
    two_class_bayes_accuracy,
    write_micro_shard,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    ADAMW_PROTOCOL_HYPERPARAMETERS,
    UPGD_W_PROTOCOL_HYPERPARAMETERS,
    IPMNISTConfig,
    cross_entropy_loss,
    init_mlp_params,
)

pytestmark = pytest.mark.unit

TINY = MicroStreamConfig(
    family="input_permutation",
    n_regimes=4,
    regime_length=25,
    dim=6,
    n_classes=3,
    n_components=2,
    component_sparsity=2,
    class_sparsity=0.5,
)


def tiny(family: str, **overrides: object) -> MicroStreamConfig:
    merged: dict[str, object] = {
        "family": family,
        "n_regimes": 4,
        "regime_length": 25,
        "dim": 6,
        "n_classes": 3,
        "n_components": 2,
        "component_sparsity": 2,
        "class_sparsity": 0.5,
        "recurrence_pool": 2,
    }
    merged.update(overrides)
    return MicroStreamConfig(**merged)  # type: ignore[arg-type]


# =============================================================================
# Canonical suite: configuration
# =============================================================================


class TestConfig:
    def test_family_codes_cover_m1_to_m4(self):
        assert FAMILIES == (
            "input_permutation",
            "label_permutation",
            "scale_shift",
            "recurrence",
        )

    def test_unknown_family_rejected(self):
        with pytest.raises(ValueError, match="family"):
            MicroStreamConfig(family="frequency_shift")

    @pytest.mark.parametrize(
        "field", ["n_regimes", "regime_length", "dim", "n_classes", "n_components"]
    )
    def test_nonpositive_ints_rejected(self, field):
        with pytest.raises(ValueError, match=field):
            MicroStreamConfig(**{"family": "input_permutation", field: 0})

    def test_component_sparsity_bounds(self):
        with pytest.raises(ValueError, match="component_sparsity"):
            tiny("input_permutation", component_sparsity=0)
        with pytest.raises(ValueError, match="component_sparsity"):
            tiny("input_permutation", component_sparsity=7)  # > dim=6
        tiny("input_permutation", component_sparsity=6)  # == dim is valid (dense)

    def test_class_sparsity_bounds(self):
        with pytest.raises(ValueError, match="class_sparsity"):
            tiny("input_permutation", class_sparsity=0.0)
        with pytest.raises(ValueError, match="class_sparsity"):
            tiny("input_permutation", class_sparsity=1.5)
        tiny("input_permutation", class_sparsity=1.0)  # dense is valid

    def test_component_scale_nonnegative(self):
        with pytest.raises(ValueError, match="component_scale"):
            tiny("input_permutation", component_scale=-0.5)

    def test_bool_is_not_a_valid_int(self):
        with pytest.raises(ValueError, match="n_regimes"):
            MicroStreamConfig(family="input_permutation", n_regimes=True)

    def test_recurrence_pool_bounds(self):
        with pytest.raises(ValueError, match="recurrence_pool"):
            tiny("recurrence", recurrence_pool=1)
        with pytest.raises(ValueError, match="recurrence_pool"):
            tiny("recurrence", recurrence_pool=5)  # > n_regimes=4
        tiny("recurrence", recurrence_pool=2)  # valid

    def test_scale_shift_bounds(self):
        with pytest.raises(ValueError, match="scale_shift"):
            tiny("scale_shift", scale_shift_min=0.0)
        with pytest.raises(ValueError, match="scale_shift"):
            tiny("scale_shift", scale_shift_min=2.0, scale_shift_max=2.0)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("mean_separation", 0.0),
            ("noise_scale", 0.0),
            ("offset_scale", -0.1),
            ("spectrum_decades", -1.0),
        ],
    )
    def test_generator_scalars_validated(self, field, value):
        with pytest.raises(ValueError, match=field):
            tiny("input_permutation", **{field: value})

    def test_n_steps(self):
        assert TINY.n_steps == 4 * 25

    def test_to_config_roundtrip(self):
        rebuilt = MicroStreamConfig(**TINY.to_config())
        assert rebuilt == TINY


# =============================================================================
# Canonical suite: per-dimension scale spectrum
# =============================================================================


class TestSpectrum:
    def test_shape_and_endpoints(self):
        config = tiny("input_permutation", spectrum_decades=2.0)
        scales = np.asarray(dim_scale_spectrum(config))
        assert scales.shape == (config.dim,)
        assert scales[0] == pytest.approx(1.0)
        assert scales[-1] == pytest.approx(10.0**-2.0)

    def test_monotone_decreasing(self):
        scales = np.asarray(dim_scale_spectrum(TINY))
        assert np.all(np.diff(scales) < 0.0)

    def test_zero_decades_is_homogeneous(self):
        config = tiny("input_permutation", spectrum_decades=0.0)
        scales = np.asarray(dim_scale_spectrum(config))
        np.testing.assert_allclose(scales, np.ones(config.dim), rtol=1e-6)


# =============================================================================
# Canonical suite: stream generation (M1-M4)
# =============================================================================


class TestGenerator:
    def test_deterministic_per_seed(self):
        a = generate_stream(TINY, seed=0)
        b = generate_stream(TINY, seed=0)
        np.testing.assert_array_equal(np.asarray(a.x), np.asarray(b.x))
        np.testing.assert_array_equal(np.asarray(a.y), np.asarray(b.y))
        np.testing.assert_array_equal(
            np.asarray(a.permutations), np.asarray(b.permutations)
        )

    def test_seeds_differ(self):
        a = generate_stream(TINY, seed=0)
        b = generate_stream(TINY, seed=1)
        assert not np.array_equal(np.asarray(a.x), np.asarray(b.x))

    def test_shapes_and_dtypes(self):
        stream = generate_stream(TINY, seed=0)
        n, d = TINY.n_steps, TINY.dim
        assert stream.x.shape == (n, d) and stream.x.dtype == jnp.float32
        assert stream.y.shape == (n,) and stream.y.dtype == jnp.int32
        assert stream.base_x.shape == (n, d)
        assert stream.base_y.shape == (n,)
        assert stream.regime_ids.shape == (n,)
        assert stream.permutations.shape == (TINY.n_regimes, d)
        assert stream.label_maps.shape == (TINY.n_regimes, TINY.n_classes)
        assert stream.scale_factors.shape == (TINY.n_regimes,)
        assert stream.regime_pool_ids.shape == (TINY.n_regimes,)
        assert stream.component_means.shape == (
            TINY.n_classes, TINY.n_components, d
        )
        assert stream.dim_sigma.shape == (d,)

    def test_regime_ids(self):
        stream = generate_stream(TINY, seed=0)
        expected = np.arange(TINY.n_steps) // TINY.regime_length
        np.testing.assert_array_equal(np.asarray(stream.regime_ids), expected)

    def test_labels_in_range(self):
        stream = generate_stream(TINY, seed=0)
        y = np.asarray(stream.y)
        base_y = np.asarray(stream.base_y)
        assert y.min() >= 0 and y.max() < TINY.n_classes
        assert base_y.min() >= 0 and base_y.max() < TINY.n_classes

    def test_observed_matches_assembly(self):
        for family in FAMILIES:
            stream = generate_stream(tiny(family), seed=3)
            x, y = assemble_observed(
                stream.base_x,
                stream.base_y,
                stream.regime_ids,
                stream.permutations,
                stream.label_maps,
                stream.scale_factors,
            )
            np.testing.assert_array_equal(np.asarray(stream.x), np.asarray(x))
            np.testing.assert_array_equal(np.asarray(stream.y), np.asarray(y))

    def test_m1_input_permutation_axis(self):
        stream = generate_stream(tiny("input_permutation"), seed=0)
        perms = np.asarray(stream.permutations)
        for row in perms:
            assert sorted(row.tolist()) == list(range(TINY.dim))
        # fresh permutation per regime (with overwhelming probability)
        assert len({tuple(row) for row in perms}) == TINY.n_regimes
        # the other two axes are inert
        np.testing.assert_array_equal(
            np.asarray(stream.label_maps),
            np.tile(np.arange(TINY.n_classes), (TINY.n_regimes, 1)),
        )
        np.testing.assert_allclose(
            np.asarray(stream.scale_factors), np.ones(TINY.n_regimes)
        )
        # x really is the coordinate-relabeled base stream
        t = TINY.regime_length  # first step of regime 1
        np.testing.assert_array_equal(
            np.asarray(stream.x[t]), np.asarray(stream.base_x[t])[perms[1]]
        )

    def test_m2_label_permutation_axis(self):
        stream = generate_stream(tiny("label_permutation"), seed=0)
        maps = np.asarray(stream.label_maps)
        for row in maps:
            assert sorted(row.tolist()) == list(range(TINY.n_classes))
        np.testing.assert_array_equal(
            np.asarray(stream.x), np.asarray(stream.base_x)
        )
        regime_ids = np.asarray(stream.regime_ids)
        np.testing.assert_array_equal(
            np.asarray(stream.y), maps[regime_ids, np.asarray(stream.base_y)]
        )

    def test_m3_scale_shift_axis(self):
        config = tiny("scale_shift", scale_shift_min=0.25, scale_shift_max=4.0)
        stream = generate_stream(config, seed=0)
        scales = np.asarray(stream.scale_factors)
        assert np.all(scales >= 0.25) and np.all(scales <= 4.0)
        assert len(np.unique(np.round(scales, 12))) > 1
        regime_ids = np.asarray(stream.regime_ids)
        np.testing.assert_allclose(
            np.asarray(stream.x),
            scales[regime_ids][:, None] * np.asarray(stream.base_x),
            rtol=1e-6,
        )
        np.testing.assert_array_equal(
            np.asarray(stream.y), np.asarray(stream.base_y)
        )

    def test_m4_recurrence_axis(self):
        config = tiny("recurrence", n_regimes=8, recurrence_pool=3)
        stream = generate_stream(config, seed=0)
        pool_ids = np.asarray(stream.regime_pool_ids)
        assert pool_ids.shape == (8,)
        assert pool_ids.min() >= 0 and pool_ids.max() < 3
        # the first `pool` regimes visit each pool element once, in order
        np.testing.assert_array_equal(pool_ids[:3], np.arange(3))
        # revisits exist beyond the introduction phase
        assert len(pool_ids) > len(np.unique(pool_ids))
        # permutation rows follow the pool assignment: equal pool id, equal row
        perms = np.asarray(stream.permutations)
        for r, p in enumerate(pool_ids):
            np.testing.assert_array_equal(perms[r], perms[int(p)])

    def test_identity_assembly(self):
        stream = generate_stream(TINY, seed=0)
        n_regimes, d, c = TINY.n_regimes, TINY.dim, TINY.n_classes
        identity_perms = jnp.tile(jnp.arange(d, dtype=jnp.int32), (n_regimes, 1))
        identity_maps = jnp.tile(jnp.arange(c, dtype=jnp.int32), (n_regimes, 1))
        ones = jnp.ones(n_regimes, dtype=jnp.float32)
        x, y = assemble_observed(
            stream.base_x, stream.base_y, stream.regime_ids,
            identity_perms, identity_maps, ones,
        )
        np.testing.assert_array_equal(np.asarray(x), np.asarray(stream.base_x))
        np.testing.assert_array_equal(np.asarray(y), np.asarray(stream.base_y))


# =============================================================================
# Canonical suite: analytic Bayes reference
# =============================================================================


class TestBayesReference:
    def test_two_class_closed_form(self):
        # unit-variance dims, means +/- e1: Mahalanobis distance 2 -> Phi(1)
        mu0 = jnp.array([1.0, 0.0], dtype=jnp.float32)
        mu1 = jnp.array([-1.0, 0.0], dtype=jnp.float32)
        sigma = jnp.ones(2, dtype=jnp.float32)
        expected = 0.5 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
        assert two_class_bayes_accuracy(mu0, mu1, sigma) == pytest.approx(
            expected, abs=1e-6
        )

    def test_two_class_scale_invariance(self):
        mu0 = jnp.array([0.3, -0.2, 0.1], dtype=jnp.float32)
        mu1 = jnp.array([-0.1, 0.4, 0.0], dtype=jnp.float32)
        sigma = jnp.array([0.5, 1.5, 0.2], dtype=jnp.float32)
        base = two_class_bayes_accuracy(mu0, mu1, sigma)
        scaled = two_class_bayes_accuracy(3.0 * mu0, 3.0 * mu1, 3.0 * sigma)
        assert scaled == pytest.approx(base, abs=1e-6)

    def test_mc_reference_matches_two_class_closed_form(self):
        # n_components=1 restores unimodal clusters, where the closed form holds
        config = tiny("input_permutation", n_classes=2, dim=8, n_components=1)
        means, dim_sigma = class_geometry(config, seed=0)
        assert means.shape == (2, 1, 8)
        exact = two_class_bayes_accuracy(means[0, 0], means[1, 0], dim_sigma)
        reference = bayes_reference(config, seed=0, n_samples=200_000)
        assert isinstance(reference, BayesReference)
        assert reference.n_samples == 200_000
        assert reference.chance == pytest.approx(0.5)
        # 5 sigma of MC noise plus float slack
        assert reference.bayes_accuracy == pytest.approx(
            exact, abs=5.0 * reference.mc_sem + 1e-4
        )

    def test_reference_deterministic_and_bounded(self):
        a = bayes_reference(TINY, seed=0, n_samples=20_000)
        b = bayes_reference(TINY, seed=0, n_samples=20_000)
        assert a == b
        assert a.chance == pytest.approx(1.0 / TINY.n_classes)
        assert a.chance < a.bayes_accuracy <= 1.0
        assert a.mc_sem == pytest.approx(
            math.sqrt(a.bayes_accuracy * (1.0 - a.bayes_accuracy) / a.n_samples)
        )

    def test_bayes_rule_invariant_under_regime_transforms(self):
        """The reference applies to every regime of all four families.

        Coordinate permutation + global scaling act covariantly on the true
        generative parameters, so the induced Bayes rule makes identical
        predictions -- the reference accuracy is regime-invariant by
        construction (label permutation composes a bijection on both sides).
        """
        config = tiny("input_permutation", dim=5, n_classes=4)
        means, dim_sigma = class_geometry(config, seed=7)
        key = jr.key(123)
        x = means[0, 0] + dim_sigma * jr.normal(key, (64, config.dim), jnp.float32)
        base_predictions = bayes_predict(means, dim_sigma, x)
        perm = jr.permutation(jr.key(9), config.dim)
        c = 3.7
        transformed = bayes_predict(
            c * means[:, :, perm], c * dim_sigma[perm], c * x[:, perm]
        )
        np.testing.assert_array_equal(
            np.asarray(base_predictions), np.asarray(transformed)
        )

    def test_stream_geometry_matches_reference_geometry(self):
        stream = generate_stream(TINY, seed=5)
        means, dim_sigma = class_geometry(TINY, seed=5)
        np.testing.assert_array_equal(
            np.asarray(stream.component_means), np.asarray(means)
        )
        np.testing.assert_array_equal(
            np.asarray(stream.dim_sigma), np.asarray(dim_sigma)
        )

    def test_component_sparsity_is_localized(self):
        config = tiny("input_permutation", dim=6, component_sparsity=2)
        means, _ = class_geometry(config, seed=3)
        # each component displaces exactly `sparsity` dims from its class
        # center, so any two components of one class differ in at most
        # 2 * sparsity dimensions
        diffs = means[:, :, None, :] - means[:, None, :, :]
        active = np.asarray(jnp.sum(jnp.abs(diffs) > 0.0, axis=-1))
        assert active.max() <= 2 * config.component_sparsity

    def test_zero_component_scale_collapses_to_unimodal(self):
        config = tiny("input_permutation", component_scale=0.0)
        means, _ = class_geometry(config, seed=1)
        np.testing.assert_array_equal(
            np.asarray(means[:, 0, :]), np.asarray(means[:, 1, :])
        )


# =============================================================================
# Canonical suite: method-ladder arms
# =============================================================================


class TestArms:
    def test_ladder_registry(self):
        assert LADDER_ARMS == (
            "sgd_raw",
            "adamw",
            "upgd_raw",
            "sgd_norm",
            "gated_norm",
            "naive_bayes",
        )
        assert set(LADDER_ARMS) == set(MICRO_ARM_REGISTRY)
        for name in LADDER_ARMS:
            spec = micro_arm_spec(name)
            assert spec.name == name
            assert spec.description
            assert all(
                isinstance(v, int | float) for v in spec.hyperparameters.values()
            )

    def test_unknown_arm_rejected(self):
        with pytest.raises(KeyError, match="unknown micro arm"):
            micro_arm_spec("rff_rls")

    def test_raw_arms_use_published_protocol_hyperparameters(self):
        assert micro_arm_spec("upgd_raw").hyperparameters == dict(
            UPGD_W_PROTOCOL_HYPERPARAMETERS
        )
        assert micro_arm_spec("adamw").hyperparameters == dict(
            ADAMW_PROTOCOL_HYPERPARAMETERS
        )

    def test_sgd_norm_matches_campaign_conditioned_floor(self):
        """sgd_norm is the campaign's sgd_ema_norm_d099 row, key for key."""
        reference = SCREENING_REGISTRY["sgd_ema_norm_d099"].hyperparameters
        assert micro_arm_spec("sgd_norm").hyperparameters == reference

    def test_gated_norm_matches_campaign_champion(self):
        """gated_norm carries the sigma0_shiftnorm_d099 champion's operative
        hyperparameters (the registered extras are inert in its factory)."""
        champion = SCREENING_REGISTRY["sigma0_shiftnorm_d099"].hyperparameters
        ours = micro_arm_spec("gated_norm").hyperparameters
        for key_name in (
            "step_size",
            "utility_decay",
            "weight_decay",
            "norm_decay",
            "norm_epsilon",
            "fast_decay",
            "shift_k",
            "shift_delta",
            "shift_refractory",
        ):
            assert ours[key_name] == champion[key_name], key_name

    def test_naive_bayes_decay_matches_campaign(self):
        ours = micro_arm_spec("naive_bayes").hyperparameters
        assert (
            ours["nb_decay"]
            == SCREENING_REGISTRY["naive_bayes"].hyperparameters["nb_decay"]
        )
        # the variance floor is rescaled to the micro spectrum (documented
        # design choice, not a campaign transplant)
        assert ours["nb_var_epsilon"] < 0.1

    def test_sgd_raw_hand_pinned_step(self):
        net = IPMNISTConfig(
            n_tasks=1, task_length=1, input_dim=4, hidden1=3, hidden2=3, n_classes=2
        )
        params = init_mlp_params(jr.key(0), net)
        spec = micro_arm_spec("sgd_raw")
        init_fn, step_fn = spec.factory(spec.hyperparameters)
        state = init_fn(params)
        x = jnp.array([0.5, -1.0, 2.0, 0.0], dtype=jnp.float32)
        y = jnp.array(1, dtype=jnp.int32)
        (_, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        lr = spec.hyperparameters["step_size"]
        new_params, _, (accuracy, loss, _) = step_fn(params, state, x, y, jr.key(1))
        for name in params:
            np.testing.assert_allclose(
                np.asarray(new_params[name]),
                np.asarray(params[name] - lr * grads[name]),
                rtol=1e-6,
                atol=1e-8,
            )
        assert float(accuracy) == float(jnp.argmax(logits) == y)
        assert float(loss) > 0.0

    def test_grad_arm_parity_with_screening_factories(self):
        """One-step cross-module pin: micro raw arms == screening controls."""
        net = IPMNISTConfig(
            n_tasks=1, task_length=1, input_dim=5, hidden1=4, hidden2=3, n_classes=3
        )
        params = init_mlp_params(jr.key(3), net)
        x = jnp.array([0.1, -0.4, 0.8, 1.2, -2.0], dtype=jnp.float32)
        y = jnp.array(2, dtype=jnp.int32)
        key = jr.key(11)
        for micro_name, screening_name in (
            ("upgd_raw", "upgd_w_control"),
            ("adamw", "adamw_control"),
        ):
            micro = micro_arm_spec(micro_name)
            m_init, m_step = micro.factory(micro.hyperparameters)
            reference = screening_spec(screening_name)
            r_init, r_step = reference.factory(reference.hyperparameters)
            m_params, _, m_metrics = m_step(params, m_init(params), x, y, key)
            r_params, _, r_metrics = r_step(params, r_init(params), x, y, key)
            for name in params:
                np.testing.assert_array_equal(
                    np.asarray(m_params[name]), np.asarray(r_params[name])
                )
            assert float(m_metrics[0]) == float(r_metrics[0])


# =============================================================================
# Canonical suite: runner
# =============================================================================


class TestRunner:
    def test_run_shapes_and_metric(self):
        result = run_micro_arm(TINY, "sgd_raw", seed=0, hidden1=8, hidden2=6)
        assert result.per_regime_accuracy.shape == (TINY.n_regimes,)
        assert result.per_regime_loss.shape == (TINY.n_regimes,)
        assert result.per_regime_plasticity.shape == (TINY.n_regimes,)
        assert np.all(result.per_regime_accuracy >= 0.0)
        assert np.all(result.per_regime_accuracy <= 1.0)
        assert result.overall_accuracy == pytest.approx(
            float(result.per_regime_accuracy.mean())
        )
        assert result.wall_clock_seconds >= 0.0
        assert result.family == TINY.family
        assert result.arm_name == "sgd_raw"
        assert result.hidden1 == 8 and result.hidden2 == 6

    def test_run_deterministic(self):
        a = run_micro_arm(TINY, "naive_bayes", seed=1, hidden1=8, hidden2=6)
        b = run_micro_arm(TINY, "naive_bayes", seed=1, hidden1=8, hidden2=6)
        np.testing.assert_array_equal(a.per_regime_accuracy, b.per_regime_accuracy)

    def test_learning_happens_on_easy_stream(self):
        config = tiny(
            "input_permutation",
            n_regimes=2,
            regime_length=150,
            dim=8,
            n_classes=3,
            n_components=1,
            class_sparsity=1.0,
            mean_separation=3.0,
            spectrum_decades=0.0,
            offset_scale=0.0,
        )
        result = run_micro_arm(config, "sgd_norm", seed=0, hidden1=16, hidden2=8)
        # online accuracy while learning ends well above chance (1/3)
        assert result.per_regime_accuracy[-1] > 0.55


# =============================================================================
# Canonical suite: shards / merge
# =============================================================================


class TestShards:
    def _result(self, seed=0, arm="sgd_raw"):
        return run_micro_arm(TINY, arm, seed=seed, hidden1=8, hidden2=6)

    def test_payload_roundtrip(self, tmp_path: Path):
        result = self._result()
        payload = micro_shard_payload(result)
        assert payload["schema"] == MICRO_SHARD_SCHEMA
        assert payload["evidence_policy"] == NONPROMOTING_POLICY
        assert payload["family"] == TINY.family
        assert payload["stream_config"] == TINY.to_config()
        path = micro_shard_path(tmp_path, TINY.family, "sgd_raw", 0)
        write_micro_shard(path, payload)
        loaded = load_micro_shard(path)
        assert loaded["arm_name"] == "sgd_raw"
        assert loaded["seed"] == 0
        np.testing.assert_allclose(
            np.asarray(loaded["per_regime_accuracy"], dtype=np.float64),
            result.per_regime_accuracy,
            atol=1e-8,
        )

    def test_shard_path_convention(self, tmp_path: Path):
        path = micro_shard_path(tmp_path, "input_permutation", "gated_norm", 2)
        assert path.name == "input_permutation_gated_norm_seed2.json"

    def test_write_refuses_existing(self, tmp_path: Path):
        payload = micro_shard_payload(self._result())
        path = micro_shard_path(tmp_path, TINY.family, "sgd_raw", 0)
        write_micro_shard(path, payload)
        with pytest.raises(FileExistsError):
            write_micro_shard(path, payload)

    def test_load_rejects_wrong_schema(self, tmp_path: Path):
        payload = micro_shard_payload(self._result())
        payload["schema"] = "something.else.v1"
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="schema"):
            load_micro_shard(path)

    def test_load_rejects_unknown_arm(self, tmp_path: Path):
        payload = micro_shard_payload(self._result())
        payload["arm_name"] = "mystery"
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="arm"):
            load_micro_shard(path)

    def test_load_rejects_bad_curve(self, tmp_path: Path):
        payload = micro_shard_payload(self._result())
        payload["per_regime_accuracy"] = payload["per_regime_accuracy"][:-1]
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="per_regime_accuracy"):
            load_micro_shard(path)

    def test_merge_ranks_and_pairs(self, tmp_path: Path):
        paths = []
        for arm in ("sgd_raw", "naive_bayes"):
            for seed in (0, 1):
                result = run_micro_arm(TINY, arm, seed=seed, hidden1=8, hidden2=6)
                path = micro_shard_path(tmp_path, TINY.family, arm, seed)
                write_micro_shard(path, micro_shard_payload(result))
                paths.append(path)
        summary = merge_micro_shards(paths, bayes_samples=20_000)
        assert summary["schema"] == MICRO_SUMMARY_SCHEMA
        assert summary["family"] == TINY.family
        assert summary["stream_config"] == TINY.to_config()
        names = [entry["arm_name"] for entry in summary["results"]]
        assert set(names) == {"sgd_raw", "naive_bayes"}
        means = [entry["average_online_accuracy_mean"] for entry in summary["results"]]
        assert means == sorted(means, reverse=True)
        assert summary["bayes_reference"]["seeds"] == [0, 1]
        assert 0.0 < summary["bayes_reference"]["bayes_accuracy_mean"] <= 1.0
        assert summary["bayes_reference"]["chance"] == pytest.approx(1.0 / 3.0)

    def test_merge_rejects_mixed_configs(self, tmp_path: Path):
        result_a = run_micro_arm(TINY, "sgd_raw", seed=0, hidden1=8, hidden2=6)
        other = tiny("input_permutation", n_regimes=3)
        result_b = run_micro_arm(other, "sgd_raw", seed=0, hidden1=8, hidden2=6)
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        write_micro_shard(path_a, micro_shard_payload(result_a))
        write_micro_shard(path_b, micro_shard_payload(result_b))
        with pytest.raises(ValueError, match="stream config"):
            merge_micro_shards([path_a, path_b], bayes_samples=1_000)


# =============================================================================
# Canonical suite: transfer validation
# =============================================================================


def _synthetic_ladder(n_regimes=8, seeds=(0, 1)):
    """Curves engineered to reproduce the campaign's full-protocol ordering."""
    curves = {
        "sgd_raw": np.full(n_regimes, 0.60),
        "adamw": np.linspace(0.80, 0.70, n_regimes),  # early strong, decays
        "upgd_raw": np.concatenate([[0.69], np.full(n_regimes - 1, 0.785)]),
        "sgd_norm": np.full(n_regimes, 0.840),
        "gated_norm": np.full(n_regimes, 0.862),
        "naive_bayes": np.full(n_regimes, 0.790),
    }
    return {
        arm: {seed: curve + 0.001 * seed for seed in seeds}
        for arm, curve in curves.items()
    }


class TestTransferValidation:
    def test_engineered_pass(self):
        report = transfer_validation(_synthetic_ladder())
        assert report["schema"] == MICRO_VALIDATION_SCHEMA
        assert report["transfer_valid"] is True
        checks = {c["name"]: c for c in report["checks"]}
        for name in (
            "conditioning_dominates",
            "gate_small_positive",
            "adam_decays",
            "adam_below_upgd_raw",
            "naive_bayes_placement",
            "champion_top",
        ):
            assert checks[name]["passed"] is True, name
            assert checks[name]["campaign_reference"]
        assert checks["adam_fast_early"]["passed"] is True
        assert checks["adam_fast_early"]["primary"] is False

    def test_gate_negative_fails(self):
        ladder = _synthetic_ladder()
        ladder["gated_norm"] = {
            seed: curve - 0.05 for seed, curve in ladder["gated_norm"].items()
        }
        report = transfer_validation(ladder)
        checks = {c["name"]: c for c in report["checks"]}
        assert checks["gate_small_positive"]["passed"] is False
        assert report["transfer_valid"] is False

    def test_nondecaying_adam_fails(self):
        ladder = _synthetic_ladder()
        ladder["adamw"] = {seed: np.linspace(0.70, 0.75, 8) for seed in (0, 1)}
        report = transfer_validation(ladder)
        checks = {c["name"]: c for c in report["checks"]}
        assert checks["adam_decays"]["passed"] is False
        assert report["transfer_valid"] is False

    def test_naive_bayes_above_conditioning_fails(self):
        ladder = _synthetic_ladder()
        ladder["naive_bayes"] = {seed: np.full(8, 0.90) for seed in (0, 1)}
        report = transfer_validation(ladder)
        checks = {c["name"]: c for c in report["checks"]}
        assert checks["naive_bayes_placement"]["passed"] is False
        assert report["transfer_valid"] is False

    def test_missing_arm_rejected(self):
        ladder = _synthetic_ladder()
        del ladder["naive_bayes"]
        with pytest.raises(ValueError, match="naive_bayes"):
            transfer_validation(ladder)

    def test_mismatched_seeds_rejected(self):
        ladder = _synthetic_ladder()
        del ladder["adamw"][1]
        with pytest.raises(ValueError, match="seed"):
            transfer_validation(ladder)

    def test_from_shards(self, tmp_path: Path):
        paths = []
        for arm in LADDER_ARMS:
            result = run_micro_arm(TINY, arm, seed=0, hidden1=8, hidden2=6)
            path = micro_shard_path(tmp_path, TINY.family, arm, 0)
            write_micro_shard(path, micro_shard_payload(result))
            paths.append(path)
        report = transfer_validation_from_shards(paths)
        assert report["schema"] == MICRO_VALIDATION_SCHEMA
        assert report["family"] == TINY.family
        assert isinstance(report["transfer_valid"], bool)
        assert {c["name"] for c in report["checks"]} >= {
            "conditioning_dominates",
            "gate_small_positive",
            "adam_decays",
        }

    def test_from_shards_rejects_non_m1(self, tmp_path: Path):
        config = tiny("scale_shift")
        result = run_micro_arm(config, "sgd_raw", seed=0, hidden1=8, hidden2=6)
        path = micro_shard_path(tmp_path, config.family, "sgd_raw", 0)
        write_micro_shard(path, micro_shard_payload(result))
        with pytest.raises(ValueError, match="input_permutation"):
            transfer_validation_from_shards([path])


# =============================================================================
# Canonical suite: CLI
# =============================================================================


@pytest.mark.integration
class TestCLI:
    ARGS = [
        "--n-regimes", "3", "--regime-length", "20", "--dim", "6",
        "--n-classes", "3", "--n-components", "2", "--component-sparsity", "2",
        "--class-sparsity", "0.5", "--hidden1", "8", "--hidden2", "6",
    ]

    def test_run_writes_shard_and_is_idempotent(self, tmp_path: Path):
        argv = [
            "run", "--family", "input_permutation", "--arm", "sgd_raw",
            "--seed", "0", "--out", str(tmp_path), *self.ARGS,
        ]
        assert main(argv) == 0
        path = micro_shard_path(tmp_path, "input_permutation", "sgd_raw", 0)
        assert path.exists()
        first = path.read_bytes()
        assert main(argv) == 0  # idempotent skip, not an overwrite
        assert path.read_bytes() == first

    def test_ladder_partial_arms_writes_summary_only(self, tmp_path: Path):
        argv = [
            "ladder", "--family", "input_permutation", "--seeds", "0",
            "--arms", "sgd_raw", "naive_bayes",
            "--out", str(tmp_path), "--bayes-samples", "5000", *self.ARGS,
        ]
        assert main(argv) == 0
        assert (tmp_path / "summary_input_permutation.json").exists()
        assert not (tmp_path / "transfer_input_permutation.json").exists()

    def test_ladder_full_writes_validation(self, tmp_path: Path):
        argv = [
            "ladder", "--family", "input_permutation", "--seeds", "0",
            "--out", str(tmp_path), "--bayes-samples", "5000", *self.ARGS,
        ]
        code = main(argv)
        transfer_path = tmp_path / "transfer_input_permutation.json"
        assert transfer_path.exists()
        report = json.loads(transfer_path.read_text(encoding="utf-8"))
        assert report["schema"] == MICRO_VALIDATION_SCHEMA
        # exit code mirrors the verdict: 0 = ordering reproduced, 2 = not
        assert code == (0 if report["transfer_valid"] else 2)


# =============================================================================
# Provisional digits-based suite (rule-discovery track compatibility)
# =============================================================================


def test_suite_registry_roles_and_disjointness() -> None:
    assert set(SEARCH_TASKS) == {"M1", "M2", "M3"}
    assert set(HOLDOUT_TASKS) == {"M4", "M1p"}
    assert set(SEARCH_TASKS) | set(HOLDOUT_TASKS) == set(MICRO_SUITE)
    assert not set(SEARCH_TASKS) & set(HOLDOUT_TASKS)
    for name in SEARCH_TASKS:
        assert MICRO_SUITE[name].role == "search"
    for name in HOLDOUT_TASKS:
        assert MICRO_SUITE[name].role == "holdout"
    assert isinstance(MICRO_SUITE_VERSION, str) and MICRO_SUITE_VERSION


def test_task_kinds_cover_the_nonstationarity_axes() -> None:
    assert MICRO_SUITE["M1"].kind == "input_permutation"
    assert MICRO_SUITE["M2"].kind == "label_permutation"
    assert MICRO_SUITE["M3"].kind == "affine_drift"
    assert MICRO_SUITE["M4"].kind == "permutation_affine"
    assert MICRO_SUITE["M1p"].kind == "input_permutation"
    # M1p is the differently-parameterized M1: different dimensionality
    # (cropped digits) and different task geometry.
    assert MICRO_SUITE["M1p"].input_dim != MICRO_SUITE["M1"].input_dim
    assert (
        MICRO_SUITE["M1p"].task_length != MICRO_SUITE["M1"].task_length
        or MICRO_SUITE["M1p"].n_tasks != MICRO_SUITE["M1"].n_tasks
    )


def test_digits_features_shapes_and_scaling() -> None:
    x_full, y_full = load_digits_features(crop=False)
    x_crop, y_crop = load_digits_features(crop=True)
    assert x_full.shape[1] == 64
    assert x_crop.shape[1] == 49
    assert x_full.shape[0] == x_crop.shape[0] == y_full.shape[0] == y_crop.shape[0]
    assert x_full.dtype == np.float32 and x_crop.dtype == np.float32
    # digits raw range 0..16 mapped into [-1, 1] (protocol MNIST convention)
    assert float(x_full.min()) >= -1.0 and float(x_full.max()) <= 1.0
    assert set(np.unique(y_full)) <= set(range(10))


def _small(config: MicroTaskConfig) -> MicroTaskConfig:
    return dataclasses.replace(config, n_tasks=3, task_length=25)


def test_stream_shapes_dtypes_and_determinism() -> None:
    for name in ("M1", "M2", "M3", "M4", "M1p"):
        config = _small(MICRO_SUITE[name])
        stream = build_micro_stream(config, seed=7)
        assert isinstance(stream, MicroStream)
        n_steps = config.n_tasks * config.task_length
        assert stream.xs.shape == (n_steps, config.input_dim)
        assert stream.ys.shape == (n_steps,)
        assert stream.example_indices.shape == (n_steps,)
        assert stream.xs.dtype == np.float32
        assert stream.ys.dtype == np.int32
        assert int(stream.ys.max()) < config.n_classes
        again = build_micro_stream(config, seed=7)
        np.testing.assert_array_equal(stream.xs, again.xs)
        np.testing.assert_array_equal(stream.ys, again.ys)
        other = build_micro_stream(config, seed=8)
        assert not np.array_equal(stream.xs, other.xs)


def test_m1_blocks_are_permutations_of_the_base_features() -> None:
    config = _small(MICRO_SUITE["M1"])
    stream = build_micro_stream(config, seed=3)
    x_base, y_base = load_digits_features(crop=False)
    # Every step's feature multiset equals the base example's multiset, and
    # labels are untouched.
    for step in (0, 30, 70):
        idx = int(stream.example_indices[step])
        np.testing.assert_allclose(
            np.sort(stream.xs[step]), np.sort(x_base[idx]), rtol=0, atol=0
        )
        assert int(stream.ys[step]) == int(y_base[idx])
    # Tasks use different permutations: same example transformed differently
    # in different task blocks (with overwhelming probability).
    first = stream.xs[: config.task_length]
    second = stream.xs[config.task_length : 2 * config.task_length]
    assert not np.array_equal(first[:5], second[:5])


def test_m2_permutes_labels_but_not_inputs() -> None:
    config = _small(MICRO_SUITE["M2"])
    stream = build_micro_stream(config, seed=3)
    x_base, y_base = load_digits_features(crop=False)
    for step in (0, 40):
        idx = int(stream.example_indices[step])
        np.testing.assert_array_equal(stream.xs[step], x_base[idx])
    # Within one task the label map is a bijection applied consistently.
    task0 = slice(0, config.task_length)
    raw = y_base[stream.example_indices[task0]]
    mapped = stream.ys[task0]
    pairs = {(int(a), int(b)) for a, b in zip(raw, mapped)}
    assert len({a for a, _ in pairs}) == len({b for _, b in pairs})


def test_m3_applies_per_task_affine_transform() -> None:
    config = _small(MICRO_SUITE["M3"])
    stream = build_micro_stream(config, seed=5)
    x_base, y_base = load_digits_features(crop=False)
    idx0 = int(stream.example_indices[0])
    assert int(stream.ys[0]) == int(y_base[idx0])
    # Affine per feature: xs = scale * x + offset with per-task constants, so
    # two steps in the same task sharing an example agree exactly; across
    # tasks the transform differs.
    assert not np.array_equal(stream.xs[0], x_base[idx0])
