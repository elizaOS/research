"""Tests for the UPGD Input-permuted MNIST replication lane.

CI-cheap: everything runs on tiny synthetic data. Real-MNIST benchmark runs
happen only through ``python -m alberta_framework.benchmarks.upgd_ipmnist``.
"""

import json

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks.upgd_ipmnist import (
    ADAMW_PROTOCOL_HYPERPARAMETERS,
    ARTIFACT_SCHEMA,
    PAPER_REFERENCE,
    PARTIAL_SCHEMA,
    UPGD_W_PROTOCOL_HYPERPARAMETERS,
    IPMNISTConfig,
    LeanUPGDState,
    _split_flat_noise,
    build_artifact,
    build_comparison,
    build_schedule,
    canonical_upgd_w,
    lean_upgd_w_update,
    merge_partial_results,
    partial_payload,
    resolve_hyperparameters,
    run_ipmnist,
    summarize_result,
    task_index_for_step,
)

TINY = IPMNISTConfig(n_tasks=2, task_length=200, input_dim=16, hidden1=32, hidden2=16)
N_TRAIN = 300


def _synthetic_dataset(
    seed: int, n_train: int, input_dim: int, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Clusterable synthetic stream: 10 gaussian prototypes plus noise."""
    key_centers, key_labels, key_noise = jr.split(jr.key(seed), 3)
    centers = jr.normal(key_centers, (n_classes, input_dim), dtype=jnp.float32)
    y = jr.randint(key_labels, (n_train,), 0, n_classes)
    x = centers[y] + 0.3 * jr.normal(key_noise, (n_train, input_dim), dtype=jnp.float32)
    return np.asarray(x), np.asarray(y.astype(jnp.int32))


class TestProtocolConstants:
    def test_default_config_matches_selected_publication_shape(self):
        config = IPMNISTConfig()
        assert config.n_tasks == 200
        assert config.task_length == 5000
        assert config.n_steps == 1_000_000
        assert config.input_dim == 784
        assert (config.hidden1, config.hidden2) == (300, 150)
        assert config.n_classes == 10
        assert config.matches_selected_publication_configuration

    def test_shrunk_config_does_not_match_selected_publication_shape(self):
        assert not TINY.matches_selected_publication_configuration

    def test_published_hyperparameters(self):
        assert UPGD_W_PROTOCOL_HYPERPARAMETERS == {
            "step_size": 0.01,
            "utility_decay": 0.9999,
            "noise_std": 0.1,
            "weight_decay": 0.01,
        }
        assert ADAMW_PROTOCOL_HYPERPARAMETERS == {
            "step_size": 1e-4,
            "beta1": 0.0,
            "beta2": 0.99,
            "eps": 1e-8,
            "weight_decay": 0.0,
        }

    def test_resolve_hyperparameters_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="unknown hyperparameters"):
            resolve_hyperparameters("upgd_w", {"lr": 0.1})
        with pytest.raises(ValueError, match="unknown learner"):
            resolve_hyperparameters("sgd", None)


class TestSchedule:
    def test_task_index_changes_exactly_at_multiples_of_task_length(self):
        length = TINY.task_length
        steps = np.arange(3 * length)
        tasks = np.asarray(task_index_for_step(steps, length))
        # Exact boundary steps: last step of task t and first step of task t+1.
        assert tasks[0] == 0
        assert tasks[length - 1] == 0
        assert tasks[length] == 1
        assert tasks[2 * length - 1] == 1
        assert tasks[2 * length] == 2
        # Each task owns exactly task_length consecutive steps.
        np.testing.assert_array_equal(np.bincount(tasks), [length, length, length])

    def test_permutations_are_valid_and_task_indexed(self):
        schedule = build_schedule(jr.key(0), TINY, N_TRAIN)
        perms = np.asarray(schedule.permutations)
        assert perms.shape == (TINY.n_tasks, TINY.input_dim)
        for row in perms:
            np.testing.assert_array_equal(np.sort(row), np.arange(TINY.input_dim))
        # The first task is itself permuted-in-general (schedule rows differ).
        assert not np.array_equal(perms[0], perms[1])

    def test_example_indices_sample_without_replacement(self):
        schedule = build_schedule(jr.key(1), TINY, N_TRAIN)
        examples = np.asarray(schedule.example_indices)
        assert examples.shape == (TINY.n_tasks, TINY.task_length)
        assert examples.min() >= 0
        assert examples.max() < N_TRAIN
        for row in examples:
            assert len(np.unique(row)) == TINY.task_length

    def test_schedule_is_deterministic_per_key(self):
        first = build_schedule(jr.key(7), TINY, N_TRAIN)
        second = build_schedule(jr.key(7), TINY, N_TRAIN)
        np.testing.assert_array_equal(
            np.asarray(first.permutations), np.asarray(second.permutations)
        )
        np.testing.assert_array_equal(
            np.asarray(first.example_indices), np.asarray(second.example_indices)
        )

    def test_schedule_rejects_dataset_smaller_than_task(self):
        with pytest.raises(ValueError, match="without replacement"):
            build_schedule(jr.key(0), TINY, TINY.task_length - 1)


@pytest.fixture(scope="module")
def debug_run():
    data_x, data_y = _synthetic_dataset(0, N_TRAIN, TINY.input_dim, TINY.n_classes)
    return run_ipmnist(
        data_x,
        data_y,
        "upgd_w",
        seeds=(0, 1),
        config=TINY,
        return_per_step=True,
    ), (data_x, data_y)


class TestAccuracyAccounting:

    def test_per_task_accuracy_is_mean_of_per_step(self, debug_run):
        result, _ = debug_run
        assert result.per_step_accuracy.shape == (2, TINY.n_tasks, TINY.task_length)
        np.testing.assert_allclose(
            result.per_task_accuracy,
            result.per_step_accuracy.mean(axis=2),
            atol=1e-6,
        )

    def test_per_step_accuracy_is_zero_or_one(self, debug_run):
        result, _ = debug_run
        assert set(np.unique(result.per_step_accuracy)) <= {0.0, 1.0}

    def test_average_online_accuracy_is_mean_over_tasks(self, debug_run):
        result, _ = debug_run
        np.testing.assert_allclose(
            result.average_online_accuracy,
            result.per_task_accuracy.mean(axis=1),
            atol=1e-7,
        )

    def test_first_step_accuracy_recomputed_externally(self, debug_run):
        """Step 0's accuracy must equal an out-of-band numpy forward pass."""
        result, (data_x, data_y) = debug_run
        for seed_index in range(2):
            permutation = result.permutations[seed_index, 0]
            example = int(result.example_indices[seed_index, 0, 0])
            x = data_x[example][permutation]
            hidden = np.maximum(
                x @ result.initial_params["w1"][seed_index]
                + result.initial_params["b1"][seed_index],
                0.0,
            )
            hidden = np.maximum(
                hidden @ result.initial_params["w2"][seed_index]
                + result.initial_params["b2"][seed_index],
                0.0,
            )
            logits = (
                hidden @ result.initial_params["w3"][seed_index]
                + result.initial_params["b3"][seed_index]
            )
            expected = float(np.argmax(logits) == data_y[example])
            assert result.per_step_accuracy[seed_index, 0, 0] == pytest.approx(expected)

    def test_losses_and_plasticity_within_bounds(self, debug_run):
        result, _ = debug_run
        assert np.all(np.isfinite(result.per_task_loss))
        assert np.all(result.per_task_loss > 0.0)
        assert np.all(result.per_task_plasticity >= 0.0)
        assert np.all(result.per_task_plasticity <= 1.0)


class TestSmoke:
    """Tiny-scale smoke: 2 tasks x 200 steps, both learners, 2 seeds."""

    @pytest.mark.parametrize("learner", ["upgd_w", "adamw"])
    def test_published_hyperparameters_run_within_bounds(self, learner):
        """Published hyperparameters are tuned for 5,000-step MNIST tasks;
        at 200 tiny synthetic steps we assert sanity, not learning."""
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        result = run_ipmnist(data_x, data_y, learner, seeds=(0, 1), config=TINY)
        assert result.per_task_accuracy.shape == (2, TINY.n_tasks)
        assert np.all(np.isfinite(result.per_task_accuracy))
        assert np.all(result.per_task_accuracy >= 0.0)
        assert np.all(result.per_task_accuracy <= 1.0)
        assert np.all(np.isfinite(result.per_task_loss))

    @pytest.mark.parametrize(
        ("learner", "overrides", "threshold"),
        [
            # Calibrated on seeds 0-5: upgd_w reaches >=0.31 mean, adamw
            # >=0.85; thresholds sit at 2x chance (0.1) and below half the
            # measured means, respectively.
            ("upgd_w", {"step_size": 0.1, "noise_std": 0.01}, 0.2),
            ("adamw", {"step_size": 0.01}, 0.4),
        ],
    )
    def test_learner_learns_above_chance_at_smoke_scale(self, learner, overrides, threshold):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        result = run_ipmnist(
            data_x, data_y, learner, seeds=(0, 1), config=TINY, hyperparameters=overrides
        )
        assert result.per_task_accuracy.mean() > threshold

    def test_runs_are_deterministic_per_seed(self):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        first = run_ipmnist(data_x, data_y, "upgd_w", seeds=(5,), config=TINY)
        second = run_ipmnist(data_x, data_y, "upgd_w", seeds=(5,), config=TINY)
        np.testing.assert_array_equal(first.per_task_accuracy, second.per_task_accuracy)

    def test_hyperparameter_override_is_recorded(self):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        result = run_ipmnist(
            data_x,
            data_y,
            "adamw",
            seeds=(0,),
            config=TINY,
            hyperparameters={"weight_decay": 0.01},
        )
        assert result.hyperparameters["weight_decay"] == pytest.approx(0.01)


class TestLeanUPGDParity:
    """The lean benchmark step must match CanonicalUPGD exactly.

    ``CanonicalUPGD(profile="official_experiment_global", mode="protecting")``
    is the audited equation source; the lean step exists purely for CPU scan
    speed. Both consume the same supplied noise, so agreement is equation
    exactness, not luck.
    """

    def test_multi_step_parity_with_canonical(self):
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        shapes = {"w1": (7, 5), "b1": (5,), "w2": (5, 3), "b2": (3,)}
        key = jr.key(42)
        key, params_key = jr.split(key)
        params = {
            name: 0.3 * jr.normal(jr.fold_in(params_key, i), shape, jnp.float32)
            for i, (name, shape) in enumerate(sorted(shapes.items()))
        }
        canonical = canonical_upgd_w(hp)
        canonical_params = dict(params)
        canonical_state = canonical.init(canonical_params)
        lean_params = dict(params)
        lean_state = LeanUPGDState(
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )
        for step in range(5):
            key, grads_key, noise_key = jr.split(key, 3)
            grads = {
                name: jr.normal(jr.fold_in(grads_key, i), shape, jnp.float32)
                for i, (name, shape) in enumerate(sorted(shapes.items()))
            }
            noise = {
                name: hp["noise_std"]
                * jr.normal(jr.fold_in(noise_key, i), shape, jnp.float32)
                for i, (name, shape) in enumerate(sorted(shapes.items()))
            }
            update = canonical.update(
                canonical_state, canonical_params, grads, jr.fold_in(key, step), noise=noise
            )
            canonical_params, canonical_state = update.params, update.state
            lean_params, lean_state = lean_upgd_w_update(
                lean_params, lean_state, grads, noise, hp
            )
            for name in shapes:
                np.testing.assert_allclose(
                    np.asarray(lean_params[name]),
                    np.asarray(canonical_params[name]),
                    atol=1e-6,
                    err_msg=f"step {step} param {name}",
                )
                np.testing.assert_allclose(
                    np.asarray(lean_state.utility[name]),
                    np.asarray(canonical_state.utility_ema[name]),
                    atol=1e-6,
                    err_msg=f"step {step} utility {name}",
                )


class TestSplitFlatNoise:
    def test_slices_in_sorted_name_order_with_exact_values(self):
        shapes = {"b1": (5,), "b2": (3,), "w1": (7, 5), "w2": (5, 3)}
        total = sum(int(np.prod(shape)) for shape in shapes.values())
        flat = jr.normal(jr.key(0), (total,), jnp.float32)
        split = _split_flat_noise(flat, shapes)
        assert set(split) == set(shapes)
        offset = 0
        for name in sorted(shapes):
            count = int(np.prod(shapes[name]))
            assert split[name].shape == shapes[name]
            np.testing.assert_array_equal(
                np.asarray(split[name]).ravel(),
                np.asarray(flat[offset:offset + count]),
            )
            offset += count
        assert offset == total


class TestRunnerTrajectoryExactness:
    """run_ipmnist's optimized inner loop vs a plain dict-based reference loop.

    The reference reproduces the exact published RNG stream contract: one
    ``jr.split`` per step of the per-seed noise-key chain, one flat
    ``N(0, sigma^2)`` draw sliced per sorted parameter name.
    """

    SMALL = IPMNISTConfig(n_tasks=2, task_length=50, input_dim=16, hidden1=32, hidden2=16)

    @pytest.mark.parametrize("learner", ["upgd_w", "adamw"])
    def test_matches_reference_loop(self, learner):
        import jax

        from alberta_framework.benchmarks.upgd_ipmnist import (
            _LEARNER_FACTORIES,
            cross_entropy_loss,
            init_mlp_params,
        )

        config = self.SMALL
        data_x, data_y = _synthetic_dataset(11, N_TRAIN, config.input_dim, config.n_classes)
        result = run_ipmnist(
            data_x, data_y, learner, seeds=(0,), config=config, return_per_step=True
        )

        hp = resolve_hyperparameters(learner)
        init_fn, step_fn = _LEARNER_FACTORIES[learner](hp)
        root = jr.key(jnp.uint32(0))
        key_init, key_schedule, key_noise = jr.split(root, 3)
        params = init_mlp_params(key_init, config)
        schedule = build_schedule(key_schedule, config, N_TRAIN)
        state = init_fn(params)
        key = key_noise
        xs = jnp.asarray(data_x, jnp.float32)
        ys = jnp.asarray(data_y, jnp.int32)
        accuracies = np.zeros((config.n_tasks, config.task_length))
        for task in range(config.n_tasks):
            permutation = schedule.permutations[task]
            for i in range(config.task_length):
                example = schedule.example_indices[task, i]
                x = xs[example][permutation]
                y = ys[example]
                (_, logits), grads = jax.value_and_grad(
                    cross_entropy_loss, has_aux=True
                )(params, x, y)
                accuracies[task, i] = float(jnp.argmax(logits) == y)
                key, step_key = jr.split(key)
                params, state = step_fn(params, state, grads, step_key)
        np.testing.assert_array_equal(result.per_step_accuracy[0], accuracies)


class TestNoisePoolMode:
    def test_noise_mode_recorded_and_defaults_to_step(self):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        exact = run_ipmnist(data_x, data_y, "upgd_w", seeds=(0,), config=TINY)
        assert exact.noise_mode == "step"
        pooled = run_ipmnist(
            data_x, data_y, "upgd_w", seeds=(0,), config=TINY,
            noise_mode="pool", noise_pool_steps=4,
        )
        assert pooled.noise_mode == "pool"

    def test_pool_mode_is_deterministic_and_bounded(self):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        first = run_ipmnist(
            data_x, data_y, "upgd_w", seeds=(2,), config=TINY,
            noise_mode="pool", noise_pool_steps=4,
        )
        second = run_ipmnist(
            data_x, data_y, "upgd_w", seeds=(2,), config=TINY,
            noise_mode="pool", noise_pool_steps=4,
        )
        np.testing.assert_array_equal(first.per_task_accuracy, second.per_task_accuracy)
        assert np.all(np.isfinite(first.per_task_loss))
        assert np.all(first.per_task_accuracy >= 0.0)
        assert np.all(first.per_task_accuracy <= 1.0)

    def test_pool_mode_changes_upgd_but_not_adamw(self):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        upgd_exact = run_ipmnist(data_x, data_y, "upgd_w", seeds=(0,), config=TINY)
        upgd_pool = run_ipmnist(
            data_x, data_y, "upgd_w", seeds=(0,), config=TINY,
            noise_mode="pool", noise_pool_steps=4,
        )
        assert not np.array_equal(upgd_exact.per_task_loss, upgd_pool.per_task_loss)

        adamw_exact = run_ipmnist(data_x, data_y, "adamw", seeds=(0,), config=TINY)
        adamw_pool = run_ipmnist(
            data_x, data_y, "adamw", seeds=(0,), config=TINY,
            noise_mode="pool", noise_pool_steps=4,
        )
        np.testing.assert_array_equal(
            adamw_exact.per_task_accuracy, adamw_pool.per_task_accuracy
        )
        np.testing.assert_array_equal(adamw_exact.per_task_loss, adamw_pool.per_task_loss)

    def test_pool_mode_rejects_invalid_pool_size(self):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        with pytest.raises(ValueError, match="noise_pool_steps"):
            run_ipmnist(
                data_x, data_y, "upgd_w", seeds=(0,), config=TINY,
                noise_mode="pool", noise_pool_steps=1,
            )

    def test_pool_mode_shard_serialization_fails_closed(self):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        pooled = run_ipmnist(
            data_x, data_y, "upgd_w", seeds=(0,), config=TINY,
            noise_mode="pool", noise_pool_steps=4,
        )
        with pytest.raises(ValueError, match="noise_mode"):
            partial_payload(pooled)


class TestPartialMerge:
    def test_partial_roundtrip_and_merge(self, tmp_path):
        data_x, data_y = _synthetic_dataset(6, N_TRAIN, TINY.input_dim, TINY.n_classes)
        shard_a = run_ipmnist(data_x, data_y, "upgd_w", seeds=(1,), config=TINY)
        shard_b = run_ipmnist(data_x, data_y, "upgd_w", seeds=(0,), config=TINY)
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        path_a.write_text(json.dumps(partial_payload(shard_a)))
        path_b.write_text(json.dumps(partial_payload(shard_b)))
        merged = merge_partial_results([path_a, path_b])
        result = merged["upgd_w"]
        assert result.seeds == (0, 1)  # sorted by seed, not file order
        assert result.per_task_accuracy.shape == (2, TINY.n_tasks)
        np.testing.assert_allclose(
            result.per_task_accuracy[1], shard_a.per_task_accuracy[0], atol=1e-9
        )
        np.testing.assert_allclose(
            result.per_task_accuracy[0], shard_b.per_task_accuracy[0], atol=1e-9
        )

    def test_merge_rejects_duplicate_seeds_and_config_mismatch(self, tmp_path):
        data_x, data_y = _synthetic_dataset(6, N_TRAIN, TINY.input_dim, TINY.n_classes)
        shard = run_ipmnist(data_x, data_y, "upgd_w", seeds=(0,), config=TINY)
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        path_a.write_text(json.dumps(partial_payload(shard)))
        path_b.write_text(json.dumps(partial_payload(shard)))
        with pytest.raises(ValueError, match="duplicate seeds"):
            merge_partial_results([path_a, path_b])

        other_config = IPMNISTConfig(
            n_tasks=2, task_length=200, input_dim=16, hidden1=32, hidden2=8
        )
        other = run_ipmnist(data_x, data_y, "upgd_w", seeds=(1,), config=other_config)
        path_b.write_text(json.dumps(partial_payload(other)))
        with pytest.raises(ValueError, match="disagree on config"):
            merge_partial_results([path_a, path_b])

    def test_v2_partial_is_single_seed_and_recursively_omits_legacy_marker(
        self, tmp_path, debug_run
    ):
        multi_seed, _ = debug_run
        with pytest.raises(ValueError, match="exactly one seed"):
            partial_payload(multi_seed)

        data_x, data_y = _synthetic_dataset(9, N_TRAIN, TINY.input_dim, TINY.n_classes)
        one_seed = run_ipmnist(data_x, data_y, "upgd_w", seeds=(3,), config=TINY)
        payload = partial_payload(one_seed)
        assert payload["schema"] == PARTIAL_SCHEMA
        assert payload["seed_id"] == 3
        assert payload["seed_count"] == 1
        assert "is_protocol_exact" not in json.dumps(payload)

        path = tmp_path / "v2.json"
        payload["is_protocol_exact"] = True
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="fields do not match"):
            merge_partial_results([path])

    def test_v2_artifact_records_scoped_match_and_exact_seed_facts(self, tmp_path):
        data_x, data_y = _synthetic_dataset(10, N_TRAIN, TINY.input_dim, TINY.n_classes)
        run = run_ipmnist(data_x, data_y, "adamw", seeds=(7,), config=TINY)
        artifact = build_artifact({"adamw": run}, TINY, tmp_path / "cache")

        assert artifact["schema"] == ARTIFACT_SCHEMA
        assert artifact["schema_version"] == 2
        assert artifact["protocol"]["matches_selected_publication_configuration"] is False
        assert artifact["protocol"]["selected_publication_configuration_match_scope"] == (
            "network_task_shape_and_horizon_only"
        )
        assert artifact["study_design"]["exact_seed_ids_by_learner"] == {"adamw": [7]}
        assert artifact["study_design"]["exact_seed_count_by_learner"] == {"adamw": 1}
        assert artifact["study_design"]["published_seed_count"] == 20
        assert artifact["evidence_policy"]["scientific_promotion_allowed"] is False
        assert "is_protocol_exact" not in json.dumps(artifact)


class TestSummariesAndComparison:
    def test_summary_and_comparison_flag_logic(self):
        data_x, data_y = _synthetic_dataset(4, N_TRAIN, TINY.input_dim, TINY.n_classes)
        result = run_ipmnist(data_x, data_y, "upgd_w", seeds=(0, 1), config=TINY)
        summary = summarize_result(result)
        assert summary["n_seeds"] == 2
        assert len(summary["per_task_accuracy_mean"]) == TINY.n_tasks
        assert summary["average_online_accuracy_mean"] == pytest.approx(
            float(result.average_online_accuracy.mean())
        )

        # Synthetic comparison: a 0.03 gap must be flagged, 0.01 must not.
        flagged = build_comparison(
            {"upgd_w": {**summary, "average_online_accuracy_mean": 0.75}}
        )
        assert flagged["learners"]["upgd_w"]["reproduction_gap_flagged"]
        clean = build_comparison(
            {"upgd_w": {**summary, "average_online_accuracy_mean": 0.79}}
        )
        assert not clean["learners"]["upgd_w"]["reproduction_gap_flagged"]
        assert clean["reference"] is PAPER_REFERENCE
