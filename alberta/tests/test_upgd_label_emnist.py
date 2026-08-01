"""Tests for the Label-permuted EMNIST replication lane.

Covers schedule exactness (task boundaries, cumulative label-permutation
composition, without-replacement sampling), plan/shard/merge accounting, and a
tiny synthetic smoke run. Benchmark executions never happen inside pytest.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks.upgd_label_emnist import (
    ADAMW_PROTOCOL_HYPERPARAMETERS,
    UPGD_W_PROTOCOL_HYPERPARAMETERS,
    LabelEMNISTConfig,
    build_artifact,
    build_comparison,
    build_plan_payload,
    build_schedule,
    load_plan,
    merge_partials,
    partial_payload,
    resolve_hyperparameters,
    run_label_emnist,
    summarize_result,
    task_index_for_step,
)

TINY = LabelEMNISTConfig(
    n_tasks=3, task_length=8, input_dim=6, hidden1=8, hidden2=4, n_classes=5
)

DATASET_META = {
    "source": "synthetic:test",
    "train_rows_used": 50,
    "x_sha256": "0" * 64,
    "y_sha256": "1" * 64,
}


def _tiny_data(n_train: int = 50) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_train, TINY.input_dim)).astype(np.float32)
    y = rng.integers(0, TINY.n_classes, size=n_train).astype(np.int32)
    return x, y


class TestConfig:
    def test_default_config_matches_selected_publication_shape(self):
        config = LabelEMNISTConfig()
        assert config.n_tasks == 400
        assert config.task_length == 2500
        assert config.n_steps == 1_000_000
        assert (config.input_dim, config.hidden1, config.hidden2) == (784, 300, 150)
        assert config.n_classes == 47
        assert config.matches_selected_publication_configuration

    def test_shrunk_config_does_not_match_selected_publication_shape(self):
        assert not TINY.matches_selected_publication_configuration

    def test_published_hyperparameters(self):
        assert UPGD_W_PROTOCOL_HYPERPARAMETERS == {
            "step_size": 0.01,
            "utility_decay": 0.9,
            "noise_std": 0.001,
            "weight_decay": 0.0,
        }
        assert ADAMW_PROTOCOL_HYPERPARAMETERS == {
            "step_size": 1e-4,
            "beta1": 0.0,
            "beta2": 0.9999,
            "eps": 1e-8,
            "weight_decay": 0.1,
        }

    def test_resolve_hyperparameters_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="unknown hyperparameters"):
            resolve_hyperparameters("upgd_w", {"sigma": 0.1})
        with pytest.raises(ValueError, match="unknown learner"):
            resolve_hyperparameters("sgd")


class TestScheduleExactness:
    def test_task_index_changes_exactly_at_multiples_of_task_length(self):
        length = TINY.task_length
        for task in range(TINY.n_tasks):
            assert task_index_for_step(task * length, length) == task
            assert task_index_for_step((task + 1) * length - 1, length) == task

    def test_label_permutations_are_valid_permutations(self):
        schedule = build_schedule(jr.key(0), TINY, n_train=50)
        assert schedule.label_permutations.shape == (TINY.n_tasks, TINY.n_classes)
        expected = np.arange(TINY.n_classes)
        for task in range(TINY.n_tasks):
            row = np.sort(np.asarray(schedule.label_permutations[task]))
            np.testing.assert_array_equal(row, expected)

    def test_label_permutations_compose_cumulatively(self):
        """Row t must equal fresh_t[row_{t-1}] with row_{-1} = identity.

        This pins the upstream ``randperm(47)[targets]`` cumulative mutation
        (the first task itself is permuted), independently recomputing the
        per-task fresh permutations from the documented key derivation.
        """
        config = LabelEMNISTConfig(
            n_tasks=5, task_length=4, input_dim=6, hidden1=8, hidden2=4, n_classes=47
        )
        key = jr.key(7)
        schedule = build_schedule(key, config, n_train=50)
        key_perm, _ = jr.split(key)
        previous = np.arange(config.n_classes)
        for task in range(config.n_tasks):
            fresh = np.asarray(jr.permutation(jr.fold_in(key_perm, task), config.n_classes))
            expected = fresh[previous]
            np.testing.assert_array_equal(
                np.asarray(schedule.label_permutations[task]), expected
            )
            previous = expected

    def test_example_indices_sample_without_replacement(self):
        n_train = 11
        schedule = build_schedule(jr.key(3), TINY, n_train=n_train)
        assert schedule.example_indices.shape == (TINY.n_tasks, TINY.task_length)
        indices = np.asarray(schedule.example_indices)
        assert indices.min() >= 0 and indices.max() < n_train
        for task in range(TINY.n_tasks):
            assert len(set(indices[task].tolist())) == TINY.task_length

    def test_schedule_is_deterministic_per_key(self):
        first = build_schedule(jr.key(9), TINY, n_train=20)
        second = build_schedule(jr.key(9), TINY, n_train=20)
        np.testing.assert_array_equal(
            np.asarray(first.label_permutations), np.asarray(second.label_permutations)
        )
        np.testing.assert_array_equal(
            np.asarray(first.example_indices), np.asarray(second.example_indices)
        )
        third = build_schedule(jr.key(10), TINY, n_train=20)
        assert not np.array_equal(
            np.asarray(first.example_indices), np.asarray(third.example_indices)
        )

    def test_schedule_rejects_dataset_smaller_than_task(self):
        with pytest.raises(ValueError, match="without replacement"):
            build_schedule(jr.key(0), TINY, n_train=TINY.task_length - 1)


class TestTinySmokeRun:
    @pytest.fixture(scope="class")
    def debug_run(self):
        x, y = _tiny_data()
        return run_label_emnist(
            x, y, "upgd_w", seeds=[0, 1], config=TINY, return_per_step=True
        )

    def test_shapes_and_bounds(self, debug_run):
        assert debug_run.per_task_accuracy.shape == (2, TINY.n_tasks)
        assert debug_run.per_step_accuracy.shape == (2, TINY.n_tasks, TINY.task_length)
        assert np.all(debug_run.per_task_accuracy >= 0.0)
        assert np.all(debug_run.per_task_accuracy <= 1.0)
        assert np.all(np.isin(debug_run.per_step_accuracy, [0.0, 1.0]))
        assert np.all(debug_run.per_task_loss > 0.0)
        assert np.all(debug_run.per_task_plasticity >= 0.0)
        assert np.all(debug_run.per_task_plasticity <= 1.0)

    def test_per_task_accuracy_is_mean_of_per_step(self, debug_run):
        np.testing.assert_allclose(
            debug_run.per_task_accuracy,
            debug_run.per_step_accuracy.mean(axis=2),
            atol=1e-6,
        )

    def test_average_online_accuracy_is_mean_over_tasks(self, debug_run):
        np.testing.assert_allclose(
            debug_run.average_online_accuracy,
            debug_run.per_task_accuracy.mean(axis=1),
            atol=1e-12,
        )

    def test_first_step_accuracy_recomputed_externally(self, debug_run):
        """The first prediction must be the initial net on the permuted label."""
        from alberta_framework.benchmarks.upgd_ipmnist import mlp_logits

        x, y = _tiny_data()
        for seed_row in range(2):
            params = {
                name: jnp.asarray(value[seed_row])
                for name, value in debug_run.initial_params.items()
            }
            example = int(debug_run.example_indices[seed_row, 0, 0])
            permuted_label = int(
                debug_run.label_permutations[seed_row, 0, int(y[example])]
            )
            logits = mlp_logits(params, jnp.asarray(x[example]))
            expected = float(int(np.argmax(np.asarray(logits))) == permuted_label)
            assert debug_run.per_step_accuracy[seed_row, 0, 0] == expected

    def test_adamw_runs_and_is_deterministic(self):
        x, y = _tiny_data()
        first = run_label_emnist(x, y, "adamw", seeds=[5], config=TINY)
        second = run_label_emnist(x, y, "adamw", seeds=[5], config=TINY)
        np.testing.assert_array_equal(first.per_task_accuracy, second.per_task_accuracy)


class TestPlanShardMergeAccounting:
    def _plan(self):
        return build_plan_payload(TINY, [0, 1], DATASET_META)

    def _result(self, learner: str, seed: int):
        x, y = _tiny_data()
        return run_label_emnist(x, y, learner, seeds=[seed], config=TINY)

    def test_plan_roundtrip_and_validation(self, tmp_path):
        from alberta_framework.benchmarks.upgd_ipmnist_v3 import atomic_write_new_json

        payload = self._plan()
        path = tmp_path / "plan.json"
        atomic_write_new_json(path, payload)
        loaded = load_plan(path)
        assert loaded["plan"]["planned_shard_count"] == 4
        assert loaded["plan_sha256"] == payload["plan_sha256"]

    def test_plan_rejects_bad_seed_lists(self):
        with pytest.raises(ValueError, match="unique"):
            build_plan_payload(TINY, [1, 1], DATASET_META)
        with pytest.raises(ValueError, match="sorted"):
            build_plan_payload(TINY, [2, 1], DATASET_META)
        with pytest.raises(ValueError, match="at least one seed"):
            build_plan_payload(TINY, [], DATASET_META)

    def test_merge_full_coverage_and_accounting(self, tmp_path):
        from alberta_framework.benchmarks.upgd_ipmnist_v3 import atomic_write_new_json

        plan = self._plan()
        paths = []
        expected: dict[tuple[str, int], float] = {}
        for learner in ("upgd_w", "adamw"):
            for seed in (0, 1):
                result = self._result(learner, seed)
                expected[(learner, seed)] = float(result.average_online_accuracy[0])
                path = tmp_path / f"{learner}_seed{seed}.json"
                atomic_write_new_json(path, partial_payload(result, plan["plan_sha256"]))
                paths.append(path)
        results, coverage = merge_partials(plan, paths)
        assert coverage["complete"] and coverage["merged_shard_count"] == 4
        for learner in ("upgd_w", "adamw"):
            assert results[learner].seeds == (0, 1)
            for row, seed in enumerate(results[learner].seeds):
                assert (
                    abs(
                        float(results[learner].average_online_accuracy[row])
                        - expected[(learner, seed)]
                    )
                    < 1e-5
                )
        artifact = build_artifact(plan, results, coverage, partial_paths=paths)
        assert artifact["coverage"]["complete"]
        assert len(artifact["partial_manifest"]) == 4
        assert set(artifact["learners"]) == {"upgd_w", "adamw"}
        summary = artifact["learners"]["upgd_w"]
        assert summary["n_seeds"] == 2
        assert (
            abs(
                summary["average_online_accuracy_mean"]
                - np.mean([expected[("upgd_w", 0)], expected[("upgd_w", 1)]])
            )
            < 1e-5
        )

    def test_merge_rejects_duplicates_missing_and_foreign_shards(self, tmp_path):
        from alberta_framework.benchmarks.upgd_ipmnist_v3 import atomic_write_new_json

        plan = self._plan()
        result = self._result("adamw", 0)
        good = tmp_path / "adamw_seed0.json"
        atomic_write_new_json(good, partial_payload(result, plan["plan_sha256"]))
        duplicate = tmp_path / "adamw_seed0_copy.json"
        atomic_write_new_json(duplicate, partial_payload(result, plan["plan_sha256"]))
        with pytest.raises(ValueError, match="duplicate shard"):
            merge_partials(plan, [good, duplicate], allow_incomplete=True)
        with pytest.raises(ValueError, match="missing planned shards"):
            merge_partials(plan, [good])
        _, coverage = merge_partials(plan, [good], allow_incomplete=True)
        assert not coverage["complete"]
        assert ["upgd_w", 0] in coverage["missing_pairs"]
        assert len(coverage["missing_pairs"]) == 3
        foreign = tmp_path / "foreign.json"
        atomic_write_new_json(foreign, partial_payload(result, "f" * 64))
        with pytest.raises(ValueError, match="different plan"):
            merge_partials(plan, [good, foreign], allow_incomplete=True)

    def test_partial_rejects_multi_seed_and_unplanned_identity(self, tmp_path):
        from alberta_framework.benchmarks.upgd_ipmnist_v3 import atomic_write_new_json

        plan = self._plan()
        x, y = _tiny_data()
        multi = run_label_emnist(x, y, "adamw", seeds=[0, 1], config=TINY)
        with pytest.raises(ValueError, match="exactly one seed"):
            partial_payload(multi, plan["plan_sha256"])
        unplanned = self._result("adamw", 7)
        path = tmp_path / "unplanned.json"
        atomic_write_new_json(path, partial_payload(unplanned, plan["plan_sha256"]))
        with pytest.raises(ValueError, match="not planned"):
            merge_partials(plan, [path], allow_incomplete=True)

    def test_summary_and_comparison_flag_logic(self):
        upgd = self._result("upgd_w", 0)
        adam = self._result("adamw", 0)
        summaries = {"upgd_w": summarize_result(upgd), "adamw": summarize_result(adam)}
        comparison = build_comparison(summaries)
        assert set(comparison["learners"]) == {"upgd_w", "adamw"}
        assert "upgd_w_beats_adamw" in comparison
        assert "upgd_w_rises" in comparison
        for entry in comparison["learners"].values():
            assert entry["reproduction_gap_flagged"] is (
                abs(entry["gap"]) > comparison["gap_threshold"]
            )
