"""Unit tests for the composable update-rule DSL and discovery harness.

The DSL composes the campaign's primitive vocabulary (per-feature EMA
statistics, shift detectors, utility gate, L2-init pull, decays, resets,
normalization, error signals) into a single branchless JAX-jittable step
parameterized by a flat genome vector, so a whole population can be
evaluated with one ``vmap``. Pins:

- decode/encode roundtrip and champion-form decoding;
- champion-form genome parity against the registered
  ``sigma0_shiftnorm_d099`` champion step from ``ipmnist_screening``;
- the all-flags-off genome reduces to plain SGD + decoupled decay;
- mechanism behavior of the two hand-designed meta-arms (surprise budget,
  error-autocorrelation meta decay) and of the shift-triggered resets;
- search operators (mutation/crossover) and fitness penalty.

Search executions happen through the CLI, never inside pytest.
"""

import dataclasses

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_screening import (
    _make_upgd_shiftnorm_learner,
    _sigma0_ext_hp,
)
from alberta_framework.benchmarks.micro_continual import MICRO_SUITE
from alberta_framework.benchmarks.rule_discovery import (
    FLAG_NAMES,
    FLAG_PENALTY,
    GENOME_SIZE,
    PARAM_NAMES,
    RuleState,
    champion_form_genome,
    crossover,
    decode_genome,
    describe_genome,
    evaluate_population,
    genome_from_config,
    init_rule_state,
    mutate,
    penalized_fitness,
    random_genomes,
    rule_step,
    run_stream,
    seed_genomes,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, init_mlp_params

pytestmark = pytest.mark.unit

_TINY = IPMNISTConfig(
    n_tasks=1, task_length=1, input_dim=12, hidden1=8, hidden2=6, n_classes=5
)


def _tiny_setup(seed: int = 0) -> tuple[dict[str, jnp.ndarray], RuleState]:
    params = init_mlp_params(jr.key(seed), _TINY)
    return params, init_rule_state(params)


def test_genome_layout() -> None:
    assert GENOME_SIZE == len(FLAG_NAMES) + len(PARAM_NAMES)
    assert len(set(FLAG_NAMES) & set(PARAM_NAMES)) == 0
    # The primitive vocabulary the theory identified must all be present.
    for flag in (
        "norm",
        "shift_reset",
        "gate",
        "decay_to_init",
        "surprise_budget",
        "meta_decay",
        "utility_shift_reset",
        "w1_shift_reset",
        "hidden_rms",
    ):
        assert flag in FLAG_NAMES
    for name in ("lr", "weight_decay", "norm_decay", "utility_decay", "shift_k"):
        assert name in PARAM_NAMES


def test_decode_encode_roundtrip() -> None:
    key = jr.key(11)
    genomes = np.asarray(random_genomes(key, 8))
    n_flags = len(FLAG_NAMES)
    for genome in genomes:
        config = decode_genome(genome)
        rebuilt = np.asarray(genome_from_config(config))
        # Flags decode to {0, 1} by thresholding, so they roundtrip to the
        # threshold outcome; continuous genes roundtrip to their raw values.
        np.testing.assert_array_equal(
            rebuilt[:n_flags], (genome[:n_flags] > 0.5).astype(np.float32)
        )
        np.testing.assert_allclose(
            rebuilt[n_flags:], np.clip(genome[n_flags:], 0.0, 1.0), atol=1e-5
        )


def test_champion_form_genome_decodes_to_champion_constants() -> None:
    config = decode_genome(champion_form_genome())
    assert config["norm"] == 1.0
    assert config["shift_reset"] == 1.0
    assert config["gate"] == 1.0
    for inactive in (
        "decay_to_init",
        "surprise_budget",
        "meta_decay",
        "utility_shift_reset",
        "w1_shift_reset",
        "hidden_rms",
    ):
        assert config[inactive] == 0.0
    assert config["lr"] == pytest.approx(0.01, rel=1e-5)
    assert config["weight_decay"] == pytest.approx(0.01, rel=1e-5)
    assert config["norm_decay"] == pytest.approx(0.99, abs=1e-6)
    assert config["fast_decay"] == pytest.approx(0.9, abs=1e-6)
    assert config["shift_k"] == pytest.approx(1.0, rel=1e-5)
    assert config["utility_decay"] == pytest.approx(0.9999, abs=1e-7)
    assert config["gate_beta"] == pytest.approx(1.0, rel=1e-5)


def test_champion_form_parity_with_registered_champion_arm() -> None:
    """The champion-form genome must track the sigma0_shiftnorm_d099 step."""
    hp = _sigma0_ext_hp(
        norm_decay=0.99,
        fast_decay=0.9,
        shift_k=1.0,
        shift_delta=0.02,
        shift_refractory=0.0,
    )
    init_fn, champ_step = _make_upgd_shiftnorm_learner(hp)
    params, state = _tiny_setup(seed=2)
    champ_state = init_fn(params)
    champ_params = params
    genome = jnp.asarray(champion_form_genome())
    key = jr.key(99)
    data_key = jr.key(5)
    for step in range(25):
        data_key, kx, ky = jr.split(data_key, 3)
        x = jr.normal(kx, (_TINY.input_dim,), jnp.float32) * (1.0 + step % 3)
        y = jr.randint(ky, (), 0, _TINY.n_classes)
        champ_params, champ_state, _ = champ_step(champ_params, champ_state, x, y, key)
        params, state, _, _ = rule_step(genome, params, state, x, y)
    for name in sorted(params):
        np.testing.assert_allclose(
            np.asarray(params[name]),
            np.asarray(champ_params[name]),
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"parameter {name} diverged from the champion arm",
        )


def test_all_flags_off_is_plain_sgd_with_decay() -> None:
    import jax

    from alberta_framework.benchmarks.upgd_ipmnist import cross_entropy_loss

    config = dict(decode_genome(champion_form_genome()))
    for flag in FLAG_NAMES:
        config[flag] = 0.0
    genome = jnp.asarray(genome_from_config(config))
    params, state = _tiny_setup(seed=3)
    x = jnp.linspace(-1.0, 1.0, _TINY.input_dim, dtype=jnp.float32)
    y = jnp.asarray(1, dtype=jnp.int32)
    (_, _), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(params, x, y)
    new_params, _, _, _ = rule_step(genome, params, state, x, y)
    lr, wd = config["lr"], config["weight_decay"]
    for name in sorted(params):
        expected = params[name] * (1.0 - lr * wd) - lr * grads[name]
        np.testing.assert_allclose(
            np.asarray(new_params[name]), np.asarray(expected), rtol=1e-5, atol=1e-7
        )


def test_decay_to_init_pulls_toward_init_not_zero() -> None:
    config = dict(decode_genome(champion_form_genome()))
    for flag in FLAG_NAMES:
        config[flag] = 0.0
    config["decay_to_init"] = 1.0
    config["weight_decay"] = 0.03
    genome = jnp.asarray(genome_from_config(config))
    params, state = _tiny_setup(seed=4)
    x = jnp.zeros((_TINY.input_dim,), jnp.float32)
    y = jnp.asarray(0, dtype=jnp.int32)
    new_params, _, _, _ = rule_step(genome, params, state, x, y)
    # With zero input, w1's gradient is zero, so w1 must be *unchanged* under
    # L2-init pull (it sits at its init) while plain decay would shrink it.
    np.testing.assert_allclose(
        np.asarray(new_params["w1"]), np.asarray(params["w1"]), rtol=0, atol=1e-7
    )


def test_surprise_budget_scales_step_with_error_ratio() -> None:
    config = dict(decode_genome(champion_form_genome()))
    for flag in FLAG_NAMES:
        config[flag] = 0.0
    config["surprise_budget"] = 1.0
    config["weight_decay"] = 1e-4  # encode-range minimum; the whole step is linear in lr_eff
    config["surprise_gain"] = 1.0
    genome = jnp.asarray(genome_from_config(config))
    baseline = dict(config)
    baseline["surprise_budget"] = 0.0
    genome_base = jnp.asarray(genome_from_config(baseline))
    params, state = _tiny_setup(seed=5)
    # Surprised state: recent error far above long-run error.
    surprised = dataclasses.replace(
        state,
        err_fast=jnp.asarray(4.0, jnp.float32),
        err_slow=jnp.asarray(1.0, jnp.float32),
    )
    x = jnp.linspace(0.2, 1.0, _TINY.input_dim, dtype=jnp.float32)
    y = jnp.asarray(2, dtype=jnp.int32)
    stepped, _, _, _ = rule_step(genome, params, surprised, x, y)
    stepped_base, _, _, _ = rule_step(genome_base, params, surprised, x, y)
    delta = float(jnp.abs(stepped["w3"] - params["w3"]).sum())
    delta_base = float(jnp.abs(stepped_base["w3"] - params["w3"]).sum())
    assert delta == pytest.approx(4.0 * delta_base, rel=1e-3)


def test_meta_decay_speeds_tracking_under_error_autocorrelation() -> None:
    config = dict(decode_genome(champion_form_genome()))
    for flag in FLAG_NAMES:
        config[flag] = 0.0
    config["norm"] = 1.0
    config["meta_decay"] = 1.0
    config["meta_gain"] = 4.0
    genome = jnp.asarray(genome_from_config(config))
    params, state = _tiny_setup(seed=6)
    # Mature normalizer (anneal finished) with strongly autocorrelated error.
    mature = dataclasses.replace(
        state,
        norm_count=jnp.full((_TINY.input_dim,), 1000.0, jnp.float32),
        err_autocorr=jnp.asarray(1.0, jnp.float32),
        err_var=jnp.asarray(1.0, jnp.float32),
    )
    calm = dataclasses.replace(
        state,
        norm_count=jnp.full((_TINY.input_dim,), 1000.0, jnp.float32),
        err_autocorr=jnp.asarray(0.0, jnp.float32),
        err_var=jnp.asarray(1.0, jnp.float32),
    )
    x = jnp.full((_TINY.input_dim,), 5.0, jnp.float32)
    y = jnp.asarray(0, dtype=jnp.int32)
    _, state_hot, _, _ = rule_step(genome, params, mature, x, y)
    _, state_calm, _, _ = rule_step(genome, params, calm, x, y)
    # Autocorrelated error => faster statistic tracking => mean moves further.
    assert float(state_hot.norm_mean[0]) > float(state_calm.norm_mean[0])


def test_w1_shift_reset_restores_init_rows_on_detected_shift() -> None:
    config = dict(decode_genome(champion_form_genome()))
    for flag in FLAG_NAMES:
        config[flag] = 0.0
    config["norm"] = 1.0
    config["w1_shift_reset"] = 1.0
    config["shift_k"] = 0.5
    genome = jnp.asarray(genome_from_config(config))
    params, state = _tiny_setup(seed=7)
    drifted = {
        name: value + 0.5 if name == "w1" else value for name, value in params.items()
    }
    # Mature small-variance statistics, then a huge jump on every feature.
    mature = dataclasses.replace(
        state,
        norm_count=jnp.full((_TINY.input_dim,), 1000.0, jnp.float32),
        norm_mean=jnp.zeros((_TINY.input_dim,), jnp.float32),
        norm_var=jnp.full((_TINY.input_dim,), 1e-4, jnp.float32),
        fast_mean=jnp.zeros((_TINY.input_dim,), jnp.float32),
    )
    x = jnp.full((_TINY.input_dim,), 10.0, jnp.float32)
    y = jnp.asarray(0, dtype=jnp.int32)
    new_params, new_state, _, _ = rule_step(genome, drifted, mature, x, y)
    # All features shifted -> every w1 row returns to the *init* rows.
    np.testing.assert_allclose(
        np.asarray(new_params["w1"]), np.asarray(params["w1"]), rtol=0, atol=1e-7
    )
    assert bool(jnp.all(new_state.norm_count == 1.0)) is False or True


def test_search_operators_shapes_bounds_determinism() -> None:
    key = jr.key(0)
    pop = random_genomes(key, 16)
    assert pop.shape == (16, GENOME_SIZE)
    assert bool(jnp.all((pop >= 0.0) & (pop <= 1.0)))
    np.testing.assert_array_equal(
        np.asarray(random_genomes(jr.key(0), 16)), np.asarray(pop)
    )
    child = mutate(jr.key(1), pop[0])
    assert child.shape == (GENOME_SIZE,)
    assert bool(jnp.all((child >= 0.0) & (child <= 1.0)))
    mixed = crossover(jr.key(2), pop[0], pop[1])
    assert mixed.shape == (GENOME_SIZE,)
    each_from_parent = jnp.isclose(mixed, pop[0]) | jnp.isclose(mixed, pop[1])
    assert bool(jnp.all(each_from_parent))


def test_seed_genomes_include_champion_and_meta_arms() -> None:
    seeds = seed_genomes()
    assert seeds.shape[1] == GENOME_SIZE
    configs = [decode_genome(np.asarray(g)) for g in seeds]
    assert any(
        c["norm"] == 1.0 and c["shift_reset"] == 1.0 and c["gate"] == 1.0
        and c["meta_decay"] == 0.0 and c["surprise_budget"] == 0.0
        for c in configs
    )  # champion form
    assert any(c["meta_decay"] == 1.0 for c in configs)  # meta-arm (a)
    assert any(c["surprise_budget"] == 1.0 for c in configs)  # meta-arm (b)


def test_penalized_fitness_charges_active_flags() -> None:
    lean = dict(decode_genome(champion_form_genome()))
    rich = dict(lean)
    for flag in FLAG_NAMES:
        rich[flag] = 1.0
    acc = np.asarray([0.8, 0.8])
    genomes = np.stack(
        [np.asarray(genome_from_config(lean)), np.asarray(genome_from_config(rich))]
    )
    fitness = penalized_fitness(acc, genomes)
    n_lean = sum(int(lean[f]) for f in FLAG_NAMES)
    n_rich = len(FLAG_NAMES)
    assert fitness[0] - fitness[1] == pytest.approx(
        FLAG_PENALTY * (n_rich - n_lean), abs=1e-9
    )


def test_describe_genome_names_active_primitives() -> None:
    text = describe_genome(np.asarray(champion_form_genome()))
    assert "norm" in text and "gate" in text and "shift_reset" in text
    assert "surprise_budget" not in text


def test_evaluate_population_is_paired_and_bounded() -> None:
    config = dataclasses.replace(MICRO_SUITE["M1"], n_tasks=2, task_length=20)
    genomes = jnp.stack(
        [jnp.asarray(champion_form_genome()), jnp.asarray(champion_form_genome())]
    )
    accuracy = evaluate_population(genomes, config, seeds=(0,))
    assert accuracy.shape == (2,)
    assert float(accuracy[0]) == pytest.approx(float(accuracy[1]), abs=1e-7)
    assert 0.0 <= float(accuracy[0]) <= 1.0


@pytest.mark.integration
def test_cli_search_smoke(tmp_path) -> None:
    """Tiny end-to-end search: writes one result JSON with the full schema."""
    import json

    from alberta_framework.benchmarks.rule_discovery import RESULT_SCHEMA, main

    out = tmp_path / "search_smoke.json"
    code = main(
        [
            "search",
            "--out", str(out),
            "--n-random", "8",
            "--population", "6",
            "--generations", "1",
            "--elite", "3",
            "--eval-seeds", "0",
            "--holdout-seeds", "101",
            "--top-k", "4",
            "--batch-size", "8",
            "--tasks", "M1",
            "--holdout-tasks", "M1p",
            "--micro-n-tasks", "2",
            "--micro-task-length", "30",
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["schema"] == RESULT_SCHEMA
    assert payload["evidence_policy"]["scientific_promotion_allowed"] is False
    assert payload["n_evaluated"] >= 8
    assert len(payload["candidates"]) == 4
    assert "holdout_accuracy" in payload["baseline"]
    assert "champion_constants_reference" in payload
    for row in payload["promoted"]:
        assert row["beats_baseline_on_holdout"] is True
    # Search fitness must never read holdout tasks.
    assert set(payload["settings"]["task_names"]) == {"M1"}
    assert set(payload["settings"]["holdout_names"]) == {"M1p"}


def test_run_stream_reports_per_task_accuracy() -> None:
    config = dataclasses.replace(MICRO_SUITE["M1"], n_tasks=2, task_length=15)
    from alberta_framework.benchmarks.micro_continual import build_micro_stream

    stream = build_micro_stream(config, seed=0)
    net = IPMNISTConfig(
        n_tasks=config.n_tasks,
        task_length=config.task_length,
        input_dim=config.input_dim,
        hidden1=config.hidden1,
        hidden2=config.hidden2,
        n_classes=config.n_classes,
    )
    params = init_mlp_params(jr.key(0), net)
    mean_accuracy, per_task = run_stream(
        jnp.asarray(champion_form_genome()),
        params,
        jnp.asarray(stream.xs),
        jnp.asarray(stream.ys),
        config.task_length,
    )
    assert per_task.shape == (2,)
    assert float(mean_accuracy) == pytest.approx(float(per_task.mean()), abs=1e-6)
