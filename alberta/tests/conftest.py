"""Pytest configuration and fixtures for Alberta Framework tests."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import jax.numpy as jnp
import jax.random as jr
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT,):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


# The standalone research repo vendors ``alberta_framework`` (the library) and
# its unit tests, but deliberately omits the ``benchmarks/`` and
# ``examples/The Alberta Plan/`` script trees (large experiment drivers, not part
# of the published package — see VENDORING.md). These modules ``load_script`` or
# ``import benchmarks.*`` at import time, so pytest cannot even collect them here;
# they fail with FileNotFoundError / ModuleNotFoundError before any assertion
# runs. Skip them at collection time when — and only when — those trees are
# absent, so a checkout that DOES vendor them (or upstream) still runs the full
# suite unchanged.
_VENDORED_OUT_TREES_PRESENT = (PROJECT_ROOT / "benchmarks").is_dir() and (
    PROJECT_ROOT / "examples"
).is_dir()

# Authoritative list of test modules whose module body reaches into the omitted
# benchmarks/examples trees. Regenerate by running the suite and collecting every
# file that fails with FileNotFoundError/ModuleNotFoundError for a
# benchmarks/ or examples/ path.
_BENCHMARK_EXAMPLE_DEPENDENT_TESTS = (
    "test_alberta_plan_remaining_todo_gate.py",
    "test_bsuite_helpers.py",
    "test_continuous_diffeml_performance.py",
    "test_d18_groupwise_decay.py",
    "test_d18_opmnist_bridge.py",
    "test_d20_multiprototype_opmnist.py",
    "test_diffeml_ablation_suite.py",
    "test_diffeml_hard_synthesis_suite.py",
    "test_diffeml_logic_benchmark.py",
    "test_diffeml_pure_eml_scale_suite.py",
    "test_production_steps.py",
    "test_rlsecd_external_acceptance_spec.py",
    "test_security_gym_oracle_experience_export.py",
    "test_step2_associative_evidence_gate.py",
    "test_step2_associative_memory_theory.py",
    "test_step2_associative_opmnist_confirmation.py",
    "test_step2_cifar_stream.py",
    "test_step2_completion_criteria.py",
    "test_step2_compositional_no_regret.py",
    "test_step2_conclusive_learner.py",
    "test_step2_context_disentanglement.py",
    "test_step2_context_inference.py",
    "test_step2_distribution_free_limits.py",
    "test_step2_expert_mixture.py",
    "test_step2_external_suite.py",
    "test_step2_formal_recursive_feature_discovery.py",
    "test_step2_new_direction_pilots.py",
    "test_step2_opmnist_protocol.py",
    "test_step2_published_stressors.py",
    "test_step2_resource_manager_stateful_external.py",
    "test_step2_scr_router_search.py",
    "test_step2_theory_falsification.py",
    "test_step2_universal_portfolio.py",
    "test_step2_universal_sieve_probe.py",
    "test_step2_upgd_ablation_configs.py",
    "test_step2_upgd_memory_opmnist.py",
    "test_step2_upgd_recursive_feature_discovery_theory.py",
    "test_step3_feature_discovery_eval.py",
    "test_throughput_smoke.py",
)

collect_ignore = (
    [] if _VENDORED_OUT_TREES_PRESENT else list(_BENCHMARK_EXAMPLE_DEPENDENT_TESTS)
)


def load_script(path: Path, name: str) -> ModuleType:
    """Load a Python script by filesystem path (works with paths containing spaces).

    Used by tests that exercise example scripts under
    ``examples/The Alberta Plan/`` whose directory name cannot be imported
    with the normal ``import`` statement.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture
def rng_key():
    """Provide a deterministic JAX random key."""
    return jr.key(42)


@pytest.fixture
def feature_dim():
    """Default feature dimension for tests."""
    return 10


@pytest.fixture
def sample_observation(feature_dim, rng_key):
    """Generate a sample observation vector."""
    return jr.normal(rng_key, (feature_dim,), dtype=jnp.float32)


@pytest.fixture
def sample_target():
    """Generate a sample target value."""
    return jnp.array([1.5], dtype=jnp.float32)
