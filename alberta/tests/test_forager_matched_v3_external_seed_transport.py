"""Pure source-contract tests for matched-v3 external two-seed transport."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_external_seed_transport as transport,
)

_PINNED_ROOT = Path("/tmp/foragax-agents-official")
_EXPECTED_DESCRIPTOR_SHA256 = (
    "66be593917a47c8eca4e1a3227407e060ebb52ac835e4207dc32fc81de7d13ad"
)


def _official_sources() -> dict[str, bytes]:
    missing = [path for path in transport.SOURCE_PATHS if not (_PINNED_ROOT / path).is_file()]
    if missing:
        pytest.skip(f"pinned source checkout unavailable: {missing!r}")
    return {path: (_PINNED_ROOT / path).read_bytes() for path in transport.SOURCE_PATHS}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _transformed_official_sources() -> dict[str, bytes]:
    sources = _official_sources()
    return {
        path: transport._apply_path_replacements(path, sources[path])
        for path in transport.SOURCE_PATHS
    }


def _minimal_semantic_sources() -> dict[str, bytes]:
    return {
        "src/continuing_main.py": b'''\
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--environment_seed", type=uint31_seed, required=True)
parser.add_argument("--agent_seed", type=uint31_seed, required=True)
args = parser.parse_args()
indices = [0]
if len(indices) != 1:
    parser.error("explicit two-seed transport requires exactly one index")
problem = Problem(
    exp,
    idx,
    collector,
    environment_seed=args.environment_seed,
    agent_seed=args.agent_seed,
)
''',
        "src/problems/BaseProblem.py": b'''\
class BaseProblem:
    def __init__(
        self,
        exp,
        idx,
        collector,
        *,
        environment_seed,
        agent_seed,
    ):
        self.environment_seed = environment_seed
        self.agent_seed = agent_seed

    def getAgent(self):
        return Agent(
            self.observations,
            self.actions,
            self.params,
            self.collector,
            self.agent_seed,
        )
''',
        "src/problems/Foragax.py": b'''\
class Foragax:
    def __init__(
        self,
        exp,
        idx,
        collector,
        *,
        environment_seed,
        agent_seed,
    ):
        super().__init__(
            exp,
            idx,
            collector,
            environment_seed=environment_seed,
            agent_seed=agent_seed,
        )
        env = Env(self.environment_seed, **self.env_params)
''',
        "src/rtu_ppo.py": b'''\
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--environment_seed", type=uint31_seed, required=True)
parser.add_argument("--agent_seed", type=uint31_seed, required=True)
args = parser.parse_args()
indices = [0]
if len(indices) != 1:
    parser.error("explicit two-seed transport requires exactly one index")


def agent_step(rng):
    return pi.sample(seed=rng)


def create_minibaches(rng, config):
    rng, _rng = jax.random.split(rng)
    return jax.random.permutation(_rng, config.rollout_steps)


def env_step(gymnax_state, rng, action):
    rng, _rng = jax.random.split(rng)
    environment_rng, environment_step_rng = jax.random.split(
        gymnax_state.environment_rng
    )
    result = gymnax_state.env_step(
        environment_step_rng,
        gymnax_state.env_state,
        action.squeeze(),
        gymnax_state.env_params,
    )
    return result, environment_rng, rng, _rng


def experiment(environment_rng, agent_rng, config: TrainConfig):
    environment_rng, reset_rng = jax.random.split(environment_rng)
    obs, env_state = env.reset(reset_rng, env.default_params)
    rng = agent_rng
    rng, _rng = jax.random.split(rng)
    network_params = network.init(_rng, init_hstate, init_x)
    rng, update_rng = jax.random.split(rng)
    rng, init_rng = jax.random.split(rng)
    rng, subkey = jax.random.split(rng)
    probe_rng = jax.random.fold_in(rng, 104729)
    probe_runner_state = (
        train_state,
        *env_step_state[1:9],
        probe_rng,
        env_step_state[10],
    )
    return obs, env_state, network_params, update_rng, init_rng, subkey, probe_runner_state


environment_rng = jax.random.key(args.environment_seed, impl="threefry2x32")
agent_rng = jax.random.key(args.agent_seed, impl="threefry2x32")
batch_experiment = jax.vmap(experiment, in_axes=(0, 0, 0))
results = batch_experiment(environment_rngs, agent_rngs, configs_stacked)
''',
    }


def _synthetic_replacement_sources() -> dict[str, bytes]:
    return {
        path: b"\n".join(
            replacement.before
            for replacement in transport._REPLACEMENTS
            if replacement.path == path
        )
        for path in transport.SOURCE_PATHS
    }


@pytest.mark.unit
def test_exact_source_set_and_raw_pins() -> None:
    assert transport.SOURCE_PATHS == (
        "src/continuing_main.py",
        "src/problems/BaseProblem.py",
        "src/problems/Foragax.py",
        "src/rtu_ppo.py",
    )
    assert dict(transport.UPSTREAM_SOURCE_SHA256_BY_PATH) == {
        "src/continuing_main.py": (
            "681c2dae9569a0bbd72c8f47a3a63d51176071308f9762f3d81855da79c3aebf"
        ),
        "src/problems/BaseProblem.py": (
            "1985825dfa257570c605a4f3704f4dc648775398008507761d76bc46d7c835d0"
        ),
        "src/problems/Foragax.py": (
            "f901d20109a35791c6ed8a8b3ddad97707645eea49461470a4bfa63ae3b40fea"
        ),
        "src/rtu_ppo.py": (
            "e75a6762690832067a24a649559a55e0aa89abba005d600f090b1bf284b3fc24"
        ),
    }
    assert dict(transport.UPSTREAM_SOURCE_SIZE_BYTES_BY_PATH) == {
        "src/continuing_main.py": 32_190,
        "src/problems/BaseProblem.py": 1_548,
        "src/problems/Foragax.py": 1_069,
        "src/rtu_ppo.py": 89_937,
    }
    assert transport.UPSTREAM_SOURCE_COMMIT == (
        "9710f60fa30da5badc451ad7ce3ff296d5070830"
    )
    assert transport.UPSTREAM_SOURCE_TREE_GIT_SHA1 == (
        "a5ad878ac4be0567c43dfd9177471c4b5a910bfa"
    )
    assert transport.UPSTREAM_SOURCE_ARCHIVE_SHA256 == (
        "1f6976de38f34a697c947891de26ad3373b294195fe82094e9d1d5b8ddfd43b6"
    )
    assert dict(transport.EXPECTED_DERIVED_SOURCE_SHA256_BY_PATH) == {
        "src/continuing_main.py": (
            "ca9748cf92107b41c1d1e6cd17d4a1a3c517fa5921c55469c1e66a73ef8d2551"
        ),
        "src/problems/BaseProblem.py": (
            "a4ab77408c1bb38dd3f4e72d830765176c38bba4b73b69fe296765a0272d87dc"
        ),
        "src/problems/Foragax.py": (
            "ff6e875511fcc574bafde7f114382dccf5303dba96f4154d5abbc16744d8e7c9"
        ),
        "src/rtu_ppo.py": (
            "1859b4cde5695fcedd5cd21280caa0df029057e1b90e364f3bace225d127f3f1"
        ),
    }


@pytest.mark.unit
def test_official_source_derivation_is_deterministic_and_ast_only() -> None:
    sources = _official_sources()
    first = transport.derive_matched_v3_external_seed_transport(sources)
    second = transport.derive_matched_v3_external_seed_transport(dict(sources))
    reversed_sources = dict(reversed(tuple(sources.items())))
    third = transport.derive_matched_v3_external_seed_transport(reversed_sources)

    assert dict(first.sources) == dict(second.sources)
    assert dict(first.sources) == dict(third.sources)
    assert dict(first.source_sha256_by_path) == dict(
        transport.EXPECTED_DERIVED_SOURCE_SHA256_BY_PATH
    )
    assert first.descriptor_sha256 == transport.EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256
    assert isinstance(first.sources, MappingProxyType)
    for path, source in first.sources.items():
        assert hashlib.sha256(source).hexdigest() == first.source_sha256_by_path[path]
        ast.parse(source, filename=path)


@pytest.mark.unit
def test_continuing_path_routes_only_explicit_roots() -> None:
    derived = transport.derive_matched_v3_external_seed_transport(_official_sources())
    continuing = derived.sources["src/continuing_main.py"]
    base = derived.sources["src/problems/BaseProblem.py"]
    foragax = derived.sources["src/problems/Foragax.py"]

    assert continuing.count(b'"--environment_seed"') == 1
    assert continuing.count(b'"--agent_seed"') == 1
    assert b"type=uint31_seed, required=True" in continuing
    assert b"environment_seed=args.environment_seed" in continuing
    assert b"agent_seed=args.agent_seed" in continuing
    assert b"exp.getRun" not in continuing
    assert b"seed_offset" not in continuing
    assert b"if len(indices) != 1:" in continuing
    assert b"explicit two-seed transport requires exactly one index" in continuing

    assert b"self.environment_seed = environment_seed" in base
    assert b"self.agent_seed = agent_seed" in base
    assert b"self.seed" not in base
    assert b"seed_offset" not in base
    assert b"self.collector,\n            self.agent_seed," in base
    assert b"Env(self.environment_seed, **self.env_params)" in foragax
    assert b"self.agent_seed" not in foragax
    assert b"self.seed" not in foragax


@pytest.mark.unit
def test_ppo_path_separates_environment_and_agent_consumers() -> None:
    source = transport.derive_matched_v3_external_seed_transport(
        _official_sources()
    ).sources["src/rtu_ppo.py"]

    assert source.count(b'"--environment_seed"') == 1
    assert source.count(b'"--agent_seed"') == 1
    assert b"def experiment(environment_rng, agent_rng, config: TrainConfig):" in source
    assert b"environment_rng, reset_rng = jax.random.split(environment_rng)" in source
    assert b"rng = agent_rng" in source
    assert b"network_params = network.init(_rng, init_hstate, init_x)" in source
    assert b"environment_rng: Any = struct.field(pytree_node=True)" in source
    assert b"environment_rng, environment_step_rng = jax.random.split(" in source
    assert b"gymnax_state.environment_rng" in source
    assert b"        environment_step_rng,\n        gymnax_state.env_state," in source
    assert b"gymnax_state.env_step(\n        _rng," not in source
    assert b"probe_rng = jax.random.fold_in(rng, 104729)" in source
    assert b"seed_offset" not in source
    assert b"exp.getRun" not in source
    assert b"if len(indices) != 1:" in source
    assert b"explicit two-seed transport requires exactly one index" in source
    assert b"batch_experiment = jax.vmap(experiment, in_axes=(0, 0, 0))" in source
    assert b"results = batch_experiment(environment_rngs, agent_rngs, configs_stacked)" in source


@pytest.mark.unit
def test_semantic_ast_validation_rejects_decoy_text_and_misrouted_roots() -> None:
    derived = _minimal_semantic_sources()
    transport._validate_derived_ast(derived)

    continuing = dict(derived)
    continuing_path = "src/continuing_main.py"
    continuing[continuing_path] = continuing[continuing_path].replace(
        b"environment_seed=args.environment_seed",
        b"environment_seed=args.agent_seed",
        1,
    ) + b'\n"environment_seed=args.environment_seed"\n'
    with pytest.raises(transport.ExternalSeedTransportError, match="semantic.*Problem"):
        transport._validate_derived_ast(continuing)

    base = dict(derived)
    base_path = "src/problems/BaseProblem.py"
    expected_agent_route = b"self.collector,\n            self.agent_seed,"
    base[base_path] = base[base_path].replace(
        expected_agent_route,
        b"self.collector,\n            self.environment_seed,",
        1,
    ) + b'\n"""self.collector,\n            self.agent_seed,"""\n'
    with pytest.raises(transport.ExternalSeedTransportError, match="semantic.*Agent"):
        transport._validate_derived_ast(base)

    ppo = dict(derived)
    ppo_path = "src/rtu_ppo.py"
    expected_environment_route = (
        b"        environment_step_rng,\n        gymnax_state.env_state,"
    )
    ppo[ppo_path] = ppo[ppo_path].replace(
        expected_environment_route,
        b"        (_rng),\n        gymnax_state.env_state,",
        1,
    ) + b'\n"""        environment_step_rng,\n        gymnax_state.env_state,"""\n'
    with pytest.raises(transport.ExternalSeedTransportError, match="semantic.*env_step"):
        transport._validate_derived_ast(ppo)


@pytest.mark.unit
def test_transform_and_derived_pin_pipeline_has_self_contained_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _synthetic_replacement_sources()
    transformed = {
        path: transport._apply_path_replacements(path, source)
        for path, source in sources.items()
    }
    for replacement in transport._REPLACEMENTS:
        assert transformed[replacement.path].count(replacement.after) == 1

    raw_sizes = MappingProxyType({path: len(source) for path, source in sources.items()})
    raw_hashes = MappingProxyType(
        {path: hashlib.sha256(source).hexdigest() for path, source in sources.items()}
    )
    derived_hashes = MappingProxyType(
        {path: hashlib.sha256(source).hexdigest() for path, source in transformed.items()}
    )
    validated: list[dict[str, bytes]] = []
    monkeypatch.setattr(transport, "UPSTREAM_SOURCE_SIZE_BYTES_BY_PATH", raw_sizes)
    monkeypatch.setattr(transport, "UPSTREAM_SOURCE_SHA256_BY_PATH", raw_hashes)
    monkeypatch.setattr(
        transport, "EXPECTED_DERIVED_SOURCE_SHA256_BY_PATH", derived_hashes
    )
    monkeypatch.setattr(
        transport,
        "_validate_derived_ast",
        lambda value: validated.append(dict(value)),
    )

    result = transport.derive_matched_v3_external_seed_transport(dict(sources))
    assert dict(result.sources) == transformed
    assert validated == [transformed]

    same_size_mutation = dict(sources)
    path = transport.SOURCE_PATHS[0]
    same_size_mutation[path] = b"X" + same_size_mutation[path][1:]
    with pytest.raises(transport.ExternalSeedTransportError, match="raw SHA-256"):
        transport.derive_matched_v3_external_seed_transport(same_size_mutation)


@pytest.mark.unit
def test_source_input_must_be_plain_exact_bytes_mapping() -> None:
    class SourceDict(dict[str, bytes]):
        pass

    class SourcePath(str):
        pass

    sources = {path: b"" for path in transport.SOURCE_PATHS}
    cases: list[object] = [
        MappingProxyType(sources),
        SourceDict(sources),
        {SourcePath(path): source for path, source in sources.items()},
        {**sources, "src/extra.py": b"pass\n"},
        {key: value for key, value in sources.items() if key != transport.SOURCE_PATHS[0]},
        {**sources, transport.SOURCE_PATHS[0]: bytearray(sources[transport.SOURCE_PATHS[0]])},
    ]
    for value in cases:
        with pytest.raises(transport.ExternalSeedTransportError):
            transport.derive_matched_v3_external_seed_transport(value)  # type: ignore[arg-type]


@pytest.mark.unit
def test_mutated_source_and_replacement_occurrence_fail_closed() -> None:
    sources = _official_sources()
    changed = dict(sources)
    changed["src/continuing_main.py"] += b"\n"
    with pytest.raises(transport.ExternalSeedTransportError, match="exact byte length"):
        transport.derive_matched_v3_external_seed_transport(changed)

    same_size = dict(sources)
    same_size["src/continuing_main.py"] = (
        b"X" + same_size["src/continuing_main.py"][1:]
    )
    with pytest.raises(transport.ExternalSeedTransportError, match="raw SHA-256"):
        transport.derive_matched_v3_external_seed_transport(same_size)

    oversized = dict(sources)
    oversized["src/problems/BaseProblem.py"] += b" " * 1_000_000
    with pytest.raises(transport.ExternalSeedTransportError, match="exact byte length"):
        transport.derive_matched_v3_external_seed_transport(oversized)

    duplicate_anchor = sources["src/continuing_main.py"].replace(
        b"UNROLL = 1\n", b"UNROLL = 1\nUNROLL = 1\n", 1
    )
    with pytest.raises(transport.ExternalSeedTransportError, match="matched 2"):
        transport._apply_path_replacements("src/continuing_main.py", duplicate_anchor)


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [True, False, -1, 2**31, 1.0, "1", None],
)
def test_uint31_validation_rejects_aliases_and_out_of_range(value: object) -> None:
    with pytest.raises(transport.ExternalSeedTransportError, match="uint31"):
        transport.validate_uint31_seed_pair(value, 0)
    with pytest.raises(transport.ExternalSeedTransportError, match="uint31"):
        transport.validate_uint31_seed_pair(0, value)


@pytest.mark.unit
def test_uint31_boundaries_and_equal_or_distinct_roots_are_allowed() -> None:
    equal = transport.validate_uint31_seed_pair(0, 0)
    distinct = transport.validate_uint31_seed_pair(0, (2**31) - 1)
    assert (equal.environment_seed, equal.agent_seed) == (0, 0)
    assert (distinct.environment_seed, distinct.agent_seed) == (0, (2**31) - 1)
    with pytest.raises(FrozenInstanceError):
        equal.environment_seed = 1  # type: ignore[misc]


@pytest.mark.unit
def test_public_environment_key_consumption_is_agent_draw_invariant() -> None:
    none = transport.public_environment_key_consumption(7, agent_key_consumption=0)
    many = transport.public_environment_key_consumption(7, agent_key_consumption=1_000_000)

    assert none.environment_key_labels == many.environment_key_labels
    assert none.environment_key_labels == (
        "reset",
        "step/0",
        "step/1",
        "step/2",
        "step/3",
        "step/4",
        "step/5",
        "step/6",
    )
    assert none.public_environment_key_count == 8
    assert many.agent_key_consumption == 1_000_000
    assert none.agent_consumption_changes_environment_schedule is False
    assert none.runtime_trace_verified is False
    for transitions, draws in ((-1, 0), (1, -1), (True, 0), (1, False)):
        with pytest.raises(transport.ExternalSeedTransportError):
            transport.public_environment_key_consumption(
                transitions, agent_key_consumption=draws
            )


@pytest.mark.unit
def test_descriptor_is_canonical_detached_and_nonpromoting() -> None:
    descriptor = transport.matched_v3_external_seed_transport_descriptor()
    raw = transport.canonical_matched_v3_external_seed_transport_descriptor_bytes()

    assert hashlib.sha256(raw).hexdigest() == (
        transport.EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256
    )
    assert transport.EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256 == (
        _EXPECTED_DESCRIPTOR_SHA256
    )
    assert transport.parse_matched_v3_external_seed_transport_descriptor(raw) == descriptor
    assert descriptor["schema_version"] == (
        "alberta.forager_matched_v3_external_seed_transport.v1"
    )
    assert descriptor["artifact_scope"] == "derived_patch_set_only"
    assert descriptor["status"] == "unexecuted_source_patch_set_contract"
    assert descriptor["seed_contract"]["equal_seeds_allowed"] is True
    assert descriptor["seed_contract"]["distinct_seeds_allowed"] is True
    assert descriptor["seed_contract"]["equal_numeric_roots_can_correlate_chains"] is True
    assert descriptor["seed_contract"]["statistical_independence_claimed"] is False
    assert "prng_implementation" not in descriptor["seed_contract"]
    assert descriptor["continuing_transport"]["jax_prng_implementation"] == (
        "runtime_default_unqualified"
    )
    assert descriptor["continuing_transport"]["numpy_agent_rng"] == (
        "numpy.random.default_rng"
    )
    assert descriptor["continuing_transport"]["prng_implementation_qualified"] is False
    assert descriptor["continuing_transport"][
        "consumer_implementation_sources_bound"
    ] is False
    assert descriptor["ppo_transport"]["prng_implementation"] == "threefry2x32"
    assert descriptor["continuing_transport"]["exactly_one_index_required"] is True
    assert descriptor["ppo_transport"]["exactly_one_index_required"] is True
    assert descriptor["continuing_transport"]["fresh_start_required_for_seed_binding"] is True
    assert descriptor["ppo_transport"]["fresh_start_required_for_seed_binding"] is False
    assert descriptor["ppo_transport"][
        "equal_numeric_roots_produce_identical_initial_key_values"
    ] is True
    assert any(
        "full dependency inventory" in limitation
        for limitation in descriptor["limitations"]
    )
    assert descriptor["public_environment_key_consumption"][
        "agent_consumption_changes_environment_schedule"
    ] is False
    assert descriptor["public_environment_key_consumption"]["runner_scope"] == (
        "ppo_derived_patch_set"
    )
    assert descriptor["public_environment_key_consumption"][
        "diagnostic_rollouts_advance_public_environment_chain"
    ] is False
    assert descriptor["public_environment_key_consumption"]["runtime_trace_verified"] is False
    assert descriptor["claims"] == {
        "execution_ready": False,
        "execution_authorized": False,
        "scientific_promotion_allowed": False,
        "performance_claim_allowed": False,
        "universal_sota_claim_allowed": False,
        "authority_granted": False,
    }

    descriptor["claims"]["execution_ready"] = True
    assert transport.matched_v3_external_seed_transport_descriptor()["claims"][
        "execution_ready"
    ] is False

    payload = transport.matched_v3_external_seed_transport_descriptor()
    payload_sha256 = payload.pop("payload_sha256")
    assert hashlib.sha256(_canonical_bytes(payload)).hexdigest() == payload_sha256

    with pytest.raises(TypeError):
        transport._DESCRIPTOR["status"] = "tampered"  # type: ignore[index]


@pytest.mark.unit
def test_parse_and_replay_reject_descriptor_mutation_and_cross_version() -> None:
    sources = _official_sources()
    raw = transport.canonical_matched_v3_external_seed_transport_descriptor_bytes()
    replayed = transport.replay_matched_v3_external_seed_transport(sources, raw)
    assert replayed.descriptor_sha256 == transport.EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256

    for changed in (raw + b"\n", raw.replace(b".v1", b".v2", 1)):
        with pytest.raises(transport.ExternalSeedTransportError):
            transport.parse_matched_v3_external_seed_transport_descriptor(changed)
        with pytest.raises(transport.ExternalSeedTransportError):
            transport.replay_matched_v3_external_seed_transport(sources, changed)


@pytest.mark.unit
def test_descriptor_parser_rejects_nonbytes_and_oversize_inputs() -> None:
    raw = transport.canonical_matched_v3_external_seed_transport_descriptor_bytes()
    with pytest.raises(transport.ExternalSeedTransportError, match="exact bytes"):
        transport.parse_matched_v3_external_seed_transport_descriptor(raw.decode())  # type: ignore[arg-type]
    with pytest.raises(transport.ExternalSeedTransportError, match="too large"):
        transport.parse_matched_v3_external_seed_transport_descriptor(
            b"x" * (transport._MAX_DESCRIPTOR_BYTES + 1)
        )


@pytest.mark.unit
def test_derived_result_and_nested_descriptor_are_immutable() -> None:
    derived = transport.derive_matched_v3_external_seed_transport(_official_sources())
    with pytest.raises(TypeError):
        derived.sources[transport.SOURCE_PATHS[0]] = b"tampered"  # type: ignore[index]
    with pytest.raises(TypeError):
        derived.descriptor["claims"]["execution_ready"] = True
    with pytest.raises(FrozenInstanceError):
        derived.descriptor_sha256 = "0" * 64  # type: ignore[misc]

    detached = copy.deepcopy(transport.matched_v3_external_seed_transport_descriptor())
    detached["source_files"][0]["path"] = "changed"
    assert transport.matched_v3_external_seed_transport_descriptor()["source_files"][0][
        "path"
    ] == transport.SOURCE_PATHS[0]


@pytest.mark.unit
def test_every_patch_record_has_exact_single_occurrence_guards() -> None:
    descriptor = transport.matched_v3_external_seed_transport_descriptor()
    records = descriptor["replacement_records"]
    assert records
    assert all(record["expected_occurrences"] == 1 for record in records)
    assert len({record["replacement_id"] for record in records}) == len(records)
    assert all(len(record["before_sha256"]) == 64 for record in records)
    assert all(len(record["after_sha256"]) == 64 for record in records)
