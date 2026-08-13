"""Pure source-patch derivation for matched-v3 external two-seed transport.

The pinned upstream continuing and PPO entrypoints derive environment and
agent randomness from one historical run seed.  This module accepts only the
exact audited source bytes and applies deterministic single-occurrence byte
replacements that introduce two required CLI roots:
``--environment_seed`` and ``--agent_seed``.

Nothing here imports or executes upstream code.  Validation is limited to raw
identity, exact byte anchors, Python AST structure, and canonical provenance.
The resulting four-file patch set is an unexecuted derivative, not a complete
dependency closure, execution-ready adapter, authority grant, or scientific
artifact.
"""

from __future__ import annotations

import ast
import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, cast

SCHEMA_VERSION: Final = "alberta.forager_matched_v3_external_seed_transport.v1"
UPSTREAM_SOURCE_COMMIT: Final = "9710f60fa30da5badc451ad7ce3ff296d5070830"
UPSTREAM_SOURCE_TREE_GIT_SHA1: Final = "a5ad878ac4be0567c43dfd9177471c4b5a910bfa"
UPSTREAM_SOURCE_ARCHIVE_SHA256: Final = (
    "1f6976de38f34a697c947891de26ad3373b294195fe82094e9d1d5b8ddfd43b6"
)
UPSTREAM_SOURCE_ARCHIVE_SIZE_BYTES: Final = 314_961_920
UINT31_MAX: Final = (1 << 31) - 1
PUBLIC_KEY_TRACE_MAX_TRANSITIONS: Final = 1_000_000
_MAX_DESCRIPTOR_BYTES: Final = 2 * 1024 * 1024

SOURCE_PATHS: Final = (
    "src/continuing_main.py",
    "src/problems/BaseProblem.py",
    "src/problems/Foragax.py",
    "src/rtu_ppo.py",
)

UPSTREAM_SOURCE_SHA256_BY_PATH: Final[Mapping[str, str]] = MappingProxyType(
    {
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
)
UPSTREAM_SOURCE_SIZE_BYTES_BY_PATH: Final[Mapping[str, int]] = MappingProxyType(
    {
        "src/continuing_main.py": 32_190,
        "src/problems/BaseProblem.py": 1_548,
        "src/problems/Foragax.py": 1_069,
        "src/rtu_ppo.py": 89_937,
    }
)


class ExternalSeedTransportError(ValueError):
    """The source set, derivation, seed pair, or descriptor is invalid."""


@dataclass(frozen=True)
class SourceReplacement:
    """One fixed single-occurrence replacement in one pinned source file."""

    path: str
    replacement_id: str
    before: bytes
    after: bytes


@dataclass(frozen=True)
class SeedPair:
    """Two exact uint31 seed roots; equality is deliberately allowed."""

    environment_seed: int
    agent_seed: int


@dataclass(frozen=True)
class PublicEnvironmentKeyConsumption:
    """Declared source-level public environment-key schedule."""

    transitions: int
    agent_key_consumption: int
    environment_key_labels: tuple[str, ...]
    public_environment_key_count: int
    agent_consumption_changes_environment_schedule: bool
    runtime_trace_verified: bool


@dataclass(frozen=True)
class DerivedExternalSeedTransport:
    """Immutable derived patch set and frozen provenance descriptor."""

    sources: Mapping[str, bytes]
    source_sha256_by_path: Mapping[str, str]
    descriptor: Mapping[str, Any]
    descriptor_sha256: str


_UINT31_HELPER = b'''\
UINT31_MAX = (1 << 31) - 1


def uint31_seed(value: str) -> int:
    try:
        seed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be a base-10 uint31") from exc
    if seed < 0 or seed > UINT31_MAX:
        raise argparse.ArgumentTypeError("seed must be in [0, 2147483647]")
    return seed
'''

_REPLACEMENTS: Final = (
    SourceReplacement(
        "src/continuing_main.py",
        "continuing_define_uint31_cli_type",
        b"UNROLL = 1\n",
        b"UNROLL = 1\n\n" + _UINT31_HELPER,
    ),
    SourceReplacement(
        "src/continuing_main.py",
        "continuing_require_two_seed_flags",
        b'parser.add_argument("--max_steps", type=int, default=None)\n',
        (
            b'parser.add_argument("--max_steps", type=int, default=None)\n'
            b'parser.add_argument("--environment_seed", type=uint31_seed, required=True)\n'
            b'parser.add_argument("--agent_seed", type=uint31_seed, required=True)\n'
        ),
    ),
    SourceReplacement(
        "src/continuing_main.py",
        "continuing_require_exactly_one_index",
        (
            b"try:\n"
            b"    indices = parse_indices(args.idxs, exp.numPermutations())\n"
            b"except ValueError as e:\n"
            b"    parser.error(str(e))\n\n"
            b"Problem = getProblem(exp.problem)\n"
        ),
        (
            b"try:\n"
            b"    indices = parse_indices(args.idxs, exp.numPermutations())\n"
            b"except ValueError as e:\n"
            b"    parser.error(str(e))\n"
            b"if len(indices) != 1:\n"
            b'    parser.error("explicit two-seed transport requires exactly one index")\n'
            b"\n"
            b"Problem = getProblem(exp.problem)\n"
        ),
    ),
    SourceReplacement(
        "src/continuing_main.py",
        "continuing_remove_implicit_run_seed",
        (
            b"    run = exp.getRun(idx)\n\n"
            b"    # set random seeds accordingly\n"
            b"    hypers = exp.get_hypers(idx)\n"
        ),
        (
            b"    # Hyperparameters remain index-driven; RNG roots are explicit CLI inputs.\n"
            b"    hypers = exp.get_hypers(idx)\n"
        ),
    ),
    SourceReplacement(
        "src/continuing_main.py",
        "continuing_pass_explicit_roots_to_problem",
        b'    problem = chk.build("p", lambda: Problem(exp, idx, collector))\n',
        (
            b'    problem = chk.build(\n'
            b'        "p",\n'
            b'        lambda: Problem(\n'
            b'            exp,\n'
            b'            idx,\n'
            b'            collector,\n'
            b'            environment_seed=args.environment_seed,\n'
            b'            agent_seed=args.agent_seed,\n'
            b'        ),\n'
            b'    )\n'
        ),
    ),
    SourceReplacement(
        "src/continuing_main.py",
        "continuing_record_explicit_seed_metadata",
        b'    meta |= {"seed": exp.getRun(idx)}\n',
        (
            b'    meta |= {\n'
            b'        "environment_seed": args.environment_seed,\n'
            b'        "agent_seed": args.agent_seed,\n'
            b'    }\n'
        ),
    ),
    SourceReplacement(
        "src/problems/BaseProblem.py",
        "base_problem_accept_explicit_roots",
        (
            b"class BaseProblem:\n"
            b"    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):\n"
        ),
        (
            b"class BaseProblem:\n"
            b"    def __init__(\n"
            b"        self,\n"
            b"        exp: ExperimentModel,\n"
            b"        idx: int,\n"
            b"        collector: Collector,\n"
            b"        *,\n"
            b"        environment_seed: int,\n"
            b"        agent_seed: int,\n"
            b"    ):\n"
        ),
    ),
    SourceReplacement(
        "src/problems/BaseProblem.py",
        "base_problem_store_unambiguous_roots",
        b'        self.seed = exp.getRun(idx) + self.exp_params.get("seed_offset", 0)\n',
        (
            b"        self.environment_seed = environment_seed\n"
            b"        self.agent_seed = agent_seed\n"
        ),
    ),
    SourceReplacement(
        "src/problems/BaseProblem.py",
        "base_problem_route_agent_root_only",
        (
            b"            self.observations, self.actions, self.params, self.collector, self.seed\n"
        ),
        (
            b"            self.observations,\n"
            b"            self.actions,\n"
            b"            self.params,\n"
            b"            self.collector,\n"
            b"            self.agent_seed,\n"
        ),
    ),
    SourceReplacement(
        "src/problems/Foragax.py",
        "foragax_problem_accept_explicit_roots",
        (
            b"class Foragax(BaseProblem):\n"
            b"    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):\n"
            b"        super().__init__(exp, idx, collector)\n"
        ),
        (
            b"class Foragax(BaseProblem):\n"
            b"    def __init__(\n"
            b"        self,\n"
            b"        exp: ExperimentModel,\n"
            b"        idx: int,\n"
            b"        collector: Collector,\n"
            b"        *,\n"
            b"        environment_seed: int,\n"
            b"        agent_seed: int,\n"
            b"    ):\n"
            b"        super().__init__(\n"
            b"            exp,\n"
            b"            idx,\n"
            b"            collector,\n"
            b"            environment_seed=environment_seed,\n"
            b"            agent_seed=agent_seed,\n"
            b"        )\n"
        ),
    ),
    SourceReplacement(
        "src/problems/Foragax.py",
        "foragax_problem_route_environment_root_only",
        b"        env = Env(self.seed, **self.env_params)\n",
        b"        env = Env(self.environment_seed, **self.env_params)\n",
    ),
    SourceReplacement(
        "src/rtu_ppo.py",
        "ppo_define_uint31_cli_type",
        b"PERIOD = 182500\n",
        b"PERIOD = 182500\n\n" + _UINT31_HELPER,
    ),
    SourceReplacement(
        "src/rtu_ppo.py",
        "ppo_carry_environment_root",
        b"    env_state: Any = struct.field(pytree_node=True)\n",
        (
            b"    env_state: Any = struct.field(pytree_node=True)\n"
            b"    environment_rng: Any = struct.field(pytree_node=True)\n"
        ),
    ),
    SourceReplacement(
        "src/rtu_ppo.py",
        "ppo_step_environment_from_environment_root",
        (
            b"    # STEP ENV\n"
            b"    obs, env_state, reward, done, info = gymnax_state.env_step(\n"
            b"        _rng, gymnax_state.env_state, action.squeeze(), gymnax_state.env_params\n"
            b"    )\n"
        ),
        (
            b"    # STEP ENV\n"
            b"    environment_rng, environment_step_rng = jax.random.split(\n"
            b"        gymnax_state.environment_rng\n"
            b"    )\n"
            b"    obs, env_state, reward, done, info = gymnax_state.env_step(\n"
            b"        environment_step_rng,\n"
            b"        gymnax_state.env_state,\n"
            b"        action.squeeze(),\n"
            b"        gymnax_state.env_params,\n"
            b"    )\n"
        ),
    ),
    SourceReplacement(
        "src/rtu_ppo.py",
        "ppo_persist_environment_root_after_step",
        (
            b"        env_params=gymnax_state.env_params,\n"
            b"        env_state=env_state,\n"
            b"    )\n"
            b"    runner_state = (\n"
        ),
        (
            b"        env_params=gymnax_state.env_params,\n"
            b"        env_state=env_state,\n"
            b"        environment_rng=environment_rng,\n"
            b"    )\n"
            b"    runner_state = (\n"
        ),
    ),
    SourceReplacement(
        "src/rtu_ppo.py",
        "ppo_accept_explicit_roots",
        b"def experiment(rng, config: TrainConfig):\n",
        b"def experiment(environment_rng, agent_rng, config: TrainConfig):\n",
    ),
    SourceReplacement(
        "src/rtu_ppo.py",
        "ppo_reset_from_environment_root",
        (
            b"    rng, reset_rng = jax.random.split(rng)\n"
            b"    obs, env_state = env.reset(reset_rng, env.default_params)\n"
        ),
        (
            b"    environment_rng, reset_rng = jax.random.split(environment_rng)\n"
            b"    obs, env_state = env.reset(reset_rng, env.default_params)\n"
            b"    rng = agent_rng\n"
        ),
    ),
    SourceReplacement(
        "src/rtu_ppo.py",
        "ppo_initialize_environment_root_carry",
        (
            b"        env_params=env.default_params,\n"
            b"        env_state=env_state,\n"
            b"    )\n"
            b"    action_dim = 4\n"
        ),
        (
            b"        env_params=env.default_params,\n"
            b"        env_state=env_state,\n"
            b"        environment_rng=environment_rng,\n"
            b"    )\n"
            b"    action_dim = 4\n"
        ),
    ),
    SourceReplacement(
        "src/rtu_ppo.py",
        "ppo_preserve_environment_root_at_rollout_boundary",
        (
            b"            env_params=gymnax_state.env_params,\n"
            b"            env_state=gymnax_state.env_state,\n"
            b"        )\n"
            b"\n"
            b"        env_step_state = (\n"
        ),
        (
            b"            env_params=gymnax_state.env_params,\n"
            b"            env_state=gymnax_state.env_state,\n"
            b"            environment_rng=gymnax_state.environment_rng,\n"
            b"        )\n"
            b"\n"
            b"        env_step_state = (\n"
        ),
    ),
    SourceReplacement(
        "src/rtu_ppo.py",
        "ppo_require_two_seed_flags",
        (
            b'    parser.add_argument("--max_steps", type=int, default=None)\n'
            b"\n"
            b"    args = parser.parse_args()\n"
        ),
        (
            b'    parser.add_argument("--max_steps", type=int, default=None)\n'
            b'    parser.add_argument("--environment_seed", type=uint31_seed, required=True)\n'
            b'    parser.add_argument("--agent_seed", type=uint31_seed, required=True)\n'
            b"\n"
            b"    args = parser.parse_args()\n"
        ),
    ),
    SourceReplacement(
        "src/rtu_ppo.py",
        "ppo_require_exactly_one_index",
        (
            b"    try:\n"
            b"        indices = parse_indices(args.idxs, exp.numPermutations())\n"
            b"    except ValueError as e:\n"
            b"        parser.error(str(e))\n"
            b"    allocate_frames = len(indices) == 1\n"
        ),
        (
            b"    try:\n"
            b"        indices = parse_indices(args.idxs, exp.numPermutations())\n"
            b"    except ValueError as e:\n"
            b"        parser.error(str(e))\n"
            b"    if len(indices) != 1:\n"
            b'        parser.error("explicit two-seed transport requires exactly one index")\n'
            b"    allocate_frames = True\n"
        ),
    ),
    SourceReplacement(
        "src/rtu_ppo.py",
        "ppo_allocate_two_root_batches",
        (
            b"    collectors = []\n"
            b"    rngs = []\n"
            b"    chks = []\n"
        ),
        (
            b"    collectors = []\n"
            b"    environment_rngs = []\n"
            b"    agent_rngs = []\n"
            b"    chks = []\n"
        ),
    ),
    SourceReplacement(
        "src/rtu_ppo.py",
        "ppo_construct_explicit_roots_without_offset",
        (
            b'        seed = exp.getRun(idx) + hypers.get("seed_offset", 0)\n'
            b"        rng = jax.random.PRNGKey(seed)\n"
            b"\n"
            b'        freeze_steps = hypers.get("freeze_after_steps", '
            b'hypers.get("freeze_steps", -1))\n'
            b"        rngs.append(rng)\n"
        ),
        (
            b"        environment_rng = jax.random.key(\n"
            b'            args.environment_seed, impl="threefry2x32"\n'
            b"        )\n"
            b"        agent_rng = jax.random.key(args.agent_seed, impl=\"threefry2x32\")\n"
            b"\n"
            b'        freeze_steps = hypers.get("freeze_after_steps", '
            b'hypers.get("freeze_steps", -1))\n'
            b"        environment_rngs.append(environment_rng)\n"
            b"        agent_rngs.append(agent_rng)\n"
        ),
    ),
    SourceReplacement(
        "src/rtu_ppo.py",
        "ppo_vmap_both_explicit_roots",
        (
            b"    batch_experiment = jax.vmap(experiment, in_axes=(0, 0))\n"
            b"    rngs = jnp.stack(rngs)\n"
            b"    configs_stacked = tree_map(lambda *xs: jnp.stack(xs), *configs)\n"
            b"    results = batch_experiment(rngs, configs_stacked)\n"
        ),
        (
            b"    batch_experiment = jax.vmap(experiment, in_axes=(0, 0, 0))\n"
            b"    environment_rngs = jnp.stack(environment_rngs)\n"
            b"    agent_rngs = jnp.stack(agent_rngs)\n"
            b"    configs_stacked = tree_map(lambda *xs: jnp.stack(xs), *configs)\n"
            b"    results = batch_experiment(environment_rngs, agent_rngs, configs_stacked)\n"
        ),
    ),
    SourceReplacement(
        "src/rtu_ppo.py",
        "ppo_record_explicit_seed_metadata",
        b'        meta |= {"seed": exp.getRun(idx)}\n',
        (
            b"        meta |= {\n"
            b'            "environment_seed": args.environment_seed,\n'
            b'            "agent_seed": args.agent_seed,\n'
            b"        }\n"
        ),
    ),
)

# Filled from the exact deterministic transform above.  These are load-bearing
# independent checks, not values calculated from caller input.
EXPECTED_DERIVED_SOURCE_SHA256_BY_PATH: Final[Mapping[str, str]] = MappingProxyType(
    {
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
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _apply_path_replacements(path: str, source: bytes) -> bytes:
    """Apply one path's fixed replacements with exact occurrence guards."""
    if type(source) is not bytes:
        raise ExternalSeedTransportError(f"{path} source must be exact bytes")
    result = source
    replacements = [item for item in _REPLACEMENTS if item.path == path]
    if not replacements:
        raise ExternalSeedTransportError(f"no replacement contract exists for {path}")
    for replacement in replacements:
        count = result.count(replacement.before)
        if count != 1:
            raise ExternalSeedTransportError(
                f"replacement {replacement.replacement_id} matched {count}; expected exactly 1"
            )
        if replacement.after in result:
            raise ExternalSeedTransportError(
                f"replacement {replacement.replacement_id} output already exists"
            )
        result = result.replace(replacement.before, replacement.after, 1)
    return result


def _decode_and_parse(path: str, source: bytes) -> ast.Module:
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExternalSeedTransportError(f"derived {path} is not UTF-8") from exc
    try:
        return ast.parse(text, filename=path)
    except SyntaxError as exc:
        raise ExternalSeedTransportError(f"derived {path} is not valid Python") from exc


def _require_count(source: bytes, needle: bytes, count: int, context: str) -> None:
    observed = source.count(needle)
    if observed != count:
        raise ExternalSeedTransportError(
            f"{context} occurrence count is {observed}; expected {count}"
        )


def _require_semantic_rendered(
    tree: ast.Module,
    node_type: type[ast.AST],
    expected: str,
    context: str,
    *,
    count: int = 1,
) -> None:
    observed = sum(
        1
        for node in ast.walk(tree)
        if type(node) is node_type and ast.unparse(node) == expected
    )
    if observed != count:
        raise ExternalSeedTransportError(
            f"semantic {context} count is {observed}; expected {count}"
        )


def _class_method(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.FunctionDef:
    matches = [
        item
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == method_name
    ]
    if len(matches) != 1:
        raise ExternalSeedTransportError(
            f"semantic {class_name}.{method_name} count is {len(matches)}; expected 1"
        )
    return matches[0]


def _require_signature(
    function: ast.FunctionDef,
    positional: tuple[str, ...],
    kwonly: tuple[str, ...],
    context: str,
) -> None:
    arguments = function.args
    observed_positional = tuple(
        item.arg for item in (*arguments.posonlyargs, *arguments.args)
    )
    observed_kwonly = tuple(item.arg for item in arguments.kwonlyargs)
    no_defaults = not arguments.defaults and all(
        item is None for item in arguments.kw_defaults
    )
    if (
        observed_positional != positional
        or observed_kwonly != kwonly
        or arguments.vararg is not None
        or arguments.kwarg is not None
        or not no_defaults
    ):
        raise ExternalSeedTransportError(f"semantic {context} signature does not match")


_INDEX_GUARD_RENDERED: Final = (
    "if len(indices) != 1:\n"
    "    parser.error('explicit two-seed transport requires exactly one index')"
)


def _validate_derived_ast(sources: Mapping[str, bytes]) -> None:
    if set(sources) != set(SOURCE_PATHS) or any(
        type(sources.get(path)) is not bytes for path in SOURCE_PATHS
    ):
        raise ExternalSeedTransportError("derived AST input must have the exact byte set")
    trees = {path: _decode_and_parse(path, sources[path]) for path in SOURCE_PATHS}

    continuing = sources["src/continuing_main.py"]
    continuing_tree = trees["src/continuing_main.py"]
    _require_count(continuing, b'"--environment_seed"', 1, "continuing environment flag")
    _require_count(continuing, b'"--agent_seed"', 1, "continuing agent flag")
    if any(token in continuing for token in (b"seed_offset", b"exp.getRun")):
        raise ExternalSeedTransportError("continuing source retains implicit seed derivation")
    for token in (
        b"environment_seed=args.environment_seed",
        b"agent_seed=args.agent_seed",
    ):
        _require_count(continuing, token, 1, "continuing explicit root routing")
    for flag in ("environment_seed", "agent_seed"):
        _require_semantic_rendered(
            continuing_tree,
            ast.Call,
            f"parser.add_argument('--{flag}', type=uint31_seed, required=True)",
            f"continuing --{flag} CLI",
        )
    _require_semantic_rendered(
        continuing_tree,
        ast.If,
        _INDEX_GUARD_RENDERED,
        "continuing exactly-one-index guard",
    )
    _require_semantic_rendered(
        continuing_tree,
        ast.Call,
        (
            "Problem(exp, idx, collector, environment_seed=args.environment_seed, "
            "agent_seed=args.agent_seed)"
        ),
        "continuing Problem root routing",
    )

    base = sources["src/problems/BaseProblem.py"]
    base_tree = trees["src/problems/BaseProblem.py"]
    for token in (
        b"self.environment_seed = environment_seed",
        b"self.agent_seed = agent_seed",
        b"self.collector,\n            self.agent_seed,",
    ):
        _require_count(base, token, 1, "BaseProblem seed routing")
    if any(token in base for token in (b"self.seed", b"seed_offset", b"exp.getRun")):
        raise ExternalSeedTransportError("BaseProblem retains ambiguous seed state")
    _require_signature(
        _class_method(base_tree, "BaseProblem", "__init__"),
        ("self", "exp", "idx", "collector"),
        ("environment_seed", "agent_seed"),
        "BaseProblem.__init__",
    )
    for expected, context in (
        ("self.environment_seed = environment_seed", "BaseProblem environment root"),
        ("self.agent_seed = agent_seed", "BaseProblem agent root"),
    ):
        _require_semantic_rendered(base_tree, ast.Assign, expected, context)
    _require_semantic_rendered(
        base_tree,
        ast.Call,
        (
            "Agent(self.observations, self.actions, self.params, self.collector, "
            "self.agent_seed)"
        ),
        "BaseProblem Agent root routing",
    )

    foragax = sources["src/problems/Foragax.py"]
    foragax_tree = trees["src/problems/Foragax.py"]
    _require_count(
        foragax,
        b"Env(self.environment_seed, **self.env_params)",
        1,
        "Foragax environment root",
    )
    if b"self.seed" in foragax:
        raise ExternalSeedTransportError("Foragax problem retains ambiguous seed state")
    _require_signature(
        _class_method(foragax_tree, "Foragax", "__init__"),
        ("self", "exp", "idx", "collector"),
        ("environment_seed", "agent_seed"),
        "Foragax.__init__",
    )
    _require_semantic_rendered(
        foragax_tree,
        ast.Call,
        (
            "super().__init__(exp, idx, collector, environment_seed=environment_seed, "
            "agent_seed=agent_seed)"
        ),
        "Foragax BaseProblem root routing",
    )
    _require_semantic_rendered(
        foragax_tree,
        ast.Call,
        "Env(self.environment_seed, **self.env_params)",
        "Foragax Env root routing",
    )

    ppo = sources["src/rtu_ppo.py"]
    ppo_tree = trees["src/rtu_ppo.py"]
    _require_count(ppo, b'"--environment_seed"', 1, "PPO environment flag")
    _require_count(ppo, b'"--agent_seed"', 1, "PPO agent flag")
    for token in (
        b"def experiment(environment_rng, agent_rng, config: TrainConfig):",
        b"environment_rng, reset_rng = jax.random.split(environment_rng)",
        b"environment_rng, environment_step_rng = jax.random.split(\n"
        b"        gymnax_state.environment_rng",
        b"        environment_step_rng,\n        gymnax_state.env_state,",
        b"rng = agent_rng",
        b"network_params = network.init(_rng, init_hstate, init_x)",
        b"probe_rng = jax.random.fold_in(rng, 104729)",
        b"results = batch_experiment(environment_rngs, agent_rngs, configs_stacked)",
    ):
        _require_count(ppo, token, 1, "PPO root separation")
    if any(token in ppo for token in (b"seed_offset", b"exp.getRun")):
        raise ExternalSeedTransportError("PPO source retains implicit seed derivation")
    if b"gymnax_state.env_step(\n        _rng," in ppo:
        raise ExternalSeedTransportError("PPO environment still consumes an agent key")
    for flag in ("environment_seed", "agent_seed"):
        _require_semantic_rendered(
            ppo_tree,
            ast.Call,
            f"parser.add_argument('--{flag}', type=uint31_seed, required=True)",
            f"PPO --{flag} CLI",
        )
    _require_semantic_rendered(
        ppo_tree,
        ast.If,
        _INDEX_GUARD_RENDERED,
        "PPO exactly-one-index guard",
    )
    experiments = [
        node
        for node in ppo_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "experiment"
    ]
    if len(experiments) != 1:
        raise ExternalSeedTransportError(
            f"semantic PPO experiment count is {len(experiments)}; expected 1"
        )
    _require_signature(
        experiments[0],
        ("environment_rng", "agent_rng", "config"),
        (),
        "PPO experiment",
    )
    semantic_nodes: tuple[tuple[type[ast.AST], str, str], ...] = (
        (
            ast.Assign,
            "environment_rng = jax.random.key(args.environment_seed, "
            "impl='threefry2x32')",
            "PPO environment root construction",
        ),
        (
            ast.Assign,
            "agent_rng = jax.random.key(args.agent_seed, impl='threefry2x32')",
            "PPO agent root construction",
        ),
        (
            ast.Assign,
            "environment_rng, reset_rng = jax.random.split(environment_rng)",
            "PPO environment reset split",
        ),
        (
            ast.Call,
            "env.reset(reset_rng, env.default_params)",
            "PPO environment reset consumer",
        ),
        (
            ast.Assign,
            "environment_rng, environment_step_rng = "
            "jax.random.split(gymnax_state.environment_rng)",
            "PPO environment step split",
        ),
        (
            ast.Call,
            (
                "gymnax_state.env_step(environment_step_rng, gymnax_state.env_state, "
                "action.squeeze(), gymnax_state.env_params)"
            ),
            "PPO environment env_step consumer",
        ),
        (ast.Assign, "rng = agent_rng", "PPO agent root carry"),
        (
            ast.Call,
            "network.init(_rng, init_hstate, init_x)",
            "PPO network initialization consumer",
        ),
        (ast.Call, "pi.sample(seed=rng)", "PPO action consumer"),
        (
            ast.Call,
            "jax.random.permutation(_rng, config.rollout_steps)",
            "PPO permutation consumer",
        ),
        (
            ast.Assign,
            "rng, update_rng = jax.random.split(rng)",
            "PPO parameter-update root split",
        ),
        (
            ast.Assign,
            "rng, init_rng = jax.random.split(rng)",
            "PPO reset root split",
        ),
        (
            ast.Assign,
            "rng, subkey = jax.random.split(rng)",
            "PPO perturbation root split",
        ),
        (
            ast.Call,
            "jax.random.fold_in(rng, 104729)",
            "PPO diagnostic root consumer",
        ),
        (
            ast.Assign,
            (
                "probe_runner_state = (train_state, *env_step_state[1:9], probe_rng, "
                "env_step_state[10])"
            ),
            "PPO diagnostic copied environment carry",
        ),
        (
            ast.Call,
            "jax.vmap(experiment, in_axes=(0, 0, 0))",
            "PPO two-root vmap",
        ),
        (
            ast.Call,
            "batch_experiment(environment_rngs, agent_rngs, configs_stacked)",
            "PPO two-root batch invocation",
        ),
    )
    for node_type, expected, context in semantic_nodes:
        _require_semantic_rendered(ppo_tree, node_type, expected, context)


def _plain_json(value: Any, context: str = "descriptor") -> Any:
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ExternalSeedTransportError(f"{context} has a non-string key")
            result[key] = _plain_json(item, f"{context}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _plain_json(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ExternalSeedTransportError(f"{context} contains a non-JSON value")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            _plain_json(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, UnicodeError, ValueError) as exc:
        raise ExternalSeedTransportError("descriptor is not canonical JSON") from exc


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in cast(dict[str, Any], value).items()}
        )
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _descriptor() -> dict[str, Any]:
    source_files = [
        {
            "path": path,
            "upstream_size_bytes": UPSTREAM_SOURCE_SIZE_BYTES_BY_PATH[path],
            "upstream_sha256": UPSTREAM_SOURCE_SHA256_BY_PATH[path],
            "derived_sha256": EXPECTED_DERIVED_SOURCE_SHA256_BY_PATH[path],
        }
        for path in SOURCE_PATHS
    ]
    records = [
        {
            "path": item.path,
            "replacement_id": item.replacement_id,
            "expected_occurrences": 1,
            "before_sha256": _sha256(item.before),
            "after_sha256": _sha256(item.after),
        }
        for item in _REPLACEMENTS
    ]
    descriptor: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "unexecuted_source_patch_set_contract",
        "artifact_scope": "derived_patch_set_only",
        "classification": "nonpromoting_external_two_seed_transport",
        "upstream": {
            "commit": UPSTREAM_SOURCE_COMMIT,
            "tree_git_sha1": UPSTREAM_SOURCE_TREE_GIT_SHA1,
            "archive_sha256": UPSTREAM_SOURCE_ARCHIVE_SHA256,
            "archive_size_bytes": UPSTREAM_SOURCE_ARCHIVE_SIZE_BYTES,
        },
        "source_files": source_files,
        "replacement_records": records,
        "seed_contract": {
            "cli_flags": ["--environment_seed", "--agent_seed"],
            "type": "exact_uint31",
            "minimum": 0,
            "maximum": UINT31_MAX,
            "equal_seeds_allowed": True,
            "distinct_seeds_allowed": True,
            "equal_numeric_roots_can_correlate_chains": True,
            "statistical_independence_claimed": False,
            "legacy_seed_offset_consumed": False,
            "legacy_seed_offset_must_not_affect_roots": True,
        },
        "continuing_transport": {
            "problem_class": "problems.Foragax.Foragax",
            "environment_constructor_root": "environment_seed",
            "agent_constructor_root": "agent_seed",
            "implicit_run_seed_used": False,
            "seed_offset_used": False,
            "exactly_one_index_required": True,
            "separate_logical_root_carries": True,
            "jax_prng_implementation": "runtime_default_unqualified",
            "numpy_agent_rng": "numpy.random.default_rng",
            "prng_implementation_qualified": False,
            "consumer_implementation_sources_bound": False,
            "fresh_start_required_for_seed_binding": True,
            "checkpoint_resume_root_binding_implemented": False,
        },
        "ppo_transport": {
            "environment_root_consumers": ["environment_reset", "environment_step"],
            "agent_root_consumers": [
                "network_initialization",
                "action_sampling",
                "minibatch_permutation",
                "parameter_update_randomness",
                "reset_and_perturb_randomness",
                "diagnostic_policy_randomness",
            ],
            "prng_implementation": "threefry2x32",
            "exactly_one_index_required": True,
            "separate_logical_root_carries": True,
            "equal_numeric_roots_produce_identical_initial_key_values": True,
            "fresh_start_required_for_seed_binding": False,
            "adapter_relationship": "derived_source_not_exact_upstream_execution",
        },
        "public_environment_key_consumption": {
            "basis": "source_ast_contract_not_runtime_trace",
            "runner_scope": "ppo_derived_patch_set",
            "reset_keys_per_run": 1,
            "step_keys_per_public_transition": 1,
            "agent_consumption_changes_environment_schedule": False,
            "diagnostic_rollouts_use_copied_environment_state": True,
            "diagnostic_rollouts_advance_public_environment_chain": False,
            "runtime_trace_verified": False,
        },
        "claims": {
            "execution_ready": False,
            "execution_authorized": False,
            "scientific_promotion_allowed": False,
            "performance_claim_allowed": False,
            "universal_sota_claim_allowed": False,
            "authority_granted": False,
        },
        "limitations": [
            "The contract performs source and AST validation only; it executes no upstream code.",
            (
                "The four patched files are not a full dependency inventory or "
                "execution source closure."
            ),
            (
                "The declared upstream archive is provenance only and is not supplied "
                "to this derivation API."
            ),
            "The continuing transform is intentionally limited to the matched Foragax problem.",
            (
                "Continuing PRNG consumer implementations were observed but their "
                "source files are not bound by the patch-set input."
            ),
            (
                "Continuing checkpoint resume can restore historical RNG state; "
                "qualification requires a fresh start."
            ),
            (
                "Equal numeric roots create correlated chains and do not establish "
                "statistical independence."
            ),
            "Runtime import, capability, RNG replay, and environment-trace probes remain required.",
            "A valid derivation does not qualify or authorize either external runner.",
        ],
    }
    descriptor["payload_sha256"] = _sha256(_canonical_json(descriptor))
    return descriptor


_DESCRIPTOR: Final[Mapping[str, Any]] = cast(
    Mapping[str, Any], _freeze_json(_descriptor())
)
EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256: Final = (
    "66be593917a47c8eca4e1a3227407e060ebb52ac835e4207dc32fc81de7d13ad"
)
if not hmac.compare_digest(
    _sha256(_canonical_json(_DESCRIPTOR)),
    EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256,
):
    raise AssertionError("canonical matched-v3 external seed-transport descriptor drifted")


def _require_uint31(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= UINT31_MAX:
        raise ExternalSeedTransportError(f"{name} must be an exact uint31")
    return value


def validate_uint31_seed_pair(environment_seed: object, agent_seed: object) -> SeedPair:
    """Validate two exact uint31 roots; equal and distinct values are valid."""
    return SeedPair(
        _require_uint31(environment_seed, "environment_seed"),
        _require_uint31(agent_seed, "agent_seed"),
    )


def public_environment_key_consumption(
    transitions: object, *, agent_key_consumption: object
) -> PublicEnvironmentKeyConsumption:
    """Model the declared public schedule; this is not runtime-trace evidence."""
    if (
        type(transitions) is not int
        or not 0 <= transitions <= PUBLIC_KEY_TRACE_MAX_TRANSITIONS
    ):
        raise ExternalSeedTransportError("transitions must be an exact bounded integer")
    if type(agent_key_consumption) is not int or agent_key_consumption < 0:
        raise ExternalSeedTransportError(
            "agent_key_consumption must be an exact nonnegative integer"
        )
    count = transitions
    labels = ("reset",) + tuple(f"step/{index}" for index in range(count))
    return PublicEnvironmentKeyConsumption(
        transitions=count,
        agent_key_consumption=agent_key_consumption,
        environment_key_labels=labels,
        public_environment_key_count=1 + count,
        agent_consumption_changes_environment_schedule=False,
        runtime_trace_verified=False,
    )


def matched_v3_external_seed_transport_descriptor() -> dict[str, Any]:
    """Return a detached plain-JSON descriptor."""
    return cast(dict[str, Any], _plain_json(_DESCRIPTOR))


def canonical_matched_v3_external_seed_transport_descriptor_bytes() -> bytes:
    """Return exact canonical descriptor bytes."""
    return _canonical_json(_DESCRIPTOR)


def parse_matched_v3_external_seed_transport_descriptor(raw: bytes) -> dict[str, Any]:
    """Accept only the exact canonical descriptor and version."""
    if type(raw) is not bytes:
        raise ExternalSeedTransportError("descriptor must be exact bytes")
    if len(raw) > _MAX_DESCRIPTOR_BYTES:
        raise ExternalSeedTransportError("descriptor is too large")
    if not hmac.compare_digest(_sha256(raw), EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256):
        raise ExternalSeedTransportError("descriptor digest does not match")
    canonical = canonical_matched_v3_external_seed_transport_descriptor_bytes()
    if raw != canonical:
        raise ExternalSeedTransportError("descriptor is not exact canonical JSON")
    return matched_v3_external_seed_transport_descriptor()


def _validate_source_input(sources: object) -> dict[str, bytes]:
    if type(sources) is not dict:
        raise ExternalSeedTransportError("sources must be one plain dict")
    source_dict = sources
    if any(type(path) is not str for path in source_dict):
        raise ExternalSeedTransportError("source paths must be exact strings")
    if set(source_dict) != set(SOURCE_PATHS):
        raise ExternalSeedTransportError("sources must have the exact file set")
    result: dict[str, bytes] = {}
    for path in SOURCE_PATHS:
        value = source_dict[path]
        if type(value) is not bytes:
            raise ExternalSeedTransportError(f"{path} source must be exact bytes")
        result[path] = value
    return result


def derive_matched_v3_external_seed_transport(
    sources: dict[str, bytes],
) -> DerivedExternalSeedTransport:
    """Derive the exact four-file patch set without importing or executing it."""
    source_dict = _validate_source_input(sources)
    derived: dict[str, bytes] = {}
    for path in SOURCE_PATHS:
        if len(source_dict[path]) != UPSTREAM_SOURCE_SIZE_BYTES_BY_PATH[path]:
            raise ExternalSeedTransportError(
                f"{path} does not have the exact byte length"
            )
        raw_digest = _sha256(source_dict[path])
        if not hmac.compare_digest(raw_digest, UPSTREAM_SOURCE_SHA256_BY_PATH[path]):
            raise ExternalSeedTransportError(f"{path} raw SHA-256 does not match the pin")
        transformed = _apply_path_replacements(path, source_dict[path])
        derived_digest = _sha256(transformed)
        if not hmac.compare_digest(
            derived_digest, EXPECTED_DERIVED_SOURCE_SHA256_BY_PATH[path]
        ):
            raise ExternalSeedTransportError(
                f"{path} derived SHA-256 does not match the frozen transform"
            )
        derived[path] = transformed
    _validate_derived_ast(derived)
    digest_map = {
        path: EXPECTED_DERIVED_SOURCE_SHA256_BY_PATH[path] for path in SOURCE_PATHS
    }
    return DerivedExternalSeedTransport(
        sources=MappingProxyType(dict(derived)),
        source_sha256_by_path=MappingProxyType(digest_map),
        descriptor=_DESCRIPTOR,
        descriptor_sha256=EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256,
    )


def replay_matched_v3_external_seed_transport(
    sources: dict[str, bytes], descriptor_raw: bytes
) -> DerivedExternalSeedTransport:
    """Replay only under the exact canonical descriptor binding."""
    parse_matched_v3_external_seed_transport_descriptor(descriptor_raw)
    return derive_matched_v3_external_seed_transport(sources)
