# Fork status: Alberta Framework

This directory began as a vendored copy of the **Alberta Framework** — a JAX
implementation of
[The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173) (Sutton,
Bowling, Pilarski 2022). It is now a **development fork**, not a
lightly-patched vendor drop: the continual-learning research campaign happens
in this tree, and the divergence from the imported snapshot is substantial and
intentional.

- **Fork point:** `lalalune/alberta` @
  `2ac35333efae45cf969ce02ec1f2703476fed6c2`
- **Canonical repository URL:** https://github.com/lalalune/alberta
  (this is the single upstream identity; the `j-klawson/alberta-framework`
  URLs that older `pyproject.toml`/`CITATION.cff` revisions pointed at are
  stale and are no longer referenced)
- **License:** Apache-2.0 (see `LICENSE`)

## Why it lives here

`eliza-robot` (`packages/research/robot`) uses the Alberta continual-RL
control subset to train robot policies that learn a sequence of tasks without
catastrophic forgetting, and benchmarks it against standard RL (PPO). The
framework is imported in-process from the robot's Python 3.12 environment,
which is why `requires-python` is `>=3.12` and the numpy floor is `>=1.26`
(brax/mujoco pin `numpy<2` there).

## Divergence from the fork point

Measured against `lalalune/alberta@2ac3533` (excluding `__pycache__`):

- **`alberta_framework/`**: 34 modified files plus 25 new top-level
  modules/subpackages — 59 changed entries in total. The two new subpackages
  are `alberta_framework/evaluation/` (27 modules: strict evidence artifacts,
  validators, and the six evaluation CLIs) and `alberta_framework/benchmarks/`
  (6 modules: the forager family and `official_foragax`). New core modules
  include `swift_td`, `stacked_horde`, `context_inference`, `state_builder`,
  `learning_signals`, `experiential_memory`, `canonical_upgd`,
  `option_value_duration`, `ftl_world_model`, `behavior_model`,
  `joint_partner_world`, `feature_bank_router`, and
  `integrated_hidden_partner`; new streams include `gauntlet`, `closed_loop`,
  `opponent`, `matrix_game`, `recurring_multiagent`, and
  `hidden_partner_mapping`.
- **`tests/`**: 83 new test files and 18 modified ones. 27 upstream test
  files are intentionally not carried (they exercise upstream-only trees).
- **Top level**: `RESEARCH_STATUS.md`, `CONTINUAL_LEARNING_EVIDENCE.md`,
  `FORAGER_BENCHMARK.md`, the `outputs/` evidence artifacts, and this file are
  fork-local. `CHANGELOG.md` continues upstream numbering (0.27.0 was cut
  here).

Because of this, "re-sync from upstream" is no longer a patch-reapplication
exercise; treat any future sync as a merge between diverged development lines.

Not carried from upstream: repository metadata and non-runtime trees such as
`.github/`, the root-level `benchmarks/` scripts tree, `docs/`, `examples/`,
and `scripts/`.

## The benchmarks-shim hazard (fixed in 0.27.0)

Upstream kept its benchmark drivers in a repository-root `benchmarks/` tree,
and `alberta_framework/__init__.py` ended with a compatibility shim that
registered that root package under the `alberta_framework.benchmarks` name.
Once this fork added a real `alberta_framework.benchmarks` subpackage, the
shim became a hazard: with any unrelated top-level `benchmarks/` directory
importable (for example an upstream checkout on `sys.path`), the shim could
bind the foreign package into `sys.modules` under the subpackage's name and
shadow the packaged integrations.

As of 0.27.0 the shim is removed: the real subpackage is imported eagerly at
the end of `alberta_framework/__init__.py` and always wins.
`tests/test_benchmarks_shim.py` pins this with a subprocess probe that puts a
dummy root `benchmarks` package on `sys.path` and asserts the packaged
subpackage still resolves.

## Continual-RL subset used by the robot package

`alberta_framework.core`: `sarsa`, `actor_critic`, `average_reward`,
`optimizers`, `learners`, `normalizers`, `types`, `upgd`, `continual_backprop`;
plus `alberta_framework.streams`. The full 12-step / `diffeml_*` / prototype
modules are present but not required by the robot integration.
