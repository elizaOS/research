# Alberta Framework — agent guide

JAX continual-learning research track for elizaOS
([The Alberta Plan](https://arxiv.org/abs/2208.11173)). This tree is a
**development fork** of `lalalune/alberta` (fork point `2ac3533`), not a
lightly-patched vendor copy — see `VENDORING.md` for the divergence summary
and the canonical upstream URL. The robot track imports the continual-RL
subset in-process; keep `requires-python >= 3.12` and the `numpy >= 1.26`
floor intact.

## Layout

```
alberta_framework/
  core/        learners, optimizers (IDBD/Autostep/SwiftTD/UPGD), Horde +
               stacked_horde, SARSA/actor-critic, average-reward control,
               world models, dreaming, options/STOMP/OaK, PrototypeAgent,
               state_builder, learning_signals, experiential_memory,
               context_inference, hidden-partner substrates
  streams/     synthetic prediction streams + gauntlet, closed_loop, opponent,
               matrix_game, recurring_multiagent, hidden_partner_mapping,
               hidden_regime_signaling
  evaluation/  strict evidence artifacts, validators, the evidence registry,
               and the evaluation CLIs
  benchmarks/  forager family (incl. matched-current campaign machinery),
               official_foragax, published-protocol replication lanes
  utils/       multi-seed experiments, statistics, metrics, export
  steps/       Step 1–12 production kernels (smoke CLIs for Steps 1–2)
outputs/       pinned evidence artifacts — see immutability rules below
tests/         325 test files, ~6,900 collected (2026-08-01); marker lanes below
```

Key documents: `RESEARCH_STATUS.md` (evidence levels L0–L3, completion
gates), `CONTINUAL_LEARNING_EVIDENCE.md` (property-by-property evidence map),
`FORAGER_BENCHMARK.md` (Foragax protocols), `VENDORING.md` (fork status),
the execution runbooks (`CONTINUAL_IA_V2_RUNBOOK.md`,
`UPGD_IPMNIST_V3_RUNBOOK.md`, `OPMNIST_DEVELOPMENT_INGESTION.md`), and
`CHANGELOG.md`.

## Running things

Always use the project venv:

```bash
.venv/bin/python -m pytest tests/<file> -q -o addopts=""   # one file
.venv/bin/python -m pytest tests -q -o addopts=""          # full suite
.venv/bin/python -m ruff check .                           # lint (line length 100)
.venv/bin/python -m mypy                                   # strict, py312
.venv/bin/alberta-evidence-status                          # evidence registry
```

`addopts` defaults to `-v`; override with `-o addopts=""` for quiet runs.
Benchmark executions happen through scripts/CLIs (`alberta-forager-benchmark`,
the evaluation CLIs), never inside pytest — tests must stay CI-cheap unless
explicitly registered as a scientific lane.

## Marker lanes

- `unit` — fast isolated behavior/contract tests; never scientific evidence.
- `integration` — spans components, persistence, or process/CLI boundaries.
- `scientific` — frozen promoted-evidence protocols; may be expensive and
  require preregistered seeds.
- `development` — calibration/exploratory protocols; must never promote
  scientific claims.
- `slow` — wall-clock heavy modules (>~30s serial); excluded from the fast
  per-PR CI lane (`-m "not slow"`).

## Evidence-promotion rules (fail-closed)

- **Never auto-promote.** Passing tests, replays, or reruns do not upgrade a
  claim. Promotion requires a frozen preregistered protocol, untouched
  held-out seeds, a versioned artifact schema, and its strict validator
  accepting the artifact.
- **Frozen seeds stay frozen.** Calibration/development seeds and consumed
  evidence seeds can never be reused for promotion. Consumed-seed replays are
  explicitly nonpromoting.
- **Pinned `outputs/` artifacts are immutable.** Never overwrite or edit
  `outputs/ftl_decision/` (sha-pinned), `outputs/continual_ia/` (historical
  chain + source snapshot), `outputs/recurring_feature/`,
  `outputs/scale_robust_feature/evidence.v2.json`, or
  `outputs/continual_multiagent/`. New runs write to NEW paths and new schema
  versions.
- **Registered source hashes are load-bearing.** Editing a registered source
  file invalidates persisted evidence until the frozen protocol is rerun; the
  registry reports `invalid` (exit 2) — that is working-as-designed, not a
  bug to silence. Check which files a claim registers before touching them.
- Thresholds are calibrated empirically on development data with ≥2x margins,
  then frozen. Retuning a threshold after seeing held-out results is
  disallowed (a failed gate stays a valid rejection).
- Library changes are failing-test-first; state is frozen chex dataclasses;
  RNG uses explicit `jr.key(...)` seeds.

## Evidence registry (5 claims)

`alberta-evidence-status` exits `0` (all accepted), `1` (valid rejection or
missing), `2` (invalid). Each claim's CLI is also
`python -m alberta_framework.evaluation.<module>`.

**Live status (2026-08-01): the registry reports all five claims `invalid`
(overall `invalid`, exit 2)** because registered source files were edited
after the artifacts were pinned. That is the fail-closed design working as
intended, not a bug; the frozen outcomes recorded in the pinned artifacts are
listed below. Renewing a claim requires rerunning its frozen protocol to a
NEW artifact path and schema version with untouched preregistered seeds.

| Claim | Frozen artifact outcome | Artifact | CLI |
|---|---|---|---|
| `recurring_pair_features` | accepted (narrow L2) | `outputs/recurring_feature/evidence.v1.json` | `alberta-recurring-feature-evidence` |
| `scale_robust_pair_features` | accepted (narrow L2) | `outputs/scale_robust_feature/evidence.v2.json` | `alberta-scale-robust-evidence` |
| `ftl_world_model_decision_fidelity` | accepted (historical chain) | `outputs/ftl_decision/evidence.v1.json` | `alberta-ftl-evidence` |
| `recurring_multiagent_coadaptation` | accepted (narrow L2) | `outputs/continual_multiagent/evidence.json` | `alberta-multiagent-evidence` |
| `continual_intelligence_amplification` | valid rejection (frozen 10% gate) | `outputs/continual_ia/evidence.json` | `alberta-ia-evidence` |

No accepted claim is an Alberta Plan completion; keep README/status wording
narrow and honest.

## Conventions

- ruff line length 100; ESM/TS conventions do not apply here — this is a pure
  Python track.
- `CLAUDE.md` and `AGENTS.md` are identical: author `CLAUDE.md`, copy to
  `AGENTS.md`.
- No git commits unless explicitly asked.
