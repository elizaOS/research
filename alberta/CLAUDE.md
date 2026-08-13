# Alberta Framework — agent guide

JAX continual-learning research track for elizaOS
([The Alberta Plan](https://arxiv.org/abs/2208.11173)). This tree is a
**development fork** of `lalalune/alberta` (fork point `2ac3533`), not a
lightly-patched vendor copy — see `VENDORING.md` for the divergence summary
and the canonical upstream URL. The robot track imports the continual-RL
subset in-process; keep `requires-python >= 3.12` and the `numpy >= 1.26`
floor intact.

**Current headline lane:** the IPMNIST screening/confirmation campaign —
development-grade and permanently nonpromoting. Best stored development
means: `sigma0_ndecay099` 0.86245 (protocol-extended: EMA input
normalization decay 0.99 + utility-gated SGD, no noise) and `adamw_cbp_r3e4`
0.80126 (protocol-pure) vs the 0.7791 published-config UPGD-W reproduction.
Theory of record: `CONTINUAL_LEARNING_THEORY.md`; forward plan:
`RESEARCH_REPORT_AGE_OF_EXPERIENCE.md`; raw record + audit:
`outputs/ipmnist_screening/`. Second active lane: the Forager matched-v3
campaign (`FORAGER_MATCHED_V3_RUNBOOK.md`, unissued). Negative and bounded
conclusions live in `NEGATIVE_RESULTS_LEDGER.md` — check it before re-trying
an idea.

## Layout

```
alberta_framework/
  core/        ~100 modules: learners, optimizers (IDBD/Autostep/SwiftTD/
               UPGD), Horde + stacked_horde, SARSA/actor-critic, average-
               reward control, world models, dreaming, options/STOMP/OaK,
               PrototypeAgent, state_builder, learning_signals,
               experiential/consolidated memory, delight/Kondo,
               feature lifecycles, embodied_safety_envelope,
               hidden-partner substrates
  streams/     synthetic prediction streams + gauntlet, closed_loop,
               opponent, matrix_game, pavlovian, hidden_partner_mapping,
               hidden_regime_signaling, recurring_multiagent
  evaluation/  strict evidence artifacts, validators, the evidence registry,
               evaluation CLIs (~98 modules; many are development lanes)
  benchmarks/  IPMNIST lanes (upgd_ipmnist, upgd_ipmnist_v3,
               ipmnist_screening, upgd_label_emnist), forager family
               (official_foragax, matched-current + matched-v3 campaign
               machinery, foragax_open_screen), slowly_changing_regression
  utils/       multi-seed experiments, statistics, metrics, export
  steps/       public Step 1–12 kernels: smoke CLIs for Steps 1–2,
               pipeline.py consumes Steps 3–4, Steps 5–12 are
               library-surface only (cited by RESEARCH_STATUS's matrix)
outputs/       evidence + campaign artifacts — see immutability rules below
tests/         ~450 test files (run `pytest --collect-only -q` for the
               current count; it grows weekly)
```

Key documents:

- Status & evidence: `RESEARCH_STATUS.md` (levels L0–L3, completion gates) ·
  `CONTINUAL_LEARNING_EVIDENCE.md` (property-by-property map)
- Active campaign: `CONTINUAL_LEARNING_THEORY.md` ·
  `RESEARCH_REPORT_AGE_OF_EXPERIENCE.md` ·
  `outputs/ipmnist_screening/{RUNBOOK,FINAL_REPORT,AUDIT,CEILING_ANALYSIS,SOTA_LANDSCAPE_2026}.md`
- Roadmap: `CONTINUAL_AGENT_IMPLEMENTATION_PLAN.md` (WP0–WP9, live — cited
  from source docstrings) · `CONTINUAL_DYAD_BENCHMARK.md` (staged,
  nonexecuting HCCL causal-core/HCCL-v1 design)
- Frozen records: `CONTINUAL_AGENT_RESEARCH.md` (2026-07-30 audit snapshot) ·
  `FORAGER_COMPARATOR_AUDIT.md` · `FORAGER_ALBERTA_CANDIDATE_AUDIT.md` ·
  `RTU_TAYLOR_CORRECTION.md` · `NEGATIVE_RESULTS_LEDGER.md` ·
  `PRUNING_REPORT.md`
- Runbooks: `UPGD_IPMNIST_V3_RUNBOOK.md` · `FORAGER_MATCHED_V3_RUNBOOK.md` ·
  `CONTINUAL_IA_V2_RUNBOOK.md` · `OPMNIST_DEVELOPMENT_INGESTION.md` ·
  `FORAGAX_OPEN_DEVELOPMENT_SCREEN.md`
- Benchmarks/infra: `FORAGER_BENCHMARK.md` ·
  `HISTORICAL_FORAGER_RECONSTRUCTED.md` · `VENDORING.md` · `CHANGELOG.md`

`README.md` is the index; if you add a root doc, link it there.

## Running things

Always use the project venv:

```bash
.venv/bin/python -m pytest tests/<file> -q -o addopts=""   # one file
.venv/bin/python -m pytest tests -q -o addopts=""          # full suite
.venv/bin/python -m pytest --collect-only -q | tail -1     # count of record
.venv/bin/python -m ruff check .                           # lint (line length 100)
.venv/bin/python -m mypy                                   # strict, py312
.venv/bin/alberta-evidence-status                          # evidence registry
```

`addopts` defaults to `-v`; override with `-o addopts=""` for quiet runs.
There are 18 console scripts — see `[project.scripts]` in `pyproject.toml`;
the ones you'll reach for are `alberta-evidence-status`,
`alberta-forager-benchmark`, `alberta-foragax-open-screen`, and the
`alberta-forager-matched-*` family. Benchmark executions happen through
scripts/CLIs, never inside pytest — tests must stay CI-cheap unless
explicitly registered as a scientific lane.

## Marker lanes

- `unit` — fast isolated behavior/contract tests; never scientific evidence.
- `integration` — spans components, persistence, or process/CLI boundaries.
- `scientific` — frozen promoted-evidence protocols; may be expensive and
  require preregistered seeds.
- `development` — calibration/exploratory protocols; must never promote
  scientific claims.
- `replication` — historical Step 1/2 replays; skip without upstream
  artifacts.
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
- **Pinned `outputs/` artifacts are immutable.** Never overwrite, edit, or
  delete `outputs/ftl_decision/` (sha-pinned), `outputs/continual_ia/`
  (historical chain + source snapshot), `outputs/recurring_feature/`,
  `outputs/scale_robust_feature/evidence.v2.json`,
  `outputs/continual_multiagent/`, `outputs/step2_canonical/`,
  `outputs/evidence_manifest.json`, the sealed/`QUARANTINED.md` forager
  roots, or the chmod-frozen negative-result dirs. New runs write to NEW
  paths and new schema versions. `outputs/ipmnist_screening/` and
  `outputs/upgd_ipmnist/` hold the active campaign's development artifacts —
  append, don't rewrite.
- **Registered source hashes are load-bearing.** Editing a registered source
  file invalidates persisted evidence until the frozen protocol is rerun; the
  registry reports `invalid` (exit 2) — that is working-as-designed, not a
  bug to silence. The 5-claim registry in
  `evaluation/evidence_manifest.py` registers 21 files; two development
  lanes hash 6 more (see `PRUNING_REPORT.md` for the full do-not-touch
  list). Check which files a claim registers before touching them.
- Thresholds are calibrated empirically on development data with ≥2x margins,
  then frozen. Retuning a threshold after seeing held-out results is
  disallowed (a failed gate stays a valid rejection).
- Library changes are failing-test-first; state is frozen chex dataclasses;
  RNG uses explicit `jr.key(...)` seeds.

## Evidence registry (5 claims)

`alberta-evidence-status` exits `0` (all accepted), `1` (valid rejection or
missing), `2` (invalid). Each claim's CLI is also
`python -m alberta_framework.evaluation.<module>`.

**Live status (2026-08-02): all five claims `invalid` (exit 2)** because
registered source files were edited after the artifacts were pinned — the
fail-closed design working as intended (a dirty worktree alone forces
`invalid`). The frozen outcomes recorded in the pinned artifacts:

| Claim | Frozen artifact outcome | Artifact | CLI |
|---|---|---|---|
| `recurring_pair_features` | accepted (narrow L2) | `outputs/recurring_feature/evidence.v1.json` | `alberta-recurring-feature-evidence` |
| `scale_robust_pair_features` | accepted (narrow L2) | `outputs/scale_robust_feature/evidence.v2.json` | `alberta-scale-robust-evidence` |
| `ftl_world_model_decision_fidelity` | accepted (historical chain) | `outputs/ftl_decision/evidence.v1.json` | `alberta-ftl-evidence` |
| `recurring_multiagent_coadaptation` | accepted (narrow L2) | `outputs/continual_multiagent/evidence.json` | `alberta-multiagent-evidence` |
| `continual_intelligence_amplification` | valid rejection (frozen 10% gate) | `outputs/continual_ia/evidence.json` | `alberta-ia-evidence` |

No accepted claim is an Alberta Plan completion; keep README/status wording
narrow and honest.

## Files that are load-bearing outside the docs

- `FORAGER_BENCHMARK.md` is hashed into Forager run provenance
  (`forager_cli._source_tree_sha256`) — edits change benchmark receipts.
- README/CHANGELOG/RESEARCH_STATUS/CONTINUAL_LEARNING_EVIDENCE/
  FORAGER_BENCHMARK/HISTORICAL_FORAGER_RECONSTRUCTED ship in the sdist
  (`pyproject.toml`) with byte-equality tests.
- `CONTINUAL_AGENT_IMPLEMENTATION_PLAN.md` section numbers are cited from
  source docstrings (delight.py, learning_signals.py, …).
- The CHANGELOG version heading is asserted by `test_release_metadata.py`.
- The robot track imports `core/{actor_critic,continual_backprop,
  initializers,normalizers,optimizers,sarsa}` via `import
  alberta_framework` — `alberta_framework/__init__.py` must stay importable,
  so every module deletion is a two-file change.

## Conventions

- ruff line length 100; ESM/TS conventions do not apply here — this is a pure
  Python track.
- `CLAUDE.md` and `AGENTS.md` are identical: author `CLAUDE.md`, copy to
  `AGENTS.md`.
- No git commits unless explicitly asked.
