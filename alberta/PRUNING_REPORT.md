# Pruning report — what can be safely removed, what must stay

Date: 2026-08-02. Method: four parallel audits (core/streams import-consumer
map over 763 first-party files; outputs/ artifact classification against the
evidence rules; root-doc staleness/overlap audit; benchmarks/steps/utils/
evaluation/tests audit). Companion: NEGATIVE_RESULTS_LEDGER.md (rule: no
lane's code is deleted before its conclusion is ledgered).

## Executive summary

- The repo is **not** mostly cruft. Of 119 core/streams modules, 90 are
  load-bearing, 15 are evidence/hash-registered (effectively immutable), 6
  are robot-track dependencies. One module is truly dead; ~23K LOC sits in
  a REVIEW band.
- The big weight was never code: **1.7 GB of the 3.2 GB was regenerable
  run bulk and caches** (extracted source trees beside their sha-bound
  tars, dataset caches beside their ARFF sources, `__pycache__`). Most is
  already deleted (see "Executed").
- ~117K lines that look dead by import-graph are the **concurrent
  session's untracked in-flight work** (WP7.x evaluation modules, the
  forager matched-v3 family). Untouched by rule: mid-process work stays.
- Root docs: **nothing is deletable**. Four files are packaging- or
  provenance-pinned (FORAGER_BENCHMARK.md is hashed into run receipts);
  the rest are live contracts or frozen negative-result records. The real
  doc problem was staleness + orphaning, fixed alongside this report.

## Executed (this pass)

| Action | Reclaim | Recovery path |
|---|---|---|
| `outputs/forager/.../sources/*/source/` extracted trees (3) | 666 MB | `tar -xf source.tar` (verified byte-exact vs inventory) |
| `outputs/forager/frozen_src_20260731/` | 9 MB | duplicate of tracked source |
| `outputs/upgd_ipmnist/reconstructed_openml_cache/` | 15 MB | RUNBOOK regeneration command |
| `outputs/ipmnist_screening/logs/`, `relearning_logs/`, driver logs | ~1 MB | noise |
| non-venv `__pycache__` (9 dirs) | 59 MB | regenerated on import |
| `diffeml_*` cluster: 11 core modules + tests (~16K LOC) | code clarity | git history (tracked, clean, unchanged since initial import) |
| EMNIST `.npy` cache (deferred until live shards finish) | 338 MB | reparse retained ARFF; digests pinned in `plan.v*.json` |

Not executed, deliberately:
- The two `rtu_schema23_*_failed_source_drift_*` dirs are chmod-frozen
  (deliberate read-only negative records) — kept, 13.5 MB.
- Tier-2 option: `sources/upstream_rng_isolated/source.tar` (301 MB) is
  byte-derivable from the upstream tar + retained 78-line patch. Available
  if further reclaim is wanted; mild loss of direct digest verifiability.
- Tier-3 (NOT recommended): the upstream `source.tar` itself is the
  self-contained root of trust for the immutable v1 forager artifacts.

## Do-not-touch list (authoritative)

1. **Evidence-registered sources (21 files across 5 claims)** — listed in
   `evaluation/evidence_manifest.py`; editing or deleting any invalidates a
   pinned artifact. Core/streams subset: ftl_world_model,
   intelligence_amplification, average_reward, oak, options,
   interaction_features, closed_loop, gauntlet, recurring_multiagent.
2. **Dev-hash-registered (6 more)**: calibrated_extended_search_control,
   behavior_model, grounded_joint_world_model, signaling_bandit,
   learning_partner, hidden_learning_partner_planning_development/scan_plan.
3. **Robot-track imports (6)**: actor_critic, continual_backprop,
   initializers, normalizers, optimizers, sarsa (re-export). Corollary:
   `alberta_framework/__init__.py` must stay importable — every module
   deletion is a two-file change.
4. **Pinned outputs/**: ftl_decision, continual_ia (52 MB, zero regenerable
   bulk), recurring_feature, scale_robust_feature/evidence.v2.json,
   continual_multiagent, plus step2_canonical, hidden_partner_development,
   slowly_changing_regression, the sealed/QUARANTINED forager roots, and
   `outputs/evidence_manifest.json` (byte-level tamper detection).
5. **Packaging/provenance-pinned docs**: FORAGER_BENCHMARK.md (hashed into
   Forager run provenance by `forager_cli._source_tree_sha256`),
   HISTORICAL_FORAGER_RECONSTRUCTED.md + README/CHANGELOG/RESEARCH_STATUS/
   CONTINUAL_LEARNING_EVIDENCE (sdist byte-equality tests).
6. **Named active lanes whose only current consumers are tests**:
   stacked_horde, streams/opponent, streams/matrix_game, swift_td,
   context_inference, streams/pavlovian (cited mechanism evidence),
   off_policy_horde/off_policy_td (subjects of the cited Baird-star
   evidence lane — import-orphans but evidence-cited; kept).
7. **All untracked in-flight work** (concurrent session): 13 core modules
   (consolidated_memory chain, option_lifecycle_audit,
   stomp_option_lifecycle, cumulant_subtask_discovery, kondo_sparse_actor,
   prototype_stomp_calibrated_search…), 36 evaluation modules, the 23
   forager_matched_v3_* benchmarks + their tests, and
   FORAGER_MATCHED_V3_RUNBOOK.md (now git-added). Decision rule when they
   conclude: commit or delete deliberately — never prune as "dead".

## Wave 2 — delete after confirming with the concurrent session / next quiet window

| Target | LOC (src+test) | Why safe | Why not yet |
|---|---|---|---|
| `core/deep_feature_lifecycle.py` | 3,227 | test-only, undocumented, superseded by live prototype_feature_lifecycle | file is M in the shared tree |
| `core/latent_world_model.py` | 2,202 | re-export-only wrapper of ftl_world_model; prototype_agent doesn't use it | __init__.py surgery (lines ~578-595) |
| `core/learning_value_router.py` | 2,314 | re-export-only; delight is live without it | __init__.py surgery |
| small leaves: prototype_basis, geometric_features, fast_slow, prototype_features, sigreg, reward_model, diagnostics, cumulant_discovery, streams/partial_observation, streams/feature_discovery | ~4,500 | re-export/test-only; diagnostics + streams/feature_discovery have zero tests | ledger entries + __init__ surgery |
| `core/partner_world_model.py` + `evaluation/partner_world_diagnostic.py` | ~1,100 | superseded by grounded_joint_world_model + joint_partner_world | confirm diagnostic not cited |
| `benchmarks/world_model_retention_development.py` + `core/shallow_ridge_world_model.py` + test | ~4,000 | SRC=0, DOC=0 (the one benchmarks module with no doc record) | ledger the baseline comparison first |
| `benchmarks/delightful_policy_gradient_development.py` + `core/delightful_actor_critic.py` + tests | ~5,500 | Kondo-exclusion already ledgered (#10) | plan doc cites it from docstrings — update plan first |
| `evaluation/continual_control_reference_suite.py` + `continual_control_campaign.py` + tests | ~7,000 | committed, zero importers, zero doc refs | ledger entry first |
| `utils/export.py` | 513 | zero callers, zero tests | public-API break; deprecate in CHANGELOG first |

Estimated wave-2 total: ~30K LOC. With wave 1 (diffeml ~16K), the tree
sheds ~45K LOC of the ~500K first-party total without touching any
evidence, robot, or active-lane surface.

## Kept-with-fix (not pruning, but the audits' real findings)

- CLAUDE.md/AGENTS.md rewritten (this pass): current headline campaign,
  corrected layout/test counts, load-bearing-files warning, replication
  marker, refreshed key-documents list.
- Orphan links fixed: RESEARCH_REPORT_AGE_OF_EXPERIENCE.md and
  FORAGER_MATCHED_V3_RUNBOOK.md now linked from README;
  RTU_TAYLOR_CORRECTION.md now pointed to from
  recurrent_trace_actor_critic.py; v3 runbook git-added.
- `forager_causal_grid_divergence_probe.py` (root): load-bearing-adjacent,
  cited 4× — should move into the package or stay documented as a root
  tool; left in place, noted here.
- `steps/` 5–12 are public API surface (RESEARCH_STATUS matrix cites every
  row) — kept; CLAUDE.md now describes them accurately.
- The 26 `test_step2_*` concluded-probe files and 25 `*_horizon.py`
  variants are consolidation candidates (parametrization), not deletions.

## The one truly dead module

`core/prototype_stomp_calibrated_search.py` (1,902 LOC, zero consumers,
zero tests) — but it is **untracked**, i.e. the concurrent session's WIP.
Flagged to its author rather than deleted. If it is still orphaned at the
next quiet window, delete.
