# OPMNIST published-scale development ingestion

This repository has a strict ingestion surface for the detached three-seed,
800-task Online Permuted MNIST run. It does not monitor, resume, launch, or
otherwise interact with benchmark workers. Synthetic tests validate the
ingestion code; they are not benchmark evidence.

Status as of 2026-08-01: **nothing has been ingested.** No bundle exists under
`outputs/opmnist_published_scale_development/` (the directory is absent). The
surface is implemented in
`alberta_framework/evaluation/opmnist_development_ingest.py` and exercised
only by `tests/test_opmnist_development_ingest.py`; the upstream 3-seed x
48M-update run it is waiting for has not produced a merged result in this
tree.

The current execution is permanently classified as a
`development_published_scale_confirmation`. It had no externally issued
pre-run manifest binding the complete source/import closure, exact commands,
runtime, and dataset bytes before seeds 0, 1, and 2 were consumed. A completed
run can therefore confirm the reported protocol scale, and it can report the
six metric comparisons, but it cannot promote a scientific, SOTA, Step-2, or
Alberta Plan completion claim.

The ingester authenticates the detached cache object against the pinned bytes
available to this repository (SHA-256
`fe4410d8dbb50f6db6482b187557c5cb8bccfbcec74eeb6abc47c858f4ffab78`,
15,469,256 bytes). This proves which known MNIST cache object was copied after
the run. It still does not prove those bytes were used during execution.

## Upstream audit

The upstream artifacts provide useful but incomplete provenance:

- `alberta.step2.opmnist_full_run_plan.v1` records the intended commands and
  protocol, but has no content digest, creation time, source closure, runtime
  binding, dataset digest, or external pre-run issuer.
- Each result has no top-level schema. Its
  `alberta.step2.upgd_memory_opmnist.manifest.v1` is created at result
  finalization and hashes only the runner and published-stressor files. It
  records `argv`, Git state, and runtime versions, but does not prove which
  source or dataset bytes were active throughout execution.
- The merge manifest hashes the three result files, merge script, and runner,
  but the upstream loader accepts duplicate keys and non-finite JSON and the
  writer can overwrite an existing path.
- `alberta.step2.opmnist_solution_gate.audit.v1` stores no digest of the merged
  artifact and delegates to a schema-tolerant gate. That gate calls the
  result-finalization two-file manifest “publishable provenance,” which is not
  sufficient under this repository's promotion rules.

For those reasons, the local validator never imports or trusts the upstream
gate implementation. It independently reconstructs its arithmetic and records
the upstream gate bytes as an input, while forcing the local receipt to remain
nonpromoting.

## Required completed inputs

Ingestion is allowed only after all of these exist:

- three final seed result JSON files, in seed order 0, 1, 2;
- three final dynamic status sidecars, in the same order, each reporting
  exactly 48,000,000 completed updates;
- the upstream merged result and solution-gate JSON;
- the plan and runbook;
- the runner, published-stressor, merge, and solution-gate source files;
- the OpenML cache object at
  `openml/openml.org/data/v1/download/52667/mnist_784.arff.gz` beneath the cache
  root reported by every seed result, with the exact pinned digest and size
  above. Symlinks are forbidden in this path and in every other input path.

The merged result and gate should first be produced using the exact commands in
the upstream runbook. The local ingester then uses the following interface
(paths are illustrative):

```bash
.venv/bin/python -m alberta_framework.evaluation.opmnist_development_ingest ingest \
  --plan /path/to/upstream-alberta/outputs/step2_opmnist_solution_full/step2_opmnist_solution_800task_3seed_plan.json \
  --runbook /path/to/upstream-alberta/outputs/step2_opmnist_solution_full/RUNBOOK.md \
  --result /path/to/upstream-alberta/outputs/step2_opmnist_solution_full/seed_splits/step2_opmnist_solution_800task_3seed_seed0_results.json \
  --result /path/to/upstream-alberta/outputs/step2_opmnist_solution_full/seed_splits/step2_opmnist_solution_800task_3seed_seed1_results.json \
  --result /path/to/upstream-alberta/outputs/step2_opmnist_solution_full/seed_splits/step2_opmnist_solution_800task_3seed_seed2_results.json \
  --status /path/to/upstream-alberta/outputs/step2_opmnist_solution_full/seed_splits/step2_opmnist_solution_800task_3seed_seed0_status.json \
  --status /path/to/upstream-alberta/outputs/step2_opmnist_solution_full/seed_splits/step2_opmnist_solution_800task_3seed_seed1_status.json \
  --status /path/to/upstream-alberta/outputs/step2_opmnist_solution_full/seed_splits/step2_opmnist_solution_800task_3seed_seed2_status.json \
  --merged-result /path/to/upstream-alberta/outputs/step2_opmnist_solution_full/step2_opmnist_solution_800task_3seed_results.json \
  --solution-gate /path/to/upstream-alberta/outputs/step2_opmnist_solution_full/step2_opmnist_solution_800task_3seed_solution_gate.json \
  --runner-source '/path/to/upstream-alberta/examples/The Alberta Plan/Step2/step2_upgd_memory_opmnist.py' \
  --published-stressors-source '/path/to/upstream-alberta/examples/The Alberta Plan/Step2/step2_published_stressors.py' \
  --merge-source /path/to/upstream-alberta/benchmarks/step2_opmnist_merge_seed_results.py \
  --solution-gate-source /path/to/upstream-alberta/benchmarks/step2_opmnist_solution_gate.py \
  --dataset-file /path/to/upstream-alberta/outputs/step2_published_mnist_openml_cache/openml/openml.org/data/v1/download/52667/mnist_784.arff.gz \
  --output outputs/opmnist_published_scale_development/run-2026-08-01-v1
```

The output directory must not exist. The ingester reads inputs through anchored
file descriptors with no-symlink traversal, validates every input, fsyncs every
staged file and directory, and publishes with Linux
`renameat2(RENAME_NOREPLACE)`. It therefore cannot replace even an empty output
directory created during the publication race. A failed copy removes its
private staging tree and reports a cleanup failure rather than hiding it.

Published files use mode `0440` and directories use `0550`. These modes protect
against accidental in-place edits; they are not a cryptographic or
administrator-proof immutability mechanism. Persist the reported receipt hash
in Git or another external trust anchor when long-term identity matters.
Validate a completed bundle independently with:

```bash
.venv/bin/python -m alberta_framework.evaluation.opmnist_development_ingest validate \
  outputs/opmnist_published_scale_development/run-2026-08-01-v1
```

## Fail-closed checks

The v1 receipt and every input JSON reject duplicate keys, `NaN`, infinities,
extra fields, boolean aliases for numeric fields, and malformed nested values.
Malformed bundles return `valid=false` (CLI exit 2) rather than raising an
indexing exception. Validation requires, independently for each seed:

- a dynamic final status of exactly 48,000,000 updates and 800 complete blocks,
  with total and latest-chunk elapsed/rate arithmetic that reconstructs;
- one final record with exact task IDs `0..799`, exact held-out view IDs
  `0..799`, a true canonical 60,000/10,000 MNIST split, prediction before every
  update, no task ID, and all 800 held-out views;
- the frozen six-method order and all six finite primitive metrics;
- an exact match between result `argv` and the corresponding plan command;
- identical runner/published-stressor hashes across all result manifests and
  the copied post-hoc source bytes.

The validator rebuilds per-method means, sample standard errors, paired
candidate differences, best-MLP selection, per-metric wins, and the set of
candidates winning all six metric means. It then cross-checks the merged result
and gate byte-for-byte through their recorded hashes and reconstructed values.
The six per-seed primitive metrics remain reported inputs: no raw prediction
trace or accumulator exists from which to reproduce them.

Bundle validation opens the root without following symlinks, enumerates an
exact closed directory/file tree, rejects symlinks and non-regular nodes, and
reads each file once from the descriptor inspected for type and mode. It also
rejects directory or file changes observed during the snapshot, then rebuilds
the receipt solely from those retained bytes.

The receipt deliberately exposes separate booleans for `protocol_complete`,
`multi_seed_full_scale`, and `all_metric_mean_win`. Regardless of those values,
`solved_opmnist_step2`, `sota_claimed`, and
`alberta_plan_step2_solved` are always `false` for this development-only run.
