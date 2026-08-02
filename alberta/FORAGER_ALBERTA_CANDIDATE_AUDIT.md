# Alberta matched-Forager candidate audit

Date: 2026-07-31

## Decision

**Internal Alberta implementation review: GO. Campaign authority: NOT CLEARED.** The
audit found no correctness, task-access, RNG-coupling, archive, or scoring blocker in the
14 Alberta candidates under the frozen four-field resource ledger. It did not close total
persistent memory or compute. The bound campaign remains content-only, unendorsed, and
nonpromoting; external trust resolution and executor-receipt verification remain required.
This is not performance evidence or an Alberta Plan completion claim. It does not
supersede the scope and statistical limitations in `FORAGER_COMPARATOR_AUDIT.md`.

Before spending the full campaign budget, run the new public-seed q-grid diagnostic in
addition to resolving the campaign's independent authority blockers. The diagnostic is
engineering-only and permanently nonpromoting. A valid rejection may motivate abandoning
v1, but it cannot retroactively amend the frozen protocol or turn seed 0 into an evidence
gate.

Execution status (2026-07-31): unchanged since this audit. The open-tuning campaign root
`outputs/forager/matched_current_open_tuning_2c3b214c_v1` still has empty `runs/` and
`completions/` directories — zero of its 210 tuning cells have executed — and the q-grid
divergence probe has not been run (no probe output artifact exists in the repository). The
hardened harness now reserves receipt schema
`alberta.forager_causal_grid_divergence_probe.v2` and default output
`outputs/forager/development/causal_q_grid_divergence_seed0_v2`; those version changes record
provenance hardening, not an execution or result.

## Scope and boundaries

The audit traced the 14 Alberta configurations through frozen configuration parsing,
agent initialization and updates, environment/agent RNG, reward-archive production,
OCI execution, archive validation, scoring, and candidate selection.

No Forager benchmark or live diagnostic was run. No reward, regret, return, or score
array was opened or inspected. Frozen outputs and registered candidate sources were not
edited. The diagnostic harness is deliberately a repository-root tool, outside
`alberta_framework/`, and executes the SHA-pinned frozen source snapshot.

## Frozen identity

- Candidate-universe SHA-256:
  `2c3b214cf29e013e3f8d88b2558bd94f75e92330bf0ddcc6afd7514279a1ee77`
- Open-protocol SHA-256:
  `b17da8af19cac570c426c74ff6bbc0e4ee0a4b95a4486c3ad5da19ceb3f8176e`
- Qualification-manifest SHA-256:
  `90182e6d9d79c4543648881f67d969d567c42163b93b2440377e5d36b2fb4d9a`
- Alberta source-archive SHA-256:
  `8f66a8cb2357e4d003adf2ac8084c75c7c46ac07cbbb8dddd6cce6e39f88bd79`
- Normalized source-inventory SHA-256:
  `e1cd51e16db0533b8a55c99cb343705be14b11084b7c8a02fb8ced66558cee6f`
- Detailed inventory JSON SHA-256:
  `3fd69fdc2f5ab373dfe8a99c494bcd41e79cbdbd8d7d5a6f5b90ee918eb6eeea`
- Snapshot-descriptor SHA-256:
  `8a390e0ed1c88e373b0e0c9a682e2e9dec79370dc02e58a3a0ff4f8233827fa7`
- Qualified OCI image SHA-256:
  `5ecaabefce6439a8731c19e7a55fedb666788242baf035e6ffca86eb31299768`
- Runtime-profile SHA-256:
  `7170418e8082babbf17ebfbbb639ee75fcd8b5ae3931d35b3fb9199ea2bfd9b3`
- Task-identity SHA-256:
  `3a353233a7eb48915220a0387d41ecafd1028b0316b04e32c09a30c70bbcb159`
- Environment RNG-schedule SHA-256:
  `51d811e6fccd2b015b1703f22775f880089bbca3fc8938421ad3e18526882cb0`

These are the last pinned qualification bindings, not newly renewed q-probe-v2 identities.
After a fresh matched-current qualification is published, its canonical artifact root and all
qualification-derived hashes and root bindings must be reviewed and repinned together before
the hardened diagnostic is run. The v2 harness does not infer authority from a manifest
sidecar or silently adopt a new qualification.

Each of the 14 Alberta configurations is byte-identical to its derived configuration,
has no allowed transform, and round-trips through its frozen parser.

| Candidate | Frozen configuration SHA-256 |
|---|---|
| `causal_e025_q050` | `1290335563481b7ac2fd3eda91ef9c63216684fd096f3ab5b16591de0870c736` |
| `causal_e025_q075` | `69a5df44db99866a0ee3967677fad66ea94c60b1bfa8317936e2c142fac34ed1` |
| `causal_e025_q090` | `e21692571fc751bdf2c4fa0e89ad43b12dbd51c72a0821d5839fc82f1031f8f4` |
| `causal_e050_q050` | `916bd37e04c39dc16c19153032fc1c3baf12a941efb3df95860ee9f03c1ef331` |
| `causal_e050_q075` | `afaa3ea47cd410a43541c85976fa6f718c5f70504494f70496385ec37ea84a63` |
| `causal_e050_q090` | `ab555510e08a98e733d01a9b145d19073bb17ba31681a459a55a978d5a4faf33` |
| `causal_e100_q050` | `00390162a1950e976a7b3e216b8c6d94a76427c38c8e30bbdc25fa583bf018a8` |
| `causal_e100_q075` | `8d7a8afdb204c1837834ef633e2524bf569180c763a34a96c883c6e2cd33fb48` |
| `causal_e100_q090` | `899658dff1eeaadf59de8dc437d1324429306b8a427a4ed67ccf54437931955c` |
| `alberta_horde_default` | `7e7e681ca3a06e6f5c9bcdf0c4de42a4775439967ac41504c3b9ebd971d0db7a` |
| `alberta_horde_eps05` | `ab402dd011e2d97df423ffa2f0203ea9fe3c01dcfc89db66d2f2fdf404b7204f` |
| `alberta_horde_recurrent64` | `870e805b046f1751cac48368b07827e3c27059d849f2a84b1c2e499e75e0f6ef` |
| `alberta_horde_step3e3` | `feb2cd34628b3d87873163e1c78d8ea0b5aba4e4652dcba67138bd3f6eba6bc5` |
| `alberta_rtu_h08_taylor` | `07571eeec0e132027c819cc3a0c8d781a0df71ecbd840947d3641e2ea3831792` |

## Prioritized findings

| Severity | Finding | Disposition |
|---|---|---|
| Authority blocker | Campaign execution/promotion authority is not cleared. | External trust resolution and executor-receipt verification remain required; the q probe cannot supply them. |
| Medium verification gap | No real-task aggregate previously demonstrated behavioral divergence among the q=.50/.75/.90 arms. | Addressed by a new nonpromoting, seed-0, fixed-10k diagnostic; it has not been run. |
| Low | The matched worker accepts the uint32 seed domain, while the Horde/RTU runner validator caps seeds at signed int32. | Inactive for the frozen campaign seeds, which are far below the signed-int32 limit. |
| Low transparency | Recurrent Horde has a substantial fixed GRU substrate not represented in the four primary resource fields. | Explicitly disclosed by the qualification supplement; budgets are intentionally unmatched, so this is not a hidden budget violation. |
| Resource-accounting limitation | Local RTU's declared `recurrent_state_elements=32` counts actor/critic carry only. | It excludes RTRL and Taylor sensitivities, AC(lambda) eligibility trees, normalization/history, and RNG state; total persistent memory and compute remain unmatched. |
| Low performance risk | Horde's nonlinear critic path uses the documented gradient-only Autostep approximation rather than exact scalar-error Table-1 semantics. | May affect performance; no contract or execution-integrity violation. |
| Low/no-op surface | `include_hint` is ineffective for array FOV observations; recurrent scale/bias fields are inactive when hidden size is zero; `freeze_after_steps=None` and causal `visit_penalty=0` are value-level no-ops. | Shared or explicit inactive settings, not accidental duplicate candidates. |

## q-grid source analysis and planned diagnostic

The q parameter is mathematically live. In
`causal_map_forager._estimated_respawn_delay` (frozen source line 872),
`respawn_quantile_z` changes the Normal upper estimate once exact respawn samples exist.
That estimate is then maxed with the censored interval upper bound, so q can be masked for
long stretches. Exact consecutive reappearance samples are reachable on
`ForagaxTwoBiomeLarge-v1`; therefore the grid is not statically dead or provably
duplicate. The missing item was a task-level behavioral check, not a source-level proof.

The relevant path is:

- `_integrate_observation` (line 1303) updates exact/censored respawn state.
- `_estimated_respawn_delay` (line 872) applies the arm's own quantile.
- `_choose_action` (line 1604) consumes the resulting causal map.
- `causal_map_step` (line 1858) performs one online update per transition.

The diagnostic harness `forager_causal_grid_divergence_probe.py` fixes seed 0, epsilon
0.05, q values 0.50/0.75/0.90, and exactly 10,000 transitions. All lanes receive identical
reset and per-transition environment keys. It commits the action executed before
`env.step` in fixed domain-separated SHA-256 Merkle trees. A canonical paired descent
proves the first divergent index against both candidate roots; it persists at most two
bounded action scalars per divergent pair and never emits a reward array. Exit meanings are:

- `0`: at least one pair diverged;
- `1`: structurally valid no-divergence rejection;
- `2`: malformed identity/runtime/execution failure;
- `3`: the final path was published but durability or replay verification is uncertain.

Version 2 changes only provenance, isolation, cleanup, and receipt validation. Its seed, q
panel, epsilon, horizon, coupled environment-key schedule, divergence gate, action disclosure
boundary, and permanently open-development/nonpromoting scientific semantics are unchanged.
It remains incapable of authorizing promotion, retroactively changing frozen v1, or turning
seed 0 into an evidence gate.

The harness now requires the canonical qualification manifest and its exact detached digest
sidecar, validates the manifest/source/runtime schemas and authority boundary, binds the
Alberta source root and snapshot descriptor to their exact safe manifest-relative paths, and
cross-checks each q arm's exact capability-receipt path, schema, source, entrypoint, and
configuration identity. The qualification root itself is portable: it may live at any supplied
canonical absolute path, while its source root must resolve exactly to the manifest-declared
`sources/alberta/source` tree below it. The v2 receipt carries the manifest and Alberta-source
relative paths, schemas, archive size, and content digests; its candidate identities and the
manifest digest transitively bind the exact configuration and capability-receipt paths. It is
replayed with its detached sidecar from an exact two-file output directory.

Every qualification input lookup rejects a symlinked final component, symlinked ancestor, or
resolved escape from that canonical root. This includes the source, manifest and sidecar,
inventory, archive, snapshot descriptor, original/derived configurations, and capability
receipts.

The parent copies exactly 14 hash-bound qualification inputs, including
`manifest.json.sha256`, into a temporary mode-0444/0755 mirror, mounts only that mirror and a
private exact harness snapshot read-only, and removes both after the child returns. The live
source tree is never mounted. Instead, the child reconstructs the pinned source archive in
private tmpfs and requires its complete extracted file inventory to equal the detailed pinned
inventory before import; the parent revalidates the manifest-bound source tree after execution.
The OCI command remains networkless, read-only, capability-dropped, and non-root, and now
clears both upper- and lower-case proxy variables explicitly.

Stdout/stderr are actively bounded. Timeout, interruption, runner failure, and completed
nonzero-child paths force-remove only a validated collision-resistant internal container name;
if removal reports failure, a separately bounded exact-name daemon query must prove that name
absent with empty stdout and stderr. Failure or overflow of that proof fails closed. Publication
still uses dirfd-anchored Linux `renameat2(RENAME_NOREPLACE)`, held-inode destination checks,
and distinct published-uncertain handling. Runner, kill, bounded-reap, and resource-close
failures are public-error normalized; all final receipt descriptors receive independent close
attempts, and a close failure after rename is published-uncertain. Linux `renameat2` remains
required, and noncooperative same-uid writers remain outside the local filesystem threat model.

## State, RNG, update, and resource audit

- The task is the exact color FOV task with aperture 9 and horizon 499,712. No Alberta
  candidate receives global position, task identity, a reward grid, or other privileged
  evaluator state.
- Environment and agent RNG streams are separate and deterministic. There is one reset
  followed by exactly 499,712 transitions in the matched worker. Horde and RTU perform one
  online update per transition and have no replay buffer.
- RTU's exact/Taylor update selection, ObGD signal use, finite guards, and RTRL state flow
  are internally coherent.
- Local RTU's protocol field `recurrent_state_elements=32` is carry-only. In addition to
  that carry, its persistent state includes 576 RTRL sensitivity elements, 576 Taylor
  sensitivity elements, 4,685 AC(lambda) eligibility elements, normalization/history,
  and RNG state. Those categories are not closed by the frozen four-field ledger.
- `alberta_horde_recurrent64` has 49,477 trainable parameters, 64 recurrent carry
  elements, and 61,248 fixed-substrate floats: 48,768 input-kernel, 12,288 recurrent-kernel,
  and 192 bias values. The qualification supplement records
  `fixed_substrate_parameter_count=61248`; it is not silently omitted.
- Candidate selection intentionally chooses one Alberta winner from 14 arms on 10 fresh
  tuning seeds, then evaluates it on a distinct 30-seed panel. Exact ties use candidate ID.
  The unequal Alberta/external panel breadth is explicit. Held-out v1 specifies three
  Alberta-vs-external contrasts among four selected inferential arms; the two fixed arms
  are descriptive only. It defines no winner among the six executed arms and cannot
  identify the best held-out member of the full registered panel.

## Worker, archive, and scorer closure

The worker's `_load_configuration` (frozen worker line 157) rejects schema or implementation
drift. Its `main` (line 404) emits only the exact finite little-endian float32 reward vector
of shape `(499712,)` into the expected deterministic NPZ archive. The parent executor does
not open reward arrays.

The container helper's `_safe_extract` and `_score` enforce bounded safe extraction and
separate scoring. The scorer's `_load_reward_archive` and `score_rewards` validate ZIP/NPY
structure, dtype, shape, horizon, finiteness, and content hashes before producing bounded
JSON scalars/hashes. `forager_matched_executor._parse_scorer_output` and
`score_seed_archive` strictly validate that output and reverify immutable inputs around
scoring. No host-side reward-array read or feedback path was found.

## Invalidation rules

| Proposed change | Frozen-v1 consequence |
|---|---|
| Change a candidate algorithm, worker, configuration, or registered source byte | Invalidates the associated frozen source/config identity. Create a new source qualification, protocol hash, and campaign root; never patch v1 in place. |
| Change the container helper, executor, scorer, archive contract, or runtime image | Requires new executor/runtime qualification and a new execution plan/root. |
| Change q thresholds or the diagnostic horizon after observing seed 0 | Not allowed for this diagnostic version. Create a separately versioned development diagnostic. |
| Add or change this external diagnostic, its tests, or this report | Does not alter frozen v1 and cannot promote or reject a scientific claim. |
| Observe no divergence at seed 0/10k | Valid engineering rejection for this diagnostic only; it does not prove q is mathematically dead and cannot rewrite the frozen protocol. |

## Verification performed

The live probe and all Forager benchmarks remained unexecuted. Focused non-Forager tests cover
the fixed q semantics, manifest and sidecar identity, exact manifest/source/snapshot/capability
schemas and paths, canonical source-root binding, the 14-file mirror, shared reset/step keys,
pre-step action capture, action commitments, proxy clearing, OCI construction, hard stream
limits, bounded termination/reaping and resource-close errors, exact-name cleanup and absence
proof, source mutation rejection, exact-integer alias rejection, output-path separation, atomic
no-replace publication, published-uncertain handling, exact destination inode checks, and strict
v2 receipt/sidecar replay.

```text
.venv/bin/python -m py_compile forager_causal_grid_divergence_probe.py tests/test_forager_causal_grid_divergence_probe.py
.venv/bin/python -m ruff check forager_causal_grid_divergence_probe.py tests/test_forager_causal_grid_divergence_probe.py
.venv/bin/python -m mypy --strict forager_causal_grid_divergence_probe.py tests/test_forager_causal_grid_divergence_probe.py
.venv/bin/python -m pytest tests/test_forager_causal_grid_divergence_probe.py -q -o addopts=""
```

A read-only `.venv/bin/alberta-evidence-status` check exited `2` in the pre-existing dirty
worktree, consistent with registered-source drift already present there. It was not
silenced or repaired and is independent of this unregistered diagnostic/report change.
