# FOV stateful-baseline screening v1

This is a **frozen configuration, execution-pending, open-development** protocol. It defines an
eight-candidate second-stage screen on current `continual-foragax==0.55.0`; it contains no
full-horizon reward results and makes no superiority, SOTA, scientific-evidence, or promotion
claim. The one-transition CPU checks in `CONFIGURATION_SMOKE_RECEIPT.json` are construction and
serialization checks only. Their first-transition rewards are not performance measurements.

## Exact task

- Environment: `ForagaxTwoBiomeLarge-v1`
- Aperture: `9`
- Observation: `color`
- Open development run indices and effective seeds: `2000001`, `2000002`
- Horizon: `102400` transitions per seed
- Upstream source: `steventango/continual-foragax-agents` commit
  `9710f60fa30da5badc451ad7ce3ff296d5070830`
- Runtime distribution: exactly `continual-foragax==0.55.0`

All source, template, configuration, image, helper, and paper hashes are recorded in
`PROTOCOL.json`. The paper cross-check is specifically arXiv `2605.01131v1`, Table 6. The paper-v1
DRQN and the current audit-checkout/XFinal DRQN are deliberately separate candidates:

| Configuration | Fixed provenance | Distinguishing settings |
| --- | --- | --- |
| `DQN-PT-architecture-control.json` | X34 PT-DQN derived control | DQN dispatch; PT fields alone removed |
| `DQN-ReDo-architecture-control.json` | XFinal ReDo derived control | DQN dispatch; ReDo fields alone removed |
| `DQN_ReDo_PostLNScore.json` | XFinal | ReDo, post-LayerNorm score, frequency 2500 updates |
| `DRQN-current-XFinal.json` | current audit checkout, XFinal | alpha `0.0001`, epsilon `0.25`, batch `32` |
| `DRQN-paper-v1.json` | paper v1 Table 6-compatible, X33 | alpha `0.001`, epsilon `0.1`, batch `4` |
| `PPO-RTU_LN_128_1_relu.json` | R1 | RTU, rollout `128`, explicit `800` updates |
| `PPO_2048_relu.json` | R1 | feedforward, rollout `2048`, explicit `50` updates |
| `PT_DQN.json` | paper v1-compatible, X34 | alpha `0.0003`, width `32`, two layers, PT state |

The X34 PT-DQN candidate changes only the environment ID and horizon from its fixed template. Its
plain-DQN control additionally changes the agent dispatch and deletes only `pm_buffer_size`,
`pt_decay`, `pt_optimizer`, and `pt_update_freq`. The two DRQN variants share a 64-unit GRU,
sequence length 32, and burn-in 16, but their optimization, exploration, and batch settings are
not interchangeable.

## Frozen score and selection

For each raw reward trace, with `e[-1] = 0`, compute
`e[t] = 0.999 * e[t-1] + 0.001 * reward[t]` without bias correction. Sample `e[t]` at zero-based
indices `0, 100, ..., 102300`, yielding 1024 samples. The per-seed score is the arithmetic mean of
sampled indices `921..1023` (103 values, the final 10% under the frozen integer boundary). The
implementation is hash-bound as `score_raw_rewards.py`; collector summaries are forbidden.

An eligible candidate's ranking value is the arithmetic mean of its two per-seed scores. Sort
descending and advance exactly the top three; exact ties are broken by configuration path in
ascending Unicode code-point order. If fewer than three candidates are eligible, advance all
eligible candidates without substitution, retuning, or reward-informed reruns. The PT and ReDo
paired contrasts are descriptive and do not alter this ranking.

## Budget asymmetries

This is operational triage, not a controlled memory or architecture ablation. DQN-family agents
schedule 25,587 gradient-update opportunities, but replay batch and auxiliary state differ. The
two PPO schedules both consume exactly 102,400 transitions, while feedforward PPO schedules 6,400
optimizer minibatch steps and RTU-PPO schedules 3,200. Persistent recurrent state, replay state,
parameter counts, diagnostic work near step 100,000, and per-update compute are not equalized.

## Validation and smoke reproduction

From the Alberta repository root, validate all derivations and upstream hashes against the audited
checkout:

```bash
.venv/bin/python outputs/forager/fov_stateful_baseline_screening_v1/validate_protocol.py \
  --upstream-root /tmp/forager-audit.XPpQoQ/continual-foragax-agents \
  --require-receipt
```

The receipt's eight smoke records were produced one configuration at a time with this exact
container contract, run from this protocol directory (replace `<config.json>`):

```bash
docker run --rm --network none --read-only --user 65532:65532 \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 512 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=2g \
  --env JAX_PLATFORM_NAME=cpu --env JAX_PLATFORMS=cpu \
  --env NUMBA_DISABLE_JIT=1 --env NVIDIA_VISIBLE_DEVICES=void \
  --env CUDA_VISIBLE_DEVICES= \
  --mount type=bind,src="$PWD",dst=/protocol,readonly \
  alberta-forager-runtime:dev2-9710f60 \
  /protocol/smoke_one_transition.py \
  --config /protocol/configs/<config.json>
```

The wrapper first validates the full configuration and exact two-index resolution. DQN-family
checks execute the real `src/continuing_main.py` with `--max_steps 1`. PPO checks execute the real
`src/rtu_ppo.py`, after smoke-only scheduling overrides to one rollout step, one minibatch, and one
update and disabling frame allocation and diagnostics. There is no video/save stub: both lanes use
the upstream save paths and each produces an actual one-transition NPZ archive.

Full screening execution remains pending and is intentionally absent from this directory. Any
future run must use new writable output paths, preserve all hashes and exact indices, retain raw
reward archives, and score those archives with `score_raw_rewards.py`. It cannot promote a claim or
overwrite any pinned Alberta evidence artifact.
