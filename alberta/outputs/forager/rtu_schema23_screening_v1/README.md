# Alberta RTU schema 2.3 FOV screening

This directory freezes an open-development comparison of six Alberta
RTU/RTRL variants on `continual-foragax==0.55.0`,
`ForagaxTwoBiomeLarge-v1`, color observations, and aperture 9. It is not a
sealed evaluation and cannot support an official or SOTA claim.

The matrix crosses hidden widths 8, 16, and 32 with the optional
parameter-diagonal Taylor correction disabled and enabled. All other
hyperparameters, causal input features, seeds, horizon, batch mode, and metric
contracts are identical. The field-of-view environment does not emit a hint,
so the retained `include_hint=true` setting is behaviorally inert and does not
provide a task label or hidden cue.

Each of the 24 variant-seed runs consumes 500,000 transitions. Schema 2.3 uses
the content-verified snapshot subprocess and writes canonical raw float32
reward and biome-regret sidecars. Reported FOV scores must be recomputed from
those traces using the unadjusted EMA with decay 0.999, initial value 0, a
sample at the first reward and every 100 rewards, and the mean over the final
10% of the sampled curve.

The single development candidate is selected by the lower endpoint of a
95%-confidence percentile-bootstrap interval over the four fixed tuning
seeds, with 10,000 resamples, bootstrap seed 2,100,000, and ascending variant
ID as the exact tie-break. The 30 evaluation seeds are declared but must not
be consumed or inspected until a completed tuning report freezes one selected
variant and a qualified matched runtime is available.

The Taylor option is an approximation, not exact moving-parameter RTRL. It
corrects only the parameter-diagonal staleness term; simultaneous movement of
several recurrent parameters can retain a first-order mixed-Hessian residual,
so it is not guaranteed to improve either sensitivity accuracy or reward.

Validation before execution:

```bash
uv run --python 3.12 python -m alberta_framework.benchmarks.forager_matrix \
  outputs/forager/rtu_schema23_screening_v1/matrix.json \
  --output-dir outputs/forager/rtu_schema23_screening_v1_execution \
  --dry-run
```

The input matrix file SHA-256 is
`a8ae7c21a24fc65f599e23e7d605e4b75ef64f433e3e25623267343c980ad35a`;
its normalized scientific configuration SHA-256 is
`994643dfeb2977faca445ee481a8e8bd4175ff35535ea3920113e569e243081b`.
