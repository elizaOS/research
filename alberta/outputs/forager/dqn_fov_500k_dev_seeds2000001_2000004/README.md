# Unsealed Forager development comparison

This directory is an integrity receipt for one four-seed, 500,000-step
development comparison on `ForagaxTwoBiomeLarge-v1` with aperture 9. It is
deliberately **not** official evidence and must not be cited as a sealed or SOTA
result.

The directory retains the raw upstream DQN NPZ files and SQLite database. The
matched Alberta RTU/RTRL development run retained scalar metrics only, so its
raw reward traces cannot be independently recomputed from this artifact. See
`DEVELOPMENT_MANIFEST.json` for exact hashes, per-seed values, metric semantics,
runtime identities, source identities, and every known admissibility gap.

On these development seeds, Alberta's mean FOV metric was
`1.5499997668875873`, versus `1.2190922828452528` for DQN. Alberta won all four
paired seeds, with a mean paired difference of `+0.3309074840423348`. This is a
screening result. A valid comparison still requires raw traces for both agents,
the same independently qualified immutable runtime, frozen code and
configuration, disjoint sealed seeds, and the preregistered statistical
protocol.
