# Causal-map 500k open-development result

This directory records one completed, unsealed run of the default
`alberta_causal_map` policy on four already-consumed field-of-view development
seeds. The run used `continual-foragax==0.55.0`, JAX 0.9.0.1 on CPU, the
reproducible OCI image identified in `receipt.v1.json`, a read-only source
mount, and 500,000 transitions per seed.

The per-seed FOV tail-EMA AUC values were `2.193037`, `1.910047`, `1.741909`,
and `1.470715`; their arithmetic mean was `1.828927`. On the same seed labels,
the difference from the earlier RTU-RTRL development receipt was positive in
all four cases and averaged `0.278927`.

This is an exploratory development result, not scientific evidence. The
configuration was not preregistered, the source closure and raw reward traces
were not persisted, and the RTU execution used a different runtime contract.
The receipt therefore prohibits inferential, promotion, or SOTA claims. A
candidate may advance only through a frozen tuning protocol followed by a
fresh, disjoint-seed, matched-runtime evaluation against the strongest
admissible learning baseline.
