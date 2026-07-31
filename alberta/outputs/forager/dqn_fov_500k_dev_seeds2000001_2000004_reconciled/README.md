# Reconciled unsealed Forager development comparison

This directory corrects and supersedes the read-only development receipt at
`outputs/forager/dqn_fov_500k_dev_seeds2000001_2000004/DEVELOPMENT_MANIFEST.json`
without modifying its historical bytes. The original receipt used a
hand-coded EMA numerator that differed at the bit level from the public
importer and did not bind the RTU receipt.

`receipt.v1.json` recomputes DQN through the public importer, binds the
immutable RTU receipt and its capture correction, records the post-output
comparator timeline, and makes the feature, resource, and runtime mismatches
explicit. On the four consumed open-development seeds, RTU's captured FOV
metric averages `1.5499997668875873`, DQN averages
`1.2190922828452653`, and the descriptive paired difference averages
`+0.3309074840423221` with four positive differences.

This is deliberately unsealed and permanently nonpromoting. It is not an
official, inferential, causal, speed, SOTA, or Alberta Plan completion result.
