# GLM-5.3-Flash NVFP4 extra captures

This directory is the registry for measurements taken from the temporary live
provider checkpoint after the baseline REAP and behavioural Atlas captures.
Each completed stage has its own directory, immutable raw artifact, and data
card. A stage is only marked complete after local copy, validation, and SHA-256
comparison with the provider copy.

| Order | Capture | Status | GPU required | Artifact |
|---:|---|---|---|---|
| 01 | Deployed checkpoint and scalar quantization-scale inventory | complete | no | `checkpoint-inventory/inventory.json` |
| 02 | Routed-expert block-scale distribution scan | complete | no | `weight-scale-scan/weight-scale-scan.json` |
| 03 | Router dynamics, margins, entropy, co-routing and burstiness | complete | yes | `router-dynamics/router-dynamics.{json,npz}` |
| 04 | Shared-vs-routed contribution and expert alignment/redundancy | complete | yes | `contributions/contributions.{json,npz}` |
| 05 | Richer KDA and sparse-indexer position/head aggregates | complete | yes | `rich-sequence/atlas-rich-sequence.json` |
| 06 | Paired Vision original/blank/mismatch/text-only effects | complete | yes | `paired-vision/paired-vision.json` |
| 07 | Sampled deployed-scale FC2 activation SQNR | complete | yes | `fc2-activation/fc2-capture.json` |
| 08 | Causal REAP route-ablation arms | complete | yes | `pruning-arms/pruning-arms.json` |

The BF16 comparison arm is intentionally absent because this provider mount
does not contain the matching BF16 checkpoint.
