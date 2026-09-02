# Capture 04 — shared/routed contribution and expert output alignment

## Status

Complete and copied from the provider on 2026-09-02. Both local artifacts
match their provider SHA-256 values, and no model process remained on the GPUs
after the run.

## What this is

Two related live measurements from the exact deployed model:

1. Shared-expert and routed-expert output components are observed after the
   runtime's routed scaling. When the normal TP path would reduce only their
   sum, diagnostic copies are separately all-reduced so the measured
   components are full-model contributions.
2. A smaller deterministic token sample replays each of the selected top-8
   experts through the deployed FlashInfer CUTLASS NVFP4 kernel with unit
   routing weight. It aggregates expert output norms, alignment with the
   weighted routed sum, and pairwise output cosine.

Only scalar sufficient statistics and expert/pair sums and counts are saved.

## Main verified facts

- 305 balanced records and 321,349 estimated tokens across the same 14-domain
  mix as Capture 03, including 40 real-image records.
- All layers 3–44 are present. Shared/routed components use 609–614 sampled
  rows per layer. Expert replays use 152–156 sampled tokens per layer and cover
  at least 257 of 288 experts in every layer.
- Early layers are routed-dominated. The minimum mean shared energy fraction is
  0.2185 at layer 5; mean shared/routed norm ratio is about 0.54 at layers 4–5.
- Layer 44 is shared-dominated: mean shared energy fraction 0.8577 and mean
  shared/routed norm ratio 3.2677.
- Mean shared/routed cosine ranges from -0.0146 at layer 5 to 0.2637 at layer
  44. Shared and routed branches therefore change both relative magnitude and
  alignment across depth.
- Across all 179,984 sampled expert-output pairs, mean cosine is only 0.01577.
  Most selected expert outputs are close to orthogonal on this sample; this is
  not evidence that the experts are interchangeable.
- The strongest eligible sampled pair is layer 3 experts 253+255 with mean
  output cosine 0.4797 over seven samples. The most opposed eligible pair is
  layer 30 experts 13+195 at -0.0204 over five samples. Low-count pair estimates
  are explicitly retained with counts and must be treated as noisy.

## Provenance and integrity

- JSON SHA-256: `8fb058fb9f42907d0f7126ef78381730644d07359129ed582bd8bf82d765a5c7`
- NPZ SHA-256: `a89ad4226ade9db40cc4391f4b158ff3a67d97830d2ea55f484daf3deb28956e`
- Collector module SHA-256: `e54d785644aff111bd70405a0bbcc2df605d5662af311aa6c648946127514c4c`
- Runner SHA-256: `82c0188b1832d90325b956c7d66cb0f2d78fea27a89c4e7b7cd65735fed5f7a6`
- Config SHA-256: `fd7f470ad2ac4192451256fb1cb6116a69b064469654f53e1cf0def57dec782e`
- Runtime hook SHA-256: `631f9464cb1b8822bcfb54ae4340d48e8e05f4ea9b7f43cdada1d7581c7cd257`
- Dataset and checkpoint hashes plus exact runtime versions are embedded in
  JSON and match the earlier captures.
- Accepted-run wall time including load and warmup: 303.9 seconds.
- Provider test suite before accepted run: 31/31 passed.

## Validation

All four TP ranks have exactly identical integer sample/pair counts. Floating
norm/cosine/component fingerprints agree within relative tolerance 1e-4 and
absolute tolerance 1e-5. For every layer, expert counts equal sampled
tokens×8, pair counts equal sampled tokens×28, all values are finite, all 42
layers are present, and no shared output is missing. Both files parse locally
and their hashes match the provider copies.

An earlier complete pass was deliberately rejected because its gate required
bit-identical floating reduction bytes across TP ranks. That is stricter than
valid floating-point reproducibility. No artifact from that pass was kept; the
accepted run used exact equality for integer counts and tolerance-checked
floating fingerprints.

## Boundaries

- This is output-space behaviour of the deployed NVFP4 kernel, not a BF16
  comparison.
- Pair estimates are sparse. The NPZ preserves sums/counts so future views can
  choose stricter support thresholds than the JSON's minimum of five.
- High output cosine alone does not establish causal redundancy or pruning
  safety; that requires the later intervention arms.
- No hidden vectors, token routes, prompts paired with activations, logits, or
  generated text are stored.

