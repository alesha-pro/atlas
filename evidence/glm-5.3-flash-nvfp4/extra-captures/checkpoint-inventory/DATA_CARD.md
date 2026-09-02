# Capture 01 — deployed NVFP4 checkpoint inventory

## Status

Complete and copied from the provider on 2026-09-02. The provider and local
artifact SHA-256 are identical.

## What this is

An inventory of the exact read-only checkpoint mounted at
`/workspace/models/GLM-5.3-Flash-NVFP4-RedHatAI`. It records the full tensor
layout, dtype and shape counts, the embedded quantization configuration, all
36,288 routed-expert `input_global_scale` values, all 36,288 routed-expert
`weight_global_scale` values, and the 45 layers of mHC attention/FFN scales.
The calibration scales remain addressable by layer, expert, and projection.

This is checkpoint evidence, not a behavioural inference measurement. It
required no GPU allocation and did not load a model instance.

## Main verified facts

- 148,498 indexed tensors across 11 safetensors files; 197,843,812,476 bytes.
- Main routed experts contain 36,288 packed `U8` weight tensors, 36,288
  `F8_E4M3` block-scale tensors, and 72,576 scalar `F32` calibration tensors.
  This is exactly 42 layers × 288 experts × 3 projections.
- The auxiliary layer 45 is kept separate: it includes 864 `F8_E4M3` tensors
  and BF16 tensors and must not be described as part of the 42-layer NVFP4
  routed-expert body.
- Shared experts and the Vision tower are BF16 in this deployed checkpoint.
- All captured scalar calibration and mHC values are finite, positive, and
  non-zero.
- `input_global_scale`: min 26.875, median 632, p95 1,624, max 1,792.
- `weight_global_scale`: min 270, median 24,576, p95 26,496, max 28,672.

## Provenance and integrity

- Artifact: `inventory.json`
- Artifact SHA-256: `cfa8fb2f249852f68aaf545d5c6feb1c35c2472182354900e6516a1d33a1d67b`
- Checkpoint `config.json` SHA-256: `29c9f4171196910e99b9c069d6b76c56e3cdcd0f436dc1bacbc9513c9a7529ac`
- Checkpoint index SHA-256: `015faae91e8189c7553f1d48ec3d0694b8c02b282d7f58af2d7b4064a81ce4c0`
- Capture module SHA-256: `31eb36eb30c96849582e37415e3f995e6f5f2510e7145aca339ed27a7203adb7`
- Runner: `scripts/capture_checkpoint_inventory.py`
- Runtime: the pinned extracted SM120 runtime wrapper.
- Tests before capture: 29 passed on the first implementation; after the
  auxiliary-layer classification correction, 11 relevant CPU tests passed.

## Validation

The JSON was parsed locally, record counts were checked against the exact
42×288×3 layout, all scale summaries report zero non-finite values, and the
provider/local SHA-256 values match byte-for-byte.

## Boundaries

- This does not measure BF16-to-NVFP4 damage; no matching BF16 checkpoint is
  available.
- Values inside the large per-block `weight_scale` tensors are not read here.
  Their dtype, shapes, counts, and placement are recorded; a sequential value
  scan is the next, slower static capture.
- No prompts, hidden states, token routes, logits, or raw model outputs are
  stored.

