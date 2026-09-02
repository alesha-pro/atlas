# Capture 02 — routed-expert NVFP4 block-scale scan

## Status

Complete and copied from the provider on 2026-09-02. The first direct copy was
rejected because JSON parsing detected an interrupted transfer. The final copy
was transferred compressed, parsed successfully, and its uncompressed SHA-256
matches the provider artifact.

## What this is

An exact histogram scan of every `F8_E4M3` block-scale value associated with
the packed NVFP4 weights of the 42 main routed-expert layers. It covers every
layer, every one of 288 experts, and all three projections. Per-tensor
summaries remain addressable by layer, expert, and projection; exact 256-code
histograms are retained globally and by layer/projection.

The scan is resumable per safetensors shard. It read weight-scale tensors only,
used CPU and storage I/O, and did not allocate a model or GPU memory.

## Main verified facts

- 36,288 scale tensors = 42 layers × 288 experts × 3 projections.
- 19,025,362,944 individual block-scale values scanned.
- All values are finite and positive; zero, negative, and non-finite counts are
  all zero.
- Overall: min 0.5625, mean 155.5104, median 160, p99 256, max 448.
- Each projection contributes 6,341,787,648 values.
- `down_proj`: mean 160.7387, median 160, p99 256.
- `gate_proj`: mean 152.6221, median 144, p99 256.
- `up_proj`: mean 153.1703, median 144, p99 256.
- The maximum finite E4M3 code (448) occurs 70,296 times, about 0.00037% of
  scanned values. This is scale-code incidence, not proof of clipped weights.
- A small set of late-layer `down_proj` tensors has unusually low mean scales;
  for example layer 44 expert 238 has mean 2.1084 and p99 5.5 while retaining
  one 448 value. The artifact preserves these outliers for later Atlas views.

## Provenance and integrity

- Artifact: `weight-scale-scan.json`
- Artifact SHA-256: `2a12b6402b9cda8c39cd953e88c6b6534800f4ed18ab4bb59ec5d1dede69b798`
- Checkpoint `config.json` SHA-256: `29c9f4171196910e99b9c069d6b76c56e3cdcd0f436dc1bacbc9513c9a7529ac`
- Checkpoint index SHA-256: `015faae91e8189c7553f1d48ec3d0694b8c02b282d7f58af2d7b4064a81ce4c0`
- Capture module SHA-256: `d570edaadedbff07e160d402a89029e696cc707951847aeeef7073aa1fec9c05`
- Runner SHA-256: `c5cf2605914e403952d7aece0b19a40d5aebe8aecc78d7bffb16ded2564b1946`
- Runtime: the pinned extracted SM120 runtime wrapper.
- Tests before capture: 13 relevant CPU tests passed.

## Validation

The final local JSON parses; the tensor count equals the exact architectural
product; global, projection, layer, and per-tensor counts reconcile; every one
of 42 main MoE layers is present; no per-tensor summary reports zero or
non-finite values; and local/provider SHA-256 hashes match byte-for-byte.

## Boundaries

- This characterizes the quantizer's deployed scale field. It does not compare
  reconstructed weights against BF16 and therefore is not quantization-error
  or quality evidence.
- The auxiliary layer 45 is excluded from the main routed-expert scan.
- Packed U8 weight codes are not read in this stage.
- No prompts, hidden states, routes, logits, or outputs are stored.

