# Capture 05 — per-head KDA and richer sparse-indexer dynamics

## Status

Complete and copied from the provider on 2026-09-02. The local JSON parses,
passes the enhanced coverage gate, and matches the provider SHA-256. No GPU
compute process remained after shutdown.

## What this is

An enhanced behavioural Atlas pass over the exact deployed NVFP4 model. It
retains all earlier signal-flow, activation, quantization, KDA, sparse-indexer,
and Vision metrics, then adds:

- KDA beta-open, log-decay, decay, and half-life separately for every one of
  64 global heads on every KDA layer;
- KDA beta/log-decay/half-life by prompt-position bucket;
- sparse-indexer selected distance by prompt-position bucket;
- sparse-indexer selected distance by returned rank bucket (0–31, 32–127,
  128–511, 512–2047).

TP-local KDA heads are concatenated in verified rank order. Only aggregate
sufficient statistics are exported.

## Main verified facts

- 305 balanced records, 321,349 estimated tokens, 14 domains, and 40 real
  image-bearing records; wall time 290.01 seconds including load and warmup.
- Coverage passed for all 45 language-flow layers, 34 KDA layers, 11 sparse
  MLA/indexer layers, 42 deployed NVFP4 FC1 layers, and 24 Vision blocks.
- 136 per-head metric groups = 34 KDA layers × 4 metrics, each with exactly 64
  global heads. The artifact also contains 40,952 finalized scalar metrics.
- Head dynamics are extremely heterogeneous. Mean beta-open ranges from
  0.0434 (layer 36, head 61) to 0.9956 (layer 34, head 17).
- Mean KDA half-life across individual heads ranges from 0.247 tokens (layer
  0, head 58) to about 2.71 million tokens (layer 36, head 33). Very large
  means arise when learned log-decay is close to zero; they should be shown on
  a log scale and not summarized by one arithmetic mean.
- At sparse layer 43, mean selected distance grows from 9.8 tokens for queries
  at positions 0–31 to 7,475 tokens for queries at 8192+. This confirms the
  indexer actually reaches far context rather than selecting only a fixed
  local window.
- At the same layer, mean selected distance also varies with returned rank:
  1,630 tokens for ranks 0–31 versus 2,276 for ranks 512–2047. Higher-ranked
  memory choices are more local on average, but still strongly non-local.

## Provenance and integrity

- Artifact: `atlas-rich-sequence.json`
- Artifact SHA-256: `d03960829640cf3b87b88aea4677c62020788bb80b817e4dd7c1c0cbf03667ed`
- Collector module SHA-256: `3357637b3715c5b06e83ecf495077d50a3603c905f256384de956d7d264f631b`
- Runner SHA-256: `1412289448df66e0b0fee3a068ab63820e02c58f417440fd6ed8a111bdf9701f`
- Config SHA-256: `c3f4918b337f3b71ef5d86f7ac869f771cd1e0d02f86edc40fbbe2f60886866a`
- Dataset SHA-256: `e2e62cd590838ae817a503e356e2f8ba7f9c1dcfcebf0759e751a3c0302ecc17`
- Checkpoint config/index hashes match Captures 01–04 and are embedded in the
  run metadata.
- Provider test suite before run: 35/35 passed.

## Validation

The enhanced gate requires the exact expected KDA and sparse-layer sets and
64 records for every per-head metric. The ordinary behavioural gate also
requires every language, KDA, sparse-indexer, deployed FC1, and Vision layer.
All gates passed. The copied JSON was parsed again locally and its SHA-256 is
byte-identical to the provider file.

## Boundaries

- Half-life can have a heavy tail near zero log-decay; head-level distributions
  are the evidence, not a single global mean.
- The fused sparse-MLA backend exposes selected token IDs but not dense
  attention probabilities. Distances and rank buckets describe retrieval
  selection, not attention mass.
- This pass describes the deployed NVFP4 checkpoint and FP8 KV configuration;
  no BF16 baseline exists.
- No raw states, per-token indexer selections, prompts paired with activations,
  logits, or generations are stored.

