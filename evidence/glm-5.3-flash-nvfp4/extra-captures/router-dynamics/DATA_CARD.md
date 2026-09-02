# Capture 03 — live router dynamics

## Status

Complete and copied from the provider on 2026-09-02. Both provider/local
SHA-256 pairs match and the model workers exited after the run. No GPU compute
process remained after shutdown.

## What this is

A live, balanced inference pass through the exact deployed NVFP4 checkpoint.
For all 42 routed-expert layers it captures aggregate top-8 router entropy,
effective expert count, concentration, top-1/top-2 margin, selection frequency,
rank occupancy, unordered expert co-routing, prompt-position buckets, and
prefill-call load burstiness. Domain aggregates cover agent, code, general,
knowledge, long-context, three languages, reasoning, systems, and four real
image-bearing Vision domains.

The NPZ contains sufficient-count arrays for later analysis. The JSON contains
scalar summaries and the top 100 expert pairs by count and independence lift.
No per-token routes or prompt-output pairs leave the workers.

## Main verified facts

- 305 records, 321,349 estimated corpus tokens, and 306,086 actual routed
  prefill tokens per MoE layer across 14 domains; 40 records contain real
  images.
- Exact coverage: layers 3–44, 288 experts, top-8 routing.
- All four TP workers produced identical aggregate signatures; rank 0 was
  accepted only after this check.
- Position coverage is complete. Token counts in buckets 0–31, 32–127,
  128–511, 512–2047, 2048–8191, and 8192+ are respectively 10,065, 27,677,
  70,587, 84,541, 90,750, and 22,466.
- Router weights normalize to one within floating-point noise at every layer.
- Mean effective top-8 breadth rises from 6.073 at layer 3 to a maximum 7.807
  at layer 43. Mean top-1/top-2 margin falls from 0.1559 at layer 3 to 0.0222
  at layer 43: late routing is much flatter within the selected eight.
- Selection-frequency Gini ranges from 0.1562 (layer 4) to 0.3654 (layer 23),
  while top-1 Gini reaches 0.5672 at layer 44. Flat within-token weights do not
  imply uniform expert usage across the corpus.
- Example frequent co-route: layer 44 experts 35+263 occur together for 16,859
  tokens (5.51%). Example affinity beyond marginal popularity: layer 44 experts
  178+267 occur 791 times with an independence lift of 60.76. These are
  descriptive associations, not causal redundancy claims.

## Provenance and integrity

- JSON SHA-256: `1af784f0adf7889e74be17729714adf12290abf0347999cc116a2f698aaed554`
- NPZ SHA-256: `8a434dfb6f6e890909e255c41a0a1baffdae67384b53e18ffbe62b9cf90b96ce`
- Dataset SHA-256: recorded inside JSON run metadata.
- Checkpoint config and index SHA-256: recorded inside JSON run metadata and
  match Capture 01.
- Collector module SHA-256: `7ffd964fe29fec19528832472accb325ced789ecdf1c0973c9d7cc8952e309cc`
- Runner SHA-256: `6e9abe4da58ff2974dd485ef2b89d3261c24e753ed50d240ac8316c90408ed68`
- Config SHA-256: `febaffcff6a7a71423ce91925e5f4995070de14a23e0a2a9379689f6f60370f0`
- Runtime hook SHA-256: `200497c357f6f66ba1201ff3dbffa78456249fb53ea3e2306b5ed6c4ba639450`
- Runtime: pinned extracted SM120 vLLM/Torch/Transformers versions recorded in
  JSON.
- Wall time including model load and warmup: 251.6 seconds.

## Validation

For every layer, selected counts equal tokens×8, every rank contains exactly
one selection per token, unordered pair counts equal tokens×C(8,2), position
counts equal routed tokens, position-expert counts equal position tokens×8,
domain token counts reconcile to the all-domain total, and all 42 layers pass.
Both artifacts parse locally and their provider/local hashes match.

The first attempted startup produced no data because the main remote checkout
did not contain the optional behavioural-Atlas module. The hook was made
dependency-independent, the complete provider test suite passed 29/29, GPUs
were confirmed empty, and the accepted run started from a fresh model instance.

## Boundaries

- Results describe the deployed NVFP4 model and this balanced corpus; no BF16
  control exists.
- Co-routing is association, not interchangeable function or safe pruning.
- Call-level load ratios describe vLLM prefill batches, not a persisted
  per-prompt time series.
- Only aggregate matrices and sufficient statistics are stored; raw routes,
  hidden states, prompts paired with activations, and generated text are absent.

