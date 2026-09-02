# Capture 07 — deployed-scale NVFP4 error at the MoE FC2 input

## Status

Complete and copied from the provider on 2026-09-02. The accepted local JSON
matches the provider SHA-256, covers every routed MoE layer, and the GPU process
table was empty after shutdown. The first fail-closed attempt is retained as
`failed-01-zero-coverage.log`; it produced no accepted data.

## What this is

The native FlashInfer CUTLASS SM120 MoE fuses FC1, SwiGLU, activation
quantization, FC2, and reduction, so its FC2 input is not exposed to a normal
PyTorch hook. This run loaded the exact same packed NVFP4 checkpoint through
vLLM's official NVFP4 emulation backend. That backend explicitly materializes
the post-SwiGLU tensor and applies the checkpoint's deployed `a2_gscale` QDQ
before FC2. The collector compares those two tensors on a deterministic sample
and persists aggregate sufficient statistics only.

This measures the activation QDQ prescribed by the deployed checkpoint. It is
not a BF16 checkpoint comparison and not a dump of the native fused kernel's
internal bits.

## Main verified facts

- All 42 routed layers (3–44) passed coverage across four TP workers.
- 10 text records from 10 domains, 11,069 estimated tokens. Each layer sampled
  1,568–1,592 route rows, or 802,816–815,104 scalar values.
- Median FC2-input SQNR is 21.049 dB. The layer range is 20.889 dB (layer 44,
  worst) to 28.504 dB (layer 34, best).
- Median relative L2 error is 8.862%; the range is 3.756% (layer 34) to 9.027%
  (layer 44).
- NVFP4 QDQ maps a substantial share of post-SwiGLU values to zero: median
  21.66%, minimum 18.90% at layer 42, maximum 24.36% at layer 3. The original
  sampled tensors are essentially nonzero, so this is a quantization effect at
  this activation boundary.
- Only about 1.21–1.36% of scalar values survive bit-exactly after QDQ.
- Layer 3 has 25.776 dB SQNR and 5.14% relative L2; layer 44 falls to 20.889 dB
  and 9.03% relative L2.

## Provenance and integrity

- Artifact: `fc2-capture.json`
- Artifact SHA-256: `e07ea71aedee6573ebeb768db20ef8ba9887530f448d21d6d107f191f58093ad`
- Accepted runtime log: `run.log`
- Accepted log SHA-256: `a389f7a29e6fcb2d3a4e8b605376d1b81777d76b2737934d9139b8448691d4da`
- Failed log SHA-256: `c1119f0764169d89151a4336eddd3c4edb7689425e90099352319f546ccd7dce`
- Dataset SHA-256: `e2e62cd590838ae817a503e356e2f8ba7f9c1dcfcebf0759e751a3c0302ecc17`
- Checkpoint config/index hashes match Captures 01–06.
- Runtime: vLLM `0.1.dev20051+g487ecf187`, Torch `2.13.0+cu130`,
  Transformers `5.15.1`; `moe_backend=emulation` for this boundary only.
- Accepted wall time: 105.41 seconds including model load and shutdown.
- Provider test suite before accepted run: 41/41 passed.

## Validation and failed evidence

The first run selected the correct emulation backend but returned zero layer
coverage because the decoder wrapper only established layer context while the
Atlas collector itself was enabled. The run failed before writing an artifact.
Its log was copied locally before retrying. The fix passes layer context while
FC2 capture is active without enabling unrelated Atlas metrics. The fresh run
then passed an exact 42-layer gate and was copied through a temporary directory,
JSON-parsed, and hash-compared.

## Boundaries

- This is the exact checkpoint-scale NVFP4 QDQ algorithm over activations
  produced from the exact packed checkpoint, but the exposing execution path is
  vLLM emulation rather than native FlashInfer CUTLASS.
- vLLM logs that differing per-expert activation scales are collapsed with
  `max()` in this backend. The artifact records the actual scalar seen at every
  layer; it should not be presented as per-expert FC2 SQNR.
- Sampled statistics span ten domains but are not a quality benchmark.
- No raw activations, prompts, generations, routes, or weights are stored.
- No BF16 baseline exists, so these numbers do not quantify total checkpoint
  degradation relative to BF16.

