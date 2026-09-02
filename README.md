<h1 align="center">Weight Atlas</h1>

<p align="center">
  An interactive canvas for taking a model apart, tensor by tensor.<br>
  <a href="https://atlas.alesha.pro"><b>atlas.alesha.pro</b></a>
</p>

<p align="center">
  <img alt="MIT" src="https://img.shields.io/badge/license-MIT-2d2a24?style=flat-square">
  <img alt="static" src="https://img.shields.io/badge/build-static%20site-b0492a?style=flat-square">
  <img alt="deps" src="https://img.shields.io/badge/runtime%20deps-0-3f6f6a?style=flat-square">
  <img alt="languages" src="https://img.shields.io/badge/UI-EN%20%2F%20RU-6a4f8c?style=flat-square">
</p>

![Every tensor of the checkpoint on one canvas](docs/screenshot.png)

A checkpoint arrives as a directory of safetensors shards and stays a black box.
Weight Atlas lays all of it out on a single pan and zoom canvas and answers one
question: **what in this model can be compressed losslessly, what falls apart,
and why.**

The picker now contains two very different autopsies:

- **Qwen3.8-27B** — 27.78B parameters across 1199 tensors, read from the
  original bf16 shards.
- **GLM-5.3-Flash NVFP4** — a 320B-total / 18B-active multimodal MoE captured
  while the released NVFP4 checkpoint was live, with 42 × 288 exact REAP
  scores, routing and contribution maps, KDA memory, sparse-indexer reach,
  causal Vision arms, deployed quantization statistics and pruning controls.

For GLM, `FC2 QDQ` is the activation quantize/dequantize measurement captured
with the checkpoint's own deployed scale. Every other GLM panel likewise shows
the evidence we actually captured, without projecting it onto another format.

## Every number is measured

Nothing here is a heuristic or a rule of thumb. Each weight tensor was actually
quantized and the error computed as SQNR in dB, `10·log₁₀(‖W‖²/‖W−Ŵ‖²)`, for
three schemes: INT8 per-channel, INT4 group-128 and FP8 e4m3. Spectra come from
real SVDs, histograms from real bins, outlier channels from real row and column
maxima.

Metrics that cannot apply to a tensor (norms, conv1d, biases) render as hatching
and say *not applicable*. They never quietly become a zero.

The scan is a single script over the shards, 116 seconds on one GPU for a 27B
model:

```bash
pip install torch safetensors
python3 scripts/weight_atlas.py /path/to/checkpoint atlas.jsonl --device cuda
```

## What is on the canvas

| Region | What you get |
|---|---|
| **start** | the model in brief: parameters, layers, the full and linear attention rhythm |
| **architecture** | group map from embedding to head, with the vision tower and the MTP draft head, every box carrying live numbers |
| **wall** | all 1199 tensors at once. Column is a layer, row is a role, so the 3:1 attention rhythm and the fragile rows read at a glance |
| **links** | scatter any metric against any other, with presets for tail length, hot channels and rank against INT4 damage |
| **depth** | how the current metric drifts down the stack, by layer and by quarter |
| **herbarium** | treemap where area is parameter count |
| **deep dive** | architecture teardown: block diagrams, interactive demos of gated attention, DeltaNet and RoPE, and the papers each piece came from |
| **living model** | the same checkpoint running, see below |

![The wall: every tensor, column is a layer, row is a role](docs/wall.png)

Click any cell, dot or node and the inspector opens with a plain-language
verdict built from that tensor's real numbers: three SQNR figures, the actual
histogram, the singular-value spectrum, channel ratios and the percentile
ladder. The metric switch in the header recolors everything at once, and the
colour scale flips between *within model* and *absolute dB* so you can compare
a tensor either against its neighbours or against the same scale for every model.

## The living model

Weights only tell you what the model is. The living model region shows what it
does, captured by running the bf16 checkpoint with hooks over a calibration mix
of english, code and agent traces.

![The living model: real activations, real attention, real damage](docs/living.png)

- Residual flow down all 64 layers, per domain, and each layer's contribution
- Activation quantizability at 12 sites, INT8 against FP8
- Attention entropy, sink, distance decay and output gates for all 16 full-attention layers
- Real attention maps on an 89-token paragraph, mean over 24 heads plus the head that differs most, hover any token pair
- The 48 linear-attention memories: write gate β, state RMS, half-life, per-head λ
- 17408 FFN neurons per layer fingerprinted by domain, with the most one-sided ones named
- Per-layer fragility: the actual KL divergence of quantizing exactly one layer to INT4
- A vision pass where the model is handed a screenshot of this very atlas and 1612 image tokens enter its context

Everything in that region is a measurement with its limits written next to it.
Where a number is a guess, the card says so.

### GLM-5.3-Flash NVFP4

GLM has its own evidence-led living region rather than forcing MoE data into the
Qwen charts:

- 12,096 expert cells switchable between exact REAP, route share and sampled
  output contribution; REAP has 14 domain slices
- router decisiveness and load inequality through all 42 routed layers
- split-half ranking stability and controls against frequency-only importance
- 34 × 64 KDA head half-lives and long-position sparse-indexer reach
- shared-versus-routed contribution, NVFP4 block-scale structure and FC2 QDQ
- four causal Vision arms and a five-arm causal REAP stress test

The architecture dossier is grounded in the released config and primary
sources. The public data bundle contains aggregates and checksums, not prompts,
generations, images, activations or raw routes.

## Bring your own checkpoint

The core weight view derives layers, components, wall rows, metric ranges and
colour domains from the data. Models with distinct runtime evidence can add a
dedicated gated section, as GLM does in `src/sections/glm.ts`.

1. Scan it and drop the result at `public/models/<slug>/atlas.jsonl`
2. Add one line to `public/models/manifest.json`:
   ```json
   { "slug": "<slug>", "name": "Model name", "note": "how it was taken" }
   ```
3. Optionally add `dossier.json` for the architecture teardown, and run the live
   pass for the living model region

Both optional files are gated: without them those two regions simply do not
appear and the rest of the canvas works unchanged. The model picker in the
header does the rest.

The live pass has a capture stage and a reduce stage. `atlas_live.py` runs the
checkpoint with hooks and writes raw artifacts; `reduce_live.py` folds those plus
the FFN neuron statistics into `live.json` and `attn_maps.json`:

```bash
python3 scripts/atlas_live.py --mode all --data data/calibration.jsonl
python3 scripts/reduce_live.py --artifacts out/live --carve out \
        --dest public/models/<slug>/live.json
```

## Run it

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # static output in dist/
```

The build is a static site with zero runtime dependencies. Host it on Pages,
S3, a spare nginx, anything that serves files. The interface is bilingual,
English by default, with an EN/RU switch and a light and dark theme in the
header.

## Under the hood

```
src/
  data.ts        # jsonl parsing; layers, groups, slots and metrics derived from the data
  color.ts       # oklch scale, block-kind colours, not-applicable hatching
  world.ts       # canvas engine: pan, zoom, flyTo, touch
  store.ts       # state (metric, scale, selection) and the shared tooltip
  panel.ts       # inspector for a tensor, a layer or a group
  i18n.ts        # every UI string, EN and RU
  ui.ts          # header, search, tours, zoom, minimap
  sections/      # intro, arch, wall, scatter, treemap, records, depth,
                 # plus dossier, Qwen live and GLM evidence views
scripts/          # see scripts/README.md for the full pipeline
  weight_atlas.py       # the scan: one jsonl line per tensor
  build_calibration.py  # the english / code / agent token mix
  run_capture.py        # FFN activations, per-neuron firing statistics
  carve_hooks.py        # the collector run_capture.py imports
  atlas_live.py         # live capture: hooks on the bf16 model, attention registry
  reduce_live.py        # folds artifacts into live.json + attn_maps.json
  build_glm_atlas.py    # reduces preserved GLM captures into public aggregates
```

Expected fields per tensor line: `name, shape, dtype, numel, mean, std, absmax,
absmean, p50…p9999, kurtosis, skew, sparsity, outlier_3s/4s/6s, dyn_range,
hist_log2[], component, layer, shard`; 2-D tensors additionally carry
`row_amax_ratio, col_amax_ratio, sqnr_int8_ch, sqnr_int4_g128, sqnr_fp8_e4m3,
sv_top[], stable_rank, sv_decay`.

Русское описание: [README.ru.md](README.ru.md).

## License

MIT
