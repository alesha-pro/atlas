# Weight Atlas

**[atlas.alesha.pro](https://atlas.alesha.pro)** — an interactive canvas for taking a
model apart, tensor by tensor.

One pan/zoom canvas where a checkpoint is laid out as all of its weight tensors.
Every number is measured, not estimated: the weights were actually quantized and
the error computed. It answers one question — **what in this model can be
compressed losslessly, what will fall apart, and for what reason.**

First model on the canvas: **Qwen3.8-27B** — 27.78B parameters, 1199 tensors,
taken off the original bf16 shards.

![canvas](docs/screenshot.png)

## What the data is

Produced by `weight_atlas.py` from the original safetensors shards, one JSON line
per tensor:

- **distribution shape** — mean/std/absmax, |w| percentiles p50…p99.99, kurtosis,
  skew, sparsity, outlier fractions beyond 3/4/6σ, dynamic range
- **histogram** — 29 log₂|w| bins, real normalized shares
- **channel structure** — row/column amax ratios (the outlier-channel problem)
- **spectrum** — top singular values, stable rank, decay
- **measured quantizability** — actual SQNR in dB for INT8 per-channel,
  INT4 group-128 and FP8 e4m3, computed as 10·log₁₀(‖W‖²/‖W−Ŵ‖²)

Metrics that do not apply (1-D tensors: norms, conv1d, biases) render as
*not applicable*, never as a zero.

## Canvas regions

| Region | What it shows |
|---|---|
| **start** | the model in brief: parameters, layers, the full/linear attention rhythm |
| **architecture** | group map (embedding → layers → attention/MLP → head, vision tower, MTP) with live numbers and a rail of every layer |
| **wall** | every tensor at once: column = layer, row = role. Rows are aligned, so the 3:1 attention rhythm and the fragile rows read instantly |
| **links** | scatter of any metric against any other, with presets (tail→INT4, hot channels→INT4, rank→INT4) |
| **depth** | per-layer average of the current metric down the stack, plus quarters |
| **herbarium** | treemap where area is parameter count |
| **records** | extreme points: worst INT4, longest tail, hottest channel, lowest rank |

Click any cell, dot or node and the inspector opens: a plain-language verdict
built from the real numbers, three SQNR figures, the actual histogram, the
singular-value spectrum, channel ratios and the percentile ladder. The metric
switch in the header recolors the whole canvas; search jumps to a tensor by name.

Interface is bilingual — **English by default**, EN/RU switch in the header.

## Run locally

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # static output in dist/ — self-contained, host it anywhere
```

## Adding a model

Nothing is hardcoded: layers, components, wall rows and metric ranges are all
derived from the data, so a new checkpoint drops in without code changes.

1. Put the atlas at `public/models/<slug>/atlas.jsonl`
2. Add an entry to `public/models/manifest.json`:
   ```json
   { "slug": "<slug>", "name": "Model name", "note": "how it was taken" }
   ```

The model picker in the header does the rest.

Expected fields per line: `name, shape, dtype, numel, mean, std, absmax, absmean,
p50…p9999, kurtosis, skew, sparsity, outlier_3s/4s/6s, dyn_range, hist_log2[],
component, layer, shard`; for 2-D tensors additionally `row_amax_ratio,
col_amax_ratio, sqnr_int8_ch, sqnr_int4_g128, sqnr_fp8_e4m3, sv_top[],
stable_rank, sv_decay`.

## Layout

```
src/
  data.ts        # jsonl parsing; layers, groups, slots and metrics derived from the data
  color.ts       # oklch scale, block-kind colors, not-applicable hatching
  world.ts       # canvas engine: pan / zoom / flyTo
  store.ts       # state (metric, selection) and the shared tooltip
  panel.ts       # inspector for a tensor, a layer or a group
  i18n.ts        # every UI string, EN + RU
  ui.ts          # header, search, tours, zoom, minimap
  sections/      # regions: intro, arch, wall, scatter, treemap, records, depth
```

Vite + TypeScript, zero runtime dependencies.

Русское описание — [README.ru.md](README.ru.md).

## License

MIT
