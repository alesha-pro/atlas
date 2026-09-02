# Producing the data

Five scripts. The first one alone gives you a working atlas; the other four add
the living model region and can be skipped entirely.

Set the checkpoint path once and every script picks it up:

```bash
export ATLAS_MODEL_DIR=/path/to/checkpoint     # or pass --model-dir
```

```
weight_atlas.py ────────────────────────────────────────────► atlas.jsonl
                                                              (the canvas)

build_calibration.py ──► calibration.jsonl ─┬─► run_capture.py ──► *.parquet ─┐
                                            │                                 ├─► reduce_live.py ──► live.json
                                            └─► atlas_live.py ──► out/live/*.json ┘        + attn_maps.json
```

## 1. weight_atlas.py: the weights

Reads the safetensors shards and writes one JSON line per tensor: distribution
shape, log₂ histogram, channel ratios, spectrum, and the measured SQNR for INT8
per-channel, INT4 group-128 and FP8 e4m3. This is the only required step.

```bash
pip install torch safetensors
python3 weight_atlas.py $ATLAS_MODEL_DIR atlas.jsonl --device cuda
```

| Flag | Meaning |
|---|---|
| `--device` | `cuda` or `cpu`. On CPU expect hours instead of minutes |
| `--no-spectral` | skip the SVDs; faster, but the spectrum column goes dark |
| `--limit N` | first N tensors only, for a smoke test |

Qwen3.8-27B: 1199 tensors in 116 s on one GPU. Drop the result at
`public/models/<slug>/atlas.jsonl` and add the model to `manifest.json`.

## 2. build_calibration.py: the calibration mix

Builds the token mix the live pass runs on: english, code and agent traces,
each to its own budget, one JSON line per document. Needs `datasets`, so it
pulls from the Hub on first run.

```bash
python3 build_calibration.py --out data/calibration.jsonl \
        --budget-en 1600000 --budget-code 1200000 --budget-agent 1200000
```

Domains are kept separate all the way through, which is what makes the per
domain curves in the flow card and the neuron fingerprints possible. Swap the
sources inside the script if your model is not an english/code/agent model.

## 3. run_capture.py: FFN activations

Runs the checkpoint over the calibration mix with hooks on every FFN and writes
per-neuron firing statistics plus layer input/output cosines. This is what
feeds the neuron card and the `io_cos` band.

```bash
python3 run_capture.py --mode selftest              # CPU, random weights, no download
python3 run_capture.py --mode gpu --data data/calibration.jsonl
```

Start with `selftest`: it builds a 2-layer model from config with random
weights and walks the entire hook path, so a broken forward signature surfaces
in seconds instead of after the model loads. Output lands in `out/`:
`neuron_stats.parquet`, `layer_io_sim.parquet`, `capture_meta.json`,
`coact_sketch.npz`.

`carve_hooks.py` holds the collector itself and is imported, not run.

## 4. atlas_live.py: everything else that happens at runtime

The main live capture. Five modes, or `all` to run them in sequence:

| Mode | What it measures | Card it fills |
|---|---|---|
| `flow` | residual RMS and per-layer Δ by domain, activation SQNR at 12 sites | signal flow, activations vs quantization |
| `linattn` | write gate β, state RMS, forget gates | linear-attention memories |
| `attn` | entropy, sink share, distance decay, output gates | how the attention layers read |
| `showcase` | attention matrices on a demo paragraph, plus an image pass | real attention maps, the vision card |
| `frag` | KL divergence from quantizing one layer to INT4 | what actually breaks |

```bash
python3 atlas_live.py --mode selftest                       # no GPU, no weights
python3 atlas_live.py --mode all --data data/calibration.jsonl
```

`--flow-bins`, `--la-bins` and `--attn-bins` control how much of the
calibration mix each mode sees; the defaults are 225, 225 and 75 bins of 2048
tokens. `attn` and `showcase` run in eager mode with a custom attention
implementation registered into the transformers attention interface, because
the maps have to be materialized. On an RTX PRO 6000 the sequence takes a few
minutes; `attn` is the slow one at batch size 1.

The vision pass wants an image named `atlas_shot.png` next to the script. Any
image works, but the card is captioned as the model looking at its own atlas,
so a screenshot of the canvas is the honest choice.

Output: `out/live/{flow,linattn,attn,frag,maps_text,maps_img}.json`.

## 5. reduce_live.py: fold it into the site

Takes whatever exists from the two capture steps, adds the per-head λ map read
straight from the `A_log` and `dt_bias` weights, and writes the two files the
canvas loads:

```bash
python3 reduce_live.py --artifacts out/live --carve out \
        --dest ../public/models/<slug>/live.json
```

`live.json` stays small, around 37 KB, because the attention matrices go to
`attn_maps.json` beside it and the card fetches them only when someone opens
it. The upper triangle of each matrix is dropped: it is empty by causality.

Every input is optional. Missing `neuron_stats.parquet` means no neuron card,
missing `attn.json` means no attention card, and so on. The script says which
inputs it skipped, and the canvas hides the cards it has no data for.

## Reproducing the published atlas

```bash
export ATLAS_MODEL_DIR=/models/Qwen3.8-27B
python3 weight_atlas.py $ATLAS_MODEL_DIR ../public/models/qwen3.8-27b/atlas.jsonl --device cuda
python3 build_calibration.py --out data/calibration.jsonl
python3 run_capture.py --mode gpu --data data/calibration.jsonl
python3 atlas_live.py --mode all --data data/calibration.jsonl
python3 reduce_live.py --artifacts out/live --carve out \
        --dest ../public/models/qwen3.8-27b/live.json
```

Needs about 60 GB of VRAM for a 27B model in bf16, or several cards with
`device_map="balanced"`, which both capture scripts use.

Numbers on the canvas come from these runs and nowhere else. If a card cannot
compute something it says so instead of showing a plausible value.

## 6. GLM-5.3-Flash NVFP4: reduce the preserved capture

GLM uses a separate reducer because its evidence is a deployed NVFP4 MoE
capture with a different structure. From the repository root:

```bash
python3 scripts/build_glm_atlas.py
```

The reducer reads `evidence/glm-5.3-flash-nvfp4/`, verifies the source shapes
while loading them, and writes two public artifacts:

- `atlas.jsonl` — 592 config-derived logical weight groups for the shared
  architecture wall
- `insights.json` — aggregated routing, exact REAP, contribution, memory,
  quantization, Vision and pruning views, plus SHA-256 source checksums

It publishes no prompts, generations, images, activations or raw routes. The
`FC2 QDQ` values are activation error captured with the deployed checkpoint's
own scale.
