# Capture 06 — paired causal Vision controls

## Status

Complete and copied from the provider on 2026-09-02. The local artifact parses,
matches the provider SHA-256, contains the expected four behavioural arms, and
the provider had no GPU compute process after model shutdown.

## What this is

A small matched intervention over 40 real Vision questions: 10 each from
ChartQA, DocVQA, MME, and OCRBench. Every question was evaluated four ways in
the same loaded TP4 model instance:

- its matching real image;
- a neutral gray image with exactly the same pixel dimensions;
- the next real image from the same benchmark domain;
- no image (a secondary control because token geometry changes).

There were separate phases for internal prefill capture and answer generation.
The internal phase used one generated token so decode could not dominate the
prefill statistics. Atlas hooks were disabled for the behavioural phase.
Prompts, references, generations, images, hidden states, and per-token routes
were never persisted.

## Main verified facts

- 40 matched records per arm, 160 prefill prompts and 160 generation prompts;
  wall time 636.94 seconds including model load and shutdown.
- Reference containment was 57.5% with the matching image, versus 5.0% with
  the same-geometry blank, 10.0% with a mismatched real image, and 10.0% with
  text only.
- Against the strongest geometry-controlled blank arm, original images won
  containment on 22/40 pairs, tied on 17, and lost on 1; mean containment
  delta was +52.5 percentage points.
- DocVQA showed the clearest dependence: 10/10 references were contained with
  the real image and 0/10 with either blank or text-only input. OCRBench was
  7/10 with the real image and 0/10 in every control arm.
- The original-vs-mismatched containment delta was +47.5 points overall.
  ChartQA was +30 points, DocVQA +90, OCRBench +70, while the tiny MME slice
  was tied and therefore inconclusive.
- The internal artifact contains 37,266 finalized scalar metrics and 136
  per-head KDA groups. Pooled across the four domains, every image-bearing arm
  covers all 45 language-flow layers and all 24 Vision blocks; text-only covers
  all 45 language-flow layers and correctly has no Vision-block observations.
- A weighted pooled diagnostic at language layer 44 measured output RMS 0.674
  for original images, 0.571 for same-geometry blanks, and 0.670 for mismatched
  images. This is an activation-distribution observation, not a quality score.

## Provenance and integrity

- Artifact: `paired-vision.json`
- Artifact SHA-256: `1f9ee224ba3d6d5cc4680482a7bbf225e5e41f999170a0e3a1f7dbcfff4894d7`
- Preserved runtime log: `run.log`
- Runtime log SHA-256: `e8fb6b39e7843354134628fb547fbcb9ff330c192188bbe5d480bbbaa9995397`
- Dataset SHA-256: `e2e62cd590838ae817a503e356e2f8ba7f9c1dcfcebf0759e751a3c0302ecc17`
- Checkpoint config SHA-256: `29c9f4171196910e99b9c069d6b76c56e3cdcd0f436dc1bacbc9513c9a7529ac`
- Checkpoint index SHA-256: `015faae91e8189c7553f1d48ec3d0694b8c02b282d7f58af2d7b4064a81ce4c0`
- Runtime: vLLM `0.1.dev20051+g487ecf187`, Torch `2.13.0+cu130`,
  Transformers `5.15.1`; native SM120 sparse MLA and NVFP4 MoE.
- Provider test suite before run: 39/39 passed.

## Validation

The output was first copied to a temporary local directory, JSON-parsed, and
compared against the provider hash before being placed here. The artifact has
all 16 arm/domain behavioural cells with 10 records each, explicit privacy
flags, and no collector errors. Pooled internal layer coverage was checked arm
by arm. The GPU process table was empty after shutdown.

## Boundaries

- This is a 40-question causal diagnostic, not a leaderboard-quality Vision
  benchmark or a claim about general multimodal quality.
- Exact match is 0% in every arm because most generations are explanatory and
  36/40 original generations reached the 64-token cap. Reference containment
  is the useful behavioural metric here; token F1 is diluted by the extra
  prose and should not be read as an ordinary VQA score.
- Blank images preserve pixel dimensions. Mismatched real images preserve the
  source domain but not necessarily the exact dimensions. Text-only changes
  multimodal token geometry.
- Small deterministic activation samples can omit an individual metric in one
  10-row domain cell. Coverage claims above are pooled across the four domains
  within each arm, where the expected layer sets are complete.
- All results describe the deployed NVFP4 checkpoint with FP8 KV cache. No
  matching BF16 checkpoint exists, so this does not measure quantization loss.

