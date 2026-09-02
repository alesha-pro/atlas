# Capture 08 — causal REAP route-ablation arms

## Status

Complete and copied from the provider on 2026-09-02. The accepted JSON matches
the provider SHA-256, all five intervention arms passed the four-rank identity
gate over all 42 routed layers, and the provider GPU process table was empty
after shutdown. No checkpoint file was changed.

## What this is

A reversible causal sensitivity test of the rankings from the merged 12.8M-token
REAP profile. For every routed layer, expert IDs were ranked by
`reap_sum / reap_count`. The same loaded native FlashInfer CUTLASS TP4 model was
run as baseline and with five interventions:

- bottom-REAP 2%: 6 experts per layer;
- bottom-REAP 5%: 14 experts per layer;
- bottom-REAP 10%: 29 experts per layer;
- deterministic random 10%: 29 experts per layer;
- top-REAP 2%: 6 experts per layer, as a positive control.

When a selected top-8 route belonged to an arm's prune set, its router weight
was zeroed and the surviving weights were renormalized to preserve the original
total router mass. If all eight routes were masked, the strongest original
route was retained. This tests contribution sensitivity under the original
top-8 selection; it does not rewrite weights, reroute to the ninth expert, or
save compute.

## Main verified facts

- Baseline plus five arms ran on 14 records, one from every calibration domain,
  including four real-image domains; 32 generated tokens and top-20 first-step
  logprobs per record were compared in memory. Wall time was 172.8 seconds.
- Bottom 2% removed 2.46% of selected route slots and 1.74% of router mass on
  this probe. All first tokens stayed identical; 35.7% of 32-token sequences
  stayed fully identical and normalized edit similarity averaged 0.775.
- Top 2% was a stronger perturbation despite hitting only 1.36% of route slots:
  it removed 1.89% of mass, left 21.4% of sequences identical, and produced
  normalized edit similarity 0.679. First-step top-20 Jaccard was 0.522 versus
  0.654 for bottom 2%. This is direct causal evidence that REAP's extremes are
  not interchangeable on the deployed NVFP4 model.
- Bottom 5% removed 4.27% router mass; 21.4% of sequences stayed identical,
  normalized edit similarity was 0.690, and first-step top-20 Jaccard was 0.623.
- Bottom 10% removed 8.66% router mass and affected 57.9% of routed tokens per
  layer on average. Sequence identity fell to 7.1%, normalized edit similarity
  was 0.636, and first-step top-20 Jaccard was 0.624.
- Random 10% removed 9.83% router mass and affected 57.3% of routed tokens.
  Sequence identity was 14.3%, normalized edit similarity 0.632, and top-20
  Jaccard 0.497. The mixed metrics do not justify declaring bottom-10% safe;
  they show why a larger evaluation is required before real pruning.
- The all-selected fallback fired 29 times in the bottom-10% arm and zero times
  in the other arms. Those events retained the strongest route and are counted
  explicitly in the artifact.
- Tiny automatic answer scores cover only 9 referenced records. Their apparent
  +11.1-point containment change for bottom 10% is noise-prone and must not be
  called a quality improvement.

## Provenance and integrity

- Artifact: `pruning-arms.json`
- Artifact SHA-256: `4bb87fbc32ceecf41afab7fb0eb2b82fb8cbb23865176cd00817eadf35bd8024`
- Accepted runtime log: `run.log`
- Accepted log SHA-256: `a59e6c217c8f7da67becaa0ed77d8acd3b7824d9ac477fb04146f6661e566bb4`
- Preserved failed log: `failed-01-missing-reference.log`
- Failed log SHA-256: `402ee0fb6f3aa35967096f9d17cd39165c7817f5d8565e6fe9ca293b25a4f73a`
- Source REAP NPZ SHA-256: `51f4a703116169e5c2230a1af74625902d02f129e4e218aa2b579fda917a83f4`
- Dataset SHA-256: `e2e62cd590838ae817a503e356e2f8ba7f9c1dcfcebf0759e751a3c0302ecc17`
- Checkpoint config/index hashes match Captures 01–07.
- Runtime: vLLM `0.1.dev20051+g487ecf187`, Torch `2.13.0+cu130`,
  Transformers `5.15.1`; native `FLASHINFER_CUTLASS` NVFP4 MoE.
- Provider test suite before accepted run: 43/43 passed.

## Validation and failed evidence

Before loading the model, local and provider REAP NPZ hashes were compared and
matched. The first attempt stopped before inference because one prompt-only
calibration row had no assistant reference; its log was copied locally. The
selector was corrected to allow prompt-only rows for sequence comparison while
scoring references only when present. The fresh run completed every arm.

For each arm, integer intervention counters had to be bit-identical on all four
TP ranks and router-mass counters had to match within strict floating tolerance.
The accepted artifact records rank-0 values only after that gate. Every arm has
all 42 routed layers and its exact per-layer expert sets for reproduction.

## Boundaries

- This masks contributions among the already selected top eight. A deployable
  pruned model should remove experts and reroute to the next available experts;
  its behavior and speed can differ.
- The intervention still executes zero-weight routes in the fused kernel, so
  these wall times say nothing about pruning speedups.
- Fourteen records are enough for a sensitivity probe, not for a pruning go/no-go
  decision or a general quality score. A real pruning candidate needs a much
  larger held-out evaluation and a physically pruned/rerouted checkpoint.
- The lists and causal effects apply to this deployed NVFP4 checkpoint. No BF16
  baseline exists.
- No prompts, generated text, token sequences, hidden states, or per-token
  routes are stored; only aggregate comparisons and reproducible expert sets.

