# Preregistration 3: scale round (3B to 7B, GPU)

Date: 2026-08-19. Frozen and pushed to the public repository before
execution; the push timestamp is the ex-ante evidence. Runs on a Colab
GPU runtime.

## Models (none previously analyzed in this project)

Qwen/Qwen2.5-3B, microsoft/Phi-3-mini-4k-instruct (3.8B),
mistralai/Mistral-7B-v0.1, Qwen/Qwen2.5-7B, allenai/OLMo-2-1124-7B.
A model that cannot load or run on the available runtime is recorded as
untestable, not failed. A model is testable for S1-S3 if it has at least
3 high-sink layers.

## Protocol

Identical to PREREGISTRATION2: 12 WikiText-103 validation windows
(stride 40, texts over 40 words), T = 64, per-layer statistics in
float64 on the attention probabilities; empirical sink column with the
16-contributing-rows restriction; sink mass excluding row 0; 12 plain
draws (z reference), 8 sinkfix draws, 8 altsink-v2 draws; high-sink
layer: sink mass > 0.4; distance ratios R_r1 and R_z as defined there.
One platform change: model forward passes may run in bfloat16 on GPU
with attentions cast to float64 before analysis. The plain-surrogate
reference is recomputed per run, so dtype effects enter both sides;
any residual precision effect is expected to be immaterial and the
anchor model is not rerun here.

## Registered predictions (thresholds identical to round 2)

- S1 (shared sink operator): per model, at 2/3 or more of high-sink
  layers, rho_real >= 2 x rho_plain AND |cos(S, sink)| > 0.7; holds in
  at least 2/3 of testable models.
- S2 (one-column sufficiency): per model, at 2/3 or more of high-sink
  layers, R_r1 < 0.4 AND R_z < 0.4; holds in at least 2/3 of testable
  models.
- S3 (dissociation): per model, at 2/3 or more of high-sink layers,
  r1_altsink2 > 0.8 AND rho_altsink2 < 0.5 x rho_real AND
  zmed_altsink2 > zmed_real; holds in at least 2/3 of testable models.
- S4 (alignment-fraction law): pooled over high-sink layers of all
  testable models, Spearman(sharedE, r1_real) <= -0.5.

## Falsification

S1 or S2 failing in 2 or more testable models rejects the sink-operator
claim at this scale. S3 failing leaves dissociation unconfirmed at this
scale. S4 failing demotes the law to sub-3B models. Post-hoc narrowings
will be labeled as such.

## Artifacts

`scale_round.py` (committed with this file, before execution) writes
`scale_round.json` on the runtime and prints per-model RESULTS blocks;
captured output is committed to this directory after the run.
