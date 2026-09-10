# Preregistration 5: training dynamics (Pythia checkpoints)

Date: 2026-09-10. Frozen by commit and pushed to the public repository
before any checkpoint is downloaded; the push timestamp is the ex-ante
evidence. Runs locally on Apple MPS in float32 (RUNLOG.md, local platform
entry; both models are far below the float32 budget, and MPS float32
reproduces the CPU protocol to four decimals).

## Question

The converged-model rounds established that a layer's heads share one
dominant operator component, identified with the sink operator at
high-sink layers, and that the shared-energy fraction predicts the
interaction geometry. Training checkpoints ask how that state is reached.

## Development calibration (before freezing, outside the registered set)

Random-init Pythia-160m and Pythia-410m from their configs (torch seed 0,
`dynamics_dev_calibration.py` -> `dynamics_dev_calibration.json`) show
that an untrained network already has a dominant shared component: shared
energy 0.93 to 0.96 (160m) and 0.88 to 0.95 (410m), because near-uniform
attention gives every head the same causal operator. That component is
not the sink operator: cos(S, sink) is 0.26 to 0.28 everywhere, sink
mass 0.06, r1 0.15 to 0.47 (degenerate, the heads are near-identical),
rho 0.88 to 0.96. Consequence: a prediction that shared energy "rises"
with training would be ill-posed. The registered predictions therefore
concern the identity of the shared mode and the law, not the level of
shared energy.

## Models and checkpoints (none previously analyzed at these checkpoints)

EleutherAI/pythia-160m (12 layers, 12 heads) and EleutherAI/pythia-410m
(24 layers, 16 heads); revisions step0, step512, step1000, step2000,
step4000, step8000, step16000, step32000, step64000, step143000 (the
final checkpoint, identical to main). Twenty (model, checkpoint) cells.
The final checkpoints were measured in rounds 1 and 2; their step143000
values must reproduce those runs within window-subset variation. That is
a sanity anchor, not a prediction.

## Protocol

Identical to PREREGISTRATION2 and 3: 12 WikiText-103 validation windows
(stride 40, texts over 40 words), T = 64, per-layer statistics in float64
on the attention probabilities; empirical sink column with the
16-contributing-rows restriction; sink mass excluding row 0; 12 plain
draws (z reference), 8 sinkfix draws, 8 altsink-v2 draws; high-sink
layer: sink mass > 0.4. Statistics come from the registered implementation
(`scale_round_v2.layer_stats`); `dynamics_round.py` writes one JSON per
(model, checkpoint) as it completes. A cell is a (model, checkpoint,
layer) triple; per-layer values are aggregated over windows as in round 3
(mean for sink mass, shared energy and the invariant deviation, median
otherwise). Checkpoint order is indexed 0 to 9 in the list above.

## Registered predictions

- D0 (untrained anchor): at step0, no layer of either model is high-sink,
  and cos(S, sink) < 0.5 in every layer. This is the premise of the
  untrained-model null family; failure revises that family before any
  other outcome is reported.
- D1 (the shared mode becomes the sink operator as the sink forms):
  pooled over all cells of both models, Spearman(sink mass, cos(S, sink))
  >= 0.5.
- D1b (S1 holds along training): among high-sink cells at any checkpoint,
  cos(S, sink) > 0.7 in at least 80 percent.
- D2 (the law holds along training, not only at convergence): pooled over
  high-sink cells of both models across all checkpoints,
  Spearman(shared energy, r1_real) <= -0.5; testable if at least 30
  high-sink cells exist.
- D3 (the mode switch is locked to sink formation): for each model, among
  layers that are high-sink at step143000, let k_s be the index of the
  first checkpoint with sink mass > 0.4 and k_c the index of the first
  checkpoint with cos(S, sink) > 0.7; the layer is consistent if k_c
  exists and |k_c - k_s| <= 1. Holds if at least 2/3 of such layers are
  consistent in every model with at least 3 such layers.
- D4 (secondary, uncorrected): the first checkpoint with any high-sink
  layer is step4000 or earlier in both models.
- D5 (secondary, exploratory): shared energy dips during training. For a
  layer, dip means min over checkpoints of shared energy is below both
  the step0 and the step143000 values by more than 0.05. Expected in at
  least 2/3 of layers per model. Reported either way, uncorrected.

Thresholds: 0.5 and -0.5 are the S4/H4 thresholds; 0.7 for cos(S, sink)
and 0.4 for sink mass are the S1 and high-sink thresholds of rounds 2 and
3; 0.5 for the step0 cosine sits well above the calibrated 0.28 and well
below 0.7.

## Falsification and reporting

D1 or D1b failing means the shared component is not carried by the sink
as it forms, which undercuts the derivation planned for the mechanism
chapter (T4.2) and is reported as such. D2 failing demotes the
alignment-fraction law to converged models. D3 failing means the mode
switch and sink formation are separated in training, and the dynamics
section reports the observed lag. Bootstrap 95 percent intervals (1000
resamples over cells) accompany D1 and D2. Post-hoc analyses are labeled
as such.

## Artifacts

`dynamics_round.py` (runner) and `eval_dynamics.py` (evaluation) are
committed with this file before any download; the runner writes
`dynamics/<model>_step<k>.json` per cell, the evaluation writes
`dynamics_results.json` and prints the scorecard.
