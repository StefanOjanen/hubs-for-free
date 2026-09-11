# Preregistration 9: the sink-profile generative model (plan T4.3)

Date: 2026-09-11. Frozen by commit and pushed to the public repository
before execution; the push timestamp is the ex-ante evidence. Runs locally
(Apple MPS for the forward passes; statistics in float64 on the CPU).

## Statement under test

At the resolution of the interaction statistics of this project, a layer's
heads are their sink-column profiles plus noise: rebuilding each head from
its real sink column alone (the entries of G_h in the empirical sink column
c and their skew mirror, shared part included) and replacing everything
else by a random skew matrix of the remaining norm with zero entries in
that column and row reproduces, layer by layer, the per-pair z against the
real layer's marginal-matched reference and the rank-one correlation r1.

## Development calibration (before freezing; labeled)

Four constructions on Qwen2.5-0.5B (8 windows, T = 64, 6 rebuilds):
dense (a_h S plus a random remainder orthogonal to S), structured (adding
the previous-token and uniform operators), sparse (matched concentration on
a random support) and sink-profile. Only the last reproduces the z profile:
Spearman 0.95 with the real per-layer z and median absolute error 1.1
z-units, against -0.32 and 18.7 for the dense rebuild; r1 Spearman 0.86
(dense 0.81). The diagnostic behind it: at high-sink layers a median 54
percent of each head's deviation energy lies in the sink column (3 percent
for random deviations). Qwen2.5-0.5B is excluded from the confirmatory
set.

## Models and protocol

The twelve other models of the rerun. 12 WikiText-103 validation windows,
stride 40, T = 64; empirical sink column with the 16-contributing-rows
restriction; 8 plain draws for the z reference; 6 rebuilds per window; per
layer, medians over windows and rebuilds (mean for sink mass and profile
share). Implementation `sinkprofile_round.py`, evaluation
`eval_sinkprofile.py`, both committed with this file. Models run in the
order of the rerun as their weights are available locally.

## Registered predictions

- P1 (primary): per model, Spearman between the real and the sink-profile
  rebuilt per-layer z is at least 0.7; holds in at least 2/3 of the twelve
  models.
- P2 (primary): per model, the median absolute error of the rebuilt z is
  at most 3 z-units; holds in at least 2/3 of models. The dense rebuild's
  Spearman and error are reported alongside as the reference the earlier
  constructions set.
- P3 (secondary): per model, Spearman between real and rebuilt r1 is at
  least 0.6 in at least 2/3 of models.
- P4 (secondary): pooled over all layers of all models, Spearman between
  real and rebuilt z is at least 0.8.
- Reported without threshold: the sink-column share of generator energy at
  high-sink layers per model.

## Falsification

P1 or P2 failing means the sink-column profile is not the whole structured
deviation at this resolution for those models, and the statement is
restricted to the models where it holds. Both passing turns the
"measured structure" of the earlier rounds into a generative statement:
the coupling geometry of a layer follows from its heads' sink profiles and
nothing else the statistics can see. Every value is reported whichever
way it falls; post-hoc analyses are labeled.

## Artifacts

`sinkprofile/<model>.json` per model, `sinkprofile_results.json` from
`eval_sinkprofile.py`, `toy_sinkprofile_dev.json` (calibration).
