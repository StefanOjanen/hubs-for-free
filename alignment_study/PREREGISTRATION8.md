# Preregistration 8: deriving the shared energy from per-head sink profiles

Date: 2026-09-11. Frozen by commit and pushed to the public repository
before execution; the push timestamp is the ex-ante evidence. Plan item
T4.2 and gate G3.

## Derivation

For causal attention with sink column c, the ideal unit sink generator is
S_c = (A_sink - A_sink^T) / 2 normalized, where A_sink puts all mass of
rows i > c on column c. For a head's generator G_h = (A_h - A_h^T) / 2,
causality zeroes the entries A_h[c, i] for i > c, so

    <G_h, S_c> = sum_{i > c} A_h[i, c] / sqrt(2 (T - 1 - c)),

and the energy of G_h along the ideal sink direction is
e_h = m_h^2 (T - 1 - c) / 2 with m_h the head's mean column-c mass over
rows i > c. With ||G_h||^2 = (1/2) sum_i (IPR_i - A_h,ii^2) (Proposition 2
of paper.md), the derived per-head fraction is e_h / ||G_h||^2, and the
derived layer value is its mean over heads. It uses each head's sink mass
and row sharpness and no cross-head information. The measured quantity is
the shared-energy fraction of rounds 2 to 6: the mean over heads of the
energy on the layer's top principal operator, which is computed from all
heads jointly. Because the principal operator maximizes the joint energy,
the derived value is expected to sit at or below the measured one.

Two further predictors are reported. Column energy,
(1/2) sum_{i > c} A_h[i, c]^2 / ||G_h||^2, bounds the energy any single
direction inside column c can carry. The two-operator predictor adds the
energy along the uniform causal operator (A_ij = 1/(i+1) for j <= i)
orthogonalized against S_c; the k-mode calibration
(`kmode_dev_calibration.json`) found that operator to be the shared mode of
untrained and low-sink layers.

## Development calibration (before freezing; labeled)

`derivation_dev.py` -> `derivation_dev.json`, Qwen2.5-0.5B, 12 windows,
T = 64. Over the 16 high-sink layers the ideal predictor tracks the
measured shared energy with linear R^2 0.95 and Spearman 0.98, sitting
below it at every layer (median gap 0.07, slope 0.62, intercept 0.35).
Over all 24 layers R^2 is 0.71 and Spearman 0.91; the gap opens at the
low-sink layers (0, 1, 2, 22, 23), where the measured shared energy of
0.35 to 0.83 is carried by the uniform or recency operator and the
sink-only predictor gives 0.03 to 0.22. The two-operator predictor
narrows the all-layer median absolute error from 0.11 to 0.04 and raises
the identity R^2 from -2.3 to 0.16. Qwen2.5-0.5B is excluded from the
confirmatory set.

## Models and protocol

The twelve other models of the rerun (gpt2, gpt2-medium, Pythia-160m,
Pythia-410m, TinyLlama-1.1B, Qwen2.5-1.5B base and Instruct, Qwen2.5-3B,
Phi-3-mini, Mistral-7B, Qwen2.5-7B, OLMo-2-7B). 12 WikiText-103
validation windows, stride 40, T = 64, per-layer means over windows;
empirical sink column with the 16-contributing-rows restriction; high-sink
layer: sink mass > 0.4. Implementation `derivation_round.py` (imports
`derivation_dev.py`), evaluation `eval_derivation.py`, both committed with
this file. Models run as their weights are available locally; the run
order and dates are recorded in RUNLOG.md.

## Registered predictions

- G1 (gate G3; primary): pooled over the high-sink layers of the twelve
  models, the linear R^2 between the ideal-sink derived value and the
  measured shared energy is at least 0.8, with a bootstrap 95 percent
  interval over layers reported.
- G2 (primary): pooled over all layers of the twelve models, Spearman
  between the derived value and the measured shared energy is at least
  0.8.
- G3 (secondary): the derived value is at or below the measured value in
  at least 90 percent of high-sink layers.
- G4 (secondary): the two-operator predictor has a pooled all-layer median
  absolute error below 0.05 and improves the identity R^2 over the
  sink-only predictor in at least 2/3 of the models.

## Falsification and reporting

G1 passing switches the mechanism chapter's framing to derived structure:
at high-sink layers the shared energy is a function of per-head sink mass
and row sharpness, not a discovered cross-head quantity. G1 failing keeps
the "discovered structure" framing and reports what the per-head profile
leaves unexplained. G2 failing means the ordering of layers by shared
energy is not recoverable from sink profiles either. Every pooled and
per-model value is reported whichever way it falls; post-hoc analyses are
labeled.

## Artifacts

`derivation/<model>.json` per model, `derivation_results.json` from
`eval_derivation.py`, `derivation_dev.json` (calibration).
