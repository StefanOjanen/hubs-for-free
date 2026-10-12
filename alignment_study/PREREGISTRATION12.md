# Preregistration 12: the field's statistics with function in the loop

Date: 2026-10-13. Frozen by commit and pushed to the public repository
before the confirmatory run. Same platform, precision policy, selection and
evaluation windows as round 11.

## Question

Round 11 showed that attention-map comparisons predict the damage of
merging two heads weakly, and that the sink is not the reason. Its
comparisons were our own. This round tests the statistics other papers use:
for pairs of heads, the Jensen-Shannon divergence between attention rows
(Clark et al. 2019) and attention-score similarity and correlation in the
two readings of CHAI's statistic (last row; every row); for single heads,
the attention "confidence" of Voita et al. (2019), the mean largest
attention weight, which that work found to mark important heads, and row
entropy. It also retests round 11's failed practical clause (F4) where it
matters for compression: many pairs merged at once.

## Development calibration (labeled, before freezing)

`standard_round.py --dev` on Qwen2.5-0.5B (`standard_dev/`,
`standard_dev_results.json`, `standard_dev.log`; Part A uses the merge
outcomes of `functional_dev_calibration.json`):

- Part A, mean within-layer Spearman with the merge KL over 8 layers:
  output perturbation 0.83; Jensen-Shannon 0.63; map cosine 0.54; last-row
  correlation 0.47; all-rows correlation 0.48. Output perturbation minus
  each: 0.20, 0.29, 0.35, 0.35, every interval above zero.
- Part B: confidence tracks sink mass (within-layer Spearman 0.86 over the
  12 high-sink layers); against the KL of ablating each head, within a
  layer, confidence -0.14, negative entropy -0.04, sink mass -0.52, output
  norm 0.77 (24 layers, 336 heads). Pooled across all layers of the model
  the output norm falls to 0.24, because norms are not comparable between
  layers; the registered clauses are within-layer.
- Part C, all layers but the first, one merged pair per KV group (23
  layers, 46 pairs): KL 0.043 when pairs are chosen by output perturbation,
  0.095 by the generator cosine, 0.111 by the last-row correlation, 0.188
  for random pairs (median of five); output perturbation first of eight
  criteria; two pairs per group: 0.127 against 0.281, 0.257 and 0.468.

## Models and plan

Round 11's four held-out models: Qwen2.5-1.5B, TinyLlama_v1.1, Mistral-7B,
Qwen2.5-7B. Fixed in `standard_round.py`: ablation of every head in layers
1, 3, ..., 27 (Qwen2.5-1.5B), 1, 3, ..., 21 (TinyLlama), 2, 10, 18, 26
(Mistral-7B) and 2, 9, 16, 23 (Qwen2.5-7B); multi-pair merges in layers 1 to
L-1 at one and two pairs per KV group for eight criteria and five random
seeds. Evaluation: 8 windows of 512 tokens for the small models, 4 for the
7B models.

Part A reuses round 11's pair-merge outcomes of the same models. Those
outcomes are known for round 11's predictors (output perturbation 0.65);
the four field statistics have not been computed on these models before
this freeze. Parts B and C measure new outcomes.

## Registered predictions

Within-layer Spearman correlations, pooled over layers with two-stage
cluster bootstrap 95 percent intervals (models, then layers; 2,000 draws).
Part C has one value per model per budget.

- S1: for each of Jensen-Shannon, map cosine, last-row correlation and
  all-rows correlation, the output perturbation's within-layer Spearman with
  the merge KL exceeds the statistic's by at least 0.10 with an interval
  above zero. Passes if all four hold.
- S2: within high-sink layers (mean sink mass over heads above 0.4),
  confidence tracks sink mass: mean within-layer Spearman at least 0.5 with
  an interval whose lower bound is at least 0.3.
- S3: against the ablation KL, the output norm's within-layer Spearman
  exceeds confidence's (S3a) and negative entropy's (S3b) by at least 0.3,
  each with an interval above zero.
- S4: the output norm's within-layer Spearman with the ablation KL is at
  least 0.4 with an interval above zero.
- S5: at one pair per KV group, the KL of merges chosen by output
  perturbation is below that of merges chosen by the generator cosine in
  every model, with a geometric mean ratio of at most 0.8.
- S6: the same against the last-row correlation (the CHAI-style choice).
- S7: output perturbation is among the two best of the eight criteria at
  one pair per group in at least three of the four models.
- Reported without thresholds: every statistic's correlations; the sign of
  confidence and sink mass against ablation damage; cross-layer pooled
  correlations per model; the two-pair budget; random baselines; the change
  in NLL.

## Falsification and reporting

S1 failing means some statistic the field uses predicts merge damage
about as well as the output perturbation; the paper's claim narrows to the
statistics that fail. S2 failing means confidence is not a sink measure in
these models. S3 or S4 failing means output norm is not a better
importance measure than attention-based heuristics. S5 to S7 failing means
the practical criterion is not established for multi-pair merging either.
Every clause and every per-model value is reported whichever way it falls.

## Artifacts

`standard/<model>.json`, `standard_results.json` from `eval_standard.py`,
notes in RUNLOG.md.
