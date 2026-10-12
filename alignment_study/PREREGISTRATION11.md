# Preregistration 11: attention weighted by what it moves (the functional round)

Date: 2026-10-12. Frozen by commit and pushed to the public repository
before the confirmatory run; the push timestamp is the ex-ante evidence.
Runs locally on Apple MPS under the precision policy of
`hubsfree/adapters.py` (float32 for the two small models, bfloat16 for the
two 7B models).

## Why this round

Every earlier round compared statistics of attention maps with the same
statistics on surrogate maps, so every result is a statement about map
statistics. The audits and the sink-profile model found that most
head-similarity statistics measure the shared sink column. Whether that
matters for what heads do was never measured. This round asks three
questions with function in the loop: is the sink functionally smaller than
its attention share; does weighting attention by the vectors it moves change
how much of head similarity the sink accounts for; and which way of comparing
two heads predicts the damage of merging them. As promised in the
manuscript (Section 5), every clause is an effect size with an interval, and
percentiles are not used as survival criteria.

## Development calibration (labeled, before freezing)

`functional_dev.py` on Qwen2.5-0.5B, the calibration model of rounds 6 and
7, excluded from the confirmatory set (`functional_dev_calibration.json`,
`functional_dev.log`): 24 layers, 12 high-sink layers, eight merged layers
with all 42 within-group pairs each, 4,096 evaluation tokens.

- F1: in high-sink layers the value vectors carried from the sink columns
  are about a sixth the size of the other positions' (median 0.16); the
  sink's share of the value-norm-weighted attention is 0.42 of its raw
  attention share (raw 0.54, weighted 0.22). The norm of the sink's
  contribution to a head's output is still 0.46 of the output norm, because
  the other positions' contributions partly cancel.
- F2: the sink-column null reproduces 0.96 of head similarity on raw maps
  and 0.47 on value-weighted maps (medians over high-sink layers).
- F3, mean within-layer Spearman between a predictor (oriented as a
  distance) and the merge KL: generator cosine 0.33, raw map distance
  0.61, map distance outside the sink columns 0.59, value-weighted distance
  0.67, first-order output perturbation 0.83, sink-mass difference 0.23.
  Differences: output perturbation minus raw map distance +0.22; raw map
  distance minus generator cosine +0.27; sink-excluded minus raw map
  distance -0.02 (interval over layers -0.05 to 0.01). The KL ranking of
  pairs within a layer is reliable (median split-half Spearman 0.95).
- F4: the pair the output perturbation picks costs a median 0.50 of the
  pair the generator cosine picks (better in 7 of 8 layers) and 0.88 of the
  pair the raw map distance picks (never worse in 8 layers).

Changed hypothesis, recorded before freezing: the proposal that motivated
this round was that removing the sink columns would make map similarity
predict function. Development contradicts it. The registered expectation
for that comparison (F3c) is therefore no gain, and the positive
predictions concern the output perturbation, which brings in the values and
output weights.

The output perturbation is a first-order approximation of the merge's
effect on the layer output, computed on selection windows disjoint from
the evaluation windows. Its correlation with merge damage is partly
expected by construction. The registered claims are the size of its
advantage over the map-based comparisons the literature uses and the cost of
the merges it chooses, not the existence of a correlation.

## Models

Held out from development, all with grouped-query attention so the merge
of round 7 is defined: Qwen/Qwen2.5-1.5B, TinyLlama/TinyLlama_v1.1,
mistralai/Mistral-7B-v0.1, Qwen/Qwen2.5-7B. Qwen2.5-1.5B and Qwen2.5-7B
were in round 7; the outcome here is newly measured (KL on new evaluation
windows) on new pairs, so no round-7 outcome enters this round.

## Protocol

Code: `alignment_study/functional_common.py` (instruments),
`functional_round.py` (runner and merge plan), `eval_functional.py`
(evaluation), committed with this file.

- Selection windows: 16 WikiText-103 validation texts of at least 256
  tokens, T = 256 (round 7's set). Evaluation windows: WikiText-103 test,
  concatenated and cut into 512-token windows, 8 windows for the two small
  models and 4 for the two 7B models (round 7's picker).
- Sink set S per layer and window: `hubsfree.stats.sink_columns`. Rows
  below 16 are excluded from every map measure. High-sink layer: mean raw
  sink share over heads above 0.4.
- F1 per head: raw sink share (attention mass on S); value-norm share
  (Kobayashi et al. 2020: share of sum over c of a_rc |W_O,h v_c| that falls
  on S); output share (norm of the S columns' contribution to the head
  output over the norm of the output); |W_O,h v_c| on S over the mean
  elsewhere.
- F2 per layer: mean pairwise cosine of the heads' flattened maps, raw and
  value-norm-weighted, against a per-row permutation within the causal
  support (diagonal kept) and the same permutation with S kept, 10 draws
  on 8 windows; reproduced share = (sink-kept null minus permutation) over
  (real minus permutation).
- F3 predictors per within-group pair (i, j), with D = a_i - a_j:
  gen_cos (cosine of the generators, round 7's criterion), tv_raw (half
  the L1 norm of D per row), tv_nosink (the same outside S), val_dist
  (|D_r V_g|), out_pert (|(1/2) D_r V_g (W_O,i - W_O,j)^T|), sink_diff
  (|sum over S of D_r|), each averaged over rows and windows.
- Merge (round 7): both heads' q_proj rows and biases replaced by their
  average and both o_proj blocks by their average; weights restored after
  each evaluation. Outcome: KL of the merged model's next-token
  distribution from the unmerged one over the unmerged model's top 256
  tokens plus a remainder bucket, mean over evaluation tokens; the change
  in NLL is recorded alongside.
- Merge plan, fixed in `functional_round.py`: Qwen2.5-1.5B layers 1, 5, 9,
  13, 17, 21, 24, 27, all 30 pairs; TinyLlama layers 1, 4, 7, 10, 13, 16,
  19, 21, 48 seeded pairs of 112; Mistral-7B layers 2, 8, 14, 20, 26, 31,
  all 48 pairs; Qwen2.5-7B layers 2, 8, 14, 20, 26, 48 seeded pairs of 84.

## Registered predictions

Pooled values with two-stage cluster bootstrap 95 percent intervals
(models resampled, then layers within a model; 2,000 draws, seed 0).
Within-layer Spearman correlations are between a predictor oriented as a
distance and the merge KL over the layer's pairs.

- Gate: median split-half Spearman of the within-layer KL ranking at least
  0.7. If it fails, F3 and F4 are reported as underpowered.
- F1: over high-sink layers, the median of (value-norm sink share) over
  (raw sink share) is at most 0.6 and its interval's upper bound at most
  0.75.
- F2: over high-sink layers, the mean drop of the sink-column null's
  reproduced share of head similarity, raw maps to value-weighted maps, is
  at least 0.25 with an interval above zero.
- F3a: mean within-layer Spearman of out_pert minus that of tv_raw is at
  least 0.10 with an interval above zero.
- F3b: tv_raw minus gen_cos is at least 0.10 with an interval above zero.
- F3c: tv_nosink minus tv_raw has an interval whose upper bound is below
  0.05 (removing the sink columns does not improve the map comparison).
- F3d: the mean within-layer Spearman of out_pert is at least 0.6 with an
  interval whose lower bound is at least 0.4.
- F4: over merged layers, the median of KL(pair chosen by out_pert) over
  KL(pair chosen by gen_cos) is at most 0.8 with an interval whose upper
  bound is below 1.
- Reported without thresholds: every predictor's within-layer Spearman
  with KL and with the change in NLL; val_dist minus tv_raw; out_pert minus
  val_dist; pick ratios out_pert over tv_raw and tv_raw over gen_cos; the
  pick's rank percentile per predictor; F1's output share and value-norm
  ratio; every per-model value.

## Falsification and reporting

- F1 or F2 failing: in these models the sink is not functionally smaller
  than its attention share, and the reading that map statistics overweight a
  functionally minor component is withdrawn.
- F3a or F3d failing: bringing in values and output weights does not
  predict merge damage better than the map, and lifting the map-only
  assumption buys no prediction.
- F3b failing: the commutator-based object is not a worse predictor than
  the plain map.
- F3c failing: removing the sink does improve the map comparison, the
  development result did not hold, and the original hypothesis gains
  support.
- F4 failing: the practical criterion is not established.

Every clause and every per-model value is reported whichever way it falls;
post-hoc analyses are labeled as such.

## Artifacts

`functional/<model>.json` per model as it completes, `functional_results.json`
from `eval_functional.py`, run notes in `RUNLOG.md`.
