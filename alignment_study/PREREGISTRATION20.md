# Preregistration 20: a benchmark of head-merging criteria against merge damage

Date: 2026-10-15. Frozen by commit and pushed before the confirmatory run.

## Question

Methods that share keys and values between attention heads, or convert
multi-head to grouped-query attention, choose which heads to merge by some
similarity: of attention scores (CHAI), of query and key weights (head-wise
shareable attention), of key and value activations or weights (AsymGQA,
DHA). Rounds 11 and 12 found that attention-map statistics predict the
damage of merging query heads weakly in grouped-query models. Nobody has
measured how well these criteria predict the damage of the merge such
methods perform. This round measures it on multi-head models and releases
the outcomes as a benchmark.

## Instrument (`alignment_study/merge_bench.py`)

Models: gpt2-medium, EleutherAI/pythia-160m and EleutherAI/pythia-410m.
Merge: two heads' key projections (weights and bias) replaced by their mean,
and likewise their value projections (mean pooling, as in Ainslie et al.
2023); queries and outputs unchanged. Damage: KL of the next-token
distribution on 4 WikiText-103 test windows of 512 tokens (top 256 plus
remainder). Every pair of heads in every layer is merged on its own; then,
per criterion, one and then two disjoint pairs per layer in layers 1 to L-1
are merged at once, chosen greedily by the criterion, against five random
draws. Criteria on 8 WikiText-103 validation windows of 256 tokens, from a
custom attention function that records post-rotary queries, keys, values and
weights (checked equal to eager attention): last-row and every-row Pearson
correlation of attention (CHAI's two readings), Jensen-Shannon divergence
(Clark et al.), map cosine, cosine of the concatenated query and key weights
and of the concatenated key and value weights, mean cosine of key vectors and
of value vectors over tokens (these four follow the families of criteria
named above, not those methods' exact pipelines), and the output
perturbation: the merge simulated on the selection windows, the mean norm of
the change in the two heads' summed contribution to the residual stream.

## Development (labeled, `alignment_study/merge_bench_dev/`)

gpt2. Capture attention equals eager attention (maximum logit difference
0.0); restore check 0. Within-layer Spearman with the single-merge KL: output
perturbation 0.67 (interval 0.59 to 0.75); Jensen-Shannon 0.15; value
cosine 0.07; map cosine 0.05; key cosine 0.05; last-row correlation 0.02;
key-value weight cosine 0.02; every-row correlation 0.01; query-key weight
cosine -0.01. One pair per layer: KL 0.110 for the output perturbation's
choice, 0.183 for the last-row correlation's (ratio 0.60), random median
0.166. Two pairs per layer: 0.313, 0.358 and 0.385. A first development
budget of a quarter of the heads per layer in every layer gave KL of 0.6 to
2.4 for every criterion including random pairs and was replaced by round 12's
budgets (`merge_bench_dev/gpt2_quarter_budget_discarded.json`, not used).

## Registered predictions

Two-stage cluster bootstrap (models, then layers; 2,000 draws).
- B1: the output perturbation's mean within-layer Spearman with merge damage
  is at least 0.4, interval above zero.
- B2: it exceeds the best other criterion's by at least 0.2, interval above
  zero.
- B3: every other criterion's mean within-layer Spearman is below 0.3.
- B4: at one pair per layer, the output perturbation's merges cost less than
  the last-row correlation's (the CHAI-style choice) in every model, with a
  geometric mean ratio of at most 0.8.
- B5: at one pair per layer, they cost less than the median of five random
  choices in every model.
- Reported: two pairs per layer, the change in NLL, per-model values, and the
  full table of pair outcomes and criteria (the benchmark).

## Falsification and reporting

B1 or B2 failing means the simulated merge does not predict merge damage
better than the similarities in use; B3 failing means some similarity in use
does predict it; B4 or B5 failing means the better predictor does not choose
cheaper merges at the smallest budget. Every clause is reported whichever way
it falls (`alignment_study/eval_merge_bench.py` writes
`alignment_study/merge_bench_results.json`).
