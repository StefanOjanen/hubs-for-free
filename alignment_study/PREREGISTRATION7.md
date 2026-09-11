# Preregistration 7: head-merging tolerance (the practical test, plan T5.1)

Date: 2026-09-11. Frozen by commit and pushed to the public repository
before execution; the push timestamp is the ex-ante evidence. Runs locally
on Apple MPS (float32 for Qwen2.5-1.5B and 3B, bfloat16 for Qwen2.5-7B).

## Question

The shared-operator findings say that a layer's heads are, to first order,
multiples of one operator plus a residual. If that description has
operational content, two heads whose skew generators are nearly parallel
should be mergeable at little cost, and layers whose heads share more
energy should tolerate merging better. This round tests both statements
on held-out perplexity. Either outcome is reported.

## Development calibration (before freezing; labeled artifact)

`merge_dev_calibration.py` -> `merge_dev_calibration.json` on Qwen2.5-0.5B
(24 layers, 14 heads in 2 KV groups; 24 evaluation windows of 512 tokens):
merging the top-cosine pair of a layer cost a median 0.0013 nats per token
against 0.0025 for a random within-group pair and 0.0087 for the
bottom-cosine pair; the top merge was cheaper than the bottom merge in 23
of 24 layers and cheaper than the random merge in 17 of 24; Spearman
between the layer's shared energy and the top-merge cost was -0.32 (-0.45
with depth partialled out). Layer 0 was the exception (top merge 0.044
nats). The plan's original expectation of Spearman 0.5 for the layer law
is not supported by the calibration; the registered effect size below is
0.3. Qwen2.5-0.5B is excluded from the confirmatory set.

## Models

Qwen/Qwen2.5-1.5B (28 layers, 12 heads, 2 KV groups), Qwen/Qwen2.5-3B (36
layers, 16 heads, 2 KV groups), Qwen/Qwen2.5-7B (28 layers, 28 heads, 4 KV
groups). Same family throughout, which is a stated limitation; the merge
operation is defined for grouped-query attention and these are the three
sizes the plan names.

## Protocol

Merge operation for query heads i and j in the same KV group of one layer:
both heads' q_proj weight rows and biases are replaced by their average and
both o_proj column blocks by their average, so the two heads compute one
attention pattern whose value output is projected by W_o,i + W_o,j; all
other weights untouched; weights restored after each evaluation
(restoration verified by recomputing the baseline). Pair selection per
layer from generator cosines cos(G_i, G_j) averaged over 16 selection
windows (WikiText-103 validation texts of at least 256 tokens, T = 256):
the top-cosine pair, the bottom-cosine pair, one seeded random pair, all
within a KV group. Evaluation: mean next-token NLL in nats per token over
32 windows of 512 tokens cut from the concatenated WikiText-103 test split
(16,384 tokens), disjoint from selection; dloss = merged minus unmerged.
Predictor: the layer's shared-energy fraction from the committed rerun
(`rerun/<model>.json`, protocol A, T = 64, 48 windows), which used
different windows from both sets here. Secondary pair sweep on
Qwen2.5-1.5B: every within-group pair (30 per layer) in layers 3, 8, 13,
18, 23 and 27. Implementation `merge_round.py` (imports the calibration
script's merge, selection and evaluation code), evaluation
`eval_merge.py`, both committed with this file.

## Registered predictions

- M1 (the generator cosine is a merge criterion; primary): per model, the
  top-cosine merge costs less than the bottom-cosine merge in at least 2/3
  of layers; holds in 3 of 3 models.
- M1b (secondary): the top-cosine merge costs less than the random
  within-group merge in more than half of the layers, per model.
- M2 (shared energy predicts tolerance; primary): pooled over the 92 layers
  of the three models, Spearman(shared energy, top-merge cost) <= -0.3 with
  a bootstrap 95 percent interval over layers that excludes zero, and the
  per-model Spearman negative in 3 of 3 models.
- M2b (secondary): the pooled Spearman with layer index partialled out of
  both variables is <= -0.3.
- M3 (secondary, Qwen2.5-1.5B only): over the 180 pairs of the pair sweep,
  Spearman(pair cosine, merge cost) <= -0.5.
- Reported without a threshold: median top-merge cost per model with a
  paired bootstrap interval over evaluation windows, and the fraction of
  layers whose top merge costs under 0.002 nats per token (about 0.2
  percent perplexity).

## Falsification and reporting

M1 failing means the attention-only cosine does not identify mergeable
pairs and the practical section of the paper is dropped to a paragraph.
M2 failing means shared energy is a description of attention geometry
without a tolerance consequence, which the README already states as the
status until this test; the section then reports the negative result. M2
passing with M2b failing is reported as a depth effect. The pooled
interval and every per-model value are reported whichever way they fall.
Post-hoc analyses are labeled as such.

## Artifacts

`merge/<model>.json` per model (per-layer costs with per-window arrays,
pair sweep), `merge_results.json` from `eval_merge.py`, run notes in
RUNLOG.md.
