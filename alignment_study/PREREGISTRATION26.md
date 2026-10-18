# Preregistration 26: closed-form fusion of key and value heads for training-free grouped-query conversion

Date: 2026-10-18. Frozen by commit and pushed before the confirmatory runs.

## Question

Round 24 converted five multi-head models to grouped-query attention without
training. Every recipe averaged the key and value projections within each group
(Ainslie et al. 2023), and the best one, averaging after Jin et al.'s alignment
in groups chosen by the simulated merge, still left perplexity at 2.0 to 4.8
times the unconverted model's at half the key-value heads in four models.
Averaging is one point of a larger family: every head keeps its own query and
output projection, so a group only has to share one key projection and one value
projection, and each head's query and output projection can absorb any linear
map. The best shared pair can be computed in closed form from the function each
head computes, its query-key bilinear form and its value-output map, and the
second moment of the attention input on calibration tokens. The result does not
depend on any head's basis, so it needs no alignment. Weight-space fits with an
input-covariance metric (CAGA, Oct 2026) and activation SVDs of the key and value
caches (Yu et al. 2024) fit keys and values without regard to the queries that
read them or the output projections that write them; KQ-SVD (Lesens et al.,
AISTATS 2026) showed, for low-rank caches of single heads, that the key
projection should preserve query-key products. This round asks whether a fusion
that weights the shared key by the queries of every head in the group, and the
shared value by the output projections, converts better than round 24's best
recipe and than the same fusion without the weighting, and whether the grouping
still matters after it.

## Instrument

`alignment_study/fuse_common.py` and `alignment_study/fuse_round.py`, evaluated
by `alignment_study/eval_fuse.py`.

Fusion (per layer and group; x the attention input, augmented with a constant 1
when the projections have biases; S = E[x x^T] on calibration tokens; L = S^(1/2)):
- Values: head h adds O_h C_h x to the residual stream. The shared value
  projection C_G = P^T L^-1 and new output projections O'_h = O_h C_h L P, with P
  the top head-dimension eigenvectors of sum_h L C_h^T W_h C_h L, minimize
  sum_h E||(O_h C_h - O'_h C_G) x||^2 for W_h = O_h^T O_h (output-weighted).
- Keys, dimensions without rotary embedding: the shared key projection and new
  queries minimize the error of every head's query-key bilinear form in the
  metric of the attention input (the key side centered, since a term constant
  over keys does not change the softmax), with the key directions weighted by the
  second moment of each head's queries (query-weighted).
- Keys, rotary planes: each plane's queries and keys are complex numbers whose
  product must be kept for every pair of tokens; the shared key plane is the top
  eigenvector of the group's whitened key planes weighted by each head's query
  energy on the plane, and each query plane is multiplied by a complex number,
  which commutes with the rotary embedding.
- Unweighted: the same fit with W_h and the query second moments replaced by
  identities (the objective of weight-space and cache fits).
Groups of one head reproduce the model; the fusion gives the same converted model
in any function-preserving basis (checked in each run).

Arms (each at g = 2 and g = 4, groups from round 24's partitions):
`fuse_u/contiguous` (unweighted, neighbouring heads), `fuse/contiguous`
(weighted, neighbouring heads), `fuse/jin` (weighted, Jin et al.'s groups),
`fuse/sim_align` (weighted, the groups round 24's simulated merge chose),
`fuse_u/sim_align` (unweighted, the same groups). Reference recipes from round 24,
evaluated with the same harness on the same models: `sim_align` (aligned mean
pooling, the same groups), `jin`, `contiguous`.

Calibration: round 24's 8 C4 validation documents of 1,024 tokens. Evaluation as
in round 24 (`zeroshot.py`): WikiText-2 perplexity over 1,024-token windows and
seven zero-shot tasks with lm-evaluation-harness prompts. dlog(a) = log(perplexity
of a / perplexity of the unconverted model).

Models: facebook/opt-1.3b, EleutherAI/pythia-1.4b, gpt2-large, allenai/OLMo-1B-hf,
microsoft/phi-2, float32 on the M1 Max (round 24's models and partitions).

## Development (labeled)

Exploratory probes (scratch scripts, recorded in the run log), on the CPU:
- After round 24's mean pooling, a least-squares refit of each layer's output
  projection on calibration tokens lowered Pythia-410m's perplexity at g = 2 from
  171 to 90 with round 24's `sim_align` groups but raised it from 2,737 to 53,208
  without alignment; a refit that also targeted the drift of the residual stream
  did worse (182). Not used.
- The closed-form fusion on Pythia-410m (16 calibration documents, 24
  WikiText-2 windows), `sim_align` groups at g = 2: unweighted 91.9, query
  weighting alone 43.1, output weighting alone 98.0, both 39.2 (aligned mean
  pooling 171.1); with both, neighbouring groups 45.9 and Jin et al.'s 44.1; at
  g = 4, `sim_align` groups 115.9 and neighbouring 143.8. Second moments measured
  layer by layer on the partly converted model did slightly worse (42.3 against
  40.1 at g = 2, 129.2 against 121.3 at g = 4, 8 documents). Not used.
- Neighbouring groups at g = 2 (4 calibration documents, 8 windows), mean pooling,
  unweighted and weighted fusion: Pythia-160m 2,589, 213, 94.7 (base 33.6);
  OPT-125m 1,505, 93.9, 87.3 (35.1); Llama-160m 873, 116, 79.9 (27.6); GPT-2
  704, 107, 80.5 (33.6).
- Code-path checks: groups of one reproduce the model to within 5e-5 to 7.6e-3
  of logits that reach 22 to 845 (OPT-125m, GPT-2, Pythia-160m, Llama-160m,
  Phi-1.5, OLMo-1B); fusing in a random function-preserving basis gives the same
  converted model to within 1e-4 to 9e-4 (Pythia-160m, Llama-160m, Phi-1.5,
  OLMo-1B). OPT-125m first gave non-finite weights for groups of one, from
  directions without energy divided by a zero eigenvalue; such directions are
  now left out (zero columns), which changes nothing when they carry energy.

Development runs (`alignment_study/fuse_dev/`, GPU, GPT-2 medium and
Pythia-410m, round 24's development partitions, the first 500 examples of each
task). Perplexity (base 21.71 and 18.03), with round 24's development results
for the reference recipes:

| | GPT-2 medium g=2 | g=4 | Pythia-410m g=2 | g=4 |
|---|---|---|---|---|
| round 24 `jin` | 121.4 | 376.4 | 382.7 | 997.9 |
| round 24 `sim_align` | 73.3 | 324.7 | 170.2 | 777.9 |
| `fuse_u/contiguous` | 60.8 | 180.5 | 117.9 | 311.7 |
| `fuse/contiguous` | 46.0 | 119.2 | 44.0 | 144.9 |
| `fuse/jin` | 39.2 | 96.3 | 41.5 | 105.9 |
| `fuse/sim_align` | 39.7 | 108.4 | 37.6 | 111.3 |
| `fuse_u/sim_align` | 48.3 | 181.7 | 97.5 | 237.3 |

Every registered direction holds in all four cells: `fuse/sim_align` below round
24's `sim_align` (dlog ratio at g = 2 0.50 and 0.33, geometric mean 0.40),
weighted below unweighted fusion in 8 of 8 comparisons, `fuse/contiguous` below
round 24's `sim_align` in 4 of 4, and `fuse/sim_align` below `fuse/contiguous`
in 4 of 4. Average accuracy of `fuse/sim_align` minus round 24's `sim_align` at
g = 2: +0.037 and +0.069 (mean +0.053, interval 0.044 to 0.063). After fusion,
Jin et al.'s groups did better than `sim_align`'s in three of the four cells (not
registered). Groups of one reproduced the models to 7.3e-4 and 5.4e-3 (logits
reaching 359 and 26), and fusing in a random basis gave the same converted model
to 6.4e-3 and 1.3e-4.

## Registered predictions

Ten cells: five models, two group sizes.
- F1: `fuse/sim_align` has a lower perplexity than round 24's `sim_align` (the
  same groups, aligned mean pooling) in at least 9 of 10 cells.
- F2 (size, g = 2): the geometric mean over models of dlog(`fuse/sim_align`) /
  dlog(`sim_align`) is at most 0.6.
- F3 (the weighting): weighted fusion has a lower perplexity than unweighted
  fusion on the same groups (neighbouring and `sim_align` groups) in at least 18
  of the 20 comparisons.
- F4 (accuracy, g = 2): `fuse/sim_align`'s average accuracy exceeds round 24's
  `sim_align` in at least 4 of 5 models, and the mean over models of the
  difference has a paired bootstrap 95% interval above zero (examples resampled
  within tasks within models, 2,000 draws).
- F5 (no search needed): `fuse/contiguous` has a lower perplexity than round 24's
  `sim_align` in at least 8 of 10 cells.
- F6 (the grouping still matters): `fuse/sim_align` has a lower perplexity than
  `fuse/contiguous` in at least 8 of 10 cells.
- Reported, not scored: every arm's perplexity and accuracy, `fuse/jin`, the
  accuracy differences with intervals at g = 4, perplexity as a multiple of the
  unconverted model's, and the checks (groups of one, basis invariance,
  restoration).

## Falsification and reporting

F1, F2 or F4 failing means the closed-form fusion does not improve on aligned
averaging as predicted; F3 failing means the query and output weighting is not
what improves the fit; F5 failing means the fusion still needs a searched
grouping to beat round 24's best recipe; F6 failing means the grouping no longer
matters once the heads are fused. Every clause is reported whichever way it
falls (`alignment_study/eval_fuse.py` writes `alignment_study/fuse_results.json`).
