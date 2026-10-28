# Preregistration 28: the weighted fusion against CAGA on CAGA's own benchmark

Date: 2026-10-28. Frozen by commit and pushed before the confirmatory run.

## Question

Rounds 26 and 27 reduced the key-value heads of trained models without training
by fitting each group's shared key and value head in closed form to what the heads
compute: the shared key weighted by the queries that read it, the shared value by
the output projections that write it. In round 26 this beat round 24's best recipe
in 10 of 10 cells and the same fit without the weighting in 20 of 20.

The closest published method is CAGA, part of CHASE (Wang, Jiang and Liu, arXiv
2610.09476, 7 October 2026). Its
Table 2 converts Llama-2-7B from 32 to 16 key-value heads without training and
reports WikiText-2 perplexity 25.1 and a seven-task zero-shot average of 44.3,
against 926 and 32.6 for direct averaging and 738 and 35.3 for Procrustes
alignment, all on CAGA's own pairing of heads. CAGA fits each pair's shared key
in the metric of the attention input's second moment, with a complex scalar per
rotary plane absorbed into each query, which is the same fit as our unweighted
key fit; it takes the shared value from a rank-d_h SVD of the stacked value
projections and absorbs the head factors into the output projections, with
neither the input metric nor any output weighting.

This round runs our fusion on that benchmark: against CAGA's published numbers,
and against our implementation of CAGA's construction on the same pairs of
heads, where no difference of evaluation protocol can enter. It adds a pairing
of heads chosen by the fusion's own residual and a functional check of the
converted models by steering.

## Instrument

`alignment_study/llama2_round.py`, evaluated by `alignment_study/eval_llama2.py`;
the weighted fusion and CAGA's key fit are `alignment_study/fuse_kv.py` as frozen
in round 27 (35249e6), unchanged.

Model: NousResearch/Llama-2-7b-hf at revision
8efe6c9b93655b934e27bd9981e3ec13e55aee9d. Both safetensors files are
byte-identical to meta-llama/Llama-2-7b-hf: their SHA-256 hashes equal the
hashes in the official repository's metadata (4ec71fd5... and 41780b5d...).
bfloat16 on the M1 Max GPU; every fit computed in float64 on the host; one run at
a time under the memory watchdog.

Constructions of a pair's shared key and value head:
- `mean`: the key and value projections averaged (Ainslie et al. 2023).
- `caga`: CAGA as its paper specifies it. Keys: minimize
  sum_h ||(W_K,h - T_h W_K) S^(1/2)||_F^2 with T_h a scale-rotation per rotary
  plane, absorbed into the queries (computed by round 27's unweighted key fit).
  Values: the rank-d_h SVD of the stacked value projections U s V^T, shared value
  projection s V^T, each head's factor U_h absorbed into its output projection.
- `fuse`: the query- and output-weighted fusion of rounds 26 and 27.

Pairings:
- `neighbour`: heads 2i and 2i + 1, the standard conversion.
- `paired`: in each layer, the minimum-cost perfect matching of the 32 heads
  (networkx min_weight_matching), the cost of a pair being the weighted fusion's
  residual: the query-weighted key residual summed over rotary planes, divided by
  the layer's total query-weighted key energy, plus the output-weighted value
  residual, divided by the layer's total output-weighted value energy. The
  residuals are computed in closed form from the weights and the second moment;
  on Llama-160m they equal a brute-force fusion of the pair to seven significant
  digits. A pairing that is not of neighbours is deployed by permuting the query
  heads and the output projection's columns, which does not change the function.

Arms: `fuse/paired`, `caga/paired`, `fuse/neighbour`, `caga/neighbour`,
`mean/neighbour`, and the unconverted model.

Calibration: the second moment of every layer's attention input over 128 C4
documents of 2,048 tokens, the first documents that long in validation shard 0
(CAGA used 128 sequences of 2,048 tokens from C4's training split).

Evaluation:
- WikiText-2 test perplexity over non-overlapping windows of 2,048 and 4,096
  tokens (the raw test split joined by blank lines, as in rounds 24 to 27). The
  paper does not state its evaluation length; these are the two in use for
  Llama-2-7B, where the unconverted model's perplexity is usually reported as
  5.47 and 5.12.
- ZS-7: BoolQ, RTE, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge and
  OpenBookQA in lm-evaluation-harness's formats (`zeroshot.py` for five tasks,
  BoolQ and RTE in the round script), the first 1,000 examples of each (RTE has
  277 and OpenBookQA 500; HellaSwag every tenth validation example), accuracy for
  BoolQ, RTE and WinoGrande and length-normalized accuracy for the other four,
  the convention the paper states for its other experiments. The plain-accuracy
  average is also reported.
- Steering check, a functional comparison in the manner of steering-based neuron
  explorers: at the middle layer (16), the write directions (down-projection
  columns) of 64 randomly chosen MLP neurons and 64 random Gaussian directions,
  unit length, are each added at every position at 0.8 times the median residual
  norm of that layer (measured on the unconverted model), on 16 C4 prompts of 128
  tokens taken from documents shorter than 2,048 tokens (so none is a calibration
  document). A direction's signature is the mean over prompts of the change in
  next-token probability at the last position. A converted model's agreement is
  the overlap of the 20 tokens each signature raises most with the unconverted
  model's, averaged over the neuron directions; the random directions and the
  cosine of the signatures are reported.

Checks: groups of one head reproduce the model (maximum logit difference, as in
round 27), and the restored model reproduces the unconverted perplexity.

## Development (labeled)

Development runs (`alignment_study/llama2_dev/`, `alignment_study/llama2_dev.log`;
GPU; 40 WikiText-2 windows; the first 200 examples of each task):

Llama-160m (12 heads, 32 documents of 2,048 tokens), perplexity at 1,024 tokens,
ZS-7 and steering agreement (neuron directions):

| arm | perplexity | ZS-7 | agreement |
|---|---|---|---|
| unconverted | 23.2 | 0.446 | 1.00 |
| fuse/paired | 52.9 | 0.437 | 0.50 |
| caga/paired | 287 | 0.380 | 0.29 |
| fuse/neighbour | 59.2 | 0.440 | 0.50 |
| caga/neighbour | 344 | 0.412 | 0.30 |
| mean/neighbour | 727 | 0.375 | 0.29 |

OpenLLaMA-3B v2 (32 heads, multi-head, CAGA's calibration of 128 documents of
2,048 tokens; three arms, to save time), perplexity at 1,024 and 2,048 tokens,
ZS-7 and agreement:

| arm | perplexity 1,024 | perplexity 2,048 | ZS-7 | agreement |
|---|---|---|---|---|
| unconverted | 8.50 | 7.60 | 0.553 | 1.00 |
| fuse/paired | 26.3 | 30.3 | 0.460 | 0.40 |
| caga/paired | 75.9 | 141.0 | 0.441 | 0.18 |
| fuse/neighbour | 29.3 | 34.2 | 0.457 | 0.40 |

The matching kept 11 of 416 neighbour pairs, lowered the summed relative
residual from 4.66 to 4.23 and the fusion's perplexity by 10 and 11 percent. The
damage grows with the window: `fuse/paired` is 3.1 times the unconverted
perplexity at 1,024 tokens and 4.0 times at 2,048, `caga/paired` 8.9 and 18.6
times. The ZS-7 difference is carried by six tasks
(BoolQ went the other way, 0.455 against 0.540; yes-no tasks swing with a damaged
model's answer bias).

Numerics (`alignment_study/llama2_numerics_check.py`, `fuse_gauge.py`; outputs
in `llama2_dev/numerics_*.json` and the development log). The fusion stores each
shared key plane as a whitened unit vector and each shared value projection in a
whitened basis; in bfloat16 these weights take new rounding. With groups of one
the converted OpenLLaMA-3B differs from the original by up to 12.5 in a logit at
the first position (median over positions 0.5) and its perplexity by 0.07
percent (7.6921 against 7.6865 on 16 windows); on Llama-160m bfloat16 rounding
costs groups of one 0.07 percent and pairs nothing measurable (62.17 in
bfloat16, 62.14 in float32). A change of gauge that makes groups of one exact in
bfloat16 leaves the function unchanged (identical perplexity in float32) but
makes pairs slightly worse in bfloat16 (62.61 against 62.17), so the round keeps
round 27's conversion without it. The first version of this check moved weights
from the GPU with `.to("cpu", torch.float64)`, which on MPS returns corrupted
values for large tensors; no committed script uses that form.

Earlier development information: the weighted fusion's perplexity at half the
key-value heads was 1.6 to 2.1 times the unconverted model's in round 26's
multi-head models with groups chosen by the simulated merge, and 3.9 times in
OpenLLaMA-3B with neighbouring heads and 8 calibration documents of 1,024 tokens
(round 27, whose evaluation is pending its last model).

## Registered predictions

- M1: `fuse/paired` has a WikiText-2 perplexity below CAGA's published 25.1 at
  both 2,048 and 4,096 tokens.
- M2: `fuse/paired`'s ZS-7 is above CAGA's published 44.3, and so is the lower end
  of its bootstrap 95% interval (examples resampled within tasks, 2,000 draws).
- M3 (same pairs): `fuse` has a lower perplexity than `caga` on both pairings at
  both lengths (4 of 4).
- M4 (same pairs, accuracy): `fuse`'s ZS-7 is above `caga`'s on both pairings, and
  the paired bootstrap 95% interval of the difference pooled over the two
  pairings is above zero.
- M5 (pairing): `fuse/paired` has a lower perplexity than `fuse/neighbour` at both
  lengths.
- M6 (steering): the agreement for neuron directions is higher for `fuse` than
  for `caga` on both pairings, and higher for `fuse/paired` than for
  `mean/neighbour`.
- Reported, not scored: every arm's perplexities, their multiples of the
  unconverted model's, ZS-7 with intervals, per-task accuracy and the
  plain-accuracy average; `caga/neighbour` and `mean/neighbour` against the
  published CAGA and averaging numbers (whose pairing differs); the random-
  direction agreement and the cosines; the rank correlation over arms between
  agreement and perplexity; the pairing residuals; the checks.

## Falsification and reporting

M1 or M2 failing means the weighted fusion does not beat CAGA's published
training-free result on its own benchmark. In development the damage of every
conversion grew with the window, so M1 at 4,096 tokens is its harder half. The
published numbers come from an evaluation whose length, subsets and harness
version the paper does not state,
and our subsets carry a sampling error of about one point in ZS-7; M3 and M4
compare the two constructions on identical pairs, data and evaluation, and
failing them means the advantage does not hold even there. M5 failing means the
fusion's own residual does not choose better pairs than neighbouring heads; M6
failing means the steering check does not separate the constructions as
perplexity does. Every clause is reported whichever way it falls
(`alignment_study/eval_llama2.py` writes `alignment_study/llama2_results.json`).
