# Preregistration 24: converting multi-head attention to grouped-query attention without training

Date: 2026-10-16. Frozen by commit and pushed before the confirmatory runs.

## Question

Rounds 20 and 22 found that simulating the merge of two heads' keys and values
on calibration tokens predicts its damage (within-layer Spearman 0.72 and
0.68 in six multi-head models), while the similarities that methods use to
choose merges average 0.24 or less, and that the KL damage tracks LAMBADA
accuracy (Spearman 0.86). Those rounds merged one or two pairs per layer.
Converting a multi-head model to grouped-query attention merges every head in
every layer. Published conversions choose the groups by position (neighbouring
heads; Ainslie et al. 2023), by the distance of the heads' key and value
weights (QCQA; Joshi et al. 2024), by the similarity of their key and value
activations (AsymGQA; Chen et al. 2024), or by the similarity of their value
caches after a function-preserving Procrustes alignment (Jin et al. 2025).
QCQA reports results without training; AsymGQA and Jin et al. report them
after fine-tuning. This round asks, without any training: does choosing the
groups by the simulated merge give better converted models on perplexity and
standard zero-shot tasks, how much of the outcome is the alignment rather than
the grouping, and does the outcome of the unaligned recipe depend on an
arbitrary basis?

## Instrument

`alignment_study/gqa_round.py`, `gqa_common.py`, `zeroshot.py` and
`gqa_convention.py`.

Conversion. Each layer's heads are partitioned into groups of g = 2 or g = 4
(half or a quarter of the key-value heads remain), and every head's key
projection (weights and bias) is replaced by the group mean, and likewise its
value projection (mean pooling, as in all four published methods); queries
and output projections stay per head, so the converted model computes exactly
what a grouped-query model with the pooled heads computes. Every layer is
converted.

Arms (nine per group size).
- Without alignment: `contiguous` (neighbouring heads; GQA), `random0` (a
  random partition), `wse` (QCQA's weight-sharing error, minimized), `act`
  (AsymGQA's activation similarity, mean best-match cosine over token vectors
  in both directions, key plus value, maximized; one grouping serves keys and
  values, groups are equal in size, and AsymGQA's accuracy-guided acceptance
  is not used), `sim` (the merge simulated on calibration tokens: mean norm of
  the change in the group's summed contribution to the residual stream,
  minimized).
- With alignment, Jin et al.'s cosine variant (generalized Procrustes on
  per-token normalized caches within each group; keys by rotations that
  commute with the rotary embedding, values by any orthogonal matrix; folded
  into the weights, which leaves the unmerged model unchanged), then mean
  pooling: `contiguous_align`, `random0_align`, `jin` (Jin et al.'s grouping:
  pairwise post-alignment value-cache cosine, summed within groups,
  maximized), `sim_align` (the simulated merge computed after the candidate
  group's alignment, minimized).

Search, identical for every scored arm: minimum-cost (maximum-similarity)
perfect matching of heads into pairs (exact for g = 2); for g = 4, the same
matching of the pairs into quadruples by the group score, then head swaps
between groups while the total improves (at most two passes). QCQA's
evolutionary search, AsymGQA's randomized search and Jin et al.'s hill
climbing are replaced by this search on their own objectives.

Calibration: the first 8 documents of at least 1,024 tokens (under the
model's tokenizer) in C4 validation shard 0, truncated to 1,024 tokens; the
same tokens for every criterion.

Evaluation (`zeroshot.py`, the prompts and scoring of lm-evaluation-harness
reproduced; on the unconverted Pythia-410m it gives LAMBADA 0.516, PIQA 0.667,
WinoGrande 0.534, ARC-Easy 0.519 and ARC-Challenge 0.243 (acc_norm) against
EleutherAI's published 0.516, 0.668, 0.537, 0.521 and 0.243): WikiText-2 test
perplexity over non-overlapping 1,024-token windows, and the average of seven
zero-shot accuracies (LAMBADA all 5,153, HellaSwag every 10th validation
example (1,005), PIQA, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA; acc_norm
for HellaSwag, ARC-Challenge and OpenBookQA). Damage of an arm:
dlog = log(perplexity / perplexity of the unconverted model).

Convention null (`gqa_convention.py`): every head's query/key space and
value/output space is rotated by a random function-preserving rotation (three
draws). Neighbouring heads are pooled without alignment, and with alignment
recomputed in the rotated basis; perplexity on the first 64 WikiText-2 windows,
and for the aligned conversion the largest logit difference between the
models converted in the random and in the original basis.

Models: facebook/opt-1.3b, EleutherAI/pythia-1.4b, gpt2-large,
allenai/OLMo-1B-hf and microsoft/phi-2 (multi-head; learned, partial-rotary
and full-rotary positions; 0.8B to 2.7B parameters), float32 on the M1 Max.

## Development (labeled)

Code-path check (`alignment_study/gqa_check/`, 2 calibration documents of 512
tokens, 200 examples per task, 8 perplexity windows): OPT-125m, Llama-160m and
Phi-1.5; capture equals eager attention exactly; alignment changes the logits
by at most 8.1e-5 in float32 (3.1e-3 and 5.6e-3 in the development models,
whose logits reach several hundred and tens; with random rotations in float64
on the CPU, at most 1e-11 for GPT-2 medium and 1e-13 for the rotary
architectures). The rewritten Gram-based alignment reproduces the first
implementation's 144 partitions on OPT-125m exactly.

Development runs (`alignment_study/gqa_dev/`, GPT-2 medium and Pythia-410m,
full calibration, first 500 examples of each task). Perplexity (base 21.71
and 18.03):

| | GPT-2 medium g=2 | g=4 | Pythia-410m g=2 | g=4 |
|---|---|---|---|---|
| contiguous | 364 | 840 | 2,653 | 2,056 |
| random0 | 380 | 1,166 | 2,563 | 2,074 |
| wse | 390 | 814 | 2,504 | 2,309 |
| act | 388 | 845 | 1,785 | 1,799 |
| sim | 314 | 906 | 1,677 | 1,678 |
| contiguous_align | 138 | 520 | 573 | 1,733 |
| random0_align | 183 | 507 | 649 | 1,728 |
| jin | 121 | 376 | 383 | 998 |
| sim_align | 73 | 325 | 170 | 778 |

Alignment lowers perplexity for both naive groupings in all eight
comparisons (ratios 0.22 to 0.84). `sim_align` has the lowest perplexity of
the nine arms in all four cells; its dlog is 0.71 and 0.74 of `jin`'s at g = 2
and 0.95 and 0.94 at g = 4. Across the nine arms, the simulated cost of each
arm's partition (with the alignment the arm uses) orders dlog at Spearman
0.90, 0.75, 0.97 and 0.90. Every unaligned arm leaves perplexity at 14 times
the base model's or more. Average accuracy (500 examples per task) of
`sim_align` minus `jin`: +0.011 and +0.009 at g = 2, +0.006 and -0.002 at
g = 4. In the code-path check, at reduced calibration and with five arms
(`contiguous`, `contiguous_align`, `jin`, `sim`, `sim_align`), `sim_align`
had the lowest perplexity in 5 of 6 cells (OPT-125m at g = 4, where `jin` was
lower, the exception) and a higher accuracy than `jin` in 3 of 3 models at
g = 2.

Convention null (`alignment_study/gqa_convention_dev/`, 64 perplexity
windows). Unaligned neighbouring pooling, original basis and three random
function-preserving bases: GPT-2 medium 374, 490, 408, 382 (g = 2) and 831,
1,025, 818, 754 (g = 4); Pythia-410m 2,594, 2,524, 2,197, 2,161 and 1,922,
1,566, 1,673, 1,775: a spread (largest over smallest) of 1.31, 1.36, 1.20
and 1.23, with the trained basis best in one cell and worst in two. Aligned
pooling gives the same perplexity in every basis to the printed precision
(143.68, 525.81, 586.23, 1,717.71), and its converted models differ across
bases by at most 7.3e-3 in any logit (float32 rounding).

## Registered predictions

Ten cells: five models, two group sizes.

- P1 (alignment): aligning before pooling lowers perplexity for `contiguous`
  and for `random0` in at least 18 of the 20 comparisons.
- P2: `sim_align` has a lower perplexity than `jin` in at least 8 of 10 cells.
- P3: `sim_align` has the lowest perplexity of the nine arms in at least 8 of
  10 cells.
- P4 (size, g = 2): the geometric mean over models of dlog(`sim_align`) /
  dlog(`jin`) is at most 0.85.
- P5 (accuracy, g = 2): `sim_align`'s average accuracy exceeds `jin`'s in at
  least 4 of 5 models, and the mean over models of the difference has a
  paired bootstrap 95% interval above zero (examples resampled within tasks
  within models, 2,000 draws).
- P6 (the simulation predicts conversion damage): across the nine arms, the
  Spearman correlation between the simulated cost of each arm's partition and
  its dlog is at least 0.6 in at least 8 of 10 cells.
- P7 (no criterion rescues unaligned pooling): every unaligned arm leaves
  perplexity at least 5 times the base model's in at least 9 of 10 cells.
- P8: without alignment, `sim` has a lower perplexity than `contiguous` at
  g = 2 in at least 4 of 5 models.
- C1 (the unaligned recipe depends on an arbitrary basis): across the
  original and three random function-preserving bases, the perplexity of
  unaligned neighbouring pooling spreads by a factor of at least 1.1 in at
  least 8 of 10 cells.
- C2 (alignment removes the dependence): in all 30 random-basis cases (10
  cells, three bases), aligned neighbouring pooling's perplexity is within
  0.1 percent of the original basis's.
- Reported, not scored: every arm's perplexity, per-task accuracies, the
  accuracy differences with intervals, `wse` and `act` against `random0`, the
  g = 4 accuracy comparisons, and the checks (capture against eager attention,
  alignment's function preservation, restoration of the unconverted model).

## Falsification and reporting

P1 failing means alignment is not what makes training-free pooling work; P2
to P5 failing means the simulated merge does not choose better groups than Jin
et al.'s criterion at this scale; P6 failing means the simulation does not
track conversion damage; P7 failing means some grouping criterion makes
unaligned pooling usable; C1 failing means the unaligned recipe's outcome does
not depend on the basis in practice; C2 failing means the alignment does not
make the conversion a function of the model alone. Every clause is reported
whichever way it falls (`alignment_study/eval_gqa.py` writes
`alignment_study/gqa_results.json`).
