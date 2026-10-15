# Preregistration 22: the merge benchmark at larger scale and on a downstream task

Date: 2026-10-15. Frozen by commit and pushed before the confirmatory runs.

## Question

Round 20 found, in three multi-head models up to 410M parameters, that no
similarity in use predicts the damage of merging two heads' keys and values
(within-layer Spearman 0.12 at most), while simulating the merge predicts it
at 0.72 and chooses merges that cost 0.28 to 0.57 as much as the CHAI-style
choice. Two questions remain before this can carry a paper: does it hold in
larger models, including OPT, whose projections and query scaling differ;
and does the KL damage track what a user would see on a task?

## Instrument (`alignment_study/merge_bench2.py`)

Round 20's merge, criteria, selection and evaluation windows and KL measure,
imported unchanged from `merge_bench.py`, with OPT support added (separate
query, key and value projections; queries scaled before the attention call,
which the simulation now takes from the call).

Part S: EleutherAI/pythia-1.4b, facebook/opt-1.3b and gpt2-large; every pair
of heads in every layer merged on its own; one and two pairs per layer in
layers 1 to L-1 chosen greedily by each criterion, with five random draws.

Part D: LAMBADA (OpenAI version, first 1,000 test examples), accuracy of
greedy prediction of every token of the last word, for the unmerged model and
for the merges chosen at one and two pairs per layer by the output
perturbation, the last-row correlation (the CHAI-style choice) and the
Jensen-Shannon divergence, and by three random draws (the same draws as the
KL runs); on round 20's three models and the three above.

## Development (labeled, `alignment_study/merge_bench2_dev/`)

facebook/opt-125m for Part S: capture attention equals eager attention
exactly, restore check 0; within-layer Spearman with merge damage: output
perturbation 0.72, every other criterion between -0.17 and 0.11; one pair per
layer, KL 0.063 against 0.130 for the CHAI-style choice. gpt2 and opt-125m
for Part D on 200 examples: base accuracy 0.37 and 0.41; the accuracy drop of
the output perturbation's merges is below the CHAI-style choice's in all four
comparisons (gpt2 -0.04 against 0.22 and 0.08 against 0.28; opt-125m 0.04
against 0.06 and 0.07 against 0.18); across 24 configurations the accuracy
drop tracks the KL damage at Spearman 0.74.

## Registered predictions

Two-stage cluster bootstrap (models, then layers; 2,000 draws) for S1 and S2.
- S1: in the three new models, the output perturbation's mean within-layer
  Spearman with merge damage is at least 0.4, interval above zero.
- S2: it exceeds the best other criterion's by at least 0.2, interval above
  zero.
- S3: every other criterion's mean is below 0.3.
- S4: at one pair per layer, the output perturbation's merges cost less KL
  than the CHAI-style choice in every new model, geometric mean ratio at most
  0.8.
- D1: at one pair per layer, the output perturbation's merges lose less
  LAMBADA accuracy than the CHAI-style choice's in at least five of the six
  models.
- D2: the same at two pairs per layer.
- D3: across all configurations (six models, two budgets, six choices), the
  accuracy drop tracks the KL damage at Spearman 0.5 or more.
- Reported: every criterion's values, the Jensen-Shannon and random choices'
  accuracies, the change in NLL, and the checks.

## Falsification and reporting

S1 to S4 failing means round 20's result does not hold at this scale or in
OPT; D1 or D2 failing means the advantage in KL does not carry to the task;
D3 failing means KL damage is a poor proxy for task accuracy here. Every
clause is reported whichever way it falls (`alignment_study/eval_merge_bench2.py`
writes `alignment_study/merge_bench2_results.json`).
