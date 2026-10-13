# Preregistration 15: head importance on multi-head decoders and a second corpus

Date: 2026-10-13. Frozen by commit and pushed before the confirmatory run.

## Question

Preregistration 12 found, on four models with grouped-query attention and on
WikiText-103, that the attention "confidence" of Voita et al. (2019) tracks a
head's sink mass and does not predict the damage of removing the head, while
the norm of the head's output does. Voita et al. worked with multi-head
attention. This round asks whether the result holds on multi-head-attention
decoders, GPT-2 and Pythia, and on a second corpus, C4.

## Development check (labeled)

`importance_mha.py --dev` on gpt2 (`importance_mha_dev/`,
`importance_mha_dev.log`): a pipeline check, not used to set thresholds,
which are preregistration 12's. Confidence tracks sink mass at 0.96 in 7
high-sink layers; within-layer Spearman with removal damage on WikiText and
C4: output norm 0.62 and 0.58, confidence -0.28 and -0.33, negative entropy
-0.30 and -0.28, sink mass -0.25 and -0.27; restore checks 0.

## Models and protocol

gpt2-medium, EleutherAI/pythia-160m and EleutherAI/pythia-410m, every head of
every layer removed one at a time (`importance_mha.py`). Head statistics on
16 WikiText-103 validation windows of 256 tokens; removal damage as the KL of
the next-token distribution on 8 windows of 512 tokens from WikiText-103 test
and, separately, from C4 validation. Each head's output is read at the input
of its layer's output projection and the head is removed by zeroing its slice
of that projection.

## Registered predictions

Preregistration 12's thresholds, two-stage cluster bootstrap intervals:
- S2: within high-sink layers, confidence tracks sink mass at 0.5 or more,
  interval lower bound 0.3 or more.
- S3a and S3b, on each corpus: output norm's within-layer Spearman with
  removal damage exceeds confidence's and negative entropy's by 0.3 or more,
  intervals above zero.
- S4, on each corpus: output norm's within-layer Spearman with removal damage
  is 0.4 or more, interval above zero.
Seven clauses in all; every one is reported whichever way it falls, with
per-model values and the cross-layer correlations.
