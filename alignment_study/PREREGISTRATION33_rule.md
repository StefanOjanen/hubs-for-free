# Round 33 draft: uptraining at a budget reviewers accept, and the rule that fixes its one free threshold

Date: 2026-10-31. Committed before the 5M-token development run on Llama-160m
(uptrain_round.py --dev, arms base, averaging, fusion rebalanced, fusion in
round 32's metrics rebalanced, learning rate 1e-4) has produced a number.

## Question

Round 30 showed the fused start beating averaging at every checkpoint of a
327,680-token uptraining. The field uptrains on 93M tokens and more, and CAGA
reports that after 100M tokens three groupings differ by 0.23 perplexity. Does
the fused start keep a lead at a budget fifteen times round 30's, or does
averaging catch up?

## Design (to be frozen as PREREGISTRATION33.md after the development run)

Models: OLMo-1B and h2o-danube-1.8b (round 30's), 5M tokens of C4 validation
text after document 10,000, attention projections only, AdamW, learning rate
2e-5 (round 30's best for both models), warmup and cosine as in round 30,
checkpoints at 0, 0.1, 0.2, 0.4, 0.6, 0.8 and 1.0 of the budget, WikiText-2
perplexity on 40 windows of 1,024 tokens, the zero-shot average at the end.
Arms: averaging (Ainslie et al.'s start), the fusion rebalanced, the fusion in
round 32's metrics rebalanced. The lead of an arm over averaging at a
checkpoint is ln(perplexity of averaging) - ln(perplexity of the arm).

Clauses:
- U1: the fused start ends below averaging in both models.
- U2: the fused start is below averaging at every checkpoint in both models.
- U3: in both models the fused start's lead at the end is at least F times its
  lead at the start.
- U4 (reported, scored only as stated): the round 32 start ends no higher than
  the plain fused start in both models.

## The rule for F

F is the Llama-160m development run's retained lead at 5M tokens, lead(end) /
lead(start) for the rebalanced fusion against averaging, rounded down to the
nearest tenth and capped at 0.5; if the development lead at the end is not
positive, F = 0.1 and U3 is kept as the weakest claim that averaging does not
catch up. The rule is fixed here so that the development value chooses the
threshold rather than the threshold being chosen after seeing the registered
models.

## Cost

About 20 GPU hours on the M1 Max (three arms, two models, 2.6 and 4.2 hours
per 5M-token arm from round 30's rates); the author decides whether to spend it.
