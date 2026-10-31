# Preregistration 33: uptraining the fused start at 5M tokens

Date: 2026-10-31. Frozen by the commit that adds this file, before the confirmatory run.
The design and the rule for the one free threshold were committed at e2c9852
(`PREREGISTRATION33_rule.md`) before the development run produced a number.

## Question

Round 30 found the fused start below averaging at every checkpoint of a
327,680-token uptraining in OLMo-1B and h2o-danube-1.8b. At a budget fifteen
times larger, does the lead persist or does averaging catch up?

## Instrument

`alignment_study/uptrain_round.py --registered=<permalink> --model=<model>
--lr=2e-5 --arms=mean,fuse,fuse_sens` (tokens 5,000,000, sequences of 1,024
tokens from C4 validation shard 0 after document 10,000, 16 sequences a step,
AdamW, warmup 2 percent with at least 10 steps, cosine to a floor of 0.1,
gradient clipping 1.0, attention projections only, float32, the fused starts
rebalanced), evaluated by `alignment_study/eval_uptrain33.py`. Calibration 128
C4 validation documents of 2,048 tokens; pairs by the fusion's residual.
WikiText-2 perplexity on windows of 1,024 tokens at 0, 0.1, 0.2, 0.4, 0.6, 0.8
and 1.0 of the budget; the seven-task zero-shot average on 500 examples each
after training (reported). Models: allenai/OLMo-1B-hf, h2oai/h2o-danube-1.8b-base.

## Development (labelled; not scored)

Llama-160m, 5M tokens at learning rate 1e-4, arms base, averaging, fusion
rebalanced, round 32 fusion rebalanced (`uptrain_dev/llama-160m_5M.json`,
`uptrain_dev/rule_result.json`): final perplexity base 23.76 (from 23.18), averaging 51.61 (from 727.2), fusion rebalanced 30.78 (from 50.63), round 32 fusion rebalanced 30.36 (from 47.35); zero-shot averages after training base 0.410, mean 0.359, fuse 0.383, fuse_sens 0.370; the fusion's lead over averaging in log perplexity 2.66 at the start and 0.52 at the end, retained 0.194; checkpoints at 0, 0.5, 1, 2, 3, 4 and 5M tokens: averaging 727.2, 163.1, 92.0, 64.4, 55.2, 53.5, 51.6; fusion 50.6, 33.6, 32.4, 31.2, 30.8, 30.9, 30.8; round 32 fusion 47.4, 33.1, 31.9, 30.9, 30.5, 30.5, 30.4. The rule gives F = 0.1.

## Registered predictions

- U1: the fused start ends below averaging in both models.
- U2: the fused start is below averaging at every checkpoint in both models.
- U3: in both models, the fused start's lead over averaging at the end (in log
  perplexity) is at least 0.1 times its lead at the start.
- U4: the round 32 start ends no higher than the plain fused start in both
  models.
- Reported, not scored: every curve, the zero-shot averages, run times.

## Falsification and reporting

U1 or U2 failing means averaging catches up within 5M tokens; U3 failing, that
most of the start's lead is gone by then; U4 failing, that round 32's metrics do
not help a start that will be trained. Every clause is reported whichever way
it falls. Cost: about 20 GPU hours; run only with the author's go.
