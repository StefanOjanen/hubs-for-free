# Preregistration 14: preregistration 12's clauses at four times the context

Date: 2026-10-13. Frozen by commit and pushed before the run.

## Question

Rounds 11 and 12 measured selection statistics on 256-token windows and
outcomes on 512-token windows. Short contexts are a stated limitation of
the project. This round repeats preregistration 12's head-importance and
multi-pair-merge clauses at 1,024 tokens, with the same instruments,
statistics and thresholds and no new development calibration.

## Models and protocol

Qwen2.5-1.5B and TinyLlama, the two fast models of rounds 11 and 12
(`standard_long.py`). Selection: 8 windows of 1,024 tokens cut from the
concatenated WikiText-103 validation split. Evaluation: 4 windows of 1,024
tokens from the concatenated test split. Ablation of every head in layers
3, 9, 15, 21, 27 (Qwen2.5-1.5B) and 3, 8, 13, 18, 21 (TinyLlama).
Multi-pair merges as in round 12: eight criteria and five random seeds, one
and two pairs per KV group in layers 1 to L-1.

## Registered predictions

Preregistration 12's S2, S3a, S3b, S4, S5, S6 and S7 with their thresholds
(`eval_long.py`): confidence tracks sink mass at 0.5 or more (interval lower
bound 0.3 or more); output norm beats confidence and negative entropy at
predicting removal damage by 0.3 or more, intervals above zero; output norm
predicts it at 0.4 or more, interval above zero; merges chosen by output
perturbation cost less than those chosen by the generator cosine and by the
last-row correlation in both models, geometric means 0.8 or less; output
perturbation among the two best of eight criteria in both models. S1 is not
tested (it needs pair merges at this length). Every clause is reported
whichever way it falls.
