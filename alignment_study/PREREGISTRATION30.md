# Preregistration 30: rebalancing fused key-value heads, for rounding and for uptraining

Date: 2026-10-30. Frozen by the commit that adds this file, before the confirmatory run.

## Question

The weighted fusion of rounds 26 to 28 halves a model's key-value heads without
training by fitting each group's shared key and value head to what the heads
compute. It builds the shared heads in a whitened basis and lets the query and
output projections absorb the inverse, so the converted model computes exactly
what the fit says but its weights are badly scaled: in TinyLlama the shared
value rows grow from a median norm of 0.46 to 2.72 (largest 49.6) while the
output columns shrink from 0.51 to 0.106 (smallest 0.0008). Two families of
rescalings leave the function unchanged (a value row against the matching
output columns of every query head of the group; a rotary plane of the shared
key against the same plane of every query of the group); fuse_balance.py sets
each pair of factors to equal norms. Development found that this lowers the
cost of rounding the attention projections to 4 bits and turns an untrainable
fused model into a trainable one. This round asks whether both hold on models
not used in development.

## Instrument

`alignment_study/balance_round.py`, evaluated by `alignment_study/eval_balance.py`;
conversion by `fuse_kv.py` (frozen in round 27) with the pair costs of
`kvfuse.py`, rebalancing by `fuse_balance.py`, export to a real grouped-query
model by `gqa_export.py`, rounding by the `rtn` function of
`llama2_quant_sink_probe.py` (asymmetric round to nearest, one scale per output
channel).

Calibration: second moments of every layer's attention input over 128 C4
validation documents of 2,048 tokens (round 28's). Pairs: the minimum-cost
perfect matching under the fusion's residual; neighbouring pairs where the
pair costs do not apply (projection biases: Qwen2.5-7B). Every model goes to
half its key-value heads.

Part Q (rounding; float32 when the weights fit, bfloat16 above 4B parameters):
allenai/OLMo-1B-hf, h2oai/h2o-danube-1.8b-base, openlm-research/open_llama_3b_v2,
NousResearch/Llama-2-7b-hf (revision 8efe6c9b), mistralai/Mistral-7B-Instruct-v0.2,
Qwen/Qwen2.5-7B. WikiText-2 perplexity on the first 8 windows of 2,048 tokens:
unconverted; fused; rebalanced; and each of the two with the attention
projections (query, key, value, output) rounded to 4 and to 3 bits. Logits of
the fused and the rebalanced model compared on one window of 512 tokens.

Part T (uptraining; float32): allenai/OLMo-1B-hf (16 to 8 key-value heads) and
h2oai/h2o-danube-1.8b-base (8 to 4). Three starts, all exported as real
grouped-query models: averaging neighbouring heads, the fusion unbalanced, the
fusion rebalanced. Each start trained at each learning rate of {5e-6, 2e-5,
8e-5, 2e-4}: the attention projections only, AdamW (betas 0.9 and 0.95, no
weight decay), linear warmup over max(10 steps, 2 percent) then cosine decay to
a tenth, gradient clipping at 1.0, 20 steps of 16 sequences of 1,024 tokens
(327,680 tokens) packed from C4 validation shard 0 after its 10,000th document,
the same sequences in the same order for every run. WikiText-2 perplexity on
the first 40 windows of 1,024 tokens at 0, 10, 20, 40, 60, 80 and 100 percent
of the steps.

## Development (labeled; not scored)

Llama-160m and TinyLlama v1.1 only (alignment_study/balance_dev/,
uptrain_dev/, uptrain_dev_*.log, run log 2026-10-30). Rebalancing changes logits
by at most 5.6e-6 (Llama-160m) and 4.9e-6 (TinyLlama) of the largest logit. With
the attention projections rounded to 4 bits per output channel, perplexity is
138.3 rebalanced against 144.1 unbalanced on Llama-160m (fused 113.7) and 62.0
against 71.8 on TinyLlama (fused 54.5): pooled, rebalancing leaves 0.63 of the
4-bit cost in log perplexity (0.83 and 0.47). At 3 bits, 189.6 against 204.1 and
217.4 against 226.7. Uptraining TinyLlama (4 to 2 key-value heads) for 20 steps:
the rebalanced fusion ends at 29.5, 21.2 and 19.4 at learning rates 5e-6, 2e-5
and 8e-5; the unbalanced fusion at 30.1 (after 24.8 at step 12) and 63.8 at
5e-6 and 2e-5 (at 1e-4 with a one-step warmup it diverged to 39,009); averaging
at 293.7, 189.4 and 100.1 (unconverted 8.38). On Llama-160m, whose fused weights
are less unbalanced, the two fused starts trained alike over 4 steps at 1e-4
(43.5 unbalanced, 44.7 rebalanced, from 50.7).

## Registered predictions

- Q1: rebalancing leaves the function unchanged: in every Part Q model the
  rebalanced perplexity is within 0.5 percent of the fused, and in the three
  float32 models no logit changes by more than 1e-4 of the largest logit.
- Q2: with the attention projections rounded to 4 bits, the rebalanced model's
  perplexity is below the unbalanced one's in every Part Q model (6 of 6).
- Q3: pooled over the six models, rebalancing leaves at most 0.75 of the 4-bit
  cost: the sum of ln(rebalanced at 4 bits / rebalanced) over the sum of
  ln(fused at 4 bits / fused) is at most 0.75.
- T1: in both Part T models, the rebalanced start's best final perplexity over
  the four learning rates is below the unbalanced start's best.
- T2: in both models, at each learning rate the rebalanced start ends at or
  below the unbalanced start (8 of 8).
- T3: in both models, the rebalanced start's best run is below the averaging
  start's best run at every checkpoint (7 of 7 in each).
- Reported, not scored: the 3-bit results; every curve; the best learning rate
  of each start; the unconverted perplexity; the run times.

## Falsification and reporting

Q1 failing would mean the rebalancing changes the model; Q2 or Q3 failing, that
its rounding benefit does not carry to these models or is small; T1 or T2
failing, that the unbalanced fusion trains as well (at its best, or at some
learning rate) and the development result was specific to TinyLlama; T3
failing, that averaging catches up within this budget. The budget is small
(327,680 tokens, far below the 5 percent of pretraining of GQA's uptraining),
so every training clause is about the head start in that regime. Every clause
is reported whichever way it falls.
