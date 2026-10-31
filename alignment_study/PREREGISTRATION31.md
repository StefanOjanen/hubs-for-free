# Preregistration 31: fitting the shared heads where attention looks

Date: 2026-10-31. Frozen by the commit that adds this file, before the confirmatory run.

## Question

The weighted fusion of rounds 26 to 30 fits each group's shared key and value head
in the metric of the attention input's second moment S = E[x x^T], counting every
token alike. A key or value error at token s matters through the attention s
receives. In development (run log, 2026-10-31), replacing S by S_att, which
weights each token by the attention it receives, lowered the converted perplexity
of TinyLlama from 58.5 to 36.5 and of OpenLLaMA-3B from 28.4 to 24.8, and raised
Llama-160m's from 105.0 to 107.6; the gain was neither a position effect nor a
first-token effect. This round asks whether it holds on models it was not
developed on, and whether it improves round 28's Llama-2-7B result. The closest
prior work is BoA (Kim et al., ICML 2025), whose attention-aware Hessians serve
quantization; the claim here is only the application to closed-form head fusion.

## Instrument

`alignment_study/attnw_round.py`, evaluated by `alignment_study/eval_attnw.py`;
conversion by `fuse_kv.py` (frozen in round 27) with the pair costs and matching
of `kvfuse.py`.

S and S_att from the same 128 C4 validation documents of 2,048 tokens (round 28's
calibration): S_att weights token s by the attention it receives, averaged over the
layer's query heads and over every 8th query position (development: the estimate
from every 8th position gave 36.40 against 36.46 from all of them on TinyLlama),
scaled so that uniform attention would give S. The attention arm uses S_att alone
(lambda = 1, the value picked by the rule committed in
`PREREGISTRATION31_lambda_rule.md` before the last development value was read),
for the key fit, the value fit and the query energies alike, as in development.
The pairs are the fusion's minimum-cost matching under S, the same in both arms
(neighbouring pairs for Qwen2.5-7B, whose projections carry biases).

Models (not used in this method's development): allenai/OLMo-1B-hf,
h2oai/h2o-danube-1.8b-base (float32), mistralai/Mistral-7B-Instruct-v0.2,
NousResearch/Llama-2-7b-hf (revision 8efe6c9b), Qwen/Qwen2.5-7B (bfloat16). Each
model is converted to half its key-value heads twice, with S and with S_att.
WikiText-2 perplexity on the first 8 windows of 2,048 tokens; for Llama-2-7B also
on the whole test set at 2,048 tokens, as in round 28. Reported per layer: the
first position's share of the received attention and the received attention's
entropy divided by log of the window length.

## Development (labeled; not scored)

Llama-160m, TinyLlama v1.1 and OpenLLaMA-3B v2 (alignment_study/attnw_fuse_dev/;
32 documents of 1,024 tokens). Perplexity at lambda 0, 0.5 and 1: Llama-160m
105.00, 104.24, 107.60 (20 windows; unconverted 67.0); TinyLlama 58.51, 40.53,
36.46 (8 windows; 7.24); OpenLLaMA-3B 28.40, 25.62, 24.83 (8 windows; 6.77). Pooled
over the three, the attention arm keeps 0.85 of the plain fusion's increase in log
perplexity (1.05, 0.77, 0.91). Controls on TinyLlama: the first position left out
of the weighting 36.56; a weighting by position alone 61.26; query energies from S
39.39. The instrument's smoke test on Llama-160m with the round's calibration: plain
113.9, attention 114.5.

## Registered predictions

- A1: the attention arm's perplexity is below the plain arm's in every model (5 of 5).
- A2: pooled over the five models, the attention arm keeps at most 0.90 of the plain
  arm's increase in log perplexity over the unconverted model (sum of
  ln(attention / unconverted) over sum of ln(plain / unconverted)).
- A3: on Llama-2-7B, the attention arm's perplexity on the whole WikiText-2 test set
  at 2,048 tokens is below round 28's registered 16.78.
- Reported, not scored: every perplexity; the plain arm's whole-test value on
  Llama-2-7B against round 28's 16.78 (the same conversion); the concentration
  statistics and whether they order the per-model gains; the run times.

## Falsification and reporting

A1 failing means the weighting hurts some model it was not developed on; A2
failing, that its pooled gain is under a tenth of the conversion's cost; A3
failing, that it does not improve the published Llama-2-7B result. Every clause is
reported whichever way it falls.
