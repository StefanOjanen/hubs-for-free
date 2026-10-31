# Preregistration 32: fitting the fusion in the metrics the attention computation implies

Date: 2026-10-31. Frozen by the commit that adds this file, before the confirmatory run.

## Question

The weighted fusion (rounds 26 to 30) fits each group's shared key and value head in
the metric of S = E[x x^T], every token alike. Round 31 replaced S by S_att, each token
weighted by the attention it receives, and failed 0 of 3 (lower in 3 of 5 models). With
the attention pattern held fixed, the attention computation itself gives two other
metrics:

- values: head h's output at query t is O_h V x_bar_t with x_bar_t = sum_s p_ts x_s, so
  the value error is exactly tr(dV S_bar dV^T O_h^T O_h) with S_bar = E_t[x_bar_t
  x_bar_t^T]. This is BoA's value Hessian (Kim et al., ICML 2025); S_att is, up to scale, its
  diagonal upper bound (Jensen's inequality over each query's attention).
- keys: to first order, a logit error d_ts moves the output by p_ts d_ts (z_s - z_bar_t),
  with z the value seen through the output projection. Its diagonal weights a key
  error at token s by w_s = sum_t p_ts |z_s - z_bar_t|^2 and a query at position t by
  u_t = sum_s p_ts |z_s - z_bar_t|^2. BoA keeps this Jacobian in its exact key Hessian and
  drops it in the relaxed form it uses, which is the plain fusion's key metric; the
  diagonal here keeps it at the cost of one pass over the calibration text.

Development (labelled; below) picked the arm with S_w for keys, u-weighted query
energies and S_bar for values ("sens"), by the rule committed before any of its values
were read (PREREGISTRATION32_rule.md, 9c17ad1). This round asks whether it beats the
plain fusion on models it was not developed on.

## Instrument

`alignment_study/attnw2_round.py`, evaluated by `alignment_study/eval_attnw2.py`; the
metrics and the per-layer fusion in `alignment_study/attnw2_common.py` (query energies
per head and rotary plane instead of a d x d S_u; biases through the ones column, the
defect of round 31); checked against the development script on Llama-160m by
`alignment_study/attnw2_check.py` (below). Conversion by fuse_kv.py's frozen
fuse_kv_group (values) and attnw2_common.fuse_kv_group_E (keys), pairs at the fusion's
minimum-cost matching under S (neighbouring pairs for Qwen2.5-7B, whose projections
carry biases), the same in both arms. S, S_w, S_bar and the energies from the same 128
C4 validation documents of 2,048 tokens (round 28's calibration), queries every 8th
position. WikiText-2 perplexity on the first 8 windows of 2,048 tokens; for Llama-2-7B
also on the whole test set at 2,048 tokens.

Models, at pinned revisions (in the script):
- new to this line of work: HuggingFaceTB/SmolLM2-1.7B, unsloth/Llama-3.2-1B (a public
  copy of Meta's Llama 3.2 1B), mistralai/Mistral-7B-v0.1;
- round 31's models, seen there with a different arm: allenai/OLMo-1B-hf,
  h2oai/h2o-danube-1.8b-base, mistralai/Mistral-7B-Instruct-v0.2,
  NousResearch/Llama-2-7b-hf, Qwen/Qwen2.5-7B.
The new tier is small because the machine had 13 GB of free disk.

## Development (labelled; not scored)

(attnw_mix_dev/, attnw_mix_dev/rule_result.json by attnw_mix_rule.py; 32 C4 documents
of 1,024 tokens, queries every 8th position)

Perplexity on WikiText-2 (Llama-160m 20 windows, the others 8) after conversion to half
the key-value heads; key metric / query energies / value metric:

| Arm | Llama-160m (unconverted 66.99) | TinyLlama (7.24) | OpenLLaMA-3B (6.77) |
|---|---|---|---|
| plain S/S/S | 105.00 | 58.51 | 28.40 |
| round31 S_att/S_att/S_att | 107.34 | 36.40 | 24.85 |
| mixv S_att/S_att/S_bar | 108.62 | 38.87 | 26.09 |
| plainmixv S/S/S_bar | 103.83 | 55.82 | 28.60 |
| sens S_w/S_u/S_bar | 102.78 | 40.56 | 26.56 |
| sensk S_w/S_u/S | 104.03 | 41.25 | 25.68 |

The rule's smallest improvement over plain: sens 2.12 percent, sensk 0.92, plainmixv
-0.70, round31 -2.23, mixv -3.45; it picks sens. Pooled over the three models, sens
keeps 0.885 of the plain arm's increase in log perplexity (round31 0.853). BoA's
metrics (plainmixv) lose to plain on OpenLLaMA-3B. The check of the round's instrument
on Llama-160m reproduces the development values: plain identical, sens 102.775 against
102.775 (relative difference 2.4e-8; attnw2_check/llama-160m.json).

## Registered predictions

- D1: the sens arm's perplexity is below the plain arm's in every model (8 of 8).
- D2: in each of the three new models, the sens arm is below the plain arm (3 of 3).
- D3: pooled over the eight models, the sens arm keeps at most 0.94 of the plain arm's
  increase in log perplexity over the unconverted model (sum of ln(sens / unconverted)
  over sum of ln(plain / unconverted)). The ceiling is the development value, 0.885,
  moved halfway toward no effect, since the arm was chosen on those models.
- D4: on Llama-2-7B, the sens arm's whole-test perplexity at 2,048 tokens is below
  round 28's registered 16.78.
- Reported, not scored: every perplexity; per-tier pooled ratios; run times.

## Gate

The plain arm must reproduce round 31's plain value (or, for Qwen2.5-7B, the corrected
23.873) within 0.1 percent in round 31's five models; otherwise the round's instrument
is defective and no clause is scored.

## Falsification and reporting

D1 or D2 failing means the sensitivity metrics hurt a model they were not developed on;
D3 failing, that the pooled gain is smaller than registered; D4 failing, that they do
not improve round 28's Llama-2-7B result. Every clause is reported whichever way it
falls.
