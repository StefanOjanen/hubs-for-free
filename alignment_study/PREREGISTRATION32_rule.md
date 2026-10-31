# Round 32 draft: the rule that picks the fusion's metrics

Date: 2026-10-31. Committed with `alignment_study/attnw_mix_dev.py` before any of
its development values are read.

## Why a new rule

Round 31 replaced the fusion's second moment S by S_att, each token weighted by
the attention it receives, for keys, values and query energies alike, and failed
0 of 3 (lower in 3 of 5 models; Llama-2-7B worse). Its weighting was chosen by a
minimax rule (smallest worst-case loss against the best development arm), which
picked a setting that was worse than the plain fusion on one development model
(Llama-160m, 107.6 against 105.0). This round's rule asks instead for an arm that
beats the plain fusion on every development model.

## Arms

Metrics for the key fit / the query energies / the value fit (`attnw_mix_dev.py`;
pairs fixed at the fusion's matching under S; 32 C4 documents of 1,024 tokens;
received attention and the mixed inputs from every 8th query position):

- plain: S / S / S (the fusion of rounds 26 to 30).
- round31: S_att / S_att / S_att.
- mixv: S_att / S_att / S_bar.
- plainmixv: S / S / S_bar (BoA's value Hessian, and its relaxed key Hessian; Kim
  et al., ICML 2025).
- sens: S_w / S_u / S_bar.
- sensk: S_w / S_u / S.

S_bar = E_t[x_bar_t x_bar_t^T] with x_bar_t = sum_s p_ts x_s, averaged over the
layer's query heads: with the attention fixed, the exact metric of the value
error. S_w weights token s by w_s = sum_t p_ts |z_s - z_bar_t|^2 and S_u weights
query position t by u_t = sum_s p_ts |z_s - z_bar_t|^2, with z the value seen
through the head's output projection (|z|^2 = v^T O_h^T O_h v): the diagonal of
the first-order output error of a logit error at (t, s), summed over heads.

## Rule

On Llama-160m (20 WikiText-2 windows of 2,048 tokens), TinyLlama v1.1 and
OpenLLaMA-3B v2 (8 windows each), compute each arm's relative improvement over
plain, (plain - arm) / plain. Pick the arm, other than plain, whose smallest
improvement over the three models is largest. If that smallest improvement is
not positive (no arm beats plain on all three), no confirmatory round is run on
these arms and the result is reported as a development negative. Ties within
0.1 percentage points go to the arm listed first above.
