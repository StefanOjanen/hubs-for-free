# Round 31 draft: the rule for the attention weighting, fixed before the last development value

Written 2026-10-31, before OpenLLaMA-3B's full-weighting arm (lambda = 1) was read, and committed then.

The attention-weighted fusion mixes the plain second moment S and the attention-weighted one S_att as
(1 - lambda) S + lambda S_att. The scored arm of round 31 uses one lambda, chosen from the grid {0, 0.5, 1} (the
values measured on all three development models) as the value with the smallest worst-case relative loss to each
development model's own best: lambda* = argmin over lambda of max over models of ppl_m(lambda) / min over the grid of
ppl_m - 1. Development models: JackFram/llama-160m, TinyLlama/TinyLlama_v1.1, openlm-research/open_llama_3b_v2
(alignment_study/attnw_fuse_dev/). Values so far (8 or 20 WikiText-2 windows of 2,048 tokens): Llama-160m 105.00,
104.24, 107.60; TinyLlama 58.51, 40.53, 36.46; OpenLLaMA-3B 28.40, 25.62, pending. Full weighting (lambda = 1) is
registered as a secondary, reported comparison whichever value the rule picks.
