# CLAUDE.md

## What this project is now

This folder began as the Emergent Coordinator Hypothesis (ECH). That framing
is retired: the headline results were unsupported (published symbolic law
R^2 = 0.982 absent from the project's own PySR logs, archived best fit
R^2 = 0.014; hub/eigengap/clustering signatures reproduce on random matrices;
depth-wise "crystallization" follows from attention locality). The active
deliverable is paper.md, "Hubs for Free: What Commutator-Based Analyses of
Attention Discover in Random Matrices". Old ECH documents are historical
case-study material; do not treat their claims as established.

## Ground rules for all work in this repo

1. Every number in any document must be regenerable from a script and
   traceable to a saved artifact. This includes numbers Claude produces:
   write the artifact first, then cite it.
2. Null models before interpretation: random matrices from the constraint
   set, marginal-matched surrogates, untrained same-architecture model,
   convention changes (sign flips, relabeling, index permutation).
3. Report adaptive fits against the best fit the identical search achieves
   on matched noise; label searched exhibits as searched.
4. No physics vocabulary (gauge, Lie algebra, phase transition) unless the
   mathematical property is verified and a null is shown.
5. Single-input results characterize (model, input); report distributions.

## Artifact status

experiments.py, results.json, and figures/fig1 to fig6 were reimplemented
from paper.md's specifications on 2026-09-02 (the August originals were
never committed); the paper's synthetic numbers are the regenerated values
and CI (.github/workflows/regenerate.yml) checks they regenerate. Still
author-held and missing from the repo: the local pilot (gptmini.py,
train_chunk.py, analyze_local.py, local_results.json, fig7). Section 7's
pilot numbers therefore have no committed artifact; remind the user to
upload them or mark the pilot as unreproduced.

Compute since 2026-09-10 is the author's M1 Max (32 GB) through Apple MPS;
the device and precision policy lives in hubsfree/adapters.py (float32
whenever the weights fit, bfloat16 above about 4B parameters) and the
measured fidelity against the CPU float32 protocol is in
alignment_study/platform_fidelity.json and RUNLOG.md. Every model run
writes results per model as it completes.

Rounds and their artifacts (all preregistered before execution, evaluation
scripts committed before results): 1 (gram_theorem.json,
robustness_addendum.json), 2 (heldout_round.json), 3
(scale_round_results.json, scale_partial/), 5 dynamics (dynamics/,
dynamics_results.json), 6 rerun (rerun/, rerun_results.json), 7
head-merging (merge/, merge_results.json; both primary clauses failed), 8
derivation (derivation/, derivation_results.json; gate failed at R^2
0.55, sink-set post hoc 0.75), 9 sink-profile generative model
(sinkprofile/, sinkprofile_results.json; 11 of 12 models), 10 new attention
designs (rerun_pr10/, sinkprofile_pr10/ and their _results.json; 4 of 4:
Falcon-7B multi-query, BLOOM-7b1 ALiBi, OPT-6.7B learned positions,
Qwen3-8B), 11 functional round (functional/, functional_results.json,
functional_dev_calibration.json; instruments in functional_common.py; 5 of
7 clauses, F3b and F4 failed: maps predict merge damage weakly, a
value-and-output comparison predicts it best, the sink is not the reason),
12 standard statistics with function (standard/, standard_results.json;
8 of 8: the largest attention weight over all tokens is a sink detector
with the wrong sign against removal damage, output norm predicts it within
layers, output perturbation beats Clark JS and CHAI-style statistics and
chooses cheaper multi-pair merges), 13 what the sink does (sink/,
sink_results.json; custom "sinkint" attention in sink_common.py must apply
its own causal mask), 14 round 12's clauses at 1,024 tokens
(standard_long/; 6 of 7), 15 round 12's head-importance clauses on GPT-2
medium and Pythia with C4 (importance_mha/; 7 of 7), 16 Voita et al.'s
confidence as they define it, sink or EOS token excluded (voita/; 5 of 6; marks
important heads in seven decoders and three OPUS-MT translation encoders;
V5 failed). Rounds 12, 14 and 15 computed confidence over all tokens; Voita
et al. exclude the end-of-sentence token, so never call the all-token
statistic theirs. 17 (audits/PREREGISTRATION17.md, audits/readings/): Target 1
in Clark et al.'s input format (same 64 percent) and Target 3 under CHAI's
every-row reading (97 percent Mistral, 43 percent OPT, C2 failed); 8 of 9.
18 (audits/PREREGISTRATION18.md, audits/rule/): a rule for which attention
claims survive nulls, six new claims; 4 of 6 (H2O heavy hitters mostly sink;
Scissorhands persistence and SnapKV hit rate survive against the rule; the
dev-informed mass-weighted versus count-based reading fits 6 of 6 and was
tested prospectively in round 23). 19 (audits/PREREGISTRATION19.md, audits/vit/):
ViT attention distance survives; artifact hypothesis fails; 3 of 5.
20 (alignment_study/PREREGISTRATION20.md, merge_bench/): KV-merge benchmark,
6,552 merges; simulated output perturbation 0.72, every similarity in use
<= 0.12; 5 of 5. 21 (alignment_study/PREREGISTRATION21.md, sink_arch/): sink
bias-like in Pythia and OPT, gate-like in GPT-2, near no-op in encoders;
constant virtual sink replaces the sink token in streaming for Qwen and
Pythia, not TinyLlama; 3 of 6. 22 (alignment_study/PREREGISTRATION22.md,
merge_bench2/): the benchmark in Pythia-1.4B, OPT-1.3B and GPT-2 large,
21,624 merges; simulation 0.68, similarities in use average <= 0.24; KL
damage tracks LAMBADA accuracy at Spearman 0.86 over 72 configurations;
task advantage over the CHAI-style choice in 5 of 6 models at one pair per
layer, 4 of 6 at two (D2 failed); 6 of 7. 23 (audits/PREREGISTRATION23.md,
audits/rule/flip/): five audited statistics computed weighted by attention
mass and by counted positions; mass s 0.52 to 0.95, counted s 0.002 to
0.27, gap >= 0.3 in all four decoder pairs; the occupancy refinement failed
on Clark's top-5 count (gap 0.37, F2); 4 of 5. 24 (alignment_study/PREREGISTRATION24.md,
gqa/, gqa_convention/): MHA-to-GQA conversion without training, nine recipes in
OPT-1.3B, Pythia-1.4B, GPT-2 large, OLMo-1B and Phi-2; aligned groups chosen by
the simulated merge lowest in 10 of 10 cells, 0.70 of Jin et al.'s
log-perplexity increase at g = 2 and +1.6 to +5.3 accuracy points; unaligned
pooling basis-dependent (spread 1.10 to 4.04), aligned not; the per-layer
simulated cost does not order the recipes in OLMo-1B or Phi-2 g = 4 (P6); 9 of
10. 26 (alignment_study/PREREGISTRATION26.md, fuse/; instrument fuse_common.py):
closed-form fusion of each group's key and value heads weighted by what the
heads compute (keys by each head's query second moment, rotary planes as complex
numbers; values by O_h^T O_h); basis-invariant, groups of one exact; beats round
24's best in 10 of 10 cells (dlog ratio 0.55 at g = 2, +2.9 to +6.6 accuracy
points) and the unweighted fit in 20 of 20; 6 of 6. 25 (alignment_study/
PREREGISTRATION25.md, xlayer/; run with the device fix 82cadd2): cross-layer KV
compression under the convention null; adjacent layers' cached K and V per-token
cosine within 0.005 of zero in four models (residual stream 0.90 to 0.94);
MiniCache's merge no better in the trained basis; XQuant's code statistic equals
independence within 0.003; B2 (aligned merging's size) and C3 (XQuant's sharing
better in the trained basis) fail; 5 of 7; the LLaMA-2-7B-80K derivative looks
misconfigured (continuation perplexity 84). 27 (alignment_study/PREREGISTRATION27.md,
fusekv/; instrument fuse_kv.py): the weighted fusion in models that already share
key-value heads, five models (TinyLlama, Qwen2.5-1.5B, OpenLLaMA-3B, Mistral-7B-
Instruct, Qwen2.5-7B); below averaging in 9 of 9 cells, dlog ratio 0.31, accuracy
+0.112; K2 fails (7 of 9): the weighting loses to the unweighted fit in the two
Qwen2.5 cells that merge every KV head into one; 3 of 4. 28 (alignment_study/PREREGISTRATION28.md, llama2/,
llama2_dev/; instrument llama2_round.py): the fusion against CAGA (Wang, Jiang and
Liu, arXiv 2610.09476) on its own Llama-2-7B benchmark, 32 to 16 KV heads without
training; pairs matched by the fusion's closed-form residual give 16.78 and 18.46
at 2,048 and 4,096 tokens (unconverted 5.47, 5.12) against CAGA's published 25.1,
ZS-7 0.504 against 0.443; CAGA's construction as the paper specifies it gives 323
to 1,348 in our harness and its pairing cannot be reproduced from the paper, so
compare constructions on identical pairs; steering agreement 0.30 against 0.05;
6 of 6. The conversion command is alignment_study/kvfuse.py (export to a real
grouped-query checkpoint: gqa_export.py). Exploratory follow-up to 28, not
preregistered (alignment_study/llama2_followup_*.py/.json/.log, llama2_quant_*,
llama2_dev/prefix160m.*; Figure 28): on pairs matched by CAGA's residual both
constructions collapse (12,142 and 11,398) through one merged pair, KV heads 8
and 18 of layer 1 once layer 0 is converted, which switches on layer 1's
super-weight neuron 7890 (down_proj[2533, 7890]) at ordinary tokens; per-layer
residuals and single-layer conversions cannot see it, one calibration window
can; averaging, CAGA's construction on other pairings and 3-bit round-to-nearest
quantization collapse other ways. 29 (alignment_study/PREREGISTRATION29.md, guard/,
guard_dev/; instrument guard_round.py): the same failure looked for in
Mistral-7B-Instruct, OpenLLaMA-3B, OLMo-1B and Qwen2.5-7B among conversions that
differ from the fusion's matching in one of layers 0 to 4 (forced-pair matchings);
gate FAILED (no member reached 5 times the reference; worst 2.21), G1 to G5
untested, 0 of 6; the one-window check (threshold 2.35) flagged none of 341
benign members; Llama-2-7B development: 7 damaged members, all pairing heads 8
and 18; post hoc, only Llama-2-7B has a dominant writer onto its super neuron's
gate (14.2 times the median; others 1.4 to 5.9), a hypothesis, not a result.
30 (alignment_study/PREREGISTRATION30.md, balance/, balance_dev/; instruments
balance_round.py, fuse_balance.py): rebalancing the fused weights (each shared
value row against its output columns, each key plane against its query planes,
to equal norms; same function) in six models for 4- and 3-bit rounding of the
attention projections and in OLMo-1B and h2o-danube-1.8b for uptraining (327,680
tokens, learning rates 5e-6 to 2e-4); 4 of 6: Q2 (4-bit lower in 6 of 6), Q3
(pooled 4-bit cost 0.48), T1 (best rebalanced run below best unbalanced) and T3
(below averaging at every checkpoint) pass; Q1 fails (rebalancing bfloat16
weights rounds twice: 0.7 and 4.2 percent in Llama-2-7B and Qwen2.5-7B; h2o
logits 3.1e-4) and T2 fails (at 5e-6 the unbalanced start ends lower). kvfuse.py
now fuses and rebalances in float32 and rounds once (--no-balance to skip); since
2026-10-31 it also matches groups of four (quad_groups: minimum-cost pairs, then
pairs of pairs by the four-head residual, group_cost) and handles projections
with biases (the Qwen2 layout; pair costs on the moment of [x, 1], the export
keeps the biases), unit-checked on tiny random models (kvfuse_tiny_check.py) and
on real ones (kvfuse_bias_check.py: Qwen2.5-0.5B exact; kvfuse_quad_check.py:
Llama-160m groups of four 265.6 against 339.7 neighbouring, checkpoint to 1.7e-4;
kvfuse_check.py: pair costs still equal round 28's). kvfuse_bench.py measures
an exported checkpoint's cache and decode speed (TinyLlama: 1.08x at batch 1).
31 (alignment_study/PREREGISTRATION31.md, PREREGISTRATION31_lambda_rule.md,
attnw/, attnw_fuse_dev/; instrument attnw_round.py): the fusion fitted in S_att
(each token weighted by the attention it receives) for keys, values and query
energies, five models; 0 of 3: A1 lower in 3 of 5 (OLMo-1B 54.7 to 44.2,
h2o-danube 38.8 to 28.8, Mistral 16.7 to 15.7; Llama-2-7B 16.0 to 16.6), A2
pooled 1.28 (0.903 over the four valid models), A3 Llama-2-7B whole test 17.235
against 16.78. Instrument defect: attnw_round.moments omits the ones column the
fusion needs for biased projections, so both Qwen2.5-7B arms are invalid
(plain 39,381 against round 30's 23.87); post hoc rerun attnw_biasfix.py
(attnw_correction/): plain 23.873 (round 30's value), weighted 27.776, so A1
still 3 of 5 and A2 0.945; post hoc, the gain tends to grow with the plain
fusion's cost (Spearman -0.69 over 8 models, Qwen2.5-7B against it). Any routine that builds second moments for the
fusion must append the ones column when kv.has_bias(l), as
fuse_kv.second_moments does. 32 (alignment_study/PREREGISTRATION32.md, frozen at bcdcd9d; rule
PREREGISTRATION32_rule.md, 9c17ad1; development attnw_mix_dev/, rule_result.json;
instruments attnw2_common.py, attnw2_round.py, eval_attnw2.py, checked by
attnw2_check.py): keys in S_w (first-order output sensitivity per token), query
energies weighted the same way, values in S_bar (BoA's value Hessian); eight
models (SmolLM2-1.7B, Llama-3.2-1B, Mistral-7B-v0.1 new; round 31's five); gate
passes (plain arms reproduce round 31 to four decimals), 2 of 4: D2 (new models
3 of 3 lower: 43.9 to 38.1, 94.6 to 80.6, 28.7 to 22.9) and D3 (pooled 0.929,
ceiling 0.94) pass; D1 fails (6 of 8; Llama-2-7B 16.0 to 18.4, Qwen2.5-7B 23.87
to 24.09) and D4 fails (whole test 19.31 against 16.78). The plain fusion stays
kvfuse.py's default. Post hoc diagnostic attnw2_llama2_diag.py (attnw2_diag/):
the sensitivity fit in layers 0-1 only gives 16.31 (plain 16.00; one-window check
1.01), so the early layers carry little of the loss and the super-weight check
sees nothing; layers 2 and up only 17.87, everywhere 18.37 (reproducing round 32), check at
most 1.06: the loss is in the deeper layers, not a super-weight event.
Quarter of the heads, development (quad_dev.py, quad_dev/): Llama-2-7B 32 to 8
KV heads collapses under the residual's groups of four (14,918, ZS-7 0.405) and
under neighbouring groups (18,953), both through layer 1 with round 29's
one-window check at 576 and 543 although heads 8 and 18 are apart; averaging
4,066 with the check at 0.93. The check's first true positives outside its
development population. Repair at the four-head level: quad_repair_dev.py
(forbid the flagged group, re-match, re-fuse; fallback: layer 1 at its matched
pairs). Mistral-7B-v0.1 at a quarter: 146 matched, 176 neighbouring, 813
averaging, check at most 2.1 (no hazard); at half 24.6. The repair on
Llama-2-7B failed: the event needs both head 8's group and the 11-and-18 group
merged; forbidding and re-matching three times fails; layer 1 at pairs with
layer 0 quartered still fires (check 360, 5,344). quad_fallback_dev.py sweeps
how many early layers must stay at pairs: one (layer 0 at pairs, the rest at
quads) passes the check at 0.98 and gives 105.5, ZS-7 0.408 at cache fraction
0.258 (quad_dev/Llama-2-7b-hf_fallback.json); layer 0's quartering is what
makes layer 1 fire. A per-layer ratio cannot be exported to a standard
Hugging Face config (one num_key_value_heads).
33 (alignment_study/PREREGISTRATION33.md, frozen at 9d73960; rule
PREREGISTRATION33_rule.md; instruments uptrain_round.py --registered --model,
eval_uptrain33.py, uptrain_rule.py; development uptrain_dev/llama-160m_5M.json:
averaging 51.6, fusion 30.8, round 32 fusion 30.4, trained unconverted 23.8 at
5M tokens, F = 0.1): uptraining at 5M tokens on OLMo-1B and h2o-danube-1.8b from
averaging, the rebalanced fusion and the rebalanced round 32 fusion, learning
rate 2e-5; U1 to U4; about 20 GPU hours, queued for 2026-11-01 to 02 behind
the quad run, its repair, chain 2 (calibration sizes, Qwen pairs, residual
against damage, whole-test values, the needle test) and a Llama-2-7B export
and benchmark (private/guards/run_pr33.sh). Run 2026-11-02 (uptrain/,
uptrain33_results.json): 2 of 4; U1 and U2 pass (the rebalanced fusion ends at
18.67 and 12.06 against averaging's 30.39 and 16.47 on OLMo-1B and h2o-danube,
leading at every checkpoint; zero-shot 0.503 and 0.512 against 0.444 and
0.456), U3 fails (retained log-lead 0.095 on OLMo-1B against 0.1; 0.120 on
h2o-danube), U4 fails (the sensitivity start 18.74 against the plain 18.67 on
OLMo-1B; 11.99 against 12.06 on h2o-danube). 34 (alignment_study/PREREGISTRATION34.md, frozen at c3cc48c after the Llama-160m
development test, quad_check_dev/; instruments quad_check_round.py and
eval_quad_check.py; run 2026-11-02 to 03, quad_check/, quad_check_results.json;
0 of 3: no false alarm in 29 benign scored members, no flag on Qwen2.5-7B's
four severe ones (layer 0 at pairs alone 4.3 times, quartered 10 times, check
0.97 to 0.98), the check's choice severe there and cheaper than the fixed rule
everywhere; Llama-2-7B reproduced its two collapses at 360 and 576; the check
detects the super-weight event only, never let it choose a ratio alone):
round 29's design at a quarter of the heads, layers 0 and 1 each unmerged, at
pairs or at groups of four with the rest quartered, damage relative to the
both-unmerged member; scored OpenLLaMA-3B, Mistral-7B-v0.1, Mistral-7B-Instruct,
Qwen2.5-7B; Llama-2-7B development; R1 to R3; runs after round 33 (runner
private/guards/run_pr34.sh <freeze permalink>). Development on 2026-11-01 (chain
2, labelled): calibration size (calib_dev/: 64 documents match 128), Qwen2.5-7B
pairs matched by the residual 70.1 against neighbours 23.9 (qwen_pairs_dev.json),
the residual against round 29's damage (residual_damage/: the populations hold
almost no damage; the parser now reads Qwen's matching labels), whole-test
perplexity for three 7B models (fulltest_dev/: the 8 windows overstate the
fused-to-unconverted ratio by a tenth to a quarter), and the needle test
(needle_dev/: Mistral-7B-Instruct retrieves a planted code 15 of 15 unconverted
and 2 of 15 after the plain fusion at half the heads, right form, wrong digits;
retrieval_pairs_check.py: the residual pairs each of the five strongest retrieval
key-value heads with a near-zero partner). The repair (retrieval_protect_dev.py,
retrieval_protect_dev/; run before round 33 on the author's decision): the
24 key-value heads serving retrieval heads at score 0.1 kept unmerged restore
15 of 15 at ppl 12.47 (cache 0.570), the 5 at 0.3 restore 14 of 15 at 13.93
(0.520), the strongest alone 7; clean random controls (heads and leftovers below
0.1) give 7 of 15 at five heads and 13 at 24; the residual's 24 costliest-to-pair
heads give 5. Retrieval under merging is graded; the retrieval heads are the
efficient exemption at small budgets; the paper's default (draft 3.8) exempts
them and reports retrieval beside perplexity. Never state a head-merging
conversion's quality from perplexity and ZS-7 alone. Random controls in
this design must restrict the odd-remainder leftover too (the leftover rule
picks the head whose exclusion makes the matching cheapest, which in layer 19
is the retrieval head). A10b and A10c (development, run 2026-11-03; residual_select_dev.py,
residual_pairs_dev.py, residual_select_dev/): over every matching of a layer
with whole-model KL as ground truth, the residual's choice sits at the median
in the early layers of five models (better half in 8 of 15, 1.77 times the
best) and is the worst of three on Qwen2.5-7B; a one-layer output perturbation
is near-best in 14 of 15 but fails in layer 0 of three models; KL is additive
over pairs (R^2 0.89 to 0.99); matching on measured single-pair KLs finds the
best or second-best in 10 of 12 layers in sample and, measured on two held-out
C4 windows, holds on Mistral-7B-v0.1 and Qwen2.5-7B and misses
Mistral-7B-Instruct's layer 0 (one catastrophic pair unflagged). The residual
is the fit criterion, not the selector; never let it choose groups alone.
The eight-window pass (run_a10cw.sh, 2026-11-04,
residual_select_dev/*_pairs_c4_w8.json): by the mean the measured matching
ranks 67, 20, 21, 23 of 105 on Mistral-7B-Instruct's layers 0, 1, 2, 16 and
28, 39, 48, 8 on Mistral-7B-v0.1's (two windows: 1, 1, 48, 8); by the
largest KL over windows 67, 1, 38, 45 and 1, 41, 48, 68: more C4
windows do not make the measured selector reliable, and the Instruct's
layer-0 pair (1, 5) costs 0.101 on WikiText against 0.001 on every C4 window
(the sequence-start effect, not an interaction). Pair diagnostics
(pair_text_diag.py, pair_start_diag.py, 2026-11-04; pair_text_diag/,
pair_start_diag/; Figure 39): the catastrophic pairs (1, 5) and (0, 1) of
Mistral-7B-Instruct's layer 0 serve the attention sink; their cost sits at the
sequence start, vanishes with the beginning-of-sequence token prepended (at
most 0.0012 nats per position on every window), appears on every window with
WikiText's opening tokens prepended, C4's included, and reaches every position
when it fires; the ranking and perplexity windows carry no such token
(bos_ppl_check.py, bos_ppl/: the plain fusion of Mistral-7B-Instruct at half the heads is at 16.68 on the 8 windows as they are and 13.48 with the token, ratio 3.01 against 2.39: 21 percent of its log-perplexity cost on these windows is the sink-start effect; within-round comparisons stand, absolute costs overstate the token-first case; on the base Mistral-7B-v0.1 the share is 3 percent, ratio 5.90 against 5.59: its cost is not the sink effect). Measure and
evaluate with the token as deployment prompts have it, or test sinkless starts
explicitly.
kvfuse.py's drafted --pairing measured is committed as an experimental option, labelled so in its docstring and not a
recipe (its Llama-160m check: kvfuse_measured_check.log). 35 (alignment_study/PREREGISTRATION35.md, frozen at 5def124 on 2026-11-04
after the development population on Mistral-7B-Instruct under the registered
protocol, retrieval_round_dev/: fusion 10 of 50, retrieval heads at 0.3 and
0.1 46 and 48, random five 7 and 12, random 24 38 and 6, the residual's five
19; instruments retrieval_protect_dev.py --registered and eval_retrieval35.py;
author's go 2026-11-02, after A10b): the retrieval-head exemption on Qwen2.5-7B and
Mistral-7B-v0.1, 50 instances per arm, E1 to E5; prerequisites run_pr35_prep.sh
(Mistral-7B-v0.1's scores, the population on the development model via --dev),
then the freeze, then run_pr35.sh <permalink> (started 2026-11-04 02:00 UTC;
retrieval_round/, retrieval_round_results.json; the eight-window pass
run_a10cw.sh and the measured-pairing check run_a10d.sh queued behind it, one
waiter per stage keyed on the previous waiter's PID). Run 2026-11-04 01:59 to
10:15 UTC (retrieval_round/, retrieval_round_results.json,
retrieval_round_answers.json by retrieval_answers.py): 2 of 5; the exemption
at 0.3 retrieves 7 of 50 on Qwen2.5-7B (plain 0, random 1 and 0, residual 0;
margin 10 missed) and 0 on Mistral-7B-v0.1 (every converted arm 0; no digit in 400 converted answers: the fused base model answers with newlines,
at which the protocol's decoding ends, or repeats; the decode diagnostic (retrieval_decode_diag.py, retrieval_diag/,
2026-11-04) shows a decode that stops only at the end of sequence retrieving
the same counts on every arm of the three models and the fused base model
continuing every prompt with one prompt-independent text: a loss of the context, not of the format; with the beginning-of-sequence token prepended to every prompt, retrieval_decode_diag.py --bos, the fused base model still retrieves 0 of 50 under both arms, so the collapse is not the sink-start failure below; Qwen2.5's tokenizer has no such token, so the test does not apply to it; the development model with the token: plain fusion 45 of 50 against 10 without, exemption 50 against 46: most of the Instruct's retrieval loss was the sink-start failure of token-less prompts; report retrieval under the deployed prompt format); E4 and E5 hold;
the development model 46 and 48 of 50 under the protocol, identical in the
rerun. The exemption repairs retrieval only where the conversion leaves the
model answering; the plain fusion at half the heads loses the planted code on
all three models (10, 0, 0) at perplexity ratios 3 to 13: report retrieval
beside perplexity, never present the exemption as a default repair. Follow-up
prepared, not run (author's decision): private/guards/run_fu35.sh, Qwen2.5-7B
under --pairing=neighbours with 50 instances. Watchdog lesson
(2026-10-31): its swap baseline is taken at start and goes stale as swap drifts;
a 4,000 MB growth limit killed a healthy 7B metrics job at 29 percent free
memory. Keep SWAPMAX at 8,000 and restart the watchdog (fresh baseline) before a
long chain; the 15 percent free-memory floor is the real protection.
Untrained 7B initializations are built on the
CPU and copied to the GPU; with two heavy runs the 65 W adapter and 32 GB
memory saturate (swap up to 37 GB), so queue 7B runs one at a time. Keep the
lid open during long GPU runs: closed, the GPU throttles to about a sixth of
its speed (2026-10-18). Do not run CPU probes alongside a GPU round. On a 45 W
adapter even one GPU job drains the battery (45 to 15 percent in two hours); the
battery guard pauses the job at 15 percent and resumes it at 30 (2026-10-28). On
MPS, never move a tensor to the host and to float64 in one call:
.to("cpu", torch.float64) returns corrupted values for large tensors in torch
2.13 (small ones raise); use .cpu().double() (2026-10-28). Another assistant
session may share this repository and GPU (2026-10-29: a parallel session's job
ran beside a 7B round and the watchdog killed both): before a GPU job or a
commit, check for running research jobs and coordinate; key waiters on process
IDs or new files, never on log lines that may already exist. A waiter whose own
command line contains its pgrep -f pattern matches itself and never exits (two
waiters deadlocked this way on 2026-10-31): wait on a PID, or bracket the pattern
(pgrep -f "[a]lignment_study/..."). Queue one waiter per stage, each keyed on the previous stage's process: a waiter that hard-wires several stages cannot take an insertion and had to be killed and replaced on 2026-11-01 (its running child survived a SIGKILL of the waiter). A waiter's no-job test must match the Python process, not any command line
that contains a script's path: pgrep -f '[P]ython -u alignment_study/'; on
2026-11-04 a parallel session's polling shell carried a script's path and
matched the path pattern, which would have held the queue until it exited. The memory watchdog
and the battery guard are private/guards/mem_watchdog.sh and battery_guard.sh
(gitignored, so a reboot keeps them; the scratchpad does not survive one); start
both before any GPU job; they match every alignment_study/*.py process. A third
guard, private/guards/disk_guard.sh, kills the job below 1.5 GB of free disk
(swap lives on the same volume; the disk was at 5 GB on 2026-11-01 with 7B runs
queued); start it too, and keep the model cache from filling the volume. A script
that loads several models must drop every reference, including views such as
fuse_kv.KV, before loading the next: on 2026-10-30 two float32 3B copies at once
panicked the machine.
Drafts for a TMLR submission and a position paper are in private/drafts/
(unpublished; the author decides on submission and posting); the scope for a
third, practical paper on the consumer-metric KV-head fusion (rounds 20 to 32)
is private/drafts/SCOPE_kvfusion_paper.md (2026-10-31); the gap and weakness
analysis with the plan for each is private/drafts/PLAN_kvfusion_gaps.md (B11:
training-free latent compression, TransMLA and GQLA, beats head merging on
published metrics; the author decides whether to extend the metric to latents);
the draft is private/drafts/paper_kvfusion/main.md with tables.md rendered by
alignment_study/make_paper_tables.py and intervals by paper_stats.py.
Instruments: functional_common.py, standard_common.py, audits/readings/readings_common.py.
Closest prior and concurrent work to cite: Papamichalis and Ruane (arXiv
2608.14712, sink convention and BERT clustering), Sandoval-Segura et al.
(ICLR 2026, output-norm head scores), Sok et al. (EMNLP 2026, BOS sink
heads), Méloux et al. (arXiv 2512.18792), Heap et al. (arXiv 2501.17727),
Fesser et al. (arXiv 2606.08105). For the KV-fusion line (rounds 20 to 33):
Ainslie et al. 2023 (GQA), Jin et al. (EMNLP 2025 Findings, aligned pooling with
distillation), CAGA (Wang, Jiang and Liu, arXiv 2610.09476; its code pairing not
reproducible), KQ-SVD, BoA (Kim et al., ICML 2025, attention-aware Hessians),
Nagel et al. 2019 (equalization), Yu et al. 2024 (super weights), and the
latent-cache family that reports smaller training-free losses at larger
reductions on its own task sets: TransMLA (Meng et al., arXiv 2502.07864; its
repository fxmeng/TransMLA no longer resolves, copies and an MLX port exist),
MHA2MLA (Ji et al. 2025), GQLA (Meng and Pan, arXiv 2605.15250), Palu (Chang et
al., arXiv 2407.21118), Eigen Attention (Saxena et al., arXiv 2408.05646), LoRC
(Zhang et al., arXiv 2410.03111). Never call CAGA "the best published
training-free conversion" without "head-merging". Audit base results: audits/clark2019/, audits/chai/ (Mistral-7B
and OPT-6.7B), audits/dewage2026/ (Mistral-7B), audits/retrieval_heads/
(Qwen2.5-7B, Mistral-7B-Instruct-v0.2 reproduced; Llama-2-7B-80K not
reproduced under our haystack and excluded); criteria in
audits/PREREGISTRATION4.md, frozen at commit ee4e267 and not edited since
(dated corrections go to audits/PREREGISTRATION4_ADDENDA.md; the OSF
registration was deferred by the author, the freeze permalink is the
registration of record); the batteries run with
--registered=<freeze permalink> and write audits/*/battery_result*.json
(per model for Targets 3 and 5); audits/eval_batteries.py (committed
before any result) regenerates audits/RESULTS.md and battery_results.json;
audits/retrieval_heads/inclusion_test.py applied the frozen rule to
Llama-2-7B-80K (excluded). alignment_study/clustered_stats.py gives the
model-clustered intervals quoted in the manuscript.
Registered failures are reported as failures in NOTE.md, README.md and
manuscript.md; never reframe them.

## Style

Direct, measured, no em-dashes, no hype. Claims sized to evidence. The paper
critiques methods, not people.
