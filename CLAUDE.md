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
quantization collapse other ways. Untrained 7B initializations are built on the
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
IDs or new files, never on log lines that may already exist.
Drafts for a TMLR submission and a position paper are in private/drafts/
(unpublished; the author decides on submission and posting).
Instruments: functional_common.py, standard_common.py, audits/readings/readings_common.py.
Closest prior and concurrent work to cite: Papamichalis and Ruane (arXiv
2608.14712, sink convention and BERT clustering), Sandoval-Segura et al.
(ICLR 2026, output-norm head scores), Sok et al. (EMNLP 2026, BOS sink
heads), Méloux et al. (arXiv 2512.18792), Heap et al. (arXiv 2501.17727),
Fesser et al. (arXiv 2606.08105). Audit base results: audits/clark2019/, audits/chai/ (Mistral-7B
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
