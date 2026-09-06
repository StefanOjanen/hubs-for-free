# Hubs for Free

![regenerate](https://github.com/StefanOjanen/hubs-for-free/actions/workflows/regenerate.yml/badge.svg)

**The attention heads of a trained language model layer are, to first
order, one shared operator.** At layers where attention concentrates on a
sink token, most of the energy of every head's skew-symmetric part lies on
a single shared component, and that component is the sink operator itself.
Head individuality lives in the residual. This repository contains the
proofs, the instruments that measure it, three preregistered studies across
thirteen models from 0.1B to 7B parameters, a null-model toolkit, and the
start of an audit program that runs the toolkit on published claims.

## Findings

**1. One shared operator, identified.** Stack the generators
G_h = (A_h - A_h^T)/2 of a layer's heads and take the top principal
component S. Across 230 high-sink layers of eleven models (Qwen2.5 0.5B to
7B base and Instruct, GPT-2 medium, Pythia-410m, TinyLlama-1.1B,
Mistral-7B, Phi-3-mini, OLMo-2-7B), S is the sink operator at every one of
them (cosine to the ideal sink generator 0.71 to 1.00, above 0.87 at 91
percent of layers), the shared-energy fraction runs 0.51 to 0.99, and the
alignment spans grouped-query KV groups, so the architecture does not
explain it. Instruction tuning leaves it unchanged.

**2. An exact cancellation, and what it means for "interaction"
analyses.** Writing G_h = a_h S + E_h, the shared-times-shared term cancels
identically in every commutator:
[G_i, G_j] = a_i [S, E_j] - a_j [S, E_i] + [E_i, E_j].
Measured on real models, the cross terms dominate the deviation term by 4x
to 25x at 100 percent of head pairs tested. Consequence: any pairwise
commutator statistic of trained attention is governed by the shared sink
mode and per-head deviations from it. A surrogate that preserves only each
row's sharpness, self-mass, and one number per row (the sink-column entry)
reproduces the real coupling statistics layer by layer, to three decimals
at the deepest layers of Mistral-7B. At this resolution these statistics
contain nothing else.

**3. Alignment, not concentration, is the mechanism.** In synthetic
ensembles and in redesigned controls, heads that are individually
concentrated but on different columns show the opposite signature: the
shared mode collapses, rank-1 norm-product geometry returns, and
commutators are elevated rather than suppressed. Confirmed on all five
sub-2B held-out models; at 7B the control loses power at 32 heads and the
dissociation is registered as unconfirmed at that scale.

**4. A one-parameter law that transfers.** The rank-1 statistic of the
coupling matrix is predicted by the layer's shared-energy fraction:
Spearman -0.76 pooled across the high-sink layers of five held-out models
at or below 1.5B, and -0.76 again in a separately preregistered 3B-to-7B
round (130 pooled layers), with the regime flip located where a fully
synthetic toy ensemble places it (shared energy 0.85 to 0.93). Three
numbers per head (a_h, e_h, and S) reproduce the r1 depth profile of the
development model at Spearman 0.81.

**5. The signatures come for free.** On random causal softmax matrices
the same pipeline produces a dominant hub, a large eigengap,
one-versus-(n-1) spectra, and 2-sigma "special" heads in 62 percent of
draws at 14 heads and 96 percent at 32. A formula search on structure
constants extracted from noise reaches the same fit quality as the
archived real-model search. On trained models the same signatures appear
for a different reason (shared-mode degeneracy rather than norm
heterogeneity); bare statistics cannot tell the two mechanisms apart, the
constrained surrogates here can. Two small proved results anchor this:
row-stochasticity forces one shared invariant direction on any attention
layer, and for causal attention 2||G||_F^2 = sum_i (IPR_i - A_ii^2)
exactly, so generator norm is sharp non-self attention and nothing more.

## Practical applications

Available now, in this repository:

- **Sanity-check an attention analysis before reporting structure.** The
  `hubsfree` toolkit runs any statistic computed on attention maps
  (clusters, hubs, eigengaps, special heads, similarity or interaction
  matrices, importance scores) against four null families: random causal
  or bidirectional softmax matched in heads and length, per-row
  marginal-matched surrogates, sink-column-preserving surrogates, and an
  untrained model of the same architecture. The report gives the real
  statistic's percentile in each null and how much of the effect the null
  reproduces. `hubsfree demo` shows the random-matrix "coordinator" in
  under a second.
- **Control for the sink before comparing heads.** The sink column
  dominates head-similarity, head-clustering, and interaction statistics.
  Any pipeline that ranks, clusters, or prunes heads from attention maps
  (attention-based importance, attention rollout, similarity-based head
  merging) should report the result with the sink column preserved in the
  null, or it is measuring how much each head attends to the sink token.
- **Base rates for "special component" claims.** The k-sigma rule flags a
  special head, neuron, or dimension among n candidates on pure noise at
  rates this repository tabulates (62 percent at n = 14, 96 percent at
  n = 32 for the 2-sigma rule); the same calculator applies to any outlier
  hunt among n components, not only attention.
- **Matched-noise floors for automated formula discovery.** Symbolic
  regression and other adaptive fits on model internals reach nontrivial
  fit quality on noise. The battery reports the best fit the identical
  search achieves on shape-matched noise, which is the line a real fit
  must clear. Directly relevant to "AI scientist" pipelines that fit laws
  to interpretability coefficients.
- **A quality protocol for AI-assisted analysis.** Artifact before
  citation, nulls before interpretation, thresholds frozen before runs,
  evaluation code committed before results, failures reported as
  failures. Applied here it caught a fabricated headline statistic (an
  R^2 of 0.982 absent from the project's own logs) and is documented
  end to end in the commit history and `paper.md`.
- **A reviewer's checklist** (paper.md, Section 6) for structure claims
  about attention, usable as is in peer review or internal model reports.

Registered but not yet demonstrated (planned tests, reported either way):

- **A per-layer redundancy diagnostic for inference efficiency.** The
  shared-energy fraction is one number per layer, computable from a dozen
  inputs, that measures how much of the heads' operator content is one
  shared component. Whether it predicts head-merging tolerance or
  KV-cache compressibility per layer is a preregistered test in the v2
  plan; if it does, it is a cheap targeting signal for compression and
  pruning. Until then it is a hypothesis.
- **Auditing published structural claims.** `audits/TARGETS.md` lists the
  first targets (Clark et al. 2019 head clustering, Kovaleva et al. 2019
  attention-pattern taxonomy, CHAI's cross-head redundancy, spectral
  outlier claims on projection weights, and Retrieval Heads as a positive
  control). Target 1's base result is reproduced and its null criteria
  are frozen in `audits/PREREGISTRATION4_DRAFT.md`; battery results
  follow, with author responses.

## Evidence standard

Three preregistrations with numeric thresholds and falsification clauses
were frozen in git before their runs (`alignment_study/PREREGISTRATION.md`,
`PREREGISTRATION2.md`, `PREREGISTRATION3.md`; the second was committed
before the held-out models were downloaded, the third pushed publicly
before execution, and evaluation scripts committed before results
existed). Registered predictions that failed are reported as failures, not
reinterpreted: the universal-breakdown claim (A1), the first dissociation
control (A7, design flaw documented and replaced), the sharp regime-edge
threshold (H4b), one-column sufficiency on TinyLlama (H2) and OLMo-2 (S2),
and the dissociation control at 32 heads (S3). Out of sample, 7 of 8
clauses passed in the sub-2B round and 3 of 4 in the 3B-to-7B round.
Additional hardening: a blind reimplementation from the written spec
reproduced the anchor values to four decimals; surrogate invariants hold
to 1e-15; per-layer bootstrap confidence intervals; adversarial review
whose objections (statistic ill-conditioning, GQA confound, circularity
risks) were tested and are answered in `alignment_study/NOTE.md`. The
synthetic battery regenerates from `experiments.py` in under a minute and
a CI job checks it on every push. Every number in every document
regenerates from a committed script into a committed artifact, with one
exception stated in `CLAUDE.md`: the small-model pilot in paper.md
Section 7, whose scripts are author-held pending upload.

## Relation to prior work

Attention sinks and their prevalence across heads are documented at the
distribution level (Xiao et al. 2023; Sun et al. 2024; Gu et al. 2025;
Barbero et al. 2025), as is similarity between heads' attention
distributions (Clark et al. 2019; Bian et al. 2021), and constrained
surrogate testing has a long lineage outside interpretability (Theiler
1992; Elsayed and Cunningham 2017; Adebayo et al. 2018). What is new here,
and was checked against the literature with adversarial search before
claiming it: the operator-level decomposition of a layer's heads and the
identification of its shared component with the sink; the commutator
cancellation and its measured dominance structure; per-row
marginal-matched and column-preserving surrogates for attention
statistics; the out-of-sample alignment-fraction law; and the toolkit that
packages the nulls.

## Repository map

- `paper.md` - methods paper: what commutator-based analyses of attention
  measure, with the full null battery and a case study.
- `experiments.py`, `results.json`, `figures/` - the synthetic null
  battery behind paper.md Sections 3 to 4.6; `check_results.py` and
  `.github/workflows/regenerate.yml` verify it regenerates.
- `alignment_study/` - the shared sink-operator studies: three
  preregistrations, all scripts, all result JSONs, run log, and the study
  note with scorecards (`NOTE.md`).
- `hubsfree/`, `tests/`, `pyproject.toml` - the null-model toolkit
  (v0.1.0.dev0): statistics, null families, battery report, CLI.
- `audits/` - audit targets, the preregistration-4 draft, and per-target
  reproductions (`clark2019/`).
- `run_qwen_protocol.py`, `qwen_results.json` - the six-prediction
  protocol on Qwen2.5-0.5B.

## Reproduce

```
./setup.sh                                  # pinned environment (requirements.txt)
source .venv/bin/activate
python experiments.py                       # synthetic null battery, under a minute on CPU
pip install -e . && hubsfree demo           # the random-matrix coordinator, one second
python -m pytest tests                      # surrogate invariants and anchors
python alignment_study/tier1_robust.py      # development-model study
python alignment_study/heldout_round.py     # five held-out models
```

Models and datasets download from Hugging Face on first use. The 3B-to-7B
round ran on a Colab T4 from `scale_colab.ipynb`; its captured outputs and
run history are in `alignment_study/scale_partial/` and
`alignment_study/RUNLOG.md`.

## Scope and open items

Results cover thirteen models up to 7B parameters, English text and code,
contexts up to 256 tokens. Where the sink sits on a mid-sequence,
window-varying column (OLMo-2, late Qwen2.5-3B layers) the sink-operator
identification weakens and one-column surrogates lose their grip; the
distinct-target dissociation control loses power at 32 heads. The
residual deviation directions carry structure not captured by the
three-number summary; one low-dispersion layer is ill-conditioned for the
correlation statistic (diagnosed, flagged). Larger models, other
languages, and long contexts remain untested. The v2 program (toolkit,
audits, derivation of the shared energy from sink statistics, training
dynamics, and the redundancy-diagnostic test) is in progress.
