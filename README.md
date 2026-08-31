# Hubs for Free

**The attention heads of a trained language model layer are, to first
order, one shared operator.** At layers where attention concentrates on
an early token, 73 to 93 percent of the energy of every head's
skew-symmetric part lies on a single shared component, and that component
is the sink operator itself (cosine 0.87 to 0.99 against the ideal sink
generator). Head individuality lives in the residual. This repository
contains the proofs, the instruments that measure it, and two
preregistered studies, including an out-of-sample confirmation on five
models the thresholds never saw.

## Findings

**1. One shared operator, identified.** Stack the generators
G_h = (A_h - A_h^T)/2 of a layer's heads and take the top principal
component S. Across eight models (Qwen2.5-0.5B/1.5B base and Instruct,
GPT-2 small/medium, Pythia-160m/410m, TinyLlama-1.1B), at every high-sink
layer tested, S is the sink operator, the shared-energy fraction is 0.73
to 0.94, and the alignment spans grouped-query KV groups (within-group
and across-group generator cosines are equal, so the architecture does
not explain it). Instruction tuning leaves it unchanged.

**2. An exact cancellation, and what it means for "interaction"
analyses.** Writing G_h = a_h S + E_h, the shared-times-shared term
cancels identically in every commutator:
[G_i, G_j] = a_i [S, E_j] - a_j [S, E_i] + [E_i, E_j].
Measured on real models, the cross terms dominate the deviation term by
4x to 25x at 100 percent of head pairs tested. Consequence: any pairwise
commutator statistic of trained attention is governed by the shared sink
mode and per-head deviations from it. A surrogate that preserves only
each row's sharpness, self-mass, and one number per row (the sink-column
entry) reproduces the real coupling statistics layer by layer, to 0.004
in the best cases. At this resolution these statistics contain nothing
else.

**3. Alignment, not concentration, is the mechanism.** In synthetic
ensembles and in redesigned controls on all five held-out models, heads
that are individually concentrated but on different columns show the
opposite signature: the shared mode collapses (rho ratio 0.14 to 0.25),
rank-1 norm-product geometry returns (r1 0.86 to 0.98), and commutators
are elevated rather than suppressed.

**4. A one-parameter law that transfers.** The rank-1 statistic of the
coupling matrix is predicted by the layer's shared-energy fraction:
Spearman -0.76 pooled across the high-sink layers of five held-out
models, with the regime flip located where a fully synthetic toy
ensemble places it (shared energy 0.85 to 0.93). Three numbers per head
(a_h, e_h, and S) reproduce the r1 depth profile of the development
model at Spearman 0.81.

**5. The signatures come for free.** On random causal softmax matrices,
the same pipeline produces a dominant hub, a large eigengap, one-versus-
(n-1) spectra, and 2-sigma "special" heads at base rates up to 97
percent. On trained models the same signatures appear for a different
reason (shared-mode degeneracy rather than norm heterogeneity). Bare
statistics cannot tell the two mechanisms apart; the constrained
surrogates in this repository can. Two small proved results anchor this:
row-stochasticity forces one shared invariant direction on any attention
layer, and for causal attention 2||G||_F^2 = sum_i (IPR_i - A_ii^2)
exactly, so generator norm is sharp non-self attention and nothing more.

## Evidence standard

Both studies were preregistered with numeric thresholds and falsification
clauses frozen in git before the runs (`alignment_study/PREREGISTRATION.md`,
`alignment_study/PREREGISTRATION2.md`; the second was committed before the
held-out models were downloaded, and the evaluation script before the
results existed). Registered predictions that failed are reported as
failures, not reinterpreted: the universal-breakdown claim (A1), the
first dissociation control (A7, design flaw documented and replaced), the
sharp regime-edge threshold (H4b), and one-column sufficiency on one of
five held-out models (TinyLlama). Out of sample, 7 of 8 registered
clauses passed. Additional hardening: a blind reimplementation from the
written spec reproduced the anchor values to four decimals; surrogate
invariants hold to 1e-15; per-layer bootstrap confidence intervals;
adversarial review whose objections (statistic ill-conditioning, GQA
confound, circularity risks) were tested and are answered in
`alignment_study/NOTE.md`. Every number regenerates from committed
scripts into committed JSONs.

## Relation to prior work

Attention sinks and their prevalence across heads are documented at the
distribution level (Xiao et al. 2023; Sun et al. 2024; Gu et al. 2025;
Barbero et al. 2025), as is similarity between heads' attention
distributions (Clark et al. 2019; Bian et al. 2021), and constrained
surrogate testing has a long lineage outside interpretability (Theiler
1992; Elsayed and Cunningham 2017; Adebayo et al. 2018). What is new
here, and was checked against the literature with adversarial search
before claiming it: the operator-level decomposition of a layer's heads
and the identification of its shared component with the sink; the
commutator cancellation and its measured dominance structure; per-row
marginal-matched and one-column-preserving surrogates for attention
statistics; and the out-of-sample alignment-fraction law.

## Repository map

- `paper.md` - methods paper: what commutator-based analyses of
  attention measure, with the full null battery and a case study.
- `alignment_study/` - the shared sink-operator studies: preregistrations,
  all scripts, all result JSONs, study note with scorecards
  (`NOTE.md`).
- `run_qwen_protocol.py`, `qwen_results.json` - the six-prediction
  protocol on Qwen2.5-0.5B.

## Reproduce

```
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets numpy
python alignment_study/tier1_robust.py      # development-model study
python alignment_study/heldout_round.py     # five held-out models
```

Models and datasets download from Hugging Face on first use; runs
complete in minutes to about an hour on a laptop CPU.

## Scope

Results cover eight models up to 1.5B parameters, English text and code,
contexts up to 256 tokens. Known open items: the residual deviation
directions carry structure not captured by the three-number summary; one
low-dispersion layer is ill-conditioned for the correlation statistic
(diagnosed, flagged); larger models remain untested.
