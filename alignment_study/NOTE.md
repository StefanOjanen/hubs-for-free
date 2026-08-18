# Shared-mode structure of attention-head generators: study note

Date: 2026-08-18. Status: preregistered study executed and adversarially
reviewed; two registered predictions failed and are reported as failures.
Every number below regenerates from the committed scripts in this directory
into the committed JSONs. Prereg: `PREREGISTRATION.md` (committed at
1323911 / 09f58e2 before the confirmatory runs; the git history is the
ex-ante evidence). Exploratory basis in `exploratory/`.

## Result in one paragraph

In three trained causal language models (Qwen2.5-0.5B, GPT-2 small,
Pythia-160m; WikiText-103 validation, T = 64), the skew-symmetric parts
G_h of a layer's attention heads concentrate on one shared operator
component. At layers where attention concentrates on the first token, that
component is the sink operator itself (|cos(S, ideal sink generator)|
0.87 to 0.99 at all 18 Qwen layers with column-0 mass above 0.3), it
carries 73 to 93 percent of generator energy, and it spans Qwen's two
KV-head groups (within-group and across-group generator cosines are equal
at high-sink layers, e.g. 0.96/0.96 at layer 16, 0.90/0.90 at layer 21),
so grouped-query attention does not explain it. In the commutator
[G_i, G_j] the shared-times-shared term cancels identically, and the
surviving commutator is empirically dominated by the shared-times-deviation
cross term a_i[S,E_j] - a_j[S,E_i] (4x to 25x the deviation-deviation term
at all six layers tested, 100 percent of head pairs). Consequently the
coupling matrix C_ij = ||[G_i,G_j]|| on trained models departs from the
rank-1 norm-product geometry that random matrices and marginal-matched
surrogates obey (r1 = 0.98), and a surrogate that additionally preserves
one number per row, the sink-column entry, reproduces the real statistic
almost exactly, layer by layer, in all three models (Qwen breakdown
layers: sinkfix tracks real r1 to 0.01-0.06; GPT-2 late layers to 0.004;
48-window check: sinkfix matches real even at high-sink layers without
breakdown). At this resolution, commutator-based "interaction" analyses of
trained attention measure per-row sharpness, self-attention mass, and the
shared sink column, and nothing else.

## Scorecard against the preregistration

- A1 (universality of the rank-1 breakdown across layers): FAIL. D_l > 0.3
  at 14/24 Qwen layers (registered threshold 16). Per the registered
  falsification clause, the universal form of the claim is rejected. The
  narrowing to high-sink layers is post-hoc and labeled as such; the
  48-window addendum shows the layer heterogeneity is real, not sampling
  noise (D_l = 1.49 [1.44, 1.58] at layer 21 versus 0.12 [0.11, 0.12] at
  layer 4, bootstrap 95 percent CIs). High sink mass is necessary but not
  sufficient for the breakdown (layer 3: col0 0.61, no breakdown).
- A2 (sink sufficiency where breakdown occurs): PASS. |sinkfix - real|
  < 0.2 at 11/14 qualifying layers (79 percent). Exceptions: layers 1
  (no sink; different shared mode), 8, 15 (moderate sink).
- A3 (shared-mode order parameter): PASS. rho(real) at least 2x rho(plain)
  in 20/24 layers; Spearman(rho, col0 mass) = 0.80. Caveat from review:
  causal masking makes all generator inner products nonnegative, so rho
  has a floor from any shared row profile; the plain-surrogate comparison
  controls part but not all of this.
- A4 (cross-model): PASS. GPT-2: 7/10 qualifying layers; Pythia-160m: 7/8.
  Under the same criteria Qwen scores 11/19 (58 percent), so Qwen is the
  weakest under the uniform criterion; the registered A1/A4 asymmetry is
  reported, not hidden. GPT-2 and Pythia keep substantially positive real
  r1 at many layers; the cross-model claim is breakdown-plus-restoration
  at high-sink layers, not near-zero r1 generally.
- A5 (decomposition identity and bound): PASS. Identity to 2.2e-15; bound
  holds everywhere; median bound/actual 4.7 to 5.3 at the three
  highest-sink layers. Wording correction from review: only the
  shared-times-shared term cancels; the surviving terms are first order in
  the deviations, so "second order" is retired. The addendum measures the
  terms directly: the cross term dominates everywhere tested.
- A6 (identity of the shared mode): PASS. 18/18 layers. At layers 22-23
  (low sink) the shared mode persists but decouples from the sink
  (cos 0.31, 0.21); the shared mode is not always the sink.
- A7 (alignment versus per-head sinkiness): FAIL. Joint criterion 6/14
  (recomputed and emitted by `robustness_addendum.py`). The altsink
  control was flawed: t_h = h mod 4 leaves four aligned clusters, one of
  which keeps column 0, so the control cannot cleanly dissociate the two
  readings. The GQA analysis partially substitutes: alignment spans KV
  groups, and altsink does collapse rho (12/14) and partially restore r1
  (for example layer 20: -0.04 real to +0.73 altsink), but the registered
  dissociation was not achieved.

## What is new (checked against literature twice, with an adversarial
refutation pass)

Unpublished per the searches (roughly 40 queries and 7 full-text reads
across the two sweeps, all "not found after N searches" claims, not proofs
of absence): pairwise commutator-norm matrices between attention heads;
per-row marginal-matched permutation surrogates for attention statistics;
cross-head operator PCA of attention maps and its sink identification;
the cancellation of the shared-mode commutator term and its consequences;
the two-mechanisms point that hub/eigengap signatures arise from norm
heterogeneity on random matrices but from shared-mode degeneracy on
trained ones, distinguishable only by constrained surrogates. Documented
prior art that any write-up must cite: distribution-level cross-head sink
concentration and massive activations (Xiao 2023; Sun 2024; Gu ICLR 2025;
Barbero 2025), distribution-level head similarity (Clark 2019; Kovaleva
2019; Bian 2021), surrogate-data lineage (Theiler 1992; Elsayed and
Cunningham 2017), randomization controls (Adebayo 2018), rank collapse
from row-stochasticity (Dong et al. 2021), composition scores (Elhage
2021), random-baseline interpretability audits (Heap et al. 2025).

## Honest limitations

Models at or below 0.5B, one corpus, T = 64, column 0 as the sink proxy,
no instruction-tuned or long-context models. The commutator method whose
behavior is explained is this project's own construction (paper.md); the
write-up must say it introduces and critiques its own instrument class.
The layer-22 anomaly (the one layer where even the plain surrogate fails
rank-1, 0.66-0.87) is unexplained. The post-hoc residual-law conjecture
(C rank-1 in residual norms) was tested and rejected
(`posthoc_residual_law.json`: median r1_resid 0.39); the addendum's term
decomposition explains why (the cross term, not the deviation-deviation
term, dominates). Statistical inference is bootstrap-over-windows only;
windows within a corpus are not fully independent even at stride 20.

## Relation to paper.md

This study answers the constructive question Section 7 left open
("characterize the pairwise alignment that breaks rank-1"): the alignment
is the shared sink operator, its cross term with per-head deviations is
the commutator's dominant content, and one preserved number per row
reproduces the phenomenology. Integration into the paper is an editorial
decision not taken here.

## Hardening round (2026-08-18, tiers 1 and 2)

Dev calibration (`tier1_robust.py`, Qwen2.5-0.5B, 24 windows): the r1
ill-conditioning diagnosis confirmed (layer 22 has cv(gn) = 0.09, lowest of
all layers); the empirical sink column equals column 0 at every high-sink
layer, validating the proxy; the per-pair z statistic revealed that the
plain surrogate has its own shared mode (the diffuse row profile), so z is
non-monotone in alignment: positive at intermediate alignment, strongly
negative at extreme alignment. The toy model (`tier2_toy.py`) reproduces
exactly this non-monotonicity and localizes the r1/z regime flip between
shared-energy 0.85 and 0.93, which is where the real Qwen layers flip
(0.83 to 0.84 intact, 0.92 to 0.94 broken). The previously unexplained
A1 heterogeneity is therefore position on a one-parameter curve indexed by
alignment fraction. Toy dissociation: shared-column concentration drives
r1 to -0.98 as alignment rises; distinct-column concentration returns r1
to +0.95, collapses the shared mode, and elevates commutators. Two lemmas
verified: causal generator inner products are never negative (min 0 over
200 ensembles), and the closed form E[<G_h,G_k>] = (1/2) sum_i i m_hi m_ki
for the plain surrogate matches sampling to 0.5 percent. Generic-deviation
test: (S, a_h, e_h) with random deviation directions predicts the r1 depth
profile (Spearman 0.81, median abs error 0.08) but not z magnitudes
(Spearman -0.34): deviation directions carry real structure.

Held-out round (`PREREGISTRATION2.md` frozen at c7451a3 before any
held-out model was downloaded; evaluation script committed at 3e5b5cb
before results): five held-out models (gpt2-medium, Pythia-410m,
TinyLlama-1.1B, Qwen2.5-1.5B base and Instruct). Scorecard: H0 pass
(plain r1 > 0.9 in 96 percent of pooled layers). H1 pass 5/5 models, at
100 percent of high-sink layers in every model (median |cos(S, sink)|
0.96 to 0.99). H2 (one-column sufficiency) pass 4/5; TinyLlama fails as
registered (57 percent versus 2/3). H3 (dissociation) pass 5/5 (median
rho ratio altsink/real 0.14 to 0.25; median r1_altsink2 0.86 to 0.98).
H4 pass: pooled Spearman(sharedE, r1) = -0.76; H4b fails as registered
(the sharedE 0.90 regime edge does not transfer across architectures;
16 pooled edge layers, not all below r1 = 0.2). H5 pass (breakdown layer
count stable from T = 64 to T = 256). H6 pass (WikiText/HumanEval
breakdown-set Jaccard 0.5; code breaks at layers 11, 16, 21, a subset of
the WikiText set). Instruction tuning leaves all statistics essentially
unchanged. Protocol notes: the HumanEval dataset id needed its namespaced
form mid-run (resume script `resume_h6.py`, H0-H5 untouched); the sink
column detector gained a minimum-contributing-rows restriction relative
to the dev run (documented in PREREGISTRATION2.md).
