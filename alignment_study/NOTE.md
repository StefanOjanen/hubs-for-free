# Shared-mode structure of attention-head generators: study note

Date: 2026-08-18. Status: preregistered study executed and adversarially
reviewed; two registered predictions failed and are reported as failures.
Every number below regenerates from the committed scripts in this directory
into the committed JSONs. Prereg: `PREREGISTRATION.md` (committed at
6c84320 / 719b0dd before the confirmatory runs; the git history is the
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
