# Shared-mode structure of attention-head generators: study note

Date: 2026-08-18. Status: preregistered study executed and adversarially
reviewed; two registered predictions failed and are reported as failures.
Every number below regenerates from the committed scripts in this directory
into the committed JSONs. Prereg: `PREREGISTRATION.md` (committed at
41769a9 / 6e57f84 before the confirmatory runs; the git history is the
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

Held-out round (`PREREGISTRATION2.md` frozen at 9ab1c28 before any
held-out model was downloaded; evaluation script committed at 39b8142
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

## Scale round (2026-09-01, preregistration 3, Colab T4)

Five models at 3B to 7B across five architecture families (Qwen2.5-3B/7B,
Mistral-7B-v0.1, Phi-3-mini-3.8B instruct, OLMo-2-7B), protocol and
thresholds frozen and publicly pushed before execution (PREREGISTRATION3.md);
run history and the v1-to-v2 implementation deviation in RUNLOG.md.
Scorecard: S1 (shared sink operator) PASS 5/5 models, at 100 percent of
high-sink layers in every model. S2 (one-column sufficiency) PASS 4/5:
100 percent in both Qwens and Phi-3, 87.5 percent in Mistral; OLMo-2
fails as registered (19 percent) because its sink sits on a mid-sequence
column that varies across windows, so a per-window pinned column does not
capture its alignment; under the registered falsification clause (2 or
more failures reject) the claim survives with OLMo documented as the
exception. S3 (dissociation) FAIL 2/5: the rho-collapse and
commutator-elevation clauses hold essentially everywhere, but the
registered r1_altsink2 > 0.8 clause fails at many high-sink layers of
the 32-head models; with targets t_h = h + 1 spanning half the causal
width at n = 32, late heads have most rows plain-permuted, so the
control's restoration weakens structurally with head count. Per the
registered clause, dissociation is unconfirmed at this scale under this
control; the synthetic and sub-2B dissociation results stand. S4
(alignment-fraction law) PASS: Spearman(sharedE, r1) = -0.76 pooled over
130 high-sink layers, matching the sub-2B round's -0.76. The law now
holds at 0.1B to 7B. New textures: Mistral is the most sink-aligned
model measured (shared energy up to 0.99, sinkfix tracking real r1 to
three decimals at its deepest layers); Qwen2.5-3B's last five layers and
all of OLMo-2 sink on mid-sequence columns rather than column 0, where
the ideal-sink identification weakens (cos 0.61 to 0.82), marking the
claim's boundary.

## Dynamics round (2026-09-10, preregistration 5, local MPS)

Pythia-160m and Pythia-410m at ten checkpoints (step0 to step143000),
protocol of rounds 2 and 3, predictions frozen and pushed (68b65f9) before
any checkpoint was downloaded. Development calibration on random-init
models from config (`dynamics_dev_calibration.json`, labeled, outside the
registered set) showed that an untrained network already has shared energy
0.88 to 0.96, carried by the uniform causal operator that every
near-uniform head shares, with cos(S, sink) 0.27, sink mass 0.06 and r1
0.15 to 0.47 (degenerate, near-identical heads). The predictions were
therefore written about the identity of the shared mode and the law, not
about the level of shared energy. Scorecard (`eval_dynamics.py` ->
`dynamics_results.json`; 360 cells, 152 high-sink): D0 pass (step0 max
sink mass 0.064, max cosine 0.276 in both models). D1 pass:
Spearman(sink mass, cos(S, sink)) = 0.935 [0.909, 0.957] over all cells.
D1b pass: cosine above 0.7 in 100 percent of the 152 high-sink cells. D2
pass: Spearman(shared energy, r1) = -0.736 [-0.802, -0.661] over high-sink
cells, against -0.76 in both converged rounds. D3 pass: the first
checkpoint at which cos(S, sink) exceeds 0.7 lies within one checkpoint of
the first at which sink mass exceeds 0.4 in 8 of 8 final-high-sink layers
of the 160m and 17 of 18 of the 410m. D4 (secondary) FAIL: the 160m's
first high-sink layer appears at step 8000, later than the registered
step 4000 (410m: step 2000). D5 (secondary, exploratory) pass: shared
energy dips below both its step0 and its final value by more than 0.05 in
75 percent (160m) and 96 percent (410m) of layers. Anchor: the step143000
cell of the 410m reproduces `heldout_round.json` to four decimals in sink
mass, shared energy and cosine, and r1 within 0.0002 (identical windows;
surrogate draws differ). Run log: `dynamics_round.log`; per-cell results
in `dynamics/`.

Post-hoc descriptions (`posthoc_dynamics.py` -> `posthoc_dynamics.json`;
labeled, not registered): (a) a generic interlude: at steps 512 and 1000,
before any sink exists, the layers sit in the random-matrix regime (410m
median r1 0.90 and 0.86, with 88 and 71 percent of layers above 0.8; 160m
0.81 and 0.77) while shared energy has fallen from about 0.9 at
initialization to 0.5 to 0.65; (b) sinks form abruptly, between steps
1000 and 2000 in the 410m (0 to 13 high-sink layers) and between 4000 and
8000 in the 160m (0 to 7), and cos(S, sink) exceeds 0.9 in the same
checkpoint; (c) along training r1 changes sign 14 times between
consecutive high-sink checkpoints, at interpolated shared energy 0.77 to
0.87 (median 0.83, IQR 0.81 to 0.85), against the aligned toy curve's zero
crossing at 0.785; (d) late reversals: four layers of the 160m (7 to 10)
lose 0.18 to 0.24 of shared energy between their peak (steps 16000 to
64000) and step143000 with sink mass steady, and r1 returns toward
positive values (layer 8: shared energy 0.82 to 0.58, r1 -0.01 to +0.64);
the 410m shows one such layer (22). Layer 0 of both models also declines,
but that continues the interlude at low sink mass and is not a reversal.
The converged state is therefore not the maximum-alignment state for
every layer; whether the reversals follow the learning-rate schedule is
open.

Reading: training does not create a shared component, it replaces one.
The uniform causal operator every untrained head carries is broken up
first; the sink then appears and becomes the shared operator within one
checkpoint; deeper alignment carries the layer through the regime flip
where the toy curve puts it. The alignment-fraction law holds at every
stage with a sink, not only at convergence.

## Instrument fixes (2026-09-10, development for preregistration 6)

Three weaknesses recorded in rounds 1 to 3 were addressed before the
rerun, each with a labeled development artifact.

Dissociation control (T3.2; `control_redesign_toy.py`,
`control_redesign_toy2.py`): the registered altsink-v2 plain-permutes the
rows above a head's target, so at 32 heads half the rows of the late heads
lose their geometry. Two alternatives were built: wrapped targets (every
row treated alike, targets a random permutation of 1..n per draw) and a
matched-geometry shift (each head's causal rows cyclically shifted by a
head-specific offset, no randomization; `hubsfree.surrogate_shift`). On
aligned toy ensembles at n in {8, 12, 14, 16, 32}, T in {64, 128, 256},
boosts 2 to 8: (i) the r1 restoration clause is limited by the causal
width, not by the control. At T = 64 no control restores r1 above 0.8 for
32 heads (median 0.51 to 0.68 at boost 8) although a genuinely misaligned
toy ensemble reaches 0.90 there, because half the rows cannot host 32
distinct targets; at T = 128 restoration is partial (0.38 to 0.75 of
trials at boost 8); at T = 256 it is complete for every control (8 of 8 at
n = 12, 16, 32). This is why S3 failed in round 3 at T = 64. (ii) The two
robust clauses, rho collapse below half and commutator elevation above the
real value, hold in every trial for the wrapped control at every (n, T)
with boost >= 4 (shared energy >= 0.67), for altsink-v2 in every row but
(32, 64, 4) at 7 of 8, and for the shift control only at boost >= 6: at
boost 4 its elevation clause fails in six rows (0 of 8 at n = 32,
T = 256), because keeping each head's row pattern also keeps the
plain-like part of the commutators. (iii) The shift control is the most
faithful on r1 at intermediate concentration, where the permuting controls
inflate it. Decision: wrapped targets as the primary S3' control on the
robust clauses at T = 256, r1 restoration secondary, shift and altsink-v2
reported alongside. Gate G2 of the plan passes on the toy at n = 32 under
the robust clauses and not under the r1 clause at T = 64.

Multi-column surrogates (T3.3; `multicolumn_dev_calibration.py`;
`hubsfree.sink_columns`): sink set = modal column plus columns whose
layer-level mass reaches 0.10, at most three. On the dev model every
high-sink layer selects one column, so the instrument reduces to the
registered one; on Qwen2.5-3B layers 3 to 30 likewise, while layers 31 to
33 select two columns and the column-set surrogate cuts R_r1 from 0.21 to
0.22 to 0.07 to 0.09 and raises the cosine to the sink-set operator at
layer 31 from 0.60 to 0.75. OLMo-2, the round-3 exception, waits for disk.

Primary statistics (T3.1): per-pair z against the plain ensemble and rho
excess over the plain surrogate are primary in preregistration 6; r1 enters
only the law (S4), restricted to layers with cv(gn) >= 0.1, with the
unrestricted value reported alongside.

## Rerun round (2026-09-10, preregistration 6, local MPS)

All thirteen models under one protocol with the fixed instruments: 48
windows from distinct documents at T = 64 (protocol A) and 6 windows at
T = 256 for the dissociation control (protocol B). Frozen and pushed
(3d5b337, 20:13 UTC) before execution; twelve models ran from 20:15 to
22:33 UTC, OLMo-2 from 22:35 to 23:25 UTC once the disk had freed
(RUNLOG.md). Qwen2.5-0.5B and Qwen2.5-3B calibrated the instruments and
are excluded from the tallies. Scorecard (`eval_rerun.py` ->
`rerun_results.json`; 11 testable models, 207 high-sink layers in
protocol A, 182 in protocol B):

- S1' pass 11 of 11, at 100 percent of high-sink layers in every model.
- S2' pass 11 of 11: R_z < 0.4 at 71 to 100 percent of high-sink layers.
  TinyLlama, the round-2 exception at 57 percent with one column, is at
  71 percent; OLMo-2, the round-3 exception at 19 percent, is at 79.
- S3' pass 11 of 11, at 100 percent of high-sink layers in every model.
  With the wrapped-target control at T = 256 the shared mode collapses to
  a median 0.07 to 0.09 of the real rho and the median per-pair z rises
  from 78 to 271 (Mistral-7B, 32 of 32 layers), 88 to 295 (Phi-3-mini, 27
  of 27) and 43 to 184 (OLMo-2, 11 of 11); r1 is restored above 0.8 at 96
  to 100 percent of layers (secondary clause).
- S4 pass: Spearman(shared energy, r1) = -0.70 [-0.76, -0.62] over the 182
  layers with cv(gn) >= 0.1 (25 ill-conditioned layers excluded), -0.75
  over all 207.
- Secondary clauses: R_r1 < 0.4 at 89 percent or more of high-sink layers
  in every model. The shift control passes the robust clauses in 11 of 11
  models, Pythia-160m lowest at 75 percent with r1 restored at 50 percent,
  as the toy predicted for that control. altsink-v2 at T = 256 passes the
  robust clauses in every model with r1 restored at 96 percent or more,
  which confirms that its round-3 failure was the 64-token causal width.
- Sanity anchor: the development model's per-layer values at 48 windows
  agree with the 12-window tier-1 run (Spearman across layers 0.99 for
  r1, shared energy, sink mass and z; largest difference 0.12 in r1) and
  its broken-layer set {9, 11, 16, 17, 20, 21} is identical.

OLMo-2 texture: the modal sink column is 0 at every high-sink layer, but
the modal column changes across 5 to 22 of the 48 windows per layer, and
the sink set has two columns in the median window (three at most). The
set captures 0.59 to 0.98 of column mass against 0.41 to 0.66 for the
single column and lifts the cosine to the sink-set operator to 0.88 to
0.91. Four of its 19 high-sink layers still have R_z above 0.4 (layers
25, 26, 29, 30: 0.50 to 0.85), so the column set resolves most, not all,
of the window-varying sink.

Reading: the three registered failures of the earlier instruments (S2 on
TinyLlama and OLMo-2, S3 at 32 heads) resolve under the fixed instruments
without moving any threshold, and S1 to S4 hold in 11 of 11
out-of-calibration models under one protocol.

## Structured deviation toy (2026-09-11, development for T4.3, labeled)

`toy_structured_dev.py` -> `toy_structured_dev.json`, development model, 8
windows, T = 64, 6 rebuilds per window. Each layer's heads were rebuilt
from their real coefficients on the shared operator S alone (the generic
construction of the hardening round), on S plus the previous-token
operator, and on S plus the previous-token and uniform operators, with a
random skew remainder of the right norm in each case; z was measured
against the real layer's plain-surrogate reference. Result: none of the
three recovers the z profile (Spearman with the real per-layer z: -0.38,
-0.35, -0.34; median absolute error 16 to 19 z-units) although all three
reproduce the r1 profile (Spearman 0.76 to 0.81). At high-sink layers the
rebuilt z is far more negative than the real one (layer 16: -21 against
-0.1; layer 21: -21 against -6), so real deviation directions produce
larger commutators than random directions of the same norm, and the two
dictionary operators carry too little energy there (previous-token 0.1
percent, uniform 0.1 percent at layers 16 and 21) to matter. Reading: the
deviation structure that sets z is head-specific and is not the recency or
uniform mode; a per-head dictionary (each head's own secondary column or
band) is the next candidate. T4.3 stays open; the r1 profile remains
predicted by (S, a_h, e_h) alone.

Follow-up diagnostic (`deviation_structure_dev.py` ->
`deviation_structure_dev.json`, labeled development): comparing each head's
real deviation E_h = G_h - a_h S with random skew matrices of the same norm
orthogonal to S, the real deviations commute with the shared operator about
three times more strongly at high-sink layers (normalized ||[S, E_h]|| 0.51
to 0.57 against 0.18 for random, ratio 2.9 to 3.2; at the low-sink first
three layers the ratio is 0.9 to 1.0) and are far more concentrated (inverse
participation ratio 9 to 25 times the random value), while no single
head-specific column carries them (top non-sink column share 5 to 11
percent against 2 for random). Reading: the structure that random rebuilds
miss is the concentration of the deviations, a consequence of peaked
attention rows, not a further low-dimensional operator. The next toy for
T4.3 is a sparse deviation with matched concentration.
Third construction (`toy_sparse_dev.py` -> `toy_sparse_dev.json`): a sparse
random deviation on a random causal support whose concentration matches the
real deviation's (support fraction 1 to 4 percent of causal entries) does no
better than the dense one (Spearman with real z -0.38, median absolute error
19.5 z-units; r1 profile still reproduced at 0.79). Matched concentration on
a random support does not raise the commutator with S, so the real
deviations' threefold larger [S, E_h] comes from where their entries sit
relative to the sink geometry, not from how many there are. T4.3 remains
open with that constraint recorded.

## Head-merging round (2026-09-11, preregistration 7, local MPS)

The practical test (plan T5.1). Qwen2.5-1.5B, 3B and 7B; per layer, the
two query heads of a KV group with the highest generator cosine merged
(query and output slices averaged) against the bottom-cosine pair and a
random within-group pair; cost = change in mean next-token NLL over 16,384
held-out WikiText-103 test tokens; predictor = the layer's shared energy
from the rerun. Frozen and pushed (6b8f8ea, 08:19 UTC) before execution;
run 08:21 to 13:25 UTC (`merge_round.log`; the 1.5B run was relaunched
without cache purging after its model completed, RUNLOG.md). Scorecard
(`eval_merge.py` -> `merge_results.json`, 92 layers):

- M1 FAIL (1 of 3 models): the top-cosine merge is cheaper than the
  bottom-cosine merge in 75 percent of layers (1.5B), 64 percent (3B) and
  54 percent (7B); the registered 2/3 holds in one model.
- M1b FAIL (secondary, 2 of 3): cheaper than the random pair in 71, 47 and
  64 percent of layers.
- M2 FAIL: pooled Spearman(shared energy, top-merge cost) = +0.13
  [-0.10, +0.32]; per model -0.00, +0.18, +0.20. M2b (depth partialled):
  -0.01. Shared energy does not predict a layer's merge tolerance.
- M3 FAIL (secondary, 1.5B, 180 pairs in six layers): Spearman(pair
  cosine, cost) = -0.28 pooled, -0.21 to -0.54 per layer; the direction is
  right and the effect small.
- Reported without threshold: the median top-merge cost is 0.0008 nats per
  token in the 1.5B (bootstrap 0.0005 to 0.0011), 0.0007 in the 3B and
  0.0013 in the 7B, and the top merge costs under 0.002 nats (about 0.2
  percent perplexity) in 82, 92 and 71 percent of layers. Median costs of
  the random and bottom pairs are of the same order (0.0006 to 0.0021). The
  one expensive merge in 92 layers is a random pair at layer 0 of the 1.5B
  (0.23 nats); layer 0 of the 1.5B is also the only layer where the top
  merge costs above 0.01.
- Weight restoration verified after every merge (baseline recomputed
  exactly).

Reading: within a KV group, merging one pair of heads costs little at almost
every layer of these models, and neither the attention-map cosine nor the
layer's shared energy says which pair or which layer. The shared-operator
description is about attention geometry; it carries no tolerance
consequence at this granularity, and the README's caution that the fraction
is a measurement, not yet a tool, stands as the result. Development
calibration on Qwen2.5-0.5B (top cheaper than bottom in 23 of 24 layers,
Spearman -0.32) did not transfer to the registered models; that gap is
itself the finding of the round.

## Derivation round (2026-09-11, preregistration 8, gate G3)

Twelve models, 310 layers (231 high-sink), 12 windows at T = 64; seven
models on the CPU from 08:26 UTC, the five larger ones on the GPU from
13:25 UTC after the merge round, all under the file frozen at b9b8ef0
(08:24 UTC). Scorecard (`eval_derivation.py` -> `derivation_results.json`):

- G1 FAIL (the gate): pooled over the 231 high-sink layers, the sink-only
  derived value explains 55 percent of the variance in measured shared
  energy (linear R^2 0.55, bootstrap 0.42 to 0.69), against the registered
  0.8. The interim value on the first seven models was 0.92; adding
  Phi-3-mini (per-model R^2 0.14), Qwen2.5-3B (0.34), OLMo-2 (0.87) and
  Qwen2.5-7B (0.93) lowered the pooled fit because the models sit on
  different offset lines: per-model intercepts run from 0.14 to 0.74, and
  in Phi-3 and Qwen2.5-3B the measured range across high-sink layers is
  narrow (0.71 to 0.97) relative to the gap. Per model, R^2 is above 0.86 in
  seven of twelve (gpt2 0.98, gpt2-medium 0.94, Pythia-160m 0.94,
  Pythia-410m 0.99, Mistral 0.93, Qwen2.5-7B 0.93, OLMo-2 0.87).
- G2 PASS: over all 310 layers the derived value orders the layers by
  shared energy at Spearman 0.86.
- G3 PASS (secondary): the derived value sits at or below the measured one
  in 100 percent of high-sink layers (median gap 0.058), as the optimality
  argument requires.
- G4 PASS (secondary): adding the uniform causal operator improves the
  identity R^2 over all layers in 12 of 12 models and lowers the pooled
  median absolute error to 0.039; at low-sink layers the sink-only value
  (median 0.20) is far below the measured (0.66) and the two-operator value
  (0.50) closes about 60 percent of that gap.

Reading, per the registered clause: the framing stays "measured structure
with a derived lower bound". Each head's sink mass and row sharpness fix a
floor under the layer's shared energy that lies within 0.06 of the measured
value at the median high-sink layer and orders layers correctly across
models, but the residual above the floor is not a fixed fraction and is not
explained by the uniform operator either; the deviation structure recorded
in the toy sections is the same open item seen from the other side.
