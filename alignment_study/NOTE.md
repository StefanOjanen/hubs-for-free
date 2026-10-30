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
  energy (linear R^2 0.55, per-layer bootstrap 0.42 to 0.69, model-clustered
  0.24 to 0.93), against the registered 0.8. The interim value on the first seven models was 0.92; adding
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

## Audit program status (2026-09-11)

Base results reproduced and criteria frozen (`audits/PREREGISTRATION4.md`);
no battery has run, since the plan gates the first battery on public
registration of the draft.

- Target 1, Clark et al. 2019: reproduced 2026-09-06 (same-layer JS 0.253
  against 0.380 across layers; nearest neighbor in the same layer for 38.9
  percent of heads, chance 7.7). Criteria frozen.
- Target 2, Kovaleva et al. 2019: not reproducible in stated form
  (classifier and annotations unreleased); logged in TARGETS.md.
- Target 4, Dewage et al. 2026: reproduced on Mistral-7B-v0.1 from the
  paper's recipe (their code URL unreachable on two dates). Outlier counts
  per matrix match Table II to within 0.3 percent (Q 1511, K 341, V 212,
  O 1450); the paper's percentages are the outliers' share of spectral
  energy (Q 89.3, K 75.7, V 43.0, O 84.4 here against 87.5, 74.7, 43.6,
  84.6), not the share of singular values (21 to 37 percent). Criteria
  frozen with four nulls including config-initialized weights.
- Target 3, CHAI 2024: partially reproduced on Mistral-7B (32 C4 documents,
  T = 1024): high cross-head correlation of last-token attention rows
  everywhere and large 0.95-clusters in layers 1 to 6 (majority clusters in
  layers 1 to 4), but a majority
  cluster in only 4 of 32 layers and no increase with depth (Spearman -0.38);
  the correlation tracks the last row's sink mass. On the paper's own OPT
  family (`facebook/opt-6.7b`) the split reverses: the depth trend holds
  (Spearman 0.49, 0.34 to 0.70) and no layer has a majority cluster (largest
  0.22 of heads). Criteria frozen for both models; the column-set surrogate is
  expected to match both statistics.
- Target 5, Retrieval Heads 2024 (positive control): reproduced on
  Qwen2.5-7B at 1K and 2K contexts (20 instances): 4.2 percent of heads above
  0.1 (paper 3 to 6), needle retrieved every time; on the paper's own
  Mistral-7B-Instruct-v0.2, 3.4 percent and again every needle retrieved;
  on the paper's primary Llama-2-7B-80K our setup retrieves only 10 of 20
  needles and finds 0.7 percent of heads above 0.1 (haystack, prompt and a
  loading deviation are the suspects), so that model is excluded from the
  battery. Criteria frozen for the two reproduced models.

With Targets 1, 3, 4 and 5 reproduced or partially reproduced and Target
2 logged, PREREGISTRATION4.md is ready to freeze on OSF; the
batteries run after that. The battery scripts for Targets 1, 3, 4 and 5
(`audits/*/battery.py`) are committed ahead of any result, each verified
only by a dry run that writes outside the repository (random maps or
matrices for 1, 3 and 4; the development model at 256 tokens for 5, whose
battery is computed online during generation so no attention rows are
stored).

Post-hoc description of the derivation residual (labeled, computed after
the scorecard from `derivation/*.json`): within every model the gap between
measured and sink-derived shared energy falls as the layer's sink mass
rises (Spearman -0.62 to -0.95 per model, -0.84 pooled over the 231
high-sink layers; gap quartiles 0.035, 0.058, 0.099), so the floor is
tight where the sink is strong and loose where it is weaker. Across
models the median gap falls with mean sink mass (Spearman -0.81) and is
largest in OLMo-2 (0.31), whose sink sits on a window-varying mid-sequence
column that the ideal generator at the modal column misses; a sink-set
version of the derivation is the natural repair to test. The pooled R^2
is low mostly because the residual varies with sink mass within models
and because two models (Phi-3-mini, Qwen2.5-3B) span a narrow measured
range; where the range is wide the sink-only floor explains 0.87 to 0.99
of the variance.
Sink-set version (`derivation_sinkset_posthoc.py` ->
`derivation_sinkset_posthoc.json`, labeled post hoc): building the ideal
generator on the layer's sink set instead of the modal column halves
OLMo-2's median gap (0.31 to 0.17; all 16 of its high-sink layers have a
set of two or more columns), leaves every single-column model unchanged,
and raises the pooled high-sink R^2 from 0.55 to 0.75 (Spearman 0.78 to
0.83) with the pooled median gap unchanged at 0.058. The registered gate
was set on the single-column derivation and stays failed; the sink-set
derivation is the version to register next, with the residual's dependence
on sink mass as its stated limit.
Where the deviation energy sits (`deviation_column_dev.py` ->
`deviation_column_dev.json`, labeled development): at the development
model's high-sink layers a median 54 percent of each head's deviation
energy E_h = G_h - a_h S lies in the sink column itself (its entries and
their skew mirrors), against 3 percent for a random skew matrix of the
same norm; the previous-token band holds 11 percent (2 to 15 percent at
high-sink layers, 19 to 35 percent at the first three layers, where the
sink is absent). The dominant deviation is therefore each head's own
sink-column profile, the way its mass over rows departs from the shared
profile shape. That is what commutes strongly with S, since both live in
the same column, and it is exactly what the sink-column surrogate
preserves and the marginal-matched surrogate destroys, which is why the
former reproduces z and the latter does not. T4.3 closes at this
resolution: the structured deviation is the per-head sink profile, and a
rebuild that keeps each head's real sink column and randomizes the rest is
the construction to register.
Fourth construction (`toy_sinkprofile_dev.py` -> `toy_sinkprofile_dev.json`):
keep each head's real sink column (its entries in column c and the skew
mirror, shared part included) and rebuild everything else as a random skew
matrix with zero entries in that column and row, scaled to the remaining
norm. This recovers the z profile that the dense, structured and sparse
rebuilds missed: Spearman with the real per-layer z 0.95 (dense -0.32),
median absolute error 1.1 z-units (dense 18.7), with the r1 profile kept
(Spearman 0.86, error 0.05). At high-sink layers the sink column holds 0.85
to 0.97 of each head's generator energy, at the low-sink first layers 0.10
to 0.28, and the rebuild is accurate in both regimes (layer 3: z 10.4 real,
10.1 rebuilt; layer 16: -0.9 and -1.0; layer 0: -1.4 and -0.9). Reading: at
the resolution of these statistics a layer's heads are their sink-column
profiles plus noise; the shared operator, the deviations that set z, and
the success of the sink-column surrogate are three views of that one
fact. This is the construction to register as the generative model of
T4.3, with the prediction that it reproduces r1 and z per layer on the
twelve other models.

## Sink-profile round (2026-09-11, preregistration 9, the generative model)

Twelve models, 310 layers; frozen and pushed (6e6b253, 14:32 UTC) before
execution, run 14:32 to 15:33 UTC on the M1 Max. Each head rebuilt from its
real sink column alone plus a random skew remainder of the remaining norm
with zero entries in that column and row; the dense rebuild (a_h S plus a
random remainder orthogonal to S) alongside. Scorecard
(`eval_sinkprofile.py` -> `sinkprofile_results.json`):

- P1 PASS 11 of 12: Spearman between real and rebuilt per-layer z is 0.85
  to 0.99 in eleven models (dense rebuild -0.66 to +0.82) and 0.31 in
  TinyLlama.
- P2 PASS 11 of 12: median absolute error of the rebuilt z is 0.5 to 1.5
  z-units in ten models and 4.5 in OLMo-2 (dense 15 to 22); TinyLlama's
  2.2 passes the error clause while failing the ordering clause.
- P3 PASS 11 of 12 (secondary): Spearman for r1 is 0.80 to 0.99; TinyLlama
  0.56.
- P4 PASS: pooled over 310 layers, Spearman(real z, rebuilt z) = 0.95
  (dense 0.13).
- Reported: the sink column holds a median 0.63 (OLMo-2) to 0.96
  (Mistral) of each head's generator energy at high-sink layers.

Post hoc, labeled (`posthoc_sinkprofile.py`): over the 231 high-sink layers
the rebuilt z matches the real one at Spearman 0.98 with a median error of
0.75 z-units; over the 79 low-sink layers 0.78 and 1.8. TinyLlama's failure
is confined to its fourteen low-sink layers (sink mass 0.03 to 0.29), where
the real z is -3 to -12 without a sink to explain it while the rebuild
gives -0.5 to -2; at its seven high-sink layers the rebuild follows the
real values (layers 4 to 8: -3.0, -9.1, -12.1, -5.3, -0.6 real against
-9.5, -10.9, -12.8, -7.2, -1.1). OLMo-2's larger error comes from its
window-varying sink column, as in every round.

Reading: at the resolution of these statistics a layer's heads are their
sink-column profiles plus noise. The shared operator (rounds 1 to 6), the
commutator cancellation, the sink-column surrogate's success, the derived
floor under the shared energy (round 8) and the deviation structure that
the toy constructions missed are one fact seen five ways: everything the
interaction statistics measure about a trained layer is in the column of
the sink token, head by head. Where a layer has no sink (TinyLlama's late
layers), the statistics measure something else, and that is the stated
boundary.

## New attention designs (2026-09-11 to 12, preregistration 10)

Four open models with attention designs absent from the thirteen measured
before: Falcon-7B (multi-query attention, one key/value head for 71 query
heads), BLOOM-7b1 (ALiBi position bias, no rotary), OPT-6.7B (learned
absolute positions, plain multi-head attention) and Qwen3-8B (the newest
Qwen generation). Frozen and pushed (de18d52, 16:40 UTC) before Falcon,
BLOOM or Qwen3 was downloaded; run 19:38 to 22:29 UTC under the exact
protocols of preregistrations 6 and 9 (`rerun_pr10/`, `sinkprofile_pr10/`;
Falcon's 71 heads took 51 minutes in the sink-profile protocol alone).
Scorecard (`rerun_pr10_results.json`, `sinkprofile_pr10_results.json`):

- All four form a first-token sink: 28 (Falcon), 19 (BLOOM), 29 (OPT) and
  29 (Qwen3) high-sink layers in protocol A, the modal sink column being
  column 0 at every one of them.
- N1 PASS 4 of 4: S1' at 100 percent of high-sink layers in every model;
  cosine of the shared component to the ideal sink operator 0.98 to 1.00
  (Falcon), 0.94 to 0.99 (BLOOM), 0.93 to 1.00 (OPT), 0.91 to 1.00 (Qwen3).
- N2 PASS 4 of 4: S2' at 82, 100, 76 and 97 percent of high-sink layers.
- N3 PASS 4 of 4: S3' at 100 percent of high-sink layers of protocol B in
  every model, r1 restored at 96 to 100 percent; BLOOM keeps only 6
  high-sink layers at T = 256 (its sink weakens with length), all passing.
- N4 PASS: Spearman(shared energy, r1) = -0.84 [-0.88, -0.78] over the 96
  high-sink layers with cv(gn) >= 0.1 (-0.87 over all 105).
- N5 PASS 4 of 4: the sink-profile rebuild reproduces the per-layer z at
  Spearman 0.90 (OPT), 0.98 (BLOOM), 0.98 (Falcon) and 0.96 (Qwen3) with
  median errors of 0.6 to 1.1 z-units (dense rebuild -0.81 to +0.79 and 17
  to 21), pooled 0.97 over 130 layers; the sink column holds 0.75 (BLOOM) to
  0.96 (Falcon) of each head's generator energy at high-sink layers. The
  secondary r1 clause holds in 3 of 4 (BLOOM 0.48).

Textures: BLOOM's shared energy is the lowest of any model measured (0.57
to 0.69 at high-sink layers) while its sink cosine is as high as the rest,
so ALiBi gives a weaker but equally sink-shaped shared operator; Falcon's
multi-query design, where every head shares one key/value set, has the
strongest sink profiles (0.96 of energy) and the cleanest rebuild (error
0.56). Reading: the shared sink operator, the column-set sufficiency, the
dissociation and the alignment-fraction law now hold across rotary,
learned-position and ALiBi positions and across multi-head, grouped-query
and multi-query attention, seventeen models in ten families, and the
generative statement of preregistration 9 holds in fifteen of sixteen
models tested (TinyLlama's low-sink layers remaining the one exception).

## Audit batteries (preregistration 4, registration of record: commit ee4e267)

Running from 2026-09-15 under `--registered=<freeze permalink>`; the
evaluation script `audits/eval_batteries.py` and the readings of the frozen
text it applies (`PREREGISTRATION4_ADDENDA.md`, addendum 4) were committed
before any result existed. `audits/RESULTS.md` is regenerated from the
result files; this section records the outcomes as they complete.

- Target 5, Qwen2.5-7B (positive control, E3): 4.2 percent of heads above
  0.1, mean top-10 score 0.43. Random rows, marginal-matched rows and
  sink-column-preserving rows put the real values at the 100th percentile
  of 200 draws and reproduce 0.008 to 0.011 of the top-10 score; the three
  untrained initializations score zero. E3 holds on this model: the
  battery does not remove a head set that copies specific context tokens.
- Target 5, Llama-2-7B-80K inclusion test: with the source repository's
  haystack, needles, insertion rule, prompt and success test the needle is
  retrieved in 20 of 20 instances (our WikiText haystack had given 10 of
  20), but 1.27 percent of heads score above 0.1 against the frozen 2 to 8
  percent band, so the model stays excluded by the rule as written. The
  band was set from the paper's 3 to 6 percent at 1K to 50K tokens on about
  600 instances; at 1K and 2K on 20 instances the fraction is lower.
- Target 3, Mistral-7B-v0.1: the registered expectation fails as written,
  in an instructive way. The sink-set surrogate (c), which keeps only the
  layer's sink columns and the diagonal of each head's last row and
  permutes the rest, reproduces 0.77 to 0.998 of the cross-head correlation
  excess over the random null (median 0.96; corrected 2026-10-12 from "0.90
  to 0.998"), so on this model the redundancy is mostly the shared sink
  column. But the real correlation sits above every one of the 200 draws
  in all 32 layers, so the frozen "matched" criterion (percentile between 5
  and 95 in two thirds of the layers) is met in 0 of 32; with 200 draws over
  32 documents the null distribution is far narrower than residuals of 0.2
  to 23 percent (median 4). The marginal-matched surrogate (b), which permutes the
  sink column away, reproduces about zero of the excess (0.000 to 0.065),
  so the clause "reproduces more than half" (addendum 4 reading) fails 0 of
  32; under the other reading of that clause (the statistic falls by more
  than half) it would hold 32 of 32. Random rows (a) and the five
  untrained models (d) are below the real value in every layer. Per-layer
  common label: shrinks (the null reproduces more than half) in 32 of 32.
  Reading: the redundancy is real and it is the sink; what the registered
  wording asked for, exact statistical indistinguishability from the
  sink-only surrogate, is stricter than the effect-size question it was
  meant to settle.
- Target 5, Mistral-7B-Instruct-v0.2 (second control model): 3.4 percent
  of heads above 0.1, every null at the 100th percentile, shrinkage 0.007
  to 0.011, untrained zero. E3 holds on both models.
- Target 1, BERT-base (Clark et al. 2019): the layer clustering D = 0.127
  survives random maps and untrained BERT (both give D of order 1e-5). The
  per-row marginal-matched surrogate reproduces 9.5 percent of D, below
  the registered 20 to 70 percent band, so the registered expectation fails
  on that clause; the separator-column surrogate ([CLS] and [SEP] columns
  kept) reproduces 64 percent of D (60 before the [SEP] fix of addendum 9)
  and puts 35.4 percent of nearest neighbors in the own layer against 39.6
  real and 7.7 by chance, inside the band. Label: shrinks. Reading:
  the clustering is not a marginals artifact (the registered guess of 20 to
  70 percent from marginals alone was wrong), it is mostly the shared
  vertical pattern on the separators, with a 40 percent residual that is
  real same-layer similarity beyond the separator columns.
- Target 4, Mistral-7B (Dewage et al. 2026): the registered expectation
  fails in all four projection types (0 of 4 pass all three clauses). The
  real outlier counts lie above all 20 draws of every null, but Gaussian
  weights matched in shape and Frobenius norm already produce 85 to 89
  percent of the reported count for the square projections (Q and O), 65
  percent for V and 40 percent for K; within-matrix permutation matches the
  Gaussian null to three digits, so the entry distribution is not the
  mechanism; row-norm-matched weights reproduce 71 to 88 percent of the
  energy share, against the registered "less than 50 percent". Label:
  shrinks in all four types.
- Post hoc (not registered, audits/dewage2026/posthoc_edge.py): the reason
  the Gaussian null produces outliers at all is that the recipe's noise
  scale, sigma^2 = median(s^2)/(1+gamma), places the bulk edge inside the
  bulk. With the scale fitted to the Marchenko-Pastur median or mean
  instead, a Gaussian matrix has zero outliers and the real matrices keep
  hundreds, but the counts fall by a factor of 1.3 to 9 and the two
  calibrated estimators differ by a factor of 1.8 to 4.6, so the published
  counts mostly reflect the estimator (corrected 2026-10-12 from "2 to 9"
  and "a property of the estimator rather than of the weights"). The
  qualitative claim, that trained attention weights carry spectral
  structure a matched Gaussian does not, survives every calibration.

## Functional round (2026-10-12, preregistration 11)

Question: do the map-level findings matter for what heads do? Frozen at
e45b49f before the run. Development on Qwen2.5-0.5B (labeled) contradicted
the motivating hypothesis that removing the sink columns would make map
similarity predict function, and the preregistration records the change.
Four held-out models with grouped-query attention (Qwen2.5-1.5B,
TinyLlama-1.1B, Mistral-7B, Qwen2.5-7B); 73 high-sink layers; 1,152 merges in
27 layers; outcome KL over 4,096 (small models) or 2,048 (7B) evaluation
tokens; weights restored exactly after every merge. Five of seven primary
clauses pass.

- Gate passes: the KL ranking of pairs is reliable (split-half 0.88;
  Mistral-7B alone 0.69, with four evaluation windows).
- F1 passes: the sink's value vectors, through the output weights, are 0.11
  of the other positions' size; its share of value-weighted attention is
  0.36 of its attention share (interval 0.32 to 0.41). Its contribution to
  the head output is still 0.47 of the output norm, because the other
  positions' contributions partly cancel. The sink is small per position,
  not a no-op.
- F2 passes: the sink-kept null's share of head similarity drops by 0.51
  (0.47 to 0.56) from raw to value-weighted maps.
- F3a passes: within a layer, the first-order output perturbation predicts
  merge KL at Spearman 0.65 against 0.38 for the plain map distance
  (difference 0.27, interval 0.14 to 0.43); the value-weighted distance
  sits between (0.51). Output perturbation first and value-weighted distance
  second in each of the four models.
- F3b FAILS: the generator cosine (0.30) is not reliably worse than the
  plain map distance (0.38): difference 0.07, interval -0.03 to 0.18,
  against 0.10 with an interval above zero. Development's 0.27 did not
  replicate.
- F3c passes: excluding the sink columns from the map distance gains
  nothing (-0.02, interval -0.06 to 0.01).
- F3d passes: 0.65, interval 0.52 to 0.78.
- F4 FAILS: the merge chosen by output perturbation is not reliably cheaper
  than the generator-cosine choice; same pair in 10 of 27 layers, cheaper in
  11, costlier in 6, median ratio 1.0. Its picks rank at the 8th percentile
  of the layer's costs against the 23rd (secondary, overlapping intervals).
- The change in NLL, round 7's outcome, is far noisier than the KL: every
  predictor's within-layer correlation with it is 0.08 to 0.26, same order.

Reading: the attention map is a weak proxy for what a head does, and the
sink is not why; what the map leaves out is what attention moves. The
output perturbation approximates the merge itself, so its correlation is
partly expected; the measured quantity is how much the map misses. The
practical claim, better single merges than the commutator criterion, is
not established.

## Standard-statistics round (2026-10-13, preregistration 12)

Frozen at fc6ae80 before the run; development on Qwen2.5-0.5B labeled. Same
four held-out models as round 11. 8 of 8 primary clauses pass.

- S1 passes: within a layer, the output perturbation predicts merge damage
  at 0.65; Jensen-Shannon 0.43, map cosine 0.35, last-row correlation 0.29,
  all-rows correlation 0.29; differences 0.22 to 0.36, intervals above zero.
  (Pair outcomes are round 11's merges; the four statistics were computed
  for the first time after this freeze.)
- S2 passes: Voita confidence tracks sink mass within high-sink layers at
  0.92 (0.86 to 0.97, 73 layers).
- S3, S4 pass: against the KL of removing a head, within a layer, output
  norm 0.53 (0.35 to 0.73), confidence -0.16 (-0.44 to -0.01), negative
  entropy -0.09, sink mass -0.42. Across layers within a model output norm
  is at chance (-0.08 to 0.24); confidence and sink mass stay negative.
- S5 to S7 pass: one merged pair per KV group in every layer but the first
  (54 to 248 pairs); pairs chosen by output perturbation cost 0.58 to 0.86 of
  the generator-cosine choice (geometric mean 0.73) and 0.66 to 0.85 of the
  CHAI-style choice (0.75), in every model; best of eight criteria in three
  models, third in TinyLlama. At two pairs per group the advantage shrinks
  to 0.99 (Mistral-7B) and 0.93 (TinyLlama), 0.52 and 0.66 in the Qwen models.

Reading: the head-importance heuristic that reads attention weights measures
the sink, and heads that use the sink more matter less when removed. The
similarity statistics the field uses are weaker predictors of what merging
two heads costs than a quantity that includes what attention moves, and the
latter chooses cheaper merges when many pairs are merged at the smaller
budget.

Post hoc (alignment_study/posthoc/sharedE_vs_merge_kl.py, not registered):
with round 11's KL outcome, a layer's shared energy against its median merge
KL is -0.38 pooled over 27 layers but positive in Mistral-7B (-0.24, -0.55,
+0.49, -0.70 per model); round 7's M2 failure is not explained by outcome
noise alone.

## What the sink does (2026-10-13, preregistration 13)

Frozen at 97d2837 before the run. Development on Qwen2.5-0.5B (labeled) after
a masking bug in the custom attention was caught by the equivalence check.
Same four models; the custom attention equals eager attention exactly in all
four. 4 of 4 primary clauses pass.

- K1: removing the first-token sink's contribution in all layers costs 28
  (Mistral-7B) to 68 (Qwen2.5-1.5B) times removing an ordinary token's.
- K2: replacing the sink weight by its mean over positions keeps 0.13 to
  0.43 of that damage (0.43 in Mistral-7B, 0.13 to 0.19 elsewhere).
- K3: removing the sink costs more than removing an ordinary token in every
  high-sink layer of three models; TinyLlama has no layer with first-token
  mass above 0.3 (its maximum is 0.16).
- K4: median mean-over-zero ratio 0.21 (0.18 to 0.64) over 33 high-sink
  layers.
- Not registered: removing the sink everywhere costs 2.3 to 3.4 times the sum
  of single-layer removals in the two fully tested models; forcing heads to
  re-spread the sink's share is costly in some single layers (up to 0.54 in
  TinyLlama) but mild in all layers at once (0.01 to 0.03), unexplained.

Reading: the sink is not a no-op. Its contribution matters, and most of it is
a constant bias per head rather than a position-dependent signal.

## Round 12's clauses at 1,024 tokens (2026-10-13, preregistration 14)

Frozen at 18901f6. Qwen2.5-1.5B and TinyLlama, selection and evaluation
windows of 1,024 tokens, same clauses and thresholds. 6 of 7 pass: S2
confidence tracks sink mass 0.91 (0.85 to 0.94); S3a, S3b output norm beats
confidence and entropy by 0.66 and 0.58 (intervals above zero but wide with
two models); S4 output norm 0.50 (0.21 to 0.74); S6 output perturbation
against the CHAI-style choice 0.75 and 0.50; S7 output perturbation best of
eight in both models. S5 FAILS narrowly: against the generator cosine 0.83 and
0.82, both below 1, geometric mean 0.82 against the registered 0.8.
Confidence against removal damage: -0.16 (-0.45 to 0.11), secondary.

## Round 12's head-importance clauses on multi-head decoders and C4 (2026-10-13, preregistration 15)

Frozen at 46fd50c. gpt2-medium, Pythia-160m and Pythia-410m, every head of
every layer removed one at a time; damage measured on WikiText-103 and on C4.
Development on gpt2 labeled. 7 of 7 clauses pass:
- S2: confidence tracks sink mass within high-sink layers at 0.86 (0.64 to
  0.93, 44 layers).
- S3a, S3b: output norm's within-layer Spearman with removal damage exceeds
  confidence's by 0.91 (WikiText) and 0.95 (C4) and negative entropy's by
  0.86 and 0.89, every interval above zero.
- S4: output norm 0.68 (0.62 to 0.73) on WikiText and 0.69 (0.61 to 0.76) on
  C4; per model 0.65 to 0.74.
- Secondary: confidence -0.24 and -0.26, negative entropy -0.18 and -0.20,
  sink mass -0.36 and -0.40 (WikiText, C4). Across all layers of a model, not
  registered: output norm 0.39 to 0.69, confidence -0.32 to -0.73. In these
  multi-head models output norm also ranks heads across layers, which it
  did not in round 12's grouped-query models (-0.08 to 0.24).

Correction found while writing this round up: the statistic called Voita et
al.'s confidence in preregistrations 12, 14 and 15 is the largest attention
weight over all tokens, sink included. Voita et al. exclude the
end-of-sentence token (their footnote 2), which in translation encoders
draws attention the way the sink does in a decoder. The registered results stand for the statistic computed; the
attribution was wrong. Preregistration 16 (frozen at f5a9539) tests their
definition.

## Voita et al.'s confidence as they define it (2026-10-13, preregistration 16)

Frozen at f5a9539. Development on gpt2, Qwen2.5-0.5B and opus-mt-en-es
labeled. Seven decoders with stored removal outcomes (rounds 12 and 15;
recomputed all-token confidence and sink mass equal the stored values exactly
in all seven) and the encoder heads of three OPUS-MT translation models,
English to Russian, German and French (18 layers, 144 heads; restore checks
0). 5 of 6 clauses pass:
- V1: with the sink columns excluded as keys, confidence's within-layer
  Spearman with removal damage is +0.32 (0.19 to 0.44), positive in all seven
  models (0.18 to 0.49).
- V2: that is 0.53 (0.36 to 0.71) above the all-token confidence (-0.21).
- V3: partial Spearman given sink mass +0.15 (0.02 to 0.28).
- V4: in the translation encoders, the EOS-excluded confidence is +0.53
  (0.41 to 0.63), 0.50 to 0.54 per model.
- V5 FAILS: in the translation encoders, excluding EOS adds 0.05 (-0.02 to
  0.13) against the registered 0.1; the all-token version is already +0.48.
  EOS takes 19 to 25 percent of encoder attention on average.
- V6: in the decoders, output norm (+0.62) exceeds the sink-excluded
  confidence by 0.31 (0.19 to 0.42).
- Secondary: on C4 (round 15's models) +0.35 against -0.26 for the all-token
  version. Excluding only the first token gives +0.26 (Qwen2.5-7B +0.02,
  whose sink set often holds other columns). Dividing by the non-sink mass
  leaves +0.11 (-0.01 to 0.22): most of what the definition carries is how
  much attention a head puts on real tokens. Translation encoders: output
  norm +0.73 (0.62 to 0.82), ahead of confidence by 0.20 (0.05 to 0.35);
  negative entropy +0.49; EOS mass -0.27.

Reading: Voita et al.'s statistic, as they define it, marks heads whose
removal matters, in their setting and, with the sink left out, in decoder
language models. Rounds 12, 14 and 15 computed it with the sink included,
which turns it into a sink detector with the wrong sign. The token they
excluded matters little in their translation encoders and decides the sign
in decoders, where the sink takes far more attention. The norm of what a
head writes predicts removal damage better in both settings.

## Two audits under the sources' own readings (2026-10-14, preregistration 17)

Frozen at 94b7734 (`audits/PREREGISTRATION17.md`); instruments checked
against the registered batteries before the freeze. 8 of 9 clauses pass.

Part K, Target 1 in Clark et al.'s input format: bert-base-cased with
segment ids on 1,000 segments, [CLS] A [SEP] B [SEP], built with the logic of
their preprocessing script from WikiText-103 validation articles (97.9
percent at 128 tokens). All five clauses pass. K0: D = 0.096, same-layer
nearest neighbor 36.1 percent (chance 7.7). K1: the null keeping the [CLS]
column and both [SEP] columns reproduces 0.638 of D (Target 1: 0.638). K2:
marginal null 0.125. K3: random and untrained nulls about zero. K4: label
shrinks, as in Target 1.

Part C, Target 3 under the every-row reading (per-row Pearson across heads
over the causal support, 64 rows, the same 32 C4 documents of 1,024 tokens):
- C1: on Mistral-7B the sink-set null reproduces a median 0.97 (0.80 to
  0.998 over layers; last row 0.96).
- C2 FAILS: on OPT-6.7B the median is 0.43 (0.08 to 0.96; last row 0.68).
  The share is below one half in 18 of 32 layers, 0.08 to 0.36 in the first
  seven, and 0.87 to 0.96 in the last seven.
- C3: marginal null 0.007 (Mistral) and 0.014 (OPT). C4: untrained models
  0.0004 and 0.002.
- Reported: every-row correlation 0.58 to 0.98 (Mistral) and 0.25 to 0.88
  (OPT); depth trend -0.36 and +0.64 (last row -0.38 and +0.49); majority
  clusters at 0.95 in 4 and 0 layers, at 0.90 in 7 and 3. The recomputed
  last-row statistics equal Target 3's exactly.

Reading: the input format does not change the BERT result. The sink account
of CHAI's redundancy holds on Mistral-7B under either reading; on OPT-6.7B
it depends on the rows read, and over every row most of the correlation in
the early and middle layers is something other than the sink columns.

Related work found after this round (2026-10-14; to be cited): Papamichalis
and Ruane (arXiv 2608.14712, August 2026) show that keeping or dropping the
sink flips 17 to 47 percent of head-similarity verdicts in ten models and
that the most prominent structure in a standard BERT head-clustering
pipeline follows from that choice; Sandoval-Segura et al. (ICLR 2026) find
that output-norm scores identify inactive heads that attention-weight
scores miss.

## Vision transformers' attention distance and artifact tokens (2026-10-15, preregistration 19)

Frozen at 8455f04 (`audits/PREREGISTRATION19.md`). 128 Imagenette images.
3 of 5 clauses pass; the artifact hypothesis is not supported.
- V5 passes: the source's claim reproduces on supervised ViT-L/16. Attention
  distance rises 24 pixels from the first to the second half of the layers,
  every head's distance correlates positively with depth, and the second
  half sits near the uniform distance (111 against 117 pixels). ViT-B/16:
  +37 pixels. Neither has artifact columns of note (0.002 and 0.08 of
  second-half attention; artifact share of the increase -0.01 and 0.02).
- V2 and V3 pass: DINOv2-L has artifact columns holding 0.16 of second-half
  attention, DINOv2-L with registers has none (registers hold 0.36).
- V1 and V4 fail: in DINOv2-L the distance does not grow with depth (-2.4
  pixels; removing the artifact columns lowers the second-half distance from
  85.6 to 78.7 pixels but the increase is undefined), and the registers model
  grows more (+4.8 pixels), not less.
Reading: the late-layer globality of the source's models is genuine, not an
artifact effect; artifacts appear in the self-supervised large model, which
does not show the depth trend. Development on DINOv2-B (share 0.47) did not
carry over to the large model.

## A benchmark of head-merging criteria (2026-10-15, preregistration 20)

Frozen at 283e5b5. Key-value merges (two heads' key and value projections
averaged, the multi-head to grouped-query conversion) in gpt2-medium,
Pythia-160m and Pythia-410m, every pair of heads in every layer (6,552
merges), damage as next-token KL. 5 of 5 clauses pass.
- B1: the simulated output perturbation predicts merge damage at within-layer
  Spearman 0.72 (0.64 to 0.79; per model 0.66, 0.70, 0.79).
- B2, B3: every criterion in use stays below 0.3: Jensen-Shannon 0.12, key
  cosine 0.10, key-value weight cosine 0.09, query-key weight cosine 0.08,
  map cosine 0.08, every-row correlation 0.06, last-row correlation 0.05,
  value cosine -0.01; output perturbation leads the best of them by 0.60
  (0.52 to 0.73).
- B4, B5: one pair per layer, the output perturbation's merges cost 0.57,
  0.28 and 0.34 of the last-row correlation's (the CHAI-style choice) and
  0.46, 0.45 and 0.28 of the random median; two pairs per layer, 0.48, 0.40
  and 0.35 of the CHAI-style choice.
- Checks: capture attention equals eager attention (0.0) and restore checks
  0 in all three models. The full table of pair outcomes and criteria is the
  benchmark (alignment_study/merge_bench/).
Reading: for the merge that grouped-query conversion performs, no similarity
in use predicts which heads can be merged; simulating the merge on a few
hundred tokens does.

## A rule for which attention claims survive nulls, tested on six claims (2026-10-15, preregistration 18)

Frozen at cdda0f1 (`audits/PREREGISTRATION18.md`). Rule as stated before
testing: claims that aggregate attention into distribution-level summaries
have s >= 0.5 (s = share reproduced by the null keeping each row's values and
the shared columns); claims about where heads look have s <= 0.2. All six
claims reproduce. The rule as stated predicts 4 of 6.
- A1, heavy hitters (H2O; OPT-6.7B): top-10-percent share of accumulated
  attention 0.70 against 0.33 random; s = 0.64. Holds.
- A2, persistence of importance (Scissorhands; OPT-6.7B): persistence beyond
  chance 0.19; s = 0.002. FAILS (predicted >= 0.5).
- A3, observation-window hit rate (SnapKV; Mistral-7B-Instruct-v0.2): 0.61
  against 0.10 random; s = 0.04. FAILS (predicted >= 0.5).
- L1, induction heads (GPT-2 medium, Pythia-410m): prefix matching 0.93 and
  0.91; s = 0.001 and 0.000. Holds.
- L2, positional heads (bert-base-cased): 0.76; s = 0.01. Holds.
- L3, syntactic heads (bert-base-cased, UD EWT): 0.78 against 0.14 random;
  s = 0.001. Holds.
The development-informed secondary reading, that statistics weighted by
attention mass are sink-dominated while statistics that count positions
above a threshold are not, fits 6 of 6; it was formed on development data
and needs a prospective test on new claims.
Reading: H2O's heavy hitters are mostly the sink; the persistence and
window observations behind Scissorhands and SnapKV, and the three located
claims, are not reproduced by the sink at all.

## The sink as bias or gate, across architectures, and a sink-free stream (2026-10-15, preregistration 21)

Frozen at 8a15d0a. 3 of 6 clauses pass.
- G1 passes: in the rotary decoder Pythia-410m a constant vector reproduces
  0.69 of the sink's effect (bias share), and removing the sink costs 16
  times an ordinary token.
- G2 FAILS: the learned-position decoders split. GPT-2 medium's sink is
  gate-like (bias share -0.06, as in development on gpt2), but OPT-6.7B's is
  bias-like (0.58). The position scheme does not separate them.
- G3 FAILS: removing the sink costs 16 and 14 times an ordinary token in
  Pythia-410m and OPT-6.7B but only 2.2 times in GPT-2 medium.
- G4 passes: in encoders the special tokens are close to no-ops. Removing
  BERT's [SEP] contributions costs 1.3 times an ordinary token, the
  translation encoder's end-of-sentence token 0.95 times (bias shares 0.19
  and 0.67).
- G5 passes: evicting the first token from a 256-token attention window
  raises next-token loss from 1.98, 2.29 and 2.73 nats (full attention) to
  5.83, 4.87 and 4.54; keeping the first four tokens (StreamingLLM) brings it
  to 2.22, 2.51 and 3.00 (TinyLlama, Qwen2.5-1.5B, Pythia-410m).
- G6 FAILS: a constant virtual sink, with no sink token kept, recovers 0.87
  of StreamingLLM's improvement in Qwen2.5-1.5B and 0.54 in Pythia-410m, but
  0.06 in TinyLlama, whose sink is not concentrated on the first token
  (round 13).
- Checks: the custom attention equals eager attention exactly in every
  float32 model; for OPT-6.7B, run in bfloat16, the maximum logit difference
  is 0.44, because the function computes attention in float32 and eager
  attention in bfloat16. OPT's edits are measured against the function's
  own unedited output (restore check 0), so they are internally consistent,
  but the registered "equal to eager" holds there only up to that precision.
Reading: the sink is a costly constant bias in most decoders tested here
(round 13's four, Pythia, OPT), a cheaper gate in GPT-2, and close to a
no-op in encoders; where it is a bias, a constant per-head vector can stand
in for the sink token in streaming attention.

## The merge benchmark at larger scale and on a task (2026-10-16, preregistration 22)

Frozen at f5d3b5e. 6 of 7 clauses pass.
- S1 to S4 pass: in Pythia-1.4B, OPT-1.3B and GPT-2 large (21,624 merges),
  the simulated merge predicts merge damage at within-layer Spearman 0.68
  (0.64 to 0.72; 0.67, 0.67 and 0.70 per model). Every similarity in use stays
  below 0.3: Jensen-Shannon 0.24, map cosine 0.18, every-row and last-row
  correlation 0.18 and 0.17, key cosine 0.07, key-value and query-key weight
  cosines 0.06 and 0.05, value cosine 0.00; the lead over the best of them is
  0.44 (0.37 to 0.51). At one pair per layer its merges cost 0.48 (Pythia),
  0.48 (OPT) and 0.64 (GPT-2 large) of the CHAI-style choice's KL.
- D3 passes: across 72 configurations (six models, two budgets, six
  choices) the LAMBADA accuracy drop tracks the KL damage at Spearman 0.86.
- D1 passes: at one pair per layer the simulated merge's choices lose less
  LAMBADA accuracy than the CHAI-style choice's in 5 of 6 models (accuracy
  drops 0.011 against 0.139 in Pythia-410m, 0.056 against 0.147 in
  Pythia-160m, 0.026 against 0.064 in Pythia-1.4B, 0.044 against 0.051 in
  GPT-2 medium, 0.027 against 0.028 in GPT-2 large; OPT-1.3B 0.016 against
  -0.006).
- D2 FAILS: at two pairs per layer, 4 of 6. The Pythia models and GPT-2 large
  favor the simulated merge (0.088 to 0.114 against 0.083 to 0.316), GPT-2
  medium does not (0.113 against 0.039, although its KL was half), and in
  OPT-1.3B the merges barely move accuracy (0.025 against 0.014).
- Checks: capture equals eager attention (0.0) and restore checks 0 in all
  three new models; LAMBADA restore equals the base accuracy in all six.
Reading: the predictive result holds at 1.3 to 1.4B parameters and in OPT;
KL damage is a good proxy for LAMBADA accuracy across configurations; the
task-level advantage is clear where merges cost accuracy (the Pythia
family) and within noise where they barely do (OPT-1.3B, GPT-2 large).

## Flipping the weighting of five audited statistics (2026-10-16, preregistration 23)

Frozen at 50fdbc4 (`audits/PREREGISTRATION23.md`). 4 of 5 clauses pass.
- F1 passes in all four decoder pairs: flipping a statistic from counted
  positions to attention mass, or back, moves the share reproduced by the
  shared-columns null in the predicted direction. Heavy hitters (OPT-6.7B):
  mass 0.64, counted 0.22. Persistence (OPT-6.7B): mass 0.52, counted 0.00.
  SnapKV hit rate (Mistral-7B-Instruct-v0.2): mass 0.64, counted 0.04.
  CHAI last row (Mistral-7B): Pearson 0.95, top-5% Jaccard 0.11.
- F2 FAILS: for Clark's statistic (bert-base-uncased), where the counted
  version takes the top 5 of about 64 positions, the occupancy reading
  predicted no separation; the Jensen-Shannon contrast is reproduced at 0.64
  and the Jaccard contrast at 0.27, a gap of 0.37. The separator columns
  raise the counted share above the other counted statistics (0.00 to 0.22)
  but do not keep it near the weighted one.
- Secondary, the plain rule: 8 of 10 statistics on the predicted side of 0.5
  or 0.2; the exceptions are Clark's counted statistic (0.27) and the counted
  heavy-hitter share (0.22).
- The recomputed originals match their rounds: Clark 0.638 (Target 1), CHAI
  0.95 (Target 3), heavy hitters 0.64, persistence 0.002 and hit rate 0.04
  (round 18).
Reading: whether nulls that keep the shared columns reproduce an attention
statistic depends on how the statistic weighs positions. Weighted by
attention mass, it is mostly the sink; counted by positions, it mostly is
not. This held prospectively on five new statistics; the finer occupancy
reading did not.

## Converting multi-head attention to grouped-query attention without training (2026-10-16 to 18, preregistration 24)

Frozen at 7f12472 (`alignment_study/PREREGISTRATION24.md`). 9 of 10 clauses pass.
- P2 and P3 pass in 10 of 10 cells: with each group's heads aligned first
  (Jin et al.'s generalized Procrustes rotations, folded into the weights) and
  the groups chosen by the simulated merge, the converted model has the
  lowest WikiText-2 perplexity of the nine recipes in every model and group
  size, and lower than Jin et al.'s grouping in all ten. Half of the
  key-value heads kept (g = 2): OPT-1.3B 33.3 against 49.1 (unconverted
  16.8), GPT-2 large 43.4 against 54.7 (19.4), Phi-2 34.9 against 57.2
  (11.0), Pythia-1.4B 63.9 against 77.5 (13.2), OLMo-1B 580 against 11,003
  (13.4).
- P4 passes: at g = 2 its increase in log perplexity is 0.56 to 0.89 of Jin
  et al.'s, geometric mean 0.70 (registered: at most 0.85).
- P5 passes: at g = 2 its average accuracy on the seven zero-shot tasks is
  higher than Jin et al.'s in all five models (+0.016 to +0.053; mean +0.033,
  interval 0.029 to 0.036). At g = 4, where every recipe is close to chance
  (0.29 to 0.36 against a chance level of 0.29), the differences are within
  noise except in OPT-1.3B (+0.021).
- P1 passes (19 of 20): aligning before pooling lowers perplexity for
  neighbouring and for random groups, except OLMo-1B's neighbouring groups at
  g = 4 (46,518 against 14,791), where both are unusable.
- P7 passes: without alignment every recipe leaves perplexity at 12.8 times
  the unconverted model's or more, in all ten cells. Neighbouring-head mean
  pooling, the standard initialization, gives 13 to 743 times at g = 2. QCQA's
  and AsymGQA's criteria land within a factor of two of a random grouping in
  every cell (0.48 to 1.50 times its perplexity).
- P8 passes (4 of 5): without alignment the simulated merge's groups beat
  neighbouring heads at g = 2 except in GPT-2 large (274.9 against 255.5).
- P6 FAILS (7 of 10): across the nine recipes the simulated cost of each
  partition orders its damage at Spearman 0.67 to 0.98 in OPT-1.3B,
  Pythia-1.4B and GPT-2 large and in Phi-2 at g = 2, but at 0.47 and 0.50 in
  OLMo-1B and 0.32 in Phi-2 at g = 4. In OLMo-1B every recipe but one is
  unusable and their order is noise; the chosen groups cost 0.92 of Jin et
  al.'s in the simulation summed over layers and give a nineteenth of the
  perplexity, so the damage compounds across layers beyond what the
  per-layer simulation sees. In Phi-2 at g = 4 the simulation misses what
  alignment does: aligned neighbouring and random groups cost almost as much
  as unaligned ones (1,248 against 1,263; 1,211 against 1,247) and reach a
  third of their perplexity.
- C1 and C2 pass: rotating every head's query-key and value-output spaces at
  random, which leaves the model unchanged, moves the perplexity of
  unaligned neighbouring pooling by a factor of 1.10 to 4.04 across the four
  bases (at least 1.1 in 9 of 10 cells; GPT-2 large at g = 4, 1.096, the
  exception), and the trained basis gives the lowest of the four in 3 of 10
  cells. Aligned pooling gives the same perplexity in every basis (within
  3e-6 relative in all 30 cases); its converted models differ by at most
  1.9e-2 in any logit in Phi-2, whose logits reach 34, and by at most 2.2e-3
  elsewhere.
- Checks: capture equals eager attention exactly in all five models; the
  alignment changes no logit by more than 1.5e-3 (4.5e-3 in Phi-2) before
  pooling; restoring the unconverted weights returns the base perplexity in
  all five.
Reading: averaging key and value projections across heads, the standard
initialization, depends on a convention the model does not have; aligned
first, the conversion is a function of the model alone. Without training, how
the groups are chosen matters once the heads are aligned, and choosing them by
the simulated merge is better than every published criterion here, by a wide
margin in perplexity and by 1.6 to 5.3 points of accuracy at half the heads.
Without alignment no criterion rescues mean pooling. Even the best conversion
stays well short of the unconverted model (perplexity 2.0 to 4.8 times at half
the heads in four models, 43 times in OLMo-1B; accuracy 0.08 to 0.23 lower),
so it is a starting point for uptraining, not a finished model; whether the
advantage survives uptraining was not tested.

## Fusing key and value heads by what the heads compute (2026-10-18 to 19, preregistration 26)

Frozen at 16404cf (`alignment_study/PREREGISTRATION26.md`). 6 of 6 clauses pass.
- F1 passes in 10 of 10 cells: on round 24's groups, the closed-form fusion
  weighted by each head's queries and output projection has a lower
  WikiText-2 perplexity than round 24's best recipe (aligned mean pooling) in
  every model and group size. Half of the key-value heads kept: OPT-1.3B 29.1
  against 33.3 (unconverted 16.8), GPT-2 large 30.4 against 43.4 (19.4),
  Phi-2 18.9 against 34.9 (11.0), Pythia-1.4B 27.3 against 63.9 (13.2),
  OLMo-1B 87.4 against 580 (13.4). A quarter: 93.7 against 183, 117 against
  238, 65.7 against 238, 71.2 against 472, 601 against 5,509.
- F2 passes: at g = 2 its increase in log perplexity is 0.46 to 0.81 of round
  24's best, geometric mean 0.55 (registered: at most 0.6). OPT-1.3B, the model
  round 24 converted best, gains least (0.81).
- F3 passes in 20 of 20: on the same groups the weighting gives a lower
  perplexity than the unweighted fit in every comparison, by factors from 1.02
  (OPT-1.3B, g = 2, round 24's groups: 29.1 against 29.6) to 3.05 (OLMo-1B,
  g = 2, neighbouring heads: 98.2 against 299).
- F4 passes: at g = 2 the average accuracy on the seven tasks is higher than
  round 24's best in all five models (+0.029 to +0.066; mean +0.044, interval
  0.041 to 0.048). At g = 4 it is higher in four (+0.020 to +0.067) and equal
  in OLMo-1B (-0.001, interval -0.010 to 0.009).
- F5 passes (9 of 10): neighbouring heads, fused with no search, beat round
  24's searched and aligned best except OPT-1.3B at g = 2 (40.5 against 33.3).
- F6 passes in 10 of 10: round 24's groups still beat neighbouring heads after
  fusion. Jin et al.'s groups, fused (reported, not scored), were lower still in
  7 of 10 cells.
- Checks: groups of one reproduce every model to at most 1.8e-2 in any logit
  (Phi-2, whose logits reach 34; at most 4.2e-3 in the others); fusing in a
  random function-preserving basis gives the same converted model to at most
  6.2e-2 (Phi-2) and 1.2e-2 elsewhere; the unconverted perplexity equals round
  24's in all five models, and restoring the weights restores it.
Reading: fitting the shared key and value head to what the heads compute,
rather than to their weights or caches, is what makes training-free
conversion work much better. At half the key-value heads the converted models
reach 1.56 to 2.07 times the unconverted perplexity in four of the five models
(round 24's best: 1.98 to 4.84) and lose 0.05 to 0.10 of accuracy; OLMo-1B
remains hard (6.5 times, 0.20). Once the heads are fused, the choice of groups
still matters but less, and the groups chosen for averaging are not the best
groups for fusion.

## Cross-layer KV cache compression under the convention null (2026-10-19, preregistration 25)

Frozen at 809f6fb (`alignment_study/PREREGISTRATION25.md`), run with the
device fix 82cadd2. 5 of 7 clauses pass.
- A1 passes: in all four models the per-token cosine between adjacent layers'
  cached keys, and between their cached values, is within 0.005 of zero
  (median over layer pairs), while the residual stream's is 0.90 to 0.94 and
  the normalized layer input's 0.85 to 0.92. MiniCache's motivating similarity
  is a property of the residual stream, not of the cache it merges.
- A2 passes: after per-head alignment fitted on other documents, adjacent
  layers' values reach a median cosine of 0.25 to 0.35, 0.21 to 0.28 above the
  same fit to shuffled tokens.
- B1 passes: MiniCache's merge from mid-depth does no better in the trained
  basis than in random function-preserving bases (trained minus the random
  median: -0.055 to -0.001 nats in three models; in Qwen2.5-7B the trained
  basis is worse by 1.19 nats).
- B2 FAILS (5 of 8): aligning heads first cuts the merge's damage to 0.21 to
  0.52 of the trained basis's in Mistral-7B-Instruct-v0.2 and Phi-3-mini at
  both depths and to 0.30 in Qwen2.5-7B from mid-depth, but only to 0.70 in
  Qwen2.5-7B from a quarter of the depth, and in the LLaMA-2-7B derivative it
  multiplies the damage (11.4 and 2.0 times).
- C1 passes: XQuant's statistic, the share of positions whose 2-bit codes
  differ by at most one level between adjacent layers, is 0.80 for keys and
  0.82 for values, within 0.003 of what independent codes with the same
  frequencies give, in every model.
- C2 passes: after alignment the exact agreement of value codes exceeds
  independence by 0.045 to 0.074.
- C3 FAILS (1 of 4): XQuant's sharing loses less in the trained basis than in a
  random basis only in Phi-3-mini (1.69 against 2.15 nats); in the other three
  a random basis does as well or better (by 0.05 to 0.17 nats). The sharing
  costs 0.40 to 2.35 nats of continuation loss, against 0.01 to 0.03 for the
  same 2-bit quantization without sharing.
- Caveat: the LLaMA-2-7B derivative's continuation perplexity is 84 in this
  setup (Mistral 4.6, Qwen 6.1, Phi-3 5.3), and 2-bit quantization lowers it;
  its configuration (rotary base 10,000 with dynamic scaling) probably does not
  match its long-context training, and round 4's audit could not reproduce its
  retrieval heads either. Its results are reported as registered.
Reading: both premises fail against their nulls. Adjacent layers' caches are
not similar token by token; the similarity MiniCache plots is the residual
stream's. XQuant's "over 80 percent within one level" is what independent
codes give. Functionally, neither method draws on the trained basis: MiniCache's
merge is no better there than in random bases, and XQuant's sharing is better
there in only one of four models. Aligning heads first exposes real shared
structure (A2, C2) and makes MiniCache's merge much cheaper in three of the
four models, but not reliably enough for the registered B2, and not in the
miscalibrated LLaMA-2-7B derivative.

## The weighted fusion against CAGA on its own benchmark (2026-10-28 to 29, preregistration 28)

Frozen at 0abdd89 (`alignment_study/PREREGISTRATION28.md`). 6 of 6 clauses pass.
Llama-2-7B (the NousResearch copy, whose weight files are byte-identical to
Meta's) converted from 32 to 16 key-value heads without training, CAGA's
setting (Wang, Jiang and Liu, arXiv 2610.09476, Table 2). Unconverted: WikiText-2
perplexity 5.47 at 2,048 tokens and 5.12 at 4,096, the values the literature
reports; seven-task average (ZS-7) 0.641.
- M1 passes: on heads paired by its own residual, the weighted fusion reaches
  perplexity 16.78 at 2,048 tokens and 18.46 at 4,096 (3.07 and 3.61 times the
  unconverted model's), below CAGA's published 25.1 at both lengths.
- M2 passes: its ZS-7 is 0.504 (interval 0.490 to 0.518), above CAGA's
  published 0.443.
- M3 passes in 4 of 4: on identical pairs it is far below our implementation
  of CAGA's construction, 16.78 against 323 at 2,048 tokens and 18.46 against
  378 at 4,096 on the matched pairs, 19.73 against 845 and 25.58 against 1,348
  on neighbouring pairs.
- M4 passes: ZS-7 higher than CAGA's construction by 0.116 on the matched pairs
  and 0.125 on neighbouring pairs (pooled 0.120, interval 0.104 to 0.137).
- M5 passes: matched pairs beat neighbouring pairs, 16.78 against 19.73 at
  2,048 tokens and 18.46 against 25.58 at 4,096. The matching keeps 11 of 512
  neighbour pairs and lowers the summed relative residual from 5.98 to 5.35.
- M6 passes: steering agreement for neuron directions is 0.30 for the fusion on
  both pairings, against 0.05 and 0.04 for CAGA's construction and 0.10 for
  averaging. The cosine of the steering signatures is 0.31 for the fusion and
  about zero for CAGA's construction (-0.004 and -0.009); across the five
  converted arms, agreement and perplexity rank together (Spearman -0.70).
- Reported: averaging neighbouring heads gives 1,081 and 1,093 (the paper's
  direct averaging on its own pairing: 926). CAGA's construction as the paper
  specifies it (its key fit, which is our unweighted key fit, and the rank-d_h
  SVD of the raw stacked value projections, both checked against the paper's
  text) lands far from the paper's 25.1 in this harness on both pairings
  tried, between the paper's averaging and our averaging on neighbouring pairs.
  The paper's pairing combines percentile ranks of a channel-support overlap
  from another component of its method and of two similarity terms it gives no
  formula for, so it could not be reproduced; an exploratory follow-up asks
  whether a pairing chosen for CAGA's own residual closes the gap. The
  plain-accuracy averages order the arms the same way (0.464 for the fused
  matched pairs, 0.356 and 0.323 for CAGA's construction, 0.317 for averaging;
  unconverted 0.589). Checks: groups of one reproduce the model to 1.13 in
  logits of up to 29.9 (bfloat16 rounding of the rewritten weights), and the
  restored model gives 5.47 again.
Reading: on CAGA's own benchmark, fitting each pair's shared key and value head
to what the heads compute, with pairs chosen by the same criterion, gives a
training-free conversion at perplexity 16.8 to 18.5 against the 25.1 published
for the best training-free method, and 6 points more zero-shot accuracy. CAGA's
construction, which shares the fusion's key fit but takes its values from the
weights alone, depends on its pairing far more than the fusion does; on the
pairs tried here its perplexity is 19 to 53 times the fusion's. (Added
2026-10-30: on a third pairing, matched by CAGA's own residual, both
constructions collapse, through one merged pair in layer 1; see the
exploratory follow-up at the end of this note.) The steering
check separates the constructions as perplexity does: of the 20 tokens a push
along an MLP neuron raises most in the original model, the fused models raise 30
percent among their own 20, CAGA's construction 4 to 5 percent. The damage of
every conversion grows with the context length, and the pairing matters most
there (25.6 against 18.5 at 4,096 tokens).

## Fusing key-value heads in models that already share them (2026-10-19 to 30, preregistration 27)

Frozen at 35249e6 (`alignment_study/PREREGISTRATION27.md`). 3 of 4 clauses pass.
Neighbouring key-value heads merged in groups of 2, and of 4 where the count
allows, in five models: TinyLlama 1.1B (32 query and 4 key-value heads),
Qwen2.5-1.5B (12 and 2), OpenLLaMA-3B (32 and 32, multi-head),
Mistral-7B-Instruct-v0.2 (32 and 8) and Qwen2.5-7B (28 and 4). WikiText-2
perplexity over 1,024-token windows; the seven tasks of `zeroshot.py`.
- K1 passes in 9 of 9 cells: the weighted fusion is below averaging everywhere.
  Half of the key-value heads kept: TinyLlama 70.9 against 587 (unconverted
  8.58), Qwen2.5-1.5B 173 against 155,351 (10.40), OpenLLaMA-3B 31.8 against
  1,344 (8.06), Mistral 19.6 against 277 (6.72), Qwen2.5-7B 37.6 against 4,717
  (7.64). A quarter: TinyLlama 341 against 2,049, OpenLLaMA-3B 237 against
  3,579, Mistral 122 against 637, Qwen2.5-7B 1,908 against 93,059.
- K2 FAILS (7 of 9): the weighting beats the unweighted fit except in the two
  Qwen2.5 cells that merge all key-value heads into one: Qwen2.5-1.5B with its two
  heads (173 against 73.5) and Qwen2.5-7B at a quarter of its four (1,908 against
  1,008). TinyLlama merged to one head the same way is not an exception (341
  against 375).
- K3 passes: at half the heads the fusion's increase in log perplexity is 0.25
  to 0.50 of averaging's (geometric mean 0.31; registered: at most 0.5).
- K4 passes: at half the heads its average accuracy exceeds averaging's in all
  five models (+0.055 to +0.164; pooled +0.112, interval 0.106 to 0.117).
- Reported: at a quarter of the heads every construction leaves Qwen2.5-7B at
  chance accuracy (0.294). Groups of one reproduce the float32 models to 1.3e-4
  in any logit and the bfloat16 models to 0.97 to 1.78 (rounding of rewritten
  weights); restoring the weights restores every unconverted perplexity. The
  last model needed three attempts (run log): the first ended without output
  after its groups-of-one check, the second was killed by the memory watchdog
  while a second 7B job ran alongside it, and the third ran alone with the
  scorer projecting the vocabulary in chunks (per-example decisions unchanged).
Reading: the fusion carries its advantage over averaging to models that already
share key-value heads and to 7B scale: at half the key-value heads it removes
50 to 75 percent of averaging's damage in log perplexity and recovers 5 to 16
points of accuracy. The weighting, which helps in seven of nine cells, hurts in
both Qwen2.5 cells where every key-value head merges into one. Both are models
whose projections carry biases, handled here through the augmented second
moment; whether the biases are the reason was not tested.

## Exploratory follow-up to round 28: one pair of heads switches on the super weight (2026-10-29 to 30, not preregistered)

Round 28 compared constructions on two pairings because CAGA's own pairing
cannot be reproduced from its paper. A parallel session of the assistant asked
whether CAGA's construction does better on pairs chosen for it, the minimum-cost
matching under CAGA's own residual (`llama2_caga_followup.py`). On Llama-2-7B
both constructions collapsed on those pairs: perplexity 11,398 for CAGA's
construction and 12,142 for the weighted fusion at 2,048 tokens (9,665 and
13,560 at 4,096), against 323 and 16.8 on round 28's matched pairs. A clean
rerun through round 28's conversion function reproduced both to every printed
digit (`llama2_followup_rerun.json`; the restored model gives 5.470). Nothing
here was registered; the scripts and outputs are committed as they ran.
- Layer by layer the two pairings look alike (`llama2_followup_diag.json`, 8
  windows of 2,048 tokens). They share 185 of 512 pairs. In the fusion's own
  residual CAGA's pairs are 1.4 percent worse in the median layer, at most 6.0
  percent outside layer 0 (17.8 percent in layer 0), and converting any single
  layer alone gives perplexities within 0.83 percent of each other.
- The collapse comes from layer 1 (`llama2_followup_bisect.json`, 4 windows,
  unconverted 6.20). Converting layers 0 to k on round 28's pairs raises
  perplexity smoothly to 20.0 at all 32 layers; on CAGA's pairs it is 6.50
  after layer 0 and 2,316 once layer 1 is added. Every layer on round 28's
  pairs except layer 1 on CAGA's gives 2,231, and every other one-layer swap
  19.5 to 20.6; every layer on CAGA's pairs except layer 1 on round 28's gives
  21.8, and every other swap 1,969 to 17,524.
- One pair is enough (`llama2_followup_mechanism.json`). With layer 0 converted
  on CAGA's pairs, merging key-value heads 8 and 18 of layer 1 and nothing else
  in that layer gives 6,008; each of layer 1's other 15 pairs gives 6.41 to
  6.53. With layer 0 on round 28's pairs the same merge gives 2,325, and with
  layer 0 unconverted 6.49 (`llama2_followup_neuron.json`). No single pair of
  layer 0 is responsible: with layer 1 on CAGA's pairs, each of layer 0's pairs
  alone gives 6.29 to 6.42.
- The unit. Unconverted, the residual stream after layer 1 holds very large
  values at two positions only, the first (800, in dimension 2533) and the
  first newline (1,960); the median position's largest entry is 0.83 (the
  massive activations of Sun et al. 2024). With layers 0 and 1 converted on
  CAGA's pairs, 418 of 2,048 positions exceed 100 and the median position's
  largest entry is 51, in dimension 2533. Dimension 2533 is written most
  strongly by neuron 7890 of layer 1's MLP, whose down-projection weight
  [2533, 7890] is the super weight Yu et al. (2024) identify in this model.
  The neuron's activation at ordinary tokens has median 0.05 unconverted and
  48.6 with layer 0 converted and heads 8 and 18 merged, with 1,760 of 2,046
  ordinary positions above 10, against 380 at the first position.
- Why the merge flips it (`llama2_followup_neuron.json`). At ordinary tokens the
  neuron's gate input is the sum of the residual stream's and every head's
  contributions, and a few heads hold it negative. Head 8 contributes -1.3
  unconverted. Converting layer 0 on CAGA's pairs gives two later newline
  tokens large activations as well (1,952 and 572), and head 8 then puts 82
  percent of its attention on newline tokens (27 percent unconverted); merged
  with head 18, it puts 8 percent there, and its contribution to the gate
  becomes +18.5 against head 11's -5.1. On round 28's pairs (8 with 1, 18 with
  11) head 8 contributes +10.6 and head 11 -11.7, and the neuron stays off
  (median 0.007).
- The other collapses of round 28 are not this (`llama2_followup_firstlayers.json`,
  8 windows, unconverted 5.03; layers 2 to 31 converted, then layer 0, then
  layer 1; in brackets the median position's norm after layer 1 relative to
  the unconverted model's 2.7). Averaging on neighbouring pairs: 1,537, 777,
  941 (1.22). CAGA's construction on round 28's pairs: 53.3, 79.3, 266 (1.19);
  on neighbouring pairs 75.9, 327, 641 (1.06); on CAGA's pairs 97.3, 177,
  12,324 (25.8). The weighted fusion on round 28's pairs: 13.1, 14.5, 16.0
  (1.02); on neighbouring pairs 15.6, 15.9, 19.9 (0.99); on CAGA's pairs 13.3,
  15.8, 12,933 (22.0). The spike appears with the one pairing that merges heads
  8 and 18, under both constructions, and with no other. Leaving layers 0 and
  1 unconverted divides CAGA's construction's perplexity by 5.0 to 8.4 on the
  pairings without the spike and the fusion's by 1.2 to 1.3; for averaging it
  makes perplexity worse (1,537 against 941).
- At small scale it does not happen: on Llama-160m (the parallel session's CPU
  probe, `llama2_dev/prefix160m.py` and `.json`, 16 windows of 512 tokens,
  unconverted 33.7) the fusion grows smoothly with depth on both pairings, to
  80.3 on the fusion's matching and 81.2 on CAGA's at all 12 layers.
- Quantization fails differently (`llama2_quant_sink_probe.json`,
  `llama2_quant_firsttoken.json`; 8 windows, unconverted 5.03). Round-to-nearest
  weights, asymmetric, every linear layer of the decoder blocks: 4 bits per
  output channel 5.67, 3 bits per output channel 399 (OmniQuant reports 539 on
  the full test set), 3 bits in groups of 128 6.23, 2 bits in groups of 128
  2,714. No configuration puts an ordinary position above 100 after layers 1
  or 2. At 3 bits per channel, keeping layer 1 in bfloat16 gives 67.8 (its
  attention alone 124, its MLP alone 92.7), keeping the last layer 106,
  keeping layer 0 359, and quantizing only layers 0 and 1 gives 5.50.
  Quantizing layer 0 lowers the first position's activation after layer 1 from
  800 to about 340 in every configuration that quantizes it, including the one
  that quantizes nothing else (perplexity 5.50), so neither a spread nor a
  weakened first-token activation accounts for the 3-bit collapse.
Reading: a per-layer criterion, ours or CAGA's, cannot see this failure. One
merged pair in layer 1, harmless alone and within 6 percent of the best
matching in the fusion's residual, switches Llama-2-7B's super-weight neuron on
at ordinary tokens once layer 0 is converted, and later layers then treat those
tokens as attention sinks. Round 28's matching avoided it by pairing heads 8 and
18 elsewhere, not by anything in its criterion. A check on one calibration
window separates the cases before any evaluation: after layer 1 the median
position's largest entry, relative to the unconverted model's, is 0.84 to 0.97
for the conversions of layers 0 and 1 that did not collapse and 21 and 62 for
the two that did (perplexity 356 and 2,316), and in the full conversions the
median position's norm is 0.99 to 1.22 times the unconverted model's for the
five arms without the spike and 22 and 26 times for the two with it. Round 28's
statement that CAGA's construction depends on its pairing far more than the
fusion does holds for the two pairings round 28 tried, not for this third one,
on which both collapse. Low-bit round-to-nearest quantization of the same model
collapses through another route.

## Does one merged pair switch on a super weight in other models? (2026-10-30, preregistration 29)

Frozen at 4d02534 (`alignment_study/PREREGISTRATION29.md`). The gate FAILS, so
G1 to G5 are untested: 0 of 6. In four models, conversions to half the
key-value heads that differ from the fusion's reference matching only in one
of layers 0 to 4 (the minimum-cost matching forced to contain one pair, or for
Qwen2.5-7B the other perfect matchings), with perplexity on 4 WikiText-2
windows relative to the reference conversion (`guard/*.json`,
`guard_results.json`, `guard.log`, `eval_guard.log`).
- Gate FAILS: no member is severely damaged (5 times or more) in any of the
  four models. Mistral-7B-Instruct-v0.2: 98 members, 0.87 to 1.39 times the
  reference (reference 19.9, unconverted 6.98). OpenLLaMA-3B: 117 members, 0.84
  to 1.05 (31.4, 8.46). OLMo-1B: 117 members, 0.97 to 1.13 (57.1, 13.3).
  Qwen2.5-7B: 10 members, 0.97 to 2.21, the largest a different matching of
  layer 0 (29.9 on neighbouring pairs, 6.10).
- G1 to G5 untested. The check flagged none of the 341 benign members: its
  largest ratio was 0.89 to 1.04 everywhere, against the frozen threshold 2.35
  and 2.98 to 3.93 for Llama-2-7B's severe members in development.
- Reported: the weights locate each model's massive activation as the
  preregistration defines it. Mistral: layer 1, dimension 2070, neuron 7310,
  the super weight Yu et al. (2024) report for this model. OpenLLaMA-3B: layer
  2, dimension 2448, neuron 6987. OLMo-1B: layer 2, dimension 1786, neuron 2281.
  Qwen2.5-7B: layer 3, dimension 458, neuron 11223. The strongest key-value
  head's write strength onto that neuron's gate is 1.4 times the layer's median
  in Mistral, 5.9 in OpenLLaMA-3B, 4.9 in OLMo-1B and 2.2 in Qwen2.5-7B, against
  14.2 for head 8 in Llama-2-7B. Members that give the strongest writer a new
  partner in the massive-activation layer stay within 0.98 to 1.05 times the
  reference (12 in Mistral, 5 in OpenLLaMA-3B, 8 in OLMo-1B). The check took
  0.5 to 1.6 seconds per member against 3.6 to 26 seconds for the 4-window
  evaluation; the round ran 9,636 seconds.
Reading: the failure found in Llama-2-7B did not appear in 342 near-optimal
conversions of four other models, so it is not a general hazard of merging
early-layer heads at this population; within this round it remains a property
of Llama-2-7B. The check stayed silent on every harmless conversion, which
says it is cheap and does not cry wolf, but its recall outside Llama-2-7B could
not be tested. Post hoc, and from five models only: the one model with a
dominant writer onto its super neuron's gate (14 times the median, against at
most 6 in the others) is the one where moving that writer breaks the model.
That pattern is a hypothesis for a later test, not a result of this round.
