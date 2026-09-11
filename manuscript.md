# One Operator per Layer: Constrained Surrogates for Attention-Structure Findings and What They Leave Behind

Working draft, 2026-09-11. Every number in this document regenerates from a
committed script into a committed artifact; Appendix B lists the artifact
for each. Sections marked PENDING wait for runs that are preregistered and
in progress; nothing in them is filled in ahead of the artifact.

## Abstract

Analyses of attention heads routinely report structure: clusters, hubs,
spectral gaps, special heads, interaction graphs. We show that the standard
head-interaction pipeline produces these signatures on random causal softmax
matrices with no training behind them, that two short proofs explain why,
and that a family of constrained surrogates separates what a trained model
adds from what the container guarantees. Surrogates that keep only each
row's sharpness and self-mass return trained layers to the random picture;
adding one number per row, the attention on the shared sink token,
reproduces the real interaction statistics layer by layer. Behind this
sits a structural fact: the skew-symmetric generators of a layer's heads
share one dominant operator component, which at high-sink layers is the
ideal sink operator (cosine 0.71 to 1.00 across 230 high-sink layers of
eleven models from 0.1B to 7B parameters) and which cancels identically in
every pairwise commutator. A one-parameter law relating a layer's
shared-energy fraction to its interaction geometry holds out of sample in
three preregistered rounds (Spearman -0.76, -0.76, -0.70), along the
training of two models (Spearman -0.74 over 152 checkpoint-layer cells),
and its regime flip sits where a synthetic ensemble of noise plus one
shared column placed it in advance. Training replaces one shared operator
with another: untrained layers share the uniform causal operator and the
sink operator takes over within one checkpoint of sink formation. Per-head
sink profiles fix a floor under the shared energy that orders layers
correctly but leaves a residual the sink does not explain. We
release the nulls as a toolkit and apply them to published findings under
preregistered criteria, reporting each outcome as registered.

## 1. Introduction

Sanity checks have a good record in interpretability. Adebayo et al. (2018)
showed that several saliency methods pass visual inspection while failing
a five-minute test against randomized models and data. This paper makes
the same move for findings about the structure of attention heads.

The pipeline under study takes a layer's attention maps A_h (one T x T
row-stochastic matrix per head), forms the skew-symmetric generators
G_h = (A_h - A_h^T) / 2, and analyzes their interactions: the coupling
matrix C_ij = ||[G_i, G_j]||_F, its spectrum, its clustering, and the
heads it singles out. Variants of every step appear in the literature on
head similarity, head redundancy, head importance, and head graphs. Run on
fourteen Gaussian-logit causal softmax matrices, the pipeline finds a hub
head that couples to everyone, a spectral gap of 59, a cluster pattern of
three singletons and one block of eleven, and a "special" head by the
two-sigma rule (Figure 1). Nothing was trained. The two-sigma rule alone
flags a special component in 62 percent of noise draws at 14 heads and 96
percent at 32 (Figure 2). A formula search on structure constants
extracted from noise reaches the same fit quality as an archived search on
a real model.

Our contribution is in three parts. First, the container: row-stochasticity
forces one shared invariant direction on every attention layer
(Proposition 1), the generator norm of a causal head is exactly its
sharpness minus its self-mass (Proposition 2), and the coupling matrix is
bounded by a rank-one matrix in the head norms (Proposition 3), so a hub
and a gap are guaranteed whenever head sharpness is heterogeneous. Second,
the surrogates: per-row marginal-matched permutations keep every quantity
the propositions name and destroy cross-head alignment; column-preserving
permutations additionally keep the shared sink column; two dissociation
controls keep every head as concentrated as it was while removing the
shared column. Third, what the surrogates find in trained models: the
heads of a layer are, to first order, multiples of one operator plus a
residual, the operator is the sink, its contribution cancels in every
commutator, and one number per layer, the shared-energy fraction, predicts
the layer's interaction geometry across models, scales and training time.
The last two parts were established under eight preregistrations whose
thresholds were frozen and publicly timestamped before the runs, and whose
failures are reported as failures.

## 2. The container: three propositions and two cancellations

Notation. A layer has n heads with causal row-stochastic maps A_h in
R^{T x T}; G_h = (A_h - A_h^T) / 2; gn_h = ||G_h||_F; C_ij = ||[G_i, G_j]||_F
for i != j; r1 is the Pearson correlation between the off-diagonal entries
of C and the rank-one matrix gn_i gn_j.

Proposition 1 (one shared direction). For row-stochastic A and B,
(AB) 1 = 1 and [A, B] 1 = 0. Every product of attention matrices fixes the
uniform direction and every commutator annihilates it, so any analysis of
head interactions built from products or commutators lives in the
(T - 1)-dimensional complement of a direction shared by all heads. A
decomposition into one universal dimension plus the rest is present in
every softmax layer before training.

Proposition 2 (generator norm is sharpness minus self-mass). For causal
row-stochastic A, 2 ||G||_F^2 = sum_i (IPR_i - A_ii^2), where
IPR_i = sum_j A_ij^2 is the collision probability of row i. Verified to
4e-15 across 700 random heads. A head that attends sharply to any token
other than itself has a large generator norm; a sink head attending to
position 0 is the canonical case.

Proposition 3 (a hub is guaranteed). C_ij <= 2 gn_i gn_j. Across the
synthetic ensembles the off-diagonal C correlates with gn_i gn_j at median
0.96 (Gaussian logits), 0.92 (Gaussian plus a sink column) and 0.69
(Dirichlet rows matched to the Gaussian ensemble's mean sharpness). A
symmetric nonnegative matrix close to rank one has one dominant
eigenvalue, and its largest row sum belongs to the largest norm: the
coupling hub is the max-norm head in 81 percent of Gaussian draws.

Commutator cancellation. Write G_h = a_h S + E_h with S the layer's shared
operator. Then [G_i, G_j] = a_i [S, E_j] - a_j [S, E_i] + [E_i, E_j]: the
shared-times-shared term vanishes identically. Interaction statistics on
trained attention therefore measure how each head deviates from the
operator the heads share. On the development model the two cross terms
outweigh the deviation term by 4 to 25 times at every head pair tested.

Two lemmas used later. Causal generator inner products are never negative
(minimum 0 over 200 ensembles), and the expected inner product under the
marginal-matched surrogate has the closed form
E<G_h, G_k> = (1/2) sum_i i m_hi m_ki with m_hi the off-diagonal causal
row mass of head h at row i divided by i, matching sampling to 0.5
percent.

Base rates. The probability that the two-sigma rule flags at least one
special component on pure noise is 0.25 at n = 8, 0.56 at 12, 0.62 at 14,
0.71 at 16, 0.74 at 17, 0.88 at 24 and 0.96 at 32. Seed 147 of the
regenerated Gaussian ensemble, the first of 114 matches among 5,000 seeds
searched, reproduces an archived real-model summary in every particular
(unique two-sigma head, cluster pattern [1, 1, 1, 11], a gap above 50) and
is labeled as a searched exhibit.

Adaptive fits. A formula search over the same operator set used in the
archived analysis reaches best R^2 between 0.005 and 0.022 (median 0.011)
on three noise datasets, against 0.014 for the archived real-model fit;
the target itself is convention-dependent, since the SVD basis is defined
up to per-vector sign and the structure constants under two equally valid
conventions correlate at 0.26.

Depth profiles. Sweeping 24 pseudo-layers of random matrices through a
locality schedule that ends in strong self-attention, the mean pairwise
commutator norm falls 42-fold, from 1.06 to 0.025, with correlation -0.999
against diagonal mass. A depth-wise "crystallization" of head interactions
follows from attention locality with no training.

## 3. The battery

Null families. (i) Random causal or bidirectional softmax matrices with
Gaussian logits and lognormal per-head temperatures, matched in n and T.
(ii) Per-row marginal-matched surrogates: each row's causal off-diagonal
entries permuted independently, so row sums, row sharpness, self-mass and
every generator norm are preserved exactly (checked to 1e-12) while the
column each row attends to is destroyed. (iii) Column-preserving
surrogates: as (ii) with the sink column set fixed, where the set is the
modal sink column plus up to two further columns whose layer-level mass
reaches 0.10. (iv) An untrained model of the same architecture. (v) Two
dissociation controls that keep every head exactly as concentrated as it
was while removing the shared column: wrapped targets (a random
permutation of 1..n assigns each head a target column; every row's maximum
is swapped into the target, wrapped into the causal range where needed,
and the rest permuted) and a matched-geometry shift (each head's causal
rows cyclically shifted by a head-specific offset).

The dissociation controls were validated on synthetic ensembles before
use. The two robust clauses, collapse of the shared mode below half and
elevation of the per-pair z above the real value, hold in every trial for
the wrapped control at every head count and width once the shared energy
exceeds 0.67. Restoring the rank-one geometry (r1 above 0.8) is limited by
the causal width, not by the control: in 64-token windows no permutation
can host 32 distinct target columns in half of the rows, so no control
restores r1 at 32 heads there (median 0.51 to 0.68) although a genuinely
misaligned ensemble reaches 0.90; at 256 tokens every control restores it
in 8 of 8 trials (median 0.95 to 0.97). This is the mechanical explanation
of a registered failure in Section 4.

Statistics. Primary: the per-pair z of C against the marginal-matched
ensemble and the excess of rho (the top eigenvalue share of the cosine
Gram of the generators) over the marginal-matched value. Secondary: r1,
which is ill-conditioned when head norms are homogeneous (cv(gn) < 0.1)
and is flagged there; the eigengap; outlier and clustering rules; the
shared-energy fraction (mean over heads of the energy on the top principal
operator of the stacked generators); the cosine of that operator to the
ideal sink-set operator.

Report. For each statistic and null: the percentile of the real value in
the null distribution, and the fraction of the real effect, measured from
the random-softmax baseline, that the constrained null already reproduces.
The toolkit `hubsfree` implements the families, the statistics and the
report with a three-command interface (extract a layer's maps from any
Hugging Face model, audit them, demo the random-matrix coordinator), tests
the invariants to 1e-12, and installs from a wheel.

## 4. What the surrogates find in trained models

### 4.1 Development model and the first preregistration

On Qwen2.5-0.5B, marginal-matched surrogates stay at the random level
(r1 above 0.9 in 96 percent of layers across the held-out set) while the
real r1 falls to -0.5 at the deepest sink layers, and the column-preserving
surrogate follows the real curve layer by layer (Figure 3). The first
preregistered round (seven clauses) established the shared sink operator
on the development model and two others: the shared component is the
ideal sink operator at every high-sink layer (A6, 18 of 18), it spans
grouped-query KV groups so the architecture does not create it (A2, 11 of
14), instruction tuning leaves it untouched, and the cross terms dominate
the commutators (A5). Two clauses failed as registered: a universal
breakdown prediction (A1, 14 of 24 layers) and the first dissociation
control (A7, 6 of 14), whose design flaw (targets t_h = h mod 4 left
aligned clusters) was documented and replaced.

### 4.2 The alignment-fraction law and its regime flip

The layer's shared-energy fraction predicts its interaction geometry. The
hardening round located a regime flip in a fully synthetic ensemble, noise
plus one shared column, between shared energy 0.85 and 0.93; the
development model's layers flip in the same interval (0.83 to 0.84 intact,
0.92 to 0.94 broken). The second preregistration froze this before five
held-out models were downloaded: pooled Spearman(shared energy, r1) = -0.76
over their high-sink layers (H4), the sink-operator identification held in
5 of 5 models at 100 percent of high-sink layers (H1), the redesigned
dissociation control held in 5 of 5 (H3), one-column sufficiency in 4 of 5
(H2; TinyLlama at 57 percent), and the context and corpus transfers held
(H5, H6). A sharp regime-edge threshold (H4b) failed as registered. The
third preregistration, pushed publicly before execution on five models from
3B to 7B in five families, gave S1 5 of 5, S2 4 of 5 (OLMo-2 fails: its
sink sits on a mid-sequence column that moves between windows), S3 2 of 5,
and the law again at -0.76 over 130 layers (Figure 5). Across the 230
high-sink layers of the eleven models measured to that point, the shared
component's cosine to the ideal sink operator ranges from 0.71 to 1.00 and
exceeds 0.87 at 91 percent of layers, carrying 0.51 to 0.99 of each head's
energy (Figure 4).

### 4.3 The instruments repaired and the set rerun

Three weaknesses were recorded as failures or limitations: r1's
ill-conditioning, the dissociation control's loss of power at 32 heads, and
the one-column surrogate's failure on window-varying sinks. The sixth
preregistration restated S1 to S4 with the repaired instruments and one
protocol for all thirteen models (48 windows per model at T = 64; six
windows at T = 256 for the dissociation control; the two calibration
models reported but excluded from tallies). Outcome, 11 testable models,
207 high-sink layers: S1' 11 of 11 at 100 percent of high-sink layers; S2'
11 of 11, with OLMo-2 at 79 percent under the column set against 19
percent with one column and TinyLlama at 71 percent; S3' 11 of 11 at 100
percent of high-sink layers at T = 256, the shared mode falling to a median
0.07 to 0.09 of its real value and the median per-pair z rising from 78 to
271 in Mistral-7B across all 32 layers (Figure 8); S4 at Spearman -0.70
(bootstrap interval -0.76 to -0.62) over the 182 layers with cv(gn) >= 0.1
and -0.75 over all 207. The development model's 48-window values agree with
its 12-window values from the first round layer by layer (Spearman 0.99),
and its broken-layer set is identical. The three earlier failures resolved
without moving any threshold.

### 4.4 Training replaces one shared operator with another

An untrained network already has a dominant shared component. Random-init
Pythia models from their configs have shared energy 0.88 to 0.96 with the
sink cosine at 0.27 and sink mass 0.06: with near-uniform attention every
head carries the same operator, and a dictionary decomposition identifies
it as the uniform causal operator at cosine 1.00 in every layer. The fifth
preregistration, frozen before any checkpoint was downloaded, therefore
asked about the identity of the shared mode rather than its level. Across
ten checkpoints each of Pythia-160m and Pythia-410m (360 checkpoint-layer
cells): D0, no layer is high-sink at step 0 and the sink cosine is below
0.5 everywhere (pass); D1, Spearman(sink mass, sink cosine) = 0.935 over all
cells (pass); D1b, the cosine exceeds 0.7 in 100 percent of the 152
high-sink cells (pass); D2, the law along training at Spearman -0.736
(pass); D3, the first checkpoint at which the cosine exceeds 0.7 lies within
one checkpoint of the first at which sink mass exceeds 0.4 in 25 of 26
layers that end high-sink (pass); D4, a secondary prediction that the 160m
forms its sink by step 4000, fails (step 8000); D5, shared energy dips
below both its initial and final values during training in 75 and 96
percent of layers (pass). The initial component is broken up first, the
layers pass through the random-matrix regime at steps 512 to 1000 with no
sink anywhere, the sink then forms abruptly and the shared component
becomes the sink operator in the same checkpoint, and deeper alignment
carries the layers through the regime flip: r1 changes sign 14 times
between consecutive high-sink checkpoints, at interpolated shared energy
0.77 to 0.87 (median 0.83), against the synthetic curve's crossing at 0.785
(Figure 7).

### 4.5 The shared energy derived from per-head sink profiles

For causal attention with sink column c, the inner product of a head's
generator with the ideal unit sink generator uses only the head's column c:
<G_h, S_c> = sum_{i > c} A_h[i, c] / sqrt(2 (T - 1 - c)), so the energy of
G_h along the sink direction is m_h^2 (T - 1 - c) / 2 with m_h the head's
mean sink mass, and ||G_h||^2 is given by Proposition 2. The derived layer
value, the mean over heads of the ratio, uses each head's sink mass and row
sharpness and no cross-head information. Because the measured shared
operator maximizes the joint energy, the derived value should sit at or
below the measured one. The eighth preregistration set the gate at a linear
R^2 of 0.8 over pooled high-sink layers.

Outcome over the twelve non-development models (310 layers, 231
high-sink): the gate fails. The sink-only derived value explains 55 percent
of the variance in measured shared energy across high-sink layers (linear
R^2 0.55, bootstrap 0.42 to 0.69), against the registered 0.8; it orders
all 310 layers at Spearman 0.86 (G2, pass), sits at or below the measured
value at every high-sink layer with a median gap of 0.058 (G3, pass), and
the two-operator predictor improves the all-layer identity fit in 12 of 12
models with a pooled median error of 0.039 (G4, pass). The pooled fit is
low because models sit on different offset lines (intercepts 0.14 to
0.74); within seven of the twelve models R^2 exceeds 0.86 (Figure 9). The
registered reading applies: per-head sink mass and row sharpness fix a
floor under the layer's shared energy that lies close to the measured
value and orders layers correctly, but the residual above the floor is not
a fixed fraction, so the shared energy is measured structure with a
derived lower bound, not derived structure. A post-hoc description of the
residual: within every model it falls as the layer's sink mass rises
(Spearman -0.84 pooled), and it is largest where the sink sits on a
window-varying mid-sequence column (OLMo-2, median gap 0.31), which points
to a sink-set version of the derivation as the next registered step; a
post-hoc run of that version halves OLMo-2's gap and raises the pooled
R^2 to 0.75, still short of the gate, with the residual's dependence on
sink mass unchanged.

### 4.6 What the deviations are: sink profiles

Rebuilding each layer's heads from their real coefficients on the shared
operator plus random skew remainders of the right norm reproduces the r1
depth profile (Spearman 0.81) but not the per-pair z profile (Spearman
-0.38); adding the previous-token and uniform operators as further
structured directions does not change this (Spearman -0.35 and -0.34),
because those operators carry at most 0.1 percent of the energy at
high-sink layers. A direct comparison shows what the rebuilds miss: real
deviations commute with the shared operator about three times more
strongly than random deviations of the same norm (normalized commutator
0.51 to 0.57 against 0.18 at high-sink layers) and are 9 to 25 times more
concentrated, while no single head-specific column carries them. The
deviations are concentrated, but a sparse random deviation with matched
concentration on a random causal support does not recover z either
(Spearman -0.38): the threefold larger commutator comes from where the
entries sit relative to the sink geometry, not from how many there are.
Measured directly, a median 54 percent of each head's deviation energy at
high-sink layers lies in the sink column itself, against 3 percent for a
random deviation: the dominant deviation is the head's own sink-column
profile, the way its mass over rows departs from the shared shape. That
is what the column-preserving surrogate keeps and the marginal-matched
surrogate destroys. A rebuild that keeps each head's real sink column and
replaces everything else by random skew entries of the remaining norm
recovers the z profile the other constructions missed (Spearman 0.95 with
the real per-layer z, median error 1.1 z-units, against -0.32 and 18.7 for
the dense rebuild) while keeping the r1 profile (Spearman 0.86). Registered as
preregistration 9 and run on the twelve other models (310 layers), the
construction reproduces the per-layer z at Spearman 0.85 to 0.99 in eleven
models with median errors of 0.5 to 1.5 z-units (4.5 in OLMo-2), against
-0.66 to +0.82 and 15 to 22 for the dense rebuild, and pooled Spearman 0.95
over all layers; r1 follows at 0.80 to 0.99. TinyLlama fails the ordering
clause (0.31) at its fourteen low-sink layers, where the real z is -3 to
-12 without a sink, while its seven high-sink layers are reproduced. At
the resolution of these statistics a layer's heads are their sink-column
profiles plus noise: the shared operator, the commutator cancellation, the
sink-column surrogate's success, the derived floor under the shared energy
and the deviation structure are one fact seen five ways, and a layer
without a sink is the stated boundary.

## 5. Audits of published findings

Procedure (preregistered per target before any battery run): reproduce the
base result with the authors' code or a fully specified reimplementation;
register the statistic, the nulls and the survival criterion; run the
battery with 200 draws per family; label the outcome survives, shrinks,
matched or not reproducible; send the note to the authors with four weeks
to respond.

Target 1, Clark et al. (2019), heads cluster by Jensen-Shannon distance
and heads in the same layer are similar. Base result reproduced on
bert-base-uncased over 38 WikiText-103 windows: mean JS between heads in
the same layer 0.253 against 0.380 for different layers; a head's nearest
neighbor is in its own layer for 38.9 percent of heads (chance 7.7). The
statistic (the contrast) and the four null families are frozen; the
registered expectation is survival against random and untrained nulls and
shrinkage between 20 and 70 percent against marginal-matched and
separator-column-preserving surrogates. Battery: PENDING registration.

Target 2, Kovaleva et al. (2019), five attention-pattern types with the
vertical pattern dominant. Not reproducible in its stated form: the
proportions rest on a classifier trained on about 400 manually annotated
maps of fine-tuned models, and neither the annotations, the classifier nor
the checkpoints were released. Logged and replaced.

Target 3, CHAI (2024), heads within a layer are redundant in which tokens
they attend to during decoding, with one or two large clusters per layer
at within-cluster correlation above 0.95 and correlation increasing with
depth. The object was reimplemented from the paper (the last token's
attention row per head over C4 documents, Pearson correlation across
heads, complete-linkage clusters at 0.95) and run on Mistral-7B as the
open substitute, at 32 documents of 1,024 tokens. Partially reproduced:
cross-head correlation is high at every layer (0.55 to 0.98) and layers 1
to 6 carry one large cluster (up to 91 percent of heads at 0.95), but only
4 of 32 layers have a majority cluster and correlation falls rather than
rises with depth (Spearman -0.38), tracking the sink mass of the last row
instead. On the paper's own OPT family (OPT-6.7B, same protocol) the split
reverses: correlation rises with depth (0.34 to 0.70, Spearman 0.49) and no
layer carries a majority cluster. Criteria frozen for both models; the
registered expectation is that the column-set surrogate matches both
statistics.

Target 4, Dewage et al. (2026), singular values of the projection weights
above the Marchenko-Pastur edge carry the learned structure. The recipe is
fully specified and was reimplemented, the stated code URL being
unreachable on two dates. On Mistral-7B the mean outlier counts per matrix
(Q 1511, K 341, V 212, O 1450) match the paper's Table II to within 0.3
percent, and the paper's percentage column (87.5, 74.7, 43.6, 84.6) is
reproduced as the outliers' share of spectral energy (89.3, 75.7, 43.0,
84.4) rather than as the share of singular values (21 to 37 percent).
Reproduced; criteria frozen. Nulls for the battery: Gaussian weights of
matched shape and norm, row-norm-matched random weights, within-matrix
permutation, and config-initialized weights of the same architecture.

Target 5, Retrieval Heads (2024), the positive control: 3 to 6 percent of
heads copy from the context during needle retrieval (retrieval score above
0.1) and masking them breaks retrieval. A compact reimplementation of the
score on Qwen2.5-7B at contexts of 1,024 and 2,048 tokens (20 instances)
retrieved the needle every time and put 4.2 percent of heads above 0.1,
with the strongest in layers 14, 22 and 23; on the paper's own
Mistral-7B-Instruct-v0.2 the same protocol gives 3.4 percent, again with
every needle retrieved. Reproduced; criteria frozen.
The battery must not remove this structure; if it does, its thresholds are
revised before any other outcome is reported.

## 6. A practical test that failed: which heads can be merged

Preregistration 7 asked whether the shared-operator description has a
tolerance consequence. For each layer of Qwen2.5-1.5B, 3B and 7B, the two
query heads of a KV group with the highest generator cosine were merged
(query and output slices averaged) and the change in next-token loss on
16,384 held-out tokens measured, against the bottom-cosine pair and a
random pair; the layer's shared energy from the rerun was the predictor.
Development calibration on Qwen2.5-0.5B had the top merge cheaper than the
bottom merge in 23 of 24 layers and a Spearman of -0.32 between shared
energy and cost. Registered on the three models: M1, top cheaper than
bottom in at least 2/3 of layers in 3 of 3 models, fails (75, 64 and 54
percent of layers; 1 of 3); M2, pooled Spearman(shared energy, top-merge
cost) at or below -0.3 with an interval excluding zero, fails (+0.13,
interval -0.10 to +0.32; per model -0.00, +0.18, +0.20); the secondary
clauses fail as well (pair-level Spearman -0.28 over 180 pairs, in the
predicted direction). What the round establishes instead: merging one pair
of heads within a KV group costs a median 0.0007 to 0.0013 nats per token,
under 0.002 nats in 71 to 92 percent of layers, and neither the
attention-map cosine nor the layer's shared energy says which pair or
which layer. One random-pair merge at layer 0 of the 1.5B cost 0.23 nats;
no other merge in 92 layers cost above 0.04. The shared-operator
description is a statement about attention geometry, and at this
granularity it has no tolerance consequence (Figure 10).

## 7. Limitations

Results cover thirteen models up to 7B parameters in six families, English
text and code, windows up to 256 tokens for the statistics and 512 for
perplexity. Where the sink sits on a mid-sequence, window-varying column
(OLMo-2, late Qwen2.5-3B layers), the sink-operator identification weakens
(cosine 0.61 to 0.82 with one column) and the column set repairs most but
not all of it (four of OLMo-2's 19 high-sink layers keep a sufficiency
residual above 0.4). Restoring the rank-one geometry under a dissociation
control needs a causal width of roughly eight tokens per head. The
deviation directions carry structure the shared-operator description does
not capture (Section 4.6). The commutator pipeline is one instrument among
many the audits address; it is the one whose failure started this work.
The dynamics result rests on one model family. The derivation gate and the practical test both failed as
registered; each is reported in full. The audit
criteria are frozen for four targets but their batteries wait for public
registration; the reproductions of Targets 3 and 5 ran at reduced scale
(fewer documents, shorter contexts) than the sources.

## 8. Recommendations

For any structure finding about attention heads: report the base rate of
the detection rule at the actual head count; run the marginal-matched
surrogate and report the fraction of the effect it reproduces; run the
column-preserving surrogate and report the same; if the finding concerns
alignment or shared structure, run a dissociation control at a width of at
least eight tokens per head; report adaptive fits against the best fit the
identical search reaches on shape-matched noise; freeze thresholds before
the runs and report failures. The toolkit runs the first four in one
command.

## Appendix A. Preregistration ledger

| Round | Frozen at | What it tested | Outcome |
|---|---|---|---|
| 1 | commits 41769a9, 6e57f84 | A1 to A7 on the development model, GPT-2, Pythia-160m | A1 fail, A2 to A6 pass, A7 fail (control redesigned) |
| 2 | 9ab1c28, before download | H0 to H6 on five held-out sub-2B models | 7 of 8 clauses pass; H2 fails on TinyLlama, H4b fails |
| 3 | 9dd433a, pushed before execution | S1 to S4 on five 3B to 7B models | S1, S2, S4 pass; S3 fails (causal width) |
| 5 | 68b65f9, pushed before download | D0 to D5, Pythia checkpoints | D0 to D3, D5 pass; D4 (secondary) fails |
| 6 | 3d5b337, pushed before execution | S1' to S4 with repaired instruments, thirteen models | 4 of 4 in 11 of 11 |
| 7 | 6b8f8ea, pushed before execution | M1 to M3, head merging | M1, M2 fail; secondary clauses fail; costs reported |
| 8 | b9b8ef0, pushed before execution | G1 to G4, derived shared energy | G1 (gate) fails at R^2 0.55; G2 to G4 pass |
| 9 | 6e6b253, pushed before execution | P1 to P4, sink-profile generative model | P1 to P4 pass (11 of 12 models; TinyLlama's low-sink layers the exception) |
| 4 (draft) | criteria for Targets 1, 3, 4, 5 frozen by commit | audits | Target 2 not reproducible; batteries pending public registration |

## Appendix B. Numbers ledger

| Number | Artifact |
|---|---|
| base rates 0.25 to 0.96; seed 147, 114 of 5,000; gap 59; median r1 0.96 / 0.92 / 0.69 | results.json (sections 4.1, 4.2), regenerated by experiments.py, checked by CI |
| noise fit R^2 0.005 to 0.022, archived 0.014; sign-convention correlation 0.26 | results.json section 4.3 |
| 42-fold fall, correlation -0.999 | results.json section 4.4 |
| A1 to A7 | alignment_study/gram_theorem.json, robustness_addendum.json, NOTE.md |
| cross terms 4 to 25 times | alignment_study/robustness_addendum.json (term_decomposition) |
| H0 to H6, -0.76 | alignment_study/heldout_round.json |
| S1 to S4, -0.76 over 130 layers | alignment_study/scale_round_results.json |
| 230 layers, cosine 0.71 to 1.00, 91 percent above 0.87 | alignment_study/heldout_round.json, scale_partial/, gram_theorem.json (README ranges) |
| toy regime flip 0.85 to 0.93; crossing 0.785 | alignment_study/tier2_toy.json, posthoc_dynamics.json |
| controls on the toy | alignment_study/control_redesign_toy.json, control_redesign_toy2.json |
| S1' to S4, -0.70 [-0.76, -0.62], Mistral z 78 to 271 | alignment_study/rerun_results.json, rerun/ |
| dev anchor Spearman 0.99 | alignment_study/rerun/Qwen2.5-0.5B.json against tier1_robust.json |
| random-init shared energy 0.88 to 0.96, cosine 0.27; uniform operator cosine 1.00 | alignment_study/dynamics_dev_calibration.json, kmode_dev_calibration.json |
| D0 to D5, 0.935, -0.736, 25 of 26, 14 crossings | alignment_study/dynamics_results.json, posthoc_dynamics.json |
| derived shared energy, G1 to G4 | alignment_study/derivation_results.json, derivation/ |
| deviation rebuilds | alignment_study/toy_structured_dev.json, toy_sparse_dev.json, toy_sinkprofile_dev.json |
| sink-profile round, P1 to P4 | alignment_study/sinkprofile_results.json, sinkprofile/, posthoc_sinkprofile.json |
| merge round, M1 to M3, costs | alignment_study/merge_results.json, merge/ |
| merge calibration | alignment_study/merge_dev_calibration.json |
| Target 1 base result | audits/clark2019/base_result.json |
| Target 3 partial reproduction | audits/chai/Mistral-7B-v0.1_base_result.json |
| Target 4 reproduction | audits/dewage2026/Mistral-7B-v0.1_base_result.json |
| Target 5 reproduction | audits/retrieval_heads/Qwen2.5-7B_base_result.json |
| MPS fidelity | alignment_study/platform_fidelity.json |
