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
release the nulls as a toolkit and apply them to four published findings
under criteria frozen before any battery ran. Three of the four shrink: the
layer clustering of BERT's heads is three fifths the shared separator
columns, the cross-head redundancy behind a KV-compression method is
mostly the shared sink column on Mistral-7B and about two thirds of it on
OPT-6.7B, and 85 to 89 percent of the Marchenko-Pastur "outliers" reported
for the square attention projections are produced by a Gaussian matrix of
the same shape and norm. Retrieval heads,
included as the positive control, are untouched by every null. Weighting
attention by the vectors it moves halves the sink's account of head
similarity, and a comparison that brings in values and output weights
predicts the damage of merging two heads far better than any map statistic
and chooses cheaper merges when many pairs are merged. The head-importance
heuristic that reads attention weights measures the sink and points the
wrong way, and the sink itself is not a no-op: removing its contribution
costs 28 to 68 times as much as removing an ordinary token's, mostly as a
constant bias. Each
registered expectation that failed is reported as a failure; one of them
failed by demanding statistical indistinguishability where the question
was an effect size.

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
The last two parts were established under nine preregistrations whose
thresholds were frozen and publicly timestamped before the runs, and whose
failures are reported as failures.

### 1.1 Related work and the lineage of the nulls

Attention as an object of study. Clark et al. (2019) and Kovaleva et al.
(2019) established the descriptive program this paper audits: heads
cluster by the distance between their maps, heads in one layer resemble
one another, and a few pattern types recur, with a "vertical" pattern on
separator tokens the most common. Michel et al. (2019) and Voita et al.
(2019) showed that most heads can be pruned, which turned head similarity
into an engineering question (CHAI, 2024, clusters heads by the correlation
of their attention rows to share keys and values). Abnar and Zuidema (2020)
propagated attention through depth (rollout), and Kobayashi et al. (2020)
showed that attention weights alone misstate a head's effect because the
value vectors carry norm structure; the debate on whether attention weights
explain anything (Jain and Wallace, 2019; Wiegreffe and Pinter, 2019)
concerns explanation, whereas this paper concerns structure that is
claimed to exist in the weights themselves.

Sinks and registers. The shared column at the center of our results is the
attention sink: Xiao et al. (2024) kept the first tokens in the cache
because removing them breaks streaming generation; Darcet et al. (2024)
showed that vision transformers store global information in high-norm
"register" tokens and that adding explicit registers removes them;
Bondarenko et al. (2023) traced quantization outliers to heads that attend
to a fixed token in order to do nothing; Sun et al. (2024) described the
massive activations that make such a token attractive; Gu et al. (2025)
and Barbero et al. (2025) studied when the sink emerges in training and
why the first token is a natural place for it. Our contribution to this
literature is not the sink itself but a measurement: what share of the
second-order statistics that other analyses report is the sink, and what
is left when it is removed while every head keeps its concentration.

Null models. The battery is assembled from three older traditions. From
network science come constrained randomizations: Maslov and Sneppen (2002)
rewired protein networks while preserving every node's degree, the
configuration model (Newman, 2003) samples graphs with a fixed degree
sequence, and both exist because degree heterogeneity alone produces hubs,
clustering and spectral gaps. Our per-row marginal-matched surrogate is
the row-stochastic analogue: row sums, row sharpness and self-mass are
fixed, and Proposition 3 states what that alone implies for the coupling
matrix. From nonlinear time series come surrogate data tests (Theiler et
al., 1992; Schreiber and Schmitz, 2000), which fix a chosen set of
statistics exactly and treat every other statistic as the test; the
column-set-preserving surrogate fixes one more statistic, the mass on the
shared columns, in this spirit. From neuroscience comes the tensor maximum
entropy null of Elsayed and Cunningham (2017), which showed that structure
reported in population recordings (low-dimensional dynamics, condition
tuning) is an expected byproduct of the marginal covariances along each
tensor mode; the question asked here of attention, whether the reported
structure is a byproduct of simpler constraints, is the same question. The
untrained-architecture null follows Adebayo et al. (2018). At the weight
level, the Marchenko and Pastur (1967) law is the null that Target 4 uses
by construction, and Martin and Mahoney (2021) showed that trained weight
matrices have heavy-tailed spectra whose "outliers" are a signature of
training rather than of any one direction, which is why the battery adds
entry-permuted and norm-matched nulls next to the Gaussian one.

| Null in this work | What it preserves | Lineage |
|---|---|---|
| random causal softmax (Gaussian logits, lognormal per-head temperature) | row-stochasticity, causal support, the sharpness distribution | random-graph baseline; randomized models of Adebayo et al. (2018) |
| per-row marginal-matched permutation | row sums, row sharpness, self-mass, the diagonal | degree-preserving rewiring (Maslov and Sneppen, 2002); configuration model (Newman, 2003); constrained surrogates (Theiler et al., 1992) |
| column-set-preserving permutation | the above plus the mass on the shared sink columns | surrogates that fix several statistics (Schreiber and Schmitz, 2000); tensor maximum entropy (Elsayed and Cunningham, 2017) |
| dissociation controls (wrapped target, matched-geometry shift) | each head's concentration and row geometry, with the shared column moved | interventions on sink and register tokens (Xiao et al., 2024; Darcet et al., 2024) |
| untrained same-architecture model | architecture, initialization, inputs | Adebayo et al. (2018) |
| weight-level: Gaussian norm-matched, row-norm-matched, permuted entries, initializer scale | shape and norm; row norms; the entry distribution; the initialization | Marchenko and Pastur (1967); Martin and Mahoney (2021) |

References for this subsection: Abnar and Zuidema, ACL 2020; Adebayo et
al., NeurIPS 2018; Barbero et al., arXiv 2025; Bondarenko, Nagel and
Blankevoort, NeurIPS 2023; CHAI, Agarwal et al., 2024; Clark et al., ACL
BlackboxNLP 2019; Darcet et al., ICLR 2024; Elsayed and Cunningham, Nature
Neuroscience 2017; Gu et al., ICLR 2025; Jain and Wallace, NAACL 2019;
Kobayashi et al., EMNLP 2020; Kovaleva et al., EMNLP 2019; Marchenko and
Pastur, Mathematics of the USSR-Sbornik 1967; Martin and Mahoney, JMLR
2021; Maslov and Sneppen, Science 2002; Michel, Levy and Neubig, NeurIPS
2019; Newman, SIAM Review 2003; Schreiber and Schmitz, Physica D 2000; Sun
et al., 2024; Theiler et al., Physica D 1992; Voita et al., ACL 2019;
Wiegreffe and Pinter, EMNLP 2019; Xiao et al., ICLR 2024.

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

Uncertainty. Layers of one model are not independent draws. Every interval
quoted for a pooled correlation is a two-stage bootstrap that resamples
models and then layers within a model (2,000 draws;
`alignment_study/clustered_stats.py`); the per-layer intervals of the round
evaluations are kept in the JSON files for comparison, and a random-effects
average of the per-model correlations is reported beside each pooled value.
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
(model-clustered bootstrap interval -0.78 to -0.57; per-layer -0.76 to
-0.62) over the 182 layers with cv(gn) >= 0.1
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
R^2 0.55, model-clustered bootstrap 0.24 to 0.93; per-layer 0.42 to 0.69),
against the registered 0.8; it orders
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

### 4.7 Other attention designs

The first thirteen models all use rotary or learned positions with
multi-head or grouped-query attention. Preregistration 10, frozen before
download, ran the protocols of preregistrations 6 and 9 on four open
models with designs outside that set: Falcon-7B (multi-query attention, one
key and value head for 71 query heads), BLOOM-7b1 (ALiBi position bias),
OPT-6.7B (learned absolute positions, plain multi-head attention) and
Qwen3-8B. All four form a first-token sink (19 to 29 high-sink layers). The
shared component is the ideal sink operator at 100 percent of high-sink
layers in every model (cosine 0.91 to 1.00); the column set suffices at 76
to 100 percent of layers; the wrapped-target control dissociates at 100
percent with r1 restored at 96 to 100 percent; the law holds at Spearman
-0.84 over 96 layers; and the sink-profile rebuild reproduces the per-layer
z at Spearman 0.90 to 0.98 per model, 0.97 pooled over 130 layers, with
median errors of 0.6 to 1.1 z-units. BLOOM's shared energy is the lowest
measured (0.57 to 0.69) at an unchanged sink cosine, so ALiBi yields a
weaker but equally sink-shaped operator; Falcon's shared key and value set
gives the strongest sink profiles (0.96 of generator energy). The
statements of Sections 4.2 to 4.5 therefore concern softmax attention with
a sink token, across ten families, three position schemes and three
attention layouts.

## 5. Audits of published findings

Procedure (preregistered per target before any battery run): reproduce the
base result with the authors' code or a fully specified reimplementation;
register the statistic, the nulls and the survival criterion; run the
battery with the frozen draw counts (200 per map-level family, 100 for
Target 1, 20 per weight-level family); label the outcome survives, shrinks,
matched or not reproducible; send the note to the authors with four weeks
to respond. The criteria were frozen by public commit (`ee4e267`) before
any battery ran; the evaluation script and the readings it applies were
committed before any result existed; every outcome below is reported as
the frozen wording decides it, with the effect sizes beside the verdict
(`audits/RESULTS.md`, regenerated from the result files).

Target 1, Clark et al. (2019), heads cluster by Jensen-Shannon distance
and heads in the same layer are similar. Base result reproduced on
bert-base-uncased over 38 WikiText-103 windows: mean JS between heads in
the same layer 0.253 against 0.380 for different layers; a head's nearest
neighbor is in its own layer for 38.9 percent of heads (chance 7.7). The
statistic (the contrast) and the four null families are frozen; the
registered expectation is survival against random and untrained nulls and
shrinkage between 20 and 70 percent against marginal-matched and
separator-column-preserving surrogates. Battery (100 draws, 38 windows):
the contrast D = 0.127 survives random maps and five untrained BERTs, both
of which give D of order 1e-5. The per-row marginal-matched surrogate
reproduces 9.5 percent of D, below the registered band, so the registered
expectation fails on that clause; the separator-column surrogate, which
keeps the [CLS] and [SEP] columns and permutes the rest of every row,
reproduces 60 percent of D and 35 of the 40 percentage points of
same-layer nearest neighbors, inside the band. Label: shrinks. The layer
clustering of BERT's heads is not a marginals effect, as the registered
guess had it; it is three fifths the shared vertical pattern on the
separators and two fifths same-layer similarity that the separator columns
do not carry.

Target 2, Kovaleva et al. (2019), five attention-pattern types with the
vertical pattern dominant. Not reproducible in its stated form: the
proportions rest on a classifier trained on about 400 manually annotated
maps of fine-tuned models, and neither the annotations, the classifier nor
the checkpoints were released. Logged and replaced.

Target 3, CHAI (2024): heads within a layer attend to the same tokens
during decoding, the redundancy its head-clustering method exploits, and
the correlation between heads increases with depth. The paper does not
define which attention rows it correlates. Its Figure 2 shows three
clusters above 0.95 within-cluster correlation for one layer and one
decoding step of LLaMA-7B, and its Figure 13 one or two large clusters for
one layer. Its code, released after the paper and not linked from it,
holds the merge patch and the clustering script but no code for the
correlation figures. We reconstructed the statistic as the Pearson
correlation of the last token's attention row across heads, averaged over
C4 documents, with complete-linkage clusters at 0.95. That reading is ours,
and the authors are asked about it. On Mistral-7B as the open substitute,
at 32 documents of 1,024 tokens, cross-head correlation is high at every
layer (0.55 to 0.98), layers 1 to 4 carry a majority cluster at 0.95 (up
to 91 percent of heads), and correlation falls rather than rises with depth
(Spearman -0.38), tracking the last row's sink mass (Spearman 0.85). On
OPT-6.7B, from the OPT family the paper evaluates, correlation rises with
depth (0.34 to 0.70, Spearman 0.49) and no layer carries a majority
cluster. Criteria frozen for both models; the registered expectation is
that the column-set surrogate matches both statistics.

Battery (200 draws). On Mistral-7B the registered expectation fails as
written. The sink-set surrogate, which keeps only each row's sink columns
and diagonal entry, reproduces a median 96 percent of the cross-head
correlation excess over the random null, and at least 77 percent in every
layer. But the real correlation lies above all 200 draws in every layer, so
the frozen "matched" criterion (percentile between 5 and 95) is met in
none. The marginal-matched surrogate, which permutes the sink column away,
reproduces 0 to 6.5 percent; random rows and five untrained models sit
below the real value everywhere. Label: shrinks in 32 of 32 layers. On
OPT-6.7B the sink-set surrogate reproduces a median 68 percent (3 to 92
percent), rising with depth (Spearman 0.97 with the layer index, 0.69 with
the last row's sink mass) and falling short of the real cluster share in
every layer; the marginal surrogate reproduces a median 1 percent, and the
real value again lies above every draw. Label: shrinks in 32 of 32 layers.
On Mistral-7B the redundancy is mostly the shared sink columns. On OPT-6.7B
they account for about two thirds of it in the median layer and little in
the early layers, where the last row puts almost no mass on the first token
and something other than the sink drives the correlation.

Target 4, Dewage et al. (2026): singular values of the attention projection
weights above the Marchenko-Pastur edge encode the dominant component of the
learned structure. The recipe is fully specified (their Equation 1) and was
reimplemented while the stated code URL was unreachable; the repository has
since become public, and its recipe is the same. On Mistral-7B the mean
outlier counts per matrix (Q 1511, K 341, V 212, O 1450) match the paper's
Table II to within 0.3 percent, and its energy ratios (87.5, 74.7, 43.6,
84.6; their Equation 2) are reproduced within 0.2 to 1.8 points (89.3,
75.7, 43.0, 84.4).
Reproduced; criteria frozen. Nulls for the battery: Gaussian weights of
matched shape and norm, row-norm-matched random weights, within-matrix
permutation, and Gaussian weights at the initializer scale. Battery (20
draws per family, 128 matrices): the registered expectation fails in all
four projection types. The real counts do lie above all 20 draws of every
null, but a Gaussian matrix matched only in shape and Frobenius norm
already produces 85 to 89 percent of the reported count in the square
projections, 65 percent in V and 40 percent in K; permuting the real
entries within a matrix matches the Gaussian null to three digits, so the
entry distribution is not the mechanism; row-norm-matched weights reproduce
71 to 88 percent of the energy share, against the registered "less than 50
percent". Label: shrinks in all four types.

Why the Gaussian null has outliers at all is worth stating, and is post hoc
rather than registered. The recipe estimates the noise scale as
sigma^2 = median(s^2) / (1 + gamma) and sets the edge at
sigma^2 (1 + sqrt(gamma))^2. For an m x n matrix with i.i.d. entries of
variance v and m <= n, the eigenvalues of W W^T are v n times a
Marchenko-Pastur law of ratio c = m / n with edge v n (1 + sqrt(c))^2; the
recipe's formula agrees only if median(s^2) = v (m + n), while the actual
median is v n times the median of the law. The edge therefore lands inside
the bulk, at a third of its true value for square matrices and three
quarters for the K and V shape. The paper's own Section VII notes that the
median-based estimate differs slightly from the exact Marchenko-Pastur
median and lists random-matrix controls as future work; the nulls above are
those controls. Refitting the scale by matching the law's median or its
mean, neither of which needs to know v, puts a Gaussian matrix at zero
outliers and leaves the real matrices with hundreds. The real counts then
fall by a factor of 1.3 to 9, and the two calibrated estimators differ from
each other by a factor of 1.8 to 4.6 (Q, four layers: 1583 by the recipe,
828 and 182 by the two fits). The qualitative claim survives every
calibration: trained attention weights carry spectral structure a matched
Gaussian does not. The published counts mostly reflect the estimator: under
the recipe a random matrix already reaches 85 to 89 percent of them in the
square projections.

Target 5, Retrieval Heads (2024), the positive control: 3 to 6 percent of
heads copy from the context during needle retrieval (retrieval score above
0.1) and masking them breaks retrieval (arXiv version 1). A compact
reimplementation of the score on Qwen2.5-7B at contexts of 1,024 and 2,048
tokens (20 instances) retrieved the needle every time and put 4.2 percent
of heads above 0.1, with the strongest in layers 14, 22 and 23; on the
paper's Mistral-7B-Instruct-v0.2 the same protocol gives 3.4 percent, again
with every needle retrieved. Our prompt ends with the first half of the
needle sentence and our score divides by the whole needle, so scores top
out near 0.6 rather than 1 and the threshold of 0.1 is correspondingly
stricter than the source's. On Llama-2-7B-80K, the paper's running example,
our haystack and prompt retrieve only half of the needles and 0.7 percent
of heads score above 0.1. The frozen inclusion rule repeated the test with the
source repository's own haystack, needles, insertion rule, prompt and
success criterion: the needle is then retrieved in 20 of 20 instances, so
the earlier failure was our haystack and prompt, but 1.3 percent of heads
score above 0.1 at 1K and 2K tokens on 20 instances, below the rule's 2
percent floor, and the model stays excluded as the rule was applied. That
repeat scored heads against the whole needle span, whereas the source's
code scores against the answer span; rerun with the answer span, 20 of
1,024 heads (1.95 percent) score above 0.1, still below the floor by one
head, so the model stays excluded as the rule is written, by that margin. Reproduced on Qwen2.5-7B and on the
paper's Mistral-7B-Instruct-v0.2; criteria frozen. The battery must not
remove this structure; if it does, its thresholds are revised before any
other outcome is reported. Battery (200 draws, 20 instances per model): on
Qwen2.5-7B and on Mistral-7B-Instruct-v0.2, random rows, marginal-matched
rows and sink-column-preserving rows put the fraction of heads above 0.1
and the top-10 score at the 100th percentile of 200 draws and reproduce
0.7 to 1.1 percent of the top-10 score; the three untrained
initializations score zero. E3 holds on both models: the surrogates that
absorb the layer clustering of Target 1 and the redundancy of Target 3 do
not touch a head set that copies specific context tokens, which is what a
permutation of where a row looks cannot produce. Because such a permutation
destroys token copying almost by construction, the control shows that the
battery can return "survives"; it does not calibrate the nulls for the
distribution-level statistics of Targets 1 and 3. What supports those two
results is the pattern across nulls: random and untrained maps reproduce
nothing, row values alone reproduce 1 to 10 percent, and only the nulls
that keep the shared columns reproduce most of the effect.

Registered expectations, with every battery complete. E3 holds on both
control models. E1, E2 and E5 fail as written, for different reasons. E1
(Target 3) failed on its wording on Mistral-7B, where the sink-only
surrogate reproduces a median 96 percent of the correlation but stays
distinguishable at 200 draws, with residuals of 0.2 to 23 percent across
layers, and partly on substance on OPT-6.7B, where it reproduces about two
thirds. E2 (Target 1) failed because the marginals reproduced less of
BERT's layer clustering than predicted, 9.5 percent against 20 to 70: a
miss in the direction against the container thesis, while the separator
columns reproduced most of it, inside the band. E5 (Target 4) failed
because row-norm-matched weights reproduced more of the energy share than
predicted, 71 to 88 percent against under 50, and because in K the permuted
null matched only 40 percent of the outlier count against the predicted
half. E4 was dropped before the freeze. The falsification clause, which
would have rejected the container-geometry thesis had every audited claim
survived with shrinkage under 20 percent, is not triggered: no audited claim
survives that way.

Only one of these failures, E1 on Mistral-7B, came from asking for
statistical indistinguishability where the question was an effect size.
Preregistrations 11 and 12 state every clause as an effect size with an
interval.

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
model-clustered interval -0.10 to +0.35; per model -0.00, +0.18, +0.20); the secondary
clauses fail as well (pair-level Spearman -0.28 over 180 pairs, in the
predicted direction). What the round establishes instead: merging one pair
of heads within a KV group costs a median 0.0007 to 0.0013 nats per token,
under 0.002 nats in 71 to 92 percent of layers, and neither the
attention-map cosine nor the layer's shared energy says which pair or
which layer. One random-pair merge at layer 0 of the 1.5B cost 0.23 nats;
no other merge in 92 layers cost above 0.04. The shared-operator
description is a statement about attention geometry, and at this
granularity it has no tolerance consequence (Figure 10).

### 6.1 The test repeated with function in the loop

Every analysis up to this point compares attention maps with surrogate
maps, so every result is a statement about map statistics. Preregistration
11 asked what changes when attention is weighted by what it moves. It
measured, on four models held out from development (Qwen2.5-1.5B,
TinyLlama-1.1B, Mistral-7B, Qwen2.5-7B), the vectors each position
contributes through a head, and it repeated the merge of round 7 on 1,152
pairs in 27 layers with a cleaner outcome: the KL divergence of the merged
model's next-token distribution from the unmerged one. Development on
Qwen2.5-0.5B had contradicted the hypothesis that motivated the round, that
removing the sink columns would make map similarity predict function; the
preregistration records the change and registers that comparison as no
gain. Every clause is an effect size with a model-clustered interval.

The sink is small per position but not negligible in output. Its value
vectors, carried through the head's output weights, are about a ninth the
size of the other positions' (median 0.11, interval 0.06 to 0.14), and its
share of value-weighted attention is 0.36 of its share of raw attention
(interval 0.32 to 0.41; F1 passes). Its contribution to a head's output
vector is nevertheless 0.47 of the output norm, because the contributions of
the other positions partly cancel. Weighting attention by value norms halves
how much of head similarity the sink column accounts for: the sink-kept null
reproduces 0.51 less of it on weighted maps than on raw maps (interval 0.47
to 0.56; F2 passes).

Map similarity is a weak proxy for what heads do, and the sink is not the
reason. Within a layer, the plain distance between two heads' maps predicts
the damage of merging them at Spearman 0.38 and the generator cosine of
round 7 at 0.30. Excluding the sink columns from the map distance changes
nothing (difference -0.02, interval -0.06 to 0.01; F3c passes). Weighting the
difference by the values raises the correlation to 0.51, and the
first-order change of the layer output, which also brings in the output
weights, to 0.65 (interval 0.52 to 0.78; F3d passes; its advantage over the
map distance is 0.27, interval 0.14 to 0.43; F3a passes). The ordering, output
perturbation first and value-weighted distance second, holds in each of the
four models. The output perturbation approximates the merge's own effect, so
its correlation is partly expected by construction; what the round measures
is how much the map leaves out.

Two registered clauses fail. The generator cosine is not reliably worse
than the plain map distance (difference 0.07, interval -0.03 to 0.18, against
a registered 0.10 with an interval above zero; F3b), so development's gap of
0.27 did not hold out of sample. And the practical criterion is not
established (F4): the merge chosen by output perturbation is not reliably
cheaper than the one chosen by the generator cosine. The two choose the same
pair in 10 of 27 layers; the former is cheaper in 11 and costlier in 6, a
median cost ratio of 1.0. Its choices rank at the 8th percentile of a layer's
merge costs against the 23rd for the generator cosine, with overlapping
intervals. The change in next-token loss, the outcome of round 7, is a much
noisier measure than the KL: every predictor's within-layer correlation with
it falls to 0.08 to 0.26, in the same order.

### 6.2 The field's statistics, tested against function

Round 11's comparisons were ours. Preregistration 12 tested the ones in
the literature on the same four held-out models. For pairs of heads it
used the Jensen-Shannon divergence between attention rows (Clark et al.
2019), the cosine of the flattened maps, and the correlation of attention
rows in both readings of CHAI's statistic, the last row and every row. For
single heads it used the confidence of Voita et al. (2019), the mean largest
attention weight, which that work found to mark important heads, and row
entropy, against the damage of removing the head. It also retested round
11's failed practical clause where it matters for compression, with many
pairs merged at once. All eight registered clauses pass. The pair-level
merge outcomes are round 11's; the removal and multi-pair outcomes are new.

Head similarity. Within a layer, the field's statistics predict merge
damage at Spearman 0.29 for both readings of the CHAI statistic, 0.35 for
map cosine and 0.43 for Jensen-Shannon divergence. The output perturbation
reaches 0.65, ahead of each by 0.22 to 0.36, every interval above zero (S1).

Head importance. Confidence is a sink detector: within a high-sink layer it
tracks the head's sink mass at Spearman 0.92 (interval 0.86 to 0.97, 73
layers; S2). Against the damage of removing a head it has the wrong sign
(-0.16, interval -0.44 to -0.01), entropy is no better (-0.09), and sink mass
itself is more strongly negative (-0.42): heads that attend to the sink more
matter less. The norm of a head's output predicts the damage at 0.53
(interval 0.35 to 0.73; S4), ahead of confidence by 0.69 and of entropy by
0.62 (S3). The comparison holds within a layer only. Ranked across all
layers of a model, output norm predicts removal damage at -0.08 to 0.24, no
better than chance, because head output norms are not comparable between
layers; confidence and sink mass stay negative across layers.

Merging many pairs. With one pair merged per KV group in every layer but the
first, 54 to 248 pairs per model, the pairs chosen by output perturbation
cost 0.58 to 0.86 of the KL of the pairs chosen by the generator cosine
(geometric mean 0.73; S5) and 0.66 to 0.85 of the pairs chosen by the
CHAI-style last-row correlation (0.75; S6), in every model. Output
perturbation is the best of eight criteria in three models and third in
TinyLlama, where the all-rows correlation is best (S7). At two pairs per
group its advantage over the generator cosine shrinks to 0.99 in Mistral-7B
and 0.93 in TinyLlama and stays at 0.52 and 0.66 in the two Qwen models. The
practical claim that failed for single merges in round 11 holds where
merges accumulate, at the smaller budget.

At four times the context (preregistration 14: 1,024-token windows,
Qwen2.5-1.5B and TinyLlama, same clauses and thresholds), six of the seven
tested clauses hold. Confidence tracks sink mass at 0.91, output norm
predicts removal damage at 0.50 and beats confidence and entropy by 0.66 and
0.58, and output perturbation is the best of eight merge criteria in both
models, at 0.75 and 0.50 of the damage of the CHAI-style choice. Against the
generator cosine its merges cost 0.83 and 0.82, cheaper in both models but
short of the registered geometric mean of 0.8, so that clause fails.

### 6.3 What the sink does

Round 11 found the sink small per position but about half of each head's
output. The literature describes attention to the sink as heads choosing to
do nothing. Preregistration 13 edited the first-token sink directly, after
the softmax, with a custom attention function that reproduces eager
attention exactly when no edit is active (checked on every model). Removing
the sink's contribution from every head in every layer costs 28 to 68 times
as much, in KL, as removing an ordinary token's (K1, all four models). In
every high-sink layer, removing it costs more than removing an ordinary
token (K3; three models, since TinyLlama has no layer whose first-token mass
reaches 0.3). Replacing the sink's weight by its average over positions,
which turns its contribution into a constant vector per head, keeps only
0.13 to 0.43 of the damage in all layers at once (K2), and a median 0.21 in
single high-sink layers (interval 0.18 to 0.64; K4). All four clauses pass.

The sink is not a no-op: its contribution matters, and most of it acts as a
constant bias rather than a signal that varies with position. The damage of
removing it from every layer at once is 2.3 to 3.4 times the sum of
removing it one layer at a time, in the two models where every layer was
tested. Forcing heads to spread the sink's share over other tokens is
costly in some single layers, up to 0.54 in TinyLlama, but applied to every
layer at once it costs less than removing the sink term (0.01 to 0.03); we
report this without an explanation.

## 7. Limitations

Results cover seventeen models up to 8B parameters in ten families, three
position schemes and three attention layouts, English text and code,
windows up to 256 tokens for the statistics and 512 for perplexity; the
functional comparisons of Section 6.2 were repeated at 1,024 tokens on two
models (six of seven clauses hold). Where the sink sits on a mid-sequence, window-varying column
(OLMo-2, late Qwen2.5-3B layers), the sink-operator identification weakens
(cosine 0.61 to 0.82 with one column) and the column set repairs most but
not all of it (four of OLMo-2's 19 high-sink layers keep a sufficiency
residual above 0.4). Restoring the rank-one geometry under a dissociation
control needs a causal width of roughly eight tokens per head. The
deviation directions carry structure the shared-operator description does
not capture (Section 4.6). The commutator pipeline is one instrument among
many the audits address; it is the one whose failure started this work.
The dynamics result rests on one model family. The derivation gate and the practical test both failed as
registered; each is reported in full. Except for Section 6.1, every
statistic is computed on attention maps; that section shows the maps leave
out most of what predicts a head's functional role, and its practical
criterion did not beat the generator cosine at choosing single merges. The audit
criteria are frozen for four targets by public commit and their batteries
run against that freeze (Section 5 reports each as it completes); the
reproductions of Targets 3 and 5 ran at reduced scale
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
| 10 | de18d52, pushed before download | N1 to N5 on Falcon-7B, BLOOM-7b1, OPT-6.7B, Qwen3-8B | N1 to N5 pass, 4 of 4 models |
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
| S1' to S4, -0.70 [-0.78, -0.57 clustered; -0.76, -0.62 per layer], Mistral z 78 to 271 | alignment_study/rerun_results.json, rerun/, clustered_stats.json |
| dev anchor Spearman 0.99 | alignment_study/rerun/Qwen2.5-0.5B.json against tier1_robust.json |
| random-init shared energy 0.88 to 0.96, cosine 0.27; uniform operator cosine 1.00 | alignment_study/dynamics_dev_calibration.json, kmode_dev_calibration.json |
| D0 to D5, 0.935, -0.736, 25 of 26, 14 crossings | alignment_study/dynamics_results.json, posthoc_dynamics.json |
| derived shared energy, G1 to G4 | alignment_study/derivation_results.json, derivation/ |
| deviation rebuilds | alignment_study/toy_structured_dev.json, toy_sparse_dev.json, toy_sinkprofile_dev.json |
| sink-profile round, P1 to P4 | alignment_study/sinkprofile_results.json, sinkprofile/, posthoc_sinkprofile.json |
| new designs, N1 to N5 | alignment_study/rerun_pr10_results.json, rerun_pr10/, sinkprofile_pr10_results.json, sinkprofile_pr10/ |
| merge round, M1 to M3, costs | alignment_study/merge_results.json, merge/ |
| merge calibration | alignment_study/merge_dev_calibration.json |
| Target 1 base result | audits/clark2019/base_result.json |
| Target 3 partial reproduction | audits/chai/Mistral-7B-v0.1_base_result.json |
| Target 4 reproduction | audits/dewage2026/Mistral-7B-v0.1_base_result.json |
| Target 5 reproduction | audits/retrieval_heads/Qwen2.5-7B_base_result.json |
| MPS fidelity | alignment_study/platform_fidelity.json |
