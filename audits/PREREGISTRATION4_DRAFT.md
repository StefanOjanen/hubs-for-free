# Preregistration 4 (DRAFT, not yet frozen): audits of published claims

Status: draft, complete for Targets 1, 3, 4 and 5 as of 2026-09-11 (Target
2 logged as not reproducible). Per-target statistics were filled in only
after each base result was reproduced (T2.1) and before any surrogate was
run (T2.2). The file is frozen by commit, pushed, and registered on OSF
before the first battery run; this header changes to FROZEN at that point.

## Common procedure

- Reproduction criterion: the source's base result R_i is reproduced within
  its own stated variability or 10 percent relative error; otherwise the
  target is labeled not reproducible and replaced by the next alternate
  in TARGETS.md, with the attempt reported.
- Nulls, 200 draws each: (a) random causal softmax matched in n and T;
  (b) per-row marginal-matched surrogates; (c) column-preserving
  surrogates keeping the per-window top-k shared columns; (d) untrained
  same-architecture model where available. For weight-level claims:
  (a') Gaussian weights with matched shape and Frobenius norm, (b')
  row-norm-matched random weights, (c') within-matrix permutation.
- Outcome labels: survives (real statistic beyond the 95th percentile of
  every null and effect-size shrinkage under 50 percent); shrinks
  (survives at least one null, shrinkage above 50 percent against
  marginal-matched or column-preserving surrogates); matched (within null
  variability); not reproducible.
- Effect size: the source's own statistic where scalar; otherwise the
  excess of the real statistic over the null median in null standard
  deviations.
- Evaluation code committed before results exist.

## Registered expectations

- E1: at least one audited hub, eigengap, or special-head style claim is
  matched by surrogates.
- E2: at least one distribution-level similarity or redundancy claim
  survives with shrinkage between 20 and 70 percent.
- E3: the positive control (retrieval heads) survives every null with
  shrinkage under 20 percent.
- E4: dropped on 2026-09-11 before freezing. It concerned Target 2
  (Kovaleva et al. 2019), whose pattern proportions are not reproducible
  (classifier and annotations unreleased); no replacement target carries a
  shared-column prevalence statement with a numeric base result. Recorded
  here so the drop is visible in the frozen document.
- E5: MP spectral outliers survive the MP null (by construction) and at
  least half of them are matched by norm-matched random weights.

Falsification: if all audited claims survive with shrinkage under 20
percent, the paper's thesis that reported structure is largely container
geometry is rejected for the audited set and stated as such; E3 failing
means the battery removes real structure and its thresholds are revised
before any other outcome is reported.

## Per-target sections (to be completed after reproduction)

### Target 1: Clark et al. 2019 (criteria frozen 2026-09-06, before any battery run)
Base result R_1, reproduced 2026-09-06 (`audits/clark2019/reproduce.py`,
bert-base-uncased, 38 WikiText-103 validation windows, T = 64, JS
divergence averaged over inputs and query positions for all 144 x 144 head
pairs): mean JS between heads in the same layer 0.253, in different layers
0.380, contrast 0.126; a head's nearest neighbor is in its own layer for
38.9 percent of heads (chance 7.7 percent). The source reports the layer
clustering qualitatively (its Figure 6), so the reproduction criterion is
qualitative agreement: same-layer JS lower than different-layer JS by more
than 0.05 and nearest-neighbor fraction above three times chance. Met.

Statistic T_1: the contrast D = mean JS(different layer) - mean JS(same
layer). Secondary: the nearest-neighbor same-layer fraction.

Nulls: (a) random bidirectional softmax maps, 144 heads with arbitrary
12 x 12 layer labels, T = 64, 100 ensembles (D is near zero by symmetry;
this gives the noise floor of D); (b) 100 per-row marginal-matched
surrogate draws of the real maps, per window; (c) 100 column-preserving
draws keeping the [CLS] column (0) and the [SEP] column (last token), the
separator columns that carry BERT's vertical pattern; (d) untrained
bert-base architecture, five random initializations, same inputs.

Effect size and shrinkage: D_real versus the null median D under (b) and
(c); shrinkage = (median D_null) / D_real, i.e. the share of the layer
clustering reproduced by the null.

Registered expectation (an instance of E2): survives against (a) and (d)
(percentile above 95); against (b) and (c) survives with shrinkage between
20 and 70 percent. Reading: heads in the same layer share more than their
marginals and their separator columns, but a substantial part of the
reported layer clustering is the shared vertical pattern.

### Target 2: Kovaleva et al. 2019 (not reproducible in stated form, 2026-09-11)
The five-class proportions rest on an unreleased classifier and unreleased
annotations of fine-tuned models (TARGETS.md, reproduction log). No
battery is run on this target. E4 is retained only if a replacement target
carries a shared-column prevalence statement with a numeric base result;
otherwise E4 is dropped before freezing and the drop is recorded here.

### Target 3: CHAI 2024 (criteria frozen 2026-09-11, before any battery run)
Base result R_3, attempted 2026-09-11 (`audits/chai/reproduce.py`,
Mistral-7B-v0.1 as the open substitute for LLaMa-7B, 32 C4 validation
documents at T = 1024 against the paper's 1024 samples at 2048; the object
reimplemented from the paper since no code exists: each head's attention
row for the last token, Pearson correlation across heads averaged over
documents, complete-linkage clusters at 0.95 and 0.90). Partially
reproduced. Cross-head correlation is high everywhere (mean off-diagonal
0.55 to 0.98) and the paper's "one or two large clusters with within-cluster
correlation above 0.95" holds in the early layers (layers 1 to 6: mean
correlation 0.93 to 0.98, largest 0.95-cluster 0.22 to 0.91 of heads),
but a majority cluster at 0.95 exists in only 4 of 32 layers (7 at 0.90), and the
paper's "correlation increases in later layers" is not reproduced: mean
correlation peaks at layers 1 to 6 and its Spearman with depth is
-0.38 (first layer 0.55, last quarter 0.76). Across layers the
mean correlation tracks the last row's mass on the sink token (0.71 at
layer 1 with correlation 0.94; 0.25 at layer 13 with 0.71). The
redundancy statement is reproduced; the depth statement is not, on this
model and at this scale. On `facebook/opt-6.7b`, the paper's own OPT family
(same protocol), the split reverses: correlation rises with depth (0.34
at layer 0 to 0.70 over the last quarter, Spearman 0.49) but no layer
carries a majority cluster at 0.95 or 0.90 (largest 0.22 of heads), and the
first-column mass of the last row is near zero until the final layers. The
battery runs on both models; the statistics and nulls below apply to each.

Statistic T_3: per layer, the mean off-diagonal Pearson correlation of the
heads' last-token attention rows and the share of heads in the largest
complete-linkage cluster at 0.95 (secondary: at 0.90).

Nulls, 200 draws each on the same documents: (a) random causal softmax
maps matched in n and T; (b) per-row marginal-matched surrogates of the
full maps (the last row's entries permuted across positions); (c)
column-set-preserving surrogates (the sink set fixed, the rest permuted);
(d) config-initialized Mistral-7B architecture, five seeds.

Effect size and shrinkage: real statistic versus null median; shrinkage =
(null median minus random median) / (real minus random median).

Registered expectation (an instance of E1): the correlation and the
cluster share are matched by the column-set surrogate (c) at 2/3 or more
of layers (the sink entry dominates a Pearson correlation between
attention rows), survive (a) and (d) at the 95th percentile, and shrink by
more than 50 percent against (b). Reading if it holds: the redundancy CHAI
exploits is the shared sink column, which is why clustering by it costs
little accuracy and why it is strongest where the sink is.

### Target 4: Dewage et al. 2026 (criteria frozen 2026-09-11, before any battery run)
Base result R_4, reproduced 2026-09-11 (`audits/dewage2026/reproduce.py`,
Mistral-7B-v0.1, 128 projection matrices, the paper's recipe: gamma =
max/min, sigma^2 = median(s^2) / (1 + gamma), lambda_plus = sigma^2
(1 + sqrt(gamma))^2, outlier s^2 > lambda_plus): mean outliers per matrix
Q 1510.8, K 340.6, V 211.5, O 1449.7 against the paper's 1511, 341,
212 and 1450 (relative error at most 0.3 percent); the share of spectral
energy in the outliers Q 89.3, K 75.7, V 43.0, O 84.4 percent against the
paper's percentages 87.5, 74.7, 43.6 and 84.6 (relative error at most 2.1
percent), which identifies the paper's percentage column as the energy
share rather than the count share (the count share is 21 to 37 percent).
Reproduction criterion (10 percent relative error) met on both.

Statistic T_4: per matrix, the number of MP outliers and their energy
share; per model, the mean over layers for each projection type.

Nulls, 200 draws each where random: (a') Gaussian weights of the same shape
and Frobenius norm (the MP null itself; expected near zero outliers);
(b') row-norm-matched random weights: each row a random Gaussian direction
scaled to the real row's norm; (c') within-matrix permutation of the real
entries (keeps the entry distribution, destroys row and column structure);
(d') the same matrices of an untrained model of the same architecture
(config-initialized weights, five seeds), which fixes the initialization's
own outlier count.

Effect size and shrinkage: the outlier count and energy share of the real
matrix versus the null median; shrinkage = (null median) / real.

Registered expectation (an instance of E5): outliers survive (a') at the
99th percentile in every matrix type; against (c') at least half of the
outlier count is matched (the entry distribution alone produces MP
outliers when it is heavy-tailed); against (b') the energy share shrinks
by less than 50 percent (row norms do not carry the structure); (d') gives
an untrained baseline of outliers that is reported and subtracted in the
discussion, not in the labels.

### Target 5: Retrieval Heads 2024, positive control (criteria frozen 2026-09-11, before any battery run)
Base result R_5, reproduced 2026-09-11 (`audits/retrieval_heads/reproduce.py`,
Qwen2.5-7B as the open model, needle-in-a-haystack at contexts of 1,024 and
2,048 tokens, five depths, two WikiText fillers, 20 instances against the
paper's about 600 at 1K to 50K; a first attempt including 4,096-token
contexts ran at over a minute per instance and was stopped): the needle
was retrieved in 20 of 20 instances; 4.2 percent of the 784 heads
(33 heads) have a retrieval score above 0.1, inside the paper's 3 to 6
percent; 0.6 percent score above 0.5; the strongest heads are L22H3 0.56, L22H4 0.56, L14H0 0.56, L23H11 0.55, L14H6 0.54.
Reproduced.

Statistic T_5: the per-head retrieval score (share of needle tokens a head
copies with its argmax attention at the matching position) and the set of
heads above 0.1.

Nulls: this statistic is defined on the attention rows of generated
tokens, so the surrogates act on those rows: (a) random causal softmax rows
matched in length; (b) per-row marginal-matched permutation of the real
rows (the argmax lands on a random position); (c) column-set-preserving
permutation (the sink set fixed); (d) config-initialized Qwen2.5-7B
architecture on the same instances. 200 draws where random.

Registered expectation E3 (the control the battery must not remove): the
retrieval-head set survives (a), (b), (c) and (d) at the 99th percentile
with the score shrinking by less than 20 percent under every null; if any
null removes it, the battery's thresholds are revised before any other
outcome is reported. A positive control that copies specific context
tokens cannot be produced by a surrogate that permutes where a row looks,
which is the point of including it.
