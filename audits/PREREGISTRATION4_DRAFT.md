# Preregistration 4 (DRAFT, not yet frozen): audits of published claims

Status: draft. Per-target statistics are filled in only after the base
result is reproduced (T2.1) and before any surrogate is run (T2.2). The
file is frozen by commit, pushed, and registered on OSF before the first
battery run; this header changes to FROZEN at that point.

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
- E4: the vertical-pattern prevalence claim is reproduced by
  column-preserving surrogates (labeled matched or shrinks).
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

### Target 2: Kovaleva et al. 2019
Base result R_2: [fraction of heads classified as vertical per layer].
Statistic T_2: [fill]. Nulls: (a), (b), (c).

### Target 3: CHAI (2024)
Base result R_3: [attention-map correlation clustering; fraction of heads
mergeable at the paper's threshold on Llama-7B or the largest open
substitute that fits the available GPU]. Statistic T_3: [fill].

### Target 4: Dewage et al. (2026)
Base result R_4: [MP outlier counts per projection matrix on one of the
paper's models]. Statistic T_4: [fill]. Nulls: (a'), (b'), (c').

### Target 5: Retrieval Heads (2024), positive control
Base result R_5: [retrieval scores per head on a needle task]. Statistic
T_5: [fill]. Nulls: (a), (b), (c).
