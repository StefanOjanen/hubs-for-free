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

### Target 1: Clark et al. 2019
Base result R_1: [fill after reproduction: JS-distance matrix between all
144 BERT-base heads on N inputs; layer-clustering statistic as in the
paper's Section 6]. Statistic T_1: [fill]. Nulls: (a), (b), (c), (d).

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
