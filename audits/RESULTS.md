# Audit batteries: results against preregistration 4

Generated 2026-09-16T12:19:33Z by `audits/eval_batteries.py` from the battery result files; regenerate with `python audits/eval_batteries.py`. Criteria: `audits/PREREGISTRATION4.md`, frozen at commit `ee4e267` and not edited since; readings fixed before any result in `audits/PREREGISTRATION4_ADDENDA.md`. Each result file records the registration reference it ran under (`registered`). Shrinkage is the share of the real effect a null reproduces (null median over real).

## Registered expectations

| Expectation | Verdict |
|---|---|
| E1: at least one hub, eigengap or special-head style claim is matched by surrogates (Target 3, column-set clause) | FAILS |
| E2: a distribution-level similarity or redundancy claim survives with shrinkage between 20 and 70 percent (Target 1) | FAILS |
| E3: the positive control survives every null with shrinkage under 20 percent (Target 5) | holds |
| E4 | dropped before the freeze (Target 2 not reproducible) |
| E5: MP outliers survive the MP null and at least half are matched by random weights (Target 4, per-target clauses) | FAILS |

Falsification clause (all audited claims survive with shrinkage under 20 percent, which would reject the container-geometry thesis for the audited set): not triggered

## Target 5, Retrieval Heads (positive control, E3)

`audits/retrieval_heads/battery_result_Mistral-7B-Instruct-v0.2.json`: mistralai/Mistral-7B-Instruct-v0.2, contexts [1024, 2048], 20 instances, 200 draws; registered under https://github.com/StefanOjanen/hubs-for-free/blob/ee4e267/audits/PREREGISTRATION4.md.

Real: 3.4 percent of heads above 0.1, mean top-10 score 0.420, max 0.568.

| Null | heads above 0.1, median | percentile of real | top-10 median | percentile of real | top-10 shrinkage |
|---|---|---|---|---|---|
| a_random | 0.00 percent | 100.0 | 0.0044 | 100.0 | 0.010 |
| b_marginal | 0.00 percent | 100.0 | 0.0044 | 100.0 | 0.010 |
| c_colset | 0.00 percent | 100.0 | 0.0030 | 100.0 | 0.007 |
| d_untrained (n=3) | 0.00 percent | see clause | 0.0000 | see clause | 0.000 |

- survives a_random: both percentiles above 99, top-10 shrinkage under 0.2: holds
- survives b_marginal: both percentiles above 99, top-10 shrinkage under 0.2: holds
- survives c_colset: both percentiles above 99, top-10 shrinkage under 0.2: holds
- survives d_untrained: real above all 3 initializations, top-10 shrinkage under 0.2: holds

E3 on this model: holds.

`audits/retrieval_heads/battery_result_Qwen2.5-7B.json`: Qwen/Qwen2.5-7B, contexts [1024, 2048], 20 instances, 200 draws; registered under https://github.com/StefanOjanen/hubs-for-free/blob/ee4e267/audits/PREREGISTRATION4.md.

Real: 4.2 percent of heads above 0.1, mean top-10 score 0.431, max 0.524.

| Null | heads above 0.1, median | percentile of real | top-10 median | percentile of real | top-10 shrinkage |
|---|---|---|---|---|---|
| a_random | 0.00 percent | 100.0 | 0.0046 | 100.0 | 0.011 |
| b_marginal | 0.00 percent | 100.0 | 0.0046 | 100.0 | 0.011 |
| c_colset | 0.00 percent | 100.0 | 0.0033 | 100.0 | 0.008 |
| d_untrained (n=3) | 0.00 percent | see clause | 0.0000 | see clause | 0.000 |

- survives a_random: both percentiles above 99, top-10 shrinkage under 0.2: holds
- survives b_marginal: both percentiles above 99, top-10 shrinkage under 0.2: holds
- survives c_colset: both percentiles above 99, top-10 shrinkage under 0.2: holds
- survives d_untrained: only medians recorded for the untrained model; compared with the medians, top-10 shrinkage under 0.2: holds

E3 on this model: holds.

## Target 1, Clark et al. 2019 (E2)

`audits/clark2019/battery_result.json`: bert-base-uncased, 38 windows, 100 draws; registered under https://github.com/StefanOjanen/hubs-for-free/blob/ee4e267/audits/PREREGISTRATION4.md.

Real contrast D = 0.1265; nearest neighbor in own layer 39.6 percent.

| Null | D median | percentile of real | shrinkage | nearest-neighbor median |
|---|---|---|---|---|
| a_random | 0.0000 | 100.0 | 0.000 | 7.6 percent |
| b_marginal | 0.0120 | 100.0 | 0.095 | 8.3 percent |
| c_colfix | 0.0762 | 100.0 | 0.602 | 34.7 percent |
| d_untrained | 0.0000 | 100.0 | 0.000 | 5.6 percent |

- survives (a) random maps: percentile above 95: holds
- survives (d) untrained BERT: real above all five initializations: holds
- (b) marginal-matched: percentile above 95 and shrinkage in [0.2, 0.7]: FAILS
- (c) separator columns kept: percentile above 95 and shrinkage in [0.2, 0.7]: holds

Common label: shrinks (battery's own label shrinks). Registered expectation: FAILS.

## Target 3, CHAI 2024 (E1)

`audits/chai/battery_result_Mistral-7B-v0.1.json`: mistralai/Mistral-7B-v0.1, 32 documents at T = 1024, 200 draws; registered under https://github.com/StefanOjanen/hubs-for-free/blob/ee4e267/audits/PREREGISTRATION4.md.

Mean real cross-head correlation over layers 0.779. Layers where: the sink-set surrogate (c) matches correlation and cluster share 0 of 32 (correlation alone 0); correlation survives (a) 32; survives (d) 32; (b) reproduces more than half 0; both surrogates reproduce under 0.2 0. Per-layer labels: {'survives': 0, 'shrinks': 32, 'matched': 0}.

| layer | real corr | real largest 0.95-cluster | a median | b median | c median | d median | b reproduced | c reproduced | c percentile (corr) | label |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.546 | 0.50 | -0.000 | 0.003 | 0.511 | 0.000 | 0.006 | 0.935 | 100.0 | shrinks |
| 1 | 0.942 | 0.83 | 0.000 | 0.001 | 0.934 | 0.000 | 0.001 | 0.991 | 100.0 | shrinks |
| 2 | 0.981 | 0.96 | 0.000 | 0.000 | 0.980 | 0.000 | 0.000 | 0.998 | 100.0 | shrinks |
| 3 | 0.971 | 0.87 | -0.000 | 0.001 | 0.966 | -0.000 | 0.001 | 0.994 | 100.0 | shrinks |
| 4 | 0.962 | 0.63 | -0.000 | 0.000 | 0.957 | 0.000 | 0.000 | 0.995 | 100.0 | shrinks |
| 5 | 0.933 | 0.55 | -0.000 | 0.001 | 0.922 | 0.000 | 0.002 | 0.988 | 100.0 | shrinks |
| 6 | 0.930 | 0.50 | -0.000 | 0.001 | 0.918 | 0.000 | 0.002 | 0.987 | 100.0 | shrinks |
| 7 | 0.858 | 0.38 | -0.000 | 0.004 | 0.834 | 0.000 | 0.004 | 0.972 | 100.0 | shrinks |
| 8 | 0.847 | 0.32 | 0.000 | 0.006 | 0.809 | 0.000 | 0.008 | 0.956 | 100.0 | shrinks |
| 9 | 0.797 | 0.38 | 0.000 | 0.003 | 0.761 | 0.000 | 0.003 | 0.954 | 100.0 | shrinks |
| 10 | 0.769 | 0.22 | 0.000 | 0.005 | 0.698 | 0.000 | 0.007 | 0.907 | 100.0 | shrinks |
| 11 | 0.731 | 0.32 | -0.000 | 0.005 | 0.685 | 0.000 | 0.006 | 0.937 | 100.0 | shrinks |
| 12 | 0.712 | 0.23 | 0.000 | 0.006 | 0.653 | 0.000 | 0.008 | 0.917 | 100.0 | shrinks |
| 13 | 0.712 | 0.19 | -0.000 | 0.016 | 0.548 | 0.000 | 0.023 | 0.769 | 100.0 | shrinks |
| 14 | 0.649 | 0.23 | 0.000 | 0.017 | 0.566 | 0.000 | 0.026 | 0.872 | 100.0 | shrinks |
| 15 | 0.661 | 0.27 | 0.000 | 0.011 | 0.584 | 0.001 | 0.017 | 0.885 | 100.0 | shrinks |
| 16 | 0.678 | 0.30 | 0.000 | 0.014 | 0.600 | 0.000 | 0.021 | 0.886 | 100.0 | shrinks |
| 17 | 0.689 | 0.25 | 0.000 | 0.020 | 0.591 | 0.000 | 0.028 | 0.858 | 100.0 | shrinks |
| 18 | 0.639 | 0.29 | 0.000 | 0.019 | 0.596 | 0.001 | 0.029 | 0.932 | 100.0 | shrinks |
| 19 | 0.701 | 0.31 | -0.000 | 0.008 | 0.661 | 0.000 | 0.012 | 0.943 | 100.0 | shrinks |
| 20 | 0.761 | 0.40 | -0.000 | 0.008 | 0.733 | 0.000 | 0.010 | 0.964 | 100.0 | shrinks |
| 21 | 0.778 | 0.37 | -0.000 | 0.006 | 0.756 | 0.000 | 0.008 | 0.971 | 100.0 | shrinks |
| 22 | 0.823 | 0.45 | -0.000 | 0.002 | 0.814 | 0.000 | 0.002 | 0.989 | 100.0 | shrinks |
| 23 | 0.802 | 0.37 | 0.000 | 0.008 | 0.782 | 0.000 | 0.010 | 0.976 | 100.0 | shrinks |
| 24 | 0.829 | 0.42 | 0.000 | 0.002 | 0.818 | 0.000 | 0.002 | 0.987 | 100.0 | shrinks |
| 25 | 0.852 | 0.46 | -0.000 | 0.002 | 0.846 | 0.000 | 0.003 | 0.993 | 100.0 | shrinks |
| 26 | 0.831 | 0.48 | -0.000 | 0.004 | 0.822 | 0.000 | 0.005 | 0.990 | 100.0 | shrinks |
| 27 | 0.821 | 0.38 | 0.000 | 0.003 | 0.810 | 0.000 | 0.003 | 0.987 | 100.0 | shrinks |
| 28 | 0.727 | 0.38 | 0.000 | 0.012 | 0.710 | 0.000 | 0.016 | 0.977 | 100.0 | shrinks |
| 29 | 0.720 | 0.37 | 0.000 | 0.010 | 0.694 | 0.000 | 0.013 | 0.964 | 100.0 | shrinks |
| 30 | 0.693 | 0.39 | 0.000 | 0.018 | 0.667 | 0.000 | 0.026 | 0.963 | 100.0 | shrinks |
| 31 | 0.600 | 0.33 | -0.000 | 0.039 | 0.539 | 0.000 | 0.065 | 0.899 | 100.0 | shrinks |

- (c) sink set kept matches correlation and cluster share in at least 22 of 32 layers: FAILS
- correlation survives (a) random rows (percentile above 95) in at least 22 of 32 layers: holds
- correlation survives (d) untrained model (above all five initializations) in at least 22 of 32 layers: holds
- (b) marginal-matched reproduces more than half of the correlation excess in at least 22 of 32 layers: FAILS

Majority label: shrinks. Registered expectation on this model: FAILS.

`audits/chai/battery_result_opt-6.7b.json`: facebook/opt-6.7b, 32 documents at T = 1024, 200 draws; registered under https://github.com/StefanOjanen/hubs-for-free/blob/ee4e267/audits/PREREGISTRATION4.md.

Mean real cross-head correlation over layers 0.598. Layers where: the sink-set surrogate (c) matches correlation and cluster share 0 of 32 (correlation alone 0); correlation survives (a) 32; survives (d) 32; (b) reproduces more than half 0; both surrogates reproduce under 0.2 4. Per-layer labels: {'survives': 9, 'shrinks': 23, 'matched': 0}.

| layer | real corr | real largest 0.95-cluster | a median | b median | c median | d median | b reproduced | c reproduced | c percentile (corr) | label |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.337 | 0.05 | -0.000 | 0.006 | 0.009 | 0.000 | 0.018 | 0.028 | 100.0 | survives |
| 1 | 0.548 | 0.13 | 0.000 | 0.002 | 0.095 | 0.001 | 0.004 | 0.173 | 100.0 | survives |
| 2 | 0.598 | 0.08 | 0.000 | 0.002 | 0.049 | 0.001 | 0.004 | 0.081 | 100.0 | survives |
| 3 | 0.570 | 0.08 | 0.000 | 0.003 | 0.067 | 0.001 | 0.006 | 0.117 | 100.0 | survives |
| 4 | 0.578 | 0.09 | 0.000 | 0.003 | 0.125 | 0.001 | 0.006 | 0.216 | 100.0 | survives |
| 5 | 0.532 | 0.13 | 0.000 | 0.008 | 0.181 | 0.003 | 0.015 | 0.340 | 100.0 | survives |
| 6 | 0.516 | 0.17 | 0.000 | 0.009 | 0.221 | 0.002 | 0.017 | 0.428 | 100.0 | survives |
| 7 | 0.539 | 0.13 | 0.000 | 0.013 | 0.212 | 0.001 | 0.023 | 0.394 | 100.0 | survives |
| 8 | 0.576 | 0.13 | 0.000 | 0.017 | 0.282 | 0.002 | 0.029 | 0.489 | 100.0 | survives |
| 9 | 0.614 | 0.13 | 0.000 | 0.014 | 0.339 | 0.003 | 0.023 | 0.552 | 100.0 | shrinks |
| 10 | 0.624 | 0.17 | -0.000 | 0.012 | 0.372 | 0.003 | 0.019 | 0.596 | 100.0 | shrinks |
| 11 | 0.614 | 0.17 | -0.000 | 0.008 | 0.392 | 0.001 | 0.013 | 0.638 | 100.0 | shrinks |
| 12 | 0.600 | 0.13 | 0.000 | 0.009 | 0.395 | 0.000 | 0.015 | 0.659 | 100.0 | shrinks |
| 13 | 0.648 | 0.17 | 0.000 | 0.009 | 0.438 | 0.002 | 0.013 | 0.676 | 100.0 | shrinks |
| 14 | 0.582 | 0.19 | -0.000 | 0.011 | 0.395 | 0.003 | 0.018 | 0.679 | 100.0 | shrinks |
| 15 | 0.587 | 0.19 | 0.000 | 0.009 | 0.409 | 0.005 | 0.015 | 0.697 | 100.0 | shrinks |
| 16 | 0.640 | 0.21 | 0.000 | 0.005 | 0.459 | 0.003 | 0.008 | 0.717 | 100.0 | shrinks |
| 17 | 0.619 | 0.18 | -0.000 | 0.008 | 0.454 | 0.002 | 0.013 | 0.733 | 100.0 | shrinks |
| 18 | 0.546 | 0.19 | -0.000 | 0.006 | 0.384 | 0.003 | 0.011 | 0.703 | 100.0 | shrinks |
| 19 | 0.543 | 0.18 | 0.000 | 0.008 | 0.367 | 0.001 | 0.015 | 0.677 | 100.0 | shrinks |
| 20 | 0.508 | 0.16 | 0.000 | 0.008 | 0.335 | 0.002 | 0.015 | 0.659 | 100.0 | shrinks |
| 21 | 0.595 | 0.21 | -0.000 | 0.005 | 0.423 | 0.003 | 0.008 | 0.710 | 100.0 | shrinks |
| 22 | 0.524 | 0.19 | 0.000 | 0.004 | 0.375 | 0.001 | 0.008 | 0.714 | 100.0 | shrinks |
| 23 | 0.527 | 0.19 | 0.000 | 0.006 | 0.376 | 0.001 | 0.012 | 0.713 | 100.0 | shrinks |
| 24 | 0.533 | 0.21 | 0.000 | 0.003 | 0.411 | 0.007 | 0.006 | 0.771 | 100.0 | shrinks |
| 25 | 0.640 | 0.28 | 0.000 | 0.000 | 0.554 | 0.003 | 0.001 | 0.865 | 100.0 | shrinks |
| 26 | 0.624 | 0.31 | 0.000 | 0.001 | 0.551 | 0.001 | 0.001 | 0.883 | 100.0 | shrinks |
| 27 | 0.663 | 0.34 | 0.000 | 0.001 | 0.584 | 0.002 | 0.001 | 0.881 | 100.0 | shrinks |
| 28 | 0.733 | 0.40 | 0.000 | 0.001 | 0.660 | 0.004 | 0.001 | 0.900 | 100.0 | shrinks |
| 29 | 0.766 | 0.48 | -0.000 | 0.001 | 0.700 | 0.002 | 0.001 | 0.913 | 100.0 | shrinks |
| 30 | 0.831 | 0.49 | 0.000 | 0.000 | 0.764 | 0.003 | 0.000 | 0.919 | 100.0 | shrinks |
| 31 | 0.784 | 0.54 | 0.000 | 0.000 | 0.723 | 0.001 | 0.000 | 0.921 | 100.0 | shrinks |

- (c) sink set kept matches correlation and cluster share in at least 22 of 32 layers: FAILS
- correlation survives (a) random rows (percentile above 95) in at least 22 of 32 layers: holds
- correlation survives (d) untrained model (above all five initializations) in at least 22 of 32 layers: holds
- (b) marginal-matched reproduces more than half of the correlation excess in at least 22 of 32 layers: FAILS

Majority label: shrinks. Registered expectation on this model: FAILS.

## Target 4, Dewage et al. 2026 (E5)

`audits/dewage2026/battery_result.json`: mistralai/Mistral-7B-v0.1, 32 layers, 20 draws per random family, initializer std 0.02; registered under https://github.com/StefanOjanen/hubs-for-free/blob/ee4e267/audits/PREREGISTRATION4.md.

| type | real outliers | real energy share | a' median | a' pct | b' median | b' count shrinkage | b' energy shrinkage | c' median | c' count shrinkage | d' baseline | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q_proj | 1510.8 | 0.893 | 1287.7 | 100 | 1379.2 | 0.913 | 0.878 | 1288.0 | 0.853 | 1287.8 | shrinks |
| k_proj | 340.6 | 0.757 | 137.7 | 100 | 263.3 | 0.773 | 0.710 | 138.2 | 0.406 | 137.5 | shrinks |
| v_proj | 211.5 | 0.430 | 137.7 | 100 | 179.4 | 0.848 | 0.797 | 137.8 | 0.651 | 137.7 | shrinks |
| o_proj | 1449.7 | 0.844 | 1287.6 | 100 | 1294.2 | 0.893 | 0.857 | 1288.0 | 0.888 | 1287.7 | shrinks |

- q_proj: survives (a') Gaussian norm-matched: percentile above 99 (above all 20 draws): holds
- q_proj: (c') permuted entries match at least half of the outlier count: holds
- q_proj: (b') row-norm-matched reproduces less than half of the energy share: FAILS
- k_proj: survives (a') Gaussian norm-matched: percentile above 99 (above all 20 draws): holds
- k_proj: (c') permuted entries match at least half of the outlier count: FAILS
- k_proj: (b') row-norm-matched reproduces less than half of the energy share: FAILS
- v_proj: survives (a') Gaussian norm-matched: percentile above 99 (above all 20 draws): holds
- v_proj: (c') permuted entries match at least half of the outlier count: holds
- v_proj: (b') row-norm-matched reproduces less than half of the energy share: FAILS
- o_proj: survives (a') Gaussian norm-matched: percentile above 99 (above all 20 draws): holds
- o_proj: (c') permuted entries match at least half of the outlier count: holds
- o_proj: (b') row-norm-matched reproduces less than half of the energy share: FAILS

Types passing all three clauses: 0 of 4. Registered expectation (all types): FAILS. The expectation list's E5 names norm-matched random weights for the half-matched clause; the per-target section names the permuted null (c'); the per-target clause is evaluated and the (b') count shrinkage is in the table.

## Target 2, Kovaleva et al. 2019

Not reproducible in its stated form (classifier and annotations unreleased); no battery, E4 dropped before the freeze.


## Post hoc, not registered: where the Gaussian null's outliers come from

`audits/dewage2026/posthoc_edge.py`, written after the Target 4 battery and changing none of its labels; layers 0, 8, 16, 24 of mistralai/Mistral-7B-v0.1. The registered Gaussian null (a') has no learned structure, yet the source recipe counts a median 1288 of its 4096 squared singular values as Marchenko-Pastur outliers in the square projections and 138 of 1024 in K and V. The reason is the recipe's noise scale. It sets sigma^2 = median(s^2) / (1 + gamma) and lambda_+ = sigma^2 (1 + sqrt(gamma))^2, but for an m x n matrix with i.i.d. entries of variance v (m <= n) the eigenvalues of W W^T are v n times a Marchenko-Pastur law of ratio c = m / n, whose edge is v n (1 + sqrt(c))^2. The recipe's formula reproduces that edge only if median(s^2) = v (m + n), whereas the actual median is v n times the median of the law. The edge is therefore placed inside the bulk and part of the bulk is counted.

Three edges applied to the same spectra (median-fit rescales the bulk by matching the MP median, robust to real outliers; mean-fit matches the MP mean, which real outliers inflate, so it undercounts; neither needs to know v; oracle uses the known entry variance and exists only for the synthetic matrices):

| type | singular values | real, recipe | real, median-fit | real, mean-fit | Gaussian, recipe | Gaussian, median-fit | Gaussian, oracle |
|---|---|---|---|---|---|---|---|
| q_proj | 4096 | 1583 | 828 | 182 | 1289 | 0 | 0 |
| k_proj | 1024 | 354 | 276 | 104 | 137 | 0 | 0 |
| v_proj | 1024 | 240 | 121 | 67 | 138 | 0 | 0 |
| o_proj | 4096 | 1462 | 513 | 202 | 1290 | 0 | 0 |

Both calibrated edges put a pure Gaussian matrix at exactly zero outliers, which is what the Marchenko-Pastur law requires and what the recipe fails to deliver. On the real weights the count depends on how the scale is estimated, by a factor of 2 to 5 between the two calibrated estimators and up to 9 against the recipe, so the reported counts (Q 1511, K 341, V 212, O 1450, reproduced here to 0.3 percent) are a property of one estimator rather than of the weights. The outliers' share of spectral energy is more stable but also falls: 0.91 to 0.71 for Q and 0.86 to 0.53 for O under the median-fit edge. What survives is that trained weights do carry spectral structure a Gaussian does not have: under either calibrated edge the real matrices have hundreds of outliers and the Gaussian has none.

