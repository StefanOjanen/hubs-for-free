# Audit batteries: results against preregistration 4

Generated 2026-09-15T17:01:24Z by `audits/eval_batteries.py` from the battery result files; regenerate with `python audits/eval_batteries.py`. Criteria: `audits/PREREGISTRATION4.md`, frozen at commit `ee4e267` and not edited since; readings fixed before any result in `audits/PREREGISTRATION4_ADDENDA.md`. Each result file records the registration reference it ran under (`registered`). Shrinkage is the share of the real effect a null reproduces (null median over real).

## Registered expectations

| Expectation | Verdict |
|---|---|
| E1: at least one hub, eigengap or special-head style claim is matched by surrogates (Target 3, column-set clause) | FAILS |
| E2: a distribution-level similarity or redundancy claim survives with shrinkage between 20 and 70 percent (Target 1) | pending |
| E3: the positive control survives every null with shrinkage under 20 percent (Target 5) | holds |
| E4 | dropped before the freeze (Target 2 not reproducible) |
| E5: MP outliers survive the MP null and at least half are matched by random weights (Target 4, per-target clauses) | pending |

Falsification clause (all audited claims survive with shrinkage under 20 percent, which would reject the container-geometry thesis for the audited set): pending

## Target 5, Retrieval Heads (positive control, E3)

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

pending

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

## Target 4, Dewage et al. 2026 (E5)

pending

## Target 2, Kovaleva et al. 2019

Not reproducible in its stated form (classifier and annotations unreleased); no battery, E4 dropped before the freeze.

