"""Audit your own attention statistic against the null battery in ten lines.

Replace `my_statistic` with any function mapping an (n_heads, T, T) array of
row-stochastic attention maps to a number, and `A` with your maps (for
example from `hubsfree extract`). The report gives, for each null family,
the percentile of your real value and the fraction of the effect that the
null already reproduces.
"""
import numpy as np
import hubsfree

A = hubsfree.random_causal_softmax(np.random.default_rng(0), n=12, T=64, sink_frac=1.0, sink_bias=3.0)  # your maps here


def my_statistic(A):                                   # example: the largest coupling row sum, a 'hub' score
    C = hubsfree.coupling(hubsfree.generators(A))
    return float(C.sum(1).max() / C.sum(1).mean())


res = hubsfree.run_battery(A, stats={"hub_score": my_statistic}, draws=100)
hubsfree.percentile_report(res)
