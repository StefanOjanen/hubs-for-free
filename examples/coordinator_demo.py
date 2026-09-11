"""The coordinator that random matrices produce, step by step.

Fourteen causal softmax heads with Gaussian logits and lognormal per-head
temperatures. No training. The standard head-interaction pipeline still
finds a hub, a spectral gap and a special head. Seed 147 is the first seed
at which the archived analysis's exact cluster pattern appears (paper.md,
Section 4.2); pass any other seed to see how often the signatures recur.
"""
import sys
import numpy as np
import hubsfree

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 147
A = hubsfree.random_causal_softmax(np.random.default_rng(seed), n=14, T=26)
G = hubsfree.generators(A)
gn = hubsfree.gnorms(G)
C = hubsfree.coupling(G)
ev = np.sort(np.linalg.eigvalsh(C))[::-1]
print(f"seed {seed}: 14 random heads, T = 26")
print(f"  hub (largest coupling row sum): head {int(C.sum(1).argmax())}")
print(f"  eigengap lambda_1 / |lambda_2|: {ev[0] / abs(ev[1]):.1f}")
print(f"  2-sigma 'special' heads by generator norm: {sorted(hubsfree.two_sigma_flags(gn))}")
print(f"  Ward clusters (4 groups): {hubsfree.ward_sizes(C)}")
print(f"  rank-1 correlation of C with the norm products: {hubsfree.rank1_corr(C, gn):.3f}  (near 1: the geometry is the norms)")
