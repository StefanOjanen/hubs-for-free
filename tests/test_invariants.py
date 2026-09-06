import numpy as np
from hubsfree import (generators, gnorms, random_causal_softmax, surrogate_plain,
                      surrogate_colfix, surrogate_altsink, coupling, rank1_corr)


def _A(seed=0, n=14, T=26):
    return random_causal_softmax(np.random.default_rng(seed), n, T)


def test_surrogates_preserve_row_sums_and_generator_norms():
    A = _A(); rng = np.random.default_rng(1)
    for fam in (surrogate_plain(A, rng, 3), surrogate_colfix(A, rng, (0,), 3), surrogate_altsink(A, rng, 3)):
        for S in fam:
            assert np.abs(A.sum(-1) - S.sum(-1)).max() < 1e-12
            assert np.abs(np.sort(A, -1) - np.sort(S, -1)).max() < 1e-12
            assert np.abs(gnorms(generators(A)) - gnorms(generators(S))).max() < 1e-12


def test_colfix_keeps_column():
    A = _A(); S = surrogate_colfix(A, np.random.default_rng(2), (0,), 1)[0]
    assert np.abs(A[:, 1:, 0] - S[:, 1:, 0]).max() == 0.0


def test_noise_is_rank1():
    A = _A(3)
    G = generators(A)
    assert rank1_corr(coupling(G), gnorms(G)) > 0.85
