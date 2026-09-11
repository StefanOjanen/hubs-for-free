import numpy as np
from hubsfree import (generators, gnorms, random_causal_softmax, surrogate_plain,
                      surrogate_colfix, surrogate_altsink, surrogate_shift, coupling, rank1_corr,
                      sink_column, sink_columns, surrogate_wrapped, run_battery, percentile_report,
                      cos_to_sink, sink_set_generator, sink_generator, shared_mode)


def _A(seed=0, n=14, T=26):
    return random_causal_softmax(np.random.default_rng(seed), n, T)


def test_surrogates_preserve_row_sums_and_generator_norms():
    A = _A(); rng = np.random.default_rng(1)
    for fam in (surrogate_plain(A, rng, 3), surrogate_colfix(A, rng, (0,), 3), surrogate_altsink(A, rng, 3), surrogate_shift(A, rng, 3)):
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


def test_bidirectional_surrogates():
    rng = np.random.default_rng(5)
    lo = rng.normal(size=(4, 10, 10)); ex = np.exp(lo); A = ex / ex.sum(-1, keepdims=True)
    S = surrogate_plain(A, rng, 2)
    assert np.abs(A.sum(-1) - S.sum(-1)).max() < 1e-12
    assert np.abs(np.diagonal(A, axis1=1, axis2=2) - np.diagonal(S[0], axis1=1, axis2=2)).max() == 0.0
    assert np.abs(np.sort(A, -1) - np.sort(S[0], -1)).max() < 1e-12
    C = surrogate_colfix(A, rng, (0, 9), 1)[0]
    assert np.abs(A[:, :, [0, 9]] - C[:, :, [0, 9]]).max() == 0.0


def test_shift_is_a_head_specific_cyclic_shift_with_distinct_offsets():
    n, T = 8, 40
    A = random_causal_softmax(np.random.default_rng(7), n, T, sink_frac=1.0, sink_bias=4.0)
    S = surrogate_shift(A, np.random.default_rng(8), 1)[0]
    offsets = []
    for h in range(n):
        # recover the offset from a row deeper than n, where d % i == d
        row = A[h, T - 1, :T - 1]
        d = next(d for d in range(1, n + 1) if np.allclose(np.roll(row, d), S[h, T - 1, :T - 1]))
        offsets.append(d)
        for i in range(2, T):
            assert np.allclose(np.roll(A[h, i, :i], d % i), S[h, i, :i])
        assert np.allclose(A[h, :2], S[h, :2])
    assert sorted(offsets) == list(range(1, n + 1))


def test_sink_columns_finds_two_shared_columns_and_one_alone():
    rng = np.random.default_rng(11)
    n, T = 8, 64
    lo = rng.normal(size=(n, T, T)); lo[:, :, 0] += 4.0; lo[:, :, 20] += 3.0
    lo = np.where(np.tril(np.ones((T, T), bool)), lo, -1e9)
    A = np.exp(lo - lo.max(-1, keepdims=True)); A /= A.sum(-1, keepdims=True)
    assert sink_column(A) == 0
    assert sink_columns(A, thresh=0.10) == [0, 20]
    B = random_causal_softmax(np.random.default_rng(12), n, T, sink_frac=1.0, sink_bias=4.0)
    assert sink_columns(B, thresh=0.10) == [0]


def test_wrapped_control_keeps_rows_and_moves_every_maximum():
    n, T = 6, 40
    A = random_causal_softmax(np.random.default_rng(21), n, T, sink_frac=1.0, sink_bias=4.0)
    S = surrogate_wrapped(A, np.random.default_rng(22), 1)[0]
    assert np.abs(np.sort(A, -1) - np.sort(S, -1)).max() < 1e-12
    assert np.abs(gnorms(generators(A)) - gnorms(generators(S))).max() < 1e-12
    targets = set()
    for h in range(n):
        cols = {int(S[h, i, :i].argmax()) for i in range(n + 1, T)}   # rows where no wrapping occurs
        assert len(cols) == 1
        targets |= cols
    assert targets == set(range(1, n + 1))


def test_battery_runs_on_causal_and_bidirectional_maps():
    A = random_causal_softmax(np.random.default_rng(31), 8, 32, sink_frac=1.0, sink_bias=3.0)
    res = run_battery(A, draws=6, seed=1)
    assert set(res) >= {"rank1_corr", "shared_energy", "sink_mass", "cos_to_sink"}
    assert set(res["rho"]) >= {"real", "random", "plain", "colfix", "wrapped", "columns"}
    rows = percentile_report(res, quiet=True)
    assert all(0.0 <= r["percentile"] <= 100.0 for r in rows)
    rng = np.random.default_rng(32); lo = rng.normal(size=(4, 12, 12)); B = np.exp(lo); B /= B.sum(-1, keepdims=True)
    resb = run_battery(B, draws=4, seed=2)
    assert "sink_mass" not in resb and "wrapped" not in resb["rho"] and resb["rho"]["columns"] == [0, 11]


def test_cos_to_sink_is_one_for_an_ideal_sink_ensemble_and_low_for_noise():
    T, n = 40, 6
    A = np.zeros((n, T, T))
    for i in range(T):
        if i > 0:
            A[:, i, 0] = 1.0
        else:
            A[:, i, i] = 1.0
    rng = np.random.default_rng(41)
    A = 0.97 * A + 0.03 * random_causal_softmax(rng, n, T)          # near-ideal shared sink, still row-stochastic and causal
    assert cos_to_sink(A, cols=[0]) > 0.99
    B = random_causal_softmax(np.random.default_rng(42), n, T)
    assert cos_to_sink(B, cols=[0]) < 0.6
    S2 = sink_set_generator(T, [0])
    assert np.allclose(S2, sink_generator(T, 0))
    assert abs(np.sqrt((sink_set_generator(T, [0, 7]) ** 2).sum()) - 1.0) < 1e-12
