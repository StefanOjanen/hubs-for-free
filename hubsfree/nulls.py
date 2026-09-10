"""Null families: random ensembles and constrained surrogates.

All surrogates permute entries within causal rows, so every row's sum,
inverse participation ratio (sharpness), and diagonal entry are preserved
exactly, hence every generator norm (Proposition 2 of the paper)."""
import numpy as np


def random_causal_softmax(rng, n, T, temp_sigma=0.5, sink_frac=0.0, sink_bias=3.0, causal=True):
    """Gaussian-logit softmax attention with lognormal per-head temperatures
    and an optional first-column bias on a fraction of heads. causal=False
    gives bidirectional maps (for encoder models such as BERT)."""
    tau = np.exp(rng.normal(0, temp_sigma, n))
    lo = rng.normal(size=(n, T, T)) / tau[:, None, None]
    if sink_frac > 0:
        k = max(1, int(round(sink_frac * n)))
        lo[rng.choice(n, k, replace=False), :, 0] += sink_bias
    if causal:
        mask = np.tril(np.ones((T, T), bool))
        lo = np.where(mask, lo, -1e9)
    ex = np.exp(lo - lo.max(-1, keepdims=True))
    return ex / ex.sum(-1, keepdims=True)


def is_causal(A, tol=1e-9):
    """True when every head is lower-triangular (causal attention)."""
    T = A.shape[-1]
    return bool(np.abs(A[:, np.triu_indices(T, 1)[0], np.triu_indices(T, 1)[1]]).max() < tol)


def _free_cols(i, T, causal, keep=()):
    cols = range(i) if causal else range(T)
    return np.array([c for c in cols if c != i and c not in keep])


def surrogate_plain(A, rng, draws=1, causal=None):
    """Permute each row's off-diagonal entries independently (within the
    causal support for causal maps, across the full row for bidirectional
    maps). Row sums, row IPR, and the diagonal are preserved exactly."""
    n, T, _ = A.shape
    causal = is_causal(A) if causal is None else causal
    S = np.repeat(A[None], draws, 0)
    for i in range(T):
        free = _free_cols(i, T, causal)
        if len(free) > 1:
            block = S[:, :, i, free].reshape(draws * n, len(free))
            S[:, :, i, free] = rng.permuted(block, axis=1).reshape(draws, n, len(free))
    return S


def surrogate_colfix(A, rng, cols=(0,), draws=1, causal=None):
    """Permute off-diagonal entries while keeping the given columns (for
    example the shared sink or separator columns) and the diagonal fixed."""
    n, T, _ = A.shape
    causal = is_causal(A) if causal is None else causal
    S = np.repeat(A[None], draws, 0)
    keep = set(cols)
    for i in range(T):
        free = _free_cols(i, T, causal, keep)
        if len(free) > 1:
            block = S[:, :, i, free].reshape(draws * n, len(free))
            S[:, :, i, free] = rng.permuted(block, axis=1).reshape(draws, n, len(free))
    return S


def surrogate_altsink(A, rng, draws=1):
    """Distinct target column per head (t_h = h + 1): each row's maximum is
    swapped into t_h, remaining off-diagonal entries permuted. Every head
    stays individually concentrated but heads no longer share a column.
    Known limitation: loses power for large head counts (see NOTE.md)."""
    n, T, _ = A.shape
    S = np.repeat(A[None], draws, 0)
    t = np.arange(n) + 1
    rows = np.arange(draws)
    for i in range(1, T):
        for h in range(n):
            block = S[:, h, i, :i]
            if t[h] < i:
                jmax = block.argmax(1)
                vmax = block[rows, jmax].copy()
                block[rows, jmax] = block[rows, t[h]]
                block[rows, t[h]] = vmax
                rest = np.array([c for c in range(i) if c != t[h]])
                block[:, rest] = rng.permuted(block[:, rest], axis=1)
            else:
                S[:, h, i, :i] = rng.permuted(block, axis=1)
    return S


def surrogate_shift(A, rng, draws=1):
    """Matched-geometry dissociation control (causal maps). Each head's causal
    row is cyclically shifted by a head-specific offset d_h, drawn per draw as
    a permutation of 1..n, so the head's dominant column moves to d_h mod i
    while the row's internal pattern is kept: every head stays exactly as
    concentrated, with exactly the same row geometry, but heads no longer
    share a column. Rows 0 and 1 (nothing to shift) are unchanged. Row sums,
    row IPR, the diagonal and every generator norm are preserved exactly.
    Toy validation: control_redesign_toy*.py in alignment_study/."""
    if not is_causal(A):
        raise NotImplementedError("surrogate_shift is defined for causal attention maps")
    n, T, _ = A.shape
    S = np.repeat(A[None], draws, 0)
    for d in range(draws):
        offs = rng.permutation(n) + 1
        for h in range(n):
            for i in range(2, T):
                S[d, h, i, :i] = np.roll(A[h, i, :i], int(offs[h]) % i)
    return S
