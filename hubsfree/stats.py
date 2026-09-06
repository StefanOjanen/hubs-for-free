"""Statistics on a stack of attention maps A with shape (n, T, T)."""
import numpy as np


def generators(A):
    """Skew-symmetric parts G_h = (A_h - A_h^T)/2."""
    return (A - A.transpose(0, 2, 1)) / 2.0


def gnorms(G):
    return np.sqrt((G ** 2).sum((1, 2)))


def coupling(G):
    """C_ij = ||[G_i, G_j]||_F for all pairs, vectorized."""
    P = np.einsum("aij,bjk->abik", G, G)
    K = P - P.transpose(1, 0, 2, 3)
    return np.sqrt((K ** 2).sum((2, 3)))


def rank1_corr(C, gn):
    """Pearson correlation of off-diagonal C with the norm products.
    Returns nan when either side is (near) constant; see cv(gn) diagnostics."""
    n = C.shape[0]
    iu = np.triu_indices(n, 1)
    x, y = C[iu], np.outer(gn, gn)[iu]
    if x.std() < 1e-15 or y.std() < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def eigengap(C):
    ev = np.linalg.eigvalsh(C)
    return float(ev[-1] / max(abs(ev[-2]), 1e-300))


def rho(G, gn=None):
    """Top-eigenvalue share of the cosine Gram of the generators (shared-mode
    order parameter, invariant to head relabeling and sign)."""
    n = G.shape[0]
    gn = gnorms(G) if gn is None else gn
    V = G.reshape(n, -1)
    M = (V @ V.T) / np.outer(gn, gn)
    return float(np.linalg.eigvalsh(M)[-1] / n)


def shared_mode(G):
    """Top principal operator S of the generator set and the decomposition
    G_h = a_h S + E_h. Returns (S, a, e, shared_energy_fraction)."""
    n, T, _ = G.shape
    V = G.reshape(n, -1)
    _, _, Vh = np.linalg.svd(V, full_matrices=False)
    S = Vh[0]
    a = V @ S
    E = V - np.outer(a, S)
    e = np.sqrt((E ** 2).sum(1))
    gn = np.sqrt((V ** 2).sum(1))
    return S.reshape(T, T), a, e, float((a ** 2 / gn ** 2).mean())


def sink_column(A, min_rows=16):
    """Empirical sink column: modal argmax over heads of column mass, over
    columns with at least min_rows contributing rows."""
    n, T, _ = A.shape
    cmax = max(T - 1 - min_rows, 1)
    best = []
    for h in range(n):
        cm = np.array([A[h, max(1, c + 1):, c].mean() for c in range(cmax)])
        best.append(int(cm.argmax()))
    vals, counts = np.unique(best, return_counts=True)
    return int(vals[counts.argmax()])


def sink_generator(T, c):
    """Ideal causal sink generator at column c, unit Frobenius norm."""
    A = np.zeros((T, T))
    for i in range(T):
        if i > c:
            A[i, c] = 1.0
        else:
            A[i, i] = 1.0
    G = (A - A.T) / 2.0
    return G / np.sqrt((G ** 2).sum())


def two_sigma_flags(x):
    """Indices deviating from the mean by more than two sample standard
    deviations (the 'special head' rule whose base rate the battery audits)."""
    m, s = x.mean(), x.std(ddof=1)
    return set(np.where(np.abs(x - m) > 2 * s)[0].tolist())


def ward_sizes(C, k=4):
    """Cluster sizes from Ward linkage on 1 - C/max(C), cut at k clusters."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    D = 1.0 - C / C.max()
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="ward")
    lab = fcluster(Z, k, criterion="maxclust")
    return sorted(np.bincount(lab)[1:].tolist())
