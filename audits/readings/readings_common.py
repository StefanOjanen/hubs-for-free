# Instruments for preregistration 17 (the audits under the sources' own
# readings), on the GPU. Each function reproduces a statistic or null of the
# registered batteries (audits/clark2019/battery.py, audits/chai/battery.py)
# and is checked against it before use (check_instruments.py):
#   js_matrix      mean over query rows of the pairwise Jensen-Shannon
#                  divergence between heads, entries clipped at 1e-12 as in
#                  the registered numpy version
#   permute_free   per (head, row) random permutation of the entries marked
#                  free, every other entry left in place: the marginal null
#                  (free = off-diagonal support) and the column-preserving null
#                  (free = support minus the kept columns and the diagonal)
#   random_maps    Gaussian-logit softmax rows with lognormal per-head
#                  temperature (sigma 0.5) over each row's support
#   row_pearson    per query row, the Pearson correlation across heads of the
#                  heads' rows over the row's support; mean over rows
import numpy as np
import torch


def js_matrix(A, chunk=16):
    """A: (n, Tq, Tk) tensor of attention rows. Returns (n, n) float64 numpy:
    the mean over query rows of the pairwise Jensen-Shannon divergence,
    computed as H(m) - (H(p) + H(q)) / 2 over head pairs i < j."""
    P = A.float().clamp(1e-12, 1.0)
    n, Tq, _ = P.shape
    iu = torch.triu_indices(n, n, 1, device=A.device)
    hp = torch.special.xlogy(P, P).sum(-1)                       # (n, Tq): sum p log p
    acc = torch.zeros(iu.shape[1], dtype=torch.float32, device=A.device)
    for s in range(0, Tq, chunk):
        p = P[:, s:s + chunk, :]                                 # (n, c, Tk)
        m = (p[iu[0]] + p[iu[1]]) / 2                            # (pairs, c, Tk)
        hm = torch.special.xlogy(m, m).sum(-1)                   # (pairs, c)
        h = hp[:, s:s + chunk]
        acc += (0.5 * (h[iu[0]] + h[iu[1]]) - hm).sum(1)
    J = torch.zeros(n, n, dtype=torch.float32, device=A.device)
    J[iu[0], iu[1]] = acc / Tq
    return (J + J.T).cpu().numpy().astype(np.float64)


def free_order(free, n):
    """Destination order for permute_free: the free columns of each row in
    column order, then the fixed ones; depends only on the mask."""
    R, T = free.shape[-2:]
    col = torch.arange(T, device=free.device, dtype=torch.float32) / T
    return torch.where(free.expand(n, R, T), col, 2.0 + col).argsort(-1)


def permute_free(A, free, gen, dst=None):
    """A: (n, R, T); free: (R, T) or (n, R, T) bool. Entries marked free are
    permuted at random within each (head, row); the rest stay in place."""
    n, R, T = A.shape
    free = free.expand(n, R, T)
    col = torch.arange(T, device=A.device, dtype=torch.float32) / T
    keys = torch.where(free, torch.rand(n, R, T, generator=gen, device=A.device), 2.0 + col)
    src = keys.argsort(-1)
    dst = free_order(free, n) if dst is None else dst
    out = torch.empty_like(A)
    out.scatter_(-1, dst, A.gather(-1, src))
    return out


def random_maps(n, R, T, support, gen, device, sigma=0.5):
    """(n, R, T) softmax rows over support (R, T) bool; zero elsewhere."""
    tau = torch.exp(torch.randn(n, generator=gen, device=device) * sigma)
    lo = torch.randn(n, R, T, generator=gen, device=device) / tau[:, None, None]
    lo = lo.masked_fill(~support[None], float("-inf"))
    return torch.softmax(lo, -1)


def row_pearson(X, support):
    """X: (n, R, T) rows; support: (R, T) bool. Per row, the Pearson
    correlation matrix across heads over the row's support (NaN set to 0, as
    np.nan_to_num(np.corrcoef) does), averaged over rows; and the same for
    cosine similarity. Returns two (n, n) float64 numpy arrays."""
    M = support[None].float()
    cnt = M.sum(-1, keepdim=True)
    mu = (X * M).sum(-1, keepdim=True) / cnt
    Xc = (X - mu) * M
    cov = torch.einsum("hrt,grt->rhg", Xc, Xc)
    sd = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2))
    corr = torch.nan_to_num(cov / (sd[:, :, None] * sd[:, None, :]), nan=0.0, posinf=0.0, neginf=0.0)
    idx = torch.arange(X.shape[0], device=X.device)
    corr[:, idx, idx] = 1.0
    Xm = X * M
    g = torch.einsum("hrt,grt->rhg", Xm, Xm); nr = torch.sqrt(torch.diagonal(g, dim1=1, dim2=2))
    cos = g / (nr[:, :, None] * nr[:, None, :]).clamp_min(1e-30)
    return corr.mean(0).cpu().numpy().astype(np.float64), cos.mean(0).cpu().numpy().astype(np.float64)
