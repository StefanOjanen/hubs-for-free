# Closed-form fusion of the key and value heads of a group, for converting multi-head attention to
# grouped-query attention without training (PREREGISTRATION26). Instead of rotating the heads of a group into
# one basis and averaging their key and value projections (Jin et al. 2025; round 24), the fusion works on the
# function each head computes, its query-key bilinear form and its value-output map, so the result does not
# depend on any head's internal basis. Every head keeps its own query and output projection, which absorb
# what the shared key and value head cannot represent; the converted model computes exactly what a
# grouped-query model with the shared heads computes.
#
# x is the attention input, augmented with a constant 1 when the projections have biases; S = E[x x^T] on
# calibration tokens and L = S^(1/2).
#   values  head h adds O_h C_h x to the residual stream (C_h: value projection, O_h: its output slice). A
#           shared value projection C_G = P^T L^-1 and new output slices O'_h = O_h C_h L P minimize
#           sum_h E||(O_h C_h - O'_h C_G) x||^2 when P holds the top hd eigenvectors of
#           sum_h L C_h^T W_h C_h L, with W_h = O_h^T O_h ("output-weighted") or the identity ("unweighted").
#   keys, dimensions without rotary embedding: the logit of query token t on key token s is
#           x_t^T A_h^T B_h x_s (A_h, B_h: query and key projections). A shared key projection B_G = R^T Lk^-1
#           and new queries A'_h = R^T Lk B_h^T A_h minimize sum_h ||L (A_h^T B_h - A'_h^T B_G) Lk||_F^2 when R
#           holds the top eigenvectors of sum_h Lk B_h^T Q_h B_h Lk, with Q_h = A_h S A_h^T, the second moment
#           of head h's queries ("query-weighted"), or the identity. Lk comes from the centered input: a term
#           that is constant over keys does not change the softmax (and a key bias only adds such a term).
#   keys, rotary planes (i, i + r/2): queries and keys of plane i are complex numbers a_h x and b_h x; their
#           product, which the rotary embedding multiplies by a phase, must be kept for every pair of tokens.
#           The shared key plane is L^-1 u, u the top eigenvector of sum_h w_h bhat_h bhat_h^H (bhat_h = L b_h,
#           w_h = E|q_h|^2 on the plane, or 1), and each head's query plane is multiplied by a complex number,
#           which commutes with the rotary embedding. Key biases are kept here, since rotation makes them
#           position dependent.
# Groups of one head reproduce the model exactly (checked by the round's scripts).
import torch

torch.set_grad_enabled(False)


def attn_input_module(arch, l):
    b = arch.blocks[l]
    if arch.kind == "gpt2":
        return b.attn.c_attn
    if arch.kind == "neox":
        return b.attention.query_key_value
    return b.self_attn.q_proj


def has_bias(arch, l):
    return arch.proj(l, 0, "q")[1] is not None


def host64(t):
    return t.detach().float().cpu().double()


def second_moments(model, arch, windows, device):
    """E[x x^T] of every layer's attention input (float64, host), augmented with a constant 1 when the layer's
    projections have biases."""
    d = arch.d
    S, n = {}, {"t": 0}

    def mk(l, aug):
        def pre(mod, inp):
            x = inp[0].detach().reshape(-1, d).float()
            if aug:
                x = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)], 1)
            m = host64(x.T @ x)
            S[l] = m if l not in S else S[l] + m
            if l == 0:
                n["t"] += x.shape[0]
        return pre
    hs = [attn_input_module(arch, l).register_forward_pre_hook(mk(l, has_bias(arch, l))) for l in range(arch.L)]
    for w in range(windows.shape[0]):
        model(input_ids=windows[w:w + 1].to(device), use_cache=False)
    for h in hs:
        h.remove()
    return {l: S[l] / n["t"] for l in S}, n["t"]


def psd_sqrt(M, eps=1e-7):
    w, U = torch.linalg.eigh(M)
    w = w.clamp_min(eps * w.max())
    return (U * w.sqrt()) @ U.mT, (U / w.sqrt()) @ U.mT


def top_eig(F, Om, k):
    """Top-k eigenvectors of F Om F^H (F: m x p, Om: p x p Hermitian PSD) through the p x p problem; directions
    without energy are returned as zero columns."""
    w, U = torch.linalg.eigh(Om)
    Oh = (U * w.clamp_min(0).sqrt()) @ U.conj().mT
    lam, Y = torch.linalg.eigh(Oh @ (F.conj().mT @ F) @ Oh)
    lam, Y = lam.flip(0)[:k], Y.flip(1)[:, :k]
    keep = lam > 1e-12 * lam[0].clamp_min(1e-300)
    V = F @ (Oh @ Y) / lam.clamp_min(1e-300).sqrt()
    return V * keep.to(V.dtype)


def get_out(arch, l, h):
    """(d, hd): head h's output slice, column convention (residual += O_h v)."""
    W = arch.out_module(l).weight
    cols = slice(h * arch.hd, (h + 1) * arch.hd)
    return host64(W[cols, :].T if arch.kind == "gpt2" else W[:, cols])


def set_out(arch, l, h, M):
    W = arch.out_module(l).weight
    cols = slice(h * arch.hd, (h + 1) * arch.hd)
    if arch.kind == "gpt2":
        W[cols, :] = M.T.to(W.dtype).to(W.device)
    else:
        W[:, cols] = M.to(W.dtype).to(W.device)


def get_proj(arch, l, h, which, aug):
    W, b, idx = arch.proj(l, h, which)
    M = host64(W[idx])
    if aug:
        M = torch.cat([M, host64(b[idx])[:, None]], 1)
    return M


def set_proj(arch, l, h, which, M, aug):
    W, b, idx = arch.proj(l, h, which)
    W[idx] = M[:, :arch.d].to(W.dtype).to(W.device)
    if aug:
        b[idx] = M[:, arch.d].to(b.dtype).to(b.device)


def layer_roots(S, d):
    """(L, L^-1, Lk, Lk^-1) for one layer's second moment."""
    L, Li = psd_sqrt(S)
    if S.shape[0] == d + 1:
        mu = S[:d, d]
        Lk, Lki = psd_sqrt(S[:d, :d] - torch.outer(mu, mu))
    else:
        Lk, Lki = L, Li
    return L, Li, Lk, Lki


def fuse_group(arch, l, G, S, roots, query_weighted=True, output_weighted=True):
    """Replace the key and value heads of the heads in G by one shared key and value head."""
    d, hd, r = arch.d, arch.hd, arch.r
    aug = S.shape[0] == d + 1
    L, Li, Lk, Lki = roots
    nr = list(range(r, hd)); half = r // 2
    A = {h: get_proj(arch, l, h, "q", aug) for h in G}
    B = {h: get_proj(arch, l, h, "k", aug) for h in G}
    Cv = {h: get_proj(arch, l, h, "v", aug) for h in G}
    O = {h: get_out(arch, l, h) for h in G}
    eye = torch.eye(hd, dtype=torch.float64)
    # values
    F = torch.cat([L @ Cv[h].T for h in G], 1)
    Om = torch.block_diag(*[(O[h].T @ O[h]) if output_weighted else eye for h in G])
    P = top_eig(F, Om, hd)
    CG = P.T @ Li
    newO = {h: O[h] @ Cv[h] @ L @ P for h in G}
    # keys, dimensions without rotary embedding (their key bias only shifts a query's logits uniformly: zero)
    BG = torch.zeros_like(B[G[0]])
    newA = {h: A[h].clone() for h in G}
    if nr:
        Kf = torch.cat([Lk @ B[h][nr, :d].T for h in G], 1)
        Gq = torch.block_diag(*[(A[h][nr] @ S @ A[h][nr].T) if query_weighted else torch.eye(len(nr), dtype=torch.float64) for h in G])
        R = top_eig(Kf, Gq, len(nr))
        BG[nr, :d] = R.T @ Lki
        for h in G:
            newA[h][nr] = R.T @ Lk @ B[h][nr, :d].T @ A[h][nr]
    # keys, rotary planes
    if half:
        Lc, Lic = L.to(torch.complex128), Li.to(torch.complex128)
        for i in range(half):
            ac = {h: torch.complex(A[h][i], A[h][i + half]) for h in G}
            bhat = {h: Lc @ torch.complex(B[h][i], B[h][i + half]) for h in G}
            ahat = {h: Lc @ ac[h].conj() for h in G}
            wq = torch.tensor([float((ahat[h].conj() @ ahat[h]).real) if query_weighted else 1.0 for h in G], dtype=torch.float64)
            u = top_eig(torch.stack([bhat[h] for h in G], 1), torch.diag(wq).to(torch.complex128), 1)[:, 0]
            u = u / u.norm().clamp_min(1e-300)
            bG = Lic @ u
            BG[i], BG[i + half] = bG.real, bG.imag
            for h in G:
                c = bhat[h] @ u.conj()
                anew = ac[h] * c.conj()
                newA[h][i], newA[h][i + half] = anew.real, anew.imag
    for h in G:
        set_proj(arch, l, h, "q", newA[h], aug)
        set_proj(arch, l, h, "k", BG, aug)
        set_proj(arch, l, h, "v", CG, aug)
        set_out(arch, l, h, newO[h])


def fuse_model(arch, groups_by_layer, S, query_weighted=True, output_weighted=True):
    for l in range(arch.L):
        roots = layer_roots(S[l], arch.d)
        for G in groups_by_layer[l]:
            fuse_group(arch, l, list(G), S[l], roots, query_weighted, output_weighted)

