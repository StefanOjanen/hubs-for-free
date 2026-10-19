# The closed-form fusion of round 26 (fuse_common.py) generalized to key-value heads that serve several query
# heads, for reducing the key-value heads of grouped-query models (PREREGISTRATION27), in llama-type layouts
# (q_proj, k_proj, v_proj, o_proj; rotate-half rotary embedding on the first r dimensions of each head).
# Merging the KV heads in a group G: every query head attached to them reads one shared key and one shared
# value. A KV head's value weighting is the sum of O_h^T O_h over its query heads, its key weighting the sum
# of their query second moments; each query head absorbs its KV head's key transform, each output slice the
# value transform. Multi-head models are the case of one query head per KV head, where this is round 26's fusion.
import torch
import fuse_common as F

torch.set_grad_enabled(False)


class KV:
    def __init__(self, model):
        c = model.config
        self.model, self.blocks = model, model.model.layers
        self.L, self.n = len(self.blocks), c.num_attention_heads
        self.nkv = getattr(c, "num_key_value_heads", None) or self.n
        self.d = c.hidden_size
        self.hd = getattr(c, "head_dim", None) or self.d // self.n
        self.rep = self.n // self.nkv
        rp = getattr(c, "rope_parameters", None) or {}
        frac = rp.get("partial_rotary_factor", None) if isinstance(rp, dict) else None
        frac = frac if frac is not None else (getattr(c, "partial_rotary_factor", None) or 1.0)
        self.r = int(self.hd * frac)

    def attn(self, l):
        return self.blocks[l].self_attn

    def has_bias(self, l):
        return self.attn(l).k_proj.bias is not None

    def params(self, l):
        a = self.attn(l)
        return [p for m in (a.q_proj, a.k_proj, a.v_proj, a.o_proj) for p in (m.weight, m.bias) if p is not None]


def second_moments(model, kv, windows, device):
    """E[x x^T] of every layer's attention input, accumulated and kept in float32 on the host (a 7B model's
    float64 statistics alone take 4 GB); each layer is promoted to float64 only when it is fused."""
    S, n = {}, {"t": 0}
    def mk(l, aug):
        def pre(mod, inp):
            x = inp[0].detach().reshape(-1, kv.d).float()
            if aug:
                x = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)], 1)
            m = (x.T @ x).float().cpu()
            S[l] = m if l not in S else S[l] + m
            if l == 0:
                n["t"] += x.shape[0]
        return pre
    hs = [kv.attn(l).k_proj.register_forward_pre_hook(mk(l, kv.has_bias(l))) for l in range(kv.L)]
    for w in range(windows.shape[0]):
        model(input_ids=windows[w:w + 1].to(device), use_cache=False)
    for h in hs:
        h.remove()
    return {l: S[l] / n["t"] for l in S}, n["t"]


def _rows(mod, i, hd, aug):
    M = F.host64(mod.weight[i * hd:(i + 1) * hd])
    if aug:
        M = torch.cat([M, F.host64(mod.bias[i * hd:(i + 1) * hd])[:, None]], 1)
    return M


def _put(mod, i, hd, M, aug, d):
    mod.weight[i * hd:(i + 1) * hd] = M[:, :d].to(mod.weight.dtype).to(mod.weight.device)
    if aug:
        mod.bias[i * hd:(i + 1) * hd] = M[:, d].to(mod.bias.dtype).to(mod.bias.device)


def mean_pool(kv, l, G):
    a, hd = kv.attn(l), kv.hd
    for mod in (a.k_proj, a.v_proj):
        m = torch.stack([mod.weight[i * hd:(i + 1) * hd] for i in G]).mean(0)
        mb = torch.stack([mod.bias[i * hd:(i + 1) * hd] for i in G]).mean(0) if mod.bias is not None else None
        for i in G:
            mod.weight[i * hd:(i + 1) * hd] = m
            if mb is not None:
                mod.bias[i * hd:(i + 1) * hd] = mb


def fuse_kv_group(kv, l, G, S, roots, qw=True, ow=True, Lsq=None):
    a, d, hd, r = kv.attn(l), kv.d, kv.hd, kv.r
    aug = S.shape[0] == d + 1
    L, Li, Lk, Lki = roots
    nr, half = list(range(r, hd)), r // 2
    Q = {i: list(range(i * kv.rep, (i + 1) * kv.rep)) for i in G}            # query heads of each KV head
    A = {h: _rows(a.q_proj, h, hd, aug) for i in G for h in Q[i]}
    B = {i: _rows(a.k_proj, i, hd, aug) for i in G}
    Cv = {i: _rows(a.v_proj, i, hd, aug) for i in G}
    O = {h: F.host64(a.o_proj.weight[:, h * hd:(h + 1) * hd]) for i in G for h in Q[i]}
    eye = torch.eye(hd, dtype=torch.float64)
    Fm = torch.cat([L @ Cv[i].T for i in G], 1)
    Om = torch.block_diag(*[sum(O[h].T @ O[h] for h in Q[i]) if ow else eye for i in G])
    P = F.top_eig(Fm, Om, hd)
    CG = P.T @ Li
    LP = L @ P                                                    # associate right to left: d^2 hd, not d^3
    newO = {h: O[h] @ (Cv[i] @ LP) for i in G for h in Q[i]}
    BG = torch.zeros_like(B[G[0]])
    newA = {h: A[h].clone() for h in A}
    if nr:
        Kf = torch.cat([Lk @ B[i][nr, :d].T for i in G], 1)
        Gq = torch.block_diag(*[sum(A[h][nr] @ S @ A[h][nr].T for h in Q[i]) if qw else torch.eye(len(nr), dtype=torch.float64) for i in G])
        R = F.top_eig(Kf, Gq, len(nr))
        BG[nr, :d] = R.T @ Lki
        for i in G:
            T = R.T @ Lk @ B[i][nr, :d].T
            for h in Q[i]:
                newA[h][nr] = T @ A[h][nr]
    if half:
        # all rotary planes at once (rows p), in real arithmetic: whitened key planes, query energy per plane,
        # then the shared plane of each p (no (d+1)^2 complex matrices are formed)
        Bh = {i: torch.complex(B[i][:half] @ L, B[i][half:r] @ L) for i in G}
        if qw:
            if Lsq is None:
                Lsq = L @ L
            E = {h: ((A[h][:half] @ Lsq) * A[h][:half]).sum(1) + ((A[h][half:r] @ Lsq) * A[h][half:r]).sum(1)
                 for i in G for h in Q[i]}
        U = torch.zeros(half, L.shape[0], dtype=torch.complex128)
        for p in range(half):
            w = torch.tensor([float(sum(E[h][p] for h in Q[i])) if qw else 1.0 for i in G], dtype=torch.float64)
            u = F.top_eig(torch.stack([Bh[i][p] for i in G], 1), torch.diag(w).to(torch.complex128), 1)[:, 0]
            U[p] = u / u.norm().clamp_min(1e-300)
        BG[:half], BG[half:r] = U.real @ Li, U.imag @ Li
        for i in G:
            cv = (Bh[i] * U.conj()).sum(1)
            for h in Q[i]:
                an = torch.complex(A[h][:half], A[h][half:r]) * cv.conj()[:, None]
                newA[h][:half], newA[h][half:r] = an.real, an.imag
    for i in G:
        _put(a.k_proj, i, hd, BG, aug, d)
        _put(a.v_proj, i, hd, CG, aug, d)
        for h in Q[i]:
            _put(a.q_proj, h, hd, newA[h], aug, d)
            a.o_proj.weight[:, h * hd:(h + 1) * hd] = newO[h].to(a.o_proj.weight.dtype).to(a.o_proj.weight.device)


def convert(kv, S, groups_by_layer, method):
    """method: 'mean', 'fuse_u' (unweighted) or 'fuse' (query- and output-weighted)."""
    for l in range(kv.L):
        if method == "mean":
            for G in groups_by_layer[l]:
                mean_pool(kv, l, G)
            continue
        S64 = S[l].double()
        roots = F.layer_roots(S64, kv.d)
        Lsq = roots[0] @ roots[0] if method == "fuse" else None          # once per layer, not per group
        for G in groups_by_layer[l]:
            fuse_kv_group(kv, l, list(G), S64, roots, qw=(method == "fuse"), ow=(method == "fuse"), Lsq=Lsq)
        del S64, roots, Lsq
