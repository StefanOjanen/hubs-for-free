# The fusion fitted in the metrics the attention computation implies (round 32 instruments; the development versions
# are attnw_mix_dev.moments and attnw_mix_dev.fuse_split, which hold every metric as a d x d matrix).
#   keys: S_w, each token s weighted by w_s = sum_t p_ts |z_s - z_bar_t|^2 (z: the value seen through the head's output
#     projection, |z|^2 = v^T O_h^T O_h v; summed over the layer's heads), the diagonal of the first-order output error
#     of a logit error at (t, s);
#   query energies: per head and rotary plane, E_h[p] = sum_t u_t |q_{t,p}|^2 with u_t = sum_s p_ts |z_s - z_bar_t|^2
#     (summed over heads), equal to the plane energies fuse_kv.fuse_kv_group computes from Lsq = S_u, without forming
#     the d x d matrix S_u;
#   values: S_bar = E_t[x_bar_t x_bar_t^T], x_bar_t = sum_s p_ts x_s, averaged over the layer's heads (BoA's value
#     Hessian; exact for the value error with the attention held fixed).
# Queries every QSTRIDE-th position, the attention input in float32, full rotary embeddings only (asserted). Where a
# layer's projections carry biases, S_w and S_bar are moments of [x, 1] (the ones column fuse_kv.second_moments appends;
# round 31's instrument left it out) and the energies are those of the biased queries.
import torch
import fuse_common as F
import fuse_kv as K
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

QSTRIDE = 8


class Stop(Exception):
    pass


def metrics(model, kv, cal, layers, device):
    """S_w and S_bar (d x d, float32, host) and the query energies E (n x half, float64, host) of the given layers, from
    the model as it is; the forward pass stops after the last of them."""
    d, hd, n, half = kv.d, kv.hd, kv.n, kv.hd // 2
    assert kv.r == kv.hd, "full rotary only"
    Sw = {l: torch.zeros(d + kv.has_bias(l), d + kv.has_bias(l)) for l in layers}
    Sb = {l: torch.zeros(d + kv.has_bias(l), d + kv.has_bias(l)) for l in layers}
    E = {l: torch.zeros(n, half, dtype=torch.float64) for l in layers}
    lin = torch.nn.functional.linear
    R = {}                                                     # R_h R_h^T = O_h^T O_h
    for l in layers:
        Ow = kv.attn(l).o_proj.weight
        R[l] = [torch.linalg.cholesky((lambda Oh: Oh.T @ Oh + 1e-8 * torch.eye(hd, dtype=torch.float64))(
            Ow[:, h * hd:(h + 1) * hd].cpu().double())).float().to(device) for h in range(n)]
    last = max(layers)
    for w in range(cal.shape[0]):
        T = cal.shape[1]
        pos = torch.arange(T, device=device)[None]
        rows = torch.arange(QSTRIDE - 1, T, QSTRIDE, device=device)
        mask = torch.ones(T, T, dtype=torch.bool, device=device).tril()[rows]

        def mk(l):
            def pre(mod, inp):
                x = inp[0][0].float()
                a = kv.attn(l)
                cos, sin = model.model.rotary_emb(x[None], position_ids=pos)
                fb = lambda m: m.bias.float() if m.bias is not None else None
                q0 = lin(x, a.q_proj.weight.float(), fb(a.q_proj)).view(1, T, n, hd).transpose(1, 2)
                k = lin(x, a.k_proj.weight.float(), fb(a.k_proj)).view(1, T, kv.nkv, hd).transpose(1, 2)
                v = lin(x, a.v_proj.weight.float(), fb(a.v_proj)).view(T, kv.nkv, hd).transpose(0, 1)
                q, k = apply_rotary_pos_emb(q0, k, cos, sin)
                ws, us = torch.zeros(T, device=x.device), torch.zeros(len(rows), device=x.device)
                A = torch.zeros(T, T, device=x.device)
                scale = getattr(a, "scaling", hd ** -0.5)
                for h in range(n):
                    p = torch.softmax(((q[0, h][rows] @ k[0, h // kv.rep].T) * scale).masked_fill(~mask, float("-inf")), -1)
                    A += p.T @ p
                    z = v[h // kv.rep] @ R[l][h]
                    zz, zb = (z * z).sum(1), p @ z
                    zbn = (zb * zb).sum(1)
                    ws += p.sum(0) * zz - 2 * (z * (p.T @ zb)).sum(1) + p.T @ zbn
                    us += p @ zz - zbn
                uw = us / us.sum() * T                         # attnw_mix_dev's S_u weights, per window
                if kv.has_bias(l):
                    x = torch.cat([x, torch.ones(T, 1, device=x.device)], 1)
                Sw[l] += ((x * (ws / ws.sum() * T)[:, None]).T @ x).cpu()
                Sb[l] += (x.T @ (A @ x) / (n * len(rows))).cpu()
                qr = q0[0][:, rows]                            # (n, rows, hd), before the rotation, which keeps plane norms
                E[l] += (((qr[..., :half] ** 2 + qr[..., half:] ** 2) * uw[None, :, None]).sum(1)).cpu().double()
                if l == last:
                    raise Stop
            return pre
        hooks = [kv.attn(l).k_proj.register_forward_pre_hook(mk(l)) for l in layers]
        try:
            model(input_ids=cal[w:w + 1].to(device), use_cache=False)
        except Stop:
            pass
        finally:
            for h in hooks:
                h.remove()
    ntok = cal.shape[0] * cal.shape[1]
    return {l: Sw[l] / ntok for l in layers}, {l: Sb[l] / ntok for l in layers}, {l: E[l] / ntok for l in layers}


def fuse_kv_group_E(kv, l, G, S, roots, E):
    """fuse_kv.fuse_kv_group's key fit with the query energies given per head and rotary plane (E[h], length half)
    instead of computed from a d x d metric; full rotary (biases through the ones column, as in fuse_kv_group). Writes
    the shared key and the queries only."""
    a, d, hd, r = kv.attn(l), kv.d, kv.hd, kv.r
    aug = S.shape[0] == d + 1
    assert r == hd and aug == kv.has_bias(l)
    L, Li, _, _ = roots
    half = r // 2
    Q = {i: list(range(i * kv.rep, (i + 1) * kv.rep)) for i in G}
    A = {h: K._rows(a.q_proj, h, hd, aug) for i in G for h in Q[i]}
    B = {i: K._rows(a.k_proj, i, hd, aug) for i in G}
    BG = torch.zeros_like(B[G[0]])
    newA = {h: A[h].clone() for h in A}
    Bh = {i: torch.complex(B[i][:half] @ L, B[i][half:r] @ L) for i in G}
    U = torch.zeros(half, L.shape[0], dtype=torch.complex128)
    for p in range(half):
        w = torch.tensor([float(sum(E[h][p] for h in Q[i])) for i in G], dtype=torch.float64)
        u = F.top_eig(torch.stack([Bh[i][p] for i in G], 1), torch.diag(w).to(torch.complex128), 1)[:, 0]
        U[p] = u / u.norm().clamp_min(1e-300)
    BG[:half], BG[half:r] = U.real @ Li, U.imag @ Li
    for i in G:
        cv = (Bh[i] * U.conj()).sum(1)
        for h in Q[i]:
            an = torch.complex(A[h][:half], A[h][half:r]) * cv.conj()[:, None]
            newA[h][:half], newA[h][half:r] = an.real, an.imag
    for i in G:
        K._put(a.k_proj, i, hd, BG, aug, d)
        for h in Q[i]:
            K._put(a.q_proj, h, hd, newA[h], aug, d)


def fuse_layer(kv, l, groups_l, Sw, E, Sb):
    """Layer l's groups fused: values and output projections by the frozen fuse_kv.fuse_kv_group in S_bar, then, from
    the original keys and queries (restored), keys and queries by fuse_kv_group_E in S_w with the plane energies E
    (n x half). attnw_mix_dev.fuse_split does the same with S_u as a d x d metric."""
    a, hd = kv.attn(l), kv.hd
    w64, b64 = Sw.double(), Sb.double()
    rw, rb = F.layer_roots(w64, kv.d), F.layer_roots(b64, kv.d)
    for G in groups_l:
        G = list(G)
        Qh = [h for i in G for h in range(i * kv.rep, (i + 1) * kv.rep)]
        krows = [slice(i * hd, (i + 1) * hd) for i in G]; qrows = [slice(h * hd, (h + 1) * hd) for h in Qh]
        keep = [(m.weight, s) for m, rr in ((a.k_proj, krows), (a.q_proj, qrows)) for s in rr]
        keep += [(m.bias, s) for m, rr in ((a.k_proj, krows), (a.q_proj, qrows)) for s in rr if m.bias is not None]
        saved = [t[s].clone() for t, s in keep]
        K.fuse_kv_group(kv, l, G, b64, rb, qw=True, ow=True, Lsq=b64)
        for (t, s), v in zip(keep, saved):
            t[s] = v
        fuse_kv_group_E(kv, l, G, w64, rw, E)
