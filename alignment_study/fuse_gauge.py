# Layer-by-layer conversion on the host in float64 with a change of gauge before the single rounding to the model's
# precision (round 28). The fusion of fuse_kv.py writes each shared key plane as a whitened unit vector, the
# queries absorbing the scale, and each shared value projection in a whitened basis, the output projections
# absorbing the rest; exact in float64, but stored in bfloat16 such weights take new rounding errors. A shared
# key plane may be multiplied by any complex number s if the queries reading it take 1/conj(s), and a shared value
# projection by any invertible matrix M if the output projections take M^-1, without changing the function. The
# gauge chosen here puts each group's shared key planes and value projection as close to the original members'
# as such factors allow (for a group of one, exactly the original weights), and the weights are rounded once.
# Constructions: "fuse" (query- and output-weighted, fuse_kv.py), "fuse_u" (unweighted), "caga" (CAGA's keys,
# which are the unweighted key fit, and its values from the SVD of the stacked value projections).
import torch
import fuse_common as F
import fuse_kv as K

torch.set_grad_enabled(False)


def caga_group(kv, l, G, S64, roots):
    a, hd = kv.attn(l), kv.hd
    Cv = torch.cat([F.host64(a.v_proj.weight[i * hd:(i + 1) * hd]) for i in G])
    O = {h: F.host64(a.o_proj.weight[:, h * hd:(h + 1) * hd]) for i in G for h in range(i * kv.rep, (i + 1) * kv.rep)}
    K.fuse_kv_group(kv, l, G, S64, roots, qw=False, ow=False)          # keys: the covariance-metric fit, a complex scalar per plane
    U, s, Vt = torch.linalg.svd(Cv, full_matrices=False)              # values: rank-d_h SVD of the stacked projections
    Vbar = s[:hd, None] * Vt[:hd]
    for k, i in enumerate(G):
        a.v_proj.weight[i * hd:(i + 1) * hd] = Vbar.to(a.v_proj.weight.dtype).to(a.v_proj.weight.device)
        for h in range(i * kv.rep, (i + 1) * kv.rep):
            a.o_proj.weight[:, h * hd:(h + 1) * hd] = (O[h] @ U[k * hd:(k + 1) * hd, :hd]).to(a.o_proj.weight.dtype).to(a.o_proj.weight.device)


def gauge_fix_layer(kv, l, groups, S, orig):
    """Rescale converted layer l (weights in float64 on the host). orig: the layer's original k and v weights."""
    a, hd, half = kv.attn(l), kv.hd, kv.hd // 2
    assert kv.r == kv.hd, "full rotary embedding only"
    Wq, Wk, Wv, Wo = a.q_proj.weight, a.k_proj.weight, a.v_proj.weight, a.o_proj.weight
    for G in groups:
        G = list(G)
        Q = {i: list(range(i * kv.rep, (i + 1) * kv.rep)) for i in G}
        r0 = slice(G[0] * hd, (G[0] + 1) * hd)
        # keys: member coefficients cv_i = <b_i, v>_S / <v, v>_S on each shared plane v; s = their RMS magnitude
        # with the phase of the largest
        BG = Wk[r0]
        v = torch.complex(BG[:half], BG[half:])
        vS = torch.complex(BG[:half] @ S, BG[half:] @ S)
        vv = (vS * v.conj()).sum(1).real.clamp_min(1e-300)
        cv = torch.stack([(torch.complex(orig["k"][i * hd:i * hd + half], orig["k"][i * hd + half:(i + 1) * hd]) * vS.conj()).sum(1) / vv for i in G])
        ph = cv[cv.abs().argmax(0), torch.arange(half)]
        s = cv.abs().pow(2).mean(0).sqrt() * ph / ph.abs().clamp_min(1e-300)
        s = torch.where(s.abs() > 0, s, torch.ones_like(s))                     # a plane without energy keeps its scale
        vn = v * s[:, None]
        for i in G:
            Wk[i * hd:i * hd + half], Wk[i * hd + half:(i + 1) * hd] = vn.real, vn.imag
            for h in Q[i]:
                q = torch.complex(Wq[h * hd:h * hd + half], Wq[h * hd + half:(h + 1) * hd]) / s.conj()[:, None]
                Wq[h * hd:h * hd + half], Wq[h * hd + half:(h + 1) * hd] = q.real, q.imag
        # values: C_i ~ R_i C_G (S-metric least squares); M = mean of R_i, unless it is poorly conditioned
        CG = Wv[r0].clone()
        R = [orig["v"][i * hd:(i + 1) * hd] @ S @ CG.T @ torch.linalg.pinv(CG @ S @ CG.T) for i in G]
        M = sum(R) / len(R)
        if not torch.isfinite(M).all() or torch.linalg.cond(M) > 1e4:
            continue
        Mi = torch.linalg.inv(M)
        for i in G:
            Wv[i * hd:(i + 1) * hd] = M @ CG
            for h in Q[i]:
                Wo[:, h * hd:(h + 1) * hd] = Wo[:, h * hd:(h + 1) * hd] @ Mi


def convert(kv, S, groups_by_layer, construction, gauge=True):
    """construction in ('fuse', 'fuse_u', 'caga'); each layer's attention weights are moved to the host in
    float64, converted, optionally put in the gauge above, rounded once to the model's dtype and moved back."""
    for l in range(kv.L):
        a = kv.attn(l)
        mods = (a.q_proj, a.k_proj, a.v_proj, a.o_proj)
        assert all(m.bias is None for m in mods), "llama layout without biases"
        dev, dt = a.q_proj.weight.device, a.q_proj.weight.dtype
        for m in mods:
            m.weight.data = m.weight.data.cpu().double()            # not .to("cpu", torch.float64): from MPS that corrupts large tensors (torch 2.13)
        orig = {"k": a.k_proj.weight.detach().clone(), "v": a.v_proj.weight.detach().clone()}
        S64 = S[l].double()
        roots = F.layer_roots(S64, kv.d)
        Lsq = roots[0] @ roots[0] if construction == "fuse" else None
        for G in groups_by_layer[l]:
            if construction == "caga":
                caga_group(kv, l, list(G), S64, roots)
            else:
                K.fuse_kv_group(kv, l, list(G), S64, roots, qw=(construction == "fuse"), ow=(construction == "fuse"), Lsq=Lsq)
        if gauge:
            gauge_fix_layer(kv, l, groups_by_layer[l], S64, orig)
        for m in mods:
            m.weight.data = m.weight.data.to(dt).to(dev)
        del S64, roots, Lsq, orig
