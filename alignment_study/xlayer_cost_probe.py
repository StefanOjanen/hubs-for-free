# Probe (exploratory, labelled): is sharing one key-value cache between adjacent layers cheaper, in the weighted
# fusion's own closed-form residual, than pairing heads within each layer at the same saving? Both halve the cache of
# a layer pair (l, l+1). Within: every layer's KV heads are paired at minimum cost (kvfuse.pair_costs). Across: every
# KV head i of layer l is paired with one KV head j of layer l+1 (minimum-cost bipartite matching) and the pair shares
# one head computed from layer l's input x_l; layer l+1's head is first replaced by its best linear prediction from
# x_l (W M, M = E[x_{l+1} x_l^T] S_l^-1), whose residual energy is added in full. Values are weighted by each head's
# output projections and keys by each head's query energy per rotary plane, exactly as in the within-layer cost, and
# every cost is the absolute weighted energy lost; the comparison is the total over both layers relative to their
# total weighted energy.
#   python alignment_study/xlayer_cost_probe.py --model=TinyLlama/TinyLlama_v1.1 [--calib=32] [--calib-len=1024]
import json
import os
import sys
import time
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "alignment_study")
import gqa_common as C
import fuse_common as F
import fuse_kv as K
import kvfuse
from scale_round_v2 import DEV
from transformers import AutoTokenizer, AutoModelForCausalLM


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME = arg("model", "TinyLlama/TinyLlama_v1.1")
N_CAL, T_CAL = int(arg("calib", 32)), int(arg("calib-len", 1024))
OUT = f"alignment_study/xlayer_fuse_dev/cost_{NAME.split('/')[-1]}.json"
torch.set_grad_enabled(False)


def head_parts(kv, l, S, W_v=None, W_k=None):
    """Per KV head: value rows, key rows (possibly replaced), output weight sum O^T O, query energy per plane."""
    a, hd, half = kv.attn(l), kv.hd, kv.hd // 2
    Wq, Wo = F.host64(a.q_proj.weight), F.host64(a.o_proj.weight)
    Wv = W_v if W_v is not None else F.host64(a.v_proj.weight)
    Wk = W_k if W_k is not None else F.host64(a.k_proj.weight)
    qe = ((Wq @ S) * Wq).sum(1).view(kv.n, hd)
    out = []
    for i in range(kv.nkv):
        Q = list(range(i * kv.rep, (i + 1) * kv.rep))
        w = qe[Q].sum(0)[:half] + qe[Q].sum(0)[half:]
        OO = sum(Wo[:, h * hd:(h + 1) * hd].T @ Wo[:, h * hd:(h + 1) * hd] for h in Q)
        out.append({"v": Wv[i * hd:(i + 1) * hd], "k": Wk[i * hd:(i + 1) * hd], "OO": OO, "w": w})
    return out


def pair_cost(p, q, S, hd):
    """Weighted energy lost when heads p and q share one key and one value head in the metric S (absolute)."""
    half = hd // 2
    GKpp, GKqq, GKpq = p["k"] @ S @ p["k"].T, q["k"] @ S @ q["k"].T, p["k"] @ S @ q["k"].T
    x = slice(0, half); y = slice(half, hd)
    kp = GKpp[x, x].diagonal() + GKpp[y, y].diagonal(); kq = GKqq[x, x].diagonal() + GKqq[y, y].diagonal()
    re = GKpq[x, x].diagonal() + GKpq[y, y].diagonal(); im = GKpq[y, x].diagonal() - GKpq[x, y].diagonal()
    a11, a22 = p["w"] * kp, q["w"] * kq
    lo = (a11 + a22) / 2 - torch.sqrt(((a11 - a22) / 2) ** 2 + p["w"] * q["w"] * (re ** 2 + im ** 2))
    Op = F.psd_sqrt(p["OO"] + 1e-12 * torch.eye(hd, dtype=torch.float64) * p["OO"].trace())[0]
    Oq = F.psd_sqrt(q["OO"] + 1e-12 * torch.eye(hd, dtype=torch.float64) * q["OO"].trace())[0]
    M = torch.cat([torch.cat([p["v"] @ S @ p["v"].T, p["v"] @ S @ q["v"].T], 1), torch.cat([q["v"] @ S @ p["v"].T, q["v"] @ S @ q["v"].T], 1)])
    Ob = torch.block_diag(Op, Oq)
    lv = torch.linalg.eigvalsh(Ob @ M @ Ob)[:hd].clamp_min(0).sum()
    return float(lo.clamp_min(0).sum()), float(lv)


def energy(p, S, hd):
    half = hd // 2
    G = p["k"] @ S @ p["k"].T
    kd = G[:half, :half].diagonal() + G[half:, half:].diagonal()
    return float((p["w"] * kd).sum()), float((p["OO"] * (p["v"] @ S @ p["v"].T)).sum())


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval().to(DEV)
    kv = K.KV(model)
    d, L, hd = kv.d, kv.L, kv.hd
    S = {l: torch.zeros(d, d, dtype=torch.float64) for l in range(L)}
    X1 = {l: torch.zeros(d, d, dtype=torch.float64) for l in range(L - 1)}
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    n = 0
    for w in range(cal.shape[0]):
        xs = {}
        hooks = [kv.attn(l).k_proj.register_forward_pre_hook((lambda l: lambda m, i: xs.__setitem__(l, i[0][0].float()))(l)) for l in range(L)]
        model(input_ids=cal[w:w + 1].to(DEV), use_cache=False)
        for h in hooks:
            h.remove()
        for l in range(L):
            S[l] += F.host64(xs[l].T @ xs[l])
            if l < L - 1:
                X1[l] += F.host64(xs[l + 1].T @ xs[l])
        n += cal.shape[1]
    S = {l: v / n for l, v in S.items()}; X1 = {l: v / n for l, v in X1.items()}
    res = {"model": NAME, "exploratory": True, "calibration": [N_CAL, T_CAL], "pairs": []}
    within_cost = {}
    for l in range(L):                                   # within-layer: absolute weighted energy lost at the matching
        parts = head_parts(kv, l, S[l])
        Ek = sum(energy(p, S[l], hd)[0] for p in parts); Ev = sum(energy(p, S[l], hd)[1] for p in parts)
        cost = kvfuse.pair_costs(kv, l, S[l])            # relative: Rk / Ek + Rv / Ev
        pairs = kvfuse.min_cost_pairs(cost)
        rk = rv = 0.0
        for i, j in pairs:
            ck, cv = pair_cost(parts[i], parts[j], S[l], hd); rk += ck; rv += cv
        within_cost[l] = {"Ek": Ek, "Ev": Ev, "rk": rk, "rv": rv}
    for l in range(0, L - 1, 2):                          # disjoint layer pairs (0,1), (2,3), ...
        Sl = S[l] + 1e-6 * S[l].diagonal().mean() * torch.eye(d, dtype=torch.float64)
        Mx = torch.linalg.solve(Sl, X1[l].T).T            # x_{l+1} ~ Mx x_l
        R = S[l + 1] - X1[l] @ torch.linalg.solve(Sl, X1[l].T)
        a1 = kv.attn(l + 1)
        Wv1, Wk1 = F.host64(a1.v_proj.weight), F.host64(a1.k_proj.weight)
        p0 = head_parts(kv, l, S[l])
        p1 = head_parts(kv, l + 1, S[l + 1], W_v=Wv1 @ Mx, W_k=Wk1 @ Mx)    # layer l+1's heads as functions of x_l
        irr = []                                                            # prediction residual of each head of l+1
        for j in range(kv.nkv):
            q = head_parts(kv, l + 1, S[l + 1])[j]
            half = hd // 2
            GR = q["k"] @ R @ q["k"].T
            irr_k = float((q["w"] * (GR[:half, :half].diagonal() + GR[half:, half:].diagonal())).sum())
            irr_v = float((q["OO"] * (q["v"] @ R @ q["v"].T)).sum())
            irr.append((irr_k, irr_v))
        cost = torch.zeros(kv.nkv, kv.nkv, dtype=torch.float64)
        parts_cost = {}
        for i in range(kv.nkv):
            for j in range(kv.nkv):
                ck, cv = pair_cost(p0[i], p1[j], S[l], hd)
                parts_cost[(i, j)] = (ck + irr[j][0], cv + irr[j][1])
        E0k, E0v, E1k, E1v = within_cost[l]["Ek"], within_cost[l]["Ev"], within_cost[l + 1]["Ek"], within_cost[l + 1]["Ev"]
        for (i, j), (ck, cv) in parts_cost.items():
            cost[i, j] = ck / (E0k + E1k) + cv / (E0v + E1v)
        ri, rj = linear_sum_assignment(cost.numpy())
        cross_k = sum(parts_cost[(i, j)][0] for i, j in zip(ri, rj)); cross_v = sum(parts_cost[(i, j)][1] for i, j in zip(ri, rj))
        within_k = within_cost[l]["rk"] + within_cost[l + 1]["rk"]; within_v = within_cost[l]["rv"] + within_cost[l + 1]["rv"]
        row = {"layers": [l, l + 1], "within_key": within_k / (E0k + E1k), "within_value": within_v / (E0v + E1v),
               "across_key": cross_k / (E0k + E1k), "across_value": cross_v / (E0v + E1v),
               "across_irreducible_key": sum(irr[j][0] for j in rj) / (E0k + E1k), "across_irreducible_value": sum(irr[j][1] for j in rj) / (E0v + E1v),
               "matching": [[int(i), int(j)] for i, j in zip(ri, rj)]}
        res["pairs"].append(row)
        log(f"layers {l},{l + 1}: within keys {row['within_key']:.3f} values {row['within_value']:.3f}; across keys {row['across_key']:.3f} "
            f"(irreducible {row['across_irreducible_key']:.3f}) values {row['across_value']:.3f} (irreducible {row['across_irreducible_value']:.3f})")
        json.dump(res, open(OUT, "w"), indent=1)
    log("XLAYER_COST_DONE")
