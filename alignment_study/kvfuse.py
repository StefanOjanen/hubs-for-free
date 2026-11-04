# Halve (or further reduce) the key-value heads of a Llama-layout model without training, and save the result as
# a real grouped-query checkpoint whose KV cache is smaller by the same factor.
#
#   python alignment_study/kvfuse.py --model meta-llama/Llama-2-7b-hf --out llama2-7b-kv16
#       [--group 2|4] [--pairing matched|neighbour|measured] [--calib 128] [--calib-len 2048] [--dtype bfloat16]
#       (measured: each pair of a layer fused alone and scored by its largest KL over held-out C4 windows, then matched;
#        groups of 2; about 28 single-pair conversions per layer. Experimental, not a recipe: out of sample it chose the
#        best or a near-best matching in some layers and the 67th of 105 in others, because a pair's damage can be
#        text-dependent by a factor of a hundred (RUNLOG 2026-11-04, the eight-window pass); measure on text like the
#        deployment's)
#
# What it does (rounds 26 to 28 of alignment_study/NOTE.md):
# 1. the second moment of every layer's attention input over C4 calibration documents (CAGA's setting: 128
#    documents of 2,048 tokens);
# 2. pairs of key-value heads, per layer: the minimum-cost perfect matching under the weighted fusion's own
#    closed-form residual (groups of 2), or neighbouring heads (any group size);
# 3. each group's shared key and value head fitted in closed form to what the heads compute: keys weighted by the
#    queries that read them (a complex scalar per rotary plane absorbed into each query), values weighted by the
#    output projections that write them (fuse_kv.py);
# 4. the export: one key and value head per group, each layer's query heads and output columns permuted so that a
#    group's heads are adjacent, num_key_value_heads set (gqa_export.py).
# The pairing functions reproduce those of the frozen round 28 instrument (llama2_round.py); kvfuse_check.py
# checks that they choose the same pairs and that the export reproduces the simulated conversion.
import argparse
import json
import os
import sys
import time
import networkx as nx
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gqa_common as C
import fuse_common as F
import fuse_kv as K
import fuse_balance as B
from gqa_export import export_gqa

torch.set_grad_enabled(False)


def pair_costs(kv, l, S):
    """Relative residual of the weighted fusion for every pair of KV heads of layer l (as in llama2_round.py)."""
    assert kv.r == kv.hd, "full rotary embedding (the llama layout)"
    aug = S.shape[0] == kv.d + 1                   # projections with biases: S of [x, 1], as fuse_kv.second_moments builds it
    assert aug == kv.has_bias(l), "S carries the ones column exactly when the projections carry biases"
    a, hd, half, n = kv.attn(l), kv.hd, kv.hd // 2, kv.nkv
    Q = [list(range(i * kv.rep, (i + 1) * kv.rep)) for i in range(n)]
    rows = lambda m: torch.cat([F.host64(m.weight), F.host64(m.bias)[:, None]], 1) if aug else F.host64(m.weight)
    Wk, Wv, Wq = (rows(m) for m in (a.k_proj, a.v_proj, a.q_proj))
    Wo = F.host64(a.o_proj.weight)
    GK, GV = Wk @ S @ Wk.T, Wv @ S @ Wv.T
    qe = ((Wq @ S) * Wq).sum(1).view(kv.n, hd)
    w = torch.stack([qe[Q[i]].sum(0)[:half] + qe[Q[i]].sum(0)[half:] for i in range(n)])
    x = lambda i: torch.arange(i * hd, i * hd + half)
    y = lambda i: torch.arange(i * hd + half, (i + 1) * hd)
    kdiag = torch.stack([GK[x(i), x(i)] + GK[y(i), y(i)] for i in range(n)])
    OO = [sum(Wo[:, h * hd:(h + 1) * hd].T @ Wo[:, h * hd:(h + 1) * hd] for h in Q[i]) for i in range(n)]
    Oh = [F.psd_sqrt(o + 1e-12 * torch.eye(hd, dtype=o.dtype) * o.trace())[0] for o in OO]
    ev = torch.tensor([float((OO[i] * GV[i * hd:(i + 1) * hd, i * hd:(i + 1) * hd]).sum()) for i in range(n)])
    Ek, Ev = float((w * kdiag).sum()), float(ev.sum())
    Rk, Rv = torch.zeros(n, n, dtype=torch.float64), torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(i + 1, n):
            re = GK[x(i), x(j)] + GK[y(i), y(j)]; im = GK[y(i), x(j)] - GK[x(i), y(j)]
            a11, a22 = w[i] * kdiag[i], w[j] * kdiag[j]
            off2 = w[i] * w[j] * (re ** 2 + im ** 2)
            lo = (a11 + a22) / 2 - torch.sqrt(((a11 - a22) / 2) ** 2 + off2)
            Mi = torch.cat([torch.cat([GV[i * hd:(i + 1) * hd, i * hd:(i + 1) * hd], GV[i * hd:(i + 1) * hd, j * hd:(j + 1) * hd]], 1),
                            torch.cat([GV[j * hd:(j + 1) * hd, i * hd:(i + 1) * hd], GV[j * hd:(j + 1) * hd, j * hd:(j + 1) * hd]], 1)])
            Ob = torch.block_diag(Oh[i], Oh[j])
            lv = torch.linalg.eigvalsh(Ob @ Mi @ Ob)[:hd].clamp_min(0).sum()
            Rk[i, j] = Rk[j, i] = float(lo.clamp_min(0).sum()); Rv[i, j] = Rv[j, i] = float(lv)
    return Rk / Ek + Rv / Ev


def min_cost_pairs(cost):
    g = nx.Graph()
    n = cost.shape[0]
    g.add_weighted_edges_from((i, j, float(cost[i, j])) for i in range(n) for j in range(i + 1, n))
    return sorted(sorted(e) for e in nx.min_weight_matching(g))


def layer_stats(kv, l, S):
    """The quantities pair_costs builds, computed once per layer, for the cost of groups of any size."""
    assert kv.r == kv.hd, "full rotary embedding (the llama layout)"
    aug = S.shape[0] == kv.d + 1
    assert aug == kv.has_bias(l), "S carries the ones column exactly when the projections carry biases"
    a, hd, half, n = kv.attn(l), kv.hd, kv.hd // 2, kv.nkv
    Q = [list(range(i * kv.rep, (i + 1) * kv.rep)) for i in range(n)]
    rows = lambda m: torch.cat([F.host64(m.weight), F.host64(m.bias)[:, None]], 1) if aug else F.host64(m.weight)
    Wk, Wv, Wq = (rows(m) for m in (a.k_proj, a.v_proj, a.q_proj))
    Wo = F.host64(a.o_proj.weight)
    GK, GV = Wk @ S @ Wk.T, Wv @ S @ Wv.T
    qe = ((Wq @ S) * Wq).sum(1).view(kv.n, hd)
    w = torch.stack([qe[Q[i]].sum(0)[:half] + qe[Q[i]].sum(0)[half:] for i in range(n)])
    x = lambda i: torch.arange(i * hd, i * hd + half)
    y = lambda i: torch.arange(i * hd + half, (i + 1) * hd)
    kdiag = torch.stack([GK[x(i), x(i)] + GK[y(i), y(i)] for i in range(n)])
    OO = [sum(Wo[:, h * hd:(h + 1) * hd].T @ Wo[:, h * hd:(h + 1) * hd] for h in Q[i]) for i in range(n)]
    Oh = [F.psd_sqrt(o + 1e-12 * torch.eye(hd, dtype=o.dtype) * o.trace())[0] for o in OO]
    ev = torch.tensor([float((OO[i] * GV[i * hd:(i + 1) * hd, i * hd:(i + 1) * hd]).sum()) for i in range(n)])
    return {"GK": GK, "GV": GV, "w": w, "Oh": Oh, "hd": hd, "half": half, "Ek": float((w * kdiag).sum()), "Ev": float(ev.sum())}


def group_cost(st, G):
    """Relative residual of the weighted fusion for one group of KV heads, from layer_stats: for the keys, per
    rotary plane, the query-weighted plane energy left after the top eigenvector; for the values, the output-weighted
    energy left after the top hd eigenvectors; each over the layer's total. For a pair it equals pair_costs' entry."""
    GK, GV, w, hd, half, m = st["GK"], st["GV"], st["w"], st["hd"], st["half"], len(G)
    x = lambda i: torch.arange(i * hd, i * hd + half)
    y = lambda i: torch.arange(i * hd + half, (i + 1) * hd)
    M = torch.zeros(half, m, m, dtype=torch.complex128)
    for ai, i in enumerate(G):
        for bj, j in enumerate(G):
            re = GK[x(i), x(j)] + GK[y(i), y(j)]; im = GK[y(i), x(j)] - GK[x(i), y(j)]
            M[:, ai, bj] = torch.sqrt(w[i] * w[j]) * torch.complex(re, im)
    lam = torch.linalg.eigvalsh(M)                                          # (half, m), ascending
    rk = float(lam[:, :-1].clamp_min(0).sum())
    idx = torch.cat([torch.arange(i * hd, (i + 1) * hd) for i in G])
    Ob = torch.block_diag(*[st["Oh"][i] for i in G])
    rv = float(torch.linalg.eigvalsh(Ob @ GV[idx][:, idx] @ Ob)[:-hd].clamp_min(0).sum())
    return rk / st["Ek"] + rv / st["Ev"]


def quad_groups(kv, l, S):
    """Groups of four: the minimum-cost pairs, then the minimum-cost matching of those pairs by the cost of their
    union (two levels; the second sees the four-head residual directly)."""
    st = layer_stats(kv, l, S)
    pairs = min_cost_pairs(pair_costs(kv, l, S))
    cost = torch.zeros(len(pairs), len(pairs), dtype=torch.float64)
    for p in range(len(pairs)):
        for q in range(p + 1, len(pairs)):
            cost[p, q] = cost[q, p] = group_cost(st, pairs[p] + pairs[q])
    return sorted(sorted(pairs[p] + pairs[q]) for p, q in min_cost_pairs(cost))


KL_TOPK = 256


def kl_base(model, X, dev):
    """The unconverted model's top-KL_TOPK log-probabilities per position on the measurement windows."""
    out = {"ids": [], "lp": []}
    for w in range(X.shape[0]):
        logp = torch.log_softmax(model(input_ids=X[w:w + 1].to(dev)).logits[0, :-1].float(), -1)
        lp, ids = logp.topk(KL_TOPK, dim=-1); out["ids"].append(ids); out["lp"].append(lp); del logp
    return out


def kl_windows(model, X, base, dev):
    """Per-window mean KL(unconverted || converted) over the top-KL_TOPK tokens plus the remainder."""
    out = []
    for w in range(X.shape[0]):
        logq = torch.log_softmax(model(input_ids=X[w:w + 1].to(dev)).logits[0, :-1].float(), -1)
        ids, lp = base["ids"][w], base["lp"][w]; lq = logq.gather(1, ids); p, q = lp.exp(), lq.exp()
        rp = (1 - p.sum(1)).clamp_min(1e-12); rq = (1 - q.sum(1)).clamp_min(1e-12)
        out.append(float(((p * (lp - lq)).sum(1) + rp * (rp.log() - rq.log())).mean())); del logq
    return out


def measured_pair_costs(model, kv, l, S64, X, base, dev):
    """Every pair of layer l's KV heads fused alone (the layer otherwise unmerged), scored by KL on the measurement
    windows; returns the cost matrix for min_cost_pairs and the pair KLs. The weights are restored after each pair."""
    n = kv.nkv; roots = F.layer_roots(S64, kv.d); Lsq = roots[0] @ roots[0]
    orig = [p.detach().clone() for p in kv.params(l)]
    cm = torch.zeros(n, n, dtype=torch.float64); pk = {}
    for i in range(n):
        for j in range(i + 1, n):
            K.fuse_kv_group(kv, l, [i, j], S64, roots, qw=True, ow=True, Lsq=Lsq)
            v = kl_windows(model, X, base, dev); cm[i, j] = cm[j, i] = max(v); pk[f"{i}-{j}"] = v        # the largest over windows guards against rare catastrophes
            for p, t in zip(kv.params(l), orig):
                p.copy_(t)
    return cm, pk


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=2, help="KV heads per group (2 halves the cache)")
    ap.add_argument("--pairing", choices=("matched", "neighbour", "measured"), default="matched",
                    help="matched: the fusion's residual; measured: each pair fused alone and scored by KL on held-out windows (groups of 2)")
    ap.add_argument("--measure-windows", type=int, default=8, help="held-out C4 windows for --pairing measured (after the calibration and check documents); a pair's cost is its largest KL over them")
    ap.add_argument("--calib", type=int, default=128)
    ap.add_argument("--calib-len", type=int, default=2048)
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--device", default=None, help="default: the GPU if there is one, else the CPU")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--no-balance", action="store_true",
                    help="skip rebalancing (fuse_balance.py; on by default: same function, weights that round and train well)")
    args = ap.parse_args()
    if args.pairing == "matched" and args.group not in (2, 4):
        sys.exit("matched pairing is for groups of 2 or 4; use --pairing neighbour for other sizes")
    if args.pairing == "measured" and args.group != 2:
        sys.exit("measured pairing is for groups of 2")
    dev = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=getattr(torch, args.dtype),
                                                 attn_implementation="sdpa").eval().to(dev)
    kv = K.KV(model)
    if kv.nkv % args.group:
        sys.exit(f"{kv.nkv} KV heads cannot be split into groups of {args.group}")
    log(f"{args.model}: {kv.L} layers, {kv.n} query heads, {kv.nkv} KV heads -> {kv.nkv // args.group}")
    cal = C.windows_c4(tok, args.calib, args.calib_len)
    S, ntok = K.second_moments(model, kv, cal, dev)
    log(f"second moments from {ntok} calibration tokens")
    measured = None
    if args.pairing == "matched":
        groups = [min_cost_pairs(pair_costs(kv, l, S[l].double())) if args.group == 2 else quad_groups(kv, l, S[l].double())
                  for l in range(kv.L)]
    elif args.pairing == "measured":
        X = C.windows_c4(tok, args.calib + 1 + args.measure_windows, args.calib_len)[-args.measure_windows:]
        base = kl_base(model, X, dev); groups, measured = [], {}
        for l in range(kv.L):
            t1 = time.time(); cm, pk = measured_pair_costs(model, kv, l, S[l].double(), X, base, dev)
            groups.append(min_cost_pairs(cm)); measured[l] = pk
            log(f"layer {l}: {len(pk)} pairs measured in {time.time() - t1:.0f}s; matching {groups[-1]}")
    else:
        groups = [[list(range(s, s + args.group)) for s in range(0, kv.nkv, args.group)] for _ in range(kv.L)]
    log(f"{args.pairing} groups chosen")
    # fuse and rebalance each layer in float32, rounding to the model's dtype once (rebalancing weights that are
    # already in bfloat16 rounds them twice: preregistration 30)
    for l in range(kv.L):
        a = kv.attn(l)
        mods = (a.q_proj, a.k_proj, a.v_proj, a.o_proj)
        for m in mods:
            m.float()
        S64 = S[l].double(); roots = F.layer_roots(S64, kv.d)
        Lsq = roots[0] @ roots[0]
        for G in groups[l]:
            K.fuse_kv_group(kv, l, list(G), S64, roots, qw=True, ow=True, Lsq=Lsq)
        if not args.no_balance:
            B.balance(kv, groups, layers=[l])
        for m in mods:
            m.to(getattr(torch, args.dtype))
        del S64, roots, Lsq
    log("fused" + ("" if args.no_balance else " and rebalanced"))
    del S
    export_gqa(model, groups, inplace=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    json.dump({"source": args.model, "revision": args.revision, "group": args.group, "pairing": args.pairing,
               "calibration": {"source": "allenai/c4 en validation shard 0", "documents": args.calib, "tokens": args.calib_len},
               "measure_windows": args.measure_windows if args.pairing == "measured" else None, "measured_pair_kl": measured,
               "kv_heads": [kv.nkv, kv.nkv // args.group], "balanced": not args.no_balance, "groups": groups}, open(os.path.join(args.out, "kvfuse.json"), "w"), indent=1)
    log(f"saved {args.out} ({kv.nkv // args.group} KV heads per layer)")


if __name__ == "__main__":
    main()
