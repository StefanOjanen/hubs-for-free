# Halve (or further reduce) the key-value heads of a Llama-layout model without training, and save the result as
# a real grouped-query checkpoint whose KV cache is smaller by the same factor.
#
#   python alignment_study/kvfuse.py --model meta-llama/Llama-2-7b-hf --out llama2-7b-kv16
#       [--group 2] [--pairing matched|neighbour] [--calib 128] [--calib-len 2048] [--dtype bfloat16]
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
from gqa_export import export_gqa

torch.set_grad_enabled(False)


def pair_costs(kv, l, S):
    """Relative residual of the weighted fusion for every pair of KV heads of layer l (as in llama2_round.py)."""
    assert kv.r == kv.hd and not kv.has_bias(l), "full rotary embedding and no biases (the llama layout)"
    a, hd, half, n = kv.attn(l), kv.hd, kv.hd // 2, kv.nkv
    Q = [list(range(i * kv.rep, (i + 1) * kv.rep)) for i in range(n)]
    Wk, Wv, Wq = (F.host64(m.weight) for m in (a.k_proj, a.v_proj, a.q_proj))
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=2, help="KV heads per group (2 halves the cache)")
    ap.add_argument("--pairing", choices=("matched", "neighbour"), default="matched")
    ap.add_argument("--calib", type=int, default=128)
    ap.add_argument("--calib-len", type=int, default=2048)
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--device", default=None, help="default: the GPU if there is one, else the CPU")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()
    if args.pairing == "matched" and args.group != 2:
        sys.exit("matched pairing is for groups of 2; use --pairing neighbour for larger groups")
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
    if args.pairing == "matched":
        groups = [min_cost_pairs(pair_costs(kv, l, S[l].double())) for l in range(kv.L)]
    else:
        groups = [[list(range(s, s + args.group)) for s in range(0, kv.nkv, args.group)] for _ in range(kv.L)]
    log(f"{args.pairing} groups chosen")
    K.convert(kv, S, groups, "fuse")
    log("fused")
    del S
    export_gqa(model, groups, inplace=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    json.dump({"source": args.model, "revision": args.revision, "group": args.group, "pairing": args.pairing,
               "calibration": {"source": "allenai/c4 en validation shard 0", "documents": args.calib, "tokens": args.calib_len},
               "kv_heads": [kv.nkv, kv.nkv // args.group], "groups": groups}, open(os.path.join(args.out, "kvfuse.json"), "w"), indent=1)
    log(f"saved {args.out} ({kv.nkv // args.group} KV heads per layer)")


if __name__ == "__main__":
    main()
