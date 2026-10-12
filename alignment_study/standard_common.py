# Instruments for preregistration 12: the statistics the field uses, with
# function in the loop. Reuses the frozen instruments of round 11
# (functional_common.py) unchanged and adds:
#
# Pair statistics (two query heads of one KV group), on the selection windows,
# rows >= ROW0:
#   js         Jensen-Shannon divergence between the heads' attention rows,
#              mean over rows (Clark et al. 2019, divergence, not distance)
#   cos_map    cosine similarity of the heads' flattened maps (attention-score
#              similarity)
#   corr_last  Pearson correlation of the heads' last attention row (our
#              reconstruction of CHAI's statistic)
#   corr_rows  Pearson correlation of each row over its causal support, mean
#              over rows (the all-rows reading of CHAI's statistic)
# Head statistics, rows >= ROW0:
#   conf       mean over rows of the largest attention weight (the
#              "confidence" of Voita et al. 2019)
#   neg_entropy minus the mean row entropy (focused heads score high)
#   sink_mass  mean attention on the layer's sink columns
#   out_norm   mean norm of the head's output vector a_r V_g W_O,h^T
#
# Outcomes on the evaluation windows (KL of the changed model's next-token
# distribution from the unchanged one, round 11's measure):
#   single-head ablation  the head's o_proj block set to zero
#   multi-pair merge      in every layer from 1 to L-1 and every KV group,
#                         b disjoint pairs chosen greedily by a criterion are
#                         merged at once (round 7's merge), b = 1 and 2
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
import functional_common as F
import merge_dev_calibration as M
from scale_round_v2 import DEV

ROW0 = F.ROW0
NEW_PAIR = ("js", "cos_map", "corr_last", "corr_rows")
OLD_PAIR = F.PREDICTORS
HEAD = ("conf", "neg_entropy", "sink_mass", "out_norm")
SIMILARITY = {"gen_cos", "cos_map", "corr_last", "corr_rows"}      # larger means more alike; every other pair statistic is a distance
CRITERIA = ("out_pert", "val_dist", "tv_raw", "gen_cos", "js", "cos_map", "corr_last", "corr_rows")
N_RANDOM = 5
torch.set_grad_enabled(False)


def layer_stats(A, Vall, Ws, Gs, n, hd, group, S):
    T = A.shape[1]; rows = np.arange(ROW0, T)
    Vg = [Vall[:, g * hd:(g + 1) * hd].astype(np.float64) for g in range(n // group)]
    heads = []
    for h in range(n):
        a = A[h][rows]
        ent = -(a * np.log(np.clip(a, 1e-30, 1.0))).sum(1).mean()
        heads.append({"conf": float(a.max(1).mean()), "neg_entropy": float(-ent), "sink_mass": float(a[:, S].sum(1).mean()),
                      "out_norm": float(F._quad_norm(a @ Vg[h // group], Gs[h]).mean())})
    X = A[:, ROW0:, :].reshape(n, -1); X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
    C = X @ X.T
    Cl = np.nan_to_num(np.corrcoef(A[:, -1, :]))
    Cr = np.zeros((n, n)); k = 0
    for r in rows:
        Cr += np.nan_to_num(np.corrcoef(A[:, r, :r + 1])); k += 1
    Cr /= max(k, 1)
    P = np.clip(A[:, rows, :], 1e-12, 1.0)
    pairs = {}
    for i in range(n):
        for j in range(i + 1, n):
            if i // group != j // group:
                continue
            p, q = P[i], P[j]; m = (p + q) / 2
            js = 0.5 * ((p * np.log(p / m)).sum(1) + (q * np.log(q / m)).sum(1))
            pairs[(i, j)] = {"js": float(js.mean()), "cos_map": float(C[i, j]), "corr_last": float(Cl[i, j]), "corr_rows": float(Cr[i, j])}
    return heads, pairs


def selection_stats(model, sel, n, hd, group, L):
    """Round-11 pair predictors and F1 shares (no F2 nulls) plus the new pair and head statistics, averaged over windows."""
    rng = np.random.default_rng(0)
    store, handles = F.capture_values(model)
    grams = {l: F.o_grams(model, l, n, hd) for l in range(L)}
    acc = {l: {"pairs": {}, "heads": [{k: [] for k in HEAD + ("raw", "val", "out")} for _ in range(n)]} for l in range(L)}
    for x in sel:
        out = model(input_ids=x.to(DEV), output_attentions=True)
        for l in range(L):
            A = out.attentions[l][0].float().cpu().numpy().astype(np.float64)
            Ws, Gs = grams[l]; V = store[l].numpy()
            m = F.layer_measures(A, V, Ws, Gs, n, hd, group, rng, do_f2=False)
            heads, pairs = layer_stats(A, V, Ws, Gs, n, hd, group, m["sink_set"])
            a = acc[l]
            for p, d in m["pairs"].items():
                a["pairs"].setdefault(p, {k: [] for k in OLD_PAIR + NEW_PAIR})
                for k in OLD_PAIR:
                    a["pairs"][p][k].append(d[k])
                for k in NEW_PAIR:
                    a["pairs"][p][k].append(pairs[p][k])
            for h in range(n):
                for k in HEAD:
                    a["heads"][h][k].append(heads[h][k])
                for k in ("raw", "val", "out"):
                    a["heads"][h][k].append(m["f1"][h][k])
        del out
    for hnd in handles:
        hnd.remove()
    return {l: {"pairs": {f"{i}-{j}": {k: float(np.mean(v)) for k, v in d.items()} for (i, j), d in acc[l]["pairs"].items()},
                "heads": [{k: float(np.mean(v)) for k, v in d.items()} for d in acc[l]["heads"]]} for l in range(L)}


def ablate(attn, h, hd, bk):
    s = slice(h * hd, (h + 1) * hd); bk.clear(); bk["ow"] = attn.o_proj.weight[:, s].clone(); attn.o_proj.weight[:, s] = 0


def unablate(attn, h, hd, bk):
    attn.o_proj.weight[:, slice(h * hd, (h + 1) * hd)] = bk["ow"]


def ablation_sweep(model, X, base, layers, n, hd, log=print):
    out, bk = {}, {}
    for l in layers:
        attn = model.model.layers[l].self_attn; out[l] = {}
        for h in range(n):
            ablate(attn, h, hd, bk); kls, dn = F.kl_eval(model, X, base); unablate(attn, h, hd, bk)
            out[l][str(h)] = {"kl": float(np.mean(kls)), "kl_windows": kls, "dnll": float(np.mean(dn))}
        log(f"  ablation layer {l}: {n} heads")
    return out


def choose_pairs(stats_l, crit, n, group, per_group, rng=None):
    keys = [tuple(map(int, k.split("-"))) for k in stats_l]
    chosen = []
    for g in range(n // group):
        cand = [p for p in keys if p[0] // group == g]
        if crit == "random":
            order = [cand[i] for i in rng.permutation(len(cand))]
        else:
            sign = -1.0 if crit in SIMILARITY else 1.0
            order = sorted(cand, key=lambda p: sign * stats_l[f"{p[0]}-{p[1]}"][crit])
        used, picked = set(), 0
        for (i, j) in order:
            if i in used or j in used:
                continue
            chosen.append((i, j)); used |= {i, j}; picked += 1
            if picked == per_group:
                break
    return chosen


def multi_merge_eval(model, X, base, stats, L, n, hd, group, per_group, crit, rng=None):
    plan = [(l, i, j) for l in range(1, L) for (i, j) in choose_pairs(stats[l]["pairs"], crit, n, group, per_group, rng)]
    bks = []
    for (l, i, j) in plan:
        bk = {}; M.merge(model.model.layers[l].self_attn, i, j, hd, bk); bks.append((l, i, j, bk))
    kls, dn = F.kl_eval(model, X, base)
    for (l, i, j, bk) in reversed(bks):
        M.restore(model.model.layers[l].self_attn, i, j, hd, bk)
    return {"kl": float(np.mean(kls)), "kl_windows": kls, "dnll": float(np.mean(dn)), "n_merged": len(plan)}


def multi_merge_all(model, X, base, stats, L, n, hd, group, budgets=(1, 2), log=print):
    res = {}
    for b in budgets:
        res[str(b)] = {}
        for c in CRITERIA:
            res[str(b)][c] = multi_merge_eval(model, X, base, stats, L, n, hd, group, b, c)
        rg = np.random.default_rng(7)
        res[str(b)]["random"] = [multi_merge_eval(model, X, base, stats, L, n, hd, group, b, "random", rg) for _ in range(N_RANDOM)]
        log(f"  multi-merge budget {b}: " + " ".join(f"{c} {res[str(b)][c]['kl']:.3f}" for c in CRITERIA) +
            f" | random median {np.median([r['kl'] for r in res[str(b)]['random']]):.3f}")
    return res
