# Instruments for the functional round: attention weighted by what it moves.
#
# Shared by functional_dev.py (labeled development calibration) and the
# confirmatory script, so both run identical code. Three measurements per
# layer, all on the selection windows of the head-merging round (WikiText-103
# validation, T = 256), and one functional outcome on disjoint evaluation
# windows (WikiText-103 test):
#
# F1, the sink's functional share. For head h, the value vector that position
# c contributes through the head is f_hc = W_O,h v_gc (v from the head's KV
# group g, W_O,h the head's slice of o_proj). Raw sink share: attention mass
# on the layer's sink columns S (hubsfree.stats.sink_columns). Value-norm
# share (Kobayashi et al. 2020): sum over S of a_rc |f_hc| divided by the sum
# over all c, averaged over rows r >= ROW0. Output share: norm of the sink
# columns' contribution to the head output divided by the norm of the head
# output.
#
# F2, head similarity before and after weighting. Mean pairwise cosine of the
# heads' flattened maps (rows >= ROW0) for the raw maps a and for the
# value-norm-weighted maps a_rc |f_hc|, with two nulls on each: a per-row
# permutation within the causal support (diagonal kept) and the same
# permutation with the sink columns kept. Reproduced share of the sink-column
# null: (sinknull - permutation) / (real - permutation).
#
# F3, pair predictors for merging two query heads i, j of one KV group (the
# merge of PREREGISTRATION7: averaged q_proj rows, averaged o_proj blocks).
# With D = a_i - a_j on rows >= ROW0:
#   gen_cos     cos(G_i, G_j), G = (A - A^T)/2 (the criterion of round 7)
#   tv_raw      mean over rows of half the L1 norm of D
#   tv_nosink   the same over columns outside S only
#   val_dist    mean over rows of |D_r V_g|
#   out_pert    mean over rows of |(1/2) D_r V_g (W_O,i - W_O,j)^T|, the
#               first-order change of the layer output under the merge
#               with the merged pattern approximated by the average pattern
#   sink_diff   mean over rows of |sum over S of D_rc|
# Outcome: KL divergence of the merged model's next-token distribution from
# the unmerged one, over the unmerged model's top-TOPK tokens plus one
# remainder bucket (a lower bound on the full KL by the data-processing
# inequality), averaged over all evaluation tokens; the change in next-token
# NLL is recorded alongside.
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import merge_dev_calibration as M
from scale_round_v2 import DEV
from hubsfree.stats import sink_columns
from hubsfree.nulls import surrogate_plain, surrogate_colfix

N_SEL, T_SEL = 16, 256          # selection windows, as in round 7
N_EVAL, T_EVAL = 8, 512         # evaluation windows for the KL outcome
TOPK = 256
ROW0 = 16                       # rows below this have too few positions
F2_WINDOWS, F2_DRAWS = 8, 10
PREDICTORS = ("gen_cos", "tv_raw", "tv_nosink", "val_dist", "out_pert", "sink_diff")
torch.set_grad_enabled(False)


def geometry(model):
    cfg = model.config
    n, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // n
    return cfg.num_hidden_layers, n, nkv, hd, n // nkv


def capture_values(model):
    store, handles = {}, []
    for l, layer in enumerate(model.model.layers):
        def hook(mod, inp, out, l=l):
            store[l] = out[0].detach().float().cpu()
        handles.append(layer.self_attn.v_proj.register_forward_hook(hook))
    return store, handles


def o_grams(model, l, n, hd):
    """Per head W_O,h (hidden x hd) as float64, and its Gram W^T W (hd x hd)."""
    W = model.model.layers[l].self_attn.o_proj.weight.detach().float().cpu().numpy().astype(np.float64)
    Ws = [W[:, h * hd:(h + 1) * hd] for h in range(n)]
    return Ws, [w.T @ w for w in Ws]


def _quad_norm(U, G):
    """Row-wise sqrt(u G u^T) for U (r x hd)."""
    return np.sqrt(np.clip(np.einsum("rd,de,re->r", U, G, U), 0, None))


def layer_measures(A, Vall, Ws, Gs, n, hd, group, rng, do_f2):
    """A: (n, T, T) attention; Vall: (T, nkv*hd) values of the layer."""
    T = A.shape[1]
    S = list(sink_columns(A))
    rows = np.arange(ROW0, T)
    notS = np.ones(T, bool); notS[S] = False
    Vg = [Vall[:, g * hd:(g + 1) * hd].astype(np.float64) for g in range(n // group)]
    fnorm = np.stack([_quad_norm(Vg[h // group], Gs[h]) for h in range(n)])          # (n, T): |f_hc|
    out = {"sink_set": S, "f1": [], "pairs": {}}
    for h in range(n):
        a = A[h][rows]                                                                 # (R, T)
        raw = a[:, S].sum(1)
        w = a * fnorm[h][None, :]
        valshare = w[:, S].sum(1) / np.maximum(w.sum(1), 1e-30)
        u_all = a @ Vg[h // group]; u_S = a[:, S] @ Vg[h // group][S]
        outshare = _quad_norm(u_S, Gs[h]) / np.maximum(_quad_norm(u_all, Gs[h]), 1e-30)
        sink_f = float(np.mean(fnorm[h][S])); other_f = float(np.mean(fnorm[h][notS])) if notS.any() else float("nan")
        out["f1"].append({"head": h, "raw": float(raw.mean()), "val": float(valshare.mean()), "out": float(outshare.mean()),
                          "fnorm_sink_over_rest": sink_f / other_f if other_f > 0 else float("nan")})
    Gen = (A - A.transpose(0, 2, 1)) / 2.0
    Gf = Gen.reshape(n, -1); Gf = Gf / np.linalg.norm(Gf, axis=1, keepdims=True)
    gcos = Gf @ Gf.T
    for i in range(n):
        for j in range(i + 1, n):
            if i // group != j // group:
                continue
            D = (A[i] - A[j])[rows]
            U = 0.5 * D @ Vg[i // group]
            Wd = Ws[i] - Ws[j]
            out["pairs"][(i, j)] = {"gen_cos": float(gcos[i, j]),
                                    "tv_raw": float(0.5 * np.abs(D).sum(1).mean()),
                                    "tv_nosink": float(0.5 * np.abs(D[:, notS]).sum(1).mean()),
                                    "val_dist": float(np.linalg.norm(D @ Vg[i // group], axis=1).mean()),
                                    "out_pert": float(_quad_norm(U, Wd.T @ Wd).mean()),
                                    "sink_diff": float(np.abs(D[:, S].sum(1)).mean())}
    if do_f2:
        out["f2"] = {}
        for kind, maps in (("raw", A), ("val", A * fnorm[:, None, :])):
            def cosmean(Ms):
                X = Ms[..., ROW0:, :].reshape(Ms.shape[-3], -1)
                X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
                C = X @ X.T
                return float(C[np.triu_indices(C.shape[0], 1)].mean())
            real = cosmean(maps)
            P = surrogate_plain(maps, rng, F2_DRAWS, causal=True)
            Cf = surrogate_colfix(maps, rng, cols=tuple(S), draws=F2_DRAWS, causal=True)
            out["f2"][kind] = {"real": real, "perm": [cosmean(P[d]) for d in range(F2_DRAWS)],
                               "sinkfix": [cosmean(Cf[d]) for d in range(F2_DRAWS)]}
    return out


def selection_measures(model, sel, layers, n, hd, group, seed=0):
    """Averages the per-window measures over the selection windows."""
    rng = np.random.default_rng(seed)
    store, handles = capture_values(model)
    grams = {l: o_grams(model, l, n, hd) for l in layers}
    acc = {l: {"pairs": {}, "f1": None, "f2": {"raw": {"real": [], "perm": [], "sinkfix": []},
                                                "val": {"real": [], "perm": [], "sinkfix": []}}, "sink_sets": []} for l in layers}
    for w, x in enumerate(sel):
        out = model(input_ids=x.to(DEV), output_attentions=True)
        for l in layers:
            A = out.attentions[l][0].float().cpu().numpy().astype(np.float64)
            Ws, Gs = grams[l]
            m = layer_measures(A, store[l].numpy(), Ws, Gs, n, hd, group, rng, do_f2=w < F2_WINDOWS)
            a = acc[l]; a["sink_sets"].append(m["sink_set"])
            for p, d in m["pairs"].items():
                a["pairs"].setdefault(p, {k: [] for k in PREDICTORS})
                for k in PREDICTORS:
                    a["pairs"][p][k].append(d[k])
            if a["f1"] is None:
                a["f1"] = [{k: [] for k in ("raw", "val", "out", "fnorm_sink_over_rest")} for _ in range(n)]
            for h, d in enumerate(m["f1"]):
                for k in ("raw", "val", "out", "fnorm_sink_over_rest"):
                    a["f1"][h][k].append(d[k])
            if "f2" in m:
                for kind in ("raw", "val"):
                    for k in ("real", "perm", "sinkfix"):
                        a["f2"][kind][k].append(m["f2"][kind][k])
        del out
    for h in handles:
        h.remove()
    res = {}
    for l in layers:
        a = acc[l]
        pairs = {f"{i}-{j}": {k: float(np.mean(v)) for k, v in d.items()} for (i, j), d in a["pairs"].items()}
        f1 = [{k: float(np.nanmean(v)) for k, v in d.items()} for d in a["f1"]]
        f2 = {}
        for kind in ("raw", "val"):
            real = float(np.mean(a["f2"][kind]["real"]))
            perm = np.array(a["f2"][kind]["perm"]).mean(0); sfx = np.array(a["f2"][kind]["sinkfix"]).mean(0)   # per draw, mean over windows
            pm, sm = float(np.median(perm)), float(np.median(sfx))
            f2[kind] = {"real": real, "perm_median": pm, "sinkfix_median": sm,
                        "reproduced": (sm - pm) / (real - pm) if abs(real - pm) > 1e-12 else None,
                        "sinkfix_pct_of_real": float((sfx < real).mean() * 100)}
        res[l] = {"pairs": pairs, "f1": f1, "f2": f2, "sink_sets": sorted({str(s) for s in a["sink_sets"]})}
    return res


def base_cache(model, X):
    """Unmerged top-TOPK log-probabilities and target NLL per evaluation window."""
    cache = {"ids": [], "lp": [], "nll": []}
    for w in range(X.shape[0]):
        x = X[w:w + 1].to(DEV)
        logp = torch.log_softmax(model(input_ids=x).logits[0, :-1].float(), -1)
        lp, ids = logp.topk(TOPK, dim=-1)
        cache["ids"].append(ids); cache["lp"].append(lp)
        cache["nll"].append(-logp.gather(1, x[0, 1:, None])[:, 0])
        del logp
    return cache


def kl_eval(model, X, base):
    """Per-window mean KL(unmerged || merged) over top-TOPK plus remainder, and mean change in NLL."""
    kls, dn = [], []
    for w in range(X.shape[0]):
        x = X[w:w + 1].to(DEV)
        logq = torch.log_softmax(model(input_ids=x).logits[0, :-1].float(), -1)
        ids, lp = base["ids"][w], base["lp"][w]
        lq = logq.gather(1, ids)
        p, q = lp.exp(), lq.exp()
        rp = (1 - p.sum(1)).clamp_min(1e-12); rq = (1 - q.sum(1)).clamp_min(1e-12)
        kl = (p * (lp - lq)).sum(1) + rp * (rp.log() - rq.log())
        nll = -logq.gather(1, x[0, 1:, None])[:, 0]
        kls.append(float(kl.mean())); dn.append(float((nll - base["nll"][w]).mean()))
        del logq
    return kls, dn


def sweep(model, X, base, layers, pairs_by_layer, hd, log=print):
    """Merge every listed pair (round-7 merge), evaluate, restore."""
    out, bk = {}, {}
    for l in layers:
        attn = model.model.layers[l].self_attn
        out[l] = {}
        for (i, j) in pairs_by_layer[l]:
            M.merge(attn, i, j, hd, bk)
            kls, dn = kl_eval(model, X, base)
            M.restore(attn, i, j, hd, bk)
            out[l][f"{i}-{j}"] = {"kl": float(np.mean(kls)), "kl_windows": kls, "dnll": float(np.mean(dn)), "dnll_windows": dn}
        log(f"  layer {l}: {len(pairs_by_layer[l])} merges")
    return out
