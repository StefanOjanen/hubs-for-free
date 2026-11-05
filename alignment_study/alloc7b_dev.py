# Development (exploratory, labelled): cost-aware allocation of key-value heads across layers at a fixed cache budget on a 7B
# model. The design is the parallel session's (private/drafts/DEV_layer_allocation.md; on CPU, TinyLlama's WikiText perplexity
# at uniform pairs' head count went from 71.56 to 37.05 and Llama-3.2-1B's from 128.19 to 96.49, each below three random
# allocations), run here under round 28's harness (bfloat16, pinned revision, 128 calibration documents of 2,048 tokens, the
# residual's pairs and groups of four, the query- and output-weighted fusion). Each layer may stay unmerged, go to its matched
# pairs or to its groups of four; the cost of a level is measured by converting that layer alone and reading the
# log-perplexity increase on 8 held-out C4 windows, as the windows are and with the beginning-of-sequence token prepended (the
# two forms differ by the sink-start effect, NOTE.md and harness_conventions.md); an exact dynamic programme picks, for each
# form, the allocation of least summed cost at the head count of uniform pairs. Arms: uniform pairs, the allocation from the
# token-less costs (dp_as_is), the allocation from the with-token costs (dp_with_token), three random allocations at the same
# head count (seeds 0, 1, 2); per arm, perplexity on the 8 registered WikiText-2 windows as they are and with the token, on
# the 8 measurement windows and on 2 further held-out C4 windows (both forms), and the registered needle protocol without the
# token for the arms named in --needle-arms. The prediction stands in the plan (private/drafts/PLAN_kvfusion_gaps.md, g22)
# before the run. A per-layer head count cannot be exported to a standard Hugging Face configuration (one
# num_key_value_heads); the result is for runtimes that accept per-layer counts. Writes alignment_study/alloc7b_dev/<model>.json.
#   python alignment_study/alloc7b_dev.py --model=mistralai/Mistral-7B-v0.1 [--cost-windows=8] [--cost-len=1024] [--seeds=0,1,2]
#                                         [--needle-arms=uniform_pairs,dp_as_is,dp_with_token] [--tag=NAME]
import gc
import json
import math
import os
import sys
import time
import numpy as np
import torch

argv0 = list(sys.argv)
sys.path.insert(0, "alignment_study")
arg = lambda n, d=None: next((a.split("=", 1)[1] for a in argv0 if a.startswith(f"--{n}=")), d)
NAME = arg("model")
if not NAME:
    sys.exit("--model=<name>")
sys.argv = [argv0[0], f"--model={NAME}"]
import retrieval_protect_dev as R                     # load (bfloat16, pinned revisions), free, needle_eval, K, C, Z, DEV, ND
sys.argv = argv0
K, C, Z, DEV, ND = R.K, R.C, R.Z, R.DEV, R.ND
import fuse_common as F
import kvfuse
N_CAL, T_CAL, N_EVAL = 128, 2048, 8                   # round 28's calibration and the registered evaluation windows
N_COST, T_COST = int(arg("cost-windows", 8)), int(arg("cost-len", 1024))
N_VAL = 2                                             # further held-out C4 windows for the arms, separate from the measurement windows
SEEDS = [int(x) for x in arg("seeds", "0,1,2").split(",")]
NEEDLE_ARMS = set(arg("needle-arms", "uniform_pairs,dp_as_is,dp_with_token").split(","))
TAG = arg("tag")
COSTS_FROM = arg("costs-from")                        # an earlier artifact of the same model whose per-layer costs are reused (the sweep is skipped)
LEVELS_ARMS = {spec.split(":")[0]: [int(x) for x in spec.split(":")[1].split(",")] for spec in (arg("levels-arms") or "").split(";") if spec}   # NAME:1,2,4,... per layer
ONLY = set(arg("only-arms").split(",")) if arg("only-arms") else None
OUT = f"alignment_study/alloc7b_dev/{NAME.split('/')[-1]}{'_' + TAG if TAG else ''}.json"
torch.set_grad_enabled(False)


def with_token(win, tok):
    """The same windows with the beginning-of-sequence token prepended and the last token dropped (bos_ppl_check's form)."""
    if tok.bos_token_id is None:
        return None
    return torch.cat([torch.full((win.shape[0], 1), tok.bos_token_id, dtype=win.dtype), win[:, :-1]], dim=1)


def as_tensor(w):
    return torch.as_tensor(w) if not torch.is_tensor(w) else w


def convert_levels(kv, S, levels, G):
    """Fuse the layers whose level is 2 (pairs) or 4 (groups of four) with the query- and output-weighted fusion; layers at
    level 1 are untouched (fuse_kv.convert would still factor every layer's moment)."""
    for l, g in enumerate(levels):
        if g == 1:
            continue
        S64 = S[l].double(); roots = F.layer_roots(S64, kv.d); Lsq = roots[0] @ roots[0]
        for grp in G[g][l]:
            K.fuse_kv_group(kv, l, list(grp), S64, roots, qw=True, ow=True, Lsq=Lsq)
        del S64, roots, Lsq


def both(model, win, win_bos):
    r = {"as_is": Z.perplexity(model, win, DEV, bs=2)}
    r["with_token"] = Z.perplexity(model, win_bos, DEV, bs=2) if win_bos is not None else None
    return r


def allocate(cost, L, H, B):
    """Exact dynamic programme: the levels per layer that minimise the summed single-layer cost at a total of at most B heads."""
    best = {0: (0.0, [])}
    for l in range(L):
        nb = {}
        for h, (c, lv) in best.items():
            for g, cg in ((1, 0.0), (2, cost[(l, 2)]), (4, cost[(l, 4)])):
                hh = h + H // g
                if hh <= B and (hh not in nb or c + cg < nb[hh][0]):
                    nb[hh] = (c + cg, lv + [g])
        best = nb
    return min(best.values(), key=lambda t: t[0])[1]


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t0 = time.time(); log = lambda s: print(s, f"({time.time() - t0:.0f}s)", flush=True)
    tok = R.AutoTokenizer.from_pretrained(NAME, revision=R.REV)
    ND.FACTS, ND.DEPTHS = R.REG_FACTS, R.REG_DEPTHS
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    held = as_tensor(C.windows_c4(tok, N_CAL + 1 + N_COST + N_VAL, T_CAL)[-(N_COST + N_VAL):])   # documents after the calibration and check documents
    cost_win, val_win = held[:N_COST, :T_COST].contiguous(), held[N_COST:].contiguous()
    cost_bos, val_bos = with_token(cost_win, tok), with_token(val_win, tok)
    win = as_tensor(Z.wikitext2_windows(tok, T=2048)[:N_EVAL]); win_bos = with_token(win, tok)
    context_ids = tok.encode(ND.haystack_text(3 * ND.CTX), add_special_tokens=False)
    model = R.load(); kv = K.KV(model); L, H = kv.L, kv.nkv
    S, ntok = K.second_moments(model, kv, cal, DEV)
    G = {2: [[[int(x) for x in p] for p in kvfuse.min_cost_pairs(kvfuse.pair_costs(kv, l, S[l].double()))] for l in range(L)],
         4: [[[int(x) for x in g] for g in kvfuse.quad_groups(kv, l, S[l].double())] for l in range(L)]}
    base = {"cost_windows": both(model, cost_win, cost_bos), "val_windows": both(model, val_win, val_bos), "wikitext": both(model, win, win_bos)}
    log(f"{NAME}: {L} layers, {H} key-value heads; unconverted perplexity wikitext {base['wikitext']['as_is']:.3f} as is, {base['wikitext']['with_token']:.3f} with the token; "
        f"cost windows ({N_COST} x {T_COST}) {base['cost_windows']['as_is']:.3f} and {base['cost_windows']['with_token']:.3f}")
    res = {"model": NAME, "revision": R.REV, "exploratory": True, "calibration": [N_CAL, T_CAL], "cost_windows": [N_COST, T_COST], "val_windows": N_VAL,
           "eval_windows": N_EVAL, "layers": L, "kv_heads": H, "budget_heads": L * H // 2, "groups": {str(g): G[g] for g in G}, "base": base, "per_layer_cost": {}, "arms": {}}
    forms = ["as_is"] + (["with_token"] if cost_bos is not None else [])
    cost = {f: {} for f in forms}
    if COSTS_FROM:
        prev = json.load(open(COSTS_FROM)); assert prev["model"] == NAME and prev["cost_windows"] == [N_COST, T_COST], "costs-from must be the same model and cost windows"
        for key, v in prev["per_layer_cost"].items():
            l, g = (int(x) for x in key.split("_"))
            for f in forms:
                cost[f][(l, g)] = v[f]
        res["per_layer_cost"] = prev["per_layer_cost"]; res["costs_from"] = COSTS_FROM
        log(f"{NAME}: per-layer costs reused from {COSTS_FROM}")
    for l in (range(L) if not COSTS_FROM else []):
        snap = [p.detach().clone() for p in kv.params(l)]
        line = []
        for g in (2, 4):
            convert_levels(kv, S, [g if i == l else 1 for i in range(L)], G)
            r = both(model, cost_win, cost_bos)
            for f in forms:
                cost[f][(l, g)] = math.log(r[f]) - math.log(base["cost_windows"][f])
            res["per_layer_cost"][f"{l}_{g}"] = {f: cost[f][(l, g)] for f in forms}
            for p, t in zip(kv.params(l), snap):
                p.copy_(t)
            line.append(f"{'pairs' if g == 2 else 'quads'} " + ", ".join(f"{cost[f][(l, g)]:.4f} {f.replace('_', ' ')}" for f in forms))
        del snap
        json.dump(res, open(OUT, "w"), indent=1)
        log(f"{NAME}, layer {l}: " + "; ".join(line) + " (log-perplexity increase, held-out C4)")
    B = L * H // 2; heads = lambda lv: sum(H // g for g in lv)
    arms = {"uniform_pairs": [2] * L, "dp_as_is": allocate(cost["as_is"], L, H, B)}
    if "with_token" in forms:
        arms["dp_with_token"] = allocate(cost["with_token"], L, H, B)
    for sd in SEEDS:
        r_ = np.random.default_rng(sd)
        while True:
            lv = [int(r_.choice([1, 2, 4])) for _ in range(L)]
            if heads(lv) == B:
                break
        arms[f"random_seed{sd}"] = lv
    for nm, lv in LEVELS_ARMS.items():                   # given allocations (controls), e.g. layer 0 whole and the two cheapest layers quartered
        assert len(lv) == L and heads(lv) == B, f"{nm}: {len(lv)} levels, {heads(lv)} heads (budget {B})"
        arms[nm] = lv
    if ONLY:
        arms = {k: v for k, v in arms.items() if k in ONLY}
    res["allocations"] = {k: v for k, v in arms.items()}
    for nm, lv in arms.items():
        log(f"{NAME}, {nm}: levels {lv} ({heads(lv)} heads; predicted " + ", ".join(f"{sum(cost[f][(l, g)] for l, g in enumerate(lv) if g > 1):.3f} {f.replace('_', ' ')}" for f in forms) + ")")
    first = True
    for nm, lv in arms.items():
        t1 = time.time()
        if not first:
            del model, kv; R.free(); model = R.load(); kv = K.KV(model)
        first = False
        convert_levels(kv, S, lv, G)
        a = {"levels": lv, "kv_heads_kept": heads(lv), "cache_fraction": heads(lv) / (L * H),
             "predicted_cost": {f: sum(cost[f][(l, g)] for l, g in enumerate(lv) if g > 1) for f in forms},
             "wikitext": both(model, win, win_bos), "cost_windows": both(model, cost_win, cost_bos), "val_windows": both(model, val_win, val_bos)}
        a["actual_cost"] = {f: math.log(a["cost_windows"][f]) - math.log(base["cost_windows"][f]) for f in forms}
        if nm in NEEDLE_ARMS:
            a["needle"] = R.needle_eval(model, tok, context_ids)
        a["seconds"] = round(time.time() - t1, 1); res["arms"][nm] = a
        json.dump(res, open(OUT, "w"), indent=1)
        log(f"{NAME}, {nm}: {a['kv_heads_kept']} heads (cache {a['cache_fraction']:.3f}); wikitext {a['wikitext']['as_is']:.3f} as is"
            + (f", {a['wikitext']['with_token']:.3f} with the token" if a["wikitext"]["with_token"] else "")
            + f"; held-out C4 {a['val_windows']['as_is']:.3f}" + (f" and {a['val_windows']['with_token']:.3f}" if a["val_windows"]["with_token"] else "")
            + "; measurement windows actual " + ", ".join(f"{a['actual_cost'][f]:.3f} {f.replace('_', ' ')}" for f in forms)
            + (f"; needle {a['needle']['hits']} of {a['needle']['of']} {a['needle']['by_depth']}" if "needle" in a else ""))
    res["seconds"] = round(time.time() - t0, 1); json.dump(res, open(OUT, "w"), indent=1)
    log("ALLOC7B_DONE")
