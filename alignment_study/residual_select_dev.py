# Development (exploratory, labelled): which local statistic selects the pairing of one layer? Round 29's
# forced-pair populations (residual_damage_dev.py) move perplexity by less than its noise in layers 1 to 4 and, in
# layer 0, by amounts the fusion's residual does not order, and on Qwen2.5-7B the residual's matching gives 70.1
# against the neighbours' 23.9. This script measures, one layer at a time with every other layer unconverted, the
# perfect matchings of that layer's key-value heads (all 105 for 8 heads, all 3 for 4; above 8, the residual's
# matching, the neighbouring one, single swaps of the residual's matching and random matchings), each fused in
# the paper's closed form: (a) the matching's summed residual (kvfuse.pair_costs, no forward pass), (b) the fused
# layer's output perturbation on one held-out C4 window (one layer forward; mean over tokens of the output change
# over the output norm), and the ground truth, (c) KL(unconverted || converted) and the change in NLL on held-out
# WikiText-2 windows through the whole model. Per layer: the Spearman of (a) and (b) with KL, the KL rank of the
# residual's choice and of the perturbation's choice among the candidates, and the KL of the neighbouring
# pairing. The layer's weights are restored after every candidate. Writes
# alignment_study/residual_select_dev/<model>.json.
#   python alignment_study/residual_select_dev.py --model=<name> [--layers=0,1,2] [--nrand=100] [--nswap=50] [--windows=2]
import gc
import itertools
import json
import os
import random
import sys
import time
import numpy as np
import torch
from scipy.stats import spearmanr

argv0 = list(sys.argv)
sys.path.insert(0, "alignment_study")
sys.argv = [argv0[0], "--dev=none"]
import quad_check_round as QC                          # REVISIONS, load (bfloat16 above 4B), Stop
sys.argv = argv0
import gqa_common as C
import fuse_common as F
import fuse_kv as K
import functional_common as FC
import kvfuse
import zeroshot as Z
from scale_round_v2 import DEV


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME = arg("model")
if not NAME:
    sys.exit("--model=<name>")
LAYERS = [int(x) for x in arg("layers", "0,1,2").split(",")]
N_RAND, N_SWAP, N_WIN = int(arg("nrand", 100)), int(arg("nswap", 50)), int(arg("windows", 2))
N_CAL, T_CAL, SEED, ALL_UP_TO = 128, 2048, 0, 8
OUT = "alignment_study/residual_select_dev"
torch.set_grad_enabled(False)


def matchings(items):
    """Every perfect matching of the list, each as sorted pairs."""
    if not items:
        yield []
        return
    a, rest = items[0], items[1:]
    for i, b in enumerate(rest):
        for m in matchings(rest[:i] + rest[i + 1:]):
            yield [[a, b]] + m


def canon(M):
    return sorted(sorted(p) for p in M)


def neighbours(n):
    return canon([[i, i + 1] for i in range(0, n, 2)])


def random_matching(n, rng):
    p = list(range(n)); rng.shuffle(p)
    return canon([p[i:i + 2] for i in range(0, n, 2)])


def swaps(M):
    """Every matching that differs from M by exchanging one head between two of its pairs."""
    out = []
    for i, j in itertools.combinations(range(len(M)), 2):
        (a, b), (c, d) = M[i], M[j]
        rest = [p for k, p in enumerate(M) if k not in (i, j)]
        out.append(canon(rest + [[a, c], [b, d]])); out.append(canon(rest + [[a, d], [b, c]]))
    return out


def candidates(n, M0, rng):
    if n <= ALL_UP_TO:
        return [canon(m) for m in matchings(list(range(n)))]
    seen, out = set(), []
    def add(M):
        k = json.dumps(M)
        if k not in seen:
            seen.add(k); out.append(M)
    add(M0); add(neighbours(n))
    sw = swaps(M0); rng.shuffle(sw)
    for M in sw[:N_SWAP]:
        add(M)
    target = len(out) + N_RAND
    while len(out) < target:
        add(random_matching(n, rng))
    return out


def rank_of(values, i):
    """1-based rank of entry i in ascending order (1 = smallest)."""
    return 1 + sum(v < values[i] for v in values)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    rng = random.Random(SEED)
    tok = QC.AutoTokenizer.from_pretrained(NAME, revision=QC.REVISIONS.get(NAME))
    model = QC.load(NAME); kv = K.KV(model); n = kv.nkv
    assert n % 2 == 0, f"{NAME}: {n} KV heads"
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    gw = C.windows_c4(tok, N_CAL + 1, T_CAL)[-1:].to(DEV)
    X = Z.wikitext2_windows(tok, T=2048)[:N_WIN]
    S, ntok = K.second_moments(model, kv, cal, DEV)
    base = FC.base_cache(model, X)
    log(f"{NAME}: {kv.L} layers, {n} KV heads, {str(next(model.parameters()).dtype)}; moments from {ntok} tokens; {N_WIN} held-out windows")
    res = {"model": NAME, "revision": QC.REVISIONS.get(NAME), "exploratory": True, "kv_heads": n, "layers": LAYERS, "windows": N_WIN,
           "calibration": [N_CAL, T_CAL], "all_matchings_up_to": ALL_UP_TO, "n_random": N_RAND, "n_swaps": N_SWAP, "seed": SEED, "per_layer": {}}
    for l in LAYERS:
        t1 = time.time()
        S64 = S[l].double(); cost = kvfuse.pair_costs(kv, l, S64); M0 = canon(kvfuse.min_cost_pairs(cost))
        roots = F.layer_roots(S64, kv.d); Lsq = roots[0] @ roots[0]
        cap = {}
        def hook(m, args, kwargs, out):
            cap["args"], cap["kwargs"] = args, kwargs
            cap["out"] = (out[0] if isinstance(out, tuple) else out).clone()
            raise QC.Stop
        h = kv.blocks[l].register_forward_hook(hook, with_kwargs=True)
        try:
            model.model(input_ids=gw, use_cache=False)
        except QC.Stop:
            pass
        h.remove()
        ref = cap["out"][0, 1:].float(); ref_norm = ref.norm(dim=-1)
        orig = [p.detach().cpu().clone() for p in kv.params(l)]
        def put():
            for p, t in zip(kv.params(l), orig):
                p.copy_(t.to(p.device))
        cands = candidates(n, M0, rng); rows = []
        for M in cands:
            put()
            for G in M:
                K.fuse_kv_group(kv, l, list(G), S64, roots, qw=True, ow=True, Lsq=Lsq)
            out = kv.blocks[l](*cap["args"], **cap["kwargs"]); out = out[0] if isinstance(out, tuple) else out
            pert = float(((out[0, 1:].float() - ref).norm(dim=-1) / ref_norm).mean())
            kls, dn = FC.kl_eval(model, X, base)
            rows.append({"matching": M, "residual": float(sum(cost[a, b] for a, b in M)), "perturbation": pert, "kl": float(np.mean(kls)),
                         "dnll": float(np.mean(dn)), "residual_choice": M == M0, "neighbours": M == neighbours(n)})
        put(); del cap, orig
        kl = [r["kl"] for r in rows]; rs = [r["residual"] for r in rows]; pt = [r["perturbation"] for r in rows]; dn = [r["dnll"] for r in rows]
        i0 = next(i for i, r in enumerate(rows) if r["residual_choice"]); ip = int(np.argmin(pt)); ib = int(np.argmin(kl))
        inb = next((i for i, r in enumerate(rows) if r["neighbours"]), None)
        st = {"candidates": len(rows), "spearman_residual_kl": float(spearmanr(rs, kl)[0]), "spearman_perturbation_kl": float(spearmanr(pt, kl)[0]),
              "spearman_residual_perturbation": float(spearmanr(rs, pt)[0]), "spearman_residual_dnll": float(spearmanr(rs, dn)[0]),
              "spearman_perturbation_dnll": float(spearmanr(pt, dn)[0]), "spearman_kl_dnll": float(spearmanr(kl, dn)[0]),
              "kl_rank_residual_choice": rank_of(kl, i0), "kl_rank_perturbation_choice": rank_of(kl, ip),
              "kl_residual_choice": kl[i0], "kl_perturbation_choice": kl[ip], "kl_best": kl[ib], "kl_median": float(np.median(kl)), "kl_worst": max(kl),
              "kl_neighbours": kl[inb] if inb is not None else None, "best_matching": rows[ib]["matching"], "residual_matching": M0,
              "perturbation_matching": rows[ip]["matching"], "seconds": round(time.time() - t1, 1)}
        res["per_layer"][str(l)] = {"stats": st, "rows": rows}
        json.dump(res, open(f"{OUT}/{NAME.split('/')[-1]}.json", "w"), indent=1)
        log(f"layer {l}: {len(rows)} matchings; Spearman with KL: residual {st['spearman_residual_kl']:.2f}, perturbation {st['spearman_perturbation_kl']:.2f}; "
            f"KL rank of the residual's choice {st['kl_rank_residual_choice']} of {len(rows)}, of the perturbation's {st['kl_rank_perturbation_choice']}; "
            f"KL residual's {kl[i0]:.4f}, perturbation's {kl[ip]:.4f}, neighbours {kl[inb] if inb is not None else float('nan'):.4f}, best {kl[ib]:.4f}, median {st['kl_median']:.4f}")
        gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(f"{OUT}/{NAME.split('/')[-1]}.json", "w"), indent=1)
    log("RESIDUAL_SELECT_DONE")
