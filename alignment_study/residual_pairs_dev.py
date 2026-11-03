# Development (exploratory, labelled): a selector from measured pair damage. residual_select_dev.py found a matching's
# whole-model KL additive over its pairs (least squares on pair indicators over the 105 matchings of an 8-head layer:
# R^2 0.89 to 0.99, leave-one-out Spearman 0.92 to 0.99), and the additive model's own minimum-cost matching ranked
# 1st to 5th of 105 where the fusion's residual ranked 57th to 66th. This script measures each pair's damage directly:
# every pair of one layer's key-value heads fused alone (28 for 8 heads, 6 for 4), the layer otherwise unmerged, KL to
# the unconverted model on the same held-out windows; then the minimum-cost matching on the measured pair KLs. Its
# whole-model KL is read from residual_select_dev's enumerated matchings (identical conversions: same calibration,
# same windows), which gives its rank without a further run, and the measured pair KLs are compared with the additive
# fit's pair effects. Writes alignment_study/residual_select_dev/<model>_pairs.json.
# --measure=c4 measures the pairs on two held-out C4 validation windows (after the calibration and check documents)
# instead of the WikiText windows that rank the matchings, so the selector is scored out of sample; output <model>_pairs_c4.json.
#   python alignment_study/residual_pairs_dev.py --model=<name> [--layers=0,1,2,16] [--windows=2] [--measure=c4]
import gc
import itertools
import json
import os
import sys
import time
import numpy as np
import torch
from scipy.stats import spearmanr

argv0 = list(sys.argv)
sys.path.insert(0, "alignment_study")
sys.argv = [argv0[0], "--model=none"]
import residual_select_dev as RS                      # QC (revisions, load, Stop), canon, the same instruments
sys.argv = argv0
QC, C, F, K, FC, kvfuse, Z, DEV = RS.QC, RS.C, RS.F, RS.K, RS.FC, RS.kvfuse, RS.Z, RS.DEV


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


NAME = arg("model")
if not NAME:
    sys.exit("--model=<name>")
LAYERS = [int(x) for x in arg("layers", "0,1,2,16").split(",")]
N_WIN, N_CAL, T_CAL = int(arg("windows", 2)), 128, 2048
MEASURE = arg("measure", "wikitext")
SRC = f"alignment_study/residual_select_dev/{NAME.split('/')[-1]}.json"
OUT = f"alignment_study/residual_select_dev/{NAME.split('/')[-1]}_pairs{'_c4' if MEASURE == 'c4' else ''}{'_w' + str(N_WIN) if N_WIN != 2 else ''}.json"
torch.set_grad_enabled(False)


def additive_effects(rows, n):
    pairs = list(itertools.combinations(range(n), 2)); idx = {p: i for i, p in enumerate(pairs)}
    X = np.zeros((len(rows), len(pairs))); y = np.array([x["kl"] for x in rows])
    for i, x in enumerate(rows):
        for p in x["matching"]:
            X[i, idx[tuple(sorted(p))]] = 1
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ w) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {p: float(w[i]) for p, i in idx.items()}, float(r2)


if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    src = json.load(open(SRC)) if os.path.exists(SRC) else None
    tok = QC.AutoTokenizer.from_pretrained(NAME, revision=QC.REVISIONS.get(NAME))
    model = QC.load(NAME); kv = K.KV(model); n = kv.nkv
    cal = C.windows_c4(tok, N_CAL, T_CAL)
    X = C.windows_c4(tok, N_CAL + 1 + N_WIN, T_CAL)[-N_WIN:] if MEASURE == "c4" else Z.wikitext2_windows(tok, T=2048)[:N_WIN]   # c4: documents after the calibration and the check window
    S, ntok = K.second_moments(model, kv, cal, DEV)
    base = FC.base_cache(model, X)
    log(f"{NAME}: {kv.L} layers, {n} KV heads; moments from {ntok} tokens; {N_WIN} held-out windows; enumerated matchings from {SRC if src else 'none'}")
    res = {"model": NAME, "revision": QC.REVISIONS.get(NAME), "exploratory": True, "kv_heads": n, "layers": LAYERS, "windows": N_WIN, "measure": MEASURE, "source": SRC if src else None, "per_layer": {}}
    for l in LAYERS:
        t1 = time.time()
        S64 = S[l].double(); cost = kvfuse.pair_costs(kv, l, S64); roots = F.layer_roots(S64, kv.d); Lsq = roots[0] @ roots[0]
        orig = [p.detach().cpu().clone() for p in kv.params(l)]
        def put():
            for p, t in zip(kv.params(l), orig):
                p.copy_(t.to(p.device))
        pair_kl, pair_kl_w = {}, {}
        for i, j in itertools.combinations(range(n), 2):
            put(); K.fuse_kv_group(kv, l, [i, j], S64, roots, qw=True, ow=True, Lsq=Lsq)
            kls, _ = FC.kl_eval(model, X, base); pair_kl[(i, j)] = float(np.mean(kls)); pair_kl_w[(i, j)] = [float(k) for k in kls]
        put(); del orig
        cm = torch.zeros(n, n, dtype=torch.float64)
        for (i, j), v in pair_kl.items():
            cm[i, j] = cm[j, i] = v
        choice = RS.canon(kvfuse.min_cost_pairs(cm)); res_choice = RS.canon(kvfuse.min_cost_pairs(cost))
        cmx = torch.zeros(n, n, dtype=torch.float64)
        for (i, j), v in pair_kl_w.items():
            cmx[i, j] = cmx[j, i] = max(v)
        choice_max = RS.canon(kvfuse.min_cost_pairs(cmx))                     # a pair's cost as its largest KL over the windows
        entry = {"pair_kl": {f"{i}-{j}": v for (i, j), v in pair_kl.items()}, "pair_kl_windows": {f"{i}-{j}": v for (i, j), v in pair_kl_w.items()}, "measured_choice_max": choice_max,
                 "pair_residual": {f"{i}-{j}": float(cost[i, j]) for i, j in pair_kl},
                 "spearman_pair_kl_vs_residual": float(spearmanr([pair_kl[p] for p in pair_kl], [float(cost[p[0], p[1]]) for p in pair_kl])[0]),
                 "measured_choice": choice, "residual_choice": res_choice, "sum_pair_kl_measured_choice": float(sum(pair_kl[tuple(p)] for p in choice)), "seconds": round(time.time() - t1, 1)}
        if src and str(l) in src["per_layer"]:
            rows = src["per_layer"][str(l)]["rows"]; eff, r2 = additive_effects(rows, n)
            kl_of = {json.dumps(RS.canon(x["matching"])): x["kl"] for x in rows}; kls = sorted(kl_of.values())
            k = json.dumps(choice); kx = json.dumps(choice_max)
            entry.update({"additive_r2": r2, "spearman_pair_kl_vs_additive_effect": float(spearmanr([pair_kl[p] for p in pair_kl], [eff[p] for p in pair_kl])[0]),
                          "measured_choice_kl": kl_of.get(k), "measured_choice_kl_rank": (1 + sum(v < kl_of[k] for v in kls)) if k in kl_of else None,
                          "measured_max_choice_kl": kl_of.get(kx), "measured_max_choice_kl_rank": (1 + sum(v < kl_of[kx] for v in kls)) if kx in kl_of else None,
                          "candidates": len(rows), "best_kl": kls[0], "median_kl": float(np.median(kls)), "residual_choice_kl_rank": src["per_layer"][str(l)]["stats"]["kl_rank_residual_choice"],
                          "perturbation_choice_kl_rank": src["per_layer"][str(l)]["stats"]["kl_rank_perturbation_choice"]})
        res["per_layer"][str(l)] = entry
        json.dump(res, open(OUT, "w"), indent=1)
        log(f"layer {l}: {len(pair_kl)} pairs measured; Spearman pair KL against residual {entry['spearman_pair_kl_vs_residual']:.2f}"
            + (f", against the additive effect {entry['spearman_pair_kl_vs_additive_effect']:.2f} (R2 {entry['additive_r2']:.3f}); the measured matching's KL rank {entry['measured_choice_kl_rank']} of {entry['candidates']} (KL {entry['measured_choice_kl'] if entry['measured_choice_kl'] is None else round(entry['measured_choice_kl'], 4)}, best {entry['best_kl']:.4f}); by the largest KL over windows rank {entry['measured_max_choice_kl_rank']}; residual's rank {entry['residual_choice_kl_rank']}, perturbation's {entry['perturbation_choice_kl_rank']}" if "additive_r2" in entry else ""))
        gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    log("RESIDUAL_PAIRS_DONE")
