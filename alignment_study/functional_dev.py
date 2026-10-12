# Development calibration for the functional round (LABELED, not registered).
# Qwen2.5-0.5B, the calibration model of rounds 6 and 7, excluded from every
# confirmatory set. Measures F1 and F2 on all layers and the merge outcome
# (KL and change in NLL) for every within-group pair in eight layers, to
# size the thresholds of the preregistration before it is frozen.
#   python alignment_study/functional_dev.py [model] [out]
import json
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
import functional_common as F
import merge_dev_calibration as M
from scale_round_v2 import DEV
from hubsfree.adapters import pick_dtype, release_memory, device_label
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr

NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
OUT = sys.argv[2] if len(sys.argv) > 2 else "alignment_study/functional_dev_calibration.json"
MERGE_LAYERS = [1, 4, 7, 10, 13, 16, 19, 22]


def as_distance(p, v):
    return -v if p == "gen_cos" else v          # every predictor oriented so that larger means less similar


def analyze(meas, sw, layers):
    per, out = {}, {}
    for l in layers:
        keys = list(sw[l])
        kl = np.array([sw[l][k]["kl"] for k in keys]); dn = np.array([sw[l][k]["dnll"] for k in keys])
        kw = np.array([sw[l][k]["kl_windows"] for k in keys])
        half = float(spearmanr(kw[:, :4].mean(1), kw[:, 4:].mean(1))[0])
        row = {"n_pairs": len(keys), "kl_median": float(np.median(kl)), "kl_min": float(kl.min()), "kl_max": float(kl.max()),
               "kl_split_half_spearman": half, "spearman_kl_dnll": float(spearmanr(kl, dn)[0]), "rho_kl": {}, "rho_dnll": {}, "pick": {}}
        for p in F.PREDICTORS:
            d = as_distance(p, np.array([meas[l]["pairs"][k][p] for k in keys]))
            row["rho_kl"][p] = float(spearmanr(d, kl)[0]); row["rho_dnll"][p] = float(spearmanr(d, dn)[0])
            pick = int(np.argmin(d))
            row["pick"][p] = {"kl": float(kl[pick]), "kl_over_best": float(kl[pick] / kl.min()), "rank_pct": float((kl < kl[pick]).mean() * 100)}
        row["random_pick_kl_mean"] = float(kl.mean())
        per[l] = row
    rg = np.random.default_rng(0)
    for p in F.PREDICTORS:
        r = np.array([per[l]["rho_kl"][p] for l in layers])
        boots = [r[rg.integers(0, len(r), len(r))].mean() for _ in range(2000)]
        out[p] = {"mean_within_layer_rho_kl": float(r.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                  "mean_within_layer_rho_dnll": float(np.mean([per[l]["rho_dnll"][p] for l in layers])),
                  "median_pick_kl_over_best": float(np.median([per[l]["pick"][p]["kl_over_best"] for l in layers])),
                  "median_pick_rank_pct": float(np.median([per[l]["pick"][p]["rank_pct"] for l in layers]))}
    diffs = {}
    for a, b in (("tv_nosink", "tv_raw"), ("val_dist", "tv_raw"), ("tv_nosink", "gen_cos"), ("val_dist", "gen_cos"), ("out_pert", "val_dist")):
        d = np.array([per[l]["rho_kl"][a] - per[l]["rho_kl"][b] for l in layers])
        boots = [d[rg.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        diffs[f"{a}_minus_{b}"] = {"mean": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    return per, out, diffs


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(NAME)
    dt = pick_dtype(NAME, DEV, "auto")
    model = AutoModelForCausalLM.from_pretrained(NAME, attn_implementation="eager", dtype=getattr(torch, dt)).eval().to(DEV)
    L, n, nkv, hd, group = F.geometry(model)
    print(NAME, "DEVICE", device_label(DEV), "DTYPE", dt, f"layers {L} heads {n} kv {nkv} hd {hd} group {group}", flush=True)
    M.N_SEL, M.T_SEL, M.N_EVAL, M.T_EVAL = F.N_SEL, F.T_SEL, F.N_EVAL, F.T_EVAL
    sel = M.selection_windows(tok); X = M.eval_windows(tok)
    meas = F.selection_measures(model, sel, list(range(L)), n, hd, group)
    print(f"selection measures done ({time.time()-t0:.0f}s)", flush=True)
    base = F.base_cache(model, X)
    pairs = {l: [tuple(map(int, k.split("-"))) for k in meas[l]["pairs"]] for l in MERGE_LAYERS}
    sw = F.sweep(model, X, base, MERGE_LAYERS, pairs, hd, log=lambda s: print(s, f"({time.time()-t0:.0f}s)", flush=True))
    check_kl, _ = F.kl_eval(model, X, base)
    per, summ, diffs = analyze(meas, sw, MERGE_LAYERS)
    f1_layers = [{"layer": l, "raw": float(np.mean([h["raw"] for h in meas[l]["f1"]])), "val": float(np.mean([h["val"] for h in meas[l]["f1"]])),
                  "out": float(np.mean([h["out"] for h in meas[l]["f1"]])), "fnorm_sink_over_rest": float(np.nanmedian([h["fnorm_sink_over_rest"] for h in meas[l]["f1"]])),
                  "f2_raw_reproduced": meas[l]["f2"]["raw"]["reproduced"], "f2_val_reproduced": meas[l]["f2"]["val"]["reproduced"],
                  "f2_raw_real": meas[l]["f2"]["raw"]["real"], "f2_val_real": meas[l]["f2"]["val"]["real"], "sink_sets": meas[l]["sink_sets"]} for l in range(L)]
    hs = [r for r in f1_layers if r["raw"] > 0.4]
    res = {"model": NAME, "dtype": dt, "label": "development calibration, not registered", "layers": L, "heads": n, "kv_heads": nkv,
           "merge_layers": MERGE_LAYERS, "eval_tokens": int(X.numel()), "restore_check_kl": float(np.mean(check_kl)),
           "f1_layers": f1_layers,
           "f1_high_sink": {"n": len(hs), "median_raw": float(np.median([r["raw"] for r in hs])) if hs else None,
                            "median_val": float(np.median([r["val"] for r in hs])) if hs else None,
                            "median_val_over_raw": float(np.median([r["val"] / r["raw"] for r in hs])) if hs else None,
                            "median_out": float(np.median([r["out"] for r in hs])) if hs else None,
                            "median_fnorm_sink_over_rest": float(np.median([r["fnorm_sink_over_rest"] for r in hs])) if hs else None,
                            "median_f2_raw_reproduced": float(np.median([r["f2_raw_reproduced"] for r in hs])) if hs else None,
                            "median_f2_val_reproduced": float(np.median([r["f2_val_reproduced"] for r in hs])) if hs else None},
           "f3_per_layer": {str(l): per[l] for l in MERGE_LAYERS}, "f3_summary": summ, "f3_differences": diffs,
           "sweep": {str(l): sw[l] for l in MERGE_LAYERS}, "selection_pairs": {str(l): meas[l]["pairs"] for l in range(L)},
           "runtime_s": round(time.time() - t0, 1)}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({"f1_high_sink": res["f1_high_sink"], "f3_summary": summ, "f3_differences": diffs, "restore_check_kl": res["restore_check_kl"]}, indent=1))
    print("FUNCTIONAL_DEV_DONE", OUT, flush=True)
    del model; release_memory(DEV)


if __name__ == "__main__":
    main()
