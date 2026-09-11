# Evaluation of PREREGISTRATION7 (head-merging round), committed before any
# result exists. Reads merge/<model>.json, writes merge_results.json and
# prints the scorecard M1, M1b, M2, M2b, M3.
import glob
import json
import numpy as np
from scipy.stats import spearmanr

D = "alignment_study/merge"
MODELS = ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]


def spear(x, y):
    return float(spearmanr(x, y)[0])


def boot_layers(x, y, B=2000, seed=0):
    rg = np.random.default_rng(seed); x, y = np.asarray(x), np.asarray(y)
    v = [spear(x[i], y[i]) for i in (rg.integers(0, len(x), len(x)) for _ in range(B))]
    return [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)]


def partial(x, y, z):
    x, y, z = map(np.asarray, (x, y, z))
    rx = x - np.polyval(np.polyfit(z, x, 1), z); ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return spear(rx, ry)


models = {}
for p in sorted(glob.glob(f"{D}/*.json")):
    c = json.load(open(p))
    if isinstance(c, dict) and "rows" in c:
        models[c["model"]] = c
if not models:
    raise SystemExit("no merge results yet")

res = {"n_models": len(models), "complete": all(m in models for m in MODELS), "per_model": {}, "M": {}}
pool_se, pool_dl, pool_layer = [], [], []
for name, c in models.items():
    rows = c["rows"]; L = len(rows)
    se = [r["sharedE"] for r in rows]; dl = [r["dloss_top"] for r in rows]
    pm = {"n_layers": L, "baseline_nll": c["baseline_nll"], "restore_check_abs_diff": c["restore_check_abs_diff"],
          "frac_top_below_bottom": round(float(np.mean([r["dloss_top"] < r["dloss_bottom"] for r in rows])), 3),
          "frac_top_below_random": round(float(np.mean([r["dloss_top"] < r["dloss_random"] for r in rows])), 3),
          "spearman_sharedE_dloss_top": round(spear(se, dl), 3),
          "spearman_sharedE_dloss_top_depth_partialled": round(partial(se, dl, list(range(L))), 3),
          "median_dloss": {k: round(float(np.median([r[f"dloss_{k}"] for r in rows])), 5) for k in ("top", "random", "bottom")},
          "frac_top_cost_below_0.002": round(float(np.mean([r["dloss_top"] < 0.002 for r in rows])), 3),
          "median_top_cos": round(float(np.median([r["top_cos"] for r in rows])), 3)}
    # paired bootstrap over evaluation windows for the median top-merge cost
    W = np.array([r["dloss_top_windows"] for r in rows])          # (L, n_eval)
    rg = np.random.default_rng(1); meds = []
    for _ in range(2000):
        idx = rg.integers(0, W.shape[1], W.shape[1]); meds.append(float(np.median(W[:, idx].mean(1))))
    pm["median_dloss_top_ci95"] = [round(float(np.percentile(meds, 2.5)), 5), round(float(np.percentile(meds, 97.5)), 5)]
    pm["M1_pass"] = pm["frac_top_below_bottom"] >= 2 / 3
    pm["M1b_pass"] = pm["frac_top_below_random"] > 0.5
    pm["M2_direction_negative"] = pm["spearman_sharedE_dloss_top"] < 0
    if c.get("pair_sweep"):
        sw = c["pair_sweep"]
        per_layer = {}
        for l in sorted({s["layer"] for s in sw}):
            xs = [s["cos"] for s in sw if s["layer"] == l]; ys = [s["dloss"] for s in sw if s["layer"] == l]
            per_layer[l] = round(spear(xs, ys), 3)
        pooled = spear([s["cos"] for s in sw], [s["dloss"] for s in sw])
        pm["M3_pair_sweep"] = {"n_pairs": len(sw), "spearman_pooled": round(pooled, 3), "spearman_per_layer": per_layer,
                               "pass": pooled <= -0.5}
    res["per_model"][name] = pm
    pool_se += se; pool_dl += dl; pool_layer += list(range(L))

tested = [res["per_model"][m] for m in MODELS if m in res["per_model"]]
res["M"]["M1"] = {"passes": sum(pm["M1_pass"] for pm in tested), "n": len(tested), "pass": bool(tested) and all(pm["M1_pass"] for pm in tested)}
res["M"]["M1b"] = {"passes": sum(pm["M1b_pass"] for pm in tested), "n": len(tested), "pass": bool(tested) and all(pm["M1b_pass"] for pm in tested), "secondary": True}
rho = spear(pool_se, pool_dl); ci = boot_layers(pool_se, pool_dl)
res["M"]["M2"] = {"spearman_pooled": round(rho, 3), "ci95": ci, "n_layers": len(pool_se),
                  "per_model_negative": sum(pm["M2_direction_negative"] for pm in tested), "n": len(tested),
                  "pass": rho <= -0.3 and ci[1] < 0 and all(pm["M2_direction_negative"] for pm in tested)}
res["M"]["M2b"] = {"spearman_pooled_depth_partialled": round(partial(pool_se, pool_dl, pool_layer), 3), "secondary": True,
                   "pass": partial(pool_se, pool_dl, pool_layer) <= -0.3}
m3 = res["per_model"].get("Qwen/Qwen2.5-1.5B", {}).get("M3_pair_sweep")
res["M"]["M3"] = dict(m3, secondary=True) if m3 else {"pass": None, "secondary": True}
json.dump(res, open("alignment_study/merge_results.json", "w"), indent=1)
print("models:", len(models), "complete:", res["complete"])
for k, v in res["M"].items():
    print(k, json.dumps(v))
for name, pm in res["per_model"].items():
    print("  ", name.split("/")[-1], json.dumps({k: v for k, v in pm.items() if k != "M3_pair_sweep"}))
