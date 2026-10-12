# Evaluation of PREREGISTRATION13 (what the sink does), committed before any
# confirmatory result exists. Reads alignment_study/sink/<model>.json (or the
# directory given), writes <dir>_results.json, prints the scorecard.
#   python alignment_study/eval_sink.py [directory]
import glob
import json
import sys
import numpy as np

D = (sys.argv[1] if len(sys.argv) > 1 else "alignment_study/sink").rstrip("/")
OUTFILE = D + "_results.json"
HS = 0.3                     # high-sink layer: first-token attention mass above 0.3 (rows >= 16, window 0)
B, SEED = 2000, 0
TH = {"K1_ratio": 5.0, "K2_ratio": 0.5, "K3_frac": 2 / 3, "K4_median": 0.5, "K4_ci_high": 0.75, "models_needed": 3}


def cluster_ci(groups, stat=np.median):
    groups = [np.asarray(g, float) for g in groups if len(g)]
    rg = np.random.default_rng(SEED); vals = []
    for _ in range(B):
        pick = rg.integers(0, len(groups), len(groups))
        vals.append(stat(np.concatenate([groups[k][rg.integers(0, len(groups[k]), len(groups[k]))] for k in pick])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


models = {}
for p in sorted(glob.glob(f"{D}/*.json")):
    c = json.load(open(p))
    if isinstance(c, dict) and "per_layer" in c:
        models[c["model"]] = c
res = {"thresholds": TH, "per_model": {}, "K": {}}
ratio_groups = []
for name, c in models.items():
    a = c["all_layers"]; m = c["first_token_mass_by_layer"]
    hs = [l for l in map(int, c["per_layer"]) if m[l] > HS]
    pl = c["per_layer"]
    z_gt_r = [pl[str(l)]["zero"]["kl"] > pl[str(l)]["zero_rand"]["kl"] for l in hs]
    mz = [pl[str(l)]["mean"]["kl"] / pl[str(l)]["zero"]["kl"] for l in hs if pl[str(l)]["zero"]["kl"] > 0]
    ratio_groups.append(mz)
    res["per_model"][name] = {"equivalence_max_abs_logit": c["equivalence_max_abs_logit"], "restore_check_kl": c["restore_check_kl"],
                              "all_layers_kl": {k: v["kl"] for k, v in a.items()},
                              "all_zero_over_rand": a["zero"]["kl"] / a["zero_rand"]["kl"], "all_mean_over_zero": a["mean"]["kl"] / a["zero"]["kl"],
                              "all_renorm_over_zero": a["renorm"]["kl"] / a["zero"]["kl"],
                              "n_layers_tested": len(pl), "n_high_sink": len(hs), "frac_zero_above_rand": float(np.mean(z_gt_r)) if hs else None,
                              "median_mean_over_zero_high_sink": float(np.median(mz)) if mz else None,
                              "max_renorm_kl_single_layer": max(v["renorm"]["kl"] for v in pl.values())}
pm = res["per_model"]; need = TH["models_needed"]
res["K"]["K1_all_layer_zero_over_random_token"] = {"per_model": {k: v["all_zero_over_rand"] for k, v in pm.items()},
                                                   "pass_": sum(v["all_zero_over_rand"] >= TH["K1_ratio"] for v in pm.values()) >= need}
res["K"]["K2_all_layer_mean_over_zero"] = {"per_model": {k: v["all_mean_over_zero"] for k, v in pm.items()},
                                           "pass_": sum(v["all_mean_over_zero"] <= TH["K2_ratio"] for v in pm.values()) >= need}
res["K"]["K3_zero_above_random_token_in_high_sink_layers"] = {"per_model": {k: v["frac_zero_above_rand"] for k, v in pm.items()},
                                                              "pass_": sum((v["frac_zero_above_rand"] or 0) >= TH["K3_frac"] for v in pm.values()) >= need}
allv = np.concatenate([np.asarray(g, float) for g in ratio_groups if len(g)]) if ratio_groups else np.array([])
k4 = {"median": float(np.median(allv)) if allv.size else None, "ci95": cluster_ci(ratio_groups) if allv.size else None, "n_layers": int(allv.size)}
k4["pass_"] = bool(allv.size) and k4["median"] <= TH["K4_median"] and k4["ci95"][1] < TH["K4_ci_high"]
res["K"]["K4_mean_over_zero_high_sink_layers"] = k4
res["K"]["primary_passes"] = int(sum(bool(v.get("pass_")) for k, v in res["K"].items() if isinstance(v, dict)))
json.dump(res, open(OUTFILE, "w"), indent=1)
for k, v in res["K"].items():
    print(k, json.dumps(v)[:300])
for name, v in pm.items():
    print(f"  {name.split('/')[-1]:24s} equiv {v['equivalence_max_abs_logit']:.1e} | all-layer KL " + " ".join(f"{k} {x:.4f}" for k, x in v["all_layers_kl"].items()) +
          f" | zero/rand {v['all_zero_over_rand']:.1f} mean/zero {v['all_mean_over_zero']:.2f} | high-sink {v['n_high_sink']}/{v['n_layers_tested']} zero>rand {v['frac_zero_above_rand']} median mean/zero {v['median_mean_over_zero_high_sink']}")
