# Evaluation of PREREGISTRATION11 (functional round), committed before any
# confirmatory result exists. Reads alignment_study/functional/<model>.json,
# writes alignment_study/functional_results.json and prints the scorecard.
# Intervals are two-stage cluster bootstraps: models resampled, then layers
# within each model (2,000 draws, seed 0), as in clustered_stats.py.
#   python alignment_study/eval_functional.py [directory]
import glob
import json
import sys
import numpy as np
from scipy.stats import spearmanr

D = sys.argv[1] if len(sys.argv) > 1 else "alignment_study/functional"
OUTFILE = "alignment_study/functional_results.json" if D.rstrip("/") == "alignment_study/functional" else D.rstrip("/") + "_results.json"
MODELS = ["Qwen/Qwen2.5-1.5B", "TinyLlama/TinyLlama_v1.1", "mistralai/Mistral-7B-v0.1", "Qwen/Qwen2.5-7B"]
PREDICTORS = ("gen_cos", "tv_raw", "tv_nosink", "val_dist", "out_pert", "sink_diff")
HS = 0.4                          # high-sink layer: mean raw sink share over heads above 0.4
B, SEED = 2000, 0
# Thresholds frozen with PREREGISTRATION11.md, sized from the labeled
# development calibration (functional_dev_calibration.json, Qwen2.5-0.5B) at
# roughly half its effects.
TH = {"RELIABILITY_MIN": 0.7,       # median split-half Spearman of the within-layer KL ranking (dev 0.95)
      "F1_ratio_max": 0.6,          # value-norm sink share over raw sink share, high-sink layers (dev 0.42)
      "F1_ratio_ci_max": 0.75,
      "F2_drop_min": 0.25,          # drop of the sink-column null's reproduced share, raw to value-weighted (dev about 0.45)
      "F3_delta_min": 0.10,         # out_pert over tv_raw, and tv_raw over gen_cos (dev 0.22 and 0.27)
      "F3c_upper_max": 0.05,        # tv_nosink minus tv_raw: no gain from removing the sink (dev -0.02, interval -0.05 to 0.01)
      "F3d_rho_min": 0.6, "F3d_ci_low_min": 0.4,   # out_pert absolute level (dev 0.83)
      "F4_ratio_max": 0.8}          # KL of the out_pert pick over KL of the gen_cos pick (dev 0.50)


def as_distance(p, v):
    return -np.asarray(v) if p == "gen_cos" else np.asarray(v)


def cluster_ci(groups, stat=np.mean):
    """groups: list over models of 1-d arrays of per-layer values."""
    groups = [np.asarray(g, float) for g in groups if len(g)]
    rg = np.random.default_rng(SEED); vals = []
    for _ in range(B):
        pick = rg.integers(0, len(groups), len(groups))
        vals.append(stat(np.concatenate([groups[k][rg.integers(0, len(groups[k]), len(groups[k]))] for k in pick])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def model_rows(c):
    """Per-layer F1, F2 values and per-merged-layer F3, F4 values for one model."""
    L = c["layers"]; f12, f34 = [], []
    for l in range(L):
        m = c["measures"][str(l)]
        raw = float(np.mean([h["raw"] for h in m["f1"]])); val = float(np.mean([h["val"] for h in m["f1"]]))
        out = float(np.mean([h["out"] for h in m["f1"]])); fr = float(np.nanmedian([h["fnorm_sink_over_rest"] for h in m["f1"]]))
        f12.append({"layer": l, "raw": raw, "val": val, "out": out, "fnorm_sink_over_rest": fr, "high_sink": raw > HS,
                    "f2_raw": m["f2"]["raw"]["reproduced"], "f2_val": m["f2"]["val"]["reproduced"],
                    "f2_raw_real": m["f2"]["raw"]["real"], "f2_val_real": m["f2"]["val"]["real"]})
    for ls, sw in c["sweep"].items():
        keys = list(sw); kl = np.array([sw[k]["kl"] for k in keys]); dn = np.array([sw[k]["dnll"] for k in keys])
        kw = np.array([sw[k]["kl_windows"] for k in keys]); h = kw.shape[1] // 2
        row = {"layer": int(ls), "n_pairs": len(keys), "kl_median": float(np.median(kl)),
               "reliability": float(spearmanr(kw[:, :h].mean(1), kw[:, h:].mean(1))[0]), "rho": {}, "rho_dnll": {}, "pick_kl": {}, "pick_rank_pct": {}}
        for p in PREDICTORS:
            d = as_distance(p, [c["measures"][ls]["pairs"][k][p] for k in keys])
            row["rho"][p] = float(spearmanr(d, kl)[0]); row["rho_dnll"][p] = float(spearmanr(d, dn)[0])
            k0 = int(np.argmin(d)); row["pick_kl"][p] = float(kl[k0]); row["pick_rank_pct"][p] = float((kl < kl[k0]).mean() * 100)
        row["best_kl"] = float(kl.min()); row["mean_kl"] = float(kl.mean())
        f34.append(row)
    return f12, f34


models = {}
for p in sorted(glob.glob(f"{D}/*.json")):
    c = json.load(open(p))
    if isinstance(c, dict) and "measures" in c and "sweep" in c:
        models[c["model"]] = c
if not models:
    raise SystemExit(f"no model results in {D}")
res = {"thresholds": TH, "n_models": len(models), "complete": all(m in models for m in MODELS), "per_model": {}, "F": {}}
G = {k: [] for k in ("f1_ratio", "f1_out", "f1_fnorm", "f2_drop", "rel", *[f"rho_{p}" for p in PREDICTORS],
                     "d_out_raw", "d_raw_gen", "d_nosink_raw", "d_val_raw", "d_out_val",
                     "f4_out_gen", "f4_out_raw", "f4_raw_gen", "rank_out", "rank_raw", "rank_gen")}
for name, c in models.items():
    f12, f34 = model_rows(c)
    hs = [r for r in f12 if r["high_sink"]]
    g = {"f1_ratio": [r["val"] / r["raw"] for r in hs], "f1_out": [r["out"] for r in hs], "f1_fnorm": [r["fnorm_sink_over_rest"] for r in hs],
         "f2_drop": [r["f2_raw"] - r["f2_val"] for r in hs if r["f2_raw"] is not None and r["f2_val"] is not None],
         "rel": [r["reliability"] for r in f34]}
    for p in PREDICTORS:
        g[f"rho_{p}"] = [r["rho"][p] for r in f34]
    for key, a, b in (("d_out_raw", "out_pert", "tv_raw"), ("d_raw_gen", "tv_raw", "gen_cos"), ("d_nosink_raw", "tv_nosink", "tv_raw"),
                      ("d_val_raw", "val_dist", "tv_raw"), ("d_out_val", "out_pert", "val_dist")):
        g[key] = [r["rho"][a] - r["rho"][b] for r in f34]
    for key, a, b in (("f4_out_gen", "out_pert", "gen_cos"), ("f4_out_raw", "out_pert", "tv_raw"), ("f4_raw_gen", "tv_raw", "gen_cos")):
        g[key] = [np.log(r["pick_kl"][a] / r["pick_kl"][b]) for r in f34]       # log ratio; below zero means the first predictor picks the cheaper merge
    for key, a in (("rank_out", "out_pert"), ("rank_raw", "tv_raw"), ("rank_gen", "gen_cos")):
        g[key] = [r["pick_rank_pct"][a] for r in f34]
    for k, v in g.items():
        G[k].append(v)
    res["per_model"][name] = {"n_layers": c["layers"], "n_high_sink": len(hs), "n_merged_layers": len(f34), "restore_check_kl": c["restore_check_kl"],
                              "f1_ratio_median": float(np.median(g["f1_ratio"])) if hs else None, "f1_out_median": float(np.median(g["f1_out"])) if hs else None,
                              "f1_fnorm_median": float(np.median(g["f1_fnorm"])) if hs else None,
                              "f2_drop_mean": float(np.mean(g["f2_drop"])) if g["f2_drop"] else None,
                              "reliability_median": float(np.median(g["rel"])),
                              "rho_mean": {p: float(np.mean(g[f"rho_{p}"])) for p in PREDICTORS},
                              "f4_median_ratio_out_over_gen": float(np.exp(np.median(g["f4_out_gen"]))),
                              "layers_f12": f12, "layers_f34": f34}


def pooled(key, stat=np.mean):
    groups = G[key]; allv = np.concatenate([np.asarray(x, float) for x in groups if len(x)])
    return {"value": float(stat(allv)), "ci95": cluster_ci(groups, stat), "n_layers": int(allv.size), "n_models": sum(1 for x in groups if len(x))}


def ratio(key):
    r = pooled(key, np.median)
    return {"median_ratio": float(np.exp(r["value"])), "ci95": [float(np.exp(x)) for x in r["ci95"]], "n_layers": r["n_layers"], "n_models": r["n_models"]}


REL = pooled("rel", np.median)
F1 = pooled("f1_ratio", np.median); F2 = pooled("f2_drop")
rho = {p: pooled(f"rho_{p}") for p in PREDICTORS}
F3a = pooled("d_out_raw"); F3b = pooled("d_raw_gen"); F3c = pooled("d_nosink_raw"); F3d = rho["out_pert"]
F4 = ratio("f4_out_gen")
res["F"]["reliability_gate"] = dict(REL, pass_=REL["value"] >= TH["RELIABILITY_MIN"])
res["F"]["F1_sink_value_share_over_attention_share"] = dict(F1, pass_=F1["value"] <= TH["F1_ratio_max"] and F1["ci95"][1] <= TH["F1_ratio_ci_max"])
res["F"]["F2_sinknull_share_drop_raw_to_valueweighted"] = dict(F2, pass_=F2["value"] >= TH["F2_drop_min"] and F2["ci95"][0] > 0)
res["F"]["F3a_rho_out_pert_minus_tv_raw"] = dict(F3a, pass_=F3a["value"] >= TH["F3_delta_min"] and F3a["ci95"][0] > 0)
res["F"]["F3b_rho_tv_raw_minus_gen_cos"] = dict(F3b, pass_=F3b["value"] >= TH["F3_delta_min"] and F3b["ci95"][0] > 0)
res["F"]["F3c_rho_tv_nosink_minus_tv_raw_no_gain"] = dict(F3c, pass_=F3c["ci95"][1] < TH["F3c_upper_max"])
res["F"]["F3d_rho_out_pert"] = dict(F3d, pass_=F3d["value"] >= TH["F3d_rho_min"] and F3d["ci95"][0] >= TH["F3d_ci_low_min"])
res["F"]["F4_pick_kl_out_pert_over_gen_cos"] = dict(F4, pass_=F4["median_ratio"] <= TH["F4_ratio_max"] and F4["ci95"][1] < 1)
res["F"]["primary_passes"] = int(sum(bool(v["pass_"]) for k, v in res["F"].items() if k.startswith("F")))
res["F"]["secondary"] = {"rho_mean_by_predictor": rho, "f1_out_share": pooled("f1_out", np.median), "f1_sink_fnorm_over_rest": pooled("f1_fnorm", np.median),
                         "d_val_minus_raw": pooled("d_val_raw"), "d_out_minus_val": pooled("d_out_val"),
                         "pick_kl_out_pert_over_tv_raw": ratio("f4_out_raw"), "pick_kl_tv_raw_over_gen_cos": ratio("f4_raw_gen"),
                         "pick_rank_pct": {"out_pert": pooled("rank_out", np.median), "tv_raw": pooled("rank_raw", np.median), "gen_cos": pooled("rank_gen", np.median)}}
json.dump(res, open(OUTFILE, "w"), indent=1)
print("models:", len(models), "complete:", res["complete"])
for k, v in res["F"].items():
    print(k, json.dumps(v)[:400])
for name, pm in res["per_model"].items():
    print(f"  {name.split('/')[-1]:24s} hs {pm['n_high_sink']:2d} | F1 ratio {pm['f1_ratio_median']} out {pm['f1_out_median']} | F2 drop {pm['f2_drop_mean']} | rel {pm['reliability_median']:.2f} | rho " +
          " ".join(f"{p} {v:+.2f}" for p, v in pm["rho_mean"].items()) + f" | F4 {pm['f4_median_ratio_out_over_gen']:.2f}")
