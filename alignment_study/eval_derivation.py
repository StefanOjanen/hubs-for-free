# Evaluation of PREREGISTRATION8 (derivation of shared energy), committed
# before any result exists. Reads derivation/<model>.json, writes
# derivation_results.json, prints the scorecard G1 to G4.
import glob
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

D = "alignment_study/derivation"
MODELS = ["gpt2", "gpt2-medium", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "TinyLlama/TinyLlama_v1.1",
          "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B", "microsoft/Phi-3-mini-4k-instruct",
          "mistralai/Mistral-7B-v0.1", "Qwen/Qwen2.5-7B", "allenai/OLMo-2-1124-7B"]
HS = 0.4


def fit(p, m):
    p, m = np.asarray(p), np.asarray(m)
    lin = np.polyfit(p, m, 1)
    return {"n": len(p), "pearson_r2_linear": round(float(pearsonr(p, m)[0] ** 2), 3),
            "r2_identity": round(float(1 - ((m - p) ** 2).sum() / ((m - m.mean()) ** 2).sum()), 3),
            "spearman": round(float(spearmanr(p, m)[0]), 3), "median_abs_err": round(float(np.median(np.abs(m - p))), 4),
            "frac_derived_below_measured": round(float(np.mean(p <= m)), 3), "slope": round(float(lin[0]), 3), "intercept": round(float(lin[1]), 3)}


def boot_r2(p, m, B=2000, seed=0):
    rg = np.random.default_rng(seed); p, m = np.asarray(p), np.asarray(m)
    v = [pearsonr(p[i], m[i])[0] ** 2 for i in (rg.integers(0, len(p), len(p)) for _ in range(B))]
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


models = {}
for f in sorted(glob.glob(f"{D}/*.json")):
    c = json.load(open(f))
    if isinstance(c, dict) and "rows" in c:
        models[c["model"]] = c
if not models:
    raise SystemExit("no derivation results yet")

res = {"n_models": len(models), "complete": all(m in models for m in MODELS), "per_model": {}, "G": {}}
hs_p, hs_m, all_p, all_m, all_d = [], [], [], [], []
improve = []
for name, c in models.items():
    rows = c["rows"]; hs = [r for r in rows if r["smass"] > HS]
    pm = {"n_layers": len(rows), "n_high_sink": len(hs)}
    if len(hs) >= 3:
        pm["high_sink_ideal"] = fit([r["derived_ideal"] for r in hs], [r["measured"] for r in hs])
    pm["all_ideal"] = fit([r["derived_ideal"] for r in rows], [r["measured"] for r in rows])
    pm["all_dict"] = fit([r["derived_dict"] for r in rows], [r["measured"] for r in rows])
    improve.append(pm["all_dict"]["r2_identity"] > pm["all_ideal"]["r2_identity"])
    res["per_model"][name] = pm
    hs_p += [r["derived_ideal"] for r in hs]; hs_m += [r["measured"] for r in hs]
    all_p += [r["derived_ideal"] for r in rows]; all_m += [r["measured"] for r in rows]; all_d += [r["derived_dict"] for r in rows]

g1 = fit(hs_p, hs_m); g1["ci95_r2"] = boot_r2(hs_p, hs_m)
res["G"]["G1"] = dict(g1, pass_=g1["pearson_r2_linear"] >= 0.8)
g2 = fit(all_p, all_m)
res["G"]["G2"] = {"spearman_all_layers": g2["spearman"], "n": g2["n"], "pass_": g2["spearman"] >= 0.8}
res["G"]["G3"] = {"frac_derived_below_measured_high_sink": g1["frac_derived_below_measured"], "pass_": g1["frac_derived_below_measured"] >= 0.9, "secondary": True}
gd = fit(all_d, all_m)
res["G"]["G4"] = {"dict_median_abs_err_all": gd["median_abs_err"], "dict_r2_identity_all": gd["r2_identity"], "ideal_r2_identity_all": g2["r2_identity"],
                  "models_where_dict_improves_identity_r2": int(sum(improve)), "n_models": len(improve),
                  "pass_": gd["median_abs_err"] < 0.05 and sum(improve) >= (2 / 3) * len(improve), "secondary": True}
json.dump(res, open("alignment_study/derivation_results.json", "w"), indent=1)
print("models:", len(models), "complete:", res["complete"])
for k, v in res["G"].items():
    print(k, json.dumps(v))
for name, pm in res["per_model"].items():
    hsf = pm.get("high_sink_ideal", {})
    print(f"   {name.split('/')[-1]:26s} layers {pm['n_layers']:2d} hs {pm['n_high_sink']:2d} | hs ideal R2 {hsf.get('pearson_r2_linear')} spearman {hsf.get('spearman')} | all ideal R2 {pm['all_ideal']['pearson_r2_linear']} spearman {pm['all_ideal']['spearman']} | dict identity R2 {pm['all_dict']['r2_identity']} mae {pm['all_dict']['median_abs_err']}")
