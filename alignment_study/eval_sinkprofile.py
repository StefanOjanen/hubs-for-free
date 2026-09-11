# Evaluation of PREREGISTRATION9 (sink-profile generative model), committed
# before any result exists. Reads sinkprofile/<model>.json, writes
# sinkprofile_results.json, prints the scorecard P1 to P4.
import glob
import json
import numpy as np
from scipy.stats import spearmanr

import sys
D = sys.argv[1] if len(sys.argv) > 1 else "alignment_study/sinkprofile"
OUTFILE = "alignment_study/sinkprofile_results.json" if D.rstrip("/") == "alignment_study/sinkprofile" else D.rstrip("/") + "_results.json"
MODELS = ["gpt2", "gpt2-medium", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "TinyLlama/TinyLlama_v1.1",
          "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B", "microsoft/Phi-3-mini-4k-instruct",
          "mistralai/Mistral-7B-v0.1", "Qwen/Qwen2.5-7B", "allenai/OLMo-2-1124-7B"]


def sp(x, y):
    return float(spearmanr(x, y)[0])


models = {}
for f in sorted(glob.glob(f"{D}/*.json")):
    c = json.load(open(f))
    if isinstance(c, dict) and "rows" in c:
        models[c["model"]] = c
if not models:
    raise SystemExit("no results yet")
res = {"n_models": len(models), "complete": all(m in models for m in MODELS) if D.rstrip("/") == "alignment_study/sinkprofile" else None, "per_model": {}, "P": {}}
pz_r, pz_p, pz_d = [], [], []
for name, c in models.items():
    rows = c["rows"]
    zr = [r["z_real"] for r in rows]; zp = [r["z_profile"] for r in rows]; zd = [r["z_dense"] for r in rows]
    rr = [r["r1_real"] for r in rows]; rp = [r["r1_profile"] for r in rows]; rd = [r["r1_dense"] for r in rows]
    pm = {"n_layers": len(rows), "spearman_z_profile": round(sp(zr, zp), 3), "spearman_z_dense": round(sp(zr, zd), 3),
          "mae_z_profile": round(float(np.median(np.abs(np.array(zr) - np.array(zp)))), 2), "mae_z_dense": round(float(np.median(np.abs(np.array(zr) - np.array(zd)))), 2),
          "spearman_r1_profile": round(sp(rr, rp), 3), "spearman_r1_dense": round(sp(rr, rd), 3),
          "mae_r1_profile": round(float(np.median(np.abs(np.array(rr) - np.array(rp)))), 3),
          "median_profile_share_high_sink": round(float(np.median([r["profile_share"] for r in rows if r["smass"] > 0.4])), 3) if any(r["smass"] > 0.4 for r in rows) else None}
    pm["P1_pass"] = pm["spearman_z_profile"] >= 0.7; pm["P2_pass"] = pm["mae_z_profile"] <= 3.0; pm["P3_pass"] = pm["spearman_r1_profile"] >= 0.6
    res["per_model"][name] = pm
    pz_r += zr; pz_p += zp; pz_d += zd
n = len(res["per_model"])
for k in ("P1", "P2", "P3"):
    passes = sum(pm[f"{k}_pass"] for pm in res["per_model"].values())
    res["P"][k] = {"passes": passes, "n": n, "pass": passes >= (2 / 3) * n, "secondary": k == "P3"}
res["P"]["P4"] = {"pooled_spearman_z_profile": round(sp(pz_r, pz_p), 3), "pooled_spearman_z_dense": round(sp(pz_r, pz_d), 3), "n_layers": len(pz_r), "pass": sp(pz_r, pz_p) >= 0.8}
json.dump(res, open(OUTFILE, "w"), indent=1)
print("models:", n, "complete:", res["complete"])
for k, v in res["P"].items():
    print(k, json.dumps(v))
for name, pm in res["per_model"].items():
    print(f"   {name.split('/')[-1]:26s} L {pm['n_layers']:2d} | z: profile rho {pm['spearman_z_profile']:+.2f} mae {pm['mae_z_profile']:5.2f} (dense {pm['spearman_z_dense']:+.2f} / {pm['mae_z_dense']:5.2f}) | r1: profile rho {pm['spearman_r1_profile']:+.2f} mae {pm['mae_r1_profile']:.3f} (dense {pm['spearman_r1_dense']:+.2f}) | profile share hs {pm['median_profile_share_high_sink']}")
