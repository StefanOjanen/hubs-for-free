# Target 4 (Dewage et al. 2026, arXiv:2608.07921), step T2.1: reproduce the
# base result only. Claim: singular values of the attention projection
# weights above the Marchenko-Pastur upper edge ("spectral outliers") carry
# the learned structure; Table II reports, per projection type, the average
# number and fraction of outliers per matrix: Mistral-7B Q 87.5 percent,
# K 74.7, V 43.6, O 84.6 (LLaMA-2-7B: 87.8, 87.5, 79.0, 79.1). Their stated
# code URL returned 404 on 2026-09-06 and again on 2026-09-11, so the recipe
# is reimplemented from the paper: for a weight matrix W (m x n) with
# singular values s_i, gamma = max(m, n) / min(m, n), sigma^2 =
# median(s_i^2) / (1 + gamma), lambda_plus = sigma^2 (1 + sqrt(gamma))^2, and
# an outlier is s_i^2 > lambda_plus. Whole projection matrices, one per
# layer and type, weights read from the safetensors shards in float32.
#   python audits/dewage2026/reproduce.py [model] [out_prefix]
import glob
import json
import os
import sys
import time
import numpy as np

NAME = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-v0.1"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "audits/dewage2026/" + NAME.split("/")[-1]
PAPER = {"mistralai/Mistral-7B-v0.1": {"q_proj": 87.5, "k_proj": 74.7, "v_proj": 43.6, "o_proj": 84.6},
         "meta-llama/Llama-2-7b-hf": {"q_proj": 87.8, "k_proj": 87.5, "v_proj": 79.0, "o_proj": 79.1}}


def mp_outliers(W):
    m, n = W.shape
    s = np.linalg.svd(W.astype(np.float64), compute_uv=False)
    gamma = max(m, n) / min(m, n)
    sigma2 = np.median(s ** 2) / (1.0 + gamma)
    lam_plus = sigma2 * (1.0 + np.sqrt(gamma)) ** 2
    k = int((s ** 2 > lam_plus).sum())
    return {"shape": [m, n], "n_sv": len(s), "outliers": k, "frac": k / len(s), "lambda_plus": float(lam_plus),
            "sigma2": float(sigma2), "top_sv2": float(s[0] ** 2), "energy_frac_outliers": float((s[:k] ** 2).sum() / (s ** 2).sum())}


def shards(name):
    from huggingface_hub import snapshot_download
    d = snapshot_download(name, allow_patterns=["*.safetensors", "*.json"])
    return sorted(glob.glob(os.path.join(d, "*.safetensors")))


if __name__ == "__main__":
    t0 = time.time()
    from safetensors import safe_open
    rows = []
    for f in shards(NAME):
        with safe_open(f, framework="pt") as sf:
            for key in sf.keys():
                if ".self_attn." in key and key.endswith(".weight") and any(p in key for p in ("q_proj", "k_proj", "v_proj", "o_proj")):
                    layer = int(key.split(".layers.")[1].split(".")[0]); ptype = key.split(".self_attn.")[1].split(".")[0]
                    W = sf.get_tensor(key).float().numpy()
                    r = {"layer": layer, "type": ptype, **mp_outliers(W)}
                    rows.append(r)
                    print(f"  L{layer:2d} {ptype:6s} {r['shape']} outliers {r['outliers']:4d}/{r['n_sv']} = {100*r['frac']:5.1f}%  energy in outliers {r['energy_frac_outliers']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: (r["layer"], r["type"]))
    summary = {}
    for ptype in ("q_proj", "k_proj", "v_proj", "o_proj"):
        rs = [r for r in rows if r["type"] == ptype]
        summary[ptype] = {"mean_outliers": round(float(np.mean([r["outliers"] for r in rs])), 1), "mean_frac_pct": round(100 * float(np.mean([r["frac"] for r in rs])), 1),
                          "paper_frac_pct": PAPER.get(NAME, {}).get(ptype), "n_layers": len(rs),
                          "depth_spearman_energy_frac": None}
        if len(rs) > 3:
            from scipy.stats import spearmanr
            summary[ptype]["depth_spearman_energy_frac"] = round(float(spearmanr([r["layer"] for r in rs], [r["energy_frac_outliers"] for r in rs])[0]), 3)
        if summary[ptype]["paper_frac_pct"]:
            summary[ptype]["rel_err_vs_paper"] = round(abs(summary[ptype]["mean_frac_pct"] - summary[ptype]["paper_frac_pct"]) / summary[ptype]["paper_frac_pct"], 3)
    json.dump({"model": NAME, "recipe": "gamma=max/min, sigma2=median(s^2)/(1+gamma), lambda_plus=sigma2(1+sqrt(gamma))^2, outlier: s^2>lambda_plus",
               "rows": rows, "summary": summary, "runtime_s": round(time.time() - t0, 1)}, open(PREFIX + "_base_result.json", "w"), indent=1)
    print("SUMMARY", json.dumps(summary)); print("DEWAGE_DONE", PREFIX)
