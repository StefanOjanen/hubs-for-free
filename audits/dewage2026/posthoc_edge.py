# POST HOC, NOT REGISTERED (audits/PREREGISTRATION4.md fixes no analysis of
# why a null reproduces a statistic; this script changes no label and no
# outcome of the Target 4 battery). It asks one question the battery raises:
# the registered Gaussian null (a'), which has no learned structure at all,
# produces a median 1288 of 4096 "Marchenko-Pastur outliers" in the square
# projections and 138 of 1024 in K and V. Why?
#
# The recipe of the source paper, reimplemented in reproduce.py and used by
# the battery, estimates the noise scale from the median of the squared
# singular values:
#     gamma = max(m, n) / min(m, n)
#     sigma^2 = median(s^2) / (1 + gamma)
#     lambda_+ = sigma^2 (1 + sqrt(gamma))^2
# For an m x n matrix with i.i.d. entries of variance v (m <= n), the
# eigenvalues of W W^T are v n times a Marchenko-Pastur law with ratio
# c = m / n, so the bulk edge sits at v n (1 + sqrt(c))^2. Matching that to
# the recipe's formula requires sigma^2 = v m, i.e. median(s^2) = v (m + n);
# the actual median is v n times the median of the MP law, which is smaller.
# The recipe therefore places lambda_+ below the true bulk edge and counts
# part of the bulk as outliers.
#
# Three edges are compared on the same spectra: the recipe's; an edge whose
# scale is fitted by matching the MP mean (the mean of s^2 is v n exactly),
# which needs no oracle but is inflated by any real outliers and therefore
# undercounts; and an edge whose scale is fitted by matching the MP median,
# which is robust to outliers (the median of s^2 in units of v n depends
# only on the aspect ratio, and is taken here from a simulated Gaussian of
# the same shape). On synthetic matrices the known entry variance gives a
# fourth, oracle edge. Every edge is applied to the real Mistral-7B weights
# of a subset of layers and to Gaussian matrices matched in shape and
# Frobenius norm, the battery's null (a').
#   python audits/dewage2026/posthoc_edge.py [--layers=0,8,16,24]
import glob
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "audits/dewage2026")
LAYERS = [int(x) for x in next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--layers=")), "0,8,16,24").split(",")]
NAME = "mistralai/Mistral-7B-v0.1"
TYPES = ("q_proj", "k_proj", "v_proj", "o_proj")
OUT = "audits/dewage2026/posthoc_edge.json"
rng = np.random.default_rng(0)


def spectrum(W):
    m, n = W.shape
    Wd = W.astype(np.float64)
    G = Wd @ Wd.T if m <= n else Wd.T @ Wd
    return np.sort(np.clip(np.linalg.eigvalsh(G), 0, None))[::-1], m, n


MED_RATIO = {}


def mp_median_ratio(m, n):
    """median(s^2) / (v n) for a Gaussian matrix of this shape, by simulation
    (a function of the aspect ratio alone; one draw suffices at these sizes)."""
    key = (min(m, n), max(m, n))
    if key not in MED_RATIO:
        g = np.random.default_rng(12345).normal(size=(m, n))
        s2s, _, _ = spectrum(g)
        MED_RATIO[key] = float(np.median(s2s) / max(m, n))
    return MED_RATIO[key]


def counts(s2, m, n, v_known=None):
    """Four edges on the same spectrum. m, n are the matrix dimensions."""
    lo, hi = min(m, n), max(m, n)
    gamma = hi / lo
    c = lo / hi
    out = {"n_sv": int(len(s2))}
    # (i) the source recipe
    lam_recipe = (np.median(s2) / (1 + gamma)) * (1 + np.sqrt(gamma)) ** 2
    # (iii) scale fitted from the bulk with no oracle: the MP mean is v n,
    # estimated by trace(G) / lo, which is exactly the mean of s2
    lam_fit = np.mean(s2) * (1 + np.sqrt(c)) ** 2
    lam_med = (np.median(s2) / mp_median_ratio(m, n)) * (1 + np.sqrt(c)) ** 2
    for key, lam in (("recipe", lam_recipe), ("mp_mean_fit", lam_fit), ("mp_median_fit", lam_med)):
        k = int((s2 > lam).sum())
        out[key] = {"lambda_plus": float(lam), "outliers": k, "frac": k / len(s2),
                    "energy_frac": float(s2[:k].sum() / s2.sum())}
    # (ii) oracle edge from the known entry variance (synthetic matrices only)
    if v_known is not None:
        lam_true = v_known * hi * (1 + np.sqrt(c)) ** 2
        k = int((s2 > lam_true).sum())
        out["oracle"] = {"lambda_plus": float(lam_true), "outliers": k, "frac": k / len(s2),
                         "energy_frac": float(s2[:k].sum() / s2.sum())}
    return out


def real_matrices(layers):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    d = snapshot_download(NAME, allow_patterns=["*.safetensors", "*.json"])
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, framework="pt") as sf:
            for key in sf.keys():
                if ".self_attn." in key and key.endswith(".weight") and any(p in key for p in TYPES):
                    l = int(key.split(".layers.")[1].split(".")[0]); t = key.split(".self_attn.")[1].split(".")[0]
                    if l in layers:
                        yield l, t, sf.get_tensor(key).float().numpy()


t0 = time.time()
res = {"model": NAME, "layers": LAYERS, "posthoc": True,
       "note": "not registered; explains the Gaussian null of the Target 4 battery, changes no label",
       "synthetic": {}, "real": []}

shapes = {}
rows = []
for l, t, W in real_matrices(set(LAYERS)):
    s2, m, n = spectrum(W)
    c = counts(s2, m, n)
    c.update({"layer": l, "type": t, "shape": [m, n], "frobenius": float(np.linalg.norm(W)),
              "entry_var": float(W.astype(np.float64).var())})
    rows.append(c); shapes[t] = (m, n, float(np.linalg.norm(W)))
    print(f"  real L{l:2d} {t:6s} {m}x{n}: recipe {c['recipe']['outliers']:5d} ({c['recipe']['frac']:.1%}, energy {c['recipe']['energy_frac']:.3f}) | mp-mean edge {c['mp_mean_fit']['outliers']:4d} | mp-median edge {c['mp_median_fit']['outliers']:4d} ({c['mp_median_fit']['frac']:.2%}, energy {c['mp_median_fit']['energy_frac']:.3f}) ({time.time()-t0:.0f}s)", flush=True)
res["real"] = rows

for t, (m, n, fro) in shapes.items():
    v = (fro ** 2) / (m * n)                       # Gaussian matched in shape and Frobenius norm, as null (a')
    G = rng.normal(scale=np.sqrt(v), size=(m, n))
    s2, _, _ = spectrum(G)
    c = counts(s2, m, n, v_known=v)
    res["synthetic"][t] = dict(c, shape=[m, n], entry_var=v)
    print(f"  gauss {t:6s} {m}x{n}: recipe {c['recipe']['outliers']:5d} ({c['recipe']['frac']:.1%}) | mp-mean edge {c['mp_mean_fit']['outliers']:4d} | mp-median edge {c['mp_median_fit']['outliers']:4d} | oracle edge {c['oracle']['outliers']:4d} ({c['oracle']['frac']:.2%}) ({time.time()-t0:.0f}s)", flush=True)

by_type = {}
for t in TYPES:
    rs = [r for r in rows if r["type"] == t]
    if rs and t in res["synthetic"]:
        by_type[t] = {"real_recipe_mean": float(np.mean([r["recipe"]["outliers"] for r in rs])),
                      "real_mp_mean_fit_mean": float(np.mean([r["mp_mean_fit"]["outliers"] for r in rs])),
                      "real_mp_median_fit_mean": float(np.mean([r["mp_median_fit"]["outliers"] for r in rs])),
                      "real_energy_recipe": float(np.mean([r["recipe"]["energy_frac"] for r in rs])),
                      "real_energy_mp_median_fit": float(np.mean([r["mp_median_fit"]["energy_frac"] for r in rs])),
                      "gauss_recipe": res["synthetic"][t]["recipe"]["outliers"],
                      "gauss_mp_median_fit": res["synthetic"][t]["mp_median_fit"]["outliers"],
                      "gauss_mp_mean_fit": res["synthetic"][t]["mp_mean_fit"]["outliers"],
                      "gauss_oracle": res["synthetic"][t]["oracle"]["outliers"],
                      "n_sv": rs[0]["n_sv"], "n_layers": len(rs)}
res["by_type"] = by_type
res["runtime_s"] = round(time.time() - t0, 1)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(by_type, indent=1)); print("POSTHOC_DONE", OUT)
