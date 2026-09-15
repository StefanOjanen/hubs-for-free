# Model-clustered inference for every pooled correlation in the manuscript
# (V3 plan, task A3). Layers of one model are not independent draws, so the
# per-layer bootstrap intervals reported by the round evaluations understate
# the uncertainty. For each pooled statistic this script recomputes the point
# estimate from the per-model result files, then gives
#   (i)  a two-stage cluster bootstrap: resample models with replacement,
#        then layers within each sampled model, 2,000 draws, percentile
#        2.5 and 97.5 (for the dynamics round the clusters are checkpoints,
#        model x training step, since only two models were trained);
#   (ii) the per-cluster estimate and a random-effects average
#        (DerSimonian and Laird on Fisher-z transformed Spearmans with
#        variance 1/(n - 3); clusters with fewer than 4 layers are listed but
#        not averaged);
#   (iii) the old per-layer bootstrap with the same seed, for comparison.
# Nothing is fitted or selected here; the pooling rules are those of the
# frozen evaluation scripts (high-sink = sink mass above 0.4, the PR6 cv(gn)
# >= 0.1 filter and calibration-model exclusion, and so on).
# Writes alignment_study/clustered_stats.json.
#   python alignment_study/clustered_stats.py
import glob
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

B, SEED = 2000, 0
HS, CV_MIN = 0.4, 0.1
CALIB = {"Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-3B"}


def spear(x, y):
    return float(spearmanr(x, y)[0])


def r2lin(x, y):
    return float(pearsonr(x, y)[0] ** 2)


def cluster_boot(groups, stat, B=B, seed=SEED):
    """groups: list of (x_array, y_array) per cluster. Two-stage resampling."""
    rg = np.random.default_rng(seed); k = len(groups); vals = []
    for _ in range(B):
        pick = rg.integers(0, k, k); xs, ys = [], []
        for g in pick:
            x, y = groups[g]; idx = rg.integers(0, len(x), len(x)); xs.append(x[idx]); ys.append(y[idx])
        x, y = np.concatenate(xs), np.concatenate(ys)
        vals.append(stat(x, y) if len(x) >= 4 and np.std(x) > 0 and np.std(y) > 0 else np.nan)
    v = np.array(vals)
    return [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)], int(np.isnan(v).sum())


def layer_boot(x, y, stat, B=B, seed=SEED):
    rg = np.random.default_rng(seed); n = len(x)
    v = [stat(x[i], y[i]) for i in (rg.integers(0, n, n) for _ in range(B))]
    return [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)]


def random_effects(groups, names):
    """DerSimonian-Laird average of per-cluster Spearmans on the Fisher-z scale."""
    per, z, w = {}, [], []
    for (x, y), nm in zip(groups, names):
        n = len(x)
        r = spear(x, y) if n >= 3 and np.std(x) > 0 and np.std(y) > 0 else None
        per[nm] = {"n": int(n), "spearman": None if r is None else round(r, 3)}
        if r is not None and n >= 4 and abs(r) < 0.9999:
            z.append(np.arctanh(r)); w.append(n - 3.0)
    if len(z) < 2:
        return per, None
    z, w = np.array(z), np.array(w); zf = (w * z).sum() / w.sum()
    Q = (w * (z - zf) ** 2).sum(); df = len(z) - 1; C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - df) / C); ws = 1 / (1 / w + tau2); zr = (ws * z).sum() / ws.sum(); se = np.sqrt(1 / ws.sum())
    return per, {"spearman": round(float(np.tanh(zr)), 3), "ci95": [round(float(np.tanh(zr - 1.96 * se)), 3), round(float(np.tanh(zr + 1.96 * se)), 3)],
                 "tau": round(float(np.sqrt(tau2)), 3), "k": int(len(z)), "heterogeneity_Q": round(float(Q), 2)}


def analyse(name, groups, names, stat=spear, note=""):
    x = np.concatenate([g[0] for g in groups]); y = np.concatenate([g[1] for g in groups])
    ci_c, n_nan = cluster_boot(groups, stat); ci_l = layer_boot(x, y, stat)
    per, re = random_effects(groups, names) if stat is spear else (
        {nm: {"n": int(len(g[0])), "r2_linear": (round(r2lin(*g), 3) if len(g[0]) >= 4 else None)} for g, nm in zip(groups, names)}, None)
    out = {"statistic": "R2 linear" if stat is r2lin else "Spearman", "pooled": round(stat(x, y), 3), "n_layers": int(len(x)), "n_clusters": len(groups),
           "ci95_cluster_bootstrap": ci_c, "cluster_draws_undefined": n_nan, "ci95_layer_bootstrap": ci_l, "random_effects": re, "per_cluster": per, "note": note}
    print(f"{name:38s} pooled {out['pooled']:+.3f} n {out['n_layers']:4d} clusters {len(groups):2d} | cluster CI {ci_c} | layer CI {ci_l} | RE {re['spearman'] if re else None} {re['ci95'] if re else ''}")
    return out


def hs_rows(rows):
    return [r for r in rows if r["smass"] > HS]


def xy(rows, kx, ky):
    return np.array([r[kx] for r in rows], float), np.array([r[ky] for r in rows], float)


res = {"B": B, "seed": SEED, "method": "two-stage cluster bootstrap (models, then layers) with per-layer bootstrap and random-effects average for comparison", "stats": {}}

# PR2 H4: heldout round, high-sink layers, sharedE vs r1
d = json.load(open("alignment_study/heldout_round.json"))
g = [(xy(hs_rows(rows), "sharedE", "r1_real")) for rows in d["models"].values()]; nm = list(d["models"])
res["stats"]["PR2_H4_law"] = analyse("PR2 H4 law (5 held-out models)", [x for x in g if len(x[0])], [n for n, x in zip(nm, g) if len(x[0])])

# PR3 S4: scale round captures
caps = ["qwen2.5-3b", "qwen2.5-7b", "mistral-7b", "phi-3-mini", "olmo-2-7b"]
g, nm = [], []
for c in caps:
    dd = json.load(open(f"alignment_study/scale_partial/{c}_capture.json")); rows = hs_rows(dd["per_layer"])
    if rows:
        g.append(xy(rows, "sharedE", "r1_real")); nm.append(dd["model"])
res["stats"]["PR3_S4_law"] = analyse("PR3 S4 law (5 scale-round models)", g, nm)


def rerun_groups(directory):
    g, nm = [], []
    for p in sorted(glob.glob(f"{directory}/*.json")):
        c = json.load(open(p))
        if not (isinstance(c, dict) and "protocol_A" in c) or c["model"] in CALIB:
            continue
        rows = [r for r in hs_rows(c["protocol_A"]["rows"]) if r["cv_gn"] >= CV_MIN]
        if rows:
            g.append(xy(rows, "sharedE", "r1_real")); nm.append(c["model"])
    return g, nm


g, nm = rerun_groups("alignment_study/rerun")
res["stats"]["PR6_S4_law"] = analyse("PR6 S4' law (11 models, cv filter)", g, nm)
g10, nm10 = rerun_groups("alignment_study/rerun_pr10")
res["stats"]["PR10_N4_law"] = analyse("PR10 N4 law (4 new designs)", g10, nm10)
res["stats"]["PR6_PR10_law_pooled"] = analyse("PR6 + PR10 law (15 models)", g + g10, nm + nm10, note="not a registered clause; the two rounds pooled for reference")

# PR5 dynamics: clusters are checkpoints
cells = {}
for p in sorted(glob.glob("alignment_study/dynamics/*.json")):
    c = json.load(open(p)); cells[(c["model"], c["step"])] = c["rows"]
nm = [f"{m}@{s}" for m, s in cells]
res["stats"]["PR5_D1_sink_vs_cos"] = analyse("PR5 D1 sink mass vs cos(S, sink)", [xy(rows, "smass", "cosS") for rows in cells.values()], nm, note="clusters are checkpoints (model x step)")
g, nm2 = [], []
for (m, s), rows in cells.items():
    h = hs_rows(rows)
    if len(h) >= 2:
        g.append(xy(h, "sharedE", "r1_real")); nm2.append(f"{m}@{s}")
res["stats"]["PR5_D2_law"] = analyse("PR5 D2 law over high-sink cells", g, nm2, note="clusters are checkpoints with at least two high-sink layers")

# PR9 / PR10 sink-profile P4
for key, directory, label in (("PR9_P4_profile", "alignment_study/sinkprofile", "PR9 P4 z rebuild (12 models)"), ("PR10_P4_profile", "alignment_study/sinkprofile_pr10", "PR10 P4 z rebuild (4 new designs)")):
    g, nm = [], []
    for p in sorted(glob.glob(f"{directory}/*.json")):
        c = json.load(open(p))
        if isinstance(c, dict) and "rows" in c:
            g.append(xy(c["rows"], "z_real", "z_profile")); nm.append(c["model"])
    res["stats"][key] = analyse(label, g, nm)
    gd = [(xy(json.load(open(p))["rows"], "z_real", "z_dense")) for p in sorted(glob.glob(f"{directory}/*.json")) if "rows" in json.load(open(p))]
    res["stats"][key + "_dense_control"] = analyse(label.replace("z rebuild", "dense control"), gd, nm)

# PR8 derivation G1 (R2 linear, high-sink) and G2 (Spearman, all layers)
g1, g2, nm = [], [], []
for p in sorted(glob.glob("alignment_study/derivation/*.json")):
    c = json.load(open(p))
    if isinstance(c, dict) and "rows" in c:
        h = hs_rows(c["rows"]); nm.append(c["model"]); g2.append(xy(c["rows"], "derived_ideal", "measured"))
        g1.append(xy(h, "derived_ideal", "measured") if h else (np.array([]), np.array([])))
keep = [i for i, x in enumerate(g1) if len(x[0]) >= 2]
res["stats"]["PR8_G1_gate_R2"] = analyse("PR8 G1 gate R2 (high-sink layers)", [g1[i] for i in keep], [nm[i] for i in keep], stat=r2lin, note="models with at least two high-sink layers")
res["stats"]["PR8_G2_all_layers"] = analyse("PR8 G2 Spearman (all layers)", g2, nm)

# PR7 merge M2
g, nm = [], []
for p in sorted(glob.glob("alignment_study/merge/*.json")):
    c = json.load(open(p))
    if isinstance(c, dict) and "rows" in c:
        g.append(xy(c["rows"], "sharedE", "dloss_top")); nm.append(c["model"])
res["stats"]["PR7_M2_sharedE_vs_merge_cost"] = analyse("PR7 M2 shared energy vs merge cost", g, nm)

json.dump(res, open("alignment_study/clustered_stats.json", "w"), indent=1)
print("CLUSTERED_DONE alignment_study/clustered_stats.json")
