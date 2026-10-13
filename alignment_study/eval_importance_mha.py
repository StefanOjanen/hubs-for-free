# Evaluation of PREREGISTRATION15, committed before any confirmatory result
# exists. Preregistration 12's S2, S3a, S3b and S4 with the same thresholds, on
# multi-head-attention decoders, with S3 and S4 judged on each of the two
# evaluation corpora (WikiText-103 test, C4 validation). Two-stage cluster
# bootstrap (models, then layers), 2,000 draws, seed 0.
#   python alignment_study/eval_importance_mha.py [directory]
import glob
import json
import sys
import numpy as np
from scipy.stats import spearmanr

D = (sys.argv[1] if len(sys.argv) > 1 else "alignment_study/importance_mha").rstrip("/")
HEAD = ("conf", "neg_entropy", "sink_mass", "out_norm")
CORPORA = ("wikitext", "c4")
HS, B, SEED = 0.4, 2000, 0
TH = {"S2_rho": 0.5, "S2_ci_low": 0.3, "S3_delta": 0.3, "S4_rho": 0.4}


def cluster_ci(groups):
    groups = [np.asarray(g, float) for g in groups if len(g)]
    rg = np.random.default_rng(SEED); vals = []
    for _ in range(B):
        pick = rg.integers(0, len(groups), len(groups))
        vals.append(np.mean(np.concatenate([groups[k][rg.integers(0, len(groups[k]), len(groups[k]))] for k in pick])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def pooled(groups):
    allv = np.concatenate([np.asarray(x, float) for x in groups if len(x)])
    return {"value": float(np.mean(allv)), "ci95": cluster_ci(groups), "n_layers": int(allv.size), "n_models": sum(1 for x in groups if len(x))}


models = {}
for p in sorted(glob.glob(f"{D}/*.json")):
    c = json.load(open(p)); models[c["model"]] = c
G = {"conf_sink": []}
for c_ in CORPORA:
    for k in ("d_conf", "d_ent") + tuple("rho_" + h for h in HEAD):
        G[f"{c_}_{k}"] = []
res = {"thresholds": TH, "per_model": {}, "S": {}}
for name, c in models.items():
    L, n = c["layers"], c["heads"]; cs = []
    for l in range(L):
        hs = c["stats"][str(l)]
        if np.mean([h["sink_mass"] for h in hs]) > HS:
            cs.append(float(spearmanr([h["conf"] for h in hs], [h["sink_mass"] for h in hs])[0]))
    G["conf_sink"].append(cs); pm = {"n_high_sink": len(cs), "restore_check_kl": c["restore_check_kl"]}
    for c_ in CORPORA:
        r = {h: [] for h in HEAD}; px = {h: [] for h in HEAD}; py = []
        for l in range(L):
            hs = c["stats"][str(l)]; kl = [c["ablation"][c_][str(l)][str(h)]["kl"] for h in range(n)]
            for h in HEAD:
                x = [hs[i][h] for i in range(n)]; r[h].append(float(spearmanr(x, kl)[0])); px[h] += x
            py += kl
        for h in HEAD:
            G[f"{c_}_rho_{h}"].append(r[h])
        G[f"{c_}_d_conf"].append([a - b for a, b in zip(r["out_norm"], r["conf"])]); G[f"{c_}_d_ent"].append([a - b for a, b in zip(r["out_norm"], r["neg_entropy"])])
        pm[c_] = {"rho_mean": {h: float(np.mean(v)) for h, v in r.items()}, "rho_pooled_over_heads": {h: float(spearmanr(px[h], py)[0]) for h in HEAD}}
    res["per_model"][name] = pm
S = res["S"]
S["S2"] = pooled(G["conf_sink"]); S["S2"]["pass_"] = S["S2"]["value"] >= TH["S2_rho"] and S["S2"]["ci95"][0] >= TH["S2_ci_low"]
for c_ in CORPORA:
    a = pooled(G[f"{c_}_d_conf"]); a["pass_"] = a["value"] >= TH["S3_delta"] and a["ci95"][0] > 0; S[f"S3a_{c_}"] = a
    b = pooled(G[f"{c_}_d_ent"]); b["pass_"] = b["value"] >= TH["S3_delta"] and b["ci95"][0] > 0; S[f"S3b_{c_}"] = b
    d = pooled(G[f"{c_}_rho_out_norm"]); d["pass_"] = d["value"] >= TH["S4_rho"] and d["ci95"][0] > 0; S[f"S4_{c_}"] = d
S["primary_passes"] = int(sum(bool(v.get("pass_")) for v in S.values() if isinstance(v, dict)))
res["secondary"] = {f"{c_}_rho_{h}": pooled(G[f"{c_}_rho_{h}"]) for c_ in CORPORA for h in HEAD}
json.dump(res, open(D + "_results.json", "w"), indent=1)
for k, v in S.items():
    print(k, json.dumps(v)[:240])
for c_ in CORPORA:
    print(c_, {h: round(res["secondary"][f"{c_}_rho_{h}"]["value"], 3) for h in HEAD})
for m, pm in res["per_model"].items():
    print(f"  {m.split('/')[-1]:16s} high-sink {pm['n_high_sink']} | " + " | ".join(f"{c_}: " + " ".join(f"{h} {v:+.2f}" for h, v in pm[c_]['rho_mean'].items()) for c_ in CORPORA))
