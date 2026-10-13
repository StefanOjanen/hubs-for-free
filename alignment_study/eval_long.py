# Evaluation of PREREGISTRATION14 (preregistration 12's clauses S2 to S7 at
# 1,024 tokens), committed before any result exists. Same statistics and
# thresholds as eval_standard.py; with two models, "every model" means both and
# S7 requires both. Reads alignment_study/standard_long/, writes
# standard_long_results.json.
import glob
import json
import numpy as np
from scipy.stats import spearmanr

D = "alignment_study/standard_long"
HEAD = ("conf", "neg_entropy", "sink_mass", "out_norm")
CRITERIA = ("out_pert", "val_dist", "tv_raw", "gen_cos", "js", "cos_map", "corr_last", "corr_rows")
HS, B, SEED = 0.4, 2000, 0
TH = {"S2_rho": 0.5, "S2_ci_low": 0.3, "S3_delta": 0.3, "S4_rho": 0.4, "S5_ratio": 0.8, "S7_rank": 2}


def cluster_ci(groups, stat=np.mean):
    groups = [np.asarray(g, float) for g in groups if len(g)]
    rg = np.random.default_rng(SEED); vals = []
    for _ in range(B):
        pick = rg.integers(0, len(groups), len(groups))
        vals.append(stat(np.concatenate([groups[k][rg.integers(0, len(groups[k]), len(groups[k]))] for k in pick])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def pooled(groups):
    allv = np.concatenate([np.asarray(x, float) for x in groups if len(x)])
    return {"value": float(np.mean(allv)), "ci95": cluster_ci(groups), "n_layers": int(allv.size)}


models = {json.load(open(p))["model"]: json.load(open(p)) for p in sorted(glob.glob(f"{D}/*.json"))}
G = {"conf_sink": [], "d_conf": [], "d_ent": [], "out_norm": [], "conf": []}
res = {"thresholds": TH, "per_model": {}, "S": {}}
for name, c in models.items():
    cs = []
    for l in range(c["layers"]):
        hs = c["stats"][str(l)]["heads"]
        if np.mean([h["sink_mass"] for h in hs]) > HS:
            cs.append(float(spearmanr([h["conf"] for h in hs], [h["sink_mass"] for h in hs])[0]))
    rB = {h: [] for h in HEAD}
    for ls, a in c["ablation"].items():
        heads = sorted(a, key=int); kl = [a[h]["kl"] for h in heads]; st = c["stats"][ls]["heads"]
        for h in HEAD:
            rB[h].append(float(spearmanr([st[int(k)][h] for k in heads], kl)[0]))
    G["conf_sink"].append(cs); G["out_norm"].append(rB["out_norm"]); G["conf"].append(rB["conf"])
    G["d_conf"].append([a - b for a, b in zip(rB["out_norm"], rB["conf"])]); G["d_ent"].append([a - b for a, b in zip(rB["out_norm"], rB["neg_entropy"])])
    C = {}
    for b, d in c["multi"].items():
        kls = {k: d[k]["kl"] for k in CRITERIA}; order = sorted(kls, key=kls.get); rnd = float(np.median([r["kl"] for r in d["random"]]))
        C[b] = {"kl": kls, "random_median": rnd, "ratio_out_over_gen": kls["out_pert"] / kls["gen_cos"], "ratio_out_over_corr_last": kls["out_pert"] / kls["corr_last"],
                "rank_out_pert": order.index("out_pert") + 1, "best": order[0]}
    res["per_model"][name] = {"restore_check_kl": c["restore_check_kl"], "n_high_sink": len(cs), "B_rho_mean": {h: float(np.mean(v)) for h, v in rB.items()}, "C": C}
S = res["S"]; pm = res["per_model"]
S["S2"] = pooled(G["conf_sink"]); S["S2"]["pass_"] = S["S2"]["value"] >= TH["S2_rho"] and S["S2"]["ci95"][0] >= TH["S2_ci_low"]
S["S3a"] = pooled(G["d_conf"]); S["S3a"]["pass_"] = S["S3a"]["value"] >= TH["S3_delta"] and S["S3a"]["ci95"][0] > 0
S["S3b"] = pooled(G["d_ent"]); S["S3b"]["pass_"] = S["S3b"]["value"] >= TH["S3_delta"] and S["S3b"]["ci95"][0] > 0
S["S4"] = pooled(G["out_norm"]); S["S4"]["pass_"] = S["S4"]["value"] >= TH["S4_rho"] and S["S4"]["ci95"][0] > 0
r5 = {m: v["C"]["1"]["ratio_out_over_gen"] for m, v in pm.items()}; r6 = {m: v["C"]["1"]["ratio_out_over_corr_last"] for m, v in pm.items()}
S["S5"] = {"per_model": r5, "geo_mean": float(np.exp(np.mean(np.log(list(r5.values()))))), }
S["S5"]["pass_"] = all(x < 1 for x in r5.values()) and S["S5"]["geo_mean"] <= TH["S5_ratio"]
S["S6"] = {"per_model": r6, "geo_mean": float(np.exp(np.mean(np.log(list(r6.values())))))}
S["S6"]["pass_"] = all(x < 1 for x in r6.values()) and S["S6"]["geo_mean"] <= TH["S5_ratio"]
S["S7"] = {"per_model": {m: v["C"]["1"]["rank_out_pert"] for m, v in pm.items()}}
S["S7"]["pass_"] = all(r <= TH["S7_rank"] for r in S["S7"]["per_model"].values())
S["primary_passes"] = int(sum(bool(v.get("pass_")) for v in S.values() if isinstance(v, dict)))
res["secondary"] = {"conf_against_ablation": pooled(G["conf"])}
json.dump(res, open(D + "_results.json", "w"), indent=1)
for k, v in S.items():
    print(k, json.dumps(v)[:300])
for m, v in pm.items():
    print(f"  {m.split('/')[-1]:22s} hs {v['n_high_sink']} | B " + " ".join(f"{h} {x:+.2f}" for h, x in v["B_rho_mean"].items()) +
          f" | C1 out/gen {v['C']['1']['ratio_out_over_gen']:.2f} out/corr_last {v['C']['1']['ratio_out_over_corr_last']:.2f} rank {v['C']['1']['rank_out_pert']} best {v['C']['1']['best']}")
