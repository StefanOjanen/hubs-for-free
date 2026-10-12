# Evaluation of PREREGISTRATION12 (standard-statistics round), committed
# before any confirmatory result exists. Reads alignment_study/standard/<model>.json
# (or the directory given), writes <dir>_results.json, prints the scorecard.
# Intervals: two-stage cluster bootstrap (models, then layers), 2,000 draws,
# seed 0. Part C has one value per model and is judged per model.
#   python alignment_study/eval_standard.py [directory]
import glob
import json
import sys
import numpy as np
from scipy.stats import spearmanr

D = (sys.argv[1] if len(sys.argv) > 1 else "alignment_study/standard").rstrip("/")
OUTFILE = D + "_results.json"
MODELS = ["Qwen/Qwen2.5-1.5B", "TinyLlama/TinyLlama_v1.1", "mistralai/Mistral-7B-v0.1", "Qwen/Qwen2.5-7B"]
SIMILARITY = {"gen_cos", "cos_map", "corr_last", "corr_rows"}
FIELD = ("js", "cos_map", "corr_last", "corr_rows")
PAIR_ALL = ("out_pert", "val_dist", "tv_raw", "gen_cos") + FIELD
HEAD = ("conf", "neg_entropy", "sink_mass", "out_norm")
CRITERIA = ("out_pert", "val_dist", "tv_raw", "gen_cos") + FIELD
HS = 0.4
B, SEED = 2000, 0
# Frozen with PREREGISTRATION12.md, about half the development effects
# (alignment_study/standard_dev/, Qwen2.5-0.5B, labeled).
TH = {"S1_delta": 0.10,              # out_pert minus each field statistic, merge KL (dev 0.20 to 0.35)
      "S2_rho": 0.5, "S2_ci_low": 0.3,   # confidence against sink mass, high-sink layers (dev 0.86)
      "S3_delta": 0.3,               # out_norm minus confidence, and minus negative entropy, ablation KL (dev 0.90 and 0.81)
      "S4_rho": 0.4,                 # out_norm against ablation KL (dev 0.77)
      "S5_ratio": 0.8,               # multi-merge KL ratios, geometric mean over models, each model below 1 (dev 0.45 and 0.39)
      "S7_rank": 2, "S7_models": 3}  # out_pert among the two best of eight criteria in at least 3 of 4 models (dev rank 1)


def dist(p, v):
    return -np.asarray(v, float) if p in SIMILARITY else np.asarray(v, float)


def cluster_ci(groups, stat=np.mean):
    groups = [np.asarray(g, float) for g in groups if len(g)]
    rg = np.random.default_rng(SEED); vals = []
    for _ in range(B):
        pick = rg.integers(0, len(groups), len(groups))
        vals.append(stat(np.concatenate([groups[k][rg.integers(0, len(groups[k]), len(groups[k]))] for k in pick])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def pooled(groups, stat=np.mean):
    allv = np.concatenate([np.asarray(x, float) for x in groups if len(x)])
    return {"value": float(stat(allv)), "ci95": cluster_ci(groups, stat), "n_layers": int(allv.size), "n_models": sum(1 for x in groups if len(x))}


def pair_outcomes(c):
    src = json.load(open(c["plan"]["pair_outcomes"]))
    return src["sweep"]


models = {}
for p in sorted(glob.glob(f"{D}/*.json")):
    c = json.load(open(p))
    if isinstance(c, dict) and "stats" in c and "multi" in c:
        models[c["model"]] = c
if not models:
    raise SystemExit(f"no model results in {D}")
res = {"thresholds": TH, "n_models": len(models), "complete": all(m in models for m in MODELS), "per_model": {}, "S": {}}
G = {k: [] for k in ["rhoA_" + p for p in PAIR_ALL] + ["dA_" + f for f in FIELD] +
     ["conf_sink"] + ["rhoB_" + h for h in HEAD] + ["dB_conf", "dB_ent"] + ["rhoB_pooled_" + h for h in HEAD]}
for name, c in models.items():
    pm = {"n_layers": c["layers"], "restore_check_kl": c["restore_check_kl"]}
    # Part A: pair statistics against the round-11 merge KL of the same model
    sw = pair_outcomes(c); rA = {p: [] for p in PAIR_ALL}
    for ls, s in sw.items():
        keys = list(s); kl = np.array([s[k]["kl"] for k in keys])
        for p in PAIR_ALL:
            rA[p].append(float(spearmanr(dist(p, [c["stats"][ls]["pairs"][k][p] for k in keys]), kl)[0]))
    for p in PAIR_ALL:
        G["rhoA_" + p].append(rA[p])
    for f in FIELD:
        G["dA_" + f].append([a - b for a, b in zip(rA["out_pert"], rA[f])])
    pm["A_rho_mean"] = {p: float(np.mean(v)) for p, v in rA.items()}
    # Part B: confidence against sink mass (selection statistics, high-sink layers), head statistics against ablation KL
    cs = []
    for l in range(c["layers"]):
        hs = c["stats"][str(l)]["heads"]
        if np.mean([h["sink_mass"] for h in hs]) > HS:
            cs.append(float(spearmanr([h["conf"] for h in hs], [h["sink_mass"] for h in hs])[0]))
    G["conf_sink"].append(cs); pm["B_conf_sink_rho_mean"] = float(np.mean(cs)) if cs else None; pm["B_n_high_sink"] = len(cs)
    rB = {h: [] for h in HEAD}; pool_x = {h: [] for h in HEAD}; pool_y = []
    for ls, a in c["ablation"].items():
        heads = sorted(a, key=int); kl = np.array([a[h]["kl"] for h in heads]); st = c["stats"][ls]["heads"]
        for h in HEAD:
            x = [st[int(k)][h] for k in heads]
            rB[h].append(float(spearmanr(x, kl)[0])); pool_x[h] += x
        pool_y += list(kl)
    for h in HEAD:
        G["rhoB_" + h].append(rB[h]); G["rhoB_pooled_" + h].append([float(spearmanr(pool_x[h], pool_y)[0])])
    G["dB_conf"].append([a - b for a, b in zip(rB["out_norm"], rB["conf"])]); G["dB_ent"].append([a - b for a, b in zip(rB["out_norm"], rB["neg_entropy"])])
    pm["B_rho_mean"] = {h: float(np.mean(v)) for h, v in rB.items()}
    pm["B_rho_pooled_over_heads"] = {h: float(spearmanr(pool_x[h], pool_y)[0]) for h in HEAD}
    # Part C: multi-pair merges
    pm["C"] = {}
    for b, d in c["multi"].items():
        rnd = float(np.median([r["kl"] for r in d["random"]]))
        kls = {k: d[k]["kl"] for k in CRITERIA}
        order = sorted(kls, key=kls.get)
        pm["C"][b] = {"kl": kls, "random_median": rnd, "ratio_out_over_gen": kls["out_pert"] / kls["gen_cos"],
                      "ratio_out_over_corr_last": kls["out_pert"] / kls["corr_last"], "ratio_out_over_random": kls["out_pert"] / rnd,
                      "rank_out_pert": order.index("out_pert") + 1, "best": order[0], "n_merged": d["out_pert"]["n_merged"]}
    res["per_model"][name] = pm

TH_ = TH or {}
S = res["S"]
S["S1_out_pert_minus_field_statistics"] = {f: pooled(G["dA_" + f]) for f in FIELD}
S["S2_confidence_vs_sink_mass"] = pooled(G["conf_sink"])
S["S3a_out_norm_minus_confidence"] = pooled(G["dB_conf"])
S["S3b_out_norm_minus_neg_entropy"] = pooled(G["dB_ent"])
S["S4_out_norm_rho"] = pooled(G["rhoB_out_norm"])
C1 = {m: pm["C"]["1"] for m, pm in res["per_model"].items()}
S["S5_multi_merge_out_over_gen_budget1"] = {"per_model": {m: v["ratio_out_over_gen"] for m, v in C1.items()},
                                            "geo_mean": float(np.exp(np.mean(np.log([v["ratio_out_over_gen"] for v in C1.values()]))))}
S["S6_multi_merge_out_over_corr_last_budget1"] = {"per_model": {m: v["ratio_out_over_corr_last"] for m, v in C1.items()},
                                                  "geo_mean": float(np.exp(np.mean(np.log([v["ratio_out_over_corr_last"] for v in C1.values()]))))}
S["S7_multi_merge_out_pert_rank_budget1"] = {m: v["rank_out_pert"] for m, v in C1.items()}
if TH:
    S["S1_out_pert_minus_field_statistics"]["pass_"] = all(v["value"] >= TH["S1_delta"] and v["ci95"][0] > 0 for f, v in S["S1_out_pert_minus_field_statistics"].items() if f in FIELD)
    S["S2_confidence_vs_sink_mass"]["pass_"] = S["S2_confidence_vs_sink_mass"]["value"] >= TH["S2_rho"] and S["S2_confidence_vs_sink_mass"]["ci95"][0] >= TH["S2_ci_low"]
    S["S3a_out_norm_minus_confidence"]["pass_"] = S["S3a_out_norm_minus_confidence"]["value"] >= TH["S3_delta"] and S["S3a_out_norm_minus_confidence"]["ci95"][0] > 0
    S["S3b_out_norm_minus_neg_entropy"]["pass_"] = S["S3b_out_norm_minus_neg_entropy"]["value"] >= TH["S3_delta"] and S["S3b_out_norm_minus_neg_entropy"]["ci95"][0] > 0
    S["S4_out_norm_rho"]["pass_"] = S["S4_out_norm_rho"]["value"] >= TH["S4_rho"] and S["S4_out_norm_rho"]["ci95"][0] > 0
    S["S5_multi_merge_out_over_gen_budget1"]["pass_"] = all(v < 1 for v in S["S5_multi_merge_out_over_gen_budget1"]["per_model"].values()) and S["S5_multi_merge_out_over_gen_budget1"]["geo_mean"] <= TH["S5_ratio"]
    S["S6_multi_merge_out_over_corr_last_budget1"]["pass_"] = all(v < 1 for v in S["S6_multi_merge_out_over_corr_last_budget1"]["per_model"].values()) and S["S6_multi_merge_out_over_corr_last_budget1"]["geo_mean"] <= TH["S5_ratio"]
    S["S7_multi_merge_out_pert_rank_budget1"]["pass_"] = sum(r <= TH["S7_rank"] for m, r in S["S7_multi_merge_out_pert_rank_budget1"].items() if m != "pass_") >= TH["S7_models"]
    S["primary_passes"] = int(sum(bool(v.get("pass_")) for k, v in S.items() if isinstance(v, dict) and k.startswith("S")))
res["secondary"] = {"A_rho_by_statistic": {p: pooled(G["rhoA_" + p]) for p in PAIR_ALL},
                    "B_rho_by_head_statistic": {h: pooled(G["rhoB_" + h]) for h in HEAD},
                    "B_rho_pooled_over_heads_per_model": {h: [x[0] for x in G["rhoB_pooled_" + h]] for h in HEAD}}
json.dump(res, open(OUTFILE, "w"), indent=1)
print("models:", len(models), "complete:", res["complete"])
for k, v in S.items():
    print(k, json.dumps(v)[:360])
print("A rho:", {p: round(v["value"], 3) for p, v in res["secondary"]["A_rho_by_statistic"].items()})
print("B rho:", {h: round(v["value"], 3) for h, v in res["secondary"]["B_rho_by_head_statistic"].items()})
for name, pm in res["per_model"].items():
    c1 = pm["C"]["1"]; c2 = pm["C"].get("2", {})
    print(f"  {name.split('/')[-1]:24s} conf~sink {pm['B_conf_sink_rho_mean']} | B " + " ".join(f"{h} {v:+.2f}" for h, v in pm["B_rho_mean"].items()) +
          f" | C1 out/gen {c1['ratio_out_over_gen']:.2f} out/corr_last {c1['ratio_out_over_corr_last']:.2f} out/random {c1['ratio_out_over_random']:.2f} rank {c1['rank_out_pert']} best {c1['best']}" +
          (f" | C2 out/gen {c2['ratio_out_over_gen']:.2f} rank {c2['rank_out_pert']}" if c2 else ""))
