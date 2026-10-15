# Evaluation of alignment_study/PREREGISTRATION22.md, committed before any
# confirmatory result exists.
#   python alignment_study/eval_merge_bench2.py [directory]
import glob
import json
import sys
import numpy as np
from scipy.stats import spearmanr

D = (sys.argv[1] if len(sys.argv) > 1 else "alignment_study/merge_bench2").rstrip("/")
CRIT = ("out_pert", "corr_last", "corr_rows", "js", "cos_map", "qk_wcos", "kv_wcos", "k_act", "v_act")
SIM = {"corr_last", "corr_rows", "cos_map", "qk_wcos", "kv_wcos", "k_act", "v_act"}
B, SEED = 2000, 0


def cluster_ci(groups):
    groups = [np.asarray(g, float) for g in groups]; rg = np.random.default_rng(SEED); vals = []
    for _ in range(B):
        pick = rg.integers(0, len(groups), len(groups))
        vals.append(np.mean(np.concatenate([groups[k][rg.integers(0, len(groups[k]), len(groups[k]))] for k in pick])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


S = [json.load(open(p)) for p in sorted(glob.glob(f"{D}/S_*.json"))]
rho = {c: [] for c in CRIT}
for m in S:
    r = {c: [] for c in CRIT}
    for l in m["single"]:
        pairs = list(m["single"][l]); kl = [m["single"][l][p]["kl"] for p in pairs]
        for c in CRIT:
            v = [m["stats"][l][p][c] for p in pairs]; r[c].append(float(spearmanr([-x for x in v] if c in SIM else v, kl)[0]))
    for c in CRIT:
        rho[c].append(r[c])
pooled = {c: {"value": float(np.mean(np.concatenate(rho[c]))), "ci95": cluster_ci(rho[c]), "per_model": [float(np.mean(x)) for x in rho[c]]} for c in CRIT}
others = [c for c in CRIT if c != "out_pert"]; best = max(others, key=lambda c: pooled[c]["value"])
diff = [np.array(a) - np.array(b) for a, b in zip(rho["out_pert"], rho[best])]
dv = {"value": float(np.mean(np.concatenate(diff))), "ci95": cluster_ci(diff)}
r1 = [m["multi"]["1"]["out_pert"]["kl"] / m["multi"]["1"]["corr_last"]["kl"] for m in S]
C = {
    "S1": {"what": "new models: out_pert within-layer Spearman >= 0.4, interval above zero", "value": pooled["out_pert"], "pass_": pooled["out_pert"]["value"] >= 0.4 and pooled["out_pert"]["ci95"][0] > 0},
    "S2": {"what": f"exceeds the best other criterion ({best}) by >= 0.2, interval above zero", "value": dv, "pass_": dv["value"] >= 0.2 and dv["ci95"][0] > 0},
    "S3": {"what": "every other criterion below 0.3", "value": {c: pooled[c]["value"] for c in others}, "pass_": all(pooled[c]["value"] < 0.3 for c in others)},
    "S4": {"what": "one pair per layer: out_pert KL below the CHAI-style choice in every new model, geometric mean ratio <= 0.8", "value": r1,
           "pass_": all(x < 1 for x in r1) and float(np.exp(np.mean(np.log(r1)))) <= 0.8},
}
Dd = [json.load(open(p)) for p in sorted(glob.glob(f"{D}/D_*.json"))]
drop = lambda m, k, c: m["base"] - m["multi"][k][c]["acc"]
for k, key in (("1", "D1"), ("2", "D2")):
    wins = [drop(m, k, "out_pert") < drop(m, k, "corr_last") for m in Dd]
    C[key] = {"what": f"{k} pair(s) per layer: out_pert's LAMBADA accuracy drop below the CHAI-style choice's in at least 5 of 6 models",
              "value": {m["model"]: [drop(m, k, "out_pert"), drop(m, k, "corr_last")] for m in Dd}, "pass_": sum(wins) >= 5}
xs, ys = [], []
for m in Dd:
    for k, v in m["multi"].items():
        for c, x in v.items():
            if x["kl"] is not None:
                xs.append(m["base"] - x["acc"]); ys.append(x["kl"])
rs = float(spearmanr(xs, ys)[0])
C["D3"] = {"what": "accuracy drop tracks KL damage across configurations: Spearman >= 0.5", "value": {"spearman": rs, "configs": len(xs)}, "pass_": rs >= 0.5}
res = {"clauses": C, "passes": int(sum(v["pass_"] for v in C.values())), "pooled": pooled, "best_other": best,
       "checks": {m["model"]: [m["capture_equivalence_max_logit_diff"], m["restore_check_kl"]] for m in S} | {m["model"] + " (D)": [m["base"], m["restore_check_acc"]] for m in Dd},
       "downstream": {m["model"]: {"base": m["base"], **{f"{k}:{c}": v["acc"] for k in m["multi"] for c, v in m["multi"][k].items()}} for m in Dd}}
json.dump(res, open(D + "_results.json", "w"), indent=1)
for k, v in C.items():
    print(k, "PASS" if v["pass_"] else "FAIL", v["what"], json.dumps(v["value"])[:220])
print("passes", res["passes"], "of", len(C))
