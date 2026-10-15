# Evaluation of alignment_study/PREREGISTRATION20.md, committed before any
# confirmatory result exists. Within-layer Spearman of each criterion (oriented
# so that larger means more predicted damage) with the single-merge KL; two-stage
# cluster bootstrap (models, then layers), 2,000 draws, seed 0.
#   python alignment_study/eval_merge_bench.py [directory]
import glob
import json
import sys
import numpy as np
from scipy.stats import spearmanr

D = (sys.argv[1] if len(sys.argv) > 1 else "alignment_study/merge_bench").rstrip("/")
CRIT = ("out_pert", "corr_last", "corr_rows", "js", "cos_map", "qk_wcos", "kv_wcos", "k_act", "v_act")
SIM = {"corr_last", "corr_rows", "cos_map", "qk_wcos", "kv_wcos", "k_act", "v_act"}
B, SEED = 2000, 0


def cluster_ci(groups):
    groups = [np.asarray(g, float) for g in groups]; rg = np.random.default_rng(SEED); vals = []
    for _ in range(B):
        pick = rg.integers(0, len(groups), len(groups))
        vals.append(np.mean(np.concatenate([groups[k][rg.integers(0, len(groups[k]), len(groups[k]))] for k in pick])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


models = [json.load(open(p)) for p in sorted(glob.glob(f"{D}/*.json")) if "single" in json.load(open(p))]
rho = {c: [] for c in CRIT}
for m in models:
    r = {c: [] for c in CRIT}
    for l in m["single"]:
        pairs = list(m["single"][l]); kl = [m["single"][l][p]["kl"] for p in pairs]
        for c in CRIT:
            v = [m["stats"][l][p][c] for p in pairs]
            r[c].append(float(spearmanr([-x for x in v] if c in SIM else v, kl)[0]))
    for c in CRIT:
        rho[c].append(r[c])
pooled = {c: {"value": float(np.mean(np.concatenate(rho[c]))), "ci95": cluster_ci(rho[c]), "per_model": [float(np.mean(x)) for x in rho[c]]} for c in CRIT}
others = [c for c in CRIT if c != "out_pert"]
best_other = max(others, key=lambda c: pooled[c]["value"])
diff = [np.array(a) - np.array(b) for a, b in zip(rho["out_pert"], rho[best_other])]
res = {"pooled": pooled, "best_other": best_other, "clauses": {}}
C = res["clauses"]
C["B1"] = {"what": "out_pert within-layer Spearman >= 0.4, interval above zero", "value": pooled["out_pert"], "pass_": pooled["out_pert"]["value"] >= 0.4 and pooled["out_pert"]["ci95"][0] > 0}
dv = {"value": float(np.mean(np.concatenate(diff))), "ci95": cluster_ci(diff)}
C["B2"] = {"what": f"out_pert exceeds the best other criterion ({best_other}) by >= 0.2, interval above zero", "value": dv, "pass_": dv["value"] >= 0.2 and dv["ci95"][0] > 0}
C["B3"] = {"what": "every other criterion below 0.3", "value": {c: pooled[c]["value"] for c in others}, "pass_": all(pooled[c]["value"] < 0.3 for c in others)}
r1 = [m["multi"]["1"]["out_pert"]["kl"] / m["multi"]["1"]["corr_last"]["kl"] for m in models]
C["B4"] = {"what": "one pair per layer: out_pert below the CHAI-style choice in every model, geometric mean ratio <= 0.8", "value": r1,
           "geo_mean": float(np.exp(np.mean(np.log(r1)))), "pass_": all(x < 1 for x in r1) and float(np.exp(np.mean(np.log(r1)))) <= 0.8}
rr = [m["multi"]["1"]["out_pert"]["kl"] / float(np.median([m["multi"]["1"][f"random{i}"]["kl"] for i in range(5)])) for m in models]
C["B5"] = {"what": "one pair per layer: out_pert below the random median in every model", "value": rr, "pass_": all(x < 1 for x in rr)}
res["passes"] = int(sum(v["pass_"] for v in C.values()))
res["multi"] = {m["model"]: {k: {c: v["kl"] for c, v in m["multi"][k].items()} for k in m["multi"]} for m in models}
res["checks"] = {m["model"]: {"capture_equivalence": m["capture_equivalence_max_logit_diff"], "restore": m["restore_check_kl"]} for m in models}
json.dump(res, open(D + "_results.json", "w"), indent=1)
for k, v in C.items():
    print(k, "PASS" if v["pass_"] else "FAIL", v["what"], json.dumps(v["value"])[:200])
print("passes", res["passes"], "of", len(C)); print({c: round(pooled[c]["value"], 3) for c in CRIT})
