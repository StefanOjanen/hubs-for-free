# Evaluation of audits/PREREGISTRATION17.md, committed before any confirmatory
# result exists. Reads clark_source_format_result.json (Part K) and
# chai_every_row_<model>.json (Part C), applies the registered clauses and
# writes audits/readings/readings_results.json.
#   python audits/readings/eval_readings.py [directory]
import json
import os
import numpy as np

import sys
D = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "audits/readings"
TARGET1_SEPARATOR_SHARE = 0.6379688      # audits/clark2019/battery_result.json, c_colfix shrinkage (rerun after addendum 9)
CHANCE_NN = 11 / 143                     # same-layer nearest neighbor by chance, 12 layers of 12 heads
TH = {"K1_band": [TARGET1_SEPARATOR_SHARE - 0.15, TARGET1_SEPARATOR_SHARE + 0.15], "K2_max": 0.25, "K3_abs_max": 0.05,
      "C12_min": 0.5, "C3_max": 0.2, "C4_max": 0.1}
CHAI = ("Mistral-7B-v0.1", "opt-6.7b")

res = {"thresholds": TH, "clauses": {}, "reported": {}}
cl = res["clauses"]
pK = f"{D}/clark_source_format_result.json"
if os.path.exists(pK):
    K = json.load(open(pK)); real = K["real"]; nu = K["nulls"]
    cl["K0"] = {"D": real["D"], "nn_same_layer": real["nn_same_layer"], "chance": CHANCE_NN,
                "pass_": real["D"] > 0 and real["nn_same_layer"] >= 2 * CHANCE_NN}
    s = nu["c_colfix"]["shrinkage"]
    cl["K1"] = {"separator_share": s, "target1": TARGET1_SEPARATOR_SHARE, "pass_": TH["K1_band"][0] <= s <= TH["K1_band"][1]}
    cl["K2"] = {"marginal_share": nu["b_marginal"]["shrinkage"], "pass_": nu["b_marginal"]["shrinkage"] < TH["K2_max"]}
    cl["K3"] = {"random_share": nu["a_random"]["shrinkage"], "untrained_share": nu["d_untrained"]["shrinkage"],
                "pass_": abs(nu["a_random"]["shrinkage"]) < TH["K3_abs_max"] and abs(nu["d_untrained"]["shrinkage"]) < TH["K3_abs_max"]}
    cl["K4"] = {"label": K["outcome"], "target1_label": "shrinks", "pass_": K["outcome"] == "shrinks"}
    res["reported"]["K"] = {"model": K["model"], "segments": K["n_segments"], "tokens_mean": K["tokens_mean"], "share_at_128": K["share_at_128"],
                            "nn_medians": {f: v["nn_median"] for f, v in nu.items()}, "percentiles_of_real": {f: v["percentile_of_real"] for f, v in nu.items()}}
C = {m: json.load(open(f"{D}/chai_every_row_{m}.json")) for m in CHAI if os.path.exists(f"{D}/chai_every_row_{m}.json")}
if len(C) == len(CHAI):
    sm = {m: c["summary"] for m, c in C.items()}
    cl["C1"] = {"colset_median": sm["Mistral-7B-v0.1"]["c_colset"]["median"], "pass_": sm["Mistral-7B-v0.1"]["c_colset"]["median"] >= TH["C12_min"]}
    cl["C2"] = {"colset_median": sm["opt-6.7b"]["c_colset"]["median"], "pass_": sm["opt-6.7b"]["c_colset"]["median"] >= TH["C12_min"]}
    cl["C3"] = {"marginal_median": {m: s["b_marginal"]["median"] for m, s in sm.items()}, "pass_": all(s["b_marginal"]["median"] <= TH["C3_max"] for s in sm.values())}
    cl["C4"] = {"untrained_median": {m: s["d_untrained"]["median"] for m, s in sm.items()}, "pass_": all(s["d_untrained"]["median"] <= TH["C4_max"] for s in sm.values())}
    rep = {}
    for m, c in C.items():
        reg = json.load(open(f"audits/chai/battery_result_{m}.json"))
        last_rep = [L["nulls"]["c_colset"]["reproduced_mean_corr"] for L in reg["layers"]]
        rep[m] = {"colset_by_layer": [round(L["nulls"]["c_colset"]["reproduced"], 3) for L in c["layers"]],
                  "colset_min_max": [sm[m]["c_colset"]["min"], sm[m]["c_colset"]["max"]],
                  "target3_last_row_colset_median": float(np.median(last_rep)),
                  "every_row_corr_range": [min(L["real"]["mean_corr"] for L in c["layers"]), max(L["real"]["mean_corr"] for L in c["layers"])],
                  "depth_spearman_every_row": sm[m]["depth_spearman_every_row"], "depth_spearman_last_row": sm[m]["depth_spearman_last_row"],
                  "layers_majority_95": sm[m]["layers_majority_95_every_row"], "layers_majority_90": sm[m]["layers_majority_90_every_row"],
                  "cosine_real_median": float(np.median([L["real_cosine"]["mean_corr"] for L in c["layers"]])),
                  "last_row_check_max_abs_diff": sm[m].get("last_row_check_max_abs_diff")}
    res["reported"]["C"] = rep
res["passes"] = int(sum(bool(v["pass_"]) for v in cl.values())); res["n_clauses"] = len(cl)
json.dump(res, open(f"{D}/readings_results.json", "w"), indent=1)
for k, v in cl.items():
    print(k, "PASS" if v["pass_"] else "FAIL", json.dumps({a: b for a, b in v.items() if a != "pass_"})[:200])
print("passes", res["passes"], "of", res["n_clauses"])
for part, r in res["reported"].items():
    print(part, json.dumps(r)[:600])
