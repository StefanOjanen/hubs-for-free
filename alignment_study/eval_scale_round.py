# Evaluates PREREGISTRATION3.md predictions S1-S4 over the five scale-round
# models, merging the per-model captures in scale_partial/ (sessions 1-3;
# provenance in RUNLOG.md). Criteria are identical to prereg 2 / the
# committed heldout_round.py evaluation. Writes scale_round_results.json.
import json
import numpy as np

CAPS = ["qwen2.5-3b_capture.json", "qwen2.5-7b_capture.json",
        "mistral-7b_capture.json", "phi-3-mini_capture.json",
        "olmo-2-7b_capture.json"]
models = {}
for fn in CAPS:
    d = json.load(open(f"alignment_study/scale_partial/{fn}"))
    models[d["model"]] = d["per_layer"]

def hs(rows):
    return [r for r in rows if r["smass"] > 0.4]

res = {"S": {}, "per_model": {}}
pooled_hs = []
for name, rows in models.items():
    H = hs(rows)
    pooled_hs += H
    testable = len(H) >= 3
    def frac(cond):
        return float(np.mean([1.0 if cond(r) else 0.0 for r in H])) if H else 0.0
    def ratios(r):
        Rr = abs(r["r1_sinkfix"] - r["r1_real"]) / max(0.05, abs(r["r1_plain"] - r["r1_real"]))
        Rz = abs(r["zmed_sinkfix"] - r["zmed_real"]) / max(1.0, abs(r["zmed_real"] - r["zmed_plain"]))
        return Rr, Rz
    pm = {"testable": testable, "n_high_sink": len(H),
          "S1_frac": frac(lambda r: r["rho_real"] >= 2 * r["rho_plain"] and r["cosS"] > 0.7),
          "S2_frac": frac(lambda r: max(ratios(r)) < 0.4),
          "S3_frac": frac(lambda r: r["r1_alt2"] > 0.8 and r["rho_alt2"] < 0.5 * r["rho_real"]
                          and r["zmed_alt2"] > r["zmed_real"])}
    for k in ("S1", "S2", "S3"):
        pm[k + "_pass"] = bool(testable and pm[k + "_frac"] >= 2 / 3)
    res["per_model"][name] = pm

test_models = [m for m, v in res["per_model"].items() if v["testable"]]
for k in ("S1", "S2", "S3"):
    passes = sum(1 for m in test_models if res["per_model"][m][k + "_pass"])
    res["S"][k] = {"passes": passes, "testable": len(test_models),
                   "pass": bool(len(test_models) and passes >= (2 / 3) * len(test_models))}

def rank(v):
    v = np.array(v); idx = np.argsort(v)
    rk = np.empty(len(v)); rk[idx] = np.arange(len(v))
    return rk

sp = float(np.corrcoef(rank([r["sharedE"] for r in pooled_hs]),
                       rank([r["r1_real"] for r in pooled_hs]))[0, 1])
res["S"]["S4"] = {"spearman": sp, "n_pooled_high_sink_layers": len(pooled_hs),
                  "pass": bool(sp <= -0.5)}
json.dump(res, open("alignment_study/scale_round_results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
