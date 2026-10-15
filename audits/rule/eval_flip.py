# Evaluation of audits/PREREGISTRATION23.md, committed before any confirmatory
# result exists.
#   python audits/rule/eval_flip.py [directory]
import glob
import json
import sys

D = (sys.argv[1] if len(sys.argv) > 1 else "audits/rule/flip").rstrip("/")
R = {}
for p in sorted(glob.glob(f"{D}/*.json")):
    r = json.load(open(p)); R.update(r["stats"])
PAIRS = {"heavy": ("heavy_mass", "heavy_count"), "persist": ("persist_mass", "persist_count"), "hit": ("hit_mass", "hit_count"),
         "chai": ("corr_last", "jaccard_last"), "clark": ("clark_js", "clark_jaccard")}
C = {}
for k in ("heavy", "persist", "hit", "chai"):
    m, c = PAIRS[k]; d = R[m]["s"] - R[c]["s"]
    C[f"F1_{k}"] = {"what": f"{k}: s(mass) - s(count) >= 0.3", "value": [R[m]["s"], R[c]["s"], d], "pass_": d >= 0.3}
m, c = PAIRS["clark"]; d = R[m]["s"] - R[c]["s"]
C["F2_clark"] = {"what": "clark (top-5 of about 64 positions): |s(mass) - s(count)| < 0.3", "value": [R[m]["s"], R[c]["s"], d], "pass_": abs(d) < 0.3}
simple = {s: {"s": v["s"], "kind": "mass" if s in ("heavy_mass", "persist_mass", "hit_mass", "corr_last", "clark_js") else "count"} for s, v in R.items()}
for s, v in simple.items():
    v["simple_rule_holds"] = (v["s"] >= 0.5) if v["kind"] == "mass" else (v["s"] <= 0.2)
res = {"clauses": C, "passes": int(sum(v["pass_"] for v in C.values())), "secondary_simple_rule": simple,
       "reproduces": {s: v["real"] > max(v["a_random"]) if s != "persist_count" else v["real"] > max(v["a_random"]) for s, v in R.items()}}
json.dump(res, open(D + "_results.json", "w"), indent=1)
for k, v in C.items():
    print(k, "PASS" if v["pass_"] else "FAIL", v["what"], [round(x, 3) for x in v["value"]])
print("passes", res["passes"], "of", len(C))
print("simple rule:", {s: (round(v["s"], 3), v["simple_rule_holds"]) for s, v in simple.items()})
