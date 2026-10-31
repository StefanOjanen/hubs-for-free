# Evaluator for PREREGISTRATION33 (committed before any registered result). Reads alignment_study/uptrain/<model>.json
# (written by uptrain_round.py --registered=... --model=...) for both models and F from
# alignment_study/uptrain_dev/rule_result.json; writes alignment_study/uptrain33_results.json.
#   python alignment_study/eval_uptrain33.py [--dir=alignment_study/uptrain]
import json
import math
import os
import sys

DIR = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")), "alignment_study/uptrain")
MODELS = ["OLMo-1B-hf", "h2o-danube-1.8b-base"]


def leads(r, arm):
    mean = {c["step"]: c["ppl"] for c in r["arms"]["mean"]["curve"]}
    return [math.log(mean[c["step"]]) - math.log(c["ppl"]) for c in r["arms"][arm]["curve"] if c["step"] in mean]


if __name__ == "__main__":
    F = json.load(open("alignment_study/uptrain_dev/rule_result.json"))["F"]
    R = {m: json.load(open(f"{DIR}/{m}.json")) for m in MODELS if os.path.exists(f"{DIR}/{m}.json")}
    per = {}
    for m, r in R.items():
        lf = leads(r, "fuse")
        per[m] = {"final": {a: r["arms"][a]["curve"][-1]["ppl"] for a in r["arms"]}, "fuse_leads": lf,
                  "retained": lf[-1] / lf[0] if lf[0] > 0 else None,
                  "sens_final": r["arms"]["fuse_sens"]["curve"][-1]["ppl"] if "fuse_sens" in r["arms"] else None,
                  "zeroshot_after": {a: r["arms"][a].get("zeroshot_after", {}).get("average") for a in r["arms"]}}
    both = len(R) == 2
    res = {"F": F, "missing": [m for m in MODELS if m not in R], "per_model": per}
    res["U1"] = {"pass": both and all(p["final"]["fuse"] < p["final"]["mean"] for p in per.values())}
    res["U2"] = {"pass": both and all(all(l > 0 for l in p["fuse_leads"]) for p in per.values())}
    res["U3"] = {"pass": both and all(p["retained"] is not None and p["retained"] >= F for p in per.values()), "retained": {m: p["retained"] for m, p in per.items()}}
    res["U4"] = {"pass": both and all(p["sens_final"] is not None and p["sens_final"] <= p["final"]["fuse"] for p in per.values())}
    res["passes"] = sum(res[k]["pass"] for k in ("U1", "U2", "U3", "U4")); res["of"] = 4
    json.dump(res, open("alignment_study/uptrain33_results.json", "w"), indent=1)
    for k in ("U1", "U2", "U3", "U4"):
        print(k, "PASS" if res[k]["pass"] else "FAIL", {kk: vv for kk, vv in res[k].items() if kk != "pass"})
    for m, p in per.items():
        print(m, "final", {a: round(v, 2) for a, v in p["final"].items()}, "retained", None if p["retained"] is None else round(p["retained"], 3))
    print("passes", res["passes"], "of", res["of"], "missing", res["missing"])
