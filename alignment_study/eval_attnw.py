# Evaluator for PREREGISTRATION31 (committed before any confirmatory result). Reads alignment_study/attnw/*.json
# (written by attnw_round.py --registered=...) and writes alignment_study/attnw_results.json.
#   python alignment_study/eval_attnw.py [--dir=alignment_study/attnw]
import glob
import json
import math
import sys

DIR = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")), "alignment_study/attnw")
MODELS = ["OLMo-1B-hf", "h2o-danube-1.8b-base", "Mistral-7B-Instruct-v0.2", "Llama-2-7b-hf", "Qwen2.5-7B"]
ROUND28_FULL = 16.78                                    # round 28, fusion on matched pairs, WikiText-2 whole test at 2,048

if __name__ == "__main__":
    R = {m: json.load(open(f"{DIR}/{m}.json")) for m in MODELS if glob.glob(f"{DIR}/{m}.json")}
    res = {"missing": [m for m in MODELS if m not in R]}
    per = {m: {"base": r["base"], "plain": r["plain"]["ppl"], "attention": r["attention"]["ppl"],
               "lower": r["attention"]["ppl"] < r["plain"]["ppl"], "pairing": r["pairing"],
               "first_position_share": sum(c["first"] for c in r["concentration"].values()) / len(r["concentration"]),
               "received_entropy": sum(c["entropy"] for c in r["concentration"].values()) / len(r["concentration"])}
           for m, r in R.items()}
    res["A1"] = {"pass": len(R) == 5 and all(v["lower"] for v in per.values()), "lower_in": sum(v["lower"] for v in per.values()), "of": len(R)}
    num = sum(math.log(v["attention"] / v["base"]) for v in per.values())
    den = sum(math.log(v["plain"] / v["base"]) for v in per.values())
    res["A2"] = {"pass": len(R) == 5 and den > 0 and num / den <= 0.90, "pooled_ratio": num / den if den else None,
                 "per_model_ratio": {m: math.log(v["attention"] / v["base"]) / math.log(v["plain"] / v["base"]) for m, v in per.items()}}
    ll = R.get("Llama-2-7b-hf")
    res["A3"] = {"pass": bool(ll) and ll["attention"]["ppl_full_2048"] < ROUND28_FULL, "round28": ROUND28_FULL,
                 "attention_full_2048": ll["attention"]["ppl_full_2048"] if ll else None,
                 "plain_full_2048": ll["plain"]["ppl_full_2048"] if ll else None}
    res["per_model"] = per
    res["passes"] = sum(bool(res[k]["pass"]) for k in ("A1", "A2", "A3"))
    res["of"] = 3
    json.dump(res, open("alignment_study/attnw_results.json", "w"), indent=1)
    for k in ("A1", "A2", "A3"):
        print(k, "PASS" if res[k]["pass"] else "FAIL", {kk: vv for kk, vv in res[k].items() if kk != "pass"})
    for m, v in per.items():
        print(m, v)
    print("passes", res["passes"], "of", res["of"], "missing", res["missing"])
