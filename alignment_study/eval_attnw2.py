# Evaluator for PREREGISTRATION32 (committed before any confirmatory result). Reads alignment_study/attnw2/*.json
# (written by attnw2_round.py --registered=...) and writes alignment_study/attnw2_results.json.
#   python alignment_study/eval_attnw2.py [--dir=alignment_study/attnw2]
import glob
import json
import math
import sys

DIR = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")), "alignment_study/attnw2")
NEW = ["SmolLM2-1.7B", "Llama-3.2-1B", "Mistral-7B-v0.1"]
SEEN = ["OLMo-1B-hf", "h2o-danube-1.8b-base", "Mistral-7B-Instruct-v0.2", "Llama-2-7b-hf", "Qwen2.5-7B"]
ROUND28_FULL = 16.78
THRESH = 0.94                                            # D3's registered ceiling (PREREGISTRATION32.md)


def pooled(per, names):
    num = sum(math.log(per[m]["sens"] / per[m]["base"]) for m in names)
    den = sum(math.log(per[m]["plain"] / per[m]["base"]) for m in names)
    return num / den if den else None


if __name__ == "__main__":
    R = {m: json.load(open(f"{DIR}/{m}.json")) for m in NEW + SEEN if glob.glob(f"{DIR}/{m}.json")}
    r31 = json.load(open("alignment_study/attnw_results.json"))["per_model"]
    q31 = json.load(open("alignment_study/attnw_correction/summary.json"))["qwen_corrected"]
    ref = {m: (q31["plain"] if m == "Qwen2.5-7B" else r31[m]["plain"]) for m in SEEN}
    per = {m: {"base": r["base"], "plain": r["plain"]["ppl"], "sens": r["sens"]["ppl"], "lower": r["sens"]["ppl"] < r["plain"]["ppl"],
               "tier": r["tier"], "pairing": r["pairing"]} for m, r in R.items()}
    res = {"missing": [m for m in NEW + SEEN if m not in R], "per_model": per}
    gate = {m: abs(per[m]["plain"] / ref[m] - 1) for m in SEEN if m in per}
    res["gate"] = {"pass": len(gate) == len(SEEN) and all(v <= 0.001 for v in gate.values()), "relative_difference": gate, "reference": ref}
    allm = [m for m in NEW + SEEN if m in per]
    res["D1"] = {"pass": len(per) == 8 and all(per[m]["lower"] for m in allm), "lower_in": sum(per[m]["lower"] for m in allm), "of": len(allm)}
    newm = [m for m in NEW if m in per]
    res["D2"] = {"pass": len(newm) == 3 and all(per[m]["lower"] for m in newm), "lower_in": sum(per[m]["lower"] for m in newm), "of": len(newm)}
    pr = pooled(per, allm) if allm else None
    res["D3"] = {"pass": len(per) == 8 and pr is not None and pr <= THRESH, "pooled_ratio": pr, "ceiling": THRESH,
                 "pooled_new": pooled(per, newm) if newm else None, "pooled_round31_models": pooled(per, [m for m in SEEN if m in per]) if any(m in per for m in SEEN) else None,
                 "per_model_ratio": {m: math.log(per[m]["sens"] / per[m]["base"]) / math.log(per[m]["plain"] / per[m]["base"]) for m in allm}}
    ll = R.get("Llama-2-7b-hf")
    res["D4"] = {"pass": bool(ll) and ll["sens"]["ppl_full_2048"] < ROUND28_FULL, "round28": ROUND28_FULL,
                 "sens_full_2048": ll["sens"]["ppl_full_2048"] if ll else None, "plain_full_2048": ll["plain"]["ppl_full_2048"] if ll else None}
    scored = res["gate"]["pass"]
    res["passes"] = sum(bool(res[k]["pass"]) for k in ("D1", "D2", "D3", "D4")) if scored else 0
    res["of"] = 4
    res["scored"] = scored
    json.dump(res, open("alignment_study/attnw2_results.json", "w"), indent=1)
    print("gate", "PASS" if res["gate"]["pass"] else "FAIL", {k: round(v, 5) for k, v in gate.items()})
    for k in ("D1", "D2", "D3", "D4"):
        print(k, "PASS" if res[k]["pass"] else "FAIL", {kk: vv for kk, vv in res[k].items() if kk not in ("pass", "per_model_ratio")})
    for m, v in per.items():
        print(m, v)
    print("passes", res["passes"], "of", res["of"], "scored", scored, "missing", res["missing"])
