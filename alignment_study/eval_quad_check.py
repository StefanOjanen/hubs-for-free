# Evaluator for PREREGISTRATION34 (committed before any confirmatory result). Reads alignment_study/quad_check/*.json
# (written by quad_check_round.py --registered=...) and writes alignment_study/quad_check_results.json. Each member
# sets layers 0 and 1 to one of unmerged, pairs or groups of four with every other layer at groups of four; its
# damage is its perplexity over the member with both early layers unmerged (the same cost everywhere else).
#   python alignment_study/eval_quad_check.py [--dir=alignment_study/quad_check]
import glob
import json
import os
import sys

DIR = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")), "alignment_study/quad_check")
OUT = "alignment_study/quad_check_results.json" if DIR.rstrip("/") == "alignment_study/quad_check" else DIR.rstrip("/") + "_results.json"   # development runs write beside their directory
SCORED = ["open_llama_3b_v2", "Mistral-7B-v0.1", "Mistral-7B-Instruct-v0.2", "Qwen2.5-7B"]
TAU, SEVERE, BENIGN = 2.35, 5.0, 2.0
FIXED = ("pairs", "quads")                               # the fixed rule: layer 0 at pairs, every other layer at the target ratio
HEADS = {"none": 1.0, "pairs": 0.5, "quads": 0.25}      # key-value heads kept in a layer, as a fraction


def analyse(r):
    mem = [m for m in r["members"] if m["kind"] == "member"]
    ref = next(m["ppl"] for m in mem if m["layer0"] == "none" and m["layer1"] == "none")
    for m in mem:
        m["relative"] = m["ppl"] / ref
        m["severe"], m["benign"] = m["relative"] >= SEVERE, m["relative"] <= BENIGN
        m["cache"] = HEADS[m["layer0"]] + HEADS[m["layer1"]]                # the two early layers' share, in layer-equivalents
    unflagged = [m for m in mem if not m["flag"]]
    chosen = min(unflagged, key=lambda m: (m["cache"], -HEADS[m["layer0"]])) if unflagged else None     # ties: keep layer 0 larger; never the outcome
    fixed = next(m for m in mem if (m["layer0"], m["layer1"]) == FIXED)
    key = lambda m: (m["layer0"], m["layer1"])
    return {"reference_ppl": ref, "severe": [key(m) for m in mem if m["severe"]], "benign": [key(m) for m in mem if m["benign"]],
            "benign_flagged": [key(m) for m in mem if m["benign"] and m["flag"]], "severe_unflagged": [key(m) for m in mem if m["severe"] and not m["flag"]],
            "chosen": key(chosen) if chosen else None, "chosen_severe": chosen["severe"] if chosen else None, "chosen_ppl": chosen["ppl"] if chosen else None,
            "chosen_cache": chosen["cache"] if chosen else None, "fixed": FIXED, "fixed_ppl": fixed["ppl"], "fixed_cache": fixed["cache"], "fixed_severe": fixed["severe"],
            "members": [{k: m[k] for k in ("layer0", "layer1", "ppl", "relative", "max_ratio", "flag", "severe", "benign", "cache")} for m in mem],
            "reported": [{k: m[k] for k in ("kind", "ppl", "max_ratio", "flag")} for m in r["members"] if m["kind"] != "member"]}


if __name__ == "__main__":
    R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{DIR}/*.json")}
    A = {m: analyse(r) for m, r in R.items()}
    scored = {m: a for m, a in A.items() if m in SCORED}
    complete = len(scored) == len(SCORED)                                   # four scored models
    res = {"missing": [m for m in SCORED if m not in R], "per_model": A, "development_model": [m for m in R if m not in SCORED]}
    n_severe, n_benign = sum(len(a["severe"]) for a in scored.values()), sum(len(a["benign"]) for a in scored.values())
    res["R1"] = {"recall_testable": n_severe > 0, "recall_pass": complete and n_severe > 0 and all(not a["severe_unflagged"] for a in scored.values()),
                 "false_alarm_pass": complete and n_benign > 0 and all(not a["benign_flagged"] for a in scored.values()), "severe_members": n_severe, "benign_members": n_benign}
    res["R1"]["pass"] = res["R1"]["false_alarm_pass"] and (res["R1"]["recall_pass"] if res["R1"]["recall_testable"] else True)
    res["R2"] = {"pass": complete and all(a["chosen"] is not None and not a["chosen_severe"] for a in scored.values()),
                 "chosen": {m: a["chosen"] for m, a in scored.items()}, "chosen_relative": {m: (a["chosen_ppl"] / a["reference_ppl"] if a["chosen"] else None) for m, a in scored.items()}}
    cheaper = {m: a for m, a in scored.items() if a["chosen"] is not None and a["chosen_cache"] < a["fixed_cache"]}
    res["R3"] = {"testable": bool(cheaper), "pass": complete and bool(cheaper) and all(not a["chosen_severe"] for a in cheaper.values()), "cheaper_on": list(cheaper),
                 "perplexity_cost_of_the_check_over_the_fixed_rule": {m: a["chosen_ppl"] / a["fixed_ppl"] for m, a in cheaper.items()}}
    res["passes"] = sum(bool(res[k]["pass"]) for k in ("R1", "R2", "R3")); res["of"] = 3
    json.dump(res, open(OUT, "w"), indent=1)
    for k in ("R1", "R2", "R3"):
        print(k, "PASS" if res[k]["pass"] else ("UNTESTABLE" if res[k].get("testable") is False else "FAIL"), {kk: vv for kk, vv in res[k].items() if kk != "pass"})
    for m, a in A.items():
        print(m, "scored" if m in SCORED else "development", "reference", round(a["reference_ppl"], 2), "chosen", a["chosen"], "severe", a["severe"], "benign flagged", a["benign_flagged"])
    print("passes", res["passes"], "of", res["of"], "missing", res["missing"])
