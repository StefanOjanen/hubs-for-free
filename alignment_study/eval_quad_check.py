# Evaluator for PREREGISTRATION34 (committed before any confirmatory result). Reads alignment_study/quad_check/*.json
# (written by quad_check_round.py --registered=...) and writes alignment_study/quad_check_results.json.
#   python alignment_study/eval_quad_check.py [--dir=alignment_study/quad_check]
import glob
import json
import os
import sys

DIR = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")), "alignment_study/quad_check")
SCORED = ["Mistral-7B-v0.1", "Mistral-7B-Instruct-v0.2", "Qwen2.5-7B"]
TAU, SEVERE, BENIGN, WITHIN, FIXED_K = 2.35, 5.0, 2.0, 0.10, 1


def analyse(r):
    q = [m for m in r["members"] if m["kind"] == "quarter"]
    best = min(m["ppl"] for m in q)
    for m in q:
        m["relative"] = m["ppl"] / best
        m["severe"], m["benign"] = m["relative"] >= SEVERE, m["relative"] <= BENIGN
    res_g = [m for m in q if m["grouping"] == "residual"]
    chosen = next((m["layers_at_pairs"] for m in sorted(res_g, key=lambda m: m["layers_at_pairs"]) if not m["flag"]), None)
    by_k = {m["layers_at_pairs"]: m for m in res_g}
    best_k = min(by_k, key=lambda k: by_k[k]["ppl"])
    return {"best_ppl": best, "severe": [(m["grouping"], m["layers_at_pairs"]) for m in q if m["severe"]],
            "benign_flagged": [(m["grouping"], m["layers_at_pairs"]) for m in q if m["benign"] and m["flag"]],
            "severe_unflagged": [(m["grouping"], m["layers_at_pairs"]) for m in q if m["severe"] and not m["flag"]],
            "chosen_k": chosen, "best_k": best_k, "chosen_ppl": by_k[chosen]["ppl"] if chosen is not None else None, "best_k_ppl": by_k[best_k]["ppl"],
            "fixed_rule_ppl": by_k[FIXED_K]["ppl"] if FIXED_K in by_k else None,
            "members": [{k: m[k] for k in ("grouping", "kind", "layers_at_pairs", "ppl", "max_ratio", "flag", "relative", "severe", "benign")} for m in q]
                       + [{k: m[k] for k in ("grouping", "kind", "ppl", "max_ratio", "flag")} for m in r["members"] if m["kind"] == "half"]}


if __name__ == "__main__":
    R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{DIR}/*.json")}
    A = {m: analyse(r) for m, r in R.items()}
    scored = {m: a for m, a in A.items() if m in SCORED}
    res = {"missing": [m for m in SCORED if m not in R], "per_model": A, "development_model": [m for m in R if m not in SCORED]}
    n_severe = sum(len(a["severe"]) for a in scored.values())
    res["R1"] = {"recall_pass": len(scored) == 3 and n_severe > 0 and all(not a["severe_unflagged"] for a in scored.values()),
                 "recall_testable": n_severe > 0, "false_alarm_pass": len(scored) == 3 and all(not a["benign_flagged"] for a in scored.values()),
                 "severe_members": n_severe}
    res["R1"]["pass"] = res["R1"]["false_alarm_pass"] and (res["R1"]["recall_pass"] if res["R1"]["recall_testable"] else True)
    res["R2"] = {"pass": len(scored) == 3 and all(a["chosen_k"] is not None and a["chosen_ppl"] <= (1 + WITHIN) * a["best_k_ppl"] for a in scored.values()),
                 "chosen_k": {m: a["chosen_k"] for m, a in scored.items()}, "best_k": {m: a["best_k"] for m, a in scored.items()}}
    differs = {m: a for m, a in scored.items() if a["chosen_k"] is not None and a["chosen_k"] != FIXED_K}
    res["R3"] = {"testable": bool(differs), "pass": bool(differs) and any(a["chosen_ppl"] <= a["fixed_rule_ppl"] for a in differs.values()),
                 "differs_on": list(differs)}
    res["passes"] = sum(bool(res[k]["pass"]) for k in ("R1", "R2", "R3")); res["of"] = 3
    json.dump(res, open("alignment_study/quad_check_results.json", "w"), indent=1)
    for k in ("R1", "R2", "R3"):
        print(k, "PASS" if res[k]["pass"] else ("UNTESTABLE" if res[k].get("testable") is False else "FAIL"), {kk: vv for kk, vv in res[k].items() if kk != "pass"})
    for m, a in A.items():
        print(m, "scored" if m in SCORED else "development", "best", round(a["best_ppl"], 2), "chosen k", a["chosen_k"], "best k", a["best_k"], "severe", a["severe"], "benign flagged", a["benign_flagged"])
    print("passes", res["passes"], "of", res["of"], "missing", res["missing"])
