# Evaluator for PREREGISTRATION35 (draft; to be committed before any confirmatory result). Reads
# alignment_study/retrieval_round/*.json (written by retrieval_protect_dev.py --registered=...) and writes
# alignment_study/retrieval_round_results.json. Scored models: Qwen2.5-7B and Mistral-7B-v0.1; Mistral-7B-Instruct-v0.2 is
# the development model, reported. A scored model whose unconverted retrieval is below 45 of 50 is excluded; one whose
# plain fusion retrieves 40 or more of 50 cannot show a repair (E1 to E3 untestable on it, not failed).
#   python alignment_study/eval_retrieval35.py [--dir=alignment_study/retrieval_round]
import glob
import json
import os
import sys

DIR = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")), "alignment_study/retrieval_round")
OUT = "alignment_study/retrieval_round_results.json" if DIR.rstrip("/") == "alignment_study/retrieval_round" else DIR.rstrip("/") + "_results.json"
SCORED = ["Qwen2.5-7B", "Mistral-7B-v0.1"]
MIN_BASE, NO_REPAIR, E1_MARGIN, E4_CACHE = 45, 40, 10, 0.55


def hits(r, arm):
    return r["arms"][arm]["needle"]["hits"] if arm in r["arms"] else None


def analyse(r):
    a = r["arms"]
    base, plain = hits(r, "unconverted"), hits(r, "a_fusion")
    b, c, f = hits(r, "b_retrieval_0.3"), hits(r, "c_retrieval_0.1"), hits(r, "f_residual")
    d = [v["needle"]["hits"] for k, v in a.items() if k.startswith("d_random")]
    e = [v["needle"]["hits"] for k, v in a.items() if k.startswith("e_random")]
    out = {"unconverted": base, "plain": plain, "b": b, "c": c, "d": d, "e": e, "f": f, "excluded": r.get("excluded", base is not None and base < MIN_BASE),
           "no_repair_possible": plain is not None and plain >= NO_REPAIR, "complete": all(x is not None for x in (base, plain, b, c, f)) and len(d) == 2 and len(e) == 2}
    if out["complete"]:
        out["E1"] = b - plain >= E1_MARGIN; out["E2"] = all(b > x for x in d); out["E3"] = b > f
        out["E4"] = a["b_retrieval_0.3"]["cache_fraction"] <= E4_CACHE and a["b_retrieval_0.3"]["ppl_8_windows"] <= a["a_fusion"]["ppl_8_windows"]
        out["cache_b"], out["ppl_b"], out["ppl_plain"] = a["b_retrieval_0.3"]["cache_fraction"], a["b_retrieval_0.3"]["ppl_8_windows"], a["a_fusion"]["ppl_8_windows"]
    return out


if __name__ == "__main__":
    R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{DIR}/*.json")}
    A = {m: analyse(r) for m, r in R.items()}
    scored = {m: A[m] for m in SCORED if m in A and not A[m]["excluded"]}
    res = {"missing": [m for m in SCORED if m not in R], "excluded": [m for m in SCORED if m in A and A[m]["excluded"]], "per_model": A,
           "development_model": [m for m in R if m not in SCORED]}
    complete = len(scored) + len(res["excluded"]) == len(SCORED) and all(a["complete"] for a in scored.values())
    testable = {m: a for m, a in scored.items() if not a["no_repair_possible"]}
    for k in ("E1", "E2", "E3"):
        res[k] = {"testable": bool(testable) and complete, "pass": complete and bool(testable) and all(a[k] for a in testable.values()),
                  "per_model": {m: a.get(k) for m, a in scored.items()}, "untestable_on": [m for m, a in scored.items() if a["no_repair_possible"]]}
    res["E4"] = {"pass": complete and bool(scored) and all(a["E4"] for a in scored.values()), "per_model": {m: a.get("E4") for m, a in scored.items()}}
    pooled_c = sum(a["c"] for a in scored.values()) if scored else None; pooled_b = sum(a["b"] for a in scored.values()) if scored else None
    pooled_e = [sum(a["e"][i] for a in scored.values()) for i in range(2)] if scored and all(len(a["e"]) == 2 for a in scored.values()) else None
    res["E5"] = {"pass": complete and bool(scored) and pooled_c is not None and pooled_e is not None and pooled_c >= pooled_b and all(pooled_c > x for x in pooled_e),
                 "pooled_c": pooled_c, "pooled_b": pooled_b, "pooled_e": pooled_e}
    res["passes"] = sum(bool(res[k]["pass"]) for k in ("E1", "E2", "E3", "E4", "E5")); res["of"] = 5
    json.dump(res, open(OUT, "w"), indent=1)
    for k in ("E1", "E2", "E3", "E4", "E5"):
        print(k, "PASS" if res[k]["pass"] else ("UNTESTABLE" if res[k].get("testable") is False else "FAIL"), {kk: vv for kk, vv in res[k].items() if kk != "pass"})
    for m, a in A.items():
        print(m, "scored" if m in SCORED else "development", {k: a[k] for k in ("unconverted", "plain", "b", "c", "d", "e", "f", "excluded")})
    print("passes", res["passes"], "of", res["of"], "missing", res["missing"], "excluded", res["excluded"])
