# Evaluator for PREREGISTRATION25 (committed before any confirmatory result). Reads alignment_study/xlayer/*.json
# and writes alignment_study/xlayer_results.json. Damage of an arm: continuation NLL (nats per token) minus the
# full cache's. Clauses A1, A2, B1, B2, C1, C2, C3 as registered in PREREGISTRATION25.md.
#   python alignment_study/eval_xlayer.py [directory]
import glob
import json
import sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "alignment_study/xlayer"
OUT = "alignment_study/xlayer_results.json" if D == "alignment_study/xlayer" else f"{D}_results.json"


def med(r, key):
    return float(np.median([p[key] for p in r["premise"]]))


if __name__ == "__main__":
    R = {}
    for p in sorted(glob.glob(f"{D}/*.json")):
        r = json.load(open(p)); R[r["model"]] = r
    models = list(R)
    clauses, summary = {}, {}
    for m, r in R.items():
        s = {"k": med(r, "k"), "v": med(r, "v"), "hidden": med(r, "hidden"), "normed_input": med(r, "normed_input"),
             "k_aligned": med(r, "k_aligned"), "v_aligned": med(r, "v_aligned"), "k_null": med(r, "k_null"), "v_null": med(r, "v_null")}
        for which in ("k", "v"):
            adj = np.median([sum(p["codes"][which]["adjacent"][:2]) for p in r["premise"]])
            ind = np.median([sum(p["codes"][which]["independence"][:2]) for p in r["premise"]])
            ex_al = np.median([p["codes"][which]["aligned"][0] for p in r["premise"]])
            ex_in = np.median([p["codes"][which]["independence"][0] for p in r["premise"]])
            s[f"codes_{which}"] = {"adjacent_le1": float(adj), "independence_le1": float(ind), "aligned_exact": float(ex_al), "independence_exact": float(ex_in)}
        dmg = {}
        for st, b in r["merge"].items():
            full = b["full"]
            dmg[st] = {a: b[a] - full for a in ("trained", "random0", "random1", "random2", "aligned", "aligned_nomatch")}
            dmg[st]["random_median"] = float(np.median([dmg[st][f"random{i}"] for i in range(3)]))
        s["merge_damage"] = dmg
        xq = r["xquant"]
        s["xquant_damage"] = {a: xq[a] - xq["full"] for a in ("kivi", "trained", "random", "aligned", "kivi_rot")}
        summary[m] = s
    half = {m: str(R[m]["layers"] // 2) for m in models}
    quarter = {m: str(R[m]["layers"] // 4) for m in models}
    a1 = {m: abs(summary[m]["k"]) <= 0.1 and abs(summary[m]["v"]) <= 0.1 and summary[m]["hidden"] > 0.5 for m in models}
    clauses["A1"] = {"what": "|median K and V cosine| <= 0.1 and median residual-stream cosine > 0.5, every model", "value": a1, "pass_": all(a1.values())}
    a2 = {m: summary[m]["v_aligned"] - summary[m]["v_null"] for m in models}
    clauses["A2"] = {"what": "median aligned V cosine exceeds the fitting null's by >= 0.1, every model", "value": a2, "pass_": all(v >= 0.1 for v in a2.values())}
    b1 = {m: summary[m]["merge_damage"][half[m]]["trained"] - summary[m]["merge_damage"][half[m]]["random_median"] for m in models}
    clauses["B1"] = {"what": "MiniCache from L/2: trained-basis damage >= median random-basis damage - 0.1 nats, every model", "value": b1,
                     "pass_": all(v >= -0.1 for v in b1.values())}
    b2 = {f"{m} {lab}": summary[m]["merge_damage"][st]["aligned"] / summary[m]["merge_damage"][st]["trained"]
          for m in models for lab, st in (("L/2", half[m]), ("L/4", quarter[m]))}
    clauses["B2"] = {"what": "aligned damage <= 0.6 x trained-basis damage in >= 7 of 8 (model, setting) cells", "value": b2,
                     "count": sum(v <= 0.6 for v in b2.values()), "pass_": sum(v <= 0.6 for v in b2.values()) >= 7}
    c1 = {f"{m} {w}": summary[m][f"codes_{w}"]["adjacent_le1"] - summary[m][f"codes_{w}"]["independence_le1"] for m in models for w in ("k", "v")}
    clauses["C1"] = {"what": "XQuant's statistic (share of |code difference| <= 1) within 0.02 of independence, keys and values, every model",
                     "value": c1, "pass_": all(abs(v) <= 0.02 for v in c1.values())}
    c2 = {m: summary[m]["codes_v"]["aligned_exact"] - summary[m]["codes_v"]["independence_exact"] for m in models}
    clauses["C2"] = {"what": "after alignment, exact value-code agreement exceeds independence by >= 0.03, every model", "value": c2,
                     "pass_": all(v >= 0.03 for v in c2.values())}
    c3 = {m: summary[m]["xquant_damage"]["random"] - summary[m]["xquant_damage"]["trained"] for m in models}
    clauses["C3"] = {"what": "XQuant's sharing loses less in the trained basis than in random bases in >= 3 of 4 models", "value": c3,
                     "count": sum(v > 0 for v in c3.values()), "pass_": sum(v > 0 for v in c3.values()) >= 3}
    res = {"models": models, "clauses": clauses, "passes": sum(c["pass_"] for c in clauses.values()), "of": len(clauses), "summary": summary}
    json.dump(res, open(OUT, "w"), indent=1)
    for k, c in clauses.items():
        print(k, "PASS" if c["pass_"] else "FAIL", c["what"], "|", json.dumps(c["value"])[:260])
    print(f"{res['passes']} of {res['of']} clauses pass; written {OUT}")
