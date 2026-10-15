# Evaluation of audits/PREREGISTRATION18.md, committed before any confirmatory
# result exists. For each claim and confirmatory model:
#   s = (median_colset - median_random) / (real - median_random)
# Inclusion: the claim reproduces (real above every random draw and every
# untrained initialization). Prediction: aggregate claims (A1, A2, A3) s >= 0.5,
# located claims (L1, L2, L3) s <= 0.2, in every confirmatory model. Score: K of
# the included claims.
#   python audits/rule/eval_rule.py [directory]
import glob
import json
import sys
import numpy as np

D = (sys.argv[1] if len(sys.argv) > 1 else "audits/rule").rstrip("/")
KIND = {"A1": "aggregate", "A2": "aggregate", "A3": "aggregate", "L1": "located", "L2": "located", "L3": "located"}
TH = {"aggregate": 0.5, "located": 0.2}
# Secondary, informed by the development run (labeled, not prospective): statistics weighted by
# attention mass have s >= 0.5; statistics that count positions above a threshold, and located
# claims, have s <= 0.2.
KIND2 = {"A1": "weighted", "A2": "count", "A3": "count", "L1": "located", "L2": "located", "L3": "located"}
TH2 = {"weighted": (">=", 0.5), "count": ("<=", 0.2), "located": ("<=", 0.2)}


def entries():
    for p in sorted(glob.glob(f"{D}/*.json")):
        r = json.load(open(p))
        if "claim" not in r:
            continue
        if r["claim"] == "A1A2":
            for c, key in (("A1", "heavy"), ("A2", "persist")):
                yield c, r["model"], r["real"][key], {f: [x for x in r[f][key]] for f in ("a_random", "b_marginal", "c_colset")}, [u[key] for u in r["untrained"]]
        else:
            yield r["claim"], r["model"], r["real"], {f: r[f] for f in ("a_random", "b_marginal", "c_colset")}, r["untrained"]


res = {"thresholds": TH, "per_model": [], "claims": {}}
for c, m, real, nulls, unt in entries():
    rnd = float(np.median(nulls["a_random"]))
    share = lambda f: (float(np.median(nulls[f])) - rnd) / (real - rnd) if abs(real - rnd) > 1e-12 else None
    reproduces = real > max(nulls["a_random"]) and real > max(unt)
    s = share("c_colset")
    hold = (s >= TH[KIND[c]]) if KIND[c] == "aggregate" else (s <= TH[KIND[c]])
    op, th2 = TH2[KIND2[c]]; hold2 = (s >= th2) if op == ">=" else (s <= th2)
    res["per_model"].append({"claim": c, "model": m, "kind": KIND[c], "real": real, "random_median": rnd, "colset_median": float(np.median(nulls["c_colset"])),
                             "marginal_median": float(np.median(nulls["b_marginal"])), "untrained": unt, "s_colset": s, "s_marginal": share("b_marginal"),
                             "s_untrained": (float(np.median(unt)) - rnd) / (real - rnd) if abs(real - rnd) > 1e-12 else None,
                             "reproduces": bool(reproduces), "prediction_holds": bool(hold), "secondary_holds": bool(hold2)})
for c in KIND:
    rows = [r for r in res["per_model"] if r["claim"] == c]
    if not rows:
        continue
    inc = all(r["reproduces"] for r in rows)
    res["claims"][c] = {"kind": KIND[c], "models": [r["model"] for r in rows], "s_colset": [r["s_colset"] for r in rows], "included": inc,
                        "prediction_holds": bool(inc and all(r["prediction_holds"] for r in rows)),
                        "secondary_kind": KIND2[c], "secondary_holds": bool(inc and all(r["secondary_holds"] for r in rows))}
inc = [c for c, v in res["claims"].items() if v["included"]]
res["K"] = int(sum(res["claims"][c]["prediction_holds"] for c in inc)); res["N"] = len(inc)
res["K_secondary"] = int(sum(res["claims"][c]["secondary_holds"] for c in inc))
json.dump(res, open(D + "_results.json", "w"), indent=1)
for r in res["per_model"]:
    print(f"{r['claim']} {r['model'].split('/')[-1]:24s} real {r['real']:.4f} random {r['random_median']:.4f} colset {r['colset_median']:.4f} "
          f"s_colset {r['s_colset']:+.3f} s_marginal {r['s_marginal']:+.3f} reproduces {r['reproduces']} holds {r['prediction_holds']}")
print("K", res["K"], "of N", res["N"], "| secondary (dev-informed):", res["K_secondary"], "of", res["N"])
