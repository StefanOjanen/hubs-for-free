# Evaluation of PREREGISTRATION5 (dynamics), committed before any checkpoint
# result exists. Reads dynamics/<model>_step<k>.json, writes
# dynamics_results.json and prints the scorecard D0 to D5.
import glob
import json
import numpy as np
from scipy.stats import spearmanr

D = "alignment_study/dynamics"
MODELS = ["EleutherAI/pythia-160m", "EleutherAI/pythia-410m"]
STEPS = [0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
HS, COS = 0.4, 0.7

tab = {}
for p in sorted(glob.glob(f"{D}/*.json")):
    c = json.load(open(p))
    tab.setdefault(c["model"], {})[c["step"]] = {r["layer"]: r for r in c["rows"]}
complete = all(set(tab.get(m, {})) == set(STEPS) for m in MODELS)
cells = [(m, s, l, r) for m in tab for s in tab[m] for l, r in tab[m][s].items()]
if not cells:
    raise SystemExit("no dynamics cells found; run dynamics_round.py first")


def spear(x, y):
    return float(spearmanr(x, y)[0])


def boot(x, y, B=1000, seed=0):
    rg = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    v = [spear(x[i], y[i]) for i in (rg.integers(0, len(x), len(x)) for _ in range(B))]
    return [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)]


res = {"complete": complete, "n_cells": len(cells), "D": {}, "per_model": {}}

# D0: step0 anchors
d0 = {}
for m in tab:
    if 0 in tab[m]:
        rows = tab[m][0].values()
        d0[m] = {"max_smass": max(r["smass"] for r in rows), "max_cosS": max(r["cosS"] for r in rows)}
res["D"]["D0"] = {"per_model": d0, "pass": bool(d0) and all(v["max_smass"] <= HS and v["max_cosS"] < 0.5 for v in d0.values())}

# D1: Spearman(sink mass, cosS) over all cells
x = [r["smass"] for *_, r in cells]; y = [r["cosS"] for *_, r in cells]
rho1 = spear(x, y)
res["D"]["D1"] = {"spearman": round(rho1, 3), "ci95": boot(x, y), "n": len(x), "pass": rho1 >= 0.5}

# D1b: cosS > 0.7 among high-sink cells
hs = [r for *_, r in cells if r["smass"] > HS]
frac = float(np.mean([r["cosS"] > COS for r in hs])) if hs else float("nan")
res["D"]["D1b"] = {"frac_cos_above_0.7": round(frac, 3), "n_high_sink": len(hs), "pass": bool(hs) and frac >= 0.8}

# D2: law over high-sink cells
if len(hs) >= 30:
    xe = [r["sharedE"] for r in hs]; yr = [r["r1_real"] for r in hs]
    rho2 = spear(xe, yr)
    res["D"]["D2"] = {"spearman": round(rho2, 3), "ci95": boot(xe, yr), "n": len(hs), "testable": True, "pass": rho2 <= -0.5}
else:
    res["D"]["D2"] = {"n": len(hs), "testable": False, "pass": None}

# D3: mode switch locked to sink formation; D4, D5 per model
d3, d4, d5 = {}, {}, {}
for m in tab:
    if set(tab[m]) != set(STEPS):
        continue
    L = len(tab[m][STEPS[-1]])
    final = tab[m][STEPS[-1]]
    layers = [l for l in range(L) if final[l]["smass"] > HS]
    cons = []
    for l in layers:
        sm = [tab[m][s][l]["smass"] for s in STEPS]; cs = [tab[m][s][l]["cosS"] for s in STEPS]
        ks = next(i for i, v in enumerate(sm) if v > HS)
        kc = next((i for i, v in enumerate(cs) if v > COS), None)
        cons.append({"layer": l, "k_s": ks, "k_c": kc, "consistent": kc is not None and abs(kc - ks) <= 1})
    f3 = float(np.mean([c["consistent"] for c in cons])) if cons else float("nan")
    d3[m] = {"n_layers": len(layers), "frac_consistent": round(f3, 3), "testable": len(layers) >= 3,
             "pass": (f3 >= 2 / 3) if len(layers) >= 3 else None, "layers": cons}
    first_hs = next((s for s in STEPS if any(r["smass"] > HS for r in tab[m][s].values())), None)
    d4[m] = {"first_high_sink_step": first_hs, "pass": first_hs is not None and first_hs <= 4000}
    dips = []
    for l in range(L):
        e = [tab[m][s][l]["sharedE"] for s in STEPS]
        dips.append(min(e) < min(e[0], e[-1]) - 0.05)
    d5[m] = {"frac_layers_with_dip": round(float(np.mean(dips)), 3), "pass": float(np.mean(dips)) >= 2 / 3}
testable3 = [v for v in d3.values() if v["testable"]]
res["D"]["D3"] = {"per_model": d3, "pass": bool(testable3) and all(v["pass"] for v in testable3)}
res["D"]["D4"] = {"per_model": d4, "pass": bool(d4) and all(v["pass"] for v in d4.values()), "secondary": True}
res["D"]["D5"] = {"per_model": d5, "pass": bool(d5) and all(v["pass"] for v in d5.values()), "secondary": True, "exploratory": True}

# anchor: final checkpoint per-layer summary for comparison with rounds 1 and 2
for m in tab:
    if STEPS[-1] in tab[m]:
        res["per_model"][m] = {"final_high_sink_layers": [l for l, r in tab[m][STEPS[-1]].items() if r["smass"] > HS],
                               "final_sharedE": [round(r["sharedE"], 3) for r in tab[m][STEPS[-1]].values()],
                               "final_r1": [round(r["r1_real"], 3) for r in tab[m][STEPS[-1]].values()],
                               "final_cosS": [round(r["cosS"], 3) for r in tab[m][STEPS[-1]].values()]}

json.dump(res, open("alignment_study/dynamics_results.json", "w"), indent=1)
print("complete:", complete, "cells:", len(cells))
for k, v in res["D"].items():
    short = {kk: vv for kk, vv in v.items() if kk not in ("per_model", "layers")}
    print(k, json.dumps(short))
    if "per_model" in v:
        for m, pm in v["per_model"].items():
            print("   ", m.split("/")[-1], json.dumps({kk: vv for kk, vv in pm.items() if kk != "layers"}))
