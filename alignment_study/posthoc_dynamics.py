# POST-HOC (not registered) descriptive analysis of the dynamics round,
# written after eval_dynamics.py produced the scorecard. Three descriptions:
# (a) the generic interlude: at steps 512 and 1000, before any sink exists,
#     how many cells sit in the generic regime (r1 > 0.8);
# (b) where along training r1 crosses zero: for each layer and each pair of
#     consecutive checkpoints with the layer high-sink at both, if r1 changes
#     sign from positive to negative, linearly interpolate shared energy at
#     the crossing; report the distribution against the toy crossing;
# (c) late-training reversals: layers whose shared energy at step143000 is
#     more than 0.1 below its maximum over the checkpoints from step2000 on
#     (step0 and the interlude are excluded: initialization has spuriously
#     high shared energy carried by the uniform causal operator).
import glob
import json
import numpy as np

STEPS = [0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
tab = {}
for p in glob.glob("alignment_study/dynamics/*.json"):
    c = json.load(open(p)); tab.setdefault(c["model"], {})[c["step"]] = {r["layer"]: r for r in c["rows"]}
toy = json.load(open("alignment_study/tier2_toy.json"))["toy_curves"]["aligned"]
pts = sorted((r["sharedE"], r["r1"]) for r in toy)
toy_cross = next((e0 + (r0 / (r0 - r1_)) * (e1 - e0) for (e0, r0), (e1, r1_) in zip(pts, pts[1:]) if r0 > 0 >= r1_), None)

out = {"toy_r1_zero_crossing_sharedE": round(float(toy_cross), 3), "generic_interlude": {}, "crossings": [], "late_reversals": {}}
for m in tab:
    short = m.split("/")[-1]
    for s in (512, 1000):
        rows = list(tab[m][s].values())
        out["generic_interlude"][f"{short}_step{s}"] = {
            "frac_r1_above_0.8": round(float(np.mean([r["r1_real"] > 0.8 for r in rows])), 3),
            "median_r1": round(float(np.median([r["r1_real"] for r in rows])), 3),
            "median_sharedE": round(float(np.median([r["sharedE"] for r in rows])), 3),
            "max_smass": round(max(r["smass"] for r in rows), 3)}
    L = len(tab[m][0])
    rev = []
    for l in range(L):
        e = [tab[m][s][l]["sharedE"] for s in STEPS]; rr = [tab[m][s][l]["r1_real"] for s in STEPS]; sm = [tab[m][s][l]["smass"] for s in STEPS]
        for k in range(len(STEPS) - 1):
            if sm[k] > 0.4 and sm[k + 1] > 0.4 and rr[k] > 0 >= rr[k + 1]:
                ec = e[k] + (rr[k] / (rr[k] - rr[k + 1])) * (e[k + 1] - e[k])
                out["crossings"].append({"model": short, "layer": l, "from_step": STEPS[k], "to_step": STEPS[k + 1],
                                         "sharedE_at_crossing": round(float(ec), 3)})
        late = e[3:]
        if max(late) - e[-1] > 0.1:
            rev.append({"layer": l, "max_sharedE_from_step2000": round(max(late), 3), "at_step": STEPS[3 + int(np.argmax(late))], "final_sharedE": round(e[-1], 3),
                        "final_smass": round(sm[-1], 3), "final_r1": round(rr[-1], 3)})
    out["late_reversals"][short] = rev
xs = [c["sharedE_at_crossing"] for c in out["crossings"]]
out["crossing_summary"] = {"n": len(xs), "median": round(float(np.median(xs)), 3), "iqr": [round(float(np.percentile(xs, 25)), 3), round(float(np.percentile(xs, 75)), 3)],
                           "min": round(min(xs), 3), "max": round(max(xs), 3)} if xs else {"n": 0}
json.dump(out, open("alignment_study/posthoc_dynamics.json", "w"), indent=1)
print("toy r1 zero crossing at sharedE", out["toy_r1_zero_crossing_sharedE"])
print("generic interlude:", json.dumps(out["generic_interlude"]))
print("crossings:", json.dumps(out["crossing_summary"]))
for c in out["crossings"]:
    print("  ", c)
print("late reversals:", json.dumps(out["late_reversals"]))
