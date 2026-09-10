# Evaluation of PREREGISTRATION6 (rerun with fixed instruments), committed
# before any result exists. Reads rerun/<model>.json, writes rerun_results.json
# and prints the scorecard S1' to S4' with secondary clauses.
#   python alignment_study/eval_rerun.py [directory]
import glob
import json
import sys
import numpy as np
from scipy.stats import spearmanr

D = sys.argv[1] if len(sys.argv) > 1 else "alignment_study/rerun"
HS, COS, RHO_X, R_MAX, SHIFT_RHO, R1_RESTORE, CV_MIN = 0.4, 0.7, 2.0, 0.4, 0.5, 0.8, 0.1
MIN_LAYERS = 3
CALIB = {"Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-3B"}   # instrument calibration models: reported, excluded from tallies and the S4 pool


def frac(flags):
    return round(float(np.mean(flags)), 3) if len(flags) else float("nan")


def spear(x, y):
    return float(spearmanr(x, y)[0])


def boot(x, y, B=1000, seed=0):
    rg = np.random.default_rng(seed); x, y = np.asarray(x), np.asarray(y)
    v = [spear(x[i], y[i]) for i in (rg.integers(0, len(x), len(x)) for _ in range(B))]
    return [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)]


models = {}
for p in sorted(glob.glob(f"{D}/*.json")):
    c = json.load(open(p))
    if isinstance(c, dict) and "model" in c and "protocol_A" in c:
        models[c["model"]] = c
if not models:
    raise SystemExit(f"no model results in {D}")

res = {"n_models": len(models), "per_model": {}, "S": {}}
pooled = []
for name, c in models.items():
    A = c["protocol_A"]["rows"]; B = c["protocol_B"]["rows"]
    hsA = [r for r in A if r["smass"] > HS]; hsB = [r for r in B if r["smass"] > HS]
    pm = {"n_layers": len(A), "n_high_sink_A": len(hsA), "n_high_sink_B": len(hsB), "testable": len(hsA) >= MIN_LAYERS,
          "layers_with_k_above_1": [r["layer"] for r in A if r["k_max"] > 1]}
    # S1': shared sink operator (rho excess and cosine to the sink-set operator)
    pm["S1_frac"] = frac([r["rho_real"] >= RHO_X * r["rho_plain"] and r["cosS"] > COS for r in hsA])
    # S2': column-set sufficiency, layer-level distance ratios from window-aggregated statistics
    Rz = [abs(r["zmed_colfix"] - r["zmed_real"]) / max(1.0, abs(r["zmed_real"] - r["zmed_plain"])) for r in hsA]
    Rr = [abs(r["r1_colfix"] - r["r1_real"]) / max(0.05, abs(r["r1_plain"] - r["r1_real"])) for r in hsA]
    pm["S2_frac"] = frac([z < R_MAX for z in Rz]); pm["S2_secondary_R_r1_frac"] = frac([x < R_MAX for x in Rr])
    pm["S2_max_R_z"] = round(max(Rz), 3) if Rz else None
    # S3': dissociation with the wrapped-target control at T = 256; shift and altsink-v2 secondary
    pm["S3_frac"] = frac([r["rho_wrapped"] < SHIFT_RHO * r["rho_real"] and r["zmed_wrapped"] > r["zmed_real"] for r in hsB])
    pm["S3_secondary_r1_restore_frac"] = frac([r["r1_wrapped"] > R1_RESTORE for r in hsB])
    pm["S3_shift_frac"] = frac([r["rho_shift"] < SHIFT_RHO * r["rho_real"] and r["zmed_shift"] > r["zmed_real"] for r in hsB])
    pm["S3_shift_r1_restore_frac"] = frac([r["r1_shift"] > R1_RESTORE for r in hsB])
    pm["S3_alt2_frac"] = frac([r["rho_alt2"] < SHIFT_RHO * r["rho_real"] and r["zmed_alt2"] > r["zmed_real"] for r in hsB])
    pm["S3_alt2_r1_restore_frac"] = frac([r["r1_alt2"] > R1_RESTORE for r in hsB])
    pm["S3_testable"] = len(hsB) >= MIN_LAYERS
    for k in ("S1", "S2"):
        pm[f"{k}_pass"] = (pm[f"{k}_frac"] >= 2 / 3) if pm["testable"] else None
    pm["S3_pass"] = (pm["S3_frac"] >= 2 / 3) if pm["S3_testable"] else None
    pm["calibration_model"] = name in CALIB
    if name not in CALIB:
        pooled += [(r["sharedE"], r["r1_real"], r["cv_gn"]) for r in hsA]
    res["per_model"][name] = pm

test = [pm for pm in res["per_model"].values() if pm["testable"] and not pm["calibration_model"]]
test3 = [pm for pm in res["per_model"].values() if pm["S3_testable"] and not pm["calibration_model"]]
for k, tt in (("S1", test), ("S2", test), ("S3", test3)):
    passes = sum(bool(pm[f"{k}_pass"]) for pm in tt)
    res["S"][k] = {"passes": passes, "testable": len(tt), "pass": bool(tt) and passes >= (2 / 3) * len(tt)}
flagged = [(e, r) for e, r, cv in pooled if cv >= CV_MIN]
if len(flagged) >= 10:
    x = [e for e, _ in flagged]; y = [r for _, r in flagged]; rho4 = spear(x, y)
    res["S"]["S4"] = {"spearman": round(rho4, 3), "ci95": boot(x, y), "n_layers": len(flagged), "n_excluded_ill_conditioned": len(pooled) - len(flagged), "pass": rho4 <= -0.5}
    xa = [e for e, _, _ in pooled]; ya = [r for _, r, _ in pooled]
    res["S"]["S4_unflagged_all_layers"] = {"spearman": round(spear(xa, ya), 3), "n_layers": len(pooled)}
else:
    res["S"]["S4"] = {"n_layers": len(flagged), "pass": None}

OUTFILE = "alignment_study/rerun_results.json" if D.rstrip("/") == "alignment_study/rerun" else D.rstrip("/") + "_results.json"
json.dump(res, open(OUTFILE, "w"), indent=1)
print(f"models: {len(models)}; testable: {len(test)}")
for k, v in res["S"].items():
    print(k, json.dumps(v))
for name, pm in res["per_model"].items():
    print("  ", name.split("/")[-1], json.dumps({k: v for k, v in pm.items() if k != "layers_with_k_above_1"}), "k>1 layers:", pm["layers_with_k_above_1"])
