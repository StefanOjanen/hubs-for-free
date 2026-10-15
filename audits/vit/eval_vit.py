# Evaluation of audits/PREREGISTRATION19.md, committed before any confirmatory
# result exists.
#   python audits/vit/eval_vit.py [directory]
import json
import sys
import numpy as np

D = (sys.argv[1] if len(sys.argv) > 1 else "audits/vit").rstrip("/")
R = {m: json.load(open(f"{D}/{m}.json")) for m in ("dinov2-large", "dinov2-with-registers-large", "vit-large-patch16-224", "vit-base-patch16-224")}
half = lambda r, k: float(np.mean(r[k][r["layers"] // 2:]))
no, reg, vl, vb = R["dinov2-large"], R["dinov2-with-registers-large"], R["vit-large-patch16-224"], R["vit-base-patch16-224"]
V = {
    "V1": {"what": "DINOv2-L: artifact share of the depth increase >= 0.4", "value": no["artifact_share_of_increase"],
           "pass_": no["artifact_share_of_increase"] is not None and no["artifact_share_of_increase"] >= 0.4},
    "V2": {"what": "DINOv2-L: second-half attention mass on artifact columns >= 0.1", "value": half(no, "mass_art"), "pass_": half(no, "mass_art") >= 0.1},
    "V3": {"what": "DINOv2-L with registers: artifact share <= 0.1 and second-half artifact mass < 0.02", "value": [reg["artifact_share_of_increase"], half(reg, "mass_art")],
           "pass_": (reg["artifact_share_of_increase"] is None or reg["artifact_share_of_increase"] <= 0.1) and half(reg, "mass_art") < 0.02},
    "V4": {"what": "registers shrink the depth increase: delta D_all with registers below the model without", "value": [reg["delta_D_all"], no["delta_D_all"]],
           "pass_": reg["delta_D_all"] < no["delta_D_all"]},
    "V5": {"what": "the source's claim reproduces on ViT-L/16: delta D_all > 0 and at least half of the heads increase with depth", "value": [vl["delta_D_all"], vl["heads_increase_with_depth_all"]],
           "pass_": vl["delta_D_all"] > 0 and vl["heads_increase_with_depth_all"] >= 0.5},
}
rep = {m: {"delta_D_all": r["delta_D_all"], "delta_D_ex": r["delta_D_ex"], "artifact_share": r["artifact_share_of_increase"], "second_half_D_all": r["second_half_D_all"],
           "second_half_D_ex": r["second_half_D_ex"], "uniform_distance": r["uniform_distance"], "second_half_mass_art": half(r, "mass_art"), "second_half_mass_cls": half(r, "mass_cls"),
           "second_half_mass_reg": half(r, "mass_reg"), "heads_increase_with_depth": r["heads_increase_with_depth_all"]} for m, r in R.items()}
res = {"clauses": V, "passes": int(sum(v["pass_"] for v in V.values())), "reported": rep}
json.dump(res, open(D + "_results.json", "w"), indent=1)
for k, v in V.items():
    print(k, "PASS" if v["pass_"] else "FAIL", v["what"], v["value"])
print("passes", res["passes"], "of", len(V))
for m, r in rep.items():
    print(m, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()})
