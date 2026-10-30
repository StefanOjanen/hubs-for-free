# Evaluator for PREREGISTRATION29 (committed before any confirmatory result). Reads alignment_study/guard/*.json
# (one file per model, written by guard_round.py --registered=...) and writes alignment_study/guard_results.json.
#   python alignment_study/eval_guard.py [--dir=alignment_study/guard]
import glob
import json
import sys

DIR = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")), "alignment_study/guard")
MODELS = ["Mistral-7B-Instruct-v0.2", "open_llama_3b_v2", "OLMo-1B-hf", "Qwen2.5-7B"]
COLLAPSE, BENIGN = 5.0, 2.0                       # severe damage and benign, as multiples of the reference conversion

if __name__ == "__main__":
    runs = {m: json.load(open(f"{DIR}/{m}.json")) for m in MODELS if glob.glob(f"{DIR}/{m}.json")}
    missing = [m for m in MODELS if m not in runs]
    per, collapses, benign = {}, [], []
    for m, r in runs.items():
        mem = r["members"]
        col = [x for x in mem if x["ppl_over_reference"] >= COLLAPSE]
        ben = [x for x in mem if x["ppl_over_reference"] <= BENIGN]
        collapses += [(m, x) for x in col]; benign += [(m, x) for x in ben]
        per[m] = {"members": len(mem), "collapses": len(col), "benign": len(ben), "flagged": sum(x["flag"] for x in mem),
                  "collapses_flagged": sum(x["flag"] for x in col), "benign_flagged": sum(x["flag"] for x in ben),
                  "benign_flag_rate": (sum(x["flag"] for x in ben) / len(ben)) if ben else None,
                  "reference_ppl": r["reference_ppl"], "base_ppl": r["base_ppl"], "tau": r["tau"],
                  "collapses_by_layer": {str(l): sum(1 for x in col if x["layer"] == l) for l in range(r["early_layers"])}}
    models_with_collapse = [m for m in per if per[m]["collapses"] > 0]
    gate = len(models_with_collapse) >= 2 and not missing
    res = {"gate": {"pass": gate, "models_with_collapse": models_with_collapse, "missing_models": missing}, "per_model": per}
    if gate:
        g1_testable = len(collapses) >= 3
        res["G1"] = {"pass": g1_testable and all(x["flag"] for _, x in collapses), "testable": g1_testable,
                     "collapses": len(collapses), "flagged": sum(x["flag"] for _, x in collapses),
                     "missed": [{"model": m, "layer": x["layer"], "label": x["label"], "ppl_over_reference": x["ppl_over_reference"],
                                 "max_ratio": x["max_ratio"]} for m, x in collapses if not x["flag"]]}
        pooled = sum(x["flag"] for _, x in benign) / len(benign) if benign else None
        worst = max((v["benign_flag_rate"] for v in per.values() if v["benign_flag_rate"] is not None), default=None)
        res["G2"] = {"pass": pooled is not None and pooled <= 0.05 and worst <= 0.10, "pooled_benign_flag_rate": pooled,
                     "largest_model_rate": worst}
        flagged = [(m, x) for m, r in runs.items() for x in r["members"] if x["flag"]]
        res["G3"] = {"pass": bool(flagged) and all(x["repaired_over_reference"] <= BENIGN for _, x in flagged),
                     "flagged": len(flagged), "repaired_within_2x": sum(x["repaired_over_reference"] <= BENIGN for _, x in flagged),
                     "failures": [{"model": m, "layer": x["layer"], "label": x["label"], "repaired_over_reference": x["repaired_over_reference"]}
                                  for m, x in flagged if x["repaired_over_reference"] > BENIGN]}
        res["G4"] = {"pass": bool(collapses) and all(x["spread_dim"] == x["base_first_position_dim"] for _, x in collapses),
                     "collapses": len(collapses), "same_dimension": sum(x["spread_dim"] == x["base_first_position_dim"] for _, x in collapses),
                     "detail": [{"model": m, "layer": x["layer"], "label": x["label"], "spread_dim": x.get("spread_dim"),
                                 "base_first_position_dim": x.get("base_first_position_dim")} for m, x in collapses]}
        # G5: (a) every severe member varies a layer at or before the model's massive-activation layer; (b) in models
        # whose massive-activation layer has more than 4 KV heads, every severe member varying that layer gives its
        # strongest writer a new partner, while at most half of the benign members varying that layer do. Models
        # without a massive-activation layer among the check's layers are not applicable; G5 needs severe members
        # in at least 2 applicable models.
        appl = {m for m, r in runs.items() if r["massive_activation_layer"] is not None}
        col5 = [(m, x) for m, x in collapses if m in appl]
        a_ok = all(x["layer"] <= runs[m]["massive_activation_layer"] for m, x in col5)
        bmods = {m for m in appl if runs[m]["kv_heads"] > 4}
        at_me = [(m, x) for m, x in col5 if m in bmods and x["layer"] == runs[m]["massive_activation_layer"]]
        ben_me = [(m, x) for m, x in benign if m in bmods and x["layer"] == runs[m]["massive_activation_layer"]]
        ben_rate = (sum(x["strongest_writer_new_partner"] for _, x in ben_me) / len(ben_me)) if ben_me else None
        b_ok = all(x["strongest_writer_new_partner"] for _, x in at_me) and (not at_me or (ben_rate is not None and ben_rate <= 0.5))
        testable = len({m for m, _ in col5}) >= 2
        res["G5"] = {"pass": testable and a_ok and b_ok, "testable": testable, "applicable_models": sorted(appl),
                     "b_models": sorted(bmods), "a_every_severe_at_or_before": a_ok, "b_strongest_writer_rule": b_ok,
                     "benign_rate_at_or_before": (sum(x["layer"] <= runs[m]["massive_activation_layer"] for m, x in benign if m in appl) /
                                                  max(1, sum(1 for m, _ in benign if m in appl))),
                     "counts_b": {"severe_giving_strongest_writer_new_partner": sum(x["strongest_writer_new_partner"] for _, x in at_me),
                                  "severe_there": len(at_me),
                                  "benign_giving_strongest_writer_new_partner": sum(x["strongest_writer_new_partner"] for _, x in ben_me),
                                  "benign_there": len(ben_me)},
                     "reported_any_writer_rule": {"severe": sum(x["touches_writer"] for _, x in at_me), "benign": sum(x["touches_writer"] for _, x in ben_me)},
                     "massive_activation_layer": {m: r["massive_activation_layer"] for m, r in runs.items()},
                     "writers": {m: (r["writers"] or {}).get("top3") for m, r in runs.items()},
                     "detail": [{"model": m, "layer": x["layer"], "label": x["label"], "new_pairs": x["new_pairs"],
                                 "strongest_writer_new_partner": x["strongest_writer_new_partner"]} for m, x in collapses]}
    else:
        for g in ("G1", "G2", "G3", "G4", "G5"):
            res[g] = {"pass": False, "untested": True}
    # reported: separation without a threshold (fraction of collapse-benign pairs ordered by the largest ratio)
    if collapses and benign:
        ok = sum((xc["max_ratio"] > xb["max_ratio"]) + 0.5 * (xc["max_ratio"] == xb["max_ratio"]) for _, xc in collapses for _, xb in benign)
        res["reported_separation"] = ok / (len(collapses) * len(benign))
    res["passes"] = sum(bool(res[g]["pass"]) for g in ("G1", "G2", "G3", "G4", "G5")) + (1 if gate else 0)
    res["of"] = 6
    json.dump(res, open("alignment_study/guard_results.json", "w"), indent=1)
    print(json.dumps({k: (v if k in ("passes", "of", "reported_separation") else {kk: vv for kk, vv in v.items() if kk not in ("detail", "per_model")})
                      for k, v in res.items() if k != "per_model"}, indent=1))
    for m, v in per.items():
        print(m, v)
