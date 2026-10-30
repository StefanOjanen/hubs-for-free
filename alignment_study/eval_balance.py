# Evaluator for PREREGISTRATION30 (committed before any confirmatory result). Reads alignment_study/balance/Q_*.json
# and T_*.json (written by balance_round.py --registered=...) and writes alignment_study/balance_results.json.
#   python alignment_study/eval_balance.py [--dir=alignment_study/balance]
import glob
import json
import math
import sys

DIR = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")), "alignment_study/balance")
Q_MODELS = ["OLMo-1B-hf", "h2o-danube-1.8b-base", "open_llama_3b_v2", "Llama-2-7b-hf", "Mistral-7B-Instruct-v0.2", "Qwen2.5-7B"]
FLOAT32 = {"OLMo-1B-hf", "h2o-danube-1.8b-base", "open_llama_3b_v2"}
T_MODELS = ["OLMo-1B-hf", "h2o-danube-1.8b-base"]

if __name__ == "__main__":
    Q = {m: json.load(open(f"{DIR}/Q_{m}.json")) for m in Q_MODELS if glob.glob(f"{DIR}/Q_{m}.json")}
    T = {m: json.load(open(f"{DIR}/T_{m}.json")) for m in T_MODELS if glob.glob(f"{DIR}/T_{m}.json")}
    res = {"missing": [m for m in Q_MODELS if m not in Q] + [f"T:{m}" for m in T_MODELS if m not in T]}
    q1 = {m: {"ppl_change": abs(r["balanced"] / r["fused"] - 1), "logit_change": r["max_logit_change_relative"]} for m, r in Q.items()}
    res["Q1"] = {"pass": len(Q) == 6 and all(v["ppl_change"] <= 0.005 for v in q1.values()) and
                 all(q1[m]["logit_change"] <= 1e-4 for m in FLOAT32 if m in q1), "per_model": q1}
    q2 = {m: {"fused_w4": r["fused_w4"], "balanced_w4": r["balanced_w4"]} for m, r in Q.items()}
    res["Q2"] = {"pass": len(Q) == 6 and all(v["balanced_w4"] < v["fused_w4"] for v in q2.values()),
                 "lower_in": sum(v["balanced_w4"] < v["fused_w4"] for v in q2.values()), "of": len(Q), "per_model": q2}
    cost_b = {m: math.log(r["balanced_w4"] / r["balanced"]) for m, r in Q.items()}
    cost_f = {m: math.log(r["fused_w4"] / r["fused"]) for m, r in Q.items()}
    pooled = sum(cost_b.values()) / sum(cost_f.values()) if Q else None
    res["Q3"] = {"pass": len(Q) == 6 and pooled is not None and pooled <= 0.75, "pooled_ratio": pooled,
                 "per_model_ratio": {m: cost_b[m] / cost_f[m] for m in Q}, "cost_rebalanced": cost_b, "cost_fused": cost_f}
    res["reported_3bit"] = {m: {"fused_w3": r["fused_w3"], "balanced_w3": r["balanced_w3"], "fused": r["fused"]} for m, r in Q.items()}

    def best(runs, start):
        rs = [r for r in runs if r["start"] == start]
        return min(rs, key=lambda r: r["final"]) if rs else None
    t1, t2, t3, rep = {}, {}, {}, {}
    for m, r in T.items():
        bb, bu, bm = best(r["runs"], "fuse_balanced"), best(r["runs"], "fuse"), best(r["runs"], "mean")
        t1[m] = {"balanced_best": bb["final"], "balanced_lr": bb["lr"], "unbalanced_best": bu["final"], "unbalanced_lr": bu["lr"],
                 "pass": bb["final"] < bu["final"]}
        pairs = []
        for lr in r["lrs"]:
            fb = next(x for x in r["runs"] if x["start"] == "fuse_balanced" and x["lr"] == lr)
            fu = next(x for x in r["runs"] if x["start"] == "fuse" and x["lr"] == lr)
            pairs.append({"lr": lr, "balanced": fb["final"], "unbalanced": fu["final"], "ok": fb["final"] <= fu["final"]})
        t2[m] = pairs
        below = [cb["ppl"] < cm["ppl"] for cb, cm in zip(bb["curve"], bm["curve"])]
        t3[m] = {"checkpoints_below": sum(below), "of": len(below), "averaging_best_lr": bm["lr"], "averaging_best_final": bm["final"]}
        rep[m] = {"base": r["base"], "curves": {f"{x['start']}@{x['lr']:g}": [c["ppl"] for c in x["curve"]] for x in r["runs"]}}
    res["T1"] = {"pass": len(T) == 2 and all(v["pass"] for v in t1.values()), "per_model": t1}
    res["T2"] = {"pass": len(T) == 2 and all(p["ok"] for v in t2.values() for p in v),
                 "at_or_below": sum(p["ok"] for v in t2.values() for p in v), "of": sum(len(v) for v in t2.values()), "per_model": t2}
    res["T3"] = {"pass": len(T) == 2 and all(v["checkpoints_below"] == v["of"] == 7 for v in t3.values()), "per_model": t3}
    res["reported_training"] = rep
    res["passes"] = sum(bool(res[k]["pass"]) for k in ("Q1", "Q2", "Q3", "T1", "T2", "T3"))
    res["of"] = 6
    json.dump(res, open("alignment_study/balance_results.json", "w"), indent=1)
    for k in ("Q1", "Q2", "Q3", "T1", "T2", "T3"):
        print(k, "PASS" if res[k]["pass"] else "FAIL", {kk: vv for kk, vv in res[k].items() if kk not in ("pass",)})
    print("passes", res["passes"], "of", res["of"], "missing", res["missing"])
