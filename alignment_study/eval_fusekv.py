# Evaluator for PREREGISTRATION27 (committed before any confirmatory result). Reads alignment_study/fusekv/*.json
# and writes alignment_study/fusekv_results.json.
#   dlog(a)  log(perplexity of a / perplexity of the unconverted model), WikiText-2
#   acc(a)   average zero-shot accuracy over the seven tasks
# Cells: (model, group size) for every group size the model's KV-head count allows (2, and 4 where possible).
#   python alignment_study/eval_fusekv.py [round directory]
import glob
import json
import sys
import numpy as np

sys.path.insert(0, "alignment_study")
from zeroshot import AVG_METRIC

D = sys.argv[1] if len(sys.argv) > 1 else "alignment_study/fusekv"
OUT = "alignment_study/fusekv_results.json" if D == "alignment_study/fusekv" else f"{D}_results.json"
METHODS = ("mean", "fuse_u", "fuse")
B = 2000
K3_MAX = 0.5


def correct(ev, task):
    key = "correct_norm" if AVG_METRIC[task] == "acc_norm" else "correct"
    return np.frombuffer(ev[task][key].encode(), dtype=np.uint8).astype(float) - 48


def pooled_ci(diffs_by_model, seed=0):
    rng = np.random.default_rng(seed)
    point = float(np.mean([np.mean([d.mean() for d in dm.values()]) for dm in diffs_by_model]))
    boots = np.empty(B)
    for i in range(B):
        boots[i] = np.mean([np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in dm.values()]) for dm in diffs_by_model])
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


if __name__ == "__main__":
    R = {}
    for p in sorted(glob.glob(f"{D}/*.json")):
        r = json.load(open(p)); R[r["model"]] = r
    models = list(R)
    cells = [(m, g) for m in models for g in R[m]["group_sizes"]]
    ev = lambda m, g, a: R[m]["evals"][f"g{g}/{a}"]
    base = {m: R[m]["evals"]["base"] for m in models}
    ppl = lambda m, g, a: ev(m, g, a)["wikitext2_ppl"]
    dlog = lambda m, g, a: float(np.log(ppl(m, g, a) / base[m]["wikitext2_ppl"]))
    table = {f"{m} g{g}": {a: {"ppl": ppl(m, g, a), "dlog": dlog(m, g, a), "acc": ev(m, g, a)["average"]} for a in METHODS} for m, g in cells}
    for m, g in cells:
        table[f"{m} g{g}"]["base"] = {"ppl": base[m]["wikitext2_ppl"], "acc": base[m]["average"]}
    n = len(cells)
    clauses = {}
    k1 = {f"{m} g{g}": ppl(m, g, "fuse") < ppl(m, g, "mean") for m, g in cells}
    clauses["K1"] = {"what": f"weighted fusion perplexity below mean pooling in >= {n - 1} of {n} cells", "value": k1,
                     "count": sum(k1.values()), "pass_": sum(k1.values()) >= n - 1}
    k2 = {f"{m} g{g}": ppl(m, g, "fuse") < ppl(m, g, "fuse_u") for m, g in cells}
    clauses["K2"] = {"what": f"weighted fusion perplexity below unweighted fusion in >= {n - 1} of {n} cells", "value": k2,
                     "count": sum(k2.values()), "pass_": sum(k2.values()) >= n - 1}
    ratios = {m: dlog(m, 2, "fuse") / dlog(m, 2, "mean") for m in models}
    ok = all(dlog(m, 2, "mean") > 0 for m in models)
    gm = float(np.exp(np.mean(np.log([max(v, 1e-9) for v in ratios.values()])))) if ok else float("nan")
    clauses["K3"] = {"what": f"g=2: geometric mean over models of dlog(fuse)/dlog(mean) <= {K3_MAX}", "value": {"per_model": ratios, "geometric_mean": gm},
                     "pass_": bool(ok and gm <= K3_MAX)}
    diffs = {m: {t: correct(ev(m, 2, "fuse"), t) - correct(ev(m, 2, "mean"), t) for t in AVG_METRIC} for m in models}
    per = {m: float(np.mean([d.mean() for d in diffs[m].values()])) for m in models}
    pt, lo, hi = pooled_ci(list(diffs.values()))
    n4 = sum(v > 0 for v in per.values())
    clauses["K4"] = {"what": f"g=2: weighted fusion accuracy above mean pooling in >= {len(models) - 1} of {len(models)} models and pooled 95% interval above zero",
                     "value": {"per_model": per, "pooled": [pt, lo, hi]}, "count": n4, "pass_": bool(n4 >= len(models) - 1 and lo > 0)}
    checks = {m: {k: R[m].get(k) for k in ("groups_of_one_max_logit_diff", "logit_scale", "restore_check_ppl", "dtype")} for m in models}
    for m in models:
        checks[m]["base_ppl"] = base[m]["wikitext2_ppl"]
    reported = {f"{m} g{g}": {"acc_diff_fuse_minus_mean": pooled_ci([{t: correct(ev(m, g, "fuse"), t) - correct(ev(m, g, "mean"), t) for t in AVG_METRIC}]),
                              "acc_diff_fuse_minus_fuse_u": pooled_ci([{t: correct(ev(m, g, "fuse"), t) - correct(ev(m, g, "fuse_u"), t) for t in AVG_METRIC}]),
                              "ppl_over_base": {a: ppl(m, g, a) / base[m]["wikitext2_ppl"] for a in METHODS}} for m, g in cells}
    # reported, not scored: how close the weighted fusion comes to the unconverted model at g = 2
    res = {"models": models, "cells": [f"{m} g{g}" for m, g in cells], "clauses": clauses,
           "passes": sum(1 for c in clauses.values() if c["pass_"]), "of": len(clauses), "table": table, "reported": reported, "checks": checks}
    json.dump(res, open(OUT, "w"), indent=1)
    for k, c in clauses.items():
        print(k, "PASS" if c["pass_"] else "FAIL", c["what"], "|", c.get("count", ""), json.dumps(c["value"])[:300])
    print(f"{res['passes']} of {res['of']} clauses pass; written {OUT}")
