# Evaluator for PREREGISTRATION26 (committed before any confirmatory result). Reads alignment_study/fuse/*.json
# (this round) and alignment_study/gqa/*.json (round 24, the reference recipes evaluated with the same harness)
# and writes alignment_study/fuse_results.json.
#   dlog(a)  log(perplexity of arm a / perplexity of the unconverted model), WikiText-2
#   acc(a)   average zero-shot accuracy over the seven tasks
# Reference recipes from round 24: sim_align (aligned mean pooling, groups chosen by the simulated merge),
# jin, contiguous (neighbouring heads, mean pooling without alignment).
#   python alignment_study/eval_fuse.py [round directory [round-24 directory]]
import glob
import json
import sys
import numpy as np

sys.path.insert(0, "alignment_study")
from zeroshot import AVG_METRIC

D = sys.argv[1] if len(sys.argv) > 1 else "alignment_study/fuse"
D24 = sys.argv[2] if len(sys.argv) > 2 else "alignment_study/gqa"
OUT = "alignment_study/fuse_results.json" if D == "alignment_study/fuse" else f"{D}_results.json"
FUSE = ("fuse_u/contiguous", "fuse/contiguous", "fuse/jin", "fuse/sim_align", "fuse_u/sim_align")
REF = ("contiguous", "jin", "sim_align")
B = 2000


def load(d):
    out = {}
    for p in sorted(glob.glob(f"{d}/*.json")):
        r = json.load(open(p))
        out[r["model"]] = r
    return out


def correct(ev, task):
    key = "correct_norm" if AVG_METRIC[task] == "acc_norm" else "correct"
    return np.frombuffer(ev[task][key].encode(), dtype=np.uint8).astype(float) - 48


def pooled_ci(diffs_by_model, seed=0):
    """Mean over models of the difference in average accuracy; paired bootstrap resampling examples within
    tasks within models."""
    rng = np.random.default_rng(seed)
    point = float(np.mean([np.mean([d.mean() for d in dm.values()]) for dm in diffs_by_model]))
    boots = np.empty(B)
    for i in range(B):
        boots[i] = np.mean([np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in dm.values()]) for dm in diffs_by_model])
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


if __name__ == "__main__":
    R, R24 = load(D), load(D24)
    models = [m for m in R if m in R24]
    cells = [(m, g) for m in models for g in (2, 4)]

    def ev(m, g, a):
        """Evaluation of arm a: this round's fusion arms, or round 24's reference recipes."""
        return R[m]["evals"][f"g{g}/{a}"] if a in FUSE else R24[m]["evals"][f"g{g}/{a}"]

    base = {m: R[m]["evals"]["base"] for m in models}
    ppl = lambda m, g, a: ev(m, g, a)["wikitext2_ppl"]
    dlog = lambda m, g, a: float(np.log(ppl(m, g, a) / base[m]["wikitext2_ppl"]))
    table = {f"{m} g{g}": {a: {"ppl": ppl(m, g, a), "dlog": dlog(m, g, a), "acc": ev(m, g, a)["average"]} for a in FUSE + REF} for m, g in cells}
    for m in models:
        for g in (2, 4):
            table[f"{m} g{g}"]["base"] = {"ppl": base[m]["wikitext2_ppl"], "acc": base[m]["average"]}
    clauses = {}
    # F1: weighted fusion beats aligned mean pooling on the same groups
    f1 = {f"{m} g{g}": ppl(m, g, "fuse/sim_align") < ppl(m, g, "sim_align") for m, g in cells}
    clauses["F1"] = {"what": "fuse/sim_align perplexity below round 24's sim_align (same groups) in >= 9 of 10 cells", "value": f1,
                     "count": sum(f1.values()), "pass_": sum(f1.values()) >= 9}
    # F2: size at g = 2
    ratios = {m: dlog(m, 2, "fuse/sim_align") / dlog(m, 2, "sim_align") for m in models}
    ok = all(dlog(m, 2, "sim_align") > 0 for m in models)
    gm = float(np.exp(np.mean(np.log([max(v, 1e-9) for v in ratios.values()])))) if ok else float("nan")
    clauses["F2"] = {"what": "g=2: geometric mean over models of dlog(fuse/sim_align)/dlog(sim_align) <= 0.6", "value": {"per_model": ratios, "geometric_mean": gm},
                     "pass_": bool(ok and gm <= 0.6)}
    # F3: weighting beats the unweighted fit on the same groups
    f3 = {f"{m} g{g} {grp}": ppl(m, g, f"fuse/{grp}") < ppl(m, g, f"fuse_u/{grp}") for m, g in cells for grp in ("contiguous", "sim_align")}
    clauses["F3"] = {"what": "weighted fusion perplexity below unweighted fusion on the same groups (contiguous, sim_align) in >= 18 of 20", "value": f3,
                     "count": sum(f3.values()), "pass_": sum(f3.values()) >= 18}
    # F4: accuracy at g = 2
    diffs = {m: {t: correct(ev(m, 2, "fuse/sim_align"), t) - correct(ev(m, 2, "sim_align"), t) for t in AVG_METRIC} for m in models}
    per = {m: float(np.mean([d.mean() for d in diffs[m].values()])) for m in models}
    pt, lo, hi = pooled_ci(list(diffs.values()))
    n4 = sum(v > 0 for v in per.values())
    clauses["F4"] = {"what": "g=2: fuse/sim_align accuracy above round 24's sim_align in >= 4 of 5 models and pooled 95% interval above zero",
                     "value": {"per_model": per, "pooled": [pt, lo, hi]}, "count": n4, "pass_": bool(n4 >= 4 and lo > 0)}
    # F5: no search needed
    f5 = {f"{m} g{g}": ppl(m, g, "fuse/contiguous") < ppl(m, g, "sim_align") for m, g in cells}
    clauses["F5"] = {"what": "fuse/contiguous (neighbouring heads, no search) perplexity below round 24's sim_align in >= 8 of 10 cells", "value": f5,
                     "count": sum(f5.values()), "pass_": sum(f5.values()) >= 8}
    # F6: grouping still matters after fusion
    f6 = {f"{m} g{g}": ppl(m, g, "fuse/sim_align") < ppl(m, g, "fuse/contiguous") for m, g in cells}
    clauses["F6"] = {"what": "fuse/sim_align perplexity below fuse/contiguous in >= 8 of 10 cells", "value": f6,
                     "count": sum(f6.values()), "pass_": sum(f6.values()) >= 8}
    checks = {m: {k: R[m].get(k) for k in ("groups_of_one_max_logit_diff", "random_basis_function_check", "basis_invariance_max_logit_diff",
                                           "logit_scale", "restore_check_ppl")} for m in models}
    for m in models:
        checks[m]["base_ppl_round24"] = R24[m]["evals"]["base"]["wikitext2_ppl"]
        checks[m]["base_ppl_this_round"] = base[m]["wikitext2_ppl"]
    reported = {f"{m} g{g}": {"acc_diff_fuse_sim_align_minus_sim_align": pooled_ci([{t: correct(ev(m, g, "fuse/sim_align"), t) - correct(ev(m, g, "sim_align"), t) for t in AVG_METRIC}]),
                              "ppl_over_base": {a: ppl(m, g, a) / base[m]["wikitext2_ppl"] for a in FUSE + REF}} for m, g in cells}
    res = {"models": models, "clauses": clauses, "passes": sum(1 for c in clauses.values() if c["pass_"]), "of": len(clauses),
           "table": table, "reported": reported, "checks": checks}
    json.dump(res, open(OUT, "w"), indent=1)
    for k, c in clauses.items():
        print(k, "PASS" if c["pass_"] else "FAIL", c["what"], "|", c.get("count", ""), json.dumps(c["value"])[:300])
    print(f"{res['passes']} of {res['of']} clauses pass; written {OUT}")
