# Evaluator for PREREGISTRATION24 (committed before any confirmatory result). Reads
# alignment_study/gqa/*.json and alignment_study/gqa_convention/*.json and writes
# alignment_study/gqa_results.json.
#   dlog(a)  log(perplexity of arm a / perplexity of the unconverted model), WikiText-2
#   acc(a)   average zero-shot accuracy over the seven tasks (zeroshot.AVG_METRIC)
# Clauses P1 to P8 and C1, C2 as registered in PREREGISTRATION24.md.
#   python alignment_study/eval_gqa.py [round directory [convention directory]]
import glob
import json
import sys
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "alignment_study")
from zeroshot import AVG_METRIC

D = sys.argv[1] if len(sys.argv) > 1 else "alignment_study/gqa"
DC = sys.argv[2] if len(sys.argv) > 2 else "alignment_study/gqa_convention"
OUT = "alignment_study/gqa_results.json" if D == "alignment_study/gqa" else f"{D}_results.json"
ARMS = ("contiguous", "random0", "wse", "act", "sim", "contiguous_align", "random0_align", "jin", "sim_align")
ALIGNED = {"contiguous_align", "random0_align", "jin", "sim_align"}
UNALIGNED = ("contiguous", "random0", "wse", "act", "sim")
B = 2000


def load(d):
    out = {}
    for p in sorted(glob.glob(f"{d}/*.json")):
        r = json.load(open(p))
        out[r["model"]] = r
    return out


def ppl(r, g, a):
    return r["evals"][f"g{g}/{a}"]["wikitext2_ppl"]


def dlog(r, g, a):
    return float(np.log(ppl(r, g, a) / r["evals"]["base"]["wikitext2_ppl"]))


def correct(ev, task):
    key = "correct_norm" if AVG_METRIC[task] == "acc_norm" else "correct"
    return np.frombuffer(ev[task][key].encode(), dtype=np.uint8).astype(float) - 48


def acc_diffs(r, g, a, b):
    """Per task: per-example correctness of arm a minus arm b."""
    ea, eb = r["evals"][f"g{g}/{a}"], r["evals"][f"g{g}/{b}"]
    return {t: correct(ea, t) - correct(eb, t) for t in AVG_METRIC}


def pooled_ci(diffs_by_model, seed=0):
    """Mean over models of the difference in average accuracy, with a paired bootstrap that resamples
    examples within tasks within models."""
    rng = np.random.default_rng(seed)
    point = float(np.mean([np.mean([d.mean() for d in dm.values()]) for dm in diffs_by_model]))
    boots = np.empty(B)
    for i in range(B):
        boots[i] = np.mean([np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in dm.values()]) for dm in diffs_by_model])
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def sim_cost(r, g, a):
    diag = r["sim_diagnostics"]
    return float(sum(diag[l][str(g)][a]["sim_align" if a in ALIGNED else "sim"] for l in diag))


if __name__ == "__main__":
    R = load(D)
    models = list(R)
    cells = [(m, g) for m in models for g in (2, 4)]
    clauses, table = {}, {}
    for m, g in cells:
        r = R[m]
        table[f"{m} g{g}"] = {a: {"ppl": ppl(r, g, a), "dlog": dlog(r, g, a), "acc": r["evals"][f"g{g}/{a}"]["average"], "sim_cost": sim_cost(r, g, a)} for a in ARMS}
        table[f"{m} g{g}"]["base"] = {"ppl": r["evals"]["base"]["wikitext2_ppl"], "acc": r["evals"]["base"]["average"]}
    # P1: alignment lowers perplexity for contiguous and random0
    p1 = {f"{m} g{g} {a}": ppl(R[m], g, a + "_align") < ppl(R[m], g, a) for m, g in cells for a in ("contiguous", "random0")}
    clauses["P1"] = {"what": "aligning before pooling lowers perplexity (contiguous, random0) in >= 18 of 20", "value": p1,
                     "count": sum(p1.values()), "of": len(p1), "pass_": sum(p1.values()) >= 18}
    # P2: sim_align below jin
    p2 = {f"{m} g{g}": ppl(R[m], g, "sim_align") < ppl(R[m], g, "jin") for m, g in cells}
    clauses["P2"] = {"what": "sim_align perplexity below jin's in >= 8 of 10 cells", "value": p2, "count": sum(p2.values()), "pass_": sum(p2.values()) >= 8}
    # P3: sim_align lowest of nine
    p3 = {f"{m} g{g}": min(ARMS, key=lambda a: ppl(R[m], g, a)) for m, g in cells}
    n3 = sum(v == "sim_align" for v in p3.values())
    clauses["P3"] = {"what": "sim_align lowest perplexity of the nine arms in >= 8 of 10 cells", "value": p3, "count": n3, "pass_": n3 >= 8}
    # P4: geometric mean of dlog ratios at g = 2
    ratios = {m: dlog(R[m], 2, "sim_align") / dlog(R[m], 2, "jin") for m in models}
    ok = all(dlog(R[m], 2, "jin") > 0 and dlog(R[m], 2, "sim_align") > 0 for m in models)
    gm = float(np.exp(np.mean(np.log(list(ratios.values()))))) if ok else float("nan")
    clauses["P4"] = {"what": "g=2: geometric mean over models of dlog(sim_align)/dlog(jin) <= 0.85", "value": {"per_model": ratios, "geometric_mean": gm},
                     "pass_": bool(ok and gm <= 0.85)}
    # P5: accuracy at g = 2
    diffs = {m: acc_diffs(R[m], 2, "sim_align", "jin") for m in models}
    per = {m: float(np.mean([d.mean() for d in diffs[m].values()])) for m in models}
    pt, lo, hi = pooled_ci(list(diffs.values()))
    n5 = sum(v > 0 for v in per.values())
    clauses["P5"] = {"what": "g=2: sim_align accuracy above jin's in >= 4 of 5 models and pooled 95% interval above zero",
                     "value": {"per_model": per, "pooled": [pt, lo, hi]}, "count": n5, "pass_": bool(n5 >= 4 and lo > 0)}
    # P6: simulated cost orders the arms' damage
    p6 = {f"{m} g{g}": float(spearmanr([sim_cost(R[m], g, a) for a in ARMS], [dlog(R[m], g, a) for a in ARMS]).correlation) for m, g in cells}
    n6 = sum(v >= 0.6 for v in p6.values())
    clauses["P6"] = {"what": "Spearman(simulated cost, dlog) across the nine arms >= 0.6 in >= 8 of 10 cells", "value": p6, "count": n6, "pass_": n6 >= 8}
    # P7: every unaligned arm >= 5x base perplexity
    p7 = {f"{m} g{g}": min(ppl(R[m], g, a) for a in UNALIGNED) / R[m]["evals"]["base"]["wikitext2_ppl"] for m, g in cells}
    n7 = sum(v >= 5 for v in p7.values())
    clauses["P7"] = {"what": "every unaligned arm >= 5x base perplexity in >= 9 of 10 cells", "value": p7, "count": n7, "pass_": n7 >= 9}
    # P8: sim below contiguous without alignment at g = 2
    p8 = {m: ppl(R[m], 2, "sim") < ppl(R[m], 2, "contiguous") for m in models}
    clauses["P8"] = {"what": "g=2: sim perplexity below contiguous in >= 4 of 5 models", "value": p8, "count": sum(p8.values()), "pass_": sum(p8.values()) >= 4}
    # C1, C2: the convention null
    RC = load(DC)
    c1, c2 = {}, {}
    for m in models:
        for g in (2, 4):
            if m not in RC:
                continue
            u = RC[m]["groups"][str(g)]["unaligned"]; al = RC[m]["groups"][str(g)]["aligned"]
            c1[f"{m} g{g}"] = max(u.values()) / min(u.values())
            for sd in ("0", "1", "2"):
                c2[f"{m} g{g} basis {sd}"] = abs(al[sd] / al["original"] - 1)
    n1 = sum(v >= 1.1 for v in c1.values())
    clauses["C1"] = {"what": "unaligned neighbouring pooling: perplexity spread across four bases >= 1.1 in >= 8 of 10 cells", "value": c1,
                     "count": n1, "pass_": n1 >= 8 and len(c1) == len(cells)}
    n2 = sum(v <= 0.001 for v in c2.values())
    clauses["C2"] = {"what": "aligned neighbouring pooling: perplexity within 0.1 percent of the original basis in all 30 cases", "value": c2,
                     "count": n2, "pass_": n2 == len(c2) == 3 * len(cells)}
    convention = {m: {g: {"aligned_vs_original_basis_max_logit_diff": RC[m]["groups"][g]["aligned_vs_original_basis"],
                          "function_check_random_bases": RC[m]["groups"][g]["function_check"]} for g in RC[m]["groups"]} for m in RC}
    reported = {f"{m} g{g}": {"acc_diff_sim_align_minus_jin": pooled_ci([acc_diffs(R[m], g, "sim_align", "jin")]),
                              "wse_vs_random0_ppl_ratio": ppl(R[m], g, "wse") / ppl(R[m], g, "random0"),
                              "act_vs_random0_ppl_ratio": ppl(R[m], g, "act") / ppl(R[m], g, "random0")} for m, g in cells}
    checks = {m: {k: R[m].get(k) for k in ("capture_vs_eager", "sdpa_vs_eager", "alignment_max_logit_diff", "restore_check_ppl")} for m in models}
    res = {"models": models, "clauses": clauses, "passes": sum(1 for c in clauses.values() if c["pass_"]), "of": len(clauses),
           "table": table, "reported": reported, "checks": checks, "convention_checks": convention}
    json.dump(res, open(OUT, "w"), indent=1)
    for k, c in clauses.items():
        print(k, "PASS" if c["pass_"] else "FAIL", c["what"], "|", c.get("count", ""), json.dumps(c["value"])[:300])
    print(f"{res['passes']} of {res['of']} clauses pass; written {OUT}")
