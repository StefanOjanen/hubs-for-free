# Evaluator for PREREGISTRATION28 (committed before any confirmatory result). Reads the round's result file and
# writes alignment_study/llama2_results.json (for a development file: <file>_results.json next to it).
#   ppl_T(a)   WikiText-2 perplexity of arm a over windows of T tokens (T = 2,048 and 4,096)
#   ZS-7(a)    mean over the seven tasks of accuracy (BoolQ, RTE, WinoGrande) or length-normalized accuracy
#              (HellaSwag, ARC-Easy, ARC-Challenge, OpenBookQA), the convention of CAGA's paper
#   overlap(a) steering agreement with the unconverted model: the mean over the 64 neuron directions of the
#              overlap of the 20 tokens whose probability each direction raises most
# Published reference (Wang et al., arXiv 2610.09476, Table 2, Llama-2-7B, 16 KV heads, before any training):
# CAGA 25.1 and 44.3; direct averaging 926 and 32.6; Procrustes alignment 738 and 35.3.
#   python alignment_study/eval_llama2.py [result file]
import json
import sys
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "alignment_study/llama2/Llama-2-7b-hf.json"
OUT = "alignment_study/llama2_results.json" if PATH == "alignment_study/llama2/Llama-2-7b-hf.json" else PATH.replace(".json", "_results.json")
CAGA_PPL, CAGA_ZS7 = 25.1, 0.443
PUBLISHED = {"caga": (25.1, 0.443), "direct averaging": (926, 0.326), "procrustes alignment": (738, 0.353)}
B = 2000


def correct(r, task, metric):
    s = r[task]["correct_norm" if metric == "acc_norm" else "correct"]
    return np.frombuffer(s.encode(), dtype=np.uint8).astype(float) - 48


def zs7_boot(r, metrics, rng):
    """Bootstrap draws of ZS-7: examples resampled within each task."""
    vs = [correct(r, t, m) for t, m in metrics.items()]
    return np.array([np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in vs]) for _ in range(B)])


def paired_diff_boot(pairs, metrics, rng):
    """Mean over the given (a, b) result pairs of ZS-7(a) - ZS-7(b), with the examples of each task resampled once
    per draw and shared by every pair (they are the same examples)."""
    d = {t: [correct(a, t, m) - correct(b, t, m) for a, b in pairs] for t, m in metrics.items()}
    point = float(np.mean([np.mean([x.mean() for x in d[t]]) for t in metrics]))
    boots = np.empty(B)
    for k in range(B):
        acc = []
        for t in metrics:
            idx = rng.integers(0, len(d[t][0]), len(d[t][0]))
            acc.append(np.mean([x[idx].mean() for x in d[t]]))
        boots[k] = np.mean(acc)
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


if __name__ == "__main__":
    R = json.load(open(PATH))
    E, L, metrics = R["evals"], R["lengths"], R["zs7_metrics"]
    rng = np.random.default_rng(0)
    ppl = lambda a, T: E[a][f"ppl_{T}"]
    clauses = {}
    m1 = {T: ppl("fuse/paired", T) for T in L}
    clauses["M1"] = {"what": f"fuse/paired perplexity below CAGA's published {CAGA_PPL} at every evaluated length {L}",
                     "value": m1, "pass_": all(v < CAGA_PPL for v in m1.values())}
    bz = zs7_boot(E["fuse/paired"], metrics, rng)
    lo, hi = float(np.quantile(bz, 0.025)), float(np.quantile(bz, 0.975))
    clauses["M2"] = {"what": f"fuse/paired ZS-7 above CAGA's published {CAGA_ZS7}, the lower end of its 95% bootstrap interval included",
                     "value": {"zs7": E["fuse/paired"]["zs7"], "ci": [lo, hi]}, "pass_": bool(E["fuse/paired"]["zs7"] > CAGA_ZS7 and lo > CAGA_ZS7)}
    m3 = {f"{p} {T}": ppl(f"fuse/{p}", T) < ppl(f"caga/{p}", T) for p in ("neighbour", "paired") for T in L}
    clauses["M3"] = {"what": "fuse perplexity below caga on the same pairs, both pairings, every length", "value": m3,
                     "count": sum(m3.values()), "pass_": all(m3.values())}
    pt, dlo, dhi = paired_diff_boot([(E[f"fuse/{p}"], E[f"caga/{p}"]) for p in ("neighbour", "paired")], metrics, rng)
    per = {p: E[f"fuse/{p}"]["zs7"] - E[f"caga/{p}"]["zs7"] for p in ("neighbour", "paired")}
    clauses["M4"] = {"what": "fuse ZS-7 above caga on both pairings, and the pooled paired 95% interval of the difference above zero",
                     "value": {"per_pairing": per, "pooled": [pt, dlo, dhi]}, "pass_": bool(all(v > 0 for v in per.values()) and dlo > 0)}
    m5 = {T: ppl("fuse/paired", T) < ppl("fuse/neighbour", T) for T in L}
    clauses["M5"] = {"what": "the matching lowers the fusion's perplexity against neighbour pairs at every length",
                     "value": {T: [ppl("fuse/paired", T), ppl("fuse/neighbour", T)] for T in L}, "pass_": all(m5.values())}
    ov = lambda a: E[a]["steer"]["overlap_neuron"]
    m6 = {"fuse>caga neighbour": ov("fuse/neighbour") > ov("caga/neighbour"), "fuse>caga paired": ov("fuse/paired") > ov("caga/paired"),
          "fuse/paired>mean/neighbour": ov("fuse/paired") > ov("mean/neighbour")}
    clauses["M6"] = {"what": "steering agreement (neuron directions) higher for fuse than for caga on both pairings and than for averaging",
                     "value": {"tests": m6, "overlap_neuron": {a: ov(a) for a in E}}, "pass_": all(m6.values())}
    arms = [a for a in E if a != "base"]
    reported = {
        "table": {a: {**{f"ppl_{T}": ppl(a, T) for T in L}, **{f"ppl_{T}_over_base": ppl(a, T) / ppl("base", T) for T in L},
                      "zs7": E[a]["zs7"], "zs7_acc": E[a]["zs7_acc"], "per_task": {t: E[a][t][m] for t, m in metrics.items()},
                      "steer": {k: v for k, v in E[a]["steer"].items() if k not in ("overlap", "cos")}} for a in E},
        "published": PUBLISHED,
        "zs7_ci": {a: [float(np.quantile(b_, q)) for q in (0.025, 0.975)] for a in E for b_ in [zs7_boot(E[a], metrics, rng)]},
        "fuse_minus_mean_zs7": paired_diff_boot([(E["fuse/neighbour"], E["mean/neighbour"])], metrics, rng),
        "fuse_minus_caga_by_pairing": {p: paired_diff_boot([(E[f"fuse/{p}"], E[f"caga/{p}"])], metrics, rng) for p in ("neighbour", "paired")},
        "steering_overlap_vs_log_ppl_spearman": None,
        "checks": {k: R.get(k) for k in ("groups_of_one_max_logit_diff", "logit_scale", "restore_check_ppl", "dtype", "revision")},
        "pairing": {"cost_neighbour_total": sum(R["pairing"]["cost_neighbour"]), "cost_paired_total": sum(R["pairing"]["cost_paired"]),
                    "neighbour_pairs_kept": R["pairing"]["neighbour_pairs_kept"]},
    }
    if len(arms) >= 3:
        from scipy.stats import spearmanr
        reported["steering_overlap_vs_log_ppl_spearman"] = float(spearmanr([ov(a) for a in arms], [np.log(ppl(a, L[0])) for a in arms])[0])
    res = {"file": PATH, "dev": R.get("dev"), "registered": R.get("registered"), "clauses": clauses,
           "passes": sum(1 for c in clauses.values() if c["pass_"]), "of": len(clauses), "reported": reported}
    json.dump(res, open(OUT, "w"), indent=1)
    for k, c in clauses.items():
        print(k, "PASS" if c["pass_"] else "FAIL", c["what"], "|", json.dumps(c["value"])[:300])
    print(f"{res['passes']} of {res['of']} clauses pass; written {OUT}")
