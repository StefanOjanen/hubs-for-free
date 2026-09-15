# Evaluation of the preregistration 4 batteries against the frozen criteria.
#
# COMMITTED BEFORE ANY BATTERY RESULT EXISTS (exercised only on the dry-run
# outputs under /tmp). Reads the battery result files, applies the
# per-target registered expectations and the common outcome labels of
# audits/PREREGISTRATION4.md (frozen at commit ee4e267) and writes
# audits/battery_results.json and audits/RESULTS.md. A missing file is
# reported as pending, never guessed. A result file marked dry_run is
# refused outside --dry-run.
#
# Readings fixed here, before results (PREREGISTRATION4_ADDENDA.md,
# addendum 4):
# - "shrinkage" is the share of the real effect reproduced by a null, the
#   frozen file's definition (null median / real; for Target 3 the excess
#   over the random-null median). "Shrinks by X percent" means shrinkage X.
# - A clause stated per layer (Target 3) holds for a model when it holds in
#   at least two thirds of the layers, the fraction the frozen file names
#   for the column-set clause.
# - Where the battery recorded a mid-rank percentile (ties in cluster
#   shares) it is used; "matched" means that percentile lies in [5, 95].
# - With five or three untrained initializations, a 95th or 99th percentile
#   clause holds only if the real statistic exceeds every draw.
# - E3 failing withholds every other verdict, as the frozen file requires.
#
#   python audits/eval_batteries.py [--dry-run]
import datetime
import glob
import json
import math
import os
import sys

DRY = "--dry-run" in sys.argv
if DRY:
    FILES = {"clark": ["/tmp/clark2019_battery_dryrun.json"], "dewage": ["/tmp/dewage2026_battery_dryrun.json"],
             "chai": ["/tmp/chai_battery_dryrun.json"], "retrieval": ["/tmp/retrieval_battery_dryrun.json"]}
    OUT_JSON, OUT_MD = "/tmp/battery_results_dryrun.json", "/tmp/RESULTS_dryrun.md"
else:
    FILES = {"clark": ["audits/clark2019/battery_result.json"], "dewage": ["audits/dewage2026/battery_result.json"],
             "chai": sorted(glob.glob("audits/chai/battery_result*.json")),
             "retrieval": sorted(glob.glob("audits/retrieval_heads/battery_result*.json"))}
    OUT_JSON, OUT_MD = "audits/battery_results.json", "audits/RESULTS.md"


def load(path):
    if not os.path.exists(path):
        return None
    r = json.load(open(path))
    if r.get("dry_run") and not DRY:
        sys.exit(f"{path} is a dry-run output; refusing to evaluate it as a result")
    return r


def label_common(percentiles, shrink):
    """Common outcome label of the frozen file. percentiles: percentile of the
    real statistic in every null family; shrink: the largest shrinkage against
    the marginal-matched and column-preserving surrogates."""
    if all(p > 95 for p in percentiles) and shrink < 0.5:
        return "survives"
    if any(p > 95 for p in percentiles) and shrink >= 0.5:
        return "shrinks"
    return "matched"


def f3(x):
    return "pending" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


# ---------------------------------------------------------------- Target 1
def eval_clark(r):
    n, real = r["nulls"], r["real"]
    pct = {f: n[f]["percentile_of_real"] for f in ("a_random", "b_marginal", "c_colfix", "d_untrained")}
    shr = {f: n[f]["shrinkage"] for f in ("b_marginal", "c_colfix")}
    shr_all = {f: n[f]["shrinkage"] for f in n}   # reported for every family; the clauses use (b) and (c)
    clauses = {
        "survives (a) random maps: percentile above 95": pct["a_random"] > 95,
        "survives (d) untrained BERT: real above all five initializations": pct["d_untrained"] > 95,
        "(b) marginal-matched: percentile above 95 and shrinkage in [0.2, 0.7]": pct["b_marginal"] > 95 and 0.2 <= shr["b_marginal"] <= 0.7,
        "(c) separator columns kept: percentile above 95 and shrinkage in [0.2, 0.7]": pct["c_colfix"] > 95 and 0.2 <= shr["c_colfix"] <= 0.7,
    }
    return {"target": r["target"], "model": r["model"], "registered": r.get("registered"), "windows": r["n_windows"], "draws": r["draws"],
            "real": real, "percentile_of_real": pct, "shrinkage": shr, "shrinkage_all": shr_all,
            "null_D_median": {f: n[f]["D_median"] for f in n}, "null_nn_median": {f: n[f]["nn_median"] for f in n},
            "label": label_common(list(pct.values()), max(shr.values())), "battery_label": r.get("outcome"),
            "clauses": clauses, "expectation_holds": all(clauses.values()),
            "all_shrinkage_under_0.2": max(shr.values()) < 0.2, "runtime_s": r.get("runtime_s")}


# ---------------------------------------------------------------- Target 3
def pct_of(ent, key):
    return ent[key].get("percentile_mid", ent[key]["percentile_of_real"])


def eval_chai(r):
    layers = r["layers"]; L = len(layers); need = math.ceil(2 * L / 3)
    rows, labels = [], []
    cnt = {"c_matched_both": 0, "c_matched_corr": 0, "a_survive_corr": 0, "d_survive_corr": 0, "b_over_half": 0, "bc_under_0.2": 0}
    for lay in layers:
        nl = lay["nulls"]
        c_corr = 5 <= pct_of(nl["c_colset"], "mean_corr") <= 95
        c_cl = 5 <= pct_of(nl["c_colset"], "largest95") <= 95
        a_ok = pct_of(nl["a_random"], "mean_corr") > 95
        d_ok = pct_of(nl["d_untrained"], "mean_corr") > 95
        b_rep = nl["b_marginal"]["reproduced_mean_corr"]; c_rep = nl["c_colset"]["reproduced_mean_corr"]
        b_ok = b_rep is not None and b_rep > 0.5
        cnt["c_matched_both"] += c_corr and c_cl; cnt["c_matched_corr"] += c_corr
        cnt["a_survive_corr"] += a_ok; cnt["d_survive_corr"] += d_ok; cnt["b_over_half"] += b_ok
        reps = [x for x in (b_rep, c_rep) if x is not None]
        cnt["bc_under_0.2"] += bool(reps) and max(reps) < 0.2
        lab = label_common([pct_of(nl[f], "mean_corr") for f in ("a_random", "b_marginal", "c_colset", "d_untrained")], max(reps) if reps else 1.0)
        labels.append(lab)
        rows.append({"layer": lay["layer"], "real_corr": lay["real"]["mean_corr"], "real_largest95": lay["real"]["largest95"],
                     "a_median": nl["a_random"]["mean_corr"]["median"], "b_median": nl["b_marginal"]["mean_corr"]["median"],
                     "c_median": nl["c_colset"]["mean_corr"]["median"], "d_median": nl["d_untrained"]["mean_corr"]["median"],
                     "b_reproduced": b_rep, "c_reproduced": c_rep, "c_pct_corr": pct_of(nl["c_colset"], "mean_corr"),
                     "c_pct_largest95": pct_of(nl["c_colset"], "largest95"), "label": lab, "sink_sets": lay.get("sink_sets")})
    clauses = {
        f"(c) sink set kept matches correlation and cluster share in at least {need} of {L} layers": cnt["c_matched_both"] >= need,
        f"correlation survives (a) random rows (percentile above 95) in at least {need} of {L} layers": cnt["a_survive_corr"] >= need,
        f"correlation survives (d) untrained model (above all five initializations) in at least {need} of {L} layers": cnt["d_survive_corr"] >= need,
        f"(b) marginal-matched reproduces more than half of the correlation excess in at least {need} of {L} layers": cnt["b_over_half"] >= need,
    }
    maj = max(set(labels), key=labels.count)
    return {"target": r["target"], "model": r["model"], "registered": r.get("registered"), "T": r["T"], "samples": r["samples"], "draws": r["draws"],
            "n_layers": L, "counts": cnt, "label_counts": {k: labels.count(k) for k in ("survives", "shrinks", "matched")}, "label": maj,
            "clauses": clauses, "expectation_holds": all(clauses.values()), "c_matched_clause": clauses[list(clauses)[0]],
            "mean_real_corr": sum(x["real_corr"] for x in rows) / L,
            "all_shrinkage_under_0.2": cnt["bc_under_0.2"] >= need and maj == "survives", "layers": rows, "runtime_s": r.get("runtime_s")}


# ---------------------------------------------------------------- Target 4
def eval_dewage(r):
    types, out = r["types"], {}
    for t, e in types.items():
        n = e["nulls"]
        a_pct = n["a_gaussian"]["outliers_percentile_of_real"]
        shr_out = {f: n[f]["outliers_shrinkage"] for f in ("a_gaussian", "b_rownorm", "c_permuted")}
        clauses = {"survives (a') Gaussian norm-matched: percentile above 99 (above all 20 draws)": a_pct > 99,
                   "(c') permuted entries match at least half of the outlier count": (shr_out["c_permuted"] or 0) >= 0.5,
                   "(b') row-norm-matched reproduces less than half of the energy share": (n["b_rownorm"]["energy_shrinkage"] or 0) < 0.5}
        pcts = [n[f]["outliers_percentile_of_real"] for f in ("a_gaussian", "b_rownorm", "c_permuted")]
        bc = max(shr_out["b_rownorm"] or 0, shr_out["c_permuted"] or 0)
        out[t] = {"real_mean_outliers": e["real_mean_outliers"], "real_mean_energy_share": e["real_mean_energy_share"],
                  "outliers_median": {f: n[f]["outliers_median"] for f in n}, "energy_median": {f: n[f]["energy_median"] for f in n},
                  "outliers_percentile_of_real": {f: n[f]["outliers_percentile_of_real"] for f in ("a_gaussian", "b_rownorm", "c_permuted")},
                  "outliers_shrinkage": shr_out, "energy_shrinkage": {f: n[f]["energy_shrinkage"] for f in ("a_gaussian", "b_rownorm", "c_permuted")},
                  "untrained_baseline_outliers": n["d_untrained"]["outliers_median"], "untrained_baseline_energy": n["d_untrained"]["energy_median"],
                  "b_rownorm_matches_half_of_count": (shr_out["b_rownorm"] or 0) >= 0.5,
                  "label": label_common(pcts, bc), "clauses": clauses, "all_clauses": all(clauses.values()), "max_shrinkage_bc": bc}
    return {"target": r["target"], "model": r["model"], "registered": r.get("registered"), "layers": r["layers"], "draws": r["draws"], "init_std": r["init_std"],
            "types": out, "expectation_holds": all(v["all_clauses"] for v in out.values()),
            "types_passing": sum(v["all_clauses"] for v in out.values()),
            "all_shrinkage_under_0.2": all(v["label"] == "survives" and v["max_shrinkage_bc"] < 0.2 for v in out.values()), "runtime_s": r.get("runtime_s")}


# ---------------------------------------------------------------- Target 5
def eval_retrieval(r):
    real, n = r["real"], r["nulls"]
    clauses, fam_rows = {}, {}
    for fam in ("a_random", "b_marginal", "c_colset"):
        e = n[fam]
        ok = e["top10_percentile_of_real"] > 99 and e["frac_percentile_of_real"] > 99 and (e["top10_shrinkage"] or 0) < 0.2
        clauses[f"survives {fam}: both percentiles above 99, top-10 shrinkage under 0.2"] = ok
        fam_rows[fam] = e
    d = n["d_untrained"]
    if "top10_values" in d:
        above = all(real["mean_top10_score"] > v for v in d["top10_values"]) and all(real["frac_heads_above_0.1"] > v for v in d["frac_values"])
        d_note = f"real above all {d['n']} initializations" if above else f"real not above every one of {d['n']} initializations"
    else:
        above = real["mean_top10_score"] > d["top10_median"] and real["frac_heads_above_0.1"] > d["frac_above_0.1_median"]
        d_note = "only medians recorded for the untrained model; compared with the medians"
    clauses[f"survives d_untrained: {d_note}, top-10 shrinkage under 0.2"] = above and (d["top10_shrinkage"] or 0) < 0.2
    return {"target": r["target"], "model": r["model"], "registered": r.get("registered"), "contexts": r["contexts"], "instances": r["instances"], "draws": r["draws"],
            "real": real, "nulls": n, "clauses": clauses, "expectation_holds": all(clauses.values()), "runtime_s": r.get("runtime_s")}


# ---------------------------------------------------------------- driver
def first(paths, fn):
    for p in paths:
        r = load(p)
        if r is not None:
            yield p, fn(r)


res = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "dry_run": DRY,
       "criteria": "audits/PREREGISTRATION4.md at commit ee4e267 (addenda in audits/PREREGISTRATION4_ADDENDA.md)",
       "targets": {"clark": dict(first(FILES["clark"], eval_clark)), "chai": dict(first(FILES["chai"], eval_chai)),
                   "dewage": dict(first(FILES["dewage"], eval_dewage)), "retrieval": dict(first(FILES["retrieval"], eval_retrieval))}}
T = res["targets"]
retr = list(T["retrieval"].values()); chai = list(T["chai"].values()); clark = list(T["clark"].values()); dew = list(T["dewage"].values())
E = {}
E["E3"] = None if not retr else all(v["expectation_holds"] for v in retr)
E["E2"] = None if not clark else clark[0]["expectation_holds"]
E["E1"] = None if not chai else any(v["c_matched_clause"] for v in chai)
E["E5"] = None if not dew else dew[0]["expectation_holds"]
E["E4"] = "dropped before the freeze (Target 2 not reproducible)"
audited = [v["all_shrinkage_under_0.2"] for v in clark + chai + dew]
res["expectations"] = E
res["falsification_all_survive_under_0.2"] = None if (not audited or len(clark + chai + dew) < 3) else all(audited)
res["verdicts_withheld"] = E["E3"] is False
json.dump(res, open(OUT_JSON, "w"), indent=1)

# ---------------------------------------------------------------- report
L = []
L.append("# Audit batteries: results against preregistration 4\n")
L.append(f"Generated {res['generated_utc']} by `audits/eval_batteries.py` from the battery result files; regenerate with "
         "`python audits/eval_batteries.py`. Criteria: `audits/PREREGISTRATION4.md`, frozen at commit `ee4e267` and not edited since; "
         "readings fixed before any result in `audits/PREREGISTRATION4_ADDENDA.md`. Each result file records the registration reference it "
         "ran under (`registered`). Shrinkage is the share of the real effect a null reproduces (null median over real).\n")
if res["verdicts_withheld"]:
    L.append("**E3 failed: the positive control did not survive every null. As the frozen file requires, every other verdict is withheld "
             "until the battery's thresholds are revised; the numbers below are reported without labels.**\n")
hold = res["verdicts_withheld"]

def verdict(b):
    return "pending" if b is None else ("withheld" if hold else ("holds" if b else "FAILS"))

L.append("## Registered expectations\n")
L.append("| Expectation | Verdict |\n|---|---|")
L.append(f"| E1: at least one hub, eigengap or special-head style claim is matched by surrogates (Target 3, column-set clause) | {verdict(E['E1'])} |")
L.append(f"| E2: a distribution-level similarity or redundancy claim survives with shrinkage between 20 and 70 percent (Target 1) | {verdict(E['E2'])} |")
L.append(f"| E3: the positive control survives every null with shrinkage under 20 percent (Target 5) | {'pending' if E['E3'] is None else ('holds' if E['E3'] else 'FAILS')} |")
L.append(f"| E4 | {E['E4']} |")
L.append(f"| E5: MP outliers survive the MP null and at least half are matched by random weights (Target 4, per-target clauses) | {verdict(E['E5'])} |")
fal = res["falsification_all_survive_under_0.2"]
L.append("\nFalsification clause (all audited claims survive with shrinkage under 20 percent, which would reject the container-geometry thesis for the audited set): "
         + ("pending" if fal is None else ("withheld" if hold else ("TRIGGERED: the thesis is rejected for the audited set" if fal else "not triggered"))) + "\n")

# Target 5 first: the control
L.append("## Target 5, Retrieval Heads (positive control, E3)\n")
if not retr:
    L.append("pending\n")
for p, v in T["retrieval"].items():
    L.append(f"`{p}`: {v['model']}, contexts {v['contexts']}, {v['instances']} instances, {v['draws']} draws; registered under {v['registered']}.\n")
    r0 = v["real"]
    L.append(f"Real: {r0['frac_heads_above_0.1']*100:.1f} percent of heads above 0.1, mean top-10 score {r0['mean_top10_score']:.3f}, max {r0['max_score']:.3f}.\n")
    L.append("| Null | heads above 0.1, median | percentile of real | top-10 median | percentile of real | top-10 shrinkage |\n|---|---|---|---|---|---|")
    for fam in ("a_random", "b_marginal", "c_colset"):
        e = v["nulls"][fam]
        L.append(f"| {fam} | {e['frac_above_0.1_median']*100:.2f} percent | {e['frac_percentile_of_real']:.1f} | {e['top10_median']:.4f} | {e['top10_percentile_of_real']:.1f} | {f3(e['top10_shrinkage'])} |")
    d = v["nulls"]["d_untrained"]
    L.append(f"| d_untrained (n={d['n']}) | {d['frac_above_0.1_median']*100:.2f} percent | see clause | {d['top10_median']:.4f} | see clause | {f3(d['top10_shrinkage'])} |")
    L.append("")
    for c, ok in v["clauses"].items():
        L.append(f"- {c}: {'holds' if ok else 'FAILS'}")
    L.append(f"\nE3 on this model: {'holds' if v['expectation_holds'] else 'FAILS'}.\n")

L.append("## Target 1, Clark et al. 2019 (E2)\n")
if not clark:
    L.append("pending\n")
for p, v in T["clark"].items():
    L.append(f"`{p}`: {v['model']}, {v['windows']} windows, {v['draws']} draws; registered under {v['registered']}.\n")
    L.append(f"Real contrast D = {v['real']['D']:.4f}; nearest neighbor in own layer {v['real']['nn_same_layer']*100:.1f} percent.\n")
    L.append("| Null | D median | percentile of real | shrinkage | nearest-neighbor median |\n|---|---|---|---|---|")
    for fam in ("a_random", "b_marginal", "c_colfix", "d_untrained"):
        L.append(f"| {fam} | {v['null_D_median'][fam]:.4f} | {v['percentile_of_real'][fam]:.1f} | {f3(v['shrinkage_all'][fam])} | {v['null_nn_median'][fam]*100:.1f} percent |")
    L.append("")
    for c, ok in v["clauses"].items():
        L.append(f"- {c}: {'withheld' if hold else ('holds' if ok else 'FAILS')}")
    L.append(f"\nCommon label: {'withheld' if hold else v['label']} (battery's own label {v['battery_label']}). Registered expectation: {verdict(v['expectation_holds'])}.\n")

L.append("## Target 3, CHAI 2024 (E1)\n")
if not chai:
    L.append("pending\n")
for p, v in T["chai"].items():
    c = v["counts"]; Ln = v["n_layers"]
    L.append(f"`{p}`: {v['model']}, {v['samples']} documents at T = {v['T']}, {v['draws']} draws; registered under {v['registered']}.\n")
    L.append(f"Mean real cross-head correlation over layers {v['mean_real_corr']:.3f}. Layers where: the sink-set surrogate (c) matches correlation and cluster share {c['c_matched_both']} of {Ln} "
             f"(correlation alone {c['c_matched_corr']}); correlation survives (a) {c['a_survive_corr']}; survives (d) {c['d_survive_corr']}; (b) reproduces more than half {c['b_over_half']}; "
             f"both surrogates reproduce under 0.2 {c['bc_under_0.2']}. Per-layer labels: {v['label_counts']}.\n")
    L.append("| layer | real corr | real largest 0.95-cluster | a median | b median | c median | d median | b reproduced | c reproduced | c percentile (corr) | label |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for row in v["layers"]:
        L.append(f"| {row['layer']} | {row['real_corr']:.3f} | {row['real_largest95']:.2f} | {row['a_median']:.3f} | {row['b_median']:.3f} | {row['c_median']:.3f} | {row['d_median']:.3f} | {f3(row['b_reproduced'])} | {f3(row['c_reproduced'])} | {row['c_pct_corr']:.1f} | {row['label']} |")
    L.append("")
    for cl, ok in v["clauses"].items():
        L.append(f"- {cl}: {'withheld' if hold else ('holds' if ok else 'FAILS')}")
    L.append(f"\nMajority label: {'withheld' if hold else v['label']}. Registered expectation on this model: {verdict(v['expectation_holds'])}.\n")

L.append("## Target 4, Dewage et al. 2026 (E5)\n")
if not dew:
    L.append("pending\n")
for p, v in T["dewage"].items():
    L.append(f"`{p}`: {v['model']}, {v['layers']} layers, {v['draws']} draws per random family, initializer std {v['init_std']}; registered under {v['registered']}.\n")
    L.append("| type | real outliers | real energy share | a' median | a' pct | b' median | b' count shrinkage | b' energy shrinkage | c' median | c' count shrinkage | d' baseline | label |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t, e in v["types"].items():
        om, os_, es = e["outliers_median"], e["outliers_shrinkage"], e["energy_shrinkage"]
        L.append(f"| {t} | {e['real_mean_outliers']:.1f} | {e['real_mean_energy_share']:.3f} | {om['a_gaussian']:.1f} | {e['outliers_percentile_of_real']['a_gaussian']:.0f} | {om['b_rownorm']:.1f} | {f3(os_['b_rownorm'])} | {f3(es['b_rownorm'])} | {om['c_permuted']:.1f} | {f3(os_['c_permuted'])} | {e['untrained_baseline_outliers']:.1f} | {'withheld' if hold else e['label']} |")
    L.append("")
    for t, e in v["types"].items():
        for cl, ok in e["clauses"].items():
            L.append(f"- {t}: {cl}: {'withheld' if hold else ('holds' if ok else 'FAILS')}")
    L.append(f"\nTypes passing all three clauses: {v['types_passing']} of 4. Registered expectation (all types): {verdict(v['expectation_holds'])}. "
             "The expectation list's E5 names norm-matched random weights for the half-matched clause; the per-target section names the permuted null (c'); "
             "the per-target clause is evaluated and the (b') count shrinkage is in the table.\n")

L.append("## Target 2, Kovaleva et al. 2019\n\nNot reproducible in its stated form (classifier and annotations unreleased); no battery, E4 dropped before the freeze.\n")
open(OUT_MD, "w").write("\n".join(L) + "\n")
print(json.dumps({"expectations": E, "falsification": res["falsification_all_survive_under_0.2"], "withheld": res["verdicts_withheld"],
                  "files": {k: list(v) for k, v in T.items()}}, indent=1))
print("EVAL_DONE", OUT_JSON, OUT_MD)
