# Evaluation of PREREGISTRATION16, committed before any confirmatory result
# exists. Within-layer Spearman of each head statistic with the damage of
# removing the head; partial Spearman of Voita et al.'s confidence given sink
# mass (ranks residualized on the sink-mass rank, then correlated); two-stage
# cluster bootstrap (models, then layers), 2,000 draws, seed 0.
#   python alignment_study/eval_voita.py [directory]
import glob
import json
import sys
import numpy as np
from scipy.stats import spearmanr, rankdata

D = (sys.argv[1] if len(sys.argv) > 1 else "alignment_study/voita").rstrip("/")
B, SEED = 2000, 0
TH = {"V1_rho": 0.2, "V2_delta": 0.3, "V4_rho": 0.2, "V5_delta": 0.1, "V6_delta": 0.2}      # V6 added after the development run
DSTAT = ("conf_voita", "conf_all", "conf_first", "conf_renorm", "sink_mass", "out_norm")
TSTAT = ("conf_voita", "conf_all", "eos_mass", "neg_entropy", "out_norm")


def cluster_ci(groups):
    groups = [np.asarray(g, float) for g in groups if len(g)]
    rg = np.random.default_rng(SEED); vals = []
    for _ in range(B):
        pick = rg.integers(0, len(groups), len(groups))
        vals.append(np.mean(np.concatenate([groups[k][rg.integers(0, len(groups[k]), len(groups[k]))] for k in pick])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def pooled(groups):
    allv = np.concatenate([np.asarray(x, float) for x in groups if len(x)])
    return {"value": float(np.mean(allv)), "ci95": cluster_ci(groups), "n_layers": int(allv.size), "n_models": sum(1 for x in groups if len(x))}


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def decoder_layers(c):
    """Per layer: head statistics (with the stored output norm) and removal KL per corpus."""
    ref = json.load(open(c["outcome_file"]))
    abl = ref["ablation"]
    corpora = {"wikitext": abl["wikitext"], "c4": abl["c4"]} if "wikitext" in abl else {"wikitext": abl}
    out = {}
    for cname, A in corpora.items():
        rows = []
        for l, heads in A.items():
            s = ref["stats"][l]; s = s["heads"] if isinstance(s, dict) else s
            st = [dict(c["stats"][l][h], out_norm=s[h]["out_norm"]) for h in range(c["heads"])]
            rows.append((st, [heads[str(h)]["kl"] for h in range(c["heads"])]))
        out[cname] = rows
    return out


def correlations(rows, keys, partial_of=None):
    r = {k: [] for k in keys}; part = []
    for st, kl in rows:
        for k in keys:
            r[k].append(float(spearmanr([h[k] for h in st], kl)[0]))
        if partial_of:
            x, z = partial_of
            part.append(partial_spearman([h[x] for h in st], kl, [h[z] for h in st]))
    return r, part


models = [json.load(open(p)) for p in sorted(glob.glob(f"{D}/*.json"))]
res = {"thresholds": TH, "per_model": {}, "V": {}, "secondary": {}}
G = {}
for c in models:
    name = c["model"]
    if c["part"] == "decoder":
        pm = {"check_max_abs_diff": c["check_max_abs_diff"]}
        for cname, rows in decoder_layers(c).items():
            r, part = correlations(rows, DSTAT, ("conf_voita", "sink_mass"))
            for k in DSTAT:
                G.setdefault(f"D_{cname}_{k}", []).append(r[k])
            G.setdefault(f"D_{cname}_delta", []).append([a - b for a, b in zip(r["conf_voita"], r["conf_all"])])
            G.setdefault(f"D_{cname}_partial", []).append(part)
            G.setdefault(f"D_{cname}_on_minus_voita", []).append([a - b for a, b in zip(r["out_norm"], r["conf_voita"])])
            pm[cname] = {"rho_mean": {k: float(np.mean(v)) for k, v in r.items()}, "partial_mean": float(np.mean(part)), "n_layers": len(rows)}
    else:
        rows = [([c["stats"][l][h] for h in range(c["heads"])], [c["ablation"][l][str(h)]["kl"] for h in range(c["heads"])]) for l in c["ablation"]]
        r, part = correlations(rows, TSTAT, ("conf_voita", "eos_mass"))
        for k in TSTAT:
            G.setdefault(f"T_{k}", []).append(r[k])
        G.setdefault("T_delta", []).append([a - b for a, b in zip(r["conf_voita"], r["conf_all"])])
        G.setdefault("T_partial", []).append(part)
        G.setdefault("T_on_minus_voita", []).append([a - b for a, b in zip(r["out_norm"], r["conf_voita"])])
        pm = {"restore_check_kl": c["restore_check_kl"], "rho_by_layer": {k: [round(v, 3) for v in r[k]] for k in TSTAT},
              "rho_mean": {k: float(np.mean(v)) for k, v in r.items()}, "partial_mean": float(np.mean(part)),
              "eos_mass_mean": float(np.mean([h["eos_mass"] for l in c["stats"] for h in c["stats"][l]])),
              "rho_pooled_over_heads": {k: float(spearmanr([h[k] for l in c["stats"] for h in c["stats"][l]],
                                                          [c["ablation"][l][str(h)]["kl"] for l in c["stats"] for h in range(c["heads"])])[0]) for k in TSTAT}}
    res["per_model"][name] = pm
V = res["V"]
if "D_wikitext_conf_voita" in G:
    v = pooled(G["D_wikitext_conf_voita"]); v["pass_"] = v["value"] >= TH["V1_rho"] and v["ci95"][0] > 0; V["V1"] = v
    v = pooled(G["D_wikitext_delta"]); v["pass_"] = v["value"] >= TH["V2_delta"] and v["ci95"][0] > 0; V["V2"] = v
    v = pooled(G["D_wikitext_partial"]); v["pass_"] = v["ci95"][0] > 0; V["V3"] = v
    v = pooled(G["D_wikitext_on_minus_voita"]); v["pass_"] = v["value"] >= TH["V6_delta"] and v["ci95"][0] > 0; V["V6"] = v
if "T_conf_voita" in G:
    v = pooled(G["T_conf_voita"]); v["pass_"] = v["value"] >= TH["V4_rho"] and v["ci95"][0] > 0; V["V4"] = v
    v = pooled(G["T_delta"]); v["pass_"] = v["value"] >= TH["V5_delta"] and v["ci95"][0] > 0; V["V5"] = v
V["primary_passes"] = int(sum(bool(x.get("pass_")) for x in V.values() if isinstance(x, dict)))
res["secondary"] = {k: pooled(g) for k, g in G.items()}
json.dump(res, open(D + "_results.json", "w"), indent=1)
for k, v in V.items():
    print(k, json.dumps(v)[:200])
for k, v in res["secondary"].items():
    print(f"  {k:28s} {v['value']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] layers {v['n_layers']}")
for m, pm in res["per_model"].items():
    keys = [k for k in pm if k in ("wikitext", "c4")]
    if keys:
        print(f"  {m.split('/')[-1]:16s} " + " | ".join(f"{k}: " + " ".join(f"{s} {x:+.2f}" for s, x in pm[k]["rho_mean"].items()) + f" partial {pm[k]['partial_mean']:+.2f}" for k in keys))
    else:
        print(f"  {m.split('/')[-1]:16s} eos {pm['eos_mass_mean']:.2f} " + " ".join(f"{s} {x:+.2f}" for s, x in pm["rho_mean"].items()) + f" partial {pm['partial_mean']:+.2f}")
