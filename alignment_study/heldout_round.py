# HELD-OUT CONFIRMATORY ROUND. Protocol and thresholds frozen in
# PREREGISTRATION2.md (commit 0f1ac3f) before any model below was
# downloaded. Evaluates H0-H6 and writes heldout_round.json.
import gc
import json
import sys
import numpy as np
import torch

sys.path.insert(0, "alignment_study")
from common import (load_model, wikitext_windows, humaneval_windows,
                    attn_layer, generators, gnorms, coup, r1, rho,
                    shared_mode, sink_column, col_mass, sink_generator,
                    sur_plain, sur_sinkfix, sur_altsink2, zstats, zmed_of)

HELDOUT = ["gpt2-medium", "EleutherAI/pythia-410m",
           "TinyLlama/TinyLlama_v1.1", "Qwen/Qwen2.5-1.5B",
           "Qwen/Qwen2.5-1.5B-Instruct"]
NWIN, STRIDE, SEQ = 12, 40, 64
KPLAIN, KFAM = 12, 8


def layer_stats(A, rg, kplain=KPLAIN, kfam=KFAM, families=True):
    n, T, _ = A.shape
    iu = np.triu_indices(n, 1)
    G = generators(A)
    gn = gnorms(G)
    C = coup(G)
    s = sink_column(A)
    Smat, a, e, frac = shared_mode(G)
    Gs_ideal = sink_generator(T, s)
    d = {"s": s, "smass": col_mass(A, s),
         "r1_real": r1(C, gn, iu), "rho_real": rho(G, gn),
         "sharedE": frac, "cv_gn": float(gn.std() / gn.mean()),
         "cosS": float(abs(np.sum(Smat / np.sqrt((Smat ** 2).sum()) * Gs_ideal)))}
    plains, r1p, rhop = [], [], []
    for _ in range(kplain):
        Gp = generators(sur_plain(A, rg))
        Cp = coup(Gp)
        plains.append(Cp)
        r1p.append(r1(Cp, gnorms(Gp), iu))
        rhop.append(rho(Gp, gnorms(Gp)))
    zr, mu, sd = zstats(C, plains, iu)
    d["zmed_real"] = zr
    d["r1_plain"] = float(np.median(r1p))
    d["rho_plain"] = float(np.median(rhop))
    d["zmed_plain"] = float(np.median([zmed_of(Cp, mu, sd, iu) for Cp in plains]))
    if families:
        zsf, r1sf, za2, r1a2, rhoa2 = [], [], [], [], []
        for _ in range(kfam):
            Gf = generators(sur_sinkfix(A, rg, s=s))
            Cf = coup(Gf)
            zsf.append(zmed_of(Cf, mu, sd, iu))
            r1sf.append(r1(Cf, gnorms(Gf), iu))
            Ga = generators(sur_altsink2(A, rg))
            Ca = coup(Ga)
            za2.append(zmed_of(Ca, mu, sd, iu))
            r1a2.append(r1(Ca, gnorms(Ga), iu))
            rhoa2.append(rho(Ga, gnorms(Ga)))
        d.update(zmed_sinkfix=float(np.median(zsf)), r1_sinkfix=float(np.median(r1sf)),
                 zmed_alt2=float(np.median(za2)), r1_alt2=float(np.median(r1a2)),
                 rho_alt2=float(np.median(rhoa2)))
    return d


def run_model(name, windows_fn, nwin=NWIN, seq=SEQ, families=True,
              kplain=KPLAIN):
    tok, model = load_model(name)
    L = model.config.num_hidden_layers
    wins = windows_fn(nwin, STRIDE, seq, tok)
    rg = np.random.default_rng(0)
    acc = {l: [] for l in range(L)}
    for w, ids in enumerate(wins):
        out = model(**ids)
        for l in range(L):
            acc[l].append(layer_stats(attn_layer(out, l), rg,
                                      kplain=kplain, families=families))
        print(name, "WIN", w, flush=True)
    rows = []
    for l in range(L):
        ds = acc[l]
        agg = {"layer": l}
        for k in ds[0]:
            vals = [d[k] for d in ds]
            agg[k] = float(np.mean(vals)) if k in ("smass", "sharedE") else float(np.median(vals))
        agg["sink_col_modal"] = int(np.bincount([d["s"] for d in ds]).argmax())
        rows.append(agg)
    del model
    gc.collect()
    return rows


res = {"models": {}, "H": {}}

for name in HELDOUT:
    try:
        res["models"][name] = run_model(name, wikitext_windows)
    except Exception as ex:
        res["models"][name] = {"error": str(ex)[:400]}
        print("MODEL FAILED", name, ex, flush=True)

# ---- H0-H4 evaluation ----
def hs(rows):
    return [r_ for r_ in rows if r_["smass"] > 0.4]

pooled_all, pooled_hs = [], []
h_eval = {"per_model": {}}
for name, rows in res["models"].items():
    if isinstance(rows, dict):
        h_eval["per_model"][name] = {"testable": False, "error": True}
        continue
    pooled_all += rows
    H = hs(rows)
    pooled_hs += H
    testable = len(H) >= 3
    def frac(cond):
        return float(np.mean([1.0 if cond(r_) else 0.0 for r_ in H])) if H else 0.0
    f1 = frac(lambda r_: r_["rho_real"] >= 2 * r_["rho_plain"] and r_["cosS"] > 0.7)
    def ratios(r_):
        Rr = abs(r_["r1_sinkfix"] - r_["r1_real"]) / max(0.05, abs(r_["r1_plain"] - r_["r1_real"]))
        Rz = abs(r_["zmed_sinkfix"] - r_["zmed_real"]) / max(1.0, abs(r_["zmed_real"] - r_["zmed_plain"]))
        return Rr, Rz
    f2 = frac(lambda r_: max(ratios(r_)) < 0.4)
    f3 = frac(lambda r_: r_["r1_alt2"] > 0.8 and r_["rho_alt2"] < 0.5 * r_["rho_real"]
              and r_["zmed_alt2"] > r_["zmed_real"])
    h_eval["per_model"][name] = {
        "testable": testable, "n_high_sink": len(H),
        "H1_frac": f1, "H1_pass": bool(testable and f1 >= 2 / 3),
        "H2_frac": f2, "H2_pass": bool(testable and f2 >= 2 / 3),
        "H3_frac": f3, "H3_pass": bool(testable and f3 >= 2 / 3),
    }

test_models = [m for m, v in h_eval["per_model"].items() if v.get("testable")]
def overall(key):
    passes = sum(1 for m in test_models if h_eval["per_model"][m][key])
    return {"passes": passes, "testable": len(test_models),
            "pass": bool(len(test_models) and passes >= (2 / 3) * len(test_models))}

h0_frac = float(np.mean([1.0 if r_["r1_plain"] > 0.9 else 0.0 for r_ in pooled_all]))
res["H"]["H0"] = {"frac_plain_r1_gt_0.9": h0_frac, "pass": bool(h0_frac >= 0.75)}
res["H"]["H1"] = overall("H1_pass")
res["H"]["H2"] = overall("H2_pass")
res["H"]["H3"] = overall("H3_pass")

def spearman(x, y):
    def rank(v):
        v = np.array(v); idx = np.argsort(v); rk = np.empty(len(v)); rk[idx] = np.arange(len(v))
        return rk
    return float(np.corrcoef(rank(x), rank(y))[0, 1])

if len(pooled_hs) >= 8:
    sp = spearman([r_["sharedE"] for r_ in pooled_hs], [r_["r1_real"] for r_ in pooled_hs])
    edge = [r_ for r_ in pooled_hs if r_["sharedE"] >= 0.90]
    h4b = all(r_["r1_real"] < 0.2 for r_ in edge)
    res["H"]["H4"] = {"spearman": sp, "pass": bool(sp <= -0.5),
                      "H4b_n_edge_layers": len(edge), "H4b_pass": bool(h4b)}
else:
    res["H"]["H4"] = {"untestable": True}
res["H"]["per_model"] = h_eval["per_model"]
json.dump(res, open("alignment_study/heldout_round.json", "w"), indent=1)
print("H0-H4 DONE", json.dumps(res["H"], default=str)[:600], flush=True)

# ---- H5: context transfer on the dev model ----
def r1_only(name, windows_fn, nwin, seq):
    rows = run_model(name, windows_fn, nwin=nwin, seq=seq, families=False,
                     kplain=8)
    return rows

q64 = r1_only("Qwen/Qwen2.5-0.5B", wikitext_windows, 6, 64)
q256 = r1_only("Qwen/Qwen2.5-0.5B", wikitext_windows, 6, 256)
c64 = sum(1 for r_ in hs(q64) if r_["r1_real"] < 0.3)
c256 = sum(1 for r_ in hs(q256) if r_["r1_real"] < 0.3)
res["H"]["H5"] = {"breakdown_layers_T64": c64, "breakdown_layers_T256": c256,
                  "pass": bool(c256 >= c64)}
res["q64"] = q64
res["q256"] = q256
json.dump(res, open("alignment_study/heldout_round.json", "w"), indent=1)
print("H5 DONE", json.dumps(res["H"]["H5"]), flush=True)

# ---- H6: corpus transfer on the dev model ----
qwiki = r1_only("Qwen/Qwen2.5-0.5B", wikitext_windows, 12, 64)
qcode = r1_only("Qwen/Qwen2.5-0.5B", humaneval_windows, 12, 64)
set_w = {r_["layer"] for r_ in hs(qwiki) if r_["r1_real"] < 0.3}
set_c = {r_["layer"] for r_ in hs(qcode) if r_["r1_real"] < 0.3}
jac = len(set_w & set_c) / max(1, len(set_w | set_c))
res["H"]["H6"] = {"wikitext_set": sorted(set_w), "code_set": sorted(set_c),
                  "jaccard": jac, "pass": bool(jac > 0.4)}
res["qwiki12"] = qwiki
res["qcode12"] = qcode
json.dump(res, open("alignment_study/heldout_round.json", "w"), indent=1)
print("H6 DONE", json.dumps(res["H"]["H6"]), flush=True)
print("ALL DONE")
