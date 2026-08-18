# Resume of heldout_round.py after the H6 dataset-id fix
# (openai_humaneval -> openai/openai_humaneval; same data). Re-runs only
# the H6 block per PREREGISTRATION2.md and completes heldout_round.json.
# layer_stats/run_model duplicated verbatim from heldout_round.py
# (families=False path) because that script executes at import.
import gc
import json
import sys
import numpy as np

sys.path.insert(0, "alignment_study")
from common import (load_model, wikitext_windows, humaneval_windows,
                    attn_layer, generators, gnorms, coup, r1, rho,
                    shared_mode, sink_column, col_mass, sink_generator,
                    sur_plain, zstats, zmed_of)

STRIDE = 40

def layer_stats(A, rg, kplain=8):
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
    return d

def run_model(name, windows_fn, nwin, seq):
    tok, model = load_model(name)
    L = model.config.num_hidden_layers
    wins = windows_fn(nwin, STRIDE, seq, tok)
    rg = np.random.default_rng(0)
    acc = {l: [] for l in range(L)}
    for w, ids in enumerate(wins):
        out = model(**ids)
        for l in range(L):
            acc[l].append(layer_stats(attn_layer(out, l), rg))
        print(name, "WIN", w, flush=True)
    rows = []
    for l in range(L):
        ds = acc[l]
        agg = {"layer": l}
        for k in ds[0]:
            vals = [d[k] for d in ds]
            agg[k] = float(np.mean(vals)) if k in ("smass", "sharedE") else float(np.median(vals))
        rows.append(agg)
    del model
    gc.collect()
    return rows

def hs(rows):
    return [r_ for r_ in rows if r_["smass"] > 0.4]

res = json.load(open("alignment_study/heldout_round.json"))
qwiki = run_model("Qwen/Qwen2.5-0.5B", wikitext_windows, 12, 64)
qcode = run_model("Qwen/Qwen2.5-0.5B", humaneval_windows, 12, 64)
set_w = {r_["layer"] for r_ in hs(qwiki) if r_["r1_real"] < 0.3}
set_c = {r_["layer"] for r_ in hs(qcode) if r_["r1_real"] < 0.3}
jac = len(set_w & set_c) / max(1, len(set_w | set_c))
res["H"]["H6"] = {"wikitext_set": sorted(set_w), "code_set": sorted(set_c),
                  "jaccard": jac, "pass": bool(jac > 0.4)}
res["qwiki12"] = qwiki
res["qcode12"] = qcode
json.dump(res, open("alignment_study/heldout_round.json", "w"), indent=1)
print("H6 DONE", json.dumps(res["H"]["H6"]))
