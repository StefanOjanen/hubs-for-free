# TIER 1 DEV CALIBRATION (Qwen2.5-0.5B, the development model; run before
# PREREGISTRATION2.md is frozen and before any held-out model is touched).
# Addresses review findings:
# 1. r1 ill-conditioning: adds CV(gn) diagnostic and a null-referenced
#    primary statistic zmed (median per-pair z of C against 16 plain draws).
# 2. Empirical sink definition: sinkfix now preserves the empirically
#    identified sink column per window, not column 0 by fiat.
# 3. altsink-v2: fully distinct per-head target columns (t_h = h+1),
#    replacing the flawed h mod 4 control.
import json
import sys
import numpy as np

sys.path.insert(0, "alignment_study")
from common import (load_model, wikitext_windows, attn_layer, generators,
                    gnorms, coup, r1, rho, shared_mode, sink_column, col_mass,
                    sur_plain, sur_sinkfix, sur_altsink2, check_invariants,
                    zstats, zmed_of)

MODEL, NWIN, STRIDE, SEQ = "Qwen/Qwen2.5-0.5B", 24, 40, 64
KPLAIN, KFAM = 16, 8
tok, model = load_model(MODEL)
L = model.config.num_hidden_layers
wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)
print("windows", len(wins), "layers", L, flush=True)

rg = np.random.default_rng(0)
per = {l: {"zmed_real": [], "zmed_sinkfix": [], "zmed_alt2": [],
           "r1_real": [], "r1_plain": [], "r1_sinkfix": [], "r1_alt2": [],
           "rho_real": [], "rho_alt2": [], "sharedE": [], "cv_gn": [],
           "sinkcol": [], "sinkmass": []} for l in range(L)}
worst_dev = 0.0
for w, ids in enumerate(wins):
    out = model(**ids)
    for l in range(L):
        A = attn_layer(out, l)
        n, T, _ = A.shape
        iu = np.triu_indices(n, 1)
        G = generators(A)
        gn = gnorms(G)
        C = coup(G)
        s = sink_column(A)
        Smat, a, e, frac = shared_mode(G)
        plains = []
        for k in range(KPLAIN):
            Sp = sur_plain(A, rg)
            if k == 0 and w == 0:
                worst_dev = max(worst_dev, max(check_invariants(A, Sp).values()))
            Gp = generators(Sp)
            plains.append(coup(Gp))
            per[l]["r1_plain"].append(r1(plains[-1], gnorms(Gp), iu))
        zr, mu, sd = zstats(C, plains, iu)
        per[l]["zmed_real"].append(zr)
        for k in range(KFAM):
            Sf = sur_sinkfix(A, rg, s=s)
            if k == 0 and w == 0:
                worst_dev = max(worst_dev, max(check_invariants(A, Sf, preserve_cols=(s,)).values()))
            Gf = generators(Sf)
            Cf = coup(Gf)
            per[l]["zmed_sinkfix"].append(zmed_of(Cf, mu, sd, iu))
            per[l]["r1_sinkfix"].append(r1(Cf, gnorms(Gf), iu))
            Sa = sur_altsink2(A, rg)
            if k == 0 and w == 0:
                worst_dev = max(worst_dev, max(check_invariants(A, Sa).values()))
            Ga = generators(Sa)
            Ca = coup(Ga)
            per[l]["zmed_alt2"].append(zmed_of(Ca, mu, sd, iu))
            per[l]["r1_alt2"].append(r1(Ca, gnorms(Ga), iu))
            per[l]["rho_alt2"].append(rho(Ga, gnorms(Ga)))
        per[l]["r1_real"].append(r1(C, gn, iu))
        per[l]["rho_real"].append(rho(G, gn))
        per[l]["sharedE"].append(frac)
        per[l]["cv_gn"].append(float(gn.std() / gn.mean()))
        per[l]["sinkcol"].append(s)
        per[l]["sinkmass"].append(col_mass(A, s))
    if w % 4 == 0:
        print("WIN", w, flush=True)

rows = []
for l in range(L):
    d = per[l]
    rows.append({
        "layer": l,
        "sink_col_modal": int(np.bincount(d["sinkcol"]).argmax()),
        "sink_mass": float(np.mean(d["sinkmass"])),
        "zmed_real": float(np.median(d["zmed_real"])),
        "zmed_sinkfix": float(np.median(d["zmed_sinkfix"])),
        "zmed_alt2": float(np.median(d["zmed_alt2"])),
        "r1_real": float(np.median(d["r1_real"])),
        "r1_plain": float(np.median(d["r1_plain"])),
        "r1_sinkfix": float(np.median(d["r1_sinkfix"])),
        "r1_alt2": float(np.median(d["r1_alt2"])),
        "rho_real": float(np.median(d["rho_real"])),
        "rho_alt2": float(np.median(d["rho_alt2"])),
        "sharedE": float(np.mean(d["sharedE"])),
        "cv_gn": float(np.median(d["cv_gn"])),
    })
res = {"model": MODEL, "n_windows": len(wins), "stride": STRIDE,
       "seq": SEQ, "k_plain": KPLAIN, "k_fam": KFAM,
       "worst_invariant_dev_sampled": worst_dev, "per_layer": rows}
json.dump(res, open("alignment_study/tier1_robust.json", "w"), indent=1)
print("worst invariant dev (sampled):", worst_dev)
print("layer | scol | smass | zmed_real | z_sinkfix | z_alt2 | r1_real | r1_sf | r1_a2 | rho/rhoA2 | sharedE | cv_gn")
for r_ in rows:
    print(f"{r_['layer']:>3} | {r_['sink_col_modal']:>3} | {r_['sink_mass']:.2f} | "
          f"{r_['zmed_real']:+7.2f} | {r_['zmed_sinkfix']:+7.2f} | {r_['zmed_alt2']:+6.2f} | "
          f"{r_['r1_real']:+.2f} | {r_['r1_sinkfix']:+.2f} | {r_['r1_alt2']:+.2f} | "
          f"{r_['rho_real']:.2f}/{r_['rho_alt2']:.2f} | {r_['sharedE']:.2f} | {r_['cv_gn']:.2f}")
