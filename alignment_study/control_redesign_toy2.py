# T3.2 development, second sweep. Same three controls as control_redesign_toy.py
# (v2 = registered altsink-v2; wrapped = every row targeted; shift = matched
# geometry by head-specific cyclic offsets), now over n in (12, 16, 32),
# T in (64, 128, 256), boosts (2, 4, 6, 8), with per-trial values saved so
# pass fractions can be computed under two criteria:
#   full   : r1_ctrl > 0.8 AND rho_ctrl < 0.5 rho_real AND z_ctrl > z_real
#            (the registered H3/S3 clause);
#   robust : rho_ctrl < 0.5 rho_real AND z_ctrl > z_real
#            (the two clauses that held everywhere in rounds 2 and 3).
# The misaligned toy world is recorded as the reference for what a genuinely
# misaligned ensemble does at the same (n, T, b). Output: control_redesign_toy2.json.
import json
import sys
import time
import numpy as np

sys.path.insert(0, "alignment_study")
from common import generators, gnorms, coup, r1, rho, shared_mode, sur_plain, zstats, zmed_of, check_invariants
from control_redesign_toy import toy_heads, ctrl_v2, ctrl_wrapped, ctrl_shift

NS, TS, BOOSTS, TRIALS, KPLAIN = (12, 16, 32), (64, 128, 256), (2.0, 4.0, 6.0, 8.0), 8, 8
CTRLS = {"v2": ctrl_v2, "wrapped": ctrl_wrapped, "shift": ctrl_shift}
rg = np.random.default_rng(1)
out = {"config": {"n": NS, "T": TS, "boosts": BOOSTS, "trials": TRIALS, "k_plain": KPLAIN}, "rows": []}
t0 = time.time()
for n in NS:
    iu = np.triu_indices(n, 1)
    for T in TS:
        for b in BOOSTS:
            tr = {k: [] for k in ["real_r1", "real_rho", "real_z", "real_sharedE", "mis_r1", "mis_rho", "mis_z"]
                  + [f"{c}_{m}" for c in CTRLS for m in ("r1", "rho", "z", "sharedE")]}
            inv = 0.0
            for _ in range(TRIALS):
                A = toy_heads(n, T, b, True, rg)
                G = generators(A); gn = gnorms(G); C = coup(G)
                plains = [coup(generators(sur_plain(A, rg))) for _ in range(KPLAIN)]
                z, mu, sd = zstats(C, plains, iu)
                tr["real_r1"].append(r1(C, gn, iu)); tr["real_rho"].append(rho(G, gn)); tr["real_z"].append(z); tr["real_sharedE"].append(shared_mode(G)[3])
                M = toy_heads(n, T, b, False, rg); Gm = generators(M); gnm = gnorms(Gm); Cm = coup(Gm)
                tr["mis_r1"].append(r1(Cm, gnm, iu)); tr["mis_rho"].append(rho(Gm, gnm)); tr["mis_z"].append(zmed_of(Cm, mu, sd, iu))
                for c, f in CTRLS.items():
                    S = f(A, rg); Gs = generators(S); gns = gnorms(Gs); Cs = coup(Gs)
                    tr[f"{c}_r1"].append(r1(Cs, gns, iu)); tr[f"{c}_rho"].append(rho(Gs, gns)); tr[f"{c}_z"].append(zmed_of(Cs, mu, sd, iu)); tr[f"{c}_sharedE"].append(shared_mode(Gs)[3])
                    inv = max(inv, max(check_invariants(A, S).values()))
            row = {"n": n, "T": T, "boost": b, "max_invariant_dev": inv, "trials": {k: [round(float(x), 4) for x in v] for k, v in tr.items()}}
            for k, v in tr.items():
                row[k] = round(float(np.median(v)), 4)
            for c in CTRLS:
                full = [(tr[f"{c}_r1"][t] > 0.8) and (tr[f"{c}_rho"][t] < 0.5 * tr["real_rho"][t]) and (tr[f"{c}_z"][t] > tr["real_z"][t]) for t in range(TRIALS)]
                robust = [(tr[f"{c}_rho"][t] < 0.5 * tr["real_rho"][t]) and (tr[f"{c}_z"][t] > tr["real_z"][t]) for t in range(TRIALS)]
                row[f"{c}_pass_full"] = round(float(np.mean(full)), 3); row[f"{c}_pass_robust"] = round(float(np.mean(robust)), 3)
            out["rows"].append(row)
            print(f"n={n:2d} T={T:3d} b={b:3.1f} | real r1 {row['real_r1']:6.3f} sE {row['real_sharedE']:.3f} | mis r1 {row['mis_r1']:6.3f} | "
                  + " ".join(f"{c}: r1 {row[c+'_r1']:6.3f} full {row[c+'_pass_full']:.2f} robust {row[c+'_pass_robust']:.2f}" for c in CTRLS)
                  + f" | inv {inv:.1e} | {time.time()-t0:.0f}s", flush=True)
            json.dump(out, open("alignment_study/control_redesign_toy2.json", "w"), indent=1)
print("CONTROL_TOY2_DONE")
