# T3.2 development: dissociation-control redesign, validated on the toy
# ensemble BEFORE any model run (plan WS3, gate G2). The registered control
# altsink-v2 (t_h = h + 1, rows i <= t_h plain-permuted) lost power at 32
# heads in round 3. Candidates:
#   v2      : the registered control, for reference.
#   wrapped : target t_h from a random permutation of 1..n per draw; rows
#             with t_h >= i use the wrapped target 1 + (t_h - 1) mod (i - 1),
#             so every row of every head gets the same treatment (row max
#             swapped into the target, other causal entries permuted).
#   shift   : matched geometry. Each head's causal row is cyclically shifted
#             by a head-specific offset d_h (random permutation of 1..n per
#             draw), so the head's dominant column moves to d_h mod i while
#             the row's internal pattern is kept. No random permutation.
# Toy worlds as in tier2_toy.py (causal softmax, lognormal temperatures,
# logit boost b on one column): aligned (all heads column 0) is the input
# to the controls; misaligned (head h -> column h + 1) is the reference for
# what a genuinely misaligned ensemble looks like at the same (n, T, b).
# Every surrogate keeps each row's sorted values, hence every generator
# norm; checked at 1e-9. Output: control_redesign_toy.json.
import json
import sys
import time
import numpy as np

sys.path.insert(0, "alignment_study")
from common import generators, gnorms, coup, r1, rho, shared_mode, sur_plain, zstats, zmed_of, check_invariants

NS, TS, BOOSTS, TRIALS, KPLAIN = (8, 14, 32), (64, 256), (0.0, 2.0, 4.0, 8.0), 8, 8
rg = np.random.default_rng(0)


def toy_heads(n, T, b, shared, rgl):
    temps = np.exp(rgl.normal(0, 0.5, n))
    A = np.zeros((n, T, T))
    mask = np.tril(np.ones((T, T), bool))
    for h in range(n):
        c = 0 if shared else h + 1
        lo = rgl.normal(0, 1, (T, T)) / temps[h]
        lo[:, c] += b
        lo = np.where(mask, lo, -1e9)
        ex = np.exp(lo - lo.max(-1, keepdims=True))
        A[h] = ex / ex.sum(-1, keepdims=True)
    return A


def ctrl_v2(A, rgl):
    n, T, _ = A.shape
    S = A.copy()
    for h in range(n):
        t = h + 1
        for i in range(1, T):
            row = S[h, i, :i].copy()
            if t < i:
                j = int(row.argmax()); row[j], row[t] = row[t], row[j]
                rest = np.array([c for c in range(i) if c != t]); row[rest] = row[rest][rgl.permutation(len(rest))]
            else:
                row = row[rgl.permutation(i)]
            S[h, i, :i] = row
    return S


def ctrl_wrapped(A, rgl):
    n, T, _ = A.shape
    S = A.copy()
    targets = rgl.permutation(n) + 1
    for h in range(n):
        t = int(targets[h])
        for i in range(2, T):
            tt = t if t < i else 1 + (t - 1) % (i - 1)
            row = S[h, i, :i].copy()
            j = int(row.argmax()); row[j], row[tt] = row[tt], row[j]
            rest = np.array([c for c in range(i) if c != tt]); row[rest] = row[rest][rgl.permutation(len(rest))]
            S[h, i, :i] = row
    return S


def ctrl_shift(A, rgl):
    n, T, _ = A.shape
    S = A.copy()
    offs = rgl.permutation(n) + 1
    for h in range(n):
        d = int(offs[h])
        for i in range(2, T):
            S[h, i, :i] = np.roll(A[h, i, :i], d % i)
    return S


CTRLS = {"v2": ctrl_v2, "wrapped": ctrl_wrapped, "shift": ctrl_shift}


if __name__ == "__main__":
    out = {"config": {"n": NS, "T": TS, "boosts": BOOSTS, "trials": TRIALS, "k_plain": KPLAIN}, "rows": []}
    t0 = time.time()
    for n in NS:
        iu = np.triu_indices(n, 1)
        for T in TS:
            for b in BOOSTS:
                acc = {k: [] for k in ["real_r1", "real_rho", "real_z", "real_sharedE", "mis_r1", "mis_rho", "mis_z", "mis_sharedE"]
                       + [f"{c}_{m}" for c in CTRLS for m in ("r1", "rho", "z", "sharedE", "inv")]}
                for _ in range(TRIALS):
                    A = toy_heads(n, T, b, True, rg)
                    G = generators(A); gn = gnorms(G); C = coup(G)
                    plains = [coup(generators(sur_plain(A, rg))) for _ in range(KPLAIN)]
                    z, mu, sd = zstats(C, plains, iu)
                    acc["real_r1"].append(r1(C, gn, iu)); acc["real_rho"].append(rho(G, gn)); acc["real_z"].append(z)
                    acc["real_sharedE"].append(shared_mode(G)[3])
                    M = toy_heads(n, T, b, False, rg)
                    Gm = generators(M); gnm = gnorms(Gm); Cm = coup(Gm)
                    acc["mis_r1"].append(r1(Cm, gnm, iu)); acc["mis_rho"].append(rho(Gm, gnm)); acc["mis_z"].append(zmed_of(Cm, mu, sd, iu))
                    acc["mis_sharedE"].append(shared_mode(Gm)[3])
                    for c, f in CTRLS.items():
                        S = f(A, rg)
                        Gs = generators(S); gns = gnorms(Gs); Cs = coup(Gs)
                        acc[f"{c}_r1"].append(r1(Cs, gns, iu)); acc[f"{c}_rho"].append(rho(Gs, gns))
                        acc[f"{c}_z"].append(zmed_of(Cs, mu, sd, iu)); acc[f"{c}_sharedE"].append(shared_mode(Gs)[3])
                        acc[f"{c}_inv"].append(max(check_invariants(A, S).values()))
                row = {"n": n, "T": T, "boost": b}
                for k, v in acc.items():
                    row[k] = round(float(np.median(v)), 4)
                for c in CTRLS:
                    row[f"{c}_pass_frac"] = round(float(np.mean([
                        (acc[f"{c}_r1"][t] > 0.8) and (acc[f"{c}_rho"][t] < 0.5 * acc["real_rho"][t]) and (acc[f"{c}_z"][t] > acc["real_z"][t])
                        for t in range(TRIALS)])), 3)
                out["rows"].append(row)
                print(f"n={n:2d} T={T:3d} b={b:3.1f} | real r1 {row['real_r1']:6.3f} rho {row['real_rho']:.3f} z {row['real_z']:7.2f} sE {row['real_sharedE']:.3f} | mis r1 {row['mis_r1']:6.3f} | "
                      + " ".join(f"{c}: r1 {row[c+'_r1']:6.3f} rho/real {row[c+'_rho']/max(row['real_rho'],1e-9):.2f} z {row[c+'_z']:7.2f} pass {row[c+'_pass_frac']:.2f}" for c in CTRLS)
                      + f" | inv {max(row[c+'_inv'] for c in CTRLS):.1e} | {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open("alignment_study/control_redesign_toy.json", "w"), indent=1)
    print("CONTROL_TOY_DONE")
