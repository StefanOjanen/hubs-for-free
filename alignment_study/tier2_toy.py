# TIER 2: theory. Three parts, all synthetic or reusing existing artifacts.
# (a) Toy ensembles: causal softmax heads with a logit boost b on one column,
#     SHARED column (all heads col 0) versus DISTINCT columns (head h -> col
#     h+1), sweeping b. Produces the curves zmed(sharedE) and r1(sharedE)
#     for the aligned and misaligned worlds. Prediction the curves encode:
#     alignment, not concentration, drives commutator suppression.
# (b) Generic-deviation prediction: for each real Qwen layer, keep the real
#     (S, a_h, e_h) per window and replace deviation DIRECTIONS with random
#     skew matrices orthogonal to S (cone constraint ignored, labeled
#     approximation). If predicted r1/zmed match the real layer values, the
#     depth profile is carried by (S, a, e) alone.
# (c) Lemma verification: (i) causality makes <G_i, G_j> >= 0; (ii) closed
#     form for the expected plain-surrogate inner product:
#     E[<G_h,G_k>] = (1/2) sum_{i>=1} rbar_hi rbar_ki / i * i = ... see note;
#     with rbar_hi = (offdiag causal row mass)/i the expectation per entry,
#     E[A_h[i,j]] = (1 - A_h[i,i] ... actually rowsum(:i))/i under row
#     permutation, so E[<G_h,G_k>] = (1/2) sum_i i * m_hi * m_ki with
#     m_hi = (sum_{j<i} A_h[i,j]) / i. Verified against sampled surrogates.
import json
import sys
import numpy as np

sys.path.insert(0, "alignment_study")
from common import (load_model, wikitext_windows, attn_layer, generators,
                    gnorms, coup, r1, rho, shared_mode, sur_plain, zstats,
                    zmed_of)

rg = np.random.default_rng(0)

# ---------- (a) toy ensembles ----------
N, T, KPLAIN, TRIALS = 14, 64, 8, 20
BOOSTS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]

def toy_heads(b, shared, rgl):
    temps = np.exp(rgl.normal(0, 0.5, N))
    A = np.zeros((N, T, T))
    for h in range(N):
        c = 0 if shared else h + 1
        lo = rgl.normal(0, 1, (T, T)) / temps[h]
        lo[:, c] += b
        mask = np.tril(np.ones((T, T), bool))
        lo = np.where(mask, lo, -1e9)
        ex = np.exp(lo - lo.max(-1, keepdims=True))
        A[h] = ex / ex.sum(-1, keepdims=True)
    return A

curves = {"aligned": [], "misaligned": []}
iu = np.triu_indices(N, 1)
for b in BOOSTS:
    for world, shared in (("aligned", True), ("misaligned", False)):
        se, zz, rr, hh = [], [], [], []
        for t in range(TRIALS):
            A = toy_heads(b, shared, rg)
            G = generators(A)
            gn = gnorms(G)
            C = coup(G)
            plains = [coup(generators(sur_plain(A, rg))) for _ in range(KPLAIN)]
            z, mu, sd = zstats(C, plains, iu)
            _, _, _, frac = shared_mode(G)
            se.append(frac); zz.append(z)
            rr.append(r1(C, gn, iu)); hh.append(rho(G, gn))
        curves[world].append({"boost": b, "sharedE": float(np.mean(se)),
                              "zmed": float(np.median(zz)),
                              "r1": float(np.median(rr)),
                              "rho": float(np.median(hh))})
        print("toy", world, "b=", b, "sharedE", round(np.mean(se), 3),
              "zmed", round(np.median(zz), 2), "r1", round(np.median(rr), 2), flush=True)

# ---------- (b) generic-deviation prediction on real Qwen layers ----------
MODEL, NWIN, STRIDE, SEQ, REPS = "Qwen/Qwen2.5-0.5B", 8, 40, 64, 6
tok, model = load_model(MODEL)
L = model.config.num_hidden_layers
wins = wikitext_windows(NWIN, STRIDE, SEQ, tok)

def random_skew_perp(S, T, rgl):
    M = rgl.normal(0, 1, (T, T))
    K = (M - M.T) / 2.0
    K -= np.sum(K * S) * S
    return K / np.sqrt((K ** 2).sum())

pred = {l: {"r1_pred": [], "r1_real": [], "z_pred": [], "z_real": []}
        for l in range(L)}
for w, ids in enumerate(wins):
    out = model(**ids)
    for l in range(L):
        A = attn_layer(out, l)
        n, Tw, _ = A.shape
        iu2 = np.triu_indices(n, 1)
        G = generators(A)
        gn = gnorms(G)
        C = coup(G)
        plains = [coup(generators(sur_plain(A, rg))) for _ in range(KPLAIN)]
        zr, mu, sd = zstats(C, plains, iu2)
        Smat, a, e, frac = shared_mode(G)
        pred[l]["r1_real"].append(r1(C, gn, iu2))
        pred[l]["z_real"].append(zr)
        for rep in range(REPS):
            Gt = np.stack([a[h] * Smat + e[h] * random_skew_perp(Smat, Tw, rg)
                           for h in range(n)])
            Ct = coup(Gt)
            pred[l]["r1_pred"].append(r1(Ct, gnorms(Gt), iu2))
            pred[l]["z_pred"].append(zmed_of(Ct, mu, sd, iu2))
    if w % 2 == 0:
        print("GD WIN", w, flush=True)

gd = []
for l in range(L):
    gd.append({"layer": l,
               "r1_real": float(np.median(pred[l]["r1_real"])),
               "r1_pred": float(np.median(pred[l]["r1_pred"])),
               "z_real": float(np.median(pred[l]["z_real"])),
               "z_pred": float(np.median(pred[l]["z_pred"]))})

def spearman(x, y):
    def rank(v):
        idx = np.argsort(v); rk = np.empty(len(v)); rk[idx] = np.arange(len(v))
        return rk
    return float(np.corrcoef(rank(np.array(x)), rank(np.array(y)))[0, 1])

gd_summary = {
    "spearman_r1": spearman([g["r1_real"] for g in gd], [g["r1_pred"] for g in gd]),
    "spearman_z": spearman([g["z_real"] for g in gd], [g["z_pred"] for g in gd]),
    "median_abs_err_r1": float(np.median([abs(g["r1_real"] - g["r1_pred"]) for g in gd])),
}

# ---------- (c) lemma verification ----------
# (i) nonnegativity of generator inner products under causality
neg = 0
for _ in range(200):
    A = toy_heads(float(rg.uniform(0, 4)), bool(rg.integers(2)), rg)
    G = generators(A)
    V = G.reshape(N, -1)
    M = V @ V.T
    neg = max(neg, float(-(M.min())))
# (ii) expected plain-surrogate inner product, closed form vs sampled
A = toy_heads(2.0, True, rg)
G = generators(A)
m = np.zeros((N, T))
for h in range(N):
    for i in range(1, T):
        m[h, i] = A[h, i, :i].sum() / i
pred_ip = np.zeros((N, N))
for h in range(N):
    for k in range(N):
        pred_ip[h, k] = 0.5 * sum(i * m[h, i] * m[k, i] for i in range(1, T))
samp = np.zeros((N, N))
NS = 200
for _ in range(NS):
    Gs = generators(sur_plain(A, rg))
    Vs = Gs.reshape(N, -1)
    samp += Vs @ Vs.T
samp /= NS
offd = ~np.eye(N, dtype=bool)
lemma = {
    "min_inner_product_over_200_ensembles": -neg,
    "closed_form_vs_sampled_offdiag_relerr_median": float(
        np.median(np.abs((pred_ip - samp) / samp)[offd])),
}

res = {"toy_curves": curves, "generic_deviation": gd,
       "generic_deviation_summary": gd_summary, "lemma_checks": lemma}
json.dump(res, open("alignment_study/tier2_toy.json", "w"), indent=1)
print(json.dumps(gd_summary))
print(json.dumps(lemma))
print("GD layer | r1_real | r1_pred | z_real | z_pred")
for g in gd:
    print(f"{g['layer']:>3} | {g['r1_real']:+.2f} | {g['r1_pred']:+.2f} | "
          f"{g['z_real']:+7.2f} | {g['z_pred']:+7.2f}")
