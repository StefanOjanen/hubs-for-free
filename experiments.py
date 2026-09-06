# experiments.py: the synthetic null battery of paper.md, Sections 3 to 4.6.
#
# PROVENANCE. The original experiments.py (August 2026) was never committed
# to this repository. This file is a reimplementation from the paper's own
# specifications (2026-09-02). Where the paper left an ensemble parameter
# implicit, the choice is stated here and the paper's numbers were updated
# to the values this script produces. Differences from the August draft are
# Monte Carlo variation plus these stated choices; no conclusion changed.
#
# Explicit choices:
#   Gaussian ensemble: logits N(0,1)/tau_h, tau_h ~ LogNormal(0, 0.5) per
#     head, causal mask, row softmax. n = 14 heads, T = 26 tokens.
#   Sink variant: 30 percent of heads (4 of 14) get +3.0 on column-0 logits.
#   Dirichlet ensemble: each causal row ~ Dirichlet(alpha, ..., alpha) with
#     alpha set so the ensemble's mean row IPR equals the Gaussian
#     ensemble's (a sharpness-matched null, chosen by rule, not tuned).
#   Two-sigma rule: |x - mean| > 2 * sample std (ddof = 1), applied to
#     generator norms and to coupling row sums; "unique shared outlier" means
#     both statistics flag exactly the same single index.
#   Ward clustering: scipy linkage on the condensed distance 1 - C/max(C),
#     method "ward", cut at 4 clusters (the published pipeline's usage).
#   Locality schedule (4.4): pseudo-layer l in 0..23, t = l/23, column-0
#     logit bias 4(1 - t), diagonal logit bias 4t, 20 draws per layer,
#     raw-attention commutator statistic F = mean_{i<j} ||[A_i, A_j]||_F.
#   L-function (4.5): a_m = |N(0,1)| + 0.1, m = 1..24, five draws; zeros
#     located by grid minima of |L| refined by Newton, Re in [-0.5, 1.6],
#     Im in (0, 140]; spacings unfolded by each draw's mean spacing.
# Runtime: a few minutes on a laptop CPU (the formula search dominates).
import json
import time
from itertools import combinations, product

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.time()
N, T = 14, 26
TRIALS = 300
OUT = {"provenance": {
    "regenerated_from_spec": True, "date": "2026-09-02",
    "note": "Reimplemented from paper.md specifications; original August script not committed."}}

# ---------------------------------------------------------------- ensembles
def softmax_causal(logits):
    T_ = logits.shape[-1]
    mask = np.tril(np.ones((T_, T_), bool))
    lo = np.where(mask, logits, -1e9)
    ex = np.exp(lo - lo.max(-1, keepdims=True))
    return ex / ex.sum(-1, keepdims=True)

def ens_gaussian(rng, n=N, T_=T, sink=False):
    tau = np.exp(rng.normal(0, 0.5, n))
    lo = rng.normal(size=(n, T_, T_)) / tau[:, None, None]
    if sink:
        k = max(1, int(round(0.3 * n)))
        heads = rng.choice(n, k, replace=False)
        lo[heads, :, 0] += 3.0
    return softmax_causal(lo)

def mean_ipr(A):
    return float((A ** 2).sum(-1).mean())

def dirichlet_alpha_matched(target_ipr, T_=T):
    """Concentration alpha such that E[sum_k p_k^2] over causal rows matches
    target_ipr; E[IPR] for Dirichlet(alpha 1_m) is (alpha+1)/(m alpha+1)."""
    ms = np.arange(1, T_ + 1)
    f = lambda a: np.mean((a + 1) / (ms * a + 1)) - target_ipr
    lo, hi = 1e-3, 50.0
    for _ in range(100):
        mid = (lo * hi) ** 0.5
        if f(mid) > 0: lo = mid
        else: hi = mid
    return float((lo * hi) ** 0.5)

_rng_cal = np.random.default_rng(12345)
GAUSS_IPR = float(np.mean([mean_ipr(ens_gaussian(_rng_cal)) for _ in range(200)]))
DIR_ALPHA = dirichlet_alpha_matched(GAUSS_IPR)

def ens_dirichlet(rng, n=N, T_=T, alpha=None):
    alpha = DIR_ALPHA if alpha is None else alpha
    A = np.zeros((n, T_, T_))
    for h in range(n):
        for i in range(T_):
            A[h, i, :i + 1] = rng.dirichlet(np.full(i + 1, alpha))
    return A

ENSEMBLES = {
    "gaussian": lambda r: ens_gaussian(r),
    "gaussian_sink": lambda r: ens_gaussian(r, sink=True),
    "dirichlet": lambda r: ens_dirichlet(r),
}

# --------------------------------------------------------------- statistics
def generators(A):
    return (A - A.transpose(0, 2, 1)) / 2.0

def gnorms(G):
    return np.sqrt((G ** 2).sum((1, 2)))

def coup(G):
    P = np.einsum("aij,bjk->abik", G, G)
    K = P - P.transpose(1, 0, 2, 3)
    return np.sqrt((K ** 2).sum((2, 3)))

def coup_raw(A):
    P = np.einsum("aij,bjk->abik", A, A)
    K = P - P.transpose(1, 0, 2, 3)
    return np.sqrt((K ** 2).sum((2, 3)))

def flags(x):
    m, s = x.mean(), x.std(ddof=1)
    return set(np.where(np.abs(x - m) > 2 * s)[0].tolist())

def ward_sizes(C, k=4):
    D = 1.0 - C / C.max()
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="ward")
    lab = fcluster(Z, k, criterion="maxclust")
    return sorted(np.bincount(lab)[1:].tolist())

def pipeline(A):
    G = generators(A)
    gn = gnorms(G)
    C = coup(G)
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    r1 = float(np.corrcoef(C[iu], np.outer(gn, gn)[iu])[0, 1])
    ev = np.linalg.eigvalsh(C)
    gap = float(ev[-1] / abs(ev[-2]))
    rs = C.sum(1)
    fg, fr = flags(gn), flags(rs)
    return {
        "r1": r1, "gap": gap,
        "hub_is_maxnorm": bool(rs.argmax() == gn.argmax()),
        "any_flag": bool(fg | fr),
        "unique_shared": bool(fg == fr and len(fg) == 1),
        "unique_index": (next(iter(fg)) if (fg == fr and len(fg) == 1) else None),
        "ward": ward_sizes(C),
        "ev": ev.tolist(),
    }

# ------------------------------------------------------- Section 3 checks
rng = np.random.default_rng(0)
p1 = 0.0
for _ in range(100):
    A = ens_gaussian(rng)
    comm = A[0] @ A[1] - A[1] @ A[0]
    p1 = max(p1, float(np.abs(comm @ np.ones(T)).max()))
p2 = 0.0
heads = 0
for _ in range(50):
    A = ens_gaussian(rng)
    G = generators(A)
    lhs = 2 * (G ** 2).sum((1, 2))
    ipr = (A ** 2).sum(-1)
    dg = np.diagonal(A, axis1=1, axis2=2)
    rhs = (ipr - dg ** 2).sum(-1)
    p2 = max(p2, float(np.abs(lhs - rhs).max()))
    heads += A.shape[0]
p3 = 0.0
for _ in range(50):
    A = ens_gaussian(rng)
    G = generators(A); gn = gnorms(G); C = coup(G)
    iu = np.triu_indices(N, 1)
    p3 = max(p3, float((C[iu] / (2 * np.outer(gn, gn)[iu])).max()))
OUT["ensemble_calibration"] = {"gaussian_mean_ipr": GAUSS_IPR, "dirichlet_alpha": DIR_ALPHA}
OUT["section3"] = {"prop1_max_abs_commutator_times_ones": p1,
                   "prop2_max_abs_identity_error": p2, "prop2_heads_checked": heads,
                   "prop3_max_ratio_C_over_bound": p3}

# ---------------------------------------------------- Section 4.1 hub-for-free
s41 = {}
fig1_example = None
for name, fn in ENSEMBLES.items():
    rng = np.random.default_rng(1)
    rows = [pipeline(fn(rng)) for _ in range(TRIALS)]
    s41[name] = {
        "median_r1": float(np.median([r["r1"] for r in rows])),
        "median_gap": float(np.median([r["gap"] for r in rows])),
        "p_hub_is_maxnorm": float(np.mean([r["hub_is_maxnorm"] for r in rows])),
        "p_ward_1_1_1_11": float(np.mean([r["ward"] == [1, 1, 1, 11] for r in rows])),
        "p_any_flag": float(np.mean([r["any_flag"] for r in rows])),
        "p_unique_shared": float(np.mean([r["unique_shared"] for r in rows])),
    }
    if name == "gaussian":
        fig1_example = ens_gaussian(np.random.default_rng(1))
OUT["section41"] = s41

# ------------------------------------------ Section 4.2 base rates across n
s42 = {"n": [], "p_any_flag": [], "p_unique_shared": [], "median_gap": [],
       "normalized_spectra": {}}
for n in (8, 12, 14, 16, 17, 24, 32):
    rng = np.random.default_rng(2)
    rows = [pipeline(ens_gaussian(rng, n=n)) for _ in range(TRIALS)]
    s42["n"].append(n)
    s42["p_any_flag"].append(float(np.mean([r["any_flag"] for r in rows])))
    s42["p_unique_shared"].append(float(np.mean([r["unique_shared"] for r in rows])))
    s42["median_gap"].append(float(np.median([r["gap"] for r in rows])))
    spec = np.median(np.array([sorted(r["ev"], reverse=True) for r in rows]), 0)
    s42["normalized_spectra"][str(n)] = (spec / spec[0]).tolist()

# searched exhibit: a Gaussian draw reproducing the published summary
found, searched, hits = None, 0, 0
for seed in range(5000):
    searched += 1
    r = pipeline(ens_gaussian(np.random.default_rng(seed)))
    if r["unique_shared"] and r["unique_index"] == 9 and r["ward"] == [1, 1, 1, 11]:
        hits += 1
        if found is None:
            found = {"seed": seed, "gap": r["gap"], "ward": r["ward"], "outlier_index": 9}
s42["searched_exhibit"] = {"seeds_searched": searched, "seeds_matching": hits,
                          "first_match": found,
                          "p_unique_outlier_at_prespecified_index": s41["gaussian"]["p_unique_shared"] / N}
OUT["section42"] = s42

# ------------------------------------ Section 4.3 structure constants & search
def sc(A):
    G = generators(A)
    Mv = G.reshape(G.shape[0], -1)
    U, S, Vh = np.linalg.svd(Mv, full_matrices=False)
    B = Vh.reshape(G.shape)
    n = G.shape[0]
    f = np.zeros((n, n, n))
    for i, j, k in product(range(n), repeat=3):
        f[i, j, k] = np.sum(B[k] * (B[i] @ B[j] - B[j] @ B[i]))
    Xa = np.array(list(product(range(n), repeat=3)), float)
    y = f.ravel()
    kp = np.abs(y) > 1e-4
    return Xa[kp], y[kp], f

def rand_tree(rg2, depth=0, maxd=4):
    if depth >= maxd or rg2.random() < 0.28:
        if rg2.random() < 0.7:
            return ["aff", int(rg2.integers(3)), rg2.normal(0, 1), rg2.normal(0, 2)]
        return ["const", rg2.normal() * rg2.choice([0.1, 1, 10])]
    if rg2.random() < 0.45:
        return [str(rg2.choice(["sin", "cos", "exp"])), rand_tree(rg2, depth + 1, maxd)]
    return [str(rg2.choice(["+", "-", "*", "/"])), rand_tree(rg2, depth + 1, maxd),
            rand_tree(rg2, depth + 1, maxd)]

def consts(t, acc):
    if t[0] == "aff": acc += [(t, 2), (t, 3)]
    elif t[0] == "const": acc.append((t, 1))
    elif t[0] in ("sin", "cos", "exp"): consts(t[1], acc)
    elif t[0] in "+-*/": consts(t[1], acc); consts(t[2], acc)
    return acc

def evt(t, Xa):
    op = t[0]
    if op == "aff": return t[2] * Xa[:, t[1]] + t[3]
    if op == "const": return np.full(Xa.shape[0], t[1])
    if op == "exp": return np.exp(np.clip(evt(t[1], Xa), -20, 20))
    if op in ("sin", "cos"): return getattr(np, op)(evt(t[1], Xa))
    a, b = evt(t[1], Xa), evt(t[2], Xa)
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    return a / np.where(np.abs(b) < 1e-9, 1e-9, b)

def r2v(g, yc, ssq):
    if not np.all(np.isfinite(g)): return -1
    gc = g - g.mean(); den = (gc ** 2).sum()
    return -1 if den < 1e-12 else float((gc @ yc) ** 2 / (den * ssq))

def search(Xa, y, seed, n_rand=8000, n_climb=600):
    rg2 = np.random.default_rng(seed); yc = y - y.mean(); ssq = (yc ** 2).sum()
    bt, best = None, -1
    for _ in range(n_rand):
        t = rand_tree(rg2); s = r2v(evt(t, Xa), yc, ssq)
        if s > best: best, bt = s, t
    cc = consts(bt, [])
    for _ in range(n_climb):
        node, idx = cc[rg2.integers(len(cc))]
        old = node[idx]; node[idx] = old + rg2.normal(0, abs(old) * 0.3 + 0.1)
        s = r2v(evt(bt, Xa), yc, ssq)
        if s > best: best = s
        else: node[idx] = old
    return best

A0 = ens_gaussian(np.random.default_rng(3))
Xa, y, f_full = sc(A0)
rng = np.random.default_rng(4)
eps = rng.choice([-1.0, 1.0], N)
f_flip = f_full * eps[:, None, None] * eps[None, :, None] * eps[None, None, :]
kp = np.abs(f_full.ravel()) > 1e-4
sign_corr = float(np.corrcoef(f_full.ravel()[kp], f_flip.ravel()[kp])[0, 1])
best_r2 = {}
for name, fn in ENSEMBLES.items():
    Xn, yn, _ = sc(fn(np.random.default_rng(5)))
    best_r2[name] = [round(search(Xn, yn, s), 4) for s in range(2)]
allr2 = [v for vs in best_r2.values() for v in vs]
grid = np.array(list(product(range(N), repeat=3)), float)
i_, j_, k_ = grid[:, 0], grid[:, 1], grid[:, 2]
pub = (j_ * (i_ - 0.32) - (-k_ + np.exp(np.sin(i_)))) / 0.0079
OUT["section43"] = {
    "n_above_threshold": int(len(y)), "rms": float(np.sqrt((y ** 2).mean())),
    "max_abs_f": float(np.abs(y).max()), "hard_bound": 2.0,
    "sign_convention_corr": sign_corr,
    "noise_search_best_r2": best_r2,
    "noise_search_r2_min": float(min(allr2)), "noise_search_r2_max": float(max(allr2)),
    "noise_search_r2_median": float(np.median(allr2)),
    "archived_real_r2": 0.014,
    "published_formula_range_0based": [float(pub.min()), float(pub.max())],
}

# ---------------------------------------- Section 4.4 depth-wise crystallization
# Two schedules over 24 pseudo-layers, both ending in strong self-attention.
#   A (shared-sink start): column-0 logit bias 4(1 - t), diagonal bias 4t.
#   B (generic start): no early bias, diagonal bias 6t.
L = 24
def run_schedule(kind):
    Fs, Ds, Ss = [], [], []
    for l in range(L):
        t = l / (L - 1)
        vals, dms, sms = [], [], []
        rng = np.random.default_rng(100 + l)
        for _ in range(20):
            lo = rng.normal(size=(N, T, T))
            idx = np.arange(T)
            if kind == "A":
                lo[:, :, 0] += 4.0 * (1 - t)
                lo[:, idx, idx] += 4.0 * t
            else:
                lo[:, idx, idx] += 6.0 * t
            A = softmax_causal(lo)
            iu = np.triu_indices(N, 1)
            vals.append(coup_raw(A)[iu].mean())
            dms.append(np.diagonal(A, axis1=1, axis2=2).mean())
            sms.append(A[:, 1:, 0].mean())
        Fs.append(float(np.mean(vals))); Ds.append(float(np.mean(dms))); Ss.append(float(np.mean(sms)))
    Fs, Ds, Ss = np.array(Fs), np.array(Ds), np.array(Ss)
    half = L // 2
    return {"meanF_by_layer": Fs.tolist(), "diag_mass_by_layer": Ds.tolist(),
            "sink_mass_by_layer": Ss.tolist(),
            "first": float(Fs[0]), "last": float(Fs[-1]), "max": float(Fs.max()),
            "argmax_layer": int(Fs.argmax()),
            "fold_decrease_first_to_last": float(Fs[0] / Fs[-1]),
            "fold_decrease_max_to_last": float(Fs.max() / Fs[-1]),
            "corr_F_diagmass_all": float(np.corrcoef(Fs, Ds)[0, 1]),
            "corr_F_diagmass_second_half": float(np.corrcoef(Fs[half:], Ds[half:])[0, 1])}
OUT["section44"] = {"schedule_A_sharedsink_to_diag": run_schedule("A"),
                    "schedule_B_generic_to_diag": run_schedule("B")}
Fs = np.array(OUT["section44"]["schedule_B_generic_to_diag"]["meanF_by_layer"])
Ds = np.array(OUT["section44"]["schedule_B_generic_to_diag"]["diag_mass_by_layer"])
FsA = np.array(OUT["section44"]["schedule_A_sharedsink_to_diag"]["meanF_by_layer"])

# ------------------------------------------------ Section 4.5 Neural L-function
def lfun(s, a):
    m = np.arange(1, 25)
    return (a[:, None] * np.exp(-np.multiply.outer(np.log(m), s))).sum(0)

def dlfun(s, a):
    m = np.arange(1, 25)
    return -(a[:, None] * np.log(m)[:, None] * np.exp(-np.multiply.outer(np.log(m), s))).sum(0)

zeros_all, spacings_all = [], []
rng = np.random.default_rng(6)
re = np.arange(-0.5, 1.6 + 1e-9, 0.01)
im = np.arange(0.05, 140.0 + 1e-9, 0.02)
RE, IM = np.meshgrid(re, im, indexing="ij")
S = (RE + 1j * IM).ravel()
for d in range(5):
    a = np.abs(rng.normal(size=24)) + 0.1
    Lv = np.abs(lfun(S, a)).reshape(RE.shape)
    loc = np.ones_like(Lv, bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0: continue
            sh = np.roll(np.roll(Lv, dx, 0), dy, 1)
            loc &= Lv <= sh
    loc[[0, -1], :] = False; loc[:, [0, -1]] = False
    cand = (RE + 1j * IM)[loc & (Lv < 1.0)]
    zs = []
    for z in cand:
        s = complex(z)
        for _ in range(30):
            v = lfun(np.array([s]), a)[0]; dv = dlfun(np.array([s]), a)[0]
            if abs(dv) < 1e-14: break
            s -= v / dv
            if abs(v) < 1e-13: break
        if abs(lfun(np.array([s]), a)[0]) < 1e-9 and -0.5 <= s.real <= 1.6 and 0 < s.imag <= 140:
            if all(abs(s - z2) > 1e-5 for z2 in zs):
                zs.append(s)
    zs = sorted(zs, key=lambda z: z.imag)
    zeros_all.append([(z.real, z.imag) for z in zs])
    ims = np.array([z.imag for z in zs])
    sp = np.diff(ims)
    spacings_all.append(sp / sp.mean())
sp_all = np.concatenate(spacings_all)
res_all = np.array([r for zs in zeros_all for r, _ in zs])
sp_sorted = np.sort(sp_all); F_emp = np.arange(1, len(sp_sorted) + 1) / len(sp_sorted)
ks_wig = float(np.abs(F_emp - (1 - np.exp(-np.pi * sp_sorted ** 2 / 4))).max())
ks_poi = float(np.abs(F_emp - (1 - np.exp(-sp_sorted))).max())
OUT["section45"] = {"n_zeros": int(len(res_all)), "mean_re": float(res_all.mean()),
                    "median_re": float(np.median(res_all)), "std_re": float(res_all.std()),
                    "ks_wigner": ks_wig, "ks_poisson": ks_poi,
                    "frac_spacings_below_0.25": float((sp_all < 0.25).mean()),
                    "poisson_expectation_below_0.25": float(1 - np.exp(-0.25))}

# ------------------------------------------------ Section 4.6 ratio numerology
phi = (1 + 5 ** 0.5) / 2
consts_lib = [1 / 2, 1 / 3, 2 / 3, 1 / np.e, 1 - 1 / np.e, 1 / phi, 1 / 2 ** 0.5,
              np.log(2), np.pi / 4, 1 / np.pi, 3 / 5, 5 / 8]
tol = abs(9 / 14 - 1 / phi)
def match_frac(n):
    return float(np.mean([any(abs(k / n - c) <= tol for c in consts_lib) for k in range(1, n + 1)]))
OUT["section46"] = {"tolerance": float(tol), "match_frac_n14": match_frac(14),
                    "match_frac_n17": match_frac(17),
                    "floor_17_over_phi_pm1_coverage": 3 / 17}

# ------------------------------------------------------------------ figures
A = fig1_example; G = generators(A); gn = gnorms(G); C = coup(G)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ax[0].imshow(C, cmap="viridis"); ax[0].set_title("coupling C_ij (noise)")
ax[1].imshow(np.outer(gn, gn), cmap="viridis"); ax[1].set_title("rank-1 ||G_i|| ||G_j||")
ev = np.sort(np.linalg.eigvalsh(C))[::-1]
ax[2].bar(range(N), ev); ax[2].set_title("eigenvalues of C"); ax[2].set_xlabel("index")
fig.tight_layout(); fig.savefig("figures/fig1_hub_for_free.png", dpi=150); plt.close(fig)

rng = np.random.default_rng(7); xs, ys = [], []
for _ in range(50):
    A = ens_gaussian(rng); G = generators(A)
    xs += list(2 * (G ** 2).sum((1, 2)))
    ipr = (A ** 2).sum(-1); dg = np.diagonal(A, axis1=1, axis2=2)
    ys += list((ipr - dg ** 2).sum(-1))
fig, ax = plt.subplots(figsize=(4.2, 4)); ax.scatter(xs, ys, s=8)
ax.plot([min(xs), max(xs)], [min(xs), max(xs)], "k--", lw=1)
ax.set_xlabel("2||G||_F^2"); ax.set_ylabel("sum_i (IPR_i - A_ii^2)"); ax.set_title("Proposition 2, 700 heads")
fig.tight_layout(); fig.savefig("figures/fig2_identity.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].plot(s42["n"], s42["p_any_flag"], "o-"); ax[0].set_ylim(0, 1)
ax[0].set_xlabel("n heads"); ax[0].set_ylabel("P(2-sigma flag | noise)")
for n, spec in s42["normalized_spectra"].items():
    ax[1].plot(range(len(spec)), spec, label=f"n={n}")
ax[1].set_xlabel("index"); ax[1].set_ylabel("eigenvalue / largest"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/fig3_baserates.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.4))
ax.hist(allr2, bins=8); ax.axvline(0.014, color="r", ls="--", label="archived real R^2 = 0.014")
ax.set_xlabel("best searched R^2 on noise"); ax.legend()
fig.tight_layout(); fig.savefig("figures/fig4_symbolic.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.plot(range(L), Fs, "o-", label="B: generic -> self-attention")
ax.plot(range(L), FsA, "s-", label="A: shared sink -> self-attention")
ax2 = ax.twinx(); ax2.plot(range(L), Ds, "--", color="gray"); ax2.set_ylabel("mean diagonal mass (B)")
ax.set_xlabel("pseudo-layer"); ax.set_ylabel("mean pairwise F (raw attention)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/fig5_cooling.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
allz = np.array([z for zs in zeros_all for z in zs])
ax[0].scatter(allz[:, 0], allz[:, 1], s=6); ax[0].axvline(0.5, color="r", ls="--")
ax[0].set_xlabel("Re(s)"); ax[0].set_ylabel("Im(s)"); ax[0].set_title("zeros of random Dirichlet polynomials")
ax[1].hist(sp_all, bins=25, density=True, alpha=0.6, label="unfolded spacings")
sg = np.linspace(0, 3.5, 200)
ax[1].plot(sg, np.pi * sg / 2 * np.exp(-np.pi * sg ** 2 / 4), label="Wigner (GOE)")
ax[1].plot(sg, np.exp(-sg), label="Poisson"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig6_lfunction.png", dpi=150); plt.close(fig)

OUT["runtime_seconds"] = round(time.time() - T0, 1)
json.dump(OUT, open("results.json", "w"), indent=1)
def _slim(o):
    if isinstance(o, dict): return {k: _slim(v) for k, v in o.items() if not k.endswith("_by_layer") and k != "normalized_spectra"}
    return o
print(json.dumps(_slim({k: v for k, v in OUT.items() if k != "section42"}), indent=1))
print("S42 base rates:", dict(zip(s42["n"], [round(p, 2) for p in s42["p_any_flag"]])))
print("S42 unique shared:", dict(zip(s42["n"], [round(p, 2) for p in s42["p_unique_shared"]])))
print("S42 median gap:", dict(zip(s42["n"], [round(g, 1) for g in s42["median_gap"]])))
print("S42 exhibit:", s42["searched_exhibit"])
print("RUNTIME", OUT["runtime_seconds"], "s")
