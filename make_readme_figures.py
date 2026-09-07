# Renders the README figures and illustrations from committed artifacts.
# Charts read results.json, alignment_study/*.json and scale_partial/*.json;
# nothing is typed in by hand. Illustrations are conceptual diagrams drawn
# from small synthetic arrays and carry no reported numbers.
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---- design tokens (reference palette, light surface) ----
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titleweight": "normal", "axes.titlesize": 11.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 9, "svg.fonttype": "none",
})
OUT = "figures/readme/"

def style(ax, title=None, sub=None):
    if title: ax.set_title(title, loc="left", pad=14 if sub else 8)
    if sub: ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")
    ax.tick_params(length=0)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)

def endlabel(ax, x, y, text, color=INK2, dx=0.6, **kw):
    ax.text(x + dx, y, text, color=color, fontsize=9, va="center", **kw)

def dot(ax, x, y, c, s=42, z=3):
    ax.scatter(x, y, s=s, color=c, edgecolors=SURF, linewidths=1.6, zorder=z)

# ---- data ----
R = json.load(open("results.json"))
QM = json.load(open("alignment_study/qwen_multilayer.json"))["per_layer"]
HO = json.load(open("alignment_study/heldout_round.json"))
GT = json.load(open("alignment_study/gram_theorem.json"))["per_layer"]
TOY = json.load(open("alignment_study/tier2_toy.json"))["toy_curves"]
SCALE = [json.load(open(f)) for f in sorted(glob.glob("alignment_study/scale_partial/*_capture.json"))]

# ============ Figure 1: the coordinator that random matrices produce ============
N, T = 14, 26
def ens_gaussian(rng, n=N, T_=T):
    tau = np.exp(rng.normal(0, 0.5, n))
    lo = rng.normal(size=(n, T_, T_)) / tau[:, None, None]
    lo = np.where(np.tril(np.ones((T_, T_), bool)), lo, -1e9)
    ex = np.exp(lo - lo.max(-1, keepdims=True))
    return ex / ex.sum(-1, keepdims=True)
seed = R["section42"]["searched_exhibit"]["first_match"]["seed"]
A = ens_gaussian(np.random.default_rng(seed))
G = (A - A.transpose(0, 2, 1)) / 2
P = np.einsum("aij,bjk->abik", G, G); C = np.sqrt(((P - P.transpose(1, 0, 2, 3)) ** 2).sum((2, 3)))
ev = np.sort(np.linalg.eigvalsh(C))[::-1]
gap = ev[0] / abs(ev[1]); hub = int(C.sum(1).argmax())
assert abs(gap - R["section42"]["searched_exhibit"]["first_match"]["gap"]) < 1e-6
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.35})
ax[0].grid(False); im = ax[0].imshow(C, cmap=CMAP); ax[0].set_xticks([]); ax[0].set_yticks([])
ax[0].add_patch(plt.Rectangle((-0.5, hub - 0.5), N, 1, fill=False, ec=S2, lw=1.6))
ax[0].text(0, N - 0.1, f"outlined row: head {hub}, the hub that couples to everyone", color=INK2, fontsize=8.5, va="top", transform=ax[0].transData)
style(ax[0], "Coupling matrix of 14 random heads", f"Gaussian-logit causal softmax, seed {seed}. No training.")
ax[1].bar(range(N), ev, width=0.55, color=S1, zorder=3)
ax[1].set_xticks(range(N)); ax[1].set_xticklabels([str(i) for i in range(N)], fontsize=8)
ax[1].set_xlabel("eigenvalue index"); ax[1].axhline(0, color=GRID, lw=0.8)
ax[1].annotate(f"gap = {gap:.0f}x", xy=(0, ev[0]), xytext=(2.4, ev[0] * 0.9), color=INK2, fontsize=9,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
style(ax[1], "Its eigenvalues: one versus the rest", "The '(n-1)+1' spectrum, for free")
fig.savefig(OUT + "fig1_coordinator_from_noise.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ============ Figure 2: false-flag rate of the 2-sigma rule ============
s42 = R["section42"]; ns, ps = s42["n"], [100 * p for p in s42["p_any_flag"]]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(ns, ps, color=S1, lw=2, solid_capstyle="round", zorder=2); dot(ax, ns, ps, S1)
for n_, p_ in zip(ns, ps):
    if n_ in (14, 32): ax.text(n_, p_ + 4, f"{p_:.0f}%", color=INK, fontsize=9.5, ha="center")
ax.set_ylim(0, 105); ax.set_xticks(ns); ax.set_xticklabels([str(n_) if n_ != 16 else "" for n_ in ns]); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0", "25", "50", "75", "100%"]); ax.set_xlabel("number of heads n")
style(ax, "How often noise produces a 'special' head", "2-sigma rule on generator norms or coupling row sums, 300 draws per n")
fig.savefig(OUT + "fig2_false_flag_rate.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ============ Figure 3: depth profile, real vs surrogates (Qwen2.5-0.5B) ============
L = [r["layer"] for r in QM]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.15], "hspace": 0.12})
series = [("plain surrogate (keeps row sharpness, self-mass)", [r["plain_r1"] for r in QM], S2),
          ("real attention", [r["real_r1"] for r in QM], S1),
          ("sink-column surrogate (also keeps one number per row)", [r["sinkfix_r1"] for r in QM], S3)]
for name, ys, c in series:
    a1.plot(L, ys, color=c, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=2, label=name)
    dot(a1, L, ys, c, s=26)
a1.axhline(0, color=GRID, lw=0.8); a1.set_ylim(-0.8, 1.1); a1.set_yticks([-0.5, 0, 0.5, 1.0])
a1.set_ylabel("rank-1 correlation r1"); a1.legend(loc="lower left", ncol=1)
endlabel(a1, L[-1], series[0][1][-1], "noise level", dx=0.5)
style(a1, "One number per row reproduces the interaction geometry", "Qwen2.5-0.5B, 24 layers, 12 windows: the sink-column surrogate tracks the real curve layer by layer")
m = [r["mean_col0"] for r in QM]
a2.fill_between(L, 0, m, color=S1, alpha=0.10); a2.plot(L, m, color=S1, lw=2); dot(a2, L, m, S1, s=20)
a2.set_ylim(0, 1); a2.set_yticks([0, 0.5, 1.0]); a2.set_ylabel("sink mass"); a2.set_xlabel("layer"); a2.set_xticks(range(0, 24, 3))
style(a2)
fig.savefig(OUT + "fig3_depth_profile.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ============ pooled high-sink layers ============
rows = []
for mname, rs in HO["models"].items():
    if isinstance(rs, list): rows += [dict(r, model=mname, round="sub-2B held-out") for r in rs]
for d in SCALE: rows += [dict(r, model=d["model"], round="3B to 7B") for r in d["per_layer"]]
for r in GT: rows.append({"model": "Qwen/Qwen2.5-0.5B", "round": "development", "smass": r["mean_col0"],
                          "cosS": r["cos_S_sink_median"], "sharedE": r["mean_shared_energy"], "r1_real": r["r1_real_median"]})
hs = [r for r in rows if r["smass"] > 0.4]

# ============ Figure 4: the shared operator is the sink ============
fig, ax = plt.subplots(figsize=(6.8, 4.2))
dot(ax, [r["smass"] for r in hs], [r["cosS"] for r in hs], S1, s=30)
ax.axhline(0.7, color=MUTED, lw=0.8); ax.text(0.905, 0.715, "0.70 registered threshold", color=INK2, fontsize=8.5, ha="right")
ax.set_xlim(0.38, 0.92); ax.set_ylim(0.6, 1.02); ax.set_xlabel("sink mass of the layer"); ax.set_ylabel("cosine of shared component to the sink operator")
n_hs = len(hs); frac87 = np.mean([r["cosS"] > 0.87 for r in hs]) * 100
ax.text(0.39, 0.625, f"{n_hs} high-sink layers, {len(set(r['model'] for r in hs))} models: all above 0.70, {frac87:.0f}% above 0.87", color=INK2, fontsize=9)
style(ax, "The shared component of a layer's heads is the sink operator", "Every high-sink layer of every model measured, 0.1B to 7B")
fig.savefig(OUT + "fig4_shared_operator_is_sink.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ============ Figure 5: the law that transfers ============
def spearman(x, y):
    def rank(v):
        v = np.array(v); idx = np.argsort(v); rk = np.empty(len(v)); rk[idx] = np.arange(len(v)); return rk
    return float(np.corrcoef(rank(x), rank(y))[0, 1])
pools = [("sub-2B held-out (5 models)", S1, [r for r in hs if r["round"] == "sub-2B held-out"]),
         ("3B to 7B (5 models)", S2, [r for r in hs if r["round"] == "3B to 7B"])]
fig, ax = plt.subplots(figsize=(7.4, 4.6))
toy = sorted(TOY["aligned"], key=lambda d: d["sharedE"])
ax.plot([d["sharedE"] for d in toy], [d["r1"] for d in toy], color=S3, lw=2, zorder=2, label="toy ensemble: noise plus one shared column")
labels = []
for name, c, pool in pools:
    sp = spearman([r["sharedE"] for r in pool], [r["r1_real"] for r in pool])
    dot(ax, [r["sharedE"] for r in pool], [r["r1_real"] for r in pool], c, s=28)
    ax.scatter([], [], color=c, s=28, label=f"{name}, Spearman {sp:.2f}")
ax.axvspan(0.85, 0.93, color=S3, alpha=0.08, lw=0); ax.text(0.89, 0.93, "toy regime flip", color=INK2, fontsize=8.5, ha="center")
ax.axhline(0, color=GRID, lw=0.8); ax.set_xlim(0.48, 1.0); ax.set_ylim(-0.8, 1.02)
ax.set_xlabel("shared-energy fraction of the layer"); ax.set_ylabel("rank-1 correlation r1"); ax.legend(loc="lower left")
style(ax, "One number per layer predicts its interaction geometry", "High-sink layers of ten held-out models; the toy curve was fixed before any model was run")
fig.savefig(OUT + "fig5_alignment_law.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ============ Figure 6: alignment versus concentration (toy worlds) ============
fig, (b1, b2) = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw={"wspace": 0.3})
for world, c, lab in (("aligned", S1, "shared column (all heads sink on the same token)"), ("misaligned", S2, "distinct columns (each head sinks elsewhere)")):
    rows_ = TOY[world]; x = [d["boost"] for d in rows_]
    b1.plot(x, [d["sharedE"] for d in rows_], color=c, lw=2, label=lab); dot(b1, x, [d["sharedE"] for d in rows_], c, s=22)
    b2.plot(x, [d["r1"] for d in rows_], color=c, lw=2, label=lab); dot(b2, x, [d["r1"] for d in rows_], c, s=22)
b1.set_ylim(0, 1.05); b1.set_xlabel("sink strength (logit boost)"); b1.set_ylabel("shared-energy fraction")
b2.set_ylim(-1.05, 1.05); b2.axhline(0, color=GRID, lw=0.8); b2.set_xlabel("sink strength (logit boost)"); b2.set_ylabel("rank-1 correlation r1")
b1.legend(loc="upper left")
style(b1, "Shared mode", "Same per-head concentration in both worlds"); style(b2, "Interaction geometry", "Only the shared column collapses it")
fig.savefig(OUT + "fig6_alignment_not_concentration.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ============ Illustrations (conceptual, SVG) ============
def causal(rng, sink=0.0, diag=0.0, n=9, noise=1.0):
    lo = rng.normal(size=(n, n)) * noise; lo[:, 0] += sink; lo[np.arange(n), np.arange(n)] += diag
    lo = np.where(np.tril(np.ones((n, n), bool)), lo, -1e9); ex = np.exp(lo - lo.max(-1, keepdims=True)); return ex / ex.sum(-1, keepdims=True)
def mat(ax, M, title=None, sub=None, vmax=None):
    ax.imshow(M, cmap=CMAP, vmin=0, vmax=vmax or M.max()); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    if title: ax.set_title(title, fontsize=10, color=INK, loc="left", pad=6)
    if sub: ax.text(0, -0.08, sub, transform=ax.transAxes, color=INK2, fontsize=8.5, va="top")
def arrow(fig, x0, y0, x1, y1):
    fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.2))

# I1: one shared operator plus a residual
rng = np.random.default_rng(3)
fig = plt.figure(figsize=(10, 3.3)); gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 0.55, 1, 0.35, 1], wspace=0.25)
heads = [causal(rng, sink=3.0) for _ in range(3)]
for k in range(3):
    mat(fig.add_subplot(gs[0, k]), heads[k], f"head {k + 1}", "sharp somewhere, and on the sink" if k == 1 else None)
S = np.zeros((9, 9)); S[1:, 0] = 1.0
mat(fig.add_subplot(gs[0, 4]), S, "shared operator S", "the sink column,\none per layer", vmax=1)
E = heads[1] - heads[1][:, :1].mean() * (S > 0); E = np.clip(E - E.min(), 0, None) * (np.tril(np.ones((9, 9))) > 0)
mat(fig.add_subplot(gs[0, 6]), E, "residual E_h", "what makes\nhead h itself", vmax=E.max() * 1.6)
fig.text(0.47, 0.5, "=  a_h", fontsize=13, color=INK, ha="center", va="center")
fig.text(0.735, 0.5, "+", fontsize=15, color=INK, ha="center", va="center")
fig.text(0.5, 0.1, "G_h = a_h S + E_h.  In every commutator [G_i, G_j] the S-with-S term cancels identically; what remains is how each head deviates from the shared operator.", ha="center", color=INK2, fontsize=9.5)
fig.savefig(OUT + "illus1_shared_operator.svg", bbox_inches="tight"); fig.savefig(OUT + "illus1_shared_operator.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# I2: the null families
rng = np.random.default_rng(4)
real = causal(rng, sink=3.0, diag=1.0)
def shuffle_rows(M, keep_cols=()):
    S_ = M.copy(); n = M.shape[0]
    for i in range(1, n):
        cols = [c for c in range(i) if c not in keep_cols]
        if len(cols) > 1: S_[i, cols] = S_[i, np.array(cols)[rng.permutation(len(cols))]]
    return S_
panels = [(real, "real attention map", "sink column, self mass,\nsharp rows"),
          (causal(rng), "random softmax", "keeps only the\ncausal shape"),
          (shuffle_rows(real), "marginal-matched\nsurrogate", "keeps each row's\nsharpness and self-mass"),
          (shuffle_rows(real, keep_cols=(0,)), "sink-column\nsurrogate", "keeps the above plus\none number per row"),
          (causal(rng, noise=0.4), "untrained model", "keeps the architecture,\nnot the training")]
fig, axs = plt.subplots(1, 5, figsize=(13, 3.6), gridspec_kw={"wspace": 0.35})
for ax_, (M, t, s) in zip(axs, panels):
    mat(ax_, M, None, s, vmax=real.max()); ax_.set_title(t, fontsize=9.5, color=INK, loc="left", pad=6)
fig.text(0.5, -0.1, "A statistic earns its meaning by how far the real map sits from each of these. The toolkit reports that distance as a percentile.", ha="center", color=INK2, fontsize=9.5)
fig.savefig(OUT + "illus2_null_families.svg", bbox_inches="tight"); fig.savefig(OUT + "illus2_null_families.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# I3: same signature, two mechanisms
fig = plt.figure(figsize=(10.5, 4.0))
gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 0.5, 1], hspace=0.55, wspace=0.15)
axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[1, 0]); axc = fig.add_subplot(gs[:, 2])
norms = np.array([0.6, 0.55, 1.9, 0.5, 0.65, 0.58, 0.62])
axa.bar(range(7), norms, width=0.55, color=S1); axa.set_xticks([]); axa.set_yticks([]); axa.grid(False)
axa.set_title("random matrices: one head happens to have a large norm", fontsize=10, loc="left", color=INK)
for sp in axa.spines.values(): sp.set_visible(False)
axb.grid(False); axb.set_xticks([]); axb.set_yticks([])
for sp in axb.spines.values(): sp.set_visible(False)
blk = np.tile(np.array([[1.0] + [0.15] * 8]), (7, 1)); blk[:, 1:] += np.random.default_rng(1).uniform(-0.08, 0.12, (7, 8)); axb.imshow(np.clip(blk, 0, 1), cmap=CMAP, aspect="auto", vmin=0, vmax=1)
axb.set_title("trained model: every head shares one column", fontsize=10, loc="left", color=INK)
axc.bar(range(7), np.sort(np.linalg.eigvalsh(np.outer(norms, norms) - np.diag(norms ** 2)))[::-1], width=0.55, color=S1); axc.set_xticks([]); axc.set_yticks([]); axc.grid(False)
for sp in axc.spines.values(): sp.set_visible(False)
axc.set_title("the same hub and eigengap", fontsize=10, loc="left", color=INK)
arrow(fig, 0.50, 0.72, 0.62, 0.55); arrow(fig, 0.50, 0.30, 0.62, 0.45)
fig.text(0.5, 0.0, "Bare statistics cannot tell the two apart. Marginal-matched surrogates reproduce the first; only a sink-column surrogate reproduces the second.", ha="center", color=INK2, fontsize=9.5)
fig.savefig(OUT + "illus3_two_mechanisms.svg", bbox_inches="tight"); fig.savefig(OUT + "illus3_two_mechanisms.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# I4: the toolkit in one line
fig = plt.figure(figsize=(12, 2.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
boxes = [("attention maps", "any model, any layer"), ("your statistic", "clusters, hubs, eigengaps,\nimportance, similarity"),
         ("four null families", "random, marginal-matched,\nsink-column, untrained"), ("percentile report", "how much of the effect\neach null reproduces")]
xs = np.linspace(0.13, 0.87, 4)
for (t, s), x in zip(boxes, xs):
    ax.add_patch(FancyBboxPatch((x - 0.105, 0.3), 0.21, 0.42, boxstyle="round,pad=0.01,rounding_size=0.02", fc="#f1f0ec", ec=GRID, lw=1))
    ax.text(x, 0.6, t, ha="center", va="center", fontsize=10.5, color=INK); ax.text(x, 0.42, s, ha="center", va="center", fontsize=8.5, color=INK2)
for x0, x1 in zip(xs[:-1], xs[1:]):
    ax.add_patch(FancyArrowPatch((x0 + 0.105, 0.51), (x1 - 0.105, 0.51), arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.2))
ax.text(0.5, 0.1, "hubsfree audit config.yaml", ha="center", fontsize=10, color=INK2, family="DejaVu Sans Mono")
fig.savefig(OUT + "illus4_toolkit.svg", bbox_inches="tight"); fig.savefig(OUT + "illus4_toolkit.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("FIGURES_DONE", sorted(glob.glob(OUT + "*")))
